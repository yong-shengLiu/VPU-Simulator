import dataclasses
from typing import List, Optional, Tuple
from enum import Enum
import math

# ==========================================
# 1. 硬體規格與操作定義 (保持不變)
# ==========================================
class OpType(Enum):
    GEMM    = 1
    SOFTMAX = 2
    LOAD    = 3
    STORE   = 4
    # 新增虛擬指令，用於模擬資源釋放
    FREE_VRF = 99 

@dataclasses.dataclass
class HardwareConfig:
    vlen_bits: int = 4096
    vregs_total: int = 32
    elen_bits: int = 16 
    
    @property
    def elements_per_reg(self):
        return self.vlen_bits // self.elen_bits

@dataclasses.dataclass
class VRFRegion:
    start_reg: int
    num_regs: int
    
    def __repr__(self):
        if self.num_regs == 1: return f"v{self.start_reg}"
        return f"v{self.start_reg}-v{self.start_reg + self.num_regs - 1}"

@dataclasses.dataclass
class MacroOpParams:
    op_id: int
    op_type: OpType
    
    # 資源管理 (用於模擬與生成)
    alloc_regs: Optional[VRFRegion] = None # 該指令分配了新資源
    free_regs: Optional[VRFRegion] = None  # 該指令釋放了資源
    
    # Synchronization
    wait_token: int = -1
    signal_token: int = -1
    
    # Dimensions (Actual Size)
    m: int = 0
    n: int = 0
    k: int = 0
    
    # Operands
    vrf_a: Optional[VRFRegion] = None
    vrf_b: Optional[VRFRegion] = None
    vrf_c: Optional[VRFRegion] = None

    def __repr__(self):
        if self.op_type == OpType.FREE_VRF:
            return f"[ID:{self.op_id:<3} FREE   ] Reclaim {self.free_regs}"
            
        sync = f"W:{self.wait_token:<2} S:{self.signal_token:<2}"
        dims = f"{self.m}x{self.n}x{self.k}"
        
        info = ""
        if self.op_type == OpType.GEMM:
            info = f"Acc:{self.vrf_c} += A x B"
        elif self.op_type == OpType.SOFTMAX:
            info = f"In/Out:{self.vrf_a}"
        elif self.op_type == OpType.STORE:
            info = f"Src:{self.vrf_c} -> MEM"
            
        alloc_str = f" [Alloc {self.alloc_regs}]" if self.alloc_regs else ""
        return f"[ID:{self.op_id:<3} {self.op_type.name:<7}] {sync} | Size:{dims:<11} | {info}{alloc_str}"

# ==========================================
# 2. 動態 VRF 管理器 (支援 Fragment 處理)
# ==========================================
class DynamicVRFManager:
    def __init__(self, config: HardwareConfig):
        self.config = config
        self.used_map = [False] * config.vregs_total
        self.total_used = 0

    def allocate_exact(self, num_elements: int) -> VRFRegion:
        """
        根據實際資料量 (num_elements) 動態計算需要的暫存器數量。
        """
        regs_needed = math.ceil(num_elements / self.config.elements_per_reg)
        
        # 尋找 First-Fit
        # 在真實 NPU 中，這裡可以結合 Register Renaming logic
        best_start = -1
        for i in range(self.config.vregs_total):
            if i + regs_needed > self.config.vregs_total:
                break
            
            # Range from i to reg needed, is not used in used_map. If all range true, return free
            is_free = all(not self.used_map[k] for k in range(i, i + regs_needed))

            # Flip the used_map status to True
            if is_free:
                for k in range(i, i + regs_needed):
                    self.used_map[k] = True
                self.total_used += regs_needed
                return VRFRegion(i, regs_needed)
                
        raise RuntimeError(f"VRF Full! Requested {regs_needed} regs. Map: {self.get_status_str()}")

    def free(self, region: VRFRegion):
        if not region: return
        for k in range(region.start_reg, region.start_reg + region.num_regs):
            self.used_map[k] = False
        self.total_used -= region.num_regs

    def get_status_str(self):
        return "".join(['█' if u else '░' for u in self.used_map])
        
    def get_utilization(self):
        return self.total_used / self.config.vregs_total

# ==========================================
# 3. 動態排程生成器
# ==========================================
class DynamicMacroGen:
    def __init__(self, hw_config: HardwareConfig):
        self.hw = hw_config
        self.vrf = DynamicVRFManager(hw_config)
        self.op_counter = 0
        self.token_counter = 0
        
    def get_id(self):
        self.op_counter += 1
        return self.op_counter
        
    def get_token(self):
        self.token_counter += 1
        return self.token_counter

    def emit_dynamic_pipeline(self, M, N, K, tile_M_max, tile_N_max):
        """
        生成動態指令流。
        特色：針對邊緣 Tile (Edge Tiles) 動態縮小 VRF 分配量。
        """
        ops = []
        
        # 預先分配固定權重 B 和輸入 A 的空間 (假設 Streaming)
        # 這裡為了演示，假設 A, B 佔用固定空間
        reg_b = self.vrf.allocate_exact(tile_N_max * K) # 簡化估計
        reg_a = self.vrf.allocate_exact(tile_M_max * K)
        
        print(f"Static Alloc: A={reg_a}, B={reg_b}")
        
        # 用於追蹤 Pipeline 依賴
        # active_tiles 是一個 queue，存儲正在飛行中的 Tile 資訊
        # (op_id, vrf_region, signal_token)
        active_tiles = []
        
        # Double/Triple Buffering Limit (最多允許幾個 Tile 同時在飛)
        # 這取決於 VRF 剩餘空間
        MAX_IN_FLIGHT = 2 
        
        num_tiles_m = (M + tile_M_max - 1) // tile_M_max
        
        for i in range(num_tiles_m):
            # 1. 計算當前 Tile 的實際大小 (Edge Case Handling)
            curr_m_start = i * tile_M_max
            curr_m_size = min(tile_M_max, M - curr_m_start)
            
            # 2. 動態分配 VRF 給 Accumulator
            # 這是關鍵：只分配「真正需要」的大小
            needed_elements = curr_m_size * tile_N_max # 假設 N 不切分
            
            # 如果 VRF 滿了，必須等待最舊的 Tile 完成並釋放
            while True:
                try:
                    curr_acc_reg = self.vrf.allocate_exact(needed_elements)
                    break # 分配成功
                except RuntimeError:
                    # 分配失敗，插入 Wait/Free 指令來釋放空間
                    if not active_tiles: raise RuntimeError("Deadlock! VRF too small even for 1 tile.")
                    oldest_tile = active_tiles.pop(0)
                    # 模擬釋放 (在生成的 code 中不一定有顯式 free，通常是最後一個 use 後自動釋放)
                    # 但在這裡我們生成一個顯式的標記方便觀察
                    ops.append(MacroOpParams(self.get_id(), OpType.FREE_VRF, free_regs=oldest_tile['reg']))
                    self.vrf.free(oldest_tile['reg'])
            
            # 3. 生成 GEMM 指令
            gemm_signal = self.get_token()
            gemm_op = MacroOpParams(
                op_id=self.get_id(),
                op_type=OpType.GEMM,
                signal_token=gemm_signal,
                # Wait logic: 如果我們複用了剛剛釋放的暫存器，硬體會自動處理 RAW/WAW
                # 這裡假設依賴是透過暫存器 ID 自動追蹤 (Scoreboarding)
                m=curr_m_size, n=tile_N_max, k=K,
                vrf_a=reg_a, vrf_b=reg_b, vrf_c=curr_acc_reg,
                alloc_regs=curr_acc_reg
            )
            ops.append(gemm_op)
            
            # 4. 生成 Softmax/Store 指令 (緊接在後，Pipeline)
            # Softmax 依賴 GEMM
            store_signal = self.get_token()
            softmax_op = MacroOpParams(
                op_id=self.get_id(),
                op_type=OpType.SOFTMAX,
                wait_token=gemm_signal, # 等 GEMM 算完
                m=curr_m_size, n=tile_N_max,
                vrf_a=curr_acc_reg # In-place
            )
            ops.append(softmax_op)
            
            store_op = MacroOpParams(
                op_id=self.get_id(),
                op_type=OpType.STORE,
                signal_token=store_signal, # Store 完代表這個 Tile 結束
                m=curr_m_size, n=tile_N_max,
                vrf_c=curr_acc_reg
            )
            ops.append(store_op)
            
            # 記錄這個 Tile，稍後釋放
            active_tiles.append({
                'reg': curr_acc_reg,
                'done_token': store_signal
            })
            
            # 打印當前 VRF 狀態
            print(f"Iter {i}: Alloc {str(curr_acc_reg):<9} (Size {curr_m_size:>3} x{tile_N_max:>3}) | Util: {self.vrf.get_utilization()*100:.1f}% | Map: {self.vrf.get_status_str()}")

        # 清理剩餘的 Tile
        while active_tiles:
            oldest = active_tiles.pop(0)
            ops.append(MacroOpParams(self.get_id(), OpType.FREE_VRF, free_regs=oldest['reg']))
            self.vrf.free(oldest['reg'])

        return ops

# ==========================================
# 4. 模擬執行
# ==========================================
if __name__ == "__main__":
    hw = HardwareConfig(vlen_bits=4096, elen_bits=16) # 256 elems per reg
    gen = DynamicMacroGen(hw)
    
    # --- 參數修正與分析 ---
    # 硬體限制: 32 VRF total.
    # 1 VRF = 256 elements.
    
    # 測試案例:
    # M = 70 (Edge Case 測試)
    # N = 64 (Accumulator Width)
    # K = 16 (Reduction Depth)
    
    # 預估資源消耗:
    # 1. Weight B (Static): K * N = 16 * 64 = 1024 elems = 4 Regs (v0-v3)
    # 2. Input A (Static): Tile_M * K = 16 * 16 = 256 elems = 1 Reg (v4)
    # 3. Accumulator C (Dynamic): Tile_M * N = 16 * 64 = 1024 elems = 4 Regs per Tile
    #    -> Double Buffer 需要 8 Regs。
    # 總共需求約 13 Regs，遠小於 32，可以成功執行！

    M_GLOBAL = 70  
    N_GLOBAL = 64
    K_GLOBAL = 16
    
    TILE_M = 16
    TILE_N = 64
    
    print(f"=== Dynamic Simulation: M={M_GLOBAL} (Tile_M={TILE_M}) ===")
    try:
        stream = gen.emit_dynamic_pipeline(M_GLOBAL, N_GLOBAL, K_GLOBAL, TILE_M, TILE_N)
        
        print("\n=== Generated Instruction Trace ===")
        print(f"{'ID':<5} {'Type':<10} {'Sync':<15} {'Size':<12} {'Info'}")
        print("-" * 80)
        for op in stream:
            print(op)
            
    except RuntimeError as e:
        print(f"\n[Error] {e}")
    
    print(f"=== Dynamic Simulation: M={M_GLOBAL} (Tile_M={TILE_M}) ===")
    stream = gen.emit_dynamic_pipeline(M_GLOBAL, N_GLOBAL, K_GLOBAL, TILE_M, TILE_N)
    
    print("\n=== Generated Instruction Trace ===")
    for op in stream:
        print(op)