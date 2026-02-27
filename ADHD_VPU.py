from dataclasses import dataclass, field
from enum import Enum, auto
from collections import deque
from typing import List, Optional, Dict

# --- 1. Hardware Spec. ---
NUM_VREGS = 32         # v0 ~ v31
LANE = 4
VLEN = 8192            # the vector length may change depending on sequence length (1024 byte)
AXI_WIDTH = 64         # 64-bit AXI bus width (8 bytes)
LSU_QUEUE_DEPTH = 16    # LSU uop queue
VALU_QUEUE_DEPTH = 16   # VALU uop queue
CIM_QUEUE_DEPTH = 32    # CIM uop queue (Tensor Core)

class UnitType(Enum):
    LSU  = auto()   # Load/Store
    VALU = auto()   # Vector Arithmetic
    CIM  = auto()   # Tensor/Matrix Core

@dataclass
class MicroOp:
    """Micro-Operation"""
    name: str
    unit_type: UnitType
    latency: int       # abstract latency in cycles
    src_regs: List[int] = field(default_factory=list) # source register IDs
    dst_regs: List[int] = field(default_factory=list) # destination register IDs
    
    def __repr__(self):
        return f"[{self.unit_type.name}] {self.name} (Lat:{self.latency})"

@dataclass
class MacroOp:
    """Macro-Operation"""
    name: str
    # this macro op will be expanded by calling expansion_func with args
    expansion_func: callable 
    args: dict = field(default_factory=dict)


@dataclass
class CSRConfig:
    """CSR Configuration for controlling the VPU"""
    # VRF idx related
    MatA_reg_base: int = 0
    MatB_reg_base: int = 4
    MatC_reg_base: int = 20

    # GEMM Tiling related
    M_tile: int = 64
    N_tile: int = 64
    K_tile: int = 32

    # Macro FSM control related
    Enable_Double_Buffer: bool = True  # 1: Ping-Pong(Normal GEMM), 0: Single Buffer(FlashAttention)

@dataclass
class TensorConfig:
    """Tensor Hardware Configuration for GEMM"""
    phys_M: int = 16
    phys_N: int = 16

@dataclass
class VALUConfig:
    """VALU Hardware Configuration for ARA"""
    lane: int = LANE
    valu_width: int = AXI_WIDTH

@dataclass
class LatencySet:
    """Latency configuration for different micro-ops, can be extended as needed."""
    Load_One_Vector: int = VLEN / AXI_WIDTH + 1 # one cycle for vreg transition
    Store_One_Vector: int = VLEN / AXI_WIDTH + 1 # one cycle for vreg transition
    VALU_VSET: int = 1
    VALU_VMV: int = VLEN / LANE / AXI_WIDTH # AXI_WIDTH is same with VALU width (Assume)
    VALU_VADD: int = VLEN / LANE / AXI_WIDTH # AXI_WIDTH is same with VALU width (Assume)
    VALU_VEXP: int = VLEN / LANE / AXI_WIDTH # AXI_WIDTH is same with VALU width (Assume)
    VALU_VGELU: int = VLEN / LANE / AXI_WIDTH # AXI_WIDTH is same with VALU width (Assume)


# --- 2. Frontend Macro to micor Expander ---
class MacroExpander:
    """
    Will take a MacroOp and expand it into a list of MicroOps based on the provided expansion function.
    """
    def expand(self, macro_op: MacroOp) -> List[MicroOp]:
        # 呼叫 Macro 定義的展開邏輯
        return macro_op.expansion_func(**macro_op.args)

class Scoreboard:
    """
    The scoreboard to check data hazards (RAW, WAW, WAR) and structural hazards (queue full).
    """
    def __init__(self):
        # 紀錄誰正在寫入該暫存器 (True = Busy/Pending Write)
        self.write_pending = [False] * NUM_VREGS
        # 紀錄有多少人正在讀取該暫存器 (Reference Count)
        self.read_count = [0] * NUM_VREGS

    def check_hazard(self, uop: MicroOp) -> bool:
        """Check hazard, if True, the uop should stall at Dispatch stage."""
        # 1. RAW
        for r in uop.src_regs:
            if self.write_pending[r]:
                return True # RAW Hazard
        
        # 2. WAW & WAR
        for r in uop.dst_regs:
            if self.write_pending[r] or self.read_count[r] > 0:
                return True # WAW or WAR Hazard
        
        return False

    def reserve(self, uop: MicroOp):
        """Dispatch success, lock the registers"""
        for r in uop.dst_regs:
            self.write_pending[r] = True
        for r in uop.src_regs:
            self.read_count[r] += 1

    def release(self, uop: MicroOp):
        """Writeback success, release the registers"""
        for r in uop.dst_regs:
            self.write_pending[r] = False
        for r in uop.src_regs:
            self.read_count[r] = max(0, self.read_count[r] - 1)

# --- 3. Backend execute uop Unit ---
class ExecutionUnit:
    """
    The abstract execution unit that can execute micro-ops. 
    It only cares about: receive uop -> countdown (Latency) -> report completion.
    """
    def __init__(self, name, scoreboard: Scoreboard):
        self.name = name
        self.scoreboard = scoreboard
        self.current_uop: Optional[MicroOp] = None
        self.remaining_cycles = 0
        self.busy = False
        
        # the execution unit can also track its own active cycles for utilization analysis
        self.total_active_cycles = 0

    def issue(self, uop: MicroOp):
        """從 Queue 接收一個 uOP"""
        self.current_uop = uop
        self.remaining_cycles = uop.latency
        self.busy = True

    def tick(self):
        """每個 Cycle 呼叫一次"""
        if self.busy:
            self.total_active_cycles += 1
            self.remaining_cycles -= 1
            if self.remaining_cycles <= 0:
                # 執行完成，釋放 Scoreboard
                # print(f"  [WB] {self.name} finished {self.current_uop.name}")
                self.scoreboard.release(self.current_uop)
                self.busy = False
                self.current_uop = None

class DecoupledQueue:
    """指令佇列，帶有深度限制"""
    def __init__(self, depth):
        self.queue = deque()
        self.depth = depth
    
    def push(self, uop: MicroOp) -> bool:
        if len(self.queue) < self.depth:
            self.queue.append(uop)
            return True
        return False # Full
    
    def pop(self) -> Optional[MicroOp]:
        if self.queue:
            return self.queue.popleft()
        return None
        
    def is_full(self):
        return len(self.queue) >= self.depth
    
    def __len__(self):
        return len(self.queue)

# --- 4. The abstract VPU ---
class ADHD_VPU:
    def __init__(self):
        self.global_cycle = 0
        
        # Components
        self.scoreboard = Scoreboard()
        self.expander   = MacroExpander()
        
        # uop Queues
        self.lsu_queue  = DecoupledQueue(LSU_QUEUE_DEPTH)
        self.valu_queue = DecoupledQueue(VALU_QUEUE_DEPTH)
        self.cim_queue  = DecoupledQueue(CIM_QUEUE_DEPTH)
        
        # Execution Units
        self.lsu_unit  = ExecutionUnit("LSU", self.scoreboard)
        self.valu_unit = ExecutionUnit("VALU", self.scoreboard)
        self.cim_unit  = ExecutionUnit("CIM", self.scoreboard)
        
        # Frontend State
        self.macro_instr_buffer = deque() # CPU 發來的 Macro Ops
        self.micro_op_buffer = deque()    # 展開後等待 Dispatch 的 uOPs
        
        # Metrics
        self.total_macro_fetched = 0
        self.total_micro_generated = 0
        self.stall_hazard_cycles = 0
        self.stall_queue_full_cycles = 0

    def fetch_macro(self, macro_ops: List[MacroOp]):
        """CPU send Macro Ops to VPU"""
        for op in macro_ops:
            self.macro_instr_buffer.append(op)
            self.total_macro_fetched += 1

    def tick(self):
        """核心模擬迴圈：一個 Clock Cycle"""
        self.global_cycle += 1
        
        # --- Stage 1: Backend Execute (Consume) ---
        # if execution unit is not busy, try to fetch uop from the corresponding queue
        if not self.lsu_unit.busy:
            op = self.lsu_queue.pop()
            if op: self.lsu_unit.issue(op)
        self.lsu_unit.tick()

        if not self.valu_unit.busy:
            op = self.valu_queue.pop()
            if op: self.valu_unit.issue(op)
        self.valu_unit.tick()
        
        if not self.cim_unit.busy:
            op = self.cim_queue.pop()
            if op: self.cim_unit.issue(op)
        self.cim_unit.tick()

        # --- Stage 2: Frontend Expansion ---
        # if micro buffer is empty, try to expand the next macro instruction
        if not self.micro_op_buffer and self.macro_instr_buffer:
            current_macro = self.macro_instr_buffer.popleft()
            uops = self.expander.expand(current_macro)
            self.micro_op_buffer.extend(uops)
            self.total_micro_generated += len(uops)
            # print(f"Cycle {self.global_cycle}: Expanded {current_macro.name} into {len(uops)} uOPs")

        # --- Stage 3: Frontend Dispatch (Issue) ---
        # Using the Score Board to check hazards for the next uop
        if self.micro_op_buffer:
            uop = self.micro_op_buffer[0] # Peek
            
            # 1. Check Hazard (Scoreboard)
            if self.scoreboard.check_hazard(uop):
                self.stall_hazard_cycles += 1
                # print(f"Cycle {self.global_cycle}: Stall Hazard on {uop}")
                return # Stall Dispatch

            # 2. Check Structural Hazard (Queue Full)
            target_queue = None
            if uop.unit_type == UnitType.LSU: target_queue = self.lsu_queue
            elif uop.unit_type == UnitType.VALU: target_queue = self.valu_queue
            elif uop.unit_type == UnitType.CIM: target_queue = self.cim_queue
            
            if target_queue.is_full():
                self.stall_queue_full_cycles += 1
                # print(f"Cycle {self.global_cycle}: Stall Queue Full for {uop}")
                return # Stall Dispatch

            # 3. Success: Dispatch & Reserve
            self.micro_op_buffer.popleft() # Remove from buffer
            target_queue.push(uop)
            self.scoreboard.reserve(uop)
            # print(f"Cycle {self.global_cycle}: Dispatched {uop}")

    def is_idle(self):
        """it seems like interrupt to notice the central controller that the VPU has finished all work."""
        return (not self.macro_instr_buffer and 
                not self.micro_op_buffer and 
                not self.lsu_unit.busy and 
                not self.valu_unit.busy and 
                not self.cim_unit.busy and
                len(self.lsu_queue) == 0 and
                len(self.valu_queue) == 0 and
                len(self.cim_queue) == 0)
    
    def print_report(self):
        print("="*40)
        print(f"Simulation Report (Total Cycles: {self.global_cycle})")
        print("="*40)
        print(f"Instruction Fetch Reduction:")
        print(f"  - CPU Macro Ops Fetched : {self.total_macro_fetched}")
        print(f"  - VPU Micro Ops Executed: {self.total_micro_generated}")
        print(f"  - Expansion Ratio       : 1:{self.total_micro_generated/self.total_macro_fetched:.1f}")
        print(f"Frontend Stalls:")
        print(f"  - Data Hazard Stalls    : {self.stall_hazard_cycles} cycles")
        print(f"  - Queue Full Stalls     : {self.stall_queue_full_cycles} cycles")
        print(f"Backend Utilization:")
        print(f"  - LSU Active  : {self.lsu_unit.total_active_cycles/self.global_cycle:.1%}")
        print(f"  - VALU Active : {self.valu_unit.total_active_cycles/self.global_cycle:.1%}")
        print(f"  - CIM Active  : {self.cim_unit.total_active_cycles/self.global_cycle:.1%}")
        print("="*40)

# --- 5. Macro template ---
def macro_flash_attn_template(csr: CSRConfig, tensor: TensorConfig, latency: LatencySet, Seq_Len=512):
    """
    【終極完整版】Macro_FlashAttention (支援 Seq_Len = 512)
    包含 Outer Loop (Q) 與 Inner Loop (K, V) 的完整硬體狀態機展開。
    """
    uops = []
    
    # --- 硬體暫存器嚴格配置 (32 VREG 零溢出分配) ---
    reg_q = 0         # v0~v3   (4KB INT8) - 常駐於 Outer Loop
    reg_k = 4         # v4~v7   (4KB INT8) - 流動於 Inner Loop
    reg_v = 8         # v8~v11  (4KB INT8) - 流動於 Inner Loop
    reg_p = 12        # v12~v15 (4KB INT8) - 流動於 Inner Loop
    reg_o_global = 16 # v16~v31 (16KB FP32) - 常駐於 Outer Loop，累積最終結果

    # =====================================================================
    # Outer Loop: Tiling Q (每次處理 64 個 Query Tokens)
    # Seq_Len = 512, M_tile = 64 -> 跑 8 次
    # =====================================================================
    for q_start in range(0, Seq_Len, csr.M_tile):
        
        # 1. [初始化] 清空 O_global (v16~v31) 以及全域 Max/Sum 純量暫存器
        uops.append(MicroOp("VALU_CLEAR_O_GLOBAL", UnitType.VALU, latency=1, src_regs=[], dst_regs=[reg_o_global + i for i in range(16)]))
        
        # 2. [LSU Load] 載入 Q_block_int8 (佔用 v0~v3)
        uops.append(MicroOp(f"LSU_LOAD_Q_TILE_{q_start}", UnitType.LSU, latency=latency.Load_One_Vector*4, dst_regs=[reg_q, reg_q+1, reg_q+2, reg_q+3]))

        # =====================================================================
        # Inner Loop: Tiling K, V (每次載入 64 個 K, V Tokens 來跟 Q 配對)
        # Seq_Len = 512, K_tile = 64 -> 跑 8 次
        # =====================================================================
        for k_start in range(0, Seq_Len, csr.K_tile):
            
            # --- Phase 1: Q * K^T -> S (16KB FP32 存在 L0 Buffer) ---
            uops.append(MicroOp("CIM_CLEAR_L0_BUFFER", UnitType.CIM, latency=1))
            uops.append(MicroOp(f"LSU_LOAD_K_TILE_{k_start}", UnitType.LSU, latency=latency.Load_One_Vector*4, dst_regs=[reg_k, reg_k+1, reg_k+2, reg_k+3]))

            # CIM 時間摺疊 (Temporal Folding): 64x64 mapped to 16x16
            for m_sub in range(0, csr.M_tile, tensor.phys_M):
                for n_sub in range(0, csr.K_tile, tensor.phys_N):
                    actual_q = reg_q + (m_sub // 16)
                    actual_k = reg_k + (n_sub // 16)
                    uops.append(MicroOp(
                        name=f"CIM_QK_16x16_{m_sub}_{n_sub}", 
                        unit_type=UnitType.CIM, latency=csr.K_tile, 
                        src_regs=[actual_q, actual_k], dst_regs=[] # Psum 留在 L0 Buffer
                    ))

            # --- Phase 2: Online Softmax (VALU 接管 L0 Buffer) ---
            # 更新全域 Max/Sum，算出局部機率 P，並量化成 INT8 寫回 reg_p (v12~v15)
            uops.append(MicroOp("VALU_SOFTMAX_UPDATE_MAX_SUM", UnitType.VALU, latency=20, src_regs=[], dst_regs=[]))
            uops.append(MicroOp("VALU_SOFTMAX_EXP_DIV_QUANT", UnitType.VALU, latency=1024, src_regs=[], dst_regs=[reg_p, reg_p+1, reg_p+2, reg_p+3]))

            # --- Phase 3: P * V -> O_partial (16KB FP32 再次存在 L0 Buffer) ---
            uops.append(MicroOp("CIM_CLEAR_L0_BUFFER", UnitType.CIM, latency=1))
            uops.append(MicroOp(f"LSU_LOAD_V_TILE_{k_start}", UnitType.LSU, latency=latency.Load_One_Vector*4, dst_regs=[reg_v, reg_v+1, reg_v+2, reg_v+3]))

            for m_sub in range(0, csr.M_tile, tensor.phys_M):
                for n_sub in range(0, csr.K_tile, tensor.phys_N):
                    actual_p = reg_p + (m_sub // 16)
                    actual_v = reg_v + (n_sub // 16)
                    uops.append(MicroOp(
                        name=f"CIM_PV_16x16_{m_sub}_{n_sub}", 
                        unit_type=UnitType.CIM, latency=csr.K_tile, 
                        src_regs=[actual_p, actual_v], dst_regs=[] # O_partial 留在 L0 Buffer
                    ))

            # --- Phase 4: O_global Update (VALU 將 L0 融合進 VRF 的 O_global) ---
            # 讀取 v16~v31，跟 L0 Buffer 的 O_partial 依照 scale 融合，寫回 v16~v31
            uops.append(MicroOp(
                name=f"VALU_GLOBAL_O_UPDATE_K_{k_start}", 
                unit_type=UnitType.VALU, latency=64, 
                src_regs=[reg_o_global + i for i in range(16)], 
                dst_regs=[reg_o_global + i for i in range(16)]
            ))
            # [Inner Loop 結束] 繼續抓下一個 K, V 來跟現在這個 Q 配對

        # =====================================================================
        # Outer Loop 收尾：這個 Q 的所有 Context 都算完了！
        # =====================================================================
        # 將 reg_o_global (16KB FP32) 量化成 INT8，並存回 SRAM 成為最終的 Output
        # 我們可以用 v12~v15 (原本放 P 的地方，現在空出來了) 當作 Quantize 後的 INT8 暫存區
        uops.append(MicroOp("VALU_QUANTIZE_O_FINAL", UnitType.VALU, latency=20, src_regs=[reg_o_global + i for i in range(16)], dst_regs=[12, 13, 14, 15]))
        uops.append(MicroOp(f"LSU_STORE_O_TILE_{q_start}", UnitType.LSU, latency=latency.Store_One_Vector*4, src_regs=[12, 13, 14, 15]))

    return uops

def macro_gemm_template(csr: CSRConfig, tensor: TensorConfig, latency:LatencySet, M_total=0, N_total=0, K_total=0):
    """
    GEMM template
    Note:
        C = A X B -> [M, N] = [M, K] X [K, N]
        N dimension in inner loop because we want to do flashattention
    """
    uops = []

    for m_start in range(0, M_total, csr.M_tile):
        for n_start in range(0, N_total, csr.N_tile):
            
            # --- [CIM] clear the accumlation registers in tensor core ---
            uops.append(MicroOp("CIM_CLEAR_PSUM_BUFFER", UnitType.CIM, latency=1, src_regs=[], dst_regs=[]))

            # [邊界保護] 計算當下真實的 Tile 大小
            current_m_tile = min(csr.M_tile, M_total - m_start)
            current_n_tile = min(csr.N_tile, N_total - n_start)

            # --- [K Dimension Accumulation Stage] ---
            for k_start in range(0, K_total, csr.K_tile):
                
                # 1. the position for VRF ping-pong buffer (if enabled)
                if csr.Enable_Double_Buffer:
                    offset = ((k_start // csr.K_tile) % 2) * 2  # assume 64x32 occupy 2 VREG with VLEN=8192
                else:
                    offset = 0
                
                reg_a = csr.MatA_reg_base + offset
                reg_b = csr.MatB_reg_base + offset
                # print(f" Ping-Pong Buffer Offset: {offset} (reg_a: v{reg_a}, reg_b: v{reg_b})")

                # 2. [LSU] Load Tile of A and B from SRAM to VRF (LSU)
                uops.append(MicroOp(f"LSU_LOAD_A_TILE", UnitType.LSU, latency=latency.Load_One_Vector*2, dst_regs=[reg_a, reg_a+1]))
                uops.append(MicroOp(f"LSU_LOAD_B_TILE", UnitType.LSU, latency=latency.Load_One_Vector*2, dst_regs=[reg_b, reg_b+1]))
                
                # 3. [CIM] 時間摺疊 (Temporal Folding) 核心邏輯！
                # FSM 在這裡將 64x64 的邏輯任務，切給 16x16 的實體陣列執行
                # Tensor Core 吃 VRF 的資料，但把 Psum 留在自己肚子裡 (不寫 dst_regs)
                for m_sub in range(0, current_m_tile, tensor.phys_M):
                    for n_sub in range(0, current_n_tile, tensor.phys_N):
                        
                        # === 🌟 架構師的精細操作：計算這個 sub-tile 真正用到的實體 VREG ===
                        actual_reg_a = reg_a if m_sub < 32 else reg_a + 1
                        actual_reg_b = reg_b if n_sub < 32 else reg_b + 1
                        
                        uops.append(MicroOp(
                            name=f"CIM_MAC_16x16_sub_{m_sub}_{n_sub}", 
                            unit_type=UnitType.CIM, 
                            latency=csr.K_tile, 
                            # ★ 關鍵：每個 uOp 只會鎖定它真正需要的 1 個 reg_a 和 1 個 reg_b！
                            src_regs=[actual_reg_a, actual_reg_b], 
                            dst_regs=[]
                        ))

            # --- [CIM, LSU] K 維度算完，把 Psum 從 Tensor Core 吐到 SRAM ---
            # 這裡我們用一個特殊的 uOP 把資料從 CIM 直接推給 LSU 存起來
            # 或者先吐回 VRF (例如 v20)，再從 VRF 存出去
            reg_c = csr.MatC_reg_base
            c_regs = [reg_c, reg_c+1, reg_c+2, reg_c+3]
            uops.append(MicroOp("CIM_QUANT_OUT", UnitType.CIM, latency=latency.Store_One_Vector*4, src_regs=[], dst_regs=c_regs))
            uops.append(MicroOp("LSU_STORE_C", UnitType.LSU, latency=latency.Store_One_Vector*4, src_regs=c_regs))
            
    return uops

def macro_gemm_gelu_template(csr: CSRConfig, tensor: TensorConfig, latency: LatencySet, M_total=0, N_total=0, K_total=0):
    """
    【算子融合版】GEMM + GELU (支援時間摺疊與邊界保護)
    特點：展示了 CIM -> VALU -> LSU 的無縫資料傳遞與 Scoreboard RAW 依賴解鎖。
    """
    uops = []

    for m_start in range(0, M_total, csr.M_tile):
        for n_start in range(0, N_total, csr.N_tile):
            
            # --- [CIM] Clear L0 Buffer ---
            uops.append(MicroOp("CIM_CLEAR_L0_BUFFER", UnitType.CIM, latency=1, src_regs=[], dst_regs=[]))

            # [邊界保護]
            current_m_tile = min(csr.M_tile, M_total - m_start)
            current_n_tile = min(csr.N_tile, N_total - n_start)

            # --- [K Dimension Accumulation Stage] (跟 GEMM 完全一樣) ---
            for k_start in range(0, K_total, csr.K_tile):
                
                # 1. Ping-Pong Offset
                offset = ((k_start // csr.K_tile) % 2) * 2 if csr.Enable_Double_Buffer else 0
                reg_a = csr.MatA_reg_base + offset
                reg_b = csr.MatB_reg_base + offset

                # 2. [LSU] Load Data
                uops.append(MicroOp(f"LSU_LOAD_A", UnitType.LSU, latency=latency.Load_One_Vector*2, dst_regs=[reg_a, reg_a+1]))
                uops.append(MicroOp(f"LSU_LOAD_B", UnitType.LSU, latency=latency.Load_One_Vector*2, dst_regs=[reg_b, reg_b+1]))
                
                # 3. [CIM] Temporal Folding
                for m_sub in range(0, current_m_tile, tensor.phys_M):
                    for n_sub in range(0, current_n_tile, tensor.phys_N):
                        actual_reg_a = reg_a if m_sub < 32 else reg_a + 1
                        actual_reg_b = reg_b if n_sub < 32 else reg_b + 1
                        
                        uops.append(MicroOp(
                            name=f"CIM_MAC_sub_{m_sub}_{n_sub}", 
                            unit_type=UnitType.CIM, 
                            latency=csr.K_tile, 
                            src_regs=[actual_reg_a, actual_reg_b], 
                            dst_regs=[] # Psum 在 L0 Buffer 累積
                        ))

            # =====================================================================
            # 🌟 [Fusion Stage: 量化 -> GELU 激勵 -> 儲存] 🌟
            # =====================================================================
            reg_c = csr.MatC_reg_base
            
            # 64x64 的 INT8 是 4KB，精確佔用 4 個 VREG (reg_c ~ reg_c+3)
            c_regs = [reg_c, reg_c+1, reg_c+2, reg_c+3]

            # Step 1 [CIM]: 從 L0 Buffer 量化成 INT8，寫入 VRF。
            # 執行此指令時，Scoreboard 會將 c_regs 標記為 "Busy (Write Pending)"
            uops.append(MicroOp("CIM_QUANT_OUT", UnitType.CIM, latency=latency.Store_One_Vector*4, src_regs=[], dst_regs=c_regs))
            
            # Step 2 [VALU]: 對 VRF 的結果做 GELU 轉換 (查表或近似)。
            # 前端派發時，發現 c_regs 正在被 CIM 寫入 -> 觸發 RAW Hazard Stall！
            # VALU 會乖乖在 Queue 裡面等，直到 CIM 寫完解鎖，VALU 才會瞬間啟動。
            uops.append(MicroOp("VALU_GELU_LUT", UnitType.VALU, latency=latency.VALU_VGELU*4, src_regs=c_regs, dst_regs=c_regs))
            
            # Step 3 [LSU]: 將做完 GELU 的結果存回 SRAM。
            # 前端派發時，發現 c_regs 又被 VALU 鎖定了 -> 再次 RAW Hazard Stall！
            # LSU 必須等 VALU 算完，才能把最終的激勵值存進 Main Memory。
            uops.append(MicroOp("LSU_STORE_C", UnitType.LSU, latency=latency.Store_One_Vector*4, src_regs=c_regs, dst_regs=[]))
            
    return uops

def macro_residual_layernorm_template(csr: CSRConfig, latency: LatencySet, Seq_Len=0, Hidden_Dim=768):
    """
    【算子融合版】Residual Add + LayerNorm
    行為: Output = LayerNorm( Input_A (from Main Branch) + Input_B (from Residual) )
    """
    uops = []
    
    # 假設我們每次處理 M 軸 (Sequence) 的一小塊，以配合 VRF 容量
    # 一個 768 維的 FP32 token 佔 3KB。為簡化，我們用 INT8 (768 Bytes) 模擬，剛好塞進 1 個 VREG
    
    for seq_idx in range(0, Seq_Len, csr.M_tile):
        # 為了雙緩衝與管線交錯，我們依舊可以做 Ping-Pong
        offset = ((seq_idx // csr.M_tile) % 2) * 4 if csr.Enable_Double_Buffer else 0
        
        reg_main = csr.MatA_reg_base + offset       # 存放 Main Branch 輸出 (例如 FFN 的結果)
        reg_residual = csr.MatB_reg_base + offset   # 存放 Residual (Shortcut) 的原始資料
        reg_out = csr.MatC_reg_base + offset        # 存放最終結果
        
        # 中繼純量暫存器 (不會引發 Structural Hazard)
        reg_mean = 28
        reg_var = 29
        
        # 1. [LSU] 載入 Main 與 Residual
        uops.append(MicroOp(f"LSU_LOAD_MAIN_{seq_idx}", UnitType.LSU, latency=latency.Load_One_Vector*2, dst_regs=[reg_main, reg_main+1]))
        uops.append(MicroOp(f"LSU_LOAD_RES_{seq_idx}", UnitType.LSU, latency=latency.Load_One_Vector*2, dst_regs=[reg_residual, reg_residual+1]))
        
        # 2. [VALU] Residual Add
        # Scoreboard 會卡住等 LSU 寫完
        uops.append(MicroOp(f"VALU_VADD_RES", UnitType.VALU, latency=int(latency.VALU_VADD*2), src_regs=[reg_main, reg_main+1, reg_residual, reg_residual+1], dst_regs=[reg_out, reg_out+1]))
        
        # 3. [VALU] LayerNorm - Mean & Variance
        # 這裡的 Latency 我們給一個稍微真實一點的數字 (處理 M_tile * Hidden_Dim)
        realistic_valu_lat = (csr.M_tile * Hidden_Dim) // (LANE * AXI_WIDTH) + 10 
        
        uops.append(MicroOp("VALU_LN_MEAN", UnitType.VALU, latency=realistic_valu_lat, src_regs=[reg_out, reg_out+1], dst_regs=[reg_mean]))
        uops.append(MicroOp("VALU_LN_VAR", UnitType.VALU, latency=realistic_valu_lat, src_regs=[reg_out, reg_out+1, reg_mean], dst_regs=[reg_var]))
        
        # 4. [VALU] LayerNorm - RSQRT & Normalize
        uops.append(MicroOp("VALU_LN_RSQRT", UnitType.VALU, latency=20, src_regs=[reg_var], dst_regs=[reg_var]))
        uops.append(MicroOp("VALU_LN_NORM", UnitType.VALU, latency=realistic_valu_lat, src_regs=[reg_out, reg_out+1, reg_mean, reg_var], dst_regs=[reg_out, reg_out+1]))
        
        # 5. [LSU] 存回 SRAM
        uops.append(MicroOp(f"LSU_STORE_LN_{seq_idx}", UnitType.LSU, latency=latency.Store_One_Vector*2, src_regs=[reg_out, reg_out+1], dst_regs=[]))
        
    return uops


# --- 6. Using Macro template to build the BERT Base ---
def build_bert_base_layer(sim: ADHD_VPU, csr: CSRConfig, tensorHW: TensorConfig, latencySet: LatencySet, seq_len: int):
    """
    建構一層完整的 BERT Base Layer
    BERT Base spec: Hidden_Dim (D) = 768, Heads = 12, Intermediate_Dim = 3072
    """
    D = 768
    D_FFN = 3072
    
    print(f"\n--- Disptaching BERT Base Layer (Sequence Length: {seq_len}) ---")
    
    # 1. Q, K, V Projections (3x GEMM)
    # Input: [N, D], Weight: [D, D] -> Output: [N, D]
    sim.fetch_macro([MacroOp("PROJ_Q", macro_gemm_template, {"csr":csr, "tensor": tensorHW, "latency": latencySet, "M_total": seq_len, "N_total": D, "K_total": D})])
    sim.fetch_macro([MacroOp("PROJ_K", macro_gemm_template, {"csr":csr, "tensor": tensorHW, "latency": latencySet, "M_total": seq_len, "N_total": D, "K_total": D})])
    sim.fetch_macro([MacroOp("PROJ_V", macro_gemm_template, {"csr":csr, "tensor": tensorHW, "latency": latencySet, "M_total": seq_len, "N_total": D, "K_total": D})])
    
    # 2. FlashAttention (Fusion of QK^T + Softmax + PV)
    # 在 12 個 Heads 上平行或序列執行。這裡我們派發 12 次 MacroOp 代表 12 Heads
    for h in range(12):
        sim.fetch_macro([MacroOp(f"FLASH_ATTN_H{h}", macro_flash_attn_template, {"csr":csr, "tensor": tensorHW, "latency": latencySet, "Seq_Len": seq_len})])
    
    # 3. Attention Output Projection (GEMM)
    sim.fetch_macro([MacroOp("ATTN_OUT_PROJ", macro_gemm_template, {"csr":csr, "tensor": tensorHW, "latency": latencySet, "M_total": seq_len, "N_total": D, "K_total": D})])
    
    # 4. Residual Add + LayerNorm 1
    sim.fetch_macro([MacroOp("RES_LN_1", macro_residual_layernorm_template, {"csr":csr, "latency": latencySet, "Seq_Len": seq_len, "Hidden_Dim": D})])
    
    # 5. FFN Layer 1 (GEMM + GELU Fusion)
    # Input: [N, D], Weight: [D, 3072] -> Output: [N, 3072]
    sim.fetch_macro([MacroOp("FFN1_GELU", macro_gemm_gelu_template, {"csr":csr, "tensor": tensorHW, "latency": latencySet, "M_total": seq_len, "N_total": D_FFN, "K_total": D})])
    
    # 6. FFN Layer 2 (GEMM)
    # Input: [N, 3072], Weight: [3072, D] -> Output: [N, D]
    sim.fetch_macro([MacroOp("FFN2", macro_gemm_template, {"csr":csr, "tensor": tensorHW, "latency": latencySet, "M_total": seq_len, "N_total": D, "K_total": D_FFN})])
    
    # 7. Residual Add + LayerNorm 2
    sim.fetch_macro([MacroOp("RES_LN_2", macro_residual_layernorm_template, {"csr":csr, "latency": latencySet, "Seq_Len": seq_len, "Hidden_Dim": D})])


def run_simulation():
    latencySet = LatencySet()
    csr = CSRConfig(MatA_reg_base=0, MatB_reg_base=4, MatC_reg_base=8, Enable_Double_Buffer=True)
    tensorHW = TensorConfig(phys_M=16, phys_N=16)
    sim = ADHD_VPU()
    
    # 設定可變的 Sequence Length
    target_seq_len = 512 # 你可以改成 1024, 4096 看看 Backend Utilization 的變化
    
    # 呼叫巨集組裝
    build_bert_base_layer(sim, csr, tensorHW, latencySet, seq_len=target_seq_len)
    
    print("--- Simulation Running... ---")
    while not sim.is_idle():
        sim.tick()
        
    sim.print_report()

# # --- 測試區 (替換原有的 run_simulation 內容) ---
# def run_simulation():
#     latencySet = LatencySet()
#     csr = CSRConfig(MatA_reg_base=0, MatB_reg_base=4, MatC_reg_base=8, Enable_Double_Buffer=True)
#     tensorHW = TensorConfig(phys_M=16, phys_N=16)
#     sim = ADHD_VPU()

#     # 假設 CPU 在迴圈裡，先 Load 資料，然後發出一個 FlashAttention Macro
#     # 我們這裡專注發射 Macro_FlashAttention
#     # TODO [Will build the BERT Base model here in future]
#     macro_gemm  = MacroOp("MACRO_GEMM", macro_gemm_template, {"csr":csr, "tensor": tensorHW, "latency": latencySet, "M_total": 512, "N_total": 768, "K_total": 768})
#     macro_gemm_gelu  = MacroOp("MACRO_GEMM", macro_gemm_gelu_template, {"csr":csr, "tensor": tensorHW, "latency": latencySet, "M_total": 512, "N_total": 3072, "K_total": 768})
#     macro_flash = MacroOp("MACRO_FLASH_ATTN", macro_flash_attn_template, {"csr":csr, "tensor": tensorHW, "latency": latencySet, "Seq_Len": 512})
    
    
#     print("--- Starting ADHD VPU Simulation (FlashAttention Fusion) ---")
#     sim.fetch_macro([macro_gemm])
#     sim.fetch_macro([macro_gemm_gelu])
#     sim.fetch_macro([macro_flash])
    
    
#     while not sim.is_idle():
#         sim.tick()
#         # if sim.global_cycle > 1000: break
        
#     sim.print_report()

if __name__ == "__main__":
    run_simulation()