from dataclasses import dataclass, field
from enum import Enum, auto
from collections import deque
from typing import List, Optional, Dict

# --- 1. Hardware Spec. ---
NUM_VREGS = 32         # v0 ~ v31
LANE = 4
VLEN = 8192            # the vector length may change depending on sequence length (1024 byte)
AXI_WIDTH = 64         # 64-bit AXI bus width (8 bytes)
LSU_QUEUE_DEPTH = 8    # LSU uop queue
VALU_QUEUE_DEPTH = 8   # VALU uop queue
CIM_QUEUE_DEPTH = 4    # CIM uop queue (Tensor Core)

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
    MatA_reg_base: int
    MatB_reg_base: int
    MatC_reg_base: int

    # Macro FSM control related
    Enable_Double_Buffer: bool  # 1: Ping-Pong(Normal GEMM), 0: Single Buffer(FlashAttention)

@dataclass
class TensorConfig:
    """Tensor Hardware Configuration for GEMM"""
    input_row:  int  # input row is variable, depending on current vrf is enough to hold the tile or not
    tensor_col: int
    tensor_row: int

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
    VALU_VADD: int
    VALU_VEXP: int
    VALU_VGELU: int


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
def macro_gemm_template(k_tiles=4, reg_a=0, reg_b=4, reg_c=8):
    """
    1. Macro_GEMM: 標準矩陣乘加 (Output Stationary)
    特點：在 K 維度累加時，Psum 留在 Tensor Core 內部，不污染 VRF (不寫入 dst_regs)。
    直到 K 迴圈結束，才透過 VCIM_OUT 寫回 VRF。
    """
    uops = []
    # 執行 K 次的 MAC 運算 (從 VRF 讀取 A 和 B)
    for k in range(k_tiles):
        uops.append(MicroOp(
            name=f"VCIM_MAC_k{k}", 
            unit_type=UnitType.CIM, 
            latency=4, 
            src_regs=[reg_a + k, reg_b + k], 
            dst_regs=[] # ★ 關鍵：Psum 在 CIM 內部，不寫回 VRF，無 Hazard！
        ))
    
    # 累加結束，將結果從 Accumulator 寫回 VRF (reg_c)
    uops.append(MicroOp(
        name=f"VCIM_OUT", 
        unit_type=UnitType.CIM, 
        latency=2, 
        src_regs=[], 
        dst_regs=[reg_c] # 鎖定 reg_c，後續指令若要讀取會被 Scoreboard 擋住
    ))
    return uops

def macro_softmax_template(v_start=0, length=4):
    """
    定義 Softmax 的展開邏輯。
    這模擬了你在 Thesis 中提到的 Softmax (Load -> Exp -> Sum -> Div -> Store)
    """
    uops = []
    # 假設每一個向量暫存器可以存 1 筆資料
    for i in range(length):
        reg = v_start + i
        # 1. Load Data (LSU)
        uops.append(MicroOp(f"VLOAD.v{reg}", UnitType.LSU, latency=10, dst_regs=[reg]))
        # 2. Exp (VALU) - RAW Hazard on reg
        uops.append(MicroOp(f"VEXP.v{reg}", UnitType.VALU, latency=4, src_regs=[reg], dst_regs=[reg]))
        # 3. Accumulate (VALU) - 假設 v30 是 accumulator
        uops.append(MicroOp(f"VADD.v30.v{reg}", UnitType.VALU, latency=2, src_regs=[reg, 30], dst_regs=[30]))
    
    # ... 省略 Div 和 Store 以簡化 demo ...
    return uops

def macro_gemm_gelu_template(k_tiles=4, reg_a=0, reg_b=4, reg_c=8):
    """
    2. Macro_GEMM_GELU: 算子融合 (GEMM 完無縫接軌 GELU)
    特點：VALU 會緊盯著 CIM 的輸出 (reg_c)。Scoreboard 會自動處理跨 Unit 的 RAW Hazard。
    """
    # 先把 GEMM 展開
    uops = macro_gemm_template(k_tiles, reg_a, reg_b, reg_c)
    
    # 緊接著插入 GELU 的硬體微指令 (分段線性或查表逼近)
    # 這些指令會因為 reg_c 還在 CIM 裡算，而被 Scoreboard 卡在 Dispatch 階段
    uops.append(MicroOp(name="VGELU_MUL_0.5", unit_type=UnitType.VALU, latency=2, src_regs=[reg_c], dst_regs=[reg_c]))
    uops.append(MicroOp(name="VGELU_TANH_APPX", unit_type=UnitType.VALU, latency=5, src_regs=[reg_c], dst_regs=[reg_c]))
    
    return uops

def macro_flash_attn_template(reg_q=0, reg_k=4, reg_v=8, reg_score=12, reg_out=16):
    """
    3. Macro_FlashAttention: Q*K -> Softmax -> Score*V
    特點：完美展示 NPU 最難的「CIM -> VALU -> CIM」異質管線交錯。
    """
    uops = []
    # Step 1: Q * K^T (假設 k_tiles=4), 輸出到 reg_score
    uops.extend(macro_gemm_template(k_tiles=4, reg_a=reg_q, reg_b=reg_k, reg_c=reg_score))
    
    # Step 2: Online Softmax (由 VALU 執行)
    # VALU 必須等 Step 1 的 VCIM_OUT 寫入 reg_score 後才能啟動
    uops.append(MicroOp("VMAX_REDUCE", unit_type=UnitType.VALU, latency=3, src_regs=[reg_score], dst_regs=[20])) # 找最大值放 v20
    uops.append(MicroOp("VSUB_EXP", unit_type=UnitType.VALU, latency=6, src_regs=[reg_score, 20], dst_regs=[reg_score])) # 減去 Max 並 Exp
    uops.append(MicroOp("VSUM_REDUCE", unit_type=UnitType.VALU, latency=3, src_regs=[reg_score], dst_regs=[21])) # 求 Sum 放 v21
    uops.append(MicroOp("VDIV", unit_type=UnitType.VALU, latency=8, src_regs=[reg_score, 21], dst_regs=[reg_score])) # 算出最終 Score 機率
    
    # Step 3: Score * V (再交回給 CIM 執行)
    # CIM 必須等 VALU 的 VDIV 寫完 reg_score 才能啟動！
    for k in range(4): # 假設 V 的維度也是 4
        uops.append(MicroOp(f"VCIM_MAC_V_k{k}", unit_type=UnitType.CIM, latency=4, src_regs=[reg_score, reg_v + k], dst_regs=[]))
    uops.append(MicroOp("VCIM_OUT_FINAL", unit_type=UnitType.CIM, latency=2, src_regs=[], dst_regs=[reg_out]))
    
    return uops

def macro_layernorm_template(reg_in=0, reg_out=4, reg_mean=20, reg_var=21):
    """
    4. Macro_LayerNorm: Mean -> Var -> Normalize
    特點：考驗 VALU 的 Reduction (歸約運算) 與 RSQRT (開根號倒數) 支援度。
    """
    uops = []
    # Step 1: 算平均值 (Mean)
    uops.append(MicroOp("VSUM_REDUCE", unit_type=UnitType.VALU, latency=3, src_regs=[reg_in], dst_regs=[reg_mean]))
    uops.append(MicroOp("VDIV_N", unit_type=UnitType.VALU, latency=4, src_regs=[reg_mean], dst_regs=[reg_mean]))
    
    # Step 2: 算變異數 (Variance) -> (x - mean)^2
    uops.append(MicroOp("VSUB_SQUARE", unit_type=UnitType.VALU, latency=5, src_regs=[reg_in, reg_mean], dst_regs=[reg_out]))
    uops.append(MicroOp("VSUM_REDUCE", unit_type=UnitType.VALU, latency=3, src_regs=[reg_out], dst_regs=[reg_var]))
    
    # Step 3: RSQRT & Normalize -> (x - mean) * rsqrt(var)
    uops.append(MicroOp("VRSQRT", unit_type=UnitType.VALU, latency=10, src_regs=[reg_var], dst_regs=[reg_var]))
    uops.append(MicroOp("VMUL_NORM", unit_type=UnitType.VALU, latency=3, src_regs=[reg_in, reg_var], dst_regs=[reg_out]))
    
    return uops

def macro_gemm_template_csr(csr: CSRConfig, tensor: TensorConfig, latency:LatencySet, M_total=0, N_total=0, K_total=0):
    """
    GEMM template
    Note:
        C = A X B -> [M, N] = [M, K] X [K, N]
        N dimension in inner loop because we want to do flashattention
    """
    uops = []

    for m_start in range(0, M_total, tensor.input_row):
        for n_start in range(0, N_total, tensor.tensor_col):
            
            # --- [Clear Stage] clear the accumlation registers in tensor core ---
            uops.append(MicroOp("VALU_VSET", UnitType.VALU, latency=latency.VALU_VSET))
            uops.append(MicroOp("VALU_VMV_0", UnitType.VALU, latency=latency.VALU_VMV, dst_regs=[20, 21]))

            # --- [Accumulation Stage] ---
            for k_start in range(0, K_total, tensor.tensor_row):
                
                # 1. 發送 Load 需求給 LSU (載入 Input 和 Weight 到 VRF)
                # 這裡假設每次 Load 一個 Tile 只需要 1 個 Vector Register
                # 為了避免 WAW/RAW Hazard 卡住 pipeline，FSM 可以簡單地做 Register 交替 (Software Pipelining)
                if csr.Enable_Double_Buffer:
                    offset = (k_start // tensor.tensor_row) % 2
                else:
                    offset = 0
                
                reg_a = csr.MatA_reg_base + offset
                reg_b = csr.MatB_reg_base + offset

                
                uops.append(MicroOp(f"LSU_LOAD_A_TILE", UnitType.LSU, latency=10, dst_regs=[reg_a]))
                uops.append(MicroOp(f"LSU_LOAD_B_TILE", UnitType.LSU, latency=10, dst_regs=[reg_b]))
                
                # 2. 發送 Compute 需求給 Tensor Core
                # Tensor Core 吃 VRF 的資料，但把 Psum 留在自己肚子裡 (不寫 dst_regs)
                uops.append(MicroOp("CIM_MAC_TILE", UnitType.CIM, latency=tensor.tensor_row, src_regs=[reg_a, reg_b], dst_regs=[]))

            # --- [Store Stage] K 維度算完，把 Psum 從 Tensor Core 吐到 SRAM ---
            # 這裡我們用一個特殊的 uOP 把資料從 CIM 直接推給 LSU 存起來
            # 或者先吐回 VRF (例如 v20)，再從 VRF 存出去
            uops.append(MicroOp("CIM_OUT", UnitType.CIM, latency=2, src_regs=[], dst_regs=[20, 21]))
            uops.append(MicroOp("LSU_STORE_C_TILE", UnitType.LSU, latency=10, src_regs=[20, 21]))
            
    return uops


# --- 測試區 (替換原有的 run_simulation 內容) ---
def run_simulation():
    latencySet = LatencySet()
    csr = CSRConfig(MatA_reg_base=0, MatB_reg_base=4, MatC_reg_base=8, Enable_Double_Buffer=True)
    tensorHW = TensorConfig(input_row=64, tensor_col=16, tensor_row=64)
    sim = ADHD_VPU()

    # 假設 CPU 在迴圈裡，先 Load 資料，然後發出一個 FlashAttention Macro
    # 我們這裡專注發射 Macro_FlashAttention
    # macro_flash = MacroOp("MACRO_FLASH_ATTN", macro_flash_attn_template, {"reg_q": 0, "reg_k": 4, "reg_v": 8, "reg_score": 12, "reg_out": 16})
    # macro_gemm  = MacroOp("MACRO_GEMM", macro_gemm_template, {"k_tiles": 4, "reg_a": 0, "reg_b": 4, "reg_c": 8})
    macro_gemm_csr  = MacroOp("MACRO_GEMM_CSR", macro_gemm_template_csr, {"tensor": tensorHW, "latency": latencySet, "M_total": 64, "N_total": 16, "K_total": 64})
    
    print("--- Starting ADHD VPU Simulation (FlashAttention Fusion) ---")
    # sim.fetch_macro([macro_flash])
    # sim.fetch_macro([macro_gemm])
    sim.fetch_macro([macro_gemm_csr])
    
    while not sim.is_idle():
        sim.tick()
        # if sim.global_cycle > 1000: break
        
    sim.print_report()

if __name__ == "__main__":
    run_simulation()

# def run_simulation():
#     sim = ADHD_VPU()
    
#     # 模擬 CPU 發送一個 Macro 指令：Softmax 處理 4 個 Vector
#     # 實際上這在傳統 CPU 需要發送 4 * 3 = 12 條指令
#     # 這裡只發送 1 條
#     softmax_macro = MacroOp("MACRO_SOFTMAX_4vec", macro_softmax_template, {"v_start": 0, "length": 4})
    
#     # 再加一個 GEMM 測試 CIM 與 LSU 平行度
#     # 假設 GEMM 計算 v0-v3 且不需要 LSU (資料已在 SRAM), 只需要很久的計算
#     def macro_gemm_template():
#         return [MicroOp("GEMM_4x4", UnitType.CIM, latency=20, src_regs=[0,1], dst_regs=[2])]
    
#     gemm_macro = MacroOp("MACRO_GEMM", macro_gemm_template)

#     print("--- Starting Simulation ---")
#     sim.fetch_macro([softmax_macro, gemm_macro])
    
#     while not sim.is_idle():
#         sim.tick()
#         # 簡單防止無窮迴圈
#         if sim.global_cycle > 1000: break
        
#     sim.print_report()

# if __name__ == "__main__":
#     run_simulation()







from dataclasses import dataclass
from typing import List

# 模擬之前定義的結構
class UnitType: LSU = 1; CIM = 2; VALU = 3
class MicroOp: 
    def __init__(self, name, unit_type, latency, src_regs=None, dst_regs=None): pass

@dataclass
class TensorConfig:
    """硬體的實體規格 (在晶片 Tape-out 後就寫死，不可改變)"""
    phys_M: int = 16  # Tensor Core 的實體 Row 數量
    phys_N: int = 16  # Tensor Core 的實體 Col 數量
    # phys_K 由硬體 datapath (512-bit DLEN) 決定，每個 cycle 吞吐 16 Bytes

@dataclass
class CSRConfig:
    """軟體可程式化的邏輯 Tiling 規格 (由 Compiler 動態設定)"""
    M_tile: int = 64
    N_tile: int = 64
    K_tile: int = 32
    Enable_Double_Buffer: bool = True
    MatA_reg_base: int = 0
    MatB_reg_base: int = 4
    MatC_reg_base: int = 20  # 如果需要寫回 VRF 才用到

@dataclass
class LatencySet:
    LSU_LOAD: int = 10
    LSU_STORE: int = 10
    CIM_OUT: int = 2

def macro_gemm_template_csr(csr: CSRConfig, tensor: TensorConfig, latency: LatencySet, M_total=0, N_total=0, K_total=0):
    """
    【新版】支援時間摺疊 (Temporal Folding) 的 GEMM FSM 展開器
    """
    uops = []

    # =====================================================================
    # 外層大迴圈：軟體 / DMA 層級的 Tiling (走 64-bit AXI 慢速通道)
    # =====================================================================
    for m_start in range(0, M_total, csr.M_tile):
        for n_start in range(0, N_total, csr.N_tile):
            
            # --- [Clear Stage] 清空 Tensor Core 底下的 16KB L0 Buffer ---
            # 絕對不是清空 VRF！我們用一個專屬的硬體控制訊號來 Clear SRAM
            uops.append(MicroOp("CIM_CLEAR_L0_BUFFER", UnitType.CIM, latency=1, src_regs=[], dst_regs=[]))

            # --- [K Dimension Accumulation Stage] ---
            for k_start in range(0, K_total, csr.K_tile):
                
                # 1. 決定 VRF 的 Ping-Pong 位置
                if csr.Enable_Double_Buffer:
                    offset = ((k_start // csr.K_tile) % 2) * 2 # 假設 64x32 佔用 2 個 VREG
                else:
                    offset = 0
                
                reg_a = csr.MatA_reg_base + offset
                reg_b = csr.MatB_reg_base + offset

                # 2. [LSU] 從 L1 SRAM 載入 64x32 和 32x64 的資料到 VRF
                # (VLEN=8192, 1個 VREG 是 1KB。64x32 = 2KB，所以佔用 2 個 VREG)
                uops.append(MicroOp(f"LSU_LOAD_A_TILE_2KB", UnitType.LSU, latency=latency.LSU_LOAD, dst_regs=[reg_a, reg_a+1]))
                uops.append(MicroOp(f"LSU_LOAD_B_TILE_2KB", UnitType.LSU, latency=latency.LSU_LOAD, dst_regs=[reg_b, reg_b+1]))
                
                # 3. [CIM] 時間摺疊 (Temporal Folding) 核心邏輯！
                # FSM 在這裡將 64x64 的邏輯任務，切給 16x16 的實體陣列執行
                for m_sub in range(0, csr.M_tile, tensor.phys_M):
                    for n_sub in range(0, csr.N_tile, tensor.phys_N):
                        
                        # 發出給 16x16 陣列的微指令。
                        # Latency = K_tile (32 cycles)，因為陣列每個 cycle 會吞吐 K 維度的 1 步 (16 Bytes)
                        uops.append(MicroOp(
                            name=f"CIM_MAC_16x16_sub_{m_sub}_{n_sub}", 
                            unit_type=UnitType.CIM, 
                            latency=csr.K_tile, 
                            src_regs=[reg_a, reg_a+1, reg_b, reg_b+1], # 從 VRF 讀取資料
                            dst_regs=[]  # Psum 像瀑布一樣掉進底下的 16KB L0 Buffer，不寫回 VRF！
                        ))

            # --- [Store Stage] K 維度完全算完，16KB 的 L0 Buffer 已經有了最終的 FP32 結果 ---
            # 針對 FlashAttention，這 16KB 的 FP32 通常會直接留在 L0 Buffer 裡繼續做 Softmax。
            # 但如果是純 GEMM，我們會把它 Quantize 成 INT8，然後存回 VRF 或 SRAM。
            uops.append(MicroOp("CIM_QUANTIZE_AND_OUT_TO_VRF", UnitType.CIM, latency=latency.CIM_OUT, src_regs=[], dst_regs=[csr.MatC_reg_base]))
            uops.append(MicroOp("LSU_STORE_C_TILE", UnitType.LSU, latency=latency.LSU_STORE, src_regs=[csr.MatC_reg_base]))
            
    return uops