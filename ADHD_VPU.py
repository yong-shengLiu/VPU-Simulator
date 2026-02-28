from dataclasses import dataclass, field
from enum import Enum, auto
from collections import deque
from typing import List, Optional, Dict

# --- 1. Hardware Spec. ---
NUM_VREGS = 32          # v0 ~ v31
LANE = 4
VLEN = 8192             # the vector length may change depending on sequence length (1024 byte)
AXI_WIDTH = 64          # 64-bit AXI bus width (8 bytes)
LSU_QUEUE_DEPTH = 16    # LSU uop queue
VALU_QUEUE_DEPTH = 16   # VALU uop queue
CIM_QUEUE_DEPTH = 32    # CIM uop queue (Tensor Core) 

# 虛擬暫存器，用來讓 Scoreboard 追蹤沒有寫入 VRF、只在 L0 Buffer 傳遞的資料依賴
VIRTUAL_L0_BUFFER_ID = 63 

class UnitType(Enum):
    LSU  = auto()   # Load/Store
    VALU = auto()   # Vector Arithmetic
    CIM  = auto()   # Tensor/Matrix Core

@dataclass
class MicroOp:
    """ Micro-Operation """
    name: str
    unit_type: UnitType
    latency: int       
    src_regs: List[int] = field(default_factory=list) 
    dst_regs: List[int] = field(default_factory=list) 
    
    # --- 解決 Deadlock 新增的 Ticket 追蹤 ---
    wait_for_writes: dict = field(default_factory=dict)
    wait_for_reads: dict = field(default_factory=dict)
    wait_for_writes_waw: dict = field(default_factory=dict)

    # --- LSU 專用的外部記憶體語意 (Memory Semantics) ---
    mem_addr: int = 0               # 該 Tile 在 SRAM/DRAM 中的起始位址
    mem_stride: int = 0             # 2D 存取時的換列跨步 (Bytes)
    is_gather_scatter: bool = False # True: block stride, False: unit stride
    index_reg: int = -1             # Block stride
    
    def __repr__(self):
        return f"[{self.unit_type.name}] {self.name} (Lat:{self.latency})"

@dataclass
class MacroOp:
    """ Macro-Operation """
    name: str
    # this macro op will be expanded by calling expansion_func with args
    expansion_func: callable 
    args: dict = field(default_factory=dict)

class ActivationType(Enum):
    NONE = auto()
    GELU = auto()
    RELU = auto()
    SILU = auto()

@dataclass
class CSRConfig:
    """
    ================================================================================
    VPU Hardware/Software Interface (CSR Mapping Specification)
    Allocated in RISC-V U-Mode Custom Read/Write Space (0x801 ~ 0x8FF)
    ================================================================================
    【外部記憶體配置區 (External Memory Subsystem)】
    [ 0x801 ] VPU_MEM_BASE_A (64-bit) | 矩陣 A 外部記憶體起始位址
    [ 0x802 ] VPU_MEM_BASE_B (64-bit) | 矩陣 B 外部記憶體起始位址
    [ 0x803 ] VPU_MEM_BASE_C (64-bit) | 矩陣 C 外部記憶體起始位址
    [ 0x804 ] VPU_MEM_BASE_D (64-bit) | 矩陣 D 外部記憶體起始位址
    
    [ 0x805 ] VPU_MEM_STRIDE (外部記憶體 2D 跨步 / Leading Dimensions)
      - Bits [15:0]  : Mem_Stride_A         (16-bit) | A 換 Row 跳躍的 Bytes
      - Bits [31:16] : Mem_Stride_B         (16-bit) | B 換 Row 跳躍的 Bytes
      - Bits [47:32] : Mem_Stride_C         (16-bit) | C 換 Row 跳躍的 Bytes
      - Bits [63:48] : Mem_Stride_D         (16-bit) | D 換 Row 跳躍的 Bytes

    [ 0x806 ] VPU_MEM_ACCESS_CFG (記憶體存取模式 - Scatter/Gather 控制) ★ NEW ★
      - Bit  [0]     : Is_Gather_A          (1-bit)  | 1: A 啟用間接定址 (如 Embedding)
      - Bits [5:1]   : Index_Reg_A          (5-bit)  | 存放 A 的 Index offset 的 VREG ID
      - Bit  [8]     : Is_Gather_B          (1-bit)  | 1: B 啟用間接定址 (如 Sparse Attention)
      - Bits [13:9]  : Index_Reg_B          (5-bit)  | 存放 B 的 Index offset 的 VREG ID
      - Bit  [16]    : Is_Scatter_C         (1-bit)  | 1: C 啟用間接寫回
      - Bits [21:17] : Index_Reg_C          (5-bit)  | 存放 C 的 Index offset 的 VREG ID
      - Bit  [24]    : Is_Gather_D          (1-bit)  | 1: D 啟用間接定址
      - Bits [29:25] : Index_Reg_D          (5-bit)  | 存放 D 的 Index offset 的 VREG ID
    --------------------------------------------------------------------------------
    
    [ 0x807 ] VPU_REG_BASE_CFG (暫存器基址與控制旗標)
      - Bits [4:0]   : MatA_reg_base        (5-bit) | 矩陣 A 基址 (通常為 Q / Main)
      - Bits [9:5]   : MatB_reg_base        (5-bit) | 矩陣 B 基址 (通常為 K / Residual)
      - Bits [14:10] : MatC_reg_base        (5-bit) | 矩陣 C 基址 (通常為 Output / O_global)
      - Bits [19:15] : MatD_reg_base        (5-bit) | 矩陣 D 基址 (FlashAttn 的 V)
      - Bits [24:20] : MatE_reg_base        (5-bit) | 矩陣 E 基址 (FlashAttn 的 P)
      - Bits [29:25] : Temp_reg_base        (5-bit) | 中繼暫存器基址 (LayerNorm Mean/Var 等)
      - Bits [30]    : Enable_Double_Buffer (1-bit) | 1: 開啟硬體自動 Ping-Pong, 0: 關閉
      - Bits [33:31] : Act_Type             (3-bit) | 激勵函數 (0:NONE, 1:GELU, 2:RELU, 3:SILU)
      - Bits [63:34] : Reserved             (30-bit)| 保留擴充

    [ 0x808 ] VPU_STRIDE_CFG (硬體暫存器跨步設定)
      - Bits [4:0]   : VREG_stride_A        (5-bit) | Tile A 佔用的 VREG 數量
      - Bits [9:5]   : VREG_stride_B        (5-bit) | Tile B 佔用的 VREG 數量
      - Bits [14:10] : VREG_stride_C        (5-bit) | Tile C 佔用的 VREG 數量
      - Bits [19:15] : VREG_stride_D        (5-bit) | Tile D 佔用的 VREG 數量
      - Bits [24:20] : VREG_stride_E        (5-bit) | Tile E 佔用的 VREG 數量
      - Bits [29:25] : VREG_stride_O        (5-bit) | Tile O_global 佔用的 VREG 數量
      - Bits [63:30] : Reserved             (34-bit)| 保留擴充

    [ 0x809 ] VPU_TILE_CFG (硬體 Tiling 邊界維度)
      - Bits [15:0]  : M_tile               (16-bit)| Tile 的 M 維度大小
      - Bits [31:16] : N_tile               (16-bit)| Tile 的 N 維度大小
      - Bits [47:32] : K_tile               (16-bit)| Tile 的 K 維度大小
      - Bits [63:48] : Reserved             (16-bit)| 保留擴充

    [ 0x80A ] VPU_MACRO_TRIGGER (執行觸發與動態巨集參數)
      *** 寫入此 CSR 即代表 CPU 發射 Macro-OP，VPU Frontend 將開始解碼 ***
      - Bits [7:0]   : Macro_Opcode         (8-bit) | 0x1: GEMM, 0x2: GEMM_GELU, 0x3: FLASH_ATTN, 0x4: RES_LN
      - Bits [23:8]  : Dim_1 / Seq_Len      (16-bit)| M_total 或 Sequence Length
      - Bits [39:24] : Dim_2                (16-bit)| N_total 或 Hidden_Dim
      - Bits [55:40] : Dim_3                (16-bit)| K_total
      - Bits [63:56] : Sub_Op_Flags         (8-bit) | 附加控制旗標 (如指定 Projection 種類等)
    ================================================================================
    """

    # --- 1. External Memory Pointers & Strides ---
    Mem_Base_A: int = 0x8000_0000  
    Mem_Base_B: int = 0x8001_0000
    Mem_Base_C: int = 0x8002_0000
    Mem_Base_D: int = 0x8003_0000
    
    Mem_Stride_A: int = 64  
    Mem_Stride_B: int = 64
    Mem_Stride_C: int = 64
    Mem_Stride_D: int = 64

    # --- 2. Memory Access Modes (Scatter/Gather) ★ NEW ★ ---
    Is_Gather_A: bool = False
    Index_Reg_A: int = 0
    Is_Gather_B: bool = False
    Index_Reg_B: int = 0
    Is_Scatter_C: bool = False
    Index_Reg_C: int = 0
    Is_Gather_D: bool = False
    Index_Reg_D: int = 0

    # --- 3. Base Register Pointers (VRF Allocation) ---
    MatA_reg_base: int = 0
    MatB_reg_base: int = 4
    MatC_reg_base: int = 20
    MatD_reg_base: int = 8    # FlashAttn 的 V
    MatE_reg_base: int = 12   # FlashAttn 的 P
    Temp_reg_base: int = 28   # 專門給 LayerNorm 放 Mean/Var，或 FlashAttn 放 Quantize 暫存區

    # --- 4. Hardware / Memory Strides ---
    VREG_stride_A: int = 2 
    VREG_stride_B: int = 2
    VREG_stride_C: int = 4
    VREG_stride_D: int = 2    
    VREG_stride_E: int = 4    
    VREG_stride_O: int = 16  

    # --- 5. Tiling Dimensions (Loop Bounds) ---
    M_tile: int = 64
    N_tile: int = 64
    K_tile: int = 32

    # --- 6. Operation Flags ---
    Enable_Double_Buffer: bool = True
    Act_Type: ActivationType = ActivationType.NONE

@dataclass
class TensorConfig:
    """ Tensor Hardware Configuration for GEMM """
    phys_M: int = 16
    phys_N: int = 16

@dataclass
class LatencySet:
    """ Latency configuration for different micro-ops, can be extended as needed """
    Load_One_Vector: int = VLEN // AXI_WIDTH + 1  # one more cycle for vreg transition
    Store_One_Vector: int = VLEN // AXI_WIDTH + 1 # one more cycle for vreg transition
    VALU_VSET: int = 1
    VALU_VMV: int = VLEN // LANE // AXI_WIDTH    # AXI_WIDTH is same with VALU width (Assume)
    VALU_VADD: int = VLEN // LANE // AXI_WIDTH   # AXI_WIDTH is same with VALU width (Assume)
    VALU_VEXP: int = VLEN // LANE // AXI_WIDTH   # AXI_WIDTH is same with VALU width (Assume)
    VALU_VGELU: int = VLEN // LANE // AXI_WIDTH  # AXI_WIDTH is same with VALU width (Assume)


# --- 2. Frontend Macro to micor Expander ---
class MacroExpander:
    """
    Will take a MacroOp and expand it into a list of MicroOps based on the provided expansion function.
    """
    def expand(self, macro_op: MacroOp) -> List[MicroOp]:
        return macro_op.expansion_func(**macro_op.args)

class Scoreboard:
    def __init__(self):
        MAX_REGS = 64 
        self.issued_writes = [0] * MAX_REGS
        self.issued_reads = [0] * MAX_REGS
        self.completed_writes = [0] * MAX_REGS
        self.completed_reads = [0] * MAX_REGS

    def allocate(self, uop: MicroOp):
        uop.wait_for_writes = {r: self.issued_writes[r] for r in set(uop.src_regs)}
        uop.wait_for_reads = {r: self.issued_reads[r] for r in set(uop.dst_regs)}
        uop.wait_for_writes_waw = {r: self.issued_writes[r] for r in set(uop.dst_regs)}

        for r in set(uop.src_regs): self.issued_reads[r] += 1
        for r in set(uop.dst_regs): self.issued_writes[r] += 1

    def can_execute(self, uop: MicroOp) -> bool:
        for r, target in uop.wait_for_writes.items():
            if self.completed_writes[r] < target: return False
        for r, target in uop.wait_for_reads.items():
            if self.completed_reads[r] < target: return False
        for r, target in uop.wait_for_writes_waw.items():
            if self.completed_writes[r] < target: return False
        return True

    def release(self, uop: MicroOp):
        for r in set(uop.src_regs): self.completed_reads[r] += 1
        for r in set(uop.dst_regs): self.completed_writes[r] += 1

# --- 3. Backend execute uop Unit ---
class ExecutionUnit:
    def __init__(self, name, scoreboard: Scoreboard):
        self.name = name
        self.scoreboard = scoreboard
        self.current_uop: Optional[MicroOp] = None
        self.remaining_cycles = 0
        self.busy = False
        self.total_active_cycles = 0
        self.stall_cycles = 0  # 紀錄這個 Unit 在 Queue 門口等資料的週期

    def issue(self, uop: MicroOp):
        self.current_uop = uop
        self.remaining_cycles = uop.latency
        self.busy = True

    def tick(self):
        if self.busy:
            self.total_active_cycles += 1
            self.remaining_cycles -= 1
            if self.remaining_cycles <= 0:
                self.scoreboard.release(self.current_uop)
                self.busy = False
                self.current_uop = None

class DecoupledQueue:
    def __init__(self, depth):
        self.queue = deque()
        self.depth = depth
    def push(self, uop: MicroOp) -> bool:
        if len(self.queue) < self.depth:
            self.queue.append(uop)
            return True
        return False
    def pop(self) -> Optional[MicroOp]:
        return self.queue.popleft() if self.queue else None
    def is_full(self): return len(self.queue) >= self.depth
    def __len__(self): return len(self.queue)

# --- 4. The abstract VPU ---
class ADHD_VPU:
    def __init__(self):
        self.global_cycle = 0
        self.scoreboard = Scoreboard()
        self.expander   = MacroExpander()
        
        self.lsu_queue  = DecoupledQueue(LSU_QUEUE_DEPTH)
        self.valu_queue = DecoupledQueue(VALU_QUEUE_DEPTH)
        self.cim_queue  = DecoupledQueue(CIM_QUEUE_DEPTH)
        
        self.lsu_unit  = ExecutionUnit("LSU", self.scoreboard)
        self.valu_unit = ExecutionUnit("VALU", self.scoreboard)
        self.cim_unit  = ExecutionUnit("CIM", self.scoreboard)
        
        self.macro_instr_buffer = deque() 
        self.micro_op_buffer = deque()    
        
        self.total_macro_fetched = 0
        self.total_micro_generated = 0
        self.stall_queue_full_cycles = 0     

    def fetch_macro(self, macro_ops: List[MacroOp]):
        for op in macro_ops:
            self.macro_instr_buffer.append(op)
            self.total_macro_fetched += 1

    def tick(self):
        self.global_cycle += 1
        
        def try_issue_from_queue(unit: ExecutionUnit, queue: DecoupledQueue):
            if not unit.busy and len(queue) > 0:
                uop = queue.queue[0] 
                if self.scoreboard.can_execute(uop):
                    queue.pop()
                    unit.issue(uop)
                else:
                    unit.stall_cycles += 1 # 紀錄是誰在等資料

        try_issue_from_queue(self.lsu_unit, self.lsu_queue)
        self.lsu_unit.tick()
        try_issue_from_queue(self.valu_unit, self.valu_queue)
        self.valu_unit.tick()
        try_issue_from_queue(self.cim_unit, self.cim_queue)
        self.cim_unit.tick()

        if not self.micro_op_buffer and self.macro_instr_buffer:
            current_macro = self.macro_instr_buffer.popleft()
            uops = self.expander.expand(current_macro)
            self.micro_op_buffer.extend(uops)
            self.total_micro_generated += len(uops)

        if self.micro_op_buffer:
            uop = self.micro_op_buffer[0] 
            target_queue = None
            if uop.unit_type == UnitType.LSU: target_queue = self.lsu_queue
            elif uop.unit_type == UnitType.VALU: target_queue = self.valu_queue
            elif uop.unit_type == UnitType.CIM: target_queue = self.cim_queue
            
            if target_queue.is_full():
                self.stall_queue_full_cycles += 1
                return 
            
            self.micro_op_buffer.popleft() 
            target_queue.push(uop)
            self.scoreboard.allocate(uop) 

    def is_idle(self):
        return (not self.macro_instr_buffer and not self.micro_op_buffer and 
                not self.lsu_unit.busy and not self.valu_unit.busy and not self.cim_unit.busy and
                len(self.lsu_queue) == 0 and len(self.valu_queue) == 0 and len(self.cim_queue) == 0)

    def print_report(self):
        print("="*50)
        print(f"Simulation Report (Total Cycles: {self.global_cycle})")
        print("="*50)
        print(f"[Instruction Fetch Reduction]")
        print(f"  - CPU Macro Ops Fetched : {self.total_macro_fetched}")
        print(f"  - VPU Micro Ops Executed: {self.total_micro_generated}")
        print(f"  - Expansion Ratio       : 1:{self.total_micro_generated/max(1, self.total_macro_fetched):.1f}")
        print(f"\n[Frontend Stall Analysis]")
        print(f"  - Queue Full Stalls : {self.stall_queue_full_cycles} cycles ({(self.stall_queue_full_cycles/self.global_cycle):.1%})")
        # print(f"  - Backend Data Hazard Stalls : {self.backend_hazard_stall_cycles} cycles ({(self.backend_hazard_stall_cycles/self.global_cycle):.1%})")
        print(f"\n[Backend Overlap & Hazard Analysis (The True Decoupling)]")
        print(f"  - LSU Active  : {self.lsu_unit.total_active_cycles/self.global_cycle:5.1%} | Wait Data: {self.lsu_unit.stall_cycles/self.global_cycle:5.1%}")
        print(f"  - VALU Active : {self.valu_unit.total_active_cycles/self.global_cycle:5.1%} | Wait Data: {self.valu_unit.stall_cycles/self.global_cycle:5.1%}")
        print(f"  - CIM Active  : {self.cim_unit.total_active_cycles/self.global_cycle:5.1%} | Wait Data: {self.cim_unit.stall_cycles/self.global_cycle:5.1%}")
        print("="*50)


# --- 5. Macro Templates (Bug Fixed) ---
def get_actual_vreg(base_reg, sub_idx, tile_size, stride):
    """安全計算 VREG offset，避免越界存取污染 Pong Buffer"""
    elements_per_vreg = max(1, tile_size // stride)
    return base_reg + (sub_idx // elements_per_vreg)

def macro_gemm_template(csr: CSRConfig, tensor: TensorConfig, latency:LatencySet, M_total=0, N_total=0, K_total=0):
    uops = []
    c_regs = [csr.MatC_reg_base + i for i in range(csr.VREG_stride_C)]

    for m_start in range(0, M_total, csr.M_tile):
        for n_start in range(0, N_total, csr.N_tile):
            uops.append(MicroOp("CIM_CLEAR_PSUM_BUFFER", UnitType.CIM, latency=1, src_regs=[], dst_regs=[]))
            current_m_tile = min(csr.M_tile, M_total - m_start)
            current_n_tile = min(csr.N_tile, N_total - n_start)

            for k_start in range(0, K_total, csr.K_tile):
                offset_a = ((k_start // csr.K_tile) % 2) * csr.VREG_stride_A if csr.Enable_Double_Buffer else 0
                offset_b = ((k_start // csr.K_tile) % 2) * csr.VREG_stride_B if csr.Enable_Double_Buffer else 0
                
                reg_a = csr.MatA_reg_base + offset_a
                reg_b = csr.MatB_reg_base + offset_b
                
                a_regs = [reg_a + i for i in range(csr.VREG_stride_A)]
                b_regs = [reg_b + i for i in range(csr.VREG_stride_B)]

                uops.append(MicroOp("LSU_LOAD_A", UnitType.LSU, latency=latency.Load_One_Vector*csr.VREG_stride_A, dst_regs=a_regs))
                uops.append(MicroOp("LSU_LOAD_B", UnitType.LSU, latency=latency.Load_One_Vector*csr.VREG_stride_B, dst_regs=b_regs))
                
                for m_sub in range(0, current_m_tile, tensor.phys_M):
                    for n_sub in range(0, current_n_tile, tensor.phys_N):
                        # 【修正】精確鎖定 VREG，不越界
                        actual_reg_a = get_actual_vreg(reg_a, m_sub, csr.M_tile, csr.VREG_stride_A)
                        actual_reg_b = get_actual_vreg(reg_b, n_sub, csr.N_tile, csr.VREG_stride_B)
                        
                        uops.append(MicroOp(
                            name=f"CIM_MAC_{m_sub}_{n_sub}", unit_type=UnitType.CIM, latency=csr.K_tile, 
                            src_regs=[actual_reg_a, actual_reg_b], dst_regs=[]
                        ))

            uops.append(MicroOp("CIM_QUANT_OUT", UnitType.CIM, latency=latency.Store_One_Vector*csr.VREG_stride_C, src_regs=[], dst_regs=c_regs))
            uops.append(MicroOp("LSU_STORE_C", UnitType.LSU, latency=latency.Store_One_Vector*csr.VREG_stride_C, src_regs=c_regs))
    return uops

def macro_gemm_gelu_template(csr: CSRConfig, tensor: TensorConfig, latency: LatencySet, M_total=0, N_total=0, K_total=0):
    uops = []
    c_regs = [csr.MatC_reg_base + i for i in range(csr.VREG_stride_C)]

    for m_start in range(0, M_total, csr.M_tile):
        for n_start in range(0, N_total, csr.N_tile):
            uops.append(MicroOp("CIM_CLEAR_L0_BUFFER", UnitType.CIM, latency=1, src_regs=[], dst_regs=[]))
            current_m_tile = min(csr.M_tile, M_total - m_start)
            current_n_tile = min(csr.N_tile, N_total - n_start)

            for k_start in range(0, K_total, csr.K_tile):
                offset_a = ((k_start // csr.K_tile) % 2) * csr.VREG_stride_A if csr.Enable_Double_Buffer else 0
                offset_b = ((k_start // csr.K_tile) % 2) * csr.VREG_stride_B if csr.Enable_Double_Buffer else 0
                
                reg_a = csr.MatA_reg_base + offset_a
                reg_b = csr.MatB_reg_base + offset_b
                
                a_regs = [reg_a + i for i in range(csr.VREG_stride_A)]
                b_regs = [reg_b + i for i in range(csr.VREG_stride_B)]

                uops.append(MicroOp("LSU_LOAD_A", UnitType.LSU, latency=latency.Load_One_Vector*csr.VREG_stride_A, dst_regs=a_regs))
                uops.append(MicroOp("LSU_LOAD_B", UnitType.LSU, latency=latency.Load_One_Vector*csr.VREG_stride_B, dst_regs=b_regs))
                
                for m_sub in range(0, current_m_tile, tensor.phys_M):
                    for n_sub in range(0, current_n_tile, tensor.phys_N):
                        actual_reg_a = get_actual_vreg(reg_a, m_sub, csr.M_tile, csr.VREG_stride_A)
                        actual_reg_b = get_actual_vreg(reg_b, n_sub, csr.N_tile, csr.VREG_stride_B)
                        
                        uops.append(MicroOp(
                            name=f"CIM_MAC_{m_sub}_{n_sub}", unit_type=UnitType.CIM, latency=csr.K_tile, 
                            src_regs=[actual_reg_a, actual_reg_b], dst_regs=[] 
                        ))

            uops.append(MicroOp("CIM_QUANT_OUT", UnitType.CIM, latency=latency.Store_One_Vector*csr.VREG_stride_C, src_regs=[], dst_regs=c_regs))
            act_name = csr.Act_Type.name if csr.Act_Type != ActivationType.NONE else "LINEAR"
            uops.append(MicroOp(f"VALU_{act_name}_LUT", UnitType.VALU, latency=latency.VALU_VGELU*csr.VREG_stride_C, src_regs=c_regs, dst_regs=c_regs))
            uops.append(MicroOp("LSU_STORE_C", UnitType.LSU, latency=latency.Store_One_Vector*csr.VREG_stride_C, src_regs=c_regs, dst_regs=[]))
    return uops

def macro_flash_attn_template(csr: CSRConfig, tensor: TensorConfig, latency: LatencySet, Seq_Len=512):
    uops = []
    
    reg_q = csr.MatA_reg_base
    reg_k = csr.MatB_reg_base
    reg_v = csr.MatD_reg_base
    reg_p = csr.MatE_reg_base
    reg_o_global = csr.MatC_reg_base
    
    q_regs = [reg_q + i for i in range(csr.VREG_stride_A)]
    k_regs = [reg_k + i for i in range(csr.VREG_stride_B)]
    v_regs = [reg_v + i for i in range(csr.VREG_stride_D)]
    p_regs = [reg_p + i for i in range(csr.VREG_stride_E)]
    o_global_regs = [reg_o_global + i for i in range(csr.VREG_stride_O)]
    quant_regs = [csr.Temp_reg_base + i for i in range(csr.VREG_stride_C)]

    for q_start in range(0, Seq_Len, csr.M_tile):
        uops.append(MicroOp("VALU_CLEAR_O_GLOBAL", UnitType.VALU, latency=1, src_regs=[], dst_regs=o_global_regs))
        uops.append(MicroOp(f"LSU_LOAD_Q", UnitType.LSU, latency=latency.Load_One_Vector*csr.VREG_stride_A, dst_regs=q_regs))

        for k_start in range(0, Seq_Len, csr.K_tile):
            # 【修正】引入 Virtual L0 Buffer ID，強制建立 CIM 與 VALU 之間的資料依賴
            uops.append(MicroOp("CIM_CLEAR_L0", UnitType.CIM, latency=1))
            uops.append(MicroOp(f"LSU_LOAD_K", UnitType.LSU, latency=latency.Load_One_Vector*csr.VREG_stride_B, dst_regs=k_regs))

            for m_sub in range(0, csr.M_tile, tensor.phys_M):
                for n_sub in range(0, csr.K_tile, tensor.phys_N):
                    actual_q = get_actual_vreg(reg_q, m_sub, csr.M_tile, csr.VREG_stride_A)
                    actual_k = get_actual_vreg(reg_k, n_sub, csr.K_tile, csr.VREG_stride_B)
                    uops.append(MicroOp(
                        name=f"CIM_QK_{m_sub}_{n_sub}", unit_type=UnitType.CIM, latency=csr.K_tile, 
                        src_regs=[actual_q, actual_k], dst_regs=[VIRTUAL_L0_BUFFER_ID] # 寫入虛擬 L0
                    ))

            uops.append(MicroOp("VALU_SOFTMAX_UPDATE", UnitType.VALU, latency=20, src_regs=[VIRTUAL_L0_BUFFER_ID], dst_regs=[]))
            uops.append(MicroOp("VALU_SOFTMAX_EXP", UnitType.VALU, latency=1024, src_regs=[VIRTUAL_L0_BUFFER_ID], dst_regs=p_regs))

            uops.append(MicroOp("CIM_CLEAR_L0", UnitType.CIM, latency=1))
            uops.append(MicroOp(f"LSU_LOAD_V", UnitType.LSU, latency=latency.Load_One_Vector*csr.VREG_stride_D, dst_regs=v_regs))

            for m_sub in range(0, csr.M_tile, tensor.phys_M):
                for n_sub in range(0, csr.K_tile, tensor.phys_N):
                    actual_p = get_actual_vreg(reg_p, m_sub, csr.M_tile, csr.VREG_stride_E)
                    actual_v = get_actual_vreg(reg_v, n_sub, csr.K_tile, csr.VREG_stride_D)
                    uops.append(MicroOp(
                        name=f"CIM_PV_{m_sub}_{n_sub}", unit_type=UnitType.CIM, latency=csr.K_tile, 
                        src_regs=[actual_p, actual_v], dst_regs=[VIRTUAL_L0_BUFFER_ID] # 寫入虛擬 L0
                    ))

            uops.append(MicroOp(
                name=f"VALU_GLOBAL_O", unit_type=UnitType.VALU, latency=64, 
                src_regs=o_global_regs + [VIRTUAL_L0_BUFFER_ID], dst_regs=o_global_regs
            ))

        uops.append(MicroOp("VALU_QUANT_O", UnitType.VALU, latency=20, src_regs=o_global_regs, dst_regs=quant_regs))
        uops.append(MicroOp(f"LSU_STORE_O", UnitType.LSU, latency=latency.Store_One_Vector*csr.VREG_stride_C, src_regs=quant_regs))

    return uops

def macro_residual_layernorm_template(csr: CSRConfig, latency: LatencySet, Seq_Len=0, Hidden_Dim=768):
    uops = []
    for seq_idx in range(0, Seq_Len, csr.M_tile):
        offset_c = ((seq_idx // csr.M_tile) % 2) * csr.VREG_stride_C if csr.Enable_Double_Buffer else 0
        
        reg_main = csr.MatA_reg_base + offset_c
        reg_residual = csr.MatB_reg_base + offset_c
        reg_out = csr.MatC_reg_base + offset_c
        
        main_regs = [reg_main + i for i in range(csr.VREG_stride_C)]
        res_regs = [reg_residual + i for i in range(csr.VREG_stride_C)]
        out_regs = [reg_out + i for i in range(csr.VREG_stride_C)]
        
        reg_mean = csr.Temp_reg_base
        reg_var  = csr.Temp_reg_base + 1
        
        uops.append(MicroOp(f"LSU_LOAD_MAIN", UnitType.LSU, latency=latency.Load_One_Vector*csr.VREG_stride_C, dst_regs=main_regs))
        uops.append(MicroOp(f"LSU_LOAD_RES", UnitType.LSU, latency=latency.Load_One_Vector*csr.VREG_stride_C, dst_regs=res_regs))
        uops.append(MicroOp(f"VALU_VADD_RES", UnitType.VALU, latency=int(latency.VALU_VADD*csr.VREG_stride_C), src_regs=main_regs + res_regs, dst_regs=out_regs))
        
        realistic_valu_lat = (csr.M_tile * Hidden_Dim) // (LANE * AXI_WIDTH) + 10 
        uops.append(MicroOp("VALU_LN_MEAN", UnitType.VALU, latency=realistic_valu_lat, src_regs=out_regs, dst_regs=[reg_mean]))
        uops.append(MicroOp("VALU_LN_VAR", UnitType.VALU, latency=realistic_valu_lat, src_regs=out_regs + [reg_mean], dst_regs=[reg_var]))
        uops.append(MicroOp("VALU_LN_RSQRT", UnitType.VALU, latency=20, src_regs=[reg_var], dst_regs=[reg_var]))
        uops.append(MicroOp("VALU_LN_NORM", UnitType.VALU, latency=realistic_valu_lat, src_regs=out_regs + [reg_mean, reg_var], dst_regs=out_regs))
        uops.append(MicroOp(f"LSU_STORE_LN", UnitType.LSU, latency=latency.Store_One_Vector*csr.VREG_stride_C, src_regs=out_regs, dst_regs=[]))
        
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

def build_vit_base_layer(sim: ADHD_VPU, csr: CSRConfig, tensorHW: TensorConfig, latencySet: LatencySet, seq_len: int = 197):
    """
    建構一層 Vision Transformer (ViT-Base) Layer
    ViT-Base spec: 
    - Patch size 16x16, Image 224x224 -> Seq_Len = (224/16)^2 + 1 (CLS token) = 197
    - Hidden_Dim (D) = 768, Heads = 12, MLP_Dim = 3072
    特點：與 BERT 高度相似，但 Sequence Length 較短，主要應用於計算機視覺 (CV)。
    """
    D = 768
    D_FFN = 3072
    
    print(f"\n--- Disptaching ViT Base Layer (Sequence Length: {seq_len}) ---")
    
    # 1. Q, K, V Projections (Vision Tokens)
    sim.fetch_macro([MacroOp("VIT_PROJ_Q", macro_gemm_template, {"csr":csr, "tensor": tensorHW, "latency": latencySet, "M_total": seq_len, "N_total": D, "K_total": D})])
    sim.fetch_macro([MacroOp("VIT_PROJ_K", macro_gemm_template, {"csr":csr, "tensor": tensorHW, "latency": latencySet, "M_total": seq_len, "N_total": D, "K_total": D})])
    sim.fetch_macro([MacroOp("VIT_PROJ_V", macro_gemm_template, {"csr":csr, "tensor": tensorHW, "latency": latencySet, "M_total": seq_len, "N_total": D, "K_total": D})])
    
    # 2. FlashAttention (Spatial Attention across patches)
    for h in range(12):
        sim.fetch_macro([MacroOp(f"VIT_FLASH_ATTN_H{h}", macro_flash_attn_template, {"csr":csr, "tensor": tensorHW, "latency": latencySet, "Seq_Len": seq_len})])
    
    # 3. Attention Output Projection
    sim.fetch_macro([MacroOp("VIT_ATTN_OUT", macro_gemm_template, {"csr":csr, "tensor": tensorHW, "latency": latencySet, "M_total": seq_len, "N_total": D, "K_total": D})])
    
    # 4. Residual Add + LayerNorm (ViT typically uses Pre-LN, but structurally it's the same fused ops)
    sim.fetch_macro([MacroOp("VIT_RES_LN_1", macro_residual_layernorm_template, {"csr":csr, "latency": latencySet, "Seq_Len": seq_len, "Hidden_Dim": D})])
    
    # 5. MLP Layer 1 (GEMM + GELU)
    sim.fetch_macro([MacroOp("VIT_MLP1_GELU", macro_gemm_gelu_template, {"csr":csr, "tensor": tensorHW, "latency": latencySet, "M_total": seq_len, "N_total": D_FFN, "K_total": D})])
    
    # 6. MLP Layer 2 (GEMM)
    sim.fetch_macro([MacroOp("VIT_MLP2", macro_gemm_template, {"csr":csr, "tensor": tensorHW, "latency": latencySet, "M_total": seq_len, "N_total": D, "K_total": D_FFN})])
    
    # 7. Residual Add + LayerNorm 2
    sim.fetch_macro([MacroOp("VIT_RES_LN_2", macro_residual_layernorm_template, {"csr":csr, "latency": latencySet, "Seq_Len": seq_len, "Hidden_Dim": D})])

def build_gpt2_prefill_layer(sim: ADHD_VPU, csr: CSRConfig, tensorHW: TensorConfig, latencySet: LatencySet, seq_len: int = 1024):
    """
    建構一層 GPT-2 Base Layer (Prefill Stage)
    GPT-2 Base spec: 
    - Hidden_Dim (D) = 768, Heads = 12, MLP_Dim = 3072
    - Context Window (Seq_Len) = 1024 (Prefill 階段，將使用長 Context 進行平行運算)
    特點：展示 Generative AI 在 Prompt 處理階段 (Prefill) 也是高度相依於相同的矩陣與注意力算子。
    """
    D = 768
    D_FFN = 3072
    
    print(f"\n--- Disptaching GPT-2 Prefill Layer (Context Length: {seq_len}) ---")
    
    # GPT-2 是 Pre-LayerNorm 架構，所以我們在 Attention 前先派發 LN (為了簡化，復用相同的 Template)
    sim.fetch_macro([MacroOp("GPT2_PRE_LN_1", macro_residual_layernorm_template, {"csr":csr, "latency": latencySet, "Seq_Len": seq_len, "Hidden_Dim": D})])

    # 1. Q, K, V Projections
    sim.fetch_macro([MacroOp("GPT2_PROJ_Q", macro_gemm_template, {"csr":csr, "tensor": tensorHW, "latency": latencySet, "M_total": seq_len, "N_total": D, "K_total": D})])
    sim.fetch_macro([MacroOp("GPT2_PROJ_K", macro_gemm_template, {"csr":csr, "tensor": tensorHW, "latency": latencySet, "M_total": seq_len, "N_total": D, "K_total": D})])
    sim.fetch_macro([MacroOp("GPT2_PROJ_V", macro_gemm_template, {"csr":csr, "tensor": tensorHW, "latency": latencySet, "M_total": seq_len, "N_total": D, "K_total": D})])
    
    # 2. FlashAttention (Causal Masking is implicitly handled by the uOP expansion logic inside the hardware)
    for h in range(12):
        sim.fetch_macro([MacroOp(f"GPT2_FLASH_ATTN_H{h}", macro_flash_attn_template, {"csr":csr, "tensor": tensorHW, "latency": latencySet, "Seq_Len": seq_len})])
    
    # 3. Attention Output
    sim.fetch_macro([MacroOp("GPT2_ATTN_OUT", macro_gemm_template, {"csr":csr, "tensor": tensorHW, "latency": latencySet, "M_total": seq_len, "N_total": D, "K_total": D})])
    
    # 4. Residual Add + Pre-LayerNorm 2
    sim.fetch_macro([MacroOp("GPT2_RES_PRE_LN_2", macro_residual_layernorm_template, {"csr":csr, "latency": latencySet, "Seq_Len": seq_len, "Hidden_Dim": D})])
    
    # 5. FFN Layer 1 (GEMM + GELU)
    sim.fetch_macro([MacroOp("GPT2_FFN1_GELU", macro_gemm_gelu_template, {"csr":csr, "tensor": tensorHW, "latency": latencySet, "M_total": seq_len, "N_total": D_FFN, "K_total": D})])
    
    # 6. FFN Layer 2
    sim.fetch_macro([MacroOp("GPT2_FFN2", macro_gemm_template, {"csr":csr, "tensor": tensorHW, "latency": latencySet, "M_total": seq_len, "N_total": D, "K_total": D_FFN})])
    
    # 7. Final Residual Add
    # (此處呼叫純 Add 或復用 macro_residual_layernorm_template 來代表運算量)
    sim.fetch_macro([MacroOp("GPT2_FINAL_RES", macro_residual_layernorm_template, {"csr":csr, "latency": latencySet, "Seq_Len": seq_len, "Hidden_Dim": D})])


def run_simulation():
    latencySet = LatencySet()
    csr = CSRConfig(MatA_reg_base=0, MatB_reg_base=4, MatC_reg_base=8, Enable_Double_Buffer=True)
    tensorHW = TensorConfig(phys_M=16, phys_N=16)
    sim = ADHD_VPU()

    model = "BERT Base" # "BERT Base", "ViT Base", "GPT-2 Base"

    # 呼叫巨集組裝
    if model == "BERT Base":
        target_seq_len = 512 # 設定可變的 Sequence Length
        build_bert_base_layer(sim, csr, tensorHW, latencySet, seq_len=target_seq_len)
    elif model == "ViT Base":
        build_vit_base_layer(sim, csr, tensorHW, latencySet)
    elif model == "GPT-2 Base":
        build_gpt2_prefill_layer(sim, csr, tensorHW, latencySet)
    
    print(f"--- Simulation Running {model} ... ---")
    while not sim.is_idle():
        sim.tick()
        
    sim.print_report()


if __name__ == "__main__":
    run_simulation()