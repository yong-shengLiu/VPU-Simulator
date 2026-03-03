import os
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import deque
from typing import List, Optional, Dict

# ==============================================================================
# 1. Hardware Specifications & Constants
# ==============================================================================
NUM_VREGS = 32          # Standard RISC-V Vector Register File (v0 ~ v31)
LANE = 4                # Number of parallel processing lanes in VALU
VLEN = 8192             # Vector length in bits (1024 bytes per vector register)
VLENB = VLEN / 8        # Vector length in bytes
AXI_WIDTH = 64          # 64-bit AXI bus width (8 bytes per transfer)
LSU_QUEUE_DEPTH = 16    # Decoupled queue depth for Load/Store Unit
VALU_QUEUE_DEPTH = 16   # Decoupled queue depth for Vector ALU
CIM_QUEUE_DEPTH = 32    # Decoupled queue depth for Compute-In-Memory (Tensor Core)

# Virtual Register ID used for Scoreboard tracking to model implicit data dependencies
# passing through the internal L0 Buffer (SRAM) without polluting the VRF.
VIRTUAL_L0_BUFFER_ID = 63 

# ★ NEW: 系統時脈設定 (假設 VPU 跑在 1 GHz)
CLOCK_FREQ_GHZ = 1.0

class UnitType(Enum):
    """ Hardware Execution Units """
    LSU  = auto()   # Load/Store Unit (Handles DMA, Scatter/Gather, 2D Strides)
    VALU = auto()   # Vector Arithmetic Logic Unit (Handles Non-linear, Add, Norm)
    CIM  = auto()   # Compute-In-Memory / Tensor Core (Handles GEMM MAC arrays)

@dataclass
class MicroOp:
    """ 
    Micro-Operation (uOP) Structure
    Representing the finest granularity of execution decoded by the Micro-sequencer.
    """
    name: str
    unit_type: UnitType
    latency: int       
    src_regs: List[int] = field(default_factory=list) 
    dst_regs: List[int] = field(default_factory=list) 
    
    # --- Ticket-based Scoreboard Tracking (Resolves Deadlocks natively) ---
    wait_for_writes: dict = field(default_factory=dict)     # RAW hazard tracking
    wait_for_reads: dict = field(default_factory=dict)      # WAR hazard tracking
    wait_for_writes_waw: dict = field(default_factory=dict) # WAW hazard tracking

    # --- 2D Address Generation Unit (AGU) / Memory Semantics ---
    mem_addr: int = 0               # Physical Base Address in SRAM/DRAM for this Tile
    mem_stride: int = 0             # Leading dimension (Bytes) for 2D Block memory access
    is_gather_scatter: bool = False # True: Indirect addressing (Sparse), False: Dense Block
    block_length: int = 0           # the number of contiguous elements in a block for scatter/gather operations
    
    def __repr__(self):
        return f"[{self.unit_type.name}] {self.name} (Lat:{self.latency}, Addr:{hex(self.mem_addr)})"

@dataclass
class MacroOp:
    """ 
    Macro-Operation (Instruction-Level Fusion)
    Dispatched by the scalar CPU to dramatically reduce Instruction Fetch bandwidth.
    """
    name: str
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
    VPU Hardware/Software Interface (CSR Mapping Specification) - [架構完全體]
    Allocated in RISC-V U-Mode Custom Read/Write Space (0x801 ~ 0x8FF)
    ================================================================================
    【外部記憶體配置區 (External Memory Subsystem)】
    [ 0x801 ] CSR_VPU_MEM_BASE_A (64-bit) | 矩陣 A 外部記憶體起始位址 (SRAM/DRAM)
    [ 0x802 ] CSR_VPU_MEM_BASE_B (64-bit) | 矩陣 B 外部記憶體起始位址
    [ 0x803 ] CSR_VPU_MEM_BASE_C (64-bit) | 矩陣 C 外部記憶體起始位址
    [ 0x804 ] CSR_VPU_MEM_BASE_D (64-bit) | 矩陣 D 外部記憶體起始位址
    
    [ 0x805 ] CSR_VPU_MEM_STRIDE (外部記憶體 2D 跨步 / Leading Dimensions)
      - Bits [15:0]  : Mem_Stride_A         (16-bit) | A 換 Row 跳躍的 Bytes
      - Bits [31:16] : Mem_Stride_B         (16-bit) | B 換 Row 跳躍的 Bytes
      - Bits [47:32] : Mem_Stride_C         (16-bit) | C 換 Row 跳躍的 Bytes
      - Bits [63:48] : Mem_Stride_D         (16-bit) | D 換 Row 跳躍的 Bytes

    [ 0x806 ] CSR_VPU_MEM_ACCESS_CFG (記憶體存取模式 - Scatter/Gather 控制)
      - Bit  [0]     : Is_Gather_A          (1-bit)  | A 啟用間接定址 (Load)
      - Bits [10:1]  : BLOCK_LEN_A          (10-bit) | A 小區段的連續元素個數
      - Bit  [11]    : Is_Gather_B          (1-bit)  | B 啟用間接定址 (Load)
      - Bits [21:12] : BLOCK_LEN_B          (10-bit) | B 小區段的連續元素個數
      - Bit  [22]    : Is_Scatter_C         (1-bit)  | C 啟用間接寫回 (Store)
      - Bits [32:23] : BLOCK_LEN_C          (10-bit) | C 小區段的連續元素個數
      - Bit  [33]    : Is_Gather_D          (1-bit)  | D 啟用間接定址 (Load)
      - Bits [43:34] : BLOCK_LEN_D          (10-bit) | D 小區段的連續元素個數
      - Bits [63:44] : Reserved             (20-bit) | 保留未來擴展

    --------------------------------------------------------------------------------
    【內部架構配置區 (Internal Architecture Config)】
    [ 0x807 ] CSR_VPU_REG_BASE_CFG (暫存器基址與控制旗標)
      - Bits [4:0]   : MatA_reg_base        (5-bit)  | 矩陣 A 基址 (Q / Main)
      - Bits [9:5]   : MatB_reg_base        (5-bit)  | 矩陣 B 基址 (K / Residual)
      - Bits [14:10] : MatC_reg_base        (5-bit)  | 矩陣 C 基址 (Output)
      - Bits [19:15] : MatD_reg_base        (5-bit)  | 矩陣 D 基址 (FlashAttn V)
      - Bits [24:20] : MatE_reg_base        (5-bit)  | 矩陣 E 基址 (FlashAttn P)
      - Bits [29:25] : Temp_reg_base        (5-bit)  | 中繼暫存器基址 (LayerNorm 等)
      - Bits [30]    : Enable_Double_Buffer (1-bit)  | 1: 開啟硬體自動 Ping-Pong
      - Bits [33:31] : Act_Type             (3-bit)  | 激勵函數種類 (0:NONE, 1:GELU, 2:RELU, 3:SILU)
      - Bits [63:34] : Reserved             (30-bit) | 保留未來擴展

    [ 0x808 ] CSR_VPU_STRIDE_CFG (硬體暫存器跨步設定 - 佔用 VREG 數)
      - Bits [4:0]   : VREG_stride_A        (5-bit)
      - Bits [9:5]   : VREG_stride_B        (5-bit)
      - Bits [14:10] : VREG_stride_C        (5-bit)
      - Bits [19:15] : VREG_stride_D        (5-bit)
      - Bits [24:20] : VREG_stride_E        (5-bit)
      - Bits [29:25] : VREG_stride_O        (5-bit)
      - Bits [63:30] : Reserved             (34-bit)

    [ 0x809 ] CSR_VPU_TILE_CFG (硬體 Tiling 邊界維度)
      - Bits [15:0]  : M_tile               (16-bit)
      - Bits [31:16] : N_tile               (16-bit)
      - Bits [47:32] : K_tile               (16-bit)
      - Bits [63:48] : Reserved             (16-bit)

    --------------------------------------------------------------------------------
    
    【執行觸發區 (Execution Trigger)】
    [ 0x80A ] CSR_VPU_MACRO_TRIGGER (執行觸發與動態巨集參數)
      *** 寫入此 CSR 即代表 CPU 發射 Macro-OP，VPU Frontend 將開始解碼 ***
      - Bits [7:0]   : Macro_Opcode         (8-bit)  | 0x1: GEMM, 0x2: GEMM_GELU, 0x3: FLASH_ATTN, 0x4: RES_LN
      - Bits [23:8]  : M_total / Seq_Len    (16-bit) | M_total 或 Sequence Length
      - Bits [39:24] : N_total              (16-bit) | N_total 或 Hidden_Dim
      - Bits [55:40] : K_total              (16-bit) | K_total
      - Bits [63:56] : Reserved             (8-bit)  | 保留未來擴展
    ================================================================================
    """

    # --- 1. Internal VRF Pointers ---
    MatA_reg_base: int = 0
    MatB_reg_base: int = 4
    MatC_reg_base: int = 20
    MatD_reg_base: int = 8    
    MatE_reg_base: int = 12   
    Temp_reg_base: int = 28   

    # --- 2. VRF Strides ---
    VREG_stride_A: int = 2 
    VREG_stride_B: int = 2
    VREG_stride_C: int = 4
    VREG_stride_D: int = 2    
    VREG_stride_E: int = 4    
    VREG_stride_O: int = 16   

    # --- 3. Tiling Dimensions ---
    M_tile: int = 64
    N_tile: int = 64
    K_tile: int = 32

    # --- 4. Operation Flags ---
    Enable_Double_Buffer: bool = True
    Act_Type: ActivationType = ActivationType.NONE

    # --- 5. External Memory Pointers & Strides ---
    Mem_Base_A: int = 0x0000_0000  
    Mem_Base_B: int = 0x0000_0000
    Mem_Base_C: int = 0x0000_0000
    Mem_Base_D: int = 0x0000_0000
    
    Mem_Stride_A: int = 64  
    Mem_Stride_B: int = 64
    Mem_Stride_C: int = 64
    Mem_Stride_D: int = 64

    # --- 6. Memory Access Modes (Scatter/Gather) ---
    Is_Gather_A: bool = False
    BLOCK_LEN_A: int = 0
    Is_Gather_B: bool = False
    BLOCK_LEN_B: int = 0
    Is_Scatter_C: bool = False
    BLOCK_LEN_C: int = 0
    Is_Gather_D: bool = False
    BLOCK_LEN_D: int = 0


    # --- Reserved for Future Extensions ---
    Macro_Op_Name: str = "GEMM"
    M_total: int = 0
    N_total: int = 0
    K_total: int = 0

@dataclass
class TensorConfig:
    """ Hardware Spatial Dimensions for the Tensor Core (CIM Array) """
    phys_M: int = 16
    phys_N: int = 16

@dataclass
class LatencySet:
    """ Cycle-accurate Latency Models for behavioral simulation """
    Load_One_Vector: int = VLEN // AXI_WIDTH + 1  
    Store_One_Vector: int = VLEN // AXI_WIDTH + 1 
    VALU_VSET: int = 1
    VALU_VMV: int = VLEN // LANE // AXI_WIDTH    
    VALU_VADD: int = VLEN // LANE // AXI_WIDTH   
    VALU_VEXP: int = VLEN // LANE // AXI_WIDTH   
    VALU_VGELU: int = VLEN // LANE // AXI_WIDTH  

# ==============================================================================
# 2. Behavioral Memory Allocator (Compiler Simulation)
# ==============================================================================
class MemoryManager:
    """
    Simulates a software-level memory allocator (e.g., TVM/MLIR memory pool).
    Provides actual memory addresses for the VPU's AGU to perform 2D accesses.
    Assuming INT8 datatype (1 byte per element) and 64-byte alignment.
    """
    def __init__(self, base_addr=0x8000_0000):
        self.start_addr = base_addr
        self.current_addr = base_addr

    def allocate(self, size_in_bytes: int) -> int:
        addr = self.current_addr
        # Align to 64 bytes (AXI Bus Width)
        self.current_addr += (size_in_bytes + 63) & ~63
        return addr

    def reset(self):
        self.current_addr = self.start_addr

# ==============================================================================
# 3. Decoupled Micro-Architecture Components
# ==============================================================================
class MacroExpander:
    """ Hardware FSM Micro-sequencer: Unrolls Macro-Ops into execution uOPs. """
    def expand(self, macro_op: MacroOp) -> List[MicroOp]:
        return macro_op.expansion_func(**macro_op.args)

class Scoreboard:
    """ 
    Ticket-based Decoupled Scoreboard
    Eliminates False-Dependencies and Self-Deadlocks using an issue/completion 
    ticket system, enabling true Out-of-Order overlapping between decoupled queues.
    """
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

class ExecutionUnit:
    """ Behavioral Model for an asynchronous Backend Execution Unit. """
    def __init__(self, name, scoreboard: Scoreboard):
        self.name = name
        self.scoreboard = scoreboard
        self.current_uop: Optional[MicroOp] = None
        self.remaining_cycles = 0
        self.busy = False
        self.total_active_cycles = 0
        self.stall_cycles = 0 
        self.total_bytes_transferred = 0

    def issue(self, uop: MicroOp):
        self.current_uop = uop
        self.remaining_cycles = uop.latency
        self.busy = True

    def tick(self):
        if self.busy:
            self.total_active_cycles += 1
            self.remaining_cycles -= 1
            
            # AXI_WIDTH 是 bits，除以 8 變成 Bytes
            if self.name == "LSU":
                self.total_bytes_transferred += (AXI_WIDTH // 8)

            
            if self.remaining_cycles <= 0:
                self.scoreboard.release(self.current_uop)
                self.busy = False
                self.current_uop = None

class DecoupledQueue:
    """ Hardware FIFO isolating the Frontend fetcher from Backend execution. """
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

class ADHD_VPU:
    """
    Autonomous Domain-Specific Heterogeneous Decoupled Processor (ADHD VPU).
    The main cycle-accurate simulator orchestrating the Decoupled Access-Execute paradigm.
    """
    def __init__(self, model_name="BERT_Base", trace_filename="vpu_csr_trace.txt", c_macro_header="vpu_macro_dispatch.h"):
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

        # --- CSR Trace Logger ---
        current_dir = os.path.dirname(os.path.abspath(__file__))
        log_dir = os.path.join(current_dir, "log")
        
        # 1. 安全機制：確保 log 資料夾存在，不存在就自動創建
        os.makedirs(log_dir, exist_ok=True)
        
        # 2. ★ 關鍵修正：將「完整的絕對路徑」存進 class instance 變數中
        self.trace_filepath = os.path.join(log_dir, trace_filename)
        self.c_filepath = os.path.join(log_dir, c_macro_header)
        
        # 3. 寫入標頭
        with open(self.trace_filepath, "w") as f:
            f.write("=========================================================\n")
            f.write(" ADHD VPU Firmware CSR Trace (Auto-Generated)\n")
            f.write("=========================================================\n\n")
        
        # ★ 寫入 C Code 標頭與 Function 開頭
        with open(self.c_filepath, "w") as f_c:
            f_c.write("// =========================================================\n")
            f_c.write("// ADHD VPU Firmware Dispatcher (Auto-Generated by Python)\n")
            f_c.write("// =========================================================\n")
            f_c.write("#include <stdint.h>\n\n")
            if model_name == "BERT_Base":
                f_c.write("static inline void dispatch_bert_base_macros() {\n")
            elif model_name == "ViT_Base":
                f_c.write("static inline void dispatch_vit_base_macros() {\n")
            elif model_name == "GPT2_Base":
                f_c.write("static inline void dispatch_gpt2_base_macros() {\n")

    def fetch_macro(self, macro_ops: List[MacroOp]):
        for op in macro_ops:
            self.macro_instr_buffer.append(op)
            self.total_macro_fetched += 1

            # 每發射一個 Macro OP，就產生對應的 CSR Trace
            self._log_csr_trace(op)
    
    def _log_csr_trace(self, op: MacroOp):
        """將 CSRConfig 與 Macro 參數打包成 64-bit Payload，並寫入 txt 檔"""
        csr = op.args["csr"]
        
        # write CSR in txt format for traceability and debugging
        with open(self.trace_filepath, "a") as f:
            f.write(f"# --- Dispatching Macro: {op.name} ({op.expansion_func.__name__}) ---\n")
            
            # [0x801 - 0x804] External Memory Base
            f.write(f"csrw 0x801, 0x{csr.Mem_Base_A:016X}  # CSR_VPU_MEM_BASE_A\n")
            f.write(f"csrw 0x802, 0x{csr.Mem_Base_B:016X}  # CSR_VPU_MEM_BASE_B\n")
            f.write(f"csrw 0x803, 0x{csr.Mem_Base_C:016X}  # CSR_VPU_MEM_BASE_C\n")
            f.write(f"csrw 0x804, 0x{csr.Mem_Base_D:016X}  # CSR_VPU_MEM_BASE_D\n")

            # [0x805] Memory Stride
            mem_stride_payload = (
                (csr.Mem_Stride_A & 0xFFFF) |
                ((csr.Mem_Stride_B & 0xFFFF) << 16) |
                ((csr.Mem_Stride_C & 0xFFFF) << 32) |
                ((csr.Mem_Stride_D & 0xFFFF) << 48)
            )
            f.write(f"csrw 0x805, 0x{mem_stride_payload:016X}  # CSR_VPU_MEM_STRIDE\n")

            # [0x806] Memory Access CFG (Scatter/Gather)
            mem_acc_payload = (
                (int(csr.Is_Gather_A) & 0x1) |
                ((csr.BLOCK_LEN_A & 0x3FF) << 1) |
                ((int(csr.Is_Gather_B) & 0x1) << 11) |
                ((csr.BLOCK_LEN_B & 0x3FF) << 12) |
                ((int(csr.Is_Scatter_C) & 0x1) << 22) |
                ((csr.BLOCK_LEN_C & 0x3FF) << 23) |
                ((int(csr.Is_Gather_D) & 0x1) << 33) |
                ((csr.BLOCK_LEN_D & 0x3FF) << 34)
            )
            f.write(f"csrw 0x806, 0x{mem_acc_payload:016X}  # CSR_VPU_MEM_ACCESS_CFG\n")

            # [0x807] Internal Reg Base CFG
            act_type_val = csr.Act_Type.value if hasattr(csr.Act_Type, 'value') else 0
            reg_base_payload = (
                (csr.MatA_reg_base & 0x1F) |
                ((csr.MatB_reg_base & 0x1F) << 5) |
                ((csr.MatC_reg_base & 0x1F) << 10) |
                ((csr.MatD_reg_base & 0x1F) << 15) |
                ((csr.MatE_reg_base & 0x1F) << 20) |
                ((csr.Temp_reg_base & 0x1F) << 25) |
                ((int(csr.Enable_Double_Buffer) & 0x1) << 30) |
                ((act_type_val & 0x7) << 31)
            )
            f.write(f"csrw 0x807, 0x{reg_base_payload:016X}  # CSR_VPU_REG_BASE_CFG\n")

            # [0x808] VREG Stride CFG
            reg_stride_payload = (
                (csr.VREG_stride_A & 0x1F) |
                ((csr.VREG_stride_B & 0x1F) << 5) |
                ((csr.VREG_stride_C & 0x1F) << 10) |
                ((csr.VREG_stride_D & 0x1F) << 15) |
                ((csr.VREG_stride_E & 0x1F) << 20) |
                ((csr.VREG_stride_O & 0x1F) << 25)
            )
            f.write(f"csrw 0x808, 0x{reg_stride_payload:016X}  # CSR_VPU_STRIDE_CFG\n")

            # [0x809] Hardware Tile CFG
            tile_payload = (
                (csr.M_tile & 0xFFFF) |
                ((csr.N_tile & 0xFFFF) << 16) |
                ((csr.K_tile & 0xFFFF) << 32)
            )
            f.write(f"csrw 0x809, 0x{tile_payload:016X}  # CSR_VPU_TILE_CFG\n")

            # [0x80A] Macro Trigger & Dynamic Dims
            # 解析 Opcode
            opcode_map = {
                "macro_gemm_template": 0x1,
                "macro_gemm_gelu_template": 0x2,
                "macro_flash_attn_template": 0x3,
                "macro_residual_layernorm_template": 0x4
            }
            func_name = op.expansion_func.__name__
            opcode = opcode_map.get(func_name, 0xFF)

            # 解析維度 (相容不同 template 的 args 命名)
            dim1 = op.args.get("M_total", op.args.get("Seq_Len", 0))
            dim2 = op.args.get("N_total", op.args.get("Hidden_Dim", 0))
            dim3 = op.args.get("K_total", 0)

            trigger_payload = (
                (opcode & 0xFF) |
                ((dim1 & 0xFFFF) << 8) |
                ((dim2 & 0xFFFF) << 24) |
                ((dim3 & 0xFFFF) << 40)
            )
            f.write(f"csrw 0x80A, 0x{trigger_payload:016X}  # CSR_VPU_MACRO_TRIGGER\n\n")

        # ★ 3. 寫入 C Header Inline Assembly
        with open(self.c_filepath, "a") as f_c:
            f_c.write(f"\n    // --- Dispatching Macro: {op.name} ---\n")
            f_c.write(f"    __asm__ volatile(\"csrw 0x801, %0\" :: \"r\"(0x{csr.Mem_Base_A:016X}ULL));\n")
            f_c.write(f"    __asm__ volatile(\"csrw 0x802, %0\" :: \"r\"(0x{csr.Mem_Base_B:016X}ULL));\n")
            f_c.write(f"    __asm__ volatile(\"csrw 0x803, %0\" :: \"r\"(0x{csr.Mem_Base_C:016X}ULL));\n")
            f_c.write(f"    __asm__ volatile(\"csrw 0x804, %0\" :: \"r\"(0x{csr.Mem_Base_D:016X}ULL));\n")
            f_c.write(f"    __asm__ volatile(\"csrw 0x805, %0\" :: \"r\"(0x{mem_stride_payload:016X}ULL));\n")
            f_c.write(f"    __asm__ volatile(\"csrw 0x806, %0\" :: \"r\"(0x{mem_acc_payload:016X}ULL));\n")
            f_c.write(f"    __asm__ volatile(\"csrw 0x807, %0\" :: \"r\"(0x{reg_base_payload:016X}ULL));\n")
            f_c.write(f"    __asm__ volatile(\"csrw 0x808, %0\" :: \"r\"(0x{reg_stride_payload:016X}ULL));\n")
            f_c.write(f"    __asm__ volatile(\"csrw 0x809, %0\" :: \"r\"(0x{tile_payload:016X}ULL));\n")
            f_c.write(f"    __asm__ volatile(\"csrw 0x80A, %0\" :: \"r\"(0x{trigger_payload:016X}ULL));\n")

    def tick(self):
        self.global_cycle += 1
        
        # --- Stage 1: Backend Execution (Out-of-Order consumption) ---
        def try_issue_from_queue(unit: ExecutionUnit, queue: DecoupledQueue):
            if not unit.busy and len(queue) > 0:
                uop = queue.queue[0] 
                if self.scoreboard.can_execute(uop):
                    queue.pop()
                    unit.issue(uop)
                else:
                    unit.stall_cycles += 1 

        try_issue_from_queue(self.lsu_unit, self.lsu_queue)
        self.lsu_unit.tick()
        try_issue_from_queue(self.valu_unit, self.valu_queue)
        self.valu_unit.tick()
        try_issue_from_queue(self.cim_unit, self.cim_queue)
        self.cim_unit.tick()

        # --- Stage 2: Macro-to-Micro Expansion (FSM Generation) ---
        if not self.micro_op_buffer and self.macro_instr_buffer:
            current_macro = self.macro_instr_buffer.popleft()
            uops = self.expander.expand(current_macro)
            self.micro_op_buffer.extend(uops)
            self.total_micro_generated += len(uops)

        # --- Stage 3: Frontend Dispatch (Strictly In-Order Push) ---
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
        print("="*60)
        print(f"Simulation Report (Total Cycles: {self.global_cycle:,})")
        print("="*60)
        print(f"[Instruction Fetch Bandwidth Reduction]")
        print(f"  - CPU Macro Ops Fetched : {self.total_macro_fetched:,}")
        print(f"  - VPU Micro Ops Executed: {self.total_micro_generated:,}")
        print(f"  - Expansion Ratio       : 1:{self.total_micro_generated/max(1, self.total_macro_fetched):.1f}")
        print(f"\n[Frontend Stall Analysis]")
        print(f"  - Queue Full Stalls     : {self.stall_queue_full_cycles:,} cycles ({(self.stall_queue_full_cycles/self.global_cycle):.1%})")
        print(f"\n[Backend Overlap & Hazard Analysis (The True Decoupling)]")
        print(f"  - LSU Active  : {self.lsu_unit.total_active_cycles/self.global_cycle:5.1%} | Wait Data (RAW/WAR): {self.lsu_unit.stall_cycles/self.global_cycle:5.1%}")
        print(f"  - VALU Active : {self.valu_unit.total_active_cycles/self.global_cycle:5.1%} | Wait Data (RAW/WAR): {self.valu_unit.stall_cycles/self.global_cycle:5.1%}")
        print(f"  - CIM Active  : {self.cim_unit.total_active_cycles/self.global_cycle:5.1%} | Wait Data (RAW/WAR): {self.cim_unit.stall_cycles/self.global_cycle:5.1%}")
        
        # ★ NEW: SRAM 頻寬分析報表
        print(f"\n[SRAM Memory Bandwidth Monitor (@ {CLOCK_FREQ_GHZ} GHz)]")
        total_mbytes = self.lsu_unit.total_bytes_transferred / (1024**2)
        
        # 公式：(總 Bytes / 10^9) / (總 Cycle / (Freq * 10^9)) = (Bytes / Cycles) * Freq
        # 這裡的 GB/s 使用 10^9 Bytes = 1 GB (硬體頻寬標準算法)
        avg_bw_gbs = (self.lsu_unit.total_bytes_transferred / self.global_cycle) * CLOCK_FREQ_GHZ
        peak_bw_gbs = (AXI_WIDTH // 8) * CLOCK_FREQ_GHZ
        
        print(f"  - Total Data Transferred: {total_mbytes:.2f} MB")
        print(f"  - Average Bandwidth     : {avg_bw_gbs:.2f} GB/s")
        print(f"  - Peak AXI Bandwidth    : {peak_bw_gbs:.2f} GB/s")
        print(f"  - Bandwidth Utilization : {(avg_bw_gbs/peak_bw_gbs):.1%}")
        print("="*60)

# ==============================================================================
# 4. Macro-OP FSM Templates (Incorporating 2D-AGU Logic)
# ==============================================================================
def get_actual_vreg(base_reg, sub_idx, tile_size, stride):
    """ Safely calculates VREG offsets avoiding Ping-Pong buffer pollution. """
    elements_per_vreg = max(1, tile_size // stride)
    return base_reg + (sub_idx // elements_per_vreg)

def macro_gemm_template(csr: CSRConfig, tensor: TensorConfig, latency:LatencySet):
    uops = []
    c_regs = [csr.MatC_reg_base + i for i in range(csr.VREG_stride_C)]

    for m_start in range(0, csr.M_total, csr.M_tile):
        for n_start in range(0, csr.N_total, csr.N_tile):
            uops.append(MicroOp("CIM_CLEAR_PSUM", UnitType.CIM, latency=1))
            current_m_tile = min(csr.M_tile, csr.M_total - m_start)  # forloop boundary check
            current_n_tile = min(csr.N_tile, csr.N_total - n_start)  # forloop boundary check

            for k_start in range(0, csr.K_total, csr.K_tile):
                # ping-pong buffer in VRF
                offset_a = ((k_start // csr.K_tile) % 2) * csr.VREG_stride_A if csr.Enable_Double_Buffer else 0
                offset_b = ((k_start // csr.K_tile) % 2) * csr.VREG_stride_B if csr.Enable_Double_Buffer else 0
                
                reg_a = csr.MatA_reg_base + offset_a
                reg_b = csr.MatB_reg_base + offset_b
                a_regs = [reg_a + i for i in range(csr.VREG_stride_A)]
                b_regs = [reg_b + i for i in range(csr.VREG_stride_B)]

                # [AGU] Compute physical memory addresses for the current 2D block
                addr_A = csr.Mem_Base_A + (m_start * csr.Mem_Stride_A) + (k_start * 1) # INT8 assumed
                addr_B = csr.Mem_Base_B + (k_start * csr.Mem_Stride_B) + (n_start * 1)

                uops.append(MicroOp(
                    name=f"LSU_LOAD_A_m{m_start}_k{k_start}", unit_type=UnitType.LSU, 
                    latency=latency.Load_One_Vector*csr.VREG_stride_A + ((csr.VREG_stride_A * VLENB // csr.BLOCK_LEN_A) if csr.Is_Gather_A else 0),
                    dst_regs=a_regs,
                    mem_addr=addr_A, mem_stride=csr.Mem_Stride_A, 
                    is_gather_scatter=csr.Is_Gather_A, block_length=csr.BLOCK_LEN_A
                ))
                uops.append(MicroOp(
                    name=f"LSU_LOAD_B_k{k_start}_n{n_start}", unit_type=UnitType.LSU, 
                    latency=latency.Load_One_Vector*csr.VREG_stride_B + ((csr.VREG_stride_B * VLENB // csr.BLOCK_LEN_B) if csr.Is_Gather_B else 0),
                    dst_regs=b_regs,
                    mem_addr=addr_B, mem_stride=csr.Mem_Stride_B, 
                    is_gather_scatter=csr.Is_Gather_B, block_length=csr.BLOCK_LEN_B
                ))

                # [CIM] Temporal Folding Matrix MAC
                for m_sub in range(0, current_m_tile, tensor.phys_M):
                    for n_sub in range(0, current_n_tile, tensor.phys_N):
                        actual_reg_a = get_actual_vreg(reg_a, m_sub, csr.M_tile, csr.VREG_stride_A)
                        actual_reg_b = get_actual_vreg(reg_b, n_sub, csr.N_tile, csr.VREG_stride_B)
                        uops.append(MicroOp(
                            name=f"CIM_MAC_{m_sub}_{n_sub}", unit_type=UnitType.CIM, latency=csr.K_tile, 
                            src_regs=[actual_reg_a, actual_reg_b]
                        ))

            # [AGU] Compute physical address for Output C
            # TODO Support the Scatter Latency in the future
            addr_C = csr.Mem_Base_C + (m_start * csr.Mem_Stride_C) + (n_start * 1)
            uops.append(MicroOp("CIM_QUANT_OUT", UnitType.CIM, latency=latency.Store_One_Vector*csr.VREG_stride_C, dst_regs=c_regs))
            uops.append(MicroOp(
                name=f"LSU_STORE_C_m{m_start}_n{n_start}", unit_type=UnitType.LSU, 
                latency=latency.Store_One_Vector*csr.VREG_stride_C + ((csr.VREG_stride_C * VLENB  // csr.BLOCK_LEN_C) if csr.Is_Scatter_C else 0),
                src_regs=c_regs,
                mem_addr=addr_C, mem_stride=csr.Mem_Stride_C,
                is_gather_scatter=csr.Is_Scatter_C, block_length=csr.BLOCK_LEN_C
            ))
    return uops

def macro_gemm_gelu_template(csr: CSRConfig, tensor: TensorConfig, latency: LatencySet):
    """ Operator Fusion: GEMM + Activation with implicit memory addressing. """
    uops = []
    c_regs = [csr.MatC_reg_base + i for i in range(csr.VREG_stride_C)]

    for m_start in range(0, csr.M_total, csr.M_tile):
        for n_start in range(0, csr.N_total, csr.N_tile):
            uops.append(MicroOp("CIM_CLEAR_L0_BUFFER", UnitType.CIM, latency=1))
            current_m_tile = min(csr.M_tile, csr.M_total - m_start)
            current_n_tile = min(csr.N_tile, csr.N_total - n_start)

            for k_start in range(0, csr.K_total, csr.K_tile):
                offset_a = ((k_start // csr.K_tile) % 2) * csr.VREG_stride_A if csr.Enable_Double_Buffer else 0
                offset_b = ((k_start // csr.K_tile) % 2) * csr.VREG_stride_B if csr.Enable_Double_Buffer else 0
                reg_a = csr.MatA_reg_base + offset_a
                reg_b = csr.MatB_reg_base + offset_b
                a_regs = [reg_a + i for i in range(csr.VREG_stride_A)]
                b_regs = [reg_b + i for i in range(csr.VREG_stride_B)]

                addr_A = csr.Mem_Base_A + (m_start * csr.Mem_Stride_A) + k_start
                addr_B = csr.Mem_Base_B + (k_start * csr.Mem_Stride_B) + n_start

                uops.append(MicroOp("LSU_LOAD_A", unit_type=UnitType.LSU, 
                    latency=latency.Load_One_Vector*csr.VREG_stride_A + ((csr.VREG_stride_A * VLENB // csr.BLOCK_LEN_A) if csr.Is_Gather_A else 0),
                    dst_regs=a_regs,
                    mem_addr=addr_A, mem_stride=csr.Mem_Stride_A, 
                    is_gather_scatter=csr.Is_Gather_A, block_length=csr.BLOCK_LEN_A
                ))
                uops.append(MicroOp("LSU_LOAD_B", unit_type=UnitType.LSU, 
                    latency=latency.Load_One_Vector*csr.VREG_stride_B + ((csr.VREG_stride_B * VLENB // csr.BLOCK_LEN_B) if csr.Is_Gather_B else 0),
                    dst_regs=b_regs,
                    mem_addr=addr_B, mem_stride=csr.Mem_Stride_B, 
                    is_gather_scatter=csr.Is_Gather_B, block_length=csr.BLOCK_LEN_B
                ))
                
                for m_sub in range(0, current_m_tile, tensor.phys_M):
                    for n_sub in range(0, current_n_tile, tensor.phys_N):
                        actual_reg_a = get_actual_vreg(reg_a, m_sub, csr.M_tile, csr.VREG_stride_A)
                        actual_reg_b = get_actual_vreg(reg_b, n_sub, csr.N_tile, csr.VREG_stride_B)
                        uops.append(MicroOp(name=f"CIM_MAC", unit_type=UnitType.CIM, latency=csr.K_tile, src_regs=[actual_reg_a, actual_reg_b]))

            addr_C = csr.Mem_Base_C + (m_start * csr.Mem_Stride_C) + n_start
            uops.append(MicroOp("CIM_QUANT_OUT", UnitType.CIM, latency=latency.Store_One_Vector*csr.VREG_stride_C, dst_regs=c_regs))
            act_name = csr.Act_Type.name if csr.Act_Type != ActivationType.NONE else "LINEAR"
            uops.append(MicroOp(f"VALU_{act_name}_LUT", UnitType.VALU, latency=latency.VALU_VGELU*csr.VREG_stride_C, src_regs=c_regs, dst_regs=c_regs))
            uops.append(MicroOp("LSU_STORE_C", unit_type=UnitType.LSU, 
                latency=latency.Store_One_Vector*csr.VREG_stride_C + ((csr.VREG_stride_C * VLENB  // csr.BLOCK_LEN_C) if csr.Is_Scatter_C else 0),
                src_regs=c_regs,
                mem_addr=addr_C, mem_stride=csr.Mem_Stride_C,
                is_gather_scatter=csr.Is_Scatter_C, block_length=csr.BLOCK_LEN_C
            ))
    return uops

def macro_flash_attn_template(csr: CSRConfig, tensor: TensorConfig, latency: LatencySet, Seq_Len=512):
    """
    FlashAttention Template with 2D-AGU & Indirect Memory Access Support
    - Q mapped to Mem_Base_A
    - K mapped to Mem_Base_B
    - V mapped to Mem_Base_D
    - Output mapped to Mem_Base_C
    """
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
        
        # [AGU] 動態計算 Q 矩陣的實體記憶體位址
        addr_Q = csr.Mem_Base_A + (q_start * csr.Mem_Stride_A)
        
        uops.append(MicroOp(
            name=f"LSU_LOAD_Q_q{q_start}", unit_type=UnitType.LSU, 
            latency=latency.Load_One_Vector*csr.VREG_stride_A, dst_regs=q_regs,
            mem_addr=addr_Q, mem_stride=csr.Mem_Stride_A,
            is_gather_scatter=csr.Is_Gather_A, index_reg=csr.Index_Reg_A
        ))

        for k_start in range(0, Seq_Len, csr.K_tile):
            uops.append(MicroOp("CIM_CLEAR_L0", UnitType.CIM, latency=1))
            
            # [AGU] 動態計算 K 矩陣的實體記憶體位址
            addr_K = csr.Mem_Base_B + (k_start * csr.Mem_Stride_B)
            
            uops.append(MicroOp(
                name=f"LSU_LOAD_K_k{k_start}", unit_type=UnitType.LSU, 
                latency=latency.Load_One_Vector*csr.VREG_stride_B, dst_regs=k_regs,
                mem_addr=addr_K, mem_stride=csr.Mem_Stride_B,
                is_gather_scatter=csr.Is_Gather_B, index_reg=csr.Index_Reg_B
            ))

            for m_sub in range(0, csr.M_tile, tensor.phys_M):
                for n_sub in range(0, csr.K_tile, tensor.phys_N):
                    actual_q = get_actual_vreg(reg_q, m_sub, csr.M_tile, csr.VREG_stride_A)
                    actual_k = get_actual_vreg(reg_k, n_sub, csr.K_tile, csr.VREG_stride_B)
                    uops.append(MicroOp(
                        name=f"CIM_QK_{m_sub}_{n_sub}", unit_type=UnitType.CIM, latency=csr.K_tile, 
                        src_regs=[actual_q, actual_k], dst_regs=[VIRTUAL_L0_BUFFER_ID]
                    ))

            uops.append(MicroOp("VALU_SOFTMAX_UPDATE", UnitType.VALU, latency=20, src_regs=[VIRTUAL_L0_BUFFER_ID], dst_regs=[]))
            uops.append(MicroOp("VALU_SOFTMAX_EXP", UnitType.VALU, latency=1024, src_regs=[VIRTUAL_L0_BUFFER_ID], dst_regs=p_regs))

            uops.append(MicroOp("CIM_CLEAR_L0", UnitType.CIM, latency=1))
            
            # [AGU] 動態計算 V 矩陣的實體記憶體位址
            addr_V = csr.Mem_Base_D + (k_start * csr.Mem_Stride_D)
            
            uops.append(MicroOp(
                name=f"LSU_LOAD_V_k{k_start}", unit_type=UnitType.LSU, 
                latency=latency.Load_One_Vector*csr.VREG_stride_D, dst_regs=v_regs,
                mem_addr=addr_V, mem_stride=csr.Mem_Stride_D,
                is_gather_scatter=csr.Is_Gather_D, index_reg=csr.Index_Reg_D
            ))

            for m_sub in range(0, csr.M_tile, tensor.phys_M):
                for n_sub in range(0, csr.K_tile, tensor.phys_N):
                    actual_p = get_actual_vreg(reg_p, m_sub, csr.M_tile, csr.VREG_stride_E)
                    actual_v = get_actual_vreg(reg_v, n_sub, csr.K_tile, csr.VREG_stride_D)
                    uops.append(MicroOp(
                        name=f"CIM_PV_{m_sub}_{n_sub}", unit_type=UnitType.CIM, latency=csr.K_tile, 
                        src_regs=[actual_p, actual_v], dst_regs=[VIRTUAL_L0_BUFFER_ID]
                    ))

            uops.append(MicroOp(
                name=f"VALU_GLOBAL_O", unit_type=UnitType.VALU, latency=64, 
                src_regs=o_global_regs + [VIRTUAL_L0_BUFFER_ID], dst_regs=o_global_regs
            ))

        uops.append(MicroOp("VALU_QUANT_O", UnitType.VALU, latency=20, src_regs=o_global_regs, dst_regs=quant_regs))
        
        # [AGU] 動態計算 Output 矩陣寫回的實體記憶體位址
        addr_O = csr.Mem_Base_C + (q_start * csr.Mem_Stride_C)
        
        uops.append(MicroOp(
            name=f"LSU_STORE_O_q{q_start}", unit_type=UnitType.LSU, 
            latency=latency.Store_One_Vector*csr.VREG_stride_C, src_regs=quant_regs, dst_regs=[],
            mem_addr=addr_O, mem_stride=csr.Mem_Stride_C,
            is_gather_scatter=csr.Is_Scatter_C, index_reg=csr.Index_Reg_C
        ))

    return uops

def macro_residual_layernorm_template(csr: CSRConfig, latency: LatencySet, Seq_Len=0, Hidden_Dim=768):
    """
    Residual Add + LayerNorm Template with 2D-AGU
    - Main branch mapped to Mem_Base_A
    - Residual branch mapped to Mem_Base_B
    - Output mapped to Mem_Base_C
    """
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
        
        # [AGU] 動態計算 Main 與 Residual 分支的實體記憶體位址
        addr_Main = csr.Mem_Base_A + (seq_idx * csr.Mem_Stride_A)
        addr_Res  = csr.Mem_Base_B + (seq_idx * csr.Mem_Stride_B)
        
        uops.append(MicroOp(
            name=f"LSU_LOAD_MAIN_{seq_idx}", unit_type=UnitType.LSU, 
            latency=latency.Load_One_Vector*csr.VREG_stride_C, dst_regs=main_regs,
            mem_addr=addr_Main, mem_stride=csr.Mem_Stride_A,
            is_gather_scatter=csr.Is_Gather_A, index_reg=csr.Index_Reg_A
        ))
        
        uops.append(MicroOp(
            name=f"LSU_LOAD_RES_{seq_idx}", unit_type=UnitType.LSU, 
            latency=latency.Load_One_Vector*csr.VREG_stride_C, dst_regs=res_regs,
            mem_addr=addr_Res, mem_stride=csr.Mem_Stride_B,
            is_gather_scatter=csr.Is_Gather_B, index_reg=csr.Index_Reg_B
        ))
        
        uops.append(MicroOp(f"VALU_VADD_RES", UnitType.VALU, latency=int(latency.VALU_VADD*csr.VREG_stride_C), src_regs=main_regs + res_regs, dst_regs=out_regs))
        
        realistic_valu_lat = (csr.M_tile * Hidden_Dim) // (LANE * AXI_WIDTH) + 10 
        uops.append(MicroOp("VALU_LN_MEAN", UnitType.VALU, latency=realistic_valu_lat, src_regs=out_regs, dst_regs=[reg_mean]))
        uops.append(MicroOp("VALU_LN_VAR", UnitType.VALU, latency=realistic_valu_lat, src_regs=out_regs + [reg_mean], dst_regs=[reg_var]))
        uops.append(MicroOp("VALU_LN_RSQRT", UnitType.VALU, latency=20, src_regs=[reg_var], dst_regs=[reg_var]))
        uops.append(MicroOp("VALU_LN_NORM", UnitType.VALU, latency=realistic_valu_lat, src_regs=out_regs + [reg_mean, reg_var], dst_regs=out_regs))
        
        # [AGU] 動態計算 LayerNorm Output 寫回的實體記憶體位址
        addr_Out = csr.Mem_Base_C + (seq_idx * csr.Mem_Stride_C)
        
        uops.append(MicroOp(
            name=f"LSU_STORE_LN_{seq_idx}", unit_type=UnitType.LSU, 
            latency=latency.Store_One_Vector*csr.VREG_stride_C, src_regs=out_regs, dst_regs=[],
            mem_addr=addr_Out, mem_stride=csr.Mem_Stride_C,
            is_gather_scatter=csr.Is_Scatter_C, index_reg=csr.Index_Reg_C
        ))
        
    return uops

# ==============================================================================
# 5. Model Builders (Software Memory Allocation & Dispatch)
# ==============================================================================

def build_bert_base_layer(sim: ADHD_VPU, csr: CSRConfig, tensorHW: TensorConfig, latencySet: LatencySet, seq_len: int, mem_mgr: MemoryManager):
    D = 768
    D_FFN = 3072
    print(f"\n--- Dispatching BERT Base Layer (Seq Length: {seq_len}) ---")
    
    # 1. Q, K, V Projections
    for proj_name in ["PROJ_Q", "PROJ_K", "PROJ_V"]:
        csr.Mem_Base_A = mem_mgr.allocate(seq_len * D) # Input [Seq, D]
        csr.Mem_Stride_A = D
        csr.Mem_Base_B = mem_mgr.allocate(D * D)       # Weight [D, D]
        csr.Mem_Stride_B = D
        csr.Mem_Base_C = mem_mgr.allocate(seq_len * D) # Output [Seq, D]
        csr.Mem_Stride_C = D
        sim.fetch_macro([MacroOp(proj_name, macro_gemm_template, {"csr":csr, "tensor": tensorHW, "latency": latencySet, "M_total": seq_len, "N_total": D, "K_total": D})])

    # 2. FlashAttention (12 Heads)
    head_dim = D // 12
    for h in range(12):
        csr.Mem_Base_A = mem_mgr.allocate(seq_len * head_dim) # Q Head
        csr.Mem_Stride_A = head_dim
        csr.Mem_Base_B = mem_mgr.allocate(seq_len * head_dim) # K Head
        csr.Mem_Stride_B = head_dim
        csr.Mem_Base_D = mem_mgr.allocate(seq_len * head_dim) # V Head
        csr.Mem_Stride_D = head_dim
        csr.Mem_Base_C = mem_mgr.allocate(seq_len * head_dim) # Out Head
        csr.Mem_Stride_C = head_dim
        sim.fetch_macro([MacroOp(f"FLASH_ATTN_H{h}", macro_flash_attn_template, {"csr":csr, "tensor": tensorHW, "latency": latencySet, "Seq_Len": seq_len})])

    # 3. Attention Output Projection
    csr.Mem_Base_A = mem_mgr.allocate(seq_len * D)
    csr.Mem_Stride_A = D
    csr.Mem_Base_B = mem_mgr.allocate(D * D)
    csr.Mem_Stride_B = D
    csr.Mem_Base_C = mem_mgr.allocate(seq_len * D)
    csr.Mem_Stride_C = D
    sim.fetch_macro([MacroOp("ATTN_OUT_PROJ", macro_gemm_template, {"csr":csr, "tensor": tensorHW, "latency": latencySet, "M_total": seq_len, "N_total": D, "K_total": D})])

    # 4. Residual Add + LayerNorm 1
    csr.Mem_Base_A = mem_mgr.allocate(seq_len * D) # Main Branch
    csr.Mem_Stride_A = D
    csr.Mem_Base_B = mem_mgr.allocate(seq_len * D) # Residual Branch
    csr.Mem_Stride_B = D
    csr.Mem_Base_C = mem_mgr.allocate(seq_len * D) # LN Output
    csr.Mem_Stride_C = D
    sim.fetch_macro([MacroOp("RES_LN_1", macro_residual_layernorm_template, {"csr":csr, "latency": latencySet, "Seq_Len": seq_len, "Hidden_Dim": D})])

    # 5. FFN Layer 1 (GEMM + GELU)
    csr.Mem_Base_A = mem_mgr.allocate(seq_len * D)
    csr.Mem_Stride_A = D
    csr.Mem_Base_B = mem_mgr.allocate(D * D_FFN)
    csr.Mem_Stride_B = D_FFN
    csr.Mem_Base_C = mem_mgr.allocate(seq_len * D_FFN)
    csr.Mem_Stride_C = D_FFN
    csr.Act_Type = ActivationType.GELU
    sim.fetch_macro([MacroOp("FFN1_GELU", macro_gemm_gelu_template, {"csr":csr, "tensor": tensorHW, "latency": latencySet, "M_total": seq_len, "N_total": D_FFN, "K_total": D})])
    csr.Act_Type = ActivationType.NONE # Reset flag

    # 6. FFN Layer 2
    csr.Mem_Base_A = mem_mgr.allocate(seq_len * D_FFN)
    csr.Mem_Stride_A = D_FFN
    csr.Mem_Base_B = mem_mgr.allocate(D_FFN * D)
    csr.Mem_Stride_B = D
    csr.Mem_Base_C = mem_mgr.allocate(seq_len * D)
    csr.Mem_Stride_C = D
    sim.fetch_macro([MacroOp("FFN2", macro_gemm_template, {"csr":csr, "tensor": tensorHW, "latency": latencySet, "M_total": seq_len, "N_total": D, "K_total": D_FFN})])

    # 7. Residual Add + LayerNorm 2
    csr.Mem_Base_A = mem_mgr.allocate(seq_len * D)
    csr.Mem_Stride_A = D
    csr.Mem_Base_B = mem_mgr.allocate(seq_len * D)
    csr.Mem_Stride_B = D
    csr.Mem_Base_C = mem_mgr.allocate(seq_len * D)
    csr.Mem_Stride_C = D
    sim.fetch_macro([MacroOp("RES_LN_2", macro_residual_layernorm_template, {"csr":csr, "latency": latencySet, "Seq_Len": seq_len, "Hidden_Dim": D})])
    
    # 模擬 Compiler Free Memory: 層結束後釋放 SRAM Workspace，供下一層重複使用
    mem_mgr.reset()

def build_vit_base_layer(sim: ADHD_VPU, csr: CSRConfig, tensorHW: TensorConfig, latencySet: LatencySet, seq_len: int, mem_mgr: MemoryManager):
    """ ViT Base: Highly similar to BERT, but typically sequence length is 197 (14x14 patches + 1 CLS). """
    D = 768
    D_FFN = 3072
    print(f"\n--- Dispatching ViT Base Layer (Seq Length: {seq_len}) ---")
    
    # [Note for Architect] ViT uses Pre-LN, so structurally it might start with LN.
    # To demonstrate memory allocation elasticity, we use the same builders but dynamic shapes.
    
    # 1. Pre-LayerNorm 1
    csr.Mem_Base_A = mem_mgr.allocate(seq_len * D)
    csr.Mem_Stride_A = D
    csr.Mem_Base_B = mem_mgr.allocate(seq_len * D) # Assuming identity residual here
    csr.Mem_Stride_B = D
    csr.Mem_Base_C = mem_mgr.allocate(seq_len * D)
    csr.Mem_Stride_C = D
    sim.fetch_macro([MacroOp("VIT_PRE_LN_1", macro_residual_layernorm_template, {"csr":csr, "latency": latencySet, "Seq_Len": seq_len, "Hidden_Dim": D})])

    # 2. Q, K, V Projections
    for proj_name in ["VIT_PROJ_Q", "VIT_PROJ_K", "VIT_PROJ_V"]:
        csr.Mem_Base_A = mem_mgr.allocate(seq_len * D)
        csr.Mem_Stride_A = D
        csr.Mem_Base_B = mem_mgr.allocate(D * D)
        csr.Mem_Stride_B = D
        csr.Mem_Base_C = mem_mgr.allocate(seq_len * D)
        csr.Mem_Stride_C = D
        sim.fetch_macro([MacroOp(proj_name, macro_gemm_template, {"csr":csr, "tensor": tensorHW, "latency": latencySet, "M_total": seq_len, "N_total": D, "K_total": D})])

    # 3. FlashAttention
    head_dim = D // 12
    for h in range(12):
        csr.Mem_Base_A = mem_mgr.allocate(seq_len * head_dim)
        csr.Mem_Stride_A = head_dim
        csr.Mem_Base_B = mem_mgr.allocate(seq_len * head_dim)
        csr.Mem_Stride_B = head_dim
        csr.Mem_Base_D = mem_mgr.allocate(seq_len * head_dim)
        csr.Mem_Stride_D = head_dim
        csr.Mem_Base_C = mem_mgr.allocate(seq_len * head_dim)
        csr.Mem_Stride_C = head_dim
        sim.fetch_macro([MacroOp(f"VIT_FLASH_ATTN_H{h}", macro_flash_attn_template, {"csr":csr, "tensor": tensorHW, "latency": latencySet, "Seq_Len": seq_len})])

    # 4. Attention Output Projection
    csr.Mem_Base_A = mem_mgr.allocate(seq_len * D)
    csr.Mem_Stride_A = D
    csr.Mem_Base_B = mem_mgr.allocate(D * D)
    csr.Mem_Stride_B = D
    csr.Mem_Base_C = mem_mgr.allocate(seq_len * D)
    csr.Mem_Stride_C = D
    sim.fetch_macro([MacroOp("VIT_ATTN_OUT", macro_gemm_template, {"csr":csr, "tensor": tensorHW, "latency": latencySet, "M_total": seq_len, "N_total": D, "K_total": D})])

    # 5. Pre-LayerNorm 2
    csr.Mem_Base_A = mem_mgr.allocate(seq_len * D)
    csr.Mem_Stride_A = D
    csr.Mem_Base_B = mem_mgr.allocate(seq_len * D)
    csr.Mem_Stride_B = D
    csr.Mem_Base_C = mem_mgr.allocate(seq_len * D)
    csr.Mem_Stride_C = D
    sim.fetch_macro([MacroOp("VIT_PRE_LN_2", macro_residual_layernorm_template, {"csr":csr, "latency": latencySet, "Seq_Len": seq_len, "Hidden_Dim": D})])

    # 6. MLP 1 & 2
    csr.Mem_Base_A = mem_mgr.allocate(seq_len * D)
    csr.Mem_Stride_A = D
    csr.Mem_Base_B = mem_mgr.allocate(D * D_FFN)
    csr.Mem_Stride_B = D_FFN
    csr.Mem_Base_C = mem_mgr.allocate(seq_len * D_FFN)
    csr.Mem_Stride_C = D_FFN
    csr.Act_Type = ActivationType.GELU
    sim.fetch_macro([MacroOp("VIT_MLP1_GELU", macro_gemm_gelu_template, {"csr":csr, "tensor": tensorHW, "latency": latencySet, "M_total": seq_len, "N_total": D_FFN, "K_total": D})])
    csr.Act_Type = ActivationType.NONE
    
    csr.Mem_Base_A = mem_mgr.allocate(seq_len * D_FFN)
    csr.Mem_Stride_A = D_FFN
    csr.Mem_Base_B = mem_mgr.allocate(D_FFN * D)
    csr.Mem_Stride_B = D
    csr.Mem_Base_C = mem_mgr.allocate(seq_len * D)
    csr.Mem_Stride_C = D
    sim.fetch_macro([MacroOp("VIT_MLP2", macro_gemm_template, {"csr":csr, "tensor": tensorHW, "latency": latencySet, "M_total": seq_len, "N_total": D, "K_total": D_FFN})])

    mem_mgr.reset()

def build_gpt2_prefill_layer(sim: ADHD_VPU, csr: CSRConfig, tensorHW: TensorConfig, latencySet: LatencySet, seq_len: int, mem_mgr: MemoryManager):
    """ GPT-2 Prefill Stage: Handling massive sequence lengths (e.g., 1024+ context window). """
    D = 768
    D_FFN = 3072
    print(f"\n--- Dispatching GPT-2 Prefill Layer (Context Length: {seq_len}) ---")

    # 1. Pre-LN
    csr.Mem_Base_A = mem_mgr.allocate(seq_len * D)
    csr.Mem_Stride_A = D
    csr.Mem_Base_B = mem_mgr.allocate(seq_len * D) 
    csr.Mem_Stride_B = D
    csr.Mem_Base_C = mem_mgr.allocate(seq_len * D)
    csr.Mem_Stride_C = D
    sim.fetch_macro([MacroOp("GPT2_PRE_LN_1", macro_residual_layernorm_template, {"csr":csr, "latency": latencySet, "Seq_Len": seq_len, "Hidden_Dim": D})])

    # 2. QKV Projection
    for proj_name in ["GPT2_PROJ_Q", "GPT2_PROJ_K", "GPT2_PROJ_V"]:
        csr.Mem_Base_A = mem_mgr.allocate(seq_len * D)
        csr.Mem_Stride_A = D
        csr.Mem_Base_B = mem_mgr.allocate(D * D)
        csr.Mem_Stride_B = D
        csr.Mem_Base_C = mem_mgr.allocate(seq_len * D)
        csr.Mem_Stride_C = D
        sim.fetch_macro([MacroOp(proj_name, macro_gemm_template, {"csr":csr, "tensor": tensorHW, "latency": latencySet, "M_total": seq_len, "N_total": D, "K_total": D})])
    
    # 3. FlashAttention
    head_dim = D // 12
    for h in range(12):
        csr.Mem_Base_A = mem_mgr.allocate(seq_len * head_dim)
        csr.Mem_Stride_A = head_dim
        csr.Mem_Base_B = mem_mgr.allocate(seq_len * head_dim)
        csr.Mem_Stride_B = head_dim
        csr.Mem_Base_D = mem_mgr.allocate(seq_len * head_dim)
        csr.Mem_Stride_D = head_dim
        csr.Mem_Base_C = mem_mgr.allocate(seq_len * head_dim)
        csr.Mem_Stride_C = head_dim
        sim.fetch_macro([MacroOp(f"GPT2_FLASH_ATTN_H{h}", macro_flash_attn_template, {"csr":csr, "tensor": tensorHW, "latency": latencySet, "Seq_Len": seq_len})])

    # 4. Attention Output
    csr.Mem_Base_A = mem_mgr.allocate(seq_len * D)
    csr.Mem_Stride_A = D
    csr.Mem_Base_B = mem_mgr.allocate(D * D)
    csr.Mem_Stride_B = D
    csr.Mem_Base_C = mem_mgr.allocate(seq_len * D)
    csr.Mem_Stride_C = D
    sim.fetch_macro([MacroOp("GPT2_ATTN_OUT", macro_gemm_template, {"csr":csr, "tensor": tensorHW, "latency": latencySet, "M_total": seq_len, "N_total": D, "K_total": D})])

    # 5. Residual Add + Pre-LayerNorm 2
    csr.Mem_Base_A = mem_mgr.allocate(seq_len * D)
    csr.Mem_Stride_A = D
    csr.Mem_Base_B = mem_mgr.allocate(seq_len * D)
    csr.Mem_Stride_B = D
    csr.Mem_Base_C = mem_mgr.allocate(seq_len * D)
    csr.Mem_Stride_C = D
    sim.fetch_macro([MacroOp("GPT2_RES_PRE_LN_2", macro_residual_layernorm_template, {"csr":csr, "latency": latencySet, "Seq_Len": seq_len, "Hidden_Dim": D})])

    # 6. FFN Layers
    csr.Mem_Base_A = mem_mgr.allocate(seq_len * D)
    csr.Mem_Stride_A = D
    csr.Mem_Base_B = mem_mgr.allocate(D * D_FFN)
    csr.Mem_Stride_B = D_FFN
    csr.Mem_Base_C = mem_mgr.allocate(seq_len * D_FFN)
    csr.Mem_Stride_C = D_FFN
    csr.Act_Type = ActivationType.GELU
    sim.fetch_macro([MacroOp("GPT2_FFN1_GELU", macro_gemm_gelu_template, {"csr":csr, "tensor": tensorHW, "latency": latencySet, "M_total": seq_len, "N_total": D_FFN, "K_total": D})])
    csr.Act_Type = ActivationType.NONE
    
    csr.Mem_Base_A = mem_mgr.allocate(seq_len * D_FFN)
    csr.Mem_Stride_A = D_FFN
    csr.Mem_Base_B = mem_mgr.allocate(D_FFN * D)
    csr.Mem_Stride_B = D
    csr.Mem_Base_C = mem_mgr.allocate(seq_len * D)
    csr.Mem_Stride_C = D
    sim.fetch_macro([MacroOp("GPT2_FFN2", macro_gemm_template, {"csr":csr, "tensor": tensorHW, "latency": latencySet, "M_total": seq_len, "N_total": D, "K_total": D_FFN})])

    # 7. Final Residual
    csr.Mem_Base_A = mem_mgr.allocate(seq_len * D)
    csr.Mem_Stride_A = D
    csr.Mem_Base_B = mem_mgr.allocate(seq_len * D)
    csr.Mem_Stride_B = D
    csr.Mem_Base_C = mem_mgr.allocate(seq_len * D)
    csr.Mem_Stride_C = D
    sim.fetch_macro([MacroOp("GPT2_FINAL_RES", macro_residual_layernorm_template, {"csr":csr, "latency": latencySet, "Seq_Len": seq_len, "Hidden_Dim": D})])

    mem_mgr.reset()

def build_subOP(sim: ADHD_VPU, csr: CSRConfig, tensorHW: TensorConfig, latencySet: LatencySet, mem_mgr: MemoryManager):
    print(f"\n Sub Macro OP for test ---")

    seq_len = 512
    Hidden_Dim = 768
    head_Dim = Hidden_Dim // 12
    FFN_Dim = 3072

    """ Attention """
    # set tile dimensions
    csr.M_tile, csr.N_tile, csr.K_tile = 64, 64, 32

    # set External Memory
    csr.Mem_Base_A = mem_mgr.allocate(seq_len * head_Dim);   csr.Mem_Stride_A = Hidden_Dim  # Q
    csr.Mem_Base_B = mem_mgr.allocate(head_Dim * seq_len);   csr.Mem_Stride_B = Hidden_Dim  # K
    csr.Mem_Base_D = mem_mgr.allocate(seq_len * head_Dim);   csr.Mem_Stride_D = Hidden_Dim  # V
    csr.Mem_Base_C = mem_mgr.allocate(seq_len * Hidden_Dim); csr.Mem_Stride_C = Hidden_Dim  # Output Head

    # Memory Access Modes (Scatter/Gather)
    csr.Is_Gather_A,  csr.BLOCK_LEN_A  = True, csr.K_tile
    csr.Is_Gather_B,  csr.BLOCK_LEN_B  = True, csr.N_tile
    csr.Is_Scatter_C, csr.BLOCK_LEN_C  = True, csr.N_tile
    csr.Is_Gather_D,  csr.BLOCK_LEN_D  = True, 0  # <--- Fix: No ZeroDivisionError protection applied inside latency logic

    # set VREG Mapping
    # TODO check logic
    csr.MatA_reg_base, csr.VREG_stride_A = 0, 2    # MatQ_tile VREG0 ~ VREG1
    csr.MatB_reg_base, csr.VREG_stride_B = 4, 2    # MatK_tile VREG4 ~ VREG5
    csr.MatD_reg_base, csr.VREG_stride_D = 8, 2    # MatV_tile VREG8 ~ VREG9
    csr.MatE_reg_base, csr.VREG_stride_E = 12, 4
    csr.MatC_reg_base, csr.VREG_stride_O = 16, 16
    csr.Enable_Double_Buffer, csr.Act_Type = True, ActivationType.NONE

    # Execution
    csr.Macro_Op_Name = "GEMM"
    csr.M_total, csr.N_total, csr.K_total = seq_len, seq_len, head_Dim
    sim.fetch_macro([MacroOp("Attention", macro_flash_attn_template, {"csr":csr, "tensor": tensorHW, "latency": latencySet})])


    """
    Projection
    """
    # set tile dimensions
    csr.M_tile = 64
    csr.N_tile = 64
    csr.K_tile = 32

    # set External Memory
    csr.Mem_Base_A = mem_mgr.allocate(seq_len * Hidden_Dim)    # MatA
    csr.Mem_Stride_A = Hidden_Dim                              # 768 elements per row
    csr.Mem_Base_B = mem_mgr.allocate(Hidden_Dim * Hidden_Dim) # MatB
    csr.Mem_Stride_B = Hidden_Dim                              # 768 elements per row
    csr.Mem_Base_C = mem_mgr.allocate(seq_len * Hidden_Dim)    # MatC
    csr.Mem_Stride_C = Hidden_Dim                              # 768 elements per row

    # Memory Access Modes (Scatter/Gather)
    csr.Is_Gather_A  = True
    csr.BLOCK_LEN_A  = csr.K_tile
    csr.Is_Gather_B  = True
    csr.BLOCK_LEN_B  = csr.N_tile
    csr.Is_Scatter_C = True
    csr.BLOCK_LEN_C  = csr.N_tile
    csr.Is_Gather_D  = True
    csr.BLOCK_LEN_D  = 0

    # set VREG Mapping
    csr.MatA_reg_base = 0  # MatA_tile VREG0 ~ VREG1
    csr.VREG_stride_A = 2
    csr.MatB_reg_base = 4  # MatB_tile VREG4 ~ VREG5
    csr.VREG_stride_B = 2
    csr.MatC_reg_base = 8  # MatC_tile VREG8 ~ VREG11
    csr.VREG_stride_C = 4

    # set operation flag
    csr.Enable_Double_Buffer = True
    csr.Act_Type = ActivationType.NONE
    
    # Execution
    csr.Macro_Op_Name = "GEMM"
    csr.M_total = seq_len
    csr.N_total = Hidden_Dim
    csr.K_total = Hidden_Dim


    sim.fetch_macro([MacroOp("Projection", 
                             macro_gemm_template, 
                             {"csr":csr, "tensor": tensorHW, "latency": latencySet})])
    


    """
    FNN + GELU
    """
    # set External Memory
    csr.Mem_Base_A = mem_mgr.allocate(seq_len * Hidden_Dim)    # MatA
    csr.Mem_Stride_A = Hidden_Dim
    csr.Mem_Base_B = mem_mgr.allocate(Hidden_Dim * FFN_Dim) # MatB
    csr.Mem_Stride_B = FFN_Dim
    csr.Mem_Base_C = mem_mgr.allocate(seq_len * FFN_Dim)    # MatC
    csr.Mem_Stride_C = FFN_Dim

    # Memory Access Modes (Scatter/Gather)
    csr.Is_Gather_A  = True
    csr.BLOCK_LEN_A  = csr.K_tile
    csr.Is_Gather_B  = True
    csr.BLOCK_LEN_B  = csr.N_tile
    csr.Is_Scatter_C = True
    csr.BLOCK_LEN_C  = csr.N_tile
    csr.Is_Gather_D  = True
    csr.BLOCK_LEN_D  = 0

    # set VREG Mapping
    csr.MatA_reg_base = 0  # MatA_tile VREG0 ~ VREG1
    csr.VREG_stride_A = 2
    csr.MatB_reg_base = 4  # MatB_tile VREG4 ~ VREG5
    csr.VREG_stride_B = 2
    csr.MatC_reg_base = 8  # MatC_tile VREG8 ~ VREG11
    csr.VREG_stride_C = 4

    # set operation flag
    csr.Enable_Double_Buffer = True
    csr.Act_Type = ActivationType.GELU
    
    # Execution
    csr.Macro_Op_Name = "GEMM_GELU"
    csr.M_total = seq_len
    csr.N_total = FFN_Dim
    csr.K_total = Hidden_Dim


    sim.fetch_macro([MacroOp("Fnn_GELU", 
                             macro_gemm_gelu_template, 
                             {"csr":csr, "tensor": tensorHW, "latency": latencySet})])

# ==============================================================================
# 6. Run Simulation
# ==============================================================================
def run_simulation():
    model = "TEST_SUBOP" # "BERT_Base", "ViT_Base", "GPT2_Base", "TEST_SUBOP"

    latencySet = LatencySet()
    # csr = CSRConfig(MatA_reg_base=0, MatB_reg_base=4, MatC_reg_base=8, Enable_Double_Buffer=True)
    csr = CSRConfig()
    tensorHW = TensorConfig(phys_M=16, phys_N=16)
    sim = ADHD_VPU(model_name=model)
    mem_mgr = MemoryManager(base_addr=0xE000_0000)


    # 呼叫巨集組裝
    if model == "BERT_Base":
        target_seq_len = 512 # 設定可變的 Sequence Length
        build_bert_base_layer(sim, csr, tensorHW, latencySet, seq_len=target_seq_len, mem_mgr=mem_mgr)
    elif model == "ViT_Base":
        target_seq_len = 197
        build_vit_base_layer(sim, csr, tensorHW, latencySet, seq_len=target_seq_len, mem_mgr=mem_mgr)
    elif model == "GPT2_Base":
        target_seq_len = 1024
        build_gpt2_prefill_layer(sim, csr, tensorHW, latencySet, seq_len=target_seq_len, mem_mgr=mem_mgr)
    elif model == "TEST_SUBOP":
        build_subOP(sim, csr, tensorHW, latencySet, mem_mgr=mem_mgr)

    print(f"--- Simulation Running {model} ... ---")
    while not sim.is_idle():
        sim.tick()
    
    # the last "}" for C code generation
    with open(sim.c_filepath, "a") as f_c:
        f_c.write("}\n")
    
    sim.print_report()

if __name__ == "__main__":
    run_simulation()