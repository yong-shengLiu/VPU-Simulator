# ==============================================================================
# TODO List
# ==============================================================================
# 1. 傳進gemm template的tile參數目前都是直接寫數字，沒有參數化: 64, 64, 32


import os
import sys
import time
import copy
import random
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import deque
from typing import List, Optional, Dict

# ==============================================================================
# 0. System Utility: Dual Logger (Terminal + File)
# ==============================================================================
class DualLogger:
    """ 將 standard output 同時印在螢幕上並存入檔案中 """
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()


# ==============================================================================
# 1. Hardware Specifications & Constants
# ==============================================================================
NUM_VREGS = 32          # Standard RISC-V Vector Register File (v0 ~ v31)
LANE = 4                # Number of parallel processing lanes in VALU
VLEN = 8192             # Vector length in bits (1024 bytes per vector register)
VLENB = VLEN // 8       # Vector length in bytes
AXI_WIDTH = 64          # 64-bit AXI bus width (8 bytes per transfer)
LSU_QUEUE_DEPTH = 16    # Decoupled queue depth for Load/Store Unit
VALU_QUEUE_DEPTH = 16   # Decoupled queue depth for Vector ALU
CIM_QUEUE_DEPTH = 32    # Decoupled queue depth for Compute-In-Memory (Tensor Core)

# Virtual Register ID used for Scoreboard tracking to model implicit data dependencies
# passing through the internal L0 Buffer (SRAM) without polluting the VRF.
VIRTUAL_L0_BUFFER_ID = 63 

# 系統時脈設定 (假設 VPU 跑在 1 GHz)
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
    NONE = 0
    GELU = 1
    RELU = 2
    SILU = 3

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
    # TODO 調整 CSR設定, stride跟base同一組。 ping-pong/NAF放到Execution Trigger
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
      - Bits [29:25] : VREG_stride_O        (5-bit)     # TODO this stride O is same function with stride C ??
      - Bits [63:30] : Reserved             (34-bit)

    [ 0x809 ] CSR_VPU_TILE_CFG (硬體 Tiling 邊界維度)
      - Bits [15:0]  : M_tile               (16-bit)
      - Bits [31:16] : N_tile               (16-bit)
      - Bits [47:32] : K_tile               (16-bit)
      - Bits [63:48] : Reserved             (16-bit)

    --------------------------------------------------------------------------------
    
    【執行觸發區 (Execution Trigger)】
    [ 0x80A ] CSR_VPU_MACRO_TRIGGER (執行觸發與動態巨集參數)
      *** 寫入此 CSR 即代表 CPU 發射 Macro-OP, VPU Frontend 將開始解碼 ***
      - Bits [7:0]   : Macro_Opcode         (8-bit)  | 0x0: GEMM, 0x1: FLASH_ATTN, 0x2: RES_LN
      - Bits [23:8]  : M_total / Seq_Len    (16-bit) | M_total 或 Sequence Length
      - Bits [39:24] : N_total              (16-bit) | N_total 或 Hidden_Dim
      - Bits [55:40] : K_total              (16-bit) | K_total
      - Bits [63:56] : Reserved             (8-bit)  | 保留未來擴展
    ================================================================================
    """
    # --- 1. External Memory Pointers & Strides ---
    Mem_Base_A: int = 0x0000_0000  
    Mem_Base_B: int = 0x0000_0000
    Mem_Base_C: int = 0x0000_0000
    Mem_Base_D: int = 0x0000_0000
    
    Mem_Stride_A: int = 64  
    Mem_Stride_B: int = 64
    Mem_Stride_C: int = 64
    Mem_Stride_D: int = 64

    # --- 2. Memory Access Modes (Scatter/Gather) ---
    Is_Gather_A: bool = False
    BLOCK_LEN_A: int = 0
    Is_Gather_B: bool = False
    BLOCK_LEN_B: int = 0
    Is_Scatter_C: bool = False
    BLOCK_LEN_C: int = 0
    Is_Gather_D: bool = False
    BLOCK_LEN_D: int = 0

    # --- 3. Internal VRF Pointers ---
    MatA_reg_base: int = 0
    MatB_reg_base: int = 4
    MatC_reg_base: int = 20
    MatD_reg_base: int = 8    
    MatE_reg_base: int = 12   
    Temp_reg_base: int = 28
    Enable_Double_Buffer: bool = True
    Act_Type: ActivationType = ActivationType.NONE

    # --- 4. VRF Strides ---
    VREG_stride_A: int = 2 
    VREG_stride_B: int = 2
    VREG_stride_C: int = 4
    VREG_stride_D: int = 2    
    VREG_stride_E: int = 4    
    VREG_stride_O: int = 16   

    # --- 5. Tiling Dimensions ---
    M_tile: int = 64
    N_tile: int = 64
    K_tile: int = 32

    # --- 6. Execution Trigger ---
    Macro_Op_Name: str = "GEMM"
    M_total: int = 0
    N_total: int = 0
    K_total: int = 0

    # --- 7. Sparsity Mask ---
    Sparse_Mask: int = 0
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

class BandwidthAnalyzer:
    def __init__(self):
        # 1. 紀錄 Causal Mask 省下的 K/V 載入量 (Software Level)
        self.causal_mask_skipped_bytes = 0 
        self.dense_kv_bytes_if_no_mask = 0  
        
        # 2. 紀錄 Content-based Dynamic Sparsity 省下的 K/V 載入量
        self.content_sparsity_skipped_bytes = 0

        # 3. 紀錄 FlashAttention 省下的 O(N^2) 中繼讀寫量 (Hardware Level)
        self.flash_saved_intermediate_bytes = 0 

    def report(self):
        print(f"\n[Sparsity & Datafusion Bandwidth Savings Analysis]")
        
        if self.dense_kv_bytes_if_no_mask > 0:
            dense_mb = self.dense_kv_bytes_if_no_mask / (1024**2)
            saved_causal_mb = self.causal_mask_skipped_bytes / (1024**2)
            saved_content_mb = self.content_sparsity_skipped_bytes / (1024**2)
            
            # 計算經過 Causal 裁切後，剩下多少合法的 Bytes
            valid_bytes_after_causal = self.dense_kv_bytes_if_no_mask - self.causal_mask_skipped_bytes
            
            causal_ratio = self.causal_mask_skipped_bytes / self.dense_kv_bytes_if_no_mask
            # Content Sparsity 的壓縮率應該建立在「剩下的合法區域」上
            content_ratio = self.content_sparsity_skipped_bytes / valid_bytes_after_causal if valid_bytes_after_causal > 0 else 0
            
            total_saved_mb = saved_causal_mb + saved_content_mb
            total_ratio = total_saved_mb / dense_mb

            print(f"  - Theoretical Dense K/V Traffic   : {dense_mb:.2f} MB")
            print(f"  - [Causal Mask] Traffic Saved     : {saved_causal_mb:.2f} MB ({causal_ratio:.1%})")
            print(f"  - [Content Sparsity] Traffic Saved: {saved_content_mb:.2f} MB ({content_ratio:.1%})")
            print(f"  -> Total K/V Bandwidth Reduction  : {total_ratio:.1%}")

        # FlashAttention Datafusion 節省分析
        saved_flash_mb = self.flash_saved_intermediate_bytes / (1024**2)
        print(f"  - [FlashAttn Datafusion] O(N^2) SRAM Traffic Saved : {saved_flash_mb:.2f} MB")

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
    def __init__(self, model_name="BERT_Base"):
        self.global_cycle = 0
        self.analyzer   = BandwidthAnalyzer()
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

        # --- CSR Trace Logger (動態檔名) ---
        current_dir = os.path.dirname(os.path.abspath(__file__))
        log_dir = os.path.join(current_dir, "log")
        
        # 1. 安全機制：確保 log 資料夾存在，不存在就自動創建
        os.makedirs(log_dir, exist_ok=True)
        
        # 2. 根據模型名稱動態建立檔名
        trace_filename = f"{model_name}_csr_trace.txt"
        c_macro_header = f"{model_name}_macro_dispatch.h"
        
        self.trace_filepath = os.path.join(log_dir, trace_filename)
        self.c_filepath = os.path.join(log_dir, c_macro_header)
        
        # 3. 寫入標頭
        with open(self.trace_filepath, "w") as f:
            f.write(f"=========================================================\n")
            f.write(f" ADHD VPU Firmware CSR Trace ({model_name})\n")
            f.write(f"=========================================================\n\n")
        
        with open(self.c_filepath, "w") as f_c:
            f_c.write("// =========================================================\n")
            f_c.write(f"// ADHD VPU Firmware Dispatcher ({model_name})\n")
            f_c.write("// =========================================================\n")
            f_c.write("#include <stdint.h>\n\n")
            f_c.write(f"static inline void dispatch_{model_name.lower()}_macros() {{\n")

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
            act_type_val = csr.Act_Type.value
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
            # 解析 Macro_Opcode
            opcode_map = {
                "macro_gemm_template": 0x0,
                "macro_flash_attn_template": 0x1,
                "macro_residual_layernorm_template": 0x2
            }
            func_name = op.expansion_func.__name__
            opcode = opcode_map.get(func_name, 0xFF)

            # 解析維度 (相容不同 template 的 args 命名)
            dim1 = csr.M_total
            dim2 = csr.N_total
            dim3 = csr.K_total

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

        # FlashAttention 和 Causal Mask 所節省的 Bandwidth
        self.analyzer.report()
        print("="*60)

# ==============================================================================
# 4. Macro-OP FSM Templates
# ==============================================================================
def get_actual_vreg(base_reg, sub_idx, tile_size, stride):
    """ Safely calculates VREG offsets avoiding Ping-Pong buffer pollution. """
    elements_per_vreg = max(1, tile_size // stride)
    return base_reg + (sub_idx // elements_per_vreg)

def _get_lsu_latency(base_lat, stride, is_sg, block_len):
    """ 安全計算包含 Scatter/Gather Penalty 的 LSU Latency，防止除以零 """
    penalty = (stride * VLENB // max(1, block_len)) if is_sg else 0
    return int(base_lat * stride + penalty)

def macro_gemm_template(csr: CSRConfig, tensor: TensorConfig, latency:LatencySet):
    """
    【一維解耦 GEMM】：外層 M, N 迴圈已交由 CPU 軟體處理。
    硬體 FSM 僅負責 K 維度的歸約與 Output Stationary。
    csr.Mem_Base_A/B/C 已經由 CPU 設定為該 Tile 的精確實體位址。
    """
    uops = []
    c_regs = [csr.MatC_reg_base + i for i in range(csr.VREG_stride_C)]

    uops.append(MicroOp("CIM_CLEAR_PSUM", UnitType.CIM, latency=1))

    # [Inner Loop]: 只有 K_total 是硬體要負責遍歷的
    for k_start in range(0, csr.K_total, csr.K_tile):
        offset_a = ((k_start // csr.K_tile) % 2) * csr.VREG_stride_A if csr.Enable_Double_Buffer else 0
        offset_b = ((k_start // csr.K_tile) % 2) * csr.VREG_stride_B if csr.Enable_Double_Buffer else 0
        
        reg_a = csr.MatA_reg_base + offset_a
        reg_b = csr.MatB_reg_base + offset_b
        a_regs = [reg_a + i for i in range(csr.VREG_stride_A)]
        b_regs = [reg_b + i for i in range(csr.VREG_stride_B)]

        # AGU 計算相對 Offset
        addr_A = csr.Mem_Base_A + k_start 
        addr_B = csr.Mem_Base_B + (k_start * csr.Mem_Stride_B)

        uops.append(MicroOp(
            name=f"LSU_LOAD_A_k{k_start}", unit_type=UnitType.LSU, 
            latency=_get_lsu_latency(latency.Load_One_Vector, csr.VREG_stride_A, csr.Is_Gather_A, csr.BLOCK_LEN_A),
            dst_regs=a_regs, mem_addr=addr_A, mem_stride=csr.Mem_Stride_A, 
            is_gather_scatter=csr.Is_Gather_A, block_length=csr.BLOCK_LEN_A
        ))
        uops.append(MicroOp(
            name=f"LSU_LOAD_B_k{k_start}", unit_type=UnitType.LSU, 
            latency=_get_lsu_latency(latency.Load_One_Vector, csr.VREG_stride_B, csr.Is_Gather_B, csr.BLOCK_LEN_B),
            dst_regs=b_regs, mem_addr=addr_B, mem_stride=csr.Mem_Stride_B, 
            is_gather_scatter=csr.Is_Gather_B, block_length=csr.BLOCK_LEN_B
        ))

        # CIM 運算
        for m_sub in range(0, csr.M_tile, tensor.phys_M):
            for n_sub in range(0, csr.N_tile, tensor.phys_N):
                actual_reg_a = get_actual_vreg(reg_a, m_sub, csr.M_tile, csr.VREG_stride_A)
                actual_reg_b = get_actual_vreg(reg_b, n_sub, csr.N_tile, csr.VREG_stride_B)
                uops.append(MicroOp(
                    name=f"CIM_MAC_{m_sub}_{n_sub}", unit_type=UnitType.CIM, latency=csr.K_tile, 
                    src_regs=[actual_reg_a, actual_reg_b]
                ))

    # Output 寫回
    uops.append(MicroOp("CIM_QUANT_OUT", UnitType.CIM, latency=latency.Store_One_Vector*csr.VREG_stride_C, dst_regs=c_regs))
    if csr.Act_Type == ActivationType.GELU:
        uops.append(MicroOp(f"VALU_GELU", UnitType.VALU, latency=latency.VALU_VGELU*csr.VREG_stride_C, src_regs=c_regs, dst_regs=c_regs))
        
    uops.append(MicroOp(
        name=f"LSU_STORE_C", unit_type=UnitType.LSU, 
        latency=_get_lsu_latency(latency.Store_One_Vector, csr.VREG_stride_C, csr.Is_Scatter_C, csr.BLOCK_LEN_C),
        src_regs=c_regs, mem_addr=csr.Mem_Base_C, mem_stride=csr.Mem_Stride_C,
        is_gather_scatter=csr.Is_Scatter_C, block_length=csr.BLOCK_LEN_C
    ))
    return uops

def macro_lsh_template(csr: CSRConfig, tensor: TensorConfig, latency: LatencySet):
    """
    【一維解耦 LSH Hashing】：
    底層本質上是一個小型 GEMM (Q * Random_Matrix)。
    核心差異：Output 階段不寫回龐大的矩陣，而是經由 VALU 執行 Sign() 二值化與 Bit-packing，
    最後由 LSU 寫回極小的 Hash ID 陣列。
    """
    uops = []
    c_regs = [csr.MatC_reg_base + i for i in range(csr.VREG_stride_C)]

    uops.append(MicroOp("CIM_CLEAR_PSUM", UnitType.CIM, latency=1))

    # [Inner Loop]: 遍歷 D 維度 (對應 GEMM 的 K)
    for k_start in range(0, csr.K_total, csr.K_tile):
        offset_a = ((k_start // csr.K_tile) % 2) * csr.VREG_stride_A if csr.Enable_Double_Buffer else 0
        offset_b = ((k_start // csr.K_tile) % 2) * csr.VREG_stride_B if csr.Enable_Double_Buffer else 0
        
        reg_a = csr.MatA_reg_base + offset_a
        reg_b = csr.MatB_reg_base + offset_b
        a_regs = [reg_a + i for i in range(csr.VREG_stride_A)]
        b_regs = [reg_b + i for i in range(csr.VREG_stride_B)]

        # AGU 計算相對 Offset
        addr_A = csr.Mem_Base_A + k_start 
        addr_B = csr.Mem_Base_B + (k_start * csr.Mem_Stride_B)

        uops.append(MicroOp(
            name=f"LSU_LOAD_Q_k{k_start}", unit_type=UnitType.LSU, 
            latency=_get_lsu_latency(latency.Load_One_Vector, csr.VREG_stride_A, csr.Is_Gather_A, csr.BLOCK_LEN_A),
            dst_regs=a_regs, mem_addr=addr_A, mem_stride=csr.Mem_Stride_A, 
            is_gather_scatter=csr.Is_Gather_A, block_length=csr.BLOCK_LEN_A
        ))
        uops.append(MicroOp(
            name=f"LSU_LOAD_R_k{k_start}", unit_type=UnitType.LSU,  # R 矩陣 (Random Projection)
            latency=_get_lsu_latency(latency.Load_One_Vector, csr.VREG_stride_B, csr.Is_Gather_B, csr.BLOCK_LEN_B),
            dst_regs=b_regs, mem_addr=addr_B, mem_stride=csr.Mem_Stride_B, 
            is_gather_scatter=csr.Is_Gather_B, block_length=csr.BLOCK_LEN_B
        ))

        # CIM 運算 (計算 Q * R)
        for m_sub in range(0, csr.M_tile, tensor.phys_M):
            for n_sub in range(0, csr.N_tile, tensor.phys_N):
                actual_reg_a = get_actual_vreg(reg_a, m_sub, csr.M_tile, csr.VREG_stride_A)
                actual_reg_b = get_actual_vreg(reg_b, n_sub, csr.N_tile, csr.VREG_stride_B)
                uops.append(MicroOp(
                    name=f"CIM_MAC_{m_sub}_{n_sub}", unit_type=UnitType.CIM, latency=csr.K_tile, 
                    src_regs=[actual_reg_a, actual_reg_b]
                ))

    # =========================================================================
    # 🌟 Output 寫回與 Hash Binarization (LSH 的精華差異！)
    # =========================================================================
    
    # 1. 將 Accumulator 的結果讀出到 VREG
    uops.append(MicroOp("CIM_READ_PSUM", UnitType.CIM, latency=1, dst_regs=c_regs))
    
    # 2. ⚡ 交給 VALU 執行 Sign() 和 Bit-packing (關鍵步驟)
    # 硬體行為：(val >= 0) ? 1 : 0，並將 32 個結果壓縮成一個 32-bit 整數
    uops.append(MicroOp(
        name=f"VALU_LSH_SIGN_PACK", unit_type=UnitType.VALU, 
        latency=latency.VALU_VGELU * csr.VREG_stride_C, # 延遲與一般非線性算子相近
        src_regs=c_regs, dst_regs=c_regs
    ))
        
    # 3. 寫回 SRAM (注意：這裡寫回的資料量極小，只有 Hash ID)
    uops.append(MicroOp(
        name=f"LSU_STORE_HASH_ID", unit_type=UnitType.LSU, 
        latency=_get_lsu_latency(latency.Store_One_Vector, csr.VREG_stride_C, csr.Is_Scatter_C, csr.BLOCK_LEN_C),
        src_regs=c_regs, mem_addr=csr.Mem_Base_C, mem_stride=csr.Mem_Stride_C,
        is_gather_scatter=csr.Is_Scatter_C, block_length=csr.BLOCK_LEN_C
    ))
    
    return uops

def macro_flash_attn_template_deprecate(csr: CSRConfig, tensor: TensorConfig, latency: LatencySet, vpu: ADHD_VPU = None):
    """
    【一維解耦 FlashAttention】：外層 Q 迴圈已交由 CPU 軟體處理。
    硬體 FSM 僅負責 K, V 的上下文序列遍歷。
    """
    uops = []
    head_dim = csr.K_total  
    
    reg_q, reg_k, reg_v, reg_p, reg_o_global = csr.MatA_reg_base, csr.MatB_reg_base, csr.MatD_reg_base, csr.MatE_reg_base, csr.MatC_reg_base
    q_regs = [reg_q + i for i in range(csr.VREG_stride_A)]
    p_regs = [reg_p + i for i in range(csr.VREG_stride_E)]
    o_global_regs = [reg_o_global + i for i in range(csr.VREG_stride_O)]
    quant_regs = [csr.Temp_reg_base + i for i in range(csr.VREG_stride_C)]

    # 1. 載入 CPU 指定好的 Q-Tile
    uops.append(MicroOp("VALU_CLEAR_O_GLOBAL", UnitType.VALU, latency=1, dst_regs=o_global_regs))
    uops.append(MicroOp(
        name=f"LSU_LOAD_Q_TILE", unit_type=UnitType.LSU, 
        latency=_get_lsu_latency(latency.Load_One_Vector, csr.VREG_stride_A, csr.Is_Gather_A, csr.BLOCK_LEN_A), 
        dst_regs=q_regs, mem_addr=csr.Mem_Base_A, mem_stride=csr.Mem_Stride_A,
        is_gather_scatter=csr.Is_Gather_A, block_length=csr.BLOCK_LEN_A
    ))

    # [Inner Loop]: 遍歷 K, V 的 Sequence (Context Length)
    for k_start in range(0, csr.N_total, csr.N_tile):
        current_n_tile = min(csr.N_tile, csr.N_total - k_start)
        uops.append(MicroOp("CIM_CLEAR_L0", UnitType.CIM, latency=1))
        
        # 🏓 Ping-Pong 邏輯
        offset_k = ((k_start // csr.N_tile) % 2) * csr.VREG_stride_B if csr.Enable_Double_Buffer else 0
        offset_v = ((k_start // csr.N_tile) % 2) * csr.VREG_stride_D if csr.Enable_Double_Buffer else 0
        reg_k_actual, reg_v_actual = reg_k + offset_k, reg_v + offset_v
        k_regs_actual = [reg_k_actual + i for i in range(csr.VREG_stride_B)]
        v_regs_actual = [reg_v_actual + i for i in range(csr.VREG_stride_D)]

        addr_K = csr.Mem_Base_B + (k_start * csr.Mem_Stride_B)
        uops.append(MicroOp(
            name=f"LSU_LOAD_K_k{k_start}", unit_type=UnitType.LSU, 
            latency=_get_lsu_latency(latency.Load_One_Vector, csr.VREG_stride_B, csr.Is_Gather_B, csr.BLOCK_LEN_B), 
            dst_regs=k_regs_actual, mem_addr=addr_K, mem_stride=csr.Mem_Stride_B,
            is_gather_scatter=csr.Is_Gather_B, block_length=csr.BLOCK_LEN_B
        ))

        # GEMM 1: Q * K^T -> S
        for m_sub in range(0, csr.M_tile, tensor.phys_M):
            for n_sub in range(0, current_n_tile, tensor.phys_N):
                actual_q = get_actual_vreg(reg_q, m_sub, csr.M_tile, csr.VREG_stride_A)
                actual_k = get_actual_vreg(reg_k_actual, n_sub, csr.N_tile, csr.VREG_stride_B)
                uops.append(MicroOp(f"CIM_QK_{m_sub}_{n_sub}", UnitType.CIM, latency=head_dim, src_regs=[actual_q, actual_k], dst_regs=[VIRTUAL_L0_BUFFER_ID]))

        uops.append(MicroOp("VALU_SOFTMAX_UPDATE", UnitType.VALU, latency=20, src_regs=[VIRTUAL_L0_BUFFER_ID]))
        uops.append(MicroOp("VALU_SOFTMAX_EXP", UnitType.VALU, latency=max(1, csr.M_tile*current_n_tile//LANE), src_regs=[VIRTUAL_L0_BUFFER_ID], dst_regs=p_regs))
        uops.append(MicroOp("CIM_CLEAR_L0", UnitType.CIM, latency=1))
        
        addr_V = csr.Mem_Base_D + (k_start * csr.Mem_Stride_D)
        uops.append(MicroOp(
            name=f"LSU_LOAD_V_k{k_start}", unit_type=UnitType.LSU, 
            latency=_get_lsu_latency(latency.Load_One_Vector, csr.VREG_stride_D, csr.Is_Gather_D, csr.BLOCK_LEN_D), 
            dst_regs=v_regs_actual, mem_addr=addr_V, mem_stride=csr.Mem_Stride_D,
            is_gather_scatter=csr.Is_Gather_D, block_length=csr.BLOCK_LEN_D
        ))

        # GEMM 2: P * V -> O
        for m_sub in range(0, csr.M_tile, tensor.phys_M):
            for d_sub in range(0, head_dim, tensor.phys_N): 
                actual_p = get_actual_vreg(reg_p, m_sub, csr.M_tile, csr.VREG_stride_E)
                actual_v = get_actual_vreg(reg_v_actual, d_sub, head_dim, csr.VREG_stride_D)
                uops.append(MicroOp(f"CIM_PV_{m_sub}_{d_sub}", UnitType.CIM, latency=current_n_tile, src_regs=[actual_p, actual_v], dst_regs=[VIRTUAL_L0_BUFFER_ID]))

        uops.append(MicroOp("VALU_GLOBAL_O", UnitType.VALU, latency=64, src_regs=o_global_regs + [VIRTUAL_L0_BUFFER_ID], dst_regs=o_global_regs))

    # [Inner Loop 結束]：量化並寫回 SRAM
    uops.append(MicroOp("VALU_QUANT_O", UnitType.VALU, latency=20, src_regs=o_global_regs, dst_regs=quant_regs))
    uops.append(MicroOp(
        name=f"LSU_STORE_O_TILE", unit_type=UnitType.LSU, 
        latency=_get_lsu_latency(latency.Store_One_Vector, csr.VREG_stride_C, csr.Is_Scatter_C, csr.BLOCK_LEN_C), 
        src_regs=quant_regs, mem_addr=csr.Mem_Base_C, mem_stride=csr.Mem_Stride_C,
        is_gather_scatter=csr.Is_Scatter_C, block_length=csr.BLOCK_LEN_C
    ))

    
    # 計算 FlashAttention 幫我們省下的 O(N^2) 中繼 SRAM 頻寬
    if vpu is not None:
        # 假設資料精度是 16-bit (2 Bytes)
        bytes_per_element = 2 
        
        # 中繼矩陣大小: M_tile * N_total
        intermediate_matrix_size = csr.M_tile * csr.N_total * bytes_per_element
        
        # 傳統 Attention 的致命傷：
        # 1. 寫出 QK^T
        # 2. 讀入 QK^T 做 Softmax
        # 3. 寫出 Softmax Prob
        # 4. 讀入 Softmax Prob 去乘 V
        # 總共 4 次 SRAM 存取！FlashAttention 把它們全部融合在 VRF/L0 Buffer 了！
        saved_traffic = intermediate_matrix_size * 4
        vpu.analyzer.flash_saved_intermediate_bytes += saved_traffic

    return uops

def macro_flash_attn_template(csr: CSRConfig, tensor: TensorConfig, latency: LatencySet, vpu: ADHD_VPU = None):
    """
    1.【一維解耦 FlashAttention】：外層 Q 迴圈已交由 CPU 軟體處理。
    硬體 FSM 僅負責 K, V 的上下文序列遍歷。
    2. 加入sparse mask在k維度上，實現content-base的功能
    """

    # 🌟 [新增] 讀取 CPU 傳下來的 Sparse_Mask (預設為全 1，代表全部要算)
    sparse_mask = getattr(csr, "Sparse_Mask", 0xFFFFFFFF)

    uops = []
    head_dim = csr.K_total
    
    reg_q, reg_k, reg_v, reg_p, reg_o_global = csr.MatA_reg_base, csr.MatB_reg_base, csr.MatD_reg_base, csr.MatE_reg_base, csr.MatC_reg_base
    q_regs = [reg_q + i for i in range(csr.VREG_stride_A)]
    p_regs = [reg_p + i for i in range(csr.VREG_stride_E)]
    o_global_regs = [reg_o_global + i for i in range(csr.VREG_stride_O)]
    quant_regs = [csr.Temp_reg_base + i for i in range(csr.VREG_stride_C)]

    # 1. 載入 CPU 指定好的 Q-Tile
    uops.append(MicroOp("VALU_CLEAR_O_GLOBAL", UnitType.VALU, latency=1, dst_regs=o_global_regs))
    uops.append(MicroOp(
        name=f"LSU_LOAD_Q_TILE", unit_type=UnitType.LSU, 
        latency=_get_lsu_latency(latency.Load_One_Vector, csr.VREG_stride_A, csr.Is_Gather_A, csr.BLOCK_LEN_A), 
        dst_regs=q_regs, mem_addr=csr.Mem_Base_A, mem_stride=csr.Mem_Stride_A,
        is_gather_scatter=csr.Is_Gather_A, block_length=csr.BLOCK_LEN_A
    ))

    # [Inner Loop]: 遍歷 K, V 的 Sequence (Context Length)
    for k_start in range(0, csr.N_total, csr.N_tile):
        # ⚡ FSM 硬體檢查：透過 Bitwise AND 判斷這一個 Block 需不需要算
        block_num = k_start // csr.N_tile
        if not (sparse_mask & (1 << block_num)):
            # 硬體直接跳過這個 Block (Bypass)，可能只消耗 1 個 Cycle 的判斷延遲
            # (在 Python 模擬器中，直接 continue 就不會產生任何 LSU/CIM/VALU 的 uOPs)
            continue

        # --- 以下是原本正常的 FlashAttention 執行步驟 ---
        current_n_tile = min(csr.N_tile, csr.N_total - k_start)
        uops.append(MicroOp("CIM_CLEAR_L0", UnitType.CIM, latency=1))
        
        # 🏓 Ping-Pong 邏輯
        offset_k = ((k_start // csr.N_tile) % 2) * csr.VREG_stride_B if csr.Enable_Double_Buffer else 0
        offset_v = ((k_start // csr.N_tile) % 2) * csr.VREG_stride_D if csr.Enable_Double_Buffer else 0
        reg_k_actual, reg_v_actual = reg_k + offset_k, reg_v + offset_v
        k_regs_actual = [reg_k_actual + i for i in range(csr.VREG_stride_B)]
        v_regs_actual = [reg_v_actual + i for i in range(csr.VREG_stride_D)]

        addr_K = csr.Mem_Base_B + (k_start * csr.Mem_Stride_B)
        uops.append(MicroOp(
            name=f"LSU_LOAD_K_k{k_start}", unit_type=UnitType.LSU, 
            latency=_get_lsu_latency(latency.Load_One_Vector, csr.VREG_stride_B, csr.Is_Gather_B, csr.BLOCK_LEN_B), 
            dst_regs=k_regs_actual, mem_addr=addr_K, mem_stride=csr.Mem_Stride_B,
            is_gather_scatter=csr.Is_Gather_B, block_length=csr.BLOCK_LEN_B
        ))

        # GEMM 1: Q * K^T -> S
        for m_sub in range(0, csr.M_tile, tensor.phys_M):
            for n_sub in range(0, current_n_tile, tensor.phys_N):
                actual_q = get_actual_vreg(reg_q, m_sub, csr.M_tile, csr.VREG_stride_A)
                actual_k = get_actual_vreg(reg_k_actual, n_sub, csr.N_tile, csr.VREG_stride_B)
                uops.append(MicroOp(f"CIM_QK_{m_sub}_{n_sub}", UnitType.CIM, latency=head_dim, src_regs=[actual_q, actual_k], dst_regs=[VIRTUAL_L0_BUFFER_ID]))

        uops.append(MicroOp("VALU_SOFTMAX_UPDATE", UnitType.VALU, latency=20, src_regs=[VIRTUAL_L0_BUFFER_ID]))
        uops.append(MicroOp("VALU_SOFTMAX_EXP", UnitType.VALU, latency=max(1, csr.M_tile*current_n_tile//LANE), src_regs=[VIRTUAL_L0_BUFFER_ID], dst_regs=p_regs))
        uops.append(MicroOp("CIM_CLEAR_L0", UnitType.CIM, latency=1))
        
        addr_V = csr.Mem_Base_D + (k_start * csr.Mem_Stride_D)
        uops.append(MicroOp(
            name=f"LSU_LOAD_V_k{k_start}", unit_type=UnitType.LSU, 
            latency=_get_lsu_latency(latency.Load_One_Vector, csr.VREG_stride_D, csr.Is_Gather_D, csr.BLOCK_LEN_D), 
            dst_regs=v_regs_actual, mem_addr=addr_V, mem_stride=csr.Mem_Stride_D,
            is_gather_scatter=csr.Is_Gather_D, block_length=csr.BLOCK_LEN_D
        ))

        # GEMM 2: P * V -> O
        for m_sub in range(0, csr.M_tile, tensor.phys_M):
            for d_sub in range(0, head_dim, tensor.phys_N): 
                actual_p = get_actual_vreg(reg_p, m_sub, csr.M_tile, csr.VREG_stride_E)
                actual_v = get_actual_vreg(reg_v_actual, d_sub, head_dim, csr.VREG_stride_D)
                uops.append(MicroOp(f"CIM_PV_{m_sub}_{d_sub}", UnitType.CIM, latency=current_n_tile, src_regs=[actual_p, actual_v], dst_regs=[VIRTUAL_L0_BUFFER_ID]))

        uops.append(MicroOp("VALU_GLOBAL_O", UnitType.VALU, latency=64, src_regs=o_global_regs + [VIRTUAL_L0_BUFFER_ID], dst_regs=o_global_regs))

    # [Inner Loop 結束]：量化並寫回 SRAM
    uops.append(MicroOp("VALU_QUANT_O", UnitType.VALU, latency=20, src_regs=o_global_regs, dst_regs=quant_regs))
    uops.append(MicroOp(
        name=f"LSU_STORE_O_TILE", unit_type=UnitType.LSU, 
        latency=_get_lsu_latency(latency.Store_One_Vector, csr.VREG_stride_C, csr.Is_Scatter_C, csr.BLOCK_LEN_C), 
        src_regs=quant_regs, mem_addr=csr.Mem_Base_C, mem_stride=csr.Mem_Stride_C,
        is_gather_scatter=csr.Is_Scatter_C, block_length=csr.BLOCK_LEN_C
    ))

    
    # 計算 FlashAttention 幫我們省下的 O(N^2) 中繼 SRAM 頻寬
    if vpu is not None:
        # 假設資料精度是 16-bit (2 Bytes)
        bytes_per_element = 2 
        
        # 中繼矩陣大小: M_tile * N_total
        intermediate_matrix_size = csr.M_tile * csr.N_total * bytes_per_element
        
        # 傳統 Attention 的致命傷：
        # 1. 寫出 QK^T
        # 2. 讀入 QK^T 做 Softmax
        # 3. 寫出 Softmax Prob
        # 4. 讀入 Softmax Prob 去乘 V
        # 總共 4 次 SRAM 存取！FlashAttention 把它們全部融合在 VRF/L0 Buffer 了！
        saved_traffic = intermediate_matrix_size * 4
        vpu.analyzer.flash_saved_intermediate_bytes += saved_traffic

    return uops

def macro_residual_layernorm_template(csr: CSRConfig, latency: LatencySet):
    """ 【一維解耦 LN】：負責處理一個 Seq_Chunk 的 LayerNorm。 """
    uops = []
    reg_main = csr.MatA_reg_base 
    reg_residual = csr.MatB_reg_base 
    reg_out = csr.MatC_reg_base 
    
    main_regs = [reg_main + i for i in range(csr.VREG_stride_C)]
    res_regs = [reg_residual + i for i in range(csr.VREG_stride_C)]
    out_regs = [reg_out + i for i in range(csr.VREG_stride_C)]
    reg_mean, reg_var  = csr.Temp_reg_base, csr.Temp_reg_base + 1
    
    uops.append(MicroOp(
        "LSU_LOAD_MAIN", UnitType.LSU, 
        latency=_get_lsu_latency(latency.Load_One_Vector, csr.VREG_stride_A, csr.Is_Gather_A, csr.BLOCK_LEN_A),
        dst_regs=main_regs, mem_addr=csr.Mem_Base_A, mem_stride=csr.Mem_Stride_A,
        is_gather_scatter=csr.Is_Gather_A, block_length=csr.BLOCK_LEN_A
    ))

    uops.append(MicroOp(
        "LSU_LOAD_RES", UnitType.LSU, 
        latency=_get_lsu_latency(latency.Load_One_Vector, csr.VREG_stride_B, csr.Is_Gather_B, csr.BLOCK_LEN_B),
        dst_regs=res_regs, mem_addr=csr.Mem_Base_B, mem_stride=csr.Mem_Stride_B,
        is_gather_scatter=csr.Is_Gather_B, block_length=csr.BLOCK_LEN_B
    ))

    uops.append(MicroOp(f"VALU_VADD_RES", UnitType.VALU, latency=int(latency.VALU_VADD*csr.VREG_stride_C), src_regs=main_regs + res_regs, dst_regs=out_regs))
    
    realistic_valu_lat = int((csr.M_tile * csr.K_total) // (LANE * AXI_WIDTH) + 10)
    uops.append(MicroOp("VALU_LN_MEAN", UnitType.VALU, latency=realistic_valu_lat, src_regs=out_regs, dst_regs=[reg_mean]))
    uops.append(MicroOp("VALU_LN_VAR", UnitType.VALU, latency=realistic_valu_lat, src_regs=out_regs + [reg_mean], dst_regs=[reg_var]))
    uops.append(MicroOp("VALU_LN_RSQRT", UnitType.VALU, latency=20, src_regs=[reg_var], dst_regs=[reg_var]))
    uops.append(MicroOp("VALU_LN_NORM", UnitType.VALU, latency=realistic_valu_lat, src_regs=out_regs + [reg_mean, reg_var], dst_regs=out_regs))
    
    uops.append(MicroOp(
        "LSU_STORE_LN", UnitType.LSU, 
        latency=_get_lsu_latency(latency.Store_One_Vector, csr.VREG_stride_C, csr.Is_Scatter_C, csr.BLOCK_LEN_C),
        src_regs=out_regs, mem_addr=csr.Mem_Base_C, mem_stride=csr.Mem_Stride_C,
        is_gather_scatter=csr.Is_Scatter_C, block_length=csr.BLOCK_LEN_C
    ))
    return uops

# ==============================================================================
# 5. Model Builders (Software Memory Allocation & Dispatch)
# ==============================================================================
def set_gemm_csr(csr: CSRConfig, A_base, B_base, C_base, A_stride, B_stride, C_stride, 
                 m_tile, n_tile, k_tile, k_total, act=ActivationType.NONE):
    """ Helper to populate standard GEMM CSR parameters """
    # ==========================================
    # 1. 執行控制與維度設定 (Execution & Dimensions)
    # ==========================================
    csr.Macro_Op_Name = "GEMM"
    csr.Act_Type = act
    csr.M_tile, csr.N_tile, csr.K_tile = m_tile, n_tile, k_tile
    csr.K_total = k_total
    csr.M_total, csr.N_total = 0, 0  # GEMM template 不使用這兩個總維度變數，清零

    # ==========================================
    # 2. 外部記憶體設定 (External Memory)
    # ==========================================
    csr.Mem_Base_A, csr.Mem_Stride_A = A_base, A_stride
    csr.Mem_Base_B, csr.Mem_Stride_B = B_base, B_stride
    csr.Mem_Base_C, csr.Mem_Stride_C = C_base, C_stride
    csr.Mem_Base_D, csr.Mem_Stride_D = 0, 0  # 矩陣 D (V) 在普通 GEMM 中用不到

    # ==========================================
    # 3. 2D AGU 存取模式 (Scatter/Gather Block Length)
    # ==========================================
    # Matrix A (M x K): 每個 row 需要連續讀取 k_tile 個元素
    csr.Is_Gather_A,  csr.BLOCK_LEN_A = True,  k_tile
    # Matrix B (K x N): 每個 row 需要連續讀取 n_tile 個元素
    csr.Is_Gather_B,  csr.BLOCK_LEN_B = True,  n_tile
    # Matrix C (M x N): 每個 row 需要連續寫回 n_tile 個元素
    csr.Is_Scatter_C, csr.BLOCK_LEN_C = True,  n_tile
    # Matrix D: 未使用
    csr.Is_Gather_D,  csr.BLOCK_LEN_D = False, 0

    # ==========================================
    # 4. 內部暫存器規劃 (VRF Allocation - 64x64 Double Buffer)
    # ==========================================
    csr.Enable_Double_Buffer = True
    csr.MatA_reg_base, csr.VREG_stride_A = 0, 2
    csr.MatB_reg_base, csr.VREG_stride_B = 4, 2
    csr.MatC_reg_base, csr.VREG_stride_C = 8, 4

    # 清空未使用的內部暫存器設定
    csr.MatD_reg_base, csr.VREG_stride_D = 0, 0
    csr.MatE_reg_base, csr.VREG_stride_E = 0, 0
    csr.Temp_reg_base, csr.VREG_stride_O = 0, 0

def set_flash_attn_csr(csr: CSRConfig, Q_base, K_base, V_base, Out_base, 
                       stride_A, stride_B, stride_D, stride_C, 
                       m_tile, n_tile, head_dim, seq_len):
    """ Helper to populate standard FlashAttention CSR parameters """
    csr.M_tile, csr.N_tile, csr.K_tile = m_tile, n_tile, head_dim
    csr.K_total, csr.N_total = head_dim, seq_len
    
    # Set External Memory
    csr.Mem_Base_A, csr.Mem_Base_B, csr.Mem_Base_D, csr.Mem_Base_C = Q_base, K_base, V_base, Out_base
    csr.Mem_Stride_A, csr.Mem_Stride_B, csr.Mem_Stride_D, csr.Mem_Stride_C = stride_A, stride_B, stride_D, stride_C
    
    # 2D Tile Scatter/Gather 自動推導
    csr.Is_Gather_A  = True; csr.BLOCK_LEN_A = head_dim
    csr.Is_Gather_B  = True; csr.BLOCK_LEN_B = head_dim
    csr.Is_Scatter_C = True; csr.BLOCK_LEN_C = head_dim
    csr.Is_Gather_D  = True; csr.BLOCK_LEN_D = head_dim
    
    # 32x32 Flash Attention 雙緩衝專用 VRF 配置
    csr.MatA_reg_base, csr.VREG_stride_A = 0, 2    
    csr.MatB_reg_base, csr.VREG_stride_B = 2, 2    
    csr.MatD_reg_base, csr.VREG_stride_D = 6, 2    
    csr.MatE_reg_base, csr.VREG_stride_E = 10, 1   
    csr.MatC_reg_base, csr.VREG_stride_O = 16, 8   
    
    csr.Enable_Double_Buffer = True
    csr.Act_Type = ActivationType.NONE
    csr.Macro_Op_Name = "FLASH_ATTN"

def set_res_ln_csr(csr: CSRConfig, A_base, B_base, C_base, 
                   stride_A, stride_B, stride_C, 
                   m_tile, k_total):
    """ Helper to populate standard Residual Add + LayerNorm CSR parameters """
    csr.M_tile, csr.K_total = m_tile, k_total
    csr.K_tile = k_total # TODO check the logic
    csr.N_tile, csr.N_total = 0, 0 # Not used in 1D-LN
    
    # Set External Memory
    csr.Mem_Base_A, csr.Mem_Base_B, csr.Mem_Base_C, csr.Mem_Base_D = A_base, B_base, C_base, 0
    csr.Mem_Stride_A, csr.Mem_Stride_B, csr.Mem_Stride_C, csr.Mem_Stride_D = stride_A, stride_B, stride_C, 0
    
    # 2D Tile Scatter/Gather 自動推導
    csr.Is_Gather_A  = True; csr.BLOCK_LEN_A = k_total
    csr.Is_Gather_B  = True; csr.BLOCK_LEN_B = k_total
    csr.Is_Scatter_C = True; csr.BLOCK_LEN_C = k_total
    csr.Is_Gather_D  = False; csr.BLOCK_LEN_D = 0
    
    # LayerNorm 專用 VRF 配置
    csr.MatA_reg_base = 0; csr.VREG_stride_A = 2  # TODO check the logic
    csr.MatB_reg_base = 4; csr.VREG_stride_B = 2  # TODO check the logic
    csr.MatC_reg_base = 8; csr.VREG_stride_C = 8  # TODO check the logic
    csr.MatD_reg_base = 0; csr.VREG_stride_D = 0
    csr.MatE_reg_base = 0; csr.VREG_stride_E = 0
    
    
    csr.Enable_Double_Buffer = False
    csr.Act_Type = ActivationType.NONE
    csr.Macro_Op_Name = "RES_LN"


def build_bert_base_layer(sim: ADHD_VPU, csr: CSRConfig, tensorHW: TensorConfig, latencySet: LatencySet, 
                          seq_len: int, layer_idx: int, hidden_state_in: int, hidden_state_out: int, 
                          mem_mgr: MemoryManager, weights: dict):
    """
    Procedure of BERT-base layer
    1. Q, K, V Projections
    2. 12 Heads FlashAttention
    3. O Projection
    4. AddNorm 1
    5. FFN1 (GELU)
    6. FFN2
    7. AddNorm 2
    """
    D = 768
    D_FFN = 3072
    head_dim = D // 12  # 64
    
    print(f"  -> Dispatching Layer {layer_idx+1}/12 ...")

    # =====================================================================
    # 1. Q, K, V Projections
    # Shape: [seq_len, D] * [D, D] -> [seq_len, D]
    # =====================================================================
    Q_out = mem_mgr.allocate(seq_len * D); W_Q = weights["W_Q"]
    K_out = mem_mgr.allocate(seq_len * D); W_K = weights["W_K"]
    V_out = mem_mgr.allocate(seq_len * D); W_V = weights["W_V"]

    for name, out_ptr, w_ptr in [("Q", Q_out, W_Q), ("K", K_out, W_K), ("V", V_out, W_V)]: # Q, K, V
        for m in range(0, seq_len, 64):
            for n in range(0, D, 64):
                set_gemm_csr(csr, hidden_state_in + (m*D), w_ptr + n, out_ptr + (m*D) + n, 
                                  D, D, D, 
                                  64, 64, 32, 
                                  D, ActivationType.NONE)
                sim.fetch_macro([MacroOp(f"L{layer_idx}_PROJ_{name}_m{m}_n{n}", macro_gemm_template, {"csr":copy.deepcopy(csr), "tensor": tensorHW, "latency": latencySet})])

    # =====================================================================
    # 2. Multi-Head Flash Attention (12 Head)
    # Shape: Q, K, V -> [seq_len, head_Dim], Context -> [seq_len, head_Dim]
    # =====================================================================
    Attn_out = mem_mgr.allocate(seq_len * D) 
    
    for h in range(12):
        for q_start in range(0, seq_len, 32):
            # Address Generation for Heads
            Q_base = Q_out + (q_start * D) + (h * head_dim)
            K_base = K_out + (h * head_dim)
            V_base = V_out + (h * head_dim)
            O_base = Attn_out + (q_start * D) + (h * head_dim)
            
            set_flash_attn_csr(csr, Q_base, K_base, V_base, O_base, 
                                    D, D, D, D, 
                                    32, 32, head_dim, 
                                    seq_len)
            sim.fetch_macro([MacroOp(f"L{layer_idx}_FLASH_ATTN_H{h}_q{q_start}", macro_flash_attn_template, {"csr":copy.deepcopy(csr), "tensor": tensorHW, "latency": latencySet, "vpu":sim})])

    # =====================================================================
    # 3. Attention Output Projection
    # Shape: [seq_len, D] * [D, D] -> [seq_len, D]
    # =====================================================================
    O_proj_out = mem_mgr.allocate(seq_len * D); W_O = weights["W_O"]
    for m in range(0, seq_len, 64):
        for n in range(0, D, 64):
            set_gemm_csr(csr, Attn_out + (m*D), W_O + n, O_proj_out + (m*D) + n, 
                              D, D, D, 
                              64, 64, 32, 
                              D, ActivationType.NONE)
            sim.fetch_macro([MacroOp(f"L{layer_idx}_ATTN_OUT_m{m}_n{n}", macro_gemm_template, {"csr":copy.deepcopy(csr), "tensor": tensorHW, "latency": latencySet})])

    # =====================================================================
    # 4. Residual Add + LayerNorm 1
    # Shape: I[seq_len, D] + Att[seq_len, D] = RES1[seq_len, D]
    # =====================================================================
    Norm1_out = mem_mgr.allocate(seq_len * D)
    for m in range(0, seq_len, 64):
        A_base = hidden_state_in + (m * D) # Main Branch (原始輸入)
        B_base = O_proj_out + (m * D)      # Residual Branch
        C_base = Norm1_out + (m * D)       # Output
        
        set_res_ln_csr(csr, A_base, B_base, C_base, 
                            D, D, D, 
                            64, D)
        sim.fetch_macro([MacroOp(f"L{layer_idx}_RES_LN1_m{m}", macro_residual_layernorm_template, {"csr":copy.deepcopy(csr), "latency": latencySet})])

    # =====================================================================
    # 5. FFN 1 (GEMM + GELU) -> dimension scale up to 3072
    # Shape: [seq_len, D] * [D, D_FFN] -> [seq_len, D_FFN]
    # =====================================================================
    FFN1_out = mem_mgr.allocate(seq_len * D_FFN); W_F1 = weights["W_1"]
    for m in range(0, seq_len, 64):
        for n in range(0, D_FFN, 64):
            set_gemm_csr(csr, Norm1_out + (m*D), W_F1 + n, FFN1_out + (m*D_FFN) + n, 
                              D, D_FFN, D_FFN, 
                              64, 64, 32, 
                              D, act=ActivationType.GELU)
            sim.fetch_macro([MacroOp(f"L{layer_idx}_FFN1_GELU_m{m}_n{n}", macro_gemm_template, {"csr":copy.deepcopy(csr), "tensor": tensorHW, "latency": latencySet})])

    # =====================================================================
    # 6. FFN 2 -> dimension scale down to 768
    # Shape: [seq_len, D_FFN] * [D_FFN, D] -> [seq_len, D]
    # =====================================================================
    FFN2_out = mem_mgr.allocate(seq_len * D); W_F2 = weights["W_2"]
    for m in range(0, seq_len, 64):
        for n in range(0, D, 64):
            set_gemm_csr(csr, FFN1_out + (m*D_FFN), W_F2 + n, FFN2_out + (m*D) + n, 
                              D_FFN, D, D, 
                              64, 64, 32, 
                              D_FFN, ActivationType.NONE)
            sim.fetch_macro([MacroOp(f"L{layer_idx}_FFN2_m{m}_n{n}", macro_gemm_template, {"csr":copy.deepcopy(csr), "tensor": tensorHW, "latency": latencySet})])

    # =====================================================================
    # 7. Residual Add + LayerNorm 2 (write back to hidden_state_out)
    # Shape: RES1[seq_len, D] + FFN2[seq_len, D] = RES2[seq_len, D]
    # =====================================================================
    for m in range(0, seq_len, 64):
        A_base = Norm1_out + (m * D)        # Main Branch
        B_base = FFN2_out + (m * D)         # Residual Branch
        C_base = hidden_state_out + (m * D) # Output (成為下一層的輸入)
        
        set_res_ln_csr(csr, A_base, B_base, C_base, 
                            D, D, D,
                            64, D)
        sim.fetch_macro([MacroOp(f"L{layer_idx}_RES_LN2_m{m}", macro_residual_layernorm_template, {"csr":copy.deepcopy(csr), "latency": latencySet})])

def build_bert_base_model(sim: ADHD_VPU, csr: CSRConfig, tensorHW: TensorConfig, latencySet: LatencySet, seq_len: int, mem_mgr: MemoryManager):
    """
    【BERT Base 全模型調度器】
    執行完整的 12 層 BERT Base 模型。利用 Ping-Pong Workspace 概念節省記憶體空間。
    """
    print(f"\n{'='*60}")
    print(f"🚀 初始化 BERT Base 模型 (12 Layers, Seq Length: {seq_len})")
    print(f"{'='*60}")
    
    D = 768
    D_FFN = 3072

    # 1. for weight size alignment
    def align64(size):
        return (size + 63) & ~63

    # 2. Static Mememory Allocation (Weight, Ping-Pong)
    ADDR_BASE = mem_mgr.start_addr
    # Accurate weight allocation
    MEM_WQ = ADDR_BASE
    MEM_WK = MEM_WQ + align64(D * D)
    MEM_WV = MEM_WK + align64(D * D)
    MEM_WO = MEM_WV + align64(D * D)
    MEM_W1 = MEM_WO + align64(D * D)
    MEM_W2 = MEM_W1 + align64(D * D_FFN)
    # ping-pong buffer of input and output
    MEM_PING = MEM_W2 + align64(D_FFN * D)
    MEM_PONG = MEM_PING + align64(seq_len * D)
    # Intermediate tensor
    SCRATCHPAD_BASE = MEM_PONG + align64(seq_len * D)
    
    # Packed the weight into MemoryManager
    weights = {
        "W_Q": MEM_WQ, "W_K": MEM_WK, "W_V": MEM_WV,
        "W_O": MEM_WO, "W_1": MEM_W1, "W_2": MEM_W2
    }

    # current address point to volatile memory region
    mem_mgr.current_addr = SCRATCHPAD_BASE
    
    for layer in range(12):
        # 決定當前層的輸入與輸出在哪個 Workspace (Ping-Pong)
        hidden_in = MEM_PING if layer % 2 == 0 else MEM_PONG
        hidden_out = MEM_PONG if layer % 2 == 0 else MEM_PING
        
        # 紀錄當前記憶體游標，以便層結束後釋放中間的 activations
        mem_checkpoint = mem_mgr.current_addr
        
        # 呼叫該層的排程器
        build_bert_base_layer(sim, csr, tensorHW, latencySet, seq_len, layer, hidden_in, hidden_out, mem_mgr, weights)
        
        # 層結束，釋放 Q,K,V 等中間變數，防止 OOM (模擬軟體 Compiler 的 Liveness Analysis)
        mem_mgr.current_addr = mem_checkpoint

    print("\n✅ 所有 12 層巨集指令派發完成！進入硬體 FSM 模擬...")

def build_vit_base_layer(sim: ADHD_VPU, csr: CSRConfig, tensorHW: TensorConfig, latencySet: LatencySet, 
                         seq_len: int, layer_idx: int, hidden_in: int, hidden_out: int, mem_mgr: MemoryManager):
    """
    【不偷懶的完整單層 ViT Base】
    Sequence Length 通常為 197。採用 Pre-LN 架構。
    所有外層迴圈皆由軟體 CPU 展開，硬體僅執行 1D-Decoupled Macro-OP。
    """
    D = 768
    D_FFN = 3072
    head_dim = D // 12  # 64
    
    print(f"  -> Dispatching ViT Layer {layer_idx+1}/12 ...")

    # =====================================================================
    # 1. Pre-LayerNorm 1 (影像特徵輸入)
    # =====================================================================
    LN1_out = mem_mgr.allocate(seq_len * D)
    for m in range(0, seq_len, 64):
        csr.M_tile = 64; csr.K_total = D
        csr.Mem_Base_A = hidden_in + (m * D)
        # 模擬 Pre-LN: 假設 B 也是 hidden_in (或 Zero Buffer) 
        csr.Mem_Base_B = hidden_in + (m * D) 
        csr.Mem_Base_C = LN1_out + (m * D)
        csr.Mem_Stride_A = csr.Mem_Stride_B = csr.Mem_Stride_C = D
        csr.MatA_reg_base, csr.MatB_reg_base, csr.MatC_reg_base = 0, 4, 8
        csr.VREG_stride_C = 4
        csr.Enable_Double_Buffer = False
        sim.fetch_macro([MacroOp(f"L{layer_idx}_VIT_PRE_LN1_m{m}", macro_residual_layernorm_template, {"csr":copy.deepcopy(csr), "latency": latencySet})])

    # =====================================================================
    # 2. Q, K, V Projections (線性投影)
    # =====================================================================
    Q_out = mem_mgr.allocate(seq_len * D); W_Q = mem_mgr.allocate(D * D)
    K_out = mem_mgr.allocate(seq_len * D); W_K = mem_mgr.allocate(D * D)
    V_out = mem_mgr.allocate(seq_len * D); W_V = mem_mgr.allocate(D * D)

    for name, out_ptr, w_ptr in [("Q", Q_out, W_Q), ("K", K_out, W_K), ("V", V_out, W_V)]:
        for m in range(0, seq_len, 64):
            for n in range(0, D, 64):
                set_gemm_csr(csr, LN1_out + (m*D), w_ptr + n, out_ptr + (m*D) + n, D, D, D, 64, 64, 32, D)
                sim.fetch_macro([MacroOp(f"L{layer_idx}_VIT_PROJ_{name}_m{m}_n{n}", macro_gemm_template, {"csr":copy.deepcopy(csr), "tensor": tensorHW, "latency": latencySet})])

    # =====================================================================
    # 3. Multi-Head Flash Attention (12 Heads)
    # =====================================================================
    Attn_out = mem_mgr.allocate(seq_len * D) 
    
    for h in range(12):
        for q_start in range(0, seq_len, 32):
            csr.M_tile, csr.N_tile, csr.K_total, csr.N_total = 32, 32, head_dim, seq_len
            
            csr.Mem_Base_A = Q_out + (q_start * D) + (h * head_dim)
            csr.Mem_Base_B = K_out + (h * head_dim)
            csr.Mem_Base_D = V_out + (h * head_dim)
            csr.Mem_Base_C = Attn_out + (q_start * D) + (h * head_dim)
            csr.Mem_Stride_A = csr.Mem_Stride_B = csr.Mem_Stride_D = csr.Mem_Stride_C = D
            
            # 32x32 Ping-Pong Allocation
            csr.MatA_reg_base, csr.VREG_stride_A = 0, 2    
            csr.MatB_reg_base, csr.VREG_stride_B = 2, 2    
            csr.MatD_reg_base, csr.VREG_stride_D = 6, 2    
            csr.MatE_reg_base, csr.VREG_stride_E = 10, 1   
            csr.MatC_reg_base, csr.VREG_stride_O = 16, 8   
            csr.Enable_Double_Buffer = True
            
            sim.fetch_macro([MacroOp(f"L{layer_idx}_VIT_FLASH_ATTN_H{h}_q{q_start}", macro_flash_attn_template, {"csr":copy.deepcopy(csr), "tensor": tensorHW, "latency": latencySet, "vpu":sim})])

    # =====================================================================
    # 4. Attention Output Projection
    # =====================================================================
    Attn_proj_out = mem_mgr.allocate(seq_len * D); W_O = mem_mgr.allocate(D * D)
    for m in range(0, seq_len, 64):
        for n in range(0, D, 64):
            set_gemm_csr(csr, Attn_out + (m*D), W_O + n, Attn_proj_out + (m*D) + n, D, D, D, 64, 64, 32, D)
            sim.fetch_macro([MacroOp(f"L{layer_idx}_VIT_ATTN_OUT_m{m}_n{n}", macro_gemm_template, {"csr":copy.deepcopy(csr), "tensor": tensorHW, "latency": latencySet})])

    # =====================================================================
    # 5. Pre-LayerNorm 2 (此處包含 Residual 1 的 ADD)
    # ViT 架構： x = x + Attn(LN1(x)) -> 然後做 LN2(x)
    # 我們利用 residual macro 同時處理 Add + LN
    # =====================================================================
    LN2_out = mem_mgr.allocate(seq_len * D)
    for m in range(0, seq_len, 64):
        csr.M_tile = 64; csr.K_total = D
        csr.Mem_Base_A = hidden_in + (m * D)       # 原始特徵 (Residual)
        csr.Mem_Base_B = Attn_proj_out + (m * D)   # Attention 輸出 (Main)
        csr.Mem_Base_C = LN2_out + (m * D)
        csr.Mem_Stride_A = csr.Mem_Stride_B = csr.Mem_Stride_C = D
        csr.MatA_reg_base, csr.MatB_reg_base, csr.MatC_reg_base = 0, 4, 8
        csr.VREG_stride_C = 4
        csr.Enable_Double_Buffer = False
        sim.fetch_macro([MacroOp(f"L{layer_idx}_VIT_RES_LN2_m{m}", macro_residual_layernorm_template, {"csr":copy.deepcopy(csr), "latency": latencySet})])

    # =====================================================================
    # 6. MLP 1 (GEMM + GELU)
    # 亮點：不再使用舊的 gemm_gelu_template，直接用統一的 gemm_template + Act_Type
    # =====================================================================
    MLP1_out = mem_mgr.allocate(seq_len * D_FFN); W_M1 = mem_mgr.allocate(D * D_FFN)
    for m in range(0, seq_len, 64):
        for n in range(0, D_FFN, 64):
            set_gemm_csr(csr, LN2_out + (m*D), W_M1 + n, MLP1_out + (m*D_FFN) + n, D, D_FFN, D_FFN, 64, 64, 32, D, act=ActivationType.GELU)
            sim.fetch_macro([MacroOp(f"L{layer_idx}_VIT_MLP1_GELU_m{m}_n{n}", macro_gemm_template, {"csr":copy.deepcopy(csr), "tensor": tensorHW, "latency": latencySet})])

    # =====================================================================
    # 7. MLP 2 -> 寫回 hidden_out
    # =====================================================================
    W_M2 = mem_mgr.allocate(D_FFN * D)
    for m in range(0, seq_len, 64):
        for n in range(0, D, 64):
            # 寫入 hidden_out，作為下一層的輸入
            set_gemm_csr(csr, MLP1_out + (m*D_FFN), W_M2 + n, hidden_out + (m*D) + n, D_FFN, D, D, 64, 64, 32, D_FFN)
            sim.fetch_macro([MacroOp(f"L{layer_idx}_VIT_MLP2_m{m}_n{n}", macro_gemm_template, {"csr":copy.deepcopy(csr), "tensor": tensorHW, "latency": latencySet})])

def build_vit_base_model(sim: ADHD_VPU, csr: CSRConfig, tensorHW: TensorConfig, latencySet: LatencySet, seq_len: int, mem_mgr: MemoryManager):
    """
    【ViT Base 全模型調度器】
    執行 12 層 Vision Transformer。使用 Workspace Ping-Pong 來節省記憶體。
    """
    print(f"\n{'='*60}")
    print(f"🖼️ 初始化 ViT Base 模型 (12 Layers, Seq Length: {seq_len})")
    print(f"{'='*60}")
    
    D = 768
    hidden_workspace_0 = mem_mgr.allocate(seq_len * D)
    hidden_workspace_1 = mem_mgr.allocate(seq_len * D)
    
    for layer in range(12):
        hidden_in = hidden_workspace_0 if layer % 2 == 0 else hidden_workspace_1
        hidden_out = hidden_workspace_1 if layer % 2 == 0 else hidden_workspace_0
        
        mem_checkpoint = mem_mgr.current_addr
        build_vit_base_layer(sim, csr, tensorHW, latencySet, seq_len, layer, hidden_in, hidden_out, mem_mgr)
        mem_mgr.current_addr = mem_checkpoint

    print("\n✅ 所有 12 層 ViT 巨集指令派發完成！進入硬體 FSM 模擬...")

def build_gpt2_base_layer(sim: ADHD_VPU, csr: CSRConfig, tensorHW: TensorConfig, latencySet: LatencySet, 
                          seq_len: int, layer_idx: int, hidden_in: int, hidden_out: int, mem_mgr: MemoryManager,
                          sparsity_ratio: float = 0.5):
    """
    【不偷懶的完整單層 GPT-2 Base (Prefill)】
    核心亮點：Causal Masking 由 CPU 動態調整 N_total 達成，硬體完全無需修改！
    """
    D = 768
    D_FFN = 3072
    head_dim = D // 12  # 64
    
    print(f"  -> Dispatching GPT-2 Layer {layer_idx+1}/12 (Prefill Phase) ...")

    # =====================================================================
    # 1. Pre-LayerNorm 1
    # =====================================================================
    LN1_out = mem_mgr.allocate(seq_len * D)
    for m in range(0, seq_len, 64):
        csr.M_tile = 64; csr.K_total = D
        csr.Mem_Base_A = hidden_in + (m * D)
        csr.Mem_Base_B = hidden_in + (m * D) # 模擬 Pre-LN
        csr.Mem_Base_C = LN1_out + (m * D)
        csr.Mem_Stride_A = csr.Mem_Stride_B = csr.Mem_Stride_C = D
        csr.MatA_reg_base, csr.MatB_reg_base, csr.MatC_reg_base = 0, 4, 8
        csr.VREG_stride_C = 4
        csr.Enable_Double_Buffer = False
        sim.fetch_macro([MacroOp(f"L{layer_idx}_GPT_PRE_LN1_m{m}", macro_residual_layernorm_template, {"csr":copy.deepcopy(csr), "latency": latencySet})])

    # =====================================================================
    # 2. Q, K, V Projections (線性投影)
    # =====================================================================
    Q_out = mem_mgr.allocate(seq_len * D); W_Q = mem_mgr.allocate(D * D)
    K_out = mem_mgr.allocate(seq_len * D); W_K = mem_mgr.allocate(D * D)
    V_out = mem_mgr.allocate(seq_len * D); W_V = mem_mgr.allocate(D * D)

    for name, out_ptr, w_ptr in [("Q", Q_out, W_Q), ("K", K_out, W_K), ("V", V_out, W_V)]:
        for m in range(0, seq_len, 64):
            for n in range(0, D, 64):
                set_gemm_csr(csr, LN1_out + (m*D), w_ptr + n, out_ptr + (m*D) + n, D, D, D, 64, 64, 32, D)
                sim.fetch_macro([MacroOp(f"L{layer_idx}_GPT_PROJ_{name}_m{m}_n{n}", macro_gemm_template, {"csr":copy.deepcopy(csr), "tensor": tensorHW, "latency": latencySet})])

    # =====================================================================
    # 🌟 2.5 LSH Hashing (模擬 NPU 計算 Hash 的「硬體代價」)
    # =====================================================================
    b_hash = 32 # 壓成 32-bit Hash ID
    R_ptr = mem_mgr.allocate(D * b_hash)
    Hash_Q_ptr = mem_mgr.allocate(seq_len * b_hash)
    Hash_K_ptr = mem_mgr.allocate(seq_len * b_hash)

    # 發射 LSH 巨集，讓 NPU 消耗 Cycle 並產生 SRAM 讀寫流量
    for m in range(0, seq_len, 64):
        set_gemm_csr(csr, Q_out + (m*D), R_ptr, Hash_Q_ptr + (m*b_hash), D, b_hash, b_hash, 64, b_hash, 32, D)
        sim.fetch_macro([MacroOp(f"L{layer_idx}_LSH_Q_m{m}", macro_lsh_template, {"csr":copy.deepcopy(csr), "tensor": tensorHW, "latency": latencySet})])

    for m in range(0, seq_len, 64):
        set_gemm_csr(csr, K_out + (m*D), R_ptr, Hash_K_ptr + (m*b_hash), D, b_hash, b_hash, 64, b_hash, 32, D)
        sim.fetch_macro([MacroOp(f"L{layer_idx}_LSH_K_m{m}", macro_lsh_template, {"csr":copy.deepcopy(csr), "tensor": tensorHW, "latency": latencySet})])

    print(f"   [CPU] LSH completed. CPU is grouping Tokens by Hash ID to build Sparse Attention schedule...")

    # =====================================================================
    # 3. 核心：Attention (12 Heads)
    # - Causal Mask
    # - Sparse Mask
    # - FlashAttention
    # =====================================================================
    Attn_out = mem_mgr.allocate(seq_len * D) 
    
    for h in range(12): # 12 個 Head 跑好跑滿
        for q_start in range(0, seq_len, 32):
            csr.M_tile, csr.N_tile, csr.K_total = 32, 32, head_dim
            
            # ✨ 【軟體定義的 Causal Masking (因果遮罩)】 ✨
            # 硬體 K, V 迴圈只會掃描到 q_start + M_tile 的位置，未來的 Token 絕對不會被讀進 VPU！
            csr.N_total = min(seq_len, q_start + csr.M_tile)

            # 🌟 [新增] 第二層防禦：Content-Based Dynamic Sparsity (軟體排程)
            sparse_mask = 0
            valid_content_blocks = 0

            # 模擬 CPU 在發射前，透過 Hash ID 產生的 Block 遮罩
            for k_start in range(0, csr.N_total, 32):
                is_valid = True
                
                # 對於對角線 (Diagonal) Block，因為包含自己，一定要保留以確保數值穩定性
                if k_start == q_start:
                    is_valid = True
                else:
                    # 對於過去的 Block，根據 sparsity_ratio 隨機剔除 (模擬 LSH 效果)
                    if random.random() < sparsity_ratio:
                        is_valid = False
                
                if is_valid:
                    block_num = k_start // 32
                    sparse_mask |= (1 << block_num)  # 將該 bit 設為 1
                    valid_content_blocks += 1

            # 將算好的 Mask 寫入 CSR 交給硬體
            csr.Sparse_Mask = sparse_mask

            # 📊 [頻寬節省量計算]
            bytes_per_element = 1
            dense_kv_bytes = (seq_len * head_dim * bytes_per_element) * 2  # 無 Mask: 完整讀取 N 個
            causal_kv_bytes = (csr.N_total * head_dim * bytes_per_element) * 2 # 只有 Causal Mask
            final_kv_bytes = (valid_content_blocks * 32 * head_dim * bytes_per_element) * 2 # Causal + Dynamic Mask
            
            # 累加統計數據
            sim.analyzer.dense_kv_bytes_if_no_mask += dense_kv_bytes
            sim.analyzer.causal_mask_skipped_bytes += (dense_kv_bytes - causal_kv_bytes)
            sim.analyzer.content_sparsity_skipped_bytes += (causal_kv_bytes - final_kv_bytes)
            
            # setup the CSR
            csr.Mem_Base_A = Q_out + (q_start * D) + (h * head_dim)
            csr.Mem_Base_B = K_out + (h * head_dim)
            csr.Mem_Base_D = V_out + (h * head_dim)
            csr.Mem_Base_C = Attn_out + (q_start * D) + (h * head_dim)
            csr.Mem_Stride_A = csr.Mem_Stride_B = csr.Mem_Stride_D = csr.Mem_Stride_C = D
            
            # 32x32 Flash Attention 雙緩衝專用 VRF 配置
            csr.MatA_reg_base, csr.VREG_stride_A = 0, 2    
            csr.MatB_reg_base, csr.VREG_stride_B = 2, 2    
            csr.MatD_reg_base, csr.VREG_stride_D = 6, 2    
            csr.MatE_reg_base, csr.VREG_stride_E = 10, 1   
            csr.MatC_reg_base, csr.VREG_stride_O = 16, 8   
            csr.Enable_Double_Buffer = True
            
            sim.fetch_macro([MacroOp(f"L{layer_idx}_GPT_CAUSAL_ATTN_H{h}_q{q_start}", macro_flash_attn_template, {"csr":copy.deepcopy(csr), "tensor": tensorHW, "latency": latencySet, "vpu":sim})])

    # =====================================================================
    # 4. Attention Output Projection
    # =====================================================================
    Attn_proj_out = mem_mgr.allocate(seq_len * D); W_O = mem_mgr.allocate(D * D)
    for m in range(0, seq_len, 64):
        for n in range(0, D, 64):
            set_gemm_csr(csr, Attn_out + (m*D), W_O + n, Attn_proj_out + (m*D) + n, D, D, D, 64, 64, 32, D)
            sim.fetch_macro([MacroOp(f"L{layer_idx}_GPT_ATTN_OUT_m{m}_n{n}", macro_gemm_template, {"csr":copy.deepcopy(csr), "tensor": tensorHW, "latency": latencySet})])

    # =====================================================================
    # 5. Pre-LayerNorm 2 (包含 Residual 1 的 ADD)
    # =====================================================================
    LN2_out = mem_mgr.allocate(seq_len * D)
    for m in range(0, seq_len, 64):
        csr.M_tile = 64; csr.K_total = D
        csr.Mem_Base_A = hidden_in + (m * D)       # 原始輸入 (Residual)
        csr.Mem_Base_B = Attn_proj_out + (m * D)   # Attention 輸出 (Main)
        csr.Mem_Base_C = LN2_out + (m * D)
        csr.Mem_Stride_A = csr.Mem_Stride_B = csr.Mem_Stride_C = D
        csr.MatA_reg_base, csr.MatB_reg_base, csr.MatC_reg_base = 0, 4, 8
        csr.VREG_stride_C = 4
        csr.Enable_Double_Buffer = False
        sim.fetch_macro([MacroOp(f"L{layer_idx}_GPT_RES_LN2_m{m}", macro_residual_layernorm_template, {"csr":copy.deepcopy(csr), "latency": latencySet})])

    # =====================================================================
    # 6. MLP 1 (GEMM + GELU)
    # =====================================================================
    MLP1_out = mem_mgr.allocate(seq_len * D_FFN); W_M1 = mem_mgr.allocate(D * D_FFN)
    for m in range(0, seq_len, 64):
        for n in range(0, D_FFN, 64):
            set_gemm_csr(csr, LN2_out + (m*D), W_M1 + n, MLP1_out + (m*D_FFN) + n, D, D_FFN, D_FFN, 64, 64, 32, D, act=ActivationType.GELU)
            sim.fetch_macro([MacroOp(f"L{layer_idx}_GPT_MLP1_GELU_m{m}_n{n}", macro_gemm_template, {"csr":copy.deepcopy(csr), "tensor": tensorHW, "latency": latencySet})])

    # =====================================================================
    # 7. MLP 2 -> 寫入 hidden_out
    # =====================================================================
    W_M2 = mem_mgr.allocate(D_FFN * D)
    for m in range(0, seq_len, 64):
        for n in range(0, D, 64):
            set_gemm_csr(csr, MLP1_out + (m*D_FFN), W_M2 + n, hidden_out + (m*D) + n, D_FFN, D, D, 64, 64, 32, D_FFN)
            sim.fetch_macro([MacroOp(f"L{layer_idx}_GPT_MLP2_m{m}_n{n}", macro_gemm_template, {"csr":copy.deepcopy(csr), "tensor": tensorHW, "latency": latencySet})])

def build_gpt2_base_model(sim: ADHD_VPU, csr: CSRConfig, tensorHW: TensorConfig, latencySet: LatencySet, seq_len: int, mem_mgr: MemoryManager):
    """
    【GPT-2 Base 全模型調度器 (Prefill Phase)】
    """
    print(f"\n{'='*60}")
    print(f"🤖 初始化 GPT-2 Base 模型 (12 Layers, Context Length: {seq_len})")
    print(f"{'='*60}")
    
    D = 768
    hidden_workspace_0 = mem_mgr.allocate(seq_len * D)
    hidden_workspace_1 = mem_mgr.allocate(seq_len * D)
    
    for layer in range(12):
        hidden_in = hidden_workspace_0 if layer % 2 == 0 else hidden_workspace_1
        hidden_out = hidden_workspace_1 if layer % 2 == 0 else hidden_workspace_0
        
        mem_checkpoint = mem_mgr.current_addr
        build_gpt2_base_layer(sim, csr, tensorHW, latencySet, seq_len, layer, hidden_in, hidden_out, mem_mgr)
        mem_mgr.current_addr = mem_checkpoint

    print("\n✅ 所有 12 層 GPT-2 巨集指令派發完成！進入硬體 FSM 模擬...")

def build_subOP(sim: ADHD_VPU, csr: CSRConfig, tensorHW: TensorConfig, latencySet: LatencySet, mem_mgr: MemoryManager):
    print(f"\n Sub Macro OP for test ---")

    seq_len = 512
    Hidden_Dim = 768
    head_Dim = Hidden_Dim // 12
    FFN_Dim = 3072

    # """ Attention WithOut PingPong"""
    # # set tile dimensions
    # csr.M_tile, csr.N_tile, csr.K_tile = 64, 64, 64

    # # set External Memory
    # csr.Mem_Base_A = mem_mgr.allocate(seq_len * head_Dim);   csr.Mem_Stride_A = Hidden_Dim  # Q
    # csr.Mem_Base_B = mem_mgr.allocate(head_Dim * seq_len);   csr.Mem_Stride_B = Hidden_Dim  # K
    # csr.Mem_Base_D = mem_mgr.allocate(seq_len * head_Dim);   csr.Mem_Stride_D = Hidden_Dim  # V
    # csr.Mem_Base_C = mem_mgr.allocate(seq_len * Hidden_Dim); csr.Mem_Stride_C = Hidden_Dim  # Output Head

    # # Memory Access Modes (Scatter/Gather)
    # csr.Is_Gather_A,  csr.BLOCK_LEN_A  = True, csr.K_tile
    # csr.Is_Gather_B,  csr.BLOCK_LEN_B  = True, csr.N_tile
    # csr.Is_Gather_D,  csr.BLOCK_LEN_D  = True, csr.N_tile
    # csr.Is_Scatter_C, csr.BLOCK_LEN_C  = True, csr.N_tile
    
    # # set VREG Mapping
    # csr.MatA_reg_base, csr.VREG_stride_A = 0, 4    # MatQ_tile VREG0 ~ VREG3
    # csr.MatB_reg_base, csr.VREG_stride_B = 4, 4    # MatK_tile VREG4 ~ VREG7
    # csr.MatD_reg_base, csr.VREG_stride_D = 8, 4    # MatV_tile VREG8 ~ VREG11
    # csr.MatE_reg_base, csr.VREG_stride_E = 12, 4   # MatP_tile VREG12 ~ VREG15
    # csr.MatC_reg_base, csr.VREG_stride_O = 16, 16  # MatOutput VREG16 ~ VREG31
    # csr.Enable_Double_Buffer, csr.Act_Type = False, ActivationType.NONE

    # # Execution
    # csr.Macro_Op_Name = "Attention"
    # csr.M_total, csr.N_total, csr.K_total = seq_len, seq_len, head_Dim
    # sim.fetch_macro([MacroOp("Attention", macro_flash_attn_template, {"csr":copy.deepcopy(csr), "tensor": tensorHW, "latency": latencySet, "vpu":sim})])


    """ Attention With PingPong"""
    # set tile dimensions
    csr.M_tile, csr.N_tile, csr.K_tile = 32, 32, 32

    # set External Memory
    csr.Mem_Base_A = mem_mgr.allocate(seq_len * head_Dim);   csr.Mem_Stride_A = Hidden_Dim  # Q
    csr.Mem_Base_B = mem_mgr.allocate(head_Dim * seq_len);   csr.Mem_Stride_B = Hidden_Dim  # K
    csr.Mem_Base_D = mem_mgr.allocate(seq_len * head_Dim);   csr.Mem_Stride_D = Hidden_Dim  # V
    csr.Mem_Base_C = mem_mgr.allocate(seq_len * Hidden_Dim); csr.Mem_Stride_C = Hidden_Dim  # Output Head

    # Memory Access Modes (Scatter/Gather)
    csr.Is_Gather_A,  csr.BLOCK_LEN_A  = True, csr.K_tile
    csr.Is_Gather_B,  csr.BLOCK_LEN_B  = True, csr.N_tile
    csr.Is_Gather_D,  csr.BLOCK_LEN_D  = True, csr.N_tile
    csr.Is_Scatter_C, csr.BLOCK_LEN_C  = True, csr.N_tile
    
    # set VREG Mapping
    csr.MatA_reg_base, csr.VREG_stride_A = 0, 2    # Q: v0~v1 (stride 2)
    csr.MatB_reg_base, csr.VREG_stride_B = 2, 2    # K: v2~v5 (stride 2, ping-pong)
    csr.MatD_reg_base, csr.VREG_stride_D = 6, 2    # V: v6~v9 (stride 2, ping-pong)
    csr.MatE_reg_base, csr.VREG_stride_E = 10, 1   # P: v10 (stride 1)
    csr.MatC_reg_base, csr.VREG_stride_O = 16, 8   # O: v16~v23 (stride 8)
    csr.Enable_Double_Buffer, csr.Act_Type = True, ActivationType.NONE

    # Execution
    csr.Macro_Op_Name = "Attention"
    csr.M_total, csr.N_total, csr.K_total = seq_len, seq_len, head_Dim
    sim.fetch_macro([MacroOp("Attention", macro_flash_attn_template, {"csr":copy.deepcopy(csr), "tensor": tensorHW, "latency": latencySet, "vpu":sim})])

    # """
    # Projection
    # """
    # # set tile dimensions
    # csr.M_tile = 64
    # csr.N_tile = 64
    # csr.K_tile = 32

    # # set External Memory
    # csr.Mem_Base_A = mem_mgr.allocate(seq_len * Hidden_Dim)    # MatA
    # csr.Mem_Stride_A = Hidden_Dim                              # 768 elements per row
    # csr.Mem_Base_B = mem_mgr.allocate(Hidden_Dim * Hidden_Dim) # MatB
    # csr.Mem_Stride_B = Hidden_Dim                              # 768 elements per row
    # csr.Mem_Base_C = mem_mgr.allocate(seq_len * Hidden_Dim)    # MatC
    # csr.Mem_Stride_C = Hidden_Dim                              # 768 elements per row

    # # Memory Access Modes (Scatter/Gather)
    # csr.Is_Gather_A  = True
    # csr.BLOCK_LEN_A  = csr.K_tile
    # csr.Is_Gather_B  = True
    # csr.BLOCK_LEN_B  = csr.N_tile
    # csr.Is_Scatter_C = True
    # csr.BLOCK_LEN_C  = csr.N_tile
    # csr.Is_Gather_D  = True
    # csr.BLOCK_LEN_D  = 0

    # # set VREG Mapping
    # csr.MatA_reg_base = 0  # MatA_tile VREG0 ~ VREG1
    # csr.VREG_stride_A = 2
    # csr.MatB_reg_base = 4  # MatB_tile VREG4 ~ VREG5
    # csr.VREG_stride_B = 2
    # csr.MatC_reg_base = 8  # MatC_tile VREG8 ~ VREG11
    # csr.VREG_stride_C = 4

    # # set operation flag
    # csr.Enable_Double_Buffer = True
    # csr.Act_Type = ActivationType.NONE
    
    # # Execution
    # csr.Macro_Op_Name = "GEMM"
    # csr.M_total = seq_len
    # csr.N_total = Hidden_Dim
    # csr.K_total = Hidden_Dim


    # sim.fetch_macro([MacroOp("Projection", 
    #                          macro_gemm_template, 
    #                          {"csr":csr, "tensor": tensorHW, "latency": latencySet})])
    


    # """
    # FNN + GELU
    # """
    # # set External Memory
    # csr.Mem_Base_A = mem_mgr.allocate(seq_len * Hidden_Dim)    # MatA
    # csr.Mem_Stride_A = Hidden_Dim
    # csr.Mem_Base_B = mem_mgr.allocate(Hidden_Dim * FFN_Dim) # MatB
    # csr.Mem_Stride_B = FFN_Dim
    # csr.Mem_Base_C = mem_mgr.allocate(seq_len * FFN_Dim)    # MatC
    # csr.Mem_Stride_C = FFN_Dim

    # # Memory Access Modes (Scatter/Gather)
    # csr.Is_Gather_A  = True
    # csr.BLOCK_LEN_A  = csr.K_tile
    # csr.Is_Gather_B  = True
    # csr.BLOCK_LEN_B  = csr.N_tile
    # csr.Is_Scatter_C = True
    # csr.BLOCK_LEN_C  = csr.N_tile
    # csr.Is_Gather_D  = True
    # csr.BLOCK_LEN_D  = 0

    # # set VREG Mapping
    # csr.MatA_reg_base = 0  # MatA_tile VREG0 ~ VREG1
    # csr.VREG_stride_A = 2
    # csr.MatB_reg_base = 4  # MatB_tile VREG4 ~ VREG5
    # csr.VREG_stride_B = 2
    # csr.MatC_reg_base = 8  # MatC_tile VREG8 ~ VREG11
    # csr.VREG_stride_C = 4

    # # set operation flag
    # csr.Enable_Double_Buffer = True
    # csr.Act_Type = ActivationType.GELU
    
    # # Execution
    # csr.Macro_Op_Name = "GEMM_GELU"
    # csr.M_total = seq_len
    # csr.N_total = FFN_Dim
    # csr.K_total = Hidden_Dim


    # sim.fetch_macro([MacroOp("Fnn_GELU", 
    #                          macro_gemm_gelu_template, 
    #                          {"csr":csr, "tensor": tensorHW, "latency": latencySet})])

# ==============================================================================
# 6. Run Simulation
# ==============================================================================
def run_simulation():
    model = "BERT_Base" # 可切換 "BERT_Base", "ViT_Base", "GPT2_Base", "TEST_SUBOP"
    
    # 建立 DualLogger (自動導向到 log/ 資料夾中，包含模型名稱)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(current_dir, "log")
    os.makedirs(log_dir, exist_ok=True)
    report_filepath = os.path.join(log_dir, f"{model}_console_output.txt")
    sys.stdout = DualLogger(report_filepath)

    latencySet = LatencySet()
    csr = CSRConfig()
    tensorHW = TensorConfig(phys_M=16, phys_N=16)
    sim = ADHD_VPU(model_name=model)
    mem_mgr = MemoryManager(base_addr=0xE000_0000)


    # 呼叫巨集組裝
    if model == "BERT_Base":
        target_seq_len = 512 # 設定可變的 Sequence Length
        build_bert_base_model(sim, csr, tensorHW, latencySet, seq_len=target_seq_len, mem_mgr=mem_mgr)
    elif model == "ViT_Base":
        target_seq_len = 197
        build_vit_base_model(sim, csr, tensorHW, latencySet, seq_len=target_seq_len, mem_mgr=mem_mgr)
    elif model == "GPT2_Base":
        target_seq_len = 1024
        build_gpt2_base_model(sim, csr, tensorHW, latencySet, seq_len=target_seq_len, mem_mgr=mem_mgr)
    elif model == "TEST_SUBOP":
        build_subOP(sim, csr, tensorHW, latencySet, mem_mgr=mem_mgr)

    print(f"--- Simulation Running {model} ... ---")
    start_time = time.time()

    while not sim.is_idle():
        sim.tick()

        # 每模擬 100 萬個 Cycle，就印出一次進度
        if sim.global_cycle % 1_000_000 == 0:
            elapsed = time.time() - start_time
            print(f"  [Heartbeat] 已經模擬了 {sim.global_cycle:,} 個 Cycles... (耗時: {elapsed:.1f} 秒)")
    

    # the last "}" for C code generation
    with open(sim.c_filepath, "a") as f_c:
        f_c.write("}\n")
    
    sim.print_report()

if __name__ == "__main__":
    run_simulation()