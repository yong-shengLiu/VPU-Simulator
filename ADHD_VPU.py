import os
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import deque
from typing import List, Optional, Dict

# ==============================================================================
# 1. Hardware Specifications & Constants
# ==============================================================================
NUM_VREGS = 32          
LANE = 4                
VLEN = 8192             
VLENB = VLEN // 8       
AXI_WIDTH = 64          
LSU_QUEUE_DEPTH = 16    
VALU_QUEUE_DEPTH = 16   
CIM_QUEUE_DEPTH = 32    
VIRTUAL_L0_BUFFER_ID = 63 
CLOCK_FREQ_GHZ = 1.0

class UnitType(Enum):
    LSU  = auto()   
    VALU = auto()   
    CIM  = auto()   

@dataclass
class MicroOp:
    name: str
    unit_type: UnitType
    latency: int       
    src_regs: List[int] = field(default_factory=list) 
    dst_regs: List[int] = field(default_factory=list) 
    
    wait_for_writes: dict = field(default_factory=dict)     
    wait_for_reads: dict = field(default_factory=dict)      
    wait_for_writes_waw: dict = field(default_factory=dict) 

    mem_addr: int = 0               
    mem_stride: int = 0             
    is_gather_scatter: bool = False 
    block_length: int = 0           
    
    def __repr__(self):
        return f"[{self.unit_type.name}] {self.name} (Lat:{self.latency}, Addr:{hex(self.mem_addr)})"

@dataclass
class MacroOp:
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
    Mem_Base_A: int = 0x0000_0000  
    Mem_Base_B: int = 0x0000_0000
    Mem_Base_C: int = 0x0000_0000
    Mem_Base_D: int = 0x0000_0000
    Mem_Stride_A: int = 64  
    Mem_Stride_B: int = 64
    Mem_Stride_C: int = 64
    Mem_Stride_D: int = 64

    Is_Gather_A: bool = False
    BLOCK_LEN_A: int = 0
    Is_Gather_B: bool = False
    BLOCK_LEN_B: int = 0
    Is_Scatter_C: bool = False
    BLOCK_LEN_C: int = 0
    Is_Gather_D: bool = False
    BLOCK_LEN_D: int = 0

    MatA_reg_base: int = 0
    MatB_reg_base: int = 4
    MatC_reg_base: int = 20
    MatD_reg_base: int = 8    
    MatE_reg_base: int = 12   
    Temp_reg_base: int = 28
    Enable_Double_Buffer: bool = True
    Act_Type: ActivationType = ActivationType.NONE

    VREG_stride_A: int = 2 
    VREG_stride_B: int = 2
    VREG_stride_C: int = 4
    VREG_stride_D: int = 2    
    VREG_stride_E: int = 4    
    VREG_stride_O: int = 16   

    M_tile: int = 64
    N_tile: int = 64
    K_tile: int = 32

    Macro_Op_Name: str = "GEMM"
    M_total: int = 0
    N_total: int = 0
    K_total: int = 0

@dataclass
class TensorConfig:
    phys_M: int = 16
    phys_N: int = 16

@dataclass
class LatencySet:
    Load_One_Vector: int = VLEN // AXI_WIDTH + 1  
    Store_One_Vector: int = VLEN // AXI_WIDTH + 1 
    VALU_VSET: int = 1
    VALU_VMV: int = VLEN // LANE // AXI_WIDTH    
    VALU_VADD: int = VLEN // LANE // AXI_WIDTH   
    VALU_VEXP: int = VLEN // LANE // AXI_WIDTH   
    VALU_VGELU: int = VLEN // LANE // AXI_WIDTH  

# ==============================================================================
# 2. Behavioral Memory Allocator 
# ==============================================================================
class MemoryManager:
    def __init__(self, base_addr=0x8000_0000):
        self.start_addr = base_addr
        self.current_addr = base_addr

    def allocate(self, size_in_bytes: int) -> int:
        addr = self.current_addr
        self.current_addr += (size_in_bytes + 63) & ~63
        return addr

    def reset(self):
        self.current_addr = self.start_addr

# ==============================================================================
# 3. Decoupled Micro-Architecture Components
# ==============================================================================
class MacroExpander:
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

class ExecutionUnit:
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
            if self.name == "LSU":
                self.total_bytes_transferred += (AXI_WIDTH // 8)
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

class ADHD_VPU:
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

        current_dir = os.path.dirname(os.path.abspath(__file__))
        log_dir = os.path.join(current_dir, "log")
        os.makedirs(log_dir, exist_ok=True)
        
        self.trace_filepath = os.path.join(log_dir, trace_filename)
        self.c_filepath = os.path.join(log_dir, c_macro_header)
        
        with open(self.trace_filepath, "w") as f:
            f.write("=========================================================\n")
            f.write(" ADHD VPU Firmware CSR Trace (Auto-Generated)\n")
            f.write("=========================================================\n\n")
        
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
            else:
                f_c.write("static inline void dispatch_sub_ops() {\n")

    def fetch_macro(self, macro_ops: List[MacroOp]):
        for op in macro_ops:
            self.macro_instr_buffer.append(op)
            self.total_macro_fetched += 1
            self._log_csr_trace(op)
    
    def _log_csr_trace(self, op: MacroOp):
        csr = op.args["csr"]
        with open(self.trace_filepath, "a") as f:
            f.write(f"# --- Dispatching Macro: {op.name} ({op.expansion_func.__name__}) ---\n")
            f.write(f"csrw 0x801, 0x{csr.Mem_Base_A:016X}\n")
            f.write(f"csrw 0x802, 0x{csr.Mem_Base_B:016X}\n")
            f.write(f"csrw 0x803, 0x{csr.Mem_Base_C:016X}\n")
            f.write(f"csrw 0x804, 0x{csr.Mem_Base_D:016X}\n\n")
            
        with open(self.c_filepath, "a") as f_c:
            f_c.write(f"\n    // --- Dispatching Macro: {op.name} ---\n")
            f_c.write(f"    __asm__ volatile(\"csrw 0x801, %0\" :: \"r\"(0x{csr.Mem_Base_A:016X}ULL));\n")
            f_c.write(f"    __asm__ volatile(\"csrw 0x802, %0\" :: \"r\"(0x{csr.Mem_Base_B:016X}ULL));\n")
            f_c.write(f"    __asm__ volatile(\"csrw 0x803, %0\" :: \"r\"(0x{csr.Mem_Base_C:016X}ULL));\n")

    def tick(self):
        self.global_cycle += 1
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
        print("="*60)
        print(f"Simulation Report (Total Cycles: {self.global_cycle:,})")
        print("="*60)
        print(f"[Instruction Fetch Bandwidth Reduction]")
        print(f"  - CPU Macro Ops Fetched : {self.total_macro_fetched:,}")
        print(f"  - VPU Micro Ops Executed: {self.total_micro_generated:,}")
        print(f"  - Expansion Ratio       : 1:{self.total_micro_generated/max(1, self.total_macro_fetched):.1f}")
        print(f"\n[Backend Overlap & Hazard Analysis (The True Decoupling)]")
        print(f"  - LSU Active  : {self.lsu_unit.total_active_cycles/self.global_cycle:5.1%} | Wait Data: {self.lsu_unit.stall_cycles/self.global_cycle:5.1%}")
        print(f"  - VALU Active : {self.valu_unit.total_active_cycles/self.global_cycle:5.1%} | Wait Data: {self.valu_unit.stall_cycles/self.global_cycle:5.1%}")
        print(f"  - CIM Active  : {self.cim_unit.total_active_cycles/self.global_cycle:5.1%} | Wait Data: {self.cim_unit.stall_cycles/self.global_cycle:5.1%}")
        print("="*60)

# ==============================================================================
# 4. Macro-OP FSM Templates (Tile-Level Granularity)
# ==============================================================================
def get_actual_vreg(base_reg, sub_idx, tile_size, stride):
    elements_per_vreg = max(1, tile_size // stride)
    return base_reg + (sub_idx // elements_per_vreg)

def _get_lsu_latency(base_lat, stride, is_sg, block_len):
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

def macro_flash_attn_template(csr: CSRConfig, tensor: TensorConfig, latency: LatencySet):
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
            dst_regs=k_regs_actual, mem_addr=addr_K, mem_stride=csr.Mem_Stride_B
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
            dst_regs=v_regs_actual, mem_addr=addr_V, mem_stride=csr.Mem_Stride_D
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
        src_regs=quant_regs, mem_addr=csr.Mem_Base_C, mem_stride=csr.Mem_Stride_C
    ))

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
    
    uops.append(MicroOp("LSU_LOAD_MAIN", UnitType.LSU, latency=_get_lsu_latency(latency.Load_One_Vector, csr.VREG_stride_C, False, 64), dst_regs=main_regs, mem_addr=csr.Mem_Base_A, mem_stride=csr.Mem_Stride_A))
    uops.append(MicroOp("LSU_LOAD_RES", UnitType.LSU, latency=_get_lsu_latency(latency.Load_One_Vector, csr.VREG_stride_C, False, 64), dst_regs=res_regs, mem_addr=csr.Mem_Base_B, mem_stride=csr.Mem_Stride_B))
    
    uops.append(MicroOp(f"VALU_VADD_RES", UnitType.VALU, latency=int(latency.VALU_VADD*csr.VREG_stride_C), src_regs=main_regs + res_regs, dst_regs=out_regs))
    
    realistic_valu_lat = int((csr.M_tile * csr.K_total) // (LANE * AXI_WIDTH) + 10)
    uops.append(MicroOp("VALU_LN_MEAN", UnitType.VALU, latency=realistic_valu_lat, src_regs=out_regs, dst_regs=[reg_mean]))
    uops.append(MicroOp("VALU_LN_VAR", UnitType.VALU, latency=realistic_valu_lat, src_regs=out_regs + [reg_mean], dst_regs=[reg_var]))
    uops.append(MicroOp("VALU_LN_RSQRT", UnitType.VALU, latency=20, src_regs=[reg_var], dst_regs=[reg_var]))
    uops.append(MicroOp("VALU_LN_NORM", UnitType.VALU, latency=realistic_valu_lat, src_regs=out_regs + [reg_mean, reg_var], dst_regs=out_regs))
    
    uops.append(MicroOp("LSU_STORE_LN", UnitType.LSU, latency=_get_lsu_latency(latency.Store_One_Vector, csr.VREG_stride_C, False, 64), src_regs=out_regs, mem_addr=csr.Mem_Base_C, mem_stride=csr.Mem_Stride_C))
    return uops

# ==============================================================================
# 5. Model Builders (Software Schedulers explicitly setting CSRs)
# ==============================================================================

def set_gemm_csr(csr: CSRConfig, A_base, B_base, C_base, A_stride, B_stride, C_stride, m_tile, n_tile, k_total, act=ActivationType.NONE):
    """ Helper to populate standard GEMM CSR parameters """
    csr.M_tile, csr.N_tile, csr.K_total = m_tile, n_tile, k_total
    csr.Mem_Base_A, csr.Mem_Base_B, csr.Mem_Base_C = A_base, B_base, C_base
    csr.Mem_Stride_A, csr.Mem_Stride_B, csr.Mem_Stride_C = A_stride, B_stride, C_stride
    
    # Standard GEMM 64x64 Double Buffer Allocation
    csr.MatA_reg_base, csr.VREG_stride_A = 0, 2
    csr.MatB_reg_base, csr.VREG_stride_B = 4, 2
    csr.MatC_reg_base, csr.VREG_stride_C = 8, 4
    csr.Enable_Double_Buffer = True
    csr.Act_Type = act

def build_bert_base_layer(sim: ADHD_VPU, csr: CSRConfig, tensorHW: TensorConfig, latencySet: LatencySet, seq_len: int, mem_mgr: MemoryManager):
    D = 768
    D_FFN = 3072
    print(f"\n--- Dispatching BERT Base Layer (Seq Length: {seq_len}) ---")
    
    # 1. Q Projection (Software Tile Outer Loops)
    Q_in = mem_mgr.allocate(seq_len * D); W_Q = mem_mgr.allocate(D * D); Q_out = mem_mgr.allocate(seq_len * D)
    for m in range(0, seq_len, 64):
        for n in range(0, D, 64):
            set_gemm_csr(csr, Q_in + (m*D), W_Q + n, Q_out + (m*D) + n, D, D, D, 64, 64, D)
            sim.fetch_macro([MacroOp(f"PROJ_Q_m{m}_n{n}", macro_gemm_template, {"csr":csr, "tensor": tensorHW, "latency": latencySet})])
    
    # K, V Projections (Simplified for simulation speed, just assigning space)
    K_out = mem_mgr.allocate(seq_len * D); V_out = mem_mgr.allocate(seq_len * D)
    
    # 2. FlashAttention (Software Q-Loop)
    head_dim = D // 12
    for h in range(1): # Only simulating 1 head for brevity
        for q_start in range(0, seq_len, 32):
            csr.M_tile, csr.N_tile, csr.K_total, csr.N_total = 32, 32, head_dim, seq_len
            
            # 【Causal Masking 的精華】：如果是 GPT，這裡的 N_total 會是 q_start + 32！
            
            csr.Mem_Base_A = Q_out + (q_start * head_dim)
            csr.Mem_Base_B, csr.Mem_Base_D = K_out, V_out
            csr.Mem_Base_C = mem_mgr.allocate(seq_len * head_dim) + (q_start * head_dim)
            csr.Mem_Stride_A = csr.Mem_Stride_B = csr.Mem_Stride_D = csr.Mem_Stride_C = head_dim
            
            # Flash Attention 32x32 Ping-Pong Allocation
            csr.MatA_reg_base, csr.VREG_stride_A = 0, 2    
            csr.MatB_reg_base, csr.VREG_stride_B = 2, 2    
            csr.MatD_reg_base, csr.VREG_stride_D = 6, 2    
            csr.MatE_reg_base, csr.VREG_stride_E = 10, 1   
            csr.MatC_reg_base, csr.VREG_stride_O = 16, 8   
            csr.Enable_Double_Buffer = True
            
            sim.fetch_macro([MacroOp(f"FLASH_ATTN_H{h}_q{q_start}", macro_flash_attn_template, {"csr":csr, "tensor": tensorHW, "latency": latencySet})])

def build_gpt2_prefill_layer(sim: ADHD_VPU, csr: CSRConfig, tensorHW: TensorConfig, latencySet: LatencySet, seq_len: int, mem_mgr: MemoryManager):
    """ GPT-2 Prefill: Demonstrating the Causal Masking capability via Software Scheduling! """
    D = 768
    print(f"\n--- Dispatching GPT-2 Prefill Layer (Context: {seq_len}) with Causal Masking ---")

    Q_out = mem_mgr.allocate(seq_len * D)
    K_out = mem_mgr.allocate(seq_len * D)
    V_out = mem_mgr.allocate(seq_len * D)
    O_out = mem_mgr.allocate(seq_len * D)
    head_dim = D // 12
    
    # ✨ FlashAttention (Software Q-Loop with CAUSAL MASKING) ✨
    for h in range(1): 
        for q_start in range(0, seq_len, 32):
            csr.M_tile, csr.N_tile, csr.K_total = 32, 32, head_dim
            
            # 【軟體可編程拓撲】：天然實現 Attention 下三角遮罩！
            # K, V 迴圈只會掃描到當前的 q_start，未來的 Token 絕對不會被讀進 VPU！
            csr.N_total = min(seq_len, q_start + csr.M_tile) 
            
            csr.Mem_Base_A = Q_out + (q_start * head_dim)
            csr.Mem_Base_B, csr.Mem_Base_D = K_out, V_out
            csr.Mem_Base_C = O_out + (q_start * head_dim)
            csr.Mem_Stride_A = csr.Mem_Stride_B = csr.Mem_Stride_D = csr.Mem_Stride_C = head_dim
            
            csr.MatA_reg_base, csr.VREG_stride_A = 0, 2    
            csr.MatB_reg_base, csr.VREG_stride_B = 2, 2    
            csr.MatD_reg_base, csr.VREG_stride_D = 6, 2    
            csr.MatE_reg_base, csr.VREG_stride_E = 10, 1   
            csr.MatC_reg_base, csr.VREG_stride_O = 16, 8   
            csr.Enable_Double_Buffer = True
            
            sim.fetch_macro([MacroOp(f"GPT_CAUSAL_ATTN_q{q_start}", macro_flash_attn_template, {"csr":csr, "tensor": tensorHW, "latency": latencySet})])

def build_subOP(sim: ADHD_VPU, csr: CSRConfig, tensorHW: TensorConfig, latencySet: LatencySet, mem_mgr: MemoryManager):
    print(f"\n Sub Macro OP for test ---")
    seq_len = 512; Hidden_Dim = 768; head_Dim = Hidden_Dim // 12
    
    csr.M_tile, csr.N_tile, csr.K_total, csr.N_total = 32, 32, head_Dim, seq_len
    csr.Mem_Base_A = mem_mgr.allocate(seq_len * head_Dim);   csr.Mem_Stride_A = head_Dim  
    csr.Mem_Base_B = mem_mgr.allocate(head_Dim * seq_len);   csr.Mem_Stride_B = head_Dim  
    csr.Mem_Base_D = mem_mgr.allocate(seq_len * head_Dim);   csr.Mem_Stride_D = head_Dim  
    csr.Mem_Base_C = mem_mgr.allocate(seq_len * Hidden_Dim); csr.Mem_Stride_C = head_Dim  

    csr.Is_Gather_A,  csr.BLOCK_LEN_A  = True, csr.K_tile
    csr.Is_Gather_B,  csr.BLOCK_LEN_B  = True, csr.N_tile
    csr.Is_Gather_D,  csr.BLOCK_LEN_D  = True, csr.N_tile
    csr.Is_Scatter_C, csr.BLOCK_LEN_C  = True, csr.N_tile
    
    csr.MatA_reg_base, csr.VREG_stride_A = 0, 2    
    csr.MatB_reg_base, csr.VREG_stride_B = 2, 2    
    csr.MatD_reg_base, csr.VREG_stride_D = 6, 2    
    csr.MatE_reg_base, csr.VREG_stride_E = 10, 1   
    csr.MatC_reg_base, csr.VREG_stride_O = 16, 8   
    csr.Enable_Double_Buffer, csr.Act_Type = True, ActivationType.NONE

    # Only fetching ONE macro op to simulate the hardware processing a single 32x32 Q-Tile
    sim.fetch_macro([MacroOp("Attention_Tile", macro_flash_attn_template, {"csr":csr, "tensor": tensorHW, "latency": latencySet})])

# ==============================================================================
# 6. Run Simulation
# ==============================================================================
def run_simulation():
    model = "BERT_Base" # "BERT_Base", "GPT2_Base", "TEST_SUBOP"

    latencySet = LatencySet()
    csr = CSRConfig()
    tensorHW = TensorConfig(phys_M=16, phys_N=16)
    sim = ADHD_VPU(model_name=model)
    mem_mgr = MemoryManager(base_addr=0xE000_0000)

    if model == "BERT_Base":
        build_bert_base_layer(sim, csr, tensorHW, latencySet, seq_len=512, mem_mgr=mem_mgr)
    elif model == "GPT2_Base":
        build_gpt2_prefill_layer(sim, csr, tensorHW, latencySet, seq_len=512, mem_mgr=mem_mgr)
    elif model == "TEST_SUBOP":
        build_subOP(sim, csr, tensorHW, latencySet, mem_mgr=mem_mgr)

    print(f"--- Simulation Running {model} ... ---")
    while not sim.is_idle():
        sim.tick()
    
    with open(sim.c_filepath, "a") as f_c:
        f_c.write("}\n")
    sim.print_report()

if __name__ == "__main__":
    run_simulation()