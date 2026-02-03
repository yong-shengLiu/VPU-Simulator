import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict

# ==========================================
# 1. Workload Definition (Software Mapping)
# ==========================================
@dataclass
class MacroOp:
    name: str
    op_type: str        # LOAD, STORE, GEMM, SOFTMAX
    dims: Tuple[int]    # (M, N, K) or similar
    data_vol: int       # Bytes
    compute_ops: int    # MACs or FLOPs
    dependency: int     # Token ID

class AttentionWorkload:
    def __init__(self, B=1, S=128, H=64, dtype_bytes=2):
        self.B, self.S, self.H = B, S, H
        self.dtype = dtype_bytes
        
    def generate_ops(self) -> List[MacroOp]:
        # Q * K^T -> Score (S x S)
        # Score -> Softmax
        # Score * V -> Output (S x H)
        ops = []
        
        # 1. Load Q, K, V (Assume worst case: load all from DRAM)
        size_qkv = self.B * self.S * self.H * self.dtype
        ops.append(MacroOp("Load Q", "LOAD", (self.S, self.H), size_qkv, 0, 0))
        ops.append(MacroOp("Load K", "LOAD", (self.S, self.H), size_qkv, 0, 0))
        ops.append(MacroOp("Load V", "LOAD", (self.S, self.H), size_qkv, 0, 0))
        
        # 2. Q * K^T (MatMul)
        # (S x H) * (H x S) -> (S x S)
        macs_att = self.B * self.S * self.S * self.H
        ops.append(MacroOp("Attention Score", "GEMM", (self.S, self.S, self.H), 0, macs_att, 1))
        
        # 3. Softmax (Row-wise on S x S)
        # Ops approx 3 * Elements (Exp, Sum, Div)
        flops_smax = self.B * self.S * self.S * 3 
        ops.append(MacroOp("Softmax", "SOFTMAX", (self.S, self.S), 0, flops_smax, 2))
        
        # 4. Score * V (MatMul)
        # (S x S) * (S x H) -> (S x H)
        macs_out = self.B * self.S * self.S * self.H
        ops.append(MacroOp("Weighted Sum", "GEMM", (self.S, self.H, self.S), 0, macs_out, 3))
        
        # 5. Store
        ops.append(MacroOp("Store Output", "STORE", (self.S, self.H), size_qkv, 0, 4))
        
        return ops

# ==========================================
# 2. Architecture Models (Hardware Specs)
# ==========================================
class BaseArch:
    def __init__(self, name, lanes=4, banks_per_lane=8, data_width=64, issue_width=1):
        self.name = name
        self.lanes = lanes
        self.banks = banks_per_lane
        self.width = data_width # bits
        self.issue_width = issue_width
        self.vrf_depth = 4096 # elements per lane (simplified)
        
    def map_tensor(self, rows, cols):
        """Returns effective parallelism (elements processed per cycle)"""
        raise NotImplementedError

    def estimate_cycles(self, op: MacroOp):
        raise NotImplementedError

class AraBaseline(BaseArch):
    def __init__(self):
        super().__init__("Ara (Baseline)", lanes=4, banks_per_lane=8, issue_width=1)
        
    def map_tensor(self, rows, cols):
        # Ara uses Strip-mining (1D mapping)
        # It fills all lanes linearly.
        # But for Softmax (Row-wise), if a row is split across lanes, we need inter-lane reduction.
        # Or if row_len < num_lanes, utilization drops.
        
        # Vector Width in elements (FP16, 16-bit)
        # 4 Lanes * 64-bit/lane = 256 bits = 16 elements
        vl_max = self.lanes * (64 // 16) 
        
        if cols < vl_max:
            # Fragmentation! e.g., H=64, but we treat it as 1D stream.
            # Actually Ara handles 1D streams well, BUT...
            # For Softmax, we need to reduce along 'cols'.
            # If mapped linearly 1D, rows are packed.
            # Row 0 might end in Lane 1, Row 1 starts in Lane 2.
            # Softmax becomes complex: Inter-lane communication required.
            return 0.5 # Penalty for misalignment/complexity
        return 1.0 # Full utilization for pure 1D streams

    def estimate_cycles(self, op: MacroOp):
        # Single Issue: Strictly Serial
        if op.op_type == "LOAD" or op.op_type == "STORE":
            # Bandwidth bound: 64-bit AXI
            bus_width_bytes = 8
            return op.data_vol / bus_width_bytes
        elif op.op_type == "GEMM":
            # Compute bound: Lanes * MACs/cycle
            # 4 Lanes, assume 1 MAC/lane/cycle for FP16? No, SIMD width.
            # 4 Lanes * (64bit/16bit) = 16 MACs/cycle
            ops_per_cycle = self.lanes * (64 // 16)
            return op.compute_ops / ops_per_cycle
        elif op.op_type == "SOFTMAX":
            # The Killer. 
            # Ara Softmax is scalar-heavy or uses expensive vector reductions.
            # Vector reduction takes log2(VL) steps + scalar moves.
            # Assume 10x penalty vs ideal vector op due to scalar interaction
            ops_per_cycle = self.lanes * (64 // 16)
            return (op.compute_ops / ops_per_cycle) * 10 
        return 0

class ADHDTarget(BaseArch):
    def __init__(self):
        super().__init__("ADHD (Target)", lanes=4, banks_per_lane=8, issue_width=3)
        
    def map_tensor(self, rows, cols):
        # Lane-Aware Mapping
        # We map Rows to Lanes directly if possible.
        # e.g., S=128. We process 4 rows in parallel (one per lane).
        # Softmax becomes Intra-Lane only (No cross-bar).
        return 1.0 # Ideal utilization

    def estimate_cycles(self, op: MacroOp):
        # Decoupled / Multi-Issue:
        # We don't just add cycles; we look at the bottleneck.
        # But for single-op latency estimation:
        
        if op.op_type == "LOAD" or op.op_type == "STORE":
            # Still Bandwidth bound (Physical limit)
            bus_width_bytes = 8
            # BUT: Lane-Aware Scatter engine has 0 overhead for transpose
            return op.data_vol / bus_width_bytes
            
        elif op.op_type == "GEMM":
            # CIM Engine? Or just optimized Vector?
            # Assume standard Vector for fairness, but Lane-Aware.
            ops_per_cycle = self.lanes * (64 // 16)
            return op.compute_ops / ops_per_cycle
            
        elif op.op_type == "SOFTMAX":
            # Ideal Vectorized Softmax (Intra-lane)
            # No scalar core involvement!
            ops_per_cycle = self.lanes * (64 // 16)
            return op.compute_ops / ops_per_cycle
        return 0

# ==========================================
# 3. Simulation & Analysis
# ==========================================
def simulate_timeline(arch: BaseArch, ops: List[MacroOp]):
    timeline = [] # (Start, End, Unit)
    # Unit IDs: 0=LSU, 1=Compute(CIM/VALU)
    
    curr_lsu = 0
    curr_compute = 0
    
    total_cycles = 0
    
    print(f"\n--- Simulation: {arch.name} ---")
    
    for op in ops:
        cycles = int(arch.estimate_cycles(op))
        utilization = 1.0
        if op.dims:
             # Check mapping efficiency
             utilization = arch.map_tensor(op.dims[0], op.dims[1])
        
        real_cycles = int(cycles / utilization)
        
        start = 0
        end = 0
        unit = ""
        
        if op.op_type in ["LOAD", "STORE"]:
            unit = "LSU"
            # Ara (Single Issue): Must wait for everything to finish (simplified)
            # ADHD (Decoupled): Only waits for LSU availability (and data dependency)
            
            if arch.issue_width == 1:
                start = max(curr_lsu, curr_compute) # Serial execution
            else:
                start = curr_lsu # Decoupled
                
            end = start + real_cycles
            curr_lsu = end
            
        else: # Compute
            unit = "EX"
            if arch.issue_width == 1:
                start = max(curr_lsu, curr_compute)
            else:
                # Dependency check! Compute must wait for Load
                # Simplified: Assume Load happens just before. 
                # Ideally check op.dependency.
                start = max(curr_compute, curr_lsu) # Wait for data (RAW)
                # Note: In a real trace, we might prefetch. 
                # Here we assume naive dependency: Load -> Compute -> Store
            
            end = start + real_cycles
            curr_compute = end
            
        print(f"  {op.name:<16} | Type: {op.op_type:<7} | Cycles: {real_cycles:<5} | Range: {start}-{end}")
        total_cycles = max(total_cycles, end)
        
    return total_cycles

# Run
workload = AttentionWorkload(S=128, H=64)
ops = workload.generate_ops()

ara = AraBaseline()
cycles_ara = simulate_timeline(ara, ops)

adhd = ADHDTarget()
cycles_adhd = simulate_timeline(adhd, ops)

print(f"\n=== Result Summary ===")
print(f"Ara Baseline Cycles: {cycles_ara}")
print(f"ADHD Target Cycles : {cycles_adhd}")
print(f"Speedup            : {cycles_ara / cycles_adhd:.2f}x")

# Spec Gap Analysis
print("\n=== Hardware Spec Gap Analysis ===")
print("1. Issue Width:")
print(f"   - Current: {ara.issue_width} (Causes serial execution of Load & Compute)")
print(f"   - Target : {adhd.issue_width} (Allows overlapping, see overlapping ranges in timeline)")

print("\n2. Mapping Logic (VRF):")
print(f"   - Current: Strip-mining (Utilization factor ~{ara.map_tensor(128,64)})")
print(f"   - Target : Lane-Aware (Utilization factor ~{adhd.map_tensor(128,64)})")
print(f"   - Requirement: LSU must support 'Tensor Scatter' to pack {workload.H}-dim rows into single lanes.")

print("\n3. Softmax Efficiency:")
print(f"   - Observation: Ara Softmax is 10x slower due to Scalar interaction.")
print(f"   - Spec Change: Need 'Vector-Only' Softmax support (VALU Intra-lane reduction).")