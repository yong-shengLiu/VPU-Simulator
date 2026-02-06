import numpy as np
from dataclasses import dataclass
from typing import List, Tuple

# ==========================================
# 1. Hardware Specification Definitions
# ==========================================
@dataclass
class HardwareSpecs:
    # LSU Specs
    axi_width_bits: int = 64
    
    # VRF Specs (from vrf.py context)
    vlen_bits: int = 4096
    nr_lanes: int = 4
    nr_banks_per_lane: int = 8
    num_vregs: int = 32 # Standard RISC-V Vector
    
    # Vector Engine (VALU/NAF) Specs
    alu_width_bits: int = 64 
    
    # CIM Specs
    cim_cols: int = 16
    cim_macs_per_cycle: int = 64
    cim_throughput_ops: int = 64

    @property
    def total_vrf_bytes(self):
        # Total VRF Capacity in Bytes
        return (self.vlen_bits * self.num_vregs) // 8

    def get_vector_throughput(self, precision_bits=8):
        ops_per_lane = self.alu_width_bits // precision_bits
        return self.nr_lanes * ops_per_lane

# ==========================================
# 2. Workload & Mapping Definition
# ==========================================
@dataclass
class TensorOperand:
    name: str
    shape: Tuple[int]
    precision: int = 8 # bits
    
    @property
    def size_bytes(self):
        elements = np.prod(self.shape)
        return (elements * self.precision) // 8

class DSEModel:
    def __init__(self, specs: HardwareSpecs):
        self.specs = specs
        
    def check_capacity(self, tensors: List[TensorOperand]):
        """
        Checks if the working set fits in VRF.
        Returns (fits: bool, utilization: float, message: str)
        """
        total_size = sum(t.size_bytes for t in tensors)
        capacity = self.specs.total_vrf_bytes
        
        utilization = total_size / capacity
        fits = total_size <= capacity
        
        details = []
        for t in tensors:
            details.append(f"{t.name}: {t.size_bytes/1024:.1f}KB")
            
        msg = (f"VRF Capacity: {capacity/1024:.1f}KB\n"
               f"Required: {total_size/1024:.1f}KB ({' + '.join(details)})\n"
               f"Utilization: {utilization*100:.1f}%")
        
        if not fits:
            msg += "\n[!] WARNING: Working set exceeds VRF capacity! Tiling is REQUIRED."
        else:
            msg += "\n[OK] Working set fits in VRF."
            
        return fits, utilization, msg

    def analyze_mapping(self, tensor: TensorOperand, strategy: str):
        """
        Analyzes efficiency based on Mapping Strategy.
        """
        rows, cols = tensor.shape[-2], tensor.shape[-1]
        
        # 1. Bandwidth Analysis (LSU)
        bytes_total = tensor.size_bytes
        lsu_cycles_ideal = bytes_total / (self.specs.axi_width_bits / 8)
        
        # 2. VRF Distribution & Compute Efficiency
        if strategy == "Strip-Mining":
            # Linear mapping efficiency
            simd_width = self.specs.get_vector_throughput(tensor.precision)
            utilization = min(1.0, cols / simd_width) if cols < simd_width else 1.0
            reduction_penalty = 10.0 # High penalty for inter-lane reduction
            
        elif strategy == "Lane-Aware":
            # Ideal Lane mapping
            utilization = 1.0 
            reduction_penalty = 1.0 # Intra-lane reduction is fast
            
        return lsu_cycles_ideal, utilization, reduction_penalty

    def simulate_tiled_schedule(self, M, N, K, strategy: str):
        """
        Simulates execution with Automatic Tiling.
        Workload: Attention Q(MxK) * K^T(KxM) -> Score(MxM) ...
        For simplicity, we analyze the Q*K^T stage with Tiling on 'M' (Sequence Length).
        """
        
        # 1. Heuristic Tiling Calculation
        vrf_limit = self.specs.total_vrf_bytes
        
        valid_tile = 0
        # Try different tile sizes for Sequence Length (M)
        for m_t in [128, 64, 32, 16]:
            # Estimated Working Set per Tile:
            # Q_block + K_block + V_block + Output_block + Score_block (approx)
            # Size = 5 matrices of size (m_t * 64)
            req_size = 5 * (m_t * 64) 
            
            if req_size <= vrf_limit:
                valid_tile = m_t
                break
        
        if valid_tile == 0:
            valid_tile = 1 # Fallback for extremely small memory
            
        num_tiles = M // valid_tile
        print(f"\n[Capacity Check & Tiling]")
        print(f"  Target Workload: Seq={M}, Hidden={K} (Full Q+K+V would be {3*M*K/1024:.1f}KB)")
        print(f"  VRF Capacity: {vrf_limit/1024:.1f}KB")
        print(f"  > Selected Tile Size (Seq): {valid_tile}")
        print(f"  > Number of Tiles: {num_tiles}")
        
        # 2. Run Timeline Simulation
        total_cycles = 0
        print(f"\n[Timeline Simulation - {strategy} (Tiled)]")
        
        # Resource Availability Timestamps
        lsu_free = 0
        compute_free = 0
        
        for t in range(num_tiles):
            # --- Analysis for ONE Tile ---
            # Data Volume per tile
            bytes_tile = valid_tile * K
            ops_gemm = valid_tile * valid_tile * K # Q*K
            ops_smax = valid_tile * valid_tile * 3
            ops_out  = valid_tile * K * valid_tile # Score*V
            
            # Latency Calculations
            # Load 3 matrices (Q, K, V parts)
            lsu_lat_tile = (bytes_tile * 3) / (self.specs.axi_width_bits / 8)
            store_lat_tile = bytes_tile / (self.specs.axi_width_bits / 8)
            
            # Compute Latency (with efficiency factors)
            _, util, red_pen = self.analyze_mapping(TensorOperand("Tile_Calc", (valid_tile, K)), strategy)
            
            cim_tpt = self.specs.cim_throughput_ops * util
            vec_tpt = (self.specs.get_vector_throughput() * util) / red_pen
            
            comp_cycles = (ops_gemm + ops_out) / cim_tpt + (ops_smax / vec_tpt)
            
            # --- Timeline Update ---
            
            if strategy == "Strip-Mining":
                # Serial Execution: Wait for previous Compute to finish before Loading next
                # Load -> Compute -> Store
                start_lsu = max(lsu_free, compute_free)
                end_lsu = start_lsu + lsu_lat_tile
                
                start_comp = end_lsu
                end_comp = start_comp + comp_cycles
                
                start_store = end_comp
                end_store = start_store + store_lat_tile
                
                lsu_free = end_store
                compute_free = end_comp
                
            else: # Lane-Aware (Decoupled/Pipelined)
                # Pipelined Execution: 
                # LSU can load Tile N+1 while Compute is busy with Tile N
                
                # 1. Load (Pre-fetch if possible)
                start_lsu = lsu_free
                end_lsu = start_lsu + lsu_lat_tile
                
                # 2. Compute (Waits for THIS Load, and Compute Unit free)
                start_comp = max(end_lsu, compute_free)
                end_comp = start_comp + comp_cycles
                
                # 3. Store (Waits for THIS Compute, and LSU free)
                start_store = max(end_comp, end_lsu) 
                end_store = start_store + store_lat_tile
                
                lsu_free = end_store
                compute_free = end_comp
                
            print(f"  Tile {t}: Load[{int(lsu_lat_tile)}] -> Comp[{int(comp_cycles)}] -> Store[{int(store_lat_tile)}]")
            
        return lsu_free

# ==========================================
# 3. Execution
# ==========================================
if __name__ == "__main__":
    specs = HardwareSpecs()
    dse = DSEModel(specs)

    # 1. Capacity Check of Full Model (Just to show it fails without tiling)
    print("=== 1. VRF Capacity Check (Full Layer without Tiling) ===")
    full_tensors = [
        TensorOperand("Q", (128, 64), 8),
        TensorOperand("K", (128, 64), 8),
        TensorOperand("V", (128, 64), 8)
    ]
    dse.check_capacity(full_tensors) 

    # 2. Tiled Simulation
    print("\n=== 2. Tiled Simulation (Strip-Mining vs Lane-Aware) ===")
    
    # Run Baseline (Ara)
    cyc_base = dse.simulate_tiled_schedule(128, 128, 64, "Strip-Mining")
    
    # Run Target (ADHD)
    cyc_opt = dse.simulate_tiled_schedule(128, 128, 64, "Lane-Aware")

    print(f"\n=== Final Comparison ===")
    print(f"Baseline (Tiled): {int(cyc_base)} cycles")
    print(f"ADHD (Tiled)    : {int(cyc_opt)} cycles")
    print(f"Speedup         : {cyc_base/cyc_opt:.2f}x")