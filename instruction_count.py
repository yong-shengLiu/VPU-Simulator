import numpy as np
import pandas as pd

def estimate_gemm_performance(M, N, P, M_tile_list=[1, 2, 4, 8]):
    """
    估算不同 M_tile 對於 GEMM 指令數與效能瓶頸的影響
    假設架構: Single Issue, VLEN=512 bits (64 bytes)
    """
    VLEN_BYTES = 512 # 4096 bits
    data_type_bytes = 1 # int8
    elements_per_vector = VLEN_BYTES // data_type_bytes # 64 elements
    
    print(f"--- Performance Projection ---")
    print(f"Workload: M={M}, N={N}, P={P}")
    print(f"Vector Width: {elements_per_vector} elements (int8)")
    
    results = []

    for m_tile in M_tile_list:
        # 1. 計算 Loop 次數
        # P 維度切分 (Strip mining)
        num_p_strips = np.ceil(P / elements_per_vector)
        
        # M 維度切分 (Tiling)
        num_m_tiles = np.ceil(M / m_tile)
        
        # Inner Loop (N) 次數
        num_n_iters = N
        
        # 2. 計算指令總數 (Total Instructions)
        # 針對每個 Inner Loop (N) 的 iteration:
        #   - Load Instructions: 1 次 (Load Weight Vector B)
        #   - Compute Instructions: m_tile 次 (MAC for each row)
        
        # 總共執行的 Inner Loop 次數 (Total micro-kernels)
        total_kernels = num_p_strips * num_m_tiles * num_n_iters
        
        total_load_insts = total_kernels * 1  # 每次 kernel 載入 1 次 B
        total_mac_insts  = total_kernels * m_tile # 每次 kernel 算 m_tile 次
        
        # 3. 估算 Cycle (Simplified Single-Issue Model)
        # 假設 Load Latency (DRAM) = 20 cycles (或是 Bus 頻寬限制)
        # 假設 MAC Latency = 1 cycle (Throughput)
        # 假設 Issue = 1 cycle
        
        # 在 Baseline Single Issue 下，這些指令是序列執行的，無法重疊
        # 這裡我們計算 "Compute-to-Memory Ratio" (指令數比例)
        # 這是論文最愛看的指標：Arithmetic Intensity
        
        ratio = total_mac_insts / total_load_insts
        
        # 4. 預估 LSU 阻塞程度 (Blocking Factor)
        # 當 LSU 發出一道指令後，它要等多少個 ALU 指令發完才能發下一道？
        # 這直接反映了 Head-of-Line Blocking 的嚴重性
        blocking_factor = m_tile 
        
        results.append({
            "M_tile": m_tile,
            "Total Load Insts": int(total_load_insts),
            "Total MAC Insts": int(total_mac_insts),
            "Comp/Load Ratio": ratio,
            "Blocking Severity": f"{m_tile}x (LSU waits {m_tile} cycles)",
            "Est. Speedup (Ideal)": f"{m_tile / (1 + m_tile / 20):.2f}x" # 粗略估計
        })
        
    df = pd.DataFrame(results)
    print(df.to_string())
    print("-" * 60)
    return df

# --- 執行估算 (使用你的 Mini-Monster Case) ---
# M=128 (Seq), N=1024 (Input), P=1024 (Output)
estimate_gemm_performance(M=128, N=1024, P=1024, M_tile_list=[1, 2, 4, 8, 16])