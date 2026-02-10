import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def analyze_gemm_instructions(M, N, P, m_tile_values):
    """
    量化分析 Outer Product GEMM 的指令數飽和效應
    架構假設:
    - VLEN = 512 bits (64 elements for INT8)
    - 支援 Vector Load A (Strided) 與 Vector Load B (Unit-stride)
    - Output Stationary (Accumulators 鎖在 VRF)
    """
    # 硬體參數
    VLEN_BYTES = 64 # 512 bits / 8 bits (INT8) elements per vector
    
    results = []
    
    # P 維度 (Width) 被 VLEN 切分
    num_p_tiles = np.ceil(P / VLEN_BYTES)
    
    for m_tile in m_tile_values:
        # M 維度 (Height) 被 m_tile 切分 (這就是 Unrolling Factor)
        num_m_blocks = np.ceil(M / m_tile)
        
        # --- 核心分析邏輯 ---
        # 總共需要執行的 "Micro-Kernels" 數量
        # 一個 Micro-Kernel = 讀入一個 B 向量，更新 m_tile 個 accumulators
        total_kernels = num_p_tiles * num_m_blocks * N
        
        # 1. Load Instructions (Overhead)
        # 每次 Kernel 需要:
        #   - 1 個 Vector Load B (Unit-stride)
        #   - 1 個 Vector Load A (Strided) -> 這裡假設硬體支援 strided vector load
        instr_load = total_kernels * 2
        
        # 2. MAC Instructions (Payload)
        # 每次 Kernel 需要:
        #   - m_tile 個 MAC 指令 (對 acc[0]...acc[m_tile-1] 做更新)
        #   注意: 這就是為什麼 MAC 指令總數幾乎不變的原因
        #   總 MAC 指令數 ~= (M * N * P) / VLEN，跟 m_tile 無關
        instr_mac = total_kernels * m_tile
        
        # 3. Store Instructions (Overhead)
        # 只有在算完 N 迴圈後，才把 Accumulators 寫回 Memory
        # 每個 Block 有 m_tile 個 accumulators
        instr_store = (num_p_tiles * num_m_blocks) * m_tile
        
        total_inst = instr_load + instr_mac + instr_store
        
        # 4. Arithmetic Intensity (計算密度指標)
        # 每一個 Load/Store 指令對應多少個運算指令
        ai_ratio = instr_mac / (instr_load + instr_store)

        results.append({
            "m_tile": m_tile,
            "Total Insts": int(total_inst),
            "MAC Insts (Fixed)": int(instr_mac),
            "Load Insts (Variable)": int(instr_load),
            "Store Insts": int(instr_store),
            "Efficiency (MAC%)": f"{instr_mac/total_inst:.1%}",
            "AI Ratio": f"{ai_ratio:.1f}"
        })

    df = pd.DataFrame(results)
    return df

# --- 執行分析 (以 BERT FFN Up-Projection 為例) ---
# M=512, N=768 (Input Dim), P=3072 (Output Dim)
# 觀察 m_tile 從 1 到 32 的變化
M, N, P = 512, 768, 3072
# M, N, P = 32, 256, 256
m_tiles = [1, 2, 4, 8, 16, 24, 32]

df = analyze_gemm_instructions(M, N, P, m_tiles)

# --- 畫圖 (Visualization) ---
plt.figure(figsize=(10, 6))

# 畫出各類指令的趨勢
plt.plot(df["m_tile"], df["Total Insts"], marker='o', linewidth=3, label="Total Instructions (Saturation!)", color='black')
plt.plot(df["m_tile"], df["Load Insts (Variable)"], marker='^', linestyle='--', label="Load Instructions (Overhead)", color='red')
plt.plot(df["m_tile"], df["MAC Insts (Fixed)"], marker='s', linestyle='-', label="MAC Instructions (Payload)", color='blue')

# 標註 Saturation Point
plt.axvline(x=16, color='green', linestyle=':', alpha=0.5)
plt.text(16.5, df["Total Insts"].max()*0.6, "Diminishing Returns\nstart here", color='green')

plt.title(f"Instruction Count Saturation Analysis (GEMM {M}x{N}x{P})")
plt.xlabel("M_Tile (Unrolling Factor)")
plt.ylabel("Number of Instructions")
plt.grid(True, which='both', linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()

# 顯示數據表
print("\n=== Instruction Count Analysis Table ===")
print(df.to_string(index=False))

# 儲存圖片
current_dir = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(current_dir, 'gemm_mtile_saturation.png')
plt.savefig(output_path)
print("\n圖表已儲存為 gemm_saturation.png")