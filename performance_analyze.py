import pandas as pd
import math

# =========================================================
# 1. 硬體規格定義 (Hardware Config)
# =========================================================
class HardwareConfig:
    def __init__(self, vlen_bits=4096, total_vregs=32, data_width_bits=8):
        self.vlen_bits = vlen_bits
        self.vlen_bytes = vlen_bits // 8
        self.total_vregs = total_vregs
        self.data_width_bits = data_width_bits
        self.data_width_bytes = data_width_bits // 8
        self.elements_per_vlen = self.vlen_bytes // self.data_width_bytes

    def __str__(self):
        return (f"Hardware[VLEN={self.vlen_bits}b ({self.vlen_bytes}B), "
                f"Regs={self.total_vregs}, "
                f"DType=INT{self.data_width_bits}]")

# =========================================================
# 2. Kernel 分析器 (GEMM Outer Product Strategy)
# =========================================================
class GemmAnalyzer:
    def __init__(self, hw_config):
        self.hw = hw_config

    def analyze(self, layer_name, M, N, K, m_tile_factor, fused_store=False, fused_load_a=False):
        """
        分析 C[M, N] = A[M, K] * B[K, N] 的效能指標
        策略: Output Stationary (Outer Product)
        
        Args:
            m_tile_factor: 一次計算多少個 C 的 Rows (佔用 Accumulators)
            fused_store: True 表示 C 算完後直接留在 VRF 給下個 Op 用 (省去 Store)
            fused_load_a: True 表示 A 是上一層留下來的 (省去 Load) - 較少見但可能
        """
        
        # --- 1. VREG Allocation Check ---
        # 需求: Accumulators (m_tile) + Vector B (1) + Vector A (1)
        regs_needed = m_tile_factor + 1 + 1
        is_valid_alloc = regs_needed <= self.hw.total_vregs
        
        # --- 2. Loop Dimensions ---
        # N 維度 (Width) 被 VLEN 切分
        num_n_tiles = math.ceil(N / self.hw.vlen_bytes) # 這裡假設 N 是 Bytes 寬度，如果是 INT8 剛好
        # 如果 N 是 Elements:
        num_n_tiles = math.ceil(N / self.hw.elements_per_vlen)
        
        # M 維度 (Height) 被 m_tile_factor 切分
        num_m_blocks = math.ceil(M / m_tile_factor)
        
        # K 維度 (Depth) 是 Inner Loop
        num_k_steps = K

        # --- 3. Instruction Counts (Frontend Analysis) ---
        # 總共的 Inner Loop 執行次數 (Tiles * Blocks * K)
        total_k_iters = num_n_tiles * num_m_blocks * num_k_steps
        
        # Load Instructions
        # Vector B: 每個 K step 讀一次 (Unit-stride)
        cnt_load_b = total_k_iters 
        
        # Vector A: 每個 K step 讀一次 (Strided Load, 讀 m_tile 個 scalar)
        cnt_load_a = total_k_iters if not fused_load_a else 0
        
        # Compute Instructions (MACs)
        # 每個 K step, 要對 m_tile 個 Accumulators 做 Outer Product 更新
        cnt_macs = total_k_iters * m_tile_factor
        
        # Store Instructions (C)
        # 只有在 K loop 結束後，每個 M Block 才寫回一次
        cnt_store_c = (num_n_tiles * num_m_blocks * m_tile_factor) if not fused_store else 0
        
        total_instr = cnt_load_b + cnt_load_a + cnt_macs + cnt_store_c

        # --- 4. Data Volume Analysis (Byte Counting) ---
        # 有效數據量 (Theoretical Minimum)
        bytes_a_ideal = M * K * self.hw.data_width_bytes
        bytes_b_ideal = K * N * self.hw.data_width_bytes
        bytes_c_ideal = M * N * self.hw.data_width_bytes
        
        # 實際數據量 (考慮 Reuse)
        # B 矩陣: 每個 M Block 重複讀取一次完整的 B (Reuse Factor = M / M_TILE ??? No)
        # 修正: Outer Loop 是 P(N), M. Inner is K.
        # 對於每個 (n_tile, m_block) 組合，我們跑 K 次。
        # 每個 m_block 都要重新讀取一次 B 的 n_tile 部分。
        # 所以 B 被讀取了 num_m_blocks 次。
        bytes_b_real = (num_n_tiles * self.hw.vlen_bytes) * K * num_m_blocks
        
        # A 矩陣: 對於每個 n_tile，A 的對應 m_block 都要重讀一次。
        # 所以 A 被讀取了 num_n_tiles 次。
        # 每次讀取: m_tile_factor * K * data_width
        bytes_a_real = (num_m_blocks * m_tile_factor * self.hw.data_width_bytes) * K * num_n_tiles if not fused_load_a else 0
        
        bytes_c_real = (M * N * self.hw.data_width_bytes) if not fused_store else 0

        total_bytes_moved = bytes_a_real + bytes_b_real + bytes_c_real
        
        # --- 5. Compute Stats ---
        total_ops = M * N * K * 2 # MAC = 2 Ops (Mul + Add)
        
        return {
            "Layer": layer_name,
            "Dim": f"{M}x{N}x{K}",
            "M_Tile": m_tile_factor,
            "Valid_Alloc": is_valid_alloc,
            "Regs_Used": regs_needed,
            "Instr_Load": cnt_load_a + cnt_load_b,
            "Instr_MAC": cnt_macs,
            "Instr_Store": cnt_store_c,
            "Bytes_Load": bytes_a_real + bytes_b_real,
            "Bytes_Store": bytes_c_real,
            "Total_Bytes": total_bytes_moved,
            "MAC_Ops": total_ops // 2,
            "AI (Ops/Byte)": round(total_ops / total_bytes_moved, 2) if total_bytes_moved > 0 else 0
        }

# =========================================================
# 3. BERT-Base Configuration & Main Execution
# =========================================================
def run_bert_analysis():
    # Setup Hardware: 4096-bit VLEN, 32 Regs, INT8
    hw = HardwareConfig(vlen_bits=4096, total_vregs=32, data_width_bits=8)
    analyzer = GemmAnalyzer(hw)
    
    # BERT-Base Parameters
    # Seq=512, Hidden=768, HeadDim=64, FFN=3072
    # GEMM Notation: C[M, N] = A[M, K] * B[K, N]
    # M: Output Height, N: Output Width, K: Reduction Depth
    
    # Layers to Analyze
    # Tuple: (Name, M, N, K, fused_store?)
    # 注意: N (Width) 對應到 VLEN 切分，M (Height) 對應到 Tile 切分
    workload = [
        # 1. Q, K, V Projections (Input: 512x768, Weight: 768x768)
        # M=512 (Seq), N=768 (Hidden), K=768 (Hidden)
        ("Proj_QKV", 512, 768, 768, False), 
        
        # 2. Attention Score (Q * K^T)
        # Q: 512x64, K^T: 64x512 -> Score: 512x512
        # M=512 (Seq), N=64 (Seq), K=512 (HeadDim)
        # Fused Store: True (Score 直接留在 VRF 給 Softmax)
        ("Attn_Score", 512, 64, 512, True), 
        
        # 3. Attention Context (Prob * V)
        # Prob: 512x512, V: 512x64 -> Out: 512x64
        # M=512 (Seq), N=512 (HeadDim), K=64 (Seq)
        # Fused Load A: True (Prob 已經在 VRF) - 但這裡分析器簡化假設 Load A 還是要算(因為 Tiling 可能切換)，先設 False 保守估計
        ("Attn_Context", 512, 512, 64, False),
        
        # 4. Output Projection
        # M=512, N=768, K=768
        ("Proj_Out", 512, 768, 768, False),
        
        # 5. FFN Up
        # M=512, N=768, K=3072
        # Fused Store: True (直接給 GeLU)
        ("FFN_Up", 512, 768, 3072, True),
        
        # 6. FFN Down
        # M=512, N=3072, K=768
        ("FFN_Down", 512, 3072, 768, False)
    ]
    
    # 策略比較: M_Tile = 4 vs 8 vs 16
    results = []
    
    # 這裡我們鎖定 M_Tile = 16 (您之前選定的)
    target_m_tile = 16
    
    print(f"=== BERT-Base Layer Analysis (VLEN=512B, INT8, M_Tile={target_m_tile}) ===\n")
    
    for name, m, n, k, fused in workload:
        res = analyzer.analyze(name, m, n, k, m_tile_factor=target_m_tile, fused_store=fused)
        results.append(res)
        
    df = pd.DataFrame(results)
    
    # 格式化輸出
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    
    # 選擇關鍵欄位顯示
    cols = ["Layer", "Dim", "Regs_Used", "Instr_Load", "Instr_MAC", "Instr_Store", "Total_Bytes", "AI (Ops/Byte)"]
    print(df[cols].to_string(index=False))
    
    print("\n[註解]")
    print("1. Instr_Load: 包含 Vector Load B (Unit) 和 Vector Load A (Strided)")
    print("2. Instr_MAC:  實際發射的 FMA 指令數")
    print("3. Total_Bytes: 包含重複讀取 (Reloading) 的開銷")
    print("4. Attn_Score 和 FFN_Up 的 Instr_Store 為 0 (假設 Fusion 生效)")

if __name__ == "__main__":
    run_bert_analysis()