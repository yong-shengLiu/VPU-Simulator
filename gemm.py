import os
import numpy as np
from contextlib import redirect_stdout

# =========================================================
# 1. 內建 MEMORY Class (保持不變)
# =========================================================
class MEMORY:
    def __init__(self, BASEADDR=0, DataWidth=64, Depth=8000000, debug=False):
        self.BASEADDR  = BASEADDR
        self.DataWidth = DataWidth
        self.Depth     = Depth # 預設 64MB 空間
        self.debug     = debug
        self.memory = np.zeros((self.Depth), dtype=np.uint64)

    def dumpMem_data(self, mode, Depth=None):
        if mode == 'rtl':
            for idx, value in enumerate(self.memory):
                for byte in range(8):
                    byte_mask  = 0b11111111 << (byte * 8)
                    byte_value = (int(value) & byte_mask) >> (byte * 8)
                    print(f"{byte_value:02X}")
        else:
            print("Unsupported mode for this simplified version.")

    def store_data(self, start_addr, size, vector):
        relative_start_addr = start_addr - self.BASEADDR
        length = len(vector)
        
        if   size == 8: align_start_addr = relative_start_addr
        elif size == 16: align_start_addr = (relative_start_addr >> 1) << 1
        elif size == 32: align_start_addr = (relative_start_addr >> 2) << 2
        elif size == 64: align_start_addr = (relative_start_addr >> 3) << 3
        else: raise ValueError("store_data: Unsupported data size")

        for idx in range(length):
            mem_addr = (align_start_addr + (idx * size // 8)) // 8
            offset   = (align_start_addr + (idx * size // 8)) % 8 * 8
            
            mask = ((1 << size) - 1) << offset
            inv_mask = ~mask & 0xFFFFFFFFFFFFFFFF
            old_value = self.memory[mem_addr]
            new_value = (int(old_value) & int(inv_mask)) | ((int(vector[idx]) << int(offset)) & int(mask))
            self.memory[mem_addr] = new_value

# =========================================================
# 2. GEMM 模擬邏輯 (加入 m_tile_factor)
# =========================================================
def igemm_simulation(C, A, B, M, N, P, m_tile_factor):
    """ 
    模擬 RISC-V GEMM (C = A * B)
    M: Rows of A and C
    N: Cols of A, Rows of B (Reduction dimension)
    P: Cols of B and C
    m_tile_factor: 模擬使用的 Vector Register 數量 (Hardware Unrolling on M)
    """
    VL_MAX = 512 # 4096 bits / 8 bits
    print(f"--- 開始 GEMM 模擬 ---")
    print(f"Dimensions: M={M}, N={N}, P={P}")
    print(f"Strategy:   VL={VL_MAX}, M_Unroll={m_tile_factor} (Using {m_tile_factor} Accumulators)")
    
    # Outer Loop 1: Strip-mining on P (Output Width)
    for p in range(0, P, VL_MAX):
        current_vl = min(P - p, VL_MAX)
        
        # Outer Loop 2: Tiling on M (Output Height) -> 這是 GEMM 贏過 GEMV 的關鍵
        for m in range(0, M, m_tile_factor):
            current_m_rows = min(M - m, m_tile_factor)
            
            # 1. 初始化 Accumulators (模擬 vmacc.vx 的 dest registers)
            # shape: (current_m_rows, current_vl)
            accs = np.zeros((current_m_rows, current_vl), dtype=int)

            # Inner Loop: Reduction on N
            for k in range(N):
                # Load Weight Vector (模擬 vle8.v) -> 這是被重複使用的資源!
                vec_b = B[k][p : p + current_vl] 
                
                # Compute for each Row in M Tile
                for i in range(current_m_rows):
                    # Load Scalar Input (模擬讀取 A 的 scalar)
                    scalar_a = A[m + i][k]
                    
                    # MAC (模擬 vmacc.vx v[i], scalar_a, v_weight)
                    accs[i] += scalar_a * vec_b
            
            # Store Result (模擬 vse8.v)
            C[m : m + current_m_rows, p : p + current_vl] = accs

# =========================================================
# 3. 主程式
# =========================================================
def gen_dram_gemm_integrated():
    # --- 關鍵變數設定 ---
    # M_TILE_FACTOR 越大，Memory Bound 越不明顯，但需要更多暫存器
    # 建議設為 4 或 8 (Ara 通常有 32 個 vector registers，用 8 個做 acc 很合理)
    M_TILE_FACTOR = 16 

    # --- 測試案例選單 (Uncomment to use) ---
    
    # [Case 1: LLaMA-7B Prefill] The "Monster" - 驗證 High Compute Intensity
    # M, N, P = 128, 4096, 4096 
    
    # [Case 2: BERT-Base Batch] The "Standard"
    # M, N, P = 64, 768, 3072
    
    # [Case 3: Academic Square] 快速測試用 (為了產生 Hex 比較快先用這個)
    M, N, P = 32, 256, 256 
    # M, N, P = 256, 256, 256 
    # M, N, P = 128, 4096, 4096 

    # --- 記憶體映射 (沿用 LLaMA-7B 的安全配置) ---
    DRAM_BASE_ADDR = 0xE0000000
    
    # 1. Matrix A (Size: M*N)
    ADDR_A        = 0x00000000 + DRAM_BASE_ADDR  
    
    # 2. Matrix B (Size: N*P) - Weight Matrix 通常最大
    ADDR_B        = 0x00100000 + DRAM_BASE_ADDR  
    
    # 3. Matrix C (Size: M*P) - Result
    # 讓出 48MB 給 B
    ADDR_C_RTL    = 0x03100000 + DRAM_BASE_ADDR  
    
    # 4. Golden
    ADDR_C_GOLDEN = 0x03200000 + DRAM_BASE_ADDR  

    print(f"Config: M={M}, N={N}, P={P}, Tile_M={M_TILE_FACTOR}")

    # --- 生成資料 ---
    np.random.seed(2026)
    A = np.random.randint(1, 5, size=(M, N)).astype(np.int8)
    B = np.random.randint(1, 5, size=(N, P)).astype(np.int8)
    C_sim = np.zeros((M, P), dtype=int)

    # --- 執行模擬 (GEMM) ---
    igemm_simulation(C_sim, A, B, M, N, P, M_TILE_FACTOR)

    # --- 驗證正確性 ---
    print("\n驗證運算正確性...")
    # 使用 numpy 矩陣乘法驗證
    C_ref = np.dot(A.astype(np.int32), B.astype(np.int32))
    
    if np.array_equal(C_sim, C_ref):
        print("[PASS] GEMM 模擬結果與 Numpy 結果一致！")
    else:
        print("[FAIL] 模擬結果錯誤！")
        diff = C_sim - C_ref
        print(f"Max Diff: {np.max(np.abs(diff))}")
        return

    # --- 準備寫入 DRAM ---
    A_uint8 = A.view(np.uint8)
    B_uint8 = B.view(np.uint8)
    C_golden_uint8 = (C_sim & 0xFF).astype(np.uint8) # Truncate to 8-bit for storage

    # --- 寫入記憶體模型 ---
    # Depth 開大一點避免 overflow
    dram = MEMORY(DataWidth=64, Depth=8000000, BASEADDR=DRAM_BASE_ADDR)
    print("正在寫入記憶體模型 (這可能需要一點時間)...")
    
    dram.store_data(ADDR_A, 8, A_uint8.flatten())
    dram.store_data(ADDR_B, 8, B_uint8.flatten())
    dram.store_data(ADDR_C_GOLDEN, 8, C_golden_uint8.flatten())

    # --- 匯出 Hex 檔案 ---
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(current_dir, "dram_gemm_rtl.hex")
    print(f"匯出 Hex: {output_path}")
    
    # 這裡會產生很大的檔案，請確保磁碟空間足夠
    with open(output_path, "w", encoding="utf-8") as f:
        with redirect_stdout(f):
            dram.dumpMem_data(mode='rtl')

    # --- 印出 C Code Header ---
    print("\n" + "="*40)
    print("   RTL C Code / Runtime Settings")
    print("="*40)
    print(f"#define M {M}")
    print(f"#define N {N}")
    print(f"#define P {P}")
    print(f"#define M_TILE {M_TILE_FACTOR} // Number of Vector Registers for Acc")
    print(f"")
    print(f"// Input Addresses")
    print(f"volatile int8_t* A_ptr = (int8_t*) 0x{ADDR_A:08X}; // Input Activation ({M}x{N})")
    print(f"volatile int8_t* B_ptr = (int8_t*) 0x{ADDR_B:08X}; // Weight Matrix ({N}x{P})")
    print(f"")
    print(f"// Output Addresses")
    print(f"volatile int8_t* C_ptr      = (int8_t*) 0x{ADDR_C_RTL:08X}; // RTL Result ({M}x{P})")
    print(f"volatile int8_t* Golden_ptr = (int8_t*) 0x{ADDR_C_GOLDEN:08X}; // Golden Ref")
    print("="*40)

if __name__ == "__main__":
    gen_dram_gemm_integrated()