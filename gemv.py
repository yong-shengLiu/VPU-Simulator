import os
import numpy as np
from contextlib import redirect_stdout

# =========================================================
# 1. 內建 MEMORY Class (來自 main_memory.py)
# =========================================================
class MEMORY:
    def __init__(self, BASEADDR=0, DataWidth=64, Depth=409600, debug=False):
        self.BASEADDR  = BASEADDR
        self.DataWidth = DataWidth
        self.Depth     = Depth
        self.debug     = debug
        self.memory = np.zeros((self.Depth), dtype=np.uint64)

    def dumpMem_data(self, mode, Depth=None):
        if mode == 'rtl':
            # 輸出 8-bit Hex 格式供 $readmemh 使用
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
# 2. GEMV 模擬邏輯
# =========================================================
def igemv_simulation(C, A, B, N, P):
    """ 模擬 RISC-V GEMV (LMUL=1, VLEN=4096b) """
    VL_MAX = 512 
    print(f"--- 開始 GEMV 模擬: M=1, N={N}, P={P} ---")
    
    # Strip-mining loop
    for p in range(0, P, VL_MAX):
        current_vl = min(P - p, VL_MAX)
        acc = np.zeros(current_vl, dtype=int) # 使用 int64 避免溢位

        # Reduction loop
        for n in range(N):
            scalar_a = A[0][n]
            vec_b = B[n][p : p + current_vl]
            acc += scalar_a * vec_b # 模擬寬 Accumulator
            
        C[0, p : p + current_vl] = acc

# =========================================================
# 3. 主程式：生成 Pattern 與 DRAM Hex
# =========================================================
def gen_dram_gemv_integrated():
    # --- 參數設定 ---
    M, N, P = 1, 4096, 11008  # LLaMA-7B SwiGLU FFN
    # M, N, P = 1, 4096, 4096   # LLaMA-7B Projection
    # M, N, P = 1, 768, 3072    # BERT-Base FFN
    # M, N, P = 1, 128, 4096    # KV cache attention
    
    # M, N, P = 1, 8, 512
    
    # --- 記憶體映射 (Memory Map) ---
    DRAM_BASE_ADDR = 0xE0000000
    # 1. Vector A (Max 4KB)
    # 範圍: 0x00000000 ~ 0x000FFFFF (給它 1MB 空間，綽綽有餘)
    ADDR_A        = 0x00000000 + DRAM_BASE_ADDR  
    # 2. Matrix B (Max ~43MB)
    # 範圍: 0x00100000 ~ 0x030FFFFF (給它 48MB 空間)
    # 0x3000000 hex = 48 MB
    ADDR_B        = 0x00100000 + DRAM_BASE_ADDR  
    # 3. Output C (RTL Result)
    # 起始點: Base + 1MB (A的偏移) + 48MB (B的空間) = Base + 49MB
    # 49 MB = 0x03100000
    ADDR_C_RTL    = 0x03100000 + DRAM_BASE_ADDR  
    # 4. Golden Reference
    # 起始點: 在 C 後面再加 1MB (0x00100000)
    ADDR_C_GOLDEN = 0x03200000 + DRAM_BASE_ADDR

    print(f"Config: M={M}, N={N}, P={P}")

    # --- 生成資料 ---
    np.random.seed(2026)
    A = np.random.randint(1, 5, size=(M, N)).astype(np.int8)
    B = np.random.randint(1, 5, size=(N, P)).astype(np.int8)
    C_sim = np.zeros((M, P), dtype=int)

    # --- 執行模擬 ---
    igemv_simulation(C_sim, A, B, N, P)

    # --- 驗證正確性 (修正溢位問題) ---
    # 關鍵修正：將 int8 轉為 int32 再做 dot，避免標準答案溢位
    C_ref = np.dot(A.astype(np.int32), B.astype(np.int32))
    
    if np.array_equal(C_sim, C_ref):
        print("\n[PASS] 模擬結果驗證成功！")
    else:
        print("\n[FAIL] 模擬結果與標準運算不符！")
        # Debug 資訊
        diff = C_sim - C_ref
        print(f"Max Diff: {np.max(np.abs(diff))}")
        return

    # --- 準備寫入 DRAM (截斷為 uint8) ---
    A_uint8 = A.view(np.uint8)
    B_uint8 = B.view(np.uint8)
    C_golden_uint8 = (C_sim & 0xFF).astype(np.uint8) # 取低 8 bits

    # --- 寫入記憶體模型 ---
    dram = MEMORY(DataWidth=64, Depth=8000000, BASEADDR=DRAM_BASE_ADDR)
    print("正在寫入記憶體模型...")
    dram.store_data(ADDR_A, 8, A_uint8.flatten())
    dram.store_data(ADDR_B, 8, B_uint8.flatten())
    dram.store_data(ADDR_C_GOLDEN, 8, C_golden_uint8.flatten())

    # --- 匯出 Hex 檔案 ---
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(current_dir, "dram_rtl.hex")
    print(f"匯出 Hex: {output_path}")
    
    with open(output_path, "w", encoding="utf-8") as f:
        with redirect_stdout(f):
            dram.dumpMem_data(mode='rtl')

    # --- 印出 RTL 設定資訊 ---
    print("\n" + "="*40)
    print("   RTL C Code / Runtime Address Settings")
    print("="*40)
    print(f"#define M {M}")
    print(f"#define N {N}")
    print(f"#define P {P}")
    print(f"")
    print(f"// Input Data Address")
    print(f"volatile int8_t* A_ptr = (int8_t*) 0x{ADDR_A:08X}; // Size: {M*N} bytes")
    print(f"volatile int8_t* B_ptr = (int8_t*) 0x{ADDR_B:08X}; // Size: {N*P} bytes")
    print(f"")
    print(f"// Output Data Address")
    print(f"volatile int8_t* C_ptr      = (int8_t*) 0x{ADDR_C_RTL:08X}; // RTL Compute Result")
    print(f"volatile int8_t* Golden_ptr = (int8_t*) 0x{ADDR_C_GOLDEN:08X}; // Golden Result")
    print("="*40)

if __name__ == "__main__":
    gen_dram_gemv_integrated()