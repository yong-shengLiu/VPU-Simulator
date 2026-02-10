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
        self.Depth     = Depth 
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
            pass

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
# 2. Tiled GEMM 模擬邏輯 (Optimization for Reuse)
# =========================================================
def igemm_simulation_vv_tiled(C, A, B, M, N, P):
    """ 
    模擬使用 Register Tiling 的 vmacc.vv (Dot Product)
    策略：
    1. Unroll M dimension (一次處理 M_TILE 個 Rows)
    2. 將 A 的 M_TILE 個 Rows 鎖在 VRF 中 (vA_0, vA_1...)
    3. Stream B 的 Columns，每個 Column 讀進來後，跟所有 cached A Rows 做運算
    
    優勢：
    - A 的 Vectors 被重複使用 P 次 (大幅減少 A 的 Memory Access)
    - B 的 Vectors 被重複使用 M_TILE 次 (在 Register File 內)
    """
    
    # 硬體參數
    VL_MAX = 512 
    M_TILE = 4   # 假設我們用 4 個 Vector Registers 來存 A 的 Rows (v0-v3)
                 # 另外需要 1 個 Vector Register 存 B (v4)
                 # 剩下的可以做 Accumulators 或 Temp
    
    print(f"--- 開始 GEMM 模擬 (Tiled Dot Product) ---")
    print(f"Strategy: Pin {M_TILE} Rows of A in VRF, Stream B Cols")
    
    # Loop M with Tiling (每次處理 M_TILE 行)
    for m_start in range(0, M, M_TILE):
        
        # 處理邊界條件 (最後可能不足 M_TILE 行)
        current_m_block = min(M - m_start, M_TILE)
        
        # Inner Loop: Reduction Dimension N (Strip-mined)
        for k_start in range(0, N, VL_MAX):
            current_vl = min(N - k_start, VL_MAX)
            
            # -------------------------------------------------
            # [Step 1: Pre-load A Rows into VRF] -> REUSE KEY!
            # -------------------------------------------------
            # 這些 Vector Registers (vA_list) 在接下來的 P 迴圈中會一直被鎖住
            # 模擬: vle8.v v0, (A_ptr_row0)
            #       vle8.v v1, (A_ptr_row1) ...
            vA_cache = []
            for i in range(current_m_block):
                row_idx = m_start + i
                # 連續讀取 A 的 Row Segment
                vec_a = A[row_idx, k_start : k_start + current_vl]
                vA_cache.append(vec_a)
            
            # -------------------------------------------------
            # [Step 2: Stream B Columns]
            # -------------------------------------------------
            for p in range(P):
                
                # Load B Vector (Column Segment)
                # 模擬: vlse8.v v_temp, (B_ptr), stride (Strided Load)
                # (如果是 Attention QK^T，這裡就是 Unit-stride Load，超快)
                vec_b = B[k_start : k_start + current_vl, p]
                
                # -------------------------------------------------
                # [Step 3: Compute Cross Product] -> M_TILE 計算並行
                # -------------------------------------------------
                # B 的這個向量，現在要跟每一個 Cached A 向量做 dot product
                
                for i in range(current_m_block):
                    row_idx = m_start + i
                    
                    # vmacc.vv + vredsum (模擬)
                    # 這裡模擬 element-wise 乘法
                    vec_res = vA_cache[i] * vec_b
                    
                    # Reduction Sum 得到 Scalar Partial Sum
                    partial_sum = np.sum(vec_res)
                    
                    # Accumulate to C (Output Stationary)
                    # 注意：真實硬體上，如果是 Strip-mining N，
                    # 這裡的 C[row_idx, p] 是在多次 k_start 迴圈中累加的
                    C[row_idx, p] += partial_sum

# =========================================================
# 3. 主程式
# =========================================================
def gen_dram_gemm_tiled():
    # --- 測試案例 ---
    # Case: Standard Block
    M, N, P = 512, 768, 768

    # --- 記憶體映射 ---
    DRAM_BASE_ADDR = 0xE0000000
    ADDR_A        = 0x00000000 + DRAM_BASE_ADDR  
    ADDR_B        = 0x00100000 + DRAM_BASE_ADDR  
    ADDR_C_RTL    = 0x03100000 + DRAM_BASE_ADDR  
    ADDR_C_GOLDEN = 0x03200000 + DRAM_BASE_ADDR  

    print(f"Config: M={M}, N={N}, P={P}")

    # --- 生成資料 ---
    np.random.seed(2026)
    A = np.random.randint(1, 5, size=(M, N)).astype(np.int8)
    B = np.random.randint(1, 5, size=(N, P)).astype(np.int8)
    C_sim = np.zeros((M, P), dtype=int)

    # --- 執行模擬 (Tiled VV Version) ---
    igemm_simulation_vv_tiled(C_sim, A, B, M, N, P)

    # --- 驗證正確性 ---
    print("\n驗證運算正確性...")
    C_ref = np.dot(A.astype(np.int32), B.astype(np.int32))
    
    if np.array_equal(C_sim, C_ref):
        print("[PASS] Tiled GEMM 模擬結果與 Numpy 結果一致！")
    else:
        print("[FAIL] 模擬結果錯誤！")
        diff = C_sim - C_ref
        print(f"Max Diff: {np.max(np.abs(diff))}")
        return

    # --- 準備寫入 DRAM ---
    A_uint8 = A.view(np.uint8)
    B_uint8 = B.view(np.uint8)
    C_golden_uint8 = (C_sim & 0xFF).astype(np.uint8)

    # --- 寫入記憶體模型 ---
    dram = MEMORY(DataWidth=64, Depth=8000000, BASEADDR=DRAM_BASE_ADDR)
    print("正在寫入記憶體模型...")
    dram.store_data(ADDR_A, 8, A_uint8.flatten())
    dram.store_data(ADDR_B, 8, B_uint8.flatten())
    dram.store_data(ADDR_C_GOLDEN, 8, C_golden_uint8.flatten())

    # --- 匯出 Hex 檔案 ---
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(current_dir, "dram_gemm_tiled_vv.hex")
    
    # with open(output_path, "w", encoding="utf-8") as f:
    #     with redirect_stdout(f):
    #         dram.dumpMem_data(mode='rtl')

    print(f"\nHex generation complete: {output_path}")

if __name__ == "__main__":
    gen_dram_gemm_tiled()