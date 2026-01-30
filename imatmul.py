import os
from contextlib import redirect_stdout
import numpy as np


def print_hex_formatted(arr, name="Array"):
    """
    格式化列印函式：
    - 每個元素轉為 8-bit hex (0x00 - 0xff)
    - 每 8 個元素換一行
    """
    print(f"--- {name} Output ---")
    
    # 1. 展平成一維陣列 (Flatten)，方便統一處理
    flat_arr = arr.flatten()
    
    for i, val in enumerate(flat_arr):
        # 2. 強制轉為 uint8 (0-255)
        # 如果是負數 (例如 -1) 會變成 0xff
        # 如果超過 255 (例如 257) 會變成 0x01
        val_8bit = val & 0xFF
        
        # 3. 列印 Hex 格式
        # 0x     : 固定前綴
        # { :02x}: 2位數十六進位，不足補0 (例如 5 -> 05)
        print(f"0x{val_8bit:02x}", end="")
        
        # 4. 處理分隔符號 (逗號與換行)
        is_last_element = (i == len(flat_arr) - 1)
        
        if not is_last_element:
            print(", ", end="")  # 不是最後一個元素，就加逗號
            
        # 5. 每 8 個元素換一行
        # (i + 1) % 8 == 0 表示是第 8, 16, 24... 個元素
        if (i + 1) % 8 == 0:
            print() # 換行
            
    print("\n") # 結束後多空一行

def imatmul_simulation(C, A, B, M, N, P):
    """
    模擬 RISC-V imatmul_4x4 的邏輯
    """
    
    # ---------------------------------------------------------
    # 1. 硬體參數模擬
    # ---------------------------------------------------------
    # 假設我們的 "Vector Length" (VL) 是 4 (對應 vsetvli 的結果)
    # 這決定了我們一次能處理 B 矩陣的幾個 columns
    VL_MAX = 512 
    BLOCK_SIZE_M = 4 # 對應 imatmul_4x4，一次處理 A 的 4 個 rows

    print(f"--- 開始運算: M={M}, N={N}, P={P} ---")
    print(f"--- 模擬 Vector Length (VL) = {VL_MAX} ---")

    # ---------------------------------------------------------
    # 2. P 軸迴圈：將 B 矩陣切成垂直條狀 (Stripes)
    # 對應 C code: for (p = 0; p < P; p += block_size_p)
    # ---------------------------------------------------------
    for p in range(0, P, VL_MAX):  # Assume P == VL_MAX
        # 計算實際的向量長度 (處理邊界情況)
        current_vl = min(P - p, VL_MAX)
        
        print(f"\n[P-Loop] 處理 B 矩陣 Column {p} 到 {p + current_vl - 1}")

        # -----------------------------------------------------
        # 3. M 軸迴圈：將 A 矩陣水平切分，一次處理 4 列
        # 對應 C code: for (m = 0; m < M; m += block_size)
        # -----------------------------------------------------
        for m in range(0, M, BLOCK_SIZE_M):
            print(f"  [M-Loop] 處理 A 矩陣 Row {m} 到 {m+3}")

            # === 初始化 Accumulators (模擬 v0, v4, v8, v12) ===
            # 這些暫存器用來累積 C 矩陣的結果
            # 它們是向量！長度為 current_vl
            acc0 = np.zeros(current_vl, dtype=int) # 模擬 v0
            acc1 = np.zeros(current_vl, dtype=int) # 模擬 v4
            acc2 = np.zeros(current_vl, dtype=int) # 模擬 v8
            acc3 = np.zeros(current_vl, dtype=int) # 模擬 v12

            # -------------------------------------------------
            # 4. N 軸迴圈：核心運算 (K 軸累積)
            # 對應 C code: while (n < N)
            # -------------------------------------------------
            for n in range(N): # Hidden dimension, accumulation times
                # A: 取出 4 個 "純量" (Scalars)
                # 對應: t0=*a, t1=*a...
                val_a0 = A[m][n]     if m < M else 0
                val_a1 = A[m+1][n]   if m+1 < M else 0
                val_a2 = A[m+2][n]   if m+2 < M else 0
                val_a3 = A[m+3][n]   if m+3 < M else 0

                # B: 取出 1 個 "向量" (Vector)
                # 對應: vle64.v v16, (%0)
                # 這是從 B 的第 n 列，取出長度為 current_vl 的片段
                vec_b = B[n][p : p + current_vl]

                # === 執行運算: vmacc.vx (Vector Multiply-Accumulate) ===
                # 邏輯: 向量 += 純量 * 向量
                acc0 += val_a0 * vec_b  # v0  += t0 * v16
                acc1 += val_a1 * vec_b  # v4  += t1 * v16
                acc2 += val_a2 * vec_b  # v8  += t2 * v16
                acc3 += val_a3 * vec_b  # v12 += t3 * v16

                # (Debug: 印出第一輪的運算細節)
                if m == 0 and p == 0 and n < 2:
                    print(f"    [N={n}] 純量 A[{val_a0}, {val_a1}..] * 向量 B{vec_b} -> 累加")

            # -------------------------------------------------
            # 5. 寫回結果
            # 對應 C code: vse64.v v0, (%0) ...
            # -------------------------------------------------
            if m < M:   C[m,     p : p + current_vl] = acc0
            if m+1 < M: C[m + 1, p : p + current_vl] = acc1
            if m+2 < M: C[m + 2, p : p + current_vl] = acc2
            if m+3 < M: C[m + 3, p : p + current_vl] = acc3

# ==========================================
# 測試資料準備
# ==========================================
if __name__ == "__main__":
    print("=== imatmul Generator testbench ===")
    print("version: 2026.01.27")

    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(current_dir, "log", "imatmul.txt")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)   # create the output path

    M, N, P = 1, 8, 512
    
    # 建立隨機矩陣
    np.random.seed(42)
    A = np.random.randint(1, 5, size=(M, N))
    B = np.random.randint(1, 5, size=(N, P))
    C_sim = np.zeros((M, P), dtype=int)
    
    # 執行模擬
    imatmul_simulation(C_sim, A, B, M, N, P)
    
    # 驗證正確性
    C_ref = np.dot(A, B)
    if np.array_equal(C_sim, C_ref):
        print("\n\033[92m[成功] 模擬結果與標準矩陣乘法一致！\033[0m")
        print("C (Result):\n", C_sim)
    else:
        print("\n[失敗] 結果不一致")

    with open(output_path, "w", encoding="utf-8") as f:
        with redirect_stdout(f):
            print("Matrix A (Hex):")
            print_hex_formatted(A, name="Matrix A")
            
            print("\nMatrix B (Hex):")
            print_hex_formatted(B, name="Matrix B")
            
            print("\nResult C (Hex):")
            print_hex_formatted(C_sim, name="Result C")