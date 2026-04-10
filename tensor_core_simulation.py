import numpy as np
import sys

# --- 參數設定 ---
M, K, N = 64, 32, 64
TILE = 16
LOG_FILENAME = "hardware_trace.log"

# 固定 Seed，確保每次執行的矩陣都長一樣
np.random.seed(42) 

# 產生數值較小的矩陣 [-5, 5]
A = np.random.randint(-5, 6, size=(M, K))
B = np.random.randint(-5, 6, size=(K, N))

# 開啟檔案準備寫入
with open(LOG_FILENAME, "w", encoding="utf-8") as f:
    
    # 定義一個 helper function 來同時印在終端機和寫入檔案 (可選)
    # 這裡我們選擇直接全部寫入檔案
    def log(text=""):
        print(text, file=f)

    # =====================================================================
    # 1. 產生給 SystemVerilog TB 的初始化程式碼
    # =====================================================================
    log("/* === 請將以下內容複製到 TB 的 prepare_data() 替換原本的迴圈 === */")
    log("begin")
    for i in range(M):
        for k in range(K):
            log(f"    A_mat[{i}][{k}] = {A[i, k]};")
    for k in range(K):
        for j in range(N):
            log(f"    B_mat[{k}][{j}] = {B[k, j]};")
    log("end")
    log("/* ========================================================= */\n")

    # =====================================================================
    # 2. 模擬硬體的 Tiling 排程，並印出每一個 Stage 的 16x16 結果
    # =====================================================================
    log("=== 開始模擬硬體 Tiling 計算過程 ===")

    for m in range(M // TILE):
        for n in range(N // TILE):
            log(f"\n" + "#"*55)
            log(f"### 正在計算 Tile Y(m={m}, n={n}) (輸出矩陣的 {m*16}~{m*16+15} Row, {n*16}~{n*16+15} Col)")
            log("#"*55)
            
            # 準備一個硬體內部的 16x16 Psum Buffer (初始為 0)
            psum_buffer = np.zeros((TILE, TILE), dtype=int)
            
            for k in range(K // TILE):
                # 切割出當下的 16x16 Activation 與 Weight
                a_tile = A[m*16 : (m+1)*16, k*16 : (k+1)*16]
                b_tile = B[k*16 : (k+1)*16, n*16 : (n+1)*16]
                
                # 進行矩陣乘法並累加上去
                mac_result = np.dot(a_tile, b_tile)
                psum_buffer += mac_result
                
                log(f"\n--- [Pass K={k}] ---")
                log(f"這個 Pass 是將 A 矩陣的 Col {k*16}~{k*16+15} 乘上 B 矩陣的 Row {k*16}~{k*16+15}")
                if k == 0:
                    log("硬體狀態：psum_src_sel = 0 (從 VRF 讀入初始 0)，算完後寫入 Internal Psum Buffer")
                else:
                    log("硬體狀態：psum_src_sel = 1 (從 Internal Psum Buffer 讀出舊值並累加)，算完後覆寫 Buffer")
                
                log(f"MAC 陣列算完這個 Pass 後，預期輸出的 16x16 數值 (請核對波形上的 int_mac_psum_out)：")
                
                # 印出 16x16 矩陣，格式化對齊
                for row in psum_buffer:
                    log("    " + " ".join(f"{val:4d}" for val in row))
                    
            log(f"\n==> [Store] Tile Y(m={m}, n={n}) 計算完畢，準備將上述最終結果寫回 VRF。")

print(f"✅ 成功！已將 Golden Model 的排程追蹤與初始化資料輸出至 '{LOG_FILENAME}'")


def write_matrix_to_file(filename, mat):
    with open(filename, "w") as f:
        for row in mat:
            # 將一整 Row 轉成字串寫入，數字間用空格隔開
            f.write(" ".join(str(val) for val in row) + "\n")

# 輸出檔案
Y_golden = np.dot(A.astype(np.int32), B.astype(np.int32))
write_matrix_to_file("A_mat.txt", A)
write_matrix_to_file("B_mat.txt", B)
write_matrix_to_file("Y_golden.txt", Y_golden)

print("✅ 成功產生 A_mat.txt, B_mat.txt, Y_golden.txt！")