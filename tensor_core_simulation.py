import numpy as np

class TensorCore16x16:
    def __init__(self):
        # 硬體內部暫存器：16x16 的 INT8 Weight Registers
        self.weights = np.zeros((16, 16), dtype=np.int8)

    def load_weights(self, w_tile):
        """
        [Phase 1: Weight Load] 
        硬體行為：花 16 個 Cycle，從 VRF 把 16 條 vector 塞進 Tensor Core 靜止不動。
        """
        self.weights = np.copy(w_tile)

    def compute_cycle(self, x_vec, psum_in):
        """
        [Phase 2: Compute]
        硬體行為：單一個 Cycle (或 Pipeline 後的有效 Throughput) 的空間運算。
        x_vec   : 來自 VRF 的 1 條 Input Vector (長度 16, INT8)
        psum_in : 來自 VRF 或 Psum Buffer 的 1 條 Psum Vector (長度 16, INT32)
        """
        # 準備輸出的 16 個 INT32 累加結果
        psum_out = np.copy(psum_in)

        # 模擬 16x16 MAC 陣列的空間展開 (Spatial Unrolling)
        for row in range(16):
            # 💡 [Input Broadcast]: 把 x_vec 的第 row 個元素，廣播給這整個橫排 (16個 Column)
            x_broadcast = np.int32(x_vec[row])

            for col in range(16):
                # 💡 [MAC & Accumulate Downwards]: 
                # psum_out 就像垂直貫穿 Column 的線，不斷把 乘積 加上去
                w_stationary = np.int32(self.weights[row, col])
                psum_out[col] += x_broadcast * w_stationary

        return psum_out


def run_npu_matmul(X, W):
    """
    NPU 軟體排程器 (Software Scheduler / Tiling Logic)
    X: M x K (INT8 Activation)
    W: K x N (INT8 Weight)
    """
    M, K = X.shape
    K_w, N = W.shape
    assert K == K_w, "矩陣內維度 K 必須一致！"

    # 最終輸出 Y: M x N (INT32)
    Y = np.zeros((M, N), dtype=np.int32)
    
    # 實例化我們的 16x16 硬體
    tc = TensorCore16x16()

    # =====================================================================
    # 💡 核心 Tiling 邏輯 (排程順序極度重要！)
    # 為了發揮 Weight Stationary 的最大威力，迴圈順序必須是：N -> K -> M
    # =====================================================================
    
    # 1. Tile over N (Output 寬度切塊)
    for n in range(0, N, 16):
        n_end = min(n + 16, N)
        
        # 2. Tile over K (Reduction 維度切塊)
        for k in range(0, K, 16):
            k_end = min(k + 16, K)

            # 處理邊界未對齊 (Padding with zeros)，硬體靠 Mask 解決
            w_tile = np.zeros((16, 16), dtype=np.int8)
            w_tile[0:(k_end-k), 0:(n_end-n)] = W[k:k_end, n:n_end]

            # [硬體呼叫] 載入 Weight (它會靜止不動直到 M 迴圈跑完！)
            tc.load_weights(w_tile)

            # 3. Tile over M (Batch/Sequence 維度，瘋狂 Streaming！)
            # 💡 在這個迴圈裡，Weight 是不用換的，資料重用率 (Data Reuse) 達到最高！
            for m in range(M):
                
                # 從 VRF 讀出一條 16 個 elements 的 Input
                x_vec = np.zeros(16, dtype=np.int8)
                x_vec[0:(k_end-k)] = X[m, k:k_end]

                # 從 VRF 讀出上一回合 (k_step) 累積的 Psum
                psum_in = np.zeros(16, dtype=np.int32)
                psum_in[0:(n_end-n)] = Y[m, n:n_end]

                # [硬體呼叫] 丟進 Tensor Core 運算
                psum_out = tc.compute_cycle(x_vec, psum_in)

                # 把算完的 Psum 寫回 VRF
                Y[m, n:n_end] = psum_out[0:(n_end-n)]

    return Y


# =====================================================================
# 測試與驗證區 (Testbench)
# =====================================================================
if __name__ == "__main__":
    # 設定任意維度的測試矩陣 (例如 BERT 的某一層: M=128, K=256, N=64)
    # 故意設定不是 16 的倍數來測試邊界 Tiling (例如 M=43, K=61, N=39)
    M, K, N = 43, 61, 39

    print(f"🚀 初始化矩陣: X({M}x{K}) * W({K}x{N})")
    
    # 產生 INT8 的隨機矩陣
    np.random.seed(42)
    X_int8 = np.random.randint(-128, 127, size=(M, K), dtype=np.int8)
    W_int8 = np.random.randint(-128, 127, size=(K, N), dtype=np.int8)

    # 1. 跑我們的 NPU 模擬器
    Y_npu = run_npu_matmul(X_int8, W_int8)

    # 2. 跑 Numpy 標準矩陣乘法 (當作 Golden Ground Truth)
    # 注意：要先轉 INT32 避免過程 Overflow
    Y_golden = np.matmul(np.int32(X_int8), np.int32(W_int8))

    # 3. 比對結果
    error = np.sum(np.abs(Y_npu - Y_golden))
    
    print("-" * 50)
    if error == 0:
        print("✅ [PASS] NPU 模擬結果與 Numpy 完全一致！資料一滴都沒漏！")
    else:
        print(f"❌ [FAIL] 發現錯誤，誤差總和: {error}")