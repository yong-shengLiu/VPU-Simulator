# ==========================================
# 硬體規格與 Tiling 參數定義
# ==========================================
N = 512        # Sequence Length
H = 768        # Hidden Size
A = 12         # Attention Heads
D_k = 64       # Head Dimension (768 / 12)
H_ff = 3072    # FFN Intermediate Size

# Tiling Config (配合您的 VRF 4096 bits)
TILE_N_ATTN = 1   # Attention 時，一次處理 1 個 Token (Row)
TILE_N_FFN  = 16  # FFN 時，一次處理 16 個 Tokens

def bert_encoder_layer(Input_X, Weights):
    """
    Input_X: [N, H] 存在 Memory (DRAM/SRAM)
    Output:  [N, H] 寫回 Memory
    """

    # ==========================================
    # Part 1: Q, K, V Projections (純 Tensor Core)
    # ==========================================
    # 這裡不需要 Fusion，因為結果太大 (512x768)，VRF 塞不下。
    # 策略：讓 Tensor Core 全速運算，結果寫回 Memory。
    Q = TensorCore.GEMM(Input_X, Weights.W_Q) # [N, H] -> Memory
    K = TensorCore.GEMM(Input_X, Weights.W_K) # [N, H] -> Memory
    V = TensorCore.GEMM(Input_X, Weights.W_V) # [N, H] -> Memory

    # Logical Reshape: [N, H] -> [A, N, D_k] (透過 Memory Stride 讀取達成)

    # ==========================================
    # Part 2: Multi-Head Attention (您的主戰場)
    # ==========================================
    # 這裡執行 "Row-wise Tiling" + "In-VRF Softmax"
    
    Attn_Out = Allocate_Memory([N, H]) 

    for h in range(A):  # Loop over 12 Heads
        
        # 針對每個 Head，我們一次處理一個 Sequence Row (Token)
        # 這就是您對抗 FlashAttention 的 "Row-wise Tiling"
        for i in range(0, N, TILE_N_ATTN): 
            
            # --- Step 2.1: Load Q Row (LSU) ---
            # 載入 Q 的第 i 行到 VRF
            # v_q 佔用 1 個 Vector Reg (64 elements * 2B = 128B)
            v_q = VPU.Load(Q[h, i, :]) 

            # --- Step 2.2: Compute Score (Tensor Core) ---
            # Q_row (1x64) * K_T (64x512) -> Score (1x512)
            # [FUSION 關鍵]: 結果直接寫入 VRF (v_score)，不寫回 Memory
            # v_score 佔用 1024B (需 LMUL=2)
            v_score = TensorCore.GEMV(v_q, K[h, :, :], dest=VRF) 

            # --- Step 2.3: Scale & Softmax (VPU) ---
            # [FUSION 關鍵]: VPU 直接讀取 v_score 進行運算
            # 這裡利用了前端 Stream-in，在 TC 算完後馬上接手
            VPU.VectorMulScalar(v_score, 1.0 / sqrt(D_k)) 
            VPU.Softmax(v_score) # In-place 或是寫入 v_score_prob
            
            # --- Step 2.4: Compute Context (Tensor Core) ---
            # Score (1x512) * V (512x64) -> Context (1x64)
            # Tensor Core 讀取 VRF 中的 v_score
            v_context = TensorCore.GEMV(v_score, V[h, :, :], dest=VRF)

            # --- Step 2.5: Store Result (LSU) ---
            # 將算完的一個 Row 寫回 Output Buffer
            VPU.Store(v_context, Attn_Out[h, i, :])

    # ==========================================
    # Part 3: Output Projection & Add-Norm
    # ==========================================
    # 這裡可以做簡單的 Fusion
    # 策略：分塊做 GEMM，結果留在 VRF 做 Add+Norm
    
    Final_Attn = Allocate_Memory([N, H])
    
    for i in range(0, N, 16): # Tiling N=16
        # GEMM
        v_gemm_out = TensorCore.GEMM(Attn_Out[i:i+16], Weights.W_O, dest=VRF)
        
        # Residual Add (Vector-Vector Add)
        v_residual = VPU.Load(Input_X[i:i+16])
        v_add_out  = VPU.Add(v_gemm_out, v_residual)
        
        # LayerNorm (Reduction operations)
        v_norm_out = VPU.LayerNorm(v_add_out)
        
        # Store
        VPU.Store(v_norm_out, Final_Attn[i:i+16])


    # ==========================================
    # Part 4: Feed-Forward Network (FFN) (最容易 Fusion)
    # ==========================================
    # 結構: Linear -> GeLU -> Linear
    # Tiling: 一次處理 16 個 Tokens (16 x 768)
    
    FFN_Out = Allocate_Memory([N, H])

    for i in range(0, N, TILE_N_FFN): # Block-wise Tiling
        
        # --- Step 4.1: Up-Projection (Tensor Core) ---
        # Input (16x768) * W1 (768x3072) -> Intermediate (16x3072)
        # 注意: 16x3072 太大 (96KB)，VRF 裝不下！
        # 修改策略：內層再切 Feature Dim (例如一次算 64 個 Features)
        
        v_final_result = VRF.Clear() # Accumulator for Down-Proj

        for f in range(0, H_ff, 64): # Feature Tiling loop
            
            # [FUSION 關鍵]: 計算一小塊中間層 (16x64) -> VRF
            v_mid = TensorCore.GEMM_Partial(Final_Attn[i:i+16], Weights.W1[:, f:f+64], dest=VRF)
            
            # [FUSION 關鍵]: VPU 直接對這一小塊做 Activation
            VPU.GeLU(v_mid) # In-place update
            
            # [FUSION 關鍵]: 馬上乘上第二層權重，累加到結果
            # Intermediate (16x64) * W2 (64x768) -> Output_Partial (16x768)
            TensorCore.GEMM_Accumulate(v_mid, Weights.W2[f:f+64, :], accum=v_final_result)

        # --- Step 4.2: Add & Norm (VPU) ---
        # 這裡 v_final_result 已經包含完整的 Down-Projection 結果
        v_residual_2 = VPU.Load(Final_Attn[i:i+16])
        v_out = VPU.Add(v_final_result, v_residual_2)
        v_out = VPU.LayerNorm(v_out)
        
        # Store
        VPU.Store(v_out, FFN_Out[i:i+16])

    return FFN_Out