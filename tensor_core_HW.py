# import numpy as np

# class TensorCoreSimulator:
#     def __init__(self, K_dim=64, N_dim=16, debug=False):
#         """
#         K_dim: MACs per column (64) - The reduction dimension per cycle
#         N_dim: Number of columns (16) - The output width per cycle
#         """
#         self.K_dim = K_dim
#         self.N_dim = N_dim
#         self.weight_buffer = np.zeros((K_dim, N_dim), dtype=np.int8)
#         self.debug = debug

#     def load_weights(self, weight_tile):
#         """
#         Simulate AXI -> SRAM Weight loading
#         weight_tile shape must be (K_dim, N_dim)
#         """
#         assert weight_tile.shape == (self.K_dim, self.N_dim), f"Weight shape mismatch! Expect {self.K_dim}x{self.N_dim}"
#         self.weight_buffer = weight_tile.copy()
#         if self.debug: print(f"[HW] Loaded Weight Tile {weight_tile.shape}")

#     def execute_cycle(self, input_vector):
#         """
#         Simulate 1 cycle of Tensor Core operation
#         input_vector: shape (1, K_dim) - Broadcast to all columns
#         Returns: partial_sum (1, N_dim) in int32
#         """
#         assert input_vector.shape == (1, self.K_dim)
#         # Hardware behavior: 64x int8 MACs per column
#         # Broadcasting input (1, 64) against Weights (64, 16)
#         # Result is (1, 16)
#         partial_sum = np.matmul(input_vector.astype(np.int32), self.weight_buffer.astype(np.int32))
#         return partial_sum

# def requantize_to_8b(data_int32):
#     """
#     Simulate the final stage before writing to VRF.
#     Scaling down and clamping to int8 range.
#     """
#     # Simple Mock Quantization: divide by constant and clamp
#     scaled = data_int32 // 64 
#     clamped = np.clip(scaled, -128, 127)
#     return clamped.astype(np.int8)

# # ==========================================
# # Strategy 1: Standard Weight Stationary
# # Scenario: Linear Projection (M x K_in) @ (K_in x N_out)
# # ==========================================
# def simulation_strategy_1_standard(tc, M, K, N):
#     print(f"\n=== Strategy 1: Standard Weight Stationary (Size: {M}x{K}x{N}) ===")
    
#     # 1. Prepare Data
#     Input_A = np.random.randint(-10, 10, (M, K)).astype(np.int8)
#     Weight_B = np.random.randint(-10, 10, (K, N)).astype(np.int8)
    
#     # Golden Reference
#     Golden_Ref = np.matmul(Input_A.astype(np.int32), Weight_B.astype(np.int32))
    
#     # Simulator Memory (VRF/Accumulators)
#     Sim_Output = np.zeros((M, N), dtype=np.int32)

#     K_tile_size = tc.K_dim # 64
#     N_tile_size = tc.N_dim # 16
    
#     for n_start in range(0, N, N_tile_size):
#         n_end = n_start + N_tile_size
#         for k_start in range(0, K, K_tile_size):
#             k_end = k_start + K_tile_size
            
#             # [Step A] Load Weight Tile (Stationary)
#             w_chunk = Weight_B[k_start:k_end, n_start:n_end]
#             tc.load_weights(w_chunk)
            
#             # [Step B] Stream Inputs (M)
#             for m in range(M):
#                 input_vec = Input_A[m:m+1, k_start:k_end] # (1, 64)
#                 psum = tc.execute_cycle(input_vec)
                
#                 # Accumulate in VRF/SRAM
#                 # Here psum is (1, 16), Sim_Output slice is (1, 16). Shapes match.
#                 Sim_Output[m:m+1, n_start:n_end] += psum
                
#     if np.allclose(Sim_Output, Golden_Ref):
#         print(">> [SUCCESS] Strategy 1 Dataflow Matches Golden Reference!")
#         Final_VRF_Content = requantize_to_8b(Sim_Output)
#     else:
#         print(">> [FAIL] Strategy 1 Dataflow Error!")

# # ==========================================
# # Strategy 2: Input-Weight Swap (For Softmax)
# # Scenario: Attention Score Q(M, D) @ K.T(D, M) -> Output(M, M)
# # ==========================================
# def simulation_strategy_2_swap(tc, Seq_Len, Head_Dim):
#     print(f"\n=== Strategy 2: Input-Weight Swap (Q@K.T for Softmax) ===")
    
#     Q_Mat = np.random.randint(-10, 10, (Seq_Len, Head_Dim)).astype(np.int8)
#     K_Mat = np.random.randint(-10, 10, (Seq_Len, Head_Dim)).astype(np.int8) 
#     KT_Mat = K_Mat.T 
    
#     Golden_Ref = np.matmul(Q_Mat.astype(np.int32), KT_Mat.astype(np.int32))
#     Sim_Output = np.zeros((Seq_Len, Seq_Len), dtype=np.int32)
    
#     # Loop over Rows of Q (Chunking M by 16)
#     for q_row_start in range(0, Seq_Len, tc.N_dim): # Step 16
#         q_row_end = q_row_start + tc.N_dim
        
#         # [Step A] Load Q chunk into Weight Buffer (Transposed!)
#         q_chunk = Q_Mat[q_row_start:q_row_end, :] # (16, 64)
#         tc.load_weights(q_chunk.T) # Load as (64, 16)
        
#         # [Step B] Stream ALL columns of K.T (rows of K)
#         for k_idx in range(Seq_Len):
#             k_vec = K_Mat[k_idx:k_idx+1, :] # (1, 64)
            
#             # Compute: Input(1,64) @ Weights(64,16) -> Output(1,16)
#             partial_results = tc.execute_cycle(k_vec) 
            
#             # FIX HERE: partial_results is (1, 16), target slice is (16,)
#             # We use .flatten() to match the shape
#             Sim_Output[q_row_start:q_row_end, k_idx] = partial_results.flatten()
        
#     if np.allclose(Sim_Output, Golden_Ref):
#         print(">> [SUCCESS] Strategy 2 Matches!")
#     else:
#         print(">> [FAIL] Strategy 2 Error")

# # ==========================================
# # Strategy 3: Tiled Accumulation (FFN Limit)
# # Scenario: FFN (M x H) @ (H x 4H)
# # ==========================================
# def simulation_strategy_3_split(tc, M, H, H_4):
#     print(f"\n=== Strategy 3: Tiled Accumulation (FFN Limit) ===")
    
#     Input = np.random.randint(-5, 5, (M, H)).astype(np.int8)
#     Weights = np.random.randint(-5, 5, (H, H_4)).astype(np.int8)
#     Golden = np.matmul(Input.astype(np.int32), Weights.astype(np.int32))
#     DRAM_Output = np.zeros((M, H_4), dtype=np.int8) 
    
#     M_block_size = 32 
#     N_tile_size = tc.N_dim 
    
#     for m_start in range(0, M, M_block_size):
#         m_end = m_start + M_block_size
#         for n_start in range(0, H_4, N_tile_size):
#             n_end = n_start + N_tile_size
            
#             vrf_acc = np.zeros((m_end - m_start, N_tile_size), dtype=np.int32)
            
#             for k_start in range(0, H, tc.K_dim):
#                 k_end = k_start + tc.K_dim
                
#                 w_chunk = Weights[k_start:k_end, n_start:n_end]
#                 tc.load_weights(w_chunk)
                
#                 for i, m_idx in enumerate(range(m_start, m_end)):
#                     input_vec = Input[m_idx:m_idx+1, k_start:k_end]
#                     psum = tc.execute_cycle(input_vec)
                    
#                     # FIX HERE: vrf_acc[i] is (16,), psum is (1, 16)
#                     # Use .flatten() to convert (1, 16) -> (16,)
#                     vrf_acc[i] += psum.flatten()
            
#             DRAM_Output[m_start:m_end, n_start:n_end] = requantize_to_8b(vrf_acc)

#     Golden_Quantized = requantize_to_8b(Golden)
#     if np.allclose(DRAM_Output, Golden_Quantized):
#         print(">> [SUCCESS] Strategy 3 Dataflow Matches!")
#     else:
#         print(">> [FAIL] Strategy 3 Error")

# # ==========================================
# # Main Execution
# # ==========================================
# if __name__ == "__main__":
#         # Define Hardware
#     # K=64 (MACs/col), N=16 (Columns)
#     TC = TensorCoreSimulator(K_dim=64, N_dim=16)
    
#     # Run Scenarios (BERT-base sizes)
#     # 1. Linear: Batch=32 (small for verification), H=768
#     simulation_strategy_1_standard(TC, M=512, K=768, N=768)
    
#     # 2. Attention Score: Seq=128 (to be faster), Head=64
#     simulation_strategy_2_swap(TC, Seq_Len=128, Head_Dim=64)
    
#     # 3. FFN: Batch=32, H=768, 4H=3072
#     simulation_strategy_3_split(TC, M=32, H=768, H_4=3072)


import numpy as np

class TensorCore_HW:
    def __init__(self, K_dim=64, N_dim=16, M_block=64, debug=False):
        self.K_dim = K_dim
        self.N_dim = N_dim
        self.M_block = M_block
        
        # TC 內部的硬體資源
        # There has 8bit weight buffer for each MAC
        # Take 32bit accumulate register for each column
        self.weight_buffer = np.zeros((K_dim, N_dim), dtype=np.int8)       # 1KB SRAM
        self.psum_accumulator = np.zeros((M_block, N_dim), dtype=np.int32) # 4KB SRAM
        
        self.debug = debug

    def load_weight_from_vrf(self, vrf_weight_data):
        """模擬從 VRF 將 8-bit Weight 載入 TC 內部 SRAM"""
        self.weight_buffer = vrf_weight_data.copy()

    def clear_accumulator(self):
        """當 K 迴圈開始前，清空內部的 32-bit 累加器"""
        self.psum_accumulator.fill(0)

    def execute_mac_cycle(self, m_idx, input_vector):
        """
        模擬 1 cycle 的 MAC 運算
        input_vector: 來自 VRF 的 1x64 8-bit vector
        m_idx: 當前處理的 Token index (用來定址內部 Accumulator)
        """
        # (1, 64) @ (64, 16) -> (1, 16)
        mac_result = np.matmul(input_vector.astype(np.int32), self.weight_buffer.astype(np.int32))
        
        # 讀改寫 (Read-Modify-Write) 完全在 TC 內部發生！不佔用 VRF 頻寬！
        self.psum_accumulator[m_idx, :] += mac_result.flatten()

    def requantize_and_output(self):
        """
        K 迴圈結束後，將內部的 32-bit 結果量化為 8-bit
        準備寫回 VRF
        """
        scaled = self.psum_accumulator // 64
        clamped = np.clip(scaled, -128, 127)
        return clamped.astype(np.int8)


def simulate_vrf_ping_pong_dataflow(TC, M, K, N):
    print(f"\n=== VRF Double-Buffering Dataflow Simulation (Size: {M}x{K}x{N}) ===")
    
    # Golden Data
    Input_A = np.random.randint(-5, 5, (M, K)).astype(np.int8)
    Weight_B = np.random.randint(-5, 5, (K, N)).astype(np.int8)
    Golden_Ref_32b = np.matmul(Input_A.astype(np.int32), Weight_B.astype(np.int32))
    Golden_Ref_8b = np.clip(Golden_Ref_32b // 64, -128, 127).astype(np.int8)
    
    # 模擬主記憶體 (DRAM)
    DRAM_Output = np.zeros((M, N), dtype=np.int8)
    
    # 硬體規格與 Tiling
    M_blk  = TC.M_block  # 64
    K_tile = TC.K_dim    # 64
    N_tile = TC.N_dim    # 16
    
    # 模擬 VRF 暫存器空間 (字典表示)
    # Ping-Pong Buffer 實作
    VRF = {
        "A_ping":   np.zeros((M_blk, K_tile), dtype=np.int8),  # v0~v7
        "A_pong":   np.zeros((M_blk, K_tile), dtype=np.int8),  # v8~v15
        "W_ping":   np.zeros((K_tile, N_tile), dtype=np.int8), # v16~v17
        "W_pong":   np.zeros((K_tile, N_tile), dtype=np.int8), # v18~v19
        "O_ping":   np.zeros((M_blk, N_tile), dtype=np.int8),  # v20~v21
        "O_pong":   np.zeros((M_blk, N_tile), dtype=np.int8),  # v22~v23
        "Reserved": np.zeros((0, 0), dtype=np.int8)            # v24~v31
    }

    # 效能計數器
    axi_read_bytes = 0
    axi_write_bytes = 0
    
    # Loop Order: M -> N -> K
    # Output Stationary: K 最裡面
    # Row first: N 在 M 的內圈
    for m_start in range(0, M, M_blk):
        m_end = m_start + M_blk

        for n_start in range(0, N, N_tile):
            n_end = n_start + N_tile

            # Clear the psum acc register (Tensor Unit)
            TC.clear_accumulator()
            
            # 使用 flag 來切換 Ping-Pong
            ping_pong_flag = True
            
            for k_start in range(0, K, K_tile):
                k_end = k_start + K_tile
                
                # --- [記憶體操作階段 (AXI -> VRF)] ---
                # 在真實硬體中，這裡的 Load 會與上一個迴圈的 Compute 平行執行
                # 決定這次資料要存入 Ping 還是 Pong
                A_buf_name = "A_ping" if ping_pong_flag else "A_pong"
                W_buf_name = "W_ping" if ping_pong_flag else "W_pong"
                
                # AXI Load Input to VRF (Load Unit)
                VRF[A_buf_name] = Input_A[m_start:m_end, k_start:k_end]
                axi_read_bytes += VRF[A_buf_name].nbytes
                
                # AXI Load Weight to VRF (Load Unit)
                VRF[W_buf_name] = Weight_B[k_start:k_end, n_start:n_end]
                axi_read_bytes += VRF[W_buf_name].nbytes
                
                # --- [運算階段 (VRF -> TC)] ---
                # 1. 從 VRF 載入 Weight 到 TC 內部 SRAM (Tensor Load Weight)
                TC.load_weight_from_vrf(VRF[W_buf_name])
                
                # 2. Vector Streaming: 將 Input 從 VRF 餵入 TC (Tensor Load Input)
                for m_idx in range(M_blk):
                    input_vec = VRF[A_buf_name][m_idx:m_idx+1, :]
                    TC.execute_mac_cycle(m_idx, input_vec)
                    
                # 切換 Ping Pong
                ping_pong_flag = not ping_pong_flag
                
            # 當 K 迴圈走完 (768 深度累加完畢)
            # --- [寫回階段 (TC -> VRF -> AXI)] ---
            
            # 1. TC 內部 32-bit Requantize，輸出 8-bit 給 VRF
            O_buf_name = "O_ping" if ping_pong_flag else "O_pong"
            VRF[O_buf_name] = TC.requantize_and_output()
            
            # 2. AXI Store: 將 VRF 的 8-bit 結果寫回 DRAM
            DRAM_Output[m_start:m_end, n_start:n_end] = VRF[O_buf_name]
            axi_write_bytes += VRF[O_buf_name].nbytes

    # 驗證正確性
    if np.allclose(DRAM_Output, Golden_Ref_8b):
        print(">> [SUCCESS] VRF Hardware Dataflow & Requantize Matches!")
        print("-" * 50)
        print(f"Total Computation (MACs): {M * K * N}")
        print(f"Total AXI Read:  {axi_read_bytes / 1024:.2f} KB")
        print(f"Total AXI Write: {axi_write_bytes / 1024:.2f} KB")
        
        # 理論最佳 AXI Read (只讀一次 Input + 一次 Weight)
        ideal_read = (M*K + K*N) / 1024
        print(f"Ideal AXI Read:  {ideal_read:.2f} KB")
        print(f"Data Read Overhead: {(axi_read_bytes/1024)/ideal_read:.2f}X (Due to Tiling)")
    else:
        print(">> [FAIL] Dataflow mismatch!")

if __name__ == "__main__":
    # BERT-base projection (e.g., Q, K, V)
    TC = TensorCore_HW(K_dim=64, N_dim=16, M_block=64)

    # Attention Projection
    simulate_vrf_ping_pong_dataflow(TC, M=512, K=768, N=768)

    # Attention Score
    simulate_vrf_ping_pong_dataflow(TC, M=512, K=64, N=512)

    # Context
    simulate_vrf_ping_pong_dataflow(TC, M=512, K=512, N=64)