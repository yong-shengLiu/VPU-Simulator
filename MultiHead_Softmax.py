import os
import numpy as np
from contextlib import redirect_stdout

def print_hex16_formatted(arr, name="Array"):
    """
    格式化列印函式 (針對 int16)：
    - 每個元素轉為 16-bit hex (0x0000 - 0xffff)
    - 每 8 個元素換一行
    """
    print(f"--- {name} Output ---")
    flat_arr = arr.flatten()
    
    for i, val in enumerate(flat_arr):
        # 強制轉為 uint16 (0-65535) 處理負數
        val_16bit = int(val) & 0xFFFF
        print(f"0x{val_16bit:04x}", end="")
        
        if i != len(flat_arr) - 1:
            print(", ", end="")
            
        if (i + 1) % 8 == 0:
            print() 
    print("\n")

def softmax_simulation(Input_Heads, Num_Heads, Seq_Len):
    """
    模擬 RISC-V VPU Softmax (Three-Pass Algorithm)
    
    Args:
        Input_Heads: 輸入資料 [Num_Heads, Seq_Len]
        Num_Heads: Head 數量
        Seq_Len: 序列長度 (M dimension)
    
    Hardware: VLEN=4096 bits, SEW=16 -> VL_MAX = 256
    """
    
    # 硬體參數
    VL_MAX = 256
    
    # 輸出容器 (Q15 format)
    Output_Heads = np.zeros_like(Input_Heads)

    print(f"=== 開始 Softmax 運算 ===")
    print(f"Config: Heads={Num_Heads}, SeqLen={Seq_Len}")
    print(f"Hardware: VL_MAX={VL_MAX} elements (int16)\n")

    # =========================================================
    # Outer Loop: Head 維度 (這是 Decouple 最能發揮的地方)
    # LSU 應該要在 Head i 計算時，預取 Head i+1
    # =========================================================
    for h in range(Num_Heads):
        print(f"==========================================")
        print(f"Processing Head #{h}")
        print(f"==========================================")
        
        # 取得當前 Head 的資料指標
        # Pointer: Input_Ptr + h * Seq_Len * 2 (bytes)
        row_data = Input_Heads[h]
        
        # -----------------------------------------------------
        # PASS 1: Find Global Max (Reduction)
        # 目的：為了數值穩定性，Softmax(x) = Softmax(x - max(x))
        # -----------------------------------------------------
        global_max = -32768 # Min int16
        
        print(f"  [Pass 1] Finding Global Max...")
        for p in range(0, Seq_Len, VL_MAX):
            current_vl = min(Seq_Len - p, VL_MAX)
            
            # Load Vector
            # asm: vle16.v v1, (ptr)
            vec_chunk = row_data[p : p + current_vl]
            
            # Vector Reduction Max
            # asm: vredmax.vs v0, v1, v0
            local_max = np.max(vec_chunk)
            if local_max > global_max:
                global_max = local_max
                
            print(f"    - Stripe {p//VL_MAX}: Local Max = {local_max}")

        print(f"    -> Global Max found: {global_max}")

        # -----------------------------------------------------
        # PASS 2: Calculate Exp & Sum (Accumulate)
        # 這裡通常是效能瓶頸 (Exp 計算慢)
        # -----------------------------------------------------
        global_sum = 0.0
        exp_buffer = np.zeros(Seq_Len, dtype=float) # 模擬中間暫存 (VRAM/SRAM)
        
        print(f"  [Pass 2] Sub Max -> Exp -> Sum...")
        for p in range(0, Seq_Len, VL_MAX):
            current_vl = min(Seq_Len - p, VL_MAX)
            
            # Load Vector
            vec_chunk = row_data[p : p + current_vl]
            
            # Vector Sub (x - max)
            # asm: vsub.vx v2, v1, max_scalar
            vec_shifted = vec_chunk - global_max
            
            # Vector Exp (硬體通常用 LUT 或 Taylor 近似)
            # 這裡用 float 模擬精度
            # asm: (Custom Exp Instruction or Loop)
            vec_exp = np.exp(vec_shifted)
            
            # Store Exp result for Pass 3 (避免重複算)
            exp_buffer[p : p + current_vl] = vec_exp
            
            # Vector Reduction Sum
            # asm: vfredsum.vs v0, v2, v0
            global_sum += np.sum(vec_exp)
            
            print(f"    - Stripe {p//VL_MAX}: Exp calculated, partial sum updated.")

        print(f"    -> Global Sum found: {global_sum:.4f}")

        # -----------------------------------------------------
        # PASS 3: Normalization (Div) & Store
        # 輸出結果定點化為 int16 (Q15 format: 1.0 = 32767)
        # -----------------------------------------------------
        print(f"  [Pass 3] Div -> Quantize -> Store...")
        
        # 避免除以零
        inv_sum = 1.0 / global_sum if global_sum != 0 else 0
        
        for p in range(0, Seq_Len, VL_MAX):
            current_vl = min(Seq_Len - p, VL_MAX)
            
            # Reload Exp Result (from buffer/cache)
            vec_exp = exp_buffer[p : p + current_vl]
            
            # Vector Scale (Div)
            # asm: vfmul.vf v4, v2, inv_sum_scalar
            vec_prob = vec_exp * inv_sum
            
            # Quantize to int16 (Q15 format)
            # 32767 represents probability 1.0
            vec_q15 = np.round(vec_prob * 32767).astype(np.int16)
            
            # Store
            # asm: vse16.v v4, (out_ptr)
            Output_Heads[h, p : p + current_vl] = vec_q15
            
            print(f"    - Stripe {p//VL_MAX}: Stored {current_vl} elements.")

    return Output_Heads

# ==========================================
# 執行與生成 Pattern
# ==========================================
if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(current_dir, "log", "softmax_pattern.txt")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 設定參數 (為了展示 Decouple，我們設 2 個 Head)
    # Seq_Len = 512，剛好是 2 倍 VL_MAX (256)，這會觸發 Strip-mining
    NUM_HEADS = 32
    SEQ_LEN = 4096
    
    # 建立隨機輸入 (模擬 Logits, 範圍 -100 ~ 100)
    np.random.seed(2026)
    Input_Data = np.random.randint(-100, 100, size=(NUM_HEADS, SEQ_LEN)).astype(np.int16)
    
    # 執行模擬
    Result_Data = softmax_simulation(Input_Data, NUM_HEADS, SEQ_LEN)
    
    # 寫入 Pattern 檔案
    with open(output_path, "w", encoding="utf-8") as f:
        with redirect_stdout(f):
            print(f"// Softmax Benchmark Pattern")
            print(f"// Format: INT16 (Q15 output)")
            print(f"// Dims: Heads={NUM_HEADS}, SeqLen={SEQ_LEN}")
            print(f"// Hardware: VLEN=4096, LMUL=1 (VL_MAX=256)\n")
            
            # print_hex16_formatted(Input_Data, name="Input Logits (A)")
            # print_hex16_formatted(Result_Data, name="Output Probabilities (C)")

    print(f"\nPattern generated at: {output_path}")