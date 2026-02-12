import numpy as np
import os
import glob

# ==========================================
# 1. 配置與常數
# ==========================================
CONFIG = {
    "N_SEQ": 512,
    "H_DIM": 768,
    "H_FF": 3072,
    "LAYERS": 12,
    "HEADS": 12 
}

# Memory Simulation Constants
INT8_MIN = -128
INT8_MAX = 127

# 路徑設定
current_dir = os.path.dirname(os.path.abspath(__file__))
GOLDEN_DIR = os.path.join(current_dir, "golden_log")   # Python 產出的黃金樣本
RTL_DIR    = os.path.join(current_dir, "rtl_log")      # 預期 RTL 產出的 dump 位置

# ==========================================
# 2. 輔助函數
# ==========================================
def to_int8(x):
    x = np.array(x, dtype=np.int32)
    return (x & 0xFF).astype(np.int8)

def to_int16(x):
    x = np.array(x, dtype=np.int32)
    return (x & 0xFFFF).astype(np.int16)

# ==========================================
# 3. 硬體行為模擬 (Kernel Simulation)
# ==========================================
def sim_gemm_int8(A, B):
    res_int32 = np.matmul(A.astype(np.int32), B.astype(np.int32))
    return to_int8(res_int32)

def sim_softmax_q88(x):
    rows, cols = x.shape
    out = np.zeros_like(x, dtype=np.int8)
    for r in range(rows):
        row_val = x[r].astype(np.int16)
        max_val = np.max(row_val)
        row_val = row_val - max_val
        probs = np.exp(row_val) / np.sum(np.exp(row_val))
        out_row = np.round(probs * 127).astype(np.int32)
        out[r] = np.clip(out_row, -128, 127).astype(np.int8)
    return out

def sim_layernorm_fake(data):
    rows, cols = data.shape
    out = np.zeros_like(data, dtype=np.int8)
    for r in range(rows):
        row = data[r].astype(np.int16)
        mean = np.mean(row).astype(np.int16)
        row = row - mean
        res = row # Fake inv_std = 1.0
        out[r] = np.clip(res, INT8_MIN, INT8_MAX).astype(np.int8)
    return out

def sim_gelu_fake(data):
    x_int32 = data.astype(np.int32)
    sqr = (x_int32 * x_int32)
    sqr_trun = to_int8(sqr).astype(np.int32)
    shft = sqr_trun >> 2
    add = shft + 10
    return to_int8(add)

def sim_add_residual(dest, src):
    res = dest.astype(np.int32) + src.astype(np.int32)
    return to_int8(res)

# ==========================================
# 4. 初始化權重
# ==========================================
def init_weights():
    np.random.seed(42)
    W = {}
    scale = 10 
    dims = [
        ('WQ', CONFIG['H_DIM'], CONFIG['H_DIM']),
        ('WK', CONFIG['H_DIM'], CONFIG['H_DIM']),
        ('WV', CONFIG['H_DIM'], CONFIG['H_DIM']),
        ('WO', CONFIG['H_DIM'], CONFIG['H_DIM']),
        ('W1', CONFIG['H_DIM'], CONFIG['H_FF']),
        ('W2', CONFIG['H_FF'], CONFIG['H_DIM'])
    ]
    for name, r, c in dims:
        W[name] = np.random.randint(-scale, scale, (r, c)).astype(np.int8)
    return W

# ==========================================
# 5. 輸出工具
# ==========================================
def dump_memory(data, filename):
    if not os.path.exists(GOLDEN_DIR):
        os.makedirs(GOLDEN_DIR)
    
    filepath = os.path.join(GOLDEN_DIR, filename)
    flat = data.flatten()
    
    with open(filepath, 'w') as f:
        for val in flat:
            u8 = val & 0xFF
            f.write(f"{u8:02X}\n")
    # print(f"Dumped: {filename}") # 減少 log 雜訊，需要時打開

# ==========================================
# 6. 單層 Encoder 模擬 (含細粒度 Dump)
# ==========================================
def run_layer_sim(in_data, W, layer_idx):
    """
    執行單層模擬，並在每個步驟後 Dump 結果
    """
    l_prefix = f"L{layer_idx:02d}" # 例如 L00, L01
    
    # --- MSA Block ---
    
    # 1. Q = X @ WQ
    Q = sim_gemm_int8(in_data, W['WQ'])
    dump_memory(Q, f"{l_prefix}_00_Q.hex")
    
    # 2. K = X @ WK
    K = sim_gemm_int8(in_data, W['WK'])
    dump_memory(K, f"{l_prefix}_01_K.hex")
    
    # 3. Scores = Q @ K.T
    Scores = sim_gemm_int8(Q, K.T)
    dump_memory(Scores, f"{l_prefix}_02_Scores.hex")
    
    # 4. Softmax
    Attn = sim_softmax_q88(Scores)
    dump_memory(Attn, f"{l_prefix}_03_Attn.hex")
    
    # 5. V = X @ WV
    V = sim_gemm_int8(in_data, W['WV'])
    dump_memory(V, f"{l_prefix}_04_V.hex")
    
    # 6. Context = Attn @ V
    Context = sim_gemm_int8(Attn, V)
    dump_memory(Context, f"{l_prefix}_05_Context.hex")
    
    # 7. Output Proj = Context @ WO
    Out_MSA = sim_gemm_int8(Context, W['WO'])
    dump_memory(Out_MSA, f"{l_prefix}_06_Out_MSA.hex")
    
    # 8. Residual + LayerNorm (Norm1)
    Res1 = sim_add_residual(Out_MSA, in_data)
    # dump_memory(Res1, f"{l_prefix}_07_Res1_PreNorm.hex") # Optional
    Norm1 = sim_layernorm_fake(Res1)
    dump_memory(Norm1, f"{l_prefix}_08_Norm1.hex")
    
    # --- FFN Block ---
    
    # 9. Inter = Norm1 @ W1
    Inter = sim_gemm_int8(Norm1, W['W1'])
    dump_memory(Inter, f"{l_prefix}_09_FFN_Inter.hex")
    
    # 10. GELU
    Act = sim_gelu_fake(Inter)
    dump_memory(Act, f"{l_prefix}_10_FFN_Act.hex")
    
    # 11. Out_FFN = Act @ W2
    Out_FFN = sim_gemm_int8(Act, W['W2'])
    dump_memory(Out_FFN, f"{l_prefix}_11_FFN_Out.hex")
    
    # 12. Residual + LayerNorm (Norm2) -> Final Output
    Res2 = sim_add_residual(Out_FFN, Norm1)
    # dump_memory(Res2, f"{l_prefix}_12_Res2_PreNorm.hex") # Optional
    Norm2 = sim_layernorm_fake(Res2)
    dump_memory(Norm2, f"{l_prefix}_13_Norm2_Final.hex")
    
    return Norm2

# ==========================================
# 7. RTL 比對工具
# ==========================================
def load_hex_file(filepath):
    """讀取 hex file 並轉成 list of integers"""
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
        # 過濾空行與非 hex 內容
        data = [int(line.strip(), 16) for line in lines if line.strip()]
        return data
    except Exception as e:
        return None

def verify_rtl_results():
    """
    自動遍歷 GOLDEN_DIR 中的所有 hex 檔案，
    並嘗試在 RTL_DIR 中尋找同名檔案進行比對。
    """
    print(f"\n🔍 Starting RTL Verification...")
    print(f"   Golden Dir: {GOLDEN_DIR}")
    print(f"   RTL Dir   : {RTL_DIR}")
    
    if not os.path.exists(RTL_DIR):
        print(f"❌ RTL directory not found. Please create '{RTL_DIR}' and place RTL dumps there.")
        return

    # 取得所有 Golden 檔案
    golden_files = sorted(glob.glob(os.path.join(GOLDEN_DIR, "*.hex")))
    
    if not golden_files:
        print("⚠️ No golden files found. Run the simulation first.")
        return

    pass_count = 0
    fail_count = 0
    missing_count = 0

    for g_path in golden_files:
        filename = os.path.basename(g_path)
        r_path = os.path.join(RTL_DIR, filename)
        
        if not os.path.exists(r_path):
            print(f"⚠️ [MISSING] {filename} not found in RTL dir.")
            missing_count += 1
            continue
            
        # Load data
        golden_data = load_hex_file(g_path)
        rtl_data = load_hex_file(r_path)
        
        if golden_data is None or rtl_data is None:
            print(f"❌ [ERROR] Could not read {filename}")
            fail_count += 1
            continue

        # Check Length
        if len(golden_data) != len(rtl_data):
            print(f"❌ [FAIL] {filename} Length Mismatch! Golden: {len(golden_data)}, RTL: {len(rtl_data)}")
            fail_count += 1
            continue
            
        # Check Content
        mismatches = 0
        first_mismatch_idx = -1
        
        for idx, (g_val, r_val) in enumerate(zip(golden_data, rtl_data)):
            # 注意: 兩邊讀進來應該都是 0-255 的 unsigned int
            if g_val != r_val:
                mismatches += 1
                if first_mismatch_idx == -1:
                    first_mismatch_idx = idx
        
        if mismatches == 0:
            print(f"✅ [PASS] {filename}")
            pass_count += 1
        else:
            print(f"❌ [FAIL] {filename} has {mismatches} mismatches.")
            print(f"   First error at index {first_mismatch_idx}: Golden=0x{golden_data[first_mismatch_idx]:02X}, RTL=0x{rtl_data[first_mismatch_idx]:02X}")
            fail_count += 1

    print("\n" + "="*40)
    print("Verification Summary")
    print("="*40)
    print(f"Total Files: {len(golden_files)}")
    print(f"PASS       : {pass_count}")
    print(f"FAIL       : {fail_count}")
    print(f"MISSING    : {missing_count}")
    print("="*40)

# ==========================================
# 8. 主程式
# ==========================================
if __name__ == "__main__":
    import sys
    
    # 模式選擇
    # 預設執行產生 Golden，如果有參數 "verify" 則執行比對
    mode = "generate"
    if len(sys.argv) > 1 and sys.argv[1] == "verify":
        mode = "verify"

    if mode == "generate":
        print(f"🚀 Generating Granular Golden Models...")
        print(f"📂 Output directory: {GOLDEN_DIR}")
        
        # 1. Init Data
        np.random.seed(123)
        input_data = np.random.randint(INT8_MIN, INT8_MAX, 
                                       (CONFIG['N_SEQ'], CONFIG['H_DIM'])).astype(np.int8)
        Weights = init_weights()
        
        # Save Initial Input
        dump_memory(input_data, "init_input.hex")
        
        # 2. Loop Layers
        current_in = input_data
        
        for i in range(CONFIG['LAYERS']):
            print(f"Simulating Layer {i}...")
            output = run_layer_sim(current_in, Weights, layer_idx=i)
            current_in = output # Ping-Pong logic
            
        print("\n✅ Generation Complete.")
        print("To verify RTL results, put RTL hex files in 'rtl_log' and run:")
        print("python bert_golden_gen_granular.py verify")
        
    elif mode == "verify":
        verify_rtl_results()