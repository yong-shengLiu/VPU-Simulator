import numpy as np

# ==========================================
# 1. 模型規格定義 (BERT-Base Config)
# ==========================================
CONFIG = {
    "N": 512,        # Sequence Length
    "H": 768,        # Hidden Size
    "A": 12,         # Attention Heads
    "D_k": 64,       # Head Dimension
    "H_ff": 3072,    # Feed-Forward Intermediate Size
    "Layers": 12     # [新增] Encoder Layer 的層數
}

# ==========================================
# Helper Functions (維持不變)
# ==========================================
def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True)) # along the last dimension (along row, Qn for each head)
    return e_x / np.sum(e_x, axis=-1, keepdims=True)

def layer_norm(x, gamma, beta, eps=1e-12):
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    x_norm = (x - mean) / np.sqrt(var + eps)
    return gamma * x_norm + beta

def gelu(x):
    return x * 0.5 * (1.0 + np.tanh(0.7978845608 * (x + 0.044715 * np.power(x, 3))))

def matmul(A, B):
    return np.matmul(A, B)

# ==========================================
# [修改] 初始化權重 (產生 12 層的參數)
# ==========================================
def init_bert_weights():
    print(f"正在初始化 {CONFIG['Layers']} 層 BERT 權重...")
    all_layers_weights = []
    
    for i in range(CONFIG['Layers']):
        W = {}
        # Attention Projections
        W['W_Q'] = np.random.randn(CONFIG['H'], CONFIG['H']) * 0.02
        W['W_K'] = np.random.randn(CONFIG['H'], CONFIG['H']) * 0.02
        W['W_V'] = np.random.randn(CONFIG['H'], CONFIG['H']) * 0.02
        W['W_O'] = np.random.randn(CONFIG['H'], CONFIG['H']) * 0.02
        
        # Feed-Forward Network
        W['W_1'] = np.random.randn(CONFIG['H'], CONFIG['H_ff']) * 0.02
        W['W_2'] = np.random.randn(CONFIG['H_ff'], CONFIG['H']) * 0.02
        
        # LayerNorm Parameters
        W['LN1_g'] = np.ones(CONFIG['H'])
        W['LN1_b'] = np.zeros(CONFIG['H'])
        W['LN2_g'] = np.ones(CONFIG['H'])
        W['LN2_b'] = np.zeros(CONFIG['H'])
        
        all_layers_weights.append(W)
        
    return all_layers_weights

# ==========================================
# [微調] 單層 Encoder Layer (增加 layer_idx 方便顯示)
# ==========================================
def bert_encoder_layer(x, weights, layer_idx):
    N, H = x.shape
    
    # 為了版面簡潔，我們只印出第一層和最後一層的詳細資訊，中間省略
    verbose = (layer_idx == 0) or (layer_idx == CONFIG['Layers'] - 1)
    
    if verbose:
        print(f"\n--- Layer {layer_idx + 1} Start ---")

    # 1. Multi-Head Self-Attention
    Q = matmul(x, weights['W_Q'])
    K = matmul(x, weights['W_K'])
    V = matmul(x, weights['W_V'])
    
    Q_split = Q.reshape(N, CONFIG['A'], CONFIG['D_k']).transpose(1, 0, 2)
    K_split = K.reshape(N, CONFIG['A'], CONFIG['D_k']).transpose(1, 0, 2)
    V_split = V.reshape(N, CONFIG['A'], CONFIG['D_k']).transpose(1, 0, 2)
    
    scores = matmul(Q_split, K_split.transpose(0, 2, 1))
    scores = scores / np.sqrt(CONFIG['D_k'])
    attn_probs = softmax(scores)
    context = matmul(attn_probs, V_split)
    
    context = context.transpose(1, 0, 2).reshape(N, H)
    attn_output = matmul(context, weights['W_O'])
    x_attn = layer_norm(x + attn_output, weights['LN1_g'], weights['LN1_b'])

    # 2. Feed-Forward Network
    intermediate = matmul(x_attn, weights['W_1'])
    activated = gelu(intermediate)
    ffn_output = matmul(activated, weights['W_2'])
    x_final = layer_norm(x_attn + ffn_output, weights['LN2_g'], weights['LN2_b'])
    
    if verbose:
        print(f"Layer {layer_idx + 1} Output Shape: {x_final.shape}")
    
    return x_final

# ==========================================
# [修改] 主程式：執行完整的 BERT Inference
# ==========================================
if __name__ == "__main__":
    # 1. 產生模擬輸入 (Batch=1, Seq=512, Hidden=768)
    # 這就是 Embedding Layer 出來的結果
    current_activation = np.random.randn(CONFIG['N'], CONFIG['H'])
    print(f"Initial Input Shape: {current_activation.shape}")
    
    # 2. 初始化所有層的權重 (List of Dictionaries)
    bert_weights = init_bert_weights()
    
    # 3. 執行 12 層堆疊運算 (Pipeline)
    print(f"\n🚀 開始執行 BERT-Base (12 Layers) Inference...")
    
    for i in range(CONFIG['Layers']):
        # [關鍵邏輯]: 上一層的輸出 (current_activation) 變成下一層的輸入
        # 每一層使用自己獨立的權重 (bert_weights[i])
        current_activation = bert_encoder_layer(current_activation, bert_weights[i], i)
        
    # 4. 最終結果
    final_output = current_activation
    print(f"\n✅ 全程 Inference 完成！")
    print(f"最終輸出形狀: {final_output.shape} (應為 [512, 768])")
    print(f"前 5 個數值範例 (Feature Vector of [CLS]):\n{final_output[0, :5]}")