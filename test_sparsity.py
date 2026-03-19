import torch
import torch.nn.functional as F
import random
import matplotlib.pyplot as plt
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset
from tqdm import tqdm

# 全域變數：控制稀疏率
GLOBAL_SPARSITY_RATIO = 0.0

# ==============================================================================
# 1. 終極優化版：攔截 PyTorch Softmax，實作 Hybrid Oracle Block Sparsity
# ==============================================================================
original_softmax = F.softmax

def custom_softmax(input, *args, **kwargs):
    if input.dim() == 4 and GLOBAL_SPARSITY_RATIO > 0:
        batch, heads, seq_len_q, seq_len_k = input.shape
        block_size = 32
        
        if seq_len_q == seq_len_k and seq_len_q % block_size == 0:
            num_blocks = seq_len_q // block_size
            
            # --- 1. 計算 Block 級別的分數 (Max Pooling) ---
            pool_input = input.view(batch * heads, 1, seq_len_q, seq_len_k)
            block_scores = F.max_pool2d(pool_input, kernel_size=block_size, stride=block_size)
            block_scores = block_scores.squeeze(1).view(batch, heads, num_blocks, num_blocks)
            
            # --- 2. 處理因果遮罩 ---
            causal_mask = torch.tril(torch.ones(num_blocks, num_blocks, device=input.device))
            block_scores = block_scores.masked_fill(causal_mask == 0, float('-inf'))
            
            # 🛡️ 修正：大模型稀疏化的「三大黃金護盾」 🛡️
            
            # 護盾 A: 保護對角線 (Current Block)
            eye_mask = torch.eye(num_blocks, device=input.device)
            block_scores = block_scores.masked_fill(eye_mask == 1, float('inf'))
            
            # 護盾 B: 保護前一個區塊 (Local Window，維持語法連貫)
            sub_eye_mask = torch.diag(torch.ones(num_blocks - 1, device=input.device), diagonal=-1)
            block_scores = block_scores.masked_fill(sub_eye_mask == 1, float('inf'))
            
            # 護盾 C: 保護第 0 個區塊 (Attention Sink，穩定 Softmax)
            block_scores[:, :, :, 0] = float('inf')
            
            # --- 3. 產生 Block Mask ---
            keep_ratio = 1.0 - GLOBAL_SPARSITY_RATIO
            block_mask = torch.zeros_like(block_scores)
            
            for row_b in range(num_blocks):
                num_valid_blocks = row_b + 1 
                
                # 確保我們至少保留那些被「護盾」保護的 Block (最多 3 個)
                min_keep = min(3, num_valid_blocks)
                # 使用 round，避免 int() 造成 10% Sparsity 卻過度砍掉 Block
                num_keep = max(min_keep, int(round(num_valid_blocks * keep_ratio)))
                
                row_scores = block_scores[:, :, row_b, :num_valid_blocks]
                _, topk_indices = torch.topk(row_scores, k=num_keep, dim=-1)
                
                block_mask[:, :, row_b].scatter_(-1, topk_indices, 1.0)
                
            # --- 4. 放大 Mask 並套用 ---
            ones_block = torch.ones((block_size, block_size), device=input.device)
            expanded_mask = torch.kron(block_mask, ones_block)
            
            input = input.masked_fill(expanded_mask == 0, -1e4)

    return original_softmax(input, *args, **kwargs)

F.softmax = custom_softmax

# ==============================================================================
# 2. 評估模型困惑度 (Perplexity) 的主程式
# ==============================================================================
def evaluate_wikitext2(model, tokenizer, device, seq_len=1024, limit_batches=100):
    test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    encodings = tokenizer("\n\n".join(test["text"]), return_tensors="pt")
    
    nlls = []
    total_len = min(encodings.input_ids.size(1), seq_len * limit_batches)
    
    for i in tqdm(range(0, total_len, seq_len), desc=f"Eval (Sparsity={GLOBAL_SPARSITY_RATIO:.1f})", leave=False):
        begin_loc = i
        end_loc = begin_loc + seq_len
        if end_loc > encodings.input_ids.size(1):
            break
            
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        
        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            nlls.append(outputs.loss)

    ppl = torch.exp(torch.stack(nlls).mean())
    return ppl.item()

# ==============================================================================
# 3. 執行實驗掃描與自動畫圖
# ==============================================================================
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Running on {device}")
    print("Loading Model and Dataset...")
    
    model_id = "gpt2" 
    tokenizer = GPT2Tokenizer.from_pretrained(model_id)
    
    try:
        model = GPT2LMHeadModel.from_pretrained(model_id, attn_implementation="eager").to(device)
    except TypeError:
        model = GPT2LMHeadModel.from_pretrained(model_id).to(device)
        
    model.eval()

    # 準備掃描的 Sparsity 列表 (0.0 到 0.8)
    # 注意：0.0 代表 Baseline (不丟棄任何合法的過去 Block)
    sparsities = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    perplexities = []

    print("\n" + "="*50)
    print(" 開始進行 Oracle Block Sparsity 敏感度測試")
    print("="*50)

    for sp in sparsities:
        GLOBAL_SPARSITY_RATIO = sp
        ppl = evaluate_wikitext2(model, tokenizer, device, seq_len=1024, limit_batches=100)
        perplexities.append(ppl)
        print(f"✅ Sparsity: {sp*100:2.0f}% | Perplexity: {ppl:.2f}")

    # ==============================================================================
    # 4. 使用 Matplotlib 繪製學術級圖表
    # ==============================================================================
    print("\n📊 正在生成圖表...")
    plt.figure(figsize=(8, 6))
    
    # 畫出折線圖，加上標記點
    plt.plot(sparsities, perplexities, marker='o', linestyle='-', color='#1f77b4', linewidth=2, markersize=8)
    
    # 標示出 Baseline 的線 (基準線)
    plt.axhline(y=perplexities[0], color='r', linestyle='--', alpha=0.5, label=f'Baseline PPL ({perplexities[0]:.2f})')
    
    # 設定圖表標題與軸標籤 (學術風)
    plt.title('GPT-2 Perplexity vs. Content-based Block Sparsity', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Sparsity Ratio (Block-wise Bypass Rate)', fontsize=12)
    plt.ylabel('Perplexity (Lower is Better)', fontsize=12)
    
    # 設定 X 軸的刻度為百分比形式
    plt.xticks(sparsities, [f"{int(s*100)}%" for s in sparsities])
    
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend(fontsize=11)
    plt.tight_layout()
    
    # 存檔
    save_path = "sparsity_vs_perplexity.png"
    plt.savefig(save_path, dpi=300)
    print(f"🎉 實驗完成！圖表已成功儲存為: {save_path}")