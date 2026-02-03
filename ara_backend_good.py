import matplotlib.pyplot as plt
import numpy as np
import os

# ==========================================
# 1. 數據輸入 (基於你提供的最新截圖)
# ==========================================
# Case Names
labels = [
    'GEMV\n(Baseline)', 
    'GEMM M=4\n(Baseline)', 
    'GEMM M=16\n(Scalar Bound)', 
    'GEMM M=16\n(Fake Scalar)'
]

# 原始數據 (Total, Compute, Load, Overlap)
data = {
    'Total':   np.array([2351688, 84618, 85223, 44041]),
    'Compute': np.array([ 909440, 42117, 35200, 35455]),
    'Load':    np.array([2130440, 34320,  8976,  8976]),
    'Overlap': np.array([ 835640, 11374,  2831,  3352])
}

# ==========================================
# 2. 數據計算 (核心邏輯)
# ==========================================
# 計算 "Solo" 時間 (只有一個單元在動，另一個被 Block 的時間)
solo_compute = data['Compute'] - data['Overlap']
solo_load    = data['Load']    - data['Overlap']

# 計算 "Effective Work" (真正有在做事的時間)
effective_work = solo_compute + solo_load + data['Overlap']

# 計算 "Overhead" (不明原因的浪費: Scalar Stall, Pipeline Bubbles)
overhead = data['Total'] - effective_work
overhead = np.maximum(overhead, 0) # 避免負數微小誤差

# 歸一化 (Normalization) -> 轉成百分比，方便跨量級比較
total = data['Total']
pct_overlap  = (data['Overlap'] / total) * 100
pct_compute  = (solo_compute    / total) * 100
pct_load     = (solo_load       / total) * 100
pct_overhead = (overhead        / total) * 100

# ==========================================
# 3. 繪圖
# ==========================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

# --- 圖 1: Efficiency Breakdown (百分比堆疊圖) ---
bar_width = 0.6
x = np.arange(len(labels))

# 繪製堆疊 (順序: Overlap -> Solo Compute -> Solo Load -> Overhead)
p1 = ax1.bar(x, pct_overlap,  bar_width, label='Overlap (Ideal Hiding)', color='#2ca02c', alpha=0.9)
p2 = ax1.bar(x, pct_compute,  bar_width, bottom=pct_overlap, label='Solo Compute (Blocking)', color='#1f77b4', alpha=0.8)
p3 = ax1.bar(x, pct_load,     bar_width, bottom=pct_overlap+pct_compute, label='Solo Load (Blocking)', color='#ff7f0e', alpha=0.8)
p4 = ax1.bar(x, pct_overhead, bar_width, bottom=pct_overlap+pct_compute+pct_load, label='System Overhead (Scalar Stall)', color='#d62728', alpha=0.8)

# 標註數值 (百分比)
for i in range(len(x)):
    # 標註 Overhead (如果是大於 2% 才標，避免太擠)
    if pct_overhead[i] > 2:
        ax1.text(i, 100 - pct_overhead[i]/2, f"{pct_overhead[i]:.1f}%", ha='center', va='center', color='white', fontweight='bold')
    # 標註 Overlap
    if pct_overlap[i] > 1:
        ax1.text(i, pct_overlap[i]/2, f"{pct_overlap[i]:.1f}%", ha='center', va='center', color='white', fontweight='bold')

ax1.set_ylabel('Execution Time Breakdown (%)')
ax1.set_title('Normalized Efficiency Analysis: Where did the cycles go?', fontsize=14, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(labels, fontsize=11)
ax1.set_ylim(0, 100)
ax1.legend(loc='lower right', bbox_to_anchor=(1.0, 1.05), ncol=2) # Legend 放上面避免遮擋

# --- 圖 2: Absolute Cycles (GEMM Optimization Path) ---
# 只看 GEMM 的三個階段，加上 "Ideal Target"
gemm_labels = ['M=16\n(Current)', 'M=16\n(Fake Scalar)', 'Dual Issue\n(Target)']
gemm_indices = [2, 3] # 對應 data 中的 index
current_val = data['Total'][2]
fake_val    = data['Total'][3]
# Ideal Target = Max(Compute, Load) for M=16 Fake Scalar
ideal_val   = max(data['Compute'][3], data['Load'][3])

values = [current_val, fake_val, ideal_val]
colors = ['#d62728', '#7f7f7f', '#9467bd'] # 紅(爛) -> 灰(過渡) -> 紫(理想)

x2 = np.arange(len(gemm_labels))
bars = ax2.bar(x2, values, color=colors, width=0.5)

# 畫出箭頭與 Gap
# 1. Scalar Bottleneck Gap
ax2.annotate(f'-{current_val - fake_val:,} cycles\n(Scalar Overhead)', 
             xy=(1, fake_val), xytext=(0.5, (current_val+fake_val)/2),
             arrowprops=dict(arrowstyle='->', lw=1.5), ha='center')

# 2. Structural Blocking Gap
ax2.annotate(f'-{fake_val - ideal_val:,} cycles\n(Issue Blocking)', 
             xy=(2, ideal_val), xytext=(1.5, (fake_val+ideal_val)/2),
             arrowprops=dict(arrowstyle='->', lw=1.5), ha='center')

ax2.bar_label(bars, fmt='{:,.0f}', padding=3, fontsize=12, fontweight='bold')
ax2.set_ylabel('Total Cycle Count')
ax2.set_title('Path to Optimization: Quantifying the Gains', fontsize=14, fontweight='bold')
ax2.set_xticks(x2)
ax2.set_xticklabels(gemm_labels, fontsize=11)
ax2.grid(axis='y', linestyle='--', alpha=0.3)

plt.tight_layout()

# 存檔
current_dir = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(current_dir, 'gemv_gemm_analysis.png')
plt.savefig(output_path)
print(f"圖表已儲存至: {output_path}")
plt.show()