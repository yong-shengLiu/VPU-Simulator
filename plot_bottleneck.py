import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. 數據輸入 (基於你的截圖 Toy Bench 16x256x256)
# ==========================================
labels = ['GEMM (M=4)', 'GEMM (M=16)\nScalar Bound', 'GEMM (M=16)\nFake Scalar']

# 原始數據
total_cycles   = np.array([84618, 85223, 44041])
compute_cycles = np.array([42117, 35200, 35455])
load_cycles    = np.array([34320,  8976,  8976])
overlap_cycles = np.array([11374,  2831,  3352])

# ==========================================
# 2. 數據處理 (拆解 Bar Chart 的組成)
# ==========================================
# Solo: 只有該單元在動，另一個在閒置的時間
solo_compute = compute_cycles - overlap_cycles
solo_load    = load_cycles - overlap_cycles

# Effective Work: 真正有在做事的時間 (Compute OR Load)
effective_work = solo_compute + solo_load + overlap_cycles

# Overhead: 總時間 - 有做事的時間 (Scalar Stall, Pipeline Bubbles)
overhead = total_cycles - effective_work
# 修正微小誤差 (避免負數)
overhead = np.maximum(overhead, 0)

# Ideal Time (Decoupled Target): Max(Compute, Load)
ideal_time = np.maximum(compute_cycles, load_cycles)

# ==========================================
# 3. 繪圖設定
# ==========================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# --- 圖表 1: Execution Time Breakdown (堆疊長條圖) ---
bar_width = 0.5
x = np.arange(len(labels))

# 繪製堆疊
p1 = ax1.bar(x, overlap_cycles, bar_width, label='Overlap (Efficient)', color='#2ca02c', alpha=0.9)
p2 = ax1.bar(x, solo_compute,   bar_width, bottom=overlap_cycles, label='Solo Compute (Blocking)', color='#1f77b4', alpha=0.8)
p3 = ax1.bar(x, solo_load,      bar_width, bottom=overlap_cycles+solo_compute, label='Solo Load (Blocking)', color='#ff7f0e', alpha=0.8)
p4 = ax1.bar(x, overhead,       bar_width, bottom=overlap_cycles+solo_compute+solo_load, label='Scalar/System Overhead', color='#d62728', alpha=0.8)

# 標註數值
for i in range(len(x)):
    # 標示 Total Cycle
    ax1.text(x[i], total_cycles[i] + 1000, f"{total_cycles[i]:,}", ha='center', fontweight='bold')
    # 標示 Hiding Efficiency
    efficiency = overlap_cycles[i] / load_cycles[i] if load_cycles[i] > 0 else 0
    ax1.text(x[i], total_cycles[i] + 5000, f"Eff: {efficiency:.2f}", ha='center', color='blue', fontsize=10)

ax1.set_ylabel('Cycle Count')
ax1.set_title('Part 1: The Anatomy of Performance Bottlenecks\n(Why is it slow?)', fontsize=14, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(labels, fontsize=11)
ax1.legend(loc='upper right')
ax1.grid(axis='y', linestyle='--', alpha=0.5)

# --- 圖表 2: Actual vs. Ideal (預測未來) ---
# 比較 "Fake Scalar" 的現狀 vs "Ideal Decoupled"
target_labels = ['Current (Fake Scalar)', 'Ideal Decoupled\n(Dual Issue Target)']
current_val = total_cycles[2] # M=16 Fake Scalar
ideal_val   = ideal_time[2]   # Max(Compute, Load) for M=16

x2 = np.arange(len(target_labels))
bars = ax2.bar(x2, [current_val, ideal_val], color=['#7f7f7f', '#9467bd'], width=0.5)

# 畫出 Gap
ax2.annotate(f'Potential Speedup: {current_val/ideal_val:.2f}x', 
             xy=(1, ideal_val), xytext=(0.5, (current_val+ideal_val)/2),
             arrowprops=dict(arrowstyle='->', lw=2),
             fontsize=12, fontweight='bold', color='purple')

ax2.bar_label(bars, fmt='{:,.0f}', padding=3)
ax2.set_title('Part 2: The Value of Dual Issue\n(Target Performance)', fontsize=14, fontweight='bold')
ax2.set_ylabel('Cycle Count')
ax2.set_xticks(x2)
ax2.set_xticklabels(target_labels, fontsize=11)
ax2.grid(axis='y', linestyle='--', alpha=0.5)

plt.tight_layout()

# 儲存圖片
import os
output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'thesis_performance_analysis.png')
plt.savefig(output_path)
print(f"圖表已儲存至: {output_path}")
plt.show()
