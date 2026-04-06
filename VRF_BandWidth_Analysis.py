import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 🔧 論文級圖表全局設定
# ==========================================
# 設定支援中文的字體
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Arial'] 
plt.rcParams['axes.unicode_minus'] = False 
plt.rcParams['font.size'] = 12

# 數據準備
workloads = ['GEMM\n(VALU 硬算, 無 CIM)', 'Softmax\n(Ping-Pong 雙緩衝)']
lsu_active = [99.3, 29.5]
valu_active = [8.9, 70.5]

# VRF Bandwidth Data (Per Lane)
# GEMM Lane 0: LSU = 3.469 (LOAD) + 0.094 (STORE) = 3.563 MB. VALU = 1.125*3 (RD) + 1.125 (WR) = 4.5 MB
# Softmax Lane 0: LSU = 0.125 (LOAD) + 0.125 (STORE) = 0.25 MB. VALU = 1.25*2 (RD) + 1.25 (WR) = 3.75 MB
lsu_bw = [3.563, 0.250]
valu_bw = [4.500, 3.750]

# ==========================================
# 📊 圖表一：管線稼動率反轉 (Grouped Bar Chart)
# ==========================================
fig1, ax1 = plt.subplots(figsize=(8, 6))
x = np.arange(len(workloads))
width = 0.35

rects1 = ax1.bar(x - width/2, lsu_active, width, label='LSU 稼動率 (%)', color='#e74c3c', edgecolor='black', linewidth=1.2)
rects2 = ax1.bar(x + width/2, valu_active, width, label='VALU 稼動率 (%)', color='#3498db', edgecolor='black', linewidth=1.2)

ax1.set_ylabel('Execution Unit Active Rate (%)', fontsize=14, fontweight='bold')
ax1.set_title('架構瓶頸轉移：Memory-Bound vs Compute-Bound', fontsize=16, fontweight='bold', pad=20)
ax1.set_xticks(x)
ax1.set_xticklabels(workloads, fontsize=14, fontweight='bold')
ax1.set_ylim(0, 120)
ax1.legend(fontsize=12, loc='upper right')
ax1.grid(axis='y', linestyle='--', alpha=0.7)

# 加上數值標籤
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax1.annotate(f'{height}%', xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 5), textcoords="offset points", ha='center', va='bottom', fontsize=12, fontweight='bold')

autolabel(rects1)
autolabel(rects2)

plt.tight_layout()
plt.savefig('fig1_active_rate_shift.png', dpi=300, bbox_inches='tight')
plt.show()

# ==========================================
# 📊 圖表二：VRF 頻寬消耗成分分析 (Stacked Bar Chart)
# ==========================================
fig2, ax2 = plt.subplots(figsize=(8, 6))
bar_width = 0.4

p1 = ax2.bar(workloads, valu_bw, bar_width, label='VALU 流量 (內部純運算)', color='#3498db', edgecolor='black', linewidth=1.2)
p2 = ax2.bar(workloads, lsu_bw, bar_width, bottom=valu_bw, label='LSU 流量 (外部記憶體搬運)', color='#f39c12', edgecolor='black', linewidth=1.2)

ax2.set_ylabel('Data Traffic per Lane (MB)', fontsize=14, fontweight='bold')
ax2.set_title('VRF 頻寬解剖：是誰吃光了暫存器頻寬？', fontsize=16, fontweight='bold', pad=20)
ax2.set_ylim(0, 9)
ax2.legend(fontsize=12, loc='upper right')
ax2.grid(axis='y', linestyle='--', alpha=0.7)

# 標上數值
for i in range(len(workloads)):
    ax2.text(i, valu_bw[i]/2, f'{valu_bw[i]:.2f} MB', ha='center', va='center', color='white', fontweight='bold', fontsize=12)
    # 避免數值太小擠在一起
    lsu_y_pos = valu_bw[i] + lsu_bw[i]/2 if lsu_bw[i] > 0.5 else valu_bw[i] + lsu_bw[i] + 0.3
    ax2.text(i, lsu_y_pos, f'{lsu_bw[i]:.2f} MB', ha='center', va='center', color='black', fontweight='bold', fontsize=12)

# 加上關鍵註解
ax2.annotate('缺乏 Reduction，\nLSU 需瘋狂搬運資料', xy=(0, 8.5), ha='center', color='#c0392b', fontweight='bold', fontsize=12)
ax2.annotate('Ping-Pong 發威，\n資料高效率重用', xy=(1, 4.5), ha='center', color='#27ae60', fontweight='bold', fontsize=12)

plt.tight_layout()
plt.savefig('fig2_vrf_bandwidth_composition.png', dpi=300, bbox_inches='tight')
plt.show()