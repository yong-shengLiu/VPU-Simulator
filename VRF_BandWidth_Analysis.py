import matplotlib.pyplot as plt
import numpy as np

# Set font for matplotlib
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 1. Data Preparation & Normalization
# ==========================================
# GEMM (128, 768) * (768, 768)
total_mac_ops = 128 * 768 * 768 # 總共 75,497,472 MACs

# Hardware Resources
macs_valu = 32  # 8(8bits vmacc) * 4 lanens = 32 MACs
macs_cim  = 256 # 16 * 16

# Total Cycles
cycles_valu = 3159942
cycles_cim = 1224569

# VRF Traffic per lane (MB)
# VALU (Lane 0): Read = 36.070, Write = 19.148 => Total = 55.218 MB/lane
# CIM (Lane 0): Read = 4.570, Write = 2.852 => Total = 7.422 MB/lane
traffic_valu_mb = 55.218 * 4  # 4 lanes total traffic
traffic_cim_mb = 7.422 * 4    # 4 lanes total traffic

# Metrics Calculation
# 1. Bandwidth Efficiency (Bytes per MAC)
bytes_per_mac_valu = (traffic_valu_mb * 1024 * 1024) / total_mac_ops
bytes_per_mac_cim = (traffic_cim_mb * 1024 * 1024) / total_mac_ops

# 2. Effective Throughput (MACs per cycle)
throughput_valu = total_mac_ops / cycles_valu
throughput_cim = total_mac_ops / cycles_cim

# 3. Hardware Utilization (%)
util_valu = (throughput_valu / macs_valu) * 100
util_cim = (throughput_cim / macs_cim) * 100

labels = ['VALU Baseline\n(1D Tiling, 32 MACs)', 'CIM Tensor Core\n(2D Tiling, 256 MACs)']

# ==========================================
# Chart 1: Bandwidth Efficiency (Bytes/MAC)
# ==========================================
fig1, ax1 = plt.subplots(figsize=(8, 6))
bars1 = ax1.bar(labels, [bytes_per_mac_valu, bytes_per_mac_cim], color=['#8e44ad', '#00ced1'], width=0.5, edgecolor='black', linewidth=1.2)

for bar in bars1:
    yval = bar.get_height()
    # Y 軸最大值縮小了，所以加高的 padding 也要縮小 (0.5 -> 0.1)
    ax1.text(bar.get_x() + bar.get_width()/2, yval + 0.1, f'{yval:.2f} Bytes/MAC', ha='center', va='bottom', fontsize=12, fontweight='bold')

# 動態調整箭頭與文字的位置
ax1.annotate('7.5x Less Memory Traffic!', 
            xy=(1, bytes_per_mac_cim + 0.3),       # 箭頭尖端
            xytext=(1.15, bytes_per_mac_cim + 1.2), # 文字方塊位置 (右上方)
            arrowprops=dict(facecolor='black', shrink=0.05, width=2, headwidth=8),
            fontsize=14, fontweight='bold', color='#00ced1', ha='center')

ax1.set_ylabel('VRF Bandwidth Efficiency (Bytes/MAC)', fontsize=12, fontweight='bold')
ax1.set_title('Normalized Data Reuse: VRF Traffic per MAC Operation', fontsize=14, fontweight='bold', pad=20)
# 根據新數據 (最大約 3.07) 重設 Y 軸上限
ax1.set_ylim(0, 4)
ax1.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('fig1_bw_efficiency.png', dpi=300)
plt.close(fig1)


# ==========================================
# Chart 2: Throughput vs Utilization
# ==========================================
c_throughput_bar = '#008080' # Teal
c_throughput_lbl = '#006666' # Darker Teal for label
c_util_bar = '#e84393'        # Magenta/Pink
c_util_lbl = '#c91c7a'       # Darker Magenta for label

fig2, ax1 = plt.subplots(figsize=(9, 6))

x = np.arange(len(labels))
width = 0.35

# Effective Throughput (Left Y-axis)
rects1 = ax1.bar(x - width/2, [throughput_valu, throughput_cim], width, label='Effective MACs / Cycle', color=c_throughput_bar, edgecolor='black', linewidth=1.2)
ax1.set_ylabel('Absolute Throughput (MACs / Cycle)', fontsize=12, fontweight='bold', color=c_throughput_lbl)
ax1.tick_params(axis='y', labelcolor=c_throughput_lbl)
ax1.set_ylim(0, 80)

# Utilization (Right Y-axis)
ax2 = ax1.twinx()
rects2 = ax2.bar(x + width/2, [util_valu, util_cim], width, label='Hardware MAC Utilization (%)', color=c_util_bar, edgecolor='black', linewidth=1.2)
ax2.set_ylabel('Hardware Utilization (%)', fontsize=12, fontweight='bold', color=c_util_lbl)
ax2.tick_params(axis='y', labelcolor=c_util_lbl)
ax2.set_ylim(0, 100)

ax1.set_xticks(x)
ax1.set_xticklabels(labels, fontsize=12, fontweight='bold')
ax1.set_title('Throughput vs. Hardware Utilization', fontsize=14, fontweight='bold', pad=40)

# Value Labels
for rect in rects1:
    height = rect.get_height()
    ax1.annotate(f'{height:.1f} MAC/cyc', xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 5), textcoords="offset points", ha='center', va='bottom', fontsize=11, fontweight='bold')

for rect in rects2:
    height = rect.get_height()
    ax2.annotate(f'{height:.1f}%', xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 5), textcoords="offset points", ha='center', va='bottom', fontsize=11, fontweight='bold')

# Add combined legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()

ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=2)


plt.subplots_adjust(top=0.85)
plt.tight_layout()

plt.savefig('fig2_throughput_utilization.png', dpi=300, bbox_inches='tight')
plt.close(fig2)