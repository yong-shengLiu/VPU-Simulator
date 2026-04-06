import matplotlib.pyplot as plt
import numpy as np

# Font configuration for the VM (using default sans-serif, avoiding Chinese to prevent missing glyphs in VM image)
plt.rcParams['font.family'] = 'sans-serif'

# NOTE the DATA from RTL simulation
labels = ['Centralized Scoreboard\n(Ara-like)', 'Decoupled Issue Queues\n(Ours)']
total_cycles = [2040953, 289084]
lsu_rates = [75.0, 29.5]
valu_rates = [25.0, 70.5]

# ==========================================
# Chart 1: Total Cycles Bar Chart
# ==========================================
fig1, ax1 = plt.subplots(figsize=(8, 6))
bars1 = ax1.bar(labels, total_cycles, color=['#e74c3c', '#2ecc71'], width=0.5)

# Add data labels on top of bars
for bar in bars1:
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2, yval + 60000, f'{int(yval):,}', ha='center', va='bottom', fontsize=12, fontweight='bold')

# Annotation for 7x speedup
ax1.annotate('7.06x Speedup!', 
            xy=(1, 400000), xytext=(0.5, 1000000),
            arrowprops=dict(facecolor='black', shrink=0.05, width=2, headwidth=8),
            fontsize=14, fontweight='bold', color='#27ae60', ha='center')

ax1.set_ylabel('Total Simulation Cycles', fontsize=12)
ax1.set_title('Performance Comparison: Total Execution Time', fontsize=14, fontweight='bold')
ax1.set_ylim(0, 2300000)
ax1.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('total_cycles_comparison.png', dpi=300)
plt.close(fig1)

# ==========================================
# Chart 2: Active Rate Stacked Bar Chart
# ==========================================
fig2, ax2 = plt.subplots(figsize=(9, 6))

bar_width = 0.5
p1 = ax2.bar(labels, valu_rates, bar_width, label='VALU Active Rate (%)', color='#3498db')
p2 = ax2.bar(labels, lsu_rates, bar_width, bottom=valu_rates, label='LSU Active Rate (%)', color='#f39c12')

# Add text inside the bars
for i in range(len(labels)):
    ax2.text(i, valu_rates[i]/2, f'{valu_rates[i]}%', ha='center', va='center', color='white', fontsize=12, fontweight='bold')
    ax2.text(i, valu_rates[i] + lsu_rates[i]/2, f'{lsu_rates[i]}%', ha='center', va='center', color='white', fontsize=12, fontweight='bold')

# Annotations
ax2.annotate('Pseudo Memory-Bound\n(Issue Starvation)', 
            xy=(0, 85), xytext=(-0.3, 105),
            ha='center', fontsize=11, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

ax2.annotate('True Compute-Bound\n(VALU Unleashed)', 
            xy=(1, 50), xytext=(1.3, 105),
            ha='center', fontsize=11, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

ax2.set_ylabel('Execution Unit Active Rate (%)', fontsize=12)
ax2.set_title('Pipeline Utilization: LSU vs VALU Active Rate', fontsize=14, fontweight='bold')
ax2.set_ylim(0, 120) # Extra space for annotations
ax2.legend(loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=2)

plt.tight_layout()
plt.savefig('active_rate_comparison.png', dpi=300)
plt.close(fig2)