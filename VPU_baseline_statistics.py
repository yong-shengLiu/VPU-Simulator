import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

# 1. 數據準備
data = {
    'Architecture': [
        'Imagine', 'VIRAM', 'SODA', 'VESPA', 'VIPERS', 'AnySP', 'VEGAS', 
        'Hwacha', 
        'Ara (Baseline)', 'Vicuna', 'Spatz', 'SiFive X280', 
        'Ascend (DaVinci)', 
        'This Work' 
    ],
    'Year': [
        2001, 2003, 2006, 2008, 2009, 2010, 2011,
        2015,
        2020, 2021, 2021, 2022,
        2021,
        2026
    ],
    'Type_Label': [
        'Type 3 (Scalar-Managed)', 'Type 3 (Scalar-Managed)', 'Type 2 (Shared Front-end)', 'Type 2 (Shared Front-end)', 'Type 2 (Shared Front-end)', 'Type 2 (Shared Front-end)', 'Type 3 (Scalar-Managed)',
        'Type 1 (Indep. Fetch)',
        'Type 3 (Scalar-Managed)', 'Type 3 (Scalar-Managed)', 'Type 3 (Scalar-Managed)', 'Type 3 (Scalar-Managed)',
        'Type 1 (Indep. Fetch)',
        'Hybrid (Macro-Op)'
    ],
    'Group': [
        'Early Exploration', 'Early Exploration', 'Early Exploration', 'Early Exploration', 'Early Exploration', 'Early Exploration', 'Early Exploration',
        'Decoupling Peak',
        'RVV Standardization (Regression)', 'RVV Standardization (Regression)', 'RVV Standardization (Regression)', 'RVV Standardization (Regression)',
        'AI/Matrix Era',
        'Target Design'
    ]
}

df = pd.DataFrame(data)

# 定義 Y 軸的整數層級 (Swimlanes)
y_map = {
    'Type 3 (Scalar-Managed)': 1,
    'Type 2 (Shared Front-end)': 2,
    'Type 1 (Indep. Fetch)': 3,
    'Hybrid (Macro-Op)': 4
}

df['Base_Y'] = df['Type_Label'].map(y_map)

# 加入手動微調 (Jitter) 以防重疊，但基於 Base_Y 進行
df['Final_Y'] = df['Base_Y'].astype(float)

offsets = {
    'Imagine': 0.1, 'VIRAM': 0.0, 'VEGAS': 0.2, 
    'SODA': 0.0, 'VESPA': 0.15, 'VIPERS': 0.30, 'AnySP': 0.45, 
    'Hwacha': 0.0, 
    'Ara (Baseline)': 0.0, 'Vicuna': -0.15, 'Spatz': -0.30, 'SiFive X280': 0.15, 
    'Ascend (DaVinci)': 0.15, 
    'This Work': 0.0 
}

for i, row in df.iterrows():
    if row['Architecture'] in offsets:
        df.at[i, 'Final_Y'] += offsets[row['Architecture']]

# 2. 設定畫布
sns.set_style("white") # 使用全白背景以便繪製色帶
fig, ax = plt.subplots(figsize=(14, 8))

# 定義顏色
colors = {
    'Early Exploration': '#95a5a6',       
    'Decoupling Peak': '#3498db',         
    'RVV Standardization (Regression)': '#e67e22', 
    'AI/Matrix Era': '#9b59b6',           
    'Target Design': '#e74c3c'            
}

# 3. 繪製背景泳道 (Swimlanes)
# 使用非常淡的顏色區分層級
band_colors = ['#f7f9f9', '#ecf0f1', '#e8f6f3', '#fcf3cf'] 
band_labels = [
    'Type 3\nScalar-Managed\n(Low Decoupling)',
    'Type 2\nShared Front-end\n(Med Decoupling)',
    'Type 1\nIndependent Fetch\n(High Decoupling)',
    'Hybrid\nMacro-Op\n(Target Design)'
]

for i, (label, y_center) in enumerate(zip(band_labels, [1, 2, 3, 4])):
    ax.axhspan(y_center - 0.45, y_center + 0.45, color=band_colors[i], alpha=0.6, zorder=0, edgecolor='none')

# 4. 繪製趨勢線 (需映射到新的 Y 軸)
trend_x = [2003, 2006, 2015, 2020, 2026]
trend_y = [1.0, 2.0, 3.0, 1.0, 4.0] # 對應到 1, 2, 3, 4 層級
ax.plot(trend_x, trend_y, color='gray', linestyle='--', alpha=0.3, linewidth=2, zorder=1)

# 5. 繪製散佈點
sns.scatterplot(
    data=df, 
    x='Year', 
    y='Final_Y', 
    hue='Group', 
    palette=colors, 
    s=500, 
    style='Group', 
    markers={'Early Exploration': 'o', 'Decoupling Peak': '^', 'RVV Standardization (Regression)': 's', 'AI/Matrix Era': 'D', 'Target Design': '*'},
    zorder=2,
    edgecolor='black',
    ax=ax,
    legend=False 
)

# 6. 加入文字標籤
for i, row in df.iterrows():
    name = row['Architecture']
    x = row['Year']
    y = row['Final_Y']
    
    y_text_offset = 0.18
    x_text_offset = 0
    
    # 針對特定點調整標籤位置以防重疊
    if name == 'Spatz': y_text_offset = -0.3
    if name == 'Vicuna': y_text_offset = -0.3; x_text_offset = 0.5
    if name == 'Ara (Baseline)': y_text_offset = -0.3
    if name == 'SiFive X280': y_text_offset = 0.18
    if name == 'Ascend (DaVinci)': y_text_offset = 0.18
    
    font_weight = 'normal'
    font_color = '#2c3e50'
    font_size = 10
    
    if name == 'This Work':
        font_weight = 'bold'
        font_color = '#c0392b'
        font_size = 13
        name = name.upper()
    
    if name == 'Hwacha':
        font_weight = 'bold'
        font_color = '#2980b9'
        
    ax.text(x + x_text_offset, y + y_text_offset, name, 
            ha='center', fontsize=font_size, color=font_color, weight=font_weight)

# 7. 加入 RVV 垂直線
ax.axvline(x=2021, color='#c0392b', linestyle='-.', linewidth=1.5, alpha=0.6, zorder=0)
ax.text(2021.2, 2.5, 'RISC-V RVV 1.0 Ratified', rotation=90, color='#c0392b', fontsize=11, fontweight='bold', va='center')

# 8. 座標軸設定
ax.set_yticks([1, 2, 3, 4])
ax.set_yticklabels(band_labels, fontsize=11, fontweight='bold') # 直接用類別名稱作為刻度
ax.set_xlabel('Year', fontsize=14, fontweight='bold')
ax.set_title('Evolution of VPU Architectures: Decoupling Capability Levels', fontsize=16, fontweight='bold', pad=20)

ax.set_xlim(1999, 2028)
ax.set_ylim(0.4, 4.6)

# 移除多餘邊框
sns.despine(left=True)
ax.yaxis.grid(False) # 關閉 Y 軸網格，因為我們有色帶了
ax.xaxis.grid(True, linestyle=':', alpha=0.5)

# 自定義圖例
from matplotlib.lines import Line2D
legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor='#95a5a6', markersize=10, label='Early Exploration'),
                   Line2D([0], [0], marker='^', color='w', markerfacecolor='#3498db', markersize=10, label='Decoupling Peak'),
                   Line2D([0], [0], marker='s', color='w', markerfacecolor='#e67e22', markersize=10, label='RVV Standardization'),
                   Line2D([0], [0], marker='D', color='w', markerfacecolor='#9b59b6', markersize=10, label='AI/Matrix Era'),
                   Line2D([0], [0], marker='*', color='w', markerfacecolor='#e74c3c', markersize=15, label='Target Design')]

ax.legend(handles=legend_elements, loc='upper left', frameon=True, fancybox=True, framealpha=0.9, fontsize=9)

plt.tight_layout()
plt.show()