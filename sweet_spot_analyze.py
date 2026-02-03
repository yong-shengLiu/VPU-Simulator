import os
import numpy as np
import matplotlib.pyplot as plt

def find_sweet_point(load_latency_cycles=20, mac_throughput_cycles=1):
    """
    模擬不同 M_tile 下的 Baseline 與 Decoupled 效能
    load_latency_cycles: 載入一個 Vector 需要的等效 Cycle 數 (含頻寬限制)
    mac_throughput_cycles: 發射一個 MAC 指令需要的 Cycle 數
    """
    
    # 測試的 Tile 大小 (受限於 32 個暫存器，最大測到 24)
    m_tiles = np.array([1, 2, 4, 8, 12, 16, 20, 24])
    
    baseline_cycles = []
    decoupled_cycles = []
    hiding_efficiency = []
    
    print(f"{'M_tile':<8} | {'Baseline':<10} | {'Decoupled':<10} | {'Speedup':<8} | {'Hiding Eff':<10}")
    print("-" * 65)

    for m in m_tiles:
        # 單次 Inner Loop 的 Cycle 估算
        t_load = load_latency_cycles
        t_compute = m * mac_throughput_cycles
        
        # Baseline: 序列執行 (Load + Compute) + Overhead (假設 10%)
        t_base = (t_load + t_compute) * 1.1
        
        # Decoupled: 平行執行 (Max(Load, Compute)) + Overhead (假設 5%)
        t_decouple = max(t_load, t_compute) * 1.05
        
        # 計算 Hiding Efficiency (Baseline 到底浪費了多少)
        # 理想時間是 t_compute (假設 Compute Bound) 或 t_load (Memory Bound)
        ideal = max(t_load, t_compute)
        overlap = t_base - ideal # 這裡簡化定義：Baseline 多出來的時間就是沒 Hide 到的
        # 重新定義你的 Hiding Efficiency: Overlap / Compute
        # 在 Baseline 中，真正的 Overlap 幾乎是 0 (Single Issue)，所以我們看 Speedup 比較直觀
        
        speedup = t_base / t_decouple
        
        baseline_cycles.append(t_base)
        decoupled_cycles.append(t_decouple)
        
        print(f"{m:<8} | {t_base:<10.1f} | {t_decouple:<10.1f} | {speedup:<8.2f}x | N/A")

    # --- 繪圖分析 ---
    plt.figure(figsize=(10, 6))
    plt.plot(m_tiles, baseline_cycles, 'o--', label='Baseline (Single Issue)', color='red')
    plt.plot(m_tiles, decoupled_cycles, 's-', label='Decoupled (Your Design)', color='green')
    
    # 標示出 "Cross Over" 點 (Load == Compute)
    # 當 M * mac == load => M = load / mac
    balance_point = load_latency_cycles / mac_throughput_cycles
    plt.axvline(x=balance_point, color='blue', linestyle=':', label=f'Balance Point (M={balance_point})')
    
    plt.xlabel('M_tile Factor (Unrolling)')
    plt.ylabel('Estimated Cycles per Loop Iteration')
    plt.title('Finding the Sweet Point: Baseline vs Decoupled')
    plt.grid(True)
    plt.legend()

    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(current_dir, 'sweet_point_analysis.png')
    plt.savefig(output_path)
    print("\n[Info] 圖表已存為 sweet_point_analysis.png")

# 假設 Ara 的 VLEN=512, DRAM Bandwidth 限制下，搬一個 Vector 可能要 16~20 cycles
find_sweet_point(load_latency_cycles=16, mac_throughput_cycles=1)