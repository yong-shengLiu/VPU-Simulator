import sys
import os
import time

# 嘗試載入畫圖套件
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("[提示] 系統未安裝 matplotlib，將只輸出文字表格。如需自動畫圖請執行: pip install matplotlib")

# 匯入你的 VPU Golden Model
try:
    import ADHD_VPU as vpu
except ImportError:
    print("錯誤: 找不到 ADHD_VPU.py，請確保它在同一個資料夾下！")
    sys.exit(1)

# ==============================================================================
# 實驗環境設定
# ==============================================================================
OUTPUT_DIR = "experiment_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)
REPORT_FILE = os.path.join(OUTPUT_DIR, "sweep_report.txt")

def log_and_print(text):
    """ 同時印出在終端機並寫入報告檔 """
    print(text)
    with open(REPORT_FILE, "a", encoding="utf-8") as f:
        f.write(text + "\n")

# 初始化報告檔
with open(REPORT_FILE, "w", encoding="utf-8") as f:
    f.write("=========================================================\n")
    f.write(" ADHD VPU 複合微架構掃描報告 (Composite Arch Sweep)\n")
    f.write("=========================================================\n\n")

def get_updated_latency_set(axi_width):
    """ 根據 AXI 頻寬動態更新 Latency 參數 """
    lat = vpu.LatencySet()
    lat.Load_One_Vector = vpu.VLEN // axi_width + 1
    lat.Store_One_Vector = vpu.VLEN // axi_width + 1
    lat.VALU_VMV = vpu.VLEN // vpu.LANE // axi_width
    lat.VALU_VADD = vpu.VLEN // vpu.LANE // axi_width
    lat.VALU_VEXP = vpu.VLEN // vpu.LANE // axi_width
    lat.VALU_VGELU = vpu.VLEN // vpu.LANE // axi_width
    return lat

def run_custom_sim(axi_width, q_depth, setup_csr_func, seq_len=256):
    """ 核心模擬執行器 (允許外部注入自訂的 CSR 設定) """
    # 更新硬體全域變數
    vpu.AXI_WIDTH = axi_width
    vpu.LSU_QUEUE_DEPTH = q_depth
    vpu.VALU_QUEUE_DEPTH = q_depth
    vpu.CIM_QUEUE_DEPTH = q_depth
    
    latencySet = get_updated_latency_set(axi_width)
    tensorHW = vpu.TensorConfig(phys_M=16, phys_N=16)
    mem_mgr = vpu.MemoryManager(base_addr=0xE000_0000)
    
    # 關閉 trace 以加速 Sweep
    sim = vpu.ADHD_VPU(model_name="Custom", trace_filename=os.devnull, c_macro_header=os.devnull)
    
    # 初始化並透過外部函數客製化 CSR
    csr = vpu.CSRConfig()
    setup_csr_func(csr) 
    
    # 執行輕量化的 BERT 測試 (為了加速 Sweep，我們縮短 Seq_Len)
    vpu.build_bert_base_layer(sim, csr, tensorHW, latencySet, seq_len=seq_len, mem_mgr=mem_mgr)
    
    while not sim.is_idle():
        sim.tick()
        
    return sim

# ==============================================================================
# [實驗一] 算術強度 vs. 頻寬資源 (Arithmetic Intensity vs. Memory Wall)
# ==============================================================================
def exp1_arithmetic_intensity_tradeoff():
    log_and_print("\n" + "="*60)
    log_and_print("🚀 [實驗一] 算術強度 vs. 頻寬資源 (Tiling Size vs AXI Width)")
    log_and_print("="*60)
    
    axi_widths = [32, 64, 128, 256]
    tile_sizes = [16, 32, 64]
    
    results_utilization = {t: [] for t in tile_sizes}
    
    for t_size in tile_sizes:
        log_and_print(f"\n--- 測試 Tile Size: {t_size}x{t_size} ---")
        def setup_csr(csr):
            csr.M_tile = t_size
            csr.N_tile = t_size
            csr.K_tile = 32
        
        for axi in axi_widths:
            sim = run_custom_sim(axi_width=axi, q_depth=16, setup_csr_func=setup_csr)
            util = sim.cim_unit.total_active_cycles / sim.global_cycle
            results_utilization[t_size].append(util)
            log_and_print(f"  AXI: {axi:3d}-bit | Total Cycles: {sim.global_cycle:9d} | CIM Utilization: {util:.1%}")

    if HAS_MATPLOTLIB:
        plt.figure(figsize=(8, 5))
        colors = ['red', 'green', 'blue']
        for idx, t_size in enumerate(tile_sizes):
            plt.plot(axi_widths, results_utilization[t_size], marker='o', color=colors[idx], label=f'Tile {t_size}x{t_size}')
            
        plt.title('CIM Utilization vs. AXI Width across different Tiling Sizes')
        plt.xlabel('SRAM AXI Interface Width (bits)')
        plt.ylabel('Compute-In-Memory (CIM) Utilization')
        plt.xscale('log', base=2)
        plt.xticks(axi_widths, labels=[str(w) for w in axi_widths])
        plt.ylim(0, 1.0)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        out_path = os.path.join(OUTPUT_DIR, 'exp1_arithmetic_intensity.png')
        plt.savefig(out_path)
        log_and_print(f"\n📊 圖表已儲存為: {out_path}")

# ==============================================================================
# [實驗二] 不規則存取 vs. 解耦佇列抗震力 (Scatter/Gather Jitter Resilience)
# ==============================================================================
def exp2_latency_jitter_resilience():
    log_and_print("\n" + "="*60)
    log_and_print("🚀 [實驗二] 記憶體破碎度 vs. 解耦佇列抗震力 (Scatter/Gather Resilience)")
    log_and_print("="*60)
    
    queue_depths = [2, 4, 8, 16, 32]
    
    modes = {
        "Dense (Continuous)": {"gather": False, "block_len": 64},
        "Mild Gather (Block=8)": {"gather": True, "block_len": 8},
        "Extreme Gather (Block=1)": {"gather": True, "block_len": 1}
    }
    
    results_cycles = {m: [] for m in modes}
    
    for mode_name, cfg in modes.items():
        log_and_print(f"\n--- 測試記憶體存取模式: {mode_name} ---")
        def setup_csr(csr):
            csr.Is_Gather_A = cfg["gather"]; csr.BLOCK_LEN_A = cfg["block_len"]
            csr.Is_Gather_B = cfg["gather"]; csr.BLOCK_LEN_B = cfg["block_len"]
            csr.Is_Scatter_C = cfg["gather"]; csr.BLOCK_LEN_C = cfg["block_len"]
            csr.Is_Gather_D = cfg["gather"]; csr.BLOCK_LEN_D = cfg["block_len"]
            
        for q_depth in queue_depths:
            sim = run_custom_sim(axi_width=64, q_depth=q_depth, setup_csr_func=setup_csr)
            results_cycles[mode_name].append(sim.global_cycle)
            log_and_print(f"  Queue Depth: {q_depth:2d} | Total Cycles: {sim.global_cycle:9d}")

    if HAS_MATPLOTLIB:
        plt.figure(figsize=(8, 5))
        markers = ['o', 's', '^']
        for idx, mode_name in enumerate(modes):
            # Normalize against the Queue=2 dense baseline just for visual clarity or plot raw
            plt.plot(queue_depths, results_cycles[mode_name], marker=markers[idx], label=mode_name)
            
        plt.title('Performance Impact of Irregular Memory Accesses (Scatter/Gather)')
        plt.xlabel('Decoupled Queue Depth (Entries)')
        plt.ylabel('Total Execution Cycles (Lower is better)')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        out_path = os.path.join(OUTPUT_DIR, 'exp2_scatter_gather_resilience.png')
        plt.savefig(out_path)
        log_and_print(f"\n📊 圖表已儲存為: {out_path}")

# ==============================================================================
# [實驗三] 雙緩衝 Ping-Pong 的空間與時間悖論 (Double Buffer Trade-off)
# ==============================================================================
def exp3_ping_pong_paradox():
    log_and_print("\n" + "="*60)
    log_and_print("🚀 [實驗三] 雙緩衝空間時間悖論 (Ping-Pong vs. Max Tile Size)")
    log_and_print("="*60)
    
    axi_widths = [16, 32, 64, 128]
    
    configs = {
        "Config A: Double Buffer ON (Tile 32x32)": {"db": True, "tile": 32},
        "Config B: Double Buffer OFF (Tile 64x64)": {"db": False, "tile": 64}
    }
    
    results_cycles = {c: [] for c in configs}
    
    for cfg_name, cfg in configs.items():
        log_and_print(f"\n--- 測試架構組態: {cfg_name} ---")
        def setup_csr(csr):
            csr.Enable_Double_Buffer = cfg["db"]
            csr.M_tile = cfg["tile"]
            csr.N_tile = cfg["tile"]
        
        for axi in axi_widths:
            sim = run_custom_sim(axi_width=axi, q_depth=16, setup_csr_func=setup_csr)
            results_cycles[cfg_name].append(sim.global_cycle)
            log_and_print(f"  AXI: {axi:3d}-bit | Total Cycles: {sim.global_cycle:9d} | LSU Active: {sim.lsu_unit.total_active_cycles/sim.global_cycle:.1%}")

    if HAS_MATPLOTLIB:
        plt.figure(figsize=(8, 5))
        for cfg_name in configs:
            plt.plot(axi_widths, results_cycles[cfg_name], marker='o', label=cfg_name)
            
        plt.title('Double Buffering vs. Larger Tiling Size under varying Bandwidths')
        plt.xlabel('SRAM AXI Interface Width (bits)')
        plt.ylabel('Total Execution Cycles (Lower is better)')
        plt.xscale('log', base=2)
        plt.xticks(axi_widths, labels=[str(w) for w in axi_widths])
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        out_path = os.path.join(OUTPUT_DIR, 'exp3_ping_pong_paradox.png')
        plt.savefig(out_path)
        log_and_print(f"\n📊 圖表已儲存為: {out_path}")

if __name__ == "__main__":
    start_time = time.time()
    log_and_print("開始執行 ADHD_VPU 複合變數敏感度分析...")
    
    exp1_arithmetic_intensity_tradeoff()
    exp2_latency_jitter_resilience()
    exp3_ping_pong_paradox()
    
    end_time = time.time()
    log_and_print("\n" + "="*60)
    log_and_print(f"✅ 所有實驗分析完成！總耗時: {end_time - start_time:.2f} 秒")
    log_and_print(f"📂 實驗數據與圖表已儲存至: ./{OUTPUT_DIR}/ 目錄下")