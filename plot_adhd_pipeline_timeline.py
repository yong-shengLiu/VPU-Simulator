import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# 1. 引入你寫好的 VPU 模擬器與 Model Builders
from NEW_ADHD_VPU import (
    ADHD_VPU, CSRConfig, LatencySet, TensorConfig, MemoryManager,
    build_bert_base_layer, build_vit_base_layer, build_gpt2_base_layer
)

def plot_model_execution(model_name="ViT_Base", max_plot_cycles=20000):
    print(f"--- 產生 {model_name} 的硬體管線 Timeline (擷取前 {max_plot_cycles} Cycles) ---")
    
    out_dir = "timeline_results"
    os.makedirs(out_dir, exist_ok=True)

    sim = ADHD_VPU(model_name=model_name)
    csr = CSRConfig()
    latencySet = LatencySet()
    tensorHW = TensorConfig(phys_M=16, phys_N=16)
    mem_mgr = MemoryManager(base_addr=0xE000_0000)

    seq_len = 64 # 為了讓圖表更聚焦，我們用較小的 seq_len 來展示單塊 Tile 的行為
    D = 768

    # 2. 根據選擇呼叫對應的單層 Builder
    if model_name == "BERT_Base":
        build_bert_base_layer(sim, csr, tensorHW, latencySet, seq_len, 0, 0x1000, 0x2000, mem_mgr)
    elif model_name == "ViT_Base":
        build_vit_base_layer(sim, csr, tensorHW, latencySet, seq_len, 0, 0x1000, 0x2000, mem_mgr)
    elif model_name == "GPT2_Base":
        build_gpt2_base_layer(sim, csr, tensorHW, latencySet, seq_len, 0, 0x1000, 0x2000, mem_mgr)
    else:
        print("未知的模型名稱！")
        return

    # 3. 建立 Tracker 來攔截 tick() 狀態
    records = []
    active = {"LSU": None, "VALU": None, "CIM": None}
    units = {"LSU": sim.lsu_unit, "VALU": sim.valu_unit, "CIM": sim.cim_unit}

    # 4. 開始模擬，但設定觀測窗上限，避免記憶體爆炸
    while not sim.is_idle() and sim.global_cycle < max_plot_cycles:
        sim.tick()
        cycle = sim.global_cycle
        
        for unit_name, unit in units.items():
            if unit.busy:
                if active[unit_name] is None:
                    active[unit_name] = {"name": unit.current_uop.name, "start": cycle}
                elif active[unit_name]["name"] != unit.current_uop.name:
                    active[unit_name]["end"] = cycle - 1
                    records.append(dict(**active[unit_name], unit=unit_name))
                    active[unit_name] = {"name": unit.current_uop.name, "start": cycle}
            else:
                if active[unit_name] is not None:
                    active[unit_name]["end"] = cycle - 1
                    records.append(dict(**active[unit_name], unit=unit_name))
                    active[unit_name] = None

    # 收尾最後的 Active 狀態
    for unit_name in units:
        if active[unit_name] is not None:
            active[unit_name]["end"] = sim.global_cycle
            records.append(dict(**active[unit_name], unit=unit_name))

    # 5. 繪製精美的 Gantt Chart
    def get_color(name):
        name = name.upper()
        # Attention 相關
        if "LOAD_Q" in name: return "royalblue"
        if "LOAD_K" in name: return "lightseagreen"
        if "LOAD_V" in name: return "mediumturquoise"
        if "CIM_QK" in name: return "crimson"
        if "CIM_PV" in name: return "darkorange"
        if "SOFTMAX" in name: return "mediumorchid"
        # GEMM (Projection / MLP) 相關
        if "LOAD_A" in name: return "cornflowerblue"
        if "LOAD_B" in name: return "lightcoral"
        if "CIM_MAC" in name: return "firebrick"
        # 共同算子
        if "STORE" in name: return "forestgreen"
        if "GELU" in name or "LN" in name: return "purple"
        if "GLOBAL" in name or "QUANT" in name: return "plum"
        if "CLEAR" in name: return "lightgray"
        return "gray"

    unit_y = {"CIM": 3, "VALU": 2, "LSU": 1}
    fig, ax = plt.subplots(figsize=(16, 6))

    for r in records:
        duration = r["end"] - r["start"] + 1
        ax.broken_barh([(r["start"], duration)], (unit_y[r["unit"]]-0.4, 0.8), 
                       facecolors=get_color(r["name"]), edgecolor='black', linewidth=0.2)

    # 圖表美化
    ax.set_ylim(0, 4)
    ax.set_yticks([1, 2, 3])
    ax.set_yticklabels(["LSU\n(Data Mover)", "VALU\n(Non-linear)", "CIM\n(Tensor Core)"], fontsize=12, fontweight='bold')
    ax.set_xlabel("Clock Cycles", fontsize=12, fontweight='bold')
    ax.set_title(f"VPU Execution Timeline Window - {model_name} (First {max_plot_cycles} Cycles)", fontsize=16, fontweight='bold', pad=20)
    ax.grid(True, axis='x', linestyle='--', alpha=0.7)

    # 建立圖例
    legend_dict = {
        "Load Weights (LSU)": "lightcoral",
        "Load Activations (LSU)": "cornflowerblue",
        "Compute GEMM MAC (CIM)": "firebrick",
        "LayerNorm / GELU (VALU)": "purple",
        "Store Output (LSU)": "forestgreen",
        "Attn QK / PV (CIM)": "crimson",
        "Attn Softmax (VALU)": "mediumorchid"
    }
    patches = [mpatches.Patch(color=c, label=l) for l, c in legend_dict.items()]
    ax.legend(handles=patches, bbox_to_anchor=(1.01, 1), loc='upper left')

    plt.tight_layout()
    output_path = os.path.join(out_dir, f"{model_name}_execution_timeline.png")
    plt.savefig(output_path, dpi=300)
    print(f"🎉 {model_name} 的 Timeline 已經儲存至: {output_path}\n")

if __name__ == "__main__":
    # 你可以在這裡切換想要畫哪一個模型
    plot_model_execution(model_name="ViT_Base", max_plot_cycles=15000)
    plot_model_execution(model_name="BERT_Base", max_plot_cycles=15000)
    plot_model_execution(model_name="GPT2_Base", max_plot_cycles=15000)