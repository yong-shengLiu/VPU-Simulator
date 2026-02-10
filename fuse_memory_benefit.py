import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

# =========================================================
# 1. 硬體規格定義 (Hardware Configuration)
# =========================================================
class HardwareConfig:
    def __init__(self, vlen_bits=4096, total_vregs=32, data_width_bits=8):
        self.vlen_bits = vlen_bits
        self.vlen_bytes = vlen_bits // 8
        self.total_vregs = total_vregs
        self.data_width_bits = data_width_bits
        self.data_width_bytes = data_width_bits // 8
        self.elements_per_vlen = self.vlen_bytes // self.data_width_bytes

    def __str__(self):
        return (f"Hardware[VLEN={self.vlen_bits}b ({self.vlen_bytes}B), "
                f"Regs={self.total_vregs}, "
                f"DType=INT{self.data_width_bits}]")

# =========================================================
# 3. Part B: Fusion 流量效益分析 (四大階段)
# =========================================================
class FusionAnalyzer:
    def __init__(self, hw):
        self.hw = hw
        # BERT-Base Parameters
        self.N = 512
        self.H = 768
        self.H_ff = 3072
        self.Heads = 12
        self.D_k = 64
        self.Batch = 1

    def calculate_traffic(self, input_shapes, output_shapes, fused_shapes):
        """
        計算 DRAM 讀寫量 (MB)
        input_shapes: List of tuples (dimensions) read from memory
        output_shapes: List of tuples (dimensions) written to memory
        fused_shapes: List of tuples (dimensions) that are kept in VRF (Saved Read + Write)
        """
        def size_bytes(shape):
            return math.prod(shape) * self.hw.data_width_bytes

        read_bytes = sum(size_bytes(s) for s in input_shapes)
        write_bytes = sum(size_bytes(s) for s in output_shapes)
        # Fused tensor 省下了一次寫出 (Write) 和一次讀入 (Read)
        saved_bytes = sum(size_bytes(s) * 2 for s in fused_shapes) 
        
        return read_bytes, write_bytes, saved_bytes

    def run_analysis(self):
        print(f"\n=== Part B: Fusion Traffic Benefit Analysis (BERT-Base, Batch={self.Batch}) ===")
        results = []

        # --- Stage 1: Score + Softmax ---
        # Unfused: Read Q, K -> Write Score -> Read Score -> Softmax -> Write Prob
        # Fused:   Read Q, K -> Score(VRF) -> Softmax(VRF) -> Write Prob
        # Intermediate Fused: Score Matrix (B, Heads, N, N)
        q_shape = (self.Batch, self.Heads, self.N, self.D_k)
        k_shape = (self.Batch, self.Heads, self.N, self.D_k)
        score_shape = (self.Batch, self.Heads, self.N, self.N) # 512x512x12 -> Huge!
        prob_shape = (self.Batch, self.Heads, self.N, self.N)
        
        r1, w1, s1 = self.calculate_traffic(
            input_shapes=[q_shape, k_shape],
            output_shapes=[prob_shape],
            fused_shapes=[score_shape]
        )
        results.append(self._format_result("1. Score + Softmax", r1, w1, s1))

        # --- Stage 2: Context + Res + Norm ---
        # Unfused: Read Prob, V -> Write Context -> Read Context, ResIn -> Add -> Write -> Read -> Norm -> Write
        # Fused:   Read Prob, V -> Context(VRF) -> Read ResIn -> Add(VRF) -> Norm(VRF) -> Write NormOut
        # Intermediate Fused: Context (B, Heads, N, D_k), AddResult (B, N, H)
        v_shape = (self.Batch, self.Heads, self.N, self.D_k)
        # Context logically is (B, N, H) after concat
        context_shape = (self.Batch, self.N, self.H) 
        res_in_shape = (self.Batch, self.N, self.H)
        norm_out_shape = (self.Batch, self.N, self.H)
        
        r2, w2, s2 = self.calculate_traffic(
            input_shapes=[prob_shape, v_shape, res_in_shape],
            output_shapes=[norm_out_shape],
            fused_shapes=[context_shape, context_shape] # Approx: Context write + Add read saved
        )
        results.append(self._format_result("2. Context + Res + Norm", r2, w2, s2))

        # --- Stage 3: FFN Up + GeLU ---
        # Unfused: Read NormOut -> UpProj -> Write UpOut -> Read UpOut -> GeLU -> Write GeLUOut
        # Fused:   Read NormOut -> UpProj(VRF) -> GeLU(VRF) -> Write GeLUOut
        # Intermediate Fused: UpProject Output (B, N, 4H) -> 512x3072
        up_out_shape = (self.Batch, self.N, self.H_ff)
        gelu_out_shape = (self.Batch, self.N, self.H_ff)
        
        r3, w3, s3 = self.calculate_traffic(
            input_shapes=[norm_out_shape],
            output_shapes=[gelu_out_shape],
            fused_shapes=[up_out_shape]
        )
        results.append(self._format_result("3. FFN Up + GeLU", r3, w3, s3))

        # --- Stage 4: FFN Down + Res + Norm ---
        # Unfused: Read GeLUOut -> DownProj -> Write DownOut -> Read DownOut, ResIn2 -> Add -> ...
        # Fused:   Read GeLUOut -> DownProj(VRF) -> Read ResIn2 -> Add(VRF) -> Norm(VRF) -> Write Final
        # Intermediate Fused: DownProject Output (B, N, H), AddResult (B, N, H)
        down_out_shape = (self.Batch, self.N, self.H)
        res_in2_shape = (self.Batch, self.N, self.H)
        final_out_shape = (self.Batch, self.N, self.H)
        
        r4, w4, s4 = self.calculate_traffic(
            input_shapes=[gelu_out_shape, res_in2_shape],
            output_shapes=[final_out_shape],
            fused_shapes=[down_out_shape, down_out_shape] # Approx: Down write + Add read saved
        )
        results.append(self._format_result("4. FFN Down + Res + Norm", r4, w4, s4))
        
        df = pd.DataFrame(results)
        
        # Summary
        total_saved = df["Saved (MB)"].sum()
        total_baseline = df["Baseline Traffic (MB)"].sum()
        total_fused = df["Fused Traffic (MB)"].sum()
        
        print(df.to_string(index=False))
        print("-" * 80)
        print(f"Total Traffic Savings (1 Encoder Layer): {total_saved:.2f} MB")
        print(f"Total Traffic Reduction:                 {100 * (1 - total_fused/total_baseline):.2f}%")
        print(f"Full BERT-Base Inference Savings (12L):  {total_saved * 12:.2f} MB")
        
    def _format_result(self, name, r, w, s):
        to_mb = 1024**2
        baseline = (r + w + s) / to_mb
        fused = (r + w) / to_mb
        saved = s / to_mb
        return {
            "Stage": name,
            "Baseline Traffic (MB)": round(baseline, 2),
            "Fused Traffic (MB)": round(fused, 2),
            "Saved (MB)": round(saved, 2),
            "Reduction": f"{100 * (1 - fused/baseline):.1f}%"
        }

# =========================================================
# 4. 主程式執行
# =========================================================
if __name__ == "__main__":
    # Setup Hardware
    hw = HardwareConfig(vlen_bits=4096, total_vregs=32, data_width_bits=8)
    
    # 2. 分析 Fusion 流量效益
    fusion_analyzer = FusionAnalyzer(hw)
    fusion_analyzer.run_analysis()