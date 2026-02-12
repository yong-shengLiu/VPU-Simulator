import enum
import math
import os
import sys

# =========================================================
# 1. 定義 Macro Op 結構
# =========================================================

class MacroOpType(enum.Enum):
    CONV   = 0
    LOAD   = 1
    STORE  = 2
    VECTOR = 3

class MemType(enum.Enum):
    DRAM     = 0
    UNI_SRAM = 2
    VRF      = 3

class DependencyTag:
    def __init__(self, ld=-1, tu=-1, st=-1):
        self.ld_tag = ld
        self.tu_tag = tu
        self.st_tag = st
    
    def __repr__(self):
        return f"DependencyTag({{ld_tag: {self.ld_tag}, tu_tag: {self.tu_tag}, st_tag: {self.st_tag}}})"

class MacroOp:
    def __init__(self, seq, op_type, dep_tag, info):
        self.seq = seq
        self.op_type = op_type
        self.dep_tag = dep_tag
        self.info = info

    def __repr__(self):
        # 格式化輸出
        base_str = f"macro_{self.seq}: MacroOp({{op_type: {self.op_type}, seq: {self.seq}, dep_tag: {self.dep_tag}"
        for k, v in self.info.items():
            base_str += f", {k}: {v}"
        base_str += "})"
        return base_str

# =========================================================
# 2. Generator 核心邏輯
# =========================================================

class BertMacroGenerator:
    def __init__(self):
        self.seq_counter = 0
        self.last_load_id = -1
        self.last_compute_id = -1
        self.last_store_id = -1
        self.VLEN_BYTES = 512
        self.M_TILE = 16
        
    def _get_next_seq(self):
        s = self.seq_counter
        self.seq_counter += 1
        return s

    def emit_load(self, src_type, dst_type, dims, name="Data"):
        seq = self._get_next_seq()
        dep = DependencyTag(ld=self.last_load_id, tu=self.last_compute_id) 
        info = {'src_type': src_type, 'dst_type': dst_type, 'dims': dims, 'comment': f"Load {name}"}
        op = MacroOp(seq, MacroOpType.LOAD, dep, info)
        print(op) # 這會被導向到檔案
        self.last_load_id = seq
        return seq

    def emit_compute(self, dims, op_name="GEMM"):
        seq = self._get_next_seq()
        dep = DependencyTag(ld=self.last_load_id, tu=self.last_compute_id)
        info = {'op_name': op_name, 'dims': dims, 'tile_m': self.M_TILE, 'strategy': 'Output Stationary'}
        op = MacroOp(seq, MacroOpType.CONV, dep, info)
        print(op)
        self.last_compute_id = seq
        return seq

    def emit_vector_op(self, op_name, dims):
        seq = self._get_next_seq()
        dep = DependencyTag(ld=-1, tu=self.last_compute_id)
        info = {'op_name': op_name, 'dims': dims, 'src_type': MemType.VRF, 'dst_type': MemType.VRF}
        op = MacroOp(seq, MacroOpType.VECTOR, dep, info)
        print(op)
        self.last_compute_id = seq
        return seq

    def emit_store(self, src_type, dst_type, dims, name="Result"):
        seq = self._get_next_seq()
        dep = DependencyTag(ld=-1, tu=self.last_compute_id, st=self.last_store_id)
        info = {'src_type': src_type, 'dst_type': dst_type, 'dims': dims, 'comment': f"Store {name}"}
        op = MacroOp(seq, MacroOpType.STORE, dep, info)
        print(op)
        self.last_store_id = seq
        return seq

    def gen_tiled_gemm(self, layer_name, M, N, K, fused_load_a=False, fused_store=False):
        print(f"\n[Layer Start] {layer_name} ({M}x{N}x{K})")
        num_n_tiles = math.ceil(N / self.VLEN_BYTES)
        num_m_blocks = math.ceil(M / self.M_TILE)
        
        for n_idx in range(num_n_tiles):
            current_n = min(self.VLEN_BYTES, N - n_idx * self.VLEN_BYTES)
            for m_idx in range(num_m_blocks):
                current_m = min(self.M_TILE, M - m_idx * self.M_TILE)
                tile_label = f"Tile[n={n_idx},m={m_idx}]"

                self.emit_load(MemType.DRAM, MemType.UNI_SRAM, f"{K}x{current_n}", f"Weight_B_{tile_label}")
                
                if not fused_load_a:
                    self.emit_load(MemType.DRAM, MemType.UNI_SRAM, f"{current_m}x{K}", f"Input_A_{tile_label}")
                
                self.emit_compute(f"{current_m}x{current_n}x{K}", f"GEMM_{tile_label}")

                if not fused_store:
                    self.emit_store(MemType.UNI_SRAM, MemType.DRAM, f"{current_m}x{current_n}", f"Output_C_{tile_label}")

# =========================================================
# 3. 執行與檔案輸出
# =========================================================
def run_generator():
    # 1. 設定輸出路徑
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(current_dir, 'bert_macro_ops.log')
    
    print(f"Generating log file to: {output_path}")

    # 2. 將 stdout 重定向到檔案
    original_stdout = sys.stdout # 保存原本的 stdout (終端機)
    
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            sys.stdout = f # 開始重定向
            
            gen = BertMacroGenerator()
            
            # BERT Parameters
            Seq, Hidden, Heads, HeadDim, FFN_Hidden = 512, 768, 12, 64, 3072

            # --- Layer 1: QKV Proj ---
            gen.gen_tiled_gemm("Q_Proj", Seq, Hidden, Hidden)
            
            # --- Layer 2: Attention Fusion ---
            gen.gen_tiled_gemm("Attn_Score_Head0", Seq, Seq, HeadDim, fused_store=True)
            gen.emit_vector_op("Vector_Softmax", f"{Seq}x{Seq}")
            gen.gen_tiled_gemm("Attn_Context_Head0", Seq, HeadDim, Seq, fused_load_a=True)

            # --- Layer 3: Residual ---
            gen.emit_load(MemType.DRAM, MemType.UNI_SRAM, f"{Seq}x{Hidden}", "Residual_Input")
            gen.emit_vector_op("Add_Norm", f"{Seq}x{Hidden}")
            gen.emit_store(MemType.VRF, MemType.DRAM, f"{Seq}x{Hidden}", "Norm_Output")

            # --- Layer 4: FFN Fusion ---
            gen.gen_tiled_gemm("FFN_Up_Proj", Seq, FFN_Hidden, Hidden, fused_store=True)
            gen.emit_vector_op("Vector_GeLU", f"{Seq}x{FFN_Hidden}")
            gen.gen_tiled_gemm("FFN_Down_Proj", Seq, Hidden, FFN_Hidden, fused_load_a=True)
            
            print("\n[End of Generation]")
            
    finally:
        sys.stdout = original_stdout # 恢復 stdout，讓這行字印在螢幕上
    
    print("Done! Check the log file.")

if __name__ == "__main__":
    run_generator()