import numpy as np
import os
from contextlib import redirect_stdout

from HLGenerator import HLGenerator
from main_memory import MEMORY
from dispatcher import DISPATCHER
from vrf import VRF
from LoadStoreU import LSU

DRAM_BASEADDR = 0xE0000000

class VPU_simulator:
    def __init__(self, debug=False):
        # Initialize all sub-modules
        self.dram       = MEMORY(BASEADDR=DRAM_BASEADDR, DataWidth=64, Depth=409600, debug=False)
        self.vrf        = VRF()
        self.dispatcher = DISPATCHER()
        self.lsu        = LSU(Memory=self.dram)  # bind main memory to lsu
        self.debug      = debug
    

    def preload_memory(self, dir_path):
        raw_pattern = np.load(dir_path)                        # load file
        byte_pattern = raw_pattern.flatten().astype(np.uint8)  # fp32 to uint8
        self.dram.init_byte_to_mem(byte_pattern)               # the preload pattern which is represent in byte


    def run(self, inst_list, arg_list):
        for inst_number, (inst, arg) in enumerate( zip(inst_list, arg_list)):

            # === Decoder instruction ===
            type = self.dispatcher.decodeCAPI(inst, arg)

            # === Dataflow for different type of instruction ===
            print(f"Inst number[{inst_number}] -> type: {type}", arg)
            
            # if type == 'vset':  sew, lmul is stored in dispatcher
            # if type == 'vstart': csr is stored in dispatcher
            if type == 'vload_a':  # [sew, vd, base_addr]
                # === load from main memory ===
                self.lsu.AxiAddrSet(self.dispatcher.vl, self.dispatcher.vstart, self.dispatcher.SEW)
                temp_vector = self.lsu.LoadMemory(arg[2], 1) # set unit stride
                self.debug and print( [f"0x{val:X}"  for val in temp_vector] )
                
                # TODO VRF element length need be calculate in lane
                # === store to VRF ===
                element_length = self.dispatcher.vl - self.dispatcher.vstart
                self.vrf.vset(self.dispatcher.SEW, self.dispatcher.LMUL)
                self.vrf.load(arg[1], self.dispatcher.vstart, element_length, temp_vector)
            
            if type == 'store_a':  # [sew, vs, base_addr]
                # TODO VRF element length need be calculate in lane
                # === load from VRF ===
                element_length = self.dispatcher.vl - self.dispatcher.vstart
                self.vrf.vset(self.dispatcher.SEW, self.dispatcher.LMUL)
                temp_vector = self.vrf.take(arg[1], self.dispatcher.vstart, element_length)

                # === store to maine memory ===
                self.lsu.AxiAddrSet(self.dispatcher.vl, self.dispatcher.vstart, self.dispatcher.SEW)
                self.lsu.StoreMemory(arg[2], 1, temp_vector) # set unit stride
            
            # === set instruction break point ===
            # if inst_number == 5: break



if __name__ == "__main__":
    
    print("=== VPU Simulator testbench ===")
    print("version: 2025.05.23")

    current_dir = os.path.dirname(os.path.abspath(__file__))
    terminal_output_path = os.path.join(current_dir, "log", "terminal.txt")
    vrf_output_path = os.path.join(current_dir, "log", "vrf_result.txt")
    dram_output_path = os.path.join(current_dir, "log", "dram_result.txt")
    os.makedirs(os.path.dirname(vrf_output_path), exist_ok=True)   # create the output path

    sim     = VPU_simulator()
    instGen = HLGenerator()


    # === Preload DRAM ===
    dir_np = os.path.join(current_dir, "pattern", "layer0.npy")
    sim.preload_memory(dir_np)

    with open(terminal_output_path, "w", encoding="utf-8") as f:
        with redirect_stdout(f):
            # === Load Matrix Insturction ===
            loadMatricInsst, loadMatricArg = instGen.CIM_Scatter_LS('load', 20, 5120, 160, DRAM_BASEADDR, 0)
            sim.run(loadMatricInsst, loadMatricArg)

            # === Store Matrix Insturction ===
            storeMatricInsst, storeMatricArg = instGen.CIM_Scatter_LS('store', 20, 160, 160, DRAM_BASEADDR, 0)
            sim.run(storeMatricInsst, storeMatricArg)


    # === Print out the current VRF memory mapping ===
    with open(vrf_output_path, "w", encoding="utf-8") as f:
        with redirect_stdout(f):
            sim.vrf.dumpVRF_data()
    print("VRF dump success")

    # === Print out the current VRF memory mapping ===
    with open(dram_output_path, "w", encoding="utf-8") as f:
        with redirect_stdout(f):
            sim.dram.dumpMem_data('debug')
    print("DRAM dump success")
    