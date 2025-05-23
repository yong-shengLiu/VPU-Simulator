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
            
            if type == 'store_a':
                print("store type not ready")
                # === load from VRF ===
                # === store to maine memory ===
            
            # === set instruction break point ===
            # if inst_number == 5: break



if __name__ == "__main__":
    
    print("=== VPU Simulator testbench ===")
    print("version: 2025.05.17")

    sim     = VPU_simulator()
    instGen = HLGenerator()


    # === Preload DRAM ===
    dir_np = 'C:/Users/david/Desktop/IwantGraduate/abstract/pattern/layer0.npy'
    sim.preload_memory(dir_np)

    # === Load Matrix Insturction ===
    loadMatricInsst, loadMatricArg = instGen.LoadMatrix(20, 5120, 160, DRAM_BASEADDR)    

    sim.run(loadMatricInsst, loadMatricArg)

    # === Print out the current VRF memory mapping ===
    with open(r"C:\Users\david\Desktop\IwantGraduate\abstract\VPU_simulator\simulation_result.txt", "w", encoding="utf-8") as f:
        with redirect_stdout(f):
            sim.vrf.dumpVRF_data()
    