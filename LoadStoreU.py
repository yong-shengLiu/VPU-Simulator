import os
import numpy as np
from main_memory import MEMORY


class LSU:
    def __init__(self, Memory, debug=False):
        self.memory = Memory # bind memory to LSU
        self.debug  = debug

        # === variable for addrgen ===
        self._vl         = 0
        self._vstart     = 0
        self._vsew       = 0
        
        self.elen        = 0
    
    def AxiAddrSet(self, vl, vstart, vsew):
        self._vl             = vl
        self._vstart         = vstart
        self._vsew           = vsew

        self.elen            = self._vl - self._vstart

    def LoadMemory(self, base_addr, stride):
        """
        this function is used to load data from Main Memory
        will return the data list with specify data size
        NOTE: "stride" is main memory byte stride defined by RVV
        """
        static_byte_addr = 0
        rtn_data         = 0
        vector_list      = []
        
        for element in range(self.elen):
            # === 64b axi addr pre-calculate ===
            if stride == 1:
                byte_addr = base_addr + ((self._vstart + element) * (self._vsew // 8))
            else:
                byte_addr = base_addr + ((self._vstart + element) * stride)
            self.debug and print(f"Fetch Addr: 0x{byte_addr:X}", end=", ")
            
            
            # === load data from Main memory per 64b===
            updata_addr = (static_byte_addr >> 3) != (byte_addr >> 3)

            if updata_addr:
                static_byte_addr = byte_addr
                rtn_data = self.memory.take64bData(static_byte_addr)  # TODO add DRAM perfomance counter here !!
                # print(f"Fetch Addr: 0x{static_byte_addr:X}")
            self.debug and print(f"data: 0x{rtn_data:X}")
            
            # === add element to the return list ===
            element_value = (rtn_data >> ((byte_addr & 0b111) * 8)) & ((1 << self._vsew) - 1)
            vector_list.append(element_value)
        

        return vector_list
    
    def StoreMemory(self, base_addr, stride, data_list):
        """
        this function is used to store data to Main Mempry
        NOTE: "stride" is main memory byte stride defined by RVV
        TODO: 
        (1) this function is not same as Ara, not only 64b store when vstart is not 0
        """
        if stride == 1:
            static_byte_addr = base_addr + (self._vstart * (self._vsew // 8))
        else:
            static_byte_addr = base_addr + (self._vstart * stride)
        byte_strb        = 0
        store_data       = 0

        for element in range(self.elen):
            # === 64b axi addr pre-calculate ===
            if stride == 1:
                byte_addr = base_addr + ((self._vstart + element) * (self._vsew // 8))
            else:
                byte_addr = base_addr + ((self._vstart + element) * stride)
            print(f"Store Addr: 0x{byte_addr:X}")

            
            # === Store data to Main memory per 64b ===
            updata_addr = ( (static_byte_addr >> 3) != (byte_addr >> 3) )

            if updata_addr:
                print(f"Store Addr: 0x{static_byte_addr:X}, strb: {byte_strb:X}, data: 0x{store_data:X}")
                self.memory.store64bData(static_byte_addr, byte_strb, store_data)  # TODO add DRAM perfomance counter here !!
                static_byte_addr = byte_addr
                byte_strb  = 0
                store_data = 0
            else:
                byte_strb = byte_strb   | (1 << (byte_addr & 0b111))  # byte strb
                store_data = store_data | (data_list[element] << ((byte_addr & 0b111) * 8))
                

if __name__ == "__main__":
    print("===== LoadStoreUnit testbench =====")
    print("version: 2025.05.23")

    current_dir = os.path.dirname(os.path.abspath(__file__))

    dram = MEMORY(BASEADDR=0xE0000000, DataWidth=64, Depth=409600, debug=False)
    lsu = LSU(Memory=dram)

    # === Preload DRAM ===
    dir_np = os.path.join(current_dir, "pattern", "layer0.npy")
    raw_pattern = np.load(dir_np)                        # load file
    byte_pattern = raw_pattern.flatten().astype(np.uint8)  # fp32 to uint8
    dram.init_byte_to_mem(byte_pattern)               # the preload pattern which is represent in byte

    # === Load test ===
    lsu.AxiAddrSet(20, 5, 16) #(vl, vstart, vsew)
    print(f"sew: {lsu._vsew}, elen: {lsu.elen}")
    vector = lsu.LoadMemory(0xE0000000, 1)  # stride is byte stride
    print([f"0x{val:X}"  for val in vector])

    # === Store test ===
    lsu.AxiAddrSet(6, 0, 16) #(vl, vstart, vsew)
    print(f"sew: {lsu._vsew}, elen: {lsu.elen}")
    lsu.StoreMemory(0xE0000000, 1, [0x1111, 0x2222, 0x3333, 0x4444, 0x5555, 0x6666])  # (self, base_addr, stride, data_list)
    

    