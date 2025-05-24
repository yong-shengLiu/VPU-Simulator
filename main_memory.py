import os
import numpy as np
from contextlib import redirect_stdout

class MEMORY:
    def __init__(self, BASEADDR=0, DataWidth=64, Depth=409600, debug=False):
        
        # === parameters ===
        self.BASEADDR  = BASEADDR
        self.DataWidth = DataWidth
        self.Depth     = Depth
        self.debug     = debug

        # === Dynamic parameters ===

        # === memory ===
        self.memory = np.zeros((self.Depth), dtype=np.uint64)
    
    
    # the function to initial bulk data to memory in byte
    def init_byte_to_mem(self, pattern):
        # === Reshape to (N, 8) ===
        dram_reshape = pattern.reshape(-1, 8)
        self.debug and print("reshape: ", dram_reshape.shape)
        

        # === load into memory ===s
        for idx, chunk in enumerate(dram_reshape):
            self.debug and print(chunk)
            
            chunk = chunk[::-1]  # reverse each chunk (little-endian)

            Dec64b = 0
            for byte in chunk:
                self.debug and print(hex(byte))
                Dec64b = (Dec64b << 8) | int(byte)
            
            self.debug and print(f"0x{Dec64b:016X}")
            self.memory[idx] = Dec64b

    def dumpMem_data(self, mode):
        """
        The function to dump all of memory
        Support mode:
        1. debug: used to dump python level memory in txt with 64b per data
        2. rtl: used to generate the hex file for rtl "readmemh" with 8b per data
        """

        if mode == 'debug':
            print("----- Memory data -----")
            print(f"Size: {self.memory.shape}, DataWidth: {self.DataWidth}")
            for idx, value in enumerate(self.memory):
                print(f"[{idx:6}] ", end="")
                print(f"0x{value:016X}", end="")
                print(" -> ", value)
        elif mode == 'rtl':
            for idx, value in enumerate(self.memory):
                for byte in range(8):
                    byte_mask  = 0b11111111 << (byte * 8)
                    byte_value = (value & byte_mask) >> (byte * 8)
                    print(f"{byte_value:02X}")

    def take64bData(self, addr):
        """
        the function to take data out of memory (addr is byte addr)
        NOTE this function can only take one 64b address support with alignment  
        """

        relative_start_addr = addr - self.BASEADDR
        align64_addr = (relative_start_addr >> 3)
        
        return self.memory[align64_addr]

    def store64bData(self, addr, byte_strb, data):
        """
        the function to store data into memory (addr is byte addr)
        NOTE this function need master to provide the byte_strb(8bit), high->store, low->don't store
        """

        relative_start_addr = addr - self.BASEADDR
        align64_addr = (relative_start_addr >> 3)

        # === store data ===
        bit_mask = 0
        for byte in range(8):
            if byte_strb & (1 << byte):
                bit_mask = bit_mask | (0b11111111 << (byte * 8))
        inv_mask = ~bit_mask & 0xFFFFFFFFFFFFFFFF  # assume 64-bit memory entry
        
        self.memory[align64_addr] = (self.memory[align64_addr] & inv_mask) | (data & bit_mask) 
        

    def take_data(self, start_addr, size, length):
        """
        the function to take data out of memory (start_addr is byte addr)
        NOTE this function is "too powerful", 
             the normal main memory will not return the data list     
        """
        
        relative_start_addr = start_addr - self.BASEADDR

        # === initial the return vector & align the start address ===
        if   size == 8:
            temp_vector = np.zeros((length), dtype=np.uint8)
            align_start_addr = relative_start_addr
        elif size == 16:
            temp_vector = np.zeros((length), dtype=np.uint16)
            align_start_addr = (relative_start_addr >> 1) << 1
        elif size == 32:
            temp_vector = np.zeros((length), dtype=np.uint32)
            align_start_addr = (relative_start_addr >> 2) << 2
        elif size == 64:
            temp_vector = np.zeros((length), dtype=np.uint64)
            align_start_addr = (relative_start_addr >> 3) << 3
        else:
            raise ValueError("take_data: Unsupported data size")

        # === take data out from memory ===
        for idx in range(length):

            # === convert byte address to mem address ===
            mem_addr = (align_start_addr + (idx * size // 8)) // 8

            # === convert byte address to bit offset ===
            offset   = (align_start_addr + (idx * size // 8)) % 8 * 8
            
            self.debug and print("mem_addr: ", mem_addr, "offset: ", offset)

            # === shift data to target byte ===
            if   size == 8:   temp_data = (self.memory[mem_addr] >> offset) & 0Xff
            elif size == 16:  temp_data = (self.memory[mem_addr] >> offset) & 0Xffff
            elif size == 32:  temp_data = (self.memory[mem_addr] >> offset) & 0Xffffffff
            elif size == 64:  temp_data = (self.memory[mem_addr] >> offset)

            temp_vector[idx] = temp_data

        return temp_vector

    def store_data(self, start_addr, size, vector):
        """
        the function to store data into memory (start_addr is byte addr)
        NOTE this function is "too powerful", 
             the normal main memory will not take the data list as input    
        """

        relative_start_addr = start_addr - self.BASEADDR

        length = len(vector)
        
        # === initial the return vector & align the start address ===
        if   size == 8:
            align_start_addr = relative_start_addr
        elif size == 16:
            align_start_addr = (relative_start_addr >> 1) << 1
        elif size == 32:
            align_start_addr = (relative_start_addr >> 2) << 2
        elif size == 64:
            align_start_addr = (relative_start_addr >> 3) << 3
        else:
            raise ValueError("store_data: Unsupported data size")

        # === take data out from memory ===
        for idx in range(length):

            # === convert byte address to mem address===
            mem_addr = (align_start_addr + (idx * size // 8)) // 8

            # === convert byte address to bit offset ===
            offset   = (align_start_addr + (idx * size // 8)) % 8 * 8
            
            self.debug and print("mem_addr: ", mem_addr, "offset: ", offset)

            # === Clear the bits at the target location & refresh the new data ====
            mask = ((1 << size) - 1) << offset
            inv_mask = ~mask & 0xFFFFFFFFFFFFFFFF  # assume 64-bit memory entry
            old_value = self.memory[mem_addr]
            new_value = (old_value & inv_mask) | ((int(vector[idx]) << offset) & mask)
            self.memory[mem_addr] = new_value


if __name__ == "__main__":
    print("===== main memory testbench =====")
    print("version: 2025.05.24")

    dram = MEMORY(DataWidth=64, Depth=409600, debug=False)

    # load the DRAM pattern (float 32b)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dir_np = os.path.join(current_dir, "pattern", "layer0.npy")
    dram_pattern = np.load(dir_np)

    # 32b to 8b
    dram_pattern = dram_pattern.flatten().astype(np.uint8)

    # the preload pattern is represent in byte
    dram.init_byte_to_mem(dram_pattern)

    # 8 bit testbench
    # print("8bit : ", [f"0x{val:02X}"  for val in dram.take_data(0, 8,  15).astype(np.uint8 )]) # align & start from 0
    # print("8bit : ", [f"0x{val:02X}"  for val in dram.take_data(2, 8,  15).astype(np.uint8 )]) # align 2Byte
    # print("8bit : ", [f"0x{val:02X}"  for val in dram.take_data(4, 8,  15).astype(np.uint8 )]) # align 4Byte
    # print("8bit : ", [f"0x{val:02X}"  for val in dram.take_data(8, 8,  15).astype(np.uint8 )]) # align 8Byte
    # print("8bit : ", [f"0x{val:02X}"  for val in dram.take_data(3, 8,  15).astype(np.uint8 )]) # non-align

    # 16 bit testbench
    # print("16bit: ", [f"0x{val:04X}"  for val in dram.take_data(0, 16, 10).astype(np.uint16)]) # align & start from 0
    # print("16bit: ", [f"0x{val:04X}"  for val in dram.take_data(2, 16, 10).astype(np.uint16)]) # align 2Byte
    # print("16bit: ", [f"0x{val:04X}"  for val in dram.take_data(4, 16, 10).astype(np.uint16)]) # align 4Byte
    # print("16bit: ", [f"0x{val:04X}"  for val in dram.take_data(8, 16, 10).astype(np.uint16)]) # align 8Byte
    # print("16bit: ", [f"0x{val:04X}"  for val in dram.take_data(3, 16, 10).astype(np.uint16)]) # non-align

    # 32 bit testbench
    # print("32bit: ", [f"0x{val:08X}"  for val in dram.take_data(0, 32, 5).astype(np.uint32)]) # align & start from 0
    # print("32bit: ", [f"0x{val:08X}"  for val in dram.take_data(2, 32, 5).astype(np.uint32)]) # align 2Byte
    # print("32bit: ", [f"0x{val:08X}"  for val in dram.take_data(4, 32, 5).astype(np.uint32)]) # align 4Byte
    # print("32bit: ", [f"0x{val:08X}"  for val in dram.take_data(8, 32, 5).astype(np.uint32)]) # align 8Byte
    # print("32bit: ", [f"0x{val:08X}"  for val in dram.take_data(3, 32, 5).astype(np.uint32)]) # non-align

    # 64 bit testbench
    # print("64bit: ", [f"0x{val:016X}" for val in dram.take_data(0, 64, 6).astype(np.uint64)]) # align & start from 0
    # print("64bit: ", [f"0x{val:016X}" for val in dram.take_data(2, 64, 6).astype(np.uint64)]) # align 2Byte
    # print("64bit: ", [f"0x{val:016X}" for val in dram.take_data(4, 64, 6).astype(np.uint64)]) # align 4Byte
    # print("64bit: ", [f"0x{val:016X}" for val in dram.take_data(8, 64, 6).astype(np.uint64)]) # align 8Byte
    # print("64bit: ", [f"0x{val:016X}" for val in dram.take_data(3, 64, 6).astype(np.uint64)]) # non-align



    # store data testbench
    # data64 = np.array([0x1212121200000000, 0x3434343400000000, 0x5656565600000000, 0x7878787800000000], dtype=np.uint64)
    # dram.store_data(0, 64, data64)

    # data32 = np.array([0x12121212, 0x34343434, 0x56565656, 0x78787878, 0x9a9a9a9a, 0xbcbcbcbc, 0xdfdfdfdf], dtype=np.uint32)
    # dram.store_data(36, 32, data32)

    # data16 = np.array([0x1212, 0x3434, 0x5656, 0x7878, 0x9a9a, 0xbcbc, 0xdfdf], dtype=np.uint16)
    # dram.store_data(88, 16, data16)

    # data8 = np.array([0x12, 0x34, 0x56, 0x78, 0x9a, 0xbc, 0xdf, 0x78, 0x99], dtype=np.uint8)
    # dram.store_data(320, 8, data8)

    dram.store64bData(0, 0b1, 0x1111111111111111) #(addr, byte_strb, data)
    dram.store64bData(8, 0b110, 0x1111111111111111) #(addr, byte_strb, data)
    dram.store64bData(16, 0b10000000, 0x1111111111111111) #(addr, byte_strb, data)

    # === Print out the current DRAM ===
    output_path = os.path.join(current_dir, "log", "dram.txt")
    with open(output_path, "w", encoding="utf-8") as f:
        with redirect_stdout(f):
            dram.dumpMem_data(mode = 'debug')
            # print("8bit : ", [f"0x{val:02X}"  for val in dram.take_data(0x1400, 8, 160)])

