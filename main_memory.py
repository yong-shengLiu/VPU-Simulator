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
    
    
    def init_byte_to_mem(self, pattern, byte_num):
        """
        The function to initial bulk data to memory in byte
        support byte_num: 1, 2, 4, 8
        """
        # === Reshape to (N, 8) ===
        dram_reshape = pattern.reshape(-1, 8//byte_num)  # Reshape per 8 bytes
        print("reshape: ", dram_reshape.shape)
        

        # === load into memory ===
        for idx, chunk in enumerate(dram_reshape):
            self.debug and print(chunk)
            
            chunk = chunk[::-1]  # reverse each chunk (little-endian)
 
            Dec64b = 0
            for word in chunk:
                self.debug and print(word)
                Dec64b = (Dec64b << (8*byte_num)) | int(word)
            
            self.debug and print(f"0x{Dec64b:016X}")
            self.memory[idx] = Dec64b

    def dumpMem_data(self, mode, Depth=None):
        """
        The function to dump all of memory
        Support mode:
        1. debug: used to dump python level memory in txt with 64b per data
        2. rtl: used to generate the hex file for rtl "readmemh" with 8b per data
        3. golden: used to generate the hex with "little-endian" 64b per row
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
                    byte_value = (int(value) & byte_mask) >> (byte * 8)
                    print(f"{byte_value:02X}")
        elif mode == 'golden':
            print("----- Memory data (little-endian) -----")
            for idx, value in enumerate(self.memory):
                # Convert 64-bit integer to little-endian byte sequence
                little_endian_value = int(value).to_bytes(8, byteorder='little', signed=False)
                print(" ".join(f"{byte:02x}" for byte in little_endian_value), end=" \n")
                
                if Depth is not None and idx >= Depth - 1:
                    break
        else:
            raise ValueError("dumpMem_data: Unsupported mode. Use 'debug', 'rtl', or 'golden'.")

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
        
        self.memory[align64_addr] = (int(self.memory[align64_addr]) & inv_mask) | (data & bit_mask) 
        

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
            # new_value = (old_value & inv_mask) | ((int(vector[idx]) << offset) & mask)
            new_value = (int(old_value) & int(inv_mask)) | ((int(vector[idx]) << int(offset)) & int(mask))
            self.memory[mem_addr] = new_value


if __name__ == "__main__":
    print("===== main memory testbench =====")
    print("version: 2025.11.03")

    dram = MEMORY(DataWidth=64, Depth=409600, debug=False)

    # load the DRAM pattern (float 32b)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # dir_np = os.path.join(current_dir, "pattern", "layer0.npy")
    dir_np = os.path.join(current_dir, "pattern", "ExpMant_Mat64_512.npy")
    dram_pattern = np.load(dir_np)

    # FP32b to uint8b
    # dram_pattern = dram_pattern.flatten().astype(np.uint8)

    print(f"Dram dtype: {dram_pattern.dtype}")
    print(f"Dram shape: {dram_pattern.shape}")
    # the preload pattern is represent in byte
    # dram.init_byte_to_mem(dram_pattern, 1) # (pattern, byte_num)
    dram.init_byte_to_mem(dram_pattern, 1) # (pattern, byte_num)

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

    data16 = np.array( [
        0x0064, 0x01b9, 0x00d2, 0x005c, 0xff64, 0x012b, 0xff80, 0x0322,
        0x03b6, 0xff11, 0x0255, 0x003b, 0x008b, 0x0368, 0xfc91, 0xfcb2,
        0xfc29, 0x02a9, 0x023a, 0x02f6, 0x03d4, 0x0265, 0xffb1, 0x023f,
        0xfcf2, 0x011f, 0xfd26, 0x038f, 0x002d, 0xff51, 0xfe1e, 0x0232,
        0xffa6, 0x008c, 0xfc26, 0x00f1, 0x00e6, 0x00ef, 0x038d, 0x0174,
        0xfee0, 0xff7f, 0x0195, 0xfc7b, 0x0156, 0x015d, 0xfdaf, 0xfd08,
        0xfe86, 0xfee9, 0x0090, 0xff82, 0x03e8, 0xfcd1, 0xfdac, 0xfd4a,
        0x013a, 0xfe07, 0xffbb, 0xfdf5, 0xfd46, 0xfce2, 0x0140, 0xfd1b,
        0xfd93, 0xfef3, 0x0291, 0xfcc7, 0x02b4, 0xfcc5, 0x03d0, 0xffc0,
        0x03d0, 0x00d7, 0x01ea, 0xfc50, 0xfe43, 0xfcf6, 0xfe5e, 0xfcf3,
        0xfe8b, 0xff50, 0xfc83, 0x018a, 0x0088, 0xfe20, 0x0030, 0xfcc0,
        0x009c, 0x036f, 0xfe8c, 0x0157, 0xfd0e, 0x01bb, 0xfe51, 0xfd77,
        0x00b1, 0xfc29, 0x02a2, 0xfc0a, 0x016c, 0xfe29, 0x01e2, 0x03b3,
        0xfdfd, 0x009c, 0x00bd, 0x0094, 0xfdc9, 0x039f, 0xff94, 0x02c5,
        0x0199, 0xfe61, 0x0283, 0xff2c, 0x030c, 0x00a6, 0x030e, 0x018a,
        0x01cd, 0x0003, 0x03a6, 0x0127, 0xff64, 0x00da, 0xfc27, 0xfe6a,
        0x0148, 0xfe52, 0x00f2, 0xff6e, 0xfd15, 0xfe63, 0x008f, 0x00ba,
        0x0098, 0x013a, 0x0138, 0xff74, 0x032c, 0xfef1, 0xff7d, 0x0323,
        0x0273, 0x01a2, 0xfccd, 0x035b, 0x01b7, 0x03fe, 0xfd32, 0x02f2,
        0xfd4d, 0x00ed, 0xfcfe, 0x02c9, 0x0275, 0x008e, 0xff42, 0xfc8e,
        0x0194, 0xffa1, 0x01c7, 0x02ee, 0x03ce, 0x02d9, 0xfc18, 0xfee1,
        0x01d7, 0xfd5f, 0x002b, 0xfc6f, 0xfd9a, 0xfc26, 0x0259, 0xfdcb,
        0xfec3, 0x036d, 0x01a3, 0xfc41, 0xfd51, 0x00f9, 0x009e, 0xfde7,
        0x0379, 0x00e9, 0x0049, 0x00b8, 0x01d7, 0xfe7f, 0xff30, 0xfdae,
        0xfd7d, 0x038e, 0x01eb, 0xffec, 0xfdd2, 0xfe09, 0xfc77, 0xff7a,
        0xfe7f, 0x0192, 0xff06, 0xfd70, 0xfc33, 0xfc8a, 0x016f, 0xffa1,
        0x004b, 0x032c, 0x03ec, 0xfdbc, 0x014e, 0xfe1b, 0xfc2a, 0x0211,
        0xfe8f, 0xff11, 0x00b5, 0x02a6, 0x0108, 0x02fb, 0xfe30, 0x0262,
        0xfd7c, 0x039f, 0x0180, 0xfdb9, 0x0394, 0x01d9, 0xfe08, 0xfdb5,
        0x0025, 0xfc35, 0xfda9, 0xff66, 0xfefe, 0xffb5, 0xfe39, 0x00b2,
        0x02e9, 0xfcf1, 0x0024, 0xfd0e, 0x01bc, 0xff2b, 0x0086, 0xfd77,
        0xfd29, 0xffe8, 0xfed8, 0x0386, 0x021f, 0x01fd, 0x033b, 0xfcab
    ], dtype=np.uint16)
    dram.store_data(0, 16, data16)

    data16 = np.array( [
        0xfc66, 0xfdbb, 0xfcd4, 0xfc5e, 0xfb66, 0xfd2d, 0xfb82, 0xff24,
        0xffb8, 0xfb13, 0xfe57, 0xfc3d, 0xfc8d, 0xff6a, 0xf893, 0xf8b4,
        0xf82b, 0xfeab, 0xfe3c, 0xfef8, 0xffd6, 0xfe67, 0xfbb3, 0xfe41,
        0xf8f4, 0xfd21, 0xf928, 0xff91, 0xfc2f, 0xfb53, 0xfa20, 0xfe34,
        0xfba8, 0xfc8e, 0xf828, 0xfcf3, 0xfce8, 0xfcf1, 0xff8f, 0xfd76,
        0xfae2, 0xfb81, 0xfd97, 0xf87d, 0xfd58, 0xfd5f, 0xf9b1, 0xf90a,
        0xfa88, 0xfaeb, 0xfc92, 0xfb84, 0xffea, 0xf8d3, 0xf9ae, 0xf94c,
        0xfd3c, 0xfa09, 0xfbbd, 0xf9f7, 0xf948, 0xf8e4, 0xfd42, 0xf91d,
        0xf995, 0xfaf5, 0xfe93, 0xf8c9, 0xfeb6, 0xf8c7, 0xffd2, 0xfbc2,
        0xffd2, 0xfcd9, 0xfdec, 0xf852, 0xfa45, 0xf8f8, 0xfa60, 0xf8f5,
        0xfa8d, 0xfb52, 0xf885, 0xfd8c, 0xfc8a, 0xfa22, 0xfc32, 0xf8c2,
        0xfc9e, 0xff71, 0xfa8e, 0xfd59, 0xf910, 0xfdbd, 0xfa53, 0xf979,
        0xfcb3, 0xf82b, 0xfea4, 0xf80c, 0xfd6e, 0xfa2b, 0xfde4, 0xffb5,
        0xf9ff, 0xfc9e, 0xfcbf, 0xfc96, 0xf9cb, 0xffa1, 0xfb96, 0xfec7,
        0xfd9b, 0xfa63, 0xfe85, 0xfb2e, 0xff0e, 0xfca8, 0xff10, 0xfd8c,
        0xfdcf, 0xfc05, 0xffa8, 0xfd29, 0xfb66, 0xfcdc, 0xf829, 0xfa6c,
        0xfd4a, 0xfa54, 0xfcf4, 0xfb70, 0xf917, 0xfa65, 0xfc91, 0xfcbc,
        0xfc9a, 0xfd3c, 0xfd3a, 0xfb76, 0xff2e, 0xfaf3, 0xfb7f, 0xff25,
        0xfe75, 0xfda4, 0xf8cf, 0xff5d, 0xfdb9, 0x0000, 0xf934, 0xfef4,
        0xf94f, 0xfcef, 0xf900, 0xfecb, 0xfe77, 0xfc90, 0xfb44, 0xf890,
        0xfd96, 0xfba3, 0xfdc9, 0xfef0, 0xffd0, 0xfedb, 0xf81a, 0xfae3,
        0xfdd9, 0xf961, 0xfc2d, 0xf871, 0xf99c, 0xf828, 0xfe5b, 0xf9cd,
        0xfac5, 0xff6f, 0xfda5, 0xf843, 0xf953, 0xfcfb, 0xfca0, 0xf9e9,
        0xff7b, 0xfceb, 0xfc4b, 0xfcba, 0xfdd9, 0xfa81, 0xfb32, 0xf9b0,
        0xf97f, 0xff90, 0xfded, 0xfbee, 0xf9d4, 0xfa0b, 0xf879, 0xfb7c,
        0xfa81, 0xfd94, 0xfb08, 0xf972, 0xf835, 0xf88c, 0xfd71, 0xfba3,
        0xfc4d, 0xff2e, 0xffee, 0xf9be, 0xfd50, 0xfa1d, 0xf82c, 0xfe13,
        0xfa91, 0xfb13, 0xfcb7, 0xfea8, 0xfd0a, 0xfefd, 0xfa32, 0xfe64,
        0xf97e, 0xffa1, 0xfd82, 0xf9bb, 0xff96, 0xfddb, 0xfa0a, 0xf9b7,
        0xfc27, 0xf837, 0xf9ab, 0xfb68, 0xfb00, 0xfbb7, 0xfa3b, 0xfcb4,
        0xfeeb, 0xf8f3, 0xfc26, 0xf910, 0xfdbe, 0xfb2d, 0xfc88, 0xf979,
        0xf92b, 0xfbea, 0xfada, 0xff88, 0xfe21, 0xfdff, 0xff3d, 0xf8ad
    ], dtype=np.uint16)
    dram.store_data(512, 16, data16)

    '''
    dram.store64bData(0, 0b1, 0x1111111111111111) #(addr, byte_strb, data)
    dram.store64bData(8, 0b110, 0x1111111111111111) #(addr, byte_strb, data)
    dram.store64bData(16, 0b10000000, 0x1111111111111111) #(addr, byte_strb, data)
    '''
    # === Print out the current DRAM ===
    output_path = os.path.join(current_dir, "log", "dram_rtl.txt")
    with open(output_path, "w", encoding="utf-8") as f:
        with redirect_stdout(f):
            dram.dumpMem_data(mode = 'rtl')
            # print("8bit : ", [f"0x{val:02X}"  for val in dram.take_data(0x1400, 8, 160)])'
    

