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
    print("version: 2025.11.17")

    dram = MEMORY(DataWidth=64, Depth=409600, debug=False)

    # load the DRAM pattern (float 32b)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # dir_np = os.path.join(current_dir, "pattern", "layer0.npy")
    dir_np = os.path.join(current_dir, "pattern", "ExpMant_Mat64_512.npy")  # garbage data
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

    # first Softmax input vector pattern (256 * 2-bytes = 4096-bit)
    Softmax_inputV1 = np.array( [
        0x0118, 0xfe29, 0xfc54, 0xfc22, 0x0282, 0x034d, 0x00da, 0x01d6,
        0x0059, 0x037b, 0x0287, 0xfc06, 0x02dc, 0xfc45, 0x01d6, 0xfd68,
        0x02e8, 0x0055, 0xfe66, 0xff62, 0xfc3a, 0xfcff, 0x015d, 0x012d,
        0x00ec, 0xff12, 0x03fa, 0x03d9, 0x017c, 0x0134, 0x0182, 0xff1d,
        0xfd15, 0x01c6, 0x0034, 0xfe7b, 0xffe3, 0x031e, 0x0379, 0xfedd,
        0x0092, 0xfe93, 0x00c1, 0xfeb4, 0xff22, 0x031f, 0xfdd1, 0x00fc,
        0xfcac, 0x02a9, 0x024c, 0xfdea, 0x0303, 0xfc78, 0xfeb0, 0xfd34,
        0xff9a, 0x025f, 0xfdd8, 0xfc6b, 0xff3d, 0xfd97, 0xfcba, 0x00a5,
        0xfe64, 0x0160, 0xfd99, 0x0389, 0xfeec, 0xfcd8, 0x0108, 0x036b,
        0xff86, 0x03a3, 0x0000, 0xff67, 0x00f6, 0x03f6, 0x0397, 0xffae,
        0x0210, 0xfffb, 0x003c, 0x0249, 0xff51, 0x01e0, 0x01b0, 0x0375,
        0xfceb, 0x01d5, 0x036b, 0x03be, 0xfc1e, 0x02e9, 0x03d9, 0x03a8,
        0xfd31, 0x03c8, 0x031f, 0x0294, 0xffd7, 0xfddc, 0x026a, 0x0363,
        0xfe21, 0x0050, 0xff8b, 0x0373, 0xfc53, 0x01db, 0x00ea, 0xfc3a,
        0x01c1, 0xfc21, 0x0210, 0x001a, 0x036f, 0xfc87, 0x02bb, 0xfc89,
        0xfec1, 0xff71, 0x03ba, 0x007f, 0xfe12, 0xfdef, 0x031b, 0xfdcf,
        0xfcff, 0xfe4f, 0x00b0, 0x006f, 0x027a, 0x007c, 0xfe4f, 0xff4e,
        0x028c, 0x0103, 0x03ac, 0xfef5, 0x006c, 0x00c0, 0x02c9, 0xfd2a,
        0xff41, 0x0348, 0xfc58, 0x0295, 0xff53, 0x02a3, 0xfc14, 0xfeec,
        0xfca1, 0x0139, 0xfe31, 0x019f, 0x038d, 0xfd04, 0x02eb, 0xfc7a,
        0xff0c, 0xff70, 0xffe9, 0x03d0, 0x0235, 0xfe79, 0xfe29, 0x02e8,
        0x030d, 0x0016, 0xfec1, 0x03f6, 0xfe87, 0xfd76, 0x030a, 0x0280,
        0x0158, 0x03ab, 0x0368, 0x01fc, 0x02e3, 0xfdfa, 0xfd21, 0x015c,
        0x01b8, 0xfd56, 0xff2a, 0x0348, 0x007e, 0x00a0, 0xfd8e, 0x0035,
        0x0030, 0xfcb6, 0x03db, 0x0092, 0xfc0d, 0x022e, 0x03d3, 0x00b8,
        0xfe8f, 0xfd80, 0x0161, 0xfd90, 0x009f, 0x00d1, 0x03b3, 0xfc94,
        0x0000, 0x01f4, 0xfd6b, 0xff1b, 0xfc81, 0x01cf, 0xfcb4, 0xff29,
        0x02fd, 0xffc7, 0x034d, 0x0221, 0x0353, 0xfd05, 0xfc97, 0xfc90,
        0x02f3, 0x0113, 0xfff9, 0xfd4f, 0x0164, 0xfe8b, 0x01b0, 0xffaf,
        0x000f, 0x0251, 0xfcbe, 0x00a1, 0xfd94, 0x0277, 0xffe9, 0x03e9,
        0xfd77, 0x03b4, 0x0268, 0xffda, 0x0282, 0x00d3, 0x013e, 0x034f,
        0xfc86, 0x02ae, 0xff0e, 0xfe9b, 0x03f4, 0x0240, 0xffe2, 0xff62
    ], dtype=np.uint16)
    dram.store_data(0, 16, Softmax_inputV1)
    # first Softmax Golden output vector pattern
        # 0x0000, 0x0000, 0x0000, 0x0000, 0x0001, 0x0002, 0x0000, 0x0000,
        # 0x0000, 0x0003, 0x0001, 0x0000, 0x0002, 0x0000, 0x0000, 0x0000,
        # 0x0002, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
        # 0x0000, 0x0000, 0x0007, 0x0004, 0x0000, 0x0000, 0x0000, 0x0000,
        # 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0002, 0x0003, 0x0000,
        # 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0002, 0x0000, 0x0000,
        # 0x0000, 0x0001, 0x0001, 0x0000, 0x0002, 0x0000, 0x0000, 0x0000,
        # 0x0000, 0x0001, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
        # 0x0000, 0x0000, 0x0000, 0x0004, 0x0000, 0x0000, 0x0000, 0x0003,
        # 0x0000, 0x0004, 0x0000, 0x0000, 0x0000, 0x0005, 0x0004, 0x0000,
        # 0x0000, 0x0000, 0x0000, 0x0001, 0x0000, 0x0000, 0x0000, 0x0003,
        # 0x0000, 0x0000, 0x0003, 0x0004, 0x0000, 0x0002, 0x0004, 0x0004,
        # 0x0000, 0x0004, 0x0002, 0x0001, 0x0000, 0x0000, 0x0001, 0x0003,
        # 0x0000, 0x0000, 0x0000, 0x0003, 0x0000, 0x0000, 0x0000, 0x0000,
        # 0x0000, 0x0000, 0x0000, 0x0000, 0x0003, 0x0000, 0x0001, 0x0000,
        # 0x0000, 0x0000, 0x0004, 0x0000, 0x0000, 0x0000, 0x0002, 0x0000,
        # 0x0000, 0x0000, 0x0000, 0x0000, 0x0001, 0x0000, 0x0000, 0x0000,
        # 0x0001, 0x0000, 0x0004, 0x0000, 0x0000, 0x0000, 0x0001, 0x0000,
        # 0x0000, 0x0002, 0x0000, 0x0001, 0x0000, 0x0001, 0x0000, 0x0000,
        # 0x0000, 0x0000, 0x0000, 0x0000, 0x0004, 0x0000, 0x0002, 0x0000,
        # 0x0000, 0x0000, 0x0000, 0x0004, 0x0001, 0x0000, 0x0000, 0x0002,
        # 0x0002, 0x0000, 0x0000, 0x0005, 0x0000, 0x0000, 0x0002, 0x0001,
        # 0x0000, 0x0004, 0x0003, 0x0000, 0x0002, 0x0000, 0x0000, 0x0000,
        # 0x0000, 0x0000, 0x0000, 0x0002, 0x0000, 0x0000, 0x0000, 0x0000,
        # 0x0000, 0x0000, 0x0004, 0x0000, 0x0000, 0x0000, 0x0004, 0x0000,
        # 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0004, 0x0000,
        # 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
        # 0x0002, 0x0000, 0x0002, 0x0000, 0x0003, 0x0000, 0x0000, 0x0000,
        # 0x0002, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
        # 0x0000, 0x0001, 0x0000, 0x0000, 0x0000, 0x0001, 0x0000, 0x0005,
        # 0x0000, 0x0004, 0x0001, 0x0000, 0x0001, 0x0000, 0x0000, 0x0002,
        # 0x0000, 0x0001, 0x0000, 0x0000, 0x0005, 0x0001, 0x0000, 0x0000


    # second Softmax input vector pattern (256 * 2-bytes = 4096-bit)
    Softmax_inputV2 = np.array( [
        0x0305, 0xfcb2, 0x01ab, 0x0250, 0x0265, 0xfe94, 0x0260, 0xfdcd,
        0xfee6, 0xff57, 0x0055, 0xfce7, 0xff41, 0xfc01, 0x01f4, 0x02d1,
        0xfd1d, 0x01a1, 0x0292, 0x03db, 0x02c0, 0xff65, 0x03d6, 0x03cb,
        0x0008, 0x0207, 0x0350, 0xffcf, 0x02e9, 0x019d, 0xfe5a, 0x0224,
        0x0091, 0xfcc0, 0xff22, 0xfc97, 0xffcf, 0xff6e, 0xff64, 0x00b1,
        0xfcfb, 0x0378, 0x0179, 0x0297, 0x032d, 0x00ab, 0xfc52, 0x01b1,
        0x008d, 0x029c, 0x0042, 0x0282, 0x03fa, 0xfece, 0xfd5e, 0xff22,
        0x0206, 0xff84, 0x00b5, 0xfd05, 0x01cf, 0xfe3e, 0xfd86, 0x02e7,
        0x0084, 0xffe0, 0x0331, 0xfcb0, 0x0192, 0xfea0, 0xfd67, 0x0166,
        0xfee7, 0xfea4, 0x038d, 0xfd98, 0x0019, 0xfc31, 0xfd4f, 0x0311,
        0x0250, 0x0074, 0xfdc8, 0x0076, 0xfc19, 0x01b4, 0x01bc, 0x012b,
        0x00e4, 0xfc97, 0xfdf9, 0x0098, 0xff27, 0x03f0, 0x0364, 0xfd37,
        0x00b8, 0x0192, 0xfd18, 0xfe80, 0x01ba, 0x0335, 0xfebc, 0xfde9,
        0x0293, 0x00ae, 0xffd0, 0xfe0d, 0xfc95, 0xfc25, 0x00a4, 0xfd87,
        0x03ce, 0xfcdc, 0xff9e, 0xff28, 0xfddc, 0x01fd, 0x0126, 0x01ce,
        0xfcaa, 0xfed2, 0x0029, 0xff6a, 0xfc53, 0xfd8d, 0x038f, 0xfd4d,
        0x02d1, 0x0294, 0xff21, 0xffbc, 0x0298, 0x0172, 0x02b2, 0x0210,
        0x0188, 0x034e, 0x0295, 0xfd6f, 0x01fc, 0xfcb2, 0xff68, 0xff2d,
        0xfd9e, 0x0381, 0xfcc2, 0xfc0a, 0xfe95, 0x03ed, 0xfe1e, 0x02a5,
        0xfd63, 0x00b1, 0x03ab, 0x01bb, 0x03d8, 0x0099, 0x03de, 0x02b2,
        0x023a, 0x031c, 0x010d, 0xfeda, 0x003a, 0xfdd0, 0x0238, 0xfd5c,
        0x009e, 0x004a, 0x0160, 0x0215, 0xfce1, 0x0100, 0xff50, 0x00ea,
        0x018d, 0x00af, 0x01dd, 0x0029, 0xffb4, 0xfe4b, 0xfdd5, 0x0190,
        0x0191, 0xfd90, 0x03c6, 0x015f, 0x0040, 0x02bb, 0xffe4, 0xffcf,
        0xfe11, 0xfd40, 0x01b1, 0x02c1, 0x016c, 0xfef3, 0x009b, 0x0082,
        0x037e, 0xff1a, 0xfd51, 0x0304, 0x0328, 0xfc63, 0xfd96, 0x0117,
        0x0250, 0x00db, 0xfd88, 0xfcf1, 0x000c, 0x0286, 0xfdbd, 0xfc9a,
        0x0069, 0xfd89, 0xfc8a, 0x0230, 0x0292, 0xff30, 0xfe5a, 0xfe38,
        0xfee3, 0x009e, 0x0039, 0xfed8, 0x0119, 0x0168, 0x0077, 0xff19,
        0x00fe, 0x00bc, 0xfeb9, 0xfe6d, 0x005e, 0x00e6, 0x00e3, 0xff10,
        0x0087, 0x03e3, 0xff6d, 0x02be, 0xfca7, 0x0300, 0x0389, 0xfe18,
        0xfc19, 0xffdd, 0xfd76, 0x03c6, 0x032e, 0x03af, 0x00d5, 0x001f
    ], dtype=np.uint16)
    dram.store_data(512, 16, Softmax_inputV2)
    # second Softmax Golden output vector pattern
        # 0x0002, 0x0000, 0x0000, 0x0001, 0x0001, 0x0000, 0x0001, 0x0000,
        # 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0002,
        # 0x0000, 0x0000, 0x0001, 0x0006, 0x0002, 0x0000, 0x0006, 0x0006,
        # 0x0000, 0x0001, 0x0004, 0x0000, 0x0002, 0x0000, 0x0000, 0x0001,
        # 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
        # 0x0000, 0x0005, 0x0000, 0x0001, 0x0003, 0x0000, 0x0000, 0x0000,
        # 0x0000, 0x0001, 0x0000, 0x0001, 0x0009, 0x0000, 0x0000, 0x0000,
        # 0x0001, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0002,
        # 0x0000, 0x0000, 0x0003, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
        # 0x0000, 0x0000, 0x0005, 0x0000, 0x0000, 0x0000, 0x0000, 0x0002,
        # 0x0001, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
        # 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0006, 0x0004, 0x0000,
        # 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0003, 0x0000, 0x0000,
        # 0x0001, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
        # 0x0006, 0x0000, 0x0000, 0x0000, 0x0000, 0x0001, 0x0000, 0x0000,
        # 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0005, 0x0000,
        # 0x0002, 0x0001, 0x0000, 0x0000, 0x0001, 0x0000, 0x0002, 0x0001,
        # 0x0000, 0x0003, 0x0001, 0x0000, 0x0001, 0x0000, 0x0000, 0x0000,
        # 0x0000, 0x0005, 0x0000, 0x0000, 0x0000, 0x0006, 0x0000, 0x0002,
        # 0x0000, 0x0000, 0x0005, 0x0000, 0x0006, 0x0000, 0x0006, 0x0002,
        # 0x0001, 0x0003, 0x0000, 0x0000, 0x0000, 0x0000, 0x0001, 0x0000,
        # 0x0000, 0x0000, 0x0000, 0x0001, 0x0000, 0x0000, 0x0000, 0x0000,
        # 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
        # 0x0000, 0x0000, 0x0006, 0x0000, 0x0000, 0x0002, 0x0000, 0x0000,
        # 0x0000, 0x0000, 0x0000, 0x0002, 0x0000, 0x0000, 0x0000, 0x0000,
        # 0x0005, 0x0000, 0x0000, 0x0002, 0x0003, 0x0000, 0x0000, 0x0000,
        # 0x0001, 0x0000, 0x0000, 0x0000, 0x0000, 0x0001, 0x0000, 0x0000,
        # 0x0000, 0x0000, 0x0000, 0x0001, 0x0001, 0x0000, 0x0000, 0x0000,
        # 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
        # 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
        # 0x0000, 0x0006, 0x0000, 0x0002, 0x0000, 0x0002, 0x0005, 0x0000,
        # 0x0000, 0x0000, 0x0000, 0x0006, 0x0003, 0x0005, 0x0000, 0x0000

    # continuous data in uint8
    # length = 512
    # data8 = np.arange(length, dtype=np.uint8)
    # dram.store_data(0, 8, data8)

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
    

