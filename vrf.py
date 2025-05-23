import os
import numpy as np
from contextlib import redirect_stdout
import tkinter as tk

# @ global variables
RegFile = 32

NONE = 5000      # the number represents no element in the vector register file (LMUL < 1)
digit_width = 5  # the width of each index number

class VRF:
    def __init__(self, NrLanes=4, VLEN=4096, NrBanks = 8, DataWidth=64, 
                 SEW=8, LMUL=1, debug=False):
        
        # === System parameters ===
        self.NrLanes   = NrLanes
        self.VLEN      = VLEN
        self.NrBanks   = NrBanks
        self.DataWidth = DataWidth
        self.debug     = debug

        self.VLENB           = self.VLEN  / 8  # vlen in bytes
        self.MaxVLenPerLane  = self.VLEN   / self.NrLanes
        self.MaxVLenBPerLane = self.VLENB / self.NrLanes
        self.VRFSizePerLane  = self.MaxVLenPerLane  * RegFile # there are 32 register in vector register file
        self.VRFBSizePerLane = self.MaxVLenBPerLane * RegFile
        self.VRFSize = int(self.VRFSizePerLane)

        # === Dynamic parameters ===
        self._SEW  = SEW     # SEW:  8, 16, 32, 64
        self._LMUL = LMUL    # LMUL: 1, 2, 4, 8 , 1/2, 1/4, 1/8
        self.VLMAX = self._LMUL * self.VLEN // self._SEW  # Maximum number of elements

        # === VRF memory structure: (Lane=4, Bank=8, Depth-per-Bank=64) ===
        self.VRF = np.zeros((self.NrLanes, self.NrBanks, 64), dtype=np.uint64)

    # Generate the VRF element index
    def Gen_idx(self, shuffle=False):

        # === create element index array ===
        if self._LMUL >= 1:
            idx_VLMAX = int(self.VLMAX)
            idx_arr = np.zeros(idx_VLMAX, dtype=int)
            virtual_LMUL = self._LMUL
        else:
            idx_VLMAX = int(self.VLEN // self._SEW)
            idx_arr = np.zeros(idx_VLMAX, dtype=int) # if LMUL < 1, the element index is limited to one VLEN
            virtual_LMUL = 1
        
        # === calculate the element index === (not shuffle)
        count_idx = 0
        for row in range(virtual_LMUL * self.VLEN // (self.NrLanes * self.NrBanks * self.DataWidth)):  # row in VRF per VLEN
            for lane in range(self.NrLanes):                                            # lane in VRF
                for bank in range(self.NrBanks):                                        # bank in lane
                    for word in range(self.DataWidth // self._SEW):                     # word in bank
                        rowoffset  = row * (self.NrLanes * self.NrBanks) * (self.DataWidth // self._SEW)
                        laneoffset = lane
                        bankoffset = bank * self.NrLanes * (self.DataWidth // self._SEW)
                        wordoffset = word * self.NrLanes

                        idx = rowoffset + laneoffset + bankoffset + wordoffset
                        if idx < self.VLMAX:
                            idx_arr[count_idx] = idx
                        else:
                            idx_arr[count_idx] = NONE

                        count_idx += 1
        
        # === reshape element idx per bank ===
        idx_arr = idx_arr.reshape(virtual_LMUL * self.VLEN // self.DataWidth, self.DataWidth // self._SEW) # (Total bank slot for VLEN, element inside the bank)

        # === re-order each bank element to fit with ara if needed ===
        if shuffle:
            for bank in range(virtual_LMUL * self.VLEN // self.DataWidth):
                if self._SEW == 32:
                    arr = [0, 1]
                elif self._SEW == 16:
                    arr = [0, 2, 1, 3]
                elif self._SEW == 8:
                    arr = [0, 4, 2, 6, 1, 5, 3, 7]
                else:
                    arr = [0]
                reordered = [idx_arr[bank][i] for i in arr]
                idx_arr[bank] = reordered
        
        # === flatten the reshape bank into 1D element idx set ==
        idx_arr = idx_arr.reshape(-1)

        # === reshape array to fit the VRF data storage dimension ===
        idx_arr = idx_arr.reshape((virtual_LMUL * self.VLEN // (self.NrLanes * self.NrBanks * self.DataWidth), 
                                   self.NrLanes,
                                   self.NrBanks,
                                   self.DataWidth // self._SEW)) # (row, lane, bank, word per bank)

        return idx_arr
    
    # function to print out the VRF element index
    def VRF_Idx(self):
        print("VRF Shape: ", self.VRF.shape)
        NumWords = int(self.VRFSize / (self.NrBanks * self.DataWidth))
        # print the VRF information
        print(f"VRF size per lane: {self.VRFBSizePerLane} byte")
        print(f"NrBanks: {self.NrBanks}")
        print(f"NumWords: {NumWords}")
        print(f"VLEN: {self.VLEN} b, SEW: {self._SEW} b, LMUL: {self._LMUL}")

        # Print Lane labels
        print("   |", end="")
        for i in range(self.NrLanes):
            label = f"lane{i}"
            width = self.NrBanks * ((self.DataWidth//self._SEW) * digit_width + 4)
            print(f"{label:^{width}}|", end="")  # 置中
        print()


        # element index array
        if self._LMUL >= 1:
            el_arr = np.zeros(int(self.VLMAX), dtype=int)
        else:
            el_arr = np.zeros(int(self.VLEN // self._SEW), dtype=int)

        # calculate the element index (shuffle)
        count_idx = 0
        
        if self._LMUL >= 1:
            virtual_LMUL = self._LMUL
        else:
            virtual_LMUL = 1

        for row in range(virtual_LMUL * self.VLEN // (self.NrLanes * self.NrBanks * self.DataWidth)):  # row in VRF per VLEN
            for lane in range(self.NrLanes):                                            # lane in VRF
                for bank in range(self.NrBanks):                                        # bank in lane
                    for word in range(self.DataWidth // self._SEW):                           # word in bank
                        rowoffset  = row * (self.NrLanes * self.NrBanks) * (self.DataWidth // self._SEW)
                        laneoffset = lane
                        bankoffset = bank * self.NrLanes * (self.DataWidth // self._SEW)
                        wordoffset = word * self.NrLanes

                        idx = rowoffset + laneoffset + bankoffset + wordoffset
                        if idx < self.VLMAX:
                            el_arr[count_idx] = idx
                        else:
                            el_arr[count_idx] = NONE

                        count_idx += 1
        
        # reshape element with the unit in bank
        el_arr = el_arr.reshape(virtual_LMUL * self.VLEN // self.DataWidth, self.DataWidth // self._SEW) # (Total bank slot for VLEN, element inside the bank)

        # print(el_arr) # (total element pre vreg, element per bank)

        # re-order each bank to fit with ara
        for bank in range(virtual_LMUL * self.VLEN // self.DataWidth):
            if self._SEW == 32:
                arr = [0, 1]
            elif self._SEW == 16:
                arr = [0, 2, 1, 3]
            elif self._SEW == 8:
                arr = [0, 4, 2, 6, 1, 5, 3, 7]
            else:
                arr = [0]
            reordered = [el_arr[bank][i] for i in arr]
            el_arr[bank] = reordered


        # print out the element index
        for reg in range(0, RegFile, virtual_LMUL):
            print(f"v{reg:>2}", end="")
            for bank in range(virtual_LMUL * self.VLEN // self.DataWidth):
                # next row of VRF
                if (bank) % ((self.NrLanes * self.NrBanks)) == 0:
                    if bank != 0:
                        print("   |", end="")
                    else:
                        print("|", end="")

                block = el_arr[bank]

                # formatted_block = " [" + "".join(f"{x:{digit_width}}" for x in block) + "] "
                formatted_block = " [" + "".join(f"{'t':>{digit_width}}" if x == NONE 
                                                                        else f"{x:{digit_width}}" for x in block) + "] "
                print(formatted_block, end="")

                # sperate to each lane
                if (bank + 1) % self.NrBanks == 0:
                    print("|", end="")

                # next row of VRF
                if (bank + 1) % ((self.NrLanes * self.NrBanks)) == 0:
                    print()

        print()

    # function to updata the SEW and LMUL
    def vset(self, EEW, EMUL):
        if EEW in [8, 16, 32, 64]:
            self._SEW = EEW
        else:
            raise ValueError("Invalid SEW value. Must be one of [8, 16, 32, 64].")

        if EMUL in [1, 2, 4, 8, 1/2, 1/4, 1/8]:
            self._LMUL = EMUL
        else:
            raise ValueError("Invalid LMUL value. Must be one of [1, 2, 4, 8, 1/2, 1/4, 1/8].")

        self.VLMAX = self._LMUL * self.VLEN // self._SEW

    # function to print out the VRF data
    def dumpVRF_data(self):
        print("----- VRF Data -----")
    
        for depth in range(64):
            for lane in range(self.NrLanes):
                for bank in range(self.NrBanks):
                    value = self.VRF[lane, bank, depth]
                    print(f"0x{value:016x}  ", end="")
                print("| ", end="")
            print()
    
    # function to load data to VRF
    def load(self, vd, vstart, elen, data):
        """
        this function is used to load data to VRF
        NOTE:
        (1) vd is take 0~31 as idx, even "LMUL" will change
        (2) the behavior is shifting each element to its correspond location
        TODO:
        need to consider the real case for 
        Ara whether to store full 64b in each bank or element one by one
        """
        idx_arr = self.Gen_idx(shuffle=True)

        for idx in range(elen):
            VRF_dimension = np.argwhere(idx_arr == (vstart + idx))[0] # (row, lane, bank, word per bank)
            if self._SEW == 64:
                val = np.uint64(int(data[idx]))
            else:
                val = np.uint64(int(data[idx]) << (VRF_dimension[3] * self._SEW))
            self.debug and print("hex: ", hex(val), "dec: ", val, "Dimension: ", VRF_dimension)
            self.VRF[ VRF_dimension[1] ][ VRF_dimension[2] ][ vd*2 + VRF_dimension[0] ] |= val

    # function to take data from VRF to do some operation
    def take(self, vs, vstart, elen):
        """
        this function is used to take data out of VRF
        TODO:
        need to consider the real case for 
        Ara whether to store full 64b in each bank or element one by one
        """
        rtn_vec = []

        idx_arr = self.Gen_idx(shuffle=True)

        for idx in range(elen):
            VRF_dimension = np.argwhere(idx_arr == (vstart + idx))[0] # (row, lane, bank, word per bank)
            
            word_8B = self.VRF[ VRF_dimension[1] ][ VRF_dimension[2] ][ vs*2 + VRF_dimension[0] ]
            
            if self._SEW == 64:
                element = word_8B
            else:
                shift_amount = int(VRF_dimension[3] * self._SEW)
                element_mask = ((1 << self._SEW) - 1)
                element = word_8B >> shift_amount & element_mask

            self.debug and print("hex: ", hex(element), "dec: ", element, "Dimension: ", VRF_dimension)
            rtn_vec.append(element)

        return rtn_vec


if __name__ == "__main__":
    print("=== VRF testbench ===")
    print("version: 2025.05.23")
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(current_dir, "log", "vrf_idx.txt")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)   # create the output path


    # === Construct a VRF ===
    vrf = VRF(SEW=8, LMUL=1)

    # === Print out the current VRF memory mapping (not synchronize with the change of vset currently)===
    with open(output_path, "w", encoding="utf-8") as f:
        with redirect_stdout(f):
            vrf.VRF_Idx()

    
    # === Print out some basic information ===
    print("Each VLEN idx Shape(row, lane, bank, word-per-bank):", vrf.Gen_idx().shape)
    print("VRF Shape(Lane, Bank, Depth-per-Bank): ", vrf.VRF.shape)


    # === Testbench for load data to VRF ===
    vrf.vset(64, 1) #(SEW, LMUL)
    print("<1> SEW: ", vrf._SEW, "LMUL: ", vrf._LMUL)
    vrf.load(0, 32, 10, [0x1111111111111111, 0x2222222222222222, 0x3333333333333333, 0x4444444444444444, 0x5555555555555555,
                        0x6666666666666666, 0x7777777777777777, 0x8888888888888888, 0x9999999999999999, 0xAAAAAAAAAAAAAAAA])  # (vd, vstart, elen, data)
    
    vrf.vset(32, 1) #(SEW, LMUL)
    print("<2> SEW: ", vrf._SEW, "LMUL: ", vrf._LMUL)
    vrf.load(1, 0, 10, [0x11111111, 0x22222222, 0x33333333, 0x44444444, 0x55555555,
                        0x66666666, 0x77777777, 0x88888888, 0x99999999, 0xAAAAAAAA])  # (vd, vstart, elen, data)
    
    vrf.vset(16, 1) #(SEW, LMUL)
    print("<3> SEW: ", vrf._SEW, "LMUL: ", vrf._LMUL)
    vrf.load(2, 0, 10, [0x1111, 0x2222, 0x3333, 0x4444, 0x5555,
                        0x6666, 0x7777, 0x8888, 0x9999, 0xAAAA])   # (vd, vstart, elen, data)
    
    vrf.vset(8, 1) #(SEW, LMUL)
    print("<4> SEW: ", vrf._SEW, "LMUL: ", vrf._LMUL)
    vrf.load(3, 0, 15, [0x11, 0x22, 0x33, 0x44, 0x55,
                        0x66, 0x77, 0x88, 0x99, 0xAA,
                        0xBB, 0xCC, 0xDD, 0xEE, 0xFF])   # (vd, vstart, elen, data)
    
    # === Print out the current VRF memory mapping ===
    with open(r"C:\Users\david\Desktop\IwantGraduate\abstract\VPU_simulator\vrf_data.txt", "w", encoding="utf-8") as f:
        with redirect_stdout(f):
            vrf.dumpVRF_data()
    

    # === Testbench for take data out from VRF ===
    vrf.vset(64, 1) #(SEW, LMUL)
    print("<5> SEW: ", vrf._SEW, "LMUL: ", vrf._LMUL)
    rtn_vec = vrf.take(0, 30, 10) # (vs, vstart, elen)
    print( [f"0x{val:X}"  for val in rtn_vec] )

    vrf.vset(32, 1) #(SEW, LMUL)
    print("<6> SEW: ", vrf._SEW, "LMUL: ", vrf._LMUL)
    rtn_vec = vrf.take(1, 0, 10) # (vs, vstart, elen)
    print( [f"0x{val:X}"  for val in rtn_vec] )

    vrf.vset(16, 1) #(SEW, LMUL)
    print("<7> SEW: ", vrf._SEW, "LMUL: ", vrf._LMUL)
    rtn_vec = vrf.take(2, 1, 10)   # (vs, vstart, elen)
    print( [f"0x{val:X}"  for val in rtn_vec] )
    
    vrf.vset(8, 1) #(SEW, LMUL)
    print("<8> SEW: ", vrf._SEW, "LMUL: ", vrf._LMUL)
    rtn_vec = vrf.take(3, 2, 15)   # (vs, vstart, elen)
    print( [f"0x{val:X}"  for val in rtn_vec] )

    # === Some long array can be taken as pattern ===
    lookup_array = np.array([
         0,  4,  8, 12, 16, 20, 24, 28,  1,  5,  9, 13, 17, 21, 25, 29,
         2,  6, 10, 14, 18, 22, 26, 30,  3,  7, 11, 15, 19, 23, 27, 31,
        32, 36, 40, 44, 48, 52, 56, 60, 33, 37, 41, 45, 49, 53, 57, 61,
        34, 38, 42, 46, 50, 54, 58, 62, 35, 39, 43, 47, 51, 55, 59, 63
    ])

    index_array = np.array([
         0,  4,  8, 12, 16, 20, 24, 28, 32,  36,  40,  44,  48,  52,  56,  60,
         1,  5,  9, 13, 17, 21, 25, 29, 33,  37,  41,  45,  49,  53,  57,  61,
         2,  6, 10, 14, 18, 22, 26, 30, 34,  38,  42,  46,  50,  54,  58,  62,
         3,  7, 11, 15, 19, 23, 27, 31, 35,  39,  43,  47,  51,  55,  59,  63,
        64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124,
        65, 69, 73, 77, 81, 85, 89, 93, 97, 101, 105, 109, 113, 117, 121, 125,
        66, 70, 74, 78, 82, 86, 90, 94, 98, 102, 106, 110, 114, 118, 122, 126,
        67, 71, 75, 79, 83, 87, 91, 95, 99, 103, 107, 111, 115, 119, 123, 127
    ])


    