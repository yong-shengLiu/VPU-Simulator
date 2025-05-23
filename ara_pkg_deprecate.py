import numpy as np
from contextlib import redirect_stdout
import tkinter as tk

# ara package
# Reference ara_pkg.sv in ara platform
# @parameter
RegFile = 32
NrLanes = 4
VLEN = 4096       # vlen in bits
VLENB = VLEN / 8  # vlen in bytes
MaxVLenPerLane  = VLEN  / NrLanes
MaxVLenBPerLane = VLENB / NrLanes
VRFSizePerLane  = MaxVLenPerLane  * RegFile # there are 32 register in vector register file
VRFBSizePerLane = MaxVLenBPerLane * RegFile
NONE = 5000 # the number represents no element in the vector register file (LMUL < 1)

# To store data into VRF
# NrLanes support: 1, 2, 4, 8, 16
# EW support: 64, 32, 16, 8
def shuffle_idx(byte_idx, nr_lanes=NrLanes, ew=64, debug=False):
    if nr_lanes == 4:
        print('Lane: 4')

# VRF
# @parameter
#  NrBanks: number of banks in the vector register file
#  VRFSize: Size of the VRF, in bits
#  DataWidth
#  vaddr_t: depth of VRF
# @IO
#  req_i, addr_i, 
#  tgt_opqueue_i ??
#  wen_i, wdata_i, be_i
#  operand_o, operand_valid_o
VRFSize = int(VRFSizePerLane)
NrBanks = NrVRFBanksPerLane = 8
DataWidth = 64
SEW  = 8      # SEW:  8, 16, 32, 64
LMUL = 1       # LMUL: 1, 2, 4, 8 , 1/2, 1/4, 1/8
VLMAX = LMUL * VLEN // SEW  # Maximum number of elements

digit_width = 5  # 每個數字用 3 格

def VRF_Idx():
    NumWords = int(VRFSize / (NrBanks * DataWidth))
    # print the VRF information
    print(f"VRF size per lane: {VRFBSizePerLane} byte")
    print(f"NrBanks: {NrBanks}")
    print(f"NumWords: {NumWords}")
    print(f"VLEN: {VLEN} b, SEW: {SEW} b, LMUL: {LMUL}")
    
    # for mul in range(0, RegFile, LMUL):
    #     print(mul)

    # create VRF to store value (lane, bank, word)
    VRF = np.zeros((NrLanes, NrBanks, DataWidth//SEW))

    # Print Lane labels
    print("   |", end="")
    for i in range(NrLanes):
        label = f"lane{i}"
        width = NrBanks * ((DataWidth//SEW) * digit_width + 4)
        print(f"{label:^{width}}|", end="")  # 置中
    print()


    # element index array
    if LMUL >= 1:
        el_arr = np.zeros(int(VLMAX), dtype=int)
    else:
        el_arr = np.zeros(int(VLEN // SEW), dtype=int)

    # calculate the element index (shuffle)
    count_idx = 0
    
    if LMUL >= 1:
        virtual_LMUL = LMUL
    else:
        virtual_LMUL = 1

    for row in range(virtual_LMUL * VLEN // (NrLanes * NrBanks * DataWidth)):  # row in VRF per VLEN
        for lane in range(NrLanes):                                            # lane in VRF
            for bank in range(NrBanks):                                        # bank in lane
                for word in range(DataWidth // SEW):                           # word in bank
                    rowoffset  = row * (NrLanes * NrBanks) * (DataWidth // SEW)
                    laneoffset = lane
                    bankoffset = bank * NrLanes * (DataWidth // SEW)
                    wordoffset = word * NrLanes

                    idx = rowoffset + laneoffset + bankoffset + wordoffset
                    if idx < VLMAX:
                        el_arr[count_idx] = idx
                    else:
                        el_arr[count_idx] = NONE

                    count_idx += 1
    
    # reshape element with the unit in bank
    el_arr = el_arr.reshape(virtual_LMUL * VLEN // DataWidth, DataWidth // SEW) # (Total bank slot for VLEN, element inside the bank)

    # print(el_arr) # (total element pre vreg, element per bank)

    # re-order each bank to fit with ara
    for bank in range(virtual_LMUL * VLEN // DataWidth):
        if SEW == 32:
            arr = [0, 1]
        elif SEW == 16:
            arr = [0, 2, 1, 3]
        elif SEW == 8:
            arr = [0, 4, 2, 6, 1, 5, 3, 7]
        else:
            arr = [0]
        reordered = [el_arr[bank][i] for i in arr]
        el_arr[bank] = reordered


    # print out the element index
    for reg in range(0, RegFile, virtual_LMUL):
        print(f"v{reg:>2}", end="")
        for bank in range(virtual_LMUL * VLEN // DataWidth):
            # next row of VRF
            if (bank) % ((NrLanes * NrBanks)) == 0:
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
            if (bank + 1) % NrBanks == 0:
                print("|", end="")

            # next row of VRF
            if (bank + 1) % ((NrLanes * NrBanks)) == 0:
                print()

    print()
    # print(el_arr.shape)

    # # reshape back to 1D
    # if LMUL >= 1:
    #     totalElement = VLMAX
    # else:
    #     totalElement = VLEN // SEW
    # el_arr = el_arr.reshape(totalElement)
    # print(el_arr.shape)
    # print(el_arr)
    # val_list = [i**2 for i in el_arr]
    # visualize_memory(el_arr, val_list, LMUL, VLEN, SEW)
    # print(val_list)

    

def visualize_memory(idx_list, val_list, LMUL=1, VLEN=128, SEW=8):
    mem_dict = {idx: val for idx, val in zip(idx_list, val_list)}
    
    root = tk.Tk()
    root.title("VRF Viewer")

    # 外層 Canvas + Scrollbar
    canvas = tk.Canvas(root, height=80)
    canvas.pack(side=tk.TOP, fill=tk.X, expand=True)

    xscrollbar = tk.Scrollbar(root, orient=tk.HORIZONTAL, command=canvas.xview)
    xscrollbar.pack(side=tk.BOTTOM, fill=tk.X)
    canvas.configure(xscrollcommand=xscrollbar.set)

    # 內部 Frame 放在 Canvas 裡
    inner_frame = tk.Frame(canvas)
    canvas.create_window((0, 0), window=inner_frame, anchor='nw')

    for i, idx in enumerate(idx_list):
        val = mem_dict.get(idx, "--")
        display = f"{idx:02d}\n{val if isinstance(val, str) else hex(val)}"
        label = tk.Label(inner_frame, text=display, width=8, height=3, borderwidth=1, relief="solid", bg="lightgreen")
        label.grid(row=0, column=i, padx=1, pady=2)

    # 更新 scroll region
    inner_frame.update_idletasks()
    canvas.config(scrollregion=canvas.bbox("all"))

    root.mainloop()

if __name__ == "__main__":
    with open("vrf_output.txt", "w", encoding="utf-8") as f:
        with redirect_stdout(f):
            VRF_Idx()

    VRF_Idx()