import numpy as np
from contextlib import redirect_stdout


class HLGenerator:
    def __init__(self, VLEN=4096, DataWidth=64, debug=False):

        # === parameters ===
        self.VLEN      = VLEN
        self.DataWidth = DataWidth
        self.debug     = debug

        # === Dynamic parameters ===
        self._SEW  = 8    # 64, 32, 16, 8
        self._LMUL = 1    # 8, 4, 2, 1
        self.VLMAX = self._LMUL * self.VLEN // self._SEW  # Maximum number of elements
        self.SEWB  = self._SEW // 8                       # byte for SEW


    def LoadMatrix(self, segment, seg_stride, seg_len, base_addr):
        """
        the function to load matrix to vrf with vector flatten (byte addresss)
        Used to Load data for CIM, "base_addr" is the byte address of the main memory
        TODO:
        (1) need to add the basd address for vrf in byte address (follow the StoreMatrix function)
        """
        
        inst_list = []
        arg_list  = []
        print(f"SEW:  {self._SEW}")
        print(f"LMUL: {self._LMUL}")
        print("========")

        # === The parameter checks whether a new C code instruction needs to be generated ====
        static_vl         = 0
        static_vstart     = 0
        static_vreg       = 0
        static_targetaddr = 0
        
        # === The parameter used to control vrf load/store flow ===
        #  TODO (below parameter maybe fetch from the VPU simulator in the future)
        vreg     = 0      # to record the current store vreg number
        vstart   = 0      # set vstart point (element idx)
        vl       = 0

        # === Generate the byte address will fetch main memory ===
        if self.debug:
            byte_addr = np.zeros((segment, seg_len), dtype=np.uint64)
            idx_count = 0
            for seg in range(segment):
                target_addr = base_addr + seg * seg_stride
                
                for len in range(seg_len):
                    if len != 0:
                        target_addr += 1
                    
                    byte_addr[seg][len] = target_addr
                    idx_count += 1
            
            for seg in range (segment):
                print(f"Seg{seg}")
                for len in range(seg_len):
                    print(byte_addr[seg][len], end="  ")
                
                print()

            print("==============")
            self.debug and print(self.VLMAX)
        
        # === strip-mining the AVL ===
        for seg in range(segment):
            print(f"Seg{seg}")
            target_addr = base_addr + seg * seg_stride
            AVL         = seg_len // self.SEWB  # application element length for each segment
            len         = 0
            
            while len < seg_len: # travel the byte for each segment
                # === The condition to check the parameter change ===
                vl_change         = False
                vstart_change     = False
                vreg_change       = False
                targetaddr_change = False

                print(f"Target Byte Addr: {target_addr:6}", end=",  ")
                if target_addr != static_targetaddr:
                    static_targetaddr = target_addr
                    targetaddr_change = True

                print(f"vreg(vd): {vreg:2}", end=",  ")
                if vreg != static_vreg:
                    static_vreg = vreg
                    vreg_change = True
                
                # === update parameter (next operation will be)===
                check_vstart = (vstart == 0)
                check_vlen   = AVL <= (self.VLMAX - vstart)

                if check_vstart and check_vlen:
                    # Case 1: vstart == 0 and VLEN is enough
                    elen = AVL
                    len  = len + elen * self.SEWB
                    if elen == (self.VLMAX - vstart):
                        vreg   = vreg + self._LMUL
                        vstart = 0
                    else:
                        vreg   = vreg
                        vstart = vstart + elen
                    self.debug and print("case1", end=",  ")
                elif check_vstart and not check_vlen:
                    # Case 2: vstart == 0 and VLEN is NOT enough
                    elen   = (self.VLMAX - vstart)
                    len    = len + elen * self.SEWB
                    vreg   = vreg + self._LMUL
                    vstart = 0
                    self.debug and print("case2", end=",  ")
                elif not check_vstart and check_vlen:
                    # Case 3: vstart != 0 and VLEN is enough
                    elen = AVL
                    len  = len + elen * self.SEWB
                    if AVL == (self.VLMAX - vstart):
                        vreg   = vreg + self._LMUL
                        vstart = 0
                    else:
                        vreg   = vreg
                        vstart = vstart + elen
                    self.debug and print("case3", end=",  ")
                else:
                    # Case 4: vstart != 0 and VLEN is NOT enough
                    elen   = (self.VLMAX - vstart)
                    len    = len + elen * self.SEWB
                    vreg   = vreg + self._LMUL
                    vstart = 0
                    self.debug and print("case4", end=",  ")

                print(f"elen: {elen:3}", end=",  ")
                
                
                if vreg_change :
                    vl = elen
                else:
                    vl = vl + elen
                print(f"vl: {vl:3}", end=",  ")

                if vl != static_vl:
                    static_vl = vl
                    vl_change = True
                
                print(f"len (byte): {len:4}", end=",  ")

                print(f"vstart: {vstart:3}")
                if vstart != static_vstart:
                    # static_vstart = vstart  # update later
                    vstart_change = True
                
                # === to check if there has new instruction needed ===
                if vl_change: 
                    inst_list.append(self.VectorCodeGen('vset',   [vl, self._SEW, self._LMUL]))
                    arg_list.append([vl, self._SEW, self._LMUL])
                if vstart_change: 
                    inst_list.append(self.VectorCodeGen('vstart', [static_vstart]))
                    arg_list.append([static_vstart])
                if vreg_change or targetaddr_change: 
                    inst_list.append(self.VectorCodeGen('vload_a', [self._SEW, vreg, target_addr-static_vstart]))  # NOTE the targets address need to minus "static_vstart"
                    arg_list.append([self._SEW, vreg, target_addr-static_vstart])

                
                # === Calculating the AVL ===
                target_addr = target_addr + len
                AVL = AVL - elen
                static_vstart = vstart
                
            print()
        
        return inst_list, arg_list
    
    def StoreMatrix(self, segment, seg_stride, seg_len, MMemeory_addr, vrf_addr):
        """
        the function to store matrix from vrf to main memory with scatter segment,
        "MMemeory_addr" is the byte address of the Main memory, 
        "vrf_addr" is the byte address of the vrf

        NOTE
        the vstart can only set with SEW=8, the other will cause unexpected behavior
        """

        inst_list = []
        arg_list  = []
        print(f"SEW:  {self._SEW}")
        print(f"LMUL: {self._LMUL}")
        print("========")

        # === The parameter checks whether a new C code instruction needs to be generated ====
        static_vstart      = 0
        static_vs          = 0
        static_target_addr = 0
        
        # === The parameter used to control vrf load/store flow ===
        #  TODO (below parameter maybe fetch from the VPU simulator in the future)
        vl       = 0
        vs       = 0      # to record the current source vs number
        vstart   = 0      # set vstart point (element idx)
        
        
        # === strip-mining the AVL ===
        for seg in range(segment):
            print(f"Seg{seg}")

            # === Pre-calculate start position of segment which in VRF and MMmemory===
            vrfaddr     = vrf_addr + seg * seg_len
            vs          = (vrfaddr // (self.VLEN // 8)) // self._LMUL * self._LMUL
            vstart      = vrfaddr % (self.VLEN // 8)

            target_addr = (MMemeory_addr + seg * seg_stride) - (vstart * self.SEWB) # NOTE the targets address need to minus "static_vstart(byte)"

            AVL         = seg_len // self.SEWB  # application element length for each segment
            len         = 0
            
            while len < seg_len: # travel the byte for each segment
                # === The condition to check the parameter change ===
                vstart_change         = False
                vs_change             = False
                target_addr_change    = False

                print(f"Source V Byte Addr: {vrfaddr:6}", end=",  ")

                print(f"vreg(vs): {vs:2}", end=",  ")
                if vs != static_vs:
                    static_vs = vs
                    vs_change   = True

                print(f"vstart: {vstart:3}", end=",  ")
                if vstart != static_vstart:
                    static_vstart = vstart
                    vstart_change = True

                
                print(f"Target Byte Addr: {target_addr:6}", end=",  ")
                if target_addr != static_target_addr:
                    static_target_addr     = target_addr
                    target_addr_change     = True
                

                # === update next vs, vstart and current execute elen, vl===
                check_vstart = (static_vstart == 0)
                check_vlen   = AVL <= (self.VLMAX - static_vstart)

                if check_vstart and check_vlen:
                    # Case 1: vstart == 0 and VLEN is enough
                    elen = AVL
                    len  = len + elen * self.SEWB
                    if elen == (self.VLMAX - vstart):
                        vs   = vs + self._LMUL
                        vstart = 0
                    else:
                        vs   = vs
                        vstart = vstart + elen
                    self.debug and print("case1", end=",  ")
                elif check_vstart and not check_vlen:
                    # Case 2: vstart == 0 and VLEN is NOT enough
                    elen   = (self.VLMAX - vstart)
                    len    = len + elen * self.SEWB
                    vs     = vs + self._LMUL
                    vstart = 0
                    self.debug and print("case2", end=",  ")
                elif not check_vstart and check_vlen:
                    # Case 3: vstart != 0 and VLEN is enough
                    elen = AVL
                    len  = len + elen * self.SEWB
                    if AVL == (self.VLMAX - vstart):
                        vs     = vs + self._LMUL
                        vstart = 0
                    else:
                        vs     = vs
                        vstart = vstart + elen
                    self.debug and print("case3", end=",  ")
                else:
                    # Case 4: vstart != 0 and VLEN is NOT enough
                    elen   = (self.VLMAX - vstart)
                    len    = len + elen * self.SEWB
                    vs     = vs + self._LMUL
                    vstart = 0
                    self.debug and print("case4", end=",  ")

                print(f"elen: {elen:3}", end=",  ")
                
                

                vl = static_vstart + elen
                print(f"vl: {vl:3}", end=",  ")
                   
                
                print(f"len (byte): {len:4}")

                
                # === to check if there has new instruction needed ===
                inst_list.append(self.VectorCodeGen('vset',   [vl, self._SEW, self._LMUL]))
                arg_list.append([vl, self._SEW, self._LMUL])
                if vstart_change: 
                    inst_list.append(self.VectorCodeGen('vstart', [static_vstart]))
                    arg_list.append([static_vstart])
                if vs_change or target_addr_change: 
                    inst_list.append(self.VectorCodeGen('vstore_a', [self._SEW, static_vs, static_target_addr]))  
                    arg_list.append([self._SEW, static_vs, static_target_addr])

                # === Calculating the AVL ===
                target_addr = target_addr + len
                vrfaddr     = vrfaddr + len
                AVL         = AVL - elen
                
            print()

        return inst_list, arg_list

    def VectorCodeGen(self, type, require_list):
        """
        this function is used to generate the VPU C code to run in RTL
        """
        
        if type == 'vset':  # [vl, sew, lmul]
            vl, sew, lmul = require_list
            return f'VSET({vl}, e{sew}, m{lmul});'

        elif type == 'vstart':  # [vstart]
            vstart = require_list[0]
            return f'write_csr(vstart, {vstart});'

        elif type == 'vload_a':  # [sew, vd, base_addr]
            sew, vd, base_addr = require_list
            return (f'asm volatile("vle{sew}.v v{vd}, (%0)" '
                    f'::"r"((uint{sew}_t*){base_addr}));')

        elif type == 'vstore_a': # [sew, vs, base_addr]
            sew, vs, base_addr = require_list
            return (f'asm volatile("vse{sew}.v v{vs}, (%0)" '
                    f'::"r"((uint{sew}_t*){base_addr}));')

        else:
            raise ValueError(f"Unsupported instruction type: {type}")
    

if __name__ == "__main__":
    
    instGenerator = HLGenerator(VLEN=4096, DataWidth=64, debug=False)
    print("=== HLGenerator testbench ===")
    print("version: 2025.05.23")

    # instGenerator.LoadMatrix(20, 5120, 160, 0)

    # === Print out the current DRAM ===
    DRAM_BASEADDR = 0xE0000000
    # with open(r"C:\Users\user\Desktop\LPHP\IwantGraduate\abstract\VPU_simulator\addr.txt", "w", encoding="utf-8") as f:
    #     with redirect_stdout(f):
    #         inst, arg = instGenerator.LoadMatrix(20, 5120, 160, DRAM_BASEADDR) #(segment, seg_stride, seg_len, base_addr)
    #         for line in inst:
    #             print(f"{line}")
    

    with open(r"C:\Users\user\Desktop\LPHP\IwantGraduate\abstract\VPU_simulator\addr.txt", "w", encoding="utf-8") as f:
        with redirect_stdout(f):
            inst, arg = instGenerator.StoreMatrix(36, 128, 72, DRAM_BASEADDR, 12960) #(segment, seg_stride, seg_len, MMemeory_addr, vrf_addr)
            for line in inst:
                print(f"{line}")