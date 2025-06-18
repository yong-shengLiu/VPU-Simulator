import os
import numpy as np
from contextlib import redirect_stdout
from VectorCodeGen import VectorCodeGenerator  # Import the VectorCodeGen class from the appropriate module
from VRF_scheduler import VRFScheduler
from Scalar_rf import Scalar_rf


class HLGenerator:
    def __init__(self, VLEN=4096, DataWidth=64, debug=False):
        self.codegen = VectorCodeGenerator()  # Initialize the VectorCodeGen class
        self.sched   = VRFScheduler()  # Initialize VRF Scheduler
        self.xrf     = Scalar_rf()  # Initialize VRF Scheduler

        # === parameters ===
        self.VLEN      = VLEN
        self.DataWidth = DataWidth
        self.debug     = debug

        # === Dynamic parameters ===
        self._SEW  = 8    # 64, 32, 16, 8
        self._LMUL = 1    # 8, 4, 2, 1
        self.VLMAX = self._LMUL * self.VLEN // self._SEW  # Maximum number of elements
        self.SEWB  = self._SEW // 8                       # byte for SEW

    def VSET(self, vl, sew=None, lmul=None):
        """
        Set the vl, SEW and LMUL for the vector operations.
        TODO: vstart
        """
        if sew is not None:
            self._SEW = sew
        if lmul is not None:
            self._LMUL = lmul
        self.VLMAX = self._LMUL * self.VLEN // self._SEW

        inst = self.codegen.VectorCodeGen('vset',   [vl, self._SEW, self._LMUL])
        arg  = [vl, self._SEW, self._LMUL]

        return inst, arg
        
    def Scatter_LS(self, mode, segment, seg_stride, seg_len, MMemeory_addr, vrf_addr, sew=None, lmul=None):
        """
        the function to load or store matrix between main memory and vrf
        NOTE:
        (1) "MMemeory_addr" is the byte address of the Main memory, 
            "vrf_addr" is the byte address of the vrf
        (2) mode can be "load" or "store"
        
        TODO:
        (1) need to support vrf segment stride (Scatter2Scatter)
        (2) vd may pass from a VRF shcedualer
        """
        
        if sew is not None:
            self._SEW = sew
        if lmul is not None:
            self._LMUL = lmul
        
        inst_list = []
        arg_list  = []
        # print(f"SEW:  {self._SEW}")
        # print(f"LMUL: {self._LMUL}")
        # print("========")

        # === The parameter checks whether a new C code instruction needs to be generated ====
        static_vstart      = 0
        static_vreg        = 0
        static_target_addr = 0
        
        # === The parameter used to control vrf load/store flow ===
        #  TODO (below parameter maybe fetch from the VPU simulator in the future)
        vl       = 0
        vreg     = 0      # to record the current vreg
        vstart   = 0      # set vstart point (element idx)
        
        
        # === strip-mining the AVL ===
        for seg in range(segment):
            # print(f"{mode} [Seg{seg}]")

            # === Pre-calculate start position of segment which in VRF and MMmemory===
            vrfaddr     = vrf_addr + seg * seg_len
            vreg        = (vrfaddr // (self.VLEN // 8)) // self._LMUL * self._LMUL
            vstart      = vrfaddr % (self.VLEN // 8)

            target_addr = (MMemeory_addr + seg * seg_stride) - (vstart * self.SEWB) # NOTE the targets address need to minus "static_vstart(byte)"

            AVL         = seg_len // self.SEWB  # application element length for each segment
            len         = 0
            
            while len < seg_len: # travel the byte for each segment
                # === The condition to check the parameter change ===
                vstart_change         = False
                vreg_change           = False
                target_addr_change    = False

                # print(f"VRF Byte Addr: {vrfaddr:6}", end=",  ")

                # print(f"vreg: {vreg:2}", end=",  ")
                if vreg != static_vreg:
                    static_vreg = vreg
                    vreg_change   = True

                # print(f"vstart: {vstart:3}", end=",  ")
                if vstart != static_vstart:
                    static_vstart = vstart
                    vstart_change = True

                
                # print(f"Target Byte Addr: {target_addr:6} (0x{target_addr:X})", end=",  ")
                if target_addr != static_target_addr:
                    static_target_addr     = target_addr
                    target_addr_change     = True
                

                # === update next vreg, vstart and current execute elen, vl===
                check_vstart = (static_vstart == 0)
                check_vlen   = AVL <= (self.VLMAX - static_vstart)

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

                # print(f"elen: {elen:3}", end=",  ")
                
                

                vl = static_vstart + elen
                # print(f"vl: {vl:3}", end=",  ")
                   
                
                # print(f"len (byte): {len:4}")

                
                # === to check if there has new instruction needed ===
                inst_list.append(self.codegen.VectorCodeGen('vset',   [vl, self._SEW, self._LMUL]))
                arg_list.append([vl, self._SEW, self._LMUL])
                if vstart_change: 
                    inst_list.append(self.codegen.VectorCodeGen('vstart', [static_vstart]))
                    arg_list.append([static_vstart])
                if vreg_change or target_addr_change or vstart_change:
                    mode == 'load'  and inst_list.append(self.codegen.VectorCodeGen('vload_a',  [self._SEW, static_vreg, static_target_addr]))
                    mode == 'store' and inst_list.append(self.codegen.VectorCodeGen('vstore_a', [self._SEW, static_vreg, static_target_addr]))
                    arg_list.append([self._SEW, static_vreg, static_target_addr])

                # === Calculating the AVL ===
                target_addr = static_target_addr + (static_vstart * self.SEWB) + len
                vrfaddr     = vrfaddr + len
                AVL         = AVL - elen
                
            # print()

        return inst_list, arg_list

    def Block_Scale_deprecate(self, Main_Base):
        
        def append_inst_arg(inst, arg, generator_func, *args, **kwargs):
            temp_inst, temp_arg = generator_func(*args, **kwargs)
            
            # If any returned value is not a list, convert it to a list
            if not isinstance(temp_inst, list):
                temp_inst = [temp_inst]
            if not isinstance(temp_arg, list):
                temp_arg = [temp_arg]

            inst.extend(temp_inst)
            arg.extend(temp_arg)
        
        def v2byte(vreg):
            return vreg*512
        inst = []
        arg  = []

        
        # === Load BF16 matrix from DRAM ===
        bf16 = self.sched.allocate('BF16', 2) # allocate BF16
        append_inst_arg(inst, arg, self.Scatter_LS, 'load', 1, 512, 512, Main_Base, v2byte(bf16[0]), sew=16, lmul=2)

        # === Seperate exp. ===
        Exp = self.sched.allocate('Exp', 1) # allocate Exp
        append_inst_arg(inst, arg, self.PurePrint, 'uint32_t scalar = 0;')
        append_inst_arg(inst, arg, self.ScalarOperation, 'equal', 'scalar', 7)# Scalar = 7
        append_inst_arg(inst, arg, self.VSET, 512, 8, 1)
        append_inst_arg(inst, arg, self.WXOperation, 'vnsrl', v2byte(Exp[0]), v2byte(bf16[0]), 'scalar')# >> 7

        # Store the result to DRAM (TODO need to remove)
        append_inst_arg(inst, arg, self.Scatter_LS, 'store', 1, 512, 512, Main_Base+1536, v2byte(Exp[0]))

        # === Seperate mantissa_plus ===
        Mant = self.sched.allocate('Mant', 1) # allocate Mant
        append_inst_arg(inst, arg, self.VSET, 512, 8, 1)
        append_inst_arg(inst, arg, self.ScalarOperation, 'equal', 'scalar', 8)
        append_inst_arg(inst, arg, self.WXOperation, 'vnsrl', v2byte(Mant[0]), v2byte(bf16[0]), 'scalar') # (element >> 8)
        append_inst_arg(inst, arg, self.ScalarOperation, 'equal', 'scalar', 0x80)
        append_inst_arg(inst, arg, self.VXOperation, 'vand', v2byte(Mant[0]), v2byte(Mant[0]), 'scalar') # (element >> 8 & 0x80)
        append_inst_arg(inst, arg, self.ScalarOperation, 'equal', 'scalar', 0x40)
        append_inst_arg(inst, arg, self.VXOperation, 'vor', v2byte(Mant[0]), v2byte(Mant[0]), 'scalar')  # (element >> 8 & 0x80) | 0x40
        append_inst_arg(inst, arg, self.ScalarOperation, 'equal', 'scalar', 1)
        temp = self.sched.allocate('temp', 1) # allocate temp
        append_inst_arg(inst, arg, self.WXOperation, 'vnsrl', v2byte(temp[0]), v2byte(bf16[0]), 'scalar')
        append_inst_arg(inst, arg, self.ScalarOperation, 'equal', 'scalar', 0x3F)
        append_inst_arg(inst, arg, self.VXOperation, 'vand', v2byte(temp[0]), v2byte(temp[0]), 'scalar') # (element >> 1 & 0x3F)
        append_inst_arg(inst, arg, self.VVOperation, 'vor', v2byte(Mant[0]), v2byte(temp[0]), v2byte(Mant[0]))      # (element >> 8 & 0x80) | 0x40 | (element >> 1 & 0x3F)
        self.sched.status()
        self.sched.free('temp')
        self.sched.free('BF16')

        # Store the result to DRAM (TODO need to remove)
        append_inst_arg(inst, arg, self.Scatter_LS, 'store', 1, 512, 512, Main_Base+2560, v2byte(Mant[0]))


        # === Find exp. max and calculate the difference ===
        mask = self.sched.allocate('mask', 1) # NOTE allocate mask, must be v0
        diff = self.sched.allocate('diff', 1) # allocate diff
        MaxExp = self.sched.allocate('MaxExp', 1) # allocate MaxExp

        append_inst_arg(inst, arg, self.PurePrint, f'VLOAD_8(v{MaxExp[0]}, 0x00);')
        append_inst_arg(inst, arg, self.PurePrint, 'uint8_t EXPMax = 0;')
        append_inst_arg(inst, arg, self.ScalarOperation, 'equal', 'scalar', 64)

        for iter in range(0, 512 // 64):
            Emask = ["0x00"] * 64
            for i in range(8):
                Emask[iter * 8 + i] = "0xff"
            bytes_str = ", ".join(Emask)

            append_inst_arg(inst, arg, self.PurePrint, f'VLOAD_8(v{mask[0]}, {bytes_str});')  # load mask

            temp = self.sched.allocate('temp', 1) # allocate temp
            append_inst_arg(inst, arg, self.VSOperation, 'vredmaxu', v2byte(temp[0]), v2byte(Exp[0]), v2byte(MaxExp[0]), mask='yes')  # find the block maximum
            append_inst_arg(inst, arg, self.XSOperation, 'vmv', v2byte(temp[0]), 'EXPMax')          # Store the block maximum to scalar
            self.sched.free('temp')
            append_inst_arg(inst, arg, self.VXOperation, 'vrsub', v2byte(diff[0]), v2byte(Exp[0]), 'EXPMax', mask='yes')  # calculate the difference


        self.sched.free('mask')
        self.sched.free('MaxExp')
        self.sched.free('Exp')
        
        # === signed shift mantissa ===
        ShiftMant = self.sched.allocate('ShiftMant', 1) # allocate ShiftMant
        append_inst_arg(inst, arg, self.VVOperation, 'vsra', v2byte(ShiftMant[0]), v2byte(Mant[0]), v2byte(diff[0]))

        self.sched.free('Mant')
        self.sched.free('diff')

        # Store the result to DRAM　(TODO need to remove)
        append_inst_arg(inst, arg, self.Scatter_LS, 'store', 1, 512, 512, Main_Base+3584, v2byte(ShiftMant[0]))

        return inst, arg

    def Block_Scale(self, Main_Base, width):
        """
        (1) Main_Base: NOTE (assume the data is loaded in a continue region)
        the base address of external memory, relate to axi bus address mapping.

        (2) width:
        the number of element will be masked to find the block maximum, support: 32, 64, 128

        TODO: (1) Revise all scalar to riscv asm
              (2) Use "with" to hidden the allocate and free
        """
        def append_inst_arg(inst, arg, generator_func, *args, **kwargs):
            temp_inst, temp_arg = generator_func(*args, **kwargs)
            
            # If any returned value is not a list, convert it to a list
            if not isinstance(temp_inst, list):
                temp_inst = [temp_inst]
            if not isinstance(temp_arg, list):
                temp_arg = [temp_arg]

            inst.extend(temp_inst)
            arg.extend(temp_arg)
        
        def v2byte(vreg):
            return vreg*512
        
        
        inst = []
        arg  = []
        total_row = 64

        
        # === find Exp. Maximum ===
        append_inst_arg(inst, arg, self.PurePrint, '// find Exp. Maximum')
        mask = self.sched.allocate('mask', 1) # NOTE allocate mask, must be v0
        MaxExp = self.sched.allocate('MaxExp', 1) # allocate MaxExp

        # append_inst_arg(inst, arg, self.VSET, 512, 8, 1)
        # append_inst_arg(inst, arg, self.PurePrint, f'VLOAD_8(v{MaxExp[0]}, 0x00);')
        append_inst_arg(inst, arg, self.Scatter_LS, 'load', 1, 512, 512, Main_Base+0x18000, v2byte(MaxExp[0]), sew=8, lmul=1)

        append_inst_arg(inst, arg, self.PurePrint, '// uint8_t EXPMax = 0;')
        EXPMax = self.xrf.allocate("EXPMax", 1)
        append_inst_arg(inst, arg, self.ScalarOperation, 'equal', EXPMax[0], 0)
        
        append_inst_arg(inst, arg, self.PurePrint, '// uint8_t Max_temp = 0;')
        Max_temp = self.xrf.allocate("Max_temp", 1)
        append_inst_arg(inst, arg, self.ScalarOperation, 'equal', Max_temp[0], 0)
        
        row_exe = 22  # NOTE a magic number

        for start_row in range(0, total_row, row_exe):
            end_row = min(start_row + row_exe, total_row)

            allocate_row = end_row - start_row

            # load exponent
            Exp = self.sched.allocate("Exp", allocate_row)
            append_inst_arg(inst, arg, self.Scatter_LS, 'load', allocate_row, 512, 512, Main_Base+(start_row*512), v2byte(Exp[0]), sew=8, lmul=1)

            # find reduction max, store to scalar
            par_Max = self.xrf.allocate("par_Max", 512 // width)
            for iter in range(0, 512 // width):
                if start_row == 0:
                    append_inst_arg(inst, arg, self.PurePrint, f'// uint8_t par_Max_{iter} = 0;')
                    append_inst_arg(inst, arg, self.ScalarOperation, 'equal', par_Max[iter], 0)
                
                # generate the mask
                # Emask = ["0x00"] * (512 // 8)
                # widthB = width // 8
                # for i in range(widthB):
                #     Emask[iter * widthB + i] = "0xff"
                # bytes_str = ", ".join(Emask)

                # append_inst_arg(inst, arg, self.PurePrint, f'VLOAD_8(v{mask[0]}, {bytes_str});')  # load mask
                append_inst_arg(inst, arg, self.Scatter_LS, 'load', 1, 512, 512, Main_Base+(0x18000+512)+512*iter, v2byte(mask[0]), sew=8, lmul=1)
                
                # find reduction maximum
                for exe in range(allocate_row):
                    temp = self.sched.allocate('temp', 1) # allocate temp
                    append_inst_arg(inst, arg, self.VSOperation, 'vredmaxu', v2byte(temp[0]), v2byte(Exp[exe]), v2byte(MaxExp[0]), mask='yes')  # find the block maximum
                    append_inst_arg(inst, arg, self.XSOperation, 'vmv', v2byte(temp[0]), 'EXPMax')          # Store the block maximum to scalar
                    self.sched.free('temp')

                    # Compare the maximum
                    append_inst_arg(inst, arg, self.PurePrint, 'if ( Max_temp < EXPMax) Max_temp = EXPMax;') # TODO: revise to assembly


                # Store the partial Maximum
                append_inst_arg(inst, arg, self.PurePrint, f'if ( par_Max_{iter} < Max_temp) par_Max_{iter} = Max_temp;') # TODO: revise to assembly
                append_inst_arg(inst, arg, self.PurePrint, '// Max_temp = 0;')
                append_inst_arg(inst, arg, self.ScalarOperation, 'equal', Max_temp[0], 0)

            # free exponent
            self.sched.free("Exp")
        
        
        self.sched.free('MaxExp')
        append_inst_arg(inst, arg, self.PurePrint, f'printf("Max0:%d, Max1:%d, Max2:%d, Max3:%d, Max4:%d, Max5:%d, Max6:%d, Max7:%d\\n", {par_Max[0]}, {par_Max[1]}, {par_Max[2]}, {par_Max[3]}, {par_Max[4]}, {par_Max[5]}, {par_Max[6]}, {par_Max[7]});')

        # === Calculate the Exp different and shift Mant ===
        append_inst_arg(inst, arg, self.PurePrint, '// Calculate the Exp different and shift Mant')
        row_exe = 10  # NOTE a magic number

        for start_row in range(0, total_row, row_exe):
            end_row = min(start_row + row_exe, total_row)

            allocate_row = end_row - start_row

            # load exponent
            Exp = self.sched.allocate("Exp", allocate_row)
            append_inst_arg(inst, arg, self.Scatter_LS, 'load', allocate_row, 512, 512, Main_Base+(start_row*512), v2byte(Exp[0]), sew=8, lmul=1)
            
            # Mask sliding to find the Differenct between Maximum
            diff = self.sched.allocate('diff', allocate_row)
            # self.sched.status()
            append_inst_arg(inst, arg, self.PurePrint, '// Max_temp = 0;')
            append_inst_arg(inst, arg, self.ScalarOperation, 'equal', Max_temp[0], 0)

            for iter in range(0, 512 // width):
                # # generate the mask
                # Emask = ["0x00"] * (512 // 8)
                # widthB = width // 8
                # for i in range(widthB):
                #     Emask[iter * widthB + i] = "0xff"
                # bytes_str = ", ".join(Emask)

                # append_inst_arg(inst, arg, self.PurePrint, f'VLOAD_8(v{mask[0]}, {bytes_str});')  # load mask

                append_inst_arg(inst, arg, self.Scatter_LS, 'load', 1, 512, 512, Main_Base+(0x18000+512)+512*iter, v2byte(mask[0]), sew=8, lmul=1)

                # Difference
                # append_inst_arg(inst, arg, self.PurePrint, f'Max_temp = par_Max_{iter};')
                for exe in range(allocate_row):
                    append_inst_arg(inst, arg, self.VXOperation, 'vrsub', v2byte(diff[exe]), v2byte(Exp[exe]), f'par_Max_{iter}', mask='yes')  # calculate the difference

            self.sched.free("Exp")
            
            append_inst_arg(inst, arg, self.Scatter_LS, 'store', allocate_row, 512, 512, Main_Base+0x10000+(start_row*512), v2byte(diff[0]), sew=8, lmul=1) # TODO this line need to delete
            
            # Load Mant and shift
            Mant = self.sched.allocate("Mant", allocate_row)
            ShiftMant = self.sched.allocate("ShiftMant", allocate_row)
            append_inst_arg(inst, arg, self.Scatter_LS, 'load', allocate_row, 512, 512, Main_Base+0x8000+(start_row*512), v2byte(Mant[0]), sew=8, lmul=1)

            for exe in range(allocate_row):
                append_inst_arg(inst, arg, self.VVOperation, 'vsra', v2byte(ShiftMant[exe]), v2byte(Mant[exe]), v2byte(diff[exe]))
                # append_inst_arg(inst, arg, self.Scatter_LS, 'store', 1, 512, 512, Main_Base+0x10000+(start_row*512)+(exe*512), v2byte(ShiftMant[exe]), sew=8, lmul=1)


            append_inst_arg(inst, arg, self.PurePrint, f'printf("Block{start_row} Store finish\\n");')
            self.sched.free("diff")
            self.sched.free("Mant")
            self.sched.free("ShiftMant")
        
        self.sched.free("mask")
        return inst, arg

    def VVOperation(self, mode, vd_addr, vs1_addr, vs2_addr):
        """
        Generates a VV (vector-vector) instruction string and its argument list.
        vs1_addr, vs2_addr, vd_addr: byte addresses of the source and destination vectors

        TODO: vset and vstart
        """

        # Convert byte address to register index
        def addr_to_reg(addr):
            return addr // (self.VLEN // 8) // self._LMUL

        vs1 = addr_to_reg(vs1_addr)
        vs2 = addr_to_reg(vs2_addr)
        vd  = addr_to_reg(vd_addr)

        # Generate instruction
        arg = [vs1, vs2, vd]
        inst = self.codegen.VectorCodeGen(f'{mode}_vv', arg)

        return inst, arg
    
    def V2VOperation(self, mode, vd_addr, vs_addr):
        """
        Generates a VV (vector to vector) instruction string and its argument list.
        vs1_addr, vs2_addr, vd_addr: byte addresses of the source and destination vectors

        TODO: vset and vstart
        """

        # Convert byte address to register index
        def addr_to_reg(addr):
            return addr // (self.VLEN // 8) // self._LMUL

        vs = addr_to_reg(vs_addr)
        vd  = addr_to_reg(vd_addr)

        # Generate instruction
        arg = [vs, vd]
        inst = self.codegen.VectorCodeGen(f'{mode}_v.v', arg)

        return inst, arg

    def VSOperation(self, mode, vd_addr, vs1_addr, vs2_addr, mask=None):
        """
        Generates a VV (vector-vector) instruction string and its argument list.
        vs1_addr, vs2_addr, vd_addr: byte addresses of the source and destination vectors

        TODO: vset and vstart
        """

        # Convert byte address to register index
        def addr_to_reg(addr):
            return addr // (self.VLEN // 8) // self._LMUL

        vs1 = addr_to_reg(vs1_addr)
        vs2 = addr_to_reg(vs2_addr)
        vd  = addr_to_reg(vd_addr)

        # Generate instruction
        arg = [vs1, vs2, vd]
        inst = self.codegen.VectorCodeGen(f'{mode}_vs', arg)

        if mask is not None:
            # Insert the mask before the final quote
            insert_str = f', v0.t'
            inst = inst.rstrip('");') + insert_str + '");'
            arg.append(mask)

        return inst, arg

    def VXOperation(self, mode, vd_addr, vs_addr, scalar, mask=None):
        """
        Generates a VX (vector-scalar) instruction string and its argument list.
        vs_addr, vd_addr: byte addresses of the source and destination vectors

        TODO: vset and vstart
        """
        def insert_mask(inst_str, mask='v0.t'):
            split_marker = '%[A]"'
            if split_marker in inst_str:
                return inst_str.replace(split_marker, f'%[A], {mask}"')
            else:
                raise ValueError("Expected assembly format not found.")

        # Convert byte address to register index
        def addr_to_reg(addr):
            return addr // (self.VLEN // 8) // self._LMUL

        vs1 = addr_to_reg(vs_addr)
        vd  = addr_to_reg(vd_addr)

        # Generate instruction
        arg = [vs1, scalar, vd]
        inst = self.codegen.VectorCodeGen(f'{mode}_vx', arg)

        

        # Example usage
        if mask is not None:
            # Insert the mask before the final quote
            inst = insert_mask(inst)
            arg.append(mask)
            
        return inst, arg

    def XVOperation(self, mode, vd_addr, scalar):
        """
        Generates a XV (scalar2vector) instruction string and its argument list.
        vs_addr, vd_addr: byte addresses of the source and destination vectors

        TODO: vset and vstart
        """

        # Convert byte address to register index
        def addr_to_reg(addr):
            return addr // (self.VLEN // 8) // self._LMUL

        vd  = addr_to_reg(vd_addr)

        # Generate instruction
        arg = [scalar, vd]
        inst = self.codegen.VectorCodeGen(f'{mode}_v.x', arg)

        return inst, arg

    def XSOperation(self, mode, vs_addr, scalar):
        """
        Generates a XS (vector[0] to scalar) instruction string and its argument list.
        vs_addr, vd_addr: byte addresses of the source and destination vectors

        TODO: vset and vstart
        """

        # Convert byte address to register index
        def addr_to_reg(addr):
            return addr // (self.VLEN // 8) // self._LMUL

        vs = addr_to_reg(vs_addr)

        # Generate instruction
        arg = [vs, scalar]
        inst = self.codegen.VectorCodeGen(f'{mode}_x.s', arg)

        return inst, arg

    def WXOperation(self, mode, vd_addr, vs_addr, scalar):
        """
        Generates a WX (Widening vector-scalar) instruction string and its argument list.
        vs_addr, vd_addr: byte addresses of the source and destination vectors

        TODO: vset and vstart
        """

        # Convert byte address to register index
        def addr_to_reg(addr):
            return addr // (self.VLEN // 8) // self._LMUL

        vs1 = addr_to_reg(vs_addr)
        vd  = addr_to_reg(vd_addr)

        # Generate instruction
        arg = [vs1, scalar, vd]
        inst = self.codegen.VectorCodeGen(f'{mode}_wx', arg)

        return inst, arg

    def VIOperation(self, mode, vd_addr, vs_addr, immediate):
        """
        Generates a VI (vector-immediate) instruction string and its argument list.
        vs_addr, vd_addr: byte addresses of the source and destination vectors

        TODO: vset and vstart
        """

        # Convert byte address to register index
        def addr_to_reg(addr):
            return addr // (self.VLEN // 8) // self._LMUL

        vs1 = addr_to_reg(vs_addr)
        vd  = addr_to_reg(vd_addr)

        # Generate instruction
        arg = [vs1, immediate, vd]
        inst = self.codegen.VectorCodeGen(f'{mode}_vi', arg)

        return inst, arg
    
    def ScalarOperation(self, mode, reg, value):
        """
        TODO: Support Arithmetic operation
        """
        inst = []
        arg  = []

        if mode == 'equal':
            if value > 0xfff:  # Needs LUI + ADDI
                upper = value >> 12
                lower = value & 0xfff
                inst.append(f'asm volatile ("lui x{reg}, {upper}");')
                inst.append(f'asm volatile ("addi x{reg}, x{reg}, {lower}");')
                arg.append([reg, upper])
                arg.append([reg, lower])
            else:
                inst.append(f'asm volatile ("addi x{reg}, x0, {value}");')
                arg = [reg, value]
            
            # inst = f'{reg} = {value};'
        return inst, arg
    
    def PurePrint(self, string):
        """
        TODO: This Only Gen C code, cannot used in Python VPU
        """
        
        inst = string
        arg = 0

        return inst, arg
if __name__ == "__main__":
    
    instGenerator = HLGenerator(VLEN=4096, DataWidth=64, debug=False)
    print("=== HLGenerator testbench ===")
    print("version: 2025.06.04")

    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(current_dir, "log", "Codeflow.txt")
    golden_path = os.path.join(current_dir, "log", "golden.txt")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)   # create the output path

    # === Print out the operation flow ===
    DRAM_BASEADDR = 0xE0000000
    with open(output_path, "w", encoding="utf-8") as f:
        with redirect_stdout(f):
            # === Testbench for CIM Load/Store ===
            # inst, arg = instGenerator.Scatter_LS('load', 20, 5120, 160, DRAM_BASEADDR, 0) #(mode, segment, seg_stride, seg_len, MMemeory_addr, vrf_addr)
            # for line in inst:
            #     print(f"{line}")
            
            # inst, arg = instGenerator.Scatter_LS('store', 20, 160, 160, DRAM_BASEADDR, 0) #(mode, segment, seg_stride, seg_len, MMemeory_addr, vrf_addr)
            # for line in inst:
            #     print(f"{line}")

            # === Testbench for Block-scale quantize ===
            inst, arg = instGenerator.Block_Scale(DRAM_BASEADDR, 64) #(Main_Base)
            for line in inst:
                print(f"{line}")

            
    
    # === Load the Golden Pattern ===
    dir_np = os.path.join(current_dir, "pattern", "conv0.npy")
    row_pattern = np.load(dir_np)
    byte_pattern = row_pattern.flatten().astype(np.uint8)

    with open(golden_path, "w", encoding="utf-8") as f:
        with redirect_stdout(f):
            for i in range(0, len(byte_pattern), 8):
                row = byte_pattern[i:i+8]
                print(" ".join(f"{b:02x}" for b in row), end=" \n")  # hex format, padded to 2 digits


    
    
        print(hex(((int(0xBE80) >> 7) & 0xFF)))
        print(hex(((int(0x3F66) >> 7) & 0xFF)))
        print(hex(((int(0x3EED) >> 7) & 0xFF)))
        print(hex(((int(0x3E4A) >> 7) & 0xFF)))
    
    arithinst, aritharg = instGenerator.WXOperation('vnsrl', 1024, 0, 'scalar') #(mode, vd_addr, vs_addr, scalar)
    print(arithinst, aritharg)
    
    # instGenerator.sched.status()

    arithinst, aritharg = instGenerator.ScalarOperation('equal', 1, 20) #(self, mode, reg, value)
    print(arithinst, aritharg)


    arithinst, aritharg = instGenerator.ScalarOperation('equal', 2, 2**12) #(self, mode, reg, value)
    print(arithinst, aritharg)
    