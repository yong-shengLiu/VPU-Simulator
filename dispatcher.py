# Global parameter
VLEN = 4096 # bit
ELEN = 64 # bit
vs = 0 #TODO

# 1. Concept of EEW & EMUL
# For register distination, EEW and EMUL will different from SEW and LMUL respectivly.
# But the fraction(EEW/EMUL) should be same
# In normal, EEW = SEW, EMUL = LMUL 



class DISPATCHER:
    def __init__(self, debug=False):
        self.debug = debug
        # === Constant parameter ===
        self.vlenb = VLEN / 8
        
        # === Dynamic parameter ===
        self.LMUL       = 1
        self.SEW        = 8
        self.vd         = 0
        self.scalar_imm = 0   # TODO a scalar regfile with 32 index

        #  === CSRs ===
        self.vstart  = 0
        self.vl      = 0
        self.vtype   = 0
        self.mstatus = 0 | vs << 9 #TODO
        
        # === RISC-V encode Mapper ===
        self.vlmul_map = {
            0: 1,
            1: 2,
            2: 3,
            3: 8,
            5: 1/8,
            6: 1/4,
            7: 1/2
        }
        self.vlmul_reverse = self._generate_reverse_map(self.vlmul_map)

        self.vsew_map = {
            0: 8,
            1: 16,
            2: 32,
            3: 64
        }
        self.vsew_reverse = self._generate_reverse_map(self.vsew_map)

    
    def _generate_reverse_map(self, map_dict):
        """Generic method to generate reverse map from any dictionary."""
        return {v: k for k, v in map_dict.items()}
    
    
    @property # Maximum Vector Length: set by vtype
    def VLMAX(self):
        return VLEN * self.LMUL // self.SEW
    
    def decodeCAPI(self, CAPI, arg):
        """
        this function is used to decode C code API, which can run in RTL
        """
        type = 'None'

        # === Check for specific instuction types
        if "VSET" in CAPI:     # [vl, sew, lmul]
            self.vl = arg[0]
            self.set_vtype(arg[2], arg[1], 0, 0)
            type = 'vset'
        
        elif "vstart" in CAPI: # [vstart]
            self.vstart = arg[0]
            type = 'vstart'
            
        elif "vle" in CAPI:   # [sew, vd, base_addr]  TODO decode uint stride, stride, index
            self.SEW        = arg[0]
            self.vd         = arg[1]
            self.scalar_imm = arg[2]
            type = 'vload_a'
        else:
            print("-> Unknown instrtuction")
        
        return type

    def decode(self, instruction):
        # CSR: vstart, vxsat, vxrm, vcsr, vtype, vl, vlenb
        ## vsetvli, vsetivli, vsetvl
        if instruction.startswith(("vsetvli", "vsetivli", "vsetvl")):
            self.debug and print("CSR setting")

        # Vector Load/Store  (generate from copilte, not sure if it is correct)
        ## unit-stride: vle8, vle16, vle32, vle64, vse8, vse16, vse32, vse64, vlm, vsm
        ## stride: vlse8, vlse16, vlse32, vlse64, vsse8, vsse16, vsse32, vsse64
        ## indexed: vluxei8, vluxei16, vluxei32, vluxei64, vloxei8, vloxei16, vloxei32, vloxei64, vsuxei8, vsuxei16, vsuxei32, vsuxei64, vsoxei8, vsoxei16, vsoxei32, vsoxei64
        ## unit-stride fault-only-first load: vle8ff, vle16ff, vle32ff, vle64ff
        ## load/store segment: vlseg<nf>e<eew>, vsseg<nf>e<eew>, vlsseg<nf>e<eew>, vssseg<nf>e<eew>, vluxseg<nf>ei<eew>, vloxseg<nf>ei<eew>, vsuxseg<nf>ei<eew>, vsoxseg<nf>ei<eew>
        ## load/store whole register: 
        

        # Vector Integer Arithmetic
        ## single-width integer add and subtract: vadd, vsub, vrsub
        if instruction.startswith(("vadd", "vsub", "vrsub")):
            self.debug and print("single-width integer add and subtract")
        ## widening integer add/subtract: vwaddu, vwsubu, vwadd, vwsub
        ## integer extension: vzext, vsext
        ## Integer Add-with-Carry / Subtract-with-Borrow Instructions: vadc, vmadc, vmmv
        ## bitwise logical: vand, vor, vxor, vsll, vsrl, vsra
        ## narrowing integer right shift: vnsrl, vnsra
        ## integer compare: vmseq, vmsne, vmsltu, vmslt, vmsleu, vmsle, vmsgtu, vmsgt, (vmsgeu, vmsge)
        ## integer min/max: vminu, vmin, vmaxu, vmax
        ## signle-width integer multiply: vmul, vmulh, vmulhu, vmulhsu
        ## integer divide: vdivu, vdiv, vremu, vrem
        ## widening integer multiply: vwmul, vwmulu, vwmulsu
        ## single-width integer multiply-add: vmacc, vnmsac, vmadd, vnmsub
        ## widening integer multiply-add: vwmaccu, vwmacc, vwmaccsu, vwmaccus
        ## integer merge instruction: vmerge
        ## integer move: vmv

        # Vector Fixed-Point Arithmetic
        ## single-width saturating add and subtract: vsaddu, vsadd, vssubu, vssub
        ## single-width averaging add and subtract: vaaddu, vaadd, vasubu, vasub
        ## single-width fractional multiply with rounding and saturation: vsmul
        ## single-width scaling shift: vssrl, vssra
        ## narrowing fixed-point clip: vnclipu, vnclip

        # Vector Floating-Point Arithmetic
        ## single-width floating-point add/subtract: vfadd, vfsub, vfrsub
        ## widening floating-point add/subtract: vfwadd, vfwsub
        ## single-width floating-point multiply/divide: vfmul, vfdiv, vfrdiv
        ## widening floating-point multiply: vfwmul
        ## single-width floating-point fused multiply-add: vfmacc, vfnmacc, vfmsac, vfnmsac, vfmadd, vfnmadd, vfmsub, vfnmsub
        ## widening floating-point fused multiply-add: vfmacc, vfwnmacc, vfwmsac, vfwnmsac
        ## foating-point square-root: vfsqrt
        ## floating-point reciprocal square-root estimate: vfrsqrt7
        ## floating-point reciprocal estimate: vfrec7
        ## floating-point min/max: vfmin, vfmax
        ## floating-point sign-injection: vfsgnj, vfsgnjn, vfsgnjx
        ## floating-point compare: vmfeq, vmfne, vmflt, vmflem vmfgt, vmfge
        ## floating-point classify: vfclass
        ## floating-point merge: vfmerge
        ## floating-point move: vfmv
        ## single-width floating-point/integer typr-convert: vfcvt
        ## widening floating-point/integer type-convert: vfwcvt
        ## narrowing floating-point/integer type-convert: vfncvt

        # Vector Reduction Operation
        ## single-width integer redution: vredsum, vredmaxu, vredmax, vredminu, vredmin, vredand, vredor, vredxor
        ## widening integer reduction: vwredsumu, vwredsum
        ## single-width floating-point reduction: vfredosum, vfredusum, vfredmax, vfredmin
        ## widening floating-point reduction: vfwredosum

        # Vector Mask Instruction
        ## mask-register logical: vmadd, vmnand, vmandn, vmxor, vmor, vmnor, vmorn, vmxnor
        ## count population in mask: vcpop
        ## find-first-set mask bit: vfirst
        ## set-before-first mask bit: vmsbf
        ## set-including-first mask bit: vmsif
        ## set-only-first mask bit: vmsof
        ## iota Instruction: viota
        ## element index: vid

        # Vector Permutation Instruction
        ## slide: vslideup, vslidedown, vslide1up, vfslide1up, vslide1down, vfslide1down
        ## register gather: vrgather, vrgatherei16
        ## compress: vcompress
        else:
            self.debug and print(f"Unknown instruction: {instruction}")

    def Bin2Asm(self, inst):
        
        # vset
        if (inst & 0b1111111 == 0b1010111):
            # type
            if inst >> 25 == 0b1000000: type = "vsetvl"
            elif inst >> 30 == 0b11:    type = "vsetivli"
            elif inst >> 31 == 0b0:     type = "vsetvli"
            print( f"{type}")

            # rs1 (AVL)
            rs1 = inst >> 15 & 0b11111
            print(f"rs1(AVL): {rs1}", end=", ")

            # rs2 (vtype)
            if type == "vsetvli":
                rs2 = inst >> 20 & 0b11111111111
                vlmul = rs2 & 0b111
                vsew  = (rs2 >> 3) & 0b111
                vta   = (rs2 >> 6) & 0b1
                vma   = (rs2 >> 7) & 0b1
                print(f"rs2(vtype) -> vma:{vma}, vta:{vta}, vsew:{vsew}. vlmul:{vlmul}" ,end=", ") # TODO: vsew, vlmul mapping
            elif type == "vsetivli":
                rs2 = inst >> 20 & 0b1111111111
                vlmul = rs2 & 0b111
                vsew  = (rs2 >> 3) & 0b111
                vta   = (rs2 >> 6) & 0b1
                vma   = (rs2 >> 7) & 0b1
                print(f"rs2(vtype) -> vma:{vma}, vta:{vta}, vsew:{vsew}. vlmul:{vlmul}" ,end=", ") # TODO: vsew, vlmul mapping
            elif type == "vsetvl":
                rs2 = inst >> 20 & 0b11111
                print(f"rs2(vtype): {rs2:05b}", end=", ")
            
            # rd
            rd = inst >> 7 & 0b11111
            print(f"rd: {rd}")

        # vload/store
        elif (inst & 0b1111111 == 0b0000111) or (inst & 0b1111111 == 0b0100111): 
            type = inst >> 5 & 0b1 # 0: vload, 1: vstore
            if type == 0:
                print("vload")
            else:
                print("vstore")
            
            # mop
            mop = inst >> 26 & 0b11
            if   mop == 0b00: mop_mode = "unit-stride"
            elif mop == 0b01: mop_mode = "indexed-unordered"
            elif mop == 0b10: mop_mode = "strided"
            elif mop == 0b11: mop_mode = "indexed-ordered"
            print(f"mop: {mop_mode}", end=", ")

            # nf
            nf = inst >> 29 & 0b111
            # mew
            mew = inst >> 28 & 0b1
            # vm
            vm = inst >> 25 & 0b1
            if vm == 0:
                print(f"vm: masked", end=", ")
            else:
                print(f"vm: umasked", end=", ")
            # lumop/sumop/rs2/vs2
            rs2 = inst >> 20 & 0b11111

            if mop_mode == "unit-stride":
                if type == 0:
                    print(f"lumop(rs2): {rs2}", end=", ")
                else:
                    print(f"sumop(rs2): {rs2}", end=", ")
            elif mop_mode == "strided":
                print(f"rs2: {rs2}", end=", ")
            elif mop_mode == "indexed-unordered" or mop_mode == "indexed-ordered":
                print(f"vs2: {rs2}", end=", ")

            # rs1
            rs1 = inst >> 15 & 0b11111
            # width
            width = inst >> 12 & 0b111
            # vd
            vd = inst >> 7 & 0b11111

            print(f"nf: {nf}, mew: {mew}, rs1: {rs1}, width: {width}, vd: {vd}")
        
        # Arithmetic
        elif (inst & 0b1111111 == 0b1010111):
            print("Arithmetic")
            
            # funct6
            funct6 = inst >> 26 & 0b111111
            # 11.1. Vector Single-Width Integer Add and Subtract
            if   (funct6 == 0b000000): print("vadd")
            elif (funct6 == 0b000010): print("vsub")
            elif (funct6 == 0b000011): print("vrsub")

            # 11.2. Vector Widening Integer Add/Subtract

            # 11.3. Vector Integer Extension

            # vm
            vm = inst >> 25 & 0b1

            # vs2
            vs2 = inst >> 20 & 0b11111

            # vs1
            vs1 = inst >> 15 & 0b11111

            # funct3
            funct3 = inst >> 12 & 0b111
            if   (funct3 == 0b000): print("OPIVV")
            elif (funct3 == 0b001): print("OPFVV")
            elif (funct3 == 0b010): print("OPMVV")
            elif (funct3 == 0b011): print("OPIVI")
            elif (funct3 == 0b100): print("OPIVX")
            elif (funct3 == 0b101): print("OPFVF")
            elif (funct3 == 0b110): print("OPMVX")
            elif (funct3 == 0b111): print("OPCFG")

            # vd
            vd = inst >> 7 & 0b11111

        else:
            print("Error: not suppport insturction")

    def set_vtype(self, vlmul, vsew, vta, vma):
        """
        this function is used to set the vtype follow rvv definition
        vlmul & vsew is in true value domain, not rvv encode domain
        TODO the RVV domain is useless currently, maybe "self.vtype" memorize the true value is enough
        """
        
        # set vlmul
        enc_vlmul = self.vlmul_reverse[vlmul]
        if enc_vlmul == 4:
            self.LMUL = 1
            self.debug and print("Error, RVV vlmul 4 is reserved")
        else:
            self.LMUL = vlmul # with default value 1
            self.debug and print(f"LMUL = {self.LMUL}", end=", ") # TODO print faction, not float

        # set vsew
        enc_vsew = self.vlmul_reverse[vsew]
        if enc_vsew > 3:
            self.SEW = 8
            self.debug and print("Error, RVV vsew > 3 is reserved")
        else:
            self.SEW = vsew # with default value 8
            self.debug and print(f"SEW = {self.SEW}" , end=", ")

        # vta
        self.vta = vta
        if vta == 1:
            self.debug and print("vta: agnostic" , end=", ")
        else:
            self.debug and print("vta: undisturbed" , end=", ")
            
        # vma
        self.vma = vma
        if vma == 1:
            self.debug and print("vma: agnostic" , end=", ")
        else:
            self.debug and print("vma: undisturbed" , end=", ")

        self.debug and print(f"VLMAX: {self.VLMAX}")
        # set vtype
        self.vtype = enc_vlmul << 0 | enc_vsew << 3 | vta << 6 | vma << 7
        return self.vtype

if __name__ == "__main__":
    print("===== dispatcher testbench =====")
    print("version: 2025.05.15")

    avl = 120
    
    dispatcher = DISPATCHER(debug=True)

    # set vtype
    # for vlmul in range(0, 8):
    #     for vsew in range(0, 4):
    #         vtype = dispatcher.set_vtype(vlmul, vsew, 0, 1)
    #         print(f"vtype: {vtype:08b}")
    #     print()
    

    # dispatcher.decode("vsetvli rd, rs1, vtypei")
    # dispatcher.decode("vadd v0, v1, v2")
    
    """
    Test the decodeCAPI function
    """
    print("Case 1")
    dispatcher.decodeCAPI('VSET(160, e8, m8);', [160, 8, 8])
    print(f"vtype: {dispatcher.vtype:X}")

    print("Case 2")
    dispatcher.decodeCAPI('write_csr(vstart, 8);', [8])
    print(f"vstart: {dispatcher.vstart}")

    print("Case 3")
    dispatcher.decodeCAPI('asm volatile("vle8.v v2, (%0)" ::"r"((uint8_t*)3758096384));', [8, 2, 3758096384])
    print(f"sew: {dispatcher.SEW},  vd: {dispatcher.vd},  addr: {dispatcher.scalar_imm:8X}")


    """
    Used to check the binary decode result
    """
    # vsetvli	t0, a0, e8, m1, ta, ma
    # print("\nCase 1: ", end="")
    # inst = 0x0c0572d7
    # dispatcher.Bin2Asm(inst)

    # # vsetvl	a2, zero, a0
    # print("\nCase 2: ", end="")
    # inst = 0x80a07657
    # dispatcher.Bin2Asm(inst)

    # # vsetivli	a1, 0xc, e8, m1, ta, ma
    # print("\nCase 3: ", end="")
    # inst = 0xcc0675d7
    # dispatcher.Bin2Asm(inst)

    # # vsetivli	a1, 0xc, e8, m1, ta, mu
    # print("\nCase 4: ", end="")
    # inst = 0xc40675d7
    # dispatcher.Bin2Asm(inst)

    # vle8.v	v0, (a0)
    # print("\nCase 5: ", end="")
    # inst = 0x02050007
    # dispatcher.Bin2Asm(inst)

    # # vle8.v	v3, (a0), v0.t
    # print("\nCase 6: ", end="")
    # inst = 0x00050187
    # dispatcher.Bin2Asm(inst)

    # # vse8.v	v3, (a0)
    # print("\nCase 7: ", end="")
    # inst = 0x020501a7
    # dispatcher.Bin2Asm(inst)

    # # vlse8.v	v1, (a0), a1
    # print("\nCase 8: ", end="")
    # inst = 0x0ab50087
    # dispatcher.Bin2Asm(inst)

    # # vlse8.v	v1, (a0), a1, v0.t
    # print("\nCase 9: ", end="")
    # inst = 0x08b50087
    # dispatcher.Bin2Asm(inst)

    # # vluxei8.v	v1, (a0), v2
    # print("\nCase 10: ", end="")
    # inst = 0x06250087
    # dispatcher.Bin2Asm(inst)

    # # vluxei8.v	v1, (a0), v2, v0.t
    # print("\nCase 11: ", end="")
    # inst = 0x04250087
    # dispatcher.Bin2Asm(inst)

    # # vle8ff.v	v1, (a0), v0.t
    # print("\nCase 12: ", end="")
    # inst = 0x01050087
    # dispatcher.Bin2Asm(inst)