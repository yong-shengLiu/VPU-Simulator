class VectorCodeGenerator:
    def __init__(self):
        self.handlers = {
            'vset':        self._handle_vset,
            'vstart':      self._handle_vstart,
            'vload_a':     self._handle_vload_a,
            'vstore_a':    self._handle_vstore_a,
        }

        # Group vv/vx/vi/vf format instructions dynamically
        self._format_instruction_list  = ([
            ('vv', ['vexp', 'vadd', 'vredsum', 'vfadd', 'vsub', 'vredor', 'vfsub', 'vredxor', 'vfredosum', 'vminu', 'vredminu', 'vfmin', 'vmin', 'vredmin', 'vfredmin', 'vmaxu', 'vfmax', 'vmax', 'vredmax', 'vfredmax', 'vand', 'vaadd', 'vfsgnjn', 'vor', 'vasubu', 'vfsgnjx', 'vxor', 'vasub', 'vrgather', 'vrgatherei16', 'vadc', 'VWXUNARY0', 'VWFUNARY0', 'vmadc', 'vsbc', 'VXUNARY0', 'VFUNARY0', 'vmsbc', 'vmerge/vmv', 'vcompress', 'vmseq', 'vmandnot', 'vmfeq', 'vmsne', 'vmand', 'vmfle', 'vmsltu', 'vmor', 'vmslt', 'vmxor', 'vmflt', 'vmsleu', 'vmornot', 'vmfne', 'vmsle', 'vmnand', 'vmnor', 'vmxnor', 'vsaddu', 'vdivu', 'vfdiv', 'vsadd', 'vdiv', 'vssubu', 'vremu', 'vssub', 'vrem', 'vsll', 'vmul', 'vsmul', 'vmulh', 'vsrl', 'vsra', 'vmadd', 'vfnmadd', 'vssrl', 'vssra', 'vnmsub', 'vfnmsub', 'vnsra', 'vmacc', 'vfnmacc', 'vnclipu', 'vnclip', 'vnmsac', 'vfnmsac', 'vwredsumu', 'vwaddu', 'vfwadd', 'vwredsum', 'vwadd', 'vfwredusum']),
            ('vs', ['vredmaxu', 'vredmax', 'vredsum']),
            ('v.v', ['vmv']),
            ('v.x', ['vmv']),
            ('x.s', ['vmv']),
            ('vx', ['vadd', 'vsub', 'vrsub', 'vminu', 'vmin', 'vmaxu', 'vmax', 'vand', 'vaadd', 'vor', 'vasubu', 'vxor', 'vasub', 'vrgather', 'vslideup', 'vslide1up', 'vslidedown', 'vslide1down', 'vadc', 'VRXUNARY0', 'vmadc', 'vsbc', 'vmsbc', 'vmerge/vmv', 'vmseq', 'vmsne', 'vmsltu', 'vmslt', 'vmsleu', 'vmsle', 'vmsgtu', 'vmsgt', 'vsaddu', 'vdivu', 'vsadd', 'vdiv', 'vssubu', 'vremu', 'vssub', 'vrem', 'vsll', 'vmul', 'vsmul', 'vmulh', 'vsrl', 'vsra', 'vmadd', 'vssrl', 'vssra', 'vnmsub', 'vnsra', 'vmacc', 'vnclipu', 'vnclip', 'vnmsac', 'vwaddu', 'vwadd']),
            ('vi', ['vadd', 'vrsub', 'vand', 'vor', 'vxor', 'vrgather', 'vslideup', 'vslidedown', 'vadc', 'vmadc', 'vmerge/vmv', 'vmseq', 'vmsne', 'vmsleu', 'vmsle', 'vmsgtu', 'vmsgt', 'vsaddu', 'vsadd', 'vsll', 'vsrl', 'vsra', 'vssrl', 'vssra', 'vnsra', 'vnclipu', 'vnclip']),
            ('vf', ['vfadd', 'vfsub', 'vfmin', 'vfmax', 'vfsgnjn', 'vfsgnjx', 'vfslide1up', 'vfslide1down', 'VRFUNARY0', 'vfmerge/vfmv', 'vmfeq', 'vmfle', 'vmflt', 'vmfne', 'vmfgt', 'vmfge', 'vfdiv', 'vfrdiv', 'vfrsub', 'vfnmadd', 'vfnmsub', 'vfnmacc', 'vfnmsac', 'vfwadd']),
            ('wx', ['vnsrl']),
        ])

        self._register_value_format_instructions(self._format_instruction_list )


    def _register_value_format_instructions(self, format_instr_list):
        for fmt, names in format_instr_list:
            for name in names:
                key = f'{name}_{fmt}'
                self.handlers[key] = self._make_format_handler(fmt, name)
    
    def _make_format_handler(self, fmt, opname):
        if fmt == 'vv':
            return lambda args: f'__asm__ volatile("{opname}.{fmt} v{args[2]}, v{args[0]}, v{args[1]}");'
        elif fmt == 'vs':
            return lambda args: f'__asm__ volatile("{opname}.{fmt} v{args[2]}, v{args[0]}, v{args[1]}");'
        elif fmt == 'v.v':
            return lambda args: f'__asm__ volatile("{opname}.v.v v{args[1]}, v{args[0]}");'
        elif fmt == 'v.x':
            return lambda args: f'__asm__ volatile("{opname}.v.x v{args[1]}, %[A]" ::[A] "r"({args[0]}));'
        elif fmt == 'x.s':
            return lambda args: f'__asm__ volatile("{opname}.x.s %0, v{args[0]}" : "=r"({args[1]}));'
        elif fmt == 'vx':
            return lambda args: f'__asm__ volatile("{opname}.{fmt} v{args[2]}, v{args[0]}, %[A]" :: [A] "r"({args[1]}));'
        elif fmt == 'vi':
            return lambda args: f'__asm__ volatile("{opname}.{fmt} v{args[2]}, v{args[0]}, {args[1]}");'
        # TODO handle vf format
        elif fmt == 'vf':
            return lambda args: f'__asm__ volatile("{opname}.{fmt} v{args[2]}, v{args[0]}, v{args[1]}");'
        elif fmt == 'wx':
            return lambda args: f'__asm__ volatile("{opname}.{fmt} v{args[2]}, v{args[0]}, %[A]" :: [A] "r"({args[1]}));'
        else:
            raise ValueError(f"Unsupported format: {fmt}")

    def VectorCodeGen(self, inst_type, require_list):
        if inst_type not in self.handlers:
            raise ValueError(f"Unsupported instruction type: {inst_type}")
        return self.handlers[inst_type](require_list)


    # === Individual handler methods ===
    def _handle_vset(self, args):
        vl, sew, lmul = args
        # return f'VSET({vl}, e{sew}, m{lmul});'
        return f'__asm__ volatile("vsetvli t0, %[A], e{sew}, m{lmul},ta,ma" ::[A] "r"({vl}));'
    

    def _handle_vstart(self, args):
        vstart = args[0]
        return f'write_csr(vstart, {vstart});'

    def _handle_vload_a(self, args):
        sew, vd, base_addr = args
        return f'__asm__ volatile("vle{sew}.v v{vd}, (%0)" :: "r"((uint{sew}_t*)(uintptr_t){base_addr}));'

    def _handle_vstore_a(self, args):
        sew, vs, base_addr = args
        return f'__asm__ volatile("vse{sew}.v v{vs}, (%0)" :: "r"((uint{sew}_t*)(uintptr_t){base_addr}));'

if __name__ == "__main__":
    print("=== VectorCodeGen testbench ===")
    print("version: 2025.11.05")

    codegen = VectorCodeGenerator()  # Initialize the VectorCodeGen class
    
    # print(codegen.VectorCodeGen('vset', [128, 8, 1]))
    # print(codegen.VectorCodeGen('vadd_vv', [1, 2, 3]))
    # print(codegen.VectorCodeGen('vor_vv', [1, 2, 3]))
    # print(codegen.VectorCodeGen('vsrl_vi', [1, 2, 3]))
    # print(codegen.VectorCodeGen('vredmaxu_vs', [1, 2, 3]))

    print(codegen.VectorCodeGen('vset', [256, 16, 1]))            # vl, sew, lmul
    print(codegen.VectorCodeGen('vmv_v.x', [0, 0]))               # scalar v
    print(codegen.VectorCodeGen('vload_a', [16, 1, 0xe0000000]))  # sew, vd, base_addr
    print(codegen.VectorCodeGen('vredmax_vs', [1, 0, 2]))         # vs1, vs2, vd
    print(codegen.VectorCodeGen('vmv_x.s', [2, "maximum"]))       # v  scalar

    print(codegen.VectorCodeGen('vsub_vx', [1, "maximum", 2]))    # vs1, scalar, vd

    print(codegen.VectorCodeGen('vexp_vv', [2, 2, 2]))            # vs1, vs2, vd

    print(codegen.VectorCodeGen('vredsum_vs', [2, 0, 1]))         # vs1, vs2, vd
    print(codegen.VectorCodeGen('vmv_x.s', [1, "summation"]))     # v  scalar

    print(codegen.VectorCodeGen('vmul_vx', [1, "reciprocal", 2]))  # vs1, scalar, vd
    print(codegen.VectorCodeGen('vsrl_vx', [1, 8, 2]))  # vs1, scalar, vd
    print(codegen.VectorCodeGen('vstore_a', [16, 2, 0xe0010000])) # sew, vs, base_addr






    # vsetvli	t0, a0, e16, m1, ta, ma
    # vmv.v.x	v0, s0
    # vle16.v	v1, (s2)                    TODO: need to revise !!!
    # vredmax.vs	v2, v1, v0
    # vmv.x.s	a1, v2

    # vsub.vx	v2, v1, a1                  TODO: need to revise !!!
    # vexp.vv	v2, v2, v2

    # vredsum.vs	v1, v2, v0
    # vmv.x.s	a1, v1
    # vmul.vx	v1, v2, a1              TODO: need to revise !!!
    # vsrl.vx	v2, v1, a0
    # vse16.v	v2, (a0)                TODO: need to revise !!!