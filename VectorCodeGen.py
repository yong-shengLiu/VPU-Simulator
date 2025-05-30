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
            ('vv', ['vadd', 'vredsum', 'vfadd', 'vsub', 'vredor', 'vfsub', 'vredxor', 'vfredosum', 'vminu', 'vredminu', 'vfmin', 'vmin', 'vredmin', 'vfredmin', 'vmaxu', 'vfmax', 'vmax', 'vredmax', 'vfredmax', 'vand', 'vaadd', 'vfsgnjn', 'vor', 'vasubu', 'vfsgnjx', 'vxor', 'vasub', 'vrgather', 'vrgatherei16', 'vadc', 'VWXUNARY0', 'VWFUNARY0', 'vmadc', 'vsbc', 'VXUNARY0', 'VFUNARY0', 'vmsbc', 'vmerge/vmv', 'vcompress', 'vmseq', 'vmandnot', 'vmfeq', 'vmsne', 'vmand', 'vmfle', 'vmsltu', 'vmor', 'vmslt', 'vmxor', 'vmflt', 'vmsleu', 'vmornot', 'vmfne', 'vmsle', 'vmnand', 'vmnor', 'vmxnor', 'vsaddu', 'vdivu', 'vfdiv', 'vsadd', 'vdiv', 'vssubu', 'vremu', 'vssub', 'vrem', 'vsll', 'vmul', 'vsmul', 'vmulh', 'vsrl', 'vsra', 'vmadd', 'vfnmadd', 'vssrl', 'vssra', 'vnmsub', 'vfnmsub', 'vnsra', 'vmacc', 'vfnmacc', 'vnclipu', 'vnclip', 'vnmsac', 'vfnmsac', 'vwredsumu', 'vwaddu', 'vfwadd', 'vwredsum', 'vwadd', 'vfwredusum']),
            ('vs', ['vredmaxu']),
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
            return lambda args: f'asm volatile("{opname}.{fmt} v{args[2]}, v{args[0]}, v{args[1]}");'
        elif fmt == 'vs':
            return lambda args: f'asm volatile("{opname}.{fmt} v{args[2]}, v{args[0]}, v{args[1]}");'
        elif fmt == 'vx':
            return lambda args: f'asm volatile("{opname}.{fmt} v{args[2]}, v{args[0]}, %[A]" :: [A] "r"({args[1]}));'
        elif fmt == 'vi':
            return lambda args: f'asm volatile("{opname}.{fmt} v{args[2]}, v{args[0]}, {args[1]}");'
        # TODO handle vf format
        elif fmt == 'vf':
            return lambda args: f'asm volatile("{opname}.{fmt} v{args[2]}, v{args[0]}, v{args[1]}");'
        elif fmt == 'wx':
            return lambda args: f'asm volatile("{opname}.{fmt} v{args[2]}, v{args[0]}, %[A]" :: [A] "r"({args[1]}));'
        else:
            raise ValueError(f"Unsupported format: {fmt}")

    def VectorCodeGen(self, inst_type, require_list):
        if inst_type not in self.handlers:
            raise ValueError(f"Unsupported instruction type: {inst_type}")
        return self.handlers[inst_type](require_list)


    # === Individual handler methods ===
    def _handle_vset(self, args):
        vl, sew, lmul = args
        return f'VSET({vl}, e{sew}, m{lmul});'

    def _handle_vstart(self, args):
        vstart = args[0]
        return f'write_csr(vstart, {vstart});'

    def _handle_vload_a(self, args):
        sew, vd, base_addr = args
        return f'asm volatile("vle{sew}.v v{vd}, (%0)" :: "r"((uint{sew}_t*){base_addr}));'

    def _handle_vstore_a(self, args):
        sew, vs, base_addr = args
        return f'asm volatile("vse{sew}.v v{vs}, (%0)" :: "r"((uint{sew}_t*){base_addr}));'

if __name__ == "__main__":
    print("=== VectorCodeGen testbench ===")
    print("version: 2025.05.29")

    codegen = VectorCodeGenerator()  # Initialize the VectorCodeGen class
    
    print(codegen.VectorCodeGen('vset', [128, 8, 1]))
    print(codegen.VectorCodeGen('vadd_vv', [1, 2, 3]))
    print(codegen.VectorCodeGen('vor_vv', [1, 2, 3]))
    print(codegen.VectorCodeGen('vsrl_vi', [1, 2, 3]))
    print(codegen.VectorCodeGen('vredmaxu_vs', [1, 2, 3]))