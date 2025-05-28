class VectorCodeGenerator:
    def __init__(self):
        self.handlers = {
            'vset':        self._handle_vset,
            'vstart':      self._handle_vstart,
            'vload_a':     self._handle_vload_a,
            'vstore_a':    self._handle_vstore_a,
            'vand_vv':     self._handle_vand_vv,
            'vand_vx':     self._handle_vand_vx,
            'vsrl_vv':     self._handle_vsrl_vv,
            'vsrl_vx':     self._handle_vsrl_vx,
            'vsra_vv':     self._handle_vsra_vv,
            'vsra_vx':     self._handle_vsra_vx,
            'vsll_vv':     self._handle_vsll_vv,
            'vsll_vx':     self._handle_vsll_vx,
            'vor_vv':      self._handle_vor_vv,
            'vor_vx':      self._handle_vor_vx,
            'vsub_vv':     self._handle_vsub_vv,
            'vsub_vx':     self._handle_vsub_vx,
            'vslideup_vx': self._handle_vslideup_vx,
            'vredmaxu_vv': self._handle_vredmaxu,
            # Add more handlers here...
        }

    def VectorCodeGen(self, type, require_list):
        if type not in self.handlers:
            raise ValueError(f"Unsupported instruction type: {type}")
        return self.handlers[type](require_list)

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

    def _handle_vand_vv(self, args):
        vs1, vs2, vd = args
        return f'asm volatile("vand.vv v{vd}, v{vs1}, v{vs2}");'

    def _handle_vand_vx(self, args):
        vs, scalar, vd = args
        return f'asm volatile("vand.vx v{vd}, v{vs}, %[A]" :: [A] "r"({scalar}));'

    def _handle_vsrl_vv(self, args):
        vs1, vs2, vd = args
        return f'asm volatile("vsrl.vv v{vd}, v{vs1}, v{vs2}");'

    def _handle_vsrl_vx(self, args):
        vs, scalar, vd = args
        return f'asm volatile("vsrl.vx v{vd}, v{vs}, %[A]" :: [A] "r"({scalar}));'
    
    def _handle_vsra_vv(self, args):
        vs1, vs2, vd = args
        return f'asm volatile("vsra.vv v{vd}, v{vs1}, v{vs2}");'

    def _handle_vsra_vx(self, args):
        vs, scalar, vd = args
        return f'asm volatile("vsra.vx v{vd}, v{vs}, %[A]" :: [A] "r"({scalar}));'

    def _handle_vsll_vv(self, args):
        vs1, vs2, vd = args
        return f'asm volatile("vsll.vv v{vd}, v{vs1}, v{vs2}");'

    def _handle_vsll_vx(self, args):
        vs, scalar, vd = args
        return f'asm volatile("vsll.vx v{vd}, v{vs}, %[A]" :: [A] "r"({scalar}));'

    def _handle_vor_vv(self, args):
        vs1, vs2, vd = args
        return f'asm volatile("vor.vv v{vd}, v{vs1}, v{vs2}");'

    def _handle_vor_vx(self, args):
        vs, scalar, vd = args
        return f'asm volatile("vor.vx v{vd}, v{vs}, %[A]" :: [A] "r"({scalar}));'
    
    def _handle_vsub_vv(self, args):
        vs1, vs2, vd = args
        return f'asm volatile("vsub.vv v{vd}, v{vs1}, v{vs2}");'

    def _handle_vsub_vx(self, args):
        vs, scalar, vd = args
        return f'asm volatile("vsub.vx v{vd}, v{vs}, %[A]" :: [A] "r"({scalar}));'

    def _handle_vslideup_vx(self, args):
        vs, scalar, vd = args
        return f'asm volatile("vslideup.vi v{vd}, v{vs}, {scalar}");'
    
    def _handle_vredmaxu(self, args):
        vs1, vs2, vd = args
        return f'asm volatile("vredmaxu.vs {vd}, v{vs1}, v{vs2}");'
