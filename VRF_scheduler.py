class VRFScheduler:
    def __init__(self, num_registers=32, reg_length=4096):
        self.num_registers = num_registers
        self.reg_length = reg_length
        self.vrf = [None] * num_registers  # None = free
        self.allocations = {}  # op_id -> list of reg indices

    def allocate(self, op_id, required_regs=1):
        free_regs = [i for i, val in enumerate(self.vrf) if val is None]

        # Allocation failed
        if len(free_regs) < required_regs:
            raise RuntimeError(f"[VRF ERROR] Not enough space to allocate {required_regs} registers for '{op_id}'")
        
        allocated = free_regs[:required_regs]

        for idx in allocated:
            self.vrf[idx] = op_id
        self.allocations[op_id] = allocated

        return allocated

    def free(self, op_id):
        if op_id not in self.allocations:
            raise RuntimeError(f"[VRF ERROR] Not found the space for freeing operation '{op_id}'")
        
        for idx in self.allocations[op_id]:
            self.vrf[idx] = None
        del self.allocations[op_id]

    def status(self):
        print("VRF Status:")
        for i, val in enumerate(self.vrf):
            print(f"V{i:02d}: {'Free' if val is None else val}")


if __name__ == "__main__":
    sched = VRFScheduler()

    sched.allocate("mask", 1)   # mask
    sched.allocate("MaxExp", 1) # all zero

    total_row = 64
    row_exe = 22

    # find Exp. Maximum
    for start_row in range(0, total_row, row_exe):
        end_row = min(start_row + row_exe, total_row)

        for i in range(start_row, end_row):
            # load exponent
            sched.allocate(f"Exp{i}", 1)

            # find reduction max, store to scalar
            sched.allocate('temp', 1)
            sched.free('temp')
        
        for i in range(start_row, end_row):
            sched.free(f"Exp{i}")
    
    
    sched.free("MaxExp")


    row_exe = 13


    # Calculate the Exp different and shift Mant
    for start_row in range(0, total_row, row_exe):
        end_row = min(start_row + row_exe, total_row)
        
        # mask sliding
        for i in range(start_row, end_row):
            # load exponent
            sched.allocate(f"Exp{i}", 1)
            
            # different
            sched.allocate(f"diff{i}", 1)

        
        for i in range(start_row, end_row):
            sched.free(f"Exp{i}")
            # load Mant. & shift
            sched.allocate(f"Mant{i}", 1)

        sched.status()
        for i in range(start_row, end_row):
            sched.free(f"diff{i}")
            sched.free(f"Mant{i}")
        
    
    sched.free("mask")
    

    