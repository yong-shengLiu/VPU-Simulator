class Scalar_rf:
    def __init__(self, num_registers=32, reg_length=32):
        self.num_registers = num_registers
        self.reg_length = reg_length
        self.xrf = [None] * num_registers  # None = free, used for schedualing
        self.values = ['x'] * num_registers  # None = free, used for store value in runtime
        self.allocations = {}  # op_id -> list of reg indices

        self.xrf[0]    = 'x0'  # x0 always be zero
        self.values[0] = 0     # x0 always be zero

    """ Scalar regfile schedualing """
    def allocate(self, op_id, required_regs=1):
        free_regs = [i for i, val in enumerate(self.xrf) if val is None]

        # Allocation failed
        if len(free_regs) < required_regs:
            raise RuntimeError(f"[XRF ERROR] Not enough space to allocate {required_regs} registers for '{op_id}'")
        
        allocated = free_regs[:required_regs]

        for idx in allocated:
            self.xrf[idx] = op_id
        self.allocations[op_id] = allocated

        return allocated

    def peek(self, op_id):
        return self.allocations.get(op_id, [])

    def free(self, op_id):
        if op_id not in self.allocations:
            raise RuntimeError(f"[XRF ERROR] Not found the space for freeing operation '{op_id}'")
        
        for idx in self.allocations[op_id]:
            self.xrf[idx] = None
        del self.allocations[op_id]

    def status(self):
        print("XRF Status:")
        for i, val in enumerate(self.xrf):
            print(f"V{i:02d}: {'Free' if val is None else val}")


    """ Scalar ALU function """
    def write(self, reg_idx, value):
        
        if reg_idx == 0:
            raise RuntimeError(f'[XRF ERROR] Cannot write the value to x0')

        self.values[reg_idx] = value

    def read(self, reg_idx):
        rtn = self.values[reg_idx]

        if rtn == 'x':
            raise RuntimeError(f'[XRF ERROR] The scalar regfile got an unknown value [reg_idx]: {rtn}')
        return rtn

    def add(self, dst, src1, src2):
        self.values[dst] = self.values[src1] + self.values[src2]

    def shift_right(self, dst, src, shift_amount):
        self.values[dst] = self.values[src] >> shift_amount

    def compare_eq(self, dst, src1, src2):
        self.values[dst] = int(self.values[src1] == self.values[src2])

if __name__ == "__main__":
    XRF = Scalar_rf()
    print("=== Scalar register file testbench ===")
    print("version: 2025.06.17")

    XRF.allocate('test', 3)

    print(XRF.peek('test'))

    XRF.write(1, 2)
    XRF.write(2, 3)
    
    XRF.add(3, 1, 2)
    
    print(f'Add Operation: {XRF.read(1)} + {XRF.read(2)} = {XRF.read(3)}')

    print(f'x0: {XRF.read(0)}')
    

    