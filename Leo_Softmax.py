
def fixQ8_to_float(hex_list):
    """
    將十六進位字串列表 (Q8.8 兩補數) 轉為 float
    例如: ['0xfd42', '0x0000'] → [-2.742, 0.0]
    """
    int_list = []
    for v in hex_list:
        val = int(v, 16)
        if val >= 0x8000:  # 處理 two's complement
            val -= 0x10000
        int_list.append(val)

    x = np.array(int_list, dtype=np.int16)
    return x.astype(np.float32) / 256.0  # Q8.8 → float

def exp_quantized(x_q8):
    """
    Quantized exponential approximation for one Q8.8 input.
    Input:  x_q8 (int or numpy int16) -- fixed-point Q8.8
    Output: exp(x) approximated in Q0.8 format (0~1 range)
    """

    # 1. log2(e) ≈ 1.5 → x * log2(e) ≈ x + x/2
    int_frac = x_q8 + (x_q8 >> 1)   # still in Q8.8
    # print(int_frac)

    # 2. split into integer and fractional parts
    integer_part = int_frac >> 8
    frac_part = int_frac - (integer_part << 8)  # still Q8.8, range [0..255]
    print(f"int: {integer_part}, frac: {frac_part}")

    # 3. compute 2^integer_part
    #    (1 << 8) represents "1.0 in Q0.8"
    if integer_part < 0:
        exp_int = (1 << 8) >> (-integer_part)
        print("<0")
    else:
        exp_int = (1 << 8) << integer_part
        print(">=0")

    # 4. approximate 2^fractional ≈ 1 + frac/2
    exp_frac = (1 << 8) + (frac_part >> 1)  # Q0.8 + Q0.8
    print(f"exp_int: {exp_int}, exp_frac: {exp_frac}")

    # 5. combine integer and fractional
    exp_out = (exp_int * exp_frac) >> 8  # back to Q0.8
    # print(f"exp_out: {exp_out}")

    return exp_out

def Softmax():
        print("Softmax is under development ...")

        # Parameters
        vec_len  = 8       # 128 elements
        ele_bit  = 16         # Q88


        # Generate random input vector
        np.random.seed(0)
        vec_float = np.random.uniform(-4, 4, vec_len)
        vec_q8 = np.round(vec_float * 256).astype(np.int16)  # float to Q8.8
        print(f"Input Vector (float):\n {vec_float}")
        
        # Print 8 elements per line
        print(f"Input Vector (Q8.8):")
        vec_hex = [f"0x{v & 0xFFFF:04x}" for v in vec_q8]
        for i in range(0, vec_len, 8):
            line = ", ".join(vec_hex[i:i+8])
            print(f"  {line},")
        
        # reduction maximum
        max_val = np.max(vec_q8)
        print(f"\nMax Value: {max_val} (0x{max_val:04x})")


        # element-wise subtraction
        diff = (vec_q8 - max_val).astype(np.int16)

        diff_hex = [f"0x{v & 0xFFFF:04x}" for v in diff]
        print("\nAfter Subtraction:")
        for i in range(0, vec_len, 8):
            line = ", ".join(f"0x{int(val, 16) & 0xFFFF:04x}" for val in diff_hex[i:i+8])
            print(line + ",")

        # element-wise exponentiation
        exp_result = np.array([exp_quantized(v) for v in diff], dtype=np.uint16)

        exp_hex = [f"0x{v & 0xFFFF:04x}" for v in exp_result]
        print("\nAfter Exponentiation:")
        for i in range(0, vec_len, 8):
            line = ", ".join(exp_hex[i:i+8])
            print(f"  {line},")
        for i in range(0, vec_len, 8):
            print(fixQ8_to_float(exp_hex))


        # reduction summation
        sum_exp = np.sum(exp_result, dtype=np.int32)
        print(f"\nSum of Exponentials: {sum_exp} (0x{sum_exp:08x})")
        
        # element-wise reciprocal
        reciprocal_q8 = np.round((1 << 16) / sum_exp).astype(np.int32)  # approximate reciprocal in Q8.8
        print(f"Reciprocal (Q8.8): {reciprocal_q8} (0x{reciprocal_q8:08x})")
        softmax_q8 = (exp_result * reciprocal_q8) >> 8

        print("\nFinal Softmax (Q0.8):")
        softmax_hex = [f"0x{v & 0xFFFF:04x}" for v in softmax_q8]
        for i in range(0, vec_len, 8):
            line = ", ".join(softmax_hex[i:i+8])
            print(f"  {line},")
        for i in range(0, vec_len, 8):
            print(fixQ8_to_float(softmax_hex))

        print(f"\nSum check: {np.sum(softmax_q8)/256:.4f} (should ≈ 1.0)")