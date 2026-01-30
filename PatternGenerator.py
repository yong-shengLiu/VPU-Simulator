import os
from contextlib import redirect_stdout
import numpy as np
import struct

pattern_seed = np.random.default_rng(seed=0)   # Fixed seed for pattern generation

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

def float32_to_bf16(float32_val, debug=False):
    """Convert a single float32 value to its BF16 bit representation."""
    # === Pack float32 ===
    packed = struct.pack('>f', float32_val)          # 4 bytes
    float_bits = struct.unpack('>I', packed)[0]      # Convert to 32-bit int
    sign     = (float_bits >> 31) & 0x1
    exponent = (float_bits >> 23) & 0xFF
    mantissa = float_bits         & 0x7FFFFF
    debug and print(f"FP32: {float32_val}, Packed-> Sign: {sign}, Exp: {exponent}, Mant: {mantissa:023b}, Converted->{(-1)**sign * 2**(exponent - 127) * (1 + mantissa / 2**23)}")

    # === Pack BF16 ===
    bf16_bits = float_bits >> 16  # Keep the top 16 bits
    sign     = (bf16_bits >> 15) & 0x1
    exponent = (bf16_bits >> 7)  & 0xFF
    mantissa = bf16_bits         & 0x7F
    print(f"BF16: {bf16_bits}, Packed-> Sign: {sign}, Exp: {exponent}, Mant: {mantissa:023b}, Converted->{(-1)**sign * 2**(exponent - 127) * (1 + mantissa / 2**7)}")

    return bf16_bits

def Gen_Matrix(row, column, datatype):
    """
    Generate a matrix with given row and column size.
    datatype: (1) BF16
    Returns: numpy array of uint16 BF16 representations.
    """
    if datatype == "BF16":
        # Generate random float32 numbers in a reasonable range
        np.random.seed(42)  # Set a fixed seed
        float_matrix = np.random.uniform(low=-1.0, high=1.0, size=(row, column)).astype(np.float32)
        
        # Convert each float32 to BF16 (uint16 values)
        bf16_matrix = np.vectorize(float32_to_bf16)(float_matrix).astype(np.uint16)
        
        return bf16_matrix
    else:
        raise ValueError(f"Unsupported datatype: {datatype}")

def Gen_golden(element_array, kernal_size, debug=False):
    H, W   = element_array.shape
    kH, kW = kernal_size

    aligned_mantissas_list = []
    Block_Max_exp = []
    exp_array_temp = []
    mantissa_array_temp = []
    diff_array_temp = []

    for row in range(0, H, kH):
        for col in range(0, W, kW):
            
            # === Extract the tile ===
            tile = element_array[row:row+kH, col:col+kW]

            # === Seperate to exp and mantissa ===
            exp_array      = np.zeros((kH, kW), dtype=np.uint8)
            mantissa_array = np.zeros((kH, kW), dtype=np.uint8)
            for r in range(kH):
                for c in range(kW):
                    element = tile[r, c] # element in row-major order

                    # === Seperate to exp and mantissa ===
                    exponent = (element >> 7) & 0xFF
                    mant_plus = (element >> 8 & 0x80) | 0x40 | (element >> 1 & 0x3F) # {sign, 1, mantissa[6:1]}
                    sign = (element >> 15) & 0x1
                    debug and print(f"Element: {hex(element)}, Exponent: {exponent}, Mant_plus: {mant_plus:08b}, Sign: {sign}")
                    exp_array[r, c]      = exponent
                    mantissa_array[r, c] = mant_plus

            exp_array_temp.append(exp_array)
            mantissa_array_temp.append(mantissa_array)

            # === find the block maxmium exponent ===
            block_max_exp = np.max(exp_array)
            debug and print(f"Block max exponent: {block_max_exp}")
            Block_Max_exp.append(block_max_exp)

            # === Calculate the different between the block maxmium ===
            shift_array = block_max_exp - exp_array
            diff_array_temp.append(shift_array)
            debug and print(f"Shift array:\n{shift_array}")

            # === shift the mantissa ===
            mantissa_array = mantissa_array.astype(np.int8)
            aligned_mantissas = mantissa_array >> shift_array  # for signed right shift
            mantissa_array = mantissa_array.astype(np.uint8)
            aligned_mantissas = aligned_mantissas.astype(np.uint8)
            debug and print(f"Aligned mantissas:\n{aligned_mantissas}")
            aligned_mantissas_list.append(aligned_mantissas)


    exp_array_temp = np.array(exp_array_temp)
    # print(f'exp_array_temp size: {exp_array_temp.shape}')

    diff_array_temp = np.array(diff_array_temp)
    # print(f'diff_array_temp size: {diff_array_temp.shape}')

    mantissa_array_temp = np.array(mantissa_array_temp)
    # print(f'mantissa_array_temp size: {mantissa_array_temp.shape}')

    aligned_mantissas_list = np.array(aligned_mantissas_list)
    # print(f'aligned_mantissas_list size: {aligned_mantissas_list.shape}')

    print(Block_Max_exp)

    return exp_array_temp, diff_array_temp, mantissa_array_temp, aligned_mantissas_list

def BlockScale():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(current_dir, "log", "Matrix.txt")
    pattern_path = os.path.join(current_dir, "pattern", "ExpMant_Mat64_512.npy")
    temp_golden = os.path.join(current_dir, "log", "temp_golden.txt")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)   # create the output path
    os.makedirs("pattern", exist_ok=True)


    # bf16_mat = Gen_Matrix(64, 512, "BF16")
    bf16_mat = Gen_Matrix(64, 512, "BF16")
    kernal_size = (64, 64)
    exp, diff, mant, aligned_mant = Gen_golden(bf16_mat, kernal_size)

    # First, transpose to (64, 8, 64), then reshape to (64, 512)
    exp_reshaped = exp.transpose(1, 0, 2).reshape(64, 512).flatten()
    diff_reshaped = diff.transpose(1, 0, 2).reshape(64, 512).flatten()
    mant_reshaped = mant.transpose(1, 0, 2).reshape(64, 512).flatten()
    aligned_mant_reshaped = aligned_mant.transpose(1, 0, 2).reshape(64, 512).flatten()

    print(f'exp size: {exp_reshaped.shape}')
    print(f'exp type: {exp_reshaped.dtype}')
    print(f'mant size: {mant_reshaped.shape}')
    print(f'aligned_mant size: {aligned_mant_reshaped.shape}')


    
    with open(temp_golden, "w", encoding="utf-8") as f:
        with redirect_stdout(f):
            print("Expont")
            for idx, element in enumerate(exp_reshaped):
                print(f'{element:02x}', end=" ")

                if (idx + 1) % 8 == 0:
                    print()

            print("Mant")
            for idx, element in enumerate(mant_reshaped):
                print(f'{element:02x}', end=" ")
                
                if (idx + 1) % 8 == 0:
                    print()
            
            print("diff Exp")
            for idx, element in enumerate(diff_reshaped):
                print(f'{element:02x}', end=" ")
                
                if (idx + 1) % 8 == 0:
                    print()


            # print("Shift Mant")
            # for idx, element in enumerate(aligned_mant_reshaped):
            #     print(f'{element:02x}', end=" ")
                
            #     if (idx + 1) % 8 == 0:
            #         print()

    zero = np.zeros((65, 512), dtype=np.uint8)
    zero_flatten = zero.flatten()

    # mask sliding
    width = 64  # bit width of the sliding mask
    widthB = width // 8  # convert bit width to byte width
    num_masks = 512 // width  # number of masks to slide through 512 bytes

    # Create a list to hold all the masks
    mask_array = []

    for i in range(num_masks):
        mask = np.zeros(512, dtype=np.uint8)
        mask[i * widthB : (i + 1) * widthB] = 0xFF
        mask_array.append(mask)

    # Stack into a 2D array if needed
    mask_array = np.stack(mask_array)
    mask_flat_array = mask_array.flatten()


    combined = np.concatenate((exp_reshaped, mant_reshaped, zero_flatten, mask_flat_array), axis=0)
    # Save as .npy file
    np.save(pattern_path, combined)

    # with open(output_path, "w", encoding="utf-8") as f:
    #     with redirect_stdout(f):
    #         np.set_printoptions(threshold=np.inf)  # Prevent truncation
    #         hex_formatter = np.vectorize(lambda x: hex(x))
    #         print(f"Generated BF16 Matrix:\n{hex_formatter(bf16_flat)}")

    np.set_printoptions(threshold=np.inf, linewidth=200)
    print(mask_array)

def FastInverse_rsqrt(number, iteration):
    threehalfs = 1.5
    x2 = number * 0.5
    y = number


    # evil floating point bit level hacking
    i = struct.unpack('I', struct.pack('f', y))[0]
    i = 0x5f3759df - (i >> 1)
    y = struct.unpack('f', struct.pack('I', i))[0]


    for _ in range(iteration):
        y = y * (threehalfs - (x2 * y * y))

    result_bits = struct.unpack('I', struct.pack('f', y))[0]
    size = struct.calcsize('I')


    if result_bits < 0 or result_bits >= (1 << (size * 8)):
        raise ValueError('result_bits out of range')

    return struct.unpack('f', struct.pack('I', result_bits))[0]

def exp_quantized(x_q8):
    """
    Quantized exponential approximation for one Q8.8 input.
    Input:  x_q8 (int or numpy int16) -- fixed-point Q8.8
    Output: exp(x) approximated in Q0.8 format (0~1 range)
    """

    # 1. log2(e) ≈ 1.5 → x * log2(e) ≈ x + x/2
    int_frac = x_q8 + (x_q8 >> 1)   # still in Q8.8
    print(int_frac)

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
        vec_len  = 256       # 128 elements
        ele_bit  = 16         # Q88


        # Generate random input vector
        # np.random.seed(0)
        vec_float = pattern_seed.uniform(-4, 4, vec_len)
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

def LayerNorm():
    """"""

def GELU():
    """"""

def to_signed16(x):
    return x - 0x10000 if x & 0x8000 else x

def imatmul(M, N, P):
    """
    C = AB with A=[MxN], B=[NxP], C=[MxP]
    """

    dtype = np.uint8
    UPPER_LIMIT = 10000
    LOWER_LIMIT = -10000

    # Matrices and results
    A = np.random.randint(LOWER_LIMIT, UPPER_LIMIT, size=(M, N)).astype(dtype)
    B = np.random.randint(LOWER_LIMIT, UPPER_LIMIT, size=(N, P)).astype(dtype)
    C = np.zeros([M, P], dtype=dtype)
    # Golden result matrix
    G = np.matmul(A, B).astype(dtype)

    print(f"A:\n{A}")
    print(f"B:\n{B}")
    print(f"G:\n{G}")
    

if __name__ == "__main__":
    print("=== Pattern Generator testbench ===")
    print("version: 2026.01.26")

    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(current_dir, "log", "Pattern.txt")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)   # create the output path


    with open(output_path, "w", encoding="utf-8") as f:
        with redirect_stdout(f):
            imatmul(4, 5, 4)
            # Softmax()
            # Softmax()

    # n = 256
    # f = FastInverse_rsqrt(n, 1)
    # print(f"inverse_rsqrt: {f}")
    # print(f"inverse_rsqrt: {1/np.sqrt(n)}")

    # print(np.log2(np.e).dtype)


    # fp32 = -3.8

    # integer = 3
    # fract   = 0.8

    # bf16 = float32_to_bf16(fp32)
    # print(bf16)

    # fixed_point = int((1 - fract) * 256) >> 1
    # result = fixed_point >> abs(integer)

    # print(result)

    # print(exp_quantized(to_signed16(0xfc66)))
    # print(exp_quantized(to_signed16(0xffb8)))
    # print(exp_quantized(to_signed16(0xfb66)))
    # print(exp_quantized(to_signed16(0xfc8d)))
    