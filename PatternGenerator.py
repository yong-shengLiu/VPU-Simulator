import os
from contextlib import redirect_stdout
import numpy as np
import struct


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
    debug and print(f"BF16: {bf16_bits}, Packed-> Sign: {sign}, Exp: {exponent}, Mant: {mantissa:023b}, Converted->{(-1)**sign * 2**(exponent - 127) * (1 + mantissa / 2**7)}")

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

    mantissa_array_temp = np.array(mantissa_array_temp)
    # print(f'mantissa_array_temp size: {mantissa_array_temp.shape}')

    aligned_mantissas_list = np.array(aligned_mantissas_list)
    # print(f'aligned_mantissas_list size: {aligned_mantissas_list.shape}')



    # exp_array_temp_flat = np.concatenate([tile.flatten() for tile in exp_array_temp])
    # mantissa_array_temp_flat = np.concatenate([tile.flatten() for tile in mantissa_array_temp])
    # diff_array_temp_flat = np.concatenate([tile.flatten() for tile in diff_array_temp])
    print(Block_Max_exp)

    return exp_array_temp, mantissa_array_temp, aligned_mantissas_list


if __name__ == "__main__":
    print("=== Pattern Generator testbench ===")
    print("version: 2025.06.04")

    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(current_dir, "log", "Matrix.txt")
    pattern_path = os.path.join(current_dir, "pattern", "ExpMant_Mat64_512.npy")
    temp_golden = os.path.join(current_dir, "log", "temp_golden.txt")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)   # create the output path
    os.makedirs("pattern", exist_ok=True)


    # bf16_mat = Gen_Matrix(64, 512, "BF16")
    bf16_mat = Gen_Matrix(64, 512, "BF16")
    kernal_size = (64, 64)
    exp, mant, aligned_mant = Gen_golden(bf16_mat, kernal_size)

    # First, transpose to (64, 8, 64), then reshape to (64, 512)
    exp_reshaped = exp.transpose(1, 0, 2).reshape(64, 512).flatten()
    mant_reshaped = mant.transpose(1, 0, 2).reshape(64, 512).flatten()
    aligned_mant_reshaped = aligned_mant.transpose(1, 0, 2).reshape(64, 512).flatten()

    print(f'exp size: {exp_reshaped.shape}')
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
            
            print("Shift Mant")
            for idx, element in enumerate(aligned_mant_reshaped):
                print(f'{element:02x}', end=" ")
                
                if (idx + 1) % 8 == 0:
                    print()


    combined = np.concatenate((exp_reshaped, mant_reshaped), axis=0)
    # Save as .npy file
    np.save(pattern_path, combined)

    # with open(output_path, "w", encoding="utf-8") as f:
    #     with redirect_stdout(f):
    #         np.set_printoptions(threshold=np.inf)  # Prevent truncation
    #         hex_formatter = np.vectorize(lambda x: hex(x))
    #         print(f"Generated BF16 Matrix:\n{hex_formatter(bf16_flat)}")
    

    