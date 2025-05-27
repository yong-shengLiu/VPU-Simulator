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

def Gen_File():
    """"
    This function is used to generate the file for VPU Main memory
    """

if __name__ == "__main__":
    print("=== Pattern Generator testbench ===")
    print("version: 2025.05.27")

    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(current_dir, "log", "Matrix.txt")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)   # create the output path


    bf16_mat = Gen_Matrix(64, 512, "BF16")
    with open(output_path, "w", encoding="utf-8") as f:
        with redirect_stdout(f):
            np.set_printoptions(threshold=np.inf)  # Prevent truncation
            print(f"Generated BF16 Matrix:\n{bf16_mat}")