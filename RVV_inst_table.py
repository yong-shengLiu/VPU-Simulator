import os
from contextlib import redirect_stdout

v_list = []
x_list = []
i_list = []
f_list = []

def add_to_lists(labels, instr):
    if 'V' in labels: v_list.append(instr)
    if 'X' in labels: x_list.append(instr)
    if 'I' in labels: i_list.append(instr)
    if 'F' in labels: f_list.append(instr)

with open("RVV_funct6_table.txt", "r") as f:  # Replace with your actual file path
    for line in f:
        tokens = line.strip().split()
        i = 0
        while i + 2 < len(tokens):
            # pattern: [addr] [labels] [instr]
            addr = tokens[i]            # e.g., '000000'
            labels = tokens[i+1]        # e.g., 'VXI'
            instr = tokens[i+2]         # e.g., 'vadd'
            add_to_lists(labels, instr)
            i += 3

current_dir = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(current_dir, "log", "supoprt_list.txt")
with open(output_path, "w", encoding="utf-8") as f:
        with redirect_stdout(f):
            print("V:", v_list)
            print("X:", x_list)
            print("I:", i_list)
            print("F:", f_list)
