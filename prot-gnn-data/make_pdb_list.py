# make_pdb_list.py
# Author  : Ethan Huang
# File    : make_pdb_list.py
# Time    : 2026/3/25

"""
Generate a list of absolute paths for all PDB files in a given directory
and save it to a specified output text file.

Usage examples:

python 2.make_pdb_list.py \
  -i /path/to/pdbs \
  -o /path/to/pdb_list.txt

"""

import os
import argparse
import sys

def generate_pdb_list(input_dir, output_file):
    # 1. Check if input directory exists
    if not os.path.exists(input_dir):
        print(f"Error: directory '{input_dir}' not found.")
        return

    # 2. Get absolute path (to avoid issues with relative paths)
    abs_input_dir = os.path.abspath(input_dir)
    print(f"Scanning directory: {abs_input_dir} ...")
    
    count = 0
    try:
        with open(output_file, 'w') as f:
            # 3. Iterate over the specified directory
            for filename in os.listdir(abs_input_dir):
                if filename.endswith(".pdb"):
                    # 4. Build full absolute path
                    full_path = os.path.join(abs_input_dir, filename)
                    f.write(full_path + "\n")
                    count += 1
    except Exception as e:
        print(f"Error writing file: {e}")
        return

    print(f"Done. {count} PDB file paths written to: {os.path.abspath(output_file)}")

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Generate a list of absolute paths for PDB files")
    
    # Input directory (required)
    parser.add_argument("--input_dir", "-i", type=str, required=True, 
                        help="Path to the folder containing .pdb files")
    
    # Output file (optional, defaults to pdb_list.txt)
    parser.add_argument("--output_file", "-o", type=str, default="pdb_list.txt", 
                        help="Path for the generated list file (default: pdb_list.txt)")
    
    args = parser.parse_args()
    
    generate_pdb_list(args.input_dir, args.output_file)