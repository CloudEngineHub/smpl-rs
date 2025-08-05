#!/usr/bin/env python3
"""
Script to read and display the contents of a .npz or .npy file
"""

import os
import numpy as np


def read_npz_or_npy(file_path):
    """Read and display contents of an npz or npy file"""
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} does not exist")
        return

    print(f"Reading file: {file_path}")
    print("=" * 50)

    data = np.load(file_path, allow_pickle=True)

    print("Keys in the npz file:", data.files)
    print("Number of arrays:", len(data.files))
    print("=" * 50)

    # Display the values for each key
    for key in data.files:
        print(f"\nKey: {key}")
        print(f"Shape: {data[key].shape}")
        print(f"Dtype: {data[key].dtype}")

        if data[key].dtype == "object":
            print(f"Data (object array):\n{data[key]}")
        else:
            print(f"Min: {data[key].min():.6f}, Max: {data[key].max():.6f}")

            if data[key].size <= 20:
                print(f"Data:\n{data[key]}")
            else:
                print(f"Data preview (first 10 elements):\n{data[key].flat[:10]}")

    data.close()


if __name__ == "__main__":
    read_npz_or_npy("data/smplx/SMPLX_NEUTRAL.npz")
