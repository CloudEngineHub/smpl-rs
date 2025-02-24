# This is a script to investigate the contents of a SMPL data `pkl`

import pickle
import numpy as np
from scipy.sparse import coo_matrix

pkl_file_path = "male.pkl"

with open(pkl_file_path, "rb") as file:
    data = pickle.load(file)

if isinstance(data, dict):
    for key, value in data.items():
        if isinstance(value, np.ndarray):
            print(f"Key: {key}")
            print(f"Shape: {value.shape}")
            print(f"Type: {value.dtype}\n")
        elif isinstance(value, coo_matrix):
            print(f"Key: {key} (Sparse Matrix)")
            print(f"Shape: {value.shape}")
            print(f"Data type: {value.dtype}\n")
        elif isinstance(value, (int, float, str)):
            print(f"Key: {key}")
            print(f"Value: {value} (Type: {type(value).__name__})\n")
        elif isinstance(value, list):
            # if len(value) <= 10:
            print(f"Key: {key} (List)")
            print(f"Value: {value}")
            # else:
            #     print(f"Key: {key} (List with {len(value)} elements)")
            #     print(f"First 5 elements: {value[:5]}")
            print()
        else:
            # For other types, just print the type
            print(f"Key: {key} (Type: {type(value).__name__})")
            print(f"Content: {value}\n")
else:
    print("The loaded data is not a dictionary. Here is the content:")
    print(data)
