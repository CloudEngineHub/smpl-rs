# This is a simple script to convert the `pkl` data files to `npz`
import pickle
import numpy as np
from scipy.sparse import coo_matrix
from rich.progress import Progress

pkl_file_path = "male.pkl"
npz_file_path = "male.npz"

with open(pkl_file_path, "rb") as file:
    data = pickle.load(file)

dense_data = {}

with Progress() as progress:
    task = progress.add_task("[cyan]Converting data...", total=len(data))

    # Convert sparse matrices to dense and prepare the data
    for key, value in data.items():
        if isinstance(value, coo_matrix):
            # Convert sparse matrix to dense
            dense_data[key] = value.toarray()
        elif isinstance(value, np.ndarray):
            # Already dense
            dense_data[key] = value
        else:
            # For other data types, store them as they are
            dense_data[key] = value

        progress.update(task, advance=1)

np.savez_compressed(npz_file_path, **dense_data)

print(f"Data has been successfully converted and saved to {npz_file_path}")
