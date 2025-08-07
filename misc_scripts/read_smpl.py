# THis is a script to see the contents of a `.smpl` file

import numpy as np

# Load the npz file
data = np.load("data/smplx/squat_ow.smpl")
# data = np.load('Dance_03_w_hands.smpl')
# data = np.load('avatar_face_expression.smpl')

# Display the keys (names of arrays stored in the npz file)
print("Keys in the npz file:", data.files)

# Display the values for each key
for key in data.files:
    print(f"\nKey: {key}")
    print(f"Value (shape: {data[key].shape}, dtype: {data[key].dtype}):")
    print(data[key])
