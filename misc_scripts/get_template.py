# This is a simple script to export the template from the npz and visualize as an obj
 
import numpy as np

# Path to the uncompressed npz file
npz_file_path = "male_dense.npz"
obj_file_path_1 = "template.obj"
obj_file_path_2 = "skeleton.obj"
obj_file_path_3 = "skin.obj"

data = np.load(npz_file_path)

v_template = data['v_template']  # Shape: (N, 3)
quad_f = data['quad_f']  # Shape: (M, 3)

with open(obj_file_path_1, "w") as file:
    for vertex in v_template:
        file.write(f"v {vertex[0]} {vertex[1]} {vertex[2]}\n")
    
    for face in quad_f:
        file.write(f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n")

print(f"First mesh saved as {obj_file_path_1}")

skel_template_v = data['skel_template_v']  # Shape: (P, 3)
skel_template_f = data['skel_template_f']  # Shape: (Q, 3)

with open(obj_file_path_2, "w") as file:
    for vertex in skel_template_v:
        file.write(f"v {vertex[0]} {vertex[1]} {vertex[2]}\n")
    
    for face in skel_template_f:
        file.write(f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n")

print(f"Second mesh saved as {obj_file_path_2}")

skin_template_v = data['skin_template_v']  # Shape: (31028, 3)
skin_template_f = data['skin_template_f']  # Shape: (61920, 3)

with open(obj_file_path_3, "w") as file:
    for vertex in skin_template_v:
        file.write(f"v {vertex[0]} {vertex[1]} {vertex[2]}\n")
    
    for face in skin_template_f:
        file.write(f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n")

print(f"Third mesh saved as {obj_file_path_3}")
