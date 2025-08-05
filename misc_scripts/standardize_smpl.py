#!/usr/bin/env python3
"""
Download the smpl-x models (SMPL-X with removed headbun NPZ)
from https://smpl-x.is.tue.mpg.de/download.php
"""
# Some smpl files for npz are using f64, some f32 and some even use objects instead of array
# Call this script to standardize them to f32 and numpy arrays only

import os, os.path as osp
import numpy as np


base_dir = osp.dirname(osp.abspath(__file__))
path_model_male = osp.join(base_dir, "../../models_lockedhead/smplx/SMPLX_MALE.npz")
path_model_female = osp.join(base_dir, "../../models_lockedhead/smplx/SMPLX_FEMALE.npz")
path_model_neutral = osp.join(
    base_dir, "../../models_lockedhead/smplx/SMPLX_NEUTRAL.npz"
)

assert osp.exists(path_model_male), f"Path to model {path_model_male} does not exist"
assert osp.exists(
    path_model_female
), f"Path to model {path_model_female} does not exist"
assert osp.exists(
    path_model_neutral
), f"Path to model {path_model_neutral} does not exist"

out_model_male = osp.join(base_dir, "../data/smplx/SMPLX_male_array_f32_slim.npz")
out_model_female = osp.join(base_dir, "../data/smplx/SMPLX_female_array_f32_slim.npz")
out_model_neutral = osp.join(base_dir, "../data/smplx/SMPLX_neutral_array_f32_slim.npz")

out_model_male_uv = osp.join(base_dir, "../data/smplx/SMPLX_male_uv.npy")
out_model_female_uv = osp.join(base_dir, "../data/smplx/SMPLX_female_uv.npy")
out_model_neutral_uv = osp.join(base_dir, "../data/smplx/SMPLX_neutral_uv.npy")

out_model_female_ft = osp.join(base_dir, "../data/smplx/SMPLX_female_ft.npy")
out_model_male_ft = osp.join(base_dir, "../data/smplx/SMPLX_male_ft.npy")
out_model_neutral_ft = osp.join(base_dir, "../data/smplx/SMPLX_neutral_ft.npy")


def standardize_npz(in_path, out_path, uv_path, ft_path):
    """
    Standardize the npz file to f32 and numpy arrays only
    """
    in_npz = np.load(in_path, allow_pickle=True)
    out_dict = {}
    for key, val in in_npz.items():
        if key in ["vt", "ft"]:
            if key == "vt":
                val = np.load(uv_path, allow_pickle=True)
            elif key == "ft":
                print(f"Loading FT from {ft_path}")
                val = np.load(ft_path, allow_pickle=True)
        print(f"Key: {key}, Type: {val.dtype}, Shape: {val.shape}")
        if val.dtype == np.float64:
            val = np.float32(val)
        elif val.dtype == np.int64:
            val = np.int32(val)
        elif val.dtype == np.uint64:
            val = np.uint32(val)
        elif val.dtype == object:
            if key == "ft":
                val = np.uint32(val)
        out_dict[key] = val
    np.savez_compressed(out_path, **out_dict)


# Then standardize the NPZs
standardize_npz(path_model_male, out_model_male, out_model_male_uv, out_model_male_ft)
standardize_npz(
    path_model_female, out_model_female, out_model_female_uv, out_model_female_ft
)
standardize_npz(
    path_model_neutral, out_model_neutral, out_model_neutral_uv, out_model_neutral_ft
)
