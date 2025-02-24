#!/usr/bin/env python3

#some smpl files for npz are using f64, some f32 and some even use objects instead of array. Call this script to standardize them to f32 and numpy arrays only
import numpy as np


# in_path="/media/rosu/Data/jobs/meshcapade/data/avatars/body_models/SMPLX_female.npz"
# in_path="/media/rosu/Data/jobs/meshcapade/data/avatars/body_models/SMPLX_MALE.npz"
# in_path ="/media/rosu/Data/jobs/meshcapade/data/avatars/body_models/smplx_lockedhead_20230207/models_lockedhead/smplx/SMPLX_MALE.npz"
# in_path ="/media/rosu/Data/jobs/meshcapade/c_ws/src/meshcapade/model-registry/models/SMPLX/neutral/SMPLX_neutral.npz"
in_path ="/media/rosu/Data/jobs/meshcapade/c_ws/src/meshcapade/model-registry/models/SMPLX/male/SMPLX_male.npz"
# out_path="/media/rosu/Data/jobs/meshcapade/data/avatars/body_models/SMPLX_female_array_f32.npz"
# out_path="/media/rosu/Data/jobs/meshcapade/data/avatars/body_models/SMPLX_male_array_f32.npz"
# out_path="/media/rosu/Data/jobs/meshcapade/data/avatars/body_models/SMPLX_neutral_array_f32.npz"
out_path="/media/rosu/Data/jobs/meshcapade/data/avatars/body_models/SMPLX_male_array_f32.npz"
in_npz = np.load(in_path, allow_pickle=True)
out_dict ={}
for key, val in in_npz.items():
    print("key is -----------------------------", key)
    print("val is shape", val.shape)
    print("val is type", val.dtype)
    if val.dtype == np.float64:
        val = np.float32(val)
    elif val.dtype == np.int64:
        val = np.int32(val)
    elif val.dtype == np.uint64:
        val = np.uint32(val)
    elif val.dtype == object:
        if key=="ft":
            val = np.uint32(val)

    out_dict[key] = val 

np.savez_compressed(out_path, **out_dict)