#!/usr/bin/env python3
from smplx import *
from gloss_py import *
from smpl_py import *
import numpy as np
import os
import torch
import pytorch3d
from pytorch3d import transforms

# data = np.load("/media/rosu/Data/jobs/meshcapade/c_ws/src/meshcapade/model-registry/poses/SMPLX/hands/relaxed.npy")
# data = np.load("/media/rosu/Data/jobs/meshcapade/c_ws/src/meshcapade/model-registry/poses/SMPLX/hands/curl.npy")
# data = np.load("/media/rosu/Data/jobs/meshcapade/c_ws/src/meshcapade/model-registry/poses/SMPLX/hands/fist.npy")
data = np.reshape(data, (2, 45))
data_left = data[0, :]
data_right = data[1, :]
data_left = torch.tensor(np.reshape(data_left, (15, 3))).reshape(1, -1)
print("data_left", data_left.shape)


path = "/media/rosu/Data/jobs/meshcapade/data/avatars/body_models/SMPLX_female.npz"
smpl = SMPLX(path, num_pca_comps=45, flat_hand_mean=True)
print("coeffs left", smpl.left_hand_components.shape)

smpl_output = smpl.forward(
    left_hand_pose=data_left,
    right_hand_pose=data_left,
    use_pca=True,
    return_full_pose=True,
)
print("full pose from smplx output is ", smpl_output.full_pose.reshape((-1, 3)))
