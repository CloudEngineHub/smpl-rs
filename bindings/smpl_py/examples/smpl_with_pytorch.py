#!/usr/bin/env python3
"""
This example shows how to run a forward pass through the smpl model
while using pytorch as the backend for computations

"""
import os
import time
import numpy as np
import torch

from gloss import Viewer
from gloss.log import gloss_setup_logger as setup_logger, LogLevel
from gloss.components import Verts, Faces

from smpl_rs.models import SmplX
from smpl_rs.types import SmplType, Gender, UpAxis
from smpl_rs.components import Betas, Follower, Pose, SmplOptions, Follow

from gloss.backend import gloss_init_burn_backend
from smpl_rs.backend import smplrs_init_burn_backend
from smpl_rs.backend import smplrs_sync_burn_gpu


# Set up the logger
# To be called only once per process. Can select between Off, Error, Warn, Info, Debug, Trace
setup_logger(log_level=LogLevel.Info)
# Initialize the backend used for burn computations
gloss_init_burn_backend("torch_cuda", 0)
smplrs_init_burn_backend("torch_cuda", 0)

if __name__ == "__main__":
    viewer = Viewer()
    smplrs_sync_burn_gpu(viewer.get_ptr_gpu())

    # get paths to all the data needed for this entity
    path_data = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "../../../data/smplx"
    )
    path_model = os.path.join(path_data, "SMPLX_neutral_array_f32_slim.npz")

    # make torch tensors on cuda
    betas_t = torch.zeros(10).cuda()
    joint_poses_t = torch.zeros((54 * 3)).cuda()
    global_trans = torch.zeros(3).cuda()

    # run forward function on smpl
    smpl = SmplX.from_npz(path_model, Gender.Female)
    betas = Betas.from_tensor(betas_t)
    pose = Pose.from_tensors(joint_poses_t, global_trans, UpAxis.Y, SmplType.SmplX)
    smpl_options = SmplOptions.default()

    smpl_output = smpl.forward(smpl_options, betas, pose)

    print("smpl_output", smpl_output)

    verts = smpl_output.verts.to_torch()
    faces = smpl_output.faces.to_torch()

    print("verts", verts)
    print("faces", faces)
    print("verts numpy", smpl_output.verts.to_numpy())
    print("faces numpy", smpl_output.faces.to_numpy())
