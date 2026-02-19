#!/usr/bin/env python3
"""
An example showing how to read an animation and run smplx for every pose and also do pose interpolation for each subframe
"""

import os
import numpy as np

import gloss
from gloss import Viewer
from gloss.log import gloss_setup_logger as setup_logger, LogLevel

import smpl_rs
from smpl_rs import SmplCache
from smpl_rs.models import SmplX
from smpl_rs.plugins import SmplPlugin
from smpl_rs.types import SmplType, Gender
from smpl_rs.components import (
    SmplParams,
    Betas,
    Animation,
    GlossInterop,
    Animation,
    SmplOptions,
)
from smpl_rs.codec import SmplCodec
from gloss.components import Verts, Faces, Normals
from gloss.backend import gloss_init_burn_backend
from smpl_rs.backend import smplrs_init_burn_backend
from smpl_rs.backend import smplrs_sync_burn_gpu


# Set up the logger
# To be called only once per process. Can select between Off, Error, Warn, Info, Debug, Trace
setup_logger(log_level=LogLevel.Info)
# Initialize the backend used for burn computations
# gloss_init_burn_backend("torch_cpu")
# smplrs_init_burn_backend("torch_cpu")
# gloss_init_burn_backend("torch_cuda",0)
# smplrs_init_burn_backend("torch_cuda",0)
# gloss_init_burn_backend("wgpu")
# smplrs_init_burn_backend("wgpu")

if __name__ == "__main__":
    viewer = Viewer()
    smplrs_sync_burn_gpu(viewer.get_ptr_gpu())

    # Get paths to all the data needed for this entity
    path_data = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "../../../data/smplx"
    )
    path_model_neutral = os.path.join(path_data, "SMPLX_neutral_array_f32_slim.npz")
    # path_smpl_file = os.path.join(path_data, "Dance_03_w_hands.smpl")
    path_smpl_file = "/home/rosu/Downloads/S1_Wrestling_Greco_Roman_01_02_michael.smpl"

    # create smpl model
    smpl_model = SmplX.from_npz(path_model_neutral, Gender.Neutral)

    # read animation from .smpl file
    smpl_codec = SmplCodec.from_file(path_smpl_file)
    betas = Betas.from_smpl_file(path_smpl_file)
    smpl_options = SmplOptions.default()
    # read anim only if there actually is one in the smpl file
    if smpl_codec.frame_count <= 1:
        print("The provided smpl file does not contain an animation (frame_count<=1)")
    else:
        animation = Animation.from_smpl_file(path_smpl_file)

        while True:
            # Interpolate poses
            for i in range(animation.num_animation_frames() - 1):
                pose_start = animation.get_pose_at_idx(i)
                pose_end = animation.get_pose_at_idx(i + 1)
                for t in np.linspace(0, 1, num=5):
                    pose_interp = pose_start.interpolate(pose_end, t, use_slerp=True)

                    # for this interpolated pose run the forward smplx model
                    smpl_output = smpl_model.forward(smpl_options, betas, pose_interp)

                    # visualize the output
                    smpl_body = viewer.get_or_create_entity(name="smpl_body")
                    smpl_body.insert(Verts(smpl_output.verts.to_numpy()))
                    smpl_body.insert(Faces(smpl_output.faces.to_numpy()))
                    smpl_body.remove(
                        Normals
                    )  # Remove the normals so gloss recalculates them at every frame

                    viewer.start_frame()

                    viewer.update()
