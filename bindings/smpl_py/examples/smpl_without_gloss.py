#!/usr/bin/env python3
"""
This example shows how to run a forward pass through the smpl model
without using the ECS system of gloss. We only use gloss for visualization of the smpl output
"""
import os
import time
import numpy as np

from gloss import Viewer
from gloss.log import gloss_setup_logger as setup_logger, LogLevel
from gloss.components import Verts, Faces

from smpl_rs.models import SmplX
from smpl_rs.types import SmplType, Gender, UpAxis
from smpl_rs.components import Betas, Follower, Pose, SmplOptions, Follow

# Set up the logger
# To be called only once per process. Can select between Off, Error, Warn, Info, Debug, Trace
setup_logger(log_level=LogLevel.Info)

if __name__ == "__main__":
    viewer = Viewer()

    # get paths to all the data needed for this entity
    path_data = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "../../../data/smplx"
    )
    path_anim = os.path.join(path_data, "apose_to_00093lazysaturdaynightfever.npz")
    path_model = os.path.join(path_data, "SMPLX_female_array_f32_slim.npz")

    # read one frame of the animation
    anim = np.load(path_anim)
    poses = anim["poses"].astype(np.float32)
    trans = anim["trans"].astype(np.float32)
    nr_frames = poses.shape[0]
    frame_to_slice = int(nr_frames / 2)
    singular_pose = poses[frame_to_slice, :]
    singular_trans = trans[frame_to_slice, :]

    # run forward function on smpl
    smpl = SmplX.from_npz(path_model, Gender.Female)
    betas = Betas(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32))
    pose = Pose.from_matrices(singular_pose, singular_trans, UpAxis.Y, SmplType.SmplH)
    smpl_options = SmplOptions.default()

    t0 = time.time()
    smpl_output = smpl.forward(smpl_options, betas, pose)
    t1 = time.time()
    print("--diff in ms: ", (t1 - t0) * 1000)

    # insert the components just to visualize
    mesh = viewer.get_or_create_entity(name="mesh")
    verts = Verts(smpl_output.verts)
    faces = Faces(smpl_output.faces)
    follow = Follow()
    mesh.insert(verts)
    mesh.insert(faces)
    mesh.insert(follow)

    # allows the camera to follow the movement of this entity
    viewer.add_resource(Follower())

    viewer.run()
