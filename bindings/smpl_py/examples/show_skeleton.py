#!/usr/bin/env python3
"""
Show a smpl body with skeleton
"""

import os
import numpy as np

from gloss import Viewer
from gloss.log import gloss_setup_logger as setup_logger, LogLevel
from gloss.components import ModelMatrix, VisLines, Verts, Edges

from smpl_rs import SmplCache
from smpl_rs.plugins import SmplPlugin
from smpl_rs.models import SmplOutputPosed, SmplX
from smpl_rs.types import SmplType, Gender
from smpl_rs.components import SmplParams, Betas, Animation, GlossInterop

# Set up the logger
# To be called only once per process. Can select between Off, Error, Warn, Info, Debug, Trace
setup_logger(log_level=LogLevel.Info)

if __name__ == "__main__":
    viewer = Viewer()
    mesh = viewer.get_or_create_entity(name="mesh")

    # get paths to all the data needed for this entity
    path_data = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "../../../data/smplx"
    )
    path_anim = os.path.join(path_data, "apose_to_00093lazysaturdaynightfever.npz")
    path_model_neutral = os.path.join(path_data, "SMPLX_neutral_array_f32_slim.npz")
    path_model_male = os.path.join(path_data, "SMPLX_male_array_f32_slim.npz")
    path_model_female = os.path.join(path_data, "SMPLX_female_array_f32_slim.npz")

    rot = (
        ModelMatrix.default()
        .with_translation(np.array([0, 0.09, 0], dtype="float32"))
        .with_rotation_euler(np.array([0.0, -1.571, 0], dtype="float32"))
    )

    # entity for the smpl model
    smpl_params = SmplParams.default()
    betas = Betas.default()
    animation = Animation.from_npz(path_anim, fps=100.0, smpl_type=SmplType.SmplH)
    interop = GlossInterop.default()
    mesh.insert(smpl_params)
    mesh.insert(betas)
    mesh.insert(animation)
    mesh.insert(interop)
    mesh.insert(rot)

    # entity for the joints
    joints = viewer.get_or_create_entity(name="joints")
    joints_line_visualisation = VisLines(show_lines=True, line_width=5.0, zbuffer=False)
    joints.insert(joints_line_visualisation)
    joints.insert(rot)

    # insert a resource which is a component that can be shared between multiple entities
    # this one just lazy loads all smpl models you might need
    smpl_cache = SmplCache.default()
    smpl_cache.set_lazy_loading(SmplType.SmplX, Gender.Neutral, path_model_neutral)
    smpl_cache.set_lazy_loading(SmplType.SmplX, Gender.Male, path_model_male)
    smpl_cache.set_lazy_loading(SmplType.SmplX, Gender.Female, path_model_female)
    viewer.add_resource(smpl_cache)

    smpl_model = SmplX.from_npz(path_model_neutral, Gender.Neutral, max_num_betas=10)

    # get the joint hierarchy
    parent_joint = smpl_model.parent_idx_per_joint
    parent_joint[0] = 0
    start_joint = np.arange(55)
    edges = np.vstack((start_joint, parent_joint)).T
    edges = edges.astype(np.uint32)

    # Insert a plugin which governs the logic functions that run on the entities
    # depending on the components they have
    # IMPORTANT: autorun is set to false, so we gain control over when the plugin is being run
    # (the plugin now runs when we call run_manual_plugins())
    viewer.insert_plugin(SmplPlugin(autorun=True))

    while True:
        viewer.start_frame()

        # Manually runs the SmplPlugin logic.
        # This will effectivelly run a forward pass of smpl so afterwards the mesh entity
        # will have the SmplOutputPosed component
        viewer.run_manual_plugins()

        # make an entity for the joints and the lines between them
        smpl_output = mesh.get(SmplOutputPosed)
        joint_positions = smpl_output.joints

        j_verts, j_edges = Verts(joint_positions), Edges(edges)
        joints.insert(j_verts)
        joints.insert(j_edges)

        viewer.update()
