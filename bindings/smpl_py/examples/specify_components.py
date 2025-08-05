#!/usr/bin/env python3
"""
Add more components to a smpl body
"""
import os
import numpy as np

from gloss import Viewer
from gloss.log import gloss_setup_logger as setup_logger, LogLevel
from gloss.components import DiffuseImg, NormalImg

from smpl_rs import SmplCache
from smpl_rs.plugins import SmplPlugin
from smpl_rs.types import (
    SmplType,
    Gender,
    AnimWrap,
    AngleType,
    UpAxis,
    HandType,
    FollowerType,
)
from smpl_rs.components import (
    SmplParams,
    Betas,
    Animation,
    GlossInterop,
    PoseOverride,
    Follow,
    Follower,
)

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
    path_anim = os.path.join(path_data, "apose_to_catwalk_001.npz")
    path_diffuse = os.path.join(path_data, "female_alb_2.png")
    path_normal = os.path.join(path_data, "female_nrm.png")
    path_model_neutral = os.path.join(path_data, "SMPLX_neutral_array_f32_slim.npz")
    path_model_male = os.path.join(path_data, "SMPLX_male_array_f32_slim.npz")
    path_model_female = os.path.join(path_data, "SMPLX_female_array_f32_slim.npz")

    # insert the needed components
    smpl_params = SmplParams(SmplType.SmplX, Gender.Female, enable_pose_corrective=True)
    betas = Betas(np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32))
    animation = Animation.from_npz(
        path_anim,
        70.0,
        wrap_behaviour=AnimWrap.Reverse,
        angle_type=AngleType.AxisAngle,
        up_axis=UpAxis.Y,
        smpl_type=SmplType.SmplH,
    )
    pose_override = PoseOverride.allow_all().overwrite_hands(HandType.Relaxed)
    diffuse = DiffuseImg(path_diffuse)
    normal = NormalImg(path_normal)
    interop = GlossInterop(with_uv=True)
    follow = Follow()
    mesh.insert(smpl_params)
    mesh.insert(betas)
    mesh.insert(animation)
    mesh.insert(pose_override)
    mesh.insert(diffuse)
    mesh.insert(normal)
    mesh.insert(interop)
    mesh.insert(follow)

    # insert a resource which is a component that can be shared between multiple entities
    # this one just lazy loads all smpl models you might need
    smpl_models = SmplCache.default()
    smpl_models.set_lazy_loading(SmplType.SmplX, Gender.Neutral, path_model_neutral)
    smpl_models.set_lazy_loading(SmplType.SmplX, Gender.Male, path_model_male)
    smpl_models.set_lazy_loading(SmplType.SmplX, Gender.Female, path_model_female)
    viewer.add_resource(smpl_models)

    # allows the camera to follow the movement of this entity
    viewer.add_resource(
        Follower(
            max_strength=30.0,
            dist_start=0.1,
            dist_end=1.0,
            follower_type=FollowerType.CamAndLights,
        )
    )

    # insert a plugin which governs the logic functions that run on the entities
    # depending on the components they have
    viewer.insert_plugin(SmplPlugin(autorun=True))
    viewer.run()
