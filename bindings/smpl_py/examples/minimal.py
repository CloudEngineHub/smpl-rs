#!/usr/bin/env python3
"""
A minimal example of loading an animated avatar
"""

import os

from gloss import Viewer
from gloss.log import gloss_setup_logger as setup_logger, LogLevel

from smpl_rs import SmplCache
from smpl_rs.plugins import SmplPlugin
from smpl_rs.types import SmplType, Gender
from smpl_rs.components import SmplParams, Betas, Animation, GlossInterop

# Set up the logger
# To be called only once per process. Can select between Off, Error, Warn, Info, Debug, Trace
setup_logger(log_level=LogLevel.Info)

if __name__ == "__main__":
    viewer = Viewer()

    smpl_body = viewer.get_or_create_entity(name="smpl_body")

    # Get paths to all the data needed for this entity
    path_data = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "../../../data/smplx"
    )
    path_anim = os.path.join(path_data, "apose_to_00093lazysaturdaynightfever.npz")
    path_model_neutral = os.path.join(path_data, "SMPLX_neutral_array_f32_slim.npz")
    path_model_male = os.path.join(path_data, "SMPLX_male_array_f32_slim.npz")
    path_model_female = os.path.join(path_data, "SMPLX_female_array_f32_slim.npz")

    # Insert the needed components
    smpl_params = SmplParams.default()
    betas = Betas.default()
    animation = Animation.from_npz(path_anim, fps=100.0, smpl_type=SmplType.SmplH)
    interop = GlossInterop.default()

    smpl_body.insert(smpl_params)
    smpl_body.insert(betas)
    smpl_body.insert(animation)
    smpl_body.insert(interop)

    smpl_models = SmplCache.default()
    smpl_models.set_lazy_loading(SmplType.SmplX, Gender.Neutral, path_model_neutral)
    smpl_models.set_lazy_loading(SmplType.SmplX, Gender.Male, path_model_male)
    smpl_models.set_lazy_loading(SmplType.SmplX, Gender.Female, path_model_female)
    viewer.add_resource(smpl_models)

    smpl_plugin = SmplPlugin(autorun=True)
    viewer.insert_plugin(smpl_plugin)

    viewer.run()
