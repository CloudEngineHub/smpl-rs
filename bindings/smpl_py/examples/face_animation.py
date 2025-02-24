#!/usr/bin/env python3
"""
Loads an animation with facial expressions
"""

import os
import numpy as np

from gloss import Viewer

from smpl_rs import SmplCache
from smpl_rs.plugins import SmplPlugin
from smpl_rs.types import SmplType, Gender, AnimWrap
from smpl_rs.components import SmplParams, Betas, Animation, GlossInterop

if __name__ == "__main__":
    viewer = Viewer()
    smpl_body = viewer.get_or_create_entity(name = "smpl_body")

    # get paths to all the data needed for this entity
    path_data=os.path.join( os.path.dirname( os.path.realpath(__file__) ),"../../../data/smplx")
    path_anim=os.path.join(path_data,"avatar_face_expression.smpl")
    path_model_neutral=os.path.join(path_data,"SMPLX_neutral_array_f32_slim.npz")
    path_model_male=os.path.join(path_data,"SMPLX_male_array_f32_slim.npz")
    path_model_female=os.path.join(path_data,"SMPLX_female_array_f32_slim.npz")

    # insert the needed components
    smpl_params = SmplParams(SmplType.SmplX, Gender.Female, enable_pose_corrective=True)
    betas = Betas( np.array([1, -2, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32) )
    animation = Animation.from_smpl_file(path_anim, wrap_behaviour=AnimWrap.Reverse)
    interop = GlossInterop(with_uv=True)

    smpl_body.insert(smpl_params)
    smpl_body.insert(betas)
    smpl_body.insert(animation)
    smpl_body.insert(interop)

    #insert a resource which is a component that can be shared between multiple entities
    #this one just lazy loads all smpl models you might need
    smpl_models = SmplCache.default()
    smpl_models.set_lazy_loading(SmplType.SmplX, Gender.Neutral, path_model_neutral)
    smpl_models.set_lazy_loading(SmplType.SmplX, Gender.Male, path_model_male)
    smpl_models.set_lazy_loading(SmplType.SmplX, Gender.Female, path_model_female)

    viewer.add_resource(smpl_models)
    viewer.insert_plugin(SmplPlugin(autorun=True))
    viewer.run()
