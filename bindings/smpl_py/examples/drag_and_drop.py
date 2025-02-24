#!/usr/bin/env python3
"""
Load the models and open an empty window such that you can drag and drop smpl files
"""

import os

from gloss import Viewer

from smpl_rs import SmplCache
from smpl_rs.plugins import SmplPlugin
from smpl_rs.types import SmplType, Gender

if __name__ == "__main__":
    viewer = Viewer()

    #get paths to all the data needed for this entity
    path_data=os.path.join( os.path.dirname( os.path.realpath(__file__) ),"../../../data/smplx")
    path_model_neutral=os.path.join(path_data,"SMPLX_neutral_array_f32_slim.npz")
    path_model_male=os.path.join(path_data,"SMPLX_male_array_f32_slim.npz")
    path_model_female=os.path.join(path_data,"SMPLX_female_array_f32_slim.npz")

    # insert a resource which is a component that can be shared between multiple entities
    # this one just lazy loads all smpl models you might need
    smpl_models = SmplCache.default()
    smpl_models.set_lazy_loading(SmplType.SmplX, Gender.Neutral, path_model_neutral)
    smpl_models.set_lazy_loading(SmplType.SmplX, Gender.Male, path_model_male)
    smpl_models.set_lazy_loading(SmplType.SmplX, Gender.Female, path_model_female)
    viewer.add_resource(smpl_models)

    # insert a plugin which governs the logic functions that run on the entities
    # depending on the components they have
    viewer.insert_plugin(SmplPlugin(autorun=True))
    viewer.run()
