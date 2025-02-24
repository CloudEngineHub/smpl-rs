#!/usr/bin/env python3
"""
Edit Color based on Animation
"""

import os
import numpy as np

from gloss import Entity, Viewer
from gloss.types import MeshColorType
from gloss.components import VisMesh, Colors

from smpl_rs import SmplCache
from smpl_rs.plugins import SmplPlugin
from smpl_rs.types import SmplType, Gender, AnimWrap
from smpl_rs.components import SmplParams, Betas, Animation, GlossInterop

if __name__ == "__main__":
    viewer = Viewer()
    mesh = viewer.get_or_create_entity(name = "mesh")

    #get paths to all the data needed for this entity
    path_data=os.path.join( os.path.dirname( os.path.realpath(__file__) ),"../../../data/smplx")
    path_anim=os.path.join(path_data,"rich_Gym_010_pushup1.npz")
    path_model_neutral=os.path.join(path_data,"SMPLX_neutral_array_f32_slim.npz")
    path_model_male=os.path.join(path_data,"SMPLX_male_array_f32_slim.npz")
    path_model_female=os.path.join(path_data,"SMPLX_female_array_f32_slim.npz")

    #insert the needed components
    smpl_params = SmplParams.default()
    betas = Betas.default()
    animation = Animation.from_npz(path_anim, 40.0,
                                   wrap_behaviour=AnimWrap.Loop, smpl_type=SmplType.SmplH)
    interop = GlossInterop(with_uv=False)
    mesh_visualisation = VisMesh(color_type=MeshColorType.PerVert)
    mesh.insert(smpl_params)
    mesh.insert(betas)
    mesh.insert(animation)
    mesh.insert(interop)
    mesh.insert(mesh_visualisation)

    # insert a resource which is a component that can be shared between multiple entities
    # this one just lazy loads all smpl models you might need
    smpl_models = SmplCache.default()
    smpl_models.set_lazy_loading(SmplType.SmplX, Gender.Neutral, path_model_neutral)
    smpl_models.set_lazy_loading(SmplType.SmplX, Gender.Male, path_model_male)
    smpl_models.set_lazy_loading(SmplType.SmplX, Gender.Female, path_model_female)
    viewer.add_resource(smpl_models)

    # insert a plugin which governs the logic functions that run on the entities
    # depending on the components they have
    viewer.insert_plugin(SmplPlugin(autorun=False))

    #read colors
    data = np.load(path_anim)
    anim_colors= data["verts_colors"]

    while True:
        viewer.start_frame()

        # manually runs the SmplPlugin logic.
        # Needs to be done here because we want the animation indices to be updated
        viewer.run_manual_plugins()

        anim = mesh.get(Animation)

        (idx_start, idx_end, weight_end)= anim.get_smooth_time_indices()
        print("start, end, w_end", idx_start, idx_end, weight_end)

        #slice from colors
        colors_start = anim_colors[idx_start,:, :]
        colors_end = anim_colors[idx_end,:, :]
        colors = colors_start * (1 - weight_end) + colors_end * weight_end #blend
        colors = colors / 255.0

        # attach the colors to the mesh
        mesh.insert(Colors(colors.astype(np.float32)))

        viewer.update()
