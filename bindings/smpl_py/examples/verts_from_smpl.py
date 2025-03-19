#!/usr/bin/env python3
"""
Gets vertex info from a .smpl file and visualise it side by side with the mesh
"""

import os
import numpy as np

from gloss import Viewer
from gloss.components import Verts, Normals

from smpl_rs import SmplCache
from smpl_rs.codec import SmplCodec
from smpl_rs.plugins import SmplPlugin
from smpl_rs.types import SmplType, Gender
from smpl_rs.components import GlossInterop, Animation

if __name__ == "__main__":
    viewer = Viewer()
    smpl_body = viewer.get_or_create_entity(name = "smpl_body")

    # get paths to all the data needed for this entity
    path_data=os.path.join( os.path.dirname( os.path.realpath(__file__) ),"../../../data/smplx")
    path_smpl=os.path.join(path_data,"squat_ow.smpl")
    assert os.path.exists(path_smpl), "File does not exist"

    # Follow instructions in the README to generate these .npz files
    path_model_neutral=os.path.join(path_data,"SMPLX_neutral_array_f32_slim.npz")
    path_model_male=os.path.join(path_data,"SMPLX_male_array_f32_slim.npz")
    path_model_female=os.path.join(path_data,"SMPLX_female_array_f32_slim.npz")

    # insert the needed components
    smpl_codec = SmplCodec.from_file(path_smpl)
    smpl_body.insert(smpl_codec.to_entity_builder())
    interop = GlossInterop(with_uv=True)
    smpl_body.insert(interop)
    #insert a resource which is a component that can be shared between multiple entities
    #this one just lazy loads all smpl models you might need
    smpl_models = SmplCache.default()
    smpl_models.set_lazy_loading(SmplType.SmplX, Gender.Neutral, path_model_neutral)
    smpl_models.set_lazy_loading(SmplType.SmplX, Gender.Male, path_model_male)
    smpl_models.set_lazy_loading(SmplType.SmplX, Gender.Female, path_model_female)

    viewer.add_resource(smpl_models)
    viewer.insert_plugin(SmplPlugin(autorun=True))
    # Create a point cloud entity
    point_cloud = viewer.get_or_create_entity(name="point_cloud")

    while True:
        viewer.start_frame()

        # Advance the animation
        anim = smpl_body.get(Animation)
        anim.advance_sec(0.0333333)  # Advance by one frame (assuming 30 FPS)

        # Update the SMPL body
        smpl_body.insert(anim)
        viewer.update()

        # Get vertices from the SMPL body
        verts = smpl_body.get(Verts) # this gets the vertices of the current pose

        # Here we visualise the mesh side by side with its corresponding point cloud
        # which we got from the vertices of the current pose
        # Create a copy of the vertices with x offset of 2
        offset_verts = verts.numpy().copy()
        offset_verts[:, 0] += 1.0  # Add 1 to x coordinates

        # Update the point cloud entity with the offset vertices
        point_cloud.insert(Verts(offset_verts))

        if anim.is_finished():
            break
