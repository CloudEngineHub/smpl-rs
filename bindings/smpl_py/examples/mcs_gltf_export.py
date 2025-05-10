#!/usr/bin/env python3
"""
An example on how to load an MCS file into a scene and 
export it as a gltf file (with or without camera)
"""

import os
import os.path as osp

from gloss import ViewerDummy

from smpl_rs import SmplCache
from smpl_rs.codec import McsCodec, GltfCodec
from smpl_rs.plugins import SmplPlugin
from smpl_rs.types import SmplType, Gender, GltfCompatibilityMode
from smpl_rs.components import GlossInterop

if __name__ == "__main__":
    viewer = ViewerDummy()

    # get paths to all the data needed for this entity
    path_data = osp.join(osp.dirname(osp.realpath(__file__)), "../../../data/smplx")
    path_data_mcs = osp.join(osp.dirname(osp.realpath(__file__)), "../../../data/mcs")
    path_mcs = os.path.join(path_data_mcs, "boxing.mcs")
    assert os.path.exists(path_mcs), f"File {path_mcs} does not exist"

    # Follow instructions in the README to generate these .npz files
    path_model_neutral = os.path.join(path_data, "SMPLX_neutral_array_f32_slim.npz")
    path_model_male = os.path.join(path_data, "SMPLX_male_array_f32_slim.npz")
    path_model_female = os.path.join(path_data, "SMPLX_female_array_f32_slim.npz")

    mcs_codec = McsCodec.from_file(path_mcs)

    print("\nInformation from the MCS file:")
    print(f"Number of frames: {mcs_codec.num_frames}")
    print(f"Number of bodies: {mcs_codec.num_bodies}")
    print(f"Has camera: {mcs_codec.has_camera}")
    print(f"Frame rate: {mcs_codec.frame_rate}")

    entity_builders = mcs_codec.to_entity_builders()

    for current_ent, builder in enumerate(entity_builders):
        entity = viewer.get_or_create_entity(name=f"mcs_entity_{current_ent}")
        entity.insert(builder)
        interop = GlossInterop(with_uv=True)
        entity.insert(interop)

    smpl_models = SmplCache.default()
    smpl_models.set_lazy_loading(SmplType.SmplX, Gender.Neutral, path_model_neutral)
    smpl_models.set_lazy_loading(SmplType.SmplX, Gender.Male, path_model_male)
    smpl_models.set_lazy_loading(SmplType.SmplX, Gender.Female, path_model_female)

    viewer.add_resource(smpl_models)
    viewer.insert_plugin(SmplPlugin(autorun=False))
    viewer.run_manual_plugins()

    # Create the writer and export as Glb
    GLTF_SAVE_PATH = "../../saved/mesh.gltf"
    gltf_codec = GltfCodec.from_scene(viewer.get_scene().ptr_idx(), export_camera = True)
    gltf_codec.save(GLTF_SAVE_PATH, GltfCompatibilityMode.Smpl)
    print(f"Saved glTF to {GLTF_SAVE_PATH}")
