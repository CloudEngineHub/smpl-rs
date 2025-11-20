#!/usr/bin/env python3
"""
Shows how to create a scene with a body and object props and export them to glb
"""

import os
import os.path as osp

from gloss import ViewerDummy
from gloss.builders import builders

from smpl_rs import SmplCache
from smpl_rs.codec import GltfCodec, SmplCodec
from smpl_rs.plugins import SmplPlugin
from smpl_rs.types import SmplType, Gender, GltfCompatibilityMode
from smpl_rs.components import GlossInterop, TransformSequence, SceneAnimation

from os import listdir
from os.path import isfile, join
import os
from pathlib import Path

if __name__ == "__main__":
    export_body = True
    export_prop = True

    # get paths to all the data needed
    path_data = osp.join(osp.dirname(osp.realpath(__file__)), "../../../data/")
    path_smpl = osp.join(osp.dirname(osp.realpath(__file__)), "../../../data/smplx")

    path_smpl_codec = os.path.join(path_data, "props/largebox/codec_spec.smpl")
    path_glb = os.path.join(path_data, "props/largebox/largebox.glb")
    path_trajectory = os.path.join(path_data, "props/largebox/object_spec.npz")

    # Follow instructions in the README to generate these .npz files
    path_model_neutral = os.path.join(path_smpl, "SMPLX_neutral_array_f32_slim.npz")
    path_model_male = os.path.join(path_smpl, "SMPLX_male_array_f32_slim.npz")
    path_model_female = os.path.join(path_smpl, "SMPLX_female_array_f32_slim.npz")

    smpl_models = SmplCache.default()
    smpl_models.set_lazy_loading(SmplType.SmplX, Gender.Neutral, path_model_neutral)
    smpl_models.set_lazy_loading(SmplType.SmplX, Gender.Male, path_model_male)
    smpl_models.set_lazy_loading(SmplType.SmplX, Gender.Female, path_model_female)

    viewer = ViewerDummy()

    viewer.add_resource(smpl_models)
    viewer.insert_plugin(
        SmplPlugin(autorun=False)
    )  # keep autorun to false since we want to call it manually one before glb export

    # read the .smpl file because we need the fps to initialize the SceneAnimation resources
    # ideally the fps would exists in the .npz so we can create glb with only the prop without needing to read the smpl file
    smpl_codec = SmplCodec.from_file(path_smpl_codec)
    viewer.add_resource(
        SceneAnimation.new_with_fps(
            num_frames=smpl_codec.frame_count, fps=smpl_codec.frame_rate
        )
    )

    # create body (can be turned off if you don't want to export the body)
    if export_body:
        smpl_ent = viewer.get_or_create_entity("smpl_ent")
        smpl_ent.insert_builder(smpl_codec.to_entity_builder())
        smpl_ent.insert(GlossInterop(with_uv=True))

    # create object prop (can be turned off if you don't want to export the prop)
    if export_prop:
        prop_ent = viewer.get_or_create_entity("prop_ent")
        prop_ent.insert_builder(builders.build_from_file(path_glb))
        prop_ent.insert(TransformSequence.new_from_npz(path_trajectory))

    # export to glb
    # run the once to initialize the systems and load the smpl models, create the SceneAnimation resource etc
    viewer.run_manual_plugins()
    # do the actual glb export
    output_path = "smpl_with_props.glb"
    gltf_codec = GltfCodec.from_scene(
        viewer.get_scene().ptr_idx(), export_camera=False, export_shape=True
    )
    gltf_codec.save(output_path, GltfCompatibilityMode.Unreal)
    print(f"Saved glTF to {output_path}")
