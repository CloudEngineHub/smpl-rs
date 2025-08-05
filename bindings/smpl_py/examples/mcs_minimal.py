#!/usr/bin/env python3
"""
A minimal example of how to use the smpl-rs bindings to load an MCS file and visualise it
"""

import os
import os.path as osp

from gloss import Viewer
from gloss.log import gloss_setup_logger as setup_logger, LogLevel

from smpl_rs import SmplCache
from smpl_rs.codec import McsCodec
from smpl_rs.plugins import SmplPlugin
from smpl_rs.types import SmplType, Gender
from smpl_rs.components import GlossInterop

setup_logger(log_level=LogLevel.Info)

if __name__ == "__main__":
    viewer = Viewer()

    # get paths to all the data needed for this entity
    path_data = osp.join(osp.dirname(osp.realpath(__file__)), "../../../data/smplx")
    path_data_mcs = osp.join(osp.dirname(osp.realpath(__file__)), "../../../data/mcs")
    path_mcs = os.path.join(path_data_mcs, "football.mcs")
    assert os.path.exists(path_mcs), "File does not exist"

    # Follow instructions in the README to generate these .npz files
    path_model_neutral = os.path.join(path_data, "SMPLX_neutral_array_f32_slim.npz")
    path_model_male = os.path.join(path_data, "SMPLX_male_array_f32_slim.npz")
    path_model_female = os.path.join(path_data, "SMPLX_female_array_f32_slim.npz")

    mcs_codec = McsCodec.from_file(path_mcs)

    print("\nInformation about the MCS file:")
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
    viewer.insert_plugin(SmplPlugin(autorun=True))
    viewer.run()
