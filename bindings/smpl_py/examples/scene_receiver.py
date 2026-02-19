#!/usr/bin/env python3
"""
Example that should be paired with scene_sender.py This is the receiver part of scene_sender.py
"""

import numpy as np
import os

from gloss import Viewer
from gloss.log import LogLevel, gloss_setup_logger as setup_logger
from gloss.types import PointColorType
from gloss.components import Verts, Colors, VisPoints
from smpl_rs import SmplCache
from smpl_rs.plugins import SmplPlugin
from smpl_rs.types import SmplType, Gender
from gloss.network import SceneReceiver, TransportConfig, SceneReceiverPlugin
from smpl_rs.network import smpl_register_components_for_receiver

# Set up the logger
# To be called only once per process. Can select between Off, Error, Warn, Info, Debug, Trace
setup_logger(log_level=LogLevel.Info)

if __name__ == "__main__":
    viewer = Viewer()

    # setup scene for network sending
    transport_config = TransportConfig(
        # local
        # address="127.0.0.1",
        # port=46378,
        # server
        address="100.115.140.113",
        port=8888,
    )

    scene_receiver = SceneReceiver(transport_config)
    smpl_register_components_for_receiver(scene_receiver.get_ptr())
    viewer.add_resource(scene_receiver)
    # plugin also
    scene_receiver_plugin = SceneReceiverPlugin(autorun=True)
    viewer.insert_plugin(scene_receiver_plugin)

    # install smpl plugin
    path_data = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "../../../data/smplx"
    )
    path_model_neutral = os.path.join(path_data, "SMPLX_neutral_array_f32_slim.npz")
    path_model_male = os.path.join(path_data, "SMPLX_male_array_f32_slim.npz")
    path_model_female = os.path.join(path_data, "SMPLX_female_array_f32_slim.npz")

    smpl_models = SmplCache.default()
    smpl_models.set_lazy_loading(SmplType.SmplX, Gender.Neutral, path_model_neutral)
    smpl_models.set_lazy_loading(SmplType.SmplX, Gender.Male, path_model_male)
    smpl_models.set_lazy_loading(SmplType.SmplX, Gender.Female, path_model_female)
    viewer.add_resource(smpl_models)

    smpl_plugin = SmplPlugin(autorun=True)
    viewer.insert_plugin(smpl_plugin)

    viewer.run()
