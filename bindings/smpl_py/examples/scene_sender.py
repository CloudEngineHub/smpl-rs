#!/usr/bin/env python3
"""
Add some data to the ECS world and send it over the network
"""

from examples import scene_receiver
import numpy as np
import os
import time

from gloss import Viewer, ViewerDummy
from gloss.log import LogLevel, gloss_setup_logger as setup_logger
from gloss.types import PointColorType
from gloss.components import Verts, Colors, VisPoints
from gloss.network import SceneSender, TransportConfig
from smpl_rs import SmplCache
from smpl_rs.plugins import SmplPlugin
from smpl_rs.types import SmplType, Gender, AnimWrap
from smpl_rs.components import (
    SmplParams,
    Betas,
    Animation,
    GlossInterop,
    SceneAnimation,
)
from gloss.backend import gloss_init_burn_backend
from smpl_rs.backend import smplrs_init_burn_backend
from smpl_rs.backend import smplrs_sync_burn_gpu
from smpl_rs.network import smpl_register_components_for_sender

# Set up the logger
# To be called only once per process. Can select between Off, Error, Warn, Info, Debug, Trace
setup_logger(log_level=LogLevel.Info)

if __name__ == "__main__":
    viewer = ViewerDummy()

    path_data = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "../../../data/smplx"
    )
    path_anim = os.path.join(path_data, "apose_to_00093lazysaturdaynightfever.npz")
    path_anim2 = os.path.join(path_data, "apose_to_catwalk_001.npz")

    # setup scene for network sending
    # if you are on a remove server run "sudo ss -lntp" and look for LocalAdress:Port that matches the IP of the serve
    transport_config = TransportConfig(
        # local
        address="127.0.0.1",
        port=46378,
        # server
        # address="0.0.0.0",
        # port=8888,
    )

    scene_sender = SceneSender(transport_config)
    smpl_register_components_for_sender(scene_sender.get_ptr())
    scene_sender.start_listening()
    scene_sender.try_connect_to_receiver()
    time.sleep(2)  # give some time to set up the connection
    viewer.add_resource(scene_sender)

    viewer.start_batch_net_sending()

    # Insert the needed components
    smpl_params = SmplParams.default()
    betas = Betas(np.array([0.0, 0.0, 0.0]).astype(np.float32))
    animation = Animation.from_npz(path_anim, fps=100.0, smpl_type=SmplType.SmplH)
    animation2 = Animation.from_npz(path_anim2, fps=100.0, smpl_type=SmplType.SmplH)
    interop = GlossInterop(with_uv=True)

    # first body
    smpl_body = viewer.get_or_create_entity(name="smpl_body")
    smpl_body.insert(smpl_params)
    smpl_body.insert(betas)
    smpl_body.insert(animation)
    smpl_body.insert(interop)

    # a second body
    smpl_body_2 = viewer.get_or_create_entity(name="smpl_body_2")
    smpl_body_2.insert(smpl_params)
    smpl_body_2.insert(betas)
    smpl_body_2.insert(animation2)
    smpl_body_2.insert(interop)

    # scene animation because by default it gets added with wrap=clamp and I want it to loop
    # scene_animation = SceneAnimation.new_with_fps_and_wrap(
    #     num_frames=300, fps=100.0, wrap_behaviour=AnimWrap.Loop
    # )
    # viewer.add_resource(scene_animation)

    viewer.end_batch_net_sending()

    while True:
        pass
