#!/usr/bin/env python3
"""
This example shows how you can  do advance the animation by a fixed time everytime
in headless mode. This is specifically in context of .mcs files (or scenes), and uses the SceneTimer
class to ensure all bodies are in sync.
This can be useful when rendering the animation to mp4s in which case you would
want to advance the animation 33ms everytime you render.
"""
import os
import os.path as osp

from gloss import ViewerHeadless
from gloss.log import gloss_setup_logger as setup_logger, LogLevel

from smpl_rs import SmplCache, SceneTimer
from smpl_rs.plugins import SmplPlugin
from smpl_rs.codec import McsCodec
from smpl_rs.types import SmplType, Gender
from smpl_rs.components import (
    GlossInterop,
    Follower,
)

# Set up the logger
# To be called only once per process. Can select between Off, Error, Warn, Info, Debug, Trace
setup_logger(log_level=LogLevel.Info)

if __name__ == "__main__":
    viewer = ViewerHeadless(800, 800)
    scene_timer = SceneTimer.from_scene(viewer.get_scene().ptr_idx())

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

    print("\nInformation from the MCS file:")
    print(f"Number of frames: {mcs_codec.num_frames}")
    print(f"Number of bodies: {mcs_codec.num_bodies}")
    print(f"Has camera: {mcs_codec.has_camera}")
    print(f"Frame rate: {mcs_codec.frame_rate}\n")

    # This gives us a list of entity builders which we can use to create entities
    entity_builders = mcs_codec.to_entity_builders()

    # Create entities in the Mcs file
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

    # allows the camera to follow the movement of the mean position of all entities
    # The plugin auto adds the Follow component to all smpl entities
    viewer.add_resource(Follower())

    # insert a plugin which governs the logic functions that run on the entities
    # depending on the components they have
    viewer.insert_plugin(SmplPlugin(autorun=True))

    # Below commented lines are for verifying that the scene timer is added as expected
    # viewer.insert_plugin(SmplPlugin(autorun=False))
    # viewer.run_manual_plugins()  # Run the plugins manually so the scene timer is
    #                              # created internally based on the smpl bodies in the scene

    # print("\nScene animation info retrieved from smpl-rs internal scene timer:")
    # print(f"Number of frames: {scene_timer.num_scene_animation_frames()}")
    # print(f"Duration: {scene_timer.duration()}\n")

    while True:
        viewer.start_frame()

        # Advance the animation
        scene_timer.pause()  # so that the SmplPlugin doesn't automatically advance the animation
        scene_timer.advance_sec(0.0333333)
        print("Animation time: ", scene_timer.get_cur_time_sec())

        viewer.update()

        if scene_timer.is_finished():
            break

        viewer.save_last_render("./image.png")
