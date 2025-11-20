#!/usr/bin/env python3
"""
Shows how to read a smpl file and get it's contents as numpy arrays
"""

from smpl_rs.codec import SmplCodec
import os
import os.path as osp

if __name__ == "__main__":

    # get paths to all the data needed
    path_data = osp.join(osp.dirname(osp.realpath(__file__)), "../../../data/")
    path_smpl = osp.join(path_data, "smplx/squat_ow.smpl")

    # read as smpl codec
    smpl_codec = SmplCodec.from_file(path_smpl)

    # get contents
    print("smpl_version", smpl_codec.smpl_version)
    print("smpl_type", smpl_codec.smpl_type)
    print("gender", smpl_codec.gender)
    print("frame_count", smpl_codec.frame_count)
    print("frame_rate", smpl_codec.frame_rate)
    print("shape_parameters", smpl_codec.shape_parameters)
    print("expression_parameters", smpl_codec.expression_parameters)
    print("body_translation", smpl_codec.body_translation)
    print("body_pose", smpl_codec.body_pose)
    print("head_pose", smpl_codec.head_pose)
    print("left_hand_pose", smpl_codec.left_hand_pose)
    print("right_hand_pose", smpl_codec.right_hand_pose)
    print("vertex_offsets", smpl_codec.vertex_offsets)
