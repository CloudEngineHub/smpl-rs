#!/usr/bin/env python3
"""
This example shows how to run a forward pass through the smpl model
while using pytorch as the backend for computations
Also running a backward pass and optimizing the inputs to the smpl model

"""
import os
import time

import numpy as np
import torch

from gloss import Viewer
from gloss.log import gloss_setup_logger as setup_logger, LogLevel
from gloss.components import Verts, Faces, Normals, VisPoints, Edges, VisLines

from smpl_rs.models import SmplX
from smpl_rs.types import SmplType, Gender, UpAxis
from smpl_rs.components import Betas, Pose, SmplOptions

from gloss.backend import gloss_init_burn_backend
from smpl_rs.backend import smplrs_init_burn_backend
from smpl_rs.backend import smplrs_sync_burn_gpu


# Set up the logger
# To be called only once per process. Can select between Off, Error, Warn, Info, Debug, Trace
setup_logger(log_level=LogLevel.Info)
# Initialize the backend used for burn computations
gloss_init_burn_backend("torch_cuda", 0)
smplrs_init_burn_backend("torch_cuda", 0)

if __name__ == "__main__":
    viewer = Viewer()
    smplrs_sync_burn_gpu(viewer.get_ptr_gpu())

    # get paths to all the data needed for this entity
    path_data = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "../../../data/smplx"
    )
    path_model = os.path.join(path_data, "SMPLX_neutral_array_f32_slim.npz")

    # make torch tensors on cuda
    betas_t = torch.zeros(10).cuda()

    # We optimize the body pose (21 joints * 3 = 63) not including global rotation
    body_pose_dim = 21 * 3  # 63 parameters for body pose (excluding global rotation)
    body_pose_t = torch.zeros(body_pose_dim).cuda().requires_grad_(True)
    global_orient_t = torch.zeros(3).cuda()  # Global orientation (root joint)
    global_trans = torch.zeros(3).cuda()

    def t_pose_loss(body_pose):
        """Penalize distance from zero pose (T-pose)"""

        latent_distance = torch.mean(body_pose.pow(2))

        return latent_distance

    # Setup optimizer
    optimizer = torch.optim.Adam([body_pose_t], lr=0.003)

    # run forward function on smpl
    smpl = SmplX.from_npz(path_model, Gender.Female)
    smpl_options = SmplOptions.default()

    # create some random points for the two hand
    hand_target = torch.rand((2, 3)).cuda()

    # Optimization loop
    num_iterations = 100
    iter_nr = 0
    while True:
        optimizer.zero_grad()

        # SMPLX structure: global_orient(3) + body_pose(63) + jaw(3) + leye(3) + reye(3) + left_hand(45) + right_hand(45) = 165
        jaw_pose = torch.zeros(3).cuda()
        leye_pose = torch.zeros(3).cuda()
        reye_pose = torch.zeros(3).cuda()
        left_hand_pose = torch.zeros(45).cuda()
        right_hand_pose = torch.zeros(45).cuda()
        full_pose = torch.cat(
            [
                global_orient_t,  # 3 params
                body_pose_t,  # 63 params
                jaw_pose,  # 3 params
                leye_pose,  # 3 params
                reye_pose,  # 3 params
                left_hand_pose,  # 45 params
                right_hand_pose,  # 45 params
            ]
        )  # Total: 165 params

        # Forward pass
        betas = Betas.from_tensor(betas_t)
        pose = Pose.from_tensors(full_pose, global_trans, UpAxis.Y, SmplType.SmplX)

        # # Benchmarking forward pass
        # min_time=1e10
        # for iter_nr in range(3000):
        #     torch.cuda.synchronize()
        #     start=time.time()
        #     smpl_output = smpl.forward(smpl_options, betas, pose)
        #     torch.cuda.synchronize()
        #     end=time.time()
        #     diff=(end-start)*1000
        #     print("forward pass time:", diff, "ms")
        #     if diff<min_time:
        #         min_time=diff
        # print("minimum forward pass time over 3000 iterations:", min_time, "ms")
        # exit(1)

        smpl_output = smpl.forward(smpl_options, betas, pose)

        loss = 0

        # Compute pose prior loss
        loss = t_pose_loss(full_pose) * 1.0

        # loss on target joints
        joints = smpl_output.joints.to_torch()
        pred_left_hand = joints[21, :]  # left hand joint index 21
        pred_right_hand = joints[20, :]
        loss += torch.mean((pred_left_hand - hand_target[0]) ** 2)
        loss += torch.mean((pred_right_hand - hand_target[1]) ** 2)

        # Backward pass
        loss.backward()
        optimizer.step()

        # if iter_nr % 20 == 0:
        print(f"Iteration {iter_nr}, loss: {loss.item():.6f}")
        if iter_nr % 250 == 0:
            # randomly choose new targets for the hands
            hand_target = torch.rand((2, 3)).cuda()

        # vis body
        mesh = viewer.get_or_create_entity(name="mesh")
        verts = Verts(smpl_output.verts.to_numpy())
        faces = Faces(smpl_output.faces.to_numpy())
        mesh.insert(verts)
        mesh.insert(faces)
        mesh.remove(Normals)  # forces the normals to be recomputed

        # vis target
        mesh = viewer.get_or_create_entity(name="target")
        verts = Verts(hand_target.cpu().numpy())
        mesh.insert(verts)
        mesh.insert(VisPoints(show_points=True, point_size=6.0))

        # vis lines
        mesh = viewer.get_or_create_entity(name="lines")
        line_verts = Verts(
            np.array(
                [
                    pred_left_hand.detach().cpu().numpy(),
                    hand_target[0].detach().cpu().numpy(),
                    pred_right_hand.detach().cpu().numpy(),
                    hand_target[1].detach().cpu().numpy(),
                ]
            )
        )
        line_edges = np.array([[0, 1], [2, 3]])
        mesh.insert(Edges(line_edges.astype(np.uint32)))
        mesh.insert(line_verts)
        mesh.insert(
            VisLines(show_lines=True, line_width=2.0, line_color=[1.0, 0.0, 0.0, 1.0])
        )

        # render
        viewer.start_frame()
        viewer.update()

        # floor
        viewer.get_scene().remove_entity("floor")

        iter_nr += 1
