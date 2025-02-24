// TODO: WHy does this exist if its commented out?

// use crate::common::animation::Animation;
// use crate::common::pose_parts::PosePart;
// use crate::common::types::{SmplType, UpAxis};
// use crate::common::{metadata::SmplMetadata, pose::Pose};
// use ndarray as nd;
// use ndarray::prelude::*;
// use std::ops::Range;

// pub struct AnimChunked {
//     pub global_trans: nd::Array2<f32>, // (nr_frames,3) global translation
//     pub global_orient: nd::Array3<f32>, // (nr_frames,1,3) global rotation as
// axis angle     pub body_pose: nd::Array3<f32>, // (nr_frames,NUM_BODY_JOINTS,
// 3) rotation of each body joint as axis angle     pub left_hand_pose:
// nd::Array3<f32>, // (nr_frames,NUM_HAND_JOINTS, 3) rotation of each hand
// joint as axis angle     pub right_hand_pose: nd::Array3<f32>, //
// (nr_frames,NUM_HAND_JOINTS, 3) rotation of each hand joint as axis angle
//     pub jaw_pose: nd::Array3<f32>, // (nr_frames,1,3) rotation of the jaw
// //TODO call this jaw_orient or jaw_rotation to be more consistent with global
// orient     pub left_eye_pose: nd::Array3<f32>, // (nr_frames,1,3) rotation of
// left eye as axis angle     pub right_eye_pose: nd::Array3<f32>, //
// (nr_frames,1,3) rotation of right eye as axis angle     pub up_axis: UpAxis,
//     pub smpl_type: SmplType,
// }
// impl Default for AnimChunked {
//     fn default() -> Self {
//         let global_trans = ndarray::Array2::<f32>::zeros((1, 3));
//         let global_orient = ndarray::Array3::<f32>::zeros((1, 1, 3));
//         let body_pose = ndarray::Array3::<f32>::zeros((1, 1, 3));
//         let left_hand_pose = ndarray::Array3::<f32>::zeros((1, 1, 3));
//         let right_hand_pose = ndarray::Array3::<f32>::zeros((1, 1, 3));
//         let jaw_pose = ndarray::Array3::<f32>::zeros((1, 1, 3));
//         let left_eye_pose = ndarray::Array3::<f32>::zeros((1, 1, 3));
//         let right_eye_pose = ndarray::Array3::<f32>::zeros((1, 1, 3));

//         Self {
//             global_trans,
//             global_orient,
//             body_pose,
//             left_hand_pose,
//             right_hand_pose,
//             jaw_pose,
//             left_eye_pose,
//             right_eye_pose,
//             up_axis: UpAxis::Y,
//             smpl_type: SmplType::SmplX,
//         }
//     }
// }

// impl AnimChunked {
//     #[allow(clippy::missing_panics_doc)]
//     pub fn new(anim: &Animation, metadata: &SmplMetadata) -> Self {
//         let p2r = &metadata.parts2jointranges;
//         let joint_poses = &anim.per_frame_joint_poses;

//         let max_range = 0..joint_poses.dim().1;
//         // log!("max_range {:?}", max_range);
//         // log!("joint_poses {:?}", joint_poses.shape());

//         let clamp_closure = |lhs: &Range<usize>, rhs: &Range<usize>| ->
// Range<usize> {             // let end = std::cmp::min(lhs.end, rhs.end);
//             // let start = std::cmp::min(lhs.start, rhs.end);
//             //if we go past the end, set the whole range to 0..0
//             if lhs.end > rhs.end {
//                 0..0
//             } else {
//                 lhs.clone()
//             }
//         };

//         let global_orient_clamped =
// clamp_closure(&p2r[PosePart::RootRotation], &max_range);         //
// log!("global_orient_clamped {:?}", global_orient_clamped);         let mut
// global_orient = joint_poses             .slice(s![.., global_orient_clamped,
// ..])             .to_owned();

//         let body_clamped = clamp_closure(&p2r[PosePart::Body], &max_range);
//         // log!("body_clamped {:?}", body_clamped);
//         let mut body_pose = joint_poses.slice(s![.., body_clamped,
// ..]).to_owned();

//         let left_hand_clamped = clamp_closure(&p2r[PosePart::LeftHand],
// &max_range);         // log!("left_hand_clamped {:?}", left_hand_clamped);
//         let mut left_hand_pose = joint_poses.slice(s![.., left_hand_clamped,
// ..]).to_owned();

//         let right_hand_clamped = clamp_closure(&p2r[PosePart::RightHand],
// &max_range);         // log!("right_hand_clamped {:?}", right_hand_clamped);
//         let mut right_hand_pose = joint_poses.slice(s![..,
// right_hand_clamped, ..]).to_owned();

//         let jaw_clamped = clamp_closure(&p2r[PosePart::Jaw], &max_range);
//         // log!("jaw_clamped {:?}", jaw_clamped);
//         let mut jaw_pose = joint_poses.slice(s![.., jaw_clamped,
// ..]).to_owned();

//         let left_eye_clamped = clamp_closure(&p2r[PosePart::LeftEye],
// &max_range);         // log!("left_eye_clamped {:?}", left_eye_clamped);
//         let mut left_eye_pose = joint_poses.slice(s![.., left_eye_clamped,
// ..]).to_owned();

//         let right_eye_clamped = clamp_closure(&p2r[PosePart::RightEye],
// &max_range);         // log!("right_eye_clamped {:?}", right_eye_clamped);
//         let mut right_eye_pose = joint_poses.slice(s![.., right_eye_clamped,
// ..]).to_owned();

//         //whatever chunks are of shape (nr_frames,0,3), we set them to 1,1,3
// so when they get put into a pose they can broadcast         if
// global_orient.dim().1 == 0 {             global_orient =
// ndarray::Array3::<f32>::zeros((1, 1, 3));         }
//         if body_pose.dim().1 == 0 {
//             body_pose = ndarray::Array3::<f32>::zeros((1, 1, 3));
//         }
//         if left_hand_pose.dim().1 == 0 {
//             left_hand_pose = ndarray::Array3::<f32>::zeros((1, 1, 3));
//         }
//         if right_hand_pose.dim().1 == 0 {
//             right_hand_pose = ndarray::Array3::<f32>::zeros((1, 1, 3));
//         }
//         if jaw_pose.dim().1 == 0 {
//             jaw_pose = ndarray::Array3::<f32>::zeros((1, 1, 3));
//         }
//         if left_eye_pose.dim().1 == 0 {
//             left_eye_pose = ndarray::Array3::<f32>::zeros((1, 1, 3));
//         }
//         if right_eye_pose.dim().1 == 0 {
//             right_eye_pose = ndarray::Array3::<f32>::zeros((1, 1, 3));
//         }

//         Self {
//             global_trans: anim.per_frame_root_trans.to_owned(),
//             global_orient,
//             body_pose,
//             left_hand_pose,
//             right_hand_pose,
//             jaw_pose,
//             left_eye_pose,
//             right_eye_pose,
//             up_axis: anim.config.up_axis,
//             smpl_type: anim.config.smpl_type,
//         }
//     }

//     #[allow(clippy::missing_panics_doc)]
//     pub fn to_pose(&self, metadata: &SmplMetadata, smpl_type: SmplType) ->
// Pose {         // log!("to_pose");
//         let mut pose = Pose::new_empty(metadata.num_joints + 1, self.up_axis,
// smpl_type);         pose.global_trans =
// self.global_trans.to_shape(3).unwrap().to_owned();         pose.joint_poses
//             .slice_mut(s![
//                 metadata.parts2jointranges[PosePart::RootRotation].clone(),
//                 ..
//             ])
//             .assign(&self.global_orient);
//         pose.joint_poses
//             .slice_mut(s![metadata.parts2jointranges[PosePart::Body].clone(),
// ..])             .assign(&self.body_pose);
//         pose.joint_poses
//             .slice_mut(s![
//                 metadata.parts2jointranges[PosePart::LeftHand].clone(),
//                 ..
//             ])
//             .assign(&self.left_hand_pose);
//         pose.joint_poses
//             .slice_mut(s![
//                 metadata.parts2jointranges[PosePart::RightHand].clone(),
//                 ..
//             ])
//             .assign(&self.right_hand_pose);
//         pose.joint_poses
//             .slice_mut(s![metadata.parts2jointranges[PosePart::Jaw].clone(),
// ..])             .assign(&self.jaw_pose);
//         pose.joint_poses
//             .slice_mut(s![
//                 metadata.parts2jointranges[PosePart::LeftEye].clone(),
//                 ..
//             ])
//             .assign(&self.left_eye_pose);
//         pose.joint_poses
//             .slice_mut(s![
//                 metadata.parts2jointranges[PosePart::RightEye].clone(),
//                 ..
//             ])
//             .assign(&self.right_eye_pose);
//         pose
//     }
// }
