use crate::common::{
    metadata::SmplMetadata,
    pose::PoseG,
    pose_parts::PosePart,
    types::{SmplType, UpAxis},
};
use burn::{
    prelude::Backend,
    tensor::{Float, Tensor},
};
use std::ops::Range;
/// Chunk ``Pose`` into various pose parts
#[derive(Debug)]
pub struct PoseChunked<B: Backend> {
    pub device: B::Device,
    pub global_trans: Tensor<B, 2>,
    pub global_orient: Option<Tensor<B, 2>>,
    pub body_pose: Option<Tensor<B, 2>>,
    pub left_hand_pose: Option<Tensor<B, 2>>,
    pub right_hand_pose: Option<Tensor<B, 2>>,
    pub jaw_pose: Option<Tensor<B, 2>>,
    pub left_eye_pose: Option<Tensor<B, 2>>,
    pub right_eye_pose: Option<Tensor<B, 2>>,
    pub up_axis: UpAxis,
    pub smpl_type: SmplType,
}
impl<B: Backend> Default for PoseChunked<B> {
    fn default() -> Self {
        let device = B::Device::default();
        let global_trans = Tensor::<B, 2, Float>::zeros([1, 3], &device.clone());
        Self {
            device,
            global_trans,
            global_orient: None,
            body_pose: None,
            left_hand_pose: None,
            right_hand_pose: None,
            jaw_pose: None,
            left_eye_pose: None,
            right_eye_pose: None,
            up_axis: UpAxis::Y,
            smpl_type: SmplType::SmplX,
        }
    }
}
impl<B: Backend> PoseChunked<B> {
    #[allow(clippy::missing_panics_doc)]
    pub fn new(pose: &PoseG<B>, metadata: &SmplMetadata) -> Self {
        if pose.smpl_type == SmplType::SmplPP {
            return Self {
                device: pose.device.clone(),
                global_trans: pose.global_trans.clone().reshape([1, 3]),
                global_orient: None,
                body_pose: Some(pose.joint_poses.clone()),
                left_hand_pose: None,
                right_hand_pose: None,
                jaw_pose: None,
                left_eye_pose: None,
                right_eye_pose: None,
                up_axis: pose.up_axis,
                smpl_type: pose.smpl_type,
            };
        }
        let p2r = &metadata.parts2jointranges;
        let joint_poses = &pose.joint_poses;
        let max_range = 0..joint_poses.dims()[0];
        let jdim = joint_poses.dims()[1];
        #[allow(clippy::if_same_then_else)]
        let slice_or_none = |joints: Tensor<B, 2>, slice: &Range<usize>, max: &Range<usize>, jdim: usize| -> Option<Tensor<B, 2>> {
            if slice.end > max.end {
                None
            } else if slice.start == 0 && slice.end == 0 {
                None
            } else {
                Some(joints.clone().slice([slice.start..slice.end, 0..jdim]))
            }
        };
        let global_orient = slice_or_none(joint_poses.clone(), &p2r[PosePart::RootRotation], &max_range, jdim);
        let body_pose = slice_or_none(joint_poses.clone(), &p2r[PosePart::Body], &max_range, jdim);
        let left_hand_pose = slice_or_none(joint_poses.clone(), &p2r[PosePart::LeftHand], &max_range, jdim);
        let right_hand_pose = slice_or_none(joint_poses.clone(), &p2r[PosePart::RightHand], &max_range, jdim);
        let jaw_pose = slice_or_none(joint_poses.clone(), &p2r[PosePart::Jaw], &max_range, jdim);
        let left_eye_pose = slice_or_none(joint_poses.clone(), &p2r[PosePart::LeftEye], &max_range, jdim);
        let right_eye_pose = slice_or_none(joint_poses.clone(), &p2r[PosePart::RightEye], &max_range, jdim);
        Self {
            device: pose.device.clone(),
            global_trans: pose.global_trans.clone().reshape([1, 3]),
            global_orient,
            body_pose,
            left_hand_pose,
            right_hand_pose,
            jaw_pose,
            left_eye_pose,
            right_eye_pose,
            up_axis: pose.up_axis,
            smpl_type: pose.smpl_type,
        }
    }
    #[allow(clippy::missing_panics_doc)]
    pub fn to_pose(&self, metadata: &SmplMetadata, smpl_type: SmplType) -> PoseG<B> {
        if smpl_type == SmplType::SmplPP {
            let mut pose = PoseG::<B>::new_empty(self.up_axis, smpl_type);
            let zeros = Tensor::<B, 2, Float>::zeros([46, 1], &self.device.clone());
            pose.joint_poses = self.body_pose.as_ref().unwrap_or(&zeros).clone();
            pose.global_trans = self.global_trans.clone().reshape([3]);
            return pose;
        }
        let mut pose = PoseG::<B>::new_empty(self.up_axis, smpl_type);
        let zeros = Tensor::<B, 2, Float>::zeros([1, 3], &self.device.clone());
        pose.global_trans = self.global_trans.clone().reshape([3]);
        let jdim = pose.joint_poses.dims()[1];
        pose.joint_poses = pose.joint_poses.clone().slice_assign(
            [metadata.parts2jointranges[PosePart::RootRotation].clone(), 0..jdim],
            self.global_orient.as_ref().unwrap_or(&zeros).clone(),
        );
        pose.joint_poses = pose.joint_poses.clone().slice_assign(
            [metadata.parts2jointranges[PosePart::Body].clone(), 0..jdim],
            self.body_pose.as_ref().unwrap_or(&zeros).clone(),
        );
        pose.joint_poses = pose.joint_poses.clone().slice_assign(
            [metadata.parts2jointranges[PosePart::LeftHand].clone(), 0..jdim],
            self.left_hand_pose.as_ref().unwrap_or(&zeros).clone(),
        );
        pose.joint_poses = pose.joint_poses.clone().slice_assign(
            [metadata.parts2jointranges[PosePart::RightHand].clone(), 0..jdim],
            self.right_hand_pose.as_ref().unwrap_or(&zeros).clone(),
        );
        pose.joint_poses = pose.joint_poses.clone().slice_assign(
            [metadata.parts2jointranges[PosePart::Jaw].clone(), 0..jdim],
            self.jaw_pose.as_ref().unwrap_or(&zeros).clone(),
        );
        pose.joint_poses = pose.joint_poses.clone().slice_assign(
            [metadata.parts2jointranges[PosePart::LeftEye].clone(), 0..jdim],
            self.left_eye_pose.as_ref().unwrap_or(&zeros).clone(),
        );
        pose.joint_poses = pose.joint_poses.clone().slice_assign(
            [metadata.parts2jointranges[PosePart::RightEye].clone(), 0..jdim],
            self.right_eye_pose.as_ref().unwrap_or(&zeros).clone(),
        );
        pose
    }
}
