use crate::common::{
    metadata::SmplMetadata,
    pose::Pose,
    pose_parts::PosePart,
    types::{SmplType, UpAxis},
};
use ndarray as nd;
use ndarray::prelude::*;
use std::ops::Range;

/// Chunk ``Pose`` into various pose parts
#[derive(Debug)]
pub struct PoseChunked {
    pub global_trans: nd::Array2<f32>,            // (1,3) global translation
    pub global_orient: Option<nd::Array2<f32>>,   // (1,3) global rotation as axis angle
    pub body_pose: Option<nd::Array2<f32>>,       // (NUM_BODY_JOINTS, 3) rotation of each body joint as axis angle
    pub left_hand_pose: Option<nd::Array2<f32>>,  // (NUM_HAND_JOINTS, 3) rotation of each hand joint as axis angle
    pub right_hand_pose: Option<nd::Array2<f32>>, // (NUM_HAND_JOINTS, 3) rotation of each hand joint as axis angle
    pub jaw_pose: Option<nd::Array2<f32>>,        /* (1,3) rotation of the jaw //TODO call this jaw_orient or jaw_rotation to be more consistent
                                                   * with global orient */
    pub left_eye_pose: Option<nd::Array2<f32>>,  // (1,3) rotation of left eye as axis angle
    pub right_eye_pose: Option<nd::Array2<f32>>, // (1,3) rotation of right eye as axis angle
    pub up_axis: UpAxis,
    pub smpl_type: SmplType,
}
impl Default for PoseChunked {
    fn default() -> Self {
        let global_trans = ndarray::Array2::<f32>::zeros((1, 3));

        Self {
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

impl PoseChunked {
    #[allow(clippy::missing_panics_doc)]
    pub fn new(pose: &Pose, metadata: &SmplMetadata) -> Self {
        if pose.smpl_type == SmplType::SmplPP {
            return Self {
                global_trans: pose.global_trans.to_shape((1, 3)).unwrap().to_owned(),
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

        let max_range = 0..joint_poses.dim().0;

        let clamp_closure = |lhs: &Range<usize>, rhs: &Range<usize>| -> Range<usize> {
            //if we go past the end, set the whole range to 0..0
            if lhs.end > rhs.end {
                0..0
            } else {
                lhs.clone()
            }
        };

        let global_orient_clamped = clamp_closure(&p2r[PosePart::RootRotation], &max_range);
        let mut global_orient = Some(joint_poses.slice(s![global_orient_clamped, ..]).to_owned());

        let body_clamped = clamp_closure(&p2r[PosePart::Body], &max_range);
        let mut body_pose = Some(joint_poses.slice(s![body_clamped, ..]).to_owned());

        let left_hand_clamped = clamp_closure(&p2r[PosePart::LeftHand], &max_range);
        let mut left_hand_pose = Some(joint_poses.slice(s![left_hand_clamped, ..]).to_owned());

        let right_hand_clamped = clamp_closure(&p2r[PosePart::RightHand], &max_range);
        let mut right_hand_pose = Some(joint_poses.slice(s![right_hand_clamped, ..]).to_owned());

        let jaw_clamped = clamp_closure(&p2r[PosePart::Jaw], &max_range);
        let mut jaw_pose = Some(joint_poses.slice(s![jaw_clamped, ..]).to_owned());

        let left_eye_clamped = clamp_closure(&p2r[PosePart::LeftEye], &max_range);
        let mut left_eye_pose = Some(joint_poses.slice(s![left_eye_clamped, ..]).to_owned());

        let right_eye_clamped = clamp_closure(&p2r[PosePart::RightEye], &max_range);
        let mut right_eye_pose = Some(joint_poses.slice(s![right_eye_clamped, ..]).to_owned());

        // whatever chunks are of shape (0,3), we set them to 1,3 so when they get put
        // into a pose they can broadcast
        if global_orient.as_ref().unwrap().dim().0 == 0 {
            global_orient = None;
        }
        if body_pose.as_ref().unwrap().dim().0 == 0 {
            body_pose = None;
        }
        if left_hand_pose.as_ref().unwrap().dim().0 == 0 {
            left_hand_pose = None;
        }
        if right_hand_pose.as_ref().unwrap().dim().0 == 0 {
            right_hand_pose = None;
        }
        if jaw_pose.as_ref().unwrap().dim().0 == 0 {
            jaw_pose = None;
        }
        if left_eye_pose.as_ref().unwrap().dim().0 == 0 {
            left_eye_pose = None;
        }
        if right_eye_pose.as_ref().unwrap().dim().0 == 0 {
            right_eye_pose = None;
        }

        Self {
            global_trans: pose.global_trans.to_shape((1, 3)).unwrap().to_owned(),
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
    pub fn to_pose(&self, metadata: &SmplMetadata, smpl_type: SmplType) -> Pose {
        // If the SMPL type is SmplPP, assign the entire pose directly
        if smpl_type == SmplType::SmplPP {
            let zeros = nd::Array2::<f32>::zeros((46, 1));
            let mut pose = Pose::new_empty(self.up_axis, smpl_type);
            pose.joint_poses.assign(self.body_pose.as_ref().unwrap_or(&zeros));
            pose.global_trans.assign(&self.global_trans.to_shape(3).unwrap().to_owned());
            return pose;
        }
        // log!("to_pose");
        let mut pose = Pose::new_empty(self.up_axis, smpl_type);
        let zeros = nd::Array2::<f32>::zeros((1, 3));
        pose.global_trans = self.global_trans.to_shape(3).unwrap().to_owned();

        pose.joint_poses
            .slice_mut(s![metadata.parts2jointranges[PosePart::RootRotation].clone(), ..])
            .assign(self.global_orient.as_ref().unwrap_or(&zeros));
        pose.joint_poses
            .slice_mut(s![metadata.parts2jointranges[PosePart::Body].clone(), ..])
            .assign(self.body_pose.as_ref().unwrap_or(&zeros));
        pose.joint_poses
            .slice_mut(s![metadata.parts2jointranges[PosePart::LeftHand].clone(), ..])
            .assign(self.left_hand_pose.as_ref().unwrap_or(&zeros));
        pose.joint_poses
            .slice_mut(s![metadata.parts2jointranges[PosePart::RightHand].clone(), ..])
            .assign(self.right_hand_pose.as_ref().unwrap_or(&zeros));
        pose.joint_poses
            .slice_mut(s![metadata.parts2jointranges[PosePart::Jaw].clone(), ..])
            .assign(self.jaw_pose.as_ref().unwrap_or(&zeros));
        pose.joint_poses
            .slice_mut(s![metadata.parts2jointranges[PosePart::LeftEye].clone(), ..])
            .assign(self.left_eye_pose.as_ref().unwrap_or(&zeros));
        pose.joint_poses
            .slice_mut(s![metadata.parts2jointranges[PosePart::RightEye].clone(), ..])
            .assign(self.right_eye_pose.as_ref().unwrap_or(&zeros));
        pose
    }
}
