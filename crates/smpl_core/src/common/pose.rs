use super::{
    metadata::smpl_metadata,
    pose_override::PoseOverride,
    types::{SmplType, UpAxis},
};
use crate::{codec::codec::SmplCodec, common::pose_parts::PosePart, smpl_h::smpl_h, smpl_x::smpl_x};
use gloss_utils::nshare::ToNalgebra;
use log::warn;
use nalgebra as na;
use nd::concatenate;
use ndarray as nd;
use ndarray::prelude::*;
use smpl_utils::numerical::interpolate_angle;
/// Component for pose
#[derive(Clone, Debug)]
pub struct Pose {
    pub joint_poses: nd::Array2<f32>,
    pub global_trans: nd::Array1<f32>,
    pub enable_pose_corrective: bool,
    pub up_axis: UpAxis,
    pub smpl_type: SmplType,
    pub non_retargeted_pose: Option<Box<Pose>>,
    pub retargeted: bool,
}
impl Pose {
    pub fn new(joint_poses: nd::Array2<f32>, global_trans: nd::Array1<f32>, up_axis: UpAxis, smpl_type: SmplType) -> Self {
        Self {
            joint_poses,
            global_trans,
            enable_pose_corrective: false,
            up_axis,
            smpl_type,
            non_retargeted_pose: None,
            retargeted: false,
        }
    }
    pub fn new_empty(up_axis: UpAxis, smpl_type: SmplType) -> Self {
        let joint_poses = match smpl_type {
            SmplType::SmplX => ndarray::Array2::<f32>::zeros((smpl_x::NUM_JOINTS + 1, 3)),
            SmplType::SmplH => ndarray::Array2::<f32>::zeros((smpl_h::NUM_JOINTS + 1, 3)),
            _ => panic!("{smpl_type:?} is not yet supported!"),
        };
        let global_trans = ndarray::Array1::<f32>::zeros(3);
        Self {
            joint_poses,
            global_trans,
            enable_pose_corrective: false,
            up_axis,
            smpl_type,
            non_retargeted_pose: None,
            retargeted: false,
        }
    }
    /// Create a new ``Pose`` component from ``SmplCodec``
    /// # Panics
    /// Will panic if the ``nr_frames`` is different than 1
    #[allow(clippy::cast_sign_loss)]
    pub fn new_from_smpl_codec(codec: &SmplCodec) -> Option<Self> {
        let nr_frames = codec.frame_count as u32;
        assert_eq!(nr_frames, 1, "For a pose the nr of frames in the codec has to be 1");
        let metadata = smpl_metadata(&codec.smpl_type());
        let body_translation = codec
            .body_translation
            .as_ref()
            .unwrap_or(&ndarray::Array2::<f32>::zeros((1, 3)))
            .index_axis(nd::Axis(0), 0)
            .to_owned();
        let body_pose = codec.body_pose.as_ref()?.index_axis(nd::Axis(0), 0).to_owned();
        let head_pose = codec
            .head_pose
            .as_ref()
            .unwrap_or(&ndarray::Array3::<f32>::zeros((1, metadata.num_face_joints, 3)))
            .index_axis(nd::Axis(0), 0)
            .into_owned();
        let left_hand_pose = codec
            .left_hand_pose
            .as_ref()
            .unwrap_or(&ndarray::Array3::<f32>::zeros((1, metadata.num_hand_joints, 3)))
            .index_axis(nd::Axis(0), 0)
            .into_owned();
        let right_hand_pose = codec
            .right_hand_pose
            .as_ref()
            .unwrap_or(&ndarray::Array3::<f32>::zeros((1, metadata.num_hand_joints, 3)))
            .index_axis(nd::Axis(0), 0)
            .into_owned();
        let joint_poses = concatenate(
            nd::Axis(0),
            &[body_pose.view(), head_pose.view(), left_hand_pose.view(), right_hand_pose.view()],
        )
        .unwrap();
        Some(Self::new(joint_poses, body_translation, UpAxis::Y, codec.smpl_type()))
    }
    /// Create new ``Pose`` component from ``.smpl`` file
    #[cfg(not(target_arch = "wasm32"))]
    #[allow(clippy::cast_possible_truncation)]
    pub fn new_from_smpl_file(path: &str) -> Option<Self> {
        let codec = SmplCodec::from_file(path);
        Self::new_from_smpl_codec(&codec)
    }
    pub fn num_active_joints(&self) -> usize {
        self.joint_poses.dim().0
    }
    pub fn apply_mask(&mut self, mask: &mut PoseOverride) {
        let metadata = smpl_metadata(&self.smpl_type);
        for part in &mask.denied_parts {
            if *part == PosePart::RootTranslation {
                self.global_trans.fill(0.0);
            } else {
                let range_of_body_part = metadata.parts2jointranges[*part].clone();
                let num_joints = self.joint_poses.dim().0;
                if range_of_body_part.start < num_joints {
                    let range_of_body_part_clamped = range_of_body_part.start..std::cmp::min(num_joints, range_of_body_part.end);
                    self.joint_poses.slice_mut(s![range_of_body_part_clamped, ..]).fill(0.0);
                }
            }
        }
        let range_left_hand = metadata.parts2jointranges[PosePart::LeftHand].clone();
        let range_right_hand = metadata.parts2jointranges[PosePart::RightHand].clone();
        if let Some(hand_type) = mask.overwrite_hands {
            let original_left = self.joint_poses.slice(s![range_left_hand.clone(), ..]);
            let original_right = self.joint_poses.slice(s![range_right_hand.clone(), ..]);
            if mask.original_left_hand.is_none() {
                mask.original_left_hand = Some(original_left.to_owned());
            }
            if mask.original_right_hand.is_none() {
                mask.original_right_hand = Some(original_right.to_owned());
            }
            self.joint_poses
                .slice_mut(s![range_left_hand, ..])
                .assign(&metadata.hand_poses[hand_type].left);
            self.joint_poses
                .slice_mut(s![range_right_hand, ..])
                .assign(&metadata.hand_poses[hand_type].right);
        } else {
            if let Some(left) = mask.original_left_hand.take() {
                self.joint_poses.slice_mut(s![range_left_hand, ..]).assign(&left);
            }
            if let Some(right) = mask.original_right_hand.take() {
                self.joint_poses.slice_mut(s![range_right_hand, ..]).assign(&right);
            }
        }
    }
    /// Interpolate between 2 poses
    #[must_use]
    pub fn interpolate(&self, other_pose: &Self, other_weight: f32) -> Pose {
        if !(0.0..=1.0).contains(&other_weight) {
            warn!("pose interpolation weight is outside the [0,1] range, will clamp. Weight is {other_weight}");
        }
        let other_weight = other_weight.clamp(0.0, 1.0);
        assert!(
            self.smpl_type == other_pose.smpl_type,
            "We can only interpolate to a pose of the same type. Origin: {:?}. Dest: {:?}",
            self.smpl_type,
            other_pose.smpl_type
        );
        let non_angle_indices = [27, 28, 37, 38];
        if self.smpl_type == SmplType::SmplPP {
            let cur_w = 1.0 - other_weight;
            let mut new_joint_poses = self.joint_poses.clone();
            for (i, ((cur_angle, other_angle), new_angle)) in self
                .joint_poses
                .iter()
                .zip(other_pose.joint_poses.iter())
                .zip(new_joint_poses.iter_mut())
                .enumerate()
            {
                if non_angle_indices.contains(&i) {
                    *new_angle = cur_w * cur_angle + other_weight * other_angle;
                } else {
                    *new_angle = interpolate_angle(*cur_angle, *other_angle, cur_w, other_weight);
                }
            }
            let new_global_trans = cur_w * &self.global_trans + other_weight * &other_pose.global_trans;
            return Pose::new(new_joint_poses, new_global_trans, self.up_axis, self.smpl_type);
        }
        let cur_w = 1.0 - other_weight;
        let new_global_trans = cur_w * &self.global_trans + other_weight * &other_pose.global_trans;
        let cur_pose_axis_angle_na = self.joint_poses.view().into_nalgebra();
        let other_pose_axis_angle_na = other_pose.joint_poses.view().into_nalgebra();
        let mut new_joint_poses = nd::Array2::<f32>::zeros((self.num_active_joints(), 3));
        for ((cur_axis, other_axis), mut new_joint) in cur_pose_axis_angle_na
            .row_iter()
            .zip(other_pose_axis_angle_na.row_iter())
            .zip(new_joint_poses.axis_iter_mut(nd::Axis(0)))
        {
            let cur_vec = na::Vector3::new(cur_axis[0], cur_axis[1], cur_axis[2]);
            let mut cur_q = na::UnitQuaternion::from_axis_angle(&na::UnitVector3::new_normalize(cur_vec), cur_axis.norm());
            if cur_axis.norm() == 0.0 {
                cur_q = na::UnitQuaternion::default();
            }
            let other_vec = na::Vector3::new(other_axis[0], other_axis[1], other_axis[2]);
            let mut other_q = na::UnitQuaternion::from_axis_angle(&na::UnitVector3::new_normalize(other_vec), other_axis.norm());
            if other_axis.norm() == 0.0 {
                other_q = na::UnitQuaternion::default();
            }
            let new_q: na::Unit<na::Quaternion<f32>> = cur_q.slerp(&other_q, other_weight);
            let axis_opt = new_q.axis();
            let angle = new_q.angle();
            if let Some(axis) = axis_opt {
                new_joint.assign(&array![axis.x * angle, axis.y * angle, axis.z * angle]);
            }
        }
        Pose::new(new_joint_poses, new_global_trans, self.up_axis, self.smpl_type)
    }
}
