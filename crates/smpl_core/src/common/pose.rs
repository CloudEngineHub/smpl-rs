use super::{
    metadata::smpl_metadata,
    pose_override::PoseOverride,
    types::{SmplType, UpAxis},
};
use crate::AppBackend;
use crate::{codec::codec::SmplCodec, common::pose_parts::PosePart, smpl_h::smpl_h, smpl_x::smpl_x};
use burn::{prelude::Backend, tensor::Tensor};
use gloss_utils::bshare::{ToBurn, ToNdArray};
use log::warn;
use nd::concatenate;
use ndarray as nd;
use smpl_utils::numerical::interpolate_angle_tensor;
/// Component for pose
#[derive(Clone, Debug)]
pub struct PoseG<B: Backend> {
    pub device: B::Device,
    pub joint_poses: Tensor<B, 2>,
    pub global_trans: Tensor<B, 1>,
    pub enable_pose_corrective: bool,
    pub up_axis: UpAxis,
    pub smpl_type: SmplType,
    pub non_retargeted_pose: Option<Box<PoseG<B>>>,
    pub retargeted: bool,
}
impl<B: Backend> PoseG<B> {
    pub fn new(joint_poses: Tensor<B, 2>, global_trans: Tensor<B, 1>, up_axis: UpAxis, smpl_type: SmplType) -> Self {
        Self {
            device: joint_poses.device(),
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
        let device = B::Device::default();
        let joint_poses = match smpl_type {
            SmplType::SmplX => Tensor::<B, 2>::zeros([smpl_x::NUM_JOINTS + 1, 3], &device),
            SmplType::SmplH => Tensor::<B, 2>::zeros([smpl_h::NUM_JOINTS + 1, 3], &device),
            _ => panic!("{smpl_type:?} is not yet supported!"),
        };
        let global_trans = Tensor::<B, 1>::zeros([3], &device);
        Self {
            device,
            joint_poses,
            global_trans,
            enable_pose_corrective: false,
            up_axis,
            smpl_type,
            non_retargeted_pose: None,
            retargeted: false,
        }
    }
    pub fn new_from_ndarray(joint_poses: nd::Array2<f32>, global_trans: nd::Array1<f32>, up_axis: UpAxis, smpl_type: SmplType) -> Self {
        let device = B::Device::default();
        Self {
            device: device.clone(),
            joint_poses: joint_poses.into_burn(&device.clone()),
            global_trans: global_trans.into_burn(&device),
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
        Some(Self::new_from_ndarray(joint_poses, body_translation, UpAxis::Y, codec.smpl_type()))
    }
    /// Create new ``Pose`` component from ``.smpl`` file
    #[cfg(not(target_arch = "wasm32"))]
    #[allow(clippy::cast_possible_truncation)]
    pub fn new_from_smpl_file(path: &str) -> Option<Self> {
        let codec = SmplCodec::from_file(path);
        Self::new_from_smpl_codec(&codec)
    }
    pub fn num_active_joints(&self) -> usize {
        self.joint_poses.dims()[0]
    }
    pub fn apply_mask(&mut self, mask: &mut PoseOverride) {
        let metadata = smpl_metadata(&self.smpl_type);
        let dim_joint = self.joint_poses.dims()[1];
        for part in &mask.denied_parts {
            if *part == PosePart::RootTranslation {
                self.global_trans = self.global_trans.clone().slice_fill([..], 0.0);
            } else {
                let range_of_body_part = metadata.parts2jointranges[*part].clone();
                let num_joints = self.joint_poses.dims()[0];
                if range_of_body_part.start < num_joints {
                    let range_of_body_part_clamped = range_of_body_part.start..std::cmp::min(num_joints, range_of_body_part.end);
                    self.joint_poses = self.joint_poses.clone().slice_fill([range_of_body_part_clamped, 0..dim_joint], 0.0);
                }
            }
        }
        let range_left_hand = metadata.parts2jointranges[PosePart::LeftHand].clone();
        let range_right_hand = metadata.parts2jointranges[PosePart::RightHand].clone();
        if let Some(hand_type) = mask.overwrite_hands {
            let original_left = self.joint_poses.clone().slice([range_left_hand.clone(), 0..dim_joint]);
            let original_right = self.joint_poses.clone().slice([range_right_hand.clone(), 0..dim_joint]);
            if mask.original_left_hand.is_none() {
                mask.original_left_hand = Some(original_left.clone().to_ndarray());
            }
            if mask.original_right_hand.is_none() {
                mask.original_right_hand = Some(original_right.clone().to_ndarray());
            }
            self.joint_poses = self
                .joint_poses
                .clone()
                .slice_assign([range_left_hand, 0..dim_joint], metadata.hand_poses[hand_type].left.to_burn(&self.device));
            self.joint_poses = self.joint_poses.clone().slice_assign(
                [range_right_hand, 0..dim_joint],
                metadata.hand_poses[hand_type].right.to_burn(&self.device),
            );
        } else {
            if let Some(left) = mask.original_left_hand.take() {
                self.joint_poses = self
                    .joint_poses
                    .clone()
                    .slice_assign([range_left_hand, 0..dim_joint], left.to_burn(&self.device));
            }
            if let Some(right) = mask.original_right_hand.take() {
                self.joint_poses = self
                    .joint_poses
                    .clone()
                    .slice_assign([range_right_hand, 0..dim_joint], right.to_burn(&self.device));
            }
        }
    }
    /// Interpolate between 2 poses
    #[must_use]
    pub fn interpolate(&self, other_pose: &Self, other_weight: f32) -> PoseG<B> {
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
        let cur_w = 1.0 - other_weight;
        if self.smpl_type == SmplType::SmplPP {
            let non_angle_indices = [27, 28, 37, 38];
            let dim_joint = self.joint_poses.dims()[1];
            let mut new_joint_poses = self.joint_poses.clone();
            #[allow(clippy::range_plus_one)]
            for (i, (cur_angle, other_angle)) in self
                .joint_poses
                .clone()
                .iter_dim(0)
                .zip(other_pose.joint_poses.clone().iter_dim(0))
                .enumerate()
            {
                if non_angle_indices.contains(&i) {
                    new_joint_poses = new_joint_poses
                        .clone()
                        .slice_assign([i..i + 1, 0..dim_joint], cur_w * cur_angle + other_weight * other_angle);
                } else {
                    let new_val = interpolate_angle_tensor(cur_angle.squeeze(0), other_angle.squeeze(0), cur_w, other_weight);
                    new_joint_poses = new_joint_poses.clone().slice_assign([i..i + 1, 0..dim_joint], new_val.unsqueeze());
                }
            }
            let new_global_trans = cur_w * self.global_trans.clone() + other_weight * other_pose.global_trans.clone();
            return PoseG::new(new_joint_poses, new_global_trans, self.up_axis, self.smpl_type);
        }
        let new_global_trans = cur_w * self.global_trans.clone() + other_weight * other_pose.global_trans.clone();
        let all_joints = Tensor::cat(vec![self.joint_poses.clone(), other_pose.joint_poses.clone()], 0);
        let all_quats = smpl_utils::numerical::axis_angle_to_quaternion(all_joints);
        let vec_quats = all_quats.split(self.joint_poses.dims()[0], 0);
        let cur_quats = vec_quats[0].clone();
        let other_quats = vec_quats[1].clone();
        let interpolated_quats = smpl_utils::numerical::quaternion_interpolate_lerp(cur_quats, other_quats, other_weight);
        let new_joint_poses = smpl_utils::numerical::quaternion_to_axis_angle_fast(interpolated_quats);
        PoseG::new(new_joint_poses, new_global_trans, self.up_axis, self.smpl_type)
    }
}
pub type Pose = PoseG<AppBackend>;
