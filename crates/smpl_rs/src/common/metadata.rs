use super::{
    pose_hands::{HandPair, HandType},
    pose_parts::PosePart,
    types::SmplType,
};
use crate::{
    smpl_h::{smpl_h, smpl_h_pose_parts},
    smpl_x::{smpl_x, smpl_x_hands::SmplXHands, smpl_x_pose_parts},
};
use enum_map::EnumMap;
use std::ops::Range;
#[derive(Default)]
pub struct SmplMetadata {
    pub num_body_joints: usize,
    pub num_hand_joints: usize,
    pub num_face_joints: usize,
    pub num_joints: usize,
    pub num_pose_blend_shapes: usize,
    pub expression_space_dim: usize,
    pub num_verts: usize,
    pub num_verts_uv_mesh: usize,
    pub num_faces: usize,
    pub shape_space_dim: usize,
    pub hand_poses: EnumMap<HandType, HandPair>,
    pub parts2jointranges: EnumMap<PosePart, Range<usize>>,
    pub joint_parents: Vec<u32>,
    pub joint_names: Vec<String>,
    pub pose_dim: usize,
}
/// # Panics
/// Will panic if the ``smpl_type`` is unknown
pub fn smpl_metadata(smpl_type: &SmplType) -> SmplMetadata {
    match smpl_type {
        SmplType::SmplH => {
            let parts = smpl_h_pose_parts::PosePartRanges::empty();
            SmplMetadata {
                num_body_joints: smpl_h::NUM_BODY_JOINTS,
                num_hand_joints: smpl_h::NUM_HAND_JOINTS,
                num_joints: smpl_h::NUM_JOINTS,
                num_pose_blend_shapes: smpl_h::NUM_POSE_BLEND_SHAPES,
                parts2jointranges: parts.parts2jointranges,
                joint_parents: smpl_h::PARENT_ID_PER_JOINT.to_vec(),
                joint_names: smpl_h::JOINT_NAMES.map(std::string::ToString::to_string).to_vec(),
                ..Default::default()
            }
        }
        SmplType::SmplX => {
            let parts = smpl_x_pose_parts::PosePartRanges::empty();
            SmplMetadata {
                num_body_joints: smpl_x::NUM_BODY_JOINTS,
                num_hand_joints: smpl_x::NUM_HAND_JOINTS,
                num_face_joints: smpl_x::NUM_FACE_JOINTS,
                num_joints: smpl_x::NUM_JOINTS,
                num_pose_blend_shapes: smpl_x::NUM_POSE_BLEND_SHAPES,
                expression_space_dim: smpl_x::EXPRESSION_SPACE_DIM,
                num_verts: smpl_x::NUM_VERTS,
                num_verts_uv_mesh: smpl_x::NUM_VERTS_UV_MESH,
                num_faces: smpl_x::NUM_FACES,
                shape_space_dim: smpl_x::SHAPE_SPACE_DIM,
                hand_poses: SmplXHands::default().type2pose,
                parts2jointranges: parts.parts2jointranges,
                joint_parents: smpl_x::PARENT_ID_PER_JOINT.to_vec(),
                joint_names: smpl_x::JOINT_NAMES.map(std::string::ToString::to_string).to_vec(),
                ..Default::default()
            }
        }
        _ => panic!("Unknown Smpl Model"),
    }
}
