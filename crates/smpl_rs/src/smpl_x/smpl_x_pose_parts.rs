use crate::{common::pose_parts::PosePart, smpl_x::smpl_x};
use enum_map::EnumMap;
use std::ops::Range;

pub struct PosePartRanges {
    pub parts2jointranges: EnumMap<PosePart, Range<usize>>,
}
#[allow(clippy::range_plus_one)]
impl PosePartRanges {
    pub fn empty() -> Self {
        let mut parts2jointranges: EnumMap<PosePart, Range<usize>> = EnumMap::default();

        //smplx ranges
        let mut cur_joints_added = 0;
        parts2jointranges[PosePart::RootRotation] = 0..1;
        cur_joints_added += 1;
        parts2jointranges[PosePart::Body] = cur_joints_added..cur_joints_added + smpl_x::NUM_BODY_JOINTS;
        cur_joints_added += smpl_x::NUM_BODY_JOINTS;
        parts2jointranges[PosePart::Jaw] = cur_joints_added..cur_joints_added + 1;
        cur_joints_added += 1;
        parts2jointranges[PosePart::LeftEye] = cur_joints_added..cur_joints_added + 1;
        cur_joints_added += 1;
        parts2jointranges[PosePart::RightEye] = cur_joints_added..cur_joints_added + 1;
        cur_joints_added += 1;
        parts2jointranges[PosePart::LeftHand] = cur_joints_added..cur_joints_added + smpl_x::NUM_HAND_JOINTS;
        cur_joints_added += smpl_x::NUM_HAND_JOINTS;
        parts2jointranges[PosePart::RightHand] = cur_joints_added..cur_joints_added + smpl_x::NUM_HAND_JOINTS;
        // cur_joints_added += smpl_x::NUM_HAND_JOINTS;

        Self { parts2jointranges }
    }
}
