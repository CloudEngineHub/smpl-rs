use crate::{common::pose_parts::PosePart, smpl_h::smpl_h};
use enum_map::EnumMap;
use std::ops::Range;
pub struct PosePartRanges {
    pub parts2jointranges: EnumMap<PosePart, Range<usize>>,
}
impl PosePartRanges {
    pub fn empty() -> Self {
        let mut parts2jointranges: EnumMap<PosePart, Range<usize>> = EnumMap::default();
        let mut cur_joints_added = 0;
        parts2jointranges[PosePart::RootRotation] = 0..1;
        cur_joints_added += 1;
        parts2jointranges[PosePart::Body] = cur_joints_added..cur_joints_added
            + smpl_h::NUM_BODY_JOINTS;
        cur_joints_added += smpl_h::NUM_BODY_JOINTS;
        parts2jointranges[PosePart::LeftHand] = cur_joints_added..cur_joints_added
            + smpl_h::NUM_HAND_JOINTS;
        cur_joints_added += smpl_h::NUM_HAND_JOINTS;
        parts2jointranges[PosePart::RightHand] = cur_joints_added..cur_joints_added
            + smpl_h::NUM_HAND_JOINTS;
        Self { parts2jointranges }
    }
}
