use crate::common::{
    metadata::{smpl_metadata, SmplMetadata},
    pose::Pose,
    types::SmplType,
};

use super::pose_chunked::PoseChunked;

/// Will remap the pose from a certain model to another one. For example from
/// smplh to smplx. This is because different models have different number of
/// joints for each part.
#[allow(dead_code)]
pub struct PoseRemap {
    origin: SmplType,
    origin_metadata: SmplMetadata,
    destination: SmplType,
    dest_metadata: SmplMetadata,
}
impl PoseRemap {
    pub fn new(origin: SmplType, destination: SmplType) -> Self {
        let origin_metadata = smpl_metadata(&origin);
        let dest_metadata = smpl_metadata(&destination);
        Self {
            origin,
            origin_metadata,
            destination,
            dest_metadata,
        }
    }

    pub fn remap(&self, pose: &Pose) -> Pose {
        //origin_pose -> chunked
        let origin_chunked = PoseChunked::new(pose, &self.origin_metadata);
        //chunked -> destination
        let mut new_pose = origin_chunked.to_pose(&self.dest_metadata, self.destination);
        //TODO ideally the remap would be a remapper.apply(pose) and it wouldn't return
        // a new one so that we avoid things like this where we need to copy state again
        new_pose.retargeted = pose.retargeted;
        new_pose
    }
}
