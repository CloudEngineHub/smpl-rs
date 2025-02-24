use super::types::{Gender, SmplType};
use crate::codec::codec::SmplCodec;

/// Params that will influence various components. For example `pose_corrective`
/// will influence animation, Pose, `PoseDestination`
#[derive(Clone)]
pub struct SmplParams {
    pub smpl_type: SmplType,
    pub gender: Gender,
    pub enable_pose_corrective: bool,
}
impl Default for SmplParams {
    fn default() -> Self {
        Self {
            smpl_type: SmplType::SmplX,
            gender: Gender::Neutral,
            enable_pose_corrective: false,
        }
    }
}
impl SmplParams {
    pub fn new(smpl_type: SmplType, gender: Gender, enable_pose_corrective: bool) -> Self {
        Self {
            smpl_type,
            gender,
            enable_pose_corrective,
        }
    }

    pub fn new_from_smpl_codec(codec: &SmplCodec) -> Self {
        Self {
            smpl_type: codec.smpl_type(),
            gender: codec.gender(),
            enable_pose_corrective: true,
        }
    }

    #[cfg(not(target_arch = "wasm32"))] //wasm cannot compile the zip library so we cannot read npz
    #[allow(clippy::cast_possible_truncation)]
    pub fn new_from_smpl_file(path: &str) -> Self {
        let codec = SmplCodec::from_file(path);
        Self::new_from_smpl_codec(&codec)
    }
}
