/// Options specifically for the forward pass of smpl family models
#[derive(Clone)]
pub struct SmplOptions {
    pub enable_pose_corrective: bool,
}
impl Default for SmplOptions {
    fn default() -> Self {
        Self {
            enable_pose_corrective: true,
        }
    }
}
impl SmplOptions {
    pub fn new(enable_pose_corrective: bool) -> Self {
        Self { enable_pose_corrective }
    }
}
