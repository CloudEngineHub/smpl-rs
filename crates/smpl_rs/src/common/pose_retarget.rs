use super::pose::Pose;
/// After adding a pose or a animation, we align the avatar to the floor and
/// store the tallness that it has; this is the most naive way of retargetting
#[derive(Clone, Copy)]
pub struct RetargetPoseYShift {
    pub y_shift: f32,
    pub dist_chest_to_feet: f32,
    pub currently_applied_y_shift: f32,
}
impl RetargetPoseYShift {
    pub fn new(y_shift: f32, dist_chest_to_feet: f32) -> Self {
        Self {
            y_shift,
            dist_chest_to_feet,
            currently_applied_y_shift: 0.0,
        }
    }
    pub fn update(&mut self, new_height: f32) {
        let old_height = self.dist_chest_to_feet;
        let diff = new_height - old_height;
        self.dist_chest_to_feet = new_height;
        self.y_shift += -diff;
    }
    #[allow(clippy::missing_panics_doc)]
    pub fn apply(&mut self, pose: &mut Pose) {
        if pose.non_retargeted_pose.is_none() {
            pose.non_retargeted_pose = Some(Box::new(pose.clone()));
        }
        pose.global_trans[1] = pose.non_retargeted_pose.as_ref().unwrap().global_trans[1]
            + self.y_shift;
        self.currently_applied_y_shift = self.y_shift;
        pose.retargeted = true;
    }
    #[allow(clippy::missing_panics_doc)]
    pub fn remove_retarget(&self, pose: &mut Pose) {
        if pose.non_retargeted_pose.is_some() {
            let original = *pose.non_retargeted_pose.take().unwrap();
            pose.clone_from(&original);
        }
        pose.retargeted = false;
    }
}
