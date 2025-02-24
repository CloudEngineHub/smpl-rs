use std::ops::AddAssign;

// use gloss_hecs::Entity;

use nalgebra as na;
use smpl_utils::numerical::{map, smootherstep};

#[derive(Clone)]
pub struct GlossInterop {
    pub with_uv: bool,
}
impl Default for GlossInterop {
    fn default() -> Self {
        Self { with_uv: true }
    }
}

/// Indication that an entity needs to be followed
#[derive(Clone)]
pub struct Follow;

// // We work with frames since its easier to work with the FPS slider
// // The interval would misbehave if we did time, since changing fps means total time changes
// // So the interval being in seconds would end up at a completely different part of the anim
// // We convert to time, given the fps at that instant
// pub struct SmplInterval {
//     pub start: usize,
//     pub end: usize,
// }

// impl SmplInterval {
//     #[allow(clippy::cast_precision_loss)]
//     pub fn is_within_interval(&self, time: f32, fps: f32) -> bool {
//         time >= (self.start as f32 / fps) && time <= (self.end as f32 / fps)
//     }
// }

/// Enum for follower type
#[derive(Clone)]
pub enum FollowerType {
    Cam,
    Lights,
    CamAndLights,
}

/// Parameters for the follower
#[derive(Clone)]
pub struct FollowParams {
    /// Strength of the follower, this value being high would mean the follower
    /// would tightly follow the entity
    pub max_strength: f32,
    pub dist_start: f32,
    pub dist_end: f32,
    /// Follower type chooses what would follow the entity - camera, lights, or
    /// both
    pub follower_type: FollowerType,
    /// Whether to follow the mean of all entities with ``GlossInterop`` in the scene
    pub follow_all: bool,
}
impl Default for FollowParams {
    fn default() -> Self {
        Self {
            max_strength: 3.0,
            dist_start: 0.1,
            dist_end: 0.5,
            follower_type: FollowerType::CamAndLights,
            follow_all: true,
        }
    }
}

/// Resource to indicate that we should follow the animation of a certain entity
#[derive(Clone)]
pub struct Follower {
    // pub entity: Entity,
    point: na::Point3<f32>,
    is_first_time: bool,
    pub params: FollowParams,
}
impl Follower {
    pub fn new(params: FollowParams) -> Self {
        Self {
            // entity,
            point: na::Point3::default(),
            is_first_time: true,
            params,
        }
    }
    /// Progress the follower by dt
    pub fn update(&mut self, point_to_follow: &na::Point3<f32>, cur_follow: &na::Point3<f32>, dt_sec: f32) {
        if self.is_first_time {
            self.is_first_time = false;
            self.point = *point_to_follow;
        } else {
            let diff = point_to_follow - cur_follow;
            let dist = diff.norm();
            let max_strength = self.params.max_strength * dt_sec; //increase the strentght if frames take longer
                                                                  //for some reason map_range doesn't clamp so the strength normalized actually
                                                                  // can have higher values than 1.0
                                                                  // let mut strength_normalized = dist.map_range(0.0..0.3, 0.0..1.0); //lower
                                                                  // than 0.15cm we don't move the camera, higher than that and we start
                                                                  // increasing the strength of movement
            let mut strength_normalized = map(dist, self.params.dist_start, self.params.dist_end, 0.0, 1.0);

            strength_normalized = smootherstep(0.0, 1.0, strength_normalized);
            strength_normalized = strength_normalized.clamp(0.2, 1.0); //instead of having the strength go from 0.0..1.0 we make it between 0.01..1.0
                                                                       // so that we never get a strenght of zero which means we completely stop moving
                                                                       // and never reach our goal.
                                                                       // println!("strength_normalized_smootstep is {}", strength_normalized);
            let strength = (strength_normalized * max_strength).clamp(0.0, 1.0); //we clamp to maximum 1 because we don't want to overshoot which can happen on
                                                                                 // web when the viewer doesn't update and therefore the dt_sec becomes quite
                                                                                 // large when we suddenly decide to render.

            let prev_followed_point = self.point;
            let cam_user_movement = cur_follow - prev_followed_point; //the user might still want to move the camera a little bit, so we add a slight
                                                                      // deviation towards the point that the user currently wants to follow
            self.point.add_assign(diff * strength);
            self.point.add_assign(cam_user_movement);
        }
    }

    /// Get the point being followed
    pub fn get_point_follow(&self, _name: &str) -> na::Point3<f32> {
        self.point
    }

    /// Reset the follower
    pub fn reset(&mut self) {
        self.is_first_time = true;
    }
}
