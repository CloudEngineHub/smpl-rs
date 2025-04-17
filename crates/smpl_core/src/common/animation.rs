use super::{
    expression::Expression,
    metadata::smpl_metadata,
    pose::Pose,
    types::{AngleType, SmplType, UpAxis},
};
use crate::codec::codec::SmplCodec;
use core::time::Duration;
use gloss_utils::nshare::{RefNdarray1, ToNalgebra};
use log::debug;
use log::warn;
use nalgebra as na;
use nd::concatenate;
use ndarray as nd;
use ndarray_npy::NpzReader;
use serde_json::Value;
use smpl_utils::{
    io::FileLoader,
    numerical::{euler2angleaxis, map},
};
use std::io::{Read, Seek};
/// Animation Wrap mode
#[derive(PartialEq, PartialOrd, Clone, Default)]
pub enum AnimWrap {
    Clamp,
    #[default]
    Loop,
    Reverse,
}
/// Animation config
#[derive(Clone)]
pub struct AnimationConfig {
    pub fps: f32,
    pub wrap_behaviour: AnimWrap,
    pub angle_type: AngleType,
    pub up_axis: UpAxis,
    pub smpl_type: SmplType,
}
impl Default for AnimationConfig {
    fn default() -> Self {
        Self {
            fps: 60.0,
            wrap_behaviour: AnimWrap::Clamp,
            angle_type: AngleType::AxisAngle,
            up_axis: UpAxis::Y,
            smpl_type: SmplType::SmplX,
        }
    }
}
/// The runner for animations
#[derive(Clone)]
#[allow(clippy::struct_excessive_bools)]
pub struct AnimationRunner {
    pub anim_current_time: Duration,
    pub anim_reversed: bool,
    pub nr_repetitions: u32,
    pub paused: bool,
    pub temporary_pause: bool,
}
impl Default for AnimationRunner {
    fn default() -> Self {
        Self {
            anim_current_time: Duration::ZERO,
            anim_reversed: false,
            nr_repetitions: 0,
            paused: false,
            temporary_pause: false,
        }
    }
}
/// Animation struct for all data regarding a certain animation
#[derive(Clone)]
pub struct Animation {
    pub per_frame_joint_poses: nd::Array3<f32>,
    pub per_frame_root_trans: nd::Array2<f32>,
    pub per_frame_expression_coeffs: Option<nd::Array2<f32>>,
    pub start_offset: usize,
    pub runner: AnimationRunner,
    pub config: AnimationConfig,
}
impl Animation {
    /// # Panics
    /// Will panic if the translation and rotation do not cover the same number
    /// of timesteps
    #[allow(clippy::cast_possible_truncation)]
    pub fn new_from_matrices(
        per_frame_joint_poses: nd::Array3<f32>,
        per_frame_global_trans: nd::Array2<f32>,
        per_frame_expression_coeffs: Option<nd::Array2<f32>>,
        config: AnimationConfig,
    ) -> Self {
        assert!(
            per_frame_joint_poses.dim().0 == per_frame_global_trans.dim().0,
            "The translation and rotation should cover the same number of timesteps"
        );
        let mut per_frame_joint_poses = per_frame_joint_poses;
        let per_frame_global_trans = per_frame_global_trans;
        if config.smpl_type == SmplType::SmplPP && config.angle_type == AngleType::Euler {
            warn!("Angle type Euler is not allowed with SMPL++");
        }
        if config.smpl_type != SmplType::SmplPP && config.angle_type == AngleType::Euler {
            let animation_frames = per_frame_joint_poses.dim().0;
            let num_active_joints = per_frame_joint_poses.dim().1;
            let mut new_per_frame_joint_poses: nd::Array3<f32> = nd::Array3::<f32>::zeros((animation_frames, num_active_joints, 3));
            for (idx_timestep, poses_for_timestep) in per_frame_joint_poses.axis_iter(nd::Axis(0)).enumerate() {
                for (idx_joint, joint_pose) in poses_for_timestep.axis_iter(nd::Axis(0)).enumerate() {
                    let angle_axis = euler2angleaxis(joint_pose[0], joint_pose[1], joint_pose[2]);
                    new_per_frame_joint_poses[(idx_timestep, idx_joint, 0)] = angle_axis.x;
                    new_per_frame_joint_poses[(idx_timestep, idx_joint, 1)] = angle_axis.y;
                    new_per_frame_joint_poses[(idx_timestep, idx_joint, 2)] = angle_axis.z;
                }
            }
            per_frame_joint_poses = new_per_frame_joint_poses;
        }
        Self {
            per_frame_joint_poses,
            per_frame_root_trans: per_frame_global_trans,
            per_frame_expression_coeffs,
            start_offset: 0,
            runner: AnimationRunner::default(),
            config,
        }
    }
    #[allow(clippy::cast_possible_truncation)]
    fn new_from_npz_reader<R: Read + Seek>(npz: &mut NpzReader<R>, config: AnimationConfig) -> Self {
        debug!("npz names is {:?}", npz.names().unwrap());
        let per_frame_joint_poses: nd::Array2<f64> = npz.by_name("poses").unwrap();
        let animation_frames = per_frame_joint_poses.nrows();
        let num_joints_3 = per_frame_joint_poses.ncols();
        let per_frame_joint_poses = per_frame_joint_poses.mapv(|x| x as f32);
        let per_frame_joint_poses = per_frame_joint_poses
            .into_shape_with_order((animation_frames, num_joints_3 / 3, 3))
            .unwrap();
        let per_frame_global_trans: nd::Array2<f64> = npz.by_name("trans").unwrap();
        let per_frame_global_trans = per_frame_global_trans.mapv(|x| x as f32);
        let per_frame_expression_coeffs: Option<nd::Array2<f64>> = npz.by_name("expressionParameters").ok();
        let per_frame_expression_coeffs = per_frame_expression_coeffs.map(|x| x.mapv(|x| x as f32));
        Self::new_from_matrices(per_frame_joint_poses, per_frame_global_trans, per_frame_expression_coeffs, config)
    }
    /// # Panics
    /// Will panic if the path cannot be opened
    /// Will panic if the translation and rotation do not cover the same number
    /// of timesteps
    #[cfg(not(target_arch = "wasm32"))]
    #[allow(clippy::cast_possible_truncation)]
    pub fn new_from_npz(anim_npz_path: &str, config: AnimationConfig) -> Self {
        let mut npz =
            NpzReader::new(std::fs::File::open(anim_npz_path).unwrap_or_else(|_| panic!("Could not find/open file: {anim_npz_path}"))).unwrap();
        Self::new_from_npz_reader(&mut npz, config)
    }
    /// # Panics
    /// Will panic if the path cannot be opened
    /// Will panic if the translation and rotation do not cover the same number
    /// of timesteps
    #[allow(clippy::cast_possible_truncation)]
    pub async fn new_from_npz_async(anim_npz_path: &str, config: AnimationConfig) -> Self {
        let reader = FileLoader::open(anim_npz_path).await;
        let mut npz = NpzReader::new(reader).unwrap();
        Self::new_from_npz_reader(&mut npz, config)
    }
    /// # Panics
    /// Will panic if the path cannot be opened
    /// Will panic if the translation and rotation do not cover the same number
    /// of timesteps
    #[allow(clippy::cast_possible_truncation)]
    #[allow(clippy::identity_op)]
    pub fn new_from_json(path: &str, _anim_fps: f32, config: AnimationConfig) -> Self {
        let file = std::fs::File::open(path).unwrap();
        let reader = std::io::BufReader::new(file);
        let v: Value = serde_json::from_reader(reader).unwrap();
        let poses = &v["poses"];
        let animation_frames = poses.as_array().unwrap().len();
        let num_active_joints = poses[0].as_array().unwrap().len() / 3;
        let per_frame_global_trans: nd::Array2<f32> = nd::Array2::<f32>::zeros((animation_frames, 3));
        let poses_json_vec: Vec<f32> = poses.as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect();
        let per_frame_joint_poses = nd::Array3::from_shape_vec((animation_frames, num_active_joints, 3), poses_json_vec).unwrap();
        Self::new_from_matrices(per_frame_joint_poses, per_frame_global_trans, None, config)
    }
    /// Create an ``Animation`` component from a ``SmplCodec``
    /// # Panics
    /// Will panic if the individual body part poses in the codec don't have the
    /// correct shape to be concatenated together into a full pose for the whole
    /// body
    #[allow(clippy::cast_sign_loss)]
    pub fn new_from_smpl_codec(codec: &SmplCodec, wrap_behaviour: AnimWrap) -> Option<Self> {
        let nr_frames = codec.frame_count as usize;
        let metadata = smpl_metadata(&codec.smpl_type());
        let body_translation = codec
            .body_translation
            .as_ref()
            .unwrap_or(&ndarray::Array2::<f32>::zeros((nr_frames, 3)))
            .clone();
        let fps = codec.frame_rate?;
        if codec.smpl_type() == SmplType::SmplPP {
            let body_pose = codec
                .body_pose
                .as_ref()
                .unwrap_or(&ndarray::Array3::<f32>::zeros((nr_frames, metadata.pose_dim, 1)))
                .clone();
            let config = AnimationConfig {
                smpl_type: SmplType::SmplPP,
                wrap_behaviour,
                fps,
                ..Default::default()
            };
            Some(Self::new_from_matrices(body_pose, body_translation, None, config))
        } else {
            let body_pose = codec
                .body_pose
                .as_ref()
                .unwrap_or(&ndarray::Array3::<f32>::zeros((nr_frames, 1 + metadata.num_body_joints, 3)))
                .clone();
            let head_pose = codec
                .head_pose
                .as_ref()
                .unwrap_or(&ndarray::Array3::<f32>::zeros((nr_frames, metadata.num_face_joints, 3)))
                .clone();
            let left_hand_pose = codec
                .left_hand_pose
                .as_ref()
                .unwrap_or(&ndarray::Array3::<f32>::zeros((nr_frames, metadata.num_hand_joints, 3)))
                .clone();
            let right_hand_pose = codec
                .right_hand_pose
                .as_ref()
                .unwrap_or(&ndarray::Array3::<f32>::zeros((nr_frames, metadata.num_hand_joints, 3)))
                .clone();
            let per_frame_joint_poses = concatenate(
                nd::Axis(1),
                &[body_pose.view(), head_pose.view(), left_hand_pose.view(), right_hand_pose.view()],
            )
            .unwrap();
            let per_frame_expression_coeffs = codec.expression_parameters.clone();
            let config = AnimationConfig {
                smpl_type: codec.smpl_type(),
                wrap_behaviour,
                fps,
                ..Default::default()
            };
            Some(Self::new_from_matrices(
                per_frame_joint_poses,
                body_translation,
                per_frame_expression_coeffs,
                config,
            ))
        }
    }
    /// Create an ``Animation`` component from a ``.smpl`` file
    #[cfg(not(target_arch = "wasm32"))]
    #[allow(clippy::cast_possible_truncation)]
    pub fn new_from_smpl_file(path: &str, wrap_behaviour: AnimWrap) -> Option<Self> {
        let codec = SmplCodec::from_file(path);
        Self::new_from_smpl_codec(&codec, wrap_behaviour)
    }
    /// Create an ``Animation`` component from a ``.smpl`` buffer
    #[allow(clippy::cast_possible_truncation)]
    pub fn new_from_smpl_buf(buf: &[u8], wrap_behaviour: AnimWrap) -> Option<Self> {
        let codec = SmplCodec::from_buf(buf);
        Self::new_from_smpl_codec(&codec, wrap_behaviour)
    }
    pub fn num_active_joints(&self) -> usize {
        self.per_frame_joint_poses.dim().1
    }
    pub fn num_animation_frames(&self) -> usize {
        self.per_frame_joint_poses.dim().0
    }
    /// Advances the animation by the amount of time elapsed since last time we
    /// got the current pose
    pub fn advance(&mut self, dt_raw: Duration, first_time: bool) {
        let duration = self.duration();
        let runner = &mut self.runner;
        let config = &self.config;
        let mut dt = dt_raw;
        if first_time {
            dt = Duration::ZERO;
        }
        let will_overflow = runner.anim_current_time + dt > duration;
        let will_underflow = runner.anim_current_time < dt && runner.anim_reversed;
        if will_overflow || will_underflow {
            if will_overflow {
                match config.wrap_behaviour {
                    AnimWrap::Clamp => {
                        dt = Duration::ZERO;
                        runner.anim_current_time = duration;
                    }
                    AnimWrap::Loop => {
                        dt = Duration::from_secs_f64(dt.as_secs_f64() % duration.as_secs_f64());
                        runner.anim_current_time = Duration::ZERO;
                        runner.nr_repetitions += 1;
                    }
                    AnimWrap::Reverse => {
                        dt = Duration::from_secs_f64(dt.as_secs_f64() % duration.as_secs_f64());
                        runner.anim_current_time = duration;
                        runner.anim_reversed = !runner.anim_reversed;
                        runner.nr_repetitions += 1;
                    }
                }
            } else {
                match config.wrap_behaviour {
                    AnimWrap::Clamp => {
                        dt = Duration::ZERO;
                        runner.anim_current_time = Duration::ZERO;
                    }
                    AnimWrap::Loop => {
                        dt = Duration::from_secs_f64(dt.as_secs_f64() % duration.as_secs_f64());
                        runner.anim_current_time = duration;
                        runner.nr_repetitions += 1;
                    }
                    AnimWrap::Reverse => {
                        dt = Duration::from_secs_f64(dt.as_secs_f64() % duration.as_secs_f64());
                        runner.anim_reversed = !runner.anim_reversed;
                        runner.nr_repetitions += 1;
                    }
                }
            }
        }
        if runner.anim_reversed {
            runner.anim_current_time = runner.anim_current_time.saturating_sub(dt);
        } else {
            runner.anim_current_time = runner.anim_current_time.saturating_add(dt);
        }
    }
    pub fn has_expression(&self) -> bool {
        self.per_frame_expression_coeffs.is_some()
    }
    #[allow(clippy::cast_precision_loss)]
    #[allow(clippy::cast_possible_truncation)]
    #[allow(clippy::cast_sign_loss)]
    pub fn get_smooth_time_indices(&self) -> (usize, usize, f32) {
        let frame_time = map(
            self.runner.anim_current_time.as_secs_f32(),
            0.0,
            self.duration().as_secs_f32(),
            0.0,
            (self.num_animation_frames() - 1) as f32,
        );
        let frame_ceil = frame_time.ceil();
        let frame_ceil = frame_ceil.clamp(0.0, (self.num_animation_frames() - 1) as f32);
        let frame_floor = frame_time.floor();
        let frame_floor = frame_floor.clamp(0.0, (self.num_animation_frames() - 1) as f32);
        let w_ceil = frame_ceil - frame_time;
        let w_ceil = 1.0 - w_ceil;
        (frame_floor as usize, frame_ceil as usize, w_ceil)
    }
    /// Get the pose and translation at the current time, interpolates if
    /// necessary
    #[allow(clippy::cast_precision_loss)]
    #[allow(clippy::cast_possible_truncation)]
    #[allow(clippy::cast_sign_loss)]
    pub fn get_current_pose(&mut self) -> Pose {
        let (frame_floor, frame_ceil, w_ceil) = self.get_smooth_time_indices();
        let anim_frame_ceil = self.get_pose_at_idx(frame_ceil);
        let anim_frame_floor = self.get_pose_at_idx(frame_floor);
        anim_frame_floor.interpolate(&anim_frame_ceil, w_ceil)
    }
    /// Get pose at a certain frame ID
    #[allow(clippy::cast_precision_loss)]
    #[allow(clippy::cast_possible_truncation)]
    #[allow(clippy::cast_sign_loss)]
    pub fn get_pose_at_idx(&self, idx: usize) -> Pose {
        let joint_poses = self.per_frame_joint_poses.index_axis(nd::Axis(0), idx).to_owned();
        let global_trans = self.per_frame_root_trans.index_axis(nd::Axis(0), idx).to_owned();
        Pose::new(joint_poses, global_trans, self.config.up_axis, self.config.smpl_type)
    }
    /// Get expression at current time
    pub fn get_current_expression(&mut self) -> Option<Expression> {
        let (frame_floor, frame_ceil, w_ceil) = self.get_smooth_time_indices();
        let expression_ceil = self.get_expression_at_idx(frame_ceil);
        let expresion_floor = self.get_expression_at_idx(frame_floor);
        expresion_floor.map(|expresion_floor| expresion_floor.interpolate(&expression_ceil.unwrap(), w_ceil))
    }
    /// Get expression at a given frame ID
    #[allow(clippy::cast_precision_loss)]
    #[allow(clippy::cast_possible_truncation)]
    #[allow(clippy::cast_sign_loss)]
    pub fn get_expression_at_idx(&self, idx: usize) -> Option<Expression> {
        if let Some(ref per_frame_expression_coeffs) = self.per_frame_expression_coeffs {
            let expr_coeffs = per_frame_expression_coeffs.index_axis(nd::Axis(0), idx).to_owned();
            Some(Expression::new(expr_coeffs))
        } else {
            None
        }
    }
    #[must_use]
    pub fn slice_time_range(&self, start_sec: f32, end_sec: f32) -> Animation {
        let mut cur_anim = self.clone();
        cur_anim.set_cur_time_as_sec(start_sec);
        let (start_idx, _, _) = cur_anim.get_smooth_time_indices();
        cur_anim.set_cur_time_as_sec(end_sec);
        let (_, end_idx, _) = cur_anim.get_smooth_time_indices();
        let nr_frames = end_idx - start_idx + 1;
        let nr_joints = cur_anim.per_frame_joint_poses.shape()[1];
        let mut new_per_frame_joint_poses = nd::Array3::<f32>::zeros((nr_frames, nr_joints, 3));
        let mut new_per_frame_root_trans = nd::Array2::<f32>::zeros((nr_frames, 3));
        for (idx_insert_to, idx_extract_from) in (start_idx..=end_idx).enumerate() {
            let joint_poses = cur_anim.per_frame_joint_poses.index_axis(nd::Axis(0), idx_extract_from);
            let trans = cur_anim.per_frame_root_trans.index_axis(nd::Axis(0), idx_extract_from);
            new_per_frame_joint_poses.index_axis_mut(nd::Axis(0), idx_insert_to).assign(&joint_poses);
            new_per_frame_root_trans.index_axis_mut(nd::Axis(0), idx_insert_to).assign(&trans);
        }
        let _new_per_frame_expression_coeffs = if let Some(ref per_frame_expression_coeffs) = cur_anim.per_frame_expression_coeffs {
            let nr_expr_coeffs = per_frame_expression_coeffs.shape()[1];
            let mut new_per_frame_expression_coeffs = nd::Array2::<f32>::zeros((nr_frames, nr_expr_coeffs));
            for (idx_insert_to, idx_extract_from) in (start_idx..=end_idx).enumerate() {
                let expr = per_frame_expression_coeffs.index_axis(nd::Axis(0), idx_extract_from);
                new_per_frame_expression_coeffs.index_axis_mut(nd::Axis(0), idx_insert_to).assign(&expr);
            }
            Some(new_per_frame_expression_coeffs)
        } else {
            None
        };
        let new_per_frame_expression_coeffs = if let Some(ref per_frame_expression_coeffs) = cur_anim.per_frame_expression_coeffs {
            let nr_expr_coeffs = per_frame_expression_coeffs.shape()[1];
            let mut new_per_frame_expression_coeffs = nd::Array2::<f32>::zeros((nr_frames, nr_expr_coeffs));
            for (idx_insert_to, idx_extract_from) in (start_idx..=end_idx).enumerate() {
                let expr = per_frame_expression_coeffs.index_axis(nd::Axis(0), idx_extract_from);
                new_per_frame_expression_coeffs.index_axis_mut(nd::Axis(0), idx_insert_to).assign(&expr);
            }
            Some(new_per_frame_expression_coeffs)
        } else {
            None
        };
        Animation::new_from_matrices(
            new_per_frame_joint_poses,
            new_per_frame_root_trans,
            new_per_frame_expression_coeffs,
            cur_anim.config.clone(),
        )
    }
    /// Rotates multiple of 90 until the axis of the body is aligned with some
    /// arbitrary vector
    pub fn align_y_axis_quadrant(&mut self, current_axis: &nd::Array1<f32>, desired_axis: &nd::Array1<f32>) {
        let mut cur_axis_xz = na::Vector2::new(current_axis[0], -current_axis[2]).normalize();
        let desired_axis_xz = na::Vector2::new(desired_axis[0], -desired_axis[2]).normalize();
        let mut best_dot = f32::MIN;
        let mut best_angle: f32 = 0.0;
        let mut cur_angle = 0.0;
        let rot_90 = na::Rotation2::new(std::f32::consts::FRAC_PI_2);
        for _iters in 0..4 {
            cur_axis_xz = rot_90 * cur_axis_xz;
            cur_angle += 90.0;
            let dot = cur_axis_xz.dot(&desired_axis_xz);
            if dot > best_dot {
                best_dot = dot;
                best_angle = cur_angle;
            }
        }
        let alignment_rot = na::Rotation3::from_euler_angles(0.0, best_angle.to_radians(), 0.0);
        ndarray::Zip::from(self.per_frame_joint_poses.outer_iter_mut())
            .and(self.per_frame_root_trans.outer_iter_mut())
            .for_each(|mut poses_for_timestep, mut trans_for_timestep| {
                let pelvis_axis_angle = poses_for_timestep.row(0).into_nalgebra();
                let pelvis_axis_angle = pelvis_axis_angle.fixed_rows::<3>(0);
                let pelvis_rot = na::Rotation3::from_scaled_axis(pelvis_axis_angle);
                let new_pelvis_rot = alignment_rot * pelvis_rot;
                poses_for_timestep.row_mut(0).assign(&new_pelvis_rot.scaled_axis().ref_ndarray1());
                let trans_for_timestep_na = trans_for_timestep.to_owned().into_nalgebra();
                let new_trans_for_timestep_na = alignment_rot * trans_for_timestep_na;
                trans_for_timestep.assign(&new_trans_for_timestep_na.ref_ndarray1());
            });
    }
    pub fn get_cur_time(&self) -> Duration {
        self.runner.anim_current_time
    }
    pub fn set_cur_time_as_sec(&mut self, time_sec: f32) {
        self.runner.anim_current_time = Duration::from_secs_f32(time_sec);
    }
    pub fn pause(&mut self) {
        self.runner.paused = true;
    }
    pub fn play(&mut self) {
        self.runner.paused = false;
    }
    /// Duration of the animation
    #[allow(clippy::cast_precision_loss)]
    pub fn duration(&self) -> Duration {
        Duration::from_secs_f32(self.num_animation_frames() as f32 / self.config.fps)
    }
    pub fn is_finished(&self) -> bool {
        self.config.wrap_behaviour == AnimWrap::Clamp && self.runner.anim_current_time >= self.duration()
    }
    pub fn nr_repetitions(&self) -> u32 {
        self.runner.nr_repetitions
    }
    /// Shift each frame of the animation by the given translation vector.
    ///
    /// # Errors
    ///
    /// Will return `Err` for array size mismatch.
    pub fn translate(&mut self, translation: &nd::Array1<f32>) -> Result<(), String> {
        if translation.len() != self.per_frame_root_trans.ncols() {
            return Err("Translation vector should be length-3 array".to_owned());
        }
        for mut row in self.per_frame_root_trans.rows_mut() {
            row += translation;
        }
        Ok(())
    }
}
