use crate::components::GlossInterop;
use crate::scene::SceneAnimation;
use gloss_renderer::network::{FromSerializable, ToSerializable};
use gloss_utils::bshare::ToNdArray;
use num_traits::FromPrimitive;
use serde::{Deserialize, Serialize};
use smpl_core::common::{
    animation::{AnimWrap, Animation, AnimationConfig, AnimationRunner},
    betas::Betas,
    smpl_params::SmplParams,
    transform_sequence::TransformSequence,
    types::{AngleType, FaceType, Gender, SmplType, UpAxis},
};
use std::time::Duration;
/// Serializable version of `SmplParams` component
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SerializableSmplParams {
    pub smpl_type: u8,
    pub gender: u8,
    pub enable_pose_corrective: bool,
}
impl ToSerializable<SerializableSmplParams> for SmplParams {
    fn to_serializable(&self) -> SerializableSmplParams {
        SerializableSmplParams {
            smpl_type: self.smpl_type as u8,
            gender: self.gender as u8,
            enable_pose_corrective: self.enable_pose_corrective,
        }
    }
}
impl FromSerializable<SerializableSmplParams> for SmplParams {
    fn from_serializable(s: &SerializableSmplParams) -> SmplParams {
        SmplParams {
            smpl_type: SmplType::from_u8(s.smpl_type).unwrap(),
            gender: Gender::from_u8(s.gender).unwrap(),
            enable_pose_corrective: s.enable_pose_corrective,
        }
    }
}
/// Serializable version of Betas component
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SerializableBetas {
    pub betas_data: Vec<f32>,
}
impl ToSerializable<SerializableBetas> for Betas {
    fn to_serializable(&self) -> SerializableBetas {
        let betas_ndarray = self.betas.to_ndarray();
        SerializableBetas {
            betas_data: betas_ndarray.to_vec(),
        }
    }
}
impl FromSerializable<SerializableBetas> for Betas {
    fn from_serializable(s: &SerializableBetas) -> Betas {
        Betas::new_from_ndarray(ndarray::Array1::from(s.betas_data.clone()))
    }
}
/// Serializable version of Animation component
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SerializableAnimation {
    pub per_frame_joint_poses_shape: (usize, usize, usize),
    pub per_frame_joint_poses_data: Vec<f32>,
    pub per_frame_root_trans_shape: (usize, usize),
    pub per_frame_root_trans_data: Vec<f32>,
    pub per_frame_expression_coeffs_shape: Option<(usize, usize)>,
    pub per_frame_expression_coeffs_data: Option<Vec<f32>>,
    pub start_offset: usize,
    pub anim_current_time_nanos: u64,
    pub anim_reversed: bool,
    pub nr_repetitions: u32,
    pub paused: bool,
    pub temporary_pause: bool,
    pub fps: f32,
    pub wrap_behaviour: u8,
    pub angle_type: u8,
    pub up_axis: u8,
    pub smpl_type: u8,
    pub face_type: u8,
}
#[allow(clippy::cast_possible_truncation)]
impl ToSerializable<SerializableAnimation> for Animation {
    fn to_serializable(&self) -> SerializableAnimation {
        let expr_shape_data = if let Some(ref expr) = self.per_frame_expression_coeffs {
            (Some(expr.dim()), Some(expr.as_slice().unwrap().to_vec()))
        } else {
            (None, None)
        };
        SerializableAnimation {
            per_frame_joint_poses_shape: self.per_frame_joint_poses.dim(),
            per_frame_joint_poses_data: self.per_frame_joint_poses.as_slice().unwrap().to_vec(),
            per_frame_root_trans_shape: self.per_frame_root_trans.dim(),
            per_frame_root_trans_data: self.per_frame_root_trans.as_slice().unwrap().to_vec(),
            per_frame_expression_coeffs_shape: expr_shape_data.0,
            per_frame_expression_coeffs_data: expr_shape_data.1,
            start_offset: self.start_offset,
            anim_current_time_nanos: self.runner.anim_current_time.as_nanos() as u64,
            anim_reversed: self.runner.anim_reversed,
            nr_repetitions: self.runner.nr_repetitions,
            paused: self.runner.paused,
            temporary_pause: self.runner.temporary_pause,
            fps: self.config.fps,
            wrap_behaviour: match self.config.wrap_behaviour {
                AnimWrap::Clamp => 0,
                AnimWrap::Loop => 1,
                AnimWrap::Reverse => 2,
            },
            angle_type: match self.config.angle_type {
                AngleType::AxisAngle => 0,
                AngleType::Euler => 1,
            },
            up_axis: match self.config.up_axis {
                UpAxis::Y => 0,
                UpAxis::Z => 1,
            },
            smpl_type: self.config.smpl_type as u8,
            face_type: self.config.face_type as u8,
        }
    }
}
impl FromSerializable<SerializableAnimation> for Animation {
    fn from_serializable(s: &SerializableAnimation) -> Animation {
        use ndarray::Array2;
        let per_frame_joint_poses = ndarray::Array3::from_shape_vec(s.per_frame_joint_poses_shape, s.per_frame_joint_poses_data.clone()).unwrap();
        let per_frame_root_trans = Array2::from_shape_vec(s.per_frame_root_trans_shape, s.per_frame_root_trans_data.clone()).unwrap();
        let per_frame_expression_coeffs =
            if let (Some(shape), Some(data)) = (&s.per_frame_expression_coeffs_shape, &s.per_frame_expression_coeffs_data) {
                Some(Array2::from_shape_vec(*shape, data.clone()).unwrap())
            } else {
                None
            };
        #[allow(clippy::match_same_arms)]
        let wrap_behaviour = match s.wrap_behaviour {
            0 => AnimWrap::Clamp,
            1 => AnimWrap::Loop,
            2 => AnimWrap::Reverse,
            _ => AnimWrap::Loop,
        };
        #[allow(clippy::match_same_arms)]
        let angle_type = match s.angle_type {
            0 => AngleType::AxisAngle,
            1 => AngleType::Euler,
            _ => AngleType::AxisAngle,
        };
        #[allow(clippy::match_same_arms)]
        let up_axis = match s.up_axis {
            0 => UpAxis::Y,
            1 => UpAxis::Z,
            _ => UpAxis::Y,
        };
        Animation {
            per_frame_joint_poses,
            per_frame_root_trans,
            per_frame_expression_coeffs,
            start_offset: s.start_offset,
            runner: AnimationRunner {
                anim_current_time: Duration::from_nanos(s.anim_current_time_nanos),
                anim_reversed: s.anim_reversed,
                nr_repetitions: s.nr_repetitions,
                paused: s.paused,
                temporary_pause: s.temporary_pause,
            },
            config: AnimationConfig {
                fps: s.fps,
                wrap_behaviour,
                angle_type,
                up_axis,
                smpl_type: SmplType::from_u8(s.smpl_type).unwrap_or(SmplType::SmplX),
                face_type: FaceType::from_u8(s.face_type).unwrap_or(FaceType::SmplX),
            },
        }
    }
}
/// Serializable version of `GlossInterop` component
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SerializableGlossInterop {
    pub with_uv: bool,
}
impl ToSerializable<SerializableGlossInterop> for GlossInterop {
    fn to_serializable(&self) -> SerializableGlossInterop {
        SerializableGlossInterop { with_uv: self.with_uv }
    }
}
impl FromSerializable<SerializableGlossInterop> for GlossInterop {
    fn from_serializable(s: &SerializableGlossInterop) -> GlossInterop {
        GlossInterop { with_uv: s.with_uv }
    }
}
/// Serializable version of `SceneAnimation` component
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SerializableSceneAnimation {
    pub num_frames: usize,
    pub anim_current_time_nanos: u64,
    pub anim_reversed: bool,
    pub nr_repetitions: u32,
    pub paused: bool,
    pub temporary_pause: bool,
    pub fps: f32,
    pub wrap_behaviour: u8,
    pub angle_type: u8,
    pub up_axis: u8,
    pub smpl_type: u8,
    pub face_type: u8,
}
impl ToSerializable<SerializableSceneAnimation> for SceneAnimation {
    fn to_serializable(&self) -> SerializableSceneAnimation {
        #[allow(clippy::cast_possible_truncation)]
        SerializableSceneAnimation {
            num_frames: self.num_frames,
            anim_current_time_nanos: self.runner.anim_current_time.as_nanos() as u64,
            anim_reversed: self.runner.anim_reversed,
            nr_repetitions: self.runner.nr_repetitions,
            paused: self.runner.paused,
            temporary_pause: self.runner.temporary_pause,
            fps: self.config.fps,
            wrap_behaviour: match self.config.wrap_behaviour {
                AnimWrap::Clamp => 0,
                AnimWrap::Loop => 1,
                AnimWrap::Reverse => 2,
            },
            angle_type: match self.config.angle_type {
                AngleType::AxisAngle => 0,
                AngleType::Euler => 1,
            },
            up_axis: match self.config.up_axis {
                UpAxis::Y => 0,
                UpAxis::Z => 1,
            },
            smpl_type: self.config.smpl_type as u8,
            face_type: self.config.face_type as u8,
        }
    }
}
impl FromSerializable<SerializableSceneAnimation> for SceneAnimation {
    fn from_serializable(s: &SerializableSceneAnimation) -> SceneAnimation {
        #[allow(clippy::match_same_arms)]
        let wrap_behaviour = match s.wrap_behaviour {
            0 => AnimWrap::Clamp,
            1 => AnimWrap::Loop,
            2 => AnimWrap::Reverse,
            _ => AnimWrap::Loop,
        };
        #[allow(clippy::match_same_arms)]
        let angle_type = match s.angle_type {
            0 => AngleType::AxisAngle,
            1 => AngleType::Euler,
            _ => AngleType::AxisAngle,
        };
        #[allow(clippy::match_same_arms)]
        let up_axis = match s.up_axis {
            0 => UpAxis::Y,
            1 => UpAxis::Z,
            _ => UpAxis::Y,
        };
        SceneAnimation {
            num_frames: s.num_frames,
            runner: AnimationRunner {
                anim_current_time: std::time::Duration::from_nanos(s.anim_current_time_nanos),
                anim_reversed: s.anim_reversed,
                nr_repetitions: s.nr_repetitions,
                paused: s.paused,
                temporary_pause: s.temporary_pause,
            },
            config: AnimationConfig {
                fps: s.fps,
                wrap_behaviour,
                angle_type,
                up_axis,
                smpl_type: SmplType::from_u8(s.smpl_type).unwrap_or(SmplType::SmplX),
                face_type: FaceType::from_u8(s.face_type).unwrap_or(FaceType::SmplX),
            },
        }
    }
}
/// Serializable version of `TransformSequence` component
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SerializableTransformSequence {
    pub translations_shape: (usize, usize),
    pub translations_data: Vec<f32>,
    pub rotations_shape: (usize, usize),
    pub rotations_data: Vec<f32>,
    pub scales_shape: usize,
    pub scales_data: Vec<f32>,
}
impl ToSerializable<SerializableTransformSequence> for TransformSequence {
    fn to_serializable(&self) -> SerializableTransformSequence {
        SerializableTransformSequence {
            translations_shape: self.translations.dim(),
            translations_data: self.translations.as_slice().unwrap().to_vec(),
            rotations_shape: self.rotations.dim(),
            rotations_data: self.rotations.as_slice().unwrap().to_vec(),
            scales_shape: self.scales.len(),
            scales_data: self.scales.as_slice().unwrap().to_vec(),
        }
    }
}
impl FromSerializable<SerializableTransformSequence> for TransformSequence {
    fn from_serializable(s: &SerializableTransformSequence) -> TransformSequence {
        TransformSequence {
            translations: ndarray::Array2::from_shape_vec(s.translations_shape, s.translations_data.clone()).unwrap(),
            rotations: ndarray::Array2::from_shape_vec(s.rotations_shape, s.rotations_data.clone()).unwrap(),
            scales: ndarray::Array1::from(s.scales_data.clone()),
        }
    }
}
gloss_renderer::impl_network_sendable_and_receivable!(
    SerializableSmplParams,
    SerializableBetas,
    SerializableAnimation,
    SerializableGlossInterop,
    SerializableSceneAnimation,
    SerializableTransformSequence,
);
