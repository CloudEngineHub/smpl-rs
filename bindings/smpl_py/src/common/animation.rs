use std::time::Duration;

use gloss_hecs::Entity;
use gloss_py_macros::PyComponent;
use gloss_renderer::scene::Scene;

use ndarray as nd;
use numpy::{PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, ToPyArray};
use pyo3::{exceptions::PyValueError, prelude::*};
use smpl_rs::common::{
    animation::{AnimWrap, Animation, AnimationConfig},
    types::{AngleType, SmplType, UpAxis},
};
use smpl_utils::convert_enum_from;

use super::{
    pose::PyPose,
    types::{PyAngleType, PySmplType, PyUpAxis},
};

#[pyclass(name = "AnimWrap", module = "smpl_rs.types", unsendable, eq, eq_int)]
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum PyAnimWrap {
    Clamp = 0,
    #[default]
    Loop,
    Reverse,
}
convert_enum_from!(PyAnimWrap, AnimWrap, Clamp, Loop, Reverse,);

#[pyclass(name = "Animation", module = "smpl_rs.components", unsendable)]
// it has to be unsendable because it does not implement Send: https://pyo3.rs/v0.19.1/class#must-be-send
#[derive(Clone, PyComponent)]
pub struct PyAnimation {
    inner: Animation,
}
#[pymethods]
impl PyAnimation {
    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (per_frame_joint_poses, per_frame_global_trans, per_expression_coeffs=None, fps=None, wrap_behaviour=None, angle_type=None, up_axis=None, smpl_type=None))]
    #[pyo3(
        text_signature = "(per_frame_joint_poses: NDArray[np.float32], per_frame_global_trans: NDArray[np.float32], per_expression_coeffs: Optional[NDArray[np.float32]] = None, fps: Optional[float] = None, wrap_behaviour: Optional[AnimWrap] = None, angle_type: Optional[AngleType] = None, up_axis: Optional[UpAxis] = None, smpl_type: Optional[SmplType] = None) -> Animation"
    )]
    pub fn from_matrices(
        per_frame_joint_poses: PyReadonlyArray2<f32>,
        per_frame_global_trans: PyReadonlyArray2<f32>,
        per_expression_coeffs: Option<PyReadonlyArray2<f32>>,
        fps: Option<f32>,
        wrap_behaviour: Option<PyAnimWrap>,
        angle_type: Option<PyAngleType>,
        up_axis: Option<PyUpAxis>,
        smpl_type: Option<PySmplType>,
    ) -> Self {
        let def = AnimationConfig::default();

        let wrap_behaviour = wrap_behaviour.map_or(def.wrap_behaviour, AnimWrap::from);
        let angle_type = angle_type.map_or(def.angle_type, AngleType::from);
        let up_axis = up_axis.map_or(def.up_axis, UpAxis::from);
        let smpl_type = smpl_type.map_or(def.smpl_type, SmplType::from);

        let config = AnimationConfig {
            fps: fps.unwrap_or(def.fps),
            wrap_behaviour,
            angle_type,
            up_axis,
            smpl_type,
        };

        let per_frame_joint_poses: nd::Array2<f32> = per_frame_joint_poses.to_owned_array();
        let nr_frames = per_frame_joint_poses.dim().0;
        let joints_3 = per_frame_joint_poses.dim().1;

        
        let per_frame_joint_poses = per_frame_joint_poses.into_shape_with_order((nr_frames, joints_3 / 3, 3)).unwrap(); 

        let per_frame_global_trans: nd::Array2<f32> = per_frame_global_trans.to_owned_array();

        //expression get it from the array or just make some default expression with
        // all zeros
        // let per_frame_expression_coeffs = per_expression_coeffs.map(numpy::PyArray::to_owned_array);
        let per_frame_expression_coeffs = per_expression_coeffs.map(|x| x.to_owned_array());

        Self {
            inner: Animation::new_from_matrices(per_frame_joint_poses, per_frame_global_trans, per_frame_expression_coeffs, config),
        }
    }
    #[staticmethod]
    #[pyo3(signature = (path_anim, fps=None, wrap_behaviour=None, angle_type=None, up_axis=None, smpl_type=None))]
    #[pyo3(
        text_signature = "(path_anim: str, fps: Optional[float] = None, wrap_behaviour: Optional[AnimWrap] = None, angle_type: Optional[AngleType] = None, up_axis: Optional[UpAxis] = None, smpl_type: Optional[SmplType] = None) -> Animation"
    )]
    pub fn from_npz(
        path_anim: &str,
        fps: Option<f32>,
        wrap_behaviour: Option<PyAnimWrap>,
        angle_type: Option<PyAngleType>,
        up_axis: Option<PyUpAxis>,
        smpl_type: Option<PySmplType>,
    ) -> Self {
        let def = AnimationConfig::default();

        let wrap_behaviour = wrap_behaviour.map_or(def.wrap_behaviour, AnimWrap::from);
        let angle_type = angle_type.map_or(def.angle_type, AngleType::from);
        let up_axis = up_axis.map_or(def.up_axis, UpAxis::from);
        let smpl_type = smpl_type.map_or(def.smpl_type, SmplType::from);

        let config = AnimationConfig {
            fps: fps.unwrap_or(def.fps),
            wrap_behaviour,
            angle_type,
            up_axis,
            smpl_type,
        };
        Self {
            inner: Animation::new_from_npz(path_anim, config), //for dance,
        }
    }

    #[staticmethod]
    #[pyo3(signature = (path_anim, wrap_behaviour=None))]
    #[pyo3(text_signature = "(path_anim: str, wrap_behaviour: Optional[AnimWrap] = None) -> Animation")]
    pub fn from_smpl_file(path_anim: &str, wrap_behaviour: Option<PyAnimWrap>) -> Self {
        Self {
            inner: Animation::new_from_smpl_file(path_anim, wrap_behaviour.unwrap_or_default().into()).unwrap(),
        }
    }

    #[pyo3(text_signature = "($self, current_axis: NDArray[np.float32], desired_axis: NDArray[np.float32]) -> None")]
    pub fn align_y_axis_quadrant(&mut self, current_axis: PyReadonlyArray1<f32>, desired_axis: PyReadonlyArray1<f32>) {
        self.inner
            .align_y_axis_quadrant(&current_axis.to_owned_array(), &desired_axis.to_owned_array());
    }
    #[pyo3(text_signature = "($self, wrap: AnimWrap) -> None")]
    pub fn set_wrap(&mut self, wrap: PyAnimWrap) {
        self.inner.config.wrap_behaviour = wrap.into();
    }
    #[pyo3(text_signature = "($self) -> Tuple[int, int, float]")]
    pub fn get_smooth_time_indices(&self) -> (usize, usize, f32) {
        self.inner.get_smooth_time_indices()
    }
    #[pyo3(text_signature = "($self, idx: int) -> Pose")]
    pub fn get_pose_at_idx(&self, idx: u32) -> PyPose {
        let pose = self.inner.get_pose_at_idx(idx as usize);
        PyPose { inner: pose }
    }
    #[getter]
    /// The root translation array for the animation.
    pub fn get_per_frame_root_trans(&self, py: Python<'_>) -> Py<PyArray2<f32>> {
        self.inner.per_frame_root_trans.to_pyarray_bound(py).into()
    }
    #[pyo3(text_signature = "($self) -> int")]
    pub fn num_animation_frames(&self) -> usize {
        self.inner.num_animation_frames()
    }
    #[pyo3(text_signature = "($self) -> None")]
    pub fn pause(&mut self) {
        self.inner.runner.paused = true;
    }
    #[pyo3(text_signature = "($self, secs: float) -> None")]
    pub fn advance_sec(&mut self, secs: f32) {
        self.inner.advance(Duration::from_secs_f32(secs), false);
    }
    #[pyo3(text_signature = "($self) -> float")]
    pub fn get_cur_time_sec(&self) -> f32 {
        self.inner.get_cur_time().as_secs_f32()
    }
    #[pyo3(text_signature = "($self) -> float")]
    pub fn duration(&self) -> f32 {
        self.inner.duration().as_secs_f32()
    }
    #[pyo3(text_signature = "($self) -> bool")]
    pub fn is_finished(&self) -> bool {
        self.inner.is_finished()
    }
    #[pyo3(text_signature = "($self, translation: NDArray[np.float32]) -> None")]
    /// Shift each frame of the animation by the given translation vector.
    ///
    /// # Errors
    /// Will return `ValueError` if the array has incorrect dimensions.
    pub fn translate(&mut self, translation: PyReadonlyArray1<f32>) -> PyResult<()> {
        self.inner
            .translate(&translation.as_array().to_owned())
            .map_err(PyErr::new::<PyValueError, _>)
    }
}
