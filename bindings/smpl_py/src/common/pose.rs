use gloss_hecs::Entity;
use gloss_py_macros::PyComponent;
use gloss_renderer::scene::Scene;
use ndarray as nd;
use numpy::PyArray2;

use numpy::PyReadonlyArray1;
use pyo3::prelude::*;

use numpy::{PyArrayMethods, ToPyArray};
use smpl_rs::common::{
    pose::Pose,
    types::{SmplType, UpAxis},
};

use super::types::{PySmplType, PyUpAxis};

#[pyclass(name = "Pose", module = "smpl_rs.components", unsendable)]
// it has to be unsendable because it does not implement Send: https://pyo3.rs/v0.19.1/class#must-be-send
#[derive(Clone, PyComponent)]
pub struct PyPose {
    pub inner: Pose,
}
#[pymethods]
impl PyPose {
    #[staticmethod]
    #[pyo3(text_signature = "(up_axis: UpAxis, smpl_type: SmplType) -> Pose")]
    pub fn new_empty(up_axis: PyUpAxis, smpl_type: PySmplType) -> Self {
        Self {
            inner: Pose::new_empty(
                // num_active_joints,
                UpAxis::from(up_axis),
                SmplType::from(smpl_type),
            ),
        }
    }
    #[staticmethod]
    #[pyo3(text_signature = "(joint_poses: NDArray[np.float32], global_trans: NDArray[np.float32], up_axis: UpAxis, smpl_type: SmplType) -> Pose")]
    pub fn from_matrices(joint_poses: PyReadonlyArray1<f32>, global_trans: PyReadonlyArray1<f32>, up_axis: PyUpAxis, smpl_type: PySmplType) -> Self {
        let joint_poses: nd::Array1<f32> = joint_poses.to_owned_array();
        let joints_3 = joint_poses.len();
        // let joint_poses = joint_poses.into_shape((joints_3 / 3, 3)).unwrap();

        let joint_poses = joint_poses.into_shape_with_order((joints_3 / 3, 3)).unwrap();

        let global_trans: nd::Array1<f32> = global_trans.to_owned_array();

        Self {
            inner: Pose::new(joint_poses, global_trans, UpAxis::from(up_axis), SmplType::from(smpl_type)),
        }
    }
    #[pyo3(signature = ())]
    #[pyo3(text_signature = "($self) -> NDArray[np.float32]")]
    pub fn joint_poses(&mut self, py: Python<'_>) -> Py<PyArray2<f32>> {
        self.inner.joint_poses.to_pyarray_bound(py).into()
    }
}
