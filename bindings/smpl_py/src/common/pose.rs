use super::types::{PySmplType, PyUpAxis};
#[cfg(feature = "burn-torch")]
use crate::tensor_utils::pytensor2burn;
use gloss_burn_multibackend::backend::MultiDevice;
use gloss_hecs::Entity;
use gloss_py_macros::PyComponent;
use gloss_renderer::scene::Scene;
use gloss_utils::bshare::ToBurn;
use gloss_utils::bshare::ToNdArray;
use ndarray as nd;
use numpy::PyArray1;
use numpy::PyArray2;
use numpy::PyReadonlyArray1;
use numpy::{PyArrayMethods, ToPyArray};
use pyo3::prelude::*;
#[cfg(feature = "burn-torch")]
use pyo3_tch::PyTensor;
use smpl_core::common::{
    pose::Pose,
    types::{SmplType, UpAxis},
};
#[pyclass(name = "Pose", module = "smpl_rs.components", unsendable)]
#[derive(Clone, PyComponent)]
pub struct PyPose {
    pub inner: Pose,
}
#[pymethods]
impl PyPose {
    #[staticmethod]
    #[allow(unused_mut)]
    #[pyo3(text_signature = "(up_axis: UpAxis, smpl_type: SmplType) -> Pose")]
    pub fn new_empty(up_axis: PyUpAxis, smpl_type: PySmplType) -> Self {
        Self {
            inner: Pose::new_empty(UpAxis::from(up_axis), SmplType::from(smpl_type)),
        }
    }
    #[staticmethod]
    #[allow(unused_mut)]
    #[pyo3(text_signature = "(joint_poses: NDArray[np.float32], global_trans: NDArray[np.float32], up_axis: UpAxis, smpl_type: SmplType) -> Pose")]
    pub fn from_matrices(joint_poses: PyReadonlyArray1<f32>, global_trans: PyReadonlyArray1<f32>, up_axis: PyUpAxis, smpl_type: PySmplType) -> Self {
        let joint_poses: nd::Array1<f32> = joint_poses.to_owned_array();
        let joints_3 = joint_poses.len();
        let mut joint_poses = joint_poses.clone().into_shape((joints_3 / 3, 3)).unwrap();

        let global_trans: nd::Array1<f32> = global_trans.to_owned_array();
        let device = MultiDevice::default();
        let joint_poses = joint_poses.to_burn(&device);
        let global_trans = global_trans.to_burn(&device);
        Self {
            inner: Pose::new(joint_poses, global_trans, UpAxis::from(up_axis), SmplType::from(smpl_type)),
        }
    }
    #[cfg(feature = "burn-torch")]
    #[staticmethod]
    #[pyo3(text_signature = "(joint_poses: PyTensor, global_trans: PyTensor, up_axis: UpAxis, smpl_type: SmplType) -> Pose")]
    pub fn from_tensors(joint_poses: PyTensor, global_trans: PyTensor, up_axis: PyUpAxis, smpl_type: PySmplType) -> Self {
        let joint_poses = pytensor2burn::<1>(joint_poses);
        let global_trans = pytensor2burn::<1>(global_trans);
        let mut joint_poses = joint_poses.clone().reshape([joint_poses.dims()[0] / 3, 3]);

        Self {
            inner: Pose::new(joint_poses, global_trans, UpAxis::from(up_axis), SmplType::from(smpl_type)),
        }
    }
    #[pyo3(text_signature = "(&self, pose_other: &Pose, weight_other: f32, use_slerp: bool) -> Pose")]
    #[must_use]
    pub fn interpolate(&self, pose_other: &Self, weight_other: f32, use_slerp: bool) -> Self {
        let pose_interp = self.inner.interpolate(&pose_other.inner, weight_other, use_slerp);
        Self { inner: pose_interp }
    }
    #[pyo3(signature = ())]
    #[pyo3(text_signature = "($self) -> NDArray[np.float32]")]
    pub fn joint_poses(&self, py: Python<'_>) -> Py<PyArray2<f32>> {
        self.inner.joint_poses.to_ndarray().to_pyarray_bound(py).into()
    }
    #[pyo3(signature = ())]
    #[pyo3(text_signature = "($self) -> NDArray[np.float32]")]
    pub fn global_trans(&self, py: Python<'_>) -> Py<PyArray1<f32>> {
        self.inner.global_trans.to_ndarray().to_pyarray_bound(py).into()
    }
}
