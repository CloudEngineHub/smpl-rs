use gloss_hecs::Entity;
use gloss_py_macros::PyComponent;
use gloss_renderer::scene::Scene;
use ndarray as nd;
use numpy::{PyArrayMethods, PyReadonlyArray2};
use pyo3::prelude::*;
use smpl_core::common::transform_sequence::TransformSequence;
#[pyclass(name = "TransformSequence", module = "smpl_rs.components", unsendable)]
#[derive(Clone, PyComponent)]
pub struct PyTransformSequence {
    pub inner: TransformSequence,
}
#[pymethods]
impl PyTransformSequence {
    #[staticmethod]
    #[pyo3(text_signature = "(path: str) -> TransformSequence")]
    pub fn new_from_npz(path: &str) -> Self {
        Self {
            inner: TransformSequence::new_from_npz(path),
        }
    }
    #[staticmethod]
    #[pyo3(text_signature = "(rot: NDArray[np.float32], trans: NDArray[np.float32]) -> TransformSequence")]
    pub fn new_from_quat_rot_trans(rot: PyReadonlyArray2<f32>, trans: PyReadonlyArray2<f32>) -> Self {
        let rot: nd::Array2<f32> = rot.to_owned_array();
        let trans: nd::Array2<f32> = trans.to_owned_array();
        Self {
            inner: TransformSequence::new_from_quat_rot_trans(&rot, &trans),
        }
    }
    #[staticmethod]
    #[pyo3(text_signature = "(rot: NDArray[np.float32], trans: NDArray[np.float32]) -> TransformSequence")]
    pub fn new_from_axisangle_rot_trans(rot: PyReadonlyArray2<f32>, trans: PyReadonlyArray2<f32>) -> Self {
        let rot: nd::Array2<f32> = rot.to_owned_array();
        let trans: nd::Array2<f32> = trans.to_owned_array();
        Self {
            inner: TransformSequence::new_from_axisangle_rot_trans(&rot, &trans),
        }
    }
}
