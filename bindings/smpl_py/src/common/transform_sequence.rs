use gloss_hecs::Entity;
use gloss_py_macros::PyComponent;
use gloss_renderer::scene::Scene;
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
}
