use super::entity_builder::PyEntityBuilderSmplRs;
use gloss_hecs::Entity;
use gloss_py_macros::PyComponent;
use gloss_renderer::scene::Scene;
use ndarray as nd;
use numpy::PyArrayMethods;
use numpy::{PyArray1, PyReadonlyArray1, ToPyArray};
use pyo3::prelude::*;
use smpl_gloss_integration::codec::SmplCodecGloss;
use smpl_rs::codec::codec::SmplCodec;
#[pyclass(name = "SmplCodec", module = "smpl_rs.codec", unsendable)]
#[derive(Clone, PyComponent)]
pub struct PySmplCodec {
    pub inner: SmplCodec,
}
#[pymethods]
impl PySmplCodec {
    #[staticmethod]
    #[pyo3(text_signature = "() -> SmplCodec")]
    #[allow(clippy::should_implement_trait)]
    pub fn default() -> Self {
        Self { inner: SmplCodec::default() }
    }
    #[staticmethod]
    #[pyo3(text_signature = "(buf: NDArray[np.uint8]) -> SmplCodec")]
    pub fn from_buf(py_buf: PyReadonlyArray1<u8>) -> Self {
        let buf: nd::Array1<u8> = py_buf.to_owned_array();
        Self {
            inner: SmplCodec::from_buf(buf.as_slice().unwrap()),
        }
    }
    #[staticmethod]
    #[pyo3(text_signature = "(path: str) -> SmplCodec")]
    pub fn from_file(path: &str) -> Self {
        Self {
            inner: SmplCodec::from_file(path),
        }
    }
    #[pyo3(text_signature = "($self) -> EntityBuilderSmplRs")]
    pub fn to_entity_builder(&self) -> PyEntityBuilderSmplRs {
        let builder = self.inner.to_entity_builder();
        PyEntityBuilderSmplRs::new(builder)
    }
    #[getter]
    pub fn get_shape_parameters(&self, py: Python<'_>) -> Option<Py<PyArray1<f32>>> {
        self.inner.shape_parameters.as_ref().map(|arr| arr.to_pyarray_bound(py).into())
    }
}
