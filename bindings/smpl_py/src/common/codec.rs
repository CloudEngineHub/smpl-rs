use super::entity_builder::PyEntityBuilderSmplRs;
use crate::common::types::{PyGender, PySmplType};
use gloss_hecs::Entity;
use gloss_py_macros::PyComponent;
use gloss_renderer::scene::Scene;
use ndarray as nd;
use numpy::PyArrayMethods;
use numpy::{PyArray1, PyArray2, PyArray3, PyReadonlyArray1, ToPyArray};
use pyo3::prelude::*;
use smpl_core::codec::codec::SmplCodec;
use smpl_core::common::types::{Gender, SmplType};
use smpl_gloss_integration::codec::SmplCodecGloss;
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
    pub fn smpl_version(&self) -> i32 {
        self.inner.smpl_version
    }
    #[getter]
    pub fn smpl_type(&self) -> PySmplType {
        let smpl_type: SmplType = self.inner.smpl_type();
        smpl_type.into()
    }
    #[getter]
    pub fn gender(&self) -> PyGender {
        let gender: Gender = self.inner.gender();
        gender.into()
    }
    #[getter]
    pub fn frame_count(&self) -> i32 {
        self.inner.frame_count
    }
    #[getter]
    pub fn frame_rate(&self) -> Option<f32> {
        self.inner.frame_rate
    }
    #[getter]
    pub fn shape_parameters(&self, py: Python<'_>) -> Option<Py<PyArray1<f32>>> {
        self.inner.shape_parameters.as_ref().map(|arr| arr.to_pyarray_bound(py).into())
    }
    #[getter]
    pub fn expression_parameters(&self, py: Python<'_>) -> Option<Py<PyArray2<f32>>> {
        self.inner.expression_parameters.as_ref().map(|arr| arr.to_pyarray_bound(py).into())
    }
    #[getter]
    pub fn body_translation(&self, py: Python<'_>) -> Option<Py<PyArray2<f32>>> {
        self.inner.body_translation.as_ref().map(|arr| arr.to_pyarray_bound(py).into())
    }
    #[getter]
    pub fn body_pose(&self, py: Python<'_>) -> Option<Py<PyArray3<f32>>> {
        self.inner.body_pose.as_ref().map(|arr| arr.to_pyarray_bound(py).into())
    }
    #[getter]
    pub fn head_pose(&self, py: Python<'_>) -> Option<Py<PyArray3<f32>>> {
        self.inner.head_pose.as_ref().map(|arr| arr.to_pyarray_bound(py).into())
    }
    #[getter]
    pub fn left_hand_pose(&self, py: Python<'_>) -> Option<Py<PyArray3<f32>>> {
        self.inner.left_hand_pose.as_ref().map(|arr| arr.to_pyarray_bound(py).into())
    }
    #[getter]
    pub fn right_hand_pose(&self, py: Python<'_>) -> Option<Py<PyArray3<f32>>> {
        self.inner.right_hand_pose.as_ref().map(|arr| arr.to_pyarray_bound(py).into())
    }
    #[getter]
    pub fn vertex_offsets(&self, py: Python<'_>) -> Option<Py<PyArray2<f32>>> {
        self.inner.vertex_offsets.as_ref().map(|arr| arr.to_pyarray_bound(py).into())
    }
}
