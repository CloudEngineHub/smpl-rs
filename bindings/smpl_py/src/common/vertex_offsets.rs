#[cfg(feature = "burn-torch")]
use crate::tensor_utils::pytensor2burn;
use gloss_burn_multibackend::backend::MultiDevice;
use gloss_hecs::Entity;
use gloss_py_macros::PyComponent;
use gloss_renderer::scene::Scene;
use gloss_utils::bshare::{ToBurn, ToNdArray};
use ndarray as nd;
use numpy::PyArrayMethods;
use numpy::{PyArray2, PyReadonlyArray2, ToPyArray};
use pyo3::prelude::*;
#[cfg(feature = "burn-torch")]
use pyo3_tch::PyTensor;
use smpl_core::common::vertex_offsets::VertexOffsets;
#[pyclass(name = "VertexOffsets", module = "smpl_rs.components", unsendable)]
#[derive(Clone, PyComponent)]
pub struct PyVertexOffsets {
    pub inner: VertexOffsets,
}
#[pymethods]
impl PyVertexOffsets {
    #[new]
    #[pyo3(text_signature = "(array: NDArray[np.float32]) -> VertexOffsets")]
    pub fn new(array: PyReadonlyArray2<f32>) -> Self {
        let offsets: nd::Array2<f32> = array.to_owned_array();
        let device = MultiDevice::default();
        let offsets = offsets.to_burn(&device);
        Self {
            inner: VertexOffsets::new(offsets),
        }
    }
    #[cfg(feature = "burn-torch")]
    #[staticmethod]
    #[pyo3(text_signature = "(tensor: PyTensor) -> VertexOffsets")]
    pub fn from_tensor(tensor: PyTensor) -> Self {
        let tensor_burn = pytensor2burn::<2>(tensor);
        Self {
            inner: VertexOffsets::new(tensor_burn),
        }
    }
    #[staticmethod]
    #[pyo3(signature = (path_smpl_file))]
    #[pyo3(text_signature = "(path_smpl_file: str) -> VertexOffsets")]
    pub fn from_smpl_file(path_smpl_file: &str) -> Option<Self> {
        VertexOffsets::new_from_smpl_file(path_smpl_file).map(|inner| Self { inner })
    }
    #[staticmethod]
    #[pyo3(signature = (path_npz))]
    #[pyo3(text_signature = "(path_npz: str) -> VertexOffsets")]
    pub fn from_npz(path_npz: &str) -> Self {
        Self {
            inner: VertexOffsets::new_from_npz(path_npz),
        }
    }
    #[getter]
    pub fn strength(&self) -> f32 {
        self.inner.strength
    }
    #[setter]
    pub fn set_strength(&mut self, strength: f32) {
        self.inner.strength = strength;
    }
    #[pyo3(text_signature = "($self) -> NDArray[np.float32]")]
    pub fn numpy(&mut self, py: Python<'_>) -> Py<PyArray2<f32>> {
        self.inner.offsets.to_ndarray().to_pyarray_bound(py).into()
    }
}
