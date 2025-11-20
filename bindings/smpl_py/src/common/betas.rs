use gloss_burn_multibackend::backend::MultiDevice;
use gloss_hecs::Entity;
use gloss_py_macros::PyComponent;
use gloss_renderer::scene::Scene;
use gloss_utils::bshare::{ToBurn, ToNdArray};
use ndarray as nd;
use numpy::PyArrayMethods;
use numpy::{PyArray1, PyReadonlyArray1, ToPyArray};
use pyo3::prelude::*;
use smpl_core::common::betas::Betas;
#[pyclass(name = "Betas", module = "smpl_rs.components", unsendable)]
#[derive(Clone, PyComponent)]
pub struct PyBetas {
    pub inner: Betas,
}
#[pymethods]
impl PyBetas {
    #[new]
    #[pyo3(text_signature = "(array: NDArray[np.float32]) -> Betas")]
    pub fn new(array: PyReadonlyArray1<f32>) -> Self {
        let betas: nd::Array1<f32> = array.to_owned_array();
        let device = MultiDevice::default();
        let betas = betas.to_burn(&device);
        Self { inner: Betas::new(betas) }
    }
    #[staticmethod]
    #[pyo3(text_signature = "() -> Betas")]
    #[allow(clippy::should_implement_trait)]
    pub fn default() -> Self {
        Self { inner: Betas::default() }
    }
    #[pyo3(text_signature = "($self) -> NDArray[np.float32]")]
    pub fn numpy(&mut self, py: Python<'_>) -> Py<PyArray1<f32>> {
        self.inner.betas.to_ndarray().to_pyarray_bound(py).into()
    }
}
