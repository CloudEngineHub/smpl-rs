use gloss_hecs::Entity;
use gloss_py_macros::PyComponent;
use gloss_renderer::scene::Scene;
use ndarray as nd;
use numpy::PyArrayMethods;
use numpy::{PyArray1, PyReadonlyArray1, ToPyArray};
use pyo3::prelude::*;
use smpl_rs::common::betas::Betas;

#[pyclass(name = "Betas", module = "smpl_rs.components", unsendable)]
// it has to be unsendable because it does not implement Send: https://pyo3.rs/v0.19.1/class#must-be-send
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
        Self { inner: Betas { betas } }
    }
    #[staticmethod]
    #[pyo3(text_signature = "() -> Betas")]
    #[allow(clippy::should_implement_trait)] //pyo3 doesn't work with traits
    pub fn default() -> Self {
        Self { inner: Betas::default() }
    }
    #[pyo3(text_signature = "($self) -> NDArray[np.float32]")]
    pub fn numpy(&mut self, py: Python<'_>) -> Py<PyArray1<f32>> {
        self.inner.betas.to_pyarray_bound(py).into()
    }
}
