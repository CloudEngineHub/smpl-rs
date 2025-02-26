use gloss_hecs::Entity;
use gloss_py_macros::PyComponent;
use gloss_renderer::scene::Scene;
use pyo3::prelude::*;
use smpl_gloss_integration::components::GlossInterop;
#[pyclass(name = "GlossInterop", module = "smpl_rs.components", unsendable)]
#[derive(Clone, PyComponent)]
pub struct PyGlossInterop {
    pub inner: GlossInterop,
}
#[pymethods]
impl PyGlossInterop {
    #[new]
    #[pyo3(text_signature = "(with_uv: bool) -> GlossInterop")]
    pub fn new(with_uv: bool) -> Self {
        Self {
            inner: GlossInterop { with_uv },
        }
    }
    #[staticmethod]
    #[pyo3(text_signature = "() -> GlossInterop")]
    #[allow(clippy::should_implement_trait)]
    pub fn default() -> Self {
        Self {
            inner: GlossInterop { with_uv: true },
        }
    }
}
