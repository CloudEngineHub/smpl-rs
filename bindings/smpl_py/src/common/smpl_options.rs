use gloss_hecs::Entity;
use gloss_py_macros::PyComponent;
use gloss_renderer::scene::Scene;

use pyo3::prelude::*;
use smpl_rs::common::smpl_options::SmplOptions;

#[pyclass(name = "SmplOptions", module = "smpl_rs.components", unsendable)]
// it has to be unsendable because it does not implement Send: https://pyo3.rs/v0.19.1/class#must-be-send
#[derive(Clone, PyComponent)]
pub struct PySmplOptions {
    pub inner: SmplOptions,
}
#[pymethods]
impl PySmplOptions {
    #[new]
    #[pyo3(text_signature = "(enable_pose_corrective: bool) -> SmplOptions")]
    pub fn new(enable_pose_corrective: bool) -> Self {
        Self {
            inner: SmplOptions { enable_pose_corrective },
        }
    }
    #[staticmethod]
    #[pyo3(text_signature = "() -> SmplOptions")]
    #[allow(clippy::should_implement_trait)] //pyo3 doesn't work with traits
    pub fn default() -> Self {
        Self {
            inner: SmplOptions {
                enable_pose_corrective: true,
            },
        }
    }
}
