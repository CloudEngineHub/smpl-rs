use super::types::{PyGender, PySmplType};
use gloss_hecs::Entity;
use gloss_py_macros::PyComponent;
use gloss_renderer::scene::Scene;
use pyo3::prelude::*;
use smpl_core::common::{
    smpl_params::SmplParams,
    types::{Gender, SmplType},
};
#[pyclass(name = "SmplParams", module = "smpl_rs.components", unsendable)]
#[derive(Clone, PyComponent)]
pub struct PySmplParams {
    pub inner: SmplParams,
}
#[pymethods]
impl PySmplParams {
    #[new]
    #[pyo3(text_signature = "(smpl_type: SmplType, gender: Gender, enable_pose_corrective: bool) -> SmplParams")]
    pub fn new(smpl_type: PySmplType, gender: PyGender, enable_pose_corrective: bool) -> Self {
        Self {
            inner: SmplParams {
                smpl_type: SmplType::from(smpl_type),
                gender: Gender::from(gender),
                enable_pose_corrective,
            },
        }
    }
    #[staticmethod]
    #[pyo3(text_signature = "() -> SmplParams")]
    #[allow(clippy::should_implement_trait)]
    pub fn default() -> Self {
        Self {
            inner: SmplParams {
                smpl_type: SmplType::SmplX,
                gender: Gender::Neutral,
                enable_pose_corrective: true,
            },
        }
    }
    #[getter]
    pub fn gender(&self) -> PyGender {
        self.inner.gender.into()
    }
    #[getter]
    pub fn smpl_type(&self) -> PySmplType {
        self.inner.smpl_type.into()
    }
}
