use super::entity_builder::PyEntityBuilderSmplRs;
use gloss_hecs::Entity;
use gloss_py_macros::PyComponent;
use gloss_renderer::scene::Scene;
use pyo3::prelude::*;
use smpl_core::codec::scene::McsCodec;
use smpl_gloss_integration::scene::McsCodecGloss;
#[pyclass(name = "McsCodec", module = "smpl_rs.codec", unsendable)]
#[derive(Clone, PyComponent)]
pub struct PyMcsCodec {
    pub inner: McsCodec,
}
#[pymethods]
impl PyMcsCodec {
    #[staticmethod]
    #[pyo3(text_signature = "(path: str) -> McsCodec")]
    pub fn from_file(path: &str) -> Self {
        Self {
            inner: McsCodec::from_file(path),
        }
    }
    #[pyo3(text_signature = "($self) -> List[EntityBuilderSmplRs]")]
    pub fn to_entity_builders(&mut self) -> Vec<PyEntityBuilderSmplRs> {
        let builder = self.inner.to_entity_builders();
        builder.into_iter().map(PyEntityBuilderSmplRs::new).collect()
    }
    #[getter]
    pub fn num_frames(&self) -> usize {
        self.inner.num_frames
    }
    #[getter]
    pub fn frame_rate(&self) -> f32 {
        self.inner.frame_rate
    }
    #[getter]
    pub fn num_bodies(&self) -> usize {
        self.inner.smpl_bodies.len()
    }
    #[getter]
    pub fn has_camera(&self) -> bool {
        self.inner.camera_track.is_some()
    }
}
