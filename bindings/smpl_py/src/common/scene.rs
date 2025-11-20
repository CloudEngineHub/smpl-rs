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
    #[staticmethod]
    #[pyo3(text_signature = "(scene_ptr_idx: int) -> McsCodec")]
    pub fn from_scene(scene_ptr_idx: u64) -> Self {
        let scene_ptr = scene_ptr_idx as *mut Scene;
        let scene: &Scene = unsafe { &*scene_ptr };
        Self {
            inner: McsCodec::from_scene(scene),
        }
    }
    #[pyo3(text_signature = "($self, path: str) -> None")]
    pub fn to_file(&mut self, path: &str) {
        self.inner.to_file(path);
    }
    #[pyo3(text_signature = "($self, with_colors: bool = True) -> List[EntityBuilderSmplRs]")]
    pub fn to_entity_builders(&mut self, with_colors: bool) -> Vec<PyEntityBuilderSmplRs> {
        let builder = self.inner.to_entity_builders(with_colors);
        builder.into_iter().map(PyEntityBuilderSmplRs::new).collect()
    }
    #[pyo3(text_signature = "($self, scene_ptr_idx: int, with_colors: bool = True) -> None")]
    pub fn insert_into_scene(&mut self, scene_ptr_idx: u64, with_colors: bool) {
        let scene_ptr = scene_ptr_idx as *mut Scene;
        let scene: &mut Scene = unsafe { &mut *scene_ptr };
        self.inner.insert_into_scene(scene, with_colors);
    }
    #[getter]
    pub fn num_frames(&self) -> usize {
        self.inner.num_frames
    }
    #[getter]
    pub fn frame_rate(&self) -> Option<f32> {
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
