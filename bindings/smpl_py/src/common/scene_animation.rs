use gloss_hecs::Entity;
use gloss_py_macros::PyComponent;
use gloss_renderer::scene::Scene;
use pyo3::prelude::*;
use smpl_gloss_integration::scene::SceneAnimation;
#[pyclass(name = "SceneAnimation", module = "smpl_rs.components", unsendable)]
#[derive(Clone, PyComponent)]
pub struct PySceneAnimation {
    pub inner: SceneAnimation,
}
#[pymethods]
impl PySceneAnimation {
    #[staticmethod]
    #[pyo3(text_signature = "(num_frames: usize, fps: f32) -> SceneAnimation")]
    pub fn new_with_fps(num_frames: usize, fps: f32) -> Self {
        Self {
            inner: SceneAnimation::new_with_fps(num_frames, fps),
        }
    }
}
