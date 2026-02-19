use super::animation::PyAnimWrap;
use gloss_hecs::Entity;
use gloss_py_macros::PyComponent;
use gloss_renderer::scene::Scene;
use pyo3::prelude::*;
use smpl_core::common::animation::{AnimWrap, AnimationConfig};
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
    #[staticmethod]
    #[pyo3(text_signature = "(num_frames: usize, fps: f32, wrap_behaviour: AnimWrap) -> SceneAnimation")]
    pub fn new_with_fps_and_wrap(num_frames: usize, fps: f32, wrap_behaviour: PyAnimWrap) -> Self {
        let anim_config = AnimationConfig {
            fps,
            wrap_behaviour: AnimWrap::from(wrap_behaviour),
            ..Default::default()
        };
        Self {
            inner: SceneAnimation::new_with_config(num_frames, anim_config),
        }
    }
}
