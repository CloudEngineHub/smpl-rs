use gloss_hecs::Entity;
use gloss_py_macros::PyComponent;
use gloss_renderer::scene::Scene;
use pyo3::prelude::*;
use smpl_gloss_integration::scene::SceneAnimation;
use std::time::Duration;
#[pyclass(name = "SceneTimer", unsendable)]
#[derive(Clone, PyComponent)]
pub struct PySceneTimer {
    pub inner: u64,
}
#[pymethods]
impl PySceneTimer {
    #[staticmethod]
    #[pyo3(text_signature = "() -> SceneTimer")]
    #[allow(clippy::should_implement_trait)]
    pub fn from_scene(scene_ptr_idx: u64) -> Self {
        Self { inner: scene_ptr_idx }
    }
    #[pyo3(text_signature = "($self) -> None")]
    pub fn pause(&self) {
        let scene_ptr = self.inner as *mut Scene;
        let scene: &Scene = unsafe { &*scene_ptr };
        if let Ok(mut scene_animation) = scene.get_resource::<&mut SceneAnimation>() {
            scene_animation.runner.paused = true;
        }
    }
    #[pyo3(text_signature = "($self) -> None")]
    pub fn play(&self) {
        let scene_ptr = self.inner as *mut Scene;
        let scene: &Scene = unsafe { &*scene_ptr };
        if let Ok(mut scene_animation) = scene.get_resource::<&mut SceneAnimation>() {
            scene_animation.runner.paused = false;
        }
    }
    #[pyo3(text_signature = "($self) -> int")]
    pub fn num_scene_animation_frames(&self) -> usize {
        let scene_ptr = self.inner as *mut Scene;
        let scene: &Scene = unsafe { &*scene_ptr };
        if let Ok(scene_animation) = scene.get_resource::<&mut SceneAnimation>() {
            scene_animation.num_frames
        } else {
            0
        }
    }
    #[pyo3(text_signature = "($self, secs: float) -> None")]
    pub fn advance_sec(&mut self, secs: f32) {
        let scene_ptr = self.inner as *mut Scene;
        let scene: &Scene = unsafe { &*scene_ptr };
        if let Ok(mut scene_animation) = scene.get_resource::<&mut SceneAnimation>() {
            scene_animation.advance(Duration::from_secs_f32(secs), false);
        }
    }
    #[pyo3(text_signature = "($self) -> float")]
    pub fn get_cur_time_sec(&self) -> f32 {
        let scene_ptr = self.inner as *mut Scene;
        let scene: &Scene = unsafe { &*scene_ptr };
        if let Ok(scene_animation) = scene.get_resource::<&mut SceneAnimation>() {
            scene_animation.get_cur_time().as_secs_f32()
        } else {
            0.0
        }
    }
    #[pyo3(text_signature = "($self) -> float")]
    pub fn duration(&self) -> f32 {
        let scene_ptr = self.inner as *mut Scene;
        let scene: &Scene = unsafe { &*scene_ptr };
        if let Ok(scene_animation) = scene.get_resource::<&mut SceneAnimation>() {
            scene_animation.duration().as_secs_f32()
        } else {
            0.0
        }
    }
    #[pyo3(text_signature = "($self) -> bool")]
    pub fn is_finished(&self) -> bool {
        let scene_ptr = self.inner as *mut Scene;
        let scene: &Scene = unsafe { &*scene_ptr };
        if let Ok(scene_animation) = scene.get_resource::<&mut SceneAnimation>() {
            scene_animation.is_finished()
        } else {
            false
        }
    }
}
