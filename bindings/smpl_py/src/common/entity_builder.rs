use gloss_hecs::{Entity, EntityBuilder};
use gloss_renderer::scene::Scene;
use pyo3::prelude::*;
#[pyclass(name = "EntityBuilderSmplRs", module = "smpl_rs.builders", unsendable)]
pub struct PyEntityBuilderSmplRs {
    pub inner: Option<EntityBuilder>,
}
impl PyEntityBuilderSmplRs {
    pub fn new(builder: EntityBuilder) -> Self {
        Self { inner: Some(builder) }
    }
}
#[pymethods]
impl PyEntityBuilderSmplRs {
    #[pyo3(text_signature = "($self, entity_bits: int, scene_ptr_idx: int) -> None")]
    pub fn insert_to_entity(&mut self, entity_bits: u64, scene_ptr_idx: u64) {
        let entity = Entity::from_bits(entity_bits).unwrap();
        let scene_ptr = scene_ptr_idx as *mut Scene;
        let scene: &mut Scene = unsafe { &mut *scene_ptr };
        scene.world.insert(entity, self.inner.take().unwrap().build()).ok();
    }
}
