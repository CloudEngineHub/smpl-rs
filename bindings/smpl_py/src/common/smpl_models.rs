use super::types::{PyGender, PySmplType};
use crate::smpl_x::smpl_x::PySmplX;
use burn::backend::Candle;
use gloss_hecs::Entity;
use gloss_renderer::scene::Scene;
use pyo3::prelude::*;
use smpl_rs::smpl_x::smpl_x_gpu::SmplXDynamic;
use smpl_rs::{
    common::smpl_model::{SmplCache, SmplCacheDynamic},
    smpl_x::smpl_x_gpu::SmplXGPU,
};
#[pyclass(name = "SmplCache", module = "smpl_rs.models", unsendable)]
pub struct PySmplModels {
    inner: Option<SmplCacheDynamic>,
}
#[pymethods]
impl PySmplModels {
    #[staticmethod]
    #[pyo3(text_signature = "() -> SmplCache")]
    #[allow(clippy::should_implement_trait)]
    pub fn default() -> Self {
        Self {
            inner: Some(SmplCacheDynamic::default()),
        }
    }
    #[pyo3(text_signature = "($self, model: SmplX, cache_models: bool) -> None")]
    pub fn add_model(&mut self, py_model: &PySmplX, cache_models: bool) {
        self.inner
            .as_mut()
            .unwrap()
            .add_model_from_dynamic_device(py_model.inner.clone(), cache_models);
    }
    #[pyo3(text_signature = "($self, smpl_type: SmplType, gender: Gender) -> Union[SmplX, SmplPP]")]
    pub fn get_model(&mut self, py: Python<'_>, smpl_type: PySmplType, gender: PyGender) -> Py<PyAny> {
        println!("get_model {:?}", self.inner.is_some());
        match &self.inner.as_ref().unwrap() {
            SmplCacheDynamic::Candle(candle_model) => {
                let ref_smpl_dyn = candle_model.get_model_box_ref(smpl_type.into(), gender.into()).unwrap();
                let box_smpl = (**ref_smpl_dyn).clone_dyn();
                match box_smpl.as_any() {
                    smpl if smpl.downcast_ref::<SmplXGPU<Candle>>().is_some() => {
                        let smplx = smpl.downcast_ref::<SmplXGPU<Candle>>().unwrap();
                        let pysmplx = PySmplX {
                            inner: SmplXDynamic::Candle(smplx.clone()),
                        };
                        Py::new(py, pysmplx).unwrap().to_object(py)
                    }
                    _ => {
                        panic!("We haven't yet implemented other models apart from SMPLX and SMPL++ for Candle")
                    }
                }
            }
            _ => panic!("SmplPy does support non-candle backends yet!"),
        }
    }
    #[pyo3(text_signature = "($self, smpl_type: SmplType, gender: Gender, path: str) -> None")]
    pub fn set_lazy_loading(&mut self, smpl_type: PySmplType, gender: PyGender, path: &str) {
        self.inner.as_mut().unwrap().set_lazy_loading(smpl_type.into(), gender.into(), path);
    }
    #[pyo3(text_signature = "(self, entity_bits: int, scene_ptr_idx: int) -> None")]
    pub fn insert_to_entity(&mut self, entity_bits: u64, scene_ptr_idx: u64) {
        let entity = Entity::from_bits(entity_bits).unwrap();
        let scene_ptr = scene_ptr_idx as *mut Scene;
        let scene: &mut Scene = unsafe { &mut *scene_ptr };
        scene.world.insert_one(entity, self.inner.take().unwrap()).ok();
    }
    #[staticmethod]
    #[pyo3(text_signature = "(entity_bits: int, scene_ptr_idx: int) -> SmplCache")]
    pub fn get(entity_bits: u64, scene_ptr_idx: u64) -> Self {
        let entity = Entity::from_bits(entity_bits).unwrap();
        let scene_ptr = scene_ptr_idx as *mut Scene;
        let scene: &mut Scene = unsafe { &mut *scene_ptr };
        let comp = scene.get_comp::<&mut SmplCache<Candle>>(&entity).unwrap();
        Self {
            inner: Some(SmplCacheDynamic::Candle(comp.clone())),
        }
    }
}
