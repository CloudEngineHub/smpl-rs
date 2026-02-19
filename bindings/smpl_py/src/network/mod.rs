use gloss_renderer::network::SceneReceiver;
use gloss_renderer::network::SceneSender;
use pyo3::prelude::*;
use smpl_gloss_integration::network::smpl_register_components_for_receiver;
use smpl_gloss_integration::network::smpl_register_components_for_sender;
#[pyfunction]
#[pyo3(name = "smpl_register_components_for_sender")]
#[pyo3(text_signature = "(scene_sender_ptr) -> None")]
pub fn py_smpl_register_components_for_sender(scene_sender_ptr: u64) {
    let scene_sender_ptr = scene_sender_ptr as *mut SceneSender;
    let scene_sender: &mut SceneSender = unsafe { &mut *scene_sender_ptr };
    smpl_register_components_for_sender(scene_sender);
}
#[pyfunction]
#[pyo3(name = "smpl_register_components_for_receiver")]
#[pyo3(text_signature = "(scene_receiver_ptr) -> None")]
pub fn py_smpl_register_components_for_receiver(scene_receiver_ptr: u64) {
    let scene_receiver_ptr = scene_receiver_ptr as *mut SceneReceiver;
    let scene_receiver: &mut SceneReceiver = unsafe { &mut *scene_receiver_ptr };
    smpl_register_components_for_receiver(scene_receiver);
}
