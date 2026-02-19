use gloss_renderer::network::{SceneReceiver, SceneSender};
pub mod serializable_components;
use crate::{components::GlossInterop, scene::SceneAnimation};
use serializable_components::{
    SerializableAnimation, SerializableBetas, SerializableGlossInterop, SerializableSceneAnimation, SerializableSmplParams,
    SerializableTransformSequence,
};
use smpl_core::common::{animation::Animation, betas::Betas, smpl_params::SmplParams, transform_sequence::TransformSequence};
pub fn smpl_register_components_for_sender(scene_sender: &mut SceneSender) {
    scene_sender
        .registry_mut()
        .register_component_simple::<SmplParams, SerializableSmplParams>();
    scene_sender.registry_mut().register_component_simple::<Betas, SerializableBetas>();
    scene_sender
        .registry_mut()
        .register_component_simple::<Animation, SerializableAnimation>();
    scene_sender
        .registry_mut()
        .register_component_simple::<GlossInterop, SerializableGlossInterop>();
    scene_sender
        .registry_mut()
        .register_component_simple::<SceneAnimation, SerializableSceneAnimation>();
    scene_sender
        .registry_mut()
        .register_component_simple::<TransformSequence, SerializableTransformSequence>();
}
pub fn smpl_register_components_for_receiver(scene_receiver: &mut SceneReceiver) {
    scene_receiver
        .registry_mut()
        .register_component_simple::<SerializableSmplParams, SmplParams>();
    scene_receiver.registry_mut().register_component_simple::<SerializableBetas, Betas>();
    scene_receiver
        .registry_mut()
        .register_component_simple::<SerializableAnimation, Animation>();
    scene_receiver
        .registry_mut()
        .register_component_simple::<SerializableGlossInterop, GlossInterop>();
    scene_receiver
        .registry_mut()
        .register_component_simple::<SerializableSceneAnimation, SceneAnimation>();
    scene_receiver
        .registry_mut()
        .register_component_simple::<SerializableTransformSequence, TransformSequence>();
}
