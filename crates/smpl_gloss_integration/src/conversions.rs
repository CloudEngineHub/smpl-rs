use burn::tensor::backend::Backend;
use gloss_hecs::{CommandBuffer, Entity};
use gloss_renderer::{
    components::{Faces, ModelMatrix, Normals, Tangents, UVs, Verts, VisMesh, VisPoints},
    scene::Scene,
};
use gloss_utils::tensor::{DynamicTensorFloat2D, DynamicTensorInt2D};
use smpl_core::common::smpl_model::SmplModel;
/// Insert vertices and vertex attributes for the entity based on changes made
/// to it, on a generic Burn Backend. We currently support - ``Candle``,
/// ``NdArray``, and ``Wgpu``
#[allow(clippy::too_many_arguments)]
#[allow(clippy::similar_names)]
#[allow(clippy::too_many_lines)]
pub fn update_entity_on_backend<B: Backend>(
    entity: Entity,
    scene: &Scene,
    commands: &mut CommandBuffer,
    with_uv: bool,
    new_verts: &DynamicTensorFloat2D,
    new_normals: &DynamicTensorFloat2D,
    new_tangents: Option<DynamicTensorFloat2D>,
    uv: DynamicTensorFloat2D,
    faces: DynamicTensorInt2D,
    _smpl_model: &dyn SmplModel<B>,
) {
    if with_uv && !scene.world.has::<UVs>(entity).unwrap() {
        commands.insert_one(entity, UVs(uv));
    }
    if with_uv {
        if let Some(tangents) = new_tangents {
            commands.insert_one(entity, Tangents(tangents.clone()));
        }
    }
    if !scene.world.has::<Faces>(entity).unwrap() {
        commands.insert_one(entity, Faces(faces));
    }
    commands.insert_one(entity, Normals(new_normals.clone()));
    commands.insert_one(entity, Verts(new_verts.clone()));
    if !scene.world.has::<VisMesh>(entity).unwrap() {
        commands.insert_one(
            entity,
            VisMesh {
                added_automatically: true,
                ..Default::default()
            },
        );
    }
    if !scene.world.has::<VisPoints>(entity).unwrap() {
        commands.insert_one(
            entity,
            VisPoints {
                added_automatically: true,
                ..Default::default()
            },
        );
    }
    if !scene.world.has::<ModelMatrix>(entity).unwrap() {
        commands.insert_one(entity, ModelMatrix::default());
    }
}
