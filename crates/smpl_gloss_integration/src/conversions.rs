use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor};
use gloss_burn_multibackend::backend::MultiBackend;
use gloss_hecs::{CommandBuffer, Entity};
use gloss_renderer::components::FacesGPU;
use gloss_renderer::components::NormalsGPU;
use gloss_renderer::components::TangentsGPU;
use gloss_renderer::components::{BoundingBox, ColorsGPU, UVsGPU, VertsGPU};
use gloss_renderer::{
    components::{Faces, ModelMatrix, Normals, Tangents, UVs, Verts, VisMesh, VisPoints},
    scene::Scene,
};
use gloss_utils::{
    bshare::ToNdArray,
    tensor::{DynamicTensorFloat2D, DynamicTensorInt2D},
};
use log::error;
use nalgebra as na;
use smpl_core::common::{pose::Pose, smpl_model::SmplModel};
use wgpu_burn_interop::interop::tensor_float2wgpu_buffer;
use wgpu_burn_interop::interop::tensor_int2wgpu_buffer;
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
#[allow(clippy::too_many_arguments)]
#[allow(clippy::cast_possible_truncation)]
pub fn update_entity_on_backend_wgpu(
    entity: Entity,
    scene: &Scene,
    gpu: &easy_wgpu::gpu::Gpu,
    commands: &mut CommandBuffer,
    with_uv: bool,
    verts: &Tensor<MultiBackend, 2>,
    normals: &Tensor<MultiBackend, 2>,
    tangents: Option<Tensor<MultiBackend, 2>>,
    uv: &Tensor<MultiBackend, 2>,
    faces: &Tensor<MultiBackend, 2, Int>,
) {
    if !with_uv {
        error!("UVs are required for WGPU backend. Currently only the case of with_uv is supported");
        return;
    }
    let verts_buf = tensor_float2wgpu_buffer(verts.clone(), wgpu::BufferUsages::VERTEX, &gpu.device().clone(), &gpu.queue().clone());
    let uv_buf = tensor_float2wgpu_buffer(uv.clone(), wgpu::BufferUsages::VERTEX, &gpu.device().clone(), &gpu.queue().clone());
    let normals_buf = tensor_float2wgpu_buffer(normals.clone(), wgpu::BufferUsages::VERTEX, &gpu.device().clone(), &gpu.queue().clone());
    let tangents_buf = tangents
        .clone()
        .map(|x| tensor_float2wgpu_buffer(x.clone(), wgpu::BufferUsages::VERTEX, &gpu.device().clone(), &gpu.queue().clone()));
    let tangents_buf = tangents_buf.unwrap();
    let faces_buf = tensor_int2wgpu_buffer(faces.clone(), wgpu::BufferUsages::INDEX, &gpu.device().clone(), &gpu.queue().clone());
    commands.insert_one(
        entity,
        VertsGPU {
            buf: verts_buf.clone(),
            nr_vertices: verts.shape().dims[0] as u32,
        },
    );
    commands.insert_one(
        entity,
        ColorsGPU {
            buf: verts_buf,
            nr_vertices: verts.shape().dims[0] as u32,
        },
    );
    commands.insert_one(
        entity,
        UVsGPU {
            buf: uv_buf,
            nr_vertices: uv.shape().dims[0] as u32,
        },
    );
    commands.insert_one(
        entity,
        NormalsGPU {
            buf: normals_buf,
            nr_vertices: normals.shape().dims[0] as u32,
        },
    );
    commands.insert_one(
        entity,
        TangentsGPU {
            buf: tangents_buf,
            nr_vertices: tangents.unwrap().shape().dims[0] as u32,
        },
    );
    commands.insert_one(
        entity,
        FacesGPU {
            buf: faces_buf,
            nr_triangles: faces.shape().dims[0] as u32,
        },
    );
    if !scene.world.has::<VisPoints>(entity).unwrap() {
        commands.insert_one(
            entity,
            VisPoints {
                added_automatically: true,
                ..Default::default()
            },
        );
    }
    if !scene.world.has::<VisMesh>(entity).unwrap() {
        commands.insert_one(
            entity,
            VisMesh {
                added_automatically: true,
                ..Default::default()
            },
        );
    }
    if let Ok(pose) = scene.get_comp::<&Pose>(&entity) {
        if !scene.world.has::<BoundingBox>(entity).unwrap() {
            let mut center = pose.global_trans.clone().to_ndarray();
            center[1] -= 0.5;
            let scale = na::Vector3::new(1.5, 1.5, 1.5);
            let center_point = na::Point3::<f32>::from_slice(center.as_slice().unwrap());
            let bounding_box = BoundingBox::from_center_and_scale(&center_point, &scale);
            commands.insert_one(entity, bounding_box);
        }
    }
}
