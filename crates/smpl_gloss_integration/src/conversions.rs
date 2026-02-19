use crate::systems::MeshData;
use burn::tensor::backend::Backend;
#[cfg(feature = "burn-torch")]
use burn::tensor::{Int, Tensor};
use gloss_burn_multibackend::backend::MultiBackend;
use gloss_hecs::{CommandBuffer, Entity};
use gloss_renderer::components::FacesGPU;
use gloss_renderer::components::NormalsGPU;
use gloss_renderer::components::TangentsGPU;
use gloss_renderer::components::{BoundingBox, ColorsGPU, UVsGPU, VertsGPU};
#[cfg(feature = "burn-torch")]
use gloss_renderer::components::{ColorsPyTensor, FacesPyTensor, NormalsPyTensor, TangentsPyTensor, UVsPyTensor, VertsPyTensor};
use gloss_renderer::{
    components::{Faces, ModelMatrix, Normals, Tangents, UVs, Verts, VisMesh, VisPoints},
    scene::Scene,
};
use gloss_utils::bshare::ToNalgebraFloat;
use gloss_utils::bshare::ToNalgebraInt;
use gloss_utils::bshare::ToNdArray;
use log::error;
use nalgebra as na;
use smpl_core::common::pose::Pose;
#[cfg(feature = "burn-torch")]
use std::sync::Arc;
use wgpu_burn_interop::interop::tensor_float2wgpu_buffer;
use wgpu_burn_interop::interop::tensor_int2wgpu_buffer;
/// Insert vertices and vertex attributes for the entity based on changes made
/// to it, on a generic Burn Backend. We currently support - ``Candle``,
/// ``NdArray``, and ``Wgpu``
#[allow(clippy::too_many_arguments)]
#[allow(clippy::similar_names)]
#[allow(clippy::too_many_lines)]
pub fn update_entity_cpu<B: Backend>(entity: Entity, scene: &Scene, commands: &mut CommandBuffer, with_uv: bool, mesh_data: MeshData<B>) {
    let verts = mesh_data.verts.to_nalgebra();
    let normals = mesh_data.normals.to_nalgebra();
    let tangents = mesh_data.tangents.map(|t| t.to_nalgebra());
    let uv = mesh_data.uv.to_nalgebra();
    let faces = mesh_data.faces.to_nalgebra();
    if with_uv && !scene.world().has::<UVs>(entity).unwrap() {
        commands.insert_one(entity, UVs(uv));
    }
    if with_uv {
        if let Some(tangents) = tangents {
            commands.insert_one(entity, Tangents(tangents.clone()));
        }
    }
    if !scene.world().has::<Faces>(entity).unwrap() {
        commands.insert_one(entity, Faces(faces));
    }
    commands.insert_one(entity, Normals(normals.clone()));
    commands.insert_one(entity, Verts(verts.clone()));
    if !scene.world().has::<VisMesh>(entity).unwrap() {
        commands.insert_one(
            entity,
            VisMesh {
                added_automatically: true,
                ..Default::default()
            },
        );
    }
    if !scene.world().has::<VisPoints>(entity).unwrap() {
        commands.insert_one(
            entity,
            VisPoints {
                added_automatically: true,
                ..Default::default()
            },
        );
    }
    if !scene.world().has::<ModelMatrix>(entity).unwrap() {
        commands.insert_one(entity, ModelMatrix::default());
    }
}
#[allow(clippy::too_many_arguments)]
#[allow(clippy::cast_possible_truncation)]
pub fn update_entity_wgpu(
    entity: Entity,
    scene: &Scene,
    gpu: &easy_wgpu::gpu::Gpu,
    commands: &mut CommandBuffer,
    with_uv: bool,
    mesh_data: MeshData<MultiBackend>,
) {
    let verts = mesh_data.verts;
    let uv = mesh_data.uv;
    let tangents = mesh_data.tangents.clone();
    let normals = mesh_data.normals;
    let faces = mesh_data.faces;
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
            buf: easy_wgpu::buffer::Buffer::new_from_buffer(verts_buf.clone()),
            nr_vertices: verts.shape().dims[0] as u32,
        },
    );
    commands.insert_one(
        entity,
        ColorsGPU {
            buf: easy_wgpu::buffer::Buffer::new_from_buffer(verts_buf),
            nr_vertices: verts.shape().dims[0] as u32,
        },
    );
    commands.insert_one(
        entity,
        UVsGPU {
            buf: easy_wgpu::buffer::Buffer::new_from_buffer(uv_buf),
            nr_vertices: uv.shape().dims[0] as u32,
        },
    );
    commands.insert_one(
        entity,
        NormalsGPU {
            buf: easy_wgpu::buffer::Buffer::new_from_buffer(normals_buf),
            nr_vertices: normals.shape().dims[0] as u32,
        },
    );
    commands.insert_one(
        entity,
        TangentsGPU {
            buf: easy_wgpu::buffer::Buffer::new_from_buffer(tangents_buf),
            nr_vertices: tangents.unwrap().shape().dims[0] as u32,
        },
    );
    commands.insert_one(
        entity,
        FacesGPU {
            buf: easy_wgpu::buffer::Buffer::new_from_buffer(faces_buf),
            nr_triangles: faces.shape().dims[0] as u32,
        },
    );
    if !scene.world().has::<VisPoints>(entity).unwrap() {
        commands.insert_one(
            entity,
            VisPoints {
                added_automatically: true,
                ..Default::default()
            },
        );
    }
    if !scene.world().has::<VisMesh>(entity).unwrap() {
        commands.insert_one(
            entity,
            VisMesh {
                added_automatically: true,
                ..Default::default()
            },
        );
    }
    if let Ok(pose) = scene.get_comp::<&Pose>(&entity) {
        if !scene.world().has::<BoundingBox>(entity).unwrap() {
            let mut center = pose.global_trans.clone().to_ndarray();
            center[1] -= 0.5;
            let scale = na::Vector3::new(1.5, 1.5, 1.5);
            let center_point = na::Point3::<f32>::from_slice(center.as_slice().unwrap());
            let bounding_box = BoundingBox::from_center_and_scale(&center_point, &scale);
            commands.insert_one(entity, bounding_box);
        }
    }
}
#[cfg(feature = "burn-torch")]
pub fn tensor_to_torch<const D: usize>(t: Tensor<MultiBackend, D>) -> tch::Tensor {
    let prim = t.into_primitive();
    match prim {
        burn::tensor::TensorPrimitive::Float(t) => match t {
            gloss_burn_multibackend::tensor::MultiFloatTensor::Candle(_) => todo!(),
            gloss_burn_multibackend::tensor::MultiFloatTensor::Wgpu(_) => todo!(),
            gloss_burn_multibackend::tensor::MultiFloatTensor::Torch(t) => t.clone().tensor,
            _ => todo!(),
        },
        burn::tensor::TensorPrimitive::QFloat(_) => todo!(),
    }
}
#[cfg(feature = "burn-torch")]
pub fn tensor_int_to_torch<const D: usize>(t: Tensor<MultiBackend, D, Int>) -> tch::Tensor {
    let prim = t.into_primitive();
    match prim {
        gloss_burn_multibackend::tensor::MultiIntTensor::Candle(_) => todo!(),
        gloss_burn_multibackend::tensor::MultiIntTensor::Wgpu(_) => todo!(),
        gloss_burn_multibackend::tensor::MultiIntTensor::Torch(t) => t.clone().tensor,
        _ => todo!(),
    }
}
#[cfg(feature = "burn-torch")]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::cast_possible_truncation)]
pub fn update_entity_cuda(
    entity: Entity,
    scene: &Scene,
    _gpu: &easy_wgpu::gpu::Gpu,
    commands: &mut CommandBuffer,
    with_uv: bool,
    mesh_data: MeshData<MultiBackend>,
) {
    let verts = mesh_data.verts.clone();
    let uv = mesh_data.uv;
    let tangents = mesh_data.tangents.clone();
    let normals = mesh_data.normals;
    let faces = mesh_data.faces;
    if !with_uv {
        error!("UVs are required for WGPU backend. Currently only the case of with_uv is supported");
        return;
    }
    commands.insert_one(
        entity,
        VertsPyTensor {
            tensor: Arc::new(tensor_to_torch(verts.clone().unsqueeze_dim::<3>(0))),
        },
    );
    commands.insert_one(
        entity,
        ColorsPyTensor {
            tensor: Arc::new(tensor_to_torch(verts.clone().unsqueeze_dim::<3>(0))),
        },
    );
    commands.insert_one(
        entity,
        UVsPyTensor {
            tensor: Arc::new(tensor_to_torch(uv.clone().unsqueeze_dim::<3>(0))),
        },
    );
    commands.insert_one(
        entity,
        NormalsPyTensor {
            tensor: Arc::new(tensor_to_torch(normals.clone().unsqueeze_dim::<3>(0))),
        },
    );
    commands.insert_one(
        entity,
        TangentsPyTensor {
            tensor: Arc::new(tensor_to_torch(tangents.clone().unwrap().unsqueeze_dim::<3>(0))),
        },
    );
    let faces: tch::Tensor = tensor_int_to_torch(faces.clone().unsqueeze_dim::<3>(0));
    let faces = faces.to_kind(tch::Kind::Int);
    commands.insert_one(entity, FacesPyTensor { tensor: Arc::new(faces) });
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
