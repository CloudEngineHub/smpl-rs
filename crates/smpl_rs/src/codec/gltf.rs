use super::scene::CameraTrack;
use crate::{
    common::{
        pose::Pose,
        types::{ChunkHeader, GltfCompatibilityMode, GltfOutputType, SmplType},
    },
    smpl_x::smpl_x,
};
use gloss_img::dynamic_image::DynImage;
use gloss_renderer::geom::Geom;
use gltf::binary::Header;
use gltf_json::validation::{Checked::Valid, USize64};
use gltf_json::{material::AlphaMode, scene::UnitQuaternion, Node};
use image::imageops::FilterType;
use image::RgbImage;
use itertools::izip;
use log::info;
use nalgebra as na;
use nalgebra::DMatrix;
use ndarray as nd;
use ndarray::prelude::*;
use smpl_utils::numerical::batch_rodrigues;
use smpl_utils::{
    log,
    vector::{
        addv3f, align_to_multiple_of_four, subv3f, to_padded_byte_vector, vec_from_array0_f, vec_from_vec, vec_to_vec, Vector2f, Vector3f, Vector4f,
        Vector4s,
    },
};
use std::borrow::Cow;
use std::{fs, mem};
use std::{
    io::{Cursor, Write},
    path::Path,
};
use utils_rs::nshare::ToNalgebra;
/// Enum for attribute ID's of a GLTF primitive (to avoid working with arbitrary
/// numbers)
#[repr(u32)]
#[derive(Debug, Clone, Copy)]
enum PrimitiveAttrIDs {
    Indices = 0,
    Positions = 1,
    Normals = 2,
    TexCoords = 3,
    Joints = 4,
    Weights = 5,
}
/// Enum for buffer view ID's (to avoid working with arbitrary numbers)
#[repr(u32)]
#[derive(Debug, Clone, Copy)]
enum BufferViewIDs {
    Index = 0,
    VertexAttr = 1,
    InvBindMat = 2,
    Keyframe = 3,
    Animation = 4,
    Deformation = 5,
}
/// Vertex definition for position and other Vertex attributes
#[derive(Copy, Clone, Debug, bytemuck::NoUninit)]
#[repr(C)]
struct Vertex {
    position: [f32; 3],
    normal: [f32; 3],
    uv: [f32; 2],
    joint_index: [u16; 4],
    joint_weight: [f32; 4],
}
/// Struct for holding GLTF texture information
#[derive(Clone, Debug)]
struct GltfTextureInfo {
    buffer_size: usize,
    image_data: Vec<u8>,
    image: gltf_json::image::Image,
    buffer_view: gltf_json::buffer::View,
    buffer_index: usize,
    texture: gltf_json::texture::Texture,
    sampler: gltf_json::texture::Sampler,
}
/// Texture indices
#[allow(clippy::struct_field_names)]
struct SmplTextures {
    diffuse_index: Option<usize>,
    normals_index: Option<usize>,
    metalic_roughtness_index: Option<usize>,
}
/// `PerBodyData` contains data for individual bodies
#[derive(Clone, Default, Debug)]
pub struct PerBodyData {
    pub diffuse_textures: Option<DynImage>,
    pub normals_textures: Option<DynImage>,
    pub metalness_textures: Option<DynImage>,
    pub roughness_textures: Option<DynImage>,
    pub positions: Option<DMatrix<f32>>,
    pub normals: Option<DMatrix<f32>>,
    pub default_joint_translations: Option<nd::Array2<f32>>,
    pub body_translation: Option<nd::Array2<f32>>,
    pub pose: Option<Pose>,
    pub body_translations: Option<nd::Array2<f32>>,
    pub body_rotations: Option<nd::Array3<f32>>,
    pub body_scales: Option<nd::Array2<f32>>,
    pub per_frame_blend_weights: Option<nd::Array2<f32>>,
}
/// The ``GltfCodec`` contains all the contents of the exported GLTF
#[derive(Debug, Clone)]
pub struct GltfCodec {
    pub num_bodies: usize,
    pub smpl_type: SmplType,
    pub gender: i32,
    pub faces: Option<DMatrix<u32>>,
    pub uvs: Option<DMatrix<f32>>,
    pub joint_index: Option<DMatrix<u32>>,
    pub joint_weight: Option<DMatrix<f32>>,
    pub default_joint_poses: Option<nd::Array2<f32>>,
    pub frame_count: Option<usize>,
    pub keyframe_times: Option<Vec<f32>>,
    pub morph_targets: Option<nd::Array3<f32>>,
    pub per_body_data: Vec<PerBodyData>,
    pub camera_track: Option<CameraTrack>,
}
impl Default for GltfCodec {
    fn default() -> Self {
        Self {
            num_bodies: 1,
            smpl_type: SmplType::SmplX,
            gender: 0,
            faces: None,
            uvs: None,
            joint_index: None,
            joint_weight: None,
            default_joint_poses: None,
            frame_count: None,
            keyframe_times: None,
            morph_targets: None,
            per_body_data: Vec::new(),
            camera_track: None,
        }
    }
}
impl GltfCodec {
    /// Export ``GltfCodec`` to a file (as a ``.gltf`` or ``.glb``)
    pub fn to_file(&mut self, name: &str, path: &str, out_type: GltfOutputType, compatibility_mode: GltfCompatibilityMode) {
        let parent_path = Path::new(path).parent();
        let file_name = Path::new(path).file_name();
        let Some(parent_path) = parent_path else {
            log!("Error: Exporting GLTF - no directory name found: {}", path);
            return;
        };
        let Some(file_name) = file_name else {
            log!("Error: Exporting GLTF - no file name found: {}", path);
            return;
        };
        let _ = fs::create_dir(parent_path);
        let target_extension: &str = match out_type {
            GltfOutputType::Standard => "gltf",
            GltfOutputType::Binary => "glb",
        };
        let file_name_with_suffix = Path::new(file_name).with_extension(target_extension);
        log!("Exporting GLTF: {}/{}", path, file_name_with_suffix.to_string_lossy());
        let binary = matches!(out_type, GltfOutputType::Binary);
        let (buffer_data, root) = self.create_buffer(name, binary, compatibility_mode);
        match out_type {
            GltfOutputType::Standard => {
                let json_path = parent_path.join(file_name_with_suffix.clone());
                let bin_path = parent_path.join("buffer0.bin");
                let writer = fs::File::create(json_path).expect("I/O error");
                gltf_json::serialize::to_writer_pretty(writer, &root).expect("Serialization error");
                let bin = to_padded_byte_vector(&buffer_data);
                let mut writer = fs::File::create(bin_path).expect("I/O error");
                writer.write_all(&bin).expect("I/O error");
                info!("Written glTF json + bin to {parent_path:?}");
            }
            GltfOutputType::Binary => {
                let json_string = gltf_json::serialize::to_string(&root).expect("Serialization error");
                let mut length = mem::size_of::<Header>() + mem::size_of::<ChunkHeader>() + json_string.len();
                align_to_multiple_of_four(&mut length);
                length += mem::size_of::<ChunkHeader>() + buffer_data.len();
                align_to_multiple_of_four(&mut length);
                let glb = gltf::binary::Glb {
                    header: gltf::binary::Header {
                        magic: *b"glTF",
                        version: 2,
                        length: length.try_into().expect("file size exceeds binary glTF limit"),
                    },
                    bin: Some(Cow::Owned(buffer_data)),
                    json: Cow::Owned(json_string.into_bytes()),
                };
                let glb_path = parent_path.join(file_name_with_suffix.clone());
                let writer = std::fs::File::create(glb_path.clone()).expect("I/O error");
                glb.to_writer(writer).expect("glTF binary output error");
                info!("Written binary glB to {glb_path:?}");
            }
        }
    }
    /// Get the ``GltfCodec`` as a u8 buffer
    pub fn to_buf(&mut self, compatibility_mode: GltfCompatibilityMode) -> Vec<u8> {
        let (buffer_data, root) = self.create_buffer("Meshcapade Avatar", true, compatibility_mode);
        let json_string = gltf_json::serialize::to_string(&root).expect("Serialization error");
        let mut length = mem::size_of::<Header>() + mem::size_of::<ChunkHeader>() + json_string.len();
        align_to_multiple_of_four(&mut length);
        length += mem::size_of::<ChunkHeader>() + buffer_data.len();
        align_to_multiple_of_four(&mut length);
        let glb = gltf::binary::Glb {
            header: gltf::binary::Header {
                magic: *b"glTF",
                version: 2,
                length: length.try_into().expect("file size exceeds binary glTF limit"),
            },
            bin: Some(Cow::Owned(buffer_data)),
            json: Cow::Owned(json_string.into_bytes()),
        };
        glb.to_vec().expect("glTF binary output error")
    }
    fn is_animated(&self) -> bool {
        self.frame_count.is_some()
    }
    /// Creates the buffer data for the GLTF
    #[allow(clippy::too_many_lines)]
    #[allow(clippy::cast_possible_truncation)]
    #[allow(clippy::cast_possible_wrap)]
    fn create_buffer(&mut self, name: &str, binary: bool, compatibility_mode: GltfCompatibilityMode) -> (Vec<u8>, gltf_json::Root) {
        assert!(self.faces.is_some(), "GltfCodec: no faces!");
        assert!(self.uvs.is_some(), "GltfCodec: no uvs!");
        let mut full_buffer_data = vec![];
        let mut accessors = vec![];
        let mut buffers = vec![];
        let mut buffer_views = vec![];
        let mut meshes = vec![];
        let mut nodes = vec![];
        let mut skins = vec![];
        let mut materials = vec![];
        let mut channels = vec![];
        let mut samplers = vec![];
        let mut images = vec![];
        let mut textures = vec![];
        let mut texture_samplers: Vec<gltf_json::texture::Sampler> = vec![];
        let mut cameras: Vec<gltf_json::camera::Camera> = vec![];
        let scene_root_node_index = u32::try_from(nodes.len()).expect("Issue converting Node idx to u32");
        let scene_root_node = Node {
            name: Some("SceneRoot".to_string()),
            children: Some(vec![]),
            ..Default::default()
        };
        nodes.push(scene_root_node);
        if let Some(camera_track) = &self.camera_track {
            let camera = gltf_json::camera::Camera {
                name: Some("MoCapadeCamera".to_string()),
                type_: gltf_json::validation::Checked::Valid(gltf_json::camera::Type::Perspective),
                perspective: Some(gltf_json::camera::Perspective {
                    yfov: camera_track.yfov,
                    znear: camera_track.znear,
                    zfar: camera_track.zfar,
                    aspect_ratio: camera_track.aspect_ratio,
                    extensions: None,
                    extras: Option::default(),
                }),
                orthographic: None,
                extensions: None,
                extras: Option::default(),
            };
            cameras.push(camera);
            let camera_track_node = Node {
                name: Some("AnimatedCamera".to_string()),
                camera: Some(gltf_json::Index::new(0)),
                ..Default::default()
            };
            let camera_node_idx = u32::try_from(nodes.len()).expect("Issue converting Node idx to u32");
            if let Some(ref mut scene_root_node_children) = nodes[0].children {
                scene_root_node_children.push(gltf_json::Index::new(camera_node_idx));
            }
            nodes.push(camera_track_node);
        }
        let node_indices: Vec<gltf_json::Index<Node>> = vec![gltf_json::Index::new(scene_root_node_index)];
        let scene: gltf_json::Scene = gltf_json::Scene {
            extensions: Option::default(),
            extras: Option::default(),
            name: Some(name.to_string()),
            nodes: node_indices,
        };
        let scenes = vec![scene];
        for (body_idx, current_body) in self.per_body_data.clone().iter_mut().enumerate() {
            assert!(current_body.positions.is_some(), "GltfCodec: no vertices for body {body_idx}!");
            assert!(current_body.normals.is_some(), "GltfCodec: no normals for body {body_idx}!");
            let mut positions = current_body.positions.clone().unwrap();
            let normals = current_body.normals.as_ref().unwrap();
            let faces = self.faces.as_ref().unwrap();
            let uvs = self.uvs.as_ref().unwrap();
            let joint_index = self.joint_index.as_ref().unwrap();
            let joint_weight = self.joint_weight.as_ref().unwrap();
            let diffuse_tex = current_body.diffuse_textures.as_ref();
            let normals_tex = current_body.normals_textures.as_ref();
            let metalness_tex = current_body.metalness_textures.as_ref();
            let roughness_tex = current_body.roughness_textures.as_ref();
            let mut vertex_attributes_array: Vec<Vertex> = vec![];
            let mut indices_array: Vec<u32> = vec![];
            let mut inverse_bind_matrices: Vec<f32> = vec![];
            let face_count = faces.shape().0;
            let vertex_count = positions.shape().0;
            let joint_count = current_body.default_joint_translations.as_ref().unwrap().shape()[0];
            for row in faces.row_iter() {
                indices_array.extend_from_slice(&[row[0], row[1], row[2]]);
            }
            let joint_rotations = batch_rodrigues(self.default_joint_poses.as_ref().unwrap());
            let mut joint_translations = current_body.default_joint_translations.clone().unwrap();
            if compatibility_mode == GltfCompatibilityMode::Unreal {
                let (min, _) = Geom::get_bounding_points(&positions, None);
                let min_vec: Vec<f32> = min.iter().copied().collect();
                let min_y = min_vec[1];
                let offset = na::RowVector3::new(0.0, min_y, 0.0);
                for i in 0..positions.nrows() {
                    let mut row = positions.row_mut(i);
                    row -= offset;
                }
                let offset_nd = ndarray::Array1::from_vec(vec![0.0, min_y, 0.0]);
                for mut row in joint_translations.axis_iter_mut(Axis(0)) {
                    row -= &offset_nd;
                }
                current_body.positions = Some(positions.clone());
                current_body.default_joint_translations = Some(joint_translations.clone());
            }
            let metadata = crate::common::metadata::smpl_metadata(&self.smpl_type);
            let bind_matrices = self.create_bind_matrices(&joint_rotations, &joint_translations, &metadata.joint_parents);
            let unreal_mapping: [usize; 10] = [0, 0, 0, 8, 7, 0, 21, 21, 20, 0];
            if compatibility_mode == GltfCompatibilityMode::Unreal {
                for j_idx in unreal_mapping {
                    let mut inverse_bind_matrix = nd::Array2::<f32>::zeros((4, 4));
                    let inverse_rotation_matrix = bind_matrices.slice(s![j_idx, 0..3, 0..3]).reversed_axes();
                    let translation: nd::Array1<f32> = if j_idx == 0 {
                        nd::Array1::from_vec(vec![0.0, 0.0, 0.0])
                    } else {
                        bind_matrices.slice(s![j_idx, 0..3, 3]).to_owned()
                    };
                    let inverse_translation = -inverse_rotation_matrix.dot(&translation);
                    inverse_bind_matrix.slice_mut(s![0..3, 0..3]).assign(&inverse_rotation_matrix);
                    inverse_bind_matrix.slice_mut(s![0..3, 3]).assign(&inverse_translation);
                    inverse_bind_matrix[(3, 3)] = 1.0;
                    inverse_bind_matrices.extend(inverse_bind_matrix.t().iter());
                }
            }
            for j_idx in 0..joint_count {
                let mut inverse_bind_matrix = nd::Array2::<f32>::zeros((4, 4));
                let inverse_rotation_matrix = bind_matrices.slice(s![j_idx, 0..3, 0..3]).reversed_axes();
                let translation: nd::Array1<f32> = bind_matrices.slice(s![j_idx, 0..3, 3]).to_owned();
                let inverse_translation = -inverse_rotation_matrix.dot(&translation);
                inverse_bind_matrix.slice_mut(s![0..3, 0..3]).assign(&inverse_rotation_matrix);
                inverse_bind_matrix.slice_mut(s![0..3, 3]).assign(&inverse_translation);
                inverse_bind_matrix[(3, 3)] = 1.0;
                inverse_bind_matrices.extend(inverse_bind_matrix.t().iter());
            }
            let num_extra_joints: usize = if compatibility_mode == GltfCompatibilityMode::Unreal { 10 } else { 0 };
            for (position, normal, uv, joint_index, joint_weight) in izip!(
                positions.row_iter(),
                normals.row_iter(),
                uvs.row_iter(),
                joint_index.row_iter(),
                joint_weight.row_iter(),
            ) {
                let jw_sum = joint_weight.iter().sum::<f32>();
                vertex_attributes_array.push(Vertex {
                    position: Vector3f::new(position[0], position[1], position[2]).into(),
                    normal: Vector3f::new(normal[0], normal[1], normal[2]).into(),
                    uv: Vector2f::new(uv[0], 1.0 - uv[1]).into(),
                    joint_index: Vector4s::new(
                        u16::try_from(joint_index[0] + u32::try_from(num_extra_joints).unwrap()).expect("Could not convert to u16!"),
                        u16::try_from(joint_index[1] + u32::try_from(num_extra_joints).unwrap()).expect("Could not convert to u16!"),
                        u16::try_from(joint_index[2] + u32::try_from(num_extra_joints).unwrap()).expect("Could not convert to u16!"),
                        u16::try_from(joint_index[3] + u32::try_from(num_extra_joints).unwrap()).expect("Could not convert to u16!"),
                    )
                    .into(),
                    joint_weight: Vector4f::new(
                        joint_weight[0] / jw_sum,
                        joint_weight[1] / jw_sum,
                        joint_weight[2] / jw_sum,
                        joint_weight[3] / jw_sum,
                    )
                    .into(),
                });
            }
            let mut texture_infos: Vec<GltfTextureInfo> = vec![];
            let mut smpl_textures = SmplTextures {
                diffuse_index: None,
                normals_index: None,
                metalic_roughtness_index: None,
            };
            if let Some(img) = diffuse_tex {
                let diffuse_tex = self.add_texture(img, texture_infos.len(), "diffuse");
                if let Some(diffuse_tex) = diffuse_tex {
                    smpl_textures.diffuse_index = Some(texture_infos.len());
                    texture_infos.push(diffuse_tex);
                }
            }
            self.prepare_normals(&mut smpl_textures, &mut texture_infos, normals_tex);
            self.prepare_metallic_roughness(&mut smpl_textures, &mut texture_infos, metalness_tex, roughness_tex);
            let mut base_color_texture: Option<gltf_json::texture::Info> = None;
            let mut normal_texture: Option<gltf_json::material::NormalTexture> = None;
            let mut metallic_roughness_texture: Option<gltf_json::texture::Info> = None;
            if let Some(diffuse_texture_index) = smpl_textures.diffuse_index {
                base_color_texture = Some(gltf_json::texture::Info {
                    index: gltf_json::Index::new(u32::try_from(diffuse_texture_index).expect("Could not convert to u32!")),
                    tex_coord: 0,
                    extensions: None,
                    extras: None,
                });
            }
            if let Some(normal_texture_index) = smpl_textures.normals_index {
                normal_texture = Some(gltf_json::material::NormalTexture {
                    scale: 1.,
                    index: gltf_json::Index::new(u32::try_from(normal_texture_index).expect("Could not convert to u32!")),
                    tex_coord: 0,
                    extensions: None,
                    extras: None,
                });
            }
            if let Some(metallic_roughness_texture_index) = smpl_textures.metalic_roughtness_index {
                metallic_roughness_texture = Some(gltf_json::texture::Info {
                    index: gltf_json::Index::new(u32::try_from(metallic_roughness_texture_index).expect("Could not convert to u32!")),
                    tex_coord: 0,
                    extensions: None,
                    extras: None,
                });
            }
            let material = gltf_json::Material {
                alpha_cutoff: None,
                alpha_mode: gltf_json::validation::Checked::<AlphaMode>::Valid(AlphaMode::Opaque),
                double_sided: false,
                name: Some("SMPL_material".to_string()),
                pbr_metallic_roughness: gltf_json::material::PbrMetallicRoughness {
                    base_color_factor: gltf_json::material::PbrBaseColorFactor([1., 1., 1., 1.]),
                    base_color_texture,
                    metallic_roughness_texture,
                    ..Default::default()
                },
                normal_texture,
                occlusion_texture: None,
                emissive_texture: None,
                emissive_factor: gltf_json::material::EmissiveFactor([0., 0., 0.]),
                extensions: None,
                extras: None,
            };
            materials.push(material);
            let mut morph_targets: Option<Vec<gltf_json::mesh::MorphTarget>> = None;
            if self.num_morph_targets() > 0 && self.is_animated() {
                let mut morph_target_accessors_start_idx = 7 + self.per_body_data[0].default_joint_translations.as_ref().unwrap().shape()[0] + 4;
                if compatibility_mode == GltfCompatibilityMode::Unreal {
                    morph_target_accessors_start_idx += 1;
                }
                morph_targets = Some(self.create_morph_targets(morph_target_accessors_start_idx));
            }
            let primitive_offset = accessors.len() as u32;
            let primitive = gltf_json::mesh::Primitive {
                attributes: {
                    let mut map = std::collections::BTreeMap::new();
                    map.insert(
                        Valid(gltf_json::mesh::Semantic::Positions),
                        gltf_json::Index::new(PrimitiveAttrIDs::Positions as u32 + primitive_offset),
                    );
                    map.insert(
                        Valid(gltf_json::mesh::Semantic::Normals),
                        gltf_json::Index::new(PrimitiveAttrIDs::Normals as u32 + primitive_offset),
                    );
                    map.insert(
                        Valid(gltf_json::mesh::Semantic::TexCoords(0)),
                        gltf_json::Index::new(PrimitiveAttrIDs::TexCoords as u32 + primitive_offset),
                    );
                    map.insert(
                        Valid(gltf_json::mesh::Semantic::Joints(0)),
                        gltf_json::Index::new(PrimitiveAttrIDs::Joints as u32 + primitive_offset),
                    );
                    map.insert(
                        Valid(gltf_json::mesh::Semantic::Weights(0)),
                        gltf_json::Index::new(PrimitiveAttrIDs::Weights as u32 + primitive_offset),
                    );
                    map
                },
                extensions: Option::default(),
                extras: Option::default(),
                indices: Some(gltf_json::Index::new(PrimitiveAttrIDs::Indices as u32 + primitive_offset)),
                material: Some(gltf_json::Index::new(body_idx as u32)),
                mode: Valid(gltf_json::mesh::Mode::Triangles),
                targets: morph_targets,
            };
            let mut morph_target_weights: Option<Vec<f32>> = None;
            if self.num_morph_targets() > 0 && self.is_animated() {
                morph_target_weights = Some(vec![0.0; self.num_morph_targets()]);
            }
            let mesh = gltf_json::Mesh {
                extensions: Option::default(),
                extras: Option::default(),
                name: Some(format!("SMPL_mesh_{body_idx}")),
                primitives: vec![primitive],
                weights: morph_target_weights,
            };
            meshes.push(mesh);
            let vertex_data = to_padded_byte_vector(&vertex_attributes_array);
            let index_data = to_padded_byte_vector(&indices_array);
            let inv_bind_mat_data = to_padded_byte_vector(&inverse_bind_matrices);
            let mut per_view_running_offset: [usize; 6] = [0, 0, 0, 0, 0, 0];
            let current_buffer_view_offset = buffer_views.len() as u32;
            let current_accessor_offset = accessors.len() as u32;
            let accessor = self.create_accessors(
                body_idx,
                vertex_count,
                face_count,
                joint_count,
                current_buffer_view_offset,
                &mut per_view_running_offset,
                num_extra_joints,
                compatibility_mode,
            );
            accessors.extend(accessor);
            let mut current_buffer_views = vec![];
            self.create_buffer_views(
                body_idx as u32,
                full_buffer_data.len(),
                vertex_count,
                face_count,
                joint_count,
                num_extra_joints,
                &mut current_buffer_views,
                compatibility_mode,
            );
            let (buffer_data, composed_buffer_views) = self.compose_buffer_views(
                body_idx,
                current_buffer_views.clone(),
                index_data.as_slice(),
                vertex_data.as_slice(),
                inv_bind_mat_data.as_slice(),
                &mut texture_infos,
                compatibility_mode,
            );
            full_buffer_data.extend(buffer_data);
            buffer_views.extend(composed_buffer_views);
            for texture in texture_infos {
                images.push(texture.image);
                textures.push(texture.texture);
                texture_samplers.push(texture.sampler);
            }
            let armature_node_index = u32::try_from(nodes.len()).expect("Issue converting Node idx to u32");
            let armature_node = Node {
                name: Some(format!("Armature_{body_idx}")),
                children: Some(vec![]),
                ..Default::default()
            };
            nodes.push(armature_node);
            if let Some(ref mut scene_armatures) = nodes[0].children {
                scene_armatures.push(gltf_json::Index::new(armature_node_index));
            }
            let mesh_skin_binding_node_index = u32::try_from(nodes.len()).expect("Issue converting Node idx to u32");
            let mesh_skin_binding_node = Node {
                mesh: Some(gltf_json::Index::new(body_idx as u32)),
                skin: Some(gltf_json::Index::new(body_idx as u32)),
                name: Some(format!("MeshSkinBinding_{body_idx}")),
                children: None,
                ..Default::default()
            };
            nodes.push(mesh_skin_binding_node);
            let root_node_index = u32::try_from(nodes.len()).expect("Issue converting Node idx to u32");
            let mut joints = vec![];
            if compatibility_mode == GltfCompatibilityMode::Unreal {
                let root_node = Node {
                    name: Some("root".to_string()),
                    translation: Some([0.0, 0.0, 0.0]),
                    children: Some(vec![]),
                    ..Default::default()
                };
                let joint_index = gltf_json::Index::<Node>::new(u32::try_from(nodes.len()).expect("Issue converting Joint idx to u32"));
                nodes.push(root_node);
                joints.push(joint_index);
                let add_empty_node = |nodes: &mut Vec<Node>,
                                      joints: &mut Vec<gltf_json::Index<Node>>,
                                      name: &str,
                                      parent_index: u32,
                                      has_children: bool,
                                      reference_bone: usize|
                 -> u32 {
                    let relative_parent_idx = parent_index - root_node_index;
                    let node_index = u32::try_from(nodes.len()).expect("Issue converting Node idx to u32");
                    let unit_quaternion = [0.0, 0.0, 0.0, 1.0];
                    let trans: Vec<f32> = if reference_bone == 0 {
                        vec![0.0, 0.0, 0.0]
                    } else {
                        let trans = vec_from_vec(&joint_translations.row(reference_bone).to_vec());
                        let parent_trans = if unreal_mapping[relative_parent_idx as usize] != 0 {
                            vec_from_vec(&joint_translations.row(unreal_mapping[relative_parent_idx as usize]).to_vec())
                        } else {
                            vec_from_vec(&[0.0, 0.0, 0.0])
                        };
                        vec_to_vec(&subv3f(&trans, &parent_trans))
                    };
                    let translation = [trans[0], trans[1], trans[2]];
                    let new_node = Node {
                        name: Some(name.to_string()),
                        rotation: Some(UnitQuaternion(unit_quaternion)),
                        translation: Some(translation),
                        children: if has_children { Some(vec![]) } else { None },
                        ..Default::default()
                    };
                    if let Some(ref mut parent_children) = nodes[parent_index as usize].children {
                        parent_children.push(gltf_json::Index::new(node_index));
                    }
                    let joint_index = gltf_json::Index::<Node>::new(node_index);
                    nodes.push(new_node);
                    joints.push(joint_index);
                    node_index
                };
                add_empty_node(&mut nodes, &mut joints, "center_of_mass", root_node_index, false, 0);
                let ik_foot_root_index = add_empty_node(&mut nodes, &mut joints, "ik_foot_root", root_node_index, true, 0);
                add_empty_node(&mut nodes, &mut joints, "ik_foot_r", ik_foot_root_index, false, 8);
                add_empty_node(&mut nodes, &mut joints, "ik_foot_l", ik_foot_root_index, false, 7);
                let ik_hand_root_index = add_empty_node(&mut nodes, &mut joints, "ik_hand_root", root_node_index, true, 0);
                let ik_hand_gun_index = add_empty_node(&mut nodes, &mut joints, "ik_hand_gun", ik_hand_root_index, true, 21);
                add_empty_node(&mut nodes, &mut joints, "ik_hand_r", ik_hand_gun_index, false, 21);
                add_empty_node(&mut nodes, &mut joints, "ik_hand_l", ik_hand_gun_index, false, 20);
                add_empty_node(&mut nodes, &mut joints, "interaction", root_node_index, false, 0);
            }
            let skeleton_root_index = self.add_skin(
                format!("Skin_{body_idx}"),
                body_idx,
                armature_node_index,
                current_accessor_offset,
                &mut nodes,
                &mut skins,
                &mut joints,
                compatibility_mode,
            );
            if compatibility_mode == GltfCompatibilityMode::Unreal {
                if let Some(ref mut root) = nodes[root_node_index as usize].children {
                    root.push(skeleton_root_index);
                }
            }
            if let Some(ref mut armature_children) = nodes[armature_node_index as usize].children {
                armature_children.push(gltf_json::Index::new(mesh_skin_binding_node_index));
                armature_children.push(gltf_json::Index::new(root_node_index));
            }
            if self.is_animated() {
                let animation_channels = self.create_animation_channels(
                    joint_count,
                    root_node_index,
                    skeleton_root_index.value(),
                    samplers.len(),
                    compatibility_mode,
                );
                let animation_samplers = self.create_animation_samplers(joint_count, current_accessor_offset, compatibility_mode);
                channels.extend(animation_channels);
                samplers.extend(animation_samplers);
            }
        }
        if self.camera_track.is_some() {
            let (cam_track_buffer_views, cam_track_buffer_data) = self.create_camera_animation_buffer_views(&mut full_buffer_data.len()).unwrap();
            let cam_track_accessors = self.create_camera_animation_accessors(buffer_views.len() as u32).unwrap();
            let (cam_track_channels, cam_track_samplers) = self
                .create_camera_animation_channels_and_samplers(accessors.len() as u32, 1, samplers.len() as u32)
                .unwrap();
            buffer_views.extend(cam_track_buffer_views);
            full_buffer_data.extend(cam_track_buffer_data);
            accessors.extend(cam_track_accessors);
            channels.extend(cam_track_channels);
            samplers.extend(cam_track_samplers);
        }
        let buffer = gltf_json::Buffer {
            byte_length: USize64::from(full_buffer_data.len()),
            extensions: Option::default(),
            extras: Option::default(),
            name: Some("scene_buffer".to_string()),
            uri: if binary { None } else { Some("buffer0.bin".into()) },
        };
        buffers.push(buffer);
        let mut animations: Vec<gltf_json::Animation> = vec![];
        if self.is_animated() {
            let animation = gltf_json::Animation {
                extensions: Option::default(),
                extras: Option::default(),
                channels,
                name: Some("Scene_animation".to_string()),
                samplers,
            };
            animations.push(animation);
        }
        let root = gltf_json::Root {
            accessors,
            animations,
            buffers,
            buffer_views,
            cameras,
            images,
            materials,
            meshes,
            nodes,
            samplers: texture_samplers,
            scenes,
            skins,
            textures,
            ..Default::default()
        };
        (full_buffer_data, root)
    }
    /// Function for creating buffer view definitions
    #[allow(clippy::too_many_arguments)]
    fn create_buffer_views(
        &self,
        body_idx: u32,
        mut running_offset: usize,
        vertex_count: usize,
        face_count: usize,
        joint_count: usize,
        num_extra_joints: usize,
        buffer_views: &mut Vec<gltf_json::buffer::View>,
        compatibility_mode: GltfCompatibilityMode,
    ) {
        let index_buffer_size = face_count * 3 * mem::size_of::<u32>();
        let index_buffer_view = gltf_json::buffer::View {
            buffer: gltf_json::Index::new(0),
            byte_length: USize64::from(index_buffer_size),
            byte_offset: Some(USize64::from(running_offset)),
            byte_stride: None,
            extensions: Option::default(),
            extras: Option::default(),
            name: Some(format!("index_buffer_view_{body_idx}")),
            target: Some(Valid(gltf_json::buffer::Target::ElementArrayBuffer)),
        };
        buffer_views.push(index_buffer_view);
        running_offset += index_buffer_size;
        let vertex_buffer_size = vertex_count * mem::size_of::<Vertex>();
        let vertex_buffer_view = gltf_json::buffer::View {
            buffer: gltf_json::Index::new(0),
            byte_length: USize64::from(vertex_buffer_size),
            byte_offset: Some(USize64::from(running_offset)),
            byte_stride: Some(gltf_json::buffer::Stride(mem::size_of::<Vertex>())),
            extensions: Option::default(),
            extras: Option::default(),
            name: Some(format!("vertex_buffer_view_{body_idx}")),
            target: Some(Valid(gltf_json::buffer::Target::ArrayBuffer)),
        };
        buffer_views.push(vertex_buffer_view);
        running_offset += vertex_buffer_size;
        let inv_bind_matrix_buffer_size = (joint_count + num_extra_joints) * 16 * mem::size_of::<f32>();
        let inverse_bind_mat_buffer_view = gltf_json::buffer::View {
            buffer: gltf_json::Index::new(0),
            byte_length: USize64::from(inv_bind_matrix_buffer_size),
            byte_offset: Some(USize64::from(running_offset)),
            byte_stride: None,
            extensions: Option::default(),
            extras: Option::default(),
            name: Some(format!("inv_bind_matrix_buffer_view_{body_idx}")),
            target: None,
        };
        buffer_views.push(inverse_bind_mat_buffer_view);
        running_offset += inv_bind_matrix_buffer_size;
        if self.is_animated() {
            let rotation_animation_buffer_size = self.frame_count.unwrap() * 4 * mem::size_of::<f32>();
            let translation_animation_buffer_size = self.frame_count.unwrap() * 3 * mem::size_of::<f32>();
            let scale_animation_buffer_size = self.frame_count.unwrap() * 3 * mem::size_of::<f32>();
            let animation_buffer_views = self.create_animation_buffer_views(
                body_idx,
                joint_count,
                rotation_animation_buffer_size,
                translation_animation_buffer_size,
                scale_animation_buffer_size,
                &mut running_offset,
                compatibility_mode,
            );
            buffer_views.extend(animation_buffer_views);
            if self.num_morph_targets() > 0 && body_idx == 0 {
                let morph_target_buffer_size = vertex_count * 3 * mem::size_of::<f32>();
                let morph_target_buffer_views = self.create_morph_target_buffer_views(morph_target_buffer_size, &mut running_offset);
                buffer_views.extend(morph_target_buffer_views);
            }
        }
    }
    /// Function for creating animation based buffer views
    #[allow(clippy::too_many_arguments)]
    fn create_animation_buffer_views(
        &self,
        body_idx: u32,
        joint_count: usize,
        rotation_buffer_size: usize,
        translation_buffer_size: usize,
        scale_buffer_size: usize,
        running_offset: &mut usize,
        compatibility_mode: GltfCompatibilityMode,
    ) -> Vec<gltf_json::buffer::View> {
        let mut animation_buffer_views: Vec<gltf_json::buffer::View> = vec![];
        let keyframe_buffer_size = self.frame_count.unwrap() * mem::size_of::<f32>();
        let keyframe_buffer_view = gltf_json::buffer::View {
            buffer: gltf_json::Index::new(0),
            byte_length: USize64::from(keyframe_buffer_size),
            byte_offset: Some(USize64::from(*running_offset)),
            byte_stride: None,
            extensions: Option::default(),
            extras: Option::default(),
            name: Some(format!("keyframe_buffer_view_{body_idx}")),
            target: None,
        };
        animation_buffer_views.push(keyframe_buffer_view);
        *running_offset += keyframe_buffer_size;
        for j_idx in 0..joint_count {
            let buffer_view_name = format!("joint_{j_idx}_animations_buffer_view_{body_idx}");
            let animation_buffer_view = gltf_json::buffer::View {
                buffer: gltf_json::Index::new(0),
                byte_length: USize64::from(rotation_buffer_size),
                byte_offset: Some(USize64::from(*running_offset)),
                byte_stride: None,
                extensions: Option::default(),
                extras: Option::default(),
                name: Some(buffer_view_name),
                target: None,
            };
            animation_buffer_views.push(animation_buffer_view);
            *running_offset += rotation_buffer_size;
        }
        let buffer_view_name = format!("root_translation_animations_buffer_view_{body_idx}");
        let animation_buffer_view = gltf_json::buffer::View {
            buffer: gltf_json::Index::new(0),
            byte_length: USize64::from(translation_buffer_size),
            byte_offset: Some(USize64::from(*running_offset)),
            byte_stride: None,
            extensions: Option::default(),
            extras: Option::default(),
            name: Some(buffer_view_name),
            target: None,
        };
        animation_buffer_views.push(animation_buffer_view);
        *running_offset += translation_buffer_size;
        let buffer_view_name = format!("root_scale_animations_buffer_view_{body_idx}");
        let animation_buffer_view = gltf_json::buffer::View {
            buffer: gltf_json::Index::new(0),
            byte_length: USize64::from(scale_buffer_size),
            byte_offset: Some(USize64::from(*running_offset)),
            byte_stride: None,
            extensions: Option::default(),
            extras: Option::default(),
            name: Some(buffer_view_name),
            target: None,
        };
        animation_buffer_views.push(animation_buffer_view);
        *running_offset += scale_buffer_size;
        if compatibility_mode == GltfCompatibilityMode::Unreal {
            let buffer_view_name = format!("pelvis_rel_translation_animations_buffer_view_{body_idx}");
            let animation_buffer_view = gltf_json::buffer::View {
                buffer: gltf_json::Index::new(0),
                byte_length: USize64::from(translation_buffer_size),
                byte_offset: Some(USize64::from(*running_offset)),
                byte_stride: None,
                extensions: Option::default(),
                extras: Option::default(),
                name: Some(buffer_view_name),
                target: None,
            };
            animation_buffer_views.push(animation_buffer_view);
            *running_offset += translation_buffer_size;
        }
        if self.num_morph_targets() > 0 {
            let morph_weights_buffer_size = self.frame_count.unwrap() * self.num_morph_targets() * mem::size_of::<f32>();
            let buffer_view_name = format!("morph_target_weights_{body_idx}");
            let morph_weights_buffer_view = gltf_json::buffer::View {
                buffer: gltf_json::Index::new(0),
                byte_length: USize64::from(morph_weights_buffer_size),
                byte_offset: Some(USize64::from(*running_offset)),
                byte_stride: None,
                extensions: Option::default(),
                extras: Option::default(),
                name: Some(buffer_view_name),
                target: None,
            };
            animation_buffer_views.push(morph_weights_buffer_view);
            *running_offset += morph_weights_buffer_size;
        }
        animation_buffer_views
    }
    /// Function for creating buffer views for morph targets
    fn create_morph_target_buffer_views(&self, morph_target_buffer_size: usize, running_offset: &mut usize) -> Vec<gltf_json::buffer::View> {
        let mut morph_targets_buffer_views: Vec<gltf_json::buffer::View> = vec![];
        for morph_target_idx in 0..self.num_morph_targets() {
            let buffer_view_name = format!("morph_{morph_target_idx}_buffer_view");
            let morph_target_buffer_view = gltf_json::buffer::View {
                buffer: gltf_json::Index::new(0),
                byte_length: USize64::from(morph_target_buffer_size),
                byte_offset: Some(USize64::from(*running_offset)),
                byte_stride: None,
                extensions: Option::default(),
                extras: Option::default(),
                name: Some(buffer_view_name),
                target: Some(Valid(gltf_json::buffer::Target::ArrayBuffer)),
            };
            morph_targets_buffer_views.push(morph_target_buffer_view);
            *running_offset += morph_target_buffer_size;
        }
        morph_targets_buffer_views
    }
    /// Function fo creating all the GLTF accessors
    #[allow(clippy::too_many_lines)]
    #[allow(clippy::too_many_arguments)]
    fn create_accessors(
        &self,
        body_idx: usize,
        vertex_count: usize,
        face_count: usize,
        joint_count: usize,
        current_buffer_view_offset: u32,
        per_view_running_offset: &mut [usize; 6],
        num_extra_joints: usize,
        compatibility_mode: GltfCompatibilityMode,
    ) -> Vec<gltf_json::Accessor> {
        let (min, max) = Geom::get_bounding_points(self.per_body_data[body_idx].positions.as_ref().unwrap(), None);
        let (min_vec, max_vec): (Vec<f32>, Vec<f32>) = (min.iter().copied().collect(), max.iter().copied().collect());
        let mut accessors: Vec<gltf_json::Accessor> = vec![];
        let indices = gltf_json::Accessor {
            buffer_view: Some(gltf_json::Index::new(BufferViewIDs::Index as u32 + current_buffer_view_offset)),
            byte_offset: Some(USize64::from(per_view_running_offset[BufferViewIDs::Index as usize])),
            count: USize64::from(face_count * 3),
            component_type: Valid(gltf_json::accessor::GenericComponentType(gltf_json::accessor::ComponentType::U32)),
            extensions: Option::default(),
            extras: Option::default(),
            type_: Valid(gltf_json::accessor::Type::Scalar),
            min: Some(gltf_json::Value::from(Vec::from([self.faces.as_ref().unwrap().min()]))),
            max: Some(gltf_json::Value::from(Vec::from([self.faces.as_ref().unwrap().max()]))),
            name: Some(format!("index_accessor_{body_idx}")),
            normalized: false,
            sparse: None,
        };
        accessors.push(indices);
        let position_element_size = 3 * mem::size_of::<f32>();
        let positions = gltf_json::Accessor {
            buffer_view: Some(gltf_json::Index::new(BufferViewIDs::VertexAttr as u32 + current_buffer_view_offset)),
            byte_offset: Some(USize64::from(per_view_running_offset[BufferViewIDs::VertexAttr as usize])),
            count: USize64::from(vertex_count),
            component_type: Valid(gltf_json::accessor::GenericComponentType(gltf_json::accessor::ComponentType::F32)),
            extensions: Option::default(),
            extras: Option::default(),
            type_: Valid(gltf_json::accessor::Type::Vec3),
            min: Some(gltf_json::Value::from(min_vec)),
            max: Some(gltf_json::Value::from(max_vec)),
            name: Some(format!("position_accessor_{body_idx}")),
            normalized: false,
            sparse: None,
        };
        per_view_running_offset[BufferViewIDs::VertexAttr as usize] += position_element_size;
        accessors.push(positions);
        let normal_element_size = 3 * mem::size_of::<f32>();
        let normals = gltf_json::Accessor {
            buffer_view: Some(gltf_json::Index::new(BufferViewIDs::VertexAttr as u32 + current_buffer_view_offset)),
            byte_offset: Some(USize64::from(per_view_running_offset[BufferViewIDs::VertexAttr as usize])),
            count: USize64::from(vertex_count),
            component_type: Valid(gltf_json::accessor::GenericComponentType(gltf_json::accessor::ComponentType::F32)),
            extensions: Option::default(),
            extras: Option::default(),
            type_: Valid(gltf_json::accessor::Type::Vec3),
            min: None,
            max: None,
            name: Some(format!("normal_accessor_{body_idx}")),
            normalized: false,
            sparse: None,
        };
        per_view_running_offset[BufferViewIDs::VertexAttr as usize] += normal_element_size;
        accessors.push(normals);
        let uv_element_size = 2 * mem::size_of::<f32>();
        let uvs = gltf_json::Accessor {
            buffer_view: Some(gltf_json::Index::new(BufferViewIDs::VertexAttr as u32 + current_buffer_view_offset)),
            byte_offset: Some(USize64::from(per_view_running_offset[BufferViewIDs::VertexAttr as usize])),
            count: USize64::from(vertex_count),
            component_type: Valid(gltf_json::accessor::GenericComponentType(gltf_json::accessor::ComponentType::F32)),
            extensions: Option::default(),
            extras: Option::default(),
            type_: Valid(gltf_json::accessor::Type::Vec2),
            min: None,
            max: None,
            name: Some("uv_accessor".to_string()),
            normalized: false,
            sparse: None,
        };
        per_view_running_offset[BufferViewIDs::VertexAttr as usize] += uv_element_size;
        accessors.push(uvs);
        let joint_index_element_size = 4 * mem::size_of::<u16>();
        let joint_indices = gltf_json::Accessor {
            buffer_view: Some(gltf_json::Index::new(BufferViewIDs::VertexAttr as u32 + current_buffer_view_offset)),
            byte_offset: Some(USize64::from(per_view_running_offset[BufferViewIDs::VertexAttr as usize])),
            count: USize64::from(vertex_count),
            component_type: Valid(gltf_json::accessor::GenericComponentType(gltf_json::accessor::ComponentType::U16)),
            extensions: Option::default(),
            extras: Option::default(),
            type_: Valid(gltf_json::accessor::Type::Vec4),
            min: None,
            max: None,
            name: Some("joint_index_accessor".to_string()),
            normalized: false,
            sparse: None,
        };
        per_view_running_offset[BufferViewIDs::VertexAttr as usize] += joint_index_element_size;
        accessors.push(joint_indices);
        let joint_weight_element_size = 4 * mem::size_of::<f32>();
        let joint_weights = gltf_json::Accessor {
            buffer_view: Some(gltf_json::Index::new(BufferViewIDs::VertexAttr as u32 + current_buffer_view_offset)),
            byte_offset: Some(USize64::from(per_view_running_offset[BufferViewIDs::VertexAttr as usize])),
            count: USize64::from(vertex_count),
            component_type: Valid(gltf_json::accessor::GenericComponentType(gltf_json::accessor::ComponentType::F32)),
            extensions: Option::default(),
            extras: Option::default(),
            type_: Valid(gltf_json::accessor::Type::Vec4),
            min: None,
            max: None,
            name: Some("joint_index_accessor".to_string()),
            normalized: false,
            sparse: None,
        };
        per_view_running_offset[BufferViewIDs::VertexAttr as usize] += joint_weight_element_size;
        accessors.push(joint_weights);
        let inv_bind_matrices = gltf_json::Accessor {
            buffer_view: Some(gltf_json::Index::new(BufferViewIDs::InvBindMat as u32 + current_buffer_view_offset)),
            byte_offset: Some(USize64::from(per_view_running_offset[BufferViewIDs::InvBindMat as usize])),
            count: USize64::from(joint_count + num_extra_joints),
            component_type: Valid(gltf_json::accessor::GenericComponentType(gltf_json::accessor::ComponentType::F32)),
            extensions: Option::default(),
            extras: Option::default(),
            type_: Valid(gltf_json::accessor::Type::Mat4),
            min: None,
            max: None,
            name: Some("inv_bind_matrices_accessor".to_string()),
            normalized: false,
            sparse: None,
        };
        accessors.push(inv_bind_matrices);
        if self.is_animated() {
            let animation_accessors =
                self.create_animation_accessors(joint_count, current_buffer_view_offset, per_view_running_offset, compatibility_mode);
            accessors.extend(animation_accessors);
            if self.num_morph_targets() > 0 && body_idx == 0 {
                let morph_target_accessors = self.create_morph_target_accessors(vertex_count, current_buffer_view_offset, per_view_running_offset);
                accessors.extend(morph_target_accessors);
            }
        }
        accessors
    }
    /// Function for creating the animation accessors
    #[allow(clippy::too_many_lines)]
    fn create_animation_accessors(
        &self,
        joint_count: usize,
        current_buffer_view_offset: u32,
        per_view_running_offset: &mut [usize; 6],
        compatibility_mode: GltfCompatibilityMode,
    ) -> Vec<gltf_json::Accessor> {
        let mut animation_accessors: Vec<gltf_json::Accessor> = vec![];
        let min_keyframe = self
            .keyframe_times
            .as_ref()
            .expect("keyframe_times should exist")
            .iter()
            .copied()
            .min_by(|a, b| a.partial_cmp(b).expect("Tried to compare a NaN"))
            .expect("keyframe_times should have elements in the vector");
        let max_keyframe = self
            .keyframe_times
            .as_ref()
            .unwrap()
            .iter()
            .copied()
            .max_by(|a, b| a.partial_cmp(b).expect("Tried to compare a NaN"))
            .unwrap();
        let keyframe_times = gltf_json::Accessor {
            buffer_view: Some(gltf_json::Index::new(BufferViewIDs::Keyframe as u32 + current_buffer_view_offset)),
            byte_offset: Some(USize64::from(per_view_running_offset[BufferViewIDs::Keyframe as usize])),
            count: USize64::from(self.frame_count.unwrap()),
            component_type: Valid(gltf_json::accessor::GenericComponentType(gltf_json::accessor::ComponentType::F32)),
            extensions: Option::default(),
            extras: Option::default(),
            type_: Valid(gltf_json::accessor::Type::Scalar),
            min: Some(gltf_json::Value::from(Vec::from([min_keyframe]))),
            max: Some(gltf_json::Value::from(Vec::from([max_keyframe]))),
            name: Some("keyframes_accessor".to_string()),
            normalized: false,
            sparse: None,
        };
        animation_accessors.push(keyframe_times);
        let mut running_buffer_view = BufferViewIDs::Animation as u32 + current_buffer_view_offset;
        for j_idx in 0..joint_count {
            let accessor_name = format!("joint_{j_idx}_animations_accessor");
            let joint_animation_accessor = gltf_json::Accessor {
                buffer_view: Some(gltf_json::Index::new(running_buffer_view)),
                byte_offset: Some(USize64::from(per_view_running_offset[BufferViewIDs::Animation as usize])),
                count: USize64::from(self.frame_count.unwrap()),
                component_type: Valid(gltf_json::accessor::GenericComponentType(gltf_json::accessor::ComponentType::F32)),
                extensions: Option::default(),
                extras: Option::default(),
                type_: Valid(gltf_json::accessor::Type::Vec4),
                min: None,
                max: None,
                name: Some(accessor_name),
                normalized: false,
                sparse: None,
            };
            animation_accessors.push(joint_animation_accessor);
            running_buffer_view += 1;
        }
        let accessor_name = "root_translation_animations_accessor".to_string();
        let body_animation_accessor = gltf_json::Accessor {
            buffer_view: Some(gltf_json::Index::new(running_buffer_view)),
            byte_offset: Some(USize64::from(per_view_running_offset[BufferViewIDs::Animation as usize])),
            count: USize64::from(self.frame_count.unwrap()),
            component_type: Valid(gltf_json::accessor::GenericComponentType(gltf_json::accessor::ComponentType::F32)),
            extensions: Option::default(),
            extras: Option::default(),
            type_: Valid(gltf_json::accessor::Type::Vec3),
            min: None,
            max: None,
            name: Some(accessor_name),
            normalized: false,
            sparse: None,
        };
        animation_accessors.push(body_animation_accessor);
        running_buffer_view += 1;
        let accessor_name = "root_scale_animations_accessor".to_string();
        let vis_animation_accessor = gltf_json::Accessor {
            buffer_view: Some(gltf_json::Index::new(running_buffer_view)),
            byte_offset: Some(USize64::from(per_view_running_offset[BufferViewIDs::Animation as usize])),
            count: USize64::from(self.frame_count.unwrap()),
            component_type: Valid(gltf_json::accessor::GenericComponentType(gltf_json::accessor::ComponentType::F32)),
            extensions: Option::default(),
            extras: Option::default(),
            type_: Valid(gltf_json::accessor::Type::Vec3),
            min: None,
            max: None,
            name: Some(accessor_name),
            normalized: false,
            sparse: None,
        };
        animation_accessors.push(vis_animation_accessor);
        running_buffer_view += 1;
        if compatibility_mode == GltfCompatibilityMode::Unreal {
            let accessor_name = "pelvis_rel_translation_animations_accessor".to_string();
            let pelvis_animation_accessor = gltf_json::Accessor {
                buffer_view: Some(gltf_json::Index::new(running_buffer_view)),
                byte_offset: Some(USize64::from(per_view_running_offset[BufferViewIDs::Animation as usize])),
                count: USize64::from(self.frame_count.unwrap()),
                component_type: Valid(gltf_json::accessor::GenericComponentType(gltf_json::accessor::ComponentType::F32)),
                extensions: Option::default(),
                extras: Option::default(),
                type_: Valid(gltf_json::accessor::Type::Vec3),
                min: None,
                max: None,
                name: Some(accessor_name),
                normalized: false,
                sparse: None,
            };
            animation_accessors.push(pelvis_animation_accessor);
            running_buffer_view += 1;
        }
        if self.num_morph_targets() > 0 {
            let accessor_name = "morph_targets_weights_accessor".to_string();
            let morph_targets_weights_accessor = gltf_json::Accessor {
                buffer_view: Some(gltf_json::Index::new(running_buffer_view)),
                byte_offset: Some(USize64::from(per_view_running_offset[BufferViewIDs::Animation as usize])),
                count: USize64::from(self.frame_count.unwrap() * self.num_morph_targets()),
                component_type: Valid(gltf_json::accessor::GenericComponentType(gltf_json::accessor::ComponentType::F32)),
                extensions: Option::default(),
                extras: Option::default(),
                type_: Valid(gltf_json::accessor::Type::Scalar),
                min: None,
                max: None,
                name: Some(accessor_name),
                normalized: false,
                sparse: None,
            };
            animation_accessors.push(morph_targets_weights_accessor);
            running_buffer_view += 1;
        }
        per_view_running_offset[BufferViewIDs::Animation as usize] += running_buffer_view as usize;
        animation_accessors
    }
    /// Function for creating accessors for morph targets
    fn create_morph_target_accessors(
        &self,
        vertex_count: usize,
        current_buffer_view_offset: u32,
        per_view_running_offset: &mut [usize; 6],
    ) -> Vec<gltf_json::Accessor> {
        let mut morph_target_accessors: Vec<gltf_json::Accessor> = vec![];
        let mut running_buffer_view = u32::try_from(per_view_running_offset[BufferViewIDs::Animation as usize]).expect("Could not convert to U32!")
            + current_buffer_view_offset;
        for morph_target_idx in 0..self.num_morph_targets() {
            let accessor_name = format!("morph_{morph_target_idx}_accessor");
            let current_morph_target = self.morph_targets.as_ref().unwrap().slice(s![morph_target_idx, .., ..]);
            let current_morph_target_na = current_morph_target.to_owned().clone().into_nalgebra();
            let (min, max) = Geom::get_bounding_points(&current_morph_target_na, None);
            let (min_vec, max_vec): (Vec<f32>, Vec<f32>) = (min.iter().copied().collect(), max.iter().copied().collect());
            let morph_target_accessor = gltf_json::Accessor {
                buffer_view: Some(gltf_json::Index::new(running_buffer_view)),
                byte_offset: Some(USize64::from(per_view_running_offset[BufferViewIDs::Deformation as usize])),
                count: USize64::from(vertex_count),
                component_type: Valid(gltf_json::accessor::GenericComponentType(gltf_json::accessor::ComponentType::F32)),
                extensions: Option::default(),
                extras: Option::default(),
                type_: Valid(gltf_json::accessor::Type::Vec3),
                min: Some(gltf_json::Value::from(min_vec)),
                max: Some(gltf_json::Value::from(max_vec)),
                name: Some(accessor_name),
                normalized: false,
                sparse: None,
            };
            morph_target_accessors.push(morph_target_accessor);
            running_buffer_view += 1;
        }
        per_view_running_offset[BufferViewIDs::Deformation as usize] += running_buffer_view as usize;
        morph_target_accessors
    }
    /// Function for creating animation channels
    #[allow(clippy::cast_possible_truncation)]
    fn create_animation_channels(
        &self,
        joint_count: usize,
        root_idx: u32,
        skeleton_root_idx: usize,
        sampler_start_idx: usize,
        compatibility_mode: GltfCompatibilityMode,
    ) -> Vec<gltf_json::animation::Channel> {
        let mut animation_channels: Vec<gltf_json::animation::Channel> = vec![];
        let mut sampler_idx = sampler_start_idx;
        for j_idx in 0..joint_count {
            let animation_target = gltf_json::animation::Target {
                extensions: Option::default(),
                extras: Option::default(),
                node: gltf_json::Index::new(u32::try_from(j_idx + skeleton_root_idx).expect("Could not convert to u32!")),
                path: gltf_json::validation::Checked::Valid(gltf_json::animation::Property::Rotation),
            };
            let channel = gltf_json::animation::Channel {
                sampler: gltf_json::Index::new(u32::try_from(sampler_idx).expect("Could not convert to u32!")),
                target: animation_target,
                extensions: Option::default(),
                extras: Option::default(),
            };
            animation_channels.push(channel);
            sampler_idx += 1;
        }
        let animation_target = gltf_json::animation::Target {
            extensions: Option::default(),
            extras: Option::default(),
            node: gltf_json::Index::new(root_idx),
            path: gltf_json::validation::Checked::Valid(gltf_json::animation::Property::Translation),
        };
        let channel = gltf_json::animation::Channel {
            sampler: gltf_json::Index::new(u32::try_from(sampler_idx).expect("Could not convert to u32!")),
            target: animation_target,
            extensions: Option::default(),
            extras: Option::default(),
        };
        animation_channels.push(channel);
        sampler_idx += 1;
        let animation_target = gltf_json::animation::Target {
            extensions: Option::default(),
            extras: Option::default(),
            node: gltf_json::Index::new(root_idx),
            path: gltf_json::validation::Checked::Valid(gltf_json::animation::Property::Scale),
        };
        let channel = gltf_json::animation::Channel {
            sampler: gltf_json::Index::new(u32::try_from(sampler_idx).expect("Could not convert to u32!")),
            target: animation_target,
            extensions: Option::default(),
            extras: Option::default(),
        };
        animation_channels.push(channel);
        sampler_idx += 1;
        let mesh_skin_binding_node_idx = root_idx - 1;
        if compatibility_mode == GltfCompatibilityMode::Unreal {
            let animation_target = gltf_json::animation::Target {
                extensions: Option::default(),
                extras: Option::default(),
                node: gltf_json::Index::new(u32::try_from(skeleton_root_idx).expect("Could not convert to u32!")),
                path: gltf_json::validation::Checked::Valid(gltf_json::animation::Property::Translation),
            };
            let channel = gltf_json::animation::Channel {
                sampler: gltf_json::Index::new(u32::try_from(sampler_idx).expect("Could not convert to u32!")),
                target: animation_target,
                extensions: Option::default(),
                extras: Option::default(),
            };
            animation_channels.push(channel);
            sampler_idx += 1;
        }
        if self.num_morph_targets() > 0 {
            let mtw_animation_target = gltf_json::animation::Target {
                extensions: Option::default(),
                extras: Option::default(),
                node: gltf_json::Index::new(mesh_skin_binding_node_idx),
                path: gltf_json::validation::Checked::Valid(gltf_json::animation::Property::MorphTargetWeights),
            };
            let channel = gltf_json::animation::Channel {
                sampler: gltf_json::Index::new(u32::try_from(sampler_idx).expect("Could not convert to u32!")),
                target: mtw_animation_target,
                extensions: Option::default(),
                extras: Option::default(),
            };
            animation_channels.push(channel);
        }
        animation_channels
    }
    /// Function for creating animation samplers
    fn create_animation_samplers(
        &self,
        joint_count: usize,
        current_buffer_view_offset: u32,
        compatibility_mode: GltfCompatibilityMode,
    ) -> Vec<gltf_json::animation::Sampler> {
        let mut animation_samplers: Vec<gltf_json::animation::Sampler> = vec![];
        let mut current_accessor = 8 + current_buffer_view_offset;
        for _ in 0..joint_count {
            let sampler = gltf_json::animation::Sampler {
                extensions: Option::default(),
                extras: Option::default(),
                input: gltf_json::Index::new(7 + current_buffer_view_offset),
                interpolation: gltf_json::validation::Checked::Valid(gltf_json::animation::Interpolation::Linear),
                output: gltf_json::Index::new(current_accessor),
            };
            animation_samplers.push(sampler);
            current_accessor += 1;
        }
        let sampler = gltf_json::animation::Sampler {
            extensions: Option::default(),
            extras: Option::default(),
            input: gltf_json::Index::new(7 + current_buffer_view_offset),
            interpolation: gltf_json::validation::Checked::Valid(gltf_json::animation::Interpolation::Linear),
            output: gltf_json::Index::new(current_accessor),
        };
        animation_samplers.push(sampler);
        current_accessor += 1;
        let sampler = gltf_json::animation::Sampler {
            extensions: Option::default(),
            extras: Option::default(),
            input: gltf_json::Index::new(7 + current_buffer_view_offset),
            interpolation: gltf_json::validation::Checked::Valid(gltf_json::animation::Interpolation::Step),
            output: gltf_json::Index::new(current_accessor),
        };
        animation_samplers.push(sampler);
        current_accessor += 1;
        if compatibility_mode == GltfCompatibilityMode::Unreal {
            let sampler = gltf_json::animation::Sampler {
                extensions: Option::default(),
                extras: Option::default(),
                input: gltf_json::Index::new(7 + current_buffer_view_offset),
                interpolation: gltf_json::validation::Checked::Valid(gltf_json::animation::Interpolation::Linear),
                output: gltf_json::Index::new(current_accessor),
            };
            animation_samplers.push(sampler);
            current_accessor += 1;
        }
        if self.num_morph_targets() > 0 {
            let sampler = gltf_json::animation::Sampler {
                extensions: Option::default(),
                extras: Option::default(),
                input: gltf_json::Index::new(7 + current_buffer_view_offset),
                interpolation: gltf_json::validation::Checked::Valid(gltf_json::animation::Interpolation::Linear),
                output: gltf_json::Index::new(current_accessor),
            };
            animation_samplers.push(sampler);
        }
        animation_samplers
    }
    /// Function for creating morph targets
    pub fn create_morph_targets(&self, pose_dirs_accessors_start_idx: usize) -> Vec<gltf_json::mesh::MorphTarget> {
        let mut pose_dirs_morph_targets: Vec<gltf_json::mesh::MorphTarget> = vec![];
        let mut running_pose_dirs_accessor = u32::try_from(pose_dirs_accessors_start_idx).expect("Not able to convert to u32");
        for _ in 0..self.num_morph_targets() {
            let morph_target = gltf_json::mesh::MorphTarget {
                positions: Some(gltf_json::Index::new(running_pose_dirs_accessor)),
                normals: None,
                tangents: None,
            };
            pose_dirs_morph_targets.push(morph_target);
            running_pose_dirs_accessor += 1;
        }
        pose_dirs_morph_targets
    }
    /// General purpose bind matrix computation from joint transformations
    pub fn create_bind_matrices(&self, rot_mat: &nd::Array3<f32>, joint_trans: &nd::Array2<f32>, joint_parents: &[u32]) -> nd::Array3<f32> {
        assert!(
            rot_mat.shape()[0] == joint_trans.shape()[0],
            "Number of rotation matrices dont match number of translation matrices!"
        );
        let num_joints = rot_mat.shape()[0];
        let mut bind_matrices = ndarray::Array3::<f32>::zeros((num_joints, 4, 4));
        bind_matrices.slice_mut(s![0, 0..3, 0..3]).assign(&rot_mat.slice(s![0, .., ..]));
        bind_matrices.slice_mut(s![0, 0..3, 3]).assign(&joint_trans.slice(s![0, ..]));
        bind_matrices[[0, 3, 3]] = 1.0;
        for j_idx in 1..num_joints {
            let parent_index = joint_parents[j_idx] as usize;
            let parent_transform = bind_matrices.index_axis(nd::Axis(0), parent_index);
            let mut local_transform = ndarray::Array2::<f32>::zeros((4, 4));
            local_transform.slice_mut(s![0..3, 0..3]).assign(&rot_mat.slice(s![j_idx, .., ..]));
            let local_translation = Array::from_vec(vec_to_vec(&compute_local_translation(j_idx, joint_parents, joint_trans)));
            local_transform.slice_mut(s![0..3, 3]).assign(&local_translation);
            local_transform[[3, 3]] = 1.0;
            let global_transform = parent_transform.dot(&local_transform);
            bind_matrices.slice_mut(s![j_idx, .., ..]).assign(&global_transform);
        }
        bind_matrices
    }
    /// Function to create animation buffer data
    pub fn create_animation_data(&self, body_idx: usize, compatibility_mode: GltfCompatibilityMode) -> Vec<u8> {
        let mut animation_data: Vec<u8> = vec![];
        let keyframe_data = to_padded_byte_vector(self.keyframe_times.as_ref().unwrap());
        let rotation_animation_data = self.per_body_data[body_idx].body_rotations.as_ref().unwrap();
        let mut translation_animation_data = self.per_body_data[body_idx].body_translations.as_ref().unwrap().clone();
        let scale_animation_data = self.per_body_data[body_idx].body_scales.as_ref().unwrap().clone();
        animation_data.extend_from_slice(keyframe_data.as_slice());
        assert_eq!(rotation_animation_data.shape()[1], translation_animation_data.shape()[0]);
        for j_idx in 0..rotation_animation_data.shape()[0] {
            let mut quaternions: Vec<f32> = vec![];
            for r_idx in 0..rotation_animation_data.shape()[1] {
                let rotation = rotation_animation_data.slice(s![j_idx, r_idx, ..]);
                let axis_angle_rotation = na::Vector3::new(rotation[0], rotation[1], rotation[2]);
                let mut quaternion_rotation =
                    na::UnitQuaternion::from_axis_angle(&na::UnitVector3::new_normalize(axis_angle_rotation), axis_angle_rotation.norm());
                if axis_angle_rotation.norm() == 0.0 {
                    quaternion_rotation = na::UnitQuaternion::default();
                }
                quaternions.extend(quaternion_rotation.as_vector().data.as_slice());
            }
            let joint_anim_data = to_padded_byte_vector(&quaternions);
            animation_data.append(&mut joint_anim_data.clone());
        }
        if compatibility_mode == GltfCompatibilityMode::Unreal {
            let mut pelvis_relative_trans = translation_animation_data.clone();
            for mut row in translation_animation_data.axis_iter_mut(Axis(0)) {
                row[1] = 0.0;
            }
            for mut row in pelvis_relative_trans.axis_iter_mut(Axis(0)) {
                row[0] = 0.0;
                row[2] = 0.0;
            }
            let trans_anim_data = to_padded_byte_vector(&translation_animation_data.to_owned().into_raw_vec_and_offset().0);
            animation_data.append(&mut trans_anim_data.clone());
            let scale_anim_data = to_padded_byte_vector(&scale_animation_data.to_owned().into_raw_vec_and_offset().0);
            animation_data.append(&mut scale_anim_data.clone());
            let pelvis_rel_anim_data = to_padded_byte_vector(&pelvis_relative_trans.to_owned().into_raw_vec_and_offset().0);
            animation_data.append(&mut pelvis_rel_anim_data.clone());
        } else {
            let trans_anim_data = to_padded_byte_vector(&translation_animation_data.to_owned().into_raw_vec_and_offset().0);
            animation_data.append(&mut trans_anim_data.clone());
            let scale_anim_data = to_padded_byte_vector(&scale_animation_data.to_owned().into_raw_vec_and_offset().0);
            animation_data.append(&mut scale_anim_data.clone());
        }
        if self.num_morph_targets() > 0 {
            let morph_target_weights_data = self.per_body_data[body_idx].per_frame_blend_weights.as_ref().unwrap();
            let weights_anim_data = to_padded_byte_vector(&morph_target_weights_data.to_owned().into_raw_vec_and_offset().0);
            animation_data.append(&mut weights_anim_data.clone());
        }
        animation_data
    }
    /// Function to compose all present buffer views and buffers
    #[allow(clippy::too_many_arguments)]
    fn compose_buffer_views(
        &self,
        body_idx: usize,
        buffer_views: Vec<gltf_json::buffer::View>,
        index_data: &[u8],
        vertex_data: &[u8],
        inv_bind_mat_data: &[u8],
        textures: &mut [GltfTextureInfo],
        compatibility_mode: GltfCompatibilityMode,
    ) -> (Vec<u8>, Vec<gltf_json::buffer::View>) {
        let mut out_data: Vec<u8> = vec![];
        let mut out_buffer_views: Vec<gltf_json::buffer::View> = vec![];
        out_data.append(&mut index_data.to_owned());
        out_data.append(&mut vertex_data.to_owned());
        out_data.append(&mut inv_bind_mat_data.to_owned());
        if self.is_animated() {
            let mut animation_data = self.create_animation_data(body_idx, compatibility_mode);
            out_data.append(&mut animation_data);
            if self.num_morph_targets() > 0 && body_idx == 0 {
                for morph_target_idx in 0..self.num_morph_targets() {
                    let posedir = self.morph_targets.as_ref().unwrap().slice(s![morph_target_idx, .., ..]).to_owned();
                    let posedir_data = to_padded_byte_vector(&posedir.to_owned().into_raw_vec_and_offset().0);
                    out_data.append(&mut posedir_data.clone());
                }
            }
        }
        out_buffer_views.extend(buffer_views);
        let mut buffer_offset = out_data.len();
        let mut buffer_index: usize = out_buffer_views.len();
        for (sampler_index, texture) in textures.iter_mut().enumerate() {
            let mut buffer_view = texture.buffer_view.clone();
            buffer_view.byte_offset = Some(USize64::from(buffer_offset));
            out_buffer_views.push(buffer_view);
            texture.buffer_index = buffer_index;
            texture.image.buffer_view = Some(gltf_json::Index::new(u32::try_from(buffer_index).expect("Issue converting to u32!")));
            texture.texture.sampler = Some(gltf_json::Index::new(u32::try_from(sampler_index).expect("Issue converting to u32!")));
            out_data.append(&mut texture.image_data.clone());
            buffer_offset += texture.buffer_size;
            buffer_index += 1;
        }
        (out_data, out_buffer_views)
    }
    /// Add a GLTF texture
    fn add_texture(&mut self, img: &DynImage, index: usize, name: &str) -> Option<GltfTextureInfo> {
        let mut image_data: Vec<u8> = vec![];
        let mut target = Cursor::new(&mut image_data);
        let image_data_buffer = img.write_to(&mut target, image::ImageFormat::Png);
        if image_data_buffer.is_ok() {
            let _ = target.flush();
            while image_data.len() % 4 != 0 {
                image_data.push(0);
            }
            let mut image_buffer_size = image_data.len();
            align_to_multiple_of_four(&mut image_buffer_size);
            let image_buffer_view = gltf_json::buffer::View {
                buffer: gltf_json::Index::new(0),
                byte_length: USize64::from(image_buffer_size),
                byte_offset: Some(USize64::from(0_usize)),
                byte_stride: Option::default(),
                extensions: Option::default(),
                extras: Option::default(),
                name: Some(name.to_string()),
                target: None,
            };
            let image = gltf_json::image::Image {
                buffer_view: Some(gltf_json::Index::new(u32::try_from(index).expect("Issue converting to u32!"))),
                mime_type: Some(gltf_json::image::MimeType("image/png".to_string())),
                name: Some(name.to_string()),
                uri: None,
                extensions: None,
                extras: None,
            };
            let texture = gltf_json::Texture {
                name: Some(name.to_string()),
                sampler: Some(gltf_json::Index::new(u32::try_from(index).expect("Issue converting to u32!"))),
                source: gltf_json::Index::new(u32::try_from(index).expect("Issue converting to u32!")),
                extensions: None,
                extras: None,
            };
            let sampler = gltf_json::texture::Sampler {
                name: Some(name.to_string()),
                mag_filter: Some(gltf_json::validation::Checked::Valid(gltf_json::texture::MagFilter::Linear)),
                min_filter: Some(gltf_json::validation::Checked::Valid(gltf_json::texture::MinFilter::Linear)),
                wrap_s: gltf_json::validation::Checked::Valid(gltf_json::texture::WrappingMode::ClampToEdge),
                wrap_t: gltf_json::validation::Checked::Valid(gltf_json::texture::WrappingMode::ClampToEdge),
                extensions: None,
                extras: None,
            };
            let texture_info = GltfTextureInfo {
                buffer_size: image_data.len(),
                image_data,
                image,
                buffer_view: image_buffer_view,
                buffer_index: 0,
                texture,
                sampler,
            };
            return Some(texture_info);
        }
        log!("add_texture FAILED: {}", name);
        None
    }
    /// Prepare normal map for GLTF
    #[allow(clippy::cast_sign_loss)]
    fn prepare_normals(&mut self, smpl_textures: &mut SmplTextures, texture_infos: &mut Vec<GltfTextureInfo>, normals_tex: Option<&DynImage>) {
        if let Some(img) = normals_tex {
            let normals_tex = self.add_texture(img, texture_infos.len(), "normals");
            if let Some(normals_tex) = normals_tex {
                smpl_textures.normals_index = Some(texture_infos.len());
                texture_infos.push(normals_tex);
            }
        }
    }
    /// Prepare metallic-roughness map for GLTF
    fn prepare_metallic_roughness(
        &mut self,
        smpl_textures: &mut SmplTextures,
        texture_infos: &mut Vec<GltfTextureInfo>,
        metalness_tex: Option<&DynImage>,
        roughness_tex: Option<&DynImage>,
    ) {
        let mut w: u32 = 0;
        let mut h: u32 = 0;
        if let Some(img) = metalness_tex {
            w = img.width();
            h = img.height();
        }
        if let Some(img) = roughness_tex {
            w = w.max(img.width());
            h = h.max(img.height());
        }
        let mut metalness: Option<Vec<u8>> = None;
        if let Some(img) = metalness_tex {
            if img.width() != w || img.height() != h {
                let resized_img = img.resize(w, h, FilterType::Gaussian);
                metalness = Some(resized_img.as_luma8().unwrap().to_vec());
            } else {
                metalness = Some(img.as_bytes().to_vec());
            }
        }
        let mut roughness: Option<Vec<u8>> = None;
        if let Some(img) = roughness_tex {
            if img.width() != w || img.height() != h {
                let resized_img = img.resize(w, h, FilterType::Gaussian);
                roughness = Some(resized_img.as_luma8().unwrap().to_vec());
            } else {
                roughness = Some(img.as_bytes().to_vec());
            }
        }
        let num_pixels: usize = (w * h) as usize;
        let mut metal_roughness_pixels: Vec<u8> = vec![];
        if let Some(metalness_pixels) = metalness {
            if let Some(roughness_pixels) = roughness {
                for (m, r) in metalness_pixels.iter().zip(roughness_pixels.iter()).take(num_pixels) {
                    metal_roughness_pixels.push(0);
                    metal_roughness_pixels.push(*r);
                    metal_roughness_pixels.push(*m);
                }
            } else {
                for &m in metalness_pixels.iter().take(num_pixels) {
                    metal_roughness_pixels.push(0);
                    metal_roughness_pixels.push(0);
                    metal_roughness_pixels.push(m);
                }
            }
        } else if let Some(roughness_pixels) = roughness {
            for &r in roughness_pixels.iter().take(num_pixels) {
                metal_roughness_pixels.push(0);
                metal_roughness_pixels.push(r);
                metal_roughness_pixels.push(0);
            }
        }
        if !metal_roughness_pixels.is_empty() {
            let metal_roughness_image = RgbImage::from_vec(w, h, metal_roughness_pixels);
            if let Some(image) = metal_roughness_image {
                let image = DynImage::from(image);
                let metallic_roughness = self.add_texture(&image, texture_infos.len(), "metal_roughness");
                if let Some(metallic_roughness) = metallic_roughness {
                    smpl_textures.metalic_roughtness_index = Some(texture_infos.len());
                    texture_infos.push(metallic_roughness);
                }
            }
        }
    }
    /// Create a new joint with the given transformations
    fn create_joint(&self, name: String, translation: &[f32], rotation: &Vector3f, children: Option<Vec<gltf_json::Index<Node>>>) -> Node {
        let cur_vec = na::Vector3::new(rotation.x, rotation.y, rotation.z);
        let mut cur_q = na::UnitQuaternion::from_axis_angle(&na::UnitVector3::new_normalize(cur_vec), rotation.norm());
        if rotation.norm() == 0.0 {
            cur_q = na::UnitQuaternion::default();
        }
        let translation: [f32; 3] = [translation[0], translation[1], translation[2]];
        let unit_quaternion = [cur_q[0], cur_q[1], cur_q[2], cur_q[3]];
        Node {
            children,
            mesh: None,
            skin: None,
            name: Some(name),
            rotation: Some(UnitQuaternion(unit_quaternion)),
            translation: Some(translation),
            ..Default::default()
        }
    }
    fn gather_children(&self, id: u32, parent_ids: &[u32], offset: u32) -> Option<Vec<gltf_json::Index<Node>>> {
        let mut children: Vec<gltf_json::Index<Node>> = vec![];
        for (p, &parent_id) in parent_ids.iter().enumerate() {
            if parent_id == id {
                let index = u32::try_from(p).expect("Index conversion error: usize value is too large to fit in a u32");
                children.push(gltf_json::Index::<Node>::new(index + offset));
            }
        }
        if !children.is_empty() {
            return Some(children);
        }
        None
    }
    /// Add a skin to GLTF
    #[allow(clippy::too_many_arguments)]
    fn add_skin(
        &mut self,
        name: String,
        body_idx: usize,
        current_armature_idx: u32,
        accessor_offset: u32,
        nodes: &mut Vec<Node>,
        skins: &mut Vec<gltf_json::Skin>,
        joints: &mut Vec<gltf_json::Index<Node>>,
        compatibility_mode: GltfCompatibilityMode,
    ) -> gltf_json::Index<Node> {
        let metadata = crate::common::metadata::smpl_metadata(&self.smpl_type);
        let joint_translations = self.per_body_data[body_idx].default_joint_translations.as_ref().unwrap();
        let skeleton_root_index = u32::try_from(nodes.len()).expect("Issue converting Node idx to u32");
        let global_translation = vec_from_array0_f(self.per_body_data[body_idx].body_translation.as_ref().unwrap());
        let mut skeleton_root_translation = compute_local_translation(0, &metadata.joint_parents, joint_translations);
        if compatibility_mode == GltfCompatibilityMode::Smpl {
            skeleton_root_translation = addv3f(&skeleton_root_translation, &global_translation);
        }
        let mut joint_rotation = Vector3f::zeros();
        if let Some(pose) = self.per_body_data[body_idx].pose.as_ref() {
            let rot = pose.joint_poses.row(0);
            joint_rotation = Vector3f::new(rot[0], rot[1], rot[2]);
        }
        let skeleton_root = self.create_joint(
            "pelvis".to_string(),
            vec_to_vec(&skeleton_root_translation).as_slice(),
            &joint_rotation,
            self.gather_children(0, &metadata.joint_parents, skeleton_root_index),
        );
        nodes.push(skeleton_root);
        joints.push(gltf_json::Index::new(skeleton_root_index));
        let joint_names = if compatibility_mode == GltfCompatibilityMode::Unreal {
            smpl_x::JOINT_NAMES_UNREAL.map(std::string::ToString::to_string).to_vec()
        } else {
            metadata.joint_names
        };
        for (j, name) in joint_names.iter().enumerate().take(metadata.num_joints + 1).skip(1) {
            if let Some(pose) = self.per_body_data[body_idx].pose.as_ref() {
                let rot = pose.joint_poses.row(j);
                joint_rotation = Vector3f::new(rot[0], rot[1], rot[2]);
            }
            let joint = self.create_joint(
                name.clone(),
                vec_to_vec(&compute_local_translation(j, &metadata.joint_parents, joint_translations)).as_slice(),
                &joint_rotation,
                self.gather_children(
                    u32::try_from(j).expect("Issue converting Joint idx to u32"),
                    &metadata.joint_parents,
                    skeleton_root_index,
                ),
            );
            let joint_index = gltf_json::Index::<Node>::new(u32::try_from(nodes.len()).expect("Issue converting Joint idx to u32"));
            nodes.push(joint);
            joints.push(joint_index);
        }
        let skin = gltf_json::Skin {
            name: Some(name),
            inverse_bind_matrices: Some(gltf_json::Index::new(accessor_offset + 6)),
            joints: joints.clone(),
            skeleton: Some(gltf_json::Index::new(current_armature_idx)),
            extensions: None,
            extras: None,
        };
        skins.push(skin);
        gltf_json::Index::<Node>::new(skeleton_root_index)
    }
    /// Create camera animation buffer views
    #[allow(clippy::cast_precision_loss)]
    fn create_camera_animation_buffer_views(&self, running_offset: &mut usize) -> Option<(Vec<gltf_json::buffer::View>, Vec<u8>)> {
        let camera_track = self.camera_track.as_ref()?;
        let mut buffer_views = Vec::new();
        let mut buffer_data = Vec::new();
        if let Some(translations) = camera_track.per_frame_translations.as_ref() {
            let trans_data = to_padded_byte_vector(translations.as_slice().unwrap());
            let trans_len = trans_data.len();
            let trans_view = gltf_json::buffer::View {
                buffer: gltf_json::Index::new(0),
                byte_length: USize64::from(trans_len),
                byte_offset: Some(USize64::from(*running_offset)),
                byte_stride: None,
                extensions: None,
                extras: Option::default(),
                name: Some("camera_translations".to_string()),
                target: None,
            };
            buffer_data.extend(trans_data);
            *running_offset += trans_len;
            buffer_views.push(trans_view);
        }
        if let Some(rotations) = camera_track.per_frame_rotations.as_ref() {
            let rot_data = to_padded_byte_vector(rotations.as_slice().unwrap());
            let rot_len = rot_data.len();
            let rot_view = gltf_json::buffer::View {
                buffer: gltf_json::Index::new(0),
                byte_length: USize64::from(rot_len),
                byte_offset: Some(USize64::from(*running_offset)),
                byte_stride: None,
                extensions: None,
                extras: Option::default(),
                name: Some("camera_rotations".to_string()),
                target: None,
            };
            buffer_data.extend(rot_data);
            *running_offset += rot_len;
            buffer_views.push(rot_view);
        }
        Some((buffer_views, buffer_data))
    }
    /// Create camera animation accessors
    fn create_camera_animation_accessors(&self, current_buffer_view_offset: u32) -> Option<Vec<gltf_json::Accessor>> {
        let camera_track = self.camera_track.as_ref()?;
        let mut accessors = Vec::new();
        let mut current_view = current_buffer_view_offset;
        if let Some(translations) = camera_track.per_frame_translations.as_ref() {
            accessors.push(gltf_json::Accessor {
                buffer_view: Some(gltf_json::Index::new(current_view)),
                byte_offset: None,
                component_type: Valid(gltf_json::accessor::GenericComponentType(gltf_json::accessor::ComponentType::F32)),
                count: USize64::from(translations.shape()[0]),
                type_: Valid(gltf_json::accessor::Type::Vec3),
                min: None,
                max: None,
                name: Some("camera_translations".to_string()),
                normalized: false,
                sparse: None,
                extensions: None,
                extras: Option::default(),
            });
            current_view += 1;
        }
        if let Some(rotations) = camera_track.per_frame_rotations.as_ref() {
            accessors.push(gltf_json::Accessor {
                buffer_view: Some(gltf_json::Index::new(current_view)),
                byte_offset: None,
                component_type: Valid(gltf_json::accessor::GenericComponentType(gltf_json::accessor::ComponentType::F32)),
                count: USize64::from(rotations.shape()[0]),
                type_: Valid(gltf_json::accessor::Type::Vec4),
                min: None,
                max: None,
                name: Some("camera_rotations".to_string()),
                normalized: false,
                sparse: None,
                extensions: None,
                extras: Option::default(),
            });
        }
        Some(accessors)
    }
    /// Create camera animation channels and samplers
    #[allow(clippy::cast_possible_truncation)]
    fn create_camera_animation_channels_and_samplers(
        &self,
        current_accessor_offset: u32,
        camera_node_index: u32,
        sampler_start_idx: u32,
    ) -> Option<(Vec<gltf_json::animation::Channel>, Vec<gltf_json::animation::Sampler>)> {
        let camera_track = self.camera_track.as_ref()?;
        let mut channels = Vec::new();
        let mut samplers = Vec::new();
        let mut current_accessor = current_accessor_offset;
        let times_accessor_index = 7;
        if camera_track.per_frame_translations.is_some() {
            samplers.push(gltf_json::animation::Sampler {
                input: gltf_json::Index::new(times_accessor_index),
                interpolation: Valid(gltf_json::animation::Interpolation::Linear),
                output: gltf_json::Index::new(current_accessor),
                extensions: None,
                extras: Option::default(),
            });
            channels.push(gltf_json::animation::Channel {
                sampler: gltf_json::Index::new(sampler_start_idx + samplers.len() as u32 - 1),
                target: gltf_json::animation::Target {
                    node: gltf_json::Index::new(camera_node_index),
                    path: Valid(gltf_json::animation::Property::Translation),
                    extensions: None,
                    extras: Option::default(),
                },
                extensions: None,
                extras: Option::default(),
            });
            current_accessor += 1;
        }
        if camera_track.per_frame_rotations.is_some() {
            samplers.push(gltf_json::animation::Sampler {
                input: gltf_json::Index::new(times_accessor_index),
                interpolation: Valid(gltf_json::animation::Interpolation::Linear),
                output: gltf_json::Index::new(current_accessor),
                extensions: None,
                extras: Option::default(),
            });
            channels.push(gltf_json::animation::Channel {
                sampler: gltf_json::Index::new(sampler_start_idx + samplers.len() as u32 - 1),
                target: gltf_json::animation::Target {
                    node: gltf_json::Index::new(camera_node_index),
                    path: Valid(gltf_json::animation::Property::Rotation),
                    extensions: None,
                    extras: Option::default(),
                },
                extensions: None,
                extras: Option::default(),
            });
        }
        Some((channels, samplers))
    }
    fn num_morph_targets(&self) -> usize {
        self.morph_targets.as_ref().map_or(0, |x| x.shape()[0])
    }
}
pub fn compute_local_translation(id: usize, parent_ids: &[u32], joint_translations: &nd::Array2<f32>) -> Vector3f {
    let trans = vec_from_vec(&joint_translations.row(id).to_vec());
    if id == 0 {
        return trans;
    }
    let parent_id = parent_ids[id] as usize;
    let parent_trans = vec_from_vec(&joint_translations.row(parent_id).to_vec());
    subv3f(&trans, &parent_trans)
}
