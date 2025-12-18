use super::codec::SmplCodec;
use crate::common::transform_sequence::TransformSequence;
use base64;
use gloss_renderer::components::ProjectionWithFov;
use gltf_json::{validation::Checked::Valid, Root, Value};
use log::info;
use ndarray as nd;
use smpl_utils::log;
use std::{
    fs::{self, File},
    io::Read,
    path::Path,
};
/// The ``McsCodec`` contains all of the contents of an ``.mcs`` file
#[derive(Clone)]
pub struct McsCodec {
    pub num_frames: usize,
    pub frame_rate: Option<f32>,
    pub smpl_bodies: Vec<SmplBody>,
    pub smpl_camera: Option<SmplCamera>,
}
/// ``SmplBody`` holds the contents of the ``.smpl`` file along with the frame presence
#[derive(Clone)]
pub struct SmplBody {
    pub frame_presence: Vec<usize>,
    pub codec: SmplCodec,
}
/// ``SmplCamera`` holds the information regarding the camera track
#[derive(Clone)]
pub struct SmplCamera {
    pub projection: ProjectionWithFov,
    pub transform_sequence: TransformSequence,
}
/// ``McsCodec`` for ``.mcs`` files
#[allow(clippy::cast_possible_truncation)]
impl McsCodec {
    /// Load `McsCodec` from a GLTF file.
    pub fn from_file(path: &str) -> Self {
        let mut file = File::open(path).expect("Failed to open GLTF file");
        let mut json_data = String::new();
        file.read_to_string(&mut json_data).expect("Failed to read GLTF file");
        let gltf: Root = serde_json::from_str(&json_data).expect("Failed to parse GLTF JSON");
        Self::from_gltf(&gltf)
    }
    /// Parse a scene GLTF into ``McsCodec``
    pub fn from_gltf(gltf: &Root) -> Self {
        let mut num_frames = 0;
        let mut smpl_bodies = Vec::new();
        if let Some(scene) = gltf.scenes.first() {
            if let Some(extensions) = &scene.extensions {
                if let Some(extension_value) = extensions.others.get("MC_scene_description") {
                    let extension = extension_value.as_object().expect("Expected extension to be an object");
                    if let Some(nf) = extension.get("num_frames").and_then(gltf_json::Value::as_u64) {
                        num_frames = nf as usize;
                    }
                    if let Some(smpl_bodies_data) = extension.get("smpl_bodies").and_then(|v| v.as_array()) {
                        smpl_bodies = Self::extract_smpl_bodies(gltf, smpl_bodies_data);
                    }
                }
            }
            Self {
                num_frames,
                frame_rate: smpl_bodies.first().and_then(|b| b.codec.frame_rate),
                smpl_bodies,
                smpl_camera: Self::extract_gltf_camera(gltf),
            }
        } else {
            panic!("Not able to find GLTF root! Check the GLTF file format!")
        }
    }
    /// Extract SMPL bodies and their `frame_presence` from the GLTF extension.
    fn extract_smpl_bodies(gltf: &Root, smpl_bodies_data: &[serde_json::Value]) -> Vec<SmplBody> {
        smpl_bodies_data
            .iter()
            .filter_map(|smpl_body_data| {
                smpl_body_data
                    .get("bufferView")
                    .and_then(gltf_json::Value::as_u64)
                    .map(|buffer_view_index| {
                        let buffer = Self::read_smpl_buffer(gltf, buffer_view_index as usize);
                        let frame_presence = smpl_body_data
                            .get("frame_presence")
                            .and_then(|v| v.as_array())
                            .map_or_else(Vec::new, |arr| arr.iter().filter_map(|v| v.as_u64().map(|n| n as usize)).collect());
                        let codec = SmplCodec::from_buf(&buffer);
                        SmplBody { frame_presence, codec }
                    })
            })
            .collect()
    }
    /// Reads buffer data based on buffer index.
    fn read_smpl_buffer(gltf: &Root, buffer_index: usize) -> Vec<u8> {
        let buffer = &gltf.buffers[buffer_index];
        buffer
            .uri
            .as_ref()
            .and_then(|uri| {
                if uri.starts_with("data:") {
                    uri.split(',')
                        .nth(1)
                        .map(|encoded_data| base64::decode(encoded_data).expect("Failed to decode Base64 data"))
                } else {
                    panic!("The data buffers must not be separate files!")
                }
            })
            .unwrap_or_default()
    }
    /// Extract camera track from `.mcs` file
    fn extract_gltf_camera(gltf: &Root) -> Option<SmplCamera> {
        if let Some(camera) = gltf.cameras.first() {
            let (yfov, znear, zfar, aspect_ratio) = match camera.type_.unwrap() {
                gltf_json::camera::Type::Perspective => (
                    camera.perspective.as_ref().map_or(std::f32::consts::FRAC_PI_2, |p| p.yfov),
                    camera.perspective.as_ref().map_or(0.1, |p| p.znear),
                    camera.perspective.as_ref().map_or(1000.0, |p| p.zfar.unwrap_or(1000.0)),
                    camera.perspective.as_ref().map_or(1.0, |p| p.aspect_ratio.unwrap_or(1.0)),
                ),
                gltf_json::camera::Type::Orthographic => {
                    panic!("Orthographic camera not supported!")
                }
            };
            if gltf.animations.is_empty() {
                let mut static_translation = nd::Array2::<f32>::zeros((1, 3));
                let mut static_rotation = nd::array![[0.0, 0.0, 0.0, 1.0]];
                for node in &gltf.nodes {
                    if node.camera.is_some() {
                        if let Some(translation) = &node.translation {
                            static_translation = nd::Array2::from_shape_vec((1, 3), vec![translation[0], translation[1], translation[2]]).unwrap();
                        }
                        if let Some(rotation) = &node.rotation {
                            let quat = rotation.0;
                            static_rotation = nd::Array2::from_shape_vec((1, 4), vec![quat[0], quat[1], quat[2], quat[3]]).unwrap();
                        }
                        break;
                    }
                }
                return Some(SmplCamera {
                    projection: ProjectionWithFov {
                        fovy: yfov,
                        near: znear,
                        far: zfar,
                        aspect_ratio,
                    },
                    transform_sequence: TransformSequence::new_from_rot_trans(&static_rotation, &static_translation),
                });
            }
            let mut per_frame_translations: Option<nd::Array2<f32>> = None;
            let mut per_frame_rotations: Option<nd::Array2<f32>> = None;
            for animation in &gltf.animations {
                for channel in &animation.channels {
                    let target = &channel.target;
                    let Some(node) = gltf.nodes.get(target.node.value()) else { continue };
                    if node.camera.is_none() {
                        continue;
                    }
                    let Some(sampler) = animation.samplers.get(channel.sampler.value()) else {
                        continue;
                    };
                    let Some(output_accessor) = gltf.accessors.get(sampler.output.value()) else {
                        continue;
                    };
                    let Some(buffer_view) = output_accessor.buffer_view.as_ref().and_then(|bv| gltf.buffer_views.get(bv.value())) else {
                        continue;
                    };
                    let Some(buffer) = gltf.buffers.get(buffer_view.buffer.value()) else {
                        continue;
                    };
                    let Some(uri) = &buffer.uri else { continue };
                    if !uri.starts_with("data:") {
                        continue;
                    }
                    let encoded_data = uri.split(',').nth(1).expect("Invalid data URI");
                    let buffer_data = base64::decode(encoded_data).expect("Failed to decode Base64 data");
                    let start = buffer_view.byte_offset.map_or(0, |x| x.0 as usize);
                    let length = buffer_view.byte_length.0 as usize;
                    let data = &buffer_data[start..start + length];
                    let floats: Vec<f32> = data.chunks(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
                    if let Valid(path) = &target.path {
                        match path {
                            gltf_json::animation::Property::Translation => {
                                let num_frames = floats.len() / 3;
                                per_frame_translations = Some(nd::Array2::from_shape_vec((num_frames, 3), floats).unwrap());
                            }
                            gltf_json::animation::Property::Rotation => {
                                let num_frames = floats.len() / 4;
                                per_frame_rotations = Some(nd::Array2::from_shape_vec((num_frames, 4), floats).unwrap());
                            }
                            _ => {}
                        }
                    }
                }
            }
            Some(SmplCamera {
                projection: ProjectionWithFov {
                    fovy: yfov,
                    near: znear,
                    far: zfar,
                    aspect_ratio,
                },
                transform_sequence: TransformSequence::new_from_rot_trans(
                    &per_frame_rotations.unwrap_or(nd::array![[0.0, 0.0, 0.0, 1.0]]),
                    &per_frame_translations.unwrap_or(nd::Array2::<f32>::zeros((1, 3))),
                ),
            })
        } else {
            None
        }
    }
    /// Export `McsCodec` to an MCS file
    pub fn to_file(&self, path: &str) {
        let parent_path = Path::new(path).parent();
        let file_name = Path::new(path).file_name();
        let Some(parent_path) = parent_path else {
            log!("Error: Exporting MCS - Something wrong with the path: {}", path);
            return;
        };
        if !parent_path.exists() {
            let _ = fs::create_dir_all(parent_path);
        }
        let Some(_) = file_name else {
            log!("Error: Exporting MCS - no file name found: {}", path);
            return;
        };
        let gltf_json = self.to_gltf_json();
        std::fs::write(path, gltf_json).expect("Failed to write MCS file");
        info!("Exported MCS file to: {path}");
    }
    /// Create the base empty Mcs structure
    pub fn create_gltf_structure(&self) -> Value {
        let mut gltf_json = serde_json::json!(
            { "asset" : { "version" : "2.0", "generator" : "smpl-rs McsCodec Exporter" },
            "scene" : 0, "scenes" : [{ "nodes" : [0], "extensions" : {
            "MC_scene_description" : { "num_frames" : self.num_frames, "smpl_bodies" : []
            } } }], "buffers" : [], "bufferViews" : [], "accessors" : [], "animations" :
            [], "extensionsUsed" : ["MC_scene_description"] }
        );
        if let Some(smpl_camera) = &self.smpl_camera {
            gltf_json["nodes"] = serde_json::json!(
                [{ "name" : "RootNode", "children" : [1] }, { "name" : "AnimatedCamera",
                "camera" : 0, "translation" : [0.0, 0.0, 0.0], "rotation" : [0.0, 0.0,
                0.0, 1.0] }]
            );
            gltf_json["cameras"] = serde_json::json!(
                [{ "type" : "perspective", "perspective" : { "yfov" : smpl_camera
                .projection.fovy, "znear" : smpl_camera.projection.near, "zfar" :
                smpl_camera.projection.far, "aspectRatio" : smpl_camera.projection
                .aspect_ratio } }]
            );
        } else {
            gltf_json["nodes"] = serde_json::json!([{ "name" : "RootNode" }]);
        }
        gltf_json
    }
    /// Add SMPL buffers to the Mcs GLTF JSON
    pub fn add_smpl_buffers_to_gltf(&self, gltf_json: &mut Value) {
        for (body_index, smpl_body) in self.smpl_bodies.iter().enumerate() {
            let buffer_data = smpl_body.codec.to_buf();
            let buffer_base64 = base64::encode(&buffer_data);
            gltf_json["buffers"].as_array_mut().unwrap().push(serde_json::json!(
                { "byteLength" : buffer_data.len(), "uri" :
                format!("data:application/octet-stream;base64,{}", buffer_base64)
                }
            ));
            gltf_json["bufferViews"].as_array_mut().unwrap().push(serde_json::json!(
                { "buffer" : body_index, "byteOffset" : 0, "byteLength" :
                buffer_data.len() }
            ));
            gltf_json["scenes"][0]["extensions"]["MC_scene_description"]["smpl_bodies"]
                .as_array_mut()
                .unwrap()
                .push(serde_json::json!(
                    { "frame_presence" : smpl_body.frame_presence, "bufferView" :
                    body_index }
                ));
        }
    }
    /// Add camera animation to the Mcs GLTF JSON
    #[allow(clippy::too_many_lines)]
    pub fn add_camera_animation(&self, gltf_json: &mut Value) {
        let buffers_start_idx = self.smpl_bodies.len();
        let num_frames = self.num_frames;
        let fps = self.frame_rate.unwrap_or(30.0);
        #[allow(clippy::cast_precision_loss)]
        let times: Vec<f32> = (0..num_frames).map(|i| i as f32 / fps).collect();
        let time_bytes = times.iter().flat_map(|f| f.to_le_bytes()).collect::<Vec<u8>>();
        let camera_positions = self.smpl_camera.as_ref().unwrap().transform_sequence.translations.clone();
        let camera_rotations = self.smpl_camera.as_ref().unwrap().transform_sequence.get_rotations_as_quaternions();
        if camera_positions.dim().0 == 1 {
            if let Some(node) = gltf_json["nodes"].as_array_mut().and_then(|nodes| nodes.get_mut(1)) {
                node["translation"] = serde_json::json!([camera_positions[[0, 0]], camera_positions[[0, 1]], camera_positions[[0, 2]]]);
                node["rotation"] = serde_json::json!([
                    camera_rotations[[0, 0]],
                    camera_rotations[[0, 1]],
                    camera_rotations[[0, 2]],
                    camera_rotations[[0, 3]]
                ]);
            }
            return;
        }
        let translation_bytes = camera_positions.iter().flat_map(|f| f.to_le_bytes()).collect::<Vec<u8>>();
        let rotation_bytes = camera_rotations.iter().flat_map(|f| f.to_le_bytes()).collect::<Vec<u8>>();
        gltf_json["buffers"].as_array_mut().unwrap().extend([
            serde_json::json!(
                { "byteLength" : time_bytes.len(), "uri" :
                format!("data:application/octet-stream;base64,{}", base64::encode(&
                time_bytes)) }
            ),
            serde_json::json!(
                { "byteLength" : translation_bytes.len(), "uri" :
                format!("data:application/octet-stream;base64,{}", base64::encode(&
                translation_bytes)) }
            ),
            serde_json::json!(
                { "byteLength" : rotation_bytes.len(), "uri" :
                format!("data:application/octet-stream;base64,{}", base64::encode(&
                rotation_bytes)) }
            ),
        ]);
        gltf_json["bufferViews"].as_array_mut().unwrap().extend([
            serde_json::json!(
                { "name" : "TimeBufferView", "buffer" : buffers_start_idx,
                "byteOffset" : 0, "byteLength" : time_bytes.len() }
            ),
            serde_json::json!(
                { "name" : "camera_track_translations_buffer_view", "buffer" :
                buffers_start_idx + 1, "byteOffset" : 0, "byteLength" :
                translation_bytes.len() }
            ),
            serde_json::json!(
                { "name" : "camera_track_rotations_buffer_view", "buffer" :
                buffers_start_idx + 2, "byteOffset" : 0, "byteLength" :
                rotation_bytes.len() }
            ),
        ]);
        let buffer_views_len = gltf_json["bufferViews"].as_array().unwrap().len();
        gltf_json["accessors"].as_array_mut().unwrap().extend([
            serde_json::json!(
                { "name" : "TimeAccessor", "bufferView" : buffer_views_len - 3,
                "componentType" : 5126, "count" : num_frames, "type" : "SCALAR",
                "min" : [times[0]], "max" : [times[num_frames - 1]] }
            ),
            serde_json::json!(
                { "name" : "camera_track_translations_accessor", "bufferView" :
                buffer_views_len - 2, "componentType" : 5126, "count" : num_frames,
                "type" : "VEC3" }
            ),
            serde_json::json!(
                { "name" : "camera_track_rotations_accessor", "bufferView" :
                buffer_views_len - 1, "componentType" : 5126, "count" : num_frames,
                "type" : "VEC4" }
            ),
        ]);
        let accessors_len = gltf_json["accessors"].as_array().unwrap().len();
        gltf_json["animations"].as_array_mut().unwrap().push(serde_json::json!(
            { "channels" : [{ "sampler" : 0, "target" : { "node" : 1, "path" :
            "translation" } }, { "sampler" : 1, "target" : { "node" : 1, "path" :
            "rotation" } }], "samplers" : [{ "input" : accessors_len - 3,
            "interpolation" : "LINEAR", "output" : accessors_len - 2 }, { "input"
            : accessors_len - 3, "interpolation" : "LINEAR", "output" :
            accessors_len - 1 }] }
        ));
    }
    /// Convert `McsCodec` to GLTF JSON string
    pub fn to_gltf_json(&self) -> String {
        let mut gltf_json = self.create_gltf_structure();
        self.add_smpl_buffers_to_gltf(&mut gltf_json);
        if self.smpl_camera.is_some() {
            self.add_camera_animation(&mut gltf_json);
        }
        serde_json::to_string_pretty(&gltf_json).unwrap()
    }
}
