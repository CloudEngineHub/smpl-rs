use super::codec::SmplCodec;
use gltf_json::{validation::Checked::Valid, Root};
use ndarray as nd;
use std::{fs::File, io::Read};

/// The ``CameraTrack`` contains the camera track data in the scene
#[derive(Debug, Clone)]
pub struct CameraTrack {
    pub yfov: f32,
    pub znear: f32,
    pub zfar: Option<f32>,
    pub aspect_ratio: Option<f32>,

    pub per_frame_translations: Option<nd::Array2<f32>>, // num_frames x 3
    pub per_frame_rotations: Option<nd::Array2<f32>>,    // num_frames x 4
}

impl Default for CameraTrack {
    fn default() -> Self {
        Self {
            yfov: 1.0,
            znear: 0.1,
            zfar: None,
            aspect_ratio: None,
            per_frame_translations: None,
            per_frame_rotations: None,
        }
    }
}

/// The ``McsCodec`` contains all of the contents of an ``.mcs`` file  
#[derive(Debug, Clone)]
pub struct McsCodec {
    pub num_frames: usize,
    pub frame_rate: f32,
    pub smpl_bodies: Vec<SmplBody>,
    pub camera_track: Option<CameraTrack>,
}

/// ``SmplBody`` holds the contents of the ``.smpl`` file along with the frame presence
#[derive(Debug, Clone)]
pub struct SmplBody {
    pub frame_presence: Vec<usize>,
    pub codec: SmplCodec,
}

/// ``McsCodec`` for ``.mcs`` files
#[allow(clippy::cast_possible_truncation)]
impl McsCodec {
    /// Load `McsCodec` from a GLTF file.
    pub fn from_file(path: &str) -> Self {
        let mut file = File::open(path).expect("Failed to open GLTF file");
        let mut json_data = String::new();
        file.read_to_string(&mut json_data).expect("Failed to read GLTF file");

        // Parse the JSON into a GLTF Root structure
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

                    // Extract `num_frames`
                    if let Some(nf) = extension.get("num_frames").and_then(gltf_json::Value::as_u64) {
                        num_frames = nf as usize;
                    }

                    // Extract `smpl_bodies`
                    if let Some(smpl_bodies_data) = extension.get("smpl_bodies").and_then(|v| v.as_array()) {
                        smpl_bodies = Self::extract_smpl_bodies(gltf, smpl_bodies_data);
                    }
                }
            }
            Self {
                num_frames,
                frame_rate: smpl_bodies.first().and_then(|b| b.codec.frame_rate).unwrap_or(30.0),
                smpl_bodies,
                camera_track: Self::extract_camera_track(gltf),
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
    fn extract_camera_track(gltf: &Root) -> Option<CameraTrack> {
        if let Some(camera) = gltf.cameras.first() {
            let (yfov, znear, zfar, aspect_ratio) = match camera.type_.unwrap() {
                gltf_json::camera::Type::Perspective => (
                    camera.perspective.as_ref().map_or(std::f32::consts::FRAC_PI_2, |p| p.yfov),
                    camera.perspective.as_ref().map_or(0.1, |p| p.znear),
                    camera.perspective.as_ref().and_then(|p| p.zfar),
                    camera.perspective.as_ref().and_then(|p| p.aspect_ratio),
                ),
                gltf_json::camera::Type::Orthographic => panic!("Orthographic camera not supported!"),
            };

            if gltf.animations.is_empty() {
                return Some(CameraTrack {
                    yfov,
                    znear,
                    zfar,
                    aspect_ratio,
                    per_frame_translations: None,
                    per_frame_rotations: None,
                });
            }

            // Look for camera animation data in the animations
            let mut per_frame_translations = None;
            let mut per_frame_rotations = None;

            // Process camera animations
            for animation in &gltf.animations {
                for channel in &animation.channels {
                    let target = &channel.target;

                    // Skip if not targeting a camera node
                    let Some(node) = gltf.nodes.get(target.node.value()) else { continue };
                    if node.camera.is_none() {
                        continue;
                    };

                    // Get animation data; We do not continue if we cannot find the sampler or output accessor
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
                    };

                    // Decode buffer data
                    let encoded_data = uri.split(',').nth(1).expect("Invalid data URI");
                    let buffer_data = base64::decode(encoded_data).expect("Failed to decode Base64 data");

                    // Extract relevant portion
                    let start = buffer_view.byte_offset.map_or(0, |x| x.0 as usize);
                    let length = buffer_view.byte_length.0 as usize;
                    let data = &buffer_data[start..start + length];

                    // Convert to f32 array
                    let floats: Vec<f32> = data.chunks(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();

                    // Store animation data based on path type
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

            Some(CameraTrack {
                yfov,
                znear,
                zfar,
                aspect_ratio,
                per_frame_translations,
                per_frame_rotations,
            })
        } else {
            None
        }
    }
}
