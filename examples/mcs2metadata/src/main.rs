use clap::Parser;
use gloss_renderer::{config::LogLevel, gloss_setup_logger};
use ndarray::s;
use num_traits::FromPrimitive;
use serde::{Deserialize, Serialize};
use smpl_core::codec::scene::McsCodec;
use smpl_core::common::types::{Gender, SmplType};
use smpl_utils::numerical::extract_extrinsics_from_rot_trans;
use std::path::Path;
#[derive(Parser, Debug)]
#[command(
    version,
    about,
    long_about = "Binary that takes an .mcs (Meshcapade Scene) file as input and prints out a JSON displaying metadata about the scene"
)]
struct Args {
    /// Input file. MUST be an mcs file.
    #[arg(short, long)]
    input: String,
    /// Show camera extrinsics for all frames (can be very large output)
    #[arg(long)]
    show_extrinsics: bool,
}
#[derive(Serialize, Deserialize)]
struct Metadata {
    file: String,
    metadata: SceneMetadata,
}
#[derive(Serialize, Deserialize)]
struct SceneMetadata {
    num_bodies: usize,
    num_frames: usize,
    frame_rate: Option<f32>,
    bodies: Vec<BodyMetadata>,
    camera: CameraMetadata,
}
#[derive(Serialize, Deserialize)]
struct BodyMetadata {
    body_id: usize,
    gender: String,
    #[serde(rename = "type")]
    smpl_type: String,
    frame_presence: FramePresence,
}
#[derive(Serialize, Deserialize)]
struct FramePresence {
    start_frame: usize,
    end_frame: usize,
    total_frames: usize,
}
#[derive(Serialize, Deserialize)]
struct CameraMetadata {
    intrinsics: CameraIntrinsics,
    #[serde(skip_serializing_if = "Option::is_none")]
    extrinsics: Option<Vec<FrameExtrinsics>>,
}
#[derive(Serialize, Deserialize)]
struct CameraIntrinsics {
    yfov: f32,
    znear: f32,
    aspect_ratio: Option<f32>,
}
#[derive(Serialize, Deserialize)]
struct FrameExtrinsics {
    frame: usize,
    extrinsics: [[f32; 4]; 4],
}
fn main() {
    gloss_setup_logger(LogLevel::Info, None);
    let args = Args::parse();
    let input_path = Path::new(&args.input);
    assert!(input_path.exists(), "Input file does not exist: {:?}", input_path);
    let input_extension = input_path.extension().and_then(|ext| ext.to_str()).unwrap_or("");
    assert!(
        input_extension.eq("mcs"),
        "Input file must have extension `.mcs`, found: {:?}",
        input_extension
    );
    let mcs_codec = McsCodec::from_file(input_path.to_str().unwrap());
    mcs_codec.to_file("data/mcs_out_check.mcs");
    let bodies: Vec<BodyMetadata> = mcs_codec
        .smpl_bodies
        .iter()
        .enumerate()
        .map(|(body_id, smpl_body)| BodyMetadata {
            body_id,
            gender: Gender::from_i32(smpl_body.codec.gender)
                .map(|g| g.to_string().to_lowercase())
                .unwrap_or("unknown".to_string()),
            smpl_type: SmplType::from_i32(smpl_body.codec.smpl_version)
                .map(|s| s.to_string().to_lowercase())
                .unwrap_or("unknown".to_string()),
            frame_presence: FramePresence {
                start_frame: smpl_body.frame_presence[0],
                end_frame: smpl_body.frame_presence[1],
                total_frames: smpl_body.frame_presence[1] - smpl_body.frame_presence[0],
            },
        })
        .collect();
    let camera_track = mcs_codec.camera_track.as_ref().unwrap();
    let intrinsics = CameraIntrinsics {
        yfov: camera_track.yfov,
        znear: camera_track.znear,
        aspect_ratio: camera_track.aspect_ratio,
    };
    let extrinsics = if args.show_extrinsics {
        match (&camera_track.per_frame_translations, &camera_track.per_frame_rotations) {
            (Some(translations), Some(rotations)) => {
                let extrinsics_3d = extract_extrinsics_from_rot_trans(translations, rotations);
                let num_frames = extrinsics_3d.shape()[0];
                Some(
                    (0..num_frames)
                        .map(|frame| {
                            let extrinsic = extrinsics_3d.slice(s![frame, .., ..]);
                            let matrix: [[f32; 4]; 4] = std::array::from_fn(|i| std::array::from_fn(|j| extrinsic[[i, j]]));
                            FrameExtrinsics { frame, extrinsics: matrix }
                        })
                        .collect(),
                )
            }
            _ => None,
        }
    } else {
        None
    };
    let camera = CameraMetadata { intrinsics, extrinsics };
    let metadata = Metadata {
        file: input_path.file_name().and_then(|name| name.to_str()).unwrap_or("unknown").to_string(),
        metadata: SceneMetadata {
            num_bodies: mcs_codec.smpl_bodies.len(),
            num_frames: mcs_codec.num_frames,
            frame_rate: mcs_codec.frame_rate,
            bodies,
            camera,
        },
    };
    match serde_json::to_string_pretty(&metadata) {
        Ok(json) => println!("{}", json),
        Err(e) => eprintln!("Failed to serialize metadata to JSON: {}", e),
    }
}
