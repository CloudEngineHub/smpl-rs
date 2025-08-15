use clap::Parser;
use gloss_renderer::viewer_dummy::ViewerDummy;
use gloss_renderer::{config::LogLevel, gloss_setup_logger};
use smpl_core::codec::codec::SmplCodec;
use smpl_core::codec::gltf::GltfCodec;
use smpl_core::codec::scene::McsCodec;
use smpl_core::common::animation::{AnimWrap, AnimationConfig};
use smpl_core::common::smpl_options::SmplOptions;
use smpl_core::common::types::{FaceType, GltfCompatibilityMode, GltfOutputType};
use smpl_core::common::{
    betas::Betas,
    smpl_model::SmplCacheDynamic,
    types::{Gender, SmplType},
};
use smpl_gloss_integration::{
    codec::SmplCodecGloss,
    components::GlossInterop,
    gltf::GltfCodecGloss,
    plugin::SmplPlugin,
    scene::{McsCodecGloss, SceneAnimation},
};
use std::path::Path;
#[derive(Parser, Debug)]
#[command(version, about, long_about = "Binary for GPU free GLB export from smpl or mcs files")]
struct Args {
    /// Input file. MUST be a smpl or mcs file.
    #[arg(short, long)]
    input: String,
    /// Output file. MUST have extension .glb.
    #[arg(short, long)]
    output: String,
    /// Compatibility mode. 0: Smpl, 1: Unreal
    #[arg(short, long)]
    compatibility_mode: u8,
    /// Path to the model data root. We assume a pattern of <model_data_root>/<gender>/SMPLX_<gender>_array_f32_slim.npz
    #[arg(short, long)]
    model_data: String,
}
fn main() {
    gloss_setup_logger(LogLevel::Info, None);
    let args = Args::parse();
    let input_path = Path::new(&args.input);
    let invalid_extension_message = "File has an invalid extension. Extension is considered invalid if:
    - there is no file name
    - there is no embedded .
    - the file name begins with . and has no other .s within";
    let input_extension = input_path.extension().expect(invalid_extension_message).to_str().unwrap();
    let output_path = Path::new(&args.output);
    let model_data_root = Path::new(&args.model_data);
    let compatibility_mode = match args.compatibility_mode {
        0 => GltfCompatibilityMode::Smpl,
        1 => GltfCompatibilityMode::Unreal,
        _ => panic!("Invalid compatibility mode! Must be 0 (Smpl) or 1 (Unreal)."),
    };
    let male_model_path = model_data_root.join("male/SMPLX_male_array_f32_slim.npz");
    let female_model_path = model_data_root.join("female/SMPLX_female_array_f32_slim.npz");
    let neutral_model_path = model_data_root.join("neutral/SMPLX_neutral_array_f32_slim.npz");
    assert!(input_path.exists(), "Input file does not exist: {:?}", input_path);
    assert!(
        output_path.extension().unwrap().to_str().unwrap().eq("glb"),
        "Output file must have extension .glb: {:?}",
        output_path.extension().unwrap().to_str().unwrap()
    );
    println!("- Found valid {:?} file at {:?}", input_extension, input_path);
    println!("- GLB will be saved to: {:?}", output_path);
    println!("- Compatibility mode: {:?}", compatibility_mode);
    let mut viewer = ViewerDummy::new(None);
    let mut smpl_models = SmplCacheDynamic::default();
    match input_extension {
        "mcs" => {
            assert!(neutral_model_path.exists(), "Neutral model data not found at {:?}", neutral_model_path);
            smpl_models.set_lazy_loading(SmplType::SmplX, Gender::Neutral, neutral_model_path.to_str().unwrap());
            let mut mcs_codec = McsCodec::from_file(input_path.to_str().unwrap());
            let builders = mcs_codec.to_entity_builders();
            for mut builder in builders {
                if !builder.has::<Betas>() {
                    builder.add(Betas::default());
                }
                let gloss_interop = GlossInterop::default();
                let smpl_options = SmplOptions::default();
                let name = viewer.scene.get_unused_name();
                viewer
                    .scene
                    .get_or_create_entity(&name)
                    .insert_builder(builder)
                    .insert(gloss_interop)
                    .insert(smpl_options);
            }
            if let Some(frame_rate) = mcs_codec.frame_rate {
                let config = AnimationConfig {
                    fps: frame_rate,
                    wrap_behaviour: AnimWrap::Loop,
                    ..Default::default()
                };
                let smpl_scene = SceneAnimation::new_with_config(mcs_codec.num_frames, config);
                viewer.scene.add_resource(smpl_scene);
            }
        }
        "smpl" => {
            assert!(male_model_path.exists(), "Male model data not found at {:?}", male_model_path);
            assert!(female_model_path.exists(), "Female model data not found at {:?}", female_model_path);
            assert!(neutral_model_path.exists(), "Neutral model data not found at {:?}", neutral_model_path);
            smpl_models.set_lazy_loading(SmplType::SmplX, Gender::Neutral, neutral_model_path.to_str().unwrap());
            smpl_models.set_lazy_loading(SmplType::SmplX, Gender::Female, female_model_path.to_str().unwrap());
            smpl_models.set_lazy_loading(SmplType::SmplX, Gender::Male, male_model_path.to_str().unwrap());
            let codec = SmplCodec::from_file(input_path.to_str().unwrap());
            let mut entity = viewer.scene.get_or_create_entity("mesh_smpl");
            let smpl_options = SmplOptions::default();
            entity
                .insert_builder(codec.to_entity_builder())
                .insert(GlossInterop::default())
                .insert(smpl_options);
        }
        _ => {
            panic!("Invalid input file extension: {:?}", input_extension);
        }
    }
    viewer.scene.add_resource(smpl_models);
    viewer.insert_plugin(&SmplPlugin::new(false));
    viewer.run_manual_plugins();
    let mut gltf_codec = GltfCodec::from_scene(&viewer.scene, None, true);
    gltf_codec.to_file(
        "Meshcapade Scene",
        output_path.to_str().unwrap(),
        GltfOutputType::Binary,
        compatibility_mode,
        FaceType::SmplX,
    );
}
