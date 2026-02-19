/// Example: Export an MCS animation file to GLB
use gloss_renderer::viewer_dummy::ViewerDummy;
use gloss_renderer::{config::LogLevel, gloss_setup_logger};
use smpl_core::codec::gltf::{GltfCodec, GltfExportOptions};
use smpl_core::codec::scene::McsCodec;
use smpl_core::common::animation::{AnimWrap, AnimationConfig};
use smpl_core::common::smpl_model::SmplCache;
use smpl_core::common::types::GltfCompatibilityMode;
use smpl_gloss_integration::gltf::GltfInteropOptions;
use smpl_gloss_integration::{
    gltf::GltfCodecGloss,
    plugin::SmplPlugin,
    scene::{McsCodecGloss, SceneAnimation},
};
fn main() {
    gloss_setup_logger(LogLevel::Info, None);
    let mut viewer = ViewerDummy::new(None);
    let mut smpl_models = SmplCache::default();
    smpl_models.lazy_load_defaults();
    let mcs_path = "data/mcs/skate_04.mcs";
    let mut mcs_codec = McsCodec::from_file(mcs_path);
    println!("MCS info:");
    println!("  frames : {}", mcs_codec.num_frames);
    println!("  bodies : {}", mcs_codec.smpl_bodies.len());
    println!("  fps    : {:?}", mcs_codec.frame_rate);
    mcs_codec.insert_into_scene(&mut viewer.scene, true);
    if let Some(frame_rate) = mcs_codec.frame_rate {
        let config = AnimationConfig {
            fps: frame_rate,
            wrap_behaviour: AnimWrap::Loop,
            ..Default::default()
        };
        let scene_anim = SceneAnimation::new_with_config(mcs_codec.num_frames, config);
        viewer.scene.add_resource(scene_anim);
    }
    viewer.scene.add_resource(smpl_models);
    viewer.insert_plugin(&SmplPlugin::new(false));
    viewer.run_manual_plugins();
    let output_path = "saved/saved_mcs_export.glb";
    let mut gltf_codec = GltfCodec::from_scene(&viewer.scene, &GltfInteropOptions::default());
    gltf_codec.to_file(
        "Meshcapade Scene",
        output_path,
        &GltfExportOptions {
            compatibility_mode: GltfCompatibilityMode::Smpl,
            ..Default::default()
        },
    );
    println!("Exported GLB to {output_path}");
}
