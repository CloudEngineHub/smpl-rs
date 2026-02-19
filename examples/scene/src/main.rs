use gloss_burn_multibackend::global_backend::init_global_burn_backend;
use gloss_burn_multibackend::global_backend::GlobalBackend;
use gloss_renderer::viewer::Viewer;
use gloss_renderer::{config::LogLevel, gloss_setup_logger};
use smpl_core::codec::scene::McsCodec;
use smpl_core::common::animation::{AnimWrap, AnimationConfig};
use smpl_core::common::smpl_model::SmplCache;
use smpl_gloss_integration::{
    plugin::SmplPlugin,
    scene::{McsCodecGloss, SceneAnimation},
};
use std::path::Path;
fn main() {
    gloss_setup_logger(LogLevel::Info, None);
    init_global_burn_backend(GlobalBackend::Candle);
    let config_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("config/config.toml");
    let mut viewer = Viewer::new(config_path.to_str());
    let mut smpl_models = SmplCache::default();
    smpl_models.lazy_load_defaults();
    let scene_path = "data/mcs/football.mcs";
    let mut mcs_codec = McsCodec::from_file(scene_path);
    mcs_codec.insert_into_scene(viewer.scene_mut(), true);
    if let Some(frame_rate) = mcs_codec.frame_rate {
        let config = AnimationConfig {
            fps: frame_rate,
            wrap_behaviour: AnimWrap::Loop,
            ..Default::default()
        };
        let smpl_scene = SceneAnimation::new_with_config(mcs_codec.num_frames, config);
        viewer.scene_mut().add_resource(smpl_scene);
    }
    viewer.scene_mut().add_resource(smpl_models);
    viewer.insert_plugin(&SmplPlugin::new(true));
    viewer.run();
}
