use gloss_renderer::viewer::Viewer;
use gloss_renderer::{config::LogLevel, gloss_setup_logger};
use smpl_gloss_integration::{
    components::GlossInterop,
    plugin::SmplPlugin,
    scene::{McsCodecGloss, SceneAnimation},
};
use smpl_rs::codec::scene::McsCodec;
use smpl_rs::common::animation::{AnimWrap, AnimationConfig};
use smpl_rs::common::{
    betas::Betas,
    smpl_model::SmplCacheDynamic,
    types::{Gender, SmplType},
};
use std::path::Path;
fn main() {
    gloss_setup_logger(LogLevel::Info, None);
    let config_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("config/config.toml");
    let mut viewer = Viewer::new(config_path.to_str());
    let mut smpl_models = SmplCacheDynamic::default();
    smpl_models.set_lazy_loading(SmplType::SmplX, Gender::Neutral, "./data/smplx/SMPLX_neutral_array_f32_slim.npz");
    let scene_path = "./data/mcs/boxing.mcs";
    let mut mcs_codec = McsCodec::from_file(scene_path);
    let builders = mcs_codec.to_entity_builders();
    for mut builder in builders {
        if !builder.has::<Betas>() {
            builder.add(Betas::default());
        }
        let gloss_interop = GlossInterop::default();
        let name = viewer.scene.get_unused_name();
        viewer.scene.get_or_create_entity(&name).insert_builder(builder).insert(gloss_interop);
    }
    let config = AnimationConfig {
        fps: mcs_codec.frame_rate,
        wrap_behaviour: AnimWrap::Reverse,
        ..Default::default()
    };
    let smpl_scene = SceneAnimation::new_with_config(mcs_codec.num_frames, config);
    viewer.scene.add_resource(smpl_scene);
    viewer.scene.add_resource(smpl_models);
    viewer.insert_plugin(&SmplPlugin::new(true));
    viewer.run();
}
