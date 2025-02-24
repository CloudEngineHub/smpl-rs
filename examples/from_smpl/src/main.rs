use gloss_renderer::{
    components::{DiffuseImg, ImgConfig, NormalImg, RoughnessImg},
    viewer::Viewer,
};

use smpl_gloss_integration::{codec::SmplCodecGloss, components::GlossInterop, plugin::SmplPlugin};

use smpl_rs::common::smpl_model::SmplCacheDynamic;

use gloss_renderer::{config::LogLevel, gloss_setup_logger};
use smpl_rs::codec::codec::SmplCodec;
use std::path::Path;

fn main() {
    gloss_setup_logger(LogLevel::Info, None); // Call only once per process
    let config_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("config/config.toml");
    let mut viewer = Viewer::new(config_path.to_str());

    // TODO: This is where we will either choose a backend or autochoose based on
    // certain conditions TODO: Native: Wgpu if GPU found; Candle if not
    // TODO: Web: Wgpu if WebGPU enabled; Candle if not
    // Currently defaults to Candle
    let mut smpl_models = SmplCacheDynamic::default();
    smpl_models.lazy_load_defaults();

    // Path from where to load entity-specific data
    let path_diffuse = "./data/smplx/female_alb_2.png";
    let path_normal = "./data/smplx/female_nrm.png";
    let path_roughness = "./data/smplx/texture_f_r.png";

    let path_codec = "./data/smplx/squat_ow.smpl";
    let codec = SmplCodec::from_file(path_codec);

    // Create the smpl entity with all the components defined by the Codec and additional components for textures,
    let mut entity = viewer.scene.get_or_create_entity("mesh_smpl");
    entity
        .insert_builder(codec.to_entity_builder())
        .insert(DiffuseImg::new_from_path(path_diffuse, &ImgConfig::default()))
        .insert(NormalImg::new_from_path(path_normal, &ImgConfig::default()))
        .insert(RoughnessImg::new_from_path(path_roughness, &ImgConfig::default()))
        .insert(GlossInterop { with_uv: true });

    // Resources shared between all entities
    viewer.scene.add_resource(smpl_models);

    // Plugins which contains all the systems (represent the logic that will run on
    // every frame depending on the entity components)
    viewer.insert_plugin(&SmplPlugin::new(true));
    viewer.run();
}
