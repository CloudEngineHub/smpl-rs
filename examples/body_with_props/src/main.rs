use gloss_burn_multibackend::global_backend::init_global_burn_backend;
use gloss_burn_multibackend::global_backend::GlobalBackend;
use gloss_renderer::{
    builders::build_from_file,
    components::{DiffuseImg, ImgConfig, NormalImg, RoughnessImg},
    viewer::Viewer,
};
use gloss_renderer::{config::LogLevel, gloss_setup_logger};
use smpl_core::codec::gltf::GltfCodec;
use smpl_core::{codec::codec::SmplCodec, common::transform_sequence::TransformSequence};
use smpl_core::{
    codec::gltf::GltfExportOptions,
    common::{smpl_model::SmplCache, types::GltfCompatibilityMode},
};
use smpl_gloss_integration::gltf::GltfCodecGloss;
use smpl_gloss_integration::{codec::SmplCodecGloss, components::GlossInterop, gltf::GltfInteropOptions, plugin::SmplPlugin};
use std::path::Path;
fn main() {
    gloss_setup_logger(LogLevel::Info, None);
    init_global_burn_backend(GlobalBackend::Candle);
    let config_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("config/config.toml");
    let mut viewer = Viewer::new(config_path.to_str());
    let mut smpl_models = SmplCache::default();
    smpl_models.lazy_load_defaults();
    let path_diffuse = "./data/smplx/female_alb_2.png";
    let path_normal = "./data/smplx/female_nrm.png";
    let path_roughness = "./data/smplx/texture_f_r.png";
    let path_smpl = "./data/props/largebox/codec_spec.smpl";
    let path_glb = "./data/props/largebox/largebox.glb";
    let path_trajectory = "./data/props/largebox/object_spec.npz";
    let mut smpl_ent = viewer.scene_mut().get_or_create_entity("smpl_ent");
    smpl_ent
        .insert_builder(SmplCodec::from_file(path_smpl).to_entity_builder())
        .insert(DiffuseImg::new_from_path(path_diffuse, &ImgConfig::default()))
        .insert(NormalImg::new_from_path(path_normal, &ImgConfig::default()))
        .insert(RoughnessImg::new_from_path(path_roughness, &ImgConfig::default()))
        .insert(GlossInterop { with_uv: true });
    let mut prop_ent = viewer.scene_mut().get_or_create_entity("prop_ent");
    prop_ent
        .insert_builder(build_from_file(path_glb))
        .insert(TransformSequence::new_from_npz(path_trajectory));
    viewer.scene_mut().add_resource(smpl_models);
    viewer.insert_plugin(&SmplPlugin::new(true));
    viewer.start_frame();
    viewer.update();
    let output_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("smpl_with_props.glb");
    let mut gltf_codec = GltfCodec::from_scene(viewer.scene(), &GltfInteropOptions::default());
    gltf_codec.to_file(
        "Meshcapade Scene",
        output_path.to_str().unwrap(),
        &GltfExportOptions {
            compatibility_mode: GltfCompatibilityMode::Unreal,
            ..Default::default()
        },
    );
}
