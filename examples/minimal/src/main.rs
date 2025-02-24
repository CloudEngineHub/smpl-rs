use gloss_renderer::{
    components::{DiffuseImg, ImgConfig, NormalImg, RoughnessImg},
    viewer::Viewer,
};

use smpl_gloss_integration::{components::GlossInterop, plugin::SmplPlugin};

use smpl_rs::common::{
    animation::{AnimWrap, Animation, AnimationConfig},
    betas::Betas,
    pose_hands::HandType,
    pose_override::PoseOverride,
    pose_parts::PosePart,
    smpl_model::SmplCacheDynamic,
    smpl_params::SmplParams,
    types::{Gender, SmplType, UpAxis},
};

use gloss_renderer::{config::LogLevel, gloss_setup_logger};
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

    // Lazy loads default models for SmplX
    smpl_models.lazy_load_defaults();

    // path from where to load entity-specific data
    let path_diffuse = "./data/smplx/female_alb_2.png";
    let path_normal = "./data/smplx/female_nrm.png";
    let path_roughness = "./data/smplx/texture_f_r.png";
    let path_anim = "./data/smplx/apose_to_catwalk_001.npz";

    //create the smpl entity itself with all the components defining shape, pose,
    // textures, etc
    let entity = viewer.scene.get_or_create_entity("mesh_smpl").entity();
    viewer
        .scene
        .world
        .insert(
            entity,
            (
                SmplParams::new(SmplType::SmplX, Gender::Female, true),
                Betas::new_empty(10),
                // Pose::new_empty(UpAxis::Y, SmplType::SmplX),
                Animation::new_from_npz(
                    path_anim,
                    AnimationConfig {
                        fps: 45.0,
                        wrap_behaviour: AnimWrap::Loop,
                        smpl_type: SmplType::SmplH,
                        up_axis: UpAxis::Y,
                        ..Default::default()
                    },
                ),
                PoseOverride::allow_all()
                    .deny(PosePart::Jaw)
                    .deny(PosePart::LeftEye)
                    .deny(PosePart::RightEye)
                    .overwrite_hands(HandType::Relaxed)
                    .build(),
                DiffuseImg::new_from_path(path_diffuse, &ImgConfig::default()),
                NormalImg::new_from_path(path_normal, &ImgConfig::default()),
                RoughnessImg::new_from_path(path_roughness, &ImgConfig::default()),
                GlossInterop { with_uv: true },
            ),
        )
        .unwrap();

    //resources shared between all entities
    viewer.scene.add_resource(smpl_models);

    //plugins which contains all the systems (represent the logic that will run on
    // every frame depending on the entity components)
    viewer.insert_plugin(&SmplPlugin::new(true));

    viewer.run();
}
