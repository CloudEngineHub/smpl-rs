use crate::{codec::SmplCodecGloss, components::GlossInterop};
use gloss_hecs::EntityBuilder;
use gloss_renderer::{
    components::{Name, VisMesh},
    scene::Scene,
};
use log::info;
use ndarray::s;
use smpl_core::{
    codec::{
        codec::SmplCodec,
        scene::{CameraTrack, McsCodec, SmplBody},
    },
    common::{
        animation::{AnimWrap, Animation, AnimationConfig, AnimationRunner},
        betas::Betas,
        pose::Pose,
        pose_override::PoseOverride,
        smpl_params::SmplParams,
    },
};
use smpl_utils::numerical::hex_to_rgb_f32;
use std::time::Duration;
const COLOR_CODES: [&str; 4] = ["#63D4BF", "#BAC2F7", "#FFEF9E", "#72B0C5"];
/// creates a ``Vec<gloss_hecs::EntityBuilder>`` from the ``McsCodec``
pub trait McsCodecGloss {
    fn from_scene(scene: &Scene) -> Self;
    fn to_entity_builders(&mut self) -> Vec<EntityBuilder>;
}
/// Trait implementation for `McsCodec`
impl McsCodecGloss for McsCodec {
    fn from_scene(scene: &Scene) -> Self {
        let mut smpl_bodies = Vec::new();
        let mut camera_track_query = scene.world.query::<&CameraTrack>();
        let camera_track = camera_track_query.iter().next().map(|c| c.1.clone());
        let (num_frames, frame_rate) = if let Ok(scene_animation) = scene.get_resource::<&SceneAnimation>() {
            (scene_animation.num_frames, Some(scene_animation.config.fps))
        } else {
            (1, None)
        };
        let mut query_state = scene.world.query::<&SmplParams>();
        for (entity, _) in query_state.iter() {
            let smpl_codec = SmplCodec::from_entity(&entity, scene, None);
            let smpl_interval = if let Ok(animation) = scene.get_comp::<&Animation>(&entity) {
                let animation_num_frames = animation.num_animation_frames();
                let animation_start_offset = animation.start_offset;
                [animation_start_offset, animation_start_offset + animation_num_frames].to_vec()
            } else {
                [0, 1].to_vec()
            };
            let smpl_body = SmplBody {
                frame_presence: smpl_interval,
                codec: smpl_codec,
            };
            smpl_bodies.push(smpl_body);
        }
        Self {
            num_frames,
            frame_rate,
            smpl_bodies,
            camera_track,
        }
    }
    #[allow(clippy::cast_precision_loss)]
    #[allow(clippy::cast_sign_loss)]
    fn to_entity_builders(&mut self) -> Vec<EntityBuilder> {
        let mut builders: Vec<EntityBuilder> = Vec::new();
        let mut camera_num_frames = 1;
        if let Some(camera_track) = &self.camera_track {
            let mut camera_builder = EntityBuilder::new();
            camera_num_frames = camera_track.per_frame_translations.as_ref().unwrap().shape()[0];
            camera_builder.add(Name("TrackedCamera".to_string()));
            camera_builder.add(camera_track.clone());
            builders.push(camera_builder);
        }
        for (i, smpl_body) in self.smpl_bodies.iter().enumerate() {
            let smpl_num_frames = smpl_body.codec.frame_count as usize;
            self.frame_rate = smpl_body.codec.frame_rate;
            let is_static = smpl_num_frames == 1;
            let start_offset = smpl_body.frame_presence[0];
            assert_eq!(
                start_offset + smpl_num_frames,
                camera_num_frames,
                "The number of frames in the smpl and camera tracks must be the same"
            );
            let color = hex_to_rgb_f32(COLOR_CODES[i % COLOR_CODES.len()]);
            let mut builder = EntityBuilder::new();
            let smpl_params = SmplParams::new_from_smpl_codec(&smpl_body.codec);
            if is_static {
                if let Some(pose) = Pose::new_from_smpl_codec(&smpl_body.codec) {
                    builder.add(pose);
                }
            } else if let Some(mut anim) = Animation::new_from_smpl_codec(&smpl_body.codec, AnimWrap::Clamp) {
                anim.start_offset = start_offset;
                builder.add(anim);
            }
            info!("Found smpl_params in the .smpl file");
            builder.add(smpl_params);
            if let Some(mut betas) = Betas::new_from_smpl_codec(&smpl_body.codec) {
                info!("Found betas in the .smpl file");
                let trimmed_betas = betas.betas.slice(s![..10]).to_owned();
                betas.betas = trimmed_betas;
                builder.add(betas);
            }
            let pose_override = PoseOverride::allow_all();
            builder.add(pose_override);
            builder.add(GlossInterop { with_uv: true });
            builder.add(VisMesh {
                solid_color: nalgebra::Vector4::<f32>::new(color.0, color.1, color.2, 1.0),
                ..Default::default()
            });
            builders.push(builder);
        }
        builders
    }
}
#[derive(Default)]
pub struct SceneAnimation {
    pub num_frames: usize,
    pub runner: AnimationRunner,
    pub config: AnimationConfig,
}
impl SceneAnimation {
    pub fn new(num_frames: usize) -> Self {
        Self {
            num_frames,
            runner: AnimationRunner::default(),
            config: AnimationConfig::default(),
        }
    }
    pub fn new_with_fps(num_frames: usize, fps: f32) -> Self {
        Self {
            num_frames,
            runner: AnimationRunner::default(),
            config: AnimationConfig { fps, ..Default::default() },
        }
    }
    pub fn new_with_config(num_frames: usize, config: AnimationConfig) -> Self {
        Self {
            num_frames,
            runner: AnimationRunner::default(),
            config,
        }
    }
    #[allow(clippy::cast_precision_loss)]
    pub fn duration(&self) -> Duration {
        Duration::from_secs_f32(self.num_frames as f32 / self.config.fps)
    }
    pub fn get_cur_time(&self) -> Duration {
        self.runner.anim_current_time
    }
    pub fn set_cur_time_as_sec(&mut self, time_sec: f32) {
        self.runner.anim_current_time = Duration::from_secs_f32(time_sec);
    }
    pub fn is_finished(&self) -> bool {
        self.config.wrap_behaviour == AnimWrap::Clamp && self.runner.anim_current_time >= self.duration()
    }
    /// Advances the animation by the amount of time elapsed since last time we
    /// got the current pose
    pub fn advance(&mut self, dt_raw: Duration, first_time: bool) {
        let duration = self.duration();
        let runner = &mut self.runner;
        let config = &self.config;
        let mut dt = dt_raw;
        if first_time {
            dt = Duration::ZERO;
        }
        let will_overflow = runner.anim_current_time + dt > duration;
        let will_underflow = runner.anim_current_time < dt && runner.anim_reversed;
        if will_overflow || will_underflow {
            if will_overflow {
                match config.wrap_behaviour {
                    AnimWrap::Clamp => {
                        dt = Duration::ZERO;
                        runner.anim_current_time = duration;
                    }
                    AnimWrap::Loop => {
                        dt = Duration::from_secs_f64(dt.as_secs_f64() % duration.as_secs_f64());
                        runner.anim_current_time = Duration::ZERO;
                        runner.nr_repetitions += 1;
                    }
                    AnimWrap::Reverse => {
                        dt = Duration::from_secs_f64(dt.as_secs_f64() % duration.as_secs_f64());
                        runner.anim_current_time = duration;
                        runner.anim_reversed = !runner.anim_reversed;
                        runner.nr_repetitions += 1;
                    }
                }
            } else {
                match config.wrap_behaviour {
                    AnimWrap::Clamp => {
                        dt = Duration::ZERO;
                        runner.anim_current_time = Duration::ZERO;
                    }
                    AnimWrap::Loop => {
                        dt = Duration::from_secs_f64(dt.as_secs_f64() % duration.as_secs_f64());
                        runner.anim_current_time = duration;
                        runner.nr_repetitions += 1;
                    }
                    AnimWrap::Reverse => {
                        dt = Duration::from_secs_f64(dt.as_secs_f64() % duration.as_secs_f64());
                        runner.anim_reversed = !runner.anim_reversed;
                        runner.nr_repetitions += 1;
                    }
                }
            }
        }
        if runner.anim_reversed {
            runner.anim_current_time = runner.anim_current_time.saturating_sub(dt);
        } else {
            runner.anim_current_time = runner.anim_current_time.saturating_add(dt);
        }
    }
}
