use crate::components::GlossInterop;
use gloss_hecs::EntityBuilder;
use gloss_renderer::components::{Name, VisMesh};
use log::info;
use ndarray::s;
use smpl_core::{
    codec::scene::McsCodec,
    common::{
        animation::{AnimWrap, Animation, AnimationConfig, AnimationRunner},
        betas::Betas,
        pose_override::PoseOverride,
        smpl_params::SmplParams,
    },
};
use smpl_utils::numerical::hex_to_rgb_f32;
use std::time::Duration;
const COLOR_CODES: [&str; 4] = ["#63D4BF", "#BAC2F7", "#FFEF9E", "#72B0C5"];
/// creates a ``Vec<gloss_hecs::EntityBuilder>`` from the ``McsCodec``
pub trait McsCodecGloss {
    fn to_entity_builders(&mut self) -> Vec<EntityBuilder>;
}
/// Trait implementation for `McsCodec`
impl McsCodecGloss for McsCodec {
    #[allow(clippy::cast_precision_loss)]
    fn to_entity_builders(&mut self) -> Vec<EntityBuilder> {
        let mut builders: Vec<EntityBuilder> = Vec::new();
        if let Some(camera_track) = &self.camera_track {
            let mut camera_builder = EntityBuilder::new();
            camera_builder.add(Name("TrackedCamera".to_string()));
            camera_builder.add(camera_track.clone());
            builders.push(camera_builder);
        }
        for (i, smpl_body) in self.smpl_bodies.iter().enumerate() {
            let start_offset = smpl_body.frame_presence[0];
            let fps = smpl_body.codec.frame_rate.unwrap();
            self.frame_rate = fps;
            let color = hex_to_rgb_f32(COLOR_CODES[i % COLOR_CODES.len()]);
            let mut builder = EntityBuilder::new();
            let smpl_params = SmplParams::new_from_smpl_codec(&smpl_body.codec);
            info!("Found smpl_params in the .smpl file");
            builder.add(smpl_params);
            if let Some(mut betas) = Betas::new_from_smpl_codec(&smpl_body.codec) {
                info!("Found betas in the .smpl file");
                let trimmed_betas = betas.betas.slice(s![..10]).to_owned();
                betas.betas = trimmed_betas;
                builder.add(betas);
            }
            if let Some(mut anim) = Animation::new_from_smpl_codec(&smpl_body.codec, AnimWrap::Clamp) {
                anim.start_offset = start_offset;
                builder.add(anim);
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
