use crate::components::{Follow, FollowParams, Follower, FollowerType, GlossInterop};
use crate::conversions::update_entity_on_backend;
use crate::scene::SceneAnimation;
use crate::{codec::SmplCodecGloss, gltf::GltfCodecGloss};
use burn::backend::{Candle, NdArray, Wgpu};
use burn::{
    prelude::*,
    tensor::{Float, Int, Tensor},
};
use core::f32;
use gloss_hecs::Entity;
use gloss_hecs::{Changed, CommandBuffer};
use gloss_renderer::plugin_manager::gui::{GuiWindow, GuiWindowType};
use gloss_renderer::{
    components::{ConfigChanges, ModelMatrix, PosLookat, Renderable, Verts},
    geom::{Geom, PerVertexNormalsWeightingType},
    plugin_manager::{
        gui::{Checkbox, Selectable, Slider, Widgets},
        Event, RunnerState,
    },
    scene::Scene,
};
use gloss_utils::abi_stable_aliases::std_types::{RNone, ROption, RString, RVec};
use gloss_utils::tensor::BurnBackend;
use gloss_utils::{
    bshare::{ToBurn, ToNalgebraFloat, ToNalgebraInt, ToNdArray},
    nshare::ToNalgebra,
    tensor::{DynamicMatrixOps, DynamicTensorFloat2D, DynamicTensorInt2D},
};
use log::{info, warn};
use nalgebra::{self as na};
use smpl_core::common::animation::AnimationConfig;
use smpl_core::common::smpl_model::{SmplCache, SmplModel};
use smpl_core::common::types::{GltfOutputType, UpAxis};
use smpl_core::common::{
    animation::Animation,
    betas::Betas,
    expression::{Expression, ExpressionOffsets},
    smpl_model::SmplCacheDynamic,
};
use smpl_core::common::{
    outputs::{SmplOutputPoseTDynamic, SmplOutputPosedDynamic},
    pose::Pose,
    pose_corrective::PoseCorrectiveDynamic,
    pose_override::PoseOverride,
    smpl_params::SmplParams,
};
use smpl_core::conversions::pose_remap::PoseRemap;
use smpl_utils::io::FileType;
/// Check all entities with ``SmplParams`` and lazy load the smpl model if
/// needed do it in two stages, first checking if we need to acually load
/// anything and in the second stage we actually load stuff, this is in order to
/// avoid accing the ``SmplModel`` mutable and then triggering all the other
/// systems to run
#[cfg(not(target_arch = "wasm32"))]
pub extern "C" fn smpl_lazy_load_model(scene: &mut Scene, _runner: &mut RunnerState) {
    let mut command_buffer = CommandBuffer::new();
    {
        let mut needs_loading = false;
        let mut query_state = scene.world.query::<&SmplParams>();
        for (_entity, smpl_params) in query_state.iter() {
            let smpl_models = scene.get_resource::<&SmplCacheDynamic>().unwrap();
            if !smpl_models.has_model(smpl_params.smpl_type, smpl_params.gender)
                && smpl_models.has_lazy_loading(smpl_params.smpl_type, smpl_params.gender)
            {
                needs_loading = true;
            }
        }
        if needs_loading {
            let mut query_state = scene.world.query::<&SmplParams>();
            for (_entity, smpl_params) in query_state.iter() {
                let mut smpl_models = scene.get_resource::<&mut SmplCacheDynamic>().unwrap();
                if !smpl_models.has_model(smpl_params.smpl_type, smpl_params.gender)
                    && smpl_models.has_lazy_loading(smpl_params.smpl_type, smpl_params.gender)
                {
                    if let Some(path) = smpl_models.get_lazy_loading(smpl_params.smpl_type, smpl_params.gender).as_ref() {
                        smpl_models.add_model_from_type(smpl_params.smpl_type, path, smpl_params.gender, 300, 100);
                    }
                }
            }
        }
    }
    command_buffer.run_on(&mut scene.world);
}
/// System to add a ``SceneAnimation`` Resource in case it doesn't already exist
pub extern "C" fn smpl_auto_add_scene(scene: &mut Scene, _runner: &mut RunnerState) {
    if !scene.has_resource::<SceneAnimation>() {
        let mut selected_num_frames = 0;
        let mut selected_fps = f32::MAX;
        let mut anim_config = AnimationConfig::default();
        let num_ents: usize;
        {
            let mut query_state = scene.world.query::<&Animation>().with::<&Renderable>();
            num_ents = query_state.iter().len();
            for (_, smpl_anim) in query_state.iter() {
                let last_frame_idx = smpl_anim.num_animation_frames() + smpl_anim.start_offset;
                selected_num_frames = selected_num_frames.max(last_frame_idx);
                selected_fps = selected_fps.min(smpl_anim.config.fps);
                anim_config = smpl_anim.config.clone();
            }
        }
        let scene_anim = match num_ents {
            0 => None,
            1 => Some(SceneAnimation::new_with_config(selected_num_frames, anim_config)),
            _ => Some(SceneAnimation::new_with_fps(selected_num_frames, selected_fps)),
        };
        if let Some(scene_anim) = scene_anim {
            scene.add_resource(scene_anim);
        }
    }
}
/// System to add a ``SmplInterval`` to entities in case it doesn't already exist
pub extern "C" fn smpl_auto_add_follow(scene: &mut Scene, _runner: &mut RunnerState) {
    let mut command_buffer = CommandBuffer::new();
    {
        let Ok(follower) = scene.get_resource::<&Follower>() else {
            return;
        };
        let follow_all = follower.params.follow_all;
        if !follow_all {
            return;
        }
        let mut query_state = scene.world.query::<&GlossInterop>().without::<&Follow>();
        for (entity, _) in query_state.iter() {
            command_buffer.insert_one(entity, Follow);
        }
    }
    command_buffer.run_on(&mut scene.world);
}
/// System to Advance Animation timer
#[allow(clippy::cast_precision_loss)]
pub extern "C" fn smpl_advance_anim(scene: &mut Scene, runner: &mut RunnerState) {
    let mut command_buffer = CommandBuffer::new();
    {
        if !scene.has_resource::<SceneAnimation>() {
            return;
        }
        let mut entities_with_anim = Vec::new();
        {
            let mut query_state = scene.world.query::<(&Animation, Changed<Animation>)>().with::<&SmplParams>();
            for (entity, (smpl_anim, changed_anim)) in query_state.iter() {
                if (!smpl_anim.runner.paused && !smpl_anim.runner.temporary_pause) || changed_anim {
                    entities_with_anim.push(entity);
                }
            }
        }
        let mut entities_within_interval = Vec::new();
        let mut entities_outside_interval = Vec::new();
        {
            let mut scene_anim = scene.get_resource::<&mut SceneAnimation>().unwrap();
            let current_global_time = scene_anim.runner.anim_current_time.as_secs_f32();
            for entity in entities_with_anim.iter() {
                let mut smpl_anim = scene.get_comp::<&mut Animation>(entity).unwrap();
                let global_fps = scene_anim.config.fps;
                smpl_anim.config.fps = global_fps;
                let start_offset = smpl_anim.start_offset;
                let is_within_interval = current_global_time >= (smpl_anim.start_offset as f32 / scene_anim.config.fps)
                    && current_global_time <= ((smpl_anim.start_offset + smpl_anim.num_animation_frames()) as f32 / scene_anim.config.fps);
                if is_within_interval {
                    smpl_anim.set_cur_time_as_sec(current_global_time - (start_offset as f32 / global_fps));
                    entities_within_interval.push(*entity);
                } else {
                    entities_outside_interval.push(*entity);
                    continue;
                }
                if !scene_anim.runner.paused && !scene_anim.runner.temporary_pause {
                    let is_added = smpl_anim.is_added();
                    smpl_anim.advance(runner.dt(), runner.is_first_time() || is_added);
                }
                let anim_frame = smpl_anim.get_current_pose();
                command_buffer.insert_one(*entity, anim_frame);
                if let Some(expression) = smpl_anim.get_current_expression() {
                    command_buffer.insert_one(*entity, expression);
                }
            }
            if !scene_anim.runner.paused && !scene_anim.runner.temporary_pause && scene_anim.num_frames != 0 {
                let is_added = scene_anim.is_added();
                scene_anim.advance(runner.dt(), runner.is_first_time() || is_added);
            }
        }
        for entity in entities_within_interval {
            if !scene.world.has::<Renderable>(entity).unwrap() {
                scene.world.insert_one(entity, Renderable).unwrap();
            }
        }
        for entity in entities_outside_interval {
            if scene.world.has::<Renderable>(entity).unwrap() {
                scene.world.remove_one::<Renderable>(entity).unwrap();
            }
        }
        runner.request_redraw();
    }
    command_buffer.run_on(&mut scene.world);
}
/// System to compute vertices if Betas has changed.
/// This internally uses the generic variant of the function suffixed with
/// ``_on_backend``
#[allow(clippy::similar_names)]
#[allow(clippy::too_many_lines)]
pub extern "C" fn smpl_betas_to_verts(scene: &mut Scene, _runner: &mut RunnerState) {
    let mut command_buffer = CommandBuffer::new();
    {
        let smpl_models_dynamic = scene.get_resource::<&SmplCacheDynamic>().unwrap();
        let changed_models = smpl_models_dynamic.is_changed();
        let mut query_state = scene.world.query::<(&SmplParams, &Betas, Changed<Betas>, Changed<SmplParams>)>();
        for (entity, (smpl_params, smpl_betas, changed_betas, changed_smpl_params)) in query_state.iter() {
            if !changed_betas && !changed_smpl_params && !changed_models {
                continue;
            }
            match &*smpl_models_dynamic {
                SmplCacheDynamic::NdArray(model) => betas_to_verts_on_backend::<NdArray>(&mut command_buffer, entity, smpl_params, smpl_betas, model),
                SmplCacheDynamic::Wgpu(model) => betas_to_verts_on_backend::<Wgpu>(&mut command_buffer, entity, smpl_params, smpl_betas, model),
                SmplCacheDynamic::Candle(model) => betas_to_verts_on_backend::<Candle>(&mut command_buffer, entity, smpl_params, smpl_betas, model),
            }
        }
    }
    command_buffer.run_on(&mut scene.world);
}
/// Function to compute vertices from betas on a generic Burn Backend. We
/// currently support - ``Candle``, ``NdArray``, and ``Wgpu``
#[allow(clippy::too_many_arguments)]
fn betas_to_verts_on_backend<B: Backend>(
    command_buffer: &mut CommandBuffer,
    entity: Entity,
    smpl_params: &SmplParams,
    smpl_betas: &Betas,
    smpl_models: &SmplCache<B>,
) where
    <B as Backend>::FloatTensorPrimitive<2>: Sync,
    <B as Backend>::IntTensorPrimitive<2>: Sync,
    B::QuantizedTensorPrimitive<1>: Sync,
    B::QuantizedTensorPrimitive<2>: Sync,
    B::QuantizedTensorPrimitive<3>: Sync,
{
    let smpl_model = smpl_models.get_model_ref(smpl_params.smpl_type, smpl_params.gender).unwrap();
    let v_burn_merged = smpl_model.betas2verts(smpl_betas);
    let joints_t_pose = smpl_model.verts2joints(v_burn_merged.clone());
    let smpl_output = SmplOutputPoseTDynamic {
        verts: v_burn_merged.clone(),
        verts_without_expression: v_burn_merged,
        joints: joints_t_pose,
    };
    command_buffer.insert_one(entity, smpl_output);
}
/// System to compute Expression offsets if Expression or ``SmplParams`` have
/// changed. This internally uses the generic variant of the function suffixed
/// with ``_on_backend``
#[allow(clippy::similar_names)]
pub extern "C" fn smpl_expression_offsets(scene: &mut Scene, _runner: &mut RunnerState) {
    let mut command_buffer = CommandBuffer::new();
    {
        let smpl_models_dynamic = scene.get_resource::<&SmplCacheDynamic>().unwrap();
        let changed_models = smpl_models_dynamic.is_changed();
        let mut query_state = scene
            .world
            .query::<(&SmplParams, &Expression, Changed<Expression>, Changed<SmplParams>)>();
        for (entity, (smpl_params, expression, changed_expression, changed_smpl_params)) in query_state.iter() {
            if !changed_expression && !changed_smpl_params && !changed_models {
                continue;
            }
            match &*smpl_models_dynamic {
                SmplCacheDynamic::NdArray(smpl_models_ndarray) => {
                    expression_offsets_on_backend::<NdArray>(&mut command_buffer, entity, smpl_params, expression, smpl_models_ndarray);
                }
                SmplCacheDynamic::Wgpu(smpl_models_wgpu) => {
                    expression_offsets_on_backend::<Wgpu>(&mut command_buffer, entity, smpl_params, expression, smpl_models_wgpu);
                }
                SmplCacheDynamic::Candle(smpl_models_candle) => {
                    expression_offsets_on_backend::<Candle>(&mut command_buffer, entity, smpl_params, expression, smpl_models_candle);
                }
            }
        }
    }
    command_buffer.run_on(&mut scene.world);
}
/// Function to compute expression offsets on a generic Burn Backend. We
/// currently support - ``Candle``, ``NdArray``, and ``Wgpu``
#[allow(clippy::too_many_arguments)]
fn expression_offsets_on_backend<B: Backend>(
    command_buffer: &mut CommandBuffer,
    entity: Entity,
    smpl_params: &SmplParams,
    expression: &Expression,
    smpl_models: &SmplCache<B>,
) where
    <B as Backend>::FloatTensorPrimitive<2>: Sync,
    <B as Backend>::IntTensorPrimitive<2>: Sync,
    B::QuantizedTensorPrimitive<1>: Sync,
    B::QuantizedTensorPrimitive<2>: Sync,
    B::QuantizedTensorPrimitive<3>: Sync,
{
    let smpl_model = smpl_models.get_model_ref(smpl_params.smpl_type, smpl_params.gender).unwrap();
    let verts_offsets_merged = smpl_model.expression2offsets(expression);
    let expr_offsets = ExpressionOffsets {
        offsets: verts_offsets_merged,
    };
    command_buffer.insert_one(entity, expr_offsets);
}
/// System to apply the expression offsets.
/// This internally uses the generic variant of the function suffixed with
/// ``_on_backend``
#[allow(clippy::too_many_lines)]
pub extern "C" fn smpl_expression_apply(scene: &mut Scene, _runner: &mut RunnerState) {
    let mut command_buffer = CommandBuffer::new();
    {
        let smpl_models_dynamic = scene.get_resource::<&SmplCacheDynamic>().unwrap();
        match &*smpl_models_dynamic {
            SmplCacheDynamic::NdArray(smpl_models_nd) => {
                apply_expression_on_backend::<NdArray>(scene, &mut command_buffer, smpl_models_nd);
            }
            SmplCacheDynamic::Wgpu(smpl_models_wgpu) => {
                apply_expression_on_backend::<Wgpu>(scene, &mut command_buffer, smpl_models_wgpu);
            }
            SmplCacheDynamic::Candle(smpl_models_cndl) => {
                apply_expression_on_backend::<Candle>(scene, &mut command_buffer, smpl_models_cndl);
            }
        }
    }
    command_buffer.run_on(&mut scene.world);
}
/// Function to apply expression offsets on a generic Burn Backend. We currently
/// support - ``Candle``, ``NdArray``, and ``Wgpu``
#[allow(clippy::too_many_arguments)]
fn apply_expression_on_backend<B: Backend>(scene: &Scene, command_buffer: &mut CommandBuffer, smpl_models: &SmplCache<B>)
where
    <B as Backend>::FloatTensorPrimitive<2>: Sync,
    <B as Backend>::IntTensorPrimitive<2>: Sync,
    B::QuantizedTensorPrimitive<1>: Sync,
    B::QuantizedTensorPrimitive<2>: Sync,
    B::QuantizedTensorPrimitive<3>: Sync,
{
    let mut query_state = scene.world.query::<(
        &SmplParams,
        &mut SmplOutputPoseTDynamic<B>,
        &ExpressionOffsets<B>,
        Changed<SmplOutputPoseTDynamic<B>>,
        Changed<ExpressionOffsets<B>>,
        Changed<SmplParams>,
    )>();
    for (entity, (smpl_params, mut smpl_t_output, expression_offsets, changed_smpl_t, changed_expression, changed_params)) in query_state.iter() {
        if !changed_smpl_t && !changed_expression && !changed_params {
            continue;
        }
        let smpl_model = smpl_models.get_model_ref(smpl_params.smpl_type, smpl_params.gender).unwrap();
        smpl_t_output.verts = smpl_t_output.verts_without_expression.clone() + expression_offsets.offsets.clone();
        smpl_t_output.joints = smpl_model.verts2joints(smpl_t_output.verts.clone());
        command_buffer.insert_one(entity, smpl_t_output.clone());
    }
}
/// System to remap the ``SmplType`` of a pose to the ``SmplType`` found in
/// ``SmplParams``
pub extern "C" fn smpl_pose_remap(scene: &mut Scene, _runner: &mut RunnerState) {
    let mut command_buffer = CommandBuffer::new();
    {
        let mut query_state = scene.world.query::<(&Pose, &SmplParams, Changed<Pose>)>();
        for (entity, (smpl_pose, smpl_params, changed_pose)) in query_state.iter() {
            if !changed_pose {
                continue;
            }
            let pose_remap = PoseRemap::new(smpl_pose.smpl_type, smpl_params.smpl_type);
            let new_pose = pose_remap.remap(smpl_pose);
            command_buffer.insert_one(entity, new_pose);
        }
    }
    command_buffer.run_on(&mut scene.world);
}
/// System for handling pose overrides
pub extern "C" fn smpl_mask_pose(scene: &mut Scene, _runner: &mut RunnerState) {
    let mut command_buffer = CommandBuffer::new();
    {
        let mut query_state = scene
            .world
            .query::<(&mut Pose, &mut PoseOverride, Changed<Pose>, Changed<PoseOverride>)>()
            .with::<&SmplParams>();
        for (_entity, (mut smpl_pose, mut pose_mask, changed_pose, changed_pose_mask)) in query_state.iter() {
            if !changed_pose && !changed_pose_mask {
                continue;
            }
            smpl_pose.apply_mask(&mut pose_mask);
        }
    }
    command_buffer.run_on(&mut scene.world);
}
/// Smpl bodies have to be assigned some pose, so if we have no Pose and no
/// Animation we set a dummy default pose
pub extern "C" fn smpl_make_dummy_pose(scene: &mut Scene, _runner: &mut RunnerState) {
    let mut command_buffer = CommandBuffer::new();
    {
        let mut query_state = scene.world.query::<&SmplParams>().without::<&Pose>().without::<&Animation>();
        for (entity, smpl_params) in query_state.iter() {
            let pose = Pose::new_empty(UpAxis::Y, smpl_params.smpl_type);
            command_buffer.insert_one(entity, pose);
        }
    }
    command_buffer.run_on(&mut scene.world);
}
/// System for computing and applying pose correctives given a pose
/// This internally uses the generic variant of the function suffixed with
/// ``_on_backend``
#[allow(clippy::similar_names)]
#[allow(clippy::too_many_lines)]
pub extern "C" fn smpl_compute_pose_correctives(scene: &mut Scene, _runner: &mut RunnerState) {
    let mut command_buffer = CommandBuffer::new();
    {
        let smpl_models_dynamic = scene.get_resource::<&SmplCacheDynamic>().unwrap();
        let smpl_models_changed = smpl_models_dynamic.is_changed();
        match &*smpl_models_dynamic {
            SmplCacheDynamic::NdArray(smpl_models_ndarray) => {
                compute_pose_correctives_on_backend::<NdArray>(scene, &mut command_buffer, smpl_models_ndarray, smpl_models_changed);
            }
            SmplCacheDynamic::Wgpu(smpl_models_wgpu) => {
                compute_pose_correctives_on_backend::<Wgpu>(scene, &mut command_buffer, smpl_models_wgpu, smpl_models_changed);
            }
            SmplCacheDynamic::Candle(smpl_models_candle) => {
                compute_pose_correctives_on_backend::<Candle>(scene, &mut command_buffer, smpl_models_candle, smpl_models_changed);
            }
        }
    }
    command_buffer.run_on(&mut scene.world);
}
/// Function to compute pose correctives given a pose on a generic Burn Backend.
/// We currently support - ``Candle``, ``NdArray``, and ``Wgpu``
#[allow(clippy::too_many_arguments)]
fn compute_pose_correctives_on_backend<B: Backend>(
    scene: &Scene,
    command_buffer: &mut CommandBuffer,
    smpl_models: &SmplCache<B>,
    smpl_models_changed: bool,
) where
    <B as Backend>::FloatTensorPrimitive<2>: Sync,
    <B as Backend>::IntTensorPrimitive<2>: Sync,
    B::QuantizedTensorPrimitive<1>: Sync,
    B::QuantizedTensorPrimitive<2>: Sync,
    B::QuantizedTensorPrimitive<3>: Sync,
{
    let mut query_state = scene.world.query::<(&SmplParams, &mut Pose, Changed<Pose>, Changed<SmplParams>)>();
    for (entity, (smpl_params, smpl_pose, changed_pose, changed_smpl_params)) in query_state.iter() {
        if (!changed_pose && !changed_smpl_params && !smpl_models_changed) || !smpl_params.enable_pose_corrective {
            continue;
        }
        let smpl_model = smpl_models.get_model_ref(smpl_params.smpl_type, smpl_params.gender).unwrap();
        let verts_offset = smpl_model.compute_pose_correctives(&smpl_pose);
        command_buffer.insert_one(entity, PoseCorrectiveDynamic::<B> { verts_offset });
    }
}
/// System for applying a pose to the given template
/// This internally uses the generic variant of the function suffixed with
/// ``_on_backend``
#[allow(clippy::too_many_lines)]
pub extern "C" fn smpl_apply_pose(scene: &mut Scene, _runner: &mut RunnerState) {
    let mut command_buffer = CommandBuffer::new();
    {
        let smpl_models_dynamic = scene.get_resource::<&SmplCacheDynamic>().unwrap();
        match &*smpl_models_dynamic {
            SmplCacheDynamic::NdArray(smpl_models_ndarray) => {
                apply_pose_on_backend::<NdArray>(scene, &mut command_buffer, smpl_models_ndarray);
            }
            SmplCacheDynamic::Wgpu(smpl_models_wgpu) => {
                apply_pose_on_backend::<Wgpu>(scene, &mut command_buffer, smpl_models_wgpu);
            }
            SmplCacheDynamic::Candle(smpl_models_candle) => {
                apply_pose_on_backend::<Candle>(scene, &mut command_buffer, smpl_models_candle);
            }
        }
    }
    command_buffer.run_on(&mut scene.world);
}
/// Function for applying pose on a generic Burn Backend. We currently support -
/// ``Candle``, ``NdArray``, and ``Wgpu``
#[allow(clippy::too_many_arguments)]
fn apply_pose_on_backend<B: Backend>(scene: &Scene, command_buffer: &mut CommandBuffer, smpl_models: &SmplCache<B>)
where
    <B as Backend>::FloatTensorPrimitive<2>: Sync,
    <B as Backend>::IntTensorPrimitive<2>: Sync,
    B::QuantizedTensorPrimitive<1>: Sync,
    B::QuantizedTensorPrimitive<2>: Sync,
    B::QuantizedTensorPrimitive<3>: Sync,
{
    let mut query_state = scene.world.query::<(
        &SmplParams,
        &mut SmplOutputPoseTDynamic<B>,
        &mut Pose,
        Option<&PoseCorrectiveDynamic<B>>,
        Changed<Pose>,
        Changed<SmplOutputPoseTDynamic<B>>,
        Changed<SmplParams>,
    )>();
    for (entity, (smpl_params, smpl_t_output, smpl_pose, pose_corrective, changed_pose, changed_t_output, changed_smpl_params)) in query_state.iter()
    {
        if !changed_pose && !changed_t_output && !changed_smpl_params {
            continue;
        }
        let smpl_model = smpl_models.get_model_ref(smpl_params.smpl_type, smpl_params.gender).unwrap();
        let lbs_weights_merged = smpl_model.lbs_weights();
        let mut verts_burn_merged = smpl_t_output.verts.clone();
        let joints_t_pose = &smpl_t_output.joints;
        let new_pose = smpl_pose.clone();
        if let Some(pose_corrective) = pose_corrective {
            if smpl_params.enable_pose_corrective {
                let v_offset_merged = &pose_corrective.verts_offset;
                verts_burn_merged = verts_burn_merged.add(v_offset_merged.clone());
            }
        }
        let (verts_posed_nd, _, _, joints_posed) =
            smpl_model.apply_pose(&verts_burn_merged, None, None, joints_t_pose, &lbs_weights_merged, &new_pose);
        command_buffer.insert_one(
            entity,
            SmplOutputPosedDynamic {
                verts: verts_posed_nd,
                joints: joints_posed,
            },
        );
    }
}
/// System to convert ``SmplOutput`` components to gloss components (``Verts``,
/// ``Faces``, ``Normals``, etc.) for the ``upload_pass``
#[allow(clippy::too_many_lines)]
pub extern "C" fn smpl_to_gloss_mesh(scene: &mut Scene, _runner: &mut RunnerState) {
    let mut command_buffer = CommandBuffer::new();
    {
        let smpl_models_dynamic = scene.get_resource::<&SmplCacheDynamic>().unwrap();
        match &*smpl_models_dynamic {
            SmplCacheDynamic::NdArray(smpl_models_ndarray) => {
                let mut query_state = scene.world.query::<(
                    &SmplParams,
                    &SmplOutputPosedDynamic<NdArray>,
                    Changed<SmplOutputPosedDynamic<NdArray>>,
                    &GlossInterop,
                )>();
                for (entity, (smpl_params, smpl_output, changed_output, gloss_interop)) in query_state.iter() {
                    if !changed_output && !smpl_models_dynamic.is_changed() {
                        continue;
                    }
                    let smpl_model = smpl_models_ndarray.get_model_ref(smpl_params.smpl_type, smpl_params.gender).unwrap();
                    let (verts, uv, normals, tangents, faces) = compute_common_mesh_data(smpl_model, &smpl_output.verts, gloss_interop.with_uv);
                    update_entity_on_backend(
                        entity,
                        scene,
                        &mut command_buffer,
                        gloss_interop.with_uv,
                        &DynamicTensorFloat2D::NdArray(verts),
                        &DynamicTensorFloat2D::NdArray(normals),
                        tangents.map(DynamicTensorFloat2D::NdArray),
                        DynamicTensorFloat2D::NdArray(uv),
                        DynamicTensorInt2D::NdArray(faces),
                        smpl_model,
                    );
                }
            }
            SmplCacheDynamic::Wgpu(smpl_models_wgpu) => {
                let mut query_state = scene.world.query::<(
                    &SmplParams,
                    &SmplOutputPosedDynamic<Wgpu>,
                    Changed<SmplOutputPosedDynamic<Wgpu>>,
                    &GlossInterop,
                )>();
                for (entity, (smpl_params, smpl_output, changed_output, gloss_interop)) in query_state.iter() {
                    if !changed_output && !smpl_models_dynamic.is_changed() {
                        continue;
                    }
                    let smpl_model = smpl_models_wgpu.get_model_ref(smpl_params.smpl_type, smpl_params.gender).unwrap();
                    let (verts, uv, normals, tangents, faces) = compute_common_mesh_data(smpl_model, &smpl_output.verts, gloss_interop.with_uv);
                    update_entity_on_backend(
                        entity,
                        scene,
                        &mut command_buffer,
                        gloss_interop.with_uv,
                        &DynamicTensorFloat2D::Wgpu(verts),
                        &DynamicTensorFloat2D::Wgpu(normals),
                        tangents.map(DynamicTensorFloat2D::Wgpu),
                        DynamicTensorFloat2D::Wgpu(uv),
                        DynamicTensorInt2D::Wgpu(faces),
                        smpl_model,
                    );
                }
            }
            SmplCacheDynamic::Candle(smpl_models_candle) => {
                let mut query_state = scene.world.query::<(
                    &SmplParams,
                    &SmplOutputPosedDynamic<Candle>,
                    Changed<SmplOutputPosedDynamic<Candle>>,
                    &GlossInterop,
                )>();
                for (entity, (smpl_params, smpl_output, changed_output, gloss_interop)) in query_state.iter() {
                    if !changed_output && !smpl_models_dynamic.is_changed() {
                        continue;
                    }
                    let smpl_model = smpl_models_candle.get_model_ref(smpl_params.smpl_type, smpl_params.gender).unwrap();
                    let (verts, uv, normals, tangents, faces) = compute_common_mesh_data(smpl_model, &smpl_output.verts, gloss_interop.with_uv);
                    update_entity_on_backend(
                        entity,
                        scene,
                        &mut command_buffer,
                        gloss_interop.with_uv,
                        &DynamicTensorFloat2D::Candle(verts),
                        &DynamicTensorFloat2D::Candle(normals),
                        tangents.map(DynamicTensorFloat2D::Candle),
                        DynamicTensorFloat2D::Candle(uv),
                        DynamicTensorInt2D::Candle(faces),
                        smpl_model,
                    );
                }
            }
        }
    }
    command_buffer.run_on(&mut scene.world);
}
/// Type alias to club all mesh data together
type MeshDataResult<B> = (
    Tensor<B, 2, Float>,
    Tensor<B, 2, Float>,
    Tensor<B, 2, Float>,
    Option<Tensor<B, 2, Float>>,
    Tensor<B, 2, Int>,
);
/// Function to compute data like Normals and Tangents on a generic Burn
/// Backend. We currently support - ``Candle``, ``NdArray``, and ``Wgpu``
fn compute_common_mesh_data<B: Backend>(smpl_model: &dyn SmplModel<B>, verts_burn: &Tensor<B, 2, Float>, with_uv: bool) -> MeshDataResult<B> {
    let device_str = format!("{:?}", verts_burn.device());
    let mapping = smpl_model.idx_split_2_merged();
    let verts_final_burn = if with_uv {
        verts_burn.clone().select(0, mapping.clone())
    } else {
        verts_burn.clone()
    };
    let uv_burn = smpl_model.uv().clone();
    let faces_burn = smpl_model.faces();
    let normals_merged_burn = match device_str.as_str() {
        "Cpu" => Geom::compute_per_vertex_normals(
            &verts_burn.to_nalgebra(),
            &faces_burn.to_nalgebra(),
            &PerVertexNormalsWeightingType::Uniform,
        )
        .to_burn(&verts_burn.device()),
        _ => Geom::compute_per_vertex_normals_burn(verts_burn, faces_burn, &PerVertexNormalsWeightingType::Uniform),
    };
    let normals_final_burn = if with_uv {
        normals_merged_burn.clone().select(0, mapping.clone())
    } else {
        normals_merged_burn
    };
    let tangents_burn = if with_uv {
        match device_str.as_str() {
            "Cpu" => Some(
                Geom::compute_tangents(
                    &verts_final_burn.to_nalgebra(),
                    &smpl_model.faces_uv().to_nalgebra(),
                    &normals_final_burn.to_nalgebra(),
                    &smpl_model.uv().to_nalgebra(),
                )
                .to_burn(&verts_burn.device()),
            ),
            _ => Some(Geom::compute_tangents_burn(
                &verts_final_burn,
                smpl_model.faces_uv(),
                &normals_final_burn,
                smpl_model.uv(),
            )),
        }
    } else {
        None
    };
    let faces_burn = if with_uv { smpl_model.faces_uv() } else { faces_burn };
    (verts_final_burn, uv_burn, normals_final_burn, tangents_burn, faces_burn.clone())
}
/// System to align a mesh at every frame to the floor
/// This internally uses the generic variant of the function suffixed with
/// ``_on_backend``
#[allow(clippy::too_many_lines)]
pub extern "C" fn smpl_align_vertical(scene: &mut Scene, _runner: &mut RunnerState) {
    let mut command_buffer = CommandBuffer::new();
    let backend = {
        let smpl_models_dynamic = scene.get_resource::<&SmplCacheDynamic>().unwrap();
        smpl_models_dynamic.get_backend()
    };
    match backend {
        BurnBackend::NdArray => align_vertical_on_backend::<NdArray>(scene),
        BurnBackend::Wgpu => align_vertical_on_backend::<Wgpu>(scene),
        BurnBackend::Candle => align_vertical_on_backend::<Candle>(scene),
    }
    command_buffer.run_on(&mut scene.world);
}
fn align_vertical_on_backend<B: Backend>(scene: &Scene)
where
    <B as Backend>::FloatTensorPrimitive<2>: Sync,
    B::QuantizedTensorPrimitive<1>: Sync,
    B::QuantizedTensorPrimitive<2>: Sync,
    B::QuantizedTensorPrimitive<3>: Sync,
{
    let mut query_state = scene
        .world
        .query::<(&Verts, &mut ModelMatrix, Changed<SmplOutputPoseTDynamic<B>>)>()
        .with::<&SmplParams>();
    for (_entity, (verts, mut model_matrix, changed_t_pose)) in query_state.iter() {
        if changed_t_pose {
            let verts_world = Geom::transform_verts(&verts.0.to_dmatrix(), &model_matrix.0);
            let min_y = verts_world.column(1).min();
            model_matrix.0.append_translation_mut(&na::Translation3::<f32>::new(0.0, -min_y, 0.0));
        }
    }
}
/// System for follower computations
#[allow(clippy::cast_precision_loss)]
pub extern "C" fn smpl_follow_anim(scene: &mut Scene, runner: &mut RunnerState) {
    let mut command_buffer = CommandBuffer::new();
    {
        let backend = {
            let smpl_models_dynamic = scene.get_resource::<&SmplCacheDynamic>().unwrap();
            smpl_models_dynamic.get_backend()
        };
        let Ok(mut follow) = scene.get_resource::<&mut Follower>() else {
            return;
        };
        let mut query_state = scene.world.query::<&Follow>().with::<&Renderable>();
        let mut goal = na::Point3::new(0.0, 0.0, 0.0);
        let mut num_ents = 0;
        for (entity, _) in query_state.iter() {
            let model_matrix = if let Ok(mm) = scene.world.get::<&mut ModelMatrix>(entity) {
                mm.0
            } else {
                ModelMatrix::default().0
            };
            let ent_goal = match backend {
                BurnBackend::NdArray => {
                    if let Some(point) = handle_goal_for_backend::<NdArray>(scene, entity, model_matrix) {
                        point
                    } else {
                        continue;
                    }
                }
                BurnBackend::Wgpu => {
                    if let Some(point) = handle_goal_for_backend::<Wgpu>(scene, entity, model_matrix) {
                        point
                    } else {
                        continue;
                    }
                }
                BurnBackend::Candle => {
                    if let Some(point) = handle_goal_for_backend::<Candle>(scene, entity, model_matrix) {
                        point
                    } else {
                        continue;
                    }
                }
            };
            goal.coords += ent_goal.coords;
            num_ents += 1;
        }
        if num_ents > 0 {
            goal.coords /= num_ents as f32;
        }
        #[allow(clippy::match_wildcard_for_single_variants)]
        match follow.params.follower_type {
            FollowerType::Cam | FollowerType::CamAndLights => {
                let cam = scene.get_current_cam().unwrap();
                if !cam.is_initialized(scene) {
                    return;
                }
                if let Ok(mut poslookat) = scene.world.get::<&mut PosLookat>(cam.entity) {
                    follow.update(&goal, &poslookat.lookat, runner.dt().as_secs_f32());
                    let point_lookat = follow.get_point_follow("cam");
                    let diff = (poslookat.lookat - point_lookat).norm();
                    if diff > 1e-7 {
                        runner.request_redraw();
                        poslookat.shift_lookat(point_lookat);
                    }
                }
            }
            _ => {}
        }
        #[allow(clippy::match_wildcard_for_single_variants)]
        match follow.params.follower_type {
            FollowerType::Lights | FollowerType::CamAndLights => {
                let lights = scene.get_lights(false);
                for light in lights.iter() {
                    if let Ok(mut poslookat) = scene.world.get::<&mut PosLookat>(*light) {
                        let point_lookat = goal;
                        let diff = (poslookat.lookat - point_lookat).norm();
                        if diff > 1e-7 {
                            runner.request_redraw();
                            poslookat.shift_lookat(point_lookat);
                        }
                    }
                }
                let point_dist_fade_center = goal;
                command_buffer.insert_one(
                    scene.get_entity_resource(),
                    ConfigChanges {
                        new_distance_fade_center: point_dist_fade_center,
                    },
                );
            }
            _ => {}
        }
    }
    command_buffer.run_on(&mut scene.world);
}
fn handle_goal_for_backend<B: Backend>(scene: &Scene, entity: Entity, model_matrix: na::SimilarityMatrix3<f32>) -> Option<na::Point3<f32>>
where
    <B as Backend>::FloatTensorPrimitive<2>: Sync,
    <B as Backend>::IntTensorPrimitive<2>: Sync,
    B::QuantizedTensorPrimitive<1>: Sync,
    B::QuantizedTensorPrimitive<2>: Sync,
    B::QuantizedTensorPrimitive<3>: Sync,
{
    if scene.world.has::<SmplOutputPosedDynamic<B>>(entity).unwrap() {
        let output_posed = scene.world.get::<&SmplOutputPosedDynamic<B>>(entity).unwrap();
        let joints_ndarray = output_posed.joints.to_ndarray();
        let pose_trans = joints_ndarray.row(0).into_nalgebra();
        let point = pose_trans.fixed_rows::<3>(0).clone_owned();
        let mut point = na::Point3::<f32> { coords: point };
        point = model_matrix * point;
        Some(point)
    } else if scene.world.has::<Verts>(entity).unwrap() && scene.world.has::<ModelMatrix>(entity).unwrap() {
        let verts = scene.world.get::<&Verts>(entity).unwrap();
        let model_matrix = scene.world.get::<&ModelMatrix>(entity).unwrap();
        Some(Geom::get_centroid(&verts.0.to_dmatrix(), Some(model_matrix.0)))
    } else {
        None
    }
}
/// Hides the floor if the camera is below it
pub extern "C" fn hide_floor_when_viewed_from_below(scene: &mut Scene, _runner: &mut RunnerState) {
    let camera = scene.get_current_cam().unwrap();
    let pos = {
        let Ok(pos_lookat) = scene.world.get::<&PosLookat>(camera.entity) else {
            warn!("rs: hide_floor_when_viewed_from_below: No PosLookat yet, camera is not initialized. Auto adding default");
            return;
        };
        pos_lookat.position
    };
    if let Some(floor) = scene.get_floor() {
        let min_y = {
            let Ok(verts) = scene.world.get::<&Verts>(floor.entity) else {
                warn!("rs: hide_floor_when_viewed_from_below: No Verts on floor");
                return;
            };
            let Ok(model_matrix) = scene.world.get::<&ModelMatrix>(floor.entity) else {
                warn!("rs: hide_floor_when_viewed_from_below: No ModelMatrix on floor");
                return;
            };
            let verts_world = Geom::transform_verts(&verts.0.to_dmatrix(), &model_matrix.0);
            verts_world.column(1).min()
        };
        if pos.coords.y < min_y {
            if scene.world.has::<Renderable>(floor.entity).unwrap() {
                scene.world.remove_one::<Renderable>(floor.entity).unwrap();
            }
        } else {
            if !scene.world.has::<Renderable>(floor.entity).unwrap() {
                scene.world.insert_one(floor.entity, Renderable).unwrap();
            }
        }
    }
}
#[allow(missing_docs)]
#[cfg(feature = "with-gui")]
#[allow(clippy::semicolon_if_nothing_returned)]
#[allow(clippy::too_many_lines)]
#[cfg_attr(target_arch = "wasm32", allow(improper_ctypes_definitions))]
pub extern "C" fn smpl_params_gui(selected_entity: ROption<Entity>, scene: &mut Scene) -> GuiWindow {
    use gloss_renderer::plugin_manager::gui::Button;
    use gloss_utils::abi_stable_aliases::std_types::ROption::RSome;
    use smpl_core::{
        codec::{codec::SmplCodec, gltf::GltfCodec},
        common::types::{Gender, GltfCompatibilityMode},
    };
    extern "C" fn enable_pose_corrective_toggle(new_val: bool, _widget_name: RString, entity: Entity, scene: &mut Scene) {
        if let Ok(mut smpl_params) = scene.world.get::<&mut SmplParams>(entity) {
            smpl_params.enable_pose_corrective = new_val;
        }
    }
    extern "C" fn save_smpl(_widget_name: RString, entity: Entity, scene: &mut Scene) {
        let codec = SmplCodec::from_entity(&entity, scene, None);
        codec.to_file("./saved.smpl");
    }
    extern "C" fn save_gltf_smpl(_widget_name: RString, _entity: Entity, scene: &mut Scene) {
        let mut codec = GltfCodec::from_scene(scene, None, true);
        let now = wasm_timer::Instant::now();
        codec.to_file(
            "Meshcapade Avatar",
            "./saved/output.gltf",
            GltfOutputType::Standard,
            GltfCompatibilityMode::Smpl,
        );
        codec.to_file(
            "Meshcapade Avatar",
            "./saved/output.glb",
            GltfOutputType::Binary,
            GltfCompatibilityMode::Smpl,
        );
        println!("Smpl mode `.gltf` export took {:?} seconds", now.elapsed());
    }
    extern "C" fn save_gltf_unreal(_widget_name: RString, _entity: Entity, scene: &mut Scene) {
        let mut codec = GltfCodec::from_scene(scene, None, true);
        let now = wasm_timer::Instant::now();
        codec.to_file(
            "Meshcapade Avatar",
            "./saved/output.gltf",
            GltfOutputType::Standard,
            GltfCompatibilityMode::Unreal,
        );
        codec.to_file(
            "Meshcapade Avatar",
            "./saved/output.glb",
            GltfOutputType::Binary,
            GltfCompatibilityMode::Unreal,
        );
        println!("Unreal mode `.gltf` export took {:?} seconds", now.elapsed());
    }
    extern "C" fn change_gender(_val: bool, widget_name: RString, entity: Entity, scene: &mut Scene) {
        if let Ok(mut smpl_params) = scene.world.get::<&mut SmplParams>(entity) {
            match widget_name.as_str() {
                "neutral" => smpl_params.gender = Gender::Neutral,
                "female" => smpl_params.gender = Gender::Female,
                "male" => smpl_params.gender = Gender::Male,
                _ => {}
            }
        }
    }
    let mut widgets = RVec::new();
    if let RSome(entity) = selected_entity {
        if let Ok(smpl_params) = scene.world.get::<&SmplParams>(entity) {
            let checkbox = Checkbox::new(
                "enable_pose_corrective",
                smpl_params.enable_pose_corrective,
                enable_pose_corrective_toggle,
            );
            let is_neutral = smpl_params.gender == Gender::Neutral;
            let is_female = smpl_params.gender == Gender::Female;
            let is_male = smpl_params.gender == Gender::Male;
            let chk_neutral = Checkbox::new("neutral", is_neutral, change_gender);
            let chk_female = Checkbox::new("female", is_female, change_gender);
            let chk_male = Checkbox::new("male", is_male, change_gender);
            let button_save_smpl = Button::new("Save as .smpl", save_smpl);
            let button_save_gltf_smpl = Button::new("Save as .gltf (SMPL)", save_gltf_smpl);
            let button_save_gltf_unreal = Button::new("Save as .gltf (UNREAL)", save_gltf_unreal);
            widgets.push(Widgets::Checkbox(chk_neutral));
            widgets.push(Widgets::Checkbox(chk_female));
            widgets.push(Widgets::Checkbox(chk_male));
            widgets.push(Widgets::Checkbox(checkbox));
            widgets.push(Widgets::Button(button_save_smpl));
            widgets.push(Widgets::Button(button_save_gltf_smpl));
            widgets.push(Widgets::Button(button_save_gltf_unreal));
        }
    }
    GuiWindow {
        window_name: RString::from("SmplParams"),
        window_type: GuiWindowType::Sidebar,
        widgets,
    }
}
#[allow(missing_docs)]
#[cfg(feature = "with-gui")]
#[allow(clippy::semicolon_if_nothing_returned)]
#[cfg_attr(target_arch = "wasm32", allow(improper_ctypes_definitions))]
pub extern "C" fn smpl_betas_gui(selected_entity: ROption<Entity>, scene: &mut Scene) -> GuiWindow {
    use gloss_utils::abi_stable_aliases::std_types::ROption::RSome;
    extern "C" fn beta_slider_change(new_val: f32, widget_name: RString, entity: Entity, scene: &mut Scene) {
        let beta_idx: usize = widget_name.split('_').last().unwrap().parse().unwrap();
        if let Ok(mut betas) = scene.world.get::<&mut Betas>(entity) {
            betas.betas[beta_idx] = new_val;
        }
    }
    let mut widgets = RVec::new();
    if let RSome(entity) = selected_entity {
        if let Ok(betas) = scene.world.get::<&Betas>(entity) {
            for i in 0..betas.betas.len() {
                let slider = Slider::new(
                    ("Beta_".to_owned() + &i.to_string()).as_str(),
                    betas.betas[i],
                    -5.0,
                    5.0,
                    RSome(80.0),
                    beta_slider_change,
                    RNone,
                );
                widgets.push(Widgets::Slider(slider));
            }
        }
    }
    GuiWindow {
        window_name: RString::from("Betas"),
        window_type: GuiWindowType::Sidebar,
        widgets,
    }
}
#[allow(missing_docs)]
#[cfg(feature = "with-gui")]
#[allow(clippy::semicolon_if_nothing_returned)]
#[cfg_attr(target_arch = "wasm32", allow(improper_ctypes_definitions))]
pub extern "C" fn smpl_expression_gui(selected_entity: ROption<Entity>, scene: &mut Scene) -> GuiWindow {
    use gloss_utils::abi_stable_aliases::std_types::ROption::RSome;
    extern "C" fn expr_slider_change(new_val: f32, widget_name: RString, entity: Entity, scene: &mut Scene) {
        let coeff_idx: usize = widget_name.split('_').last().unwrap().parse().unwrap();
        if let Ok(mut coeffs) = scene.world.get::<&mut Expression>(entity) {
            coeffs.expr_coeffs[coeff_idx] = new_val;
        }
    }
    let mut widgets = RVec::new();
    if let RSome(entity) = selected_entity {
        if let Ok(expression) = scene.world.get::<&Expression>(entity) {
            for i in 0..expression.expr_coeffs.len() {
                let slider = Slider::new(
                    ("Coeff_".to_owned() + &i.to_string()).as_str(),
                    expression.expr_coeffs[i],
                    -5.0,
                    5.0,
                    RSome(80.0),
                    expr_slider_change,
                    RNone,
                );
                widgets.push(Widgets::Slider(slider));
            }
        }
    }
    GuiWindow {
        window_name: RString::from("Expression"),
        window_type: GuiWindowType::Sidebar,
        widgets,
    }
}
#[allow(missing_docs)]
#[cfg(feature = "with-gui")]
#[allow(clippy::semicolon_if_nothing_returned)]
#[cfg_attr(target_arch = "wasm32", allow(improper_ctypes_definitions))]
pub extern "C" fn smpl_anim_scroll_gui(_selected_entity: ROption<Entity>, scene: &mut Scene) -> GuiWindow {
    use gloss_renderer::plugin_manager::gui::{Button, WindowPivot, WindowPosition, WindowPositionType};
    use gloss_utils::abi_stable_aliases::std_types::ROption::RSome;
    extern "C" fn scene_anim_slider_change(new_val: f32, _widget_name: RString, _entity: Entity, scene: &mut Scene) {
        if let Ok(mut scene_anim) = scene.get_resource::<&mut SceneAnimation>() {
            scene_anim.set_cur_time_as_sec(new_val);
            scene_anim.runner.temporary_pause = true;
        }
    }
    extern "C" fn scene_anim_slider_no_change(_widget_name: RString, _entity: Entity, scene: &mut Scene) {
        if let Ok(mut scene_anim) = scene.get_resource::<&mut SceneAnimation>() {
            scene_anim.runner.temporary_pause = false;
        }
    }
    extern "C" fn scene_button_play_pause(_widget_name: RString, _entity: Entity, scene: &mut Scene) {
        if let Ok(mut scene_anim) = scene.get_resource::<&mut SceneAnimation>() {
            scene_anim.runner.paused = !scene_anim.runner.paused;
        }
    }
    #[allow(clippy::cast_possible_truncation)]
    #[allow(clippy::cast_sign_loss)]
    extern "C" fn scene_button_next_frame(_widget_name: RString, _entity: Entity, scene: &mut Scene) {
        if let Ok(mut scene_anim) = scene.get_resource::<&mut SceneAnimation>() {
            let nr_frames = scene_anim.num_frames;
            let duration = scene_anim.duration();
            let dt_between_frames = duration / nr_frames as u32;
            scene_anim.advance(dt_between_frames, false);
        }
    }
    extern "C" fn scene_fps_slider_change(new_val: f32, _widget_name: RString, _entity: Entity, scene: &mut Scene) {
        if let Ok(mut scene_anim) = scene.get_resource::<&mut SceneAnimation>() {
            let cur_time = scene_anim.get_cur_time();
            let prev_fps = scene_anim.config.fps;
            let multiplier_duration = prev_fps / new_val;
            scene_anim.set_cur_time_as_sec(cur_time.as_secs_f32() * multiplier_duration);
            scene_anim.config.fps = new_val;
        }
    }
    extern "C" fn follow_anim(new_val: bool, _widget_name: RString, _entity: Entity, scene: &mut Scene) {
        if new_val {
            scene.add_resource(Follower::new(FollowParams::default()));
        } else {
            let _ = scene.remove_resource::<Follower>();
        }
    }
    let mut widgets = RVec::new();
    if let Ok(scene_anim) = scene.get_resource::<&mut SceneAnimation>() {
        if scene_anim.num_frames != 0 {
            let max_duration = scene_anim.duration().as_secs_f32();
            let cur_time = scene_anim.get_cur_time().as_secs_f32();
            let slider_anim = Slider::new(
                "AnimTime",
                cur_time,
                0.0,
                max_duration,
                RSome(400.0),
                scene_anim_slider_change,
                RSome(scene_anim_slider_no_change as extern "C" fn(RString, Entity, &mut Scene)),
            );
            let button_play_pause = Button::new("play/pause", scene_button_play_pause);
            let button_next_frame = Button::new("next_frame", scene_button_next_frame);
            let slider_fps = Slider::new("FPS", scene_anim.config.fps, 1.0, 120.0, RSome(100.0), scene_fps_slider_change, RNone);
            let chk_follow_anim = Checkbox::new("follow", scene.has_resource::<Follower>(), follow_anim);
            widgets.push(Widgets::Slider(slider_anim));
            widgets.push(Widgets::Horizontal(RVec::from(vec![
                Widgets::Button(button_play_pause),
                Widgets::Button(button_next_frame),
                Widgets::Slider(slider_fps),
            ])));
            widgets.push(Widgets::Horizontal(RVec::from(vec![Widgets::Checkbox(chk_follow_anim)])));
        }
    }
    GuiWindow {
        window_name: RString::from("Scene"),
        window_type: GuiWindowType::FloatWindow(WindowPivot::CenterBottom, WindowPosition([0.5, 1.0]), WindowPositionType::Fixed),
        widgets,
    }
}
#[allow(missing_docs)]
#[cfg(feature = "with-gui")]
#[allow(clippy::semicolon_if_nothing_returned)]
#[cfg_attr(target_arch = "wasm32", allow(improper_ctypes_definitions))]
pub extern "C" fn smpl_hand_pose_gui(selected_entity: ROption<Entity>, scene: &mut Scene) -> GuiWindow {
    use gloss_renderer::plugin_manager::gui::SelectableList;
    use gloss_utils::abi_stable_aliases::std_types::ROption::RSome;
    use log::warn;
    use smpl_core::common::pose_hands::HandType;
    extern "C" fn set_hand_pose_type(widget_name: RString, entity: Entity, scene: &mut Scene) {
        let hand_type = match widget_name.to_string().as_str() {
            "Flat" => Some(HandType::Flat),
            "Relaxed" => Some(HandType::Relaxed),
            "Curled" => Some(HandType::Curled),
            "Fist" => Some(HandType::Fist),
            _ => {
                warn!("HandType not known");
                None
            }
        };
        let mut command_buffer = CommandBuffer::new();
        if let Some(hand_type) = hand_type {
            info!("setting to {hand_type:?}");
            if let Ok(mut pose_mask) = scene.world.get::<&mut PoseOverride>(entity) {
                info!("we already have a pose mask");
                pose_mask.set_overwrite_hands(hand_type);
            } else {
                info!("inserting a new pose mask");
                let pose_mask = PoseOverride::allow_all().overwrite_hands(hand_type).build();
                command_buffer.insert_one(entity, pose_mask);
            }
        } else {
            info!("removing overwrite");
            if let Ok(mut pose_mask) = scene.world.get::<&mut PoseOverride>(entity) {
                info!("removing overwrite and we have posemask");
                if pose_mask.get_overwrite_hands_type().is_some() {
                    pose_mask.remove_overwrite_hands();
                }
            }
        }
        command_buffer.run_on(&mut scene.world);
    }
    let mut widgets = RVec::new();
    if let RSome(entity) = selected_entity {
        if let Ok(pose_mask) = scene.world.get::<&PoseOverride>(entity) {
            let hand_type_overwrite = pose_mask.get_overwrite_hands_type();
            let mut selectable_vec = RVec::new();
            selectable_vec.push(Selectable::new("None", hand_type_overwrite.is_none(), set_hand_pose_type));
            selectable_vec.push(Selectable::new("Flat", hand_type_overwrite == Some(HandType::Flat), set_hand_pose_type));
            selectable_vec.push(Selectable::new(
                "Relaxed",
                hand_type_overwrite == Some(HandType::Relaxed),
                set_hand_pose_type,
            ));
            selectable_vec.push(Selectable::new(
                "Curled",
                hand_type_overwrite == Some(HandType::Curled),
                set_hand_pose_type,
            ));
            selectable_vec.push(Selectable::new("Fist", hand_type_overwrite == Some(HandType::Fist), set_hand_pose_type));
            let selectable_list = SelectableList::new(selectable_vec, true);
            widgets.push(Widgets::SelectableList(selectable_list));
        }
    }
    GuiWindow {
        window_name: RString::from("HandOverwrite"),
        window_type: GuiWindowType::Sidebar,
        widgets,
    }
}
#[allow(missing_docs)]
#[cfg(feature = "with-gui")]
#[allow(clippy::cast_precision_loss)]
#[cfg_attr(target_arch = "wasm32", allow(improper_ctypes_definitions))]
pub extern "C" fn smpl_event_dropfile(scene: &mut Scene, _runner: &mut RunnerState, event: &Event) -> bool {
    use crate::scene::McsCodecGloss;
    use log::warn;
    use smpl_core::codec::{codec::SmplCodec, scene::McsCodec};
    use std::path::PathBuf;
    let mut handled = false;
    match event {
        Event::DroppedFile(path) => {
            let path_buf = PathBuf::from(path.to_string());
            let filetype = match path_buf.extension() {
                Some(extension) => FileType::find_match(extension.to_str().unwrap_or("")),
                None => FileType::Unknown,
            };
            if scene.has_resource::<SceneAnimation>() {
                scene.remove_resource::<SceneAnimation>().unwrap();
            }
            match filetype {
                FileType::Smpl => {
                    info!("handling dropped smpl file {}", path);
                    let codec = SmplCodec::from_file(path);
                    let mut builder = codec.to_entity_builder();
                    if !builder.has::<Betas>() {
                        warn!("The .smpl file didn't have any shape_parameters associated, we are defaulting to the mean smpl shape");
                        builder.add(Betas::default());
                    }
                    let gloss_interop = GlossInterop::default();
                    let name = scene.get_unused_name();
                    scene.get_or_create_entity(&name).insert_builder(builder).insert(gloss_interop);
                    handled = true;
                }
                FileType::Mcs => {
                    info!("handling dropped mcs file {}", path);
                    let mut codec = McsCodec::from_file(path);
                    let builders = codec.to_entity_builders();
                    for mut builder in builders {
                        if !builder.has::<Betas>() {
                            warn!("The .smpl file didn't have any shape_parameters associated, we are defaulting to the mean smpl shape");
                            builder.add(Betas::default());
                        }
                        let gloss_interop = GlossInterop::default();
                        let name = scene.get_unused_name();
                        scene.get_or_create_entity(&name).insert_builder(builder).insert(gloss_interop);
                        handled = true;
                    }
                    let smpl_scene = SceneAnimation::new_with_fps(codec.num_frames, codec.frame_rate);
                    scene.add_resource(smpl_scene);
                }
                _ => {
                    info!("No known filetype {}", path);
                }
            }
        }
    }
    handled
}
