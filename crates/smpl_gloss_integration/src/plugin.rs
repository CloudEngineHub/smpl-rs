#[cfg(feature = "with-gui")]
use crate::systems::smpl_anim_scroll_gui;
use crate::systems::smpl_auto_add_follow;
use crate::systems::smpl_auto_add_scene;
#[cfg(feature = "with-gui")]
use crate::systems::smpl_betas_gui;
#[cfg(feature = "with-gui")]
use crate::systems::smpl_event_dropfile;
#[cfg(feature = "with-gui")]
use crate::systems::smpl_expression_gui;
#[cfg(feature = "with-gui")]
use crate::systems::smpl_hand_pose_gui;
#[cfg(not(target_arch = "wasm32"))]
use crate::systems::smpl_lazy_load_model;
#[cfg(feature = "with-gui")]
use crate::systems::smpl_params_gui;
use crate::systems::{hide_floor_when_viewed_from_below, smpl_advance_anim};
use crate::systems::{
    smpl_apply_pose, smpl_betas_to_verts, smpl_compute_pose_correctives, smpl_expression_apply, smpl_expression_offsets, smpl_follow_anim,
    smpl_make_dummy_pose, smpl_mask_pose, smpl_pose_remap, smpl_to_gloss_mesh,
};
use gloss_renderer::plugin_manager::{EventSystem, GuiSystem, LogicSystem, Plugin};
#[derive(Clone)]
pub struct SmplPlugin {
    pub autorun: bool,
}
impl SmplPlugin {
    pub fn new(autorun: bool) -> Self {
        Self { autorun }
    }
}
impl Plugin for SmplPlugin {
    fn autorun(&self) -> bool {
        self.autorun
    }
    #[allow(clippy::vec_init_then_push)]
    fn event_systems(&self) -> Vec<EventSystem> {
        let mut vec = Vec::new();
        cfg_if::cfg_if! {
            if #[cfg(feature = "with-gui")] { vec
            .push(EventSystem::new(smpl_event_dropfile)
            .with_name("smpl_event_dropfile")); } else {}
        }
        vec
    }
    fn logic_systems(&self) -> Vec<LogicSystem> {
        let mut vec = Vec::new();
        #[cfg(not(target_arch = "wasm32"))]
        vec.push(LogicSystem::new(smpl_lazy_load_model).with_name("smpl_lazy_load_model"));
        let mut rest = vec![
            LogicSystem::new(smpl_auto_add_scene).with_name("smpl_auto_add_scene"),
            LogicSystem::new(smpl_auto_add_follow).with_name("smpl_auto_add_follow"),
            LogicSystem::new(smpl_advance_anim).with_name("smpl_advance_anim"),
            LogicSystem::new(smpl_betas_to_verts).with_name("smpl_betas_to_verts"),
            LogicSystem::new(smpl_expression_offsets).with_name("smpl_expression_offsets"),
            LogicSystem::new(smpl_expression_apply).with_name("smpl_expression_apply"),
            LogicSystem::new(smpl_make_dummy_pose).with_name("smpl_make_dummy_pose"),
            LogicSystem::new(smpl_pose_remap).with_name("smpl_pose_remap"),
            LogicSystem::new(smpl_mask_pose).with_name("smpl_mask_pose"),
            LogicSystem::new(smpl_compute_pose_correctives).with_name("smpl_compute_pose_correctives"),
            LogicSystem::new(smpl_apply_pose).with_name("smpl_apply_pose"),
            LogicSystem::new(smpl_to_gloss_mesh).with_name("smpl_to_gloss_mesh"),
            LogicSystem::new(smpl_follow_anim).with_name("smpl_follow_anim"),
            LogicSystem::new(hide_floor_when_viewed_from_below).with_name("hide_floor_when_viewed_from_below"),
        ];
        vec.append(&mut rest);
        vec
    }
    #[allow(clippy::vec_init_then_push)]
    fn gui_systems(&self) -> Vec<GuiSystem> {
        let mut vec = Vec::new();
        cfg_if::cfg_if! {
            if #[cfg(feature = "with-gui")] { vec.push(GuiSystem::new(smpl_params_gui));
            vec.push(GuiSystem::new(smpl_betas_gui)); vec
            .push(GuiSystem::new(smpl_expression_gui)); vec
            .push(GuiSystem::new(smpl_anim_scroll_gui)); vec
            .push(GuiSystem::new(smpl_hand_pose_gui)); } else {}
        }
        vec
    }
}
