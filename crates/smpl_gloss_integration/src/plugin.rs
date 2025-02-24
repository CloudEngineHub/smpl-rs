use crate::systems::smpl_auto_add_follow;
// use crate::systems::smpl_auto_add_interval;
use crate::systems::smpl_auto_add_scene;
#[cfg(feature = "with-gui")]
use crate::systems::smpl_event_dropfile;
use crate::systems::{hide_floor_when_viewed_from_below, smpl_advance_anim};

#[cfg(feature = "with-gui")]
use crate::systems::smpl_anim_scroll_gui;
// #[cfg(feature = "with-gui")]
// use crate::systems::fps_display_gui;
#[cfg(feature = "with-gui")]
use crate::systems::smpl_betas_gui;
#[cfg(feature = "with-gui")]
use crate::systems::smpl_expression_gui;
#[cfg(feature = "with-gui")]
use crate::systems::smpl_hand_pose_gui;
#[cfg(not(target_arch = "wasm32"))]
use crate::systems::smpl_lazy_load_model;
#[cfg(feature = "with-gui")]
use crate::systems::smpl_params_gui;
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
            if #[cfg(feature = "with-gui")]{
                vec.push(EventSystem::new(smpl_event_dropfile).with_name("smpl_event_dropfile"));
            }
            // TODO: is there a reason to do this? why do anything at all if the scope is empty?
            else{
            }
        }
        vec
    }
    fn logic_systems(&self) -> Vec<LogicSystem> {
        let mut vec = Vec::new();
        #[cfg(not(target_arch = "wasm32"))]
        vec.push(LogicSystem::new(smpl_lazy_load_model).with_name("smpl_lazy_load_model")); // Involves reading from file which is not a thing on wasm

        let mut rest = vec![
            LogicSystem::new(smpl_auto_add_scene).with_name("smpl_auto_add_scene"), // Automatically add a scene resource if absent
            // LogicSystem::new(smpl_auto_add_interval).with_name("smpl_auto_add_interval"), // Automatically add SmplInterval's to Entities if absent
            LogicSystem::new(smpl_auto_add_follow).with_name("smpl_auto_add_follow"), // Automatically add Follow to all ents if FollowerParams.follow_all is true
            LogicSystem::new(smpl_advance_anim).with_name("smpl_advance_anim"),       // Anim -> Anim + dt
            LogicSystem::new(smpl_betas_to_verts).with_name("smpl_betas_to_verts"),   // Changed(Betas) -> SmplOutputPoseT
            LogicSystem::new(smpl_expression_offsets).with_name("smpl_expression_offsets"), // Changed(expression) -> ExpressionOffsets
            LogicSystem::new(smpl_expression_apply).with_name("smpl_expression_apply"), //Changed(SmplOutputPoseT, ExpressionOffset) -> SmplOutputPoseT
            LogicSystem::new(smpl_make_dummy_pose).with_name("smpl_make_dummy_pose"),
            LogicSystem::new(smpl_pose_remap).with_name("smpl_pose_remap"),
            LogicSystem::new(smpl_mask_pose).with_name("smpl_mask_pose"),
            LogicSystem::new(smpl_compute_pose_correctives).with_name("smpl_compute_pose_correctives"), // Changed(pose) -> VertsOffset
            LogicSystem::new(smpl_apply_pose).with_name("smpl_apply_pose"), // Changed(SmplTPose,pose,ExpressionOffset) -> SmplOutputPosed
            LogicSystem::new(smpl_to_gloss_mesh).with_name("smpl_to_gloss_mesh"),
            LogicSystem::new(smpl_follow_anim).with_name("smpl_follow_anim"),
            LogicSystem::new(hide_floor_when_viewed_from_below).with_name("hide_floor_when_viewed_from_below"),
            // LogicSystem::new(smpl_align_vertical).with_name("smpl_align_vertical"),
            // We don't enable smpl_align_vertical because it messes with what you expect
            // the certain animations to look like and where the origin is.
            // But this could potentially be useful in certain settings
        ];

        vec.append(&mut rest);
        vec
    }

    #[allow(clippy::vec_init_then_push)]
    fn gui_systems(&self) -> Vec<GuiSystem> {
        let mut vec = Vec::new();
        cfg_if::cfg_if! {
            if #[cfg(feature = "with-gui")]{
                vec.push(GuiSystem::new(smpl_params_gui));
                vec.push(GuiSystem::new(smpl_betas_gui));
                vec.push(GuiSystem::new(smpl_expression_gui));
                vec.push(GuiSystem::new(smpl_anim_scroll_gui));
                vec.push(GuiSystem::new(smpl_hand_pose_gui));
                // vec.push(GuiSystem::new(fps_display_gui));
            }
            else {}
        }
        vec
    }
}
