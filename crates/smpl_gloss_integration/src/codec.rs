use std::cmp::Ordering;

use gloss_hecs::{Entity, EntityBuilder};
use gloss_renderer::scene::Scene;

use log::info;
use nd::concatenate;
use ndarray as nd;
use smpl_rs::{
    codec::codec::SmplCodec,
    common::{
        animation::{AnimWrap, Animation},
        betas::Betas,
        metadata::smpl_metadata,
        pose::Pose,
        pose_override::PoseOverride,
        pose_retarget::RetargetPoseYShift,
        smpl_params::SmplParams,
    },
    conversions::{pose_chunked::PoseChunked, pose_remap::PoseRemap},
};
use smpl_utils::log;

/// Creates a ``SmplCodec`` from an entity by extracting components from it or
/// creates a ``gloss_hecs::EntityBuilder`` from the ``SmplCodec``
pub trait SmplCodecGloss {
    fn from_entity(entity: &Entity, scene: &Scene, max_texture_size: Option<u32>) -> SmplCodec;
    fn to_entity_builder(&self) -> EntityBuilder;
}

/// Trait implementation for `SmplCodec`
impl SmplCodecGloss for SmplCodec {
    #[allow(clippy::too_many_lines)]
    #[allow(clippy::cast_possible_truncation)]
    #[allow(clippy::cast_possible_wrap)]
    fn from_entity(entity: &Entity, scene: &Scene, _max_texture_size: Option<u32>) -> SmplCodec {
        let mut codec = SmplCodec::default();

        let smpl_params = scene.get_comp::<&SmplParams>(entity).expect("Entity should have SmplParams component");
        let smpl_version = smpl_params.smpl_type as i32;
        let gender = smpl_params.gender as i32;

        codec.smpl_version = smpl_version;
        codec.gender = gender;

        if let Ok(betas) = scene.get_comp::<&Betas>(entity) {
            codec.shape_parameters = Some(betas.betas.clone());
        }

        // If we have only pose but not animation then we only add components from the
        // Pose
        if scene.world.has::<Pose>(*entity).unwrap() && !scene.world.has::<Animation>(*entity).unwrap() {
            log!("we are writing a pose in the codec");
            let pose = scene.get_comp::<&Pose>(entity).unwrap();
            let metadata = smpl_metadata(&smpl_params.smpl_type);
            let chunked = PoseChunked::new(&pose, &metadata);

            // Assign the chunked parts of the pose into the codec

            // Translation
            codec.body_translation = Some(chunked.global_trans);

            // Body
            if chunked.global_orient.is_some() && chunked.body_pose.is_some() {
                let body_pose_with_root =
                    concatenate(nd::Axis(0), &[chunked.global_orient.unwrap().view(), chunked.body_pose.unwrap().view()]).unwrap();
                codec.body_pose = Some(body_pose_with_root.insert_axis(nd::Axis(0)));
            }

            // Head
            if chunked.jaw_pose.is_some() && chunked.left_eye_pose.is_some() && chunked.right_eye_pose.is_some() {
                let head_pose = concatenate(
                    nd::Axis(0),
                    &[
                        chunked.jaw_pose.unwrap().view(),
                        chunked.left_eye_pose.unwrap().view(),
                        chunked.right_eye_pose.unwrap().view(),
                    ],
                )
                .unwrap();
                codec.head_pose = Some(head_pose.insert_axis(nd::Axis(0)));
            }

            // Left hand
            if let Some(left_hand_pose) = chunked.left_hand_pose {
                codec.left_hand_pose = Some(left_hand_pose.insert_axis(nd::Axis(0)));
            }

            // Right hand
            if let Some(right_hand_pose) = chunked.right_hand_pose {
                codec.right_hand_pose = Some(right_hand_pose.insert_axis(nd::Axis(0)));
            }

            // We have animation and we may also have or not have a Pose but we
            // don't care about it
        } else if scene.world.has::<Animation>(*entity).unwrap() {
            log!("we are writing a animation in the codec");
            let anim = scene.get_comp::<&Animation>(entity).unwrap();
            let metadata = smpl_metadata(&smpl_params.smpl_type);
            let nr_frames = anim.num_animation_frames();

            // We make some dense matrices for the codec, we will later prune the ones that
            // have only zeros
            let mut full_body_translation = nd::Array2::<f32>::zeros((nr_frames, 3));
            let mut full_body_pose = nd::Array3::<f32>::zeros((nr_frames, metadata.num_body_joints + 1, 3));
            let mut full_head_pose = nd::Array3::<f32>::zeros((nr_frames, metadata.num_face_joints, 3));
            let mut full_left_hand_pose = nd::Array3::<f32>::zeros((nr_frames, metadata.num_hand_joints, 3));
            let mut full_right_hand_pose = nd::Array3::<f32>::zeros((nr_frames, metadata.num_hand_joints, 3));

            // Run through the animation frame by frame and get the poses, remap them to the
            // current model, mask them and retarget them and then finally we add it our
            // array
            for time_idx in 0..anim.num_animation_frames() {
                let mut pose = anim.get_pose_at_idx(time_idx);

                // Here we put all the transformations that a pose can go through

                // 1. Remap
                let pose_remap = PoseRemap::new(pose.smpl_type, smpl_params.smpl_type);
                pose = pose_remap.remap(&pose);

                // 2. Mask
                if let Ok(ref pose_mask) = scene.get_comp::<&PoseOverride>(entity) {
                    let mut new_pose_mask = PoseOverride::clone(pose_mask);
                    pose.apply_mask(&mut new_pose_mask);
                }

                // 3. Retarget
                if let Ok(ref pose_retarget) = scene.get_comp::<&RetargetPoseYShift>(entity) {
                    let mut pose_retarget_local = RetargetPoseYShift::clone(pose_retarget); //we don't want to internally change the retarget because the saving shouldn't
                                                                                            // modify stuff in the world so we just clone a local one.
                    pose_retarget_local.apply(&mut pose);
                }

                // Chunk it and put everything into the full matrices
                let chunked = PoseChunked::new(&pose, &metadata);

                // Translation
                full_body_translation
                    .index_axis_mut(nd::Axis(0), time_idx)
                    .assign(&chunked.global_trans.to_shape(3).unwrap());

                // Body
                if chunked.global_orient.is_some() && chunked.body_pose.is_some() {
                    let body_pose_with_root =
                        concatenate(nd::Axis(0), &[chunked.global_orient.unwrap().view(), chunked.body_pose.unwrap().view()]).unwrap();
                    full_body_pose.index_axis_mut(nd::Axis(0), time_idx).assign(&body_pose_with_root);
                }

                // Head
                if chunked.jaw_pose.is_some() && chunked.left_eye_pose.is_some() && chunked.right_eye_pose.is_some() {
                    let head_pose = concatenate(
                        nd::Axis(0),
                        &[
                            chunked.jaw_pose.unwrap().view(),
                            chunked.left_eye_pose.unwrap().view(),
                            chunked.right_eye_pose.unwrap().view(),
                        ],
                    )
                    .unwrap();
                    full_head_pose.index_axis_mut(nd::Axis(0), time_idx).assign(&head_pose);
                }

                // Left hand
                if let Some(left_hand_pose) = chunked.left_hand_pose {
                    full_left_hand_pose.index_axis_mut(nd::Axis(0), time_idx).assign(&left_hand_pose);
                }

                // Right hand
                if let Some(right_hand_pose) = chunked.right_hand_pose {
                    full_right_hand_pose.index_axis_mut(nd::Axis(0), time_idx).assign(&right_hand_pose);
                }
            }

            // Put everything in the codec
            codec.frame_count = nr_frames as i32;
            codec.frame_rate = Some(anim.config.fps);
            codec.body_translation = Some(full_body_translation);
            codec.body_pose = Some(full_body_pose);
            codec.head_pose = Some(full_head_pose);
            codec.left_hand_pose = Some(full_left_hand_pose);
            codec.right_hand_pose = Some(full_right_hand_pose);
        }

        codec
    }

    fn to_entity_builder(&self) -> EntityBuilder {
        let mut builder = EntityBuilder::new();

        // Params
        let smpl_params = SmplParams::new_from_smpl_codec(self);
        info!("Found smpl_params in the .smpl file");
        builder.add(smpl_params);

        // Betas
        let betas = Betas::new_from_smpl_codec(self);
        if let Some(betas) = betas {
            info!("Found betas in the .smpl file");
            builder.add(betas);
        }

        // Anim or Pose
        match self.frame_count.cmp(&1) {
            Ordering::Greater => {
                let anim = Animation::new_from_smpl_codec(self, AnimWrap::default()).expect("The framecount is >1 so the animation should be valid");
                info!("Found animation in the .smpl file");
                builder.add(anim);
            }
            Ordering::Equal => {
                let pose = Pose::new_from_smpl_codec(self).expect("The framecount is =1 so the pose should be valid");
                info!("Found pose in the .smpl file");
                builder.add(pose);
            }
            Ordering::Less => {
                // There is no animation and no pose so there is nothing to do
                // here
            }
        }

        builder
    }
}
