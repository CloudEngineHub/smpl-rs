use burn::prelude::Backend;
use gloss_img::dynamic_image::DynImage;
use gloss_renderer::{
    components::{DiffuseImg, MetalnessImg, Name, NormalImg, RoughnessImg},
    scene::Scene,
};
use log::info;
use ndarray::{self as nd, s};
use smpl_rs::{
    codec::{gltf::PerBodyData, scene::CameraTrack},
    common::{
        animation::Animation, betas::Betas, expression::Expression, pose_override::PoseOverride, pose_retarget::RetargetPoseYShift,
        smpl_model::SmplCacheDynamic, smpl_options::SmplOptions, types::UpAxis,
    },
};
use utils_rs::{
    bshare::{ToNalgebraFloat, ToNalgebraInt, ToNdArray},
    nshare::ToNalgebra,
};

use smpl_rs::{
    codec::gltf::GltfCodec,
    common::{metadata::smpl_metadata, pose::Pose, smpl_model::SmplCache, smpl_params::SmplParams},
    conversions::pose_remap::PoseRemap,
};

use burn::backend::{Candle, NdArray, Wgpu};
use image::imageops::FilterType;
use nalgebra::DMatrix;
use smpl_rs::common::types::SmplType;
use smpl_utils::array::{Gather2D, Gather3D};
use std::f32::consts::PI;

use crate::scene::SceneAnimation;

/// Creates a ``GltfCodec`` from an entity by extracting components from it
pub trait GltfCodecGloss {
    fn from_scene(scene: &Scene, max_texture_size: Option<u32>, ignore_ent: Option<String>) -> GltfCodec;
}

fn get_image(image: &DynImage, to_gray: bool, max_texture_size: Option<u32>) -> DynImage {
    let mut image = image.clone();

    if to_gray {
        image = image.grayscale();
    }

    if let Some(force_image_size) = max_texture_size {
        if image.width() > force_image_size {
            image.resize(force_image_size, force_image_size, FilterType::Gaussian)
        } else {
            image
        }
    } else {
        image
    }
}

/// Trait implementation for ``GltfCodec``
impl GltfCodecGloss for GltfCodec {
    /// Get a ``GltfCodec`` from the scene
    fn from_scene(scene: &Scene, max_texture_size: Option<u32>, ignore_ent: Option<String>) -> GltfCodec {
        let smpl_models = scene.get_resource::<&SmplCacheDynamic>().unwrap();

        match &*smpl_models {
            SmplCacheDynamic::NdArray(models) => from_scene_on_backend::<NdArray>(scene, models, max_texture_size, &ignore_ent),
            SmplCacheDynamic::Wgpu(models) => from_scene_on_backend::<Wgpu>(scene, models, max_texture_size, &ignore_ent),
            SmplCacheDynamic::Candle(models) => from_scene_on_backend::<Candle>(scene, models, max_texture_size, &ignore_ent),
        }
    }
}

/// Function to get a ``GltfCodec`` from an entity on a generic Burn backend. We
/// currently support - ``Candle``, ``NdArray``, and ``Wgpu``
#[allow(clippy::too_many_lines)]
#[allow(clippy::trivially_copy_pass_by_ref)]
fn from_scene_on_backend<B: Backend>(
    scene: &Scene,
    smpl_models: &SmplCache<B>,
    max_texture_size: Option<u32>,
    ignore_ent: &Option<String>,
) -> GltfCodec
where
    <B as Backend>::FloatTensorPrimitive<2>: Sync, // TODO: Check if we still need these
    <B as Backend>::IntTensorPrimitive<2>: Sync,
    B::QuantizedTensorPrimitive<1>: std::marker::Sync,
    B::QuantizedTensorPrimitive<2>: std::marker::Sync,
    B::QuantizedTensorPrimitive<3>: std::marker::Sync,
{
    let now = wasm_timer::Instant::now();

    let mut gltf_codec = GltfCodec::default();

    let scene_anim = scene.get_resource::<&SceneAnimation>().unwrap();

    let nr_frames = scene_anim.num_frames;
    let fps = scene_anim.config.fps;

    // Add camera track to the codec
    let mut cameras_query = scene.world.query::<&CameraTrack>();

    for (_, camera_track) in cameras_query.iter() {
        gltf_codec.camera_track = Some(camera_track.clone());
    }

    // Any  with `SmplParams` is a candidate for export
    let mut query = scene.world.query::<(&SmplParams, &Name)>();
    let num_bodies = query.iter().len();
    gltf_codec.num_bodies = num_bodies;

    //figure out if we have to write pose correctives or expressions
    //check over all the bodies even if one of them has PoseCorrectives or Expression
    let mut should_export_posedirs = false;
    let mut should_export_exprdirs = false;
    let mut num_expression_blend_shapes = 0;
    for (entity, (smpl_params, _name)) in query.iter() {
        //pose dirs are exported if there is an animation and that has pose correctives enabled. If there is just a static pose, we bake the pose correctives
        if scene.world.has::<Animation>(entity).unwrap() && smpl_params.enable_pose_corrective {
            should_export_posedirs = true;
        }
        //expr dirs are exported if there is an animation with expressions and we have expression space in the model
        let smpl_model = smpl_models.get_model_ref(smpl_params.smpl_type, smpl_params.gender).unwrap();
        if let Ok(anim) = scene.get_comp::<&Animation>(&entity) {
            if anim.has_expression() && smpl_model.get_expression_dirs().is_some() {
                should_export_exprdirs = true;
                num_expression_blend_shapes = smpl_model.get_expression_dirs().unwrap().shape().dims[1];
            }
        }
    }
    // println!("should_export_posedirs {}", should_export_posedirs);
    // println!("should_export_exprdirs {}", should_export_exprdirs);

    // Add body entities to the codec
    for (body_idx, (entity, (smpl_params, name))) in query.iter().enumerate() {
        // We might want to ignore a certain entity
        if let Some(ent_to_ignore) = ignore_ent {
            if *ent_to_ignore == name.0 {
                continue; // Skip this entity
            }
        }
        let smpl_version = smpl_params.smpl_type;
        let gender = smpl_params.gender as i32;
        let mut current_body = PerBodyData::default();

        assert!(smpl_version != SmplType::SmplPP, "GLTF export for SMPL++ is not supported yet!");

        let smpl_model = smpl_models.get_model_ref(smpl_params.smpl_type, smpl_params.gender).unwrap();

        let Ok(betas) = scene.get_comp::<&Betas>(&entity) else {
            panic!("Betas component does not exist!");
        };

        let default_pose = Pose::new_empty(UpAxis::Y, smpl_params.smpl_type);
        let default_expression = Expression::new_empty(10);

        let mut smpl_output = smpl_model.forward(&SmplOptions::default(), &betas, &default_pose, Some(&default_expression));

        smpl_output.compute_normals();
        smpl_output = smpl_model.create_body_with_uv(&smpl_output);

        let metadata = smpl_metadata(&smpl_params.smpl_type);

        //figure out how many dimensions we write for the blendshape weights
        let mut num_total_blendshapes = 0;
        if should_export_posedirs {
            //we add +1 blendshape for the pose (explained in the section below where we export pose blendshapes)
            num_total_blendshapes += metadata.num_pose_blend_shapes + 1;
        }
        if should_export_exprdirs {
            num_total_blendshapes += num_expression_blend_shapes;
        }
        // println!("num_total_blendshapes {}", num_total_blendshapes);

        // Add metadata to the codec
        gltf_codec.smpl_type = smpl_version;
        gltf_codec.gender = gender;

        current_body.pose = Some(default_pose.clone());
        gltf_codec.default_joint_poses = Some(default_pose.clone().joint_poses);

        current_body.body_translation = Some(default_pose.clone().global_trans.to_shape((1, 3)).unwrap().to_owned());

        let verts_na = smpl_output.verts.to_nalgebra();
        let normals_na = smpl_output.normals.as_ref().expect("SMPL Output is missing normals!").to_nalgebra();
        let faces_na = smpl_output.faces.to_nalgebra();
        let uvs_na = smpl_output.uvs.as_ref().expect("SMPL Output is missing UVs!").to_nalgebra();

        // This is the info for T pose
        current_body.positions = Some(verts_na);
        current_body.normals = Some(normals_na);
        gltf_codec.faces = Some(faces_na);
        gltf_codec.uvs = Some(uvs_na);

        let smpl_joints = smpl_output.joints.clone().to_ndarray();
        let joint_count = smpl_joints.shape()[0];
        let lbs_weights = smpl_model.lbs_weights_split().to_ndarray();

        let vertex_count = smpl_output.verts.dims()[0];
        let mut skin_vertex_index = DMatrix::<u32>::zeros(vertex_count, 4);
        let mut skin_vertex_weight = DMatrix::<f32>::zeros(vertex_count, 4);

        // Process each vertex's weights
        for (vertex_id, row) in lbs_weights.outer_iter().enumerate() {
            let mut vertex_weights: Vec<(usize, f32)> = row.iter().enumerate().map(|(index, &weight)| (index, weight)).collect();

            // Sort weights by descending order of weight
            vertex_weights.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            // Ensure exactly 4 weights per vertex
            assert_eq!(vertex_weights.len().min(4), 4, "Illegal vertex weights");

            // Process up to 4 weights per vertex
            for (i, (index, weight)) in vertex_weights.iter().take(4).enumerate() {
                skin_vertex_index[(vertex_id, i)] = u32::try_from(*index).expect("Cannot convert to u32!");
                skin_vertex_weight[(vertex_id, i)] = *weight;
            }
        }

        gltf_codec.joint_index = Some(skin_vertex_index);
        gltf_codec.joint_weight = Some(skin_vertex_weight);

        // Add textures and materials to codec
        let diffuse_img = scene.get_comp::<&DiffuseImg>(&entity);
        if let Ok(diffuse_img) = diffuse_img {
            if let Some(img) = &diffuse_img.generic_img.cpu_img {
                current_body.diffuse_textures = Some(get_image(img, false, max_texture_size));
            }
        }

        let normals_img = scene.get_comp::<&NormalImg>(&entity);
        if let Ok(normals_img) = normals_img {
            if let Some(img) = &normals_img.generic_img.cpu_img {
                current_body.normals_textures = Some(get_image(img, false, max_texture_size));
            }
        }

        let metalness_img = scene.get_comp::<&MetalnessImg>(&entity);
        if let Ok(metalness_img) = metalness_img {
            if let Some(img) = &metalness_img.generic_img.cpu_img {
                current_body.metalness_textures = Some(get_image(img, true, max_texture_size));
            }
        }

        let roughness_img = scene.get_comp::<&RoughnessImg>(&entity);
        if let Ok(roughness_img) = roughness_img {
            if let Some(img) = &roughness_img.generic_img.cpu_img {
                current_body.roughness_textures = Some(get_image(img, true, max_texture_size));
            }
        }

        // Add pose info to codec. This is done for exporting static meshes without animations
        if scene.world.has::<Pose>(entity).unwrap() && !scene.world.has::<Animation>(entity).unwrap() {
            let Ok(pose_ref) = scene.get_comp::<&Pose>(&entity) else {
                panic!("Pose component doesn't exist");
            };
            let current_pose: &Pose = &pose_ref;
            let current_body_translation = current_pose.global_trans.to_shape((1, 3)).unwrap().to_owned();

            // This location of the Vec already has this for Default pose
            // We replace it only if we have to export a static scene.
            current_body.pose = Some(current_pose.clone());
            current_body.body_translation = Some(current_body_translation);

            // If pose correctives present and Animations are not, just add the offsets
            // to vertices; No need to deal with morph targets and weights
            // since we have only a single deformation
            if smpl_params.enable_pose_corrective {
                let vertex_offsets_merged = smpl_model.compute_pose_correctives(current_pose).to_ndarray();
                let mapping = &smpl_model.idx_split_2_merged_vec();
                let cols = vec![0, 1, 2];
                let vertex_offsets = vertex_offsets_merged.gather(mapping, &cols).into_nalgebra();
                current_body.positions = Some(current_body.positions.as_ref().unwrap() + vertex_offsets);
            }
        }

        // Add animation information if Animation component present
        #[allow(clippy::cast_precision_loss)]
        if scene.world.has::<Animation>(entity).unwrap() {
            info!("Processing Animation for body {:?}", body_idx);

            let anim = scene.get_comp::<&Animation>(&entity).unwrap();

            gltf_codec.frame_count = Some(nr_frames);

            // We make some dense matrices for animation data in the codec
            let mut keyframe_times: Vec<f32> = Vec::new();
            let mut current_body_rotations = nd::Array3::<f32>::zeros((joint_count, nr_frames, 3));
            let mut current_body_translations = nd::Array2::<f32>::zeros((nr_frames, 3));
            let mut current_body_scales = nd::Array2::<f32>::zeros((nr_frames, 3));
            let mut current_per_frame_blend_weights = nd::Array2::<f32>::zeros((nr_frames, num_total_blendshapes));

            /*
                ====================================== NOTE ======================================
                The values for morph targets and blend weights used below are a reformulation.
                This was done to work around certain limitations in GLTF. This reformulation is briefly described here

                At any frame the deformation of a shaped body is `T_frame = T_default + PoseBlendShapes.dot(BlendWeights_frame)`
                We normalise BlendWeights_frame to 0 -> 1 as `NewBlendWeights_frame = (BlendWeights_frame + PI) / 2PI` -- since we know its range
                Moving stuff around this gives `BlendWeights_frame = 2PI * NewBlendWeights_frame - PI`
                Replacing this in the top eqn we get `T_frame = T_default + PoseBlendShapes.dot(2PI * NewBlendWeights_frame - PI)`
                Expanding as `T_frame = T_default + 2PI * PoseBlendShapes.dot(NewBlendWeights_frame) - PoseBlendShapes.dot(PI)` where PI is an array of pi's
                                                the same shape as BlendWeights_frame
                This gives us `T_frame = (T_default - PoseBlendShapes.dot(PI)) + (2PI * PoseBlendShapes).dot(NewBlendWeights_frame)`
                The first bracketed term above is a constant and comparing to the first equation we get -
                    * NewT_default          = T_default - PoseBlendShapes.dot(PI)
                    * NewPoseBlendShapes    = 2PI * PoseBlendShapes
                    * NewBlendWeights_frame = (BlendWeights_frame + PI) / 2PI

                The only consideration now is that we need to add `-PoseBlendShapes.dot(PI)` to the template mesh
                This is done by making `PoseBlendShapes.dot(-PI)` a new morph target whose weights are always set to 1

                This sticks to the constraints of the GLTF spec while being mathematically equivalent
                ==================================================================================
            */

            if should_export_posedirs || should_export_exprdirs {
                let mut full_morph_targets = nd::Array3::<f32>::zeros((num_total_blendshapes, vertex_count, 3));

                let mut running_idx_morph_target = 0;

                // Add reformulated pose morph targets to the Codec
                if should_export_posedirs {
                    // 1 extra one for the template offset (Refer to note above)
                    let mut pose_morph_targets = nd::Array3::<f32>::zeros((metadata.num_pose_blend_shapes + 1, vertex_count, 3));

                    let nr_elem_merged = smpl_model.get_pose_dirs().dims()[0] / 3; // This is for the merged mesh - (10k * 3) x num_pose_blend_shapes
                    let pose_dirs_merged = smpl_model
                        .get_pose_dirs()
                        .to_ndarray()
                        .into_shape_with_order((nr_elem_merged, 3, metadata.num_pose_blend_shapes))
                        .unwrap();

                    let mapping = smpl_model.idx_split_2_merged_vec();
                    let cols = vec![0, 1, 2];
                    let depth = (0..metadata.num_pose_blend_shapes).collect::<Vec<_>>().into_boxed_slice();

                    // Get the original pose blend shapes in desired shape (split not merged)
                    let pose_blend_shapes = pose_dirs_merged
                        .gather(mapping, &cols, &depth)
                        .into_shape_with_order((vertex_count, 3, metadata.num_pose_blend_shapes))
                        .unwrap()
                        .permuted_axes([2, 0, 1]);

                    // Morph targets used are as described in the note above
                    let morph_targets = (2.0 * PI) * pose_blend_shapes.clone();

                    // This is the offset to be added to the template as mentioned above
                    let pi = nd::Array1::<f32>::from_elem(metadata.num_pose_blend_shapes, -PI);
                    let pi_array = pi.insert_axis(nd::Axis(1)).insert_axis(nd::Axis(2));
                    assert_eq!(pose_blend_shapes.shape()[0], pi_array.len());

                    // Does a dot product without reshaping multiple times
                    let template_offset = (pose_blend_shapes * &pi_array).sum_axis(nd::Axis(0));

                    pose_morph_targets
                        .slice_mut(s![0..metadata.num_pose_blend_shapes, .., ..])
                        .assign(&morph_targets);
                    pose_morph_targets
                        .slice_mut(s![metadata.num_pose_blend_shapes, .., ..])
                        .assign(&template_offset);

                    // println!("Final pose_morph_targets is {:?}", pose_morph_targets.shape());
                    // gltf_codec.morph_targets = Some(pose_morph_targets);

                    //set to the full morph targets
                    #[allow(clippy::range_plus_one)]
                    full_morph_targets
                        .slice_mut(s![
                            running_idx_morph_target..running_idx_morph_target + metadata.num_pose_blend_shapes + 1,
                            ..,
                            ..
                        ])
                        .assign(&pose_morph_targets);

                    running_idx_morph_target += metadata.num_pose_blend_shapes + 1;
                }

                //add also the expression space
                #[allow(unused_assignments)]
                if should_export_exprdirs {
                    if let Some(expr_dirs) = smpl_model.get_expression_dirs() {
                        // println!("num_expression_morph_targets {}", num_expression_blend_shapes);

                        // let mut expression_morph_targets = nd::Array3::<f32>::zeros((gltf_codec.num_expression_morph_targets, vertex_count, 3));

                        let nr_elem_merged = expr_dirs.dims()[0] / 3; // This is for the merged mesh - (10k * 3) x num_expression_blend_shapes
                        let expression_dirs_merged = expr_dirs
                            .to_ndarray()
                            .into_shape_with_order((nr_elem_merged, 3, num_expression_blend_shapes))
                            .unwrap();

                        //split the expression space because we are exporting mesh with UVs
                        let mapping = smpl_model.idx_split_2_merged_vec();
                        let cols = vec![0, 1, 2];
                        let depth = (0..metadata.expression_space_dim).collect::<Vec<_>>().into_boxed_slice();

                        // Get the original expression blend shapes in desired shape (split not merged)
                        //has shape nr_blend_shapes x 10K x 3
                        let expression_dirs_split = expression_dirs_merged
                            .gather(mapping, &cols, &depth)
                            .into_shape_with_order((vertex_count, 3, num_expression_blend_shapes))
                            .unwrap()
                            .permuted_axes([2, 0, 1]);

                        //done! assign to gltf
                        // println!("Final expression morph targets is {:?}", expression_dirs_split.shape());
                        // gltf_codec.expression_morph_targets = Some(expression_dirs_merged_split);

                        //set to the full morph targets
                        full_morph_targets
                            .slice_mut(s![
                                running_idx_morph_target..running_idx_morph_target + num_expression_blend_shapes,
                                ..,
                                ..
                            ])
                            .assign(&expression_dirs_split);

                        running_idx_morph_target += num_expression_blend_shapes;
                    }
                }

                //finally assign to codec
                gltf_codec.morph_targets = Some(full_morph_targets);
            }

            // Run through the animation frame by frame and get the poses, remap them to the
            // current model, mask them and retarget them and then finally we add it our
            // array. Do this only if within `SmplInterval`
            for global_frame_idx in 0..nr_frames {
                keyframe_times.push((global_frame_idx as f32) / fps);

                // Early continue if outside `SmplInterval`
                if global_frame_idx < anim.start_offset || global_frame_idx > anim.start_offset + anim.num_animation_frames() {
                    continue;
                }
                // local frame id for a single animaton
                let mut local_frame_idx = global_frame_idx - anim.start_offset;

                // The frame right after the interval copies the last frame animation data
                // This is done so that the last pose isnt interpolated with uninitialised values
                if global_frame_idx == (anim.start_offset + anim.num_animation_frames()) {
                    local_frame_idx -= 1;
                }

                let mut pose = anim.get_pose_at_idx(local_frame_idx);

                // 1. Remap
                let pose_remap = PoseRemap::new(pose.smpl_type, smpl_params.smpl_type);
                pose = pose_remap.remap(&pose);

                // 2. Mask
                if let Ok(ref pose_mask) = scene.get_comp::<&PoseOverride>(&entity) {
                    let mut new_pose_mask = PoseOverride::clone(pose_mask);
                    pose.apply_mask(&mut new_pose_mask);
                }

                // 3. Retarget
                if let Ok(ref pose_retarget) = scene.get_comp::<&RetargetPoseYShift>(&entity) {
                    let mut pose_retarget_local = RetargetPoseYShift::clone(pose_retarget); //we don't want to internally change the retarget because the saving shouldn't
                                                                                            // modify stuff in the world so we just clone a local one.
                    pose_retarget_local.apply(&mut pose);
                }

                // Joint translations don't change so we just need to animate joint rotations
                current_body_rotations.slice_mut(s![.., global_frame_idx, ..]).assign(&pose.joint_poses);

                // The only changing translation is the skeleton root translation
                let mut skeleton_root_translation = pose.global_trans.to_owned();
                let root_translation = smpl_output.joints.to_ndarray().slice(s![0, ..]).to_owned();

                skeleton_root_translation = skeleton_root_translation + root_translation;

                current_body_translations
                    .slice_mut(s![global_frame_idx, ..])
                    .assign(&skeleton_root_translation);

                if global_frame_idx < (anim.start_offset + anim.num_animation_frames()) {
                    current_body_scales.slice_mut(s![global_frame_idx, ..]).assign(&nd::Array1::ones(3));
                }

                // Add pose corrective vertex offsets info to codec
                let mut running_idx_morph_target = 0;
                if should_export_posedirs {
                    // Recalculate new normalised blend weights
                    let pose_blend_weights = &smpl_model.compute_pose_feature(&pose);

                    let rescaled_pose_blend_weights = pose_blend_weights.map(|&elem| (elem + PI) / (2.0 * PI));

                    current_per_frame_blend_weights
                        .slice_mut(s![global_frame_idx, 0..metadata.num_pose_blend_shapes])
                        .assign(&rescaled_pose_blend_weights);

                    if global_frame_idx == (anim.start_offset + anim.num_animation_frames()) {
                        current_per_frame_blend_weights
                            .slice_mut(s![global_frame_idx..nr_frames, 0..metadata.num_pose_blend_shapes])
                            .assign(&rescaled_pose_blend_weights);
                    }
                    running_idx_morph_target += metadata.num_pose_blend_shapes + 1;
                }
                //add blend weights for the expressions
                #[allow(unused_assignments)]
                if should_export_exprdirs {
                    let expr_opt = anim.get_expression_at_idx(local_frame_idx);
                    if let Some(expr) = expr_opt.as_ref() {
                        //get the coeffs, and slice from it because expression coeffs might be smaller than the total number of expression blendshapes
                        let max_nr_expr_coeffs = num_expression_blend_shapes.min(expr.expr_coeffs.len());
                        let expr_coeffs = expr.expr_coeffs.slice(s![0..max_nr_expr_coeffs]);
                        // println!("max_nr_expr_coeffs {}", max_nr_expr_coeffs);
                        // println!("expr_coeffs {:?}", expr_coeffs);

                        current_per_frame_blend_weights
                            .slice_mut(s![
                                global_frame_idx,
                                running_idx_morph_target..running_idx_morph_target + max_nr_expr_coeffs
                            ])
                            .assign(&expr_coeffs);
                    }
                    running_idx_morph_target += num_expression_blend_shapes;
                }
            }
            gltf_codec.keyframe_times = Some(keyframe_times);

            current_body.body_scales = Some(current_body_scales);
            current_body.body_translations = Some(current_body_translations);
            current_body.body_rotations = Some(current_body_rotations);

            if should_export_posedirs {
                // Set weights of the offset morph target to always be 1
                current_per_frame_blend_weights
                    .slice_mut(s![.., metadata.num_pose_blend_shapes])
                    .assign(&nd::Array1::<f32>::from_elem(nr_frames, 1.0));
            }
            if should_export_posedirs || should_export_exprdirs {
                current_body.per_frame_blend_weights = Some(current_per_frame_blend_weights);
            }
        }

        current_body.default_joint_translations = Some(smpl_joints);
        gltf_codec.per_body_data.push(current_body);
    }
    info!(
        "Writing {} body scene to GltfCodec: Took {} seconds for {} frames",
        num_bodies,
        now.elapsed().as_secs(),
        nr_frames
    );
    gltf_codec
}
