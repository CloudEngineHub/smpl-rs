use burn::tensor::{backend::Backend, Float, Int, Tensor};
use gloss_utils::nshare::{RefNdarray2, ToNalgebra};
use nalgebra as na;
use nalgebra::clamp;
use ndarray as nd;
use ndarray::prelude::*;
use std::{
    f32::consts::PI,
    ops::{Div, SubAssign},
};
pub fn hex_to_rgb(hex: &str) -> (u8, u8, u8) {
    let hex = hex.trim_start_matches('#');
    let r = u8::from_str_radix(&hex[0..2], 16).unwrap_or(0);
    let g = u8::from_str_radix(&hex[2..4], 16).unwrap_or(0);
    let b = u8::from_str_radix(&hex[4..6], 16).unwrap_or(0);
    (r, g, b)
}
pub fn hex_to_rgb_f32(hex: &str) -> (f32, f32, f32) {
    let (r, g, b) = hex_to_rgb(hex);
    (f32::from(r) / 255.0, f32::from(g) / 255.0, f32::from(b) / 255.0)
}
pub fn interpolate_angle(cur_angle: f32, other_angle: f32, _cur_w: f32, other_w: f32) -> f32 {
    let mut diff = other_angle - cur_angle;
    if diff.abs() > PI {
        if diff > 0.0 {
            diff -= 2.0 * PI;
        } else {
            diff += 2.0 * PI;
        }
    }
    cur_angle + other_w * diff
}
pub fn interpolate_angle_tensor<B: Backend>(cur_angle: Tensor<B, 1>, other_angle: Tensor<B, 1>, _cur_w: f32, other_w: f32) -> Tensor<B, 1> {
    let mut diff = other_angle - cur_angle.clone();
    assert!(cur_angle.dims() == [1]);
    let abs_diff = diff.clone().abs();
    let needs_adjustment = abs_diff.greater_elem(PI);
    let two_pi = Tensor::<B, 1>::from_floats([2.0 * PI], &cur_angle.device());
    let neg_two_pi = Tensor::<B, 1>::from_floats([-2.0 * PI], &cur_angle.device());
    let positive_mask = diff.clone().greater_elem(0.0);
    let negative_mask = diff.clone().lower_elem(0.0);
    let pos_adjustment = positive_mask.clone().float() * neg_two_pi.clone();
    let neg_adjustment = negative_mask.clone().float() * two_pi.clone();
    let total_adjustment = pos_adjustment + neg_adjustment;
    let adjustment = needs_adjustment.float() * total_adjustment;
    diff = diff + adjustment;
    cur_angle + other_w * diff
}
pub fn axis_angle_to_quaternion<B: Backend>(axis_angle: Tensor<B, 2>) -> Tensor<B, 2> {
    let eps = 1e-6f32;
    let angle: Tensor<B, 1> = axis_angle.clone().powf_scalar(2.0).sum_dim(1).squeeze_dims(&[1]).clamp_min(1e-8).sqrt();
    let denom = angle.clone().unsqueeze_dim(1) + eps;
    let axis = axis_angle / denom;
    let half_angle = angle * 0.5;
    let cos_half = half_angle.clone().cos();
    let sin_half = half_angle.sin();
    let qxyz = axis.clone().slice([0..axis.dims()[0], 0..3]) * sin_half.clone().unsqueeze_dim(1);
    let qw = cos_half.unsqueeze_dim(1);
    Tensor::cat(vec![qxyz, qw], 1)
}
pub fn quaternion_to_axis_angle<B: Backend>(quat: Tensor<B, 2>) -> Tensor<B, 2> {
    let eps = 1e-6f32;
    let nr_rows = quat.dims()[0];
    let qxyz = quat.clone().slice([0..nr_rows, 0..3]);
    let qw = quat.slice([0..nr_rows, 3..4]).squeeze(1);
    let vec_norm = qxyz.clone().powf_scalar(2.0).sum_dim(1).clamp_min(1e-8).sqrt().squeeze_dims(&[1]);
    let abs_qw = qw.abs();
    let safe_qw = abs_qw.clone() + eps;
    let half_angle_tan = vec_norm.clone() / safe_qw;
    let small_rotation_mask = abs_qw.greater_elem(0.9);
    let small_angle_approx = 2.0 * vec_norm.clone();
    let x = half_angle_tan.clone();
    let atan_approx = x.clone() / (1.0 + 0.28 * x.powf_scalar(2.0));
    let large_angle_approx = 2.0 * atan_approx;
    let small_mask_float = small_rotation_mask.clone().float();
    let angle: Tensor<B, 1> = small_mask_float.clone() * small_angle_approx + (1.0 - small_mask_float) * large_angle_approx;
    let small_angle_mask = vec_norm.clone().lower_elem(eps);
    let safe_vec_norm = vec_norm.clone() + eps;
    let angle_over_norm = angle.unsqueeze_dim(1) / safe_vec_norm.unsqueeze_dim(1);
    let axis_angle = qxyz * angle_over_norm;
    let small_angle_mask_3d = small_angle_mask.float().unsqueeze_dim(1);
    (1.0 - small_angle_mask_3d) * axis_angle
}
pub fn quaternion_to_axis_angle_fast<B: Backend>(quat: Tensor<B, 2>) -> Tensor<B, 2> {
    let eps = 1e-6f32;
    let nr_rows = quat.dims()[0];
    let qxyz = quat.clone().slice([0..nr_rows, 0..3]);
    let qw: Tensor<B, 1> = quat.slice([0..nr_rows, 3..4]).squeeze(1);
    let w_negative_mask = qw.clone().lower_elem(0.0);
    let qw: Tensor<B, 1> = w_negative_mask.clone().float() * (-qw.clone()) + (1.0 - w_negative_mask.clone().float()) * qw.clone();
    let qxyz = w_negative_mask.clone().float().unsqueeze_dim(1) * (-qxyz.clone()) + (1.0 - w_negative_mask.float().unsqueeze_dim(1)) * qxyz.clone();
    let clamped_w = qw.clone().clamp(0.0, 1.0);
    let one_minus_w: Tensor<B, 1> = 1.0 - clamped_w.clone();
    let sqrt_term = one_minus_w.sqrt();
    let acos_w =
        sqrt_term * (1.570_728_8 + clamped_w.clone() * (-0.212_114_4 + clamped_w.clone() * (0.074_261_0 + clamped_w.clone() * -0.018_729_3)));
    let angle: Tensor<B, 1> = 2.0 * acos_w;
    let one_minus_square: Tensor<B, 1> = 1.0 - clamped_w.clone() * clamped_w;
    let sin_half_angle: Tensor<B, 1> = one_minus_square.sqrt();
    let denom = sin_half_angle + eps;
    let axis = qxyz / denom.unsqueeze_dim(1);
    axis * angle.unsqueeze_dim(1)
}
pub fn quaternion_interpolate_slerp<B: Backend>(lhs: Tensor<B, 2>, other: Tensor<B, 2>, other_weight: f32) -> Tensor<B, 2> {
    let eps = 1e-6f32;
    let lhs_norm = lhs.clone().powf_scalar(2.0).sum_dim(1).clamp_min(1e-8).sqrt() + eps;
    let other_norm = other.clone().powf_scalar(2.0).sum_dim(1).clamp_min(1e-8).sqrt() + eps;
    let lhs_normalized = lhs / lhs_norm;
    let other_normalized = other / other_norm;
    let dot: Tensor<B, 1> = (lhs_normalized.clone() * other_normalized.clone()).sum_dim(1).squeeze_dims(&[1]);
    let negative_dot_mask = dot.clone().lower_elem(0.0);
    let negative_dot_mask_float: Tensor<B, 1> = negative_dot_mask.clone().float();
    let dot_mask_float: Tensor<B, 1> = 1.0 - negative_dot_mask_float;
    let sign_corrected_other =
        negative_dot_mask.clone().float().unsqueeze_dim(1) * (-other_normalized.clone()) + dot_mask_float.unsqueeze_dim(1) * other_normalized.clone();
    let corrected_dot: Tensor<B, 1> = dot.clone().abs();
    let close_threshold = 0.9995f32;
    let very_close_mask = corrected_dot.clone().greater_elem(close_threshold);
    let lerp_result = lhs_normalized.clone() * (1.0 - other_weight) + sign_corrected_other.clone() * other_weight;
    let lerp_norm = lerp_result.clone().powf_scalar(2.0).sum_dim(1).clamp_min(1e-8).sqrt() + eps;
    let lerp_normalized = lerp_result / lerp_norm;
    let one_minus_dot_sq: Tensor<B, 1> = 1.0 - corrected_dot.clone().powf_scalar(2.0);
    let sqrt_term = one_minus_dot_sq.clamp_min(1e-8).sqrt();
    let safe_dot = corrected_dot.clone() + eps;
    let ratio = sqrt_term / safe_dot;
    let theta_approx: Tensor<B, 1> = ratio.clone() / (1.0 + 0.28 * ratio.clone().powf_scalar(2.0));
    let sin_theta = theta_approx.clone().sin();
    let safe_sin_theta = sin_theta.clone() + eps;
    let weight_lhs = ((1.0 - other_weight) * theta_approx.clone()).sin() / safe_sin_theta.clone();
    let weight_other = (other_weight * theta_approx).sin() / safe_sin_theta;
    let slerp_result = lhs_normalized.clone() * weight_lhs.unsqueeze_dim(1) + sign_corrected_other * weight_other.unsqueeze_dim(1);
    let inv_very_close_mask_float: Tensor<B, 1> = 1.0 - very_close_mask.clone().float();
    very_close_mask.clone().float().unsqueeze_dim(1) * lerp_normalized + inv_very_close_mask_float.unsqueeze_dim(1) * slerp_result
}
pub fn quaternion_interpolate_lerp<B: Backend>(lhs: Tensor<B, 2>, other: Tensor<B, 2>, other_weight: f32) -> Tensor<B, 2> {
    let eps = 1e-6f32;
    let dot: Tensor<B, 1> = (lhs.clone() * other.clone()).sum_dim(1).squeeze_dims(&[1]);
    let negative_dot_mask = dot.lower_elem(0.0);
    let negative_dot_mask_float: Tensor<B, 1> = negative_dot_mask.float();
    let positive_dot_mask_float: Tensor<B, 1> = 1.0 - negative_dot_mask_float.clone();
    let sign_corrected_other = negative_dot_mask_float.clone().unsqueeze_dim(1) * (-other.clone()) + positive_dot_mask_float.unsqueeze_dim(1) * other;
    let lerp_result = lhs * (1.0 - other_weight) + sign_corrected_other * other_weight;
    let lerp_norm_sq = lerp_result.clone().powf_scalar(2.0).sum_dim(1);
    lerp_result / (lerp_norm_sq.clamp_min(1e-8).sqrt() + eps)
}
pub fn map(value: f32, in_min: f32, in_max: f32, out_min: f32, out_max: f32) -> f32 {
    let value_clamped = clamp(value, in_min, in_max);
    out_min + (out_max - out_min) * (value_clamped - in_min) / (in_max - in_min)
}
pub fn smootherstep(low: f32, high: f32, val: f32) -> f32 {
    let t = map(val, low, high, 0.0, 1.0);
    t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
}
pub fn batch_rodrigues(full_pose: &nd::Array2<f32>) -> nd::Array3<f32> {
    let mut rotations_per_join = ndarray::Array3::<f32>::zeros((full_pose.shape()[0], 3, 3));
    for (idx, v) in full_pose.axis_iter(nd::Axis(0)).enumerate() {
        let angle = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        let rot_dir = full_pose.row(idx).to_owned().div(angle + 1e-6);
        let cos = angle.cos();
        let sin = angle.sin();
        let (rx, ry, rz) = (rot_dir[0], rot_dir[1], rot_dir[2]);
        let k = array![[0.0, -rz, ry], [rz, 0.0, -rx], [-ry, rx, 0.0]];
        let identity = ndarray::Array2::<f32>::eye(3);
        let rot_mat = identity + sin * k.clone() + (1.0 - cos) * k.dot(&k);
        rotations_per_join.slice_mut(s![idx, .., ..]).assign(&rot_mat);
    }
    rotations_per_join
}
#[allow(clippy::let_and_return)]
pub fn batch_rodrigues_burn<B: Backend>(full_pose: &Tensor<B, 2, Float>) -> Tensor<B, 3, Float> {
    let eps = Tensor::<B, 1, Float>::from_floats([1e-6], &full_pose.device());
    let angle = full_pose.clone().powf_scalar(2.0).sum_dim(1).clamp_min(1e-8).sqrt();
    let denom = angle.clone() + eps.unsqueeze_dim(0);
    let k = full_pose.clone() / denom;
    let kx: Tensor<B, 1> = k.clone().slice_dim(1, 0..1).squeeze(1);
    let ky: Tensor<B, 1> = k.clone().slice_dim(1, 1..2).squeeze(1);
    let kz: Tensor<B, 1> = k.clone().slice_dim(1, 2..3).squeeze(1);
    let zero: Tensor<B, 2> = Tensor::<B, 1, Float>::zeros_like(&kx).unsqueeze_dim(1);
    let k11 = zero.clone();
    let k12 = -kz.clone().unsqueeze_dim(1);
    let k13 = ky.clone().unsqueeze_dim(1);
    let k21 = kz.clone().unsqueeze_dim(1);
    let k22 = zero.clone();
    let k23 = -kx.clone().unsqueeze_dim(1);
    let k31 = -ky.clone().unsqueeze_dim(1);
    let k32 = kx.clone().unsqueeze_dim(1);
    let k33 = zero;
    let k_mat = Tensor::cat(
        vec![
            Tensor::cat(vec![k11, k12, k13], 1),
            Tensor::cat(vec![k21, k22, k23], 1),
            Tensor::cat(vec![k31, k32, k33], 1),
        ],
        1,
    )
    .reshape([-1, 3, 3]);
    let cos = angle.clone().cos().unsqueeze_dim(2);
    let sin = angle.clone().sin().unsqueeze_dim(2);
    let eye = Tensor::<B, 2, Float>::eye(3, &full_pose.device()).unsqueeze_dim(0);
    let eye = eye.repeat(&[full_pose.dims()[0], 1, 1]);
    let k_sq = k_mat.clone().matmul(k_mat.clone());
    let rot_mat = eye + sin * k_mat + (Tensor::ones_like(&cos) - cos) * k_sq;
    rot_mat
}
#[allow(clippy::let_and_return)]
pub fn batch_rodrigues_burn_2<B: Backend>(full_pose: &Tensor<B, 2, Float>) -> Tensor<B, 3, Float> {
    let eps = Tensor::<B, 1, Float>::from_floats([1e-6], &full_pose.device());
    let angle = full_pose.clone().powf_scalar(2.0).sum_dim(1).clamp_min(1e-8).sqrt().squeeze(1);
    let denom = angle.clone().unsqueeze_dim(1) + eps.unsqueeze_dim(0);
    let k = full_pose.clone() / denom;
    let kx: Tensor<B, 1> = k.clone().slice_dim(1, 0..1).squeeze(1);
    let ky: Tensor<B, 1> = k.clone().slice_dim(1, 1..2).squeeze(1);
    let kz: Tensor<B, 1> = k.clone().slice_dim(1, 2..3).squeeze(1);
    let cos = angle.clone().cos();
    let sin = angle.clone().sin();
    let one = Tensor::<B, 1, Float>::ones_like(&cos);
    let one_minus_cos = one.clone() - cos.clone();
    let r11 = cos.clone() + one_minus_cos.clone() * kx.clone() * kx.clone();
    let r12 = one_minus_cos.clone() * kx.clone() * ky.clone() - sin.clone() * kz.clone();
    let r13 = one_minus_cos.clone() * kx.clone() * kz.clone() + sin.clone() * ky.clone();
    let r21 = one_minus_cos.clone() * ky.clone() * kx.clone() + sin.clone() * kz.clone();
    let r22 = cos.clone() + one_minus_cos.clone() * ky.clone() * ky.clone();
    let r23 = one_minus_cos.clone() * ky.clone() * kz.clone() - sin.clone() * kx.clone();
    let r31 = one_minus_cos.clone() * kz.clone() * kx.clone() - sin.clone() * ky.clone();
    let r32 = one_minus_cos.clone() * kz.clone() * ky.clone() + sin.clone() * kx.clone();
    let r33 = cos.clone() + one_minus_cos.clone() * kz.clone() * kz.clone();
    let rot_mat = Tensor::stack(
        vec![
            Tensor::stack::<2>(vec![r11, r12, r13], 1),
            Tensor::stack::<2>(vec![r21, r22, r23], 1),
            Tensor::stack::<2>(vec![r31, r32, r33], 1),
        ],
        1,
    );
    rot_mat
}
#[allow(clippy::let_and_return)]
pub fn batch_rodrigues_burn_3<B: Backend>(full_pose: &Tensor<B, 2, Float>) -> Tensor<B, 3, Float> {
    let device = full_pose.device();
    let angle: Tensor<B, 1> = full_pose.clone().powi_scalar(2).sum_dim(1).clamp_min(1e-8).sqrt().squeeze(1);
    let denom = angle.clone().unsqueeze_dim(1) + 1e-6;
    let k = full_pose.clone() / denom;
    let k_3_1 = k.clone().unsqueeze_dim(2);
    let k_1_3 = k.clone().unsqueeze_dim(1);
    let kk_t = k_3_1 * k_1_3;
    let kx = k.clone().slice_dim(1, 0..1).squeeze(1);
    let ky = k.clone().slice_dim(1, 1..2).squeeze(1);
    let kz = k.clone().slice_dim(1, 2..3).squeeze(1);
    let zero = Tensor::<B, 1, Float>::zeros_like(&kx);
    let row1 = Tensor::stack::<2>(vec![zero.clone(), -kz.clone(), ky.clone()], 1);
    let row2 = Tensor::stack::<2>(vec![kz.clone(), zero.clone(), -kx.clone()], 1);
    let row3 = Tensor::stack::<2>(vec![-ky.clone(), kx.clone(), zero.clone()], 1);
    let k = Tensor::stack(vec![row1, row2, row3], 1);
    let cos: Tensor<B, 3> = angle.clone().cos().unsqueeze_dim::<2>(1).unsqueeze_dim(2);
    let sin: Tensor<B, 3> = angle.clone().sin().unsqueeze_dim::<2>(1).unsqueeze_dim(2);
    let one_minus_cos = 1.0 - cos.clone();
    let eye = Tensor::<B, 2, Float>::eye(3, &device).unsqueeze_dim(0);
    let rot = cos * eye + one_minus_cos * kk_t + sin * k;
    rot
}
pub fn euler2angleaxis(euler_x: f32, euler_y: f32, euler_z: f32) -> na::Vector3<f32> {
    let c1 = f32::cos(euler_x / 2.0);
    let c2 = f32::cos(euler_y / 2.0);
    let c3 = f32::cos(euler_z / 2.0);
    let s1: f32 = f32::sin(euler_x / 2.0);
    let s2 = f32::sin(euler_y / 2.0);
    let s3 = f32::sin(euler_z / 2.0);
    let rot = na::Quaternion::new(
        c1 * c2 * c3 - s1 * s2 * s3,
        s1 * c2 * c3 + c1 * s2 * s3,
        c1 * s2 * c3 - s1 * c2 * s3,
        c1 * c2 * s3 + s1 * s2 * c3,
    );
    let rot = na::UnitQuaternion::new_normalize(rot);
    rot.scaled_axis()
}
/// Interpolates between two axis angles using a slerp
pub fn interpolate_axis_angle(this_axis: &nd::Array1<f32>, other_axis: &nd::Array1<f32>, other_weight: f32) -> nd::Array1<f32> {
    let this_axis_na = this_axis.clone().into_nalgebra();
    let other_axis_na = other_axis.clone().into_nalgebra();
    let cur_r = na::Rotation3::new(this_axis_na.fixed_rows(0));
    let other_r = na::Rotation3::new(other_axis_na.fixed_rows(0));
    let new_r = cur_r.slerp(&other_r, other_weight);
    let axis_angle = new_r.scaled_axis();
    let new_axis_angle_nd = array![axis_angle.x, axis_angle.y, axis_angle.z];
    new_axis_angle_nd
}
/// Interpolates betwen batch of axis angles where the batch is shape
/// [``nr_joints``, 3]
pub fn interpolate_axis_angle_batch(this_axis: &nd::Array2<f32>, other_axis: &nd::Array2<f32>, other_weight: f32) -> nd::Array2<f32> {
    let this_axis_na = this_axis.clone().into_nalgebra();
    let other_axis_na = other_axis.clone().into_nalgebra();
    let mut new_axis_angles = nd::Array2::<f32>::zeros(this_axis_na.shape());
    for ((this_axis, other_axis), mut new_joint) in this_axis_na
        .row_iter()
        .zip(other_axis_na.row_iter())
        .zip(new_axis_angles.axis_iter_mut(nd::Axis(0)))
    {
        let cur_r = na::Rotation3::new(this_axis.transpose().fixed_rows(0));
        let other_r = na::Rotation3::new(other_axis.transpose().fixed_rows(0));
        let new_r = cur_r.slerp(&other_r, other_weight);
        let axis_angle = new_r.scaled_axis();
        new_joint.assign(&array![axis_angle.x, axis_angle.y, axis_angle.z]);
    }
    new_axis_angles
}
#[allow(clippy::missing_panics_doc)]
#[allow(clippy::similar_names)]
#[allow(clippy::cast_sign_loss)]
pub fn batch_rigid_transform(
    parent_idx_per_joint: Vec<u32>,
    rot_mats: &nd::Array3<f32>,
    joints: &nd::Array2<f32>,
    num_joints: usize,
) -> (nd::Array2<f32>, nd::Array3<f32>) {
    let mut rel_joints = joints.clone();
    let parent_idx_data_u32 = parent_idx_per_joint;
    let parent_idx_per_joint = nd::Array1::from_vec(parent_idx_data_u32);
    for (idx_cur, idx_parent) in parent_idx_per_joint.iter().enumerate().skip(1) {
        let parent_joint_position = joints.row(*idx_parent as usize);
        rel_joints.row_mut(idx_cur).sub_assign(&parent_joint_position);
    }
    let mut transforms_mat = ndarray::Array3::<f32>::zeros((num_joints + 1, 4, 4));
    for idx in 0..=num_joints {
        let rot = rot_mats.slice(s![idx, .., ..]).to_owned();
        let t = rel_joints.row(idx).to_owned();
        transforms_mat.slice_mut(s![idx, 0..3, 0..3]).assign(&rot);
        transforms_mat.slice_mut(s![idx, 0..3, 3]).assign(&t);
        transforms_mat.slice_mut(s![idx, 3, 0..4]).assign(&array![0.0, 0.0, 0.0, 1.0]);
    }
    let mut transform_chain = Vec::new();
    transform_chain.push(transforms_mat.slice(s![0, 0..4, 0..4]).to_owned().into_shape((4, 4)).unwrap());
    for i in 1..=num_joints {
        let mat_1 = &transform_chain[parent_idx_per_joint[[i]] as usize];
        let mat_2 = transforms_mat.slice(s![i, 0..4, 0..4]);
        let curr_res = mat_1.dot(&mat_2);
        transform_chain.push(curr_res);
    }
    let mut posed_joints = joints.clone();
    for (i, tf) in transform_chain.iter().enumerate() {
        let t = tf.slice(s![0..3, 3]);
        posed_joints.row_mut(i).assign(&t);
    }
    let mut rel_transforms = ndarray::Array3::<f32>::zeros((num_joints + 1, 4, 4));
    for (i, transform) in transform_chain.iter().enumerate() {
        let (jx, jy, jz) = (joints.row(i)[0], joints.row(i)[1], joints.row(i)[2]);
        let joint_homogen = array![jx, jy, jz, 0.0];
        let transformed_joint = transform.dot(&joint_homogen);
        let mut transformed_joint_4 = nd::Array2::<f32>::zeros((4, 4));
        transformed_joint_4.slice_mut(s![0..4, 3]).assign(&transformed_joint);
        transformed_joint_4 = transform - transformed_joint_4;
        rel_transforms.slice_mut(s![i, .., ..]).assign(&transformed_joint_4);
    }
    (posed_joints, rel_transforms)
}
/// Burn-only batch rigid transform
pub fn batch_rigid_transform_burn<B: Backend>(
    parent_idx_per_joint_t: Tensor<B, 1, Int>,
    parent_idx_per_joint: &nd::Array1<u32>,
    rot_mats: Tensor<B, 3>,
    joints: Tensor<B, 2>,
) -> (Tensor<B, 2>, Tensor<B, 3>) {
    let num_joints = joints.dims()[0];
    let parent_idx_per_joint_t = parent_idx_per_joint_t.slice_fill(0..1, 0);
    let parent_joints = joints.clone().select(0, parent_idx_per_joint_t);
    let rel_joints = joints.clone() - parent_joints;
    let rel_joints = rel_joints.slice_assign([0..1, 0..3], joints.clone().slice([0..1, 0..3]));
    let eye_row = Tensor::zeros([num_joints, 1, 4], &joints.device());
    let eye_row = eye_row.slice_fill([0..num_joints, 0..1, 3..4], 1.0);
    let t_col = rel_joints.reshape([num_joints, 3, 1]);
    let upper = Tensor::cat(vec![rot_mats, t_col], 2);
    let transforms = Tensor::cat(vec![upper, eye_row], 1);
    let mut transform_chain: Vec<Tensor<B, 2>> = Vec::new();
    #[allow(clippy::needless_range_loop)]
    #[allow(clippy::single_range_in_vec_init)]
    #[allow(clippy::range_plus_one)]
    for j in 0..num_joints {
        let parent = parent_idx_per_joint[j] as usize;
        let t_j = transforms.clone().slice([j..j + 1]);
        let t_j = t_j.squeeze(0);
        if j == 0 {
            transform_chain.push(t_j);
        } else {
            let parent_t = transform_chain[parent].clone();
            transform_chain.push(parent_t.matmul(t_j));
        }
    }
    let transform_chain = Tensor::stack(transform_chain, 0);
    let posed_joints = transform_chain.clone().slice([0..num_joints, 0..3, 3..4]).squeeze(2);
    let joints_homo = joints.pad((0, 1, 0, 0), 0.0);
    let joints_homo = joints_homo.unsqueeze_dim(2);
    let transformed_joint: Tensor<B, 2> = transform_chain.clone().matmul(joints_homo).squeeze(2);
    let mut transformed_joint_4 = Tensor::zeros_like(&transform_chain.clone());
    transformed_joint_4 = transformed_joint_4.slice_assign([0..num_joints, 0..4, 3..4], transformed_joint.unsqueeze_dim(2));
    let rel_transforms = transform_chain - transformed_joint_4;
    (posed_joints, rel_transforms)
}
/// Faster Burn implementation of `batch_rigid_transform` (single-skeleton version)
/// - `parent_idx_per_joint_t`: Tensor<Int> shape [J] (index tensor)
/// - `parent_idx_per_joint`: `ndarray::Array1`<u32> (cpu-side parent indices, same shape)
/// - `rot_mats`: Tensor shape [J,3,3]
/// - joints: Tensor shape [J,3]
///   instead of doing a sequential loop over the 55 joints to accumulate the transforms, we do log J iterations.
///   Assume the tree is like
///   root (0)
///   |
///   1
///   |
///   2
///   |
///   3
///   On the first iteration, each joint knows about its local transform
///   chain[0] = L[0]    (root, special case)
///   chain[1] = L[1]
///   chain[2] = L[2]
///   chain[3] = L[3]
///   Then we accumulate the transform to the parent
///   chain[j] = chain[parent[j]] · chain[j]
///   root (0)        chain[0] = L[0]
///   1               chain[1] = L[0]·L[1]
///   2               chain[2] = L[1]·L[2]
///   3               chain[3] = L[2]·L[3]
///   Then we accumulate the transform to the grandparent
///   chain[j] = chain[parent^2[j]] · chain[j]
///   root (0)        chain[0] = L[0]
///   1               chain[1] = L[0]·L[1]
///   2               chain[2] = L[0]·L[1]·L[2]
///   3               chain[3] = L[0]·L[1]·L[2]·L[3]
pub fn batch_rigid_transform_burn_fast<B: Backend>(
    mut parent_idx_per_joint_t: Tensor<B, 1, Int>,
    _parent_idx_per_joint: &nd::Array1<u32>,
    rot_mats: Tensor<B, 3>,
    joints: Tensor<B, 2>,
    kinematic_tree_depth: usize,
) -> (Tensor<B, 2>, Tensor<B, 3>) {
    let num_joints = joints.dims()[0];
    parent_idx_per_joint_t = parent_idx_per_joint_t.clone().slice_fill(0..1, 0);
    let parent_joints = joints.clone().select(0, parent_idx_per_joint_t.clone());
    let mut rel_joints = joints.clone() - parent_joints;
    rel_joints = rel_joints.slice_assign([0..1, 0..3], joints.clone().slice([0..1, 0..3]));
    let t_col = rel_joints.reshape([num_joints, 3, 1]);
    let upper = Tensor::cat(vec![rot_mats, t_col], 2);
    let mut eye_row = Tensor::zeros([num_joints, 1, 4], &joints.device());
    eye_row = eye_row.slice_fill([0..num_joints, 0..1, 3..4], 1.0);
    let transforms = Tensor::cat(vec![upper, eye_row], 1);
    let mut transform_chain = transforms.clone();
    let identity = Tensor::eye(4, &joints.device()).unsqueeze_dim(0);
    transform_chain = transform_chain.slice_assign([0..1, 0..4, 0..4], identity.clone());
    let mut parent_pow = parent_idx_per_joint_t.clone();
    #[allow(clippy::cast_possible_truncation)]
    #[allow(clippy::cast_sign_loss)]
    #[allow(clippy::cast_precision_loss)]
    let max_steps = if num_joints <= 1 {
        0usize
    } else {
        (kinematic_tree_depth as f32).log2().ceil() as usize
    };
    for _ in 0..max_steps {
        let parent_transforms = transform_chain.clone().select(0, parent_pow.clone());
        let new_chain = parent_transforms.matmul(transform_chain.clone());
        parent_pow = parent_pow.clone().select(0, parent_pow.clone());
        transform_chain = new_chain;
    }
    let root_transform = transforms.clone().slice([0..1, 0..4, 0..4]);
    let transform_chain = root_transform.matmul(transform_chain);
    let posed_joints = transform_chain.clone().slice([0..num_joints, 0..3, 3..4]).squeeze(2);
    let joints_homo = joints.pad((0, 1, 0, 0), 0.0).unsqueeze_dim(2);
    let transformed_joint: Tensor<B, 2> = transform_chain.clone().matmul(joints_homo).squeeze(2);
    let mut transformed_joint_4 = Tensor::zeros_like(&transform_chain.clone());
    transformed_joint_4 = transformed_joint_4.slice_assign([0..num_joints, 0..4, 3..4], transformed_joint.unsqueeze_dim(2));
    let rel_transforms = transform_chain - transformed_joint_4;
    (posed_joints, rel_transforms)
}
/// Converts a 2D array of quaternions of shape Nx4 (each row being a quaternion in format xyzw) and a 2D array of translations of shape Nx3 to extrinsics as an 3D array of Nx4x4
pub fn extract_extrinsics_from_rot_trans(translations: &ndarray::Array2<f32>, rotations: &ndarray::Array2<f32>) -> ndarray::Array3<f32> {
    let num_frames = translations.shape()[0].min(rotations.shape()[0]);
    let mut extrinsics = ndarray::Array3::<f32>::zeros((num_frames, 4, 4));
    for frame in 0..num_frames {
        let trans = nalgebra::Vector3::new(translations[(frame, 0)], translations[(frame, 1)], translations[(frame, 2)]);
        let quat = nalgebra::UnitQuaternion::new_normalize(nalgebra::Quaternion::new(
            rotations[(frame, 3)],
            rotations[(frame, 0)],
            rotations[(frame, 1)],
            rotations[(frame, 2)],
        ));
        let transform = nalgebra::Isometry3::from_parts(trans.into(), quat);
        let matrix_4x4 = transform.to_homogeneous();
        extrinsics.slice_mut(s![frame, .., ..]).assign(&matrix_4x4.ref_ndarray2());
    }
    extrinsics
}
pub fn compute_tree_depth(parent_idx_per_joint: &nd::Array1<u32>) -> usize {
    let mut max_depth = 0;
    for i in 0..parent_idx_per_joint.len() {
        let mut depth = 0;
        let mut current_idx = i;
        loop {
            let parent = parent_idx_per_joint[current_idx];
            if parent == 0 || parent >= u32::try_from(parent_idx_per_joint.len()).unwrap() {
                depth += 1;
                break;
            }
            depth += 1;
            current_idx = parent as usize;
        }
        max_depth = max_depth.max(depth);
    }
    max_depth
}
