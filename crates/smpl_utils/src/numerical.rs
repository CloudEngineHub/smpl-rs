use nalgebra as na;
use nalgebra::clamp;
use ndarray as nd;
use ndarray::prelude::*;
use std::{
    f32::consts::PI,
    ops::{Div, SubAssign},
};
use utils_rs::nshare::ToNalgebra;
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
pub fn euler2angleaxis(euler_x: f32, euler_y: f32, euler_z: f32) -> na::Vector3<f32> {
    let c1 = f32::cos(euler_x / 2.0);
    let c2 = f32::cos(euler_y / 2.0);
    let c3 = f32::cos(euler_z / 2.0);
    let s1 = f32::sin(euler_x / 2.0);
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
    transform_chain.push(transforms_mat.slice(s![0, 0..4, 0..4]).to_owned().into_shape_with_order((4, 4)).unwrap());
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
