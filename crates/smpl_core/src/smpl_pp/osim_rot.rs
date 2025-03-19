use burn::{prelude::Backend, tensor::Tensor};
use gloss_utils::bshare::tensor_to_data_float;
pub fn axis_angle_to_matrix<B: Backend>(axis_angle: Tensor<B, 1>) -> Tensor<B, 2> {
    let quaternion = axis_angle_to_quaternion(axis_angle);
    quaternion_to_matrix(&quaternion)
}
pub fn axis_angle_to_quaternion<B: Backend>(axis_angle: Tensor<B, 1>) -> Tensor<B, 1> {
    let angle = tensor_to_data_float(
        &axis_angle.clone().powi_scalar(2).sum_dim(0).sqrt(),
    )[0];
    let half_angle = angle * 0.5;
    let eps = 1e-6;
    let small_angle = angle.abs() < eps;
    let sin_half_angle_over_angle = if small_angle {
        0.5 - (angle * angle) / 48.0
    } else {
        half_angle.sin() / angle
    };
    let cos_half_angle = half_angle.cos();
    let quaternion = Tensor::cat(
        [
            Tensor::from_floats([cos_half_angle].as_slice(), &axis_angle.device()),
            axis_angle.mul_scalar(sin_half_angle_over_angle),
        ]
            .to_vec(),
        0,
    );
    quaternion
}
pub fn quaternion_to_matrix<B: Backend>(quaternions: &Tensor<B, 1>) -> Tensor<B, 2> {
    let r = tensor_to_data_float(&quaternions.clone().slice([0..1; 1]))[0];
    let i = tensor_to_data_float(&quaternions.clone().slice([1..2; 1]))[0];
    let j = tensor_to_data_float(&quaternions.clone().slice([2..3; 1]))[0];
    let k = tensor_to_data_float(&quaternions.clone().slice([3..4; 1]))[0];
    let two_s = 2.0 / (r * r + i * i + j * j + k * k);
    let mat_values = vec![
        1.0 - two_s * (j * j + k * k), two_s * (i * j - k * r), two_s * (i * k + j * r),
        two_s * (i * j + k * r), 1.0 - two_s * (i * i + k * k), two_s * (j * k - i * r),
        two_s * (i * k - j * r), two_s * (j * k + i * r), 1.0 - two_s * (i * i + j * j),
    ];
    Tensor::<B, 1>::from_floats(mat_values.as_slice(), &quaternions.device())
        .reshape([3, 3])
}
#[allow(clippy::single_range_in_vec_init)]
#[allow(clippy::range_plus_one)]
pub fn euler_angles_to_matrix<B: Backend>(
    euler_angles: &Tensor<B, 1>,
    convention: &str,
) -> Tensor<B, 2> {
    assert!(euler_angles.dims() [0] == 3, "Invalid input euler angles.");
    assert!(convention.len() == 3, "Convention must have 3 letters.");
    let mut matrices = vec![];
    for (i, c) in convention.chars().enumerate() {
        let angle = tensor_to_data_float(&euler_angles.clone().slice([i..i + 1]))[0];
        matrices
            .push(axis_angle_rotation(&c.to_string(), angle, &euler_angles.device()));
    }
    matrices.into_iter().reduce(Tensor::matmul).unwrap()
}
pub fn axis_angle_rotation<B: Backend>(
    axis: &str,
    angle: f32,
    device: &B::Device,
) -> Tensor<B, 2> {
    let (cos, sin) = (angle.cos(), angle.sin());
    let r_flat = match axis {
        "X" => vec![1.0, 0.0, 0.0, 0.0, cos, - sin, 0.0, sin, cos],
        "Y" => vec![cos, 0.0, sin, 0.0, 1.0, 0.0, - sin, 0.0, cos],
        "Z" => vec![cos, - sin, 0.0, sin, cos, 0.0, 0.0, 0.0, 1.0],
        _ => panic!("Invalid axis: {axis}"),
    };
    Tensor::<B, 1>::from_floats(r_flat.as_slice(), device).reshape([3, 3])
}
pub trait OsimJoint<B: Backend>: Send + Sync {
    fn q_to_translation(&self, q: Tensor<B, 1>) -> Tensor<B, 1> {
        Tensor::zeros([3], &q.device())
    }
    fn q_to_rot(&self, q: Tensor<B, 1>) -> Tensor<B, 2>;
    fn nb_dof(&self) -> usize;
}
impl<B: Backend> Clone for Box<dyn OsimJoint<B> + Send + Sync> {
    #[allow(unconditional_recursion)]
    fn clone(&self) -> Box<dyn OsimJoint<B> + Send + Sync> {
        self.clone()
    }
}
#[derive(Clone)]
pub struct CustomJoint<B: Backend> {
    axis: Tensor<B, 2>,
    axis_flip: Tensor<B, 1>,
    nb_dof: usize,
}
impl<B: Backend> CustomJoint<B> {
    pub fn new(axis: &[f32], axis_flip: &[f32], device: &B::Device) -> Self {
        let axis: Tensor<B, 2> = Tensor::<B, 1>::from_floats(axis, device)
            .reshape([axis.len() / 3, 3]);
        let axis_flip = Tensor::from_floats(axis_flip, device);
        Self {
            axis,
            axis_flip: axis_flip.clone(),
            nb_dof: axis_flip.dims()[0],
        }
    }
}
impl<B: Backend> OsimJoint<B> for CustomJoint<B>
where
    B::FloatTensorPrimitive<2>: Sync,
    B::FloatTensorPrimitive<1>: Sync,
    B::QuantizedTensorPrimitive<1>: std::marker::Sync,
    B::QuantizedTensorPrimitive<2>: std::marker::Sync,
{
    #[allow(clippy::single_range_in_vec_init)]
    #[allow(clippy::range_plus_one)]
    fn q_to_rot(&self, q: Tensor<B, 1>) -> Tensor<B, 2> {
        let mut rp: Tensor<B, 2> = Tensor::eye(3, &q.device());
        for i in 0..self.nb_dof {
            let axis = self.axis.clone().slice([i..i + 1, 0..3]).squeeze(0);
            let angle_axis = q
                .clone()
                .slice([i..i + 1])
                .mul(self.axis_flip.clone().slice([i..i + 1]))
                .mul(axis);
            let rp_i = axis_angle_to_matrix(angle_axis);
            rp = rp_i.matmul(rp);
        }
        rp
    }
    fn nb_dof(&self) -> usize {
        self.nb_dof
    }
}
#[derive(Clone)]
pub struct CustomJoint1D<B: Backend> {
    axis: Tensor<B, 1>,
    axis_flip: Tensor<B, 1>,
    nb_dof: usize,
}
impl<B: Backend> CustomJoint1D<B> {
    #[allow(clippy::cast_precision_loss)]
    pub fn new(axis: &[f32], axis_flip: &[f32], device: &B::Device) -> Self {
        let axis = Tensor::from_floats(axis, device)
            .div(Tensor::from_floats(vec![axis.len() as f32].as_slice(), device));
        let axis_flip = Tensor::from_floats(axis_flip, device);
        Self { axis, axis_flip, nb_dof: 1 }
    }
}
impl<B: Backend> OsimJoint<B> for CustomJoint1D<B>
where
    B::FloatTensorPrimitive<2>: Sync,
    B::FloatTensorPrimitive<1>: Sync,
    B::QuantizedTensorPrimitive<1>: std::marker::Sync,
    B::QuantizedTensorPrimitive<2>: std::marker::Sync,
{
    #[allow(clippy::single_range_in_vec_init)]
    #[allow(clippy::range_plus_one)]
    fn q_to_rot(&self, q: Tensor<B, 1>) -> Tensor<B, 2> {
        let angle_axis = q
            .slice([0..1])
            .mul(self.axis_flip.clone())
            .mul(self.axis.clone());
        axis_angle_to_matrix(angle_axis)
    }
    fn nb_dof(&self) -> usize {
        self.nb_dof
    }
}
#[derive(Clone)]
pub struct WalkerKnee {
    nb_dof: usize,
}
impl WalkerKnee {
    pub fn new() -> Self {
        Self { nb_dof: 1 }
    }
}
impl Default for WalkerKnee {
    fn default() -> Self {
        Self::new()
    }
}
impl<B: Backend> OsimJoint<B> for WalkerKnee {
    #[allow(clippy::single_range_in_vec_init)]
    #[allow(clippy::range_plus_one)]
    fn q_to_rot(&self, q: Tensor<B, 1>) -> Tensor<B, 2> {
        let mut theta_i = Tensor::zeros([3], &q.device());
        theta_i = theta_i.slice_assign([2..3], q.slice([0..1]).neg());
        axis_angle_to_matrix(theta_i)
    }
    fn nb_dof(&self) -> usize {
        self.nb_dof
    }
}
#[derive(Clone)]
pub struct PinJoint<B: Backend> {
    parent_frame_ori: Tensor<B, 1>,
    nb_dof: usize,
}
impl<B: Backend> PinJoint<B> {
    pub fn new(parent_frame_ori: &[f32], device: &B::Device) -> Self {
        let parent_frame_ori = Tensor::from_floats(parent_frame_ori, device);
        Self {
            parent_frame_ori,
            nb_dof: 1,
        }
    }
}
impl<B: Backend> OsimJoint<B> for PinJoint<B>
where
    B::FloatTensorPrimitive<2>: Sync,
    B::FloatTensorPrimitive<1>: Sync,
    B::QuantizedTensorPrimitive<1>: std::marker::Sync,
{
    #[allow(clippy::single_range_in_vec_init)]
    #[allow(clippy::range_plus_one)]
    fn q_to_rot(&self, q: Tensor<B, 1>) -> Tensor<B, 2> {
        let ra_i = euler_angles_to_matrix(&self.parent_frame_ori.clone(), "XYZ");
        let z_axis = Tensor::<
            B,
            1,
        >::from_floats([0.0, 0.0, 1.0].as_slice(), &q.device());
        let axis = ra_i.matmul(z_axis.unsqueeze_dim(1)).squeeze(1);
        let axis_angle = q.slice([0..1]).mul(axis);
        axis_angle_to_matrix(axis_angle)
    }
    fn nb_dof(&self) -> usize {
        self.nb_dof
    }
}
#[derive(Clone)]
pub struct ConstantCurvatureJoint<B: Backend> {
    custom_joint: CustomJoint<B>,
}
impl<B: Backend> ConstantCurvatureJoint<B> {
    pub fn new(axis: &[f32], axis_flip: &[f32], device: &B::Device) -> Self {
        Self {
            custom_joint: CustomJoint::new(axis, axis_flip, device),
        }
    }
}
impl<B: Backend> OsimJoint<B> for ConstantCurvatureJoint<B>
where
    B::FloatTensorPrimitive<2>: Sync,
    B::FloatTensorPrimitive<1>: Sync,
    B::QuantizedTensorPrimitive<1>: std::marker::Sync,
    B::QuantizedTensorPrimitive<2>: std::marker::Sync,
{
    fn q_to_rot(&self, q: Tensor<B, 1>) -> Tensor<B, 2> {
        self.custom_joint.q_to_rot(q)
    }
    fn nb_dof(&self) -> usize {
        self.custom_joint.nb_dof()
    }
}
#[derive(Clone)]
pub struct EllipsoidJoint<B: Backend> {
    custom_joint: CustomJoint<B>,
}
impl<B: Backend> EllipsoidJoint<B> {
    pub fn new(axis: &[f32], axis_flip: &[f32], device: &B::Device) -> Self {
        Self {
            custom_joint: CustomJoint::new(axis, axis_flip, device),
        }
    }
}
impl<B: Backend> OsimJoint<B> for EllipsoidJoint<B>
where
    B::FloatTensorPrimitive<2>: Sync,
    B::FloatTensorPrimitive<1>: Sync,
    B::QuantizedTensorPrimitive<1>: std::marker::Sync,
    B::QuantizedTensorPrimitive<2>: std::marker::Sync,
{
    fn q_to_rot(&self, q: Tensor<B, 1>) -> Tensor<B, 2> {
        self.custom_joint.q_to_rot(q)
    }
    fn nb_dof(&self) -> usize {
        self.custom_joint.nb_dof()
    }
}
