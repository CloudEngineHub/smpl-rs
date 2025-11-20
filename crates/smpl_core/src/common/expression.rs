use crate::{common::types::FaceType, AppBackend};
use burn::{
    prelude::Backend,
    tensor::{Float, Tensor},
};
use gloss_utils::bshare::ToBurn;
use log::warn;
use ndarray as nd;
/// Component for Smpl Expressions or Expression Parameters
#[derive(Clone)]
pub struct ExpressionG<B: Backend> {
    pub device: B::Device,
    pub expr_coeffs: Tensor<B, 1>,
    pub expr_type: FaceType,
}
impl<B: Backend> Default for ExpressionG<B> {
    fn default() -> Self {
        let device = B::Device::default();
        let num_coeffs = 10;
        let expr_coeffs = Tensor::<B, 1>::zeros([num_coeffs], &device);
        Self {
            device,
            expr_coeffs,
            expr_type: FaceType::SmplX,
        }
    }
}
impl<B: Backend> ExpressionG<B> {
    pub fn new(expr_coeffs: Tensor<B, 1>, expr_type: FaceType) -> Self {
        Self {
            device: expr_coeffs.device(),
            expr_coeffs,
            expr_type,
        }
    }
    pub fn new_empty(num_coeffs: usize, expr_type: FaceType) -> Self {
        let device = B::Device::default();
        let expr_coeffs = Tensor::<B, 1>::zeros([num_coeffs], &device);
        Self {
            device,
            expr_coeffs,
            expr_type,
        }
    }
    pub fn new_from_ndarray(expr_coeffs: nd::Array1<f32>, expr_type: FaceType) -> Self {
        let device = B::Device::default();
        Self::new(expr_coeffs.into_burn(&device), expr_type)
    }
    #[must_use]
    pub fn interpolate(&self, other_pose: &Self, other_weight: f32) -> Self {
        if !(0.0..=1.0).contains(&other_weight) {
            warn!("pose interpolation weight is outside the [0,1] range, will clamp. Weight is {other_weight}");
        }
        let other_weight = other_weight.clamp(0.0, 1.0);
        let cur_w = 1.0 - other_weight;
        let new_expression = cur_w * self.expr_coeffs.clone() + other_weight * other_pose.expr_coeffs.clone();
        Self::new(new_expression, self.expr_type)
    }
}
/// ``ExpressionOffsets`` is the result of smpl.expression2offsets(expression)
/// which contains vertex offset for that expression
#[derive(Clone)]
pub struct ExpressionOffsetsG<B: Backend> {
    pub offsets: Tensor<B, 2, Float>,
}
pub type ExpressionOffsets = ExpressionOffsetsG<AppBackend>;
pub type Expression = ExpressionG<AppBackend>;
