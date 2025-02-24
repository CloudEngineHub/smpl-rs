use burn::{
    prelude::Backend,
    tensor::{Float, Tensor},
};
use log::warn;
use ndarray as nd;

/// Component for Smpl Expressions or Expression Parameters
#[derive(Clone)]
pub struct Expression {
    pub expr_coeffs: nd::Array1<f32>,
}
impl Default for Expression {
    fn default() -> Self {
        let num_coeffs = 10; //TODO parametrize this
        let expr_coeffs = ndarray::Array1::<f32>::zeros(num_coeffs);
        Self { expr_coeffs }
    }
}

impl Expression {
    pub fn new(expr_coeffs: nd::Array1<f32>) -> Self {
        Self { expr_coeffs }
    }

    pub fn new_empty(num_coeffs: usize) -> Self {
        let expr_coeffs = ndarray::Array1::<f32>::zeros(num_coeffs);
        Self { expr_coeffs }
    }

    #[must_use]
    pub fn interpolate(&self, other_pose: &Self, other_weight: f32) -> Self {
        if !(0.0..=1.0).contains(&other_weight) {
            warn!("pose interpolation weight is outside the [0,1] range, will clamp. Weight is {other_weight}");
        }
        let other_weight = other_weight.clamp(0.0, 1.0);

        let cur_w = 1.0 - other_weight;

        //solo similar for the expression
        let new_expression = cur_w * &self.expr_coeffs + other_weight * &other_pose.expr_coeffs;

        Self::new(new_expression)
    }
}

/// ``ExpressionOffsets`` is the result of smpl.expression2offsets(expression)
/// which contains vertex offset for that expression
#[derive(Clone)]
pub struct ExpressionOffsets<B: Backend> {
    pub offsets: Tensor<B, 2, Float>,
}
