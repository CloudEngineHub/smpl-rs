use burn::{
    prelude::Backend,
    tensor::{Float, Tensor},
};
use ndarray as nd;
/// Component for Pose corrective vertex offsets
pub struct PoseCorrective {
    pub verts_offset: nd::Array2<f32>,
}
/// Component for Pose corrective vertex offsets. This component is generic over
/// burn backend.
pub struct PoseCorrectiveDynamic<B: Backend> {
    pub verts_offset: Tensor<B, 2, Float>,
}
