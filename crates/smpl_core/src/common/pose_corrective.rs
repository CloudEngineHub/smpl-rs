use crate::AppBackend;
use burn::{
    prelude::Backend,
    tensor::{Float, Tensor},
};
/// Component for Pose corrective vertex offsets. This component is generic over
/// burn backend.
pub struct PoseCorrectiveG<B: Backend> {
    pub verts_offset: Tensor<B, 2, Float>,
}
pub type PoseCorrective = PoseCorrectiveG<AppBackend>;
