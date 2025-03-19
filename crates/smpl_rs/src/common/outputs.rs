use burn::{prelude::Backend, tensor::{Float, Int, Tensor}};
use gloss_renderer::geom::{Geom, PerVertexNormalsWeightingType};
/// Component for shaped and un-posed mesh. This would be the output of the
/// ``betas_to_verts`` system. This component is a generic over burn backend
#[derive(Clone)]
pub struct SmplOutputPoseTDynamic<B: Backend> {
    pub verts: Tensor<B, 2, Float>,
    pub verts_without_expression: Tensor<B, 2, Float>,
    pub joints: Tensor<B, 2, Float>,
}
/// Component for a posed mesh. This would be the output of the ``apply_pose``
/// system. This component is a generic over burn backend
#[derive(Clone)]
pub struct SmplOutputPosedDynamic<B: Backend> {
    pub joints: Tensor<B, 2, Float>,
    pub verts: Tensor<B, 2, Float>,
}
/// Component for the final shaped and posed mesh. This would be the output of
/// ``smpl_model.forward()`` This component is a generic over burn backend
#[derive(Clone)]
pub struct SmplOutputDynamic<B: Backend> {
    pub verts: Tensor<B, 2, Float>,
    pub faces: Tensor<B, 2, Int>,
    pub normals: Option<Tensor<B, 2, Float>>,
    pub uvs: Option<Tensor<B, 2, Float>>,
    pub joints: Tensor<B, 2, Float>,
}
impl<B: Backend> SmplOutputDynamic<B> {
    pub fn compute_normals(&mut self) {
        let normals = Geom::compute_per_vertex_normals_burn(
            &self.verts,
            &self.faces,
            &PerVertexNormalsWeightingType::Area,
        );
        self.normals = Some(normals);
    }
}
