use burn::prelude::Backend;
use gloss_renderer::geom::{Geom, PerVertexNormalsWeightingType};
use smpl_rs::common::outputs::SmplOutputDynamic;
use utils_rs::bshare::{ToBurn, ToNalgebraFloat, ToNalgebraInt};
/// Add some gloss specific functions
pub trait SmplOutputGloss {
    fn compute_normals(&mut self);
}
impl<B: Backend> SmplOutputGloss for SmplOutputDynamic<B> {
    /// Compute Normals for a ``SmplOutputDynamic`` component
    fn compute_normals(&mut self) {
        let v_na = self.verts.clone().into_nalgebra();
        let f_na = self.faces.clone().into_nalgebra();
        let normals = Geom::compute_per_vertex_normals(&v_na, &f_na, &PerVertexNormalsWeightingType::Area);
        self.normals = Some(normals.to_burn(&self.verts.device()));
    }
}
