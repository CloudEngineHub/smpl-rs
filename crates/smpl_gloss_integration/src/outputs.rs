use burn::prelude::Backend;
use gloss_geometry::geom::{self, PerVertexNormalsWeightingType};
use gloss_utils::bshare::{ToBurn, ToNalgebraFloat, ToNalgebraInt};
use smpl_core::common::outputs::SmplOutputDynamic;
/// Add some gloss specific functions
pub trait SmplOutputGloss {
    fn compute_normals(&mut self);
}
impl<B: Backend> SmplOutputGloss for SmplOutputDynamic<B> {
    /// Compute Normals for a ``SmplOutputDynamic`` component
    fn compute_normals(&mut self) {
        let v_na = self.verts.clone().into_nalgebra();
        let f_na = self.faces.clone().into_nalgebra();
        let normals = geom::compute_per_vertex_normals(&v_na, &f_na, &PerVertexNormalsWeightingType::Area);
        self.normals = Some(normals.to_burn(&self.verts.device()));
    }
}
