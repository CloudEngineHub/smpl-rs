use crate::common::{
    betas::PyBetas,
    expression::PyExpression,
    outputs::PySmplOutput,
    pose::PyPose,
    smpl_options::PySmplOptions,
    types::{PyGender, PySmplType},
};
use gloss_hecs::Entity;
use gloss_py_macros::PyComponent;
use gloss_renderer::scene::Scene;
use gloss_utils::bshare::ToNdArray;
use numpy::{PyArray1, PyArray2, ToPyArray};
use pyo3::prelude::*;
use smpl_core::common::smpl_model::SmplModel;
use smpl_core::{common::types::Gender, smpl_x::smpl_x_gpu::SmplXGPU};
#[pyclass(name = "SmplX", module = "smpl_rs.models", unsendable)]
#[derive(Clone, PyComponent)]
pub struct PySmplX {
    pub inner: SmplXGPU,
}
#[pymethods]
impl PySmplX {
    #[staticmethod]
    #[pyo3(
        signature = (
            path,
            pygender,
            max_num_betas = None,
            max_expression_blendshapes = None
        )
    )]
    #[pyo3(
        text_signature = "(path: str, gender: Gender, max_num_betas: Optional[int] = None, max_expression_blendshapes: Optional[int] = None) -> SmplX"
    )]
    pub fn from_npz(path: &str, pygender: PyGender, max_num_betas: Option<u32>, max_expression_blendshapes: Option<u32>) -> Self {
        Self {
            inner: SmplXGPU::new_from_npz(
                path,
                Gender::from(pygender),
                max_num_betas.unwrap_or(300) as usize,
                max_expression_blendshapes.unwrap_or(100) as usize,
            ),
        }
    }
    #[allow(clippy::type_complexity)]
    #[pyo3(signature = (options, betas, pose, expression = None))]
    #[pyo3(text_signature = "($self, options: SmplOptions, betas: Betas, pose: Pose, expression: Optional[Expression] = None) -> SmplOutput")]
    pub fn forward(&mut self, options: &PySmplOptions, betas: &PyBetas, pose: &PyPose, expression: Option<&PyExpression>) -> PySmplOutput {
        let smpl_output = self
            .inner
            .forward(&options.inner, &betas.inner, &pose.inner, expression.map(|x| &x.inner));
        PySmplOutput { inner: smpl_output }
    }
    #[pyo3(text_signature = "($self, smpl_merged: SmplOutput) -> SmplOutput")]
    pub fn create_body_with_uv(&mut self, py_smpl_merged: &PySmplOutput) -> PySmplOutput {
        let smpl_output = self.inner.create_body_with_uv(&py_smpl_merged.inner);
        PySmplOutput { inner: smpl_output }
    }
    #[getter]
    pub fn smpl_type(&self) -> PySmplType {
        self.inner.smpl_type.into()
    }
    #[getter]
    pub fn gender(&self) -> PyGender {
        self.inner.gender.into()
    }
    #[getter]
    pub fn verts_template(&self, py: Python<'_>) -> Py<PyArray2<f32>> {
        self.inner.verts_template.to_ndarray().to_pyarray_bound(py).into()
    }
    #[getter]
    pub fn faces(&self, py: Python<'_>) -> Py<PyArray2<u32>> {
        self.inner.faces.to_ndarray().to_pyarray_bound(py).into()
    }
    #[getter]
    pub fn uv(&self, py: Python<'_>) -> Py<PyArray2<f32>> {
        self.inner.uv.to_ndarray().to_pyarray_bound(py).into()
    }
    #[getter]
    pub fn shape_dirs(&self, py: Python<'_>) -> Py<PyArray2<f32>> {
        self.inner.shape_dirs.to_ndarray().to_pyarray_bound(py).into()
    }
    #[getter]
    pub fn pose_dirs(&self, py: Python<'_>) -> Option<Py<PyArray2<f32>>> {
        self.inner.pose_dirs.as_ref().map(|v| v.to_ndarray().to_pyarray_bound(py).into())
    }
    #[getter]
    pub fn joint_regressor(&self, py: Python<'_>) -> Py<PyArray2<f32>> {
        self.inner.joint_regressor.to_ndarray().to_pyarray_bound(py).into()
    }
    #[getter]
    pub fn parent_idx_per_joint(&self, py: Python<'_>) -> Py<PyArray1<u32>> {
        self.inner.parent_idx_per_joint.to_ndarray().to_pyarray_bound(py).into()
    }
    #[getter]
    pub fn lbs_weights(&self, py: Python<'_>) -> Py<PyArray2<f32>> {
        self.inner.lbs_weights.to_ndarray().to_pyarray_bound(py).into()
    }
    #[getter]
    pub fn lbs_weights_split(&self, py: Python<'_>) -> Py<PyArray2<f32>> {
        self.inner.lbs_weights_split.to_ndarray().to_pyarray_bound(py).into()
    }
    #[getter]
    pub fn idx_split_2_merged(&self, py: Python<'_>) -> Py<PyArray1<usize>> {
        self.inner.idx_vuv_2_vnouv_vec.to_pyarray_bound(py).into()
    }
}
