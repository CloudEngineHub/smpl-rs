use crate::common::types::PyFaceType;
use gloss_hecs::Entity;
use gloss_py_macros::PyComponent;
use gloss_renderer::scene::Scene;
use ndarray as nd;
use numpy::PyArrayMethods;
use numpy::{PyArray1, PyReadonlyArray1, ToPyArray};
use pyo3::prelude::*;
use smpl_core::common::expression::Expression;
use smpl_core::common::types::FaceType;
#[pyclass(name = "Expression", module = "smpl_rs.components", unsendable)]
#[derive(Clone, PyComponent)]
pub struct PyExpression {
    pub inner: Expression,
}
#[pymethods]
impl PyExpression {
    #[new]
    #[pyo3(signature = (array, face_type = None))]
    #[pyo3(text_signature = "(array: NDArray[np.float32], face_type: Optional[FaceType] = None) -> Expression")]
    pub fn new(array: PyReadonlyArray1<f32>, face_type: Option<PyFaceType>) -> Self {
        let expr_coeffs: nd::Array1<f32> = array.to_owned_array();
        let face_type = face_type.map_or(FaceType::SmplX, FaceType::from);
        Self {
            inner: Expression {
                expr_coeffs,
                expr_type: face_type,
            },
        }
    }
    #[staticmethod]
    #[pyo3(signature = (num_coeffs, face_type = None))]
    #[pyo3(text_signature = "(num_coeffs: int, face_type: Optional[FaceType] = None) -> Expression")]
    pub fn new_empty(num_coeffs: usize, face_type: Option<PyFaceType>) -> Self {
        let face_type = face_type.map_or(FaceType::SmplX, FaceType::from);
        Self {
            inner: Expression::new_empty(num_coeffs, face_type),
        }
    }
    #[staticmethod]
    #[pyo3(text_signature = "() -> Expression")]
    #[allow(clippy::should_implement_trait)]
    pub fn default() -> Self {
        Self {
            inner: Expression::new_empty(1, FaceType::SmplX),
        }
    }
    #[pyo3(text_signature = "($self) -> NDArray[np.float32]")]
    pub fn numpy(&mut self, py: Python<'_>) -> Py<PyArray1<f32>> {
        self.inner.expr_coeffs.to_pyarray_bound(py).into()
    }
}
