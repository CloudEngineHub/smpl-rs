use crate::common::burn_tensor::BurnTensorType;
use crate::common::burn_tensor::PyBurnTensor;
use gloss_burn_multibackend::backend::MultiDevice;
use gloss_hecs::Entity;
use gloss_py_macros::PyComponent;
use gloss_renderer::scene::Scene;
use gloss_utils::bshare::{ToBurn, ToNdArray};
use numpy::{PyArray2, PyArrayMethods, PyReadonlyArray2, ToPyArray};
use pyo3::prelude::*;
use smpl_core::common::outputs::{SmplOutput, SmplOutputPoseT, SmplOutputPosed};
#[pyclass(name = "SmplOutputPoseT", module = "smpl_rs.models", unsendable)]
#[derive(Clone, PyComponent)]
pub struct PySmplOutputPoseT {
    pub inner: SmplOutputPoseT,
}
#[pymethods]
impl PySmplOutputPoseT {
    #[getter]
    pub fn verts(&mut self, py: Python<'_>) -> Py<PyArray2<f32>> {
        self.inner.verts.to_ndarray().to_pyarray_bound(py).into()
    }
    #[pyo3(text_signature = "($self) -> NDArray[np.float32]")]
    pub fn verts_without_expression(&mut self, py: Python<'_>) -> Py<PyArray2<f32>> {
        self.inner.verts_without_expression.to_ndarray().to_pyarray_bound(py).into()
    }
}
#[pyclass(name = "SmplOutputPosed", module = "smpl_rs.models", unsendable)]
#[derive(Clone, PyComponent)]
pub struct PySmplOutputPosed {
    pub inner: SmplOutputPosed,
}
#[pymethods]
impl PySmplOutputPosed {
    #[getter]
    pub fn joints(&mut self, py: Python<'_>) -> Py<PyArray2<f32>> {
        self.inner.joints.to_ndarray().to_pyarray_bound(py).into()
    }
    #[getter]
    pub fn verts(&mut self, py: Python<'_>) -> Py<PyArray2<f32>> {
        self.inner.verts.to_ndarray().to_pyarray_bound(py).into()
    }
}
#[pyclass(name = "SmplOutput", module = "smpl_rs.models", unsendable)]
#[derive(Clone, PyComponent)]
pub struct PySmplOutput {
    pub inner: SmplOutput,
}
#[pymethods]
impl PySmplOutput {
    #[getter]
    pub fn verts(&mut self) -> PyBurnTensor {
        PyBurnTensor {
            inner: BurnTensorType::FloatDim2(self.inner.verts.clone()),
        }
    }
    #[getter]
    pub fn faces(&mut self) -> PyBurnTensor {
        PyBurnTensor {
            inner: BurnTensorType::IntDim2(self.inner.faces.clone()),
        }
    }
    #[getter]
    pub fn uvs(&mut self) -> Option<PyBurnTensor> {
        self.inner.uvs.as_ref().map(|x| PyBurnTensor {
            inner: BurnTensorType::FloatDim2(x.clone()),
        })
    }
    #[getter]
    pub fn normals(&mut self) -> Option<PyBurnTensor> {
        self.inner.normals.as_ref().map(|x| PyBurnTensor {
            inner: BurnTensorType::FloatDim2(x.clone()),
        })
    }
    #[getter]
    pub fn joints(&mut self) -> PyBurnTensor {
        PyBurnTensor {
            inner: BurnTensorType::FloatDim2(self.inner.joints.clone()),
        }
    }
    #[pyo3(text_signature = "($self) -> None")]
    pub fn compute_normals(&mut self) {
        self.inner.compute_normals();
    }
    #[setter]
    fn set_verts(&mut self, v: PyReadonlyArray2<f32>) {
        let device = MultiDevice::default();
        self.inner.verts = v.to_owned_array().to_burn(&device);
    }
}
