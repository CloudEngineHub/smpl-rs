#[cfg(feature = "burn-torch")]
use crate::tensor_utils::{burn2pytensor, burn2pytensor_int};
use burn::tensor::{Int, Tensor};
use gloss_burn_multibackend::backend::MultiBackend;
use gloss_utils::bshare::ToNdArray;
use numpy::ToPyArray;
use pyo3::prelude::*;
#[cfg(feature = "burn-torch")]
use pyo3_tch::PyTensor;
#[derive(Clone)]
pub enum BurnTensorType {
    FloatDim1(Tensor<MultiBackend, 1>),
    FloatDim2(Tensor<MultiBackend, 2>),
    FloatDim3(Tensor<MultiBackend, 3>),
    IntDim1(Tensor<MultiBackend, 1, Int>),
    IntDim2(Tensor<MultiBackend, 2, Int>),
    IntDim3(Tensor<MultiBackend, 3, Int>),
}
#[pyclass(name = "BurnTensor", unsendable)]
#[derive(Clone)]
pub struct PyBurnTensor {
    pub inner: BurnTensorType,
}
#[pymethods]
impl PyBurnTensor {
    #[cfg(feature = "burn-torch")]
    #[pyo3(text_signature = "($self) -> PyTensor")]
    pub fn to_torch(&self) -> PyTensor {
        match &self.inner {
            BurnTensorType::FloatDim1(t) => burn2pytensor::<1>(t.clone()),
            BurnTensorType::FloatDim2(t) => burn2pytensor::<2>(t.clone()),
            BurnTensorType::FloatDim3(t) => burn2pytensor::<3>(t.clone()),
            BurnTensorType::IntDim1(t) => burn2pytensor_int::<1>(t.clone()),
            BurnTensorType::IntDim2(t) => burn2pytensor_int::<2>(t.clone()),
            BurnTensorType::IntDim3(t) => burn2pytensor_int::<3>(t.clone()),
        }
    }
    #[pyo3(text_signature = "($self) -> NDArray[np.float32]")]
    pub fn to_numpy(&self, py: Python<'_>) -> PyObject {
        match &self.inner {
            BurnTensorType::FloatDim1(t) => t.to_ndarray().to_pyarray_bound(py).into(),
            BurnTensorType::FloatDim2(t) => t.to_ndarray().to_pyarray_bound(py).into(),
            BurnTensorType::FloatDim3(t) => t.to_ndarray().to_pyarray_bound(py).into(),
            BurnTensorType::IntDim1(t) => t.to_ndarray().to_pyarray_bound(py).into(),
            BurnTensorType::IntDim2(t) => t.to_ndarray().to_pyarray_bound(py).into(),
            BurnTensorType::IntDim3(t) => t.to_ndarray().to_pyarray_bound(py).into(),
        }
    }
}
