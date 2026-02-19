use burn::prelude::Int;
use burn::tensor::{Tensor, TensorPrimitive};
use burn_tch::TchTensor;
use gloss_burn_multibackend::backend::MultiBackend;
use gloss_burn_multibackend::tensor::{MultiFloatTensor, MultiIntTensor};
use pyo3_tch::PyTensor;
pub fn pytensor2burn<const D: usize>(py_tensor: PyTensor) -> Tensor<MultiBackend, D> {
    let tch_tensor = TchTensor::new(py_tensor.0);
    let tensor_multibackend = MultiFloatTensor::Torch(tch_tensor);
    let tensor = Tensor::from_primitive(TensorPrimitive::Float(tensor_multibackend));
    tensor
}
pub fn burn2pytensor<const D: usize>(burn_tensor: Tensor<MultiBackend, D>) -> PyTensor {
    let out_primitive = burn_tensor.into_primitive();
    if let TensorPrimitive::Float(tch_wrapper) = out_primitive {
        match tch_wrapper {
            MultiFloatTensor::Torch(tch) => {
                let raw_tch: tch::Tensor = tch.tensor;
                PyTensor(raw_tch)
            }
            _ => {
                unreachable!("expected only Torch backend");
            }
        }
    } else {
        unreachable!("expected only Float primitives");
    }
}
pub fn burn2pytensor_int<const D: usize>(burn_tensor: Tensor<MultiBackend, D, Int>) -> PyTensor {
    let out_primitive = burn_tensor.into_primitive();
    match out_primitive {
        MultiIntTensor::Torch(tch) => {
            let raw_tch: tch::Tensor = tch.tensor;
            PyTensor(raw_tch)
        }
        _ => {
            unreachable!("expected only Torch backend");
        }
    }
}
