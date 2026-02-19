#![allow(clippy::needless_pass_by_value)]
pub mod common;
pub mod global_backend;
pub mod network;
pub mod smpl_x;
#[cfg(feature = "burn-torch")]
pub mod tensor_utils;
use crate::common::types::PyFaceType;
use crate::global_backend::init_global_burn_backend;
use crate::global_backend::smplrs_sync_burn_gpu;
use crate::network::py_smpl_register_components_for_receiver;
use crate::network::py_smpl_register_components_for_sender;
use common::{
    animation::{PyAnimWrap, PyAnimation},
    betas::PyBetas,
    codec::PySmplCodec,
    entity_builder::PyEntityBuilderSmplRs,
    expression::PyExpression,
    follower::{PyFollow, PyFollower, PyFollowerType},
    gltf::PyGltfCodec,
    interop::PyGlossInterop,
    outputs::{PySmplOutput, PySmplOutputPoseT, PySmplOutputPosed},
    pose::PyPose,
    pose_hands::PyHandType,
    pose_override::PyPoseOverride,
    scene::PyMcsCodec,
    scene_animation::PySceneAnimation,
    scene_timer::PySceneTimer,
    smpl_models::PySmplModels,
    smpl_options::PySmplOptions,
    smpl_params::PySmplParams,
    transform_sequence::PyTransformSequence,
    types::{PyAngleType, PyGender, PyGltfCompatibilityMode, PySmplType, PyUpAxis},
    vertex_offsets::PyVertexOffsets,
};
use pyo3::prelude::*;
use smpl_x::{plugin::PySmplPlugin, smpl_x::PySmplX};
/// A Python module implemented in Rust using tch to manipulate `PyTorch`
/// objects.
#[pymodule]
#[pyo3(name = "smpl_rs")]
#[allow(clippy::missing_errors_doc)]
pub fn extension(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    let models_module = PyModule::new_bound(_py, "models")?;

    let components_module = PyModule::new_bound(_py, "components")?;
    let types_module = PyModule::new_bound(_py, "types")?;
    let builders_module = PyModule::new_bound(_py, "builders")?;
    let plugins_module = PyModule::new_bound(_py, "plugins")?;
    let codec_module = PyModule::new_bound(_py, "codec")?;
    let backend_module = PyModule::new_bound(_py, "backend")?;
    let network_module = PyModule::new_bound(_py, "network")?;
    m.add_class::<PySmplModels>()?;
    m.add_class::<PySceneTimer>()?;
    add_submod_models(_py, &models_module)?;
    add_submod_components(_py, &components_module)?;
    add_submod_types(_py, &types_module)?;
    add_submod_builders(_py, &builders_module)?;
    add_submod_plugins(_py, &plugins_module)?;
    add_submod_codec(_py, &codec_module)?;
    add_submod_backend(_py, &backend_module)?;
    add_submod_network(_py, &network_module)?;
    let sys = _py.import_bound("sys")?.getattr("modules")?;
    sys.set_item("smpl_rs.models", models_module.as_ref())?;
    sys.set_item("smpl_rs.components", components_module.as_ref())?;
    sys.set_item("smpl_rs.types", types_module.as_ref())?;
    sys.set_item("smpl_rs.builders", builders_module.as_ref())?;
    sys.set_item("smpl_rs.plugins", plugins_module.as_ref())?;
    sys.set_item("smpl_rs.codec", codec_module.as_ref())?;
    sys.set_item("smpl_rs.backend", backend_module.as_ref())?;
    sys.set_item("smpl_rs.network", network_module.as_ref())?;
    m.add_submodule(&models_module)?;
    m.add_submodule(&components_module)?;
    m.add_submodule(&types_module)?;
    m.add_submodule(&builders_module)?;
    m.add_submodule(&plugins_module)?;
    m.add_submodule(&codec_module)?;
    m.add_submodule(&backend_module)?;
    m.add_submodule(&network_module)?;
    Ok(())
}
#[pymodule]
fn add_submod_models(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PySmplOutputPosed>()?;
    m.add_class::<PySmplOutputPoseT>()?;
    m.add_class::<PySmplOutput>()?;
    m.add_class::<PySmplX>()?;
    Ok(())
}
#[pymodule]
fn add_submod_components(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PySmplParams>()?;
    m.add_class::<PySmplOptions>()?;
    m.add_class::<PyBetas>()?;
    m.add_class::<PyExpression>()?;
    m.add_class::<PyAnimation>()?;
    m.add_class::<PyGlossInterop>()?;
    m.add_class::<PyPose>()?;
    m.add_class::<PyPoseOverride>()?;
    m.add_class::<PyFollow>()?;
    m.add_class::<PyFollower>()?;
    m.add_class::<PyTransformSequence>()?;
    m.add_class::<PySceneAnimation>()?;
    m.add_class::<PyVertexOffsets>()?;
    Ok(())
}
#[pymodule]
fn add_submod_types(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyAnimWrap>()?;
    m.add_class::<PyGender>()?;
    m.add_class::<PyAngleType>()?;
    m.add_class::<PyUpAxis>()?;
    m.add_class::<PySmplType>()?;
    m.add_class::<PyGltfCompatibilityMode>()?;
    m.add_class::<PyHandType>()?;
    m.add_class::<PyFollowerType>()?;
    m.add_class::<PyFaceType>()?;
    Ok(())
}
#[pymodule]
fn add_submod_builders(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyEntityBuilderSmplRs>()?;
    Ok(())
}
#[pymodule]
fn add_submod_plugins(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PySmplPlugin>()?;
    Ok(())
}
#[pymodule]
fn add_submod_codec(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PySmplCodec>()?;
    m.add_class::<PyMcsCodec>()?;
    m.add_class::<PyGltfCodec>()?;
    Ok(())
}
#[pymodule]
fn add_submod_backend(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(init_global_burn_backend, m)?)?;
    m.add_function(wrap_pyfunction!(smplrs_sync_burn_gpu, m)?)?;
    Ok(())
}
#[pymodule]
fn add_submod_network(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_smpl_register_components_for_receiver, m)?)?;
    m.add_function(wrap_pyfunction!(py_smpl_register_components_for_sender, m)?)?;
    Ok(())
}
