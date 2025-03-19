use pyo3::prelude::*;
use smpl_core::common::types::{AngleType, Gender, GltfCompatibilityMode, SmplType, UpAxis};
use smpl_utils::{convert_enum_from, convert_enum_into};
#[pyclass(name = "UpAxis", module = "smpl_rs.types", unsendable, eq, eq_int)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PyUpAxis {
    Y = 0,
    Z,
}
convert_enum_from!(PyUpAxis, UpAxis, Y, Z,);
#[pyclass(name = "Gender", module = "smpl_rs.types", unsendable, eq, eq_int)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PyGender {
    Neutral = 0,
    Male,
    Female,
}
convert_enum_from!(PyGender, Gender, Neutral, Male, Female,);
convert_enum_into!(Gender, PyGender, Neutral, Male, Female,);
#[pyclass(name = "SmplType", module = "smpl_rs.types", unsendable, eq, eq_int)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PySmplType {
    Smpl = 0,
    SmplH,
    SmplX,
    Supr,
    SmplPP,
}
convert_enum_from!(PySmplType, SmplType, Smpl, SmplH, SmplX, Supr, SmplPP,);
convert_enum_into!(SmplType, PySmplType, Smpl, SmplH, SmplX, Supr, SmplPP,);
#[pyclass(name = "AngleType", module = "smpl_rs.types", unsendable, eq, eq_int)]
#[derive(Clone, Copy, PartialEq)]
pub enum PyAngleType {
    AxisAngle = 0,
    Euler,
}
convert_enum_from!(PyAngleType, AngleType, AxisAngle, Euler,);
#[pyclass(name = "GltfCompatibilityMode", module = "smpl_rs.types", unsendable, eq, eq_int)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PyGltfCompatibilityMode {
    Smpl = 0,
    Unreal,
}
convert_enum_from!(PyGltfCompatibilityMode, GltfCompatibilityMode, Smpl, Unreal,);
convert_enum_into!(GltfCompatibilityMode, PyGltfCompatibilityMode, Smpl, Unreal,);
