use pyo3::prelude::*;
use smpl_core::common::pose_hands::HandType;
use smpl_utils::convert_enum_from;
#[pyclass(name = "HandType", module = "smpl_rs.types", unsendable, eq, eq_int)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PyHandType {
    Flat,
    Relaxed,
    Curled,
    Fist,
}
convert_enum_from!(PyHandType, HandType, Flat, Relaxed, Curled, Fist,);
