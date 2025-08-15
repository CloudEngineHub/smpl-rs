use enum_map::Enum;
use num_derive::FromPrimitive;
use strum_macros::Display;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;
/// Various ``SmplModel`` types
#[derive(Clone, Copy, Debug, Enum, FromPrimitive, PartialEq, Display)]
pub enum SmplType {
    Smpl = 0,
    SmplH,
    SmplX,
    Supr,
    SmplPP,
}
#[derive(Clone, Copy, Debug, Enum, FromPrimitive, PartialEq)]
pub enum FaceType {
    SmplX = 0,
    ARKit,
}
#[derive(Clone, Copy, PartialEq)]
pub enum AngleType {
    AxisAngle,
    Euler,
}
#[derive(Clone, Copy, PartialEq, Debug, Enum, FromPrimitive, Display)]
pub enum Gender {
    Neutral = 0,
    Male,
    Female,
}
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum UpAxis {
    Y,
    Z,
}
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum GltfOutputType {
    Standard,
    Binary,
}
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
pub enum GltfCompatibilityMode {
    Smpl,
    Unreal,
}
/// Dummy class just to be able to get the right size for chunk header (never
/// really used)
#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct ChunkHeader {
    /// The length of the chunk data in byte excluding the header.
    length: u32,
    /// Chunk type.
    ty: gltf::binary::ChunkType,
}
