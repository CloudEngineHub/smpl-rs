use enum_map::Enum;
use ndarray as nd;
use strum_macros::EnumIter;
/// Enum for hand type
#[derive(Eq, PartialEq, Copy, Clone, Hash, Debug, Enum, EnumIter)]
pub enum HandType {
    Flat,
    Relaxed,
    Curled,
    Fist,
}
#[derive(Default)]
pub struct HandPair {
    pub left: nd::Array2<f32>,
    pub right: nd::Array2<f32>,
}
