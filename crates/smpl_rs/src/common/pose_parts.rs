use enum_map::Enum;
use strum_macros::EnumIter;
/// Enum for pose parts, for chunking
#[derive(Eq, PartialEq, Copy, Clone, Hash, Debug, Enum, EnumIter)]
pub enum PosePart {
    RootTranslation,
    RootRotation,
    Body,
    LeftHand,
    RightHand,
    Jaw,
    LeftEye,
    RightEye,
}
