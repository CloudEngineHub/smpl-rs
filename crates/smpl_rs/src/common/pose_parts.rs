use enum_map::Enum;
use strum_macros::EnumIter; // 0.17.1

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
