#![recursion_limit = "256"]
pub mod codec;
pub mod common;
pub mod conversions;
pub mod smpl_h;
pub mod smpl_x;
use gloss_burn_multibackend::backend::MultiBackend;
pub type AppBackend = MultiBackend;
