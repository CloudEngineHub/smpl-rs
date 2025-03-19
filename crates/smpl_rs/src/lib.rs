#![deny(missing_docs)]
//! ## Crate Items Overview
//!
//! This section provides quick links to the main items in smpl-rs.
//!
//! ### Modules
//! - [`smpl_core`](crate::smpl_core) - The core functionality of smpl-rs.
//! - [`smpl_gloss_integration`](crate::smpl_gloss_integration) - The integration between smpl-rs and gloss-renderer.
//! - [`smpl_utils`](crate::smpl_utils) - Utility functions and helpers.
//!
//! ## Examples
//! Below are the examples you can explore in the `examples/` folder of the
//! repository:
//!
//! - **Animation from .npz Matrices**: [animation_from_matrices.py](https://github.com/Meshcapade/smpl-rs/bindings/smpl_py/examples/animation_from_matrices.py)
//! - **Show Animation with Skeleton**: [show_skeleton.py](https://github.com/Meshcapade/smpl-rs/bindings/smpl_py/examples/show_skeleton.py)
//! - **Minimal Example**: [minimal.py](https://github.com/Meshcapade/smpl-rs/bindings/smpl_py/examples/minimal.py)
//! - **Export a glTF from animation**: [smpl_gltf_export.py](https://github.com/Meshcapade/smpl-rs/bindings/smpl_py/examples/smpl_gltf_export.py)
//!
//! These examples demonstrate various features of smpl-rs and can be run
//! directly.
pub use smpl_core;
pub use smpl_gloss_integration;
pub use smpl_utils;
