use super::types::PyGltfCompatibilityMode;
use crate::common::types::PyFaceType;
use gloss_hecs::Entity;
use gloss_py_macros::PyComponent;
use gloss_renderer::scene::Scene;
use log::info;
use pyo3::prelude::*;
use smpl_core::codec::gltf::GltfCodec;
use smpl_core::common::types::GltfOutputType;
use smpl_gloss_integration::gltf::GltfCodecGloss;
#[pyclass(name = "GltfCodec", module = "smpl_rs.codec", unsendable)]
#[derive(Clone, PyComponent)]
pub struct PyGltfCodec {
    pub inner: GltfCodec,
}
#[pymethods]
impl PyGltfCodec {
    #[staticmethod]
    #[pyo3(text_signature = "() -> GltfCodec")]
    #[allow(clippy::should_implement_trait)]
    pub fn default() -> Self {
        Self { inner: GltfCodec::default() }
    }
    #[staticmethod]
    #[pyo3(text_signature = "(scene_ptr_idx: int, export_camera: bool = True) -> GltfCodec")]
    pub fn from_scene(scene_ptr_idx: u64, export_camera: bool) -> Self {
        let scene_ptr = scene_ptr_idx as *mut Scene;
        let scene: &Scene = unsafe { &*scene_ptr };
        Self {
            inner: GltfCodec::from_scene(scene, None, export_camera),
        }
    }
    #[pyo3(signature = (path, compatibility_mode = None, out_face_type = None))]
    #[pyo3(
        text_signature = "($self, path: str, compatibility_mode: Optional[GltfCompatibilityMode] = None, out_face_type: Optional[FaceType] = None) -> None"
    )]
    fn save(&mut self, path: &str, compatibility_mode: Option<PyGltfCompatibilityMode>, out_face_type: Option<PyFaceType>) {
        let output_type = if std::path::Path::new(path).extension().is_some_and(|ext| ext.eq_ignore_ascii_case("gltf")) {
            GltfOutputType::Standard
        } else if std::path::Path::new(path).extension().is_some_and(|ext| ext.eq_ignore_ascii_case("glb")) {
            GltfOutputType::Binary
        } else {
            panic!("Unsupported file extension. Use `.gltf` or `.glb`.");
        };
        let compatibility_mode = compatibility_mode.unwrap_or(PyGltfCompatibilityMode::Smpl);
        let face_mode = out_face_type.unwrap_or(PyFaceType::SmplX);
        self.inner
            .to_file("Meshcapade Avatar", path, output_type, compatibility_mode.into(), face_mode.into());
        info!("Saved glTF to {}", path);
    }
}
