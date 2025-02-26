use super::types::PyGltfCompatibilityMode;
use gloss_hecs::Entity;
use gloss_py_macros::PyComponent;
use gloss_renderer::scene::Scene;
use log::info;
use pyo3::prelude::*;
use smpl_gloss_integration::gltf::GltfCodecGloss;
use smpl_rs::codec::gltf::GltfCodec;
use smpl_rs::common::types::GltfOutputType;
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
    #[pyo3(text_signature = "(scene_ptr_idx: int) -> GltfCodec")]
    pub fn from_scene(scene_ptr_idx: u64) -> Self {
        let scene_ptr = scene_ptr_idx as *mut Scene;
        let scene: &Scene = unsafe { &*scene_ptr };
        Self {
            inner: GltfCodec::from_scene(scene, None, None),
        }
    }
    #[pyo3(signature = (path, compatibility_mode = None))]
    #[pyo3(text_signature = "($self, path: str, compatibility_mode: Optional[GltfCompatibilityMode] = None) -> None")]
    fn save(&mut self, path: &str, compatibility_mode: Option<PyGltfCompatibilityMode>) {
        let output_type = if std::path::Path::new(path)
            .extension()
            .map_or(false, |ext| ext.eq_ignore_ascii_case("gltf"))
        {
            GltfOutputType::Standard
        } else if std::path::Path::new(path)
            .extension()
            .map_or(false, |ext| ext.eq_ignore_ascii_case("glb"))
        {
            GltfOutputType::Binary
        } else {
            panic!("Unsupported file extension. Use `.gltf` or `.glb`.");
        };
        let compatibility_mode = compatibility_mode.unwrap_or(PyGltfCompatibilityMode::Smpl);
        self.inner.to_file("Meshcapade Avatar", path, output_type, compatibility_mode.into());
        info!("Saved glTF to {}", path);
    }
}
