use gloss_renderer::plugin_manager::Plugins;

use pyo3::prelude::*;
use smpl_gloss_integration::plugin::SmplPlugin;

#[pyclass(name = "SmplPlugin", module = "smpl_rs.plugins", unsendable)]
// it has to be unsendable because it does not implement Send: https://pyo3.rs/v0.19.1/class#must-be-send
pub struct PySmplPlugin {
    inner: SmplPlugin,
}

#[pymethods]
impl PySmplPlugin {
    #[new]
    #[pyo3(text_signature = "(autorun: bool) -> SmplPlugin")]
    pub fn new(autorun: bool) -> Self {
        PySmplPlugin {
            inner: SmplPlugin::new(autorun),
        }
    }
    #[pyo3(text_signature = "($self, plugin_ptr_idx: int) -> None")]
    pub fn insert_plugin(&mut self, plugin_ptr_idx: u64) {
        let plugin_ptr: *mut Plugins = plugin_ptr_idx as *mut Plugins;
        let plugin_list: &mut Plugins = unsafe { &mut *plugin_ptr };
        plugin_list.insert_plugin(&self.inner);
    }
}
