pub struct FileLoader {}
#[cfg(not(target_arch = "wasm32"))]
use std::fs::File;
#[cfg(not(target_arch = "wasm32"))]
impl FileLoader {
    /// # Panics
    /// Will panic if the path cannot be opened
    #[allow(clippy::unused_async)]
    pub async fn open(file_path: &str) -> File {
        File::open(file_path).unwrap()
    }
}
#[cfg(target_arch = "wasm32")]
use {
    std::io::Cursor,
    wasm_bindgen::JsCast,
    wasm_bindgen_futures::JsFuture,
    web_sys::{Request, RequestInit, RequestMode, Response},
};
#[cfg(target_arch = "wasm32")]
impl FileLoader {
    pub async fn open(file_path: &str) -> Cursor<Vec<u8>> {
        let result = fetch_as_binary(file_path).await.unwrap();
        Cursor::new(result)
    }
}
#[cfg(target_arch = "wasm32")]
pub async fn fetch_as_binary(url: &str) -> Result<Vec<u8>, String> {
    let opts = RequestInit::new();
    opts.set_method("GET");
    opts.set_mode(RequestMode::Cors);
    let request = match Request::new_with_str_and_init(&url, &opts) {
        Ok(request) => request,
        Err(_e) => return Err("Failed to create request".to_string()),
    };
    let window = web_sys::window().unwrap();
    let response = match JsFuture::from(window.fetch_with_request(&request)).await {
        Ok(response) => response,
        Err(_e) => return Err("Failed to fetch".to_string()),
    };
    let response: Response = match response.dyn_into() {
        Ok(response) => response,
        Err(_e) => return Err("Failed to dyn_into Response".to_string()),
    };
    let buffer = match response.array_buffer() {
        Ok(buffer) => buffer,
        Err(_e) => return Err("Failed to get as array buffer".to_string()),
    };
    let buffer = match JsFuture::from(buffer).await {
        Ok(buffer) => buffer,
        Err(_e) => return Err("Failed to ...?".to_string()),
    };
    Ok(js_sys::Uint8Array::new(&buffer).to_vec())
}
/// associating a extension with a enum
use strum::IntoEnumIterator;
use strum_macros::EnumIter;
#[derive(Debug, EnumIter)]
pub enum FileType {
    Smpl,
    Gltf,
    Mcs,
    Unknown,
}
impl FileType {
    pub fn value(&self) -> &'static [&'static str] {
        match self {
            Self::Smpl => &["smpl"],
            Self::Gltf => &["gltf"],
            Self::Mcs => &["mcs"],
            Self::Unknown => &[""],
        }
    }
    pub fn find_match(ext: &str) -> Self {
        Self::iter()
            .find(|filetype| filetype.value().contains(&(ext.to_lowercase()).as_str()))
            .unwrap_or(FileType::Unknown)
    }
}
