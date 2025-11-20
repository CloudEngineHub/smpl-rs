use crate::codec::codec::SmplCodec;
use log::info;
use ndarray as nd;
use ndarray_npy::NpzReader;
use smpl_utils::io::FileLoader;
use std::io::{Read, Seek};
/// Component for free vertex deformation of the SMPL template defined in T-Pose
#[derive(Clone)]
pub struct VertexOffsets {
    pub offsets: nd::Array2<f32>,
    pub strength: f32,
}
impl VertexOffsets {
    pub fn new(offsets: nd::Array2<f32>) -> Self {
        Self { offsets, strength: 1.0 }
    }
    #[allow(clippy::cast_possible_truncation)]
    fn new_from_npz_reader<R: Read + Seek>(npz: &mut NpzReader<R>) -> Self {
        info!("NPZ keys - {:?}", npz.names().unwrap());
        let offsets: nd::Array2<f64> = npz.by_name("vertexOffsets").unwrap();
        let offsets = offsets.mapv(|x| x as f32);
        Self { offsets, strength: 1.0 }
    }
    #[cfg(not(target_arch = "wasm32"))]
    /// # Panics
    /// Will panic if the file cannot be read
    pub fn new_from_npz(npz_path: &str) -> Self {
        let mut npz = NpzReader::new(std::fs::File::open(npz_path).unwrap()).unwrap();
        Self::new_from_npz_reader(&mut npz)
    }
    /// # Panics
    /// Will panic if the file cannot be read
    pub async fn new_from_npz_async(npz_path: &str) -> Self {
        let reader = FileLoader::open(npz_path).await;
        let mut npz = NpzReader::new(reader).unwrap();
        Self::new_from_npz_reader(&mut npz)
    }
    /// Create a new ``VertexOffsets`` component from a ``SmplCodec``
    pub fn new_from_smpl_codec(codec: &SmplCodec) -> Option<Self> {
        codec.vertex_offsets.as_ref().map(|offsets| Self {
            offsets: offsets.clone(),
            strength: 1.0,
        })
    }
    /// Create a new ``VertexOffsets`` component from a ``.smpl`` file
    pub fn new_from_smpl_file(path: &str) -> Option<Self> {
        let codec = SmplCodec::from_file(path);
        Self::new_from_smpl_codec(&codec)
    }
}
