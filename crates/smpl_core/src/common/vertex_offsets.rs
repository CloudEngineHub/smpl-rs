use crate::{codec::codec::SmplCodec, AppBackend};
use burn::{prelude::Backend, tensor::Tensor};
use gloss_utils::bshare::ToBurn;
use log::info;
use ndarray as nd;
use ndarray_npy::NpzReader;
use smpl_utils::io::FileLoader;
use std::io::{Read, Seek};
/// Component for free vertex deformation of the SMPL template defined in T-Pose
#[derive(Clone)]
pub struct VertexOffsetsG<B: Backend> {
    pub device: B::Device,
    pub offsets: Tensor<B, 2>,
    pub strength: f32,
}
impl<B: Backend> VertexOffsetsG<B> {
    pub fn new(offsets: Tensor<B, 2>) -> Self {
        Self {
            device: offsets.device(),
            offsets,
            strength: 1.0,
        }
    }
    pub fn new_from_ndarray(offsets: nd::Array2<f32>) -> Self {
        let device = B::Device::default();
        let offsets = offsets.into_burn(&device);
        Self::new(offsets)
    }
    #[allow(clippy::cast_possible_truncation)]
    fn new_from_npz_reader<R: Read + Seek>(npz: &mut NpzReader<R>) -> Self {
        info!("NPZ keys - {:?}", npz.names().unwrap());
        let offsets: nd::Array2<f64> = npz.by_name("vertexOffsets.npy").unwrap();
        let offsets = offsets.mapv(|x| x as f32);
        let device = B::Device::default();
        let offsets = offsets.into_burn(&device);
        Self {
            device,
            offsets,
            strength: 1.0,
        }
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
        codec.vertex_offsets.as_ref().map(|offsets| Self::new_from_ndarray(offsets.clone()))
    }
    /// Create a new ``VertexOffsets`` component from a ``.smpl`` file
    pub fn new_from_smpl_file(path: &str) -> Option<Self> {
        let codec = SmplCodec::from_file(path);
        Self::new_from_smpl_codec(&codec)
    }
}
pub type VertexOffsets = VertexOffsetsG<AppBackend>;
