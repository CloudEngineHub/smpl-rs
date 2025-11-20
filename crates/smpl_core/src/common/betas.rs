use crate::{codec::codec::SmplCodec, AppBackend};
use burn::{prelude::Backend, tensor::Tensor};
use gloss_utils::bshare::ToBurn;
use log::info;
use ndarray as nd;
use ndarray::prelude::*;
use ndarray_npy::NpzReader;
use smpl_utils::io::FileLoader;
use std::io::{Read, Seek};
/// Component for Smpl Betas or Shape Parameters
#[derive(Clone)]
pub struct BetasG<B: Backend> {
    pub device: B::Device,
    pub betas: Tensor<B, 1>,
}
impl<B: Backend> Default for BetasG<B> {
    fn default() -> Self {
        let device = B::Device::default();
        let num_betas = 10;
        let betas = Tensor::<B, 1>::zeros([num_betas], &device);
        Self { device, betas }
    }
}
impl<B: Backend> BetasG<B> {
    pub fn new(betas: Tensor<B, 1>) -> Self {
        Self {
            device: betas.device(),
            betas,
        }
    }
    pub fn new_empty(num_betas: usize) -> Self {
        let device = B::Device::default();
        let betas = Tensor::<B, 1>::zeros([num_betas], &device);
        Self { device, betas }
    }
    pub fn new_from_ndarray(betas: nd::Array1<f32>) -> Self {
        let device = B::Device::default();
        let betas = betas.into_burn(&device);
        Self::new(betas)
    }
    /// # Panics
    /// Will panic if the file cannot be read
    #[allow(clippy::cast_possible_truncation)]
    fn new_from_npz_reader<R: Read + Seek>(npz: &mut NpzReader<R>, truncate_nr_betas: Option<usize>) -> Self {
        info!("NPZ keys - {:?}", npz.names().unwrap());
        let device = B::Device::default();
        let betas: nd::Array1<f64> = npz.by_name("betas").unwrap();
        let mut betas = betas.mapv(|x| x as f32);
        if let Some(truncate_nr_betas) = truncate_nr_betas {
            if truncate_nr_betas < betas.len() {
                betas = betas.slice(s![0..truncate_nr_betas]).to_owned();
            }
        }
        let betas = betas.into_burn(&device);
        Self { device, betas }
    }
    #[cfg(not(target_arch = "wasm32"))]
    /// # Panics
    /// Will panic if the file cannot be read
    #[allow(clippy::cast_possible_truncation)]
    pub fn new_from_npz(npz_path: &str, truncate_nr_betas: Option<usize>) -> Self {
        let mut npz = NpzReader::new(std::fs::File::open(npz_path).unwrap()).unwrap();
        Self::new_from_npz_reader(&mut npz, truncate_nr_betas)
    }
    /// # Panics
    /// Will panic if the file cannot be read
    #[allow(clippy::cast_possible_truncation)]
    pub async fn new_from_npz_async(npz_path: &str, truncate_nr_betas: Option<usize>) -> Self {
        let reader = FileLoader::open(npz_path).await;
        let mut npz = NpzReader::new(reader).unwrap();
        Self::new_from_npz_reader(&mut npz, truncate_nr_betas)
    }
    /// Create a new ``Betas`` component from a ``SmplCodec``
    pub fn new_from_smpl_codec(codec: &SmplCodec) -> Option<Self> {
        codec.shape_parameters.as_ref().map(|betas| Self::new_from_ndarray(betas.clone()))
    }
    /// Create a new ``Betas`` component from a ``.smpl`` file
    pub fn new_from_smpl_file(path: &str) -> Option<Self> {
        let codec = SmplCodec::from_file(path);
        Self::new_from_smpl_codec(&codec)
    }
}
pub type Betas = BetasG<AppBackend>;
