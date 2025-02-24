use log::info;
use ndarray as nd;
use ndarray::prelude::*;
use ndarray_npy::NpzReader;

use crate::codec::codec::SmplCodec;
use smpl_utils::io::FileLoader;
use std::io::{Read, Seek};

/// Component for Smpl Betas or Shape Parameters
#[derive(Clone)]
pub struct Betas {
    pub betas: nd::Array1<f32>,
}
impl Default for Betas {
    fn default() -> Self {
        let num_betas = 10;
        let betas = ndarray::Array1::<f32>::zeros(num_betas);
        Self { betas }
    }
}

impl Betas {
    pub fn new(betas: nd::Array1<f32>) -> Self {
        Self { betas }
    }

    pub fn new_empty(num_betas: usize) -> Self {
        let betas = ndarray::Array1::<f32>::zeros(num_betas);
        Self { betas }
    }

    /// # Panics
    /// Will panic if the file cannot be read
    #[allow(clippy::cast_possible_truncation)]
    fn new_from_npz_reader<R: Read + Seek>(npz: &mut NpzReader<R>, truncate_nr_betas: Option<usize>) -> Self {
        info!("NPZ keys - {:?}", npz.names().unwrap());

        let betas: nd::Array1<f64> = npz.by_name("betas").unwrap();
        let mut betas = betas.mapv(|x| x as f32);
        //get only 10 betas because the measurement regressor only has 10 for now
        if let Some(truncate_nr_betas) = truncate_nr_betas {
            if truncate_nr_betas < betas.len() {
                betas = betas.slice(s![0..truncate_nr_betas]).to_owned();
            }
        }
        Self { betas }
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
        codec.shape_parameters.as_ref().map(|betas| Self { betas: betas.clone() })
    }

    /// Create a new ``Betas`` component from a ``.smpl`` file
    pub fn new_from_smpl_file(path: &str) -> Option<Self> {
        let codec = SmplCodec::from_file(path);
        Self::new_from_smpl_codec(&codec)
    }
}
