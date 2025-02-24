use std::any::Any;

use super::{
    betas::Betas,
    expression::Expression,
    outputs::SmplOutputDynamic,
    pose::Pose,
    smpl_options::SmplOptions,
    types::{Gender, SmplType},
};
use crate::smpl_x::smpl_x_gpu::SmplXDynamic;
use burn::{
    backend::{Candle, NdArray, Wgpu},
    prelude::Backend,
    tensor::{Float, Int, Tensor},
};
use dyn_clone::DynClone;
use enum_map::EnumMap;
use ndarray as nd;
use utils_rs::tensor::BurnBackend;
use pollster;
use reqwest;
// SmplModel is a super trait of gloss_hecs::Component but since I don't want
// smpl-rs to be dependant in hecs we just make it a supertrait of what a
// Component means which is Send+Sync+static
/// Trait for a Smpl based model. Smpl-rs expects all Smpl models to implement
/// this.
pub trait SmplModel<B: Backend>: Send + Sync + 'static + Any + DynClone {
    fn smpl_type(&self) -> SmplType;
    fn gender(&self) -> Gender;

    fn forward(&self, options: &SmplOptions, betas: &Betas, pose_raw: &Pose, expression: Option<&Expression>) -> SmplOutputDynamic<B>;

    fn create_body_with_uv(&self, smpl_output: &SmplOutputDynamic<B>) -> SmplOutputDynamic<B>;

    fn expression2offsets(&self, expression: &Expression) -> Tensor<B, 2, Float>;

    fn betas2verts(&self, betas: &Betas) -> Tensor<B, 2, Float>;

    fn verts2joints(&self, verts_t_pose: Tensor<B, 2, Float>) -> Tensor<B, 2, Float>;

    fn compute_pose_correctives(&self, pose: &Pose) -> Tensor<B, 2, Float>;

    fn compute_pose_feature(&self, pose: &Pose) -> nd::Array1<f32>;

    #[allow(clippy::type_complexity)]
    fn apply_pose(
        &self,
        verts_t_pose: &Tensor<B, 2, Float>,
        normals: Option<&Tensor<B, 2, Float>>,
        tangents: Option<&Tensor<B, 2, Float>>,
        joints: &Tensor<B, 2, Float>,
        lbs_weights: &Tensor<B, 2, Float>, /* TODO this should be some sort of ArrayRef but is curently blocked by https://github.com/rust-ndarray/ndarray/issues/879 */
        pose: &Pose,
    ) -> (
        Tensor<B, 2, Float>,         // v
        Option<Tensor<B, 2, Float>>, // n
        Option<Tensor<B, 2, Float>>, // t
        Tensor<B, 2, Float>,         // joints
    );
    //accesors
    fn faces(&self) -> &Tensor<B, 2, Int>;

    fn faces_uv(&self) -> &Tensor<B, 2, Int>;

    fn uv(&self) -> &Tensor<B, 2, Float>;

    fn lbs_weights(&self) -> Tensor<B, 2, Float>;

    fn lbs_weights_split(&self) -> Tensor<B, 2, Float>;

    fn idx_split_2_merged(&self) -> Tensor<B, 1, Int>;

    fn idx_split_2_merged_vec(&self) -> &Vec<usize>;

    fn set_pose_dirs(&mut self, posedirs: Tensor<B, 2, Float>);

    fn get_pose_dirs(&self) -> Tensor<B, 2, Float>;

    fn get_expression_dirs(&self) -> Option<Tensor<B, 2, Float>>;

    // Misc
    //https://stackoverflow.com/a/67251812
    fn clone_dyn(&self) -> Box<dyn SmplModel<B>>;

    // https://stackoverflow.com/questions/33687447/how-to-get-a-reference-to-a-concrete-type-from-a-trait-object
    fn as_any(&self) -> &dyn Any;
}

// So that we can clone a Box<dyn SmplModel> for the python bindings https://github.com/dtolnay/dyn-clone
// dyn_clone::clone_trait_object!(SmplModel);
impl<B: Backend> Clone for Box<dyn SmplModel<B>> {
    #[allow(unconditional_recursion)]
    fn clone(&self) -> Box<dyn SmplModel<B>> {
        self.clone()
    }
}
/// A mapping from ``Gender`` to ``SmplModel``
#[derive(Default, Clone)]
pub struct Gender2Model<B: Backend> {
    gender_to_model: EnumMap<Gender, Option<Box<dyn SmplModel<B>>>>,
}

#[derive(Default, Clone)]
pub struct Gender2Path {
    gender_to_path: EnumMap<Gender, Option<String>>,
}

/// A Dynamic Backend Cache for storing and easy access to ``SmplModels``
/// This internally uses ``SmplCache<B>``
#[allow(clippy::large_enum_variant)]
#[derive(Clone)]
pub enum SmplCacheDynamic {
    NdArray(SmplCache<NdArray>),
    Wgpu(SmplCache<Wgpu>),
    Candle(SmplCache<Candle>),
}

impl Default for SmplCacheDynamic {
    fn default() -> Self {
        SmplCacheDynamic::Candle(SmplCache::default())
    }
}

impl SmplCacheDynamic {
    /// Get the Burn Backend the Cache was created using
    pub fn get_backend(&self) -> BurnBackend {
        match self {
            SmplCacheDynamic::NdArray(_) => BurnBackend::NdArray,
            SmplCacheDynamic::Wgpu(_) => BurnBackend::Wgpu,
            SmplCacheDynamic::Candle(_) => BurnBackend::Candle,
        }
    }
    /// Check whether the Cache has a certain model
    pub fn has_model(&self, smpl_type: SmplType, gender: Gender) -> bool {
        match self {
            SmplCacheDynamic::NdArray(models) => models.has_model(smpl_type, gender),
            SmplCacheDynamic::Wgpu(models) => models.has_model(smpl_type, gender),
            SmplCacheDynamic::Candle(models) => models.has_model(smpl_type, gender),
        }
    }
    /// Clear the Cache
    pub fn remove_all_models(&mut self) {
        match self {
            SmplCacheDynamic::NdArray(models) => models.remove_all_models(),
            SmplCacheDynamic::Wgpu(models) => models.remove_all_models(),
            SmplCacheDynamic::Candle(models) => models.remove_all_models(),
        }
    }
    pub fn has_lazy_loading(&self, smpl_type: SmplType, gender: Gender) -> bool {
        match self {
            SmplCacheDynamic::NdArray(models) => models.has_lazy_loading(smpl_type, gender),
            SmplCacheDynamic::Wgpu(models) => models.has_lazy_loading(smpl_type, gender),
            SmplCacheDynamic::Candle(models) => models.has_lazy_loading(smpl_type, gender),
        }
    }
    pub fn get_lazy_loading(&self, smpl_type: SmplType, gender: Gender) -> Option<String> {
        match self {
            SmplCacheDynamic::NdArray(models) => models.get_lazy_loading(smpl_type, gender),
            SmplCacheDynamic::Wgpu(models) => models.get_lazy_loading(smpl_type, gender),
            SmplCacheDynamic::Candle(models) => models.get_lazy_loading(smpl_type, gender),
        }
    }
    /// Set lazy loading using default paths
    pub fn lazy_load_defaults(&mut self) {
        self.set_lazy_loading(SmplType::SmplX, Gender::Neutral, "./data/smplx/SMPLX_neutral_array_f32_slim.npz");
        self.set_lazy_loading(SmplType::SmplX, Gender::Male, "./data/smplx/SMPLX_male_array_f32_slim.npz");
        self.set_lazy_loading(SmplType::SmplX, Gender::Female, "./data/smplx/SMPLX_female_array_f32_slim.npz");
    }
    /// Set lazy loading explicitly
    pub fn set_lazy_loading(&mut self, smpl_type: SmplType, gender: Gender, path: &str) {
        match self {
            SmplCacheDynamic::NdArray(models) => models.set_lazy_loading(smpl_type, gender, path),
            SmplCacheDynamic::Wgpu(models) => models.set_lazy_loading(smpl_type, gender, path),
            SmplCacheDynamic::Candle(models) => models.set_lazy_loading(smpl_type, gender, path),
        }
    }
    /// Add a Smpl Model created on a certain Burn Backend
    pub fn add_model_from_dynamic_device(&mut self, model: SmplXDynamic, cache_models: bool) {
        match (self, model) {
            (SmplCacheDynamic::NdArray(models), SmplXDynamic::NdArray(model_ndarray)) => {
                models.add_model(model_ndarray, cache_models);
            }
            (SmplCacheDynamic::Wgpu(models), SmplXDynamic::Wgpu(model_wgpu)) => {
                models.add_model(model_wgpu, cache_models);
            }
            (SmplCacheDynamic::Candle(models), SmplXDynamic::Candle(model_candle)) => {
                models.add_model(model_candle, cache_models);
            }
            // Optional: Handle cases where the model type doesn't match the backend (e.g., error logging)
            _ => {
                eprintln!("Model and backend type mismatch!");
            }
        }
    }
    pub fn add_model_from_type(&mut self, smpl_type: SmplType, path: &str, gender: Gender, max_num_betas: usize, num_expression_components: usize) {
        match smpl_type {
            SmplType::SmplX => {
                let new_model = SmplXDynamic::new_from_npz(self, path, gender, max_num_betas, num_expression_components);
                self.add_model_from_dynamic_device(new_model, true);
            }
            _ => panic!("Model loading for {smpl_type:?} if not supported yet!"),
        };
    }
}

/// A Cache for storing and easy access to ``SmplModels`` which is generic over
/// Burn Backend
#[derive(Default, Clone)]
pub struct SmplCache<B: Backend> {
    type_to_model: EnumMap<SmplType, Gender2Model<B>>,
    type_to_path: EnumMap<SmplType, Gender2Path>,
}

impl<B: Backend> SmplCache<B> {
    pub fn add_model<T: SmplModel<B>>(&mut self, model: T, cache_models: bool) {
        let smpl_type = model.smpl_type();
        let gender = model.gender();

        // if we don't cache we already remove all the other models before adding this
        // one
        if !cache_models {
            self.type_to_model = EnumMap::default();
        }
        self.type_to_model[smpl_type].gender_to_model[gender] = Some(Box::new(model));
    }
    pub fn remove_all_models(&mut self) {
        self.type_to_model = EnumMap::default();
    }

    #[allow(clippy::borrowed_box)]
    pub fn get_model_box_ref(&self, smpl_type: SmplType, gender: Gender) -> Option<&Box<dyn SmplModel<B>>> {
        self.type_to_model[smpl_type].gender_to_model[gender].as_ref()
    }

    #[allow(clippy::redundant_closure_for_method_calls)]
    pub fn get_model_ref(&self, smpl_type: SmplType, gender: Gender) -> Option<&dyn SmplModel<B>> {
        let opt = &self.type_to_model[smpl_type].gender_to_model[gender];
        let model = opt.as_ref().map(|x| x.as_ref());
        model
    }

    #[allow(clippy::redundant_closure_for_method_calls)]
    pub fn get_model_mut(&mut self, smpl_type: SmplType, gender: Gender) -> Option<&mut dyn SmplModel<B>> {
        // model
        let opt = &mut self.type_to_model[smpl_type].gender_to_model[gender];
        let model = opt.as_mut().map(|x| x.as_mut());
        model
    }

    pub fn has_model(&self, smpl_type: SmplType, gender: Gender) -> bool {
        self.type_to_model[smpl_type].gender_to_model[gender].is_some()
    }

    pub fn has_lazy_loading(&self, smpl_type: SmplType, gender: Gender) -> bool {
        self.type_to_path[smpl_type].gender_to_path[gender].is_some()
    }

    pub fn get_lazy_loading(&self, smpl_type: SmplType, gender: Gender) -> Option<String> {
        self.type_to_path[smpl_type].gender_to_path[gender].clone()
    }

    fn get_default_url(gender: Gender) -> &'static str {
        match gender {
            Gender::Neutral => "https://download.is.tue.mpg.de/smplx_neutral.npz",
            Gender::Male => "https://download.is.tue.mpg.de/smplx_male.npz",
            Gender::Female => "https://download.is.tue.mpg.de/smplx_female.npz",
        }
    }
    // TODO: Needs to be tested with real URL's 
    fn ensure_model_file(&self, smpl_type: SmplType, gender: Gender) -> std::io::Result<()> {
        if let Some(path) = self.get_lazy_loading(smpl_type, gender) {
            if !std::path::Path::new(&path).exists() {
                // Create parent directories if they don't exist
                if let Some(parent) = std::path::Path::new(&path).parent() {
                    std::fs::create_dir_all(parent)?;
                }
                
                let url = Self::get_default_url(gender);
                println!("Downloading SMPLX model for {gender:?} from {url}");
                
                let bytes = pollster::block_on(async {
                    let response = reqwest::get(url)
                        .await
                        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;
                        
                    response.bytes()
                        .await
                        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))
                })?;
                
                std::fs::write(&path, bytes)?;
                println!("Successfully downloaded model to {path}");
            }
        }
        Ok(())
    }

    pub fn set_lazy_loading(&mut self, smpl_type: SmplType, gender: Gender, path: &str) {
        self.type_to_path[smpl_type].gender_to_path[gender] = Some(path.to_string());
        // Try to download if file doesn't exist
        if let Err(e) = self.ensure_model_file(smpl_type, gender) {
            eprintln!("Failed to download model: {e}");
        }
    }
}
