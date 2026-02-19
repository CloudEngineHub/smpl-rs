use super::{
    expression::ExpressionG,
    pose::PoseG,
    smpl_options::SmplOptions,
    types::{Gender, SmplType},
};
use crate::{
    common::{betas::BetasG, outputs::SmplOutputG, vertex_offsets::VertexOffsetsG},
    AppBackend,
};
use burn::{
    prelude::Backend,
    tensor::{Float, Int, Tensor},
};
use dyn_clone::DynClone;
use enum_map::EnumMap;
use gloss_geometry::csr::VertexFaceCSRBurn;
use std::any::Any;
pub trait FaceModel<B: Backend>: Send + Sync + 'static + Any + DynClone {
    fn expression2offsets(&self, expression: &ExpressionG<B>) -> Tensor<B, 2, Float>;
    fn get_face_model(&self) -> &dyn FaceModel<B>;
}
impl<B: Backend> Clone for Box<dyn FaceModel<B>> {
    #[allow(unconditional_recursion)]
    fn clone(&self) -> Box<dyn FaceModel<B>> {
        self.clone()
    }
}
/// Trait for a Smpl based model. Smpl-rs expects all Smpl models to implement
/// this.
pub trait SmplModel<B: Backend>: Send + Sync + 'static + Any + DynClone {
    fn smpl_type(&self) -> SmplType;
    fn gender(&self) -> Gender;
    fn device(&self) -> B::Device;
    fn forward(
        &self,
        options: &SmplOptions,
        betas: &BetasG<B>,
        pose_raw: &PoseG<B>,
        expression: Option<&ExpressionG<B>>,
        vertex_offsets: Option<&VertexOffsetsG<B>>,
    ) -> SmplOutputG<B>;
    fn create_body_with_uv(&self, smpl_output: &SmplOutputG<B>) -> SmplOutputG<B>;
    fn get_face_model(&self) -> &dyn FaceModel<B>;
    fn betas2verts(&self, betas: &BetasG<B>) -> Tensor<B, 2, Float>;
    fn verts2joints(&self, verts_t_pose: Tensor<B, 2, Float>) -> Tensor<B, 2, Float>;
    fn compute_pose_correctives(&self, pose: &PoseG<B>) -> Tensor<B, 2, Float>;
    fn compute_pose_feature(&self, pose: &PoseG<B>) -> Tensor<B, 1, Float>;
    #[allow(clippy::type_complexity)]
    fn apply_pose(
        &self,
        verts_t_pose: &Tensor<B, 2, Float>,
        joints: &Tensor<B, 2, Float>,
        lbs_weights: &Tensor<B, 2, Float>,
        pose: &PoseG<B>,
    ) -> (Tensor<B, 2, Float>, Tensor<B, 2, Float>);
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
    fn vertex_face_csr(&self) -> Option<VertexFaceCSRBurn<B>>;
    fn vertex_face_uv_csr(&self) -> Option<VertexFaceCSRBurn<B>>;
    fn kinematic_tree_depth(&self) -> usize;
    fn clone_dyn(&self) -> Box<dyn SmplModel<B>>;
    fn as_any(&self) -> &dyn Any;
}
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
/// A Cache for storing and easy access to ``SmplModels`` which is generic over
/// Burn Backend
#[derive(Default, Clone)]
pub struct SmplCacheG<B: Backend> {
    type_to_model: EnumMap<SmplType, Gender2Model<B>>,
    type_to_path: EnumMap<SmplType, Gender2Path>,
}
impl<B: Backend> SmplCacheG<B> {
    pub fn add_model<T: SmplModel<B> + FaceModel<B>>(&mut self, model: T, cache_models: bool) {
        let smpl_type = model.smpl_type();
        self.add_model_under_type(smpl_type, model, cache_models);
    }
    pub fn add_model_under_type<T: SmplModel<B> + FaceModel<B>>(&mut self, smpl_type: SmplType, model: T, cache_models: bool) {
        let gender = model.gender();
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
    pub fn get_face_model_ref(&self, smpl_type: SmplType, gender: Gender) -> Option<&dyn FaceModel<B>> {
        let opt = &self.type_to_model[smpl_type].gender_to_model[gender];
        opt.as_ref().map(|model| model.get_face_model())
    }
    #[allow(clippy::redundant_closure_for_method_calls)]
    pub fn get_model_mut(&mut self, smpl_type: SmplType, gender: Gender) -> Option<&mut dyn SmplModel<B>> {
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
    pub fn lazy_load_defaults(&mut self) {
        self.set_lazy_loading(SmplType::SmplX, Gender::Neutral, "./data/smplx/SMPLX_neutral_array_f32_slim.npz");
        self.set_lazy_loading(SmplType::SmplX, Gender::Male, "./data/smplx/SMPLX_male_array_f32_slim.npz");
        self.set_lazy_loading(SmplType::SmplX, Gender::Female, "./data/smplx/SMPLX_female_array_f32_slim.npz");
    }
    pub fn set_lazy_loading(&mut self, smpl_type: SmplType, gender: Gender, path: &str) {
        self.type_to_path[smpl_type].gender_to_path[gender] = Some(path.to_string());
        #[cfg(not(target_arch = "wasm32"))]
        assert!(
            std::path::Path::new(&path).exists(),
            "File at path {path} does not exist. Please follow the data download instructions in the README."
        );
    }
    #[cfg(not(target_arch = "wasm32"))]
    pub fn add_model_from_type(&mut self, smpl_type: SmplType, path: &str, gender: Gender, max_num_betas: usize, num_expression_components: usize) {
        match smpl_type {
            SmplType::SmplX | SmplType::SmplXS => {
                use crate::smpl_x::smpl_x_gpu::SmplXGPUG;
                let new_model = SmplXGPUG::new_from_npz(path, gender, max_num_betas, num_expression_components);
                self.add_model_under_type(smpl_type, new_model, true);
            }
            _ => panic!("Model loading for {smpl_type:?} if not supported yet!"),
        }
    }
}
pub type SmplCache = SmplCacheG<AppBackend>;
