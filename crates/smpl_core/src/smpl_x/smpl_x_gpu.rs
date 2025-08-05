use crate::{
    common::{
        betas::Betas,
        expression::Expression,
        outputs::SmplOutputDynamic,
        pose::Pose,
        smpl_model::{FaceModel, SmplCacheDynamic, SmplModel},
        smpl_options::SmplOptions,
        types::{Gender, SmplType, UpAxis},
    },
    conversions::pose_remap::PoseRemap,
};
use burn::tensor::{backend::Backend, Float, Int, Tensor};
use gloss_utils::bshare::{tensor_to_data_float, tensor_to_data_int, ToBurn};
use gloss_utils::nshare::ToNalgebra;
use log::{info, warn};
use nalgebra as na;
use ndarray as nd;
use ndarray::prelude::*;
use ndarray_npy::NpzReader;
use smpl_utils::numerical::{batch_rigid_transform, batch_rodrigues};
use smpl_utils::{array::Gather2D, io::FileLoader};
use std::ops::Sub;
use std::{
    any::Any,
    io::{Read, Seek},
};
pub const NUM_BODY_JOINTS: usize = 21;
pub const NUM_HAND_JOINTS: usize = 15;
pub const NUM_FACE_JOINTS: usize = 3;
pub const NUM_JOINTS: usize = NUM_BODY_JOINTS + 2 * NUM_HAND_JOINTS + NUM_FACE_JOINTS;
pub const NECK_IDX: usize = 12;
pub const NUM_VERTS: usize = 10475;
pub const NUM_VERTS_UV_MESH: usize = 11307;
pub const NUM_FACES: usize = 20908;
pub const FULL_SHAPE_SPACE_DIM: usize = 400;
pub const SHAPE_SPACE_DIM: usize = 300;
pub const EXPRESSION_SPACE_DIM: usize = 100;
pub const NUM_POSE_BLEND_SHAPES: usize = NUM_JOINTS * 9;
use burn::backend::{Candle, NdArray, Wgpu};
#[allow(clippy::large_enum_variant)]
#[derive(Clone)]
pub enum SmplXDynamic {
    NdArray(SmplXGPU<NdArray>),
    Wgpu(SmplXGPU<Wgpu>),
    Candle(SmplXGPU<Candle>),
}
#[allow(clippy::return_self_not_must_use)]
impl SmplXDynamic {
    #[cfg(not(target_arch = "wasm32"))]
    pub fn new_from_npz(models: &SmplCacheDynamic, path: &str, gender: Gender, max_num_betas: usize, num_expression_components: usize) -> Self {
        match models {
            SmplCacheDynamic::Wgpu(_) => {
                info!("Initializing with Wgpu Backend");
                let model = SmplXGPU::<Wgpu>::new_from_npz(path, gender, max_num_betas, num_expression_components);
                SmplXDynamic::Wgpu(model)
            }
            SmplCacheDynamic::NdArray(_) => {
                info!("Initializing with NdArray Backend");
                let model = SmplXGPU::<NdArray>::new_from_npz(path, gender, max_num_betas, num_expression_components);
                SmplXDynamic::NdArray(model)
            }
            SmplCacheDynamic::Candle(_) => {
                info!("Initializing with Candle Backend");
                let model = SmplXGPU::<Candle>::new_from_npz(path, gender, max_num_betas, num_expression_components);
                SmplXDynamic::Candle(model)
            }
        }
    }
    pub fn new_from_reader<R: Read + Seek>(
        models: &SmplCacheDynamic,
        reader: R,
        gender: Gender,
        max_num_betas: usize,
        max_num_expression_components: usize,
    ) -> Self {
        match models {
            SmplCacheDynamic::Wgpu(_) => {
                info!("Initializing from reader with Wgpu Backend");
                let model = SmplXGPU::<Wgpu>::new_from_reader(reader, gender, max_num_betas, max_num_expression_components);
                SmplXDynamic::Wgpu(model)
            }
            SmplCacheDynamic::NdArray(_) => {
                info!("Initializing from reader with NdArray Backend");
                let model = SmplXGPU::<NdArray>::new_from_reader(reader, gender, max_num_betas, max_num_expression_components);
                SmplXDynamic::NdArray(model)
            }
            SmplCacheDynamic::Candle(_) => {
                info!("Initializing from reader with Candle Backend");
                let model = SmplXGPU::<Candle>::new_from_reader(reader, gender, max_num_betas, max_num_expression_components);
                SmplXDynamic::Candle(model)
            }
        }
    }
    pub async fn new_from_npz_async(
        models: &SmplCacheDynamic,
        path: &str,
        gender: Gender,
        max_num_betas: usize,
        num_expression_components: usize,
    ) -> Self {
        match models {
            SmplCacheDynamic::Wgpu(_) => {
                info!("Initializing with Wgpu Backend");
                let model = SmplXGPU::<Wgpu>::new_from_npz_async(path, gender, max_num_betas, num_expression_components).await;
                SmplXDynamic::Wgpu(model)
            }
            SmplCacheDynamic::NdArray(_) => {
                info!("Initializing with NdArray Backend");
                let model = SmplXGPU::<NdArray>::new_from_npz_async(path, gender, max_num_betas, num_expression_components).await;
                SmplXDynamic::NdArray(model)
            }
            SmplCacheDynamic::Candle(_) => {
                info!("Initializing with Candle Backend");
                let model = SmplXGPU::<Candle>::new_from_npz_async(path, gender, max_num_betas, num_expression_components).await;
                SmplXDynamic::Candle(model)
            }
        }
    }
}
#[derive(Clone)]
pub struct SmplXGPU<B: Backend> {
    pub device: B::Device,
    pub smpl_type: SmplType,
    pub gender: Gender,
    pub verts_template: Tensor<B, 2, Float>,
    pub faces: Tensor<B, 2, Int>,
    pub faces_uv_mesh: Tensor<B, 2, Int>,
    pub uv: Tensor<B, 2, Float>,
    pub shape_dirs: Tensor<B, 2, Float>,
    pub expression_dirs: Option<Tensor<B, 2, Float>>,
    pub pose_dirs: Option<Tensor<B, 2, Float>>,
    pub joint_regressor: Tensor<B, 2, Float>,
    pub parent_idx_per_joint: Tensor<B, 1, Int>,
    pub lbs_weights: Tensor<B, 2, Float>,
    pub verts_ones: Tensor<B, 2, Float>,
    pub idx_vuv_2_vnouv: Tensor<B, 1, Int>,
    pub faces_na: na::DMatrix<u32>,
    pub faces_uv_mesh_na: na::DMatrix<u32>,
    pub uv_na: na::DMatrix<f32>,
    pub idx_vuv_2_vnouv_vec: Vec<usize>,
    pub lbs_weights_split: Tensor<B, 2>,
    pub lbs_weights_nd: nd::ArcArray2<f32>,
    pub lbs_weights_split_nd: nd::ArcArray2<f32>,
}
impl<B: Backend> SmplXGPU<B> {
    /// # Panics
    /// Will panic if the matrices don't match the expected sizes
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::too_many_lines)]
    pub fn new_from_matrices(
        gender: Gender,
        verts_template: &nd::Array2<f32>,
        faces: &nd::Array2<u32>,
        faces_uv_mesh: &nd::Array2<u32>,
        uv: &nd::Array2<f32>,
        shape_dirs: &nd::Array3<f32>,
        expression_dirs: Option<nd::Array3<f32>>,
        pose_dirs: Option<nd::Array3<f32>>,
        joint_regressor: &nd::Array2<f32>,
        parent_idx_per_joint: &nd::Array1<u32>,
        lbs_weights: nd::Array2<f32>,
        max_num_betas: usize,
        max_num_expression_components: usize,
    ) -> Self {
        let device = B::Device::default();
        let b_verts_template = verts_template.to_burn(&device);
        let b_faces = faces.to_burn(&device);
        let b_faces_uv_mesh = faces_uv_mesh.to_burn(&device);
        let b_uv = uv.to_burn(&device);
        let actual_num_betas = max_num_betas.min(shape_dirs.shape()[2]);
        let shape_dirs = shape_dirs
            .slice_axis(Axis(2), ndarray::Slice::from(0..actual_num_betas))
            .to_owned()
            .into_shape_with_order((NUM_VERTS * 3, actual_num_betas))
            .unwrap();
        let b_shape_dirs = shape_dirs.to_burn(&device);
        let b_expression_dirs = expression_dirs.map(|expression_dirs| {
            let actual_num_expression_components = max_num_expression_components.min(expression_dirs.shape()[2]);
            let expression_dirs = expression_dirs
                .slice_axis(nd::Axis(2), nd::Slice::from(0..actual_num_expression_components))
                .into_shape_with_order((NUM_VERTS * 3, actual_num_expression_components))
                .unwrap()
                .to_owned();
            expression_dirs.to_burn(&device)
        });
        let b_pose_dirs = pose_dirs.map(|pose_dirs| {
            let pose_dirs = pose_dirs.into_shape_with_order((NUM_VERTS * 3, NUM_JOINTS * 9)).unwrap();
            pose_dirs.to_burn(&device)
        });
        let b_joint_regressor = joint_regressor.to_burn(&device);
        let b_parent_idx_per_joint = parent_idx_per_joint.to_burn(&device).reshape([NUM_JOINTS + 1]);
        let b_lbs_weights = lbs_weights.to_burn(&device);
        #[allow(clippy::cast_possible_wrap)]
        let faces_uv_mesh_i32: nd::Array2<i32> = faces_uv_mesh.mapv(|x| x as i32);
        let ft: nd::ArcArray2<i32> = faces_uv_mesh_i32.into();
        let max_v_uv_idx = *ft.iter().max_by_key(|&x| x).unwrap();
        let max_v_uv_idx_usize = usize::try_from(max_v_uv_idx).unwrap_or_else(|_| panic!("Cannot cast max_v_uv_idx to usize"));
        let mut idx_vuv_2_vnouv = nd::ArcArray1::<i32>::zeros(max_v_uv_idx_usize + 1);
        for (fuv, fnouv) in ft.axis_iter(nd::Axis(0)).zip(faces.axis_iter(nd::Axis(0))) {
            let uv_0 = fuv[[0]];
            let uv_1 = fuv[[1]];
            let uv_2 = fuv[[2]];
            let nouv_0 = fnouv[[0]];
            let nouv_1 = fnouv[[1]];
            let nouv_2 = fnouv[[2]];
            idx_vuv_2_vnouv[usize::try_from(uv_0).unwrap_or_else(|_| panic!("Cannot cast uv_0 to usize"))] =
                i32::try_from(nouv_0).unwrap_or_else(|_| panic!("Cannot cast nouv_0 to i32"));
            idx_vuv_2_vnouv[usize::try_from(uv_1).unwrap_or_else(|_| panic!("Cannot cast uv_1 to usize"))] =
                i32::try_from(nouv_1).unwrap_or_else(|_| panic!("Cannot cast nouv_1 to i32"));
            idx_vuv_2_vnouv[usize::try_from(uv_2).unwrap_or_else(|_| panic!("Cannot cast uv_2 to usize"))] =
                i32::try_from(nouv_2).unwrap_or_else(|_| panic!("Cannot cast nouv_2 to i32"));
        }
        let idx_vuv_2_vnouv_vec: Vec<i32> = idx_vuv_2_vnouv.mapv(|x| x).into_raw_vec_and_offset().0;
        let idx_vuv_2_vnouv_slice: &[i32] = &idx_vuv_2_vnouv_vec;
        let b_idx_vuv_2_vnouv = Tensor::<B, 1, Int>::from_ints(idx_vuv_2_vnouv_slice, &device);
        let idx_vuv_2_vnouv_vec: Vec<usize> = idx_vuv_2_vnouv
            .to_vec()
            .iter()
            .map(|&x| usize::try_from(x).unwrap_or_else(|_| panic!("Cannot cast negative value to usize")))
            .collect();
        let faces_na = faces.view().into_nalgebra().clone_owned().map(|x| x);
        let faces_uv_mesh_na = ft
            .view()
            .into_nalgebra()
            .clone_owned()
            .map(|x| u32::try_from(x).unwrap_or_else(|_| panic!("Cannot cast value to u32")));
        let uv_na = uv.view().into_nalgebra().clone_owned();
        let cols: Vec<usize> = (0..lbs_weights.ncols()).collect();
        let lbs_weights_split: nd::ArcArray2<f32> = lbs_weights.to_owned().gather(&idx_vuv_2_vnouv_vec, &cols).into();
        let b_lbs_weights_split =
            Tensor::<B, 1>::from_floats(lbs_weights_split.as_slice().unwrap(), &device).reshape([idx_vuv_2_vnouv_vec.len(), NUM_JOINTS + 1]);
        let verts_ones = Tensor::<B, 2>::ones([NUM_VERTS, 1], &device);
        let lbs_weights_nd: nd::ArcArray2<f32> = lbs_weights.into();
        let cols: Vec<usize> = (0..lbs_weights_nd.ncols()).collect();
        let lbs_weights_split_nd = lbs_weights_nd.to_owned().gather(&idx_vuv_2_vnouv_vec, &cols).into();
        info!("Initialised burn on Backend: {:?}", B::name());
        info!("Device: {:?}", &device);
        Self {
            smpl_type: SmplType::SmplX,
            gender,
            device,
            verts_template: b_verts_template,
            faces: b_faces,
            faces_uv_mesh: b_faces_uv_mesh,
            uv: b_uv,
            shape_dirs: b_shape_dirs,
            expression_dirs: b_expression_dirs,
            pose_dirs: b_pose_dirs,
            joint_regressor: b_joint_regressor,
            parent_idx_per_joint: b_parent_idx_per_joint,
            lbs_weights: b_lbs_weights,
            verts_ones,
            idx_vuv_2_vnouv: b_idx_vuv_2_vnouv,
            faces_na,
            faces_uv_mesh_na,
            uv_na,
            idx_vuv_2_vnouv_vec,
            lbs_weights_split: b_lbs_weights_split,
            lbs_weights_nd,
            lbs_weights_split_nd,
        }
    }
    /// # Panics
    /// Will panic if the path cannot be opened
    fn new_from_npz_reader<R: Read + Seek>(
        npz: &mut NpzReader<R>,
        gender: Gender,
        max_num_betas: usize,
        max_num_expression_components: usize,
    ) -> Self {
        let verts_template: nd::Array2<f32> = npz.by_name("v_template").unwrap();
        let faces: nd::Array2<u32> = npz.by_name("f").unwrap();
        let uv: nd::Array2<f32> = npz.by_name("vt").unwrap();
        let full_shape_dirs: nd::Array3<f32> = npz.by_name("shapedirs").unwrap();
        let (shape_dirs, expression_dirs) = if let Ok(expression_dirs) = npz.by_name("expressiondirs") {
            (full_shape_dirs, Some(expression_dirs))
        } else {
            let num_available_betas = full_shape_dirs.shape()[2];
            let num_full_betas = 300;
            let num_betas_to_use = num_full_betas.min(max_num_betas).min(num_available_betas);
            let shape_dirs = full_shape_dirs.slice_axis(nd::Axis(2), nd::Slice::from(0..num_betas_to_use)).to_owned();
            let expression_dirs = if full_shape_dirs.shape()[2] > 300 {
                Some(
                    full_shape_dirs
                        .slice_axis(nd::Axis(2), nd::Slice::from(300..300 + max_num_expression_components.min(100)))
                        .to_owned(),
                )
            } else {
                None
            };
            (shape_dirs, expression_dirs)
        };
        let pose_dirs: Option<nd::Array3<f32>> = npz.by_name("posedirs").ok();
        let joint_regressor: nd::Array2<f32> = npz.by_name("J_regressor").unwrap();
        let parent_idx_per_joint: nd::Array2<i32> = npz.by_name("kintree_table").unwrap();
        #[allow(clippy::cast_sign_loss)]
        let parent_idx_per_joint = parent_idx_per_joint.mapv(|x| x as u32);
        let parent_idx_per_joint = parent_idx_per_joint
            .slice_axis(nd::Axis(0), nd::Slice::from(0..1))
            .to_owned()
            .into_shape_with_order(NUM_JOINTS + 1)
            .unwrap();
        let lbs_weights: nd::Array2<f32> = npz.by_name("weights").unwrap();
        let ft: nd::Array2<u32> = npz.by_name("ft").unwrap();
        if pose_dirs.is_none() {
            warn!("No pose_dirs loaded from npz");
        }
        Self::new_from_matrices(
            gender,
            &verts_template,
            &faces,
            &ft,
            &uv,
            &shape_dirs,
            expression_dirs,
            pose_dirs,
            &joint_regressor,
            &parent_idx_per_joint,
            lbs_weights,
            max_num_betas,
            max_num_expression_components,
        )
    }
    #[cfg(not(target_arch = "wasm32"))]
    /// # Panics
    /// Will panic if the path cannot be opened
    pub fn new_from_npz(model_path: &str, gender: Gender, max_num_betas: usize, max_num_expression_components: usize) -> Self {
        let mut npz = NpzReader::new(std::fs::File::open(model_path).unwrap()).unwrap();
        Self::new_from_npz_reader(&mut npz, gender, max_num_betas, max_num_expression_components)
    }
    /// # Panics
    /// Will panic if the path cannot be opened
    /// Will panic if the translation and rotation do not cover the same number
    /// of timesteps
    #[allow(clippy::cast_possible_truncation)]
    pub async fn new_from_npz_async(model_path: &str, gender: Gender, max_num_betas: usize, max_num_expression_components: usize) -> Self {
        let reader = FileLoader::open(model_path).await;
        let mut npz = NpzReader::new(reader).unwrap();
        Self::new_from_npz_reader(&mut npz, gender, max_num_betas, max_num_expression_components)
    }
    /// # Panics
    /// Will panic if the path cannot be opened
    /// Will panic if the translation and rotation do not cover the same number
    /// of timesteps
    #[allow(clippy::cast_possible_truncation)]
    pub fn new_from_reader<R: Read + Seek>(reader: R, gender: Gender, max_num_betas: usize, max_num_expression_components: usize) -> Self {
        let mut npz = NpzReader::new(reader).unwrap();
        Self::new_from_npz_reader(&mut npz, gender, max_num_betas, max_num_expression_components)
    }
    /// # Panics
    /// Will panic if the path cannot be opened
    /// Will panic if the translation and rotation do not cover the same number
    /// of timesteps
    #[allow(clippy::cast_possible_truncation)]
    pub fn read_pose_dirs_from_reader<R: Read + Seek>(reader: R, device: &B::Device) -> Tensor<B, 2, Float> {
        let mut npz = NpzReader::new(reader).unwrap();
        let pose_dirs: Option<nd::Array3<f32>> = Some(npz.by_name("pose_dirs").unwrap());
        let b_pose_dirs =
            pose_dirs.map(|pose_dirs| Tensor::<B, 1>::from_floats(pose_dirs.as_slice().unwrap(), device).reshape([NUM_VERTS * 3, NUM_JOINTS * 9]));
        b_pose_dirs.unwrap()
    }
}
impl<B: Backend> FaceModel<B> for SmplXGPU<B>
where
    B::IntTensorPrimitive<1>: Sync,
    B::IntTensorPrimitive<2>: Sync,
    B::FloatTensorPrimitive<2>: Sync,
    B::QuantizedTensorPrimitive<2>: std::marker::Sync,
{
    #[allow(clippy::missing_panics_doc)]
    #[allow(non_snake_case)]
    #[allow(clippy::let_and_return)]
    fn expression2offsets(&self, expression: &Expression) -> Tensor<B, 2, Float> {
        let device = self.verts_template.device();
        let offsets = if let Some(ref expression_dirs) = self.expression_dirs {
            let input_nr_expression_coeffs = expression.expr_coeffs.len();
            let model_nr_expression_coeffs = expression_dirs.shape().dims[1];
            let nr_expression_coeffs = input_nr_expression_coeffs.min(model_nr_expression_coeffs);
            let expr_sliced = expression.expr_coeffs.slice(s![0..nr_expression_coeffs]);
            let expr_tensor = Tensor::<B, 1, Float>::from_floats(expr_sliced.as_slice().unwrap(), &device);
            let expression_dirs_sliced = expression_dirs.clone().slice([0..expression_dirs.dims()[0], 0..nr_expression_coeffs]);
            let v_expr_offsets = expression_dirs_sliced.matmul(expr_tensor.reshape([-1, 1]));
            v_expr_offsets.reshape([NUM_VERTS, 3])
        } else {
            Tensor::<B, 2, Float>::zeros([NUM_VERTS, 3], &device)
        };
        offsets
    }
}
impl<B: Backend> SmplModel<B> for SmplXGPU<B>
where
    B::FloatTensorPrimitive<2>: Sync,
    B::IntTensorPrimitive<2>: Sync,
    B::IntTensorPrimitive<1>: Sync,
    B::QuantizedTensorPrimitive<1>: std::marker::Sync,
    B::QuantizedTensorPrimitive<2>: std::marker::Sync,
{
    fn clone_dyn(&self) -> Box<dyn SmplModel<B>> {
        Box::new(self.clone())
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn smpl_type(&self) -> SmplType {
        self.smpl_type
    }
    fn gender(&self) -> Gender {
        self.gender
    }
    fn get_face_model(&self) -> &dyn FaceModel<B> {
        self
    }
    #[allow(clippy::missing_panics_doc)]
    #[allow(non_snake_case)]
    fn forward(&self, options: &SmplOptions, betas: &Betas, pose_raw: &Pose, expression: Option<&Expression>) -> SmplOutputDynamic<B> {
        let mut verts_t_pose = self.betas2verts(betas);
        if let Some(expression) = expression {
            verts_t_pose = verts_t_pose + self.expression2offsets(expression);
        }
        let pose_remap = PoseRemap::new(pose_raw.smpl_type, SmplType::SmplX);
        let pose = pose_remap.remap(pose_raw);
        let joints_t_pose = self.verts2joints(verts_t_pose.clone());
        if options.enable_pose_corrective {
            let verts_offset = self.compute_pose_correctives(&pose);
            verts_t_pose = verts_t_pose + verts_offset;
        }
        let (verts_posed_nd, _, _, joints_posed) = self.apply_pose(&verts_t_pose, None, None, &joints_t_pose, &self.lbs_weights, &pose);
        SmplOutputDynamic {
            verts: verts_posed_nd,
            faces: self.faces.clone(),
            normals: None,
            uvs: None,
            joints: joints_posed,
        }
    }
    fn create_body_with_uv(&self, smpl_merged: &SmplOutputDynamic<B>) -> SmplOutputDynamic<B> {
        let cols_tensor = Tensor::<B, 1, Int>::from_ints([0, 1, 2], &self.device);
        let mapping_tensor = self.idx_split_2_merged();
        let v_burn_split = smpl_merged.verts.clone().select(0, mapping_tensor.clone());
        let v_burn_split = v_burn_split.select(1, cols_tensor.clone());
        let n_burn_split = smpl_merged
            .normals
            .as_ref()
            .map(|n| n.clone().select(0, mapping_tensor).select(1, cols_tensor));
        SmplOutputDynamic {
            verts: v_burn_split,
            faces: self.faces_uv_mesh.clone(),
            normals: n_burn_split,
            uvs: Some(self.uv.clone()),
            joints: smpl_merged.joints.clone(),
        }
    }
    #[allow(clippy::missing_panics_doc)]
    #[allow(non_snake_case)]
    #[allow(clippy::let_and_return)]
    fn betas2verts(&self, betas: &Betas) -> Tensor<B, 2, Float> {
        let device = self.verts_template.device();
        let input_nr_betas = betas.betas.len();
        let model_nr_betas = self.shape_dirs.shape().dims[1];
        let nr_betas = input_nr_betas.min(model_nr_betas);
        let betas_sliced = betas.betas.slice(s![0..nr_betas]);
        let betas_tensor = Tensor::<B, 1, Float>::from_floats(betas_sliced.as_slice().unwrap(), &device);
        let shape_dirs_sliced = self.shape_dirs.clone().slice([0..self.shape_dirs.dims()[0], 0..nr_betas]);
        let v_beta_offsets = shape_dirs_sliced.matmul(betas_tensor.reshape([-1, 1]));
        let v_beta_offsets_reshaped = v_beta_offsets.reshape([NUM_VERTS, 3]);
        let verts_t_pose = v_beta_offsets_reshaped.add(self.verts_template.clone());
        verts_t_pose
    }
    fn verts2joints(&self, verts_t_pose: Tensor<B, 2, Float>) -> Tensor<B, 2, Float> {
        self.joint_regressor.clone().matmul(verts_t_pose)
    }
    #[allow(clippy::missing_panics_doc)]
    fn compute_pose_correctives(&self, pose: &Pose) -> Tensor<B, 2, Float> {
        let offsets = if let Some(pose_dirs) = &self.pose_dirs {
            let full_pose = &pose.joint_poses;
            assert!(
                full_pose.dim().0 == NUM_JOINTS + 1,
                "The pose does not have the correct number of joints for this model. Maybe you need to add a PoseRemapper component?\n {:?} != {:?}",
                full_pose.dim().0,
                NUM_JOINTS + 1
            );
            let mut rot_mats = batch_rodrigues(full_pose);
            let identity = ndarray::Array2::<f32>::eye(3);
            let pose_feature = (rot_mats.slice_mut(s![1.., .., ..]).sub(&identity))
                .into_shape_with_order(NUM_JOINTS * 9)
                .unwrap();
            let b_pose_feature = Tensor::<B, 1, Float>::from_floats(pose_feature.as_slice().unwrap(), &self.device).reshape([NUM_JOINTS * 9, 1]);
            let new_pose_dirs = pose_dirs.clone();
            let all_pose_offsets = new_pose_dirs.matmul(b_pose_feature);
            all_pose_offsets.reshape([NUM_VERTS, 3])
        } else {
            Tensor::<B, 2, Float>::zeros([NUM_VERTS, 3], &self.device)
        };
        offsets
    }
    #[allow(clippy::missing_panics_doc)]
    fn compute_pose_feature(&self, pose: &Pose) -> nd::Array1<f32> {
        let full_pose = &pose.joint_poses;
        assert!(
            full_pose.dim().0 == NUM_JOINTS + 1,
            "The pose does not have the correct number of joints for this model. Maybe you need to add a PoseRemapper component?\n {:?} != {:?}",
            full_pose.dim().0,
            NUM_JOINTS + 1
        );
        let mut rot_mats = batch_rodrigues(full_pose);
        let identity = ndarray::Array2::<f32>::eye(3);
        let pose_feature = (rot_mats.slice_mut(s![1.., .., ..]).sub(&identity))
            .into_shape_with_order(NUM_JOINTS * 9)
            .unwrap();
        pose_feature
    }
    #[allow(clippy::missing_panics_doc)]
    #[allow(non_snake_case)]
    #[allow(clippy::cast_precision_loss)]
    #[allow(clippy::cast_sign_loss)]
    #[allow(clippy::too_many_lines)]
    #[allow(clippy::similar_names)]
    fn apply_pose(
        &self,
        verts_t_pose: &Tensor<B, 2, Float>,
        normals: Option<&Tensor<B, 2, Float>>,
        tangents: Option<&Tensor<B, 2, Float>>,
        joints: &Tensor<B, 2, Float>,
        lbs_weights: &Tensor<B, 2, Float>,
        pose: &Pose,
    ) -> (
        Tensor<B, 2, Float>,
        Option<Tensor<B, 2, Float>>,
        Option<Tensor<B, 2, Float>>,
        Tensor<B, 2, Float>,
    ) {
        assert!(
            verts_t_pose.shape().dims[0] == lbs_weights.shape().dims[0],
            "Verts and LBS weights should match"
        );
        let full_pose = &pose.joint_poses;
        assert!(
            full_pose.shape()[0] == NUM_JOINTS + 1,
            "The pose does not have the correct number of joints for this model."
        );
        let rot_mats = batch_rodrigues(full_pose);
        let joints_data = tensor_to_data_float(joints);
        let shape = joints.shape().dims;
        let nd_joints = nd::Array2::from_shape_vec((shape[0], shape[1]), joints_data).expect("Shape mismatch during tensor to ndarray conversion");
        let parent_idx_data_i32: Vec<i32> = tensor_to_data_int(&self.parent_idx_per_joint);
        let parent_idx_data_u32: Vec<u32> = parent_idx_data_i32.into_iter().map(|x| x as u32).collect();
        let (posed_joints_nd, rel_transforms_nd) = batch_rigid_transform(parent_idx_data_u32, &rot_mats, &nd_joints, NUM_JOINTS);
        let posed_joints = posed_joints_nd.to_burn(&self.device);
        let nr_verts = verts_t_pose.shape().dims[0];
        let nr_joints = posed_joints.shape().dims[0];
        let v_posed = verts_t_pose.clone();
        let W = lbs_weights;
        let A_nd = rel_transforms_nd.into_shape_with_order((NUM_JOINTS + 1, 16)).unwrap();
        let A = A_nd.to_burn(&self.device);
        let T = W.clone().matmul(A).reshape([nr_verts, 4, 4]);
        let dims_3 = 3;
        let rot0 = T.clone().slice([0..nr_verts, 0..1, 0..dims_3]).squeeze(1);
        let rot1 = T.clone().slice([0..nr_verts, 1..2, 0..dims_3]).squeeze(1);
        let rot2 = T.clone().slice([0..nr_verts, 2..3, 0..dims_3]).squeeze(1);
        let trans: Tensor<B, 2> = T.slice([0..nr_verts, 0..dims_3, 3..4]).squeeze(2);
        let verts_final_0 = rot0.clone().mul(v_posed.clone()).sum_dim(1);
        let verts_final_1 = rot1.clone().mul(v_posed.clone()).sum_dim(1);
        let verts_final_2 = rot2.clone().mul(v_posed.clone()).sum_dim(1);
        let verts_final = Tensor::<B, 1>::stack(vec![verts_final_0.squeeze(1), verts_final_1.squeeze(1), verts_final_2.squeeze(1)], 1);
        let verts_final = verts_final.add(trans);
        let mut normals_final = if let Some(normals) = normals {
            let normals_0 = rot0.clone().mul(normals.clone()).sum_dim(1);
            let normals_1 = rot1.clone().mul(normals.clone()).sum_dim(1);
            let normals_2 = rot2.clone().mul(normals.clone()).sum_dim(1);
            let normals_final = Tensor::<B, 1>::stack(vec![normals_0.squeeze(1), normals_1.squeeze(1), normals_2.squeeze(1)], 1);
            Some(normals_final)
        } else {
            None
        };
        let mut tangents_final = if let Some(tangents) = tangents {
            let tangents_3 = tangents.clone().slice([0..nr_verts, 0..3]);
            let tangents_0 = rot0.mul(tangents_3.clone()).sum_dim(1);
            let tangents_1 = rot1.mul(tangents_3.clone()).sum_dim(1);
            let tangents_2 = rot2.mul(tangents_3.clone()).sum_dim(1);
            let handedness: Tensor<B, 1> = tangents.clone().slice([0..nr_verts, 3..4]).squeeze(1);
            let tangents_final = Tensor::<B, 1>::stack(vec![tangents_0.squeeze(1), tangents_1.squeeze(1), tangents_2.squeeze(1), handedness], 1);
            Some(tangents_final)
        } else {
            None
        };
        let trans_pose_nd = pose.global_trans.clone();
        let trans_pose = trans_pose_nd.to_burn(&self.device);
        let trans_pose_broadcasted_v = trans_pose.clone().reshape([1, 3]).expand(verts_final.shape());
        let trans_pose_broadcasted_p = trans_pose.reshape([1, 3]).expand(posed_joints.shape());
        let mut verts_final_modified = verts_final.clone().add(trans_pose_broadcasted_v.clone());
        let mut posed_joints_modified = posed_joints.clone().add(trans_pose_broadcasted_p.clone());
        if pose.up_axis == UpAxis::Z {
            let vcol0: Tensor<B, 1> = verts_final_modified.clone().slice([0..nr_verts, 0..1]).squeeze(1);
            let vcol1: Tensor<B, 1> = verts_final_modified.clone().slice([0..nr_verts, 1..2]).squeeze(1);
            let vcol2: Tensor<B, 1> = verts_final_modified.clone().slice([0..nr_verts, 2..3]).squeeze(1);
            let verts_new_col1 = vcol2;
            let verts_new_col2 = vcol1.mul_scalar(-1.0);
            verts_final_modified = Tensor::stack::<2>(vec![vcol0, verts_new_col1, verts_new_col2], 1);
            let jcol0: Tensor<B, 1> = posed_joints_modified.clone().slice([0..nr_joints, 0..1]).squeeze(1);
            let jcol1: Tensor<B, 1> = posed_joints_modified.clone().slice([0..nr_joints, 1..2]).squeeze(1);
            let jcol2: Tensor<B, 1> = posed_joints_modified.clone().slice([0..nr_joints, 2..3]).squeeze(1);
            let joints_new_col1 = jcol2;
            let joints_new_col2 = jcol1.mul_scalar(-1.0);
            posed_joints_modified = Tensor::stack::<2>(vec![jcol0, joints_new_col1, joints_new_col2], 1);
            if let Some(ref mut normals) = normals_final {
                let ncol0: Tensor<B, 1> = normals.clone().slice([0..nr_verts, 0..1]).squeeze(1);
                let ncol1: Tensor<B, 1> = normals.clone().slice([0..nr_verts, 1..2]).squeeze(1);
                let ncol2: Tensor<B, 1> = normals.clone().slice([0..nr_verts, 2..3]).squeeze(1);
                let normals_new_col1 = ncol2;
                let normals_new_col2 = ncol1.mul_scalar(-1.0);
                let normals_final_modified = Tensor::stack::<2>(vec![ncol0, normals_new_col1, normals_new_col2], 1);
                *normals = normals_final_modified;
            }
            if let Some(ref mut tangents) = tangents_final {
                let tcol0: Tensor<B, 1> = tangents.clone().slice([0..nr_verts, 0..1]).squeeze(1);
                let tcol1: Tensor<B, 1> = tangents.clone().slice([0..nr_verts, 1..2]).squeeze(1);
                let tcol2: Tensor<B, 1> = tangents.clone().slice([0..nr_verts, 2..3]).squeeze(1);
                let tangents_new_col1 = tcol2;
                let tangents_new_col2 = tcol1.mul_scalar(-1.0);
                let handedness = tangents.clone().slice([0..nr_verts, 3..4]).squeeze(1);
                let tangents_final_modified = Tensor::stack::<2>(vec![tcol0, tangents_new_col1, tangents_new_col2, handedness], 1);
                *tangents = tangents_final_modified;
            }
        }
        (verts_final_modified, normals_final.clone(), tangents_final.clone(), posed_joints_modified)
    }
    fn faces(&self) -> &Tensor<B, 2, Int> {
        &self.faces
    }
    fn faces_uv(&self) -> &Tensor<B, 2, Int> {
        &self.faces_uv_mesh
    }
    fn uv(&self) -> &Tensor<B, 2, Float> {
        &self.uv
    }
    fn lbs_weights(&self) -> Tensor<B, 2, Float> {
        self.lbs_weights.clone()
    }
    fn lbs_weights_split(&self) -> Tensor<B, 2, Float> {
        self.lbs_weights_split.clone()
    }
    fn idx_split_2_merged(&self) -> Tensor<B, 1, Int> {
        self.idx_vuv_2_vnouv.clone()
    }
    fn idx_split_2_merged_vec(&self) -> &Vec<usize> {
        &self.idx_vuv_2_vnouv_vec
    }
    fn set_pose_dirs(&mut self, pose_dirs: Tensor<B, 2, Float>) {
        self.pose_dirs = Some(pose_dirs);
    }
    fn get_pose_dirs(&self) -> Tensor<B, 2, Float> {
        if let Some(pose_dirs_tensor) = self.pose_dirs.clone() {
            pose_dirs_tensor
        } else {
            panic!("pose_dirs is not available!");
        }
    }
    fn get_expression_dirs(&self) -> Option<Tensor<B, 2, Float>> {
        self.expression_dirs.clone()
    }
}
