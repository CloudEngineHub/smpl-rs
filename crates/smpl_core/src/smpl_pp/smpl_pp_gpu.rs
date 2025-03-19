use log::info;
use nalgebra as na;
use ndarray as nd;
use ndarray::prelude::*;
use ndarray_npy::NpzReader;
use burn::{
    backend::{Candle, NdArray, Wgpu},
    tensor::{backend::Backend, Float, Int, Tensor},
};
use core::f32;
use gloss_utils::nshare::ToNalgebra;
use std::{
    any::Any, f32::consts::PI, io::{Read, Seek},
    ops::Sub,
};
use crate::{
    common::{
        betas::Betas, expression::Expression, outputs::SmplOutputDynamic, pose::Pose,
        smpl_model::{SmplCacheDynamic, SmplModel},
        smpl_options::SmplOptions, types::{Gender, SmplType},
    },
    conversions::pose_remap::PoseRemap,
};
use super::osim_rot::{
    ConstantCurvatureJoint, CustomJoint, EllipsoidJoint, OsimJoint, PinJoint, WalkerKnee,
};
use smpl_utils::{array::Gather2D, io::FileLoader, numerical::batch_rodrigues};
use gloss_utils::bshare::{tensor_to_data_float, ToBurn};
pub const NUM_VERTS: usize = 31028;
pub const NUM_JOINTS: usize = 24;
pub const NUM_BODY_JOINTS: usize = 24;
#[allow(clippy::large_enum_variant)]
#[derive(Clone)]
pub enum SmplPPDynamic {
    NdArray(SmplPPGPU<NdArray>),
    Wgpu(SmplPPGPU<Wgpu>),
    Candle(SmplPPGPU<Candle>),
}
#[allow(clippy::return_self_not_must_use)]
impl SmplPPDynamic {
    #[cfg(not(target_arch = "wasm32"))]
    pub fn new_from_npz(
        models: &SmplCacheDynamic,
        path: &str,
        gender: Gender,
        max_num_betas: usize,
        num_expression_components: usize,
    ) -> Self {
        match models {
            SmplCacheDynamic::Wgpu(_) => {
                println!("Initializing with Wgpu Backend");
                let model = SmplPPGPU::<
                    Wgpu,
                >::new_from_npz(path, gender, max_num_betas, num_expression_components);
                SmplPPDynamic::Wgpu(model)
            }
            SmplCacheDynamic::NdArray(_) => {
                println!("Initializing with NdArray Backend");
                let model = SmplPPGPU::<
                    NdArray,
                >::new_from_npz(path, gender, max_num_betas, num_expression_components);
                SmplPPDynamic::NdArray(model)
            }
            SmplCacheDynamic::Candle(_) => {
                println!("Initializing with Candle Backend");
                let model = SmplPPGPU::<
                    Candle,
                >::new_from_npz(path, gender, max_num_betas, num_expression_components);
                SmplPPDynamic::Candle(model)
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
                println!("Initializing from reader with Wgpu Backend");
                let model = SmplPPGPU::<
                    Wgpu,
                >::new_from_reader(
                    reader,
                    gender,
                    max_num_betas,
                    max_num_expression_components,
                );
                SmplPPDynamic::Wgpu(model)
            }
            SmplCacheDynamic::NdArray(_) => {
                println!("Initializing from reader with NdArray Backend");
                let model = SmplPPGPU::<
                    NdArray,
                >::new_from_reader(
                    reader,
                    gender,
                    max_num_betas,
                    max_num_expression_components,
                );
                SmplPPDynamic::NdArray(model)
            }
            SmplCacheDynamic::Candle(_) => {
                println!("Initializing from reader with Candle Backend");
                let model = SmplPPGPU::<
                    Candle,
                >::new_from_reader(
                    reader,
                    gender,
                    max_num_betas,
                    max_num_expression_components,
                );
                SmplPPDynamic::Candle(model)
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
                println!("Initializing with Wgpu Backend");
                let model = SmplPPGPU::<
                    Wgpu,
                >::new_from_npz_async(
                        path,
                        gender,
                        max_num_betas,
                        num_expression_components,
                    )
                    .await;
                SmplPPDynamic::Wgpu(model)
            }
            SmplCacheDynamic::NdArray(_) => {
                println!("Initializing with NdArray Backend");
                let model = SmplPPGPU::<
                    NdArray,
                >::new_from_npz_async(
                        path,
                        gender,
                        max_num_betas,
                        num_expression_components,
                    )
                    .await;
                SmplPPDynamic::NdArray(model)
            }
            SmplCacheDynamic::Candle(_) => {
                println!("Initializing with Candle Backend");
                let model = SmplPPGPU::<
                    Candle,
                >::new_from_npz_async(
                        path,
                        gender,
                        max_num_betas,
                        num_expression_components,
                    )
                    .await;
                SmplPPDynamic::Candle(model)
            }
        }
    }
}
#[derive(Clone)]
pub struct SmplPPGPU<B: Backend> {
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
    pub parents: Vec<u32>,
    pub children: Vec<u32>,
    pub joint_idx_fixed_beta: Vec<u32>,
    pub kintree_table: Tensor<B, 2, Int>,
    pub joint_regressor_osim: Tensor<B, 2, Float>,
    pub per_joint_rot: Tensor<B, 3, Float>,
    pub parameter_mapping: Tensor<B, 1, Float>,
    pub tpose_transfo: Tensor<B, 3, Float>,
    pub apose_transfo: Tensor<B, 3, Float>,
    pub apose_rel_transfo: Tensor<B, 3, Float>,
    pub skin_weights: Tensor<B, 2, Float>,
    pub joints_dict: Vec<Box<dyn OsimJoint<B> + Send + Sync>>,
}
impl<B: Backend> SmplPPGPU<B>
where
    B::FloatTensorPrimitive<2>: Sync,
    B::FloatTensorPrimitive<1>: Sync,
    B::QuantizedTensorPrimitive<1>: std::marker::Sync,
    B::QuantizedTensorPrimitive<2>: std::marker::Sync,
    B::QuantizedTensorPrimitive<3>: std::marker::Sync,
{
    /// # Panics
    /// Will panic if the matrices don't match the expected sizes
    /// Amalthea reference: <https://gitlab.com/meshcapade/core/amalthea/-/blob/main/amalthea/core/utils/models/smplpp/smplpp_model.py?ref_type=heads#L47>
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::too_many_lines)]
    #[allow(clippy::similar_names)]
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
        lbs_weights: &nd::Array2<f32>,
        max_num_betas: usize,
        max_num_expression_components: usize,
        parameter_mapping: &nd::Array1<f32>,
        tpose_transfo: &nd::Array3<f32>,
        apose_transfo: &nd::Array3<f32>,
        apose_rel_transfo: &nd::Array3<f32>,
        kintree_table: &nd::Array2<u32>,
        joint_regressor_osim: &nd::Array2<f32>,
        per_joint_rot: &nd::Array3<f32>,
        parents: Vec<u32>,
        children: Vec<u32>,
        joint_idx_fixed_beta: Vec<u32>,
        skin_weights: &nd::Array2<f32>,
    ) -> Self {
        let device = B::Device::default();
        let b_verts_template = verts_template.to_burn(&device);
        let b_faces = faces.to_burn(&device);
        let b_faces_uv_mesh = faces_uv_mesh.to_burn(&device);
        let b_uv = uv.to_burn(&device);
        let shape_dirs = shape_dirs
            .slice_axis(Axis(2), ndarray::Slice::from(0..max_num_betas))
            .to_owned()
            .into_shape_with_order((NUM_VERTS * 3, max_num_betas))
            .unwrap();
        let b_shape_dirs = shape_dirs.to_burn(&device);
        let b_parameter_mapping = parameter_mapping.to_burn(&device);
        let b_tpose_transfo = tpose_transfo.to_burn(&device);
        let b_apose_transfo = apose_transfo.to_burn(&device);
        let b_apose_rel_transfo = apose_rel_transfo.to_burn(&device);
        let b_kintree_table = kintree_table.to_burn(&device);
        let b_joint_regressor_osim = joint_regressor_osim.to_burn(&device);
        let b_per_joint_rot = per_joint_rot.to_burn(&device);
        let b_skin_weights = skin_weights.to_burn(&device);
        let b_expression_dirs = expression_dirs
            .map(|expression_dirs| {
                let expression_dirs = expression_dirs
                    .slice_axis(
                        nd::Axis(2),
                        nd::Slice::from(0..max_num_expression_components),
                    )
                    .into_shape_with_order((
                        NUM_VERTS * 3,
                        max_num_expression_components,
                    ))
                    .unwrap()
                    .to_owned();
                expression_dirs.to_burn(&device)
            });
        let b_pose_dirs = pose_dirs
            .map(|pose_dirs| {
                let pose_dirs = pose_dirs
                    .into_shape_with_order((NUM_VERTS * 3, 207))
                    .unwrap();
                pose_dirs.to_burn(&device)
            });
        let b_joint_regressor = joint_regressor.to_burn(&device);
        let b_parent_idx_per_joint = parent_idx_per_joint
            .to_burn(&device)
            .reshape([NUM_JOINTS]);
        let b_lbs_weights = lbs_weights.to_burn(&device);
        #[allow(clippy::cast_possible_wrap)]
        let faces_uv_mesh_i32: nd::Array2<i32> = faces_uv_mesh.mapv(|x| x as i32);
        let ft: nd::ArcArray2<i32> = faces_uv_mesh_i32.into();
        let max_v_uv_idx = *ft.iter().max_by_key(|&x| x).unwrap();
        let max_v_uv_idx_usize = usize::try_from(max_v_uv_idx)
            .unwrap_or_else(|_| panic!("Cannot cast max_v_uv_idx to usize"));
        let mut idx_vuv_2_vnouv = nd::ArcArray1::<i32>::zeros(max_v_uv_idx_usize + 1);
        for (fuv, fnouv) in ft.axis_iter(nd::Axis(0)).zip(faces.axis_iter(nd::Axis(0))) {
            let uv_0 = fuv[[0]];
            let uv_1 = fuv[[1]];
            let uv_2 = fuv[[2]];
            let nouv_0 = fnouv[[0]];
            let nouv_1 = fnouv[[1]];
            let nouv_2 = fnouv[[2]];
            idx_vuv_2_vnouv[usize::try_from(uv_0)
                .unwrap_or_else(|_| panic!("Cannot cast uv_0 to usize"))] = i32::try_from(
                    nouv_0,
                )
                .unwrap_or_else(|_| panic!("Cannot cast nouv_0 to i32"));
            idx_vuv_2_vnouv[usize::try_from(uv_1)
                .unwrap_or_else(|_| panic!("Cannot cast uv_1 to usize"))] = i32::try_from(
                    nouv_1,
                )
                .unwrap_or_else(|_| panic!("Cannot cast nouv_1 to i32"));
            idx_vuv_2_vnouv[usize::try_from(uv_2)
                .unwrap_or_else(|_| panic!("Cannot cast uv_2 to usize"))] = i32::try_from(
                    nouv_2,
                )
                .unwrap_or_else(|_| panic!("Cannot cast nouv_2 to i32"));
        }
        let idx_vuv_2_vnouv_vec: Vec<i32> = idx_vuv_2_vnouv
            .mapv(|x| x)
            .into_raw_vec_and_offset()
            .0;
        let idx_vuv_2_vnouv_slice: &[i32] = &idx_vuv_2_vnouv_vec;
        let b_idx_vuv_2_vnouv = Tensor::<
            B,
            1,
            Int,
        >::from_ints(idx_vuv_2_vnouv_slice, &device);
        let idx_vuv_2_vnouv_vec: Vec<usize> = idx_vuv_2_vnouv
            .to_vec()
            .iter()
            .map(|&x| {
                usize::try_from(x)
                    .unwrap_or_else(|_| panic!("Cannot cast negative value to usize"))
            })
            .collect();
        let faces_na = faces.view().into_nalgebra().clone_owned().map(|x| x);
        let faces_uv_mesh_na = ft
            .view()
            .into_nalgebra()
            .clone_owned()
            .map(|x| {
                u32::try_from(x).unwrap_or_else(|_| panic!("Cannot cast value to u32"))
            });
        let uv_na = uv.view().into_nalgebra().clone_owned();
        let cols: Vec<usize> = (0..lbs_weights.ncols()).collect();
        let lbs_weights_split: nd::ArcArray2<f32> = lbs_weights
            .to_owned()
            .gather(&idx_vuv_2_vnouv_vec, &cols)
            .into();
        let b_lbs_weights_split = Tensor::<
            B,
            1,
        >::from_floats(lbs_weights_split.as_slice().unwrap(), &device)
            .reshape([idx_vuv_2_vnouv_vec.len(), NUM_JOINTS]);
        let verts_ones = Tensor::<B, 2>::ones([NUM_VERTS, 1], &device);
        #[rustfmt::skip]
        let joints_dict: Vec<Box<dyn OsimJoint<B> + Send + Sync>> = vec![
            Box::new(CustomJoint::new(& [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0], &
            [1.0, 1.0, 1.0], & device)), Box::new(CustomJoint::new(& [0.0, 0.0, 1.0, 1.0,
            0.0, 0.0, 0.0, 1.0, 0.0], & [1.0, 1.0, 1.0], & device)),
            Box::new(WalkerKnee::new()), Box::new(PinJoint::new(& [0.175_895, -
            0.105_208, 0.0186_622], & device)), Box::new(PinJoint::new(& [- 1.76819,
            0.906_223, 1.819_6], & device)), Box::new(PinJoint::new(& [- PI, 0.619_901,
            0.0], & device)), Box::new(CustomJoint::new(& [0.0, 0.0, 1.0, 1.0, 0.0, 0.0,
            0.0, 1.0, 0.0], & [1.0, - 1.0, - 1.0], & device)),
            Box::new(WalkerKnee::new()), Box::new(PinJoint::new(& [0.175_895, -
            0.105_208, 0.018_662_2], & device)), Box::new(PinJoint::new(& [1.76819, -
            0.906_223, 1.8196], & device)), Box::new(PinJoint::new(& [- PI, - 0.619_901,
            0.0], & device)), Box::new(ConstantCurvatureJoint::new(& [1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 1.0, 0.0], & [1.0, 1.0, 1.0], & device)),
            Box::new(ConstantCurvatureJoint::new(& [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
            1.0, 0.0], & [1.0, 1.0, 1.0], & device)),
            Box::new(ConstantCurvatureJoint::new(& [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
            1.0, 0.0], & [1.0, 1.0, 1.0], & device)), Box::new(EllipsoidJoint::new(&
            [0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0], & [1.0, - 1.0, - 1.0], &
            device)), Box::new(CustomJoint::new(& [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
            0.0, 1.0], & [1.0, 1.0, 1.0], & device)), Box::new(CustomJoint::new(&
            [0.0494, 0.0366, 0.998_108_25], & [1.0], & device)),
            Box::new(CustomJoint::new(& [- 0.017_160_99, 0.992_665_64, - 0.119_667_96], &
            [1.0], & device)), Box::new(CustomJoint::new(& [1.0, 0.0, 0.0, 0.0, 0.0, -
            1.0], & [1.0, 1.0], & device)), Box::new(EllipsoidJoint::new(& [0.0, 1.0,
            0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0], & [1.0, 1.0, 1.0], & device)),
            Box::new(CustomJoint::new(& [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], &
            [1.0, 1.0, 1.0], & device)), Box::new(CustomJoint::new(& [- 0.0494, - 0.0366,
            0.998_108_25], & [1.0], & device)), Box::new(CustomJoint::new(&
            [0.017_160_99, - 0.992_665_64, - 0.119_667_96], & [1.0], & device)),
            Box::new(CustomJoint::new(& [- 1.0, 0.0, 0.0, 0.0, 0.0, - 1.0], & [1.0, 1.0],
            & device)),
        ];
        info!("Chosen Dynamic Backend: {}", B::name().to_uppercase());
        info!("On Device: {:?}", & device);
        Self {
            smpl_type: SmplType::SmplPP,
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
            parents,
            children,
            joint_idx_fixed_beta,
            kintree_table: b_kintree_table,
            joint_regressor_osim: b_joint_regressor_osim,
            per_joint_rot: b_per_joint_rot,
            parameter_mapping: b_parameter_mapping,
            tpose_transfo: b_tpose_transfo,
            apose_transfo: b_apose_transfo,
            apose_rel_transfo: b_apose_rel_transfo,
            skin_weights: b_skin_weights,
            joints_dict,
        }
    }
    /// # Panics
    /// Will panic if the path cannot be opened
    /// Amalthea reference: <https://gitlab.com/meshcapade/core/amalthea/-/blob/main/amalthea/core/utils/models/smplpp/smplpp_model.py?ref_type=heads#L47>
    #[allow(clippy::too_many_lines)]
    #[allow(clippy::cast_sign_loss)]
    #[allow(clippy::cast_possible_truncation)]
    fn new_from_npz_reader<R: Read + Seek>(
        npz: &mut NpzReader<R>,
        gender: Gender,
        max_num_betas: usize,
        max_num_expression_components: usize,
    ) -> Self {
        println!("\nNames are {:?}", npz.names());
        let verts_template: nd::Array2<f32> = npz.by_name("v_template").unwrap();
        println!("\nTemplate Mesh Shape  {:?}", verts_template.shape());
        let faces_orig: nd::Array2<i32> = npz.by_name("quad_f").unwrap();
        let faces: nd::Array2<u32> = faces_orig.mapv(|elem| elem as u32);
        let uv_orig: nd::Array2<f64> = npz.by_name("quad_vt").unwrap();
        let uv: nd::Array2<f32> = uv_orig.mapv(|elem| elem as f32);
        let full_shape_dirs_orig: nd::Array3<f64> = npz.by_name("shapedirs").unwrap();
        let full_shape_dirs: nd::Array3<f32> = full_shape_dirs_orig
            .mapv(|elem| elem as f32);
        let (shape_dirs, expression_dirs) = if let Ok(expression_dirs) = npz
            .by_name("expressiondirs")
        {
            (full_shape_dirs, Some(expression_dirs))
        } else {
            let shape_dirs = full_shape_dirs
                .slice_axis(nd::Axis(2), nd::Slice::from(0..max_num_betas.min(300)))
                .to_owned();
            let expression_dirs = if full_shape_dirs.shape()[2] > 300 {
                Some(
                    full_shape_dirs
                        .slice_axis(
                            nd::Axis(2),
                            nd::Slice::from(
                                300..300 + max_num_expression_components.min(100),
                            ),
                        )
                        .to_owned(),
                )
            } else {
                None
            };
            (shape_dirs, expression_dirs)
        };
        let pose_dirs = None;
        let joint_regressor_orig: nd::Array2<f64> = npz.by_name("J_regressor").unwrap();
        let joint_regressor: nd::Array2<f32> = joint_regressor_orig
            .mapv(|elem| elem as f32);
        let parent_idx_per_joint_orig: nd::Array2<i64> = npz
            .by_name("kintree_table")
            .unwrap();
        #[allow(clippy::cast_sign_loss)]
        let parent_idx_per_joint = parent_idx_per_joint_orig.mapv(|x| x as u32);
        let parent_idx_per_joint = parent_idx_per_joint
            .slice_axis(nd::Axis(0), nd::Slice::from(0..1))
            .to_owned()
            .into_shape_with_order(NUM_JOINTS)
            .unwrap();
        let lbs_weights_orig: nd::Array2<f64> = npz.by_name("weights").unwrap();
        let lbs_weights: nd::Array2<f32> = lbs_weights_orig.mapv(|elem| elem as f32);
        let ft_orig: nd::Array2<i64> = npz.by_name("quad_ft").unwrap();
        let ft: nd::Array2<u32> = ft_orig.mapv(|elem| elem as u32);
        let parameter_mapping_orig: nd::Array1<f64> = npz
            .by_name("parameter_mapping")
            .unwrap();
        let parameter_mapping: nd::Array1<f32> = parameter_mapping_orig
            .mapv(|elem| elem as f32);
        let tpose_transfo_orig: nd::Array3<f64> = npz.by_name("tpose_transfo").unwrap();
        let tpose_transfo: nd::Array3<f32> = tpose_transfo_orig.mapv(|elem| elem as f32);
        let apose_transfo_orig: nd::Array3<f64> = npz.by_name("apose_transfo").unwrap();
        let apose_transfo: nd::Array3<f32> = apose_transfo_orig.mapv(|elem| elem as f32);
        let apose_rel_transfo_orig: nd::Array3<f64> = npz
            .by_name("apose_rel_transfo")
            .unwrap();
        let apose_rel_transfo: nd::Array3<f32> = apose_rel_transfo_orig
            .mapv(|elem| elem as f32);
        let kintree_table_orig: nd::Array2<i64> = npz
            .by_name("osim_kintree_table")
            .unwrap();
        let kintree_table: nd::Array2<u32> = kintree_table_orig.mapv(|elem| elem as u32);
        let joint_regressor_osim_orig: nd::Array2<f64> = npz
            .by_name("J_regressor_osim")
            .unwrap();
        let joint_regressor_osim: nd::Array2<f32> = joint_regressor_osim_orig
            .mapv(|elem| elem as f32);
        let per_joint_rot_orig: nd::Array3<f64> = npz.by_name("per_joint_rot").unwrap();
        let per_joint_rot: nd::Array3<f32> = per_joint_rot_orig.mapv(|elem| elem as f32);
        let joint_idx_fixed_beta: Vec<u32> = vec![0, 5, 10, 13, 18, 23];
        let mut id_to_col = std::collections::HashMap::new();
        for i in 0..kintree_table.shape()[1] {
            id_to_col.insert(kintree_table[(1, i)], i as u32);
        }
        let parent: Vec<u32> = (1..kintree_table.shape()[1])
            .map(|it| id_to_col[&kintree_table[(0, it)]])
            .collect();
        let mut child_array: Vec<u32> = vec![];
        for i in 0..NUM_JOINTS {
            let j_array: Vec<usize> = kintree_table
                .index_axis(nd::Axis(0), 0)
                .indexed_iter()
                .filter_map(|(idx, &val)| if val == i as u32 { Some(idx) } else { None })
                .collect();
            let child_index = if j_array.is_empty() {
                0
            } else {
                let j = j_array[0];
                if j >= kintree_table.shape()[1] {
                    0
                } else {
                    kintree_table[(1, j)] as u32
                }
            };
            child_array.push(child_index);
        }
        let skin_weights_orig: nd::Array2<f64> = npz.by_name("skin_weights").unwrap();
        let skin_weights: nd::Array2<f32> = skin_weights_orig.mapv(|elem| elem as f32);
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
            &lbs_weights,
            max_num_betas,
            max_num_expression_components,
            &parameter_mapping,
            &tpose_transfo,
            &apose_transfo,
            &apose_rel_transfo,
            &kintree_table,
            &joint_regressor_osim,
            &per_joint_rot,
            parent,
            child_array,
            joint_idx_fixed_beta,
            &skin_weights,
        )
    }
    /// # Panics
    /// Will panic if the path cannot be opened
    #[cfg(not(target_arch = "wasm32"))]
    pub fn new_from_npz(
        model_path: &str,
        gender: Gender,
        max_num_betas: usize,
        max_num_expression_components: usize,
    ) -> Self {
        let mut npz = NpzReader::new(std::fs::File::open(model_path).unwrap()).unwrap();
        Self::new_from_npz_reader(
            &mut npz,
            gender,
            max_num_betas,
            max_num_expression_components,
        )
    }
    /// # Panics
    /// Will panic if the path cannot be opened
    /// Will panic if the translation and rotation do not cover the same number
    /// of timesteps
    #[allow(clippy::cast_possible_truncation)]
    pub async fn new_from_npz_async(
        model_path: &str,
        gender: Gender,
        max_num_betas: usize,
        max_num_expression_components: usize,
    ) -> Self {
        let reader = FileLoader::open(model_path).await;
        let mut npz = NpzReader::new(reader).unwrap();
        Self::new_from_npz_reader(
            &mut npz,
            gender,
            max_num_betas,
            max_num_expression_components,
        )
    }
    /// # Panics
    /// Will panic if the path cannot be opened
    /// Will panic if the translation and rotation do not cover the same number
    /// of timesteps
    #[allow(clippy::cast_possible_truncation)]
    pub fn new_from_reader<R: Read + Seek>(
        reader: R,
        gender: Gender,
        max_num_betas: usize,
        max_num_expression_components: usize,
    ) -> Self {
        let mut npz = NpzReader::new(reader).unwrap();
        Self::new_from_npz_reader(
            &mut npz,
            gender,
            max_num_betas,
            max_num_expression_components,
        )
    }
    /// # Panics
    /// Will panic if the path cannot be opened
    /// Will panic if the translation and rotation do not cover the same number
    /// of timesteps
    #[allow(clippy::cast_possible_truncation)]
    pub fn read_pose_dirs_from_reader<R: Read + Seek>(
        reader: R,
        device: &B::Device,
    ) -> Tensor<B, 2, Float> {
        let mut npz = NpzReader::new(reader).unwrap();
        let pose_dirs: Option<nd::Array3<f32>> = Some(npz.by_name("pose_dirs").unwrap());
        let b_pose_dirs = pose_dirs
            .map(|pose_dirs| {
                Tensor::<B, 1>::from_floats(pose_dirs.as_slice().unwrap(), device)
                    .reshape([NUM_VERTS * 3, NUM_JOINTS * 9])
            });
        b_pose_dirs.unwrap()
    }
    /// Amalthea Reference: <https://gitlab.com/meshcapade/core/amalthea/-/blob/main/amalthea/core/utils/models/skel/utils.py?ref_type=heads#L90>
    #[allow(clippy::many_single_char_names)]
    pub fn rotation_matrix_from_vectors(
        &self,
        vec1: Tensor<B, 2>,
        vec2: Tensor<B, 2>,
    ) -> Tensor<B, 3> {
        let nj = vec1.dims()[0];
        let a = vec1.clone().div(vec1.powi_scalar(2).sum_dim(1).sqrt());
        let b = vec2.clone().div(vec2.powi_scalar(2).sum_dim(1).sqrt());
        let v = gloss_utils::tensor::cross_product(&a, &b);
        let c = a.mul(b).sum_dim(1).squeeze(1);
        let s = v
            .clone()
            .powi_scalar(2)
            .sum_dim(1)
            .squeeze(1)
            .sqrt()
            .add_scalar(f32::EPSILON);
        let v0 = Tensor::zeros([nj, 1], &v.device());
        let kmat_l1 = Tensor::cat(
            [
                v0.clone(),
                v.clone().slice([0..nj, 2..3]).neg(),
                v.clone().slice([0..nj, 1..2]),
            ]
                .to_vec(),
            1,
        );
        let kmat_l2 = Tensor::cat(
            [
                v.clone().slice([0..nj, 2..3]),
                v0.clone(),
                v.clone().slice([0..nj, 0..1]).neg(),
            ]
                .to_vec(),
            1,
        );
        let kmat_l3 = Tensor::cat(
            [
                v.clone().slice([0..nj, 1..2]).neg(),
                v.clone().slice([0..nj, 0..1]),
                v0.clone(),
            ]
                .to_vec(),
            1,
        );
        let kmat = Tensor::stack([kmat_l1, kmat_l2, kmat_l3].to_vec(), 1);
        let identity = Tensor::eye(3, &v.device());
        let kmat_squared = kmat.clone().matmul(kmat.clone());
        let one_sub_c_div_s2 = Tensor::<B, 1>::from_floats([1.0], &v.device())
            .sub(c)
            .div(s.powi_scalar(2))
            .unsqueeze_dim::<2>(1)
            .unsqueeze_dim::<3>(1);
        let kmat_scaled = kmat_squared.mul(one_sub_c_div_s2);
        let identity_expanded = identity.unsqueeze_dim(0).repeat(&[nj, 1, 1]);
        identity_expanded + kmat + kmat_scaled
    }
    /// Amalthea Reference: <https://gitlab.com/meshcapade/core/amalthea/-/blob/main/amalthea/core/utils/models/skel/skel_model.py?ref_type=heads#L677>
    /// We assume method to be ``learn_adjust``; The other 2 methods will not be
    /// used.
    pub fn compute_bone_orientation(
        &self,
        j: &Tensor<B, 2>,
        j_: Tensor<B, 2>,
    ) -> Tensor<B, 3> {
        let nj = j_.dims()[0];
        let mut bone_vect = Tensor::zeros(j_.dims(), &j.device());
        #[allow(clippy::cast_possible_wrap)]
        let children_indices = Tensor::from_ints(
            self
                .children
                .clone()
                .into_iter()
                .map(|v| v as i32)
                .collect::<Vec<i32>>()
                .as_slice(),
            &j.device(),
        );
        bone_vect = bone_vect
            .slice_assign([0..nj, 0..3], j_.select(0, children_indices.clone()));
        bone_vect = bone_vect
            .clone()
            .slice_assign(
                [16..17, 0..3],
                bone_vect
                    .clone()
                    .slice([16..17, 0..3])
                    .add(bone_vect.clone().slice([17..18, 0..3])),
            );
        bone_vect = bone_vect
            .clone()
            .slice_assign(
                [21..22, 0..3],
                bone_vect
                    .clone()
                    .slice([21..22, 0..3])
                    .add(bone_vect.slice([22..23, 0..3])),
            );
        bone_vect = bone_vect
            .clone()
            .slice_assign([12..13, 0..3], bone_vect.clone().slice([11..12, 0..3]));
        let mut osim_vect = self
            .apose_rel_transfo
            .clone()
            .slice([0..self.apose_rel_transfo.dims()[0], 0..3, 3..4])
            .reshape([nj, 3])
            .clone();
        osim_vect = osim_vect
            .clone()
            .slice_assign([0..nj, 0..3], osim_vect.clone().select(0, children_indices));
        osim_vect = osim_vect
            .clone()
            .slice_assign(
                [16..17, 0..3],
                osim_vect
                    .clone()
                    .slice([16..17, 0..3])
                    .add(osim_vect.clone().slice([17..18, 0..3])),
            );
        osim_vect = osim_vect
            .clone()
            .slice_assign(
                [21..22, 0..3],
                osim_vect
                    .clone()
                    .slice([21..22, 0..3])
                    .add(osim_vect.clone().slice([22..23, 0..3])),
            );
        let gk_learned = self.per_joint_rot.clone().reshape([nj, 3, 3]).clone();
        let osim_vect_corr = gk_learned
            .clone()
            .matmul(osim_vect.unsqueeze_dim(2))
            .squeeze(2);
        let mut gk = self.rotation_matrix_from_vectors(osim_vect_corr, bone_vect);
        let nan_mask = gk.clone().not_equal(gk.clone());
        gk = gk.mask_fill(nan_mask, 0.0);
        let identity_matrix = Tensor::eye(3, &j.device());
        #[allow(clippy::range_plus_one)]
        for idx in &self.joint_idx_fixed_beta {
            gk = gk
                .slice_assign(
                    [*idx as usize..*idx as usize + 1, 0..3, 0..3],
                    identity_matrix.clone().unsqueeze_dim(0),
                );
        }
        gk.matmul(gk_learned)
    }
    /// Amalthea Reference: <https://gitlab.com/meshcapade/core/amalthea/-/blob/main/amalthea/core/utils/models/skel/skel_model.py?ref_type=heads#L249>
    #[allow(clippy::single_range_in_vec_init)]
    #[allow(clippy::range_plus_one)]
    pub fn pose_params_to_rot(
        &self,
        osim_poses: &Tensor<B, 1>,
    ) -> (Tensor<B, 3>, Tensor<B, 2>) {
        let nj = self.joints_dict.len();
        let device = osim_poses.device();
        let ident = Tensor::eye(3, &device);
        let mut rp: Tensor<B, 3> = ident.unsqueeze_dim(0).repeat(&[nj, 1, 1]);
        let tp = Tensor::zeros([nj, 3], &device);
        let mut start_index = 0;
        for (i, joint_object) in self.joints_dict.iter().enumerate() {
            let nb_dof = joint_object.nb_dof();
            let end_index = start_index + nb_dof;
            let joint_pose = osim_poses.clone().slice([start_index..end_index]);
            let joint_rot = joint_object.q_to_rot(joint_pose);
            rp = rp.slice_assign([i..i + 1, 0..3, 0..3], joint_rot.unsqueeze_dim(0));
            start_index = end_index;
        }
        (rp, tp)
    }
    /// Amalthea Reference: <https://gitlab.com/meshcapade/core/amalthea/-/blob/main/amalthea/core/utils/models/skel/joints_def.py?ref_type=heads>
    fn right_scapula(
        &self,
        angle_abduction: f32,
        angle_elevation: f32,
        _angle_rot: f32,
        thorax_width: f32,
        thorax_height: f32,
    ) -> [f32; 3] {
        let pi_over_4 = PI / 4.0;
        let radius_x = (thorax_width / 4.0) * (angle_elevation - pi_over_4).cos();
        let radius_y = thorax_width / 4.0;
        let radius_z = thorax_height / 2.0;
        [
            -radius_x * angle_abduction.cos(),
            -radius_z * (angle_elevation - pi_over_4).sin(),
            radius_y * angle_abduction.sin(),
        ]
    }
    /// Amalthea Reference: <https://gitlab.com/meshcapade/core/amalthea/-/blob/main/amalthea/core/utils/models/skel/joints_def.py?ref_type=heads>
    fn left_scapula(
        &self,
        mut angle_abduction: f32,
        mut angle_elevation: f32,
        _angle_rot: f32,
        thorax_width: f32,
        thorax_height: f32,
    ) -> [f32; 3] {
        let pi_over_4 = PI / 4.0;
        angle_abduction = -angle_abduction;
        angle_elevation = -angle_elevation;
        let radius_x = (thorax_width / 4.0) * (angle_elevation - pi_over_4).cos();
        let radius_y = thorax_width / 4.0;
        let radius_z = thorax_height / 2.0;
        [
            radius_x * angle_abduction.cos(),
            -radius_z * (angle_elevation - pi_over_4).sin(),
            radius_y * angle_abduction.sin(),
        ]
    }
    /// Amalthea Reference: <https://gitlab.com/meshcapade/core/amalthea/-/blob/main/amalthea/core/utils/models/skel/joints_def.py?ref_type=heads>
    #[allow(clippy::many_single_char_names)]
    fn curve_rust_1d(&self, angle: f32, t: f32, l: f32) -> (f32, f32) {
        let x;
        let y;
        if angle.abs() < 1e-5 {
            x = l * t * t * angle / 2.0;
            y = l * t * (1.0 - (t * t * t / 6.0) * angle * angle);
        } else {
            let r = l / angle;
            x = r * (1.0 - (t * angle).cos());
            y = r * (t * angle).sin();
        }
        (x, y)
    }
    /// Amalthea Reference: <https://gitlab.com/meshcapade/core/amalthea/-/blob/main/amalthea/core/utils/models/skel/joints_def.py?ref_type=heads>
    fn curve_rust_3d(
        &self,
        angle_x: f32,
        angle_y: f32,
        t: f32,
        l: f32,
    ) -> (f32, f32, f32) {
        let (x1, y1) = self.curve_rust_1d(angle_x, t, l);
        let tx = (-x1, y1, 0.0);
        let (x2, y2) = self.curve_rust_1d(angle_y, t, l);
        let ty = (0.0, y2, -x2);
        (tx.0 + ty.0, tx.1 + ty.1, tx.2 + ty.2)
    }
    fn matmul_chain(&self, rot_list: &[Tensor<B, 3>]) -> Tensor<B, 3> {
        let mut r_tot = rot_list[rot_list.len() - 1].clone();
        for i in (0..rot_list.len() - 1).rev() {
            r_tot = rot_list[i].clone().matmul(r_tot);
        }
        r_tot
    }
}
impl<B: Backend> SmplModel<B> for SmplPPGPU<B>
where
    B::FloatTensorPrimitive<3>: Sync,
    B::FloatTensorPrimitive<2>: Sync,
    B::FloatTensorPrimitive<1>: Sync,
    B::IntTensorPrimitive<2>: Sync,
    B::IntTensorPrimitive<1>: Sync,
    B::QuantizedTensorPrimitive<1>: std::marker::Sync,
    B::QuantizedTensorPrimitive<2>: std::marker::Sync,
    B::QuantizedTensorPrimitive<3>: std::marker::Sync,
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
    #[allow(clippy::missing_panics_doc)]
    #[allow(non_snake_case)]
    fn forward(
        &self,
        options: &SmplOptions,
        betas: &Betas,
        pose_raw: &Pose,
        expression: Option<&Expression>,
    ) -> SmplOutputDynamic<B> {
        let mut verts_t_pose = self.betas2verts(betas);
        if let Some(expression) = expression {
            verts_t_pose = verts_t_pose + self.expression2offsets(expression);
        }
        let pose_remap = PoseRemap::new(pose_raw.smpl_type, SmplType::SmplPP);
        let pose = pose_remap.remap(pose_raw);
        let joints_t_pose = self.verts2joints(verts_t_pose.clone());
        if options.enable_pose_corrective {
            let verts_offset = self.compute_pose_correctives(&pose);
            verts_t_pose = verts_t_pose + verts_offset;
        }
        let (verts_posed_nd, _, _, joints_posed) = self
            .apply_pose(
                &verts_t_pose,
                None,
                None,
                &joints_t_pose,
                &self.lbs_weights,
                &pose,
            );
        SmplOutputDynamic {
            verts: verts_posed_nd,
            faces: self.faces.clone(),
            normals: None,
            uvs: None,
            joints: joints_posed,
        }
    }
    fn create_body_with_uv(
        &self,
        smpl_merged: &SmplOutputDynamic<B>,
    ) -> SmplOutputDynamic<B> {
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
        let betas_slice = betas.betas.as_slice().unwrap();
        let betas_tensor = Tensor::<B, 1, Float>::from_floats(betas_slice, &device);
        let input_nr_betas = betas_tensor.shape().dims[0];
        let shape_dirs_sliced = self
            .shape_dirs
            .clone()
            .slice([0..self.shape_dirs.dims()[0], 0..input_nr_betas]);
        let v_beta_offsets = shape_dirs_sliced
            .matmul(betas_tensor.reshape([input_nr_betas, 1]));
        let v_beta_offsets_reshaped = v_beta_offsets.reshape([NUM_VERTS, 3]);
        let verts_t_pose = v_beta_offsets_reshaped.add(self.verts_template.clone());
        verts_t_pose
    }
    #[allow(clippy::missing_panics_doc)]
    #[allow(non_snake_case)]
    #[allow(clippy::let_and_return)]
    fn expression2offsets(&self, expression: &Expression) -> Tensor<B, 2, Float> {
        let device = self.verts_template.device();
        let offsets = if let Some(ref expression_dirs) = self.expression_dirs {
            let input_nr_expression_coeffs = expression.expr_coeffs.len();
            let expression_dirs_sliced = expression_dirs
                .clone()
                .slice([0..expression_dirs.dims()[0], 0..input_nr_expression_coeffs]);
            let expr_coeffs_tensor = Tensor::<
                B,
                1,
                Float,
            >::from_floats(expression.expr_coeffs.as_slice().unwrap(), &device);
            let v_expr_offsets = expression_dirs_sliced
                .matmul(expr_coeffs_tensor.reshape([input_nr_expression_coeffs, 1]));
            v_expr_offsets.reshape([NUM_VERTS, 3])
        } else {
            Tensor::<B, 2, Float>::zeros([NUM_VERTS, 3], &device)
        };
        offsets
    }
    fn verts2joints(&self, verts_t_pose: Tensor<B, 2, Float>) -> Tensor<B, 2, Float> {
        self.joint_regressor_osim.clone().matmul(verts_t_pose)
    }
    #[allow(clippy::missing_panics_doc)]
    fn compute_pose_correctives(&self, pose: &Pose) -> Tensor<B, 2, Float> {
        let offsets = if let Some(pose_dirs) = &self.pose_dirs {
            let full_pose = &pose.joint_poses;
            assert!(
                full_pose.dim().0 == NUM_JOINTS + 1,
                "The pose does not have the correct number of joints for this model. Maybe you need to add a PoseRemapper component?\n {:?} != {:?}",
                full_pose.dim().0, NUM_JOINTS + 1
            );
            let mut rot_mats = batch_rodrigues(full_pose);
            let identity = ndarray::Array2::<f32>::eye(3);
            let pose_feature = (rot_mats.slice_mut(s![1.., .., ..]).sub(&identity))
                .into_shape_with_order(NUM_JOINTS * 9)
                .unwrap();
            let b_pose_feature = Tensor::<
                B,
                1,
                Float,
            >::from_floats(pose_feature.as_slice().unwrap(), &self.device)
                .reshape([NUM_JOINTS * 9, 1]);
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
            full_pose.dim().0, NUM_JOINTS + 1
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
    #[allow(clippy::too_many_lines)]
    #[allow(clippy::range_plus_one)]
    fn apply_pose(
        &self,
        verts_t_pose: &Tensor<B, 2, Float>,
        _normals: Option<&Tensor<B, 2, Float>>,
        _tangents: Option<&Tensor<B, 2, Float>>,
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
        let full_pose = pose.joint_poses.clone();
        let skel_pose: nd::Array1<f32> = full_pose.to_shape(46).unwrap().to_owned();
        let nj = joints.shape().dims[0];
        let ns = self.verts_template.dims()[0];
        let J = self.joint_regressor_osim.clone().matmul(verts_t_pose.clone());
        let mut J_ = J.clone();
        assert!(full_pose.shape() [0] == 46, "The pose is not of the expected shape.");
        for i in 1..nj {
            let parent_idx = self.parents[i - 1] as usize;
            let joint_slice = J.clone().slice([i..i + 1, 0..3]);
            let parent_slice = J.clone().slice([parent_idx..parent_idx + 1, 0..3]);
            J_ = J_.slice_assign([i..i + 1, 0..3], joint_slice - parent_slice);
        }
        let t: Tensor<B, 2> = J_.clone();
        let Rk01: Tensor<B, 3> = self.compute_bone_orientation(&J.clone(), J_.clone());
        let Ra = self
            .apose_rel_transfo
            .clone()
            .slice([0..self.apose_rel_transfo.dims()[0], 0..3, 0..3])
            .reshape([nj, 3, 3]);
        let (Rp, _) = self.pose_params_to_rot(&skel_pose.to_burn(&self.device));
        let R = self
            .matmul_chain(
                &[
                    Rk01.clone(),
                    Ra.clone().permute([0, 2, 1]),
                    Rp.clone(),
                    Ra.clone(),
                    Rk01.permute([0, 2, 1]),
                ],
            );
        let mut t_posed = t.clone();
        let thorax_width = tensor_to_data_float(
            &(J.clone().slice([19..20, 0..3]) - J.clone().slice([14..15, 0..3]))
                .powf_scalar(2.0)
                .sum()
                .sqrt(),
        )[0];
        let thorax_height = tensor_to_data_float(
            &(J.clone().slice([12..13, 0..3]) - J.clone().slice([11..12, 0..3]))
                .powf_scalar(2.0)
                .sum()
                .sqrt(),
        )[0];
        let angle_abduction = skel_pose[26];
        let angle_elevation = skel_pose[27];
        let angle_rot = skel_pose[28];
        let angle_zero = 0.0;
        let right_scap_array = self
            .right_scapula(
                angle_abduction,
                angle_elevation,
                angle_rot,
                thorax_width,
                thorax_height,
            );
        let right_scap_zero_array = self
            .right_scapula(
                angle_zero,
                angle_zero,
                angle_zero,
                thorax_width,
                thorax_height,
            );
        let right_scap: Tensor<B, 1> = Tensor::from_floats(
            right_scap_array,
            &self.device,
        );
        let right_scap_zero: Tensor<B, 1> = Tensor::from_floats(
            right_scap_zero_array,
            &self.device,
        );
        let right_scap_diff = right_scap - right_scap_zero;
        let t_posed_14: Tensor<B, 1> = t_posed.clone().slice([14..15, 0..3]).squeeze(0);
        t_posed = t_posed
            .clone()
            .slice_assign(
                [14..15, 0..3],
                t_posed_14.add(right_scap_diff).unsqueeze_dim(0),
            );
        let angle_abduction = skel_pose[36];
        let angle_elevation = skel_pose[37];
        let angle_rot = skel_pose[38];
        let angle_zero = 0.0;
        let left_scap_array = self
            .left_scapula(
                angle_abduction,
                angle_elevation,
                angle_rot,
                thorax_width,
                thorax_height,
            );
        let left_scap_zero_array = self
            .left_scapula(
                angle_zero,
                angle_zero,
                angle_zero,
                thorax_width,
                thorax_height,
            );
        let left_scap: Tensor<B, 1> = Tensor::from_floats(left_scap_array, &self.device);
        let left_scap_zero: Tensor<B, 1> = Tensor::from_floats(
            left_scap_zero_array,
            &self.device,
        );
        let left_scap_diff = left_scap - left_scap_zero;
        let t_posed_19: Tensor<B, 1> = t_posed.clone().slice([19..20, 0..3]).squeeze(0);
        t_posed = t_posed
            .clone()
            .slice_assign(
                [19..20, 0..3],
                t_posed_19.add(left_scap_diff).unsqueeze_dim(0),
            );
        let lumbar_bending = skel_pose[17];
        let lumbar_extension = skel_pose[18];
        let angle_zero = 0.0;
        let interp_t = 1.0;
        let l = tensor_to_data_float(&(J.clone().slice([11..12, 1..2])))[0]
            - tensor_to_data_float(&J.clone().slice([0..1, 1..2]))[0].abs();
        let lumbar_curve_array = self
            .curve_rust_3d(lumbar_bending, lumbar_extension, interp_t, l);
        let lumbar_curve_zero_array = self
            .curve_rust_3d(angle_zero, angle_zero, interp_t, l);
        let lumbar_curve_array: [f32; 3] = lumbar_curve_array.into();
        let lumbar_curve_zero_array: [f32; 3] = lumbar_curve_zero_array.into();
        let lumbar_curve: Tensor<B, 1> = Tensor::from_floats(
            lumbar_curve_array,
            &self.device,
        );
        let lumbar_curve_zero: Tensor<B, 1> = Tensor::from_floats(
            lumbar_curve_zero_array,
            &self.device,
        );
        let lumbar_curve_diff = lumbar_curve - lumbar_curve_zero;
        let t_posed_11: Tensor<B, 1> = t_posed.clone().slice([11..12, 0..3]).squeeze(0);
        t_posed = t_posed
            .clone()
            .slice_assign(
                [11..12, 0..3],
                t_posed_11.add(lumbar_curve_diff).unsqueeze_dim(0),
            );
        let thorax_bending = skel_pose[20];
        let thorax_extension = skel_pose[21];
        let l = tensor_to_data_float(&(J.clone().slice([12..13, 1..2])))[0]
            - tensor_to_data_float(&J.clone().slice([11..12, 1..2]))[0].abs();
        let thorax_curve_array = self
            .curve_rust_3d(thorax_bending, thorax_extension, interp_t, l);
        let thorax_curve_zero_array = self
            .curve_rust_3d(angle_zero, angle_zero, interp_t, l);
        let thorax_curve_array: [f32; 3] = thorax_curve_array.into();
        let thorax_curve_zero_array: [f32; 3] = thorax_curve_zero_array.into();
        let thorax_curve: Tensor<B, 1> = Tensor::from_floats(
            thorax_curve_array,
            &self.device,
        );
        let thorax_curve_zero: Tensor<B, 1> = Tensor::from_floats(
            thorax_curve_zero_array,
            &self.device,
        );
        let thorax_curve_diff = thorax_curve - thorax_curve_zero;
        let t_posed_12: Tensor<B, 1> = t_posed.clone().slice([12..13, 0..3]).squeeze(0);
        t_posed = t_posed
            .clone()
            .slice_assign(
                [12..13, 0..3],
                t_posed_12.add(thorax_curve_diff).unsqueeze_dim(0),
            );
        let head_bending = skel_pose[23];
        let head_extension = skel_pose[24];
        let l = tensor_to_data_float(&(J.clone().slice([13..14, 1..2])))[0]
            - tensor_to_data_float(&J.clone().slice([12..13, 1..2]))[0].abs();
        let head_curve_array = self
            .curve_rust_3d(head_bending, head_extension, interp_t, l);
        let head_curve_zero_array = self
            .curve_rust_3d(angle_zero, angle_zero, interp_t, l);
        let head_curve_array: [f32; 3] = head_curve_array.into();
        let head_curve_zero_array: [f32; 3] = head_curve_zero_array.into();
        let head_curve: Tensor<B, 1> = Tensor::from_floats(
            head_curve_array,
            &self.device,
        );
        let head_curve_zero: Tensor<B, 1> = Tensor::from_floats(
            head_curve_zero_array,
            &self.device,
        );
        let head_curve_diff = head_curve - head_curve_zero;
        let t_posed_13: Tensor<B, 1> = t_posed.clone().slice([13..14, 0..3]).squeeze(0);
        t_posed = t_posed
            .clone()
            .slice_assign(
                [13..14, 0..3],
                t_posed_13.add(head_curve_diff).unsqueeze_dim(0),
            );
        let G_: Tensor<B, 3> = Tensor::cat(
            [R.clone(), t_posed.clone().unsqueeze_dim(2)].to_vec(),
            2,
        );
        let pad_row: Tensor<B, 3> = Tensor::<
            B,
            1,
        >::from_floats([0.0, 0.0, 0.0, 1.0], &self.device)
            .unsqueeze_dim::<2>(0)
            .expand([nj, 1, 4]);
        let G_: Tensor<B, 3> = Tensor::cat([G_.clone(), pad_row].to_vec(), 1);
        let mut G: Vec<Tensor<B, 2>> = vec![
            G_.clone().slice([0..1, 0..4, 0..4]).squeeze(0).clone()
        ];
        for i in 1..nj {
            let parent_index = self.parents[i - 1] as usize;
            let G_parent = G[parent_index].clone();
            let G_local = G_.clone().slice([i..i + 1, 0..4, 0..4]).squeeze(0).clone();
            let G_new = G_parent.matmul(G_local);
            G.push(G_new);
        }
        let G: Tensor<B, 3> = Tensor::stack(G, 0);
        let rest: Tensor<B, 3> = Tensor::cat(
                [J.clone(), Tensor::zeros([nj, 1], &self.device)].to_vec(),
                1,
            )
            .reshape([nj, 4, 1]);
        let zeros: Tensor<B, 3> = Tensor::zeros([nj, 4, 3], &self.device);
        let rest: Tensor<B, 3> = Tensor::cat([zeros, rest].to_vec(), 2);
        let rest_transformed: Tensor<B, 3> = G.clone().matmul(rest);
        let gskin: Tensor<B, 3> = G - rest_transformed;
        let T: Tensor<B, 3> = self
            .skin_weights
            .clone()
            .matmul(gskin.clone().reshape([nj, 16]))
            .reshape([ns, 4, 4]);
        let rest_shape_h: Tensor<B, 2> = Tensor::cat(
            [
                verts_t_pose.clone(),
                Tensor::ones([verts_t_pose.dims()[0], 1], &self.device),
            ]
                .to_vec(),
            1,
        );
        let rest_shape_h_expanded: Tensor<B, 3> = rest_shape_h.unsqueeze_dim(2);
        let v_posed: Tensor<B, 2> = T
            .matmul(rest_shape_h_expanded.clone())
            .squeeze(2)
            .slice([0..ns, 0..3]);
        let v_trans = v_posed + pose.global_trans.to_burn(&self.device).unsqueeze_dim(0);
        (v_trans.clone().reshape(v_trans.shape()), None, None, joints.clone())
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
