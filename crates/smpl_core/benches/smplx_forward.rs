use burn::prelude::Tensor;
use burn::{
    backend::{Candle, NdArray, Wgpu},
    prelude::Backend,
};
use divan::black_box;
use divan::Bencher;
use smpl_core::common::betas::BetasG;
use smpl_core::common::pose::PoseG;
use smpl_core::common::{
    smpl_model::SmplModel,
    smpl_options::SmplOptions,
    types::{Gender, SmplType, UpAxis},
};
use smpl_core::smpl_x::smpl_x_gpu::SmplXGPUG;
use smpl_utils::numerical::batch_rigid_transform_burn;
use smpl_utils::numerical::batch_rodrigues_burn_2;
use smpl_utils::numerical::{batch_rigid_transform_burn_fast, batch_rodrigues_burn, batch_rodrigues_burn_3};
const SMPLX_NEUTRAL_PATH: &str = "../../data/smplx/SMPLX_neutral_array_f32_slim.npz";
fn main() {
    divan::main();
}
#[divan::bench(types = [Candle, NdArray, Wgpu])]
fn smplx_forward<B>(bencher: Bencher)
where
    B: Backend,
{
    let smpl_model = SmplXGPUG::<B>::new_from_npz(SMPLX_NEUTRAL_PATH, Gender::Neutral, 300, 100);
    let smpl_options = SmplOptions::default();
    let betas = BetasG::new_empty(10);
    let pose = PoseG::new_empty(UpAxis::Y, SmplType::SmplX);
    bencher.bench_local(move || {
        black_box(smpl_model.forward(black_box(&smpl_options), black_box(&betas), black_box(&pose), black_box(None)));
        B::sync(&smpl_model.device);
    });
}
#[divan::bench(types = [Candle, NdArray, Wgpu])]
fn smplx_betas2verts<B>(bencher: Bencher)
where
    B: Backend,
{
    let smpl_model = SmplXGPUG::<B>::new_from_npz(SMPLX_NEUTRAL_PATH, Gender::Neutral, 300, 100);
    let betas = BetasG::new_empty(10);
    bencher.bench_local(move || {
        black_box(smpl_model.betas2verts(black_box(&betas)));
        B::sync(&smpl_model.device);
    });
}
#[divan::bench(types = [Candle, NdArray, Wgpu])]
fn smplx_verts2joints<B>(bencher: Bencher)
where
    B: Backend,
{
    let smpl_model = SmplXGPUG::<B>::new_from_npz(SMPLX_NEUTRAL_PATH, Gender::Neutral, 300, 100);
    let betas = BetasG::new_empty(10);
    let verts_t_pose = smpl_model.betas2verts(&betas);
    bencher.bench_local(move || {
        black_box(smpl_model.verts2joints(black_box(verts_t_pose.clone())));
        B::sync(&smpl_model.device);
    });
}
#[divan::bench(types = [Candle, NdArray, Wgpu])]
fn smplx_pose_correctives<B>(bencher: Bencher)
where
    B: Backend,
{
    let smpl_model = SmplXGPUG::<B>::new_from_npz(SMPLX_NEUTRAL_PATH, Gender::Neutral, 300, 100);
    let pose = PoseG::new_empty(UpAxis::Y, SmplType::SmplX);
    bencher.bench_local(move || {
        black_box(smpl_model.compute_pose_correctives(black_box(&pose)));
        B::sync(&smpl_model.device);
    });
}
#[divan::bench(types = [Candle, NdArray, Wgpu])]
fn smplx_apply_pose<B>(bencher: Bencher)
where
    B: Backend,
{
    let smpl_model = SmplXGPUG::<B>::new_from_npz(SMPLX_NEUTRAL_PATH, Gender::Neutral, 300, 100);
    let betas = BetasG::new_empty(10);
    let pose = PoseG::new_empty(UpAxis::Y, SmplType::SmplX);
    let verts_t_pose = smpl_model.betas2verts(&betas);
    let joints_t_pose = smpl_model.verts2joints(verts_t_pose.clone());
    bencher.bench_local(move || {
        black_box(smpl_model.apply_pose(
            black_box(&verts_t_pose),
            black_box(&joints_t_pose),
            black_box(&smpl_model.lbs_weights),
            black_box(&pose),
        ));
        B::sync(&smpl_model.device);
    });
}
#[divan::bench(types = [Candle, NdArray, Wgpu])]
fn smplx_batch_rodrigues<B>(bencher: Bencher)
where
    B: Backend,
{
    let smpl_model = SmplXGPUG::<B>::new_from_npz(SMPLX_NEUTRAL_PATH, Gender::Neutral, 300, 100);
    let pose = PoseG::new_empty(UpAxis::Y, SmplType::SmplX);
    bencher.bench_local(move || {
        let full_pose: Tensor<B, 2> = black_box(pose.joint_poses.clone());
        black_box(batch_rodrigues_burn(black_box(&full_pose)));
        B::sync(&smpl_model.device);
    });
}
#[divan::bench(types = [Candle, NdArray, Wgpu])]
fn smplx_batch_rodrigues_2<B>(bencher: Bencher)
where
    B: Backend,
{
    let smpl_model = SmplXGPUG::<B>::new_from_npz(SMPLX_NEUTRAL_PATH, Gender::Neutral, 300, 100);
    let pose = PoseG::new_empty(UpAxis::Y, SmplType::SmplX);
    let full_pose: Tensor<B, 2> = pose.joint_poses.clone();
    bencher.bench_local(move || {
        black_box(batch_rodrigues_burn_2(black_box(&full_pose)));
        B::sync(&smpl_model.device);
    });
}
#[divan::bench(types = [Candle, NdArray, Wgpu])]
fn smplx_batch_rodrigues_3<B>(bencher: Bencher)
where
    B: Backend,
{
    let smpl_model = SmplXGPUG::<B>::new_from_npz(SMPLX_NEUTRAL_PATH, Gender::Neutral, 300, 100);
    let pose = PoseG::new_empty(UpAxis::Y, SmplType::SmplX);
    let full_pose: Tensor<B, 2> = pose.joint_poses.clone();
    bencher.bench_local(move || {
        black_box(batch_rodrigues_burn_3(black_box(&full_pose)));
        B::sync(&smpl_model.device);
    });
}
#[divan::bench(types = [Candle, NdArray, Wgpu])]
fn smplx_batch_rigid_transform<B>(bencher: Bencher)
where
    B: Backend,
{
    let smpl_model = SmplXGPUG::<B>::new_from_npz(SMPLX_NEUTRAL_PATH, Gender::Neutral, 300, 100);
    let betas = BetasG::new_empty(10);
    let pose = PoseG::new_empty(UpAxis::Y, SmplType::SmplX);
    let full_pose: Tensor<B, 2> = pose.joint_poses.clone();
    let rot_mats_t = batch_rodrigues_burn_2(&full_pose);
    let verts_t_pose = smpl_model.betas2verts(&betas);
    let joints = smpl_model.verts2joints(verts_t_pose.clone());
    bencher.bench_local(move || {
        black_box(batch_rigid_transform_burn(
            black_box(smpl_model.parent_idx_per_joint.clone()),
            black_box(&smpl_model.parent_idx_per_joint_nd),
            black_box(rot_mats_t.clone()),
            black_box(joints.clone()),
        ));
        B::sync(&smpl_model.device);
    });
}
#[divan::bench(types = [Candle, NdArray, Wgpu])]
fn smplx_batch_rigid_transform_fast<B>(bencher: Bencher)
where
    B: Backend,
{
    let smpl_model = SmplXGPUG::<B>::new_from_npz(SMPLX_NEUTRAL_PATH, Gender::Neutral, 300, 100);
    let betas = BetasG::new_empty(10);
    let pose = PoseG::new_empty(UpAxis::Y, SmplType::SmplX);
    let full_pose: Tensor<B, 2> = pose.joint_poses.clone();
    let rot_mats_t = batch_rodrigues_burn_2(&full_pose);
    let verts_t_pose = smpl_model.betas2verts(&betas);
    let joints = smpl_model.verts2joints(verts_t_pose.clone());
    bencher.bench_local(move || {
        black_box(batch_rigid_transform_burn_fast(
            black_box(smpl_model.parent_idx_per_joint.clone()),
            black_box(&smpl_model.parent_idx_per_joint_nd),
            black_box(rot_mats_t.clone()),
            black_box(joints.clone()),
        ));
        B::sync(&smpl_model.device);
    });
}
