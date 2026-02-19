use easy_wgpu::gpu::Gpu;
use gloss_burn_multibackend::global_backend::GlobalBackend;
use log::info;
use pyo3::prelude::*;
use wgpu_burn_global_device::global_device::init_global_device;
#[pyfunction]
#[pyo3(name = "smplrs_init_burn_backend")]
#[pyo3(text_signature = "(backend: string, idx_gpu: Optional[usize] = None) -> None")]
pub fn init_global_burn_backend(backend_name: &str, idx_gpu: Option<usize>) {
    let backend = match backend_name {
        "candle" => GlobalBackend::Candle,
        "ndarray" => GlobalBackend::NdArray,
        "wgpu" => GlobalBackend::Wgpu,
        "torch_cpu" => GlobalBackend::TorchCpu,
        "torch_cuda" => GlobalBackend::TorchCuda(idx_gpu.expect("idx_gpu must be provided when using torch_cuda backend")),
        _ => {
            panic!("Unknown backend: {backend_name}");
        }
    };
    gloss_burn_multibackend::global_backend::init_global_burn_backend(backend);
}
#[pyfunction]
pub fn smplrs_sync_burn_gpu(gpu_ptr_idx: u64) {
    let gpu_ptr = gpu_ptr_idx as *mut Gpu;
    let gpu: &mut Gpu = unsafe { &mut *gpu_ptr };
    info!("smplrs syncing gpu: {:?}", gpu.adapter());
    init_global_device(gpu.instance(), gpu.adapter(), gpu.device(), gpu.queue());
}
