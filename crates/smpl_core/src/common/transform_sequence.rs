use gloss_renderer::components::ModelMatrix;
use nalgebra as na;
use ndarray as nd;
use ndarray::s;
use ndarray_npy::NpzReader;
use std::time::Duration;
fn map(value: f32, from_min: f32, from_max: f32, to_min: f32, to_max: f32) -> f32 {
    to_min + (value - from_min) * (to_max - to_min) / (from_max - from_min)
}
/// The ``TransformSequence`` contains the rigid pose sequence of a model over time.
/// It can contain per-frame translations, rotations (in angle-axis format) and scales.
#[derive(Debug, Clone)]
pub struct TransformSequence {
    pub translations: nd::Array2<f32>,
    pub rotations: nd::Array2<f32>,
    pub scales: nd::Array1<f32>,
}
impl TransformSequence {
    /// Create a new `TransformSequence` from given npz file path.
    /// # Panics
    /// Will panic if the number of frames in translations and rotations do not match.
    pub fn new_from_npz(path: &str) -> Self {
        let mut npz = NpzReader::new(std::fs::File::open(path).unwrap()).unwrap();
        let translations: nd::Array2<f32> = npz.by_name("translation").unwrap();
        let rotations: nd::Array2<f32> = npz.by_name("rotation").unwrap();
        let scales: nd::Array1<f32> = npz.by_name("objectSize").unwrap();
        assert_eq!(
            translations.shape()[0],
            rotations.shape()[0],
            "Number of frames in translations and rotations must match"
        );
        Self {
            translations,
            rotations,
            scales,
        }
    }
    /// Create an empty `TransformSequence` with given number of frames.
    pub fn new_empty(num_frames: usize) -> Self {
        let translations = nd::Array2::<f32>::zeros((num_frames, 3));
        let rotations = nd::Array2::<f32>::zeros((num_frames, 3));
        let scales = nd::Array1::<f32>::zeros(num_frames);
        Self {
            translations,
            rotations,
            scales,
        }
    }
    /// Duration of the animation
    #[allow(clippy::cast_precision_loss)]
    pub fn duration(&self, fps: f32) -> Duration {
        Duration::from_secs_f32(self.num_frames() as f32 / fps)
    }
    #[allow(clippy::cast_precision_loss)]
    #[allow(clippy::cast_possible_truncation)]
    #[allow(clippy::cast_sign_loss)]
    pub fn get_smooth_time_indices(&self, time_sec: f32, fps: f32) -> (usize, usize, f32) {
        let frame_time = map(time_sec, 0.0, self.duration(fps).as_secs_f32(), 0.0, (self.num_frames() - 1) as f32);
        let frame_ceil = frame_time.ceil();
        let frame_ceil = frame_ceil.clamp(0.0, (self.num_frames() - 1) as f32);
        let frame_floor = frame_time.floor();
        let frame_floor = frame_floor.clamp(0.0, (self.num_frames() - 1) as f32);
        let w_ceil = frame_ceil - frame_time;
        let w_ceil = 1.0 - w_ceil;
        (frame_floor as usize, frame_ceil as usize, w_ceil)
    }
    pub fn get_transform_at_idx(&self, idx: u32) -> ModelMatrix {
        let translation = self.translations.slice(s![idx as usize, ..]);
        let rotation = self.rotations.slice(s![idx as usize, ..]);
        let scale = self.scales[idx as usize];
        let translation_vec = na::Vector3::from_row_slice(translation.as_slice().unwrap());
        let rotation_vec = na::Vector3::from_row_slice(rotation.as_slice().unwrap());
        let rotation_matrix = if rotation_vec.norm() > 0.0 {
            na::Rotation3::from_axis_angle(&na::UnitVector3::new_normalize(rotation_vec), rotation_vec.norm())
        } else {
            na::Rotation3::identity()
        };
        let similarity = na::SimilarityMatrix3::from_parts(na::Translation3::from(translation_vec), rotation_matrix, scale);
        ModelMatrix(similarity)
    }
    #[allow(clippy::cast_possible_truncation)]
    pub fn get_transform_at_time(&self, time_sec: f32, fps: f32) -> ModelMatrix {
        let (frame_floor, frame_ceil, w_ceil) = self.get_smooth_time_indices(time_sec, fps);
        let mm_floor = self.get_transform_at_idx(frame_floor as u32);
        let mm_ceil = self.get_transform_at_idx(frame_ceil as u32);
        mm_floor.interpolate(&mm_ceil, w_ceil)
    }
    /// Get the number of frames in the `TransformSequence`.
    pub fn num_frames(&self) -> usize {
        self.translations.shape()[0]
    }
}
