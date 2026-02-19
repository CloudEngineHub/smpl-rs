use crate::common::types::{Gender, SmplType};
use log::info;
use ndarray as nd;
use ndarray_npy::{NpzReader, NpzWriter, ReadNpzError, ReadableElement};
use num_traits::FromPrimitive;
use smpl_utils::log;
use std::{
    ffi::OsStr,
    fs::{self, File},
    io::{Cursor, Read, Seek, Write},
    path::Path,
};
/// The ``SmplCodec`` contains all of the contents of a ``.smpl`` file
#[derive(Debug, Clone)]
pub struct SmplCodec {
    pub smpl_version: i32,
    pub gender: i32,
    pub shape_parameters: Option<nd::Array1<f32>>,
    pub expression_parameters: Option<nd::Array2<f32>>,
    pub frame_count: i32,
    pub frame_rate: Option<f32>,
    pub body_translation: Option<nd::Array2<f32>>,
    pub body_pose: Option<nd::Array3<f32>>,
    pub head_pose: Option<nd::Array3<f32>>,
    pub left_hand_pose: Option<nd::Array3<f32>>,
    pub right_hand_pose: Option<nd::Array3<f32>>,
    pub vertex_offsets: Option<nd::Array2<f32>>,
}
impl Default for SmplCodec {
    fn default() -> Self {
        Self {
            smpl_version: 2,
            gender: 0,
            shape_parameters: None,
            expression_parameters: None,
            frame_count: 1,
            frame_rate: None,
            body_translation: None,
            body_pose: None,
            head_pose: None,
            left_hand_pose: None,
            right_hand_pose: None,
            vertex_offsets: None,
        }
    }
}
impl SmplCodec {
    /// # Panics
    /// Will panic if it can't create the file
    pub fn to_file(&self, path: &str) {
        let parent_path = Path::new(path).parent();
        let file_name = Path::new(path).file_name();
        let Some(parent_path) = parent_path else {
            log!("Error: Exporting SMPL - Something wrong with the path: {}", path);
            return;
        };
        if !parent_path.exists() {
            let _ = fs::create_dir_all(parent_path);
        }
        let Some(_) = file_name else {
            log!("Error: Exporting SMPL - no file name found: {}", path);
            return;
        };
        let mut path_with_suffix = path.to_string();
        let extension = Path::new(path).extension().and_then(OsStr::to_str);
        if let Some(ext) = extension {
            if ext != "smpl" {
                path_with_suffix += ".smpl";
            }
        }
        info!("saving smpl codec in {path_with_suffix}");
        let mut npz = NpzWriter::new_compressed(File::create(path_with_suffix).unwrap());
        self.write_to_npz(&mut npz);
        npz.finish().unwrap();
    }
    /// # Panics
    /// Will panic if it can't write the npz
    pub fn to_buf(&self) -> Vec<u8> {
        let vec = Vec::new();
        let mut cursor = Cursor::new(vec);
        let mut npz = NpzWriter::new_compressed(&mut cursor);
        self.write_to_npz(&mut npz);
        let out = npz.finish().unwrap();
        out.to_owned().into_inner()
    }
    /// # Panics
    /// Will panic if it can't write the npz
    pub fn write_to_npz<W: Write + Seek>(&self, npz: &mut NpzWriter<W>) {
        npz.add_array("smplVersion.npy", &nd::Array0::<i32>::from_elem((), self.smpl_version))
            .unwrap();
        npz.add_array("gender.npy", &nd::Array0::<i32>::from_elem((), self.gender)).unwrap();
        if let Some(shape_params) = &self.shape_parameters {
            npz.add_array("shapeParameters.npy", shape_params).unwrap();
        }
        if let Some(expression_parameters) = &self.expression_parameters {
            npz.add_array("expressionParameters.npy", expression_parameters).unwrap();
        }
        npz.add_array("frameCount.npy", &nd::Array0::<i32>::from_elem((), self.frame_count))
            .unwrap();
        if let Some(frame_rate) = self.frame_rate {
            npz.add_array("frameRate.npy", &nd::Array0::<f32>::from_elem((), frame_rate)).unwrap();
        }
        if let Some(body_translation) = &self.body_translation {
            npz.add_array("bodyTranslation.npy", body_translation).unwrap();
        }
        if let Some(body_pose) = &self.body_pose {
            npz.add_array("bodyPose.npy", body_pose).unwrap();
        }
        if let Some(head_pose) = &self.head_pose {
            npz.add_array("headPose.npy", head_pose).unwrap();
        }
        if let Some(left_hand_pose) = &self.left_hand_pose {
            npz.add_array("leftHandPose.npy", left_hand_pose).unwrap();
        }
        if let Some(right_hand_pose) = &self.right_hand_pose {
            npz.add_array("rightHandPose.npy", right_hand_pose).unwrap();
        }
        if let Some(vertex_offsets) = &self.vertex_offsets {
            npz.add_array("vertexOffsets.npy", vertex_offsets).unwrap();
        }
    }
    /// Wrapper around [`NpzReader::by_name`] that handles both `"key"` and `"key.npy"` entry names
    /// for backwards compatibility with `.smpl` files generated with a different `ndarray` version.
    fn npz_by_name<R, S, D>(npz: &mut NpzReader<R>, name: &str) -> Result<nd::ArrayBase<S, D>, ReadNpzError>
    where
        R: Read + Seek,
        S: nd::DataOwned,
        S::Elem: ReadableElement,
        D: nd::Dimension,
    {
        npz.by_name(name).or_else(|_| {
            let alt = match name.strip_suffix(".npy") {
                Some(stem) => stem.to_string(),
                None => format!("{name}.npy"),
            };
            npz.by_name(&alt)
        })
    }
    fn from_npz_reader<R: Read + Seek>(npz: &mut NpzReader<R>) -> Self {
        let smpl_version_arr: nd::Array0<i32> = Self::npz_by_name(npz, "smplVersion.npy").expect("smplVersion.npy should exist and be a int32");
        let smpl_version = smpl_version_arr.into_scalar();
        let gender_arr: nd::Array0<i32> = Self::npz_by_name(npz, "gender.npy").expect("gender.npy should exist and be a int32");
        let gender = gender_arr.into_scalar();
        let shape_parameters: Option<nd::Array1<f32>> = Self::npz_by_name(npz, "shapeParameters.npy").ok();
        let expression_parameters: Option<nd::Array2<f32>> = Self::npz_by_name(npz, "expressionParameters.npy").ok();
        let frame_count_arr: nd::Array0<i32> = Self::npz_by_name(npz, "frameCount.npy").expect("frameCount.npy should exist and be a int32");
        let frame_count = frame_count_arr.into_scalar();
        let body_translation: Option<nd::Array2<f32>> = Self::npz_by_name(npz, "bodyTranslation.npy").ok();
        let (head_pose, left_hand_pose, right_hand_pose) = if smpl_version == 4 {
            (None, None, None)
        } else {
            (
                Self::npz_by_name(npz, "headPose.npy").ok(),
                Self::npz_by_name(npz, "leftHandPose.npy").ok(),
                Self::npz_by_name(npz, "rightHandPose.npy").ok(),
            )
        };
        let body_pose: Option<nd::Array3<f32>> = if smpl_version == 4 {
            Self::npz_by_name(npz, "bodyPose.npy")
                .ok()
                .map(|arr2: nd::Array2<f32>| arr2.insert_axis(nd::Axis(2)))
        } else {
            Self::npz_by_name(npz, "bodyPose.npy").ok()
        };
        let frame_rate = if frame_count > 1 {
            let fps_arr: nd::Array0<f32> =
                Self::npz_by_name(npz, "frameRate.npy").expect("frameRate.npy should exist and be a f32. It's required because frameCount >1");
            Some(fps_arr.into_scalar())
        } else {
            None
        };
        let vertex_offsets: Option<nd::Array2<f32>> = Self::npz_by_name(npz, "vertexOffsets.npy").ok();
        Self {
            smpl_version,
            gender,
            shape_parameters,
            expression_parameters,
            frame_count,
            frame_rate,
            body_translation,
            body_pose,
            head_pose,
            left_hand_pose,
            right_hand_pose,
            vertex_offsets,
        }
    }
    /// # Panics
    /// Will panic if it can't open the file
    pub fn from_file(path: &str) -> Self {
        let mut npz = NpzReader::new(std::fs::File::open(path).unwrap_or_else(|_| panic!("Could not find/open file: {path}"))).unwrap();
        Self::from_npz_reader(&mut npz)
    }
    /// # Panics
    /// Will panic if it can't open the buf as npz
    pub fn from_buf(buf: &[u8]) -> Self {
        let reader = Cursor::new(buf);
        let mut npz = NpzReader::new(reader).unwrap();
        Self::from_npz_reader(&mut npz)
    }
    /// # Panics
    /// Will panic if the ``smpl_version`` cannot be parsed to the enum
    pub fn smpl_type(&self) -> SmplType {
        FromPrimitive::from_i32(self.smpl_version).unwrap()
    }
    /// # Panics
    /// Will panic if the ``gender`` cannot be parsed to the enum
    pub fn gender(&self) -> Gender {
        FromPrimitive::from_i32(self.gender).unwrap()
    }
}
