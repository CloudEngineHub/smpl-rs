use log::info;
use ndarray as nd;

use ndarray_npy::{NpzReader, NpzWriter};
use std::{
    ffi::OsStr,
    fs::File,
    io::{Cursor, Read, Seek, Write},
    path::Path,
};

use crate::common::types::{Gender, SmplType};
use num_traits::FromPrimitive;

// SmplType:
// 0 - SMPL
// 1 - SMPLH
// 2 - SMPLX (locked head version)
// 3 - SUPR
// 4 - SMPLPP

// Gender:
// 0 - Neutral
// 1 - Male
// 2 - Female

/// The ``SmplCodec`` contains all of the contents of a ``.smpl`` file  
#[derive(Debug, Clone)]
pub struct SmplCodec {
    pub smpl_version: i32,
    pub gender: i32,

    pub shape_parameters: Option<nd::Array1<f32>>,
    pub expression_parameters: Option<nd::Array2<f32>>, // nr_frames x [10-100]

    pub frame_count: i32,
    pub frame_rate: Option<f32>,                   // Required if frame_count > 1
    pub body_translation: Option<nd::Array2<f32>>, // nr_frames x 3
    pub body_pose: Option<nd::Array3<f32>>,        // nr_frames x nr_joints x 3
    pub head_pose: Option<nd::Array3<f32>>,        // nr_frames x nr_joints x 3
    pub left_hand_pose: Option<nd::Array3<f32>>,   // nr_frames x nr_joints x 3
    pub right_hand_pose: Option<nd::Array3<f32>>,  // nr_frames x nr_joints x 3
}
impl Default for SmplCodec {
    fn default() -> Self {
        Self {
            smpl_version: 2, // SmplX
            gender: 0,       // Neutral
            shape_parameters: None,
            expression_parameters: None,
            frame_count: 1,
            frame_rate: None,
            body_translation: None,
            body_pose: None,
            head_pose: None,
            left_hand_pose: None,
            right_hand_pose: None,
        }
    }
}

impl SmplCodec {
    /// # Panics
    /// Will panic if it can't create the file
    pub fn to_file(&self, path: &str) {
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
        npz.add_array("smplVersion", &nd::Array0::<i32>::from_elem((), self.smpl_version))
            .unwrap();

        npz.add_array("gender", &nd::Array0::<i32>::from_elem((), self.gender)).unwrap();

        if let Some(shape_params) = &self.shape_parameters {
            npz.add_array("shapeParameters", shape_params).unwrap();
        }

        if let Some(expression_parameters) = &self.expression_parameters {
            npz.add_array("expressionParameters", expression_parameters).unwrap();
        }

        npz.add_array("frameCount", &nd::Array0::<i32>::from_elem((), self.frame_count)).unwrap();

        if let Some(frame_rate) = self.frame_rate {
            npz.add_array("frameRate", &nd::Array0::<f32>::from_elem((), frame_rate)).unwrap();
        }

        if let Some(body_translation) = &self.body_translation {
            npz.add_array("bodyTranslation", body_translation).unwrap();
        }

        if let Some(body_pose) = &self.body_pose {
            npz.add_array("bodyPose", body_pose).unwrap();
        }

        if let Some(head_pose) = &self.head_pose {
            npz.add_array("headPose", head_pose).unwrap();
        }

        if let Some(left_hand_pose) = &self.left_hand_pose {
            npz.add_array("leftHandPose", left_hand_pose).unwrap();
        }

        if let Some(right_hand_pose) = &self.right_hand_pose {
            npz.add_array("rightHandPose", right_hand_pose).unwrap();
        }
    }

    fn from_npz_reader<R: Read + Seek>(npz: &mut NpzReader<R>) -> Self {
        // params required
        let smpl_version_arr: nd::Array0<i32> = npz.by_name("smplVersion").expect("smplVersion.npy should exist and be a int32");

        let smpl_version = smpl_version_arr.into_scalar();
        let gender_arr: nd::Array0<i32> = npz.by_name("gender").expect("gender.npy should exist and be a int32");
        let gender = gender_arr.into_scalar();

        // betas and expression
        let shape_parameters: Option<nd::Array1<f32>> = npz.by_name("shapeParameters").ok();
        let expression_parameters: Option<nd::Array2<f32>> = npz.by_name("expressionParameters").ok();

        // anim
        let frame_count_arr: nd::Array0<i32> = npz.by_name("frameCount").expect("frameCount.npy should exist and be a int32");
        let frame_count = frame_count_arr.into_scalar();

        let body_translation: Option<nd::Array2<f32>> = npz.by_name("bodyTranslation").ok(); //(num_animation_frames,3)
        let (head_pose, left_hand_pose, right_hand_pose) = if smpl_version == 4 {
            (None, None, None)
        } else {
            (
                npz.by_name("headPose").ok(),
                npz.by_name("leftHandPose").ok(),
                npz.by_name("rightHandPose").ok(),
            )
        };
        // Handle body_pose depending on the smpl_version
        let body_pose: Option<nd::Array3<f32>> = if smpl_version == 4 {
            npz.by_name("bodyPose").ok().map(|arr2: nd::Array2<f32>| arr2.insert_axis(nd::Axis(2)))
        } else {
            npz.by_name("bodyPose").ok()
        };

        let frame_rate = if frame_count > 1 {
            let fps_arr: nd::Array0<f32> = npz
                .by_name("frameRate")
                .expect("frameRate.npy should exist and be a f32. It's required because frameCount >1");
            Some(fps_arr.into_scalar())
        } else {
            None
        };

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
        }
    }

    /// # Panics
    /// Will panic if it can't open the file
    pub fn from_file(path: &str) -> Self {
        let mut npz = NpzReader::new(std::fs::File::open(path).unwrap_or_else(|_| panic!("Could not find/open file: {path}"))).unwrap();

        // println!("names is {:?}", npz.names());

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
