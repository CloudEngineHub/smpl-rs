use gloss_hecs::Entity;
use gloss_py_macros::PyComponent;
use gloss_renderer::scene::Scene;
use pyo3::prelude::*;
use smpl_gloss_integration::components::{Follow, FollowParams, Follower, FollowerType};
use smpl_utils::convert_enum_from;
#[pyclass(name = "FollowerType", module = "smpl_rs.types", unsendable, eq, eq_int)]
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum PyFollowerType {
    Cam = 0,
    #[default]
    Lights,
    CamAndLights,
}
convert_enum_from!(PyFollowerType, FollowerType, Cam, Lights, CamAndLights,);
#[pyclass(name = "Follower", module = "smpl_rs.components", unsendable)]
#[derive(Clone, PyComponent)]
pub struct PyFollower {
    pub inner: Follower,
}
#[pymethods]
impl PyFollower {
    #[new]
    #[pyo3(
        signature = (
            max_strength = None,
            dist_start = None,
            dist_end = None,
            follower_type = None,
            follow_all = None
        )
    )]
    #[pyo3(
        text_signature = "(max_strength: Optional[float] = None, dist_start: Optional[float] = None, dist_end: Optional[float] = None, follower_type: Optional[FollowerType] = None, follow_all: Optional[bool] = None) -> Follower"
    )]
    pub fn new(
        max_strength: Option<f32>,
        dist_start: Option<f32>,
        dist_end: Option<f32>,
        follower_type: Option<PyFollowerType>,
        follow_all: Option<bool>,
    ) -> Self {
        let def = FollowParams::default();
        let params = FollowParams {
            max_strength: max_strength.unwrap_or(def.max_strength),
            dist_start: dist_start.unwrap_or(def.dist_start),
            dist_end: dist_end.unwrap_or(def.dist_end),
            follower_type: follower_type.map_or(def.follower_type, FollowerType::from),
            follow_all: follow_all.unwrap_or(def.follow_all),
        };
        Self {
            inner: Follower::new(params),
        }
    }
}
#[pyclass(name = "Follow", module = "smpl_rs.components", unsendable)]
#[derive(Clone, PyComponent)]
pub struct PyFollow {
    pub inner: Follow,
}
#[pymethods]
#[allow(clippy::new_without_default)]
impl PyFollow {
    #[new]
    #[pyo3(signature = ())]
    #[pyo3(text_signature = "() -> Follow")]
    pub fn new() -> Self {
        Self { inner: Follow }
    }
}
