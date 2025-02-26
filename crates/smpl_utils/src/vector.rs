use na::{Point3, Vector2, Vector3, Vector4};
use ndarray as nd;
extern crate nalgebra as na;
pub type Color4f = [f32; 4];
pub fn color_to_v4(color: &Color4f) -> Vector4f {
    Vector4f::new(color[0], color[1], color[2], color[3])
}
pub fn u8arr_to_v4(color: [u8; 3]) -> Color4f {
    [f32::from(color[0]) / 255., f32::from(color[1]) / 255., f32::from(color[2]) / 255., 1.]
}
pub type Vector2f = Vector2<f32>;
pub type Point3f = Point3<f32>;
pub type Point3d = Point3<f64>;
pub type Vector3f = Vector3<f32>;
pub type Vector4f = Vector4<f32>;
pub type Vector4s = Vector4<u16>;
pub type Vector4u = Vector4<u32>;
pub fn addv2f_scaled(a: &Vector2f, b: &Vector2f, scale: f32) -> Vector2f {
    Vector2::new(a.x + b.x * scale, a.y + b.y * scale)
}
pub fn subv3f(a: &Vector3f, b: &Vector3f) -> Vector3f {
    Vector3::new(a.x - b.x, a.y - b.y, a.z - b.z)
}
pub fn addv3f(a: &Vector3f, b: &Vector3f) -> Vector3f {
    Vector3::new(a.x + b.x, a.y + b.y, a.z + b.z)
}
pub fn addv3f_scaled(a: &Vector3f, b: &Vector3f, scale: f32) -> Vector3f {
    Vector3::new(a.x + b.x * scale, a.y + b.y * scale, a.z + b.z * scale)
}
pub fn mulv3f(a: &Vector3f, s: f32) -> Vector3f {
    Vector3::new(a.x * s, a.y * s, a.z * s)
}
pub fn mulv3d(a: &Vector3d, s: f64) -> Vector3d {
    Vector3d::new(a.x * s, a.y * s, a.z * s)
}
pub fn len_sqrv3f(a: &Vector3f) -> f32 {
    a.x * a.x + a.y * a.y + a.z * a.z
}
pub fn dotv3f(a: &Vector3f, b: &Vector3f) -> f32 {
    a.x * b.x + a.y * b.y + a.z * b.z
}
pub fn vec_from_array_f(vertices: &nd::Array2<f32>, row_index: usize) -> Vector3f {
    let row = vertices.row(row_index);
    Vector3f::new(row[0], row[1], row[2])
}
pub fn array_to_vec3(arr: &Option<nd::Array2<f32>>) -> [f32; 3] {
    let v = arr
        .as_ref()
        .unwrap_or(&ndarray::Array2::<f32>::zeros((1, 3)))
        .index_axis(nd::Axis(0), 0)
        .to_owned();
    fixed_vec3(&v.to_vec())
}
pub fn vec_from_array3_f(vertices: &nd::Array3<f32>, col_index: usize, row_index: usize) -> Vector3f {
    Vector3f::new(
        vertices[[col_index, row_index, 0]],
        vertices[[col_index, row_index, 1]],
        vertices[[col_index, row_index, 2]],
    )
}
pub fn vec_from_array0_f(vertices: &nd::Array2<f32>) -> Vector3f {
    let row = vertices.row(0);
    Vector3f::new(row[0], row[1], row[2])
}
pub fn set_vec_from_array_f(vertices: &nd::Array2<f32>, row_index: usize, v: &mut Vector3f) {
    let row = vertices.row(row_index);
    v.x = row[0];
    v.y = row[1];
    v.z = row[2];
}
pub fn vec_from_vec(v: &[f32]) -> Vector3f {
    Vector3f::new(v[0], v[1], v[2])
}
pub fn vec_to_vec(v: &Vector3f) -> Vec<f32> {
    vec![v.x, v.y, v.z]
}
pub fn to_fixed_vec3(v: &Vector3f) -> [f32; 3] {
    [v.x, v.y, v.z]
}
pub fn fixed_vec3(v: &[f32]) -> [f32; 3] {
    [v[0], v[1], v[2]]
}
pub fn vec_from_fixed(v: &[f32; 3]) -> Vector3f {
    Vector3f::new(v[0], v[1], v[2])
}
pub type Line2D = Vec<Vector2f>;
pub fn len_v2f(a: &Vector2f) -> f32 {
    (a.x * a.x + a.y * a.y).sqrt()
}
pub type Vector2d = Vector2<f64>;
pub type Vector3d = Vector3<f64>;
pub fn v3d_from_v3f(v: &Vector3f) -> Vector3d {
    Vector3d::new(f64::from(v.x), f64::from(v.y), f64::from(v.z))
}
#[allow(clippy::cast_possible_truncation)]
pub fn v3f_from_v3d(v: &Vector3d) -> Vector3f {
    Vector3f::new(v.x as f32, v.y as f32, v.z as f32)
}
pub fn p3d_from_v3d(v: &Vector3d) -> Point3d {
    Point3d::new(v.x, v.y, v.z)
}
pub fn p3d_from_p3f(v: &Vector3f) -> Point3f {
    Point3f::new(v.x, v.y, v.z)
}
#[allow(clippy::cast_possible_truncation)]
pub fn p3f_from_p3d(v: &Vector3d) -> Point3f {
    Point3f::new(v.x as f32, v.y as f32, v.z as f32)
}
pub fn p3f_from_v3f(v: &Vector3f) -> Point3f {
    Point3f::new(v.x, v.y, v.z)
}
pub fn subv3d(a: &Vector3d, b: &Vector3d) -> Vector3d {
    Vector3::new(a.x - b.x, a.y - b.y, a.z - b.z)
}
pub fn addv3d(a: &Vector3d, b: &Vector3d) -> Vector3d {
    Vector3::new(a.x + b.x, a.y + b.y, a.z + b.z)
}
pub fn addv3d_scaled(a: &Vector3d, b: &Vector3d, scale: f64) -> Vector3d {
    Vector3::new(a.x + b.x * scale, a.y + b.y * scale, a.z + b.z * scale)
}
pub fn subv3f3d(a: &Vector3f, b: &Vector3f) -> Vector3d {
    Vector3::new(f64::from(a.x - b.x), f64::from(a.y - b.y), f64::from(a.z - b.z))
}
pub fn addv3f3d(a: &Vector3f, b: &Vector3f) -> Vector3d {
    Vector3::new(f64::from(a.x + b.x), f64::from(a.y + b.y), f64::from(a.z + b.z))
}
pub fn len_sqrv3d(a: &Vector3d) -> f64 {
    a.x * a.x + a.y * a.y + a.z * a.z
}
pub fn dotv3d(a: &Vector3d, b: &Vector3d) -> f64 {
    a.x * b.x + a.y * b.y + a.z * b.z
}
pub fn vec_from_array_d(vertices: &nd::Array2<f32>, row_index: usize) -> Vector3d {
    let row = vertices.row(row_index);
    Vector3d::new(f64::from(row[0]), f64::from(row[1]), f64::from(row[2]))
}
pub fn set_vec_from_array_d(vertices: &nd::Array2<f32>, row_index: usize, v: &mut Vector3d) {
    let row = vertices.row(row_index);
    v.x = f64::from(row[0]);
    v.y = f64::from(row[1]);
    v.z = f64::from(row[2]);
}
pub type Matrix3f = na::SimilarityMatrix3<f32>;
pub fn align_to_multiple_of_four(n: &mut usize) {
    *n = (*n + 3) & !3;
}
pub fn to_padded_byte_vector<T: bytemuck::NoUninit>(vec: &[T]) -> Vec<u8> {
    let arr_8: &[u8] = bytemuck::cast_slice(vec);
    let mut new_vec = arr_8.to_vec();
    while new_vec.len() % 4 != 0 {
        new_vec.push(0);
    }
    new_vec
}
