use ndarray as nd;
use num_traits;

//there is currently no scatter or gather in ndarray but you can implement it with //https://github.com/rust-ndarray/ndarray/issues/526#issuecomment-434917877
//TODO do not use vecs, use arrays
pub trait Gather1D<T: nd::ScalarOperand + num_traits::identities::Zero + Copy> {
    fn gather(&self, indices: &[usize]) -> nd::Array1<T>; //perform the equivalent from python of dst=array[indices] where self is the
                                                          // array
}
impl<T: nd::ScalarOperand + num_traits::identities::Zero + Copy> Gather1D<T> for nd::Array1<T> {
    fn gather(&self, indices: &[usize]) -> nd::Array1<T> {
        let mut res = nd::Array1::<T>::zeros(indices.len());
        for (i_out, &i_in) in indices.iter().enumerate() {
            res[i_out] = self[i_in];
        }
        res
    }
}
pub trait Gather2D<T: nd::ScalarOperand + num_traits::identities::Zero + Copy> {
    fn gather(&self, indices_rows: &[usize], indices_cols: &[usize]) -> nd::Array2<T>; //perform the equivalent from python of dst=src[indices_rows, indices_cols]
                                                                                       // where self is the src
}
impl<T: nd::ScalarOperand + num_traits::identities::Zero + Copy> Gather2D<T> for nd::Array2<T> {
    fn gather(&self, indices_rows: &[usize], indices_cols: &[usize]) -> nd::Array2<T> {
        // let mut res = nd::Array1::<T>::zeros((indices.len()));
        let mut res = nd::Array2::zeros((indices_rows.len(), indices_cols.len()));
        for (i_out, &i_in) in indices_rows.iter().enumerate() {
            for (j_out, &j_in) in indices_cols.iter().enumerate() {
                // res[(i_out)] = arr[i_in as usize];
                res[(i_out, j_out)] = self[(i_in, j_in)];
            }
        }
        res
    }
}

pub trait Gather3D<T: nd::ScalarOperand + num_traits::identities::Zero + Copy> {
    fn gather(&self, indices_rows: &[usize], indices_cols: &[usize], indices_depth: &[usize]) -> nd::Array3<T>; //perform the equivalent from python of dst=src[indices_rows, indices_cols,
                                                                                                                // indices_depth] where self is the src. An extension of Gather to 3D arrays
}
impl<T: nd::ScalarOperand + num_traits::identities::Zero + Copy> Gather3D<T> for nd::Array3<T> {
    fn gather(&self, indices_rows: &[usize], indices_cols: &[usize], indices_depth: &[usize]) -> nd::Array3<T> {
        // let mut res = nd::Array1::<T>::zeros((indices.len()));
        let mut res = nd::Array3::zeros((indices_rows.len(), indices_cols.len(), indices_depth.len()));
        for (i_out, &i_in) in indices_rows.iter().enumerate() {
            for (j_out, &j_in) in indices_cols.iter().enumerate() {
                for (k_out, &k_in) in indices_depth.iter().enumerate() {
                    // res[(i_out)] = arr[i_in as usize];
                    res[(i_out, j_out, k_out)] = self[(i_in, j_in, k_in)];
                }
            }
        }
        res
    }
}

pub trait Scatter1D<T: nd::ScalarOperand + num_traits::identities::Zero + Copy> {
    fn scatter(&self, indices: &[usize], dst: &mut nd::Array1<T>); //perform the equivalent from python of dst[indices]=src where self is the src
}
impl<T: nd::ScalarOperand + num_traits::identities::Zero + Copy> Scatter1D<T> for nd::Array1<T> {
    fn scatter(&self, indices: &[usize], dst: &mut nd::Array1<T>) {
        for (i_in, &i_out) in indices.iter().enumerate() {
            dst[i_out] = self[i_in];
        }
    }
}

pub trait Scatter2D<T: nd::ScalarOperand + num_traits::identities::Zero + Copy> {
    fn scatter(&self, indices_rows: &[usize], indices_cols: &[usize], dst: &mut nd::Array2<T>); //perform the equivalent from python of array[indices_rows,indices_cols]=src
                                                                                                // where self is the array
}
impl<T: nd::ScalarOperand + num_traits::identities::Zero + Copy> Scatter2D<T> for nd::Array2<T> {
    fn scatter(&self, indices_rows: &[usize], indices_cols: &[usize], dst: &mut nd::Array2<T>) {
        for (i_in, &i_out) in indices_rows.iter().enumerate() {
            for (j_in, &j_out) in indices_cols.iter().enumerate() {
                dst[(i_out, j_out)] = self[(i_in, j_in)];
            }
        }
    }
}
