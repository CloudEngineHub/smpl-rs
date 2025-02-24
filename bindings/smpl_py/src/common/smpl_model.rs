//base class for all smpl models
//example of dynamic return type
//https://github.com/daemontus/pyo3/blob/48c90d95863dd582bbbb70f2ff776660820723dc/guide/src/class.md
//https://github.com/PyO3/pyo3/issues/1637

use pyo3::prelude::*;

#[pyclass(subclass)]
struct SmplModel {}

// #[pymethods]
// impl BaseClass {
//     #[new]
//     fn new() -> Self {
//         BaseClass { val1: 10 }
//     }

//     pub fn method(&self) -> PyResult<usize> {
//         Ok(self.val1)
//     }
// }
