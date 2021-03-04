use cpp::cpp;
use cpp::cpp_class;
use std::ops;

cpp! {{
    #include <Eigen/Core>
    #include <Eigen/Dense>
    typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> MatX;
    typedef double Scalar;
    typedef Eigen::Index Index;
}}

use f64 as Scalar;
use usize as Index;

cpp_class!(pub unsafe struct MatX as "MatX");
impl MatX {
    pub fn new() -> Self {
        unsafe {
            cpp!([] -> MatX as "MatX" {
                return MatX();
            })
        }
    }

    pub fn new_with_size(x: Index, y: Index) -> Self {
        unsafe {
            cpp!([x as "Index", y as "Index"] -> MatX as "MatX" {
                return MatX(x, y);
            })
        }
    }

    pub fn set(&mut self, x: Index, y: Index, value: Scalar) {
        unsafe {
            cpp!([self as "MatX*", x as "Index", y as "Index", value as "Scalar"] {
                (*self)(x, y) = value;
            })
        }
    }

    pub fn get(&self, x: Index, y: Index) -> Scalar {
        unsafe {
            cpp!([self as "const MatX*", x as "Index", y as "Index"] -> Scalar as "Scalar" {
                return (*self)(x, y);
            })
        }
    }

    pub fn resize(&mut self, x: Index, y: Index) {
        unsafe {
            cpp!([self as "MatX*", x as "Index", y as "Index"] {
                return self->resize(x, y);
            })
        }
    }

    pub fn rows(&self) -> Index {
        unsafe {
            cpp!([self as "const MatX*"] -> Index as "Index" {
                return self->rows();
            })
        }
    }

    pub fn cols(&self) -> Index {
        unsafe {
            cpp!([self as "const MatX*"] -> Index as "Index" {
                return self->cols();
            })
        }
    }

    pub fn size(&self) -> Index {
        unsafe {
            cpp!([self as "const MatX*"] -> Index as "Index" {
                return self->size();
            })
        }
    }
}

impl ops::Add<&MatX> for &MatX {
    type Output = MatX;

    fn add(self, rhs: &MatX) -> MatX {
        unsafe {
            cpp!([self as "const MatX*", rhs as "const MatX*"] -> MatX as "MatX" {
                return *self + *rhs;
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eigenmatx_set_get() {
        let mut mat = MatX::new_with_size(1, 1);
        mat.set(0, 0, 5.0);
        assert_eq!(mat.get(0, 0), 5.0);
    }

    #[test]
    fn eigenmatx_resize() {
        let mut mat = MatX::new_with_size(1, 1);
        assert_eq!(mat.rows(), 1);
        assert_eq!(mat.cols(), 1);
        assert_eq!(mat.size(), 1);
        mat.resize(3, 2);
        assert_eq!(mat.rows(), 3);
        assert_eq!(mat.cols(), 2);
        assert_eq!(mat.size(), 6);
    }

    #[test]
    fn eigenmatx_add() {
        let mut mat1 = MatX::new_with_size(1, 1);
        let mut mat2 = MatX::new_with_size(1, 1);
        mat1.set(0, 0, 2.0);
        mat2.set(0, 0, 3.0);
        let mat3 = &mat1 + &mat2;
        assert_eq!(mat3.size(), 1);
        assert_eq!(mat3.get(0, 0), 5.0);
    }
}
