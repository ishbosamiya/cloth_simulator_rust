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

    unsafe fn data_raw(&self) -> *const Scalar {
        cpp!([self as "const MatX*"] -> *const Scalar as "const Scalar*" {
            return self->data();
        })
    }

    pub fn data(&self) -> &[Scalar] {
        unsafe {
            return std::slice::from_raw_parts(self.data_raw(), self.size());
        }
    }
}

impl ops::Neg for &MatX {
    type Output = MatX;

    fn neg(self) -> Self::Output {
        unsafe {
            cpp!([self as "const MatX*"] -> MatX as "MatX" {
                return -(*self);
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

impl ops::AddAssign<&MatX> for MatX {
    fn add_assign(&mut self, other: &MatX) {
        unsafe {
            cpp!([self as "MatX*", other as "const MatX*"] {
                *self += *other;
            })
        }
    }
}

impl ops::Sub<&MatX> for &MatX {
    type Output = MatX;

    fn sub(self, rhs: &MatX) -> MatX {
        unsafe {
            cpp!([self as "const MatX*", rhs as "const MatX*"] -> MatX as "MatX" {
                return *self - *rhs;
            })
        }
    }
}

impl ops::SubAssign<&MatX> for MatX {
    fn sub_assign(&mut self, other: &MatX) {
        unsafe {
            cpp!([self as "MatX*", other as "const MatX*"] {
                *self -= *other;
            })
        }
    }
}

impl ops::Mul<&MatX> for &MatX {
    type Output = MatX;

    fn mul(self, rhs: &MatX) -> MatX {
        unsafe {
            cpp!([self as "const MatX*", rhs as "const MatX*"] -> MatX as "MatX" {
                return *self * *rhs;
            })
        }
    }
}

impl ops::Mul<Scalar> for &MatX {
    type Output = MatX;

    fn mul(self, rhs: Scalar) -> MatX {
        unsafe {
            cpp!([self as "const MatX*", rhs as "Scalar"] -> MatX as "MatX" {
                return *self * rhs;
            })
        }
    }
}

impl ops::Mul<&MatX> for Scalar {
    type Output = MatX;

    fn mul(self, rhs: &MatX) -> MatX {
        unsafe {
            cpp!([self as "Scalar", rhs as "const MatX*"] -> MatX as "MatX" {
                return self * *rhs;
            })
        }
    }
}

impl ops::MulAssign<Scalar> for MatX {
    fn mul_assign(&mut self, rhs: Scalar) {
        unsafe {
            cpp!([self as "MatX*", rhs as "Scalar"] {
                *self *= rhs;
            })
        }
    }
}

impl ops::Div<Scalar> for &MatX {
    type Output = MatX;

    fn div(self, rhs: Scalar) -> MatX {
        unsafe {
            cpp!([self as "const MatX*", rhs as "Scalar"] -> MatX as "MatX" {
                return *self / rhs;
            })
        }
    }
}

impl ops::DivAssign<Scalar> for MatX {
    fn div_assign(&mut self, rhs: Scalar) {
        unsafe {
            cpp!([self as "MatX*", rhs as "Scalar"] {
                *self /= rhs;
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
    fn eigenmatx_data() {
        let mut mat = MatX::new_with_size(2, 2);
        mat.set(0, 0, 1.0);
        mat.set(0, 1, 2.0);
        mat.set(1, 0, 3.0);
        mat.set(1, 1, 4.0);
        // Due to column major storage, the sequence in memory is
        // 1, 3, 2, 4 compared to 1, 2, 3, 4 if it was row major
        assert_eq!(mat.data(), [1.0, 3.0, 2.0, 4.0]);
        assert_eq!(mat.get(0, 0), 1.0);
        assert_eq!(mat.get(0, 1), 2.0);
        assert_eq!(mat.get(1, 0), 3.0);
        assert_eq!(mat.get(1, 1), 4.0);
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

    #[test]
    fn eigenmatx_addassign() {
        let mut mat1 = MatX::new_with_size(1, 1);
        let mut mat2 = MatX::new_with_size(1, 1);
        mat1.set(0, 0, 2.0);
        mat2.set(0, 0, 3.0);
        mat1 += &mat2;
        assert_eq!(mat1.size(), 1);
        assert_eq!(mat1.get(0, 0), 5.0);
    }

    #[test]
    fn eigenmatx_neg() {
        let mut mat1 = MatX::new_with_size(2, 1);
        mat1.set(0, 0, 2.0);
        mat1.set(1, 0, -3.0);
        let mat2 = -&mat1;
        assert_eq!(mat2.size(), 2);
        assert_eq!(mat2.get(0, 0), -2.0);
        assert_eq!(mat2.get(1, 0), 3.0);
    }

    #[test]
    fn eigenmatx_sub() {
        let mut mat1 = MatX::new_with_size(1, 1);
        let mut mat2 = MatX::new_with_size(1, 1);
        mat1.set(0, 0, 2.0);
        mat2.set(0, 0, 3.0);
        let mat3 = &mat1 - &mat2;
        assert_eq!(mat3.size(), 1);
        assert_eq!(mat3.get(0, 0), -1.0);
    }

    #[test]
    fn eigenmatx_subassign() {
        let mut mat1 = MatX::new_with_size(1, 1);
        let mut mat2 = MatX::new_with_size(1, 1);
        mat1.set(0, 0, 2.0);
        mat2.set(0, 0, 3.0);
        mat1 -= &mat2;
        assert_eq!(mat1.size(), 1);
        assert_eq!(mat1.get(0, 0), -1.0);
    }

    #[test]
    fn eigenmatx_mul() {
        // MatX*MatX
        {
            let mut mat1 = MatX::new_with_size(1, 2);
            mat1.set(0, 0, 2.0);
            mat1.set(0, 1, 3.0);
            let mut mat2 = MatX::new_with_size(2, 2);
            mat2.set(0, 0, 1.0);
            mat2.set(0, 1, 2.0);
            mat2.set(1, 0, 3.0);
            mat2.set(1, 1, 4.0);
            let mat3 = &mat1 * &mat2;
            assert_eq!(mat3.data(), [11.0, 16.0]);
        }

        // MatX*Scalar
        {
            let mut mat1 = MatX::new_with_size(1, 2);
            mat1.set(0, 0, 2.0);
            mat1.set(0, 1, 3.0);
            let scalar = 5.0;
            assert_eq!((&mat1 * scalar).data(), [10.0, 15.0]);
        }

        // Scalar*MatX
        {
            let mut mat1 = MatX::new_with_size(1, 2);
            mat1.set(0, 0, 2.0);
            mat1.set(0, 1, 3.0);
            let scalar = 5.0;
            assert_eq!((scalar * &mat1).data(), [10.0, 15.0]);
        }
    }

    #[test]
    fn eigenmatx_mulassign() {
        let mut mat1 = MatX::new_with_size(1, 2);
        mat1.set(0, 0, 2.0);
        mat1.set(0, 1, 3.0);
        mat1 *= 5.0;
        assert_eq!(mat1.data(), [10.0, 15.0]);
    }

    #[test]
    fn eigenmatx_div() {
        // MatX/Scalar
        let mut mat1 = MatX::new_with_size(1, 2);
        mat1.set(0, 0, 2.0);
        mat1.set(0, 1, 3.0);
        let scalar = 5.0;
        assert_eq!((&mat1 / scalar).data(), [2.0 / 5.0, 3.0 / 5.0]);
    }

    #[test]
    fn eigenmatx_divassign() {
        let mut mat1 = MatX::new_with_size(1, 2);
        mat1.set(0, 0, 2.0);
        mat1.set(0, 1, 3.0);
        mat1 /= 5.0;
        assert_eq!(mat1.data(), [2.0 / 5.0, 3.0 / 5.0]);
    }
}
