use cpp::cpp;
use cpp::cpp_class;

cpp! {{
    #include <Eigen/Core>
    #include <Eigen/Dense>
    typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> MatX;
    typedef double Scalar;
    typedef size_t Index;
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
}
