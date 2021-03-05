use cpp::cpp;
use cpp::cpp_class;
use std::ops;

cpp! {{
    #include <Eigen/Core>
    #include <Eigen/Dense>
    #include <Eigen/Sparse>
    #include <memory>
    typedef double Scalar;
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> MatX;
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> VecX;
    typedef Eigen::SparseMatrix<Scalar> SparseMatrix;
    typedef Eigen::Triplet<Scalar> Triplet;
    typedef Eigen::SimplicialLLT<SparseMatrix, Eigen::Upper> SimplicialLLT_;
    typedef std::unique_ptr<SimplicialLLT_> SimplicialLLT;
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

    pub fn new_random(x: Index, y: Index) -> Self {
        unsafe {
            cpp!([x as "Index", y as "Index"] -> MatX as "MatX" {
                return MatX::Random(x, y);
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

    unsafe fn data_raw_mut(&mut self) -> *mut Scalar {
        cpp!([self as "MatX*"] -> *mut Scalar as "Scalar*" {
            return self->data();
        })
    }

    pub fn data(&self) -> &[Scalar] {
        unsafe {
            return std::slice::from_raw_parts(self.data_raw(), self.size());
        }
    }

    pub fn data_mut(&mut self) -> &mut [Scalar] {
        unsafe {
            return std::slice::from_raw_parts_mut(self.data_raw_mut(), self.size());
        }
    }

    pub fn transpose(&self) -> MatX {
        unsafe {
            cpp!([self as "const MatX*"] -> MatX as "MatX"{
                return self->transpose();
            })
        }
    }

    pub fn transpose_in_place(&mut self) {
        unsafe {
            cpp!([self as "MatX*"]{
                self->transposeInPlace();
            })
        }
    }

    pub fn adjoint(&self) -> MatX {
        unsafe {
            cpp!([self as "const MatX*"] -> MatX as "MatX"{
                return self->adjoint();
            })
        }
    }

    pub fn adjoint_in_place(&mut self) {
        unsafe {
            cpp!([self as "MatX*"]{
                self->adjointInPlace();
            })
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

impl ops::Mul<&VecX> for &MatX {
    type Output = VecX;

    fn mul(self, rhs: &VecX) -> VecX {
        unsafe {
            cpp!([self as "const MatX*", rhs as "const VecX*"] -> VecX as "VecX" {
                return *self * *rhs;
            })
        }
    }
}

impl ops::Mul<&MatX> for &VecX {
    type Output = MatX;

    fn mul(self, rhs: &MatX) -> MatX {
        unsafe {
            cpp!([self as "const VecX*", rhs as "const MatX*"] -> MatX as "MatX" {
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

cpp_class!(pub unsafe struct VecX as "VecX");
impl VecX {
    pub fn new() -> Self {
        unsafe {
            cpp!([] -> VecX as "VecX" {
                return VecX();
            })
        }
    }

    pub fn new_with_size(size: Index) -> Self {
        unsafe {
            cpp!([size as "Index"] -> VecX as "VecX" {
                return VecX(size);
            })
        }
    }

    pub fn new_random(size: Index) -> Self {
        unsafe {
            cpp!([size as "Index"] -> VecX as "VecX" {
                return VecX::Random(size);
            })
        }
    }

    pub fn set(&mut self, x: Index, value: Scalar) {
        unsafe {
            cpp!([self as "VecX*", x as "Index", value as "Scalar"] {
                (*self)(x) = value;
            })
        }
    }

    pub fn get(&self, x: Index) -> Scalar {
        unsafe {
            cpp!([self as "const VecX*", x as "Index"] -> Scalar as "Scalar" {
                return (*self)(x);
            })
        }
    }

    pub fn resize(&mut self, x: Index) {
        unsafe {
            cpp!([self as "VecX*", x as "Index"] {
                return self->resize(x);
            })
        }
    }

    pub fn size(&self) -> Index {
        unsafe {
            cpp!([self as "const VecX*"] -> Index as "Index" {
                return self->size();
            })
        }
    }

    unsafe fn data_raw(&self) -> *const Scalar {
        cpp!([self as "const VecX*"] -> *const Scalar as "const Scalar*" {
            return self->data();
        })
    }

    unsafe fn data_raw_mut(&mut self) -> *mut Scalar {
        cpp!([self as "VecX*"] -> *mut Scalar as "Scalar*" {
            return self->data();
        })
    }

    pub fn data(&self) -> &[Scalar] {
        unsafe {
            return std::slice::from_raw_parts(self.data_raw(), self.size());
        }
    }

    pub fn data_mut(&mut self) -> &mut [Scalar] {
        unsafe {
            return std::slice::from_raw_parts_mut(self.data_raw_mut(), self.size());
        }
    }

    pub fn transpose(&self) -> MatX {
        unsafe {
            cpp!([self as "const VecX*"] -> MatX as "MatX"{
                return self->transpose();
            })
        }
    }

    pub fn dot(&self, other: &VecX) -> Scalar {
        unsafe {
            cpp!([self as "const VecX*", other as "const VecX*"] -> Scalar as "Scalar" {
                return self->dot(*other);
            })
        }
    }

    pub fn cross(&self, other: &VecX) -> VecX {
        assert_eq!(self.size(), 3);
        assert_eq!(other.size(), 3);
        unsafe {
            cpp!([self as "const VecX*", other as "const VecX*"] -> VecX as "VecX" {
                VecX res(3);
                auto vec = Eigen::Matrix<Scalar, 3, 1>((*self)[0],
                                                       (*self)[1],
                                                       (*self)[2]).cross(
                           Eigen::Matrix<Scalar, 3, 1>((*other)[0],
                                                       (*other)[1],
                                                       (*other)[2]));
                res(0) = vec(0);
                res(1) = vec(1);
                res(2) = vec(2);
                return res;
            })
        }
    }
}

impl ops::Neg for &VecX {
    type Output = VecX;

    fn neg(self) -> Self::Output {
        unsafe {
            cpp!([self as "const VecX*"] -> VecX as "VecX" {
                return -(*self);
            })
        }
    }
}

impl ops::Add<&VecX> for &VecX {
    type Output = VecX;

    fn add(self, rhs: &VecX) -> VecX {
        unsafe {
            cpp!([self as "const VecX*", rhs as "const VecX*"] -> VecX as "VecX" {
                return *self + *rhs;
            })
        }
    }
}

impl ops::AddAssign<&VecX> for VecX {
    fn add_assign(&mut self, other: &VecX) {
        unsafe {
            cpp!([self as "VecX*", other as "const VecX*"] {
                *self += *other;
            })
        }
    }
}

impl ops::Sub<&VecX> for &VecX {
    type Output = VecX;

    fn sub(self, rhs: &VecX) -> VecX {
        unsafe {
            cpp!([self as "const VecX*", rhs as "const VecX*"] -> VecX as "VecX" {
                return *self - *rhs;
            })
        }
    }
}

impl ops::SubAssign<&VecX> for VecX {
    fn sub_assign(&mut self, other: &VecX) {
        unsafe {
            cpp!([self as "VecX*", other as "const VecX*"] {
                *self -= *other;
            })
        }
    }
}

impl ops::Mul<Scalar> for &VecX {
    type Output = VecX;

    fn mul(self, rhs: Scalar) -> VecX {
        unsafe {
            cpp!([self as "const VecX*", rhs as "Scalar"] -> VecX as "VecX" {
                return *self * rhs;
            })
        }
    }
}

impl ops::Mul<&VecX> for Scalar {
    type Output = VecX;

    fn mul(self, rhs: &VecX) -> VecX {
        unsafe {
            cpp!([self as "Scalar", rhs as "const VecX*"] -> VecX as "VecX" {
                return self * *rhs;
            })
        }
    }
}

impl ops::MulAssign<Scalar> for VecX {
    fn mul_assign(&mut self, rhs: Scalar) {
        unsafe {
            cpp!([self as "VecX*", rhs as "Scalar"] {
                *self *= rhs;
            })
        }
    }
}

impl ops::Div<Scalar> for &VecX {
    type Output = VecX;

    fn div(self, rhs: Scalar) -> VecX {
        unsafe {
            cpp!([self as "const VecX*", rhs as "Scalar"] -> VecX as "VecX" {
                return *self / rhs;
            })
        }
    }
}

impl ops::DivAssign<Scalar> for VecX {
    fn div_assign(&mut self, rhs: Scalar) {
        unsafe {
            cpp!([self as "VecX*", rhs as "Scalar"] {
                *self /= rhs;
            })
        }
    }
}

cpp_class!(pub unsafe struct Triplet as "Triplet");
impl Triplet {
    pub fn new(x: Index, y: Index, value: Scalar) -> Self {
        unsafe {
            cpp!([x as "Index", y as "Index", value as "Scalar"] -> Triplet as "Triplet" {
                return Triplet(x, y, value);
            })
        }
    }
}

cpp_class!(pub unsafe struct SimplicialLLT as "SimplicialLLT");
impl SimplicialLLT {
    pub fn new() -> Self {
        unsafe {
            cpp!([] -> SimplicialLLT as "SimplicialLLT" {
                return std::unique_ptr<SimplicialLLT_>(new SimplicialLLT_);
            })
        }
    }

    pub fn analyze_pattern(&mut self, mat: &SparseMatrix) {
        unsafe {
            cpp!([self as "SimplicialLLT", mat as "const SparseMatrix*"] {
                self->analyzePattern(*mat);
            })
        }
    }

    pub fn compute(&self, mat: &SparseMatrix) {
        unsafe {
            cpp!([self as "SimplicialLLT", mat as "const SparseMatrix*"] {
                self->compute(*mat);
            })
        }
    }

    pub fn determinant(&self) -> Scalar {
        unsafe {
            cpp!([self as "SimplicialLLT"] -> Scalar as "Scalar" {
                return self->determinant();
            })
        }
    }

    pub fn factorize(&self, mat: &SparseMatrix) {
        unsafe {
            cpp!([self as "SimplicialLLT", mat as "const SparseMatrix*"] {
                self->factorize(*mat);
            })
        }
    }

    pub fn info(&self) -> ComputationInfo {
        let value;
        unsafe {
            value = cpp!([self as "SimplicialLLT"] -> i32 as "int32_t" {
                auto info = self->info();
                if (info == Eigen::Success) {
                    return 1;
                }
                else {
                    return 2;
                }
            });
        }
        if value == 1 {
            return ComputationInfo::Success;
        } else if value == 2 {
            return ComputationInfo::NumericalIssue;
        } else {
            panic!("eigen: couldn't set the correct ComputationInfo");
        }
    }

    /// Solves matrix equation Ax=b
    pub fn solve(&self, b: &VecX) -> VecX {
        unsafe {
            cpp!([self as "SimplicialLLT", b as "const VecX*"] -> VecX as "VecX" {
                return self->solve(*b);
            })
        }
    }
}

pub enum ComputationInfo {
    Success,
    NumericalIssue,
    NoConvergence,
    InvalidInput,
}

cpp_class!(pub unsafe struct SparseMatrix as "SparseMatrix");
impl SparseMatrix {
    pub fn new() -> Self {
        unsafe {
            cpp!([] -> SparseMatrix as "SparseMatrix" {
                return SparseMatrix();
            })
        }
    }

    pub fn new_with_size(x: Index, y: Index) -> Self {
        unsafe {
            cpp!([x as "Index", y as "Index"] -> SparseMatrix as "SparseMatrix" {
                return SparseMatrix(x, y);
            })
        }
    }

    pub fn new_from_mat(mat: &SparseMatrix) -> Self {
        unsafe {
            cpp!([mat as "const SparseMatrix*"] -> SparseMatrix as "SparseMatrix" {
                return SparseMatrix(*mat);
            })
        }
    }

    pub fn resize(&mut self, x: Index, y: Index) {
        unsafe {
            cpp!([self as "SparseMatrix*", x as "Index", y as "Index"] {
                return self->resize(x, y);
            })
        }
    }

    pub fn rows(&self) -> Index {
        unsafe {
            cpp!([self as "const SparseMatrix*"] -> Index as "Index" {
                return self->rows();
            })
        }
    }

    pub fn cols(&self) -> Index {
        unsafe {
            cpp!([self as "const SparseMatrix*"] -> Index as "Index" {
                return self->cols();
            })
        }
    }

    pub fn non_zeros(&self) -> Index {
        unsafe {
            cpp!([self as "const SparseMatrix*"] -> Index as "Index" {
                return self->nonZeros();
            })
        }
    }

    pub fn reserve(&mut self, nnz: Index) {
        unsafe {
            cpp!([self as "SparseMatrix*", nnz as "Index"] {
                return self->reserve(nnz);
            })
        }
    }

    pub fn insert(&mut self, x: Index, y: Index, value: Scalar) {
        unsafe {
            cpp!([self as "SparseMatrix*", x as "Index", y as "Index", value as "Scalar"] {
                self->insert(x, y) = value;
            })
        }
    }

    pub fn make_compressed(&mut self) {
        unsafe {
            cpp!([self as "SparseMatrix*"] {
                return self->makeCompressed();
            })
        }
    }

    pub fn transpose(&self) -> SparseMatrix {
        unsafe {
            cpp!([self as "const SparseMatrix*"] -> SparseMatrix as "SparseMatrix"{
                return self->transpose();
            })
        }
    }

    pub fn adjoint(&self) -> SparseMatrix {
        unsafe {
            cpp!([self as "const SparseMatrix*"] -> SparseMatrix as "SparseMatrix"{
                return self->adjoint();
            })
        }
    }

    pub fn set_from_triplets(&mut self, triplets: &[Triplet]) {
        let triplets_len = triplets.len();
        let triplets_ptr = triplets.as_ptr();
        unsafe {
            cpp!([self as "SparseMatrix*", triplets_ptr as "const Triplet*", triplets_len as "Index"]{
                return self->setFromTriplets(triplets_ptr, triplets_ptr + triplets_len);
            })
        }
    }
}

impl ops::Neg for &SparseMatrix {
    type Output = SparseMatrix;

    fn neg(self) -> Self::Output {
        unsafe {
            cpp!([self as "const SparseMatrix*"] -> SparseMatrix as "SparseMatrix" {
                return -(*self);
            })
        }
    }
}

impl ops::Add<&SparseMatrix> for &SparseMatrix {
    type Output = SparseMatrix;

    fn add(self, rhs: &SparseMatrix) -> SparseMatrix {
        unsafe {
            cpp!([self as "const SparseMatrix*", rhs as "const SparseMatrix*"] -> SparseMatrix as "SparseMatrix" {
                return *self + *rhs;
            })
        }
    }
}

impl ops::AddAssign<&SparseMatrix> for SparseMatrix {
    fn add_assign(&mut self, other: &SparseMatrix) {
        unsafe {
            cpp!([self as "SparseMatrix*", other as "const SparseMatrix*"] {
                *self += *other;
            })
        }
    }
}

impl ops::Sub<&SparseMatrix> for &SparseMatrix {
    type Output = SparseMatrix;

    fn sub(self, rhs: &SparseMatrix) -> SparseMatrix {
        unsafe {
            cpp!([self as "const SparseMatrix*", rhs as "const SparseMatrix*"] -> SparseMatrix as "SparseMatrix" {
                return *self - *rhs;
            })
        }
    }
}

impl ops::SubAssign<&SparseMatrix> for SparseMatrix {
    fn sub_assign(&mut self, other: &SparseMatrix) {
        unsafe {
            cpp!([self as "SparseMatrix*", other as "const SparseMatrix*"] {
                *self -= *other;
            })
        }
    }
}

impl ops::Mul<&SparseMatrix> for &SparseMatrix {
    type Output = SparseMatrix;

    fn mul(self, rhs: &SparseMatrix) -> SparseMatrix {
        unsafe {
            cpp!([self as "const SparseMatrix*", rhs as "const SparseMatrix*"] -> SparseMatrix as "SparseMatrix" {
                return *self * *rhs;
            })
        }
    }
}

impl ops::Mul<&VecX> for &SparseMatrix {
    type Output = VecX;

    fn mul(self, rhs: &VecX) -> VecX {
        unsafe {
            cpp!([self as "const SparseMatrix*", rhs as "const VecX*"] -> VecX as "VecX" {
                return *self * *rhs;
            })
        }
    }
}

impl ops::Mul<&MatX> for &SparseMatrix {
    type Output = MatX;

    fn mul(self, rhs: &MatX) -> MatX {
        unsafe {
            cpp!([self as "const SparseMatrix*", rhs as "const MatX*"] -> MatX as "MatX" {
                return *self * *rhs;
            })
        }
    }
}

impl ops::Mul<&SparseMatrix> for &MatX {
    type Output = MatX;

    fn mul(self, rhs: &SparseMatrix) -> MatX {
        unsafe {
            cpp!([self as "const MatX*", rhs as "const SparseMatrix*"] -> MatX as "MatX" {
                return *self * *rhs;
            })
        }
    }
}

impl ops::Mul<&SparseMatrix> for &VecX {
    type Output = MatX;

    fn mul(self, rhs: &SparseMatrix) -> MatX {
        unsafe {
            cpp!([self as "const VecX*", rhs as "const SparseMatrix*"] -> MatX as "MatX" {
                return *self * *rhs;
            })
        }
    }
}

impl ops::Mul<Scalar> for &SparseMatrix {
    type Output = SparseMatrix;

    fn mul(self, rhs: Scalar) -> SparseMatrix {
        unsafe {
            cpp!([self as "const SparseMatrix*", rhs as "Scalar"] -> SparseMatrix as "SparseMatrix" {
                return *self * rhs;
            })
        }
    }
}

impl ops::Mul<&SparseMatrix> for Scalar {
    type Output = SparseMatrix;

    fn mul(self, rhs: &SparseMatrix) -> SparseMatrix {
        unsafe {
            cpp!([self as "Scalar", rhs as "const SparseMatrix*"] -> SparseMatrix as "SparseMatrix" {
                return self * *rhs;
            })
        }
    }
}

impl ops::MulAssign<Scalar> for SparseMatrix {
    fn mul_assign(&mut self, rhs: Scalar) {
        unsafe {
            cpp!([self as "SparseMatrix*", rhs as "Scalar"] {
                *self *= rhs;
            })
        }
    }
}

impl ops::Div<Scalar> for &SparseMatrix {
    type Output = SparseMatrix;

    fn div(self, rhs: Scalar) -> SparseMatrix {
        unsafe {
            cpp!([self as "const SparseMatrix*", rhs as "Scalar"] -> SparseMatrix as "SparseMatrix" {
                return *self / rhs;
            })
        }
    }
}

impl ops::DivAssign<Scalar> for SparseMatrix {
    fn div_assign(&mut self, rhs: Scalar) {
        unsafe {
            cpp!([self as "SparseMatrix*", rhs as "Scalar"] {
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
    fn eigenmatx_data_mut() {
        // Due to column major storage, the sequence in memory is
        // 1, 3, 2, 4 compared to 1, 2, 3, 4 if it was row major
        {
            let mut mat = MatX::new_with_size(2, 2);
            let data = mat.data_mut();
            data[0] = 1.0;
            data[1] = 3.0;
            data[2] = 2.0;
            data[3] = 4.0;
            assert_eq!(mat.get(0, 0), 1.0);
            assert_eq!(mat.get(0, 1), 2.0);
            assert_eq!(mat.get(1, 0), 3.0);
            assert_eq!(mat.get(1, 1), 4.0);
        }
        {
            let mut mat = MatX::new_with_size(2, 2);
            mat.data_mut().swap_with_slice(&mut [1.0, 3.0, 2.0, 4.0]);
            assert_eq!(mat.get(0, 0), 1.0);
            assert_eq!(mat.get(0, 1), 2.0);
            assert_eq!(mat.get(1, 0), 3.0);
            assert_eq!(mat.get(1, 1), 4.0);
        }
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
    fn eigenmatx_vecx_mul() {
        let mut mat1 = MatX::new_with_size(2, 3);
        mat1.data_mut()
            .swap_with_slice(&mut [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let mut vec1 = VecX::new_with_size(3);
        vec1.data_mut().swap_with_slice(&mut [1.0, 2.0, 3.0]);
        assert_eq!((&mat1 * &vec1).data(), [22.0, 28.0]);

        let mut mat2 = MatX::new_with_size(1, 3);
        mat2.data_mut().swap_with_slice(&mut [1.0, 2.0, 3.0]);
        let mut vec2 = VecX::new_with_size(2);
        vec2.data_mut().swap_with_slice(&mut [1.0, 2.0]);
        assert_eq!((&vec2 * &mat2).data(), [1.0, 2.0, 2.0, 4.0, 3.0, 6.0]);
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

    #[test]
    fn eigenmatx_transpose() {
        // generate new matrix
        {
            let mut mat1 = MatX::new_with_size(1, 2);
            mat1.data_mut().swap_with_slice(&mut [1.0, 2.0]);
            let mat2 = mat1.transpose();
            assert_eq!(mat2.rows(), 2);
            assert_eq!(mat2.cols(), 1);
            assert_eq!(mat2.data(), [1.0, 2.0]);
        }
        // inplace
        {
            let mut mat1 = MatX::new_with_size(1, 2);
            mat1.data_mut().swap_with_slice(&mut [1.0, 2.0]);
            mat1.transpose_in_place();
            assert_eq!(mat1.rows(), 2);
            assert_eq!(mat1.cols(), 1);
            assert_eq!(mat1.data(), [1.0, 2.0]);
        }
    }

    #[test]
    fn eigenvecx_set_get() {
        let mut vec = VecX::new_with_size(3);
        vec.set(0, 1.0);
        vec.set(1, 2.0);
        vec.set(2, 3.0);
        assert_eq!(vec.get(0), 1.0);
        assert_eq!(vec.get(1), 2.0);
        assert_eq!(vec.get(2), 3.0);
    }

    #[test]
    fn eigenvecx_resize() {
        let mut vec = VecX::new();
        assert_eq!(vec.size(), 0);
        vec.resize(2);
        assert_eq!(vec.size(), 2);
    }

    #[test]
    fn eigenvecx_data() {
        let mut vec = VecX::new_with_size(3);
        vec.set(0, 1.0);
        vec.set(1, 2.0);
        vec.set(2, 3.0);
        assert_eq!(vec.data(), [1.0, 2.0, 3.0]);
    }

    #[test]
    fn eigenvecx_data_mut() {
        let mut vec = VecX::new_with_size(3);
        vec.data_mut().swap_with_slice(&mut [1.0, 2.0, 3.0]);
        assert_eq!(vec.get(0), 1.0);
        assert_eq!(vec.get(1), 2.0);
        assert_eq!(vec.get(2), 3.0);
    }

    #[test]
    fn eigenvecx_transpose() {
        let mut vec = VecX::new_with_size(3);
        vec.data_mut().swap_with_slice(&mut [1.0, 2.0, 3.0]);
        assert_eq!(vec.size(), 3);
        let vec_t = vec.transpose();
        assert_eq!(vec_t.size(), 3);
        assert_eq!(vec_t.rows(), 1);
        assert_eq!(vec_t.cols(), 3);
        assert_eq!(vec_t.data(), [1.0, 2.0, 3.0]);
    }

    #[test]
    fn eigenvecx_dot() {
        let mut vec1 = VecX::new_with_size(3);
        vec1.data_mut().swap_with_slice(&mut [1.0, 2.0, 3.0]);
        let mut vec2 = VecX::new_with_size(3);
        vec2.data_mut().swap_with_slice(&mut [3.0, 2.0, 1.0]);
        assert_eq!(vec1.dot(&vec2), 10.0);
    }

    #[test]
    fn eigenvecx_cross() {
        let mut vec1 = VecX::new_with_size(3);
        vec1.data_mut().swap_with_slice(&mut [1.0, 2.0, 3.0]);
        let mut vec2 = VecX::new_with_size(3);
        vec2.data_mut().swap_with_slice(&mut [3.0, 2.0, 1.0]);
        assert_eq!(vec1.cross(&vec2).data(), [-4.0, 8.0, -4.0]);
    }

    #[test]
    fn eigenvecx_add() {
        let mut vec1 = VecX::new_with_size(1);
        let mut vec2 = VecX::new_with_size(1);
        vec1.set(0, 2.0);
        vec2.set(0, 3.0);
        let vec3 = &vec1 + &vec2;
        assert_eq!(vec3.size(), 1);
        assert_eq!(vec3.get(0), 5.0);
    }

    #[test]
    fn eigenvecx_addassign() {
        let mut vec1 = VecX::new_with_size(1);
        let mut vec2 = VecX::new_with_size(1);
        vec1.set(0, 2.0);
        vec2.set(0, 3.0);
        vec1 += &vec2;
        assert_eq!(vec1.size(), 1);
        assert_eq!(vec1.get(0), 5.0);
    }

    #[test]
    fn eigenvecx_neg() {
        let mut vec1 = VecX::new_with_size(2);
        vec1.set(0, 2.0);
        vec1.set(1, -3.0);
        let vec2 = -&vec1;
        assert_eq!(vec2.size(), 2);
        assert_eq!(vec2.get(0), -2.0);
        assert_eq!(vec2.get(1), 3.0);
    }

    #[test]
    fn eigenvecx_sub() {
        let mut vec1 = VecX::new_with_size(1);
        let mut vec2 = VecX::new_with_size(1);
        vec1.set(0, 2.0);
        vec2.set(0, 3.0);
        let vec3 = &vec1 - &vec2;
        assert_eq!(vec3.size(), 1);
        assert_eq!(vec3.get(0), -1.0);
    }

    #[test]
    fn eigenvecx_subassign() {
        let mut vec1 = VecX::new_with_size(1);
        let mut vec2 = VecX::new_with_size(1);
        vec1.set(0, 2.0);
        vec2.set(0, 3.0);
        vec1 -= &vec2;
        assert_eq!(vec1.size(), 1);
        assert_eq!(vec1.get(0), -1.0);
    }

    #[test]
    fn eigenvecx_mul() {
        // VecX*Scalar
        {
            let mut vec1 = VecX::new_with_size(2);
            vec1.set(0, 2.0);
            vec1.set(1, 3.0);
            let scalar = 5.0;
            assert_eq!((&vec1 * scalar).data(), [10.0, 15.0]);
        }

        // Scalar*VecX
        {
            let mut vec1 = VecX::new_with_size(2);
            vec1.set(0, 2.0);
            vec1.set(1, 3.0);
            let scalar = 5.0;
            assert_eq!((scalar * &vec1).data(), [10.0, 15.0]);
        }
    }

    #[test]
    fn eigenvecx_mulassign() {
        let mut vec1 = VecX::new_with_size(2);
        vec1.set(0, 2.0);
        vec1.set(1, 3.0);
        vec1 *= 5.0;
        assert_eq!(vec1.data(), [10.0, 15.0]);
    }

    #[test]
    fn eigenvecx_div() {
        // VecX/Scalar
        let mut vec1 = VecX::new_with_size(2);
        vec1.set(0, 2.0);
        vec1.set(1, 3.0);
        let scalar = 5.0;
        assert_eq!((&vec1 / scalar).data(), [2.0 / 5.0, 3.0 / 5.0]);
    }

    #[test]
    fn eigenvecx_divassign() {
        let mut vec1 = VecX::new_with_size(2);
        vec1.set(0, 2.0);
        vec1.set(1, 3.0);
        vec1 /= 5.0;
        assert_eq!(vec1.data(), [2.0 / 5.0, 3.0 / 5.0]);
    }

    // TODO(ish): test all SparseMatrix functions
    // TODO(ish): test all SimplicialLLT functions
}
