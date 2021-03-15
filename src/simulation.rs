use nalgebra_glm as glm;

use crate::eigen::{SimplicialLLT, SparseMatrix, VecX};

pub mod cloth {
    use super::*;
    use crate::mesh;

    pub struct ClothNodeData {
        pub prev_pos: glm::DVec3,
    }

    type ClothVertData = ();

    pub struct ClothEdgeData {
        pub rest_len: f64,
    }

    type ClothFaceData = ();

    pub type Node = mesh::Node<ClothNodeData>;
    pub type Vert = mesh::Vert<ClothVertData>;
    pub type Edge = mesh::Edge<ClothEdgeData>;
    pub type Face = mesh::Face<ClothFaceData>;
    pub type Mesh = mesh::Mesh<ClothNodeData, ClothVertData, ClothEdgeData, ClothFaceData>;
}

pub struct Simulation {
    cloth: cloth::Mesh,
    mass_matrix: SparseMatrix,
    time_step: f64,
    d: VecX,
    prefactored_mat: SimplicialLLT,
}

impl Simulation {
    pub fn new(cloth: cloth::Mesh, time_step: f64) -> Self {
        // TODO(ish): initialize mass_matrix, d appropriately
        return Self {
            cloth,
            mass_matrix: SparseMatrix::new(),
            time_step,
            d: VecX::new(),
            prefactored_mat: SimplicialLLT::new(),
        };
    }

    /// TODO(ish)
    fn get_l(&self) -> SparseMatrix {
        return SparseMatrix::new();
    }

    /// TODO(ish)
    fn get_j(&self) -> SparseMatrix {
        return SparseMatrix::new();
    }

    /// TODO(ish)
    fn get_external_forces(&self) -> VecX {
        return VecX::new();
    }

    /// TODO(ish)
    fn get_prefactored_system_matrix(&self) -> &SimplicialLLT {
        return &self.prefactored_mat;
    }

    /// TODO(ish)
    fn get_rhs(&self) -> VecX {
        return VecX::new();
    }

    /// Solves for the vector d from the paper
    /// Currently supports only direct edges as springs, will need to add better contraints later
    fn solve_d(&mut self) {
        assert_eq!(self.d.size(), 3 * self.get_num_springs());
        for (i, (_, edge)) in self.cloth.get_edges().iter().enumerate() {
            let edge_verts = edge.get_verts().unwrap();
            let vert_1 = self.cloth.get_vert(edge_verts.0).unwrap();
            let vert_2 = self.cloth.get_vert(edge_verts.1).unwrap();
            let node_1 = self
                .cloth
                .get_node(vert_1.get_node_index().unwrap())
                .unwrap();
            let node_2 = self
                .cloth
                .get_node(vert_2.get_node_index().unwrap())
                .unwrap();
            let p_diff = node_1.pos - node_2.pos;
            let res = edge.extra_data.as_ref().unwrap().rest_len * (p_diff) / glm::length(&p_diff);

            self.d.set_v3_glm(i, &res);
        }
    }

    pub fn next_step(&mut self, num_iterations: usize) {
        // TODO(ish): set initial guess (y)
        for _ in 0..num_iterations {
            self.solver_local_step();
            self.solver_global_step();
        }
    }

    fn solver_local_step(&mut self) {
        self.solve_d();
    }

    fn solver_global_step(&mut self) {
        let rhs = self.get_rhs();
        let x = self.get_prefactored_system_matrix().solve(&rhs);
        self.set_cloth_info_from_x(&x);
    }

    /// TODO(ish)
    fn set_cloth_info_from_x(&mut self, x: &VecX) {}

    fn get_num_springs(&self) -> usize {
        return self.cloth.get_edges().len();
    }
}

impl VecX {
    #[inline]
    fn set_v3_glm(&mut self, index: usize, val: &glm::DVec3) {
        let data = self.data_mut();
        data[index + 0] = val[0];
        data[index + 1] = val[1];
        data[index + 2] = val[2];
    }
}
