use nalgebra_glm as glm;

use crate::eigen;
use eigen::{SimplicialLLT, SparseMatrix, VecX};

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

    impl Mesh {
        pub fn setup(&mut self) {
            for (_, node) in self.get_nodes_mut() {
                let extra_data = ClothNodeData { prev_pos: node.pos };
                node.extra_data = Some(extra_data);
            }
            let mut edge_data = Vec::new();
            for (_, edge) in self.get_edges().iter() {
                let edge_verts = edge.get_verts().unwrap();
                let vert_1 = self.get_vert(edge_verts.0).unwrap();
                let vert_2 = self.get_vert(edge_verts.1).unwrap();
                let node_1 = self.get_node(vert_1.get_node_index().unwrap()).unwrap();
                let node_2 = self.get_node(vert_2.get_node_index().unwrap()).unwrap();
                let len = glm::length(&(node_1.pos - node_2.pos));
                let extra_data = ClothEdgeData { rest_len: len };

                edge_data.push((extra_data, edge.get_self_index()));
            }
            for (extra_data, edge_index) in edge_data {
                let edge = self.get_edge_mut(edge_index).unwrap();
                edge.extra_data = Some(extra_data);
            }
        }
    }
}

pub struct Simulation {
    pub cloth: cloth::Mesh,
    cloth_mass: f64,
    mass_matrix: Option<SparseMatrix>,
    time_step: f64,
    d: VecX,
    prefactored_solver: SimplicialLLT,
    l: SparseMatrix,
    j: SparseMatrix,
    spring_stiffness: f64,
}

impl Simulation {
    pub fn new(cloth: cloth::Mesh, cloth_mass: f64, time_step: f64, spring_stiffness: f64) -> Self {
        return Self {
            cloth,
            cloth_mass,
            mass_matrix: None,
            time_step,
            d: VecX::new(),
            prefactored_solver: SimplicialLLT::new(),
            l: SparseMatrix::new(),
            j: SparseMatrix::new(),
            spring_stiffness,
        };
    }

    /// Panics if mass_matrix is not initialized
    fn get_mass_matrix(&self) -> &SparseMatrix {
        return &self.mass_matrix.as_ref().unwrap();
    }

    fn compute_mass_matrix(&mut self) {
        let num_nodes = self.cloth.get_nodes().len();
        let mass = self.cloth_mass / num_nodes as f64;
        self.mass_matrix = Some(SparseMatrix::new());
        let mass_matrix = self.mass_matrix.as_mut().unwrap();
        mass_matrix.resize(3 * num_nodes, 3 * num_nodes);

        let mut triplets = Vec::new();
        for i in 0..(3 * num_nodes) {
            triplets.push(eigen::Triplet::new(i, i, mass));
        }

        mass_matrix.set_from_triplets(&triplets);
    }

    fn get_l(&self) -> &SparseMatrix {
        return &self.l;
    }

    fn get_j(&self) -> &SparseMatrix {
        return &self.j;
    }

    fn get_y(&self) -> VecX {
        let mut y = VecX::new_with_size(3 * self.cloth.get_nodes().len());
        for (i, (_, node)) in self.cloth.get_nodes().iter().enumerate() {
            // y = 2*q_n - q_(n-1)
            let y_i = &(2.0 * &node.pos) - &node.extra_data.as_ref().unwrap().prev_pos;
            y.set_v3_glm(i, &y_i);
        }

        return y;
    }

    /// Gives f_ext, the external force on each node of the mesh
    fn get_external_forces(&self) -> VecX {
        let mut f = VecX::new_with_size(3 * self.cloth.get_nodes().len());
        for i in 0..self.cloth.get_nodes().len() {
            // Currently limited to gravity
            f.set_v3_glm(i, &glm::make_vec3(&[0.0, -9.8, 0.0]));
        }
        return self.get_mass_matrix() * &f;
    }

    fn compute_l(&mut self) {
        // assert_eq!(
        //     self.cloth.get_nodes().capacity(),
        //     self.cloth.get_nodes().len()
        // ); // TODO(ish): make sure that there isn't an element within nodes that isn't assigned to a value
        let num_nodes = self.cloth.get_nodes().len();
        self.l.resize(3 * num_nodes, 3 * num_nodes);

        let mut l_triplets = Vec::new();

        for (_, edge) in self.cloth.get_edges().iter() {
            let edge_verts = edge.get_verts().unwrap();
            let vert_1 = self.cloth.get_vert(edge_verts.0).unwrap();
            let vert_2 = self.cloth.get_vert(edge_verts.1).unwrap();
            let node_1_index = generational_arena::Index::from(vert_1.get_node_index().unwrap())
                .into_raw_parts()
                .0;
            let node_2_index = generational_arena::Index::from(vert_2.get_node_index().unwrap())
                .into_raw_parts()
                .0;

            triplet_3_push(
                &mut l_triplets,
                node_1_index,
                node_1_index,
                self.spring_stiffness,
            );

            triplet_3_push(
                &mut l_triplets,
                node_2_index,
                node_2_index,
                self.spring_stiffness,
            );

            triplet_3_push(
                &mut l_triplets,
                node_1_index,
                node_2_index,
                -self.spring_stiffness,
            );

            triplet_3_push(
                &mut l_triplets,
                node_2_index,
                node_1_index,
                -self.spring_stiffness,
            );
        }

        self.l.set_from_triplets(&l_triplets);
    }

    fn compute_j(&mut self) {
        // assert_eq!(
        //     self.cloth.get_edges().capacity(),
        //     self.cloth.get_edges().len()
        // );
        // assert_eq!(
        //     self.cloth.get_nodes().capacity(),
        //     self.cloth.get_nodes().len()
        // ); // TODO(ish): make sure that there isn't an element within nodes that isn't assigned to a value
        let num_nodes = self.cloth.get_nodes().len();
        let num_edges = self.cloth.get_edges().len();
        self.j.resize(3 * num_nodes, 3 * num_edges);

        let mut j_triplets = Vec::new();

        for (edge_index, edge) in self.cloth.get_edges().iter() {
            let edge_index = generational_arena::Index::from(edge_index)
                .into_raw_parts()
                .0;
            let edge_verts = edge.get_verts().unwrap();
            let vert_1 = self.cloth.get_vert(edge_verts.0).unwrap();
            let vert_2 = self.cloth.get_vert(edge_verts.1).unwrap();
            let node_1_index = generational_arena::Index::from(vert_1.get_node_index().unwrap())
                .into_raw_parts()
                .0;
            let node_2_index = generational_arena::Index::from(vert_2.get_node_index().unwrap())
                .into_raw_parts()
                .0;

            triplet_3_push(
                &mut j_triplets,
                node_1_index,
                edge_index,
                self.spring_stiffness,
            );

            triplet_3_push(
                &mut j_triplets,
                node_2_index,
                edge_index,
                -self.spring_stiffness,
            );
        }

        self.j.set_from_triplets(&j_triplets);
    }

    /// Gets the system matrix (M + h*h*L) and prefactorizes the solver with it
    /// Also precomputes L and J matrices
    fn prefactorize_and_precompute(&mut self) {
        // Precompute
        self.compute_l();
        self.compute_j();

        // Prefactorize
        // TODO(ish): Might have to add some regularization component
        let system_matrix =
            self.get_mass_matrix() + &(self.time_step * self.time_step * self.get_l());
        self.prefactored_solver.analyze_pattern(&system_matrix);
        self.prefactored_solver.factorize(&system_matrix);
    }

    fn get_prefactored_system_matrix_solver(&self) -> &SimplicialLLT {
        return &self.prefactored_solver;
    }

    /// Returns the rhs for the global step solve
    fn get_rhs(&self) -> VecX {
        let m = self.get_mass_matrix();
        let y = &self.get_y();
        let h = &self.time_step;
        let j = self.get_j();
        let d = &self.d;
        let f_ext = &self.get_external_forces();

        // M*y + h*h*J*d + h*h*f_ext
        return &(m * y) + &(h * h * &(&(j * d) + f_ext));
    }

    /// Solves for the vector d from the paper
    /// Currently supports only direct edges as springs, will need to add better contraints later
    fn solve_d(&mut self) {
        self.d.resize(3 * self.get_num_springs());
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
        // TODO(ish): this can be optimized if the mesh structure
        // doesn't change between previous step and current step
        self.compute_mass_matrix();

        // TODO(ish): this can be optimized, instead of transforming
        // to y and then back, can do it in place with on single
        // iteration
        self.set_cloth_info_from_x(&self.get_y());

        // TODO(ish): might be able to optimize by running
        // prefactorize_and_precompute only if the system configuration
        // has changed, eg: change in mesh connectivity
        self.prefactorize_and_precompute();
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
        let x = self.get_prefactored_system_matrix_solver().solve(&rhs);
        self.set_cloth_info_from_x(&x);
    }

    /// Store the solved position data into the cloth nodes
    fn set_cloth_info_from_x(&mut self, x: &VecX) {
        // Set the previous pos to current pos and set current pos to x
        assert_eq!(x.size(), 3 * self.cloth.get_nodes().len());
        for (i, (_, node)) in self.cloth.get_nodes_mut().iter_mut().enumerate() {
            if i == 0 {
                continue;
            }
            let pos = x.get_v3_glm(i);
            node.extra_data.as_mut().unwrap().prev_pos = node.pos;
            node.pos = pos;
        }
    }

    fn get_num_springs(&self) -> usize {
        return self.cloth.get_edges().len();
    }
}

impl VecX {
    #[inline]
    fn set_v3_glm(&mut self, index: usize, val: &glm::DVec3) {
        let data = self.data_mut();
        data[3 * index + 0] = val[0];
        data[3 * index + 1] = val[1];
        data[3 * index + 2] = val[2];
    }

    #[inline]
    fn get_v3_glm(&self, index: usize) -> glm::DVec3 {
        let data = self.data();
        return glm::vec3(
            data[3 * index + 0],
            data[3 * index + 1],
            data[3 * index + 2],
        );
    }
}

fn triplet_3_push(triplets: &mut Vec<eigen::Triplet>, i1: usize, i2: usize, value: f64) {
    triplets.push(eigen::Triplet::new(3 * i1 + 0, 3 * i2 + 0, value));
    triplets.push(eigen::Triplet::new(3 * i1 + 1, 3 * i2 + 1, value));
    triplets.push(eigen::Triplet::new(3 * i1 + 2, 3 * i2 + 2, value));
}
