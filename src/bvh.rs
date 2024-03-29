use generational_arena::{Arena, Index};
use nalgebra_glm as glm;
use rand::{Rng, SeedableRng};

use crate::drawable::Drawable;
use crate::gpu_immediate::*;
use crate::shader::Shader;

const MAX_TREETYPE: u8 = 32;

lazy_static! {
    static ref BVHTREE_KDOP_AXES: Vec<glm::TVec3<Scalar>> = {
        let v = vec![
            glm::vec3(1.0, 0.0, 0.0),
            glm::vec3(0.0, 1.0, 0.0),
            glm::vec3(0.0, 0.0, 1.0),
            glm::vec3(1.0, 1.0, 1.0),
            glm::vec3(1.0, -1.0, 1.0),
            glm::vec3(1.0, 1.0, -1.0),
            glm::vec3(1.0, -1.0, -1.0),
            glm::vec3(1.0, 1.0, 0.0),
            glm::vec3(1.0, 0.0, 1.0),
            glm::vec3(0.0, 1.0, 1.0),
            glm::vec3(1.0, -1.0, 0.0),
            glm::vec3(1.0, 0.0, -1.0),
            glm::vec3(0.0, 1.0, -1.0),
        ];
        assert_eq!(v.len(), 13);
        v
    };
}

type Scalar = f64;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct BVHNodeIndex(pub Index);

impl BVHNodeIndex {
    fn unknown() -> Self {
        Self(Index::from_raw_parts(usize::MAX, u64::MAX))
    }
}

struct BVHNode<T>
where
    T: Copy,
{
    children: Vec<BVHNodeIndex>,  // Indices of the child nodes
    parent: Option<BVHNodeIndex>, // Parent index

    bv: Vec<Scalar>,       // Bounding volume axis data
    elem_index: Option<T>, // Index of element stored in the node
    totnode: u8,           // How many nodes are used, used for speedup
    main_axis: u8,         // Axis used to split this node
}

impl<T> BVHNode<T>
where
    T: Copy,
{
    fn new() -> Self {
        Self {
            children: Vec::new(),
            parent: None,

            bv: Vec::new(),
            elem_index: None,
            totnode: 0,
            main_axis: 0,
        }
    }

    fn min_max_init(&mut self, start_axis: u8, stop_axis: u8) {
        let bv = &mut self.bv;
        for axis_iter in start_axis..stop_axis {
            bv[(2 * axis_iter) as usize] = Scalar::MAX;
            bv[((2 * axis_iter) + 1) as usize] = -Scalar::MAX;
        }
    }

    fn create_kdop_hull(
        &mut self,
        start_axis: u8,
        stop_axis: u8,
        co_many: Vec<glm::TVec3<Scalar>>,
        moving: bool,
    ) {
        if !moving {
            self.min_max_init(start_axis, stop_axis);
        }
        let bv = &mut self.bv;

        assert_eq!(bv.len(), (stop_axis * 2) as usize);
        for co in co_many {
            for axis_iter in start_axis..stop_axis {
                let axis_iter = axis_iter as usize;
                let new_min_max = glm::dot(&co, &BVHTREE_KDOP_AXES[axis_iter]);
                if new_min_max < bv[2 * axis_iter] {
                    bv[2 * axis_iter] = new_min_max;
                }
                if new_min_max > bv[(2 * axis_iter) + 1] {
                    bv[(2 * axis_iter) + 1] = new_min_max;
                }
            }
        }
    }

    fn overlap_test(&self, other: &BVHNode<T>, start_axis: u8, stop_axis: u8) -> bool {
        let bv1 = &self.bv;
        let bv2 = &other.bv;
        for axis_iter in start_axis..stop_axis {
            let axis_iter = axis_iter as usize;
            if bv1[(2 * axis_iter)] > bv2[(2 * axis_iter) + 1]
                || bv2[(2 * axis_iter)] > bv1[(2 * axis_iter) + 1]
            {
                return false;
            }
        }

        true
    }
}

#[derive(Debug)]
pub enum BVHError {
    IndexOutOfRange,
    DifferentNumPoints,
}

impl std::fmt::Display for BVHError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BVHError::IndexOutOfRange => write!(f, "Index given is out of range"),
            BVHError::DifferentNumPoints => write!(f, "Different number of points given"),
        }
    }
}

impl std::error::Error for BVHError {}

pub struct BVHTree<T>
where
    T: Copy,
{
    nodes: Vec<BVHNodeIndex>,
    node_array: Arena<BVHNode<T>>, // Where the actual nodes are stored

    epsilon: Scalar, // Epsilon for inflation of the kdop
    totleaf: usize,
    totbranch: usize,
    start_axis: u8,
    stop_axis: u8,
    axis: u8,      // kdop type (6 => OBB, 8 => AABB, etc.)
    tree_type: u8, // Type of tree (4 => QuadTree, etc.)
}

struct BVHBuildHelper {
    totleafs: usize,
    leafs_per_child: [usize; 32], // Min number of leafs that are archievable from a node at depth N
    branches_on_level: [usize; 32], // Number of nodes at depth N (tree_type^N)
    remain_leafs: usize, // Number of leafs that are placed on the level that is not 100% filled
}

impl BVHBuildHelper {
    fn new(
        totleafs: usize,
        leafs_per_child: [usize; 32],
        branches_on_level: [usize; 32],
        remain_leafs: usize,
    ) -> Self {
        Self {
            totleafs,
            leafs_per_child,
            branches_on_level,
            remain_leafs,
        }
    }

    /// Return the min index of all the leafs achievable with the given branch
    fn implicit_leafs_index(&self, depth: usize, child_index: usize) -> usize {
        let min_leaf_index = child_index * self.leafs_per_child[depth - 1];
        if min_leaf_index <= self.remain_leafs {
            min_leaf_index
        } else if self.leafs_per_child[depth] != 0 {
            self.totleafs
                - (self.branches_on_level[depth - 1] - child_index) * self.leafs_per_child[depth]
        } else {
            self.remain_leafs
        }
    }
}

struct BVHDivNodesData<'a> {
    brances_array_start: usize,
    tree_offset: isize,
    data: &'a BVHBuildHelper,
    depth: usize,
    i: usize,
    first_of_next_level: usize,
}

impl<'a> BVHDivNodesData<'a> {
    fn new(
        brances_array_start: usize,
        tree_offset: isize,
        data: &'a BVHBuildHelper,
        depth: usize,
        i: usize,
        first_of_next_level: usize,
    ) -> Self {
        Self {
            brances_array_start,
            tree_offset,
            data,
            depth,
            i,
            first_of_next_level,
        }
    }
}

pub struct BVHTreeOverlap<T>
where
    T: Copy,
{
    pub index_1: T,
    pub index_2: T,
}

impl<T> BVHTreeOverlap<T>
where
    T: Copy,
{
    fn new(index_1: T, index_2: T) -> Self {
        Self { index_1, index_2 }
    }
}

impl<T> BVHTree<T>
where
    T: Copy,
{
    pub fn new(max_size: usize, epsilon: Scalar, tree_type: u8, axis: u8) -> Self {
        assert!(
            (2..=MAX_TREETYPE).contains(&tree_type),
            "tree_type must be >= 2 and <= {}",
            MAX_TREETYPE
        );

        // epsilon must be >= Scalar::EPSILON so that tangent rays can still hit a bounding volume
        let epsilon = epsilon.max(Scalar::EPSILON);

        let start_axis;
        let stop_axis;
        if axis == 26 {
            start_axis = 0;
            stop_axis = 13;
        } else if axis == 18 {
            start_axis = 7;
            stop_axis = 13;
        } else if axis == 14 {
            start_axis = 0;
            stop_axis = 7;
        } else if axis == 8 {
            // AABB
            start_axis = 0;
            stop_axis = 4;
        } else if axis == 6 {
            // OBB
            start_axis = 0;
            stop_axis = 3;
        } else {
            panic!("axis shouldn't be any other value");
        }

        let numnodes =
            max_size + implicit_needed_branches(tree_type, max_size) + tree_type as usize;
        let mut nodes = Vec::with_capacity(numnodes);
        nodes.resize(numnodes, BVHNodeIndex::unknown());
        let mut node_array = Arena::with_capacity(numnodes);

        for _ in 0..numnodes {
            node_array.insert(BVHNode::new());
        }

        for i in 0..numnodes {
            let node = node_array.get_unknown_gen_mut(i).unwrap().0;
            node.bv.resize(axis.into(), 0.0);
            node.children
                .resize(tree_type.into(), BVHNodeIndex::unknown());
        }

        Self {
            nodes,
            node_array,

            epsilon,
            totleaf: 0,
            totbranch: 0,
            start_axis,
            stop_axis,
            axis,
            tree_type,
        }
    }

    pub fn insert(&mut self, index: T, co_many: Vec<glm::TVec3<Scalar>>) {
        assert!(self.totbranch == 0);

        self.nodes[self.totleaf] = BVHNodeIndex(self.node_array.get_unknown_index(self.totleaf));
        let node = self.node_array.get_unknown_mut(self.totleaf);

        self.totleaf += 1;

        node.create_kdop_hull(self.start_axis, self.stop_axis, co_many, false);
        node.elem_index = Some(index);

        // Inflate bv by epsilon
        for axis_iter in self.start_axis..self.stop_axis {
            let axis_iter = axis_iter as usize;
            node.bv[(2 * axis_iter)] -= self.epsilon; // min
            node.bv[(2 * axis_iter) + 1] += self.epsilon; // max
        }
    }

    fn refit_kdop_hull(&mut self, node_index: BVHNodeIndex, start: usize, end: usize) {
        {
            let node = self.node_array.get_mut(node_index.0).unwrap();
            node.min_max_init(self.start_axis, self.stop_axis);
        }

        for j in start..end {
            let (node, node_2) = self.node_array.get2_mut(node_index.0, self.nodes[j].0);
            let node = node.unwrap();
            let bv = &mut node.bv;
            let node_bv = &mut node_2.unwrap().bv;

            for axis_iter in self.start_axis..self.stop_axis {
                let axis_iter = axis_iter as usize;

                let new_min = node_bv[(2 * axis_iter)];
                if new_min < bv[(2 * axis_iter)] {
                    bv[(2 * axis_iter)] = new_min;
                }

                let new_max = node_bv[(2 * axis_iter) + 1];
                if new_max > bv[(2 * axis_iter) + 1] {
                    bv[(2 * axis_iter) + 1] = new_max;
                }
            }
        }
    }

    fn build_implicit_helper(&self) -> BVHBuildHelper {
        let totleafs = self.totleaf;
        let tree_type = self.tree_type as usize;

        // calculate smallest tree_type^n such that tree_type^n >= self.num_leafs
        let mut leafs_per_child: [usize; 32] = [0; 32];
        leafs_per_child[0] = 1;
        while leafs_per_child[0] < totleafs {
            leafs_per_child[0] *= tree_type;
        }

        let mut branches_on_level: [usize; 32] = [0; 32];
        let mut depth = 1;
        branches_on_level[0] = 1;
        while depth < 32 && (leafs_per_child[depth - 1] != 0) {
            branches_on_level[depth] = branches_on_level[depth - 1] * tree_type;
            leafs_per_child[depth] = leafs_per_child[depth - 1] / tree_type;
            depth += 1;
        }

        let remain = totleafs - leafs_per_child[1];
        let nnodes = (remain + tree_type - 2) / (tree_type - 1);
        let remain_leafs = remain + nnodes;

        BVHBuildHelper::new(totleafs, leafs_per_child, branches_on_level, remain_leafs)
    }

    fn bvh_insertion_sort(&mut self, lo: usize, hi: usize, axis: usize) {
        for i in lo..hi {
            let mut j = i;
            let node_t_index = self.nodes[i];
            let node_t = self.node_array.get(node_t_index.0).unwrap();
            if j != lo {
                let mut node_j_minus_one = self.node_array.get(self.nodes[j - 1].0).unwrap();
                while (j != lo) && (node_t.bv[axis] < node_j_minus_one.bv[axis]) {
                    self.nodes[j] = self.nodes[j - 1];
                    j -= 1;
                    node_j_minus_one = self.node_array.get(self.nodes[j - 1].0).unwrap();
                }
            }
            self.nodes[j] = node_t_index;
        }
    }

    fn bvh_partition(
        &mut self,
        lo: usize,
        hi: usize,
        node_x_index: BVHNodeIndex,
        axis: usize,
    ) -> usize {
        let mut i = lo;
        let mut j = hi;
        let node_x = self.node_array.get(node_x_index.0).unwrap();
        loop {
            let mut node_a_i = self.node_array.get(self.nodes[i].0).unwrap();
            while node_a_i.bv[axis] < node_x.bv[axis] {
                i += 1;
                node_a_i = self.node_array.get(self.nodes[i].0).unwrap();
            }

            j -= 1;
            let mut node_a_j = self.node_array.get(self.nodes[j].0).unwrap();
            while node_x.bv[axis] < node_a_j.bv[axis] {
                j -= 1;
                node_a_j = self.node_array.get(self.nodes[j].0).unwrap();
            }

            if i >= j {
                return i;
            }

            self.nodes.swap(i, j);

            i += 1;
        }
    }

    fn bvh_median_of_3(&self, lo: usize, mid: usize, hi: usize, axis: usize) -> BVHNodeIndex {
        let node_lo = self.node_array.get(self.nodes[lo].0).unwrap();
        let node_mid = self.node_array.get(self.nodes[mid].0).unwrap();
        let node_hi = self.node_array.get(self.nodes[hi].0).unwrap();

        if node_mid.bv[axis] < node_lo.bv[axis] {
            if node_hi.bv[axis] < node_mid.bv[axis] {
                self.nodes[mid]
            } else if node_hi.bv[axis] < node_lo.bv[axis] {
                self.nodes[hi]
            } else {
                self.nodes[lo]
            }
        } else if node_hi.bv[axis] < node_mid.bv[axis] {
            if node_hi.bv[axis] < node_lo.bv[axis] {
                self.nodes[lo]
            } else {
                self.nodes[hi]
            }
        } else {
            self.nodes[mid]
        }
    }

    fn partition_nth_element(&mut self, mut begin: usize, mut end: usize, n: usize, axis: usize) {
        while (end - begin) > 3 {
            let cut = self.bvh_partition(
                begin,
                end,
                self.bvh_median_of_3(begin, (begin + end) / 2, end - 1, axis),
                axis,
            );

            if cut <= n {
                begin = cut;
            } else {
                end = cut;
            }
        }

        self.bvh_insertion_sort(begin, end, axis);
    }

    fn split_leafs(&mut self, nth: &[usize], partitions: usize, split_axis: usize) {
        for i in 0..(partitions - 1) {
            if nth[i] >= nth[partitions] {
                break;
            }

            self.partition_nth_element(nth[i], nth[partitions], nth[i + 1], split_axis);
        }
    }

    fn non_recursive_bvh_div_nodes_task_cb(&mut self, data: &BVHDivNodesData, j: usize) {
        let parent_level_index = j - data.i;

        let mut nth_positions: [usize; (MAX_TREETYPE + 1) as usize] =
            [0; (MAX_TREETYPE + 1) as usize];

        let parent_leafs_begin = data
            .data
            .implicit_leafs_index(data.depth, parent_level_index);
        let parent_leafs_end = data
            .data
            .implicit_leafs_index(data.depth, parent_level_index + 1);

        let parent_index = BVHNodeIndex(
            self.node_array
                .get_unknown_index(data.brances_array_start + j),
        );

        // calculate the bounding box of this branch and chooses the
        // longest axis as the axis to divide the leaves
        self.refit_kdop_hull(parent_index, parent_leafs_begin, parent_leafs_end);
        let parent = self.node_array.get_mut(parent_index.0).unwrap();
        let split_axis = get_largest_axis(&parent.bv);

        // Save split axis (this can be used on raytracing to speedup the query time)
        parent.main_axis = split_axis / 2;

        // Split the childs along the split_axis, note: its not needed
        // to sort the whole leafs array.
        // Only to assure that the elements are partitioned on a way
        // that each child takes the elements it would take in case
        // the whole array was sorted.
        // Split_leafs takes care of that "sort" problem.
        nth_positions[0] = parent_leafs_begin;
        nth_positions[self.tree_type as usize] = parent_leafs_end;
        for k in 1..self.tree_type {
            let k = k as usize;
            let child_index =
                ((j * self.tree_type as usize) as isize + data.tree_offset + k as isize) as usize;
            let child_level_index = child_index - data.first_of_next_level;
            nth_positions[k] = data
                .data
                .implicit_leafs_index(data.depth + 1, child_level_index);
        }

        self.split_leafs(&nth_positions, self.tree_type.into(), split_axis.into());

        // setup children and totnode counters
        let mut totnode = 0;
        for k in 0..self.tree_type {
            let k = k as usize;
            let child_index =
                ((j * self.tree_type as usize) as isize + data.tree_offset + k as isize) as usize;
            let child_level_index = child_index - data.first_of_next_level;

            let child_leafs_begin = data
                .data
                .implicit_leafs_index(data.depth + 1, child_level_index);
            let child_leafs_end = data
                .data
                .implicit_leafs_index(data.depth + 1, child_level_index + 1);

            #[allow(clippy::comparison_chain)]
            if child_leafs_end - child_leafs_begin > 1 {
                let child_index = BVHNodeIndex(
                    self.node_array
                        .get_unknown_index(data.brances_array_start + child_index),
                );
                let parent = self.node_array.get_mut(parent_index.0).unwrap();
                parent.children[k] = child_index;
                let child = self.node_array.get_mut(child_index.0).unwrap();
                child.parent = Some(parent_index);
            } else if child_leafs_end - child_leafs_begin == 1 {
                let child_index = self.nodes[child_leafs_begin];
                let parent = self.node_array.get_mut(parent_index.0).unwrap();
                parent.children[k] = child_index;
                let child = self.node_array.get_mut(child_index.0).unwrap();
                child.parent = Some(parent_index);
            } else {
                break;
            }
            totnode += 1;
        }

        let parent = self.node_array.get_mut(parent_index.0).unwrap();
        parent.totnode = totnode;
    }

    fn non_recursive_bvh_div_nodes(&mut self, branches_array_start: usize, num_leafs: usize) {
        let tree_type = self.tree_type;
        let tree_offset: isize = 2 - tree_type as isize;
        let num_branches = implicit_needed_branches(tree_type, num_leafs);

        if num_leafs == 1 {
            let root_index =
                BVHNodeIndex(self.node_array.get_unknown_index(branches_array_start + 1)); // TODO(ish): verify this
            self.refit_kdop_hull(root_index, 0, num_leafs);

            let root = self.node_array.get_mut(root_index.0).unwrap();
            root.main_axis = get_largest_axis(&root.bv) / 2;
            root.totnode = 1;
            root.children[0] = self.nodes[0];
            let root_child_index = root.children[0];
            let child = self.node_array.get_mut(root_child_index.0).unwrap();
            child.parent = Some(root_index);
            return;
        }

        let data = self.build_implicit_helper();

        let mut cb_data = BVHDivNodesData::new(branches_array_start, tree_offset, &data, 0, 0, 0);

        // loop tree levels, (log N) loops
        let mut i = 1;
        let mut depth = 1;
        while i <= num_branches {
            let first_of_next_level: usize =
                ((i as isize * tree_type as isize) + tree_offset) as usize;
            // index of last branch on this level
            let i_stop = first_of_next_level.min(num_branches + 1);

            // Loop all branches on this level
            cb_data.first_of_next_level = first_of_next_level;
            cb_data.i = i;
            cb_data.depth = depth;

            // TODO(ish): make this parallel, refer to Blender's code
            for i_task in i..i_stop {
                self.non_recursive_bvh_div_nodes_task_cb(&cb_data, i_task);
            }

            i = first_of_next_level;
            depth += 1;
        }
    }

    /// Call balance() after inserting the nodes using insert()
    /// This function should be called only once
    pub fn balance(&mut self) {
        assert_eq!(self.totbranch, 0);

        self.non_recursive_bvh_div_nodes(self.totleaf - 1, self.totleaf);

        self.totbranch = implicit_needed_branches(self.tree_type, self.totleaf);
        for i in 0..self.totbranch {
            self.nodes[self.totleaf + i] =
                BVHNodeIndex(self.node_array.get_unknown_index(self.totleaf + i));
        }
    }

    pub fn update_node(
        &mut self,
        node_index: usize,
        co_many: Vec<glm::TVec3<Scalar>>,
        co_moving_many: Vec<glm::TVec3<Scalar>>,
    ) -> Result<(), BVHError> {
        if node_index > self.totleaf {
            return Err(BVHError::IndexOutOfRange);
        }
        if !co_moving_many.is_empty() && co_many.len() != co_moving_many.len() {
            return Err(BVHError::DifferentNumPoints);
        }

        let node = self.node_array.get_unknown_mut(node_index);

        node.create_kdop_hull(self.start_axis, self.stop_axis, co_many, false);

        if !co_moving_many.is_empty() {
            node.create_kdop_hull(self.start_axis, self.stop_axis, co_moving_many, true);
        }

        // Inflate bv by epsilon
        for axis_iter in self.start_axis..self.stop_axis {
            let axis_iter = axis_iter as usize;
            node.bv[(2 * axis_iter)] -= self.epsilon; // min
            node.bv[(2 * axis_iter) + 1] += self.epsilon; // max
        }

        Ok(())
    }

    fn node_join(&mut self, nodes_index: usize) {
        let node_index = self.nodes[nodes_index];
        {
            let node = self.node_array.get_mut(node_index.0).unwrap();
            node.min_max_init(self.start_axis, self.stop_axis);
        }

        for i in 0..self.tree_type {
            let i = i as usize;
            let node = self.node_array.get(node_index.0).unwrap();
            let child_index = node.children[i];
            let (node, child) = self.node_array.get2_mut(node_index.0, child_index.0);
            if let Some(child) = child {
                let node = node.unwrap();
                for axis_iter in self.start_axis..self.stop_axis {
                    let axis_iter = axis_iter as usize;
                    // update minimum
                    if child.bv[(2 * axis_iter)] < node.bv[(2 * axis_iter)] {
                        node.bv[(2 * axis_iter)] = child.bv[(2 * axis_iter)];
                    }
                    // update maximum
                    if child.bv[(2 * axis_iter) + 1] > node.bv[(2 * axis_iter) + 1] {
                        node.bv[(2 * axis_iter) + 1] = child.bv[(2 * axis_iter) + 1];
                    }
                }
            } else {
                break;
            }
        }
    }

    pub fn update_tree(&mut self) {
        let root_start = self.totleaf;
        let mut index = self.totleaf + self.totbranch - 1;

        while index >= root_start {
            self.node_join(index);
            index -= 1;
        }
    }

    fn overlap_thread_num(&self) -> usize {
        let node = self.node_array.get(self.nodes[self.totleaf].0).unwrap();
        self.tree_type.min(node.totnode).into()
    }

    #[allow(clippy::too_many_arguments)]
    fn overlap_traverse_callback<F>(
        &self,
        other: &BVHTree<T>,
        node_1_index: BVHNodeIndex,
        node_2_index: BVHNodeIndex,
        start_axis: u8,
        stop_axis: u8,
        callback: &F,
        r_overlap_pairs: &mut Vec<BVHTreeOverlap<T>>,
    ) where
        F: Fn(T, T) -> bool,
    {
        let node_1 = self.node_array.get(node_1_index.0).unwrap();
        let node_2 = other.node_array.get(node_2_index.0).unwrap();
        if node_1.overlap_test(node_2, start_axis, stop_axis) {
            // check if node_1 is a leaf node
            if node_1.totnode == 0 {
                // check if node_2 is a leaf node
                if node_2.totnode == 0 {
                    // the two nodes if equal, all the children will
                    // also match. This happens when overlap between
                    // the same tree is checked for.
                    if node_1_index == node_2_index {
                        return;
                    }

                    // Only difference to BVHTree::overlap_traverse
                    if callback(node_1.elem_index.unwrap(), node_2.elem_index.unwrap()) {
                        let overlap = BVHTreeOverlap::new(
                            node_1.elem_index.unwrap(),
                            node_2.elem_index.unwrap(),
                        );
                        r_overlap_pairs.push(overlap);
                    }
                } else {
                    for j in 0..other.tree_type {
                        let child_index = node_2.children[j as usize];
                        if other.node_array.get(child_index.0).is_some() {
                            self.overlap_traverse_callback(
                                other,
                                node_1_index,
                                child_index,
                                start_axis,
                                stop_axis,
                                callback,
                                r_overlap_pairs,
                            );
                        }
                    }
                }
            } else {
                for j in 0..self.tree_type {
                    let child_index = node_1.children[j as usize];
                    if self.node_array.get(child_index.0).is_some() {
                        self.overlap_traverse_callback(
                            other,
                            child_index,
                            node_2_index,
                            start_axis,
                            stop_axis,
                            callback,
                            r_overlap_pairs,
                        );
                    }
                }
            }
        }
    }

    fn overlap_traverse(
        &self,
        other: &BVHTree<T>,
        node_1_index: BVHNodeIndex,
        node_2_index: BVHNodeIndex,
        start_axis: u8,
        stop_axis: u8,
        r_overlap_pairs: &mut Vec<BVHTreeOverlap<T>>,
    ) {
        let node_1 = self.node_array.get(node_1_index.0).unwrap();
        let node_2 = other.node_array.get(node_2_index.0).unwrap();
        if node_1.overlap_test(node_2, start_axis, stop_axis) {
            // check if node_1 is a leaf node
            if node_1.totnode == 0 {
                // check if node_2 is a leaf node
                if node_2.totnode == 0 {
                    // the two nodes if equal, all the children will
                    // also match. This happens when overlap between
                    // the same tree is checked for.
                    if node_1_index == node_2_index {
                        return;
                    }

                    let overlap =
                        BVHTreeOverlap::new(node_1.elem_index.unwrap(), node_2.elem_index.unwrap());
                    r_overlap_pairs.push(overlap);
                } else {
                    for j in 0..other.tree_type {
                        let child_index = node_2.children[j as usize];
                        if other.node_array.get(child_index.0).is_some() {
                            self.overlap_traverse(
                                other,
                                node_1_index,
                                child_index,
                                start_axis,
                                stop_axis,
                                r_overlap_pairs,
                            );
                        }
                    }
                }
            } else {
                for j in 0..self.tree_type {
                    let child_index = node_1.children[j as usize];
                    if self.node_array.get(child_index.0).is_some() {
                        self.overlap_traverse(
                            other,
                            child_index,
                            node_2_index,
                            start_axis,
                            stop_axis,
                            r_overlap_pairs,
                        );
                    }
                }
            }
        }
    }

    pub fn overlap<F>(
        &self,
        other: &BVHTree<T>,
        callback: Option<&F>,
    ) -> Option<Vec<BVHTreeOverlap<T>>>
    where
        F: Fn(T, T) -> bool,
    {
        // TODO(ish): add multithreading support
        let use_threading = false;
        let root_node_len = self.overlap_thread_num();
        let _thread_num;
        if use_threading {
            _thread_num = root_node_len;
        } else {
            _thread_num = 1;
        }

        assert!(
            !(self.axis != other.axis
                && (self.axis == 14 || other.axis == 14)
                && (self.axis == 18 || other.axis == 18)),
            "trees not compatible for overlap check"
        );

        let root_1_index = self.nodes[self.totleaf];
        let root_2_index = other.nodes[other.totleaf];

        let start_axis = self.start_axis.min(other.start_axis);
        let stop_axis = self.stop_axis.min(other.stop_axis);

        // fast check root nodes for collision before expensive split and traversal
        let root_1 = self.node_array.get(root_1_index.0).unwrap();
        let root_2 = other.node_array.get(root_2_index.0).unwrap();
        if !root_1.overlap_test(root_2, start_axis, stop_axis) {
            return None;
        }

        if use_threading {
            panic!("Multithreading not implemented yet for BVHTree::overlap()");
        } else {
            let mut overlap_pairs = Vec::new();
            if let Some(callback) = callback {
                self.overlap_traverse_callback(
                    other,
                    root_1_index,
                    root_2_index,
                    start_axis,
                    stop_axis,
                    callback,
                    &mut overlap_pairs,
                );
            } else {
                self.overlap_traverse(
                    other,
                    root_1_index,
                    root_2_index,
                    start_axis,
                    stop_axis,
                    &mut overlap_pairs,
                );
            }
            if overlap_pairs.is_empty() {
                None
            } else {
                Some(overlap_pairs)
            }
        }
    }

    fn recursive_draw(
        &self,
        node_index: BVHNodeIndex,
        pos_attr: usize,
        color_attr: usize,
        imm: &mut GPUImmediate,
        draw_level: usize,
        current_level: usize,
    ) {
        let node = self.node_array.get(node_index.0).unwrap();

        if current_level == draw_level {
            let x1 = node.bv[0] as f32;
            let x2 = node.bv[1] as f32;
            let y1 = node.bv[(2)] as f32;
            let y2 = node.bv[(2) + 1] as f32;
            let z1 = node.bv[(2 * 2)] as f32;
            let z2 = node.bv[(2 * 2) + 1] as f32;

            draw_box(imm, x1, x2, y1, y2, z1, z2, pos_attr, color_attr);

            return; // don't need to go below this level anyway to render
        }

        if node.totnode != 0 {
            for i in 0..self.tree_type {
                let child_index = node.children[i as usize];
                if self.node_array.get(child_index.0).is_some() {
                    self.recursive_draw(
                        child_index,
                        pos_attr,
                        color_attr,
                        imm,
                        draw_level,
                        current_level + 1,
                    );
                }
            }
        }
    }
}

fn draw_line(
    imm: &mut GPUImmediate,
    p1: &glm::Vec3,
    p2: &glm::Vec3,
    pos_attr: usize,
    color_attr: usize,
) {
    imm.attr_4f(color_attr, 0.8, 0.3, 0.8, 1.0);
    imm.vertex_3f(pos_attr, p1[0], p1[1], p1[2]);
    imm.attr_4f(color_attr, 0.8, 0.3, 0.8, 1.0);
    imm.vertex_3f(pos_attr, p2[0], p2[1], p2[2]);
}

#[allow(clippy::too_many_arguments)]
fn draw_box(
    imm: &mut GPUImmediate,
    x1: f32,
    x2: f32,
    y1: f32,
    y2: f32,
    z1: f32,
    z2: f32,
    pos_attr: usize,
    color_attr: usize,
) {
    let v1 = glm::vec3(x1, y1, z1);
    let v2 = glm::vec3(x2, y1, z1);
    let v3 = glm::vec3(x2, y2, z1);
    let v4 = glm::vec3(x1, y2, z1);
    let v5 = glm::vec3(x1, y1, z2);
    let v6 = glm::vec3(x2, y1, z2);
    let v7 = glm::vec3(x2, y2, z2);
    let v8 = glm::vec3(x1, y2, z2);

    draw_line(imm, &v1, &v2, pos_attr, color_attr);
    draw_line(imm, &v2, &v3, pos_attr, color_attr);
    draw_line(imm, &v3, &v4, pos_attr, color_attr);
    draw_line(imm, &v4, &v1, pos_attr, color_attr);

    draw_line(imm, &v5, &v6, pos_attr, color_attr);
    draw_line(imm, &v6, &v7, pos_attr, color_attr);
    draw_line(imm, &v7, &v8, pos_attr, color_attr);
    draw_line(imm, &v8, &v5, pos_attr, color_attr);

    draw_line(imm, &v1, &v5, pos_attr, color_attr);
    draw_line(imm, &v2, &v6, pos_attr, color_attr);
    draw_line(imm, &v3, &v7, pos_attr, color_attr);
    draw_line(imm, &v4, &v8, pos_attr, color_attr);
}

fn draw_triangle(
    imm: &mut GPUImmediate,
    p1: &glm::Vec3,
    p2: &glm::Vec3,
    p3: &glm::Vec3,
    color: &glm::Vec4,
    pos_attr: usize,
    color_attr: usize,
) {
    imm.attr_4f(color_attr, color[0], color[1], color[2], color[3]);
    imm.vertex_3f(pos_attr, p1[0], p1[1], p1[2]);
    imm.attr_4f(color_attr, color[0], color[1], color[2], color[3]);
    imm.vertex_3f(pos_attr, p2[0], p2[1], p2[2]);
    imm.attr_4f(color_attr, color[0], color[1], color[2], color[3]);
    imm.vertex_3f(pos_attr, p3[0], p3[1], p3[2]);
}

#[allow(clippy::too_many_arguments)]
fn draw_box_fill(
    imm: &mut GPUImmediate,
    x1: f32,
    x2: f32,
    y1: f32,
    y2: f32,
    z1: f32,
    z2: f32,
    color: &glm::Vec4,
    pos_attr: usize,
    color_attr: usize,
) {
    let v1 = glm::vec3(x1, y1, z1);
    let v2 = glm::vec3(x2, y1, z1);
    let v3 = glm::vec3(x2, y2, z1);
    let v4 = glm::vec3(x1, y2, z1);
    let v5 = glm::vec3(x1, y1, z2);
    let v6 = glm::vec3(x2, y1, z2);
    let v7 = glm::vec3(x2, y2, z2);
    let v8 = glm::vec3(x1, y2, z2);

    draw_triangle(imm, &v2, &v6, &v5, color, pos_attr, color_attr);
    draw_triangle(imm, &v5, &v1, &v2, color, pos_attr, color_attr);

    draw_triangle(imm, &v3, &v7, &v6, color, pos_attr, color_attr);
    draw_triangle(imm, &v6, &v2, &v3, color, pos_attr, color_attr);

    draw_triangle(imm, &v6, &v7, &v8, color, pos_attr, color_attr);
    draw_triangle(imm, &v8, &v5, &v6, color, pos_attr, color_attr);

    draw_triangle(imm, &v1, &v5, &v8, color, pos_attr, color_attr);
    draw_triangle(imm, &v8, &v4, &v1, color, pos_attr, color_attr);

    draw_triangle(imm, &v3, &v2, &v1, color, pos_attr, color_attr);
    draw_triangle(imm, &v1, &v4, &v3, color, pos_attr, color_attr);

    draw_triangle(imm, &v4, &v8, &v7, color, pos_attr, color_attr);
    draw_triangle(imm, &v7, &v3, &v4, color, pos_attr, color_attr);
}

pub struct BVHDrawData<'a> {
    imm: &'a mut GPUImmediate,
    shader: &'a Shader,
    draw_level: usize,
}

impl<'a> BVHDrawData<'a> {
    pub fn new(imm: &'a mut GPUImmediate, shader: &'a Shader, draw_level: usize) -> Self {
        Self {
            imm,
            shader,
            draw_level,
        }
    }
}

impl<T> Drawable<BVHDrawData<'_>, ()> for BVHTree<T>
where
    T: Copy,
{
    fn draw(&self, draw_data: &mut BVHDrawData) -> Result<(), ()> {
        let imm = &mut draw_data.imm;
        let shader = &draw_data.shader;
        let draw_level = draw_data.draw_level;
        shader.use_shader();

        let format = imm.get_cleared_vertex_format();
        let pos_attr = format.add_attribute(
            "in_pos\0".to_string(),
            GPUVertCompType::F32,
            3,
            GPUVertFetchMode::Float,
        );
        let color_attr = format.add_attribute(
            "in_color\0".to_string(),
            GPUVertCompType::F32,
            4,
            GPUVertFetchMode::Float,
        );

        imm.begin_at_most(GPUPrimType::Lines, self.nodes.len() * 12 * 2, shader);

        self.recursive_draw(
            self.nodes[self.totleaf],
            pos_attr,
            color_attr,
            imm,
            draw_level,
            0,
        );

        imm.end();

        Ok(())
    }
}

#[allow(clippy::upper_case_acronyms)]
pub trait AABB {
    type ElementIndex;
    fn give_aabb(&self, elem_index: Self::ElementIndex) -> Vec<Scalar>;
}

pub struct BVHOverlapDrawData<'a, T> {
    imm: &'a mut GPUImmediate,
    shader: &'a Shader,
    give_aabb_1: &'a dyn AABB<ElementIndex = T>,
    give_aabb_2: &'a dyn AABB<ElementIndex = T>,
}

impl<'a, T> BVHOverlapDrawData<'a, T> {
    pub fn new(
        imm: &'a mut GPUImmediate,
        shader: &'a Shader,
        give_aabb_1: &'a dyn AABB<ElementIndex = T>,
        give_aabb_2: &'a dyn AABB<ElementIndex = T>,
    ) -> Self {
        Self {
            imm,
            shader,
            give_aabb_1,
            give_aabb_2,
        }
    }
}

impl<T> Drawable<BVHOverlapDrawData<'_, T>, ()> for Vec<BVHTreeOverlap<T>>
where
    T: Copy,
{
    fn draw(&self, draw_data: &mut BVHOverlapDrawData<T>) -> Result<(), ()> {
        let imm = &mut draw_data.imm;
        let shader = &draw_data.shader;
        let give_aabb_1 = &draw_data.give_aabb_1;
        let give_aabb_2 = &draw_data.give_aabb_2;
        shader.use_shader();

        let format = imm.get_cleared_vertex_format();
        let pos_attr = format.add_attribute(
            "in_pos\0".to_string(),
            GPUVertCompType::F32,
            3,
            GPUVertFetchMode::Float,
        );
        let color_attr = format.add_attribute(
            "in_color\0".to_string(),
            GPUVertCompType::F32,
            4,
            GPUVertFetchMode::Float,
        );

        unsafe {
            gl::Enable(gl::BLEND);
            gl::BlendFunc(gl::SRC_ALPHA, gl::ONE_MINUS_SRC_ALPHA);
        }

        let mut rng = rand_chacha::ChaChaRng::seed_from_u64(10);

        // Need self.len() (number of overlap pairs) * 6 (number of
        // faces in a box) * 2 (triangles per face) * 3 (verts per
        // triangle) * 2 (both boxes)
        imm.begin(GPUPrimType::Tris, self.len() * 6 * 2 * 3 * 2, shader);

        for overlap in self {
            let color = glm::vec4(rng.gen(), rng.gen(), rng.gen(), 0.3);
            let bv = give_aabb_1.give_aabb(overlap.index_1);
            assert_eq!(bv.len(), 6);
            draw_box_fill(
                imm,
                bv[0] as f32,
                bv[1] as f32,
                bv[2] as f32,
                bv[3] as f32,
                bv[4] as f32,
                bv[5] as f32,
                &color,
                pos_attr,
                color_attr,
            );

            let bv = give_aabb_2.give_aabb(overlap.index_2);
            assert_eq!(bv.len(), 6);
            draw_box_fill(
                imm,
                bv[0] as f32,
                bv[1] as f32,
                bv[2] as f32,
                bv[3] as f32,
                bv[4] as f32,
                bv[5] as f32,
                &color,
                pos_attr,
                color_attr,
            );
        }

        imm.end();

        unsafe {
            gl::Disable(gl::BLEND);
        }

        Ok(())
    }
}

fn implicit_needed_branches(tree_type: u8, leafs: usize) -> usize {
    1.max(leafs + tree_type as usize - 3) / (tree_type - 1) as usize
}

fn get_largest_axis(bv: &[Scalar]) -> u8 {
    let middle_point_x = bv[1] - bv[0]; // x axis
    let middle_point_y = bv[3] - bv[2]; // y axis
    let middle_point_z = bv[5] - bv[4]; // z axis

    if middle_point_x > middle_point_y {
        if middle_point_x > middle_point_z {
            1 // max x axis
        } else {
            5 // max z axis
        }
    } else if middle_point_y > middle_point_z {
        3 // max y axis
    } else {
        5 // max z axis
    }
}

trait ArenaFunctions {
    type Output;

    fn get_unknown_index(&self, i: usize) -> Index;
    fn get_unknown(&self, i: usize) -> &Self::Output;
    fn get_unknown_mut(&mut self, i: usize) -> &mut Self::Output;
}

impl<T> ArenaFunctions for Arena<T> {
    type Output = T;

    #[inline]
    fn get_unknown_index(&self, i: usize) -> Index {
        return self.get_unknown_gen(i).unwrap().1;
    }

    #[inline]
    fn get_unknown(&self, i: usize) -> &Self::Output {
        return self.get_unknown_gen(i).unwrap().0;
    }

    #[inline]
    fn get_unknown_mut(&mut self, i: usize) -> &mut Self::Output {
        return self.get_unknown_gen_mut(i).unwrap().0;
    }
}
