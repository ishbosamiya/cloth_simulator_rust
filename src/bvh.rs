use generational_arena::{Arena, Index};
use nalgebra_glm as glm;

const MAX_TREETYPE: u8 = 32;

lazy_static! {
    static ref BVHTREE_KDOP_AXES: Vec<glm::TVec3<Scalar>> = {
        let mut v = Vec::with_capacity(13);
        v.push(glm::vec3(1.0, 0.0, 0.0));
        v.push(glm::vec3(0.0, 1.0, 0.0));
        v.push(glm::vec3(0.0, 0.0, 1.0));
        v.push(glm::vec3(1.0, 1.0, 1.0));
        v.push(glm::vec3(1.0, -1.0, 1.0));
        v.push(glm::vec3(1.0, 1.0, -1.0));
        v.push(glm::vec3(1.0, -1.0, -1.0));
        v.push(glm::vec3(1.0, 1.0, 0.0));
        v.push(glm::vec3(1.0, 0.0, 1.0));
        v.push(glm::vec3(1.0, -1.0, 0.0));
        v.push(glm::vec3(1.0, 0.0, -1.0));
        v.push(glm::vec3(0.0, 1.0, -1.0));
        assert_eq!(v.len(), 13);
        v
    };
}

type Scalar = f64;

struct BVHNodeIndex(pub Index);

struct BVHNode<T> {
    children: Vec<BVHNodeIndex>,  // Indices of the child nodes
    parent: Option<BVHNodeIndex>, // Parent index

    bv: Vec<Scalar>,       // Bounding volume axis data
    elem_index: Option<T>, // Index of element stored in the node
    totnode: u8,           // How many nodes are used, used for speedup
    main_axis: u8,         // Axis used to split this node
}

impl<T> BVHNode<T> {
    fn new() -> Self {
        return Self {
            children: Vec::new(),
            parent: None,

            bv: Vec::new(),
            elem_index: None,
            totnode: 0,
            main_axis: 0,
        };
    }

    fn min_max_init(&mut self, start_axis: u8, stop_axis: u8) {
        let bv = &mut self.bv;
        for axis_iter in start_axis..stop_axis {
            bv[(axis_iter + 0) as usize] = Scalar::MAX;
            bv[(axis_iter + 1) as usize] = -Scalar::MAX;
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

pub struct BVHTree<T> {
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

impl<T> BVHTree<T> {
    pub fn new(max_size: usize, epsilon: Scalar, tree_type: u8, axis: u8) -> Self {
        assert!(
            tree_type >= 2 && tree_type <= MAX_TREETYPE,
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
        let nodes = Vec::with_capacity(numnodes);
        let mut node_array = Arena::with_capacity(numnodes);

        for _ in 0..numnodes {
            node_array.insert(BVHNode::new());
        }

        for i in 0..numnodes {
            let node = node_array.get_unknown_gen_mut(i).unwrap().0;
            node.bv.resize(axis.into(), 0.0);
            // TODO(ish): might have to initialize the node.children here but should be possible elsewhere
        }

        return Self {
            nodes,
            node_array,

            epsilon,
            totleaf: 0,
            totbranch: 0,
            start_axis,
            stop_axis,
            axis,
            tree_type,
        };
    }

    pub fn insert(&mut self, index: T, co_many: Vec<glm::TVec3<Scalar>>) {
        assert!(self.totbranch <= 0);

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

    pub fn update(
        &mut self,
        node_index: usize,
        co_many: Vec<glm::TVec3<Scalar>>,
        co_moving_many: Vec<glm::TVec3<Scalar>>,
    ) -> Result<(), BVHError> {
        if node_index > self.totleaf {
            return Err(BVHError::IndexOutOfRange);
        }
        if co_moving_many.len() > 0 {
            if co_many.len() != co_moving_many.len() {
                return Err(BVHError::DifferentNumPoints);
            }
        }

        let node = self.node_array.get_unknown_mut(node_index);

        node.create_kdop_hull(self.start_axis, self.stop_axis, co_many, false);

        if co_moving_many.len() > 0 {
            node.create_kdop_hull(self.start_axis, self.stop_axis, co_moving_many, true);
        }

        // Inflate bv by epsilon
        for axis_iter in self.start_axis..self.stop_axis {
            let axis_iter = axis_iter as usize;
            node.bv[(2 * axis_iter)] -= self.epsilon; // min
            node.bv[(2 * axis_iter) + 1] += self.epsilon; // max
        }

        return Ok(());
    }
}

fn implicit_needed_branches(tree_type: u8, leafs: usize) -> usize {
    return 1.max(leafs + tree_type as usize - 3) / (tree_type - 1) as usize;
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
