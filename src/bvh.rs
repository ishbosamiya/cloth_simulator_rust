use generational_arena::{Arena, Index};

const MAX_TREETYPE: u8 = 32;

type Scalar = f64;

struct BVHNodeIndex(pub Index);
struct BVIndex(pub Index);

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
}

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
}

fn implicit_needed_branches(tree_type: u8, leafs: usize) -> usize {
    return 1.max(leafs + tree_type as usize - 3) / (tree_type - 1) as usize;
}
