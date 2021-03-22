use generational_arena::{Arena, Index};

const MAX_TREETYPE: u8 = 32;

type Scalar = f64;

struct BVHNodeIndex(pub Index);
struct BVIndex(pub Index);

struct BVHNode<T> {
    children: Vec<BVHNodeIndex>, // Indices of the child nodes
    parent: BVHNodeIndex,        // Parent index

    bv: BVIndex,   // Bounding volume start index within node_bv of the tree
    elem_index: T, // Index of element stored in the node
    totnode: u8,   // How many nodes are used, used for speedup
    main_axis: u8, // Axis used to split this node
}

pub struct BVHTree<T> {
    nodes: Vec<BVHNodeIndex>,
    node_array: Arena<BVHNode<T>>, // Where the actual nodes are stored
    node_child: Vec<BVHNodeIndex>,
    node_bv: Arena<Scalar>, // Where the actual bounding volume info is stored

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
        let node_array = Arena::with_capacity(numnodes);
        let node_child = Vec::with_capacity(tree_type as usize * numnodes);
        let node_bv = Arena::with_capacity(axis as usize * numnodes);

        // TODO(ish): initialize and link up everything

        return Self {
            nodes,
            node_array,
            node_child,
            node_bv,

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
