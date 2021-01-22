use nalgebra_glm as glm;

use std::cell::RefCell;
use std::rc::{Rc, Weak};

pub struct Node {
    id: usize,
    pos: glm::DVec3,

    verts: IncidentVerts,
}

pub struct Vertex {
    id: usize,
    uv: Option<glm::DVec2>,

    node: RefToNode,
    edges: IncidentEdges,
}

pub struct Edge {
    id: usize,

    verts: Option<(AdjacentVert, AdjacentVert)>,
    faces: IncidentFaces,
}

pub struct Face {
    id: usize,
    normal: Option<glm::DVec3>,

    edges: AdjacentEdges,
}

pub struct Mesh {
    nodes: Vec<Rc<RefCell<Node>>>,
    verts: Vec<Rc<RefCell<Vertex>>>,
    edges: Vec<Rc<RefCell<Edge>>>,
    faces: Vec<Rc<RefCell<Face>>>,
}

type IncidentEdges = Vec<Weak<RefCell<Edge>>>;
type AdjacentVert = Weak<RefCell<Vertex>>;
type IncidentFaces = Vec<Weak<RefCell<Face>>>;
type AdjacentEdges = IncidentEdges;
type IncidentVerts = Vec<Weak<RefCell<Vertex>>>;
type RefToNode = Weak<RefCell<Node>>;
type RefToVert = Weak<RefCell<Vertex>>;
type RefToEdge = Weak<RefCell<Edge>>;
type RefToFace = Weak<RefCell<Face>>;

impl Mesh {
    pub fn new() -> Mesh {
        return Mesh {
            nodes: Vec::new(),
            verts: Vec::new(),
            edges: Vec::new(),
            faces: Vec::new(),
        };
    }
}

impl Face {
    pub fn new(id: usize) -> Face {
        return Face {
            id,
            normal: None,

            edges: Vec::new(),
        };
    }
}

impl Edge {
    pub fn new(id: usize) -> Edge {
        return Edge {
            id,

            verts: None,
            faces: Vec::new(),
        };
    }
}

impl Vertex {
    pub fn new(id: usize) -> Vertex {
        return Vertex {
            id,
            uv: None,

            node: Weak::new(),
            edges: Vec::new(),
        };
    }
}

impl Node {
    pub fn new(id: usize, pos: glm::DVec3) -> Node {
        return Node {
            id,
            pos,

            verts: Vec::new(),
        };
    }
}
