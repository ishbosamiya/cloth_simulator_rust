use nalgebra_glm as glm;

use std::cell::RefCell;
use std::rc::{Rc, Weak};

pub struct Node<ExtraData = ()> {
    id: usize,
    pos: glm::DVec3,
    extra_data: Option<ExtraData>,

    verts: IncidentVerts,
}

pub struct Vertex<ExtraData = ()> {
    id: usize,
    uv: Option<glm::DVec2>,
    extra_data: Option<ExtraData>,

    node: RefToNode,
    edges: IncidentEdges,
}

pub struct Edge<ExtraData = ()> {
    id: usize,
    extra_data: Option<ExtraData>,

    verts: Option<(AdjacentVert, AdjacentVert)>,
    faces: IncidentFaces,
}

pub struct Face<ExtraData = ()> {
    id: usize,
    normal: Option<glm::DVec3>,
    extra_data: Option<ExtraData>,

    edges: AdjacentEdges,
}

pub struct Mesh<ExtraDataVertex = (), ExtraDataEdge = (), ExtraDataFace = ()> {
    verts: Vec<Rc<RefCell<Vertex<ExtraDataVertex>>>>,
    edges: Vec<Rc<RefCell<Edge<ExtraDataEdge>>>>,
    faces: Vec<Rc<RefCell<Face<ExtraDataFace>>>>,
}

type IncidentEdges = Vec<Weak<RefCell<Edge>>>;
type AdjacentVert = Weak<RefCell<Vertex>>;
type IncidentFaces = Vec<Weak<RefCell<Face>>>;
type AdjacentEdges = IncidentEdges;
type RefToNode = Weak<RefCell<Node>>;
type IncidentVerts = Vec<Weak<RefCell<Vertex>>>;

impl<ExtraDataVertex, ExtraDataEdge, ExtraDataFace>
    Mesh<ExtraDataVertex, ExtraDataEdge, ExtraDataFace>
{
    pub fn new() -> Mesh<ExtraDataVertex, ExtraDataEdge, ExtraDataFace> {
        return Mesh {
            verts: Vec::new(),
            edges: Vec::new(),
            faces: Vec::new(),
        };
    }
}

impl<ExtraData> Face<ExtraData> {
    pub fn new(id: usize) -> Face<ExtraData> {
        return Face {
            id,
            normal: None,
            extra_data: None,

            edges: Vec::new(),
        };
    }
}

impl<ExtraData> Edge<ExtraData> {
    pub fn new(id: usize) -> Edge<ExtraData> {
        return Edge {
            id,
            extra_data: None,

            verts: None,
            faces: Vec::new(),
        };
    }
}

impl<ExtraData> Vertex<ExtraData> {
    pub fn new(id: usize) -> Vertex<ExtraData> {
        return Vertex {
            id,
            uv: None,
            extra_data: None,

            node: Weak::new(),
            edges: Vec::new(),
        };
    }
}

impl<ExtraData> Node<ExtraData> {
    pub fn new(id: usize, pos: glm::DVec3) -> Node<ExtraData> {
        return Node {
            id,
            pos,
            extra_data: None,
            verts: Vec::new(),
        };
    }
}
