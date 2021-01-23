use itertools::Itertools;
use nalgebra_glm as glm;

use std::cell::RefCell;
use std::path::Path;
use std::rc::{Rc, Weak};

use crate::drawable::Drawable;
use crate::gl_mesh::GLMesh;
use crate::meshreader::{MeshReader, MeshReaderError};

pub struct Node {
    pub pos: glm::DVec3,

    verts: IncidentVerts,
}

pub struct Vertex {
    pub uv: Option<glm::DVec2>,

    node: RefToNode,
    edges: IncidentEdges,
}

pub struct Edge {
    verts: Option<(AdjacentVert, AdjacentVert)>,
    faces: IncidentFaces,
}

pub struct Face {
    pub normal: Option<glm::DVec3>,

    edges: AdjacentEdges,
}

pub struct Mesh {
    nodes: Vec<Rc<RefCell<Node>>>,
    verts: Vec<Rc<RefCell<Vertex>>>,
    edges: Vec<Rc<RefCell<Edge>>>,
    faces: Vec<Rc<RefCell<Face>>>,

    gl_mesh: Option<GLMesh>,
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

            gl_mesh: None,
        };
    }

    pub fn add_empty_node(&mut self, pos: glm::DVec3) -> RefToNode {
        self.nodes.push(Rc::new(RefCell::new(Node::new(pos))));
        return Rc::downgrade(self.nodes.last().unwrap());
    }

    pub fn add_empty_vert(&mut self) -> RefToVert {
        self.verts.push(Rc::new(RefCell::new(Vertex::new())));
        return Rc::downgrade(self.verts.last().unwrap());
    }

    pub fn add_empty_edge(&mut self) -> RefToEdge {
        self.edges.push(Rc::new(RefCell::new(Edge::new())));
        return Rc::downgrade(self.edges.last().unwrap());
    }

    pub fn add_empty_face(&mut self) -> RefToFace {
        self.faces.push(Rc::new(RefCell::new(Face::new())));
        return Rc::downgrade(self.faces.last().unwrap());
    }

    pub fn read(&mut self, path: &Path) -> Result<(), MeshReaderError> {
        let data = MeshReader::read(path)?;

        // Create all the nodes
        for pos in data.positions {
            self.add_empty_node(pos);
        }
        // Create all the verts
        for uv in data.uvs {
            let vert = self.add_empty_vert().upgrade().unwrap();
            vert.borrow_mut().uv = Some(uv);
        }
        // TODO(ish): add support to store vertex normals
        for face_i in data.face_indices {
            let mut face_edges: AdjacentEdges = Vec::new();
            for (pos_index, uv_index, _) in &face_i {
                // Update vert to store the node
                self.verts[*uv_index].borrow_mut().node = Rc::downgrade(&self.nodes[*pos_index]);
                // Update node to store the vert
                self.nodes[*pos_index]
                    .borrow_mut()
                    .verts
                    .push(Rc::downgrade(&self.verts[*uv_index]));
            }
            // Update edges and store them to update the face
            for ((_, vi_1, _), (_, vi_2, _)) in face_i.into_iter().circular_tuple_windows() {
                let edge = match self.verts[vi_1]
                    .borrow()
                    .get_connecting_edge(&Rc::downgrade(&self.verts[vi_2]))
                {
                    Some(edge_ref) => Some(edge_ref),
                    None => None,
                };

                let edge = match edge {
                    Some(edge) => edge,
                    None => {
                        let edge_ref = self.add_empty_edge();
                        let edge_refcell = edge_ref.upgrade().unwrap();
                        let mut edge = edge_refcell.borrow_mut();
                        edge.verts = Some((
                            Rc::downgrade(&self.verts[vi_1]),
                            Rc::downgrade(&self.verts[vi_2]),
                        ));
                        edge_ref
                    }
                };
                self.verts[vi_1].borrow_mut().edges.push(edge.clone());
                self.verts[vi_2].borrow_mut().edges.push(edge.clone());
                face_edges.push(edge);
            }
            // Update face
            let face_ref = self.add_empty_face();
            let face_refcell = face_ref.upgrade().unwrap();
            let mut face = face_refcell.borrow_mut();
            face.edges = face_edges;
            for edge in &face.edges {
                add_as_set(
                    &mut edge.upgrade().unwrap().borrow_mut().faces,
                    &face_ref.clone(),
                );
            }
        }

        return Ok(());
    }

    pub fn generate_gl_mesh(&mut self) {}
}

#[derive(Debug, Copy, Clone)]
pub enum MeshDrawError {
    GenerateGLMeshFirst,
    ErrorWhileDrawing,
}

impl std::fmt::Display for MeshDrawError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MeshDrawError::GenerateGLMeshFirst => {
                write!(f, "Generate GLMesh before calling draw()")
            }
            MeshDrawError::ErrorWhileDrawing => {
                write!(f, "Error while drawing Mesh")
            }
        }
    }
}

impl std::error::Error for MeshDrawError {}

impl From<()> for MeshDrawError {
    fn from(_err: ()) -> MeshDrawError {
        return MeshDrawError::ErrorWhileDrawing;
    }
}

impl Drawable<MeshDrawError> for Mesh {
    fn draw(&self) -> Result<(), MeshDrawError> {
        match self.gl_mesh {
            None => return Err(MeshDrawError::GenerateGLMeshFirst),
            Some(_) => (),
        }
        self.gl_mesh.as_ref().unwrap().draw()?;
        return Ok(());
    }
}

impl Face {
    pub fn new() -> Face {
        return Face {
            normal: None,

            edges: Vec::new(),
        };
    }
}

impl Edge {
    pub fn new() -> Edge {
        return Edge {
            verts: None,
            faces: Vec::new(),
        };
    }
}

impl Vertex {
    pub fn new() -> Vertex {
        return Vertex {
            uv: None,

            node: Weak::new(),
            edges: Vec::new(),
        };
    }

    pub fn get_connecting_edge(&self, other: &RefToVert) -> Option<RefToEdge> {
        for edge_weak in &self.edges {
            let edge = edge_weak.upgrade().unwrap();
            let edge = edge.borrow();
            match &edge.verts {
                Some((vert_1, vert_2)) => {
                    if vert_1.ptr_eq(other) || vert_2.ptr_eq(other) {
                        return Some(edge_weak.clone());
                    }
                }
                None => {
                    return None;
                }
            }
        }
        return None;
    }
}

impl Node {
    pub fn new(pos: glm::DVec3) -> Node {
        return Node {
            pos,

            verts: Vec::new(),
        };
    }
}

fn add_as_set<T>(vec: &mut Vec<Weak<RefCell<T>>>, val: &Weak<RefCell<T>>) {
    for v in vec.iter() {
        if v.ptr_eq(val) {
            return;
        }
    }
    vec.push(val.clone());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mesh_read_test() {
        // TODO(ish): add more comprehensive relation tests
        let mut mesh = Mesh::new();
        mesh.read(&Path::new("tests/obj_test_01.obj")).unwrap();
        assert_eq!(mesh.faces.len(), 2);
        for face in &mesh.faces {
            assert_eq!(face.borrow().edges.len(), 3);
        }
        assert_eq!(mesh.edges.len(), 6);
        for edge in &mesh.edges {
            assert_eq!(edge.borrow().faces.len(), 1);
            match edge.borrow().verts {
                Some(_) => assert!(true),
                None => assert!(false),
            }
        }
        assert_eq!(mesh.verts.len(), 6);
        for vert in &mesh.verts {
            assert_eq!(vert.borrow().edges.len(), 2);
        }
        assert_eq!(mesh.nodes.len(), 5);
        for node in &mesh.nodes {
            let len = node.borrow().verts.len();
            assert!(len == 0 || len == 1 || len == 2);
        }
    }
}
