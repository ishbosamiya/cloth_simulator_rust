use itertools::Itertools;
use nalgebra_glm as glm;

use std::cell::RefCell;
use std::convert::TryInto;
use std::path::Path;
use std::rc::{Rc, Weak};

use crate::drawable::Drawable;
use crate::gl_mesh::{GLMesh, GLVert};
use crate::meshreader::{MeshReader, MeshReaderError};

pub struct Node {
    pub pos: glm::DVec3,
    pub normal: Option<glm::DVec3>,

    verts: IncidentVerts,
}

pub struct Vertex {
    id: usize,
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

#[derive(Debug)]
pub enum MeshError {
    MeshReader(MeshReaderError),
    NoUV,
}

impl From<MeshReaderError> for MeshError {
    fn from(err: MeshReaderError) -> MeshError {
        return MeshError::MeshReader(err);
    }
}

impl std::fmt::Display for MeshError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MeshError::MeshReader(error) => write!(f, "{}", error),
            MeshError::NoUV => write!(f, "No UV information found"),
        }
    }
}

impl std::error::Error for MeshError {}

type IncidentEdges = Vec<Weak<RefCell<Edge>>>;
type AdjacentVert = Weak<RefCell<Vertex>>;
type AdjacentVerts = Vec<Weak<RefCell<Vertex>>>;
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
        let id = self.verts.len();
        self.verts.push(Rc::new(RefCell::new(Vertex::new(id))));
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

    pub fn read(&mut self, path: &Path) -> Result<(), MeshError> {
        let data = MeshReader::read(path)?;

        if data.uvs.len() == 0 || data.face_has_uv == false {
            return Err(MeshError::NoUV);
        }

        // Create all the nodes
        for pos in data.positions {
            self.add_empty_node(pos);
        }
        // Create all the verts
        for uv in data.uvs {
            let vert = self.add_empty_vert().upgrade().unwrap();
            vert.borrow_mut().uv = Some(uv);
        }
        for face_i in data.face_indices {
            let mut face_edges: AdjacentEdges = Vec::new();
            for (pos_index, uv_index, normal_index) in &face_i {
                // Update vert to store the node
                self.verts[*uv_index].borrow_mut().node = Rc::downgrade(&self.nodes[*pos_index]);
                // Update node to store the vert
                let mut node = self.nodes[*pos_index].borrow_mut();
                node.verts.push(Rc::downgrade(&self.verts[*uv_index]));
                if data.face_has_normal && data.normals.len() > 0 {
                    node.set_normal(data.normals[*normal_index]);
                }
            }
            // Update edges and store them to update the face
            for ((_, vi_1, _), (_, vi_2, _)) in face_i.into_iter().circular_tuple_windows() {
                // Always create a new edge to ensure that the mesh's faces
                // can be oriented, otherwise there will be mesh winding cannot be made consistent
                let edge_ref = self.add_empty_edge();
                let edge_refcell = edge_ref.upgrade().unwrap();
                let mut edge = edge_refcell.borrow_mut();
                edge.verts = Some((
                    Rc::downgrade(&self.verts[vi_1]),
                    Rc::downgrade(&self.verts[vi_2]),
                ));
                self.verts[vi_1].borrow_mut().edges.push(edge_ref.clone());
                self.verts[vi_2].borrow_mut().edges.push(edge_ref.clone());
                face_edges.push(edge_ref);
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
            face.sort_edges();
        }
        // Any node without a vert gets a new vert without uv
        let mut loose_nodes: Vec<RefToNode> = Vec::new();
        for node_rc in &self.nodes {
            let node = node_rc.borrow();
            if node.verts.len() == 0 {
                loose_nodes.push(Rc::downgrade(node_rc));
            }
        }
        for node_weak in &loose_nodes {
            let node_refcell = node_weak.upgrade().unwrap();
            let mut node = node_refcell.borrow_mut();
            let vert_weak = self.add_empty_vert();
            let vert_refcell = vert_weak.upgrade().unwrap();
            let mut vert = vert_refcell.borrow_mut();
            vert.node = node_weak.clone();
            node.verts.push(vert_weak);
        }
        // Add the lines to the mesh
        for line in data.line_indices {
            for (node_index_1, node_index_2) in line.iter().tuple_windows() {
                let node_1_rc = self.nodes[*node_index_1].clone();
                let node_2_rc = self.nodes[*node_index_2].clone();
                // Assuming that only new edges are to be created with the first vert node.verts
                assert!(node_1_rc.borrow().verts.len() > 0 && node_2_rc.borrow().verts.len() > 0);
                let vert_1_weak = &node_1_rc.borrow().verts[0];
                let vert_2_weak = &node_2_rc.borrow().verts[0];
                let edge_weak = self.add_empty_edge();
                let edge_refcell = edge_weak.upgrade().unwrap();
                let mut edge = edge_refcell.borrow_mut();
                edge.verts = Some((vert_1_weak.clone(), vert_2_weak.clone()));
                let vert_1_refcell = vert_1_weak.upgrade().unwrap();
                let vert_2_refcell = vert_2_weak.upgrade().unwrap();
                let mut vert_1 = vert_1_refcell.borrow_mut();
                let mut vert_2 = vert_2_refcell.borrow_mut();
                vert_1.edges.push(edge_weak.clone());
                vert_2.edges.push(edge_weak);
            }
        }

        return Ok(());
    }

    pub fn generate_gl_mesh(&mut self, use_face_normal: bool) {
        fn store_in_gl_vert(
            gl_verts: &mut Vec<GLVert>,
            vert: &Vertex,
            node: &Node,
            face_rc: &Rc<RefCell<Face>>,
            use_face_normal: bool,
        ) {
            match vert.uv {
                Some(uv) => {
                    if use_face_normal {
                        match face_rc.borrow().normal {
                            Some(normal) => gl_verts.push(GLVert::new(
                                glm::convert(node.pos),
                                glm::convert(uv),
                                glm::convert(normal),
                            )),
                            None => gl_verts.push(GLVert::new(
                                glm::convert(node.pos),
                                glm::convert(uv),
                                glm::zero(),
                            )),
                        }
                    } else {
                        match node.normal {
                            Some(normal) => gl_verts.push(GLVert::new(
                                glm::convert(node.pos),
                                glm::convert(uv),
                                glm::convert(normal),
                            )),
                            None => gl_verts.push(GLVert::new(
                                glm::convert(node.pos),
                                glm::convert(uv),
                                glm::zero(),
                            )),
                        }
                    }
                }
                None => gl_verts.push(GLVert::new(
                    glm::convert(node.pos),
                    glm::zero(),
                    glm::zero(),
                )),
            }
        }
        let mut gl_verts: Vec<GLVert> = Vec::new();
        let mut gl_indices: Vec<gl::types::GLuint> = Vec::new();
        for face_rc in &self.faces {
            let verts = face_rc.borrow().get_adjacent_verts();
            let vert_1_weak = &verts[0];
            let vert_1_refcell = vert_1_weak.upgrade().unwrap();
            let vert_1 = vert_1_refcell.borrow();
            let node_1_refcell = vert_1.node.upgrade().unwrap();
            let node_1 = node_1_refcell.borrow();
            let id1 = gl_verts.len();
            store_in_gl_vert(&mut gl_verts, &vert_1, &node_1, &face_rc, use_face_normal);
            for (vert_2_weak, vert_3_weak) in verts.iter().skip(1).tuple_windows() {
                let vert_2_refcell = vert_2_weak.upgrade().unwrap();
                let vert_2 = vert_2_refcell.borrow();
                let node_2_refcell = vert_2.node.upgrade().unwrap();
                let node_2 = node_2_refcell.borrow();
                let vert_3_refcell = vert_3_weak.upgrade().unwrap();
                let vert_3 = vert_3_refcell.borrow();
                let node_3_refcell = vert_3.node.upgrade().unwrap();
                let node_3 = node_3_refcell.borrow();
                let id2 = gl_verts.len();
                store_in_gl_vert(&mut gl_verts, &vert_2, &node_2, &face_rc, use_face_normal);
                let id3 = gl_verts.len();
                store_in_gl_vert(&mut gl_verts, &vert_3, &node_3, &face_rc, use_face_normal);
                gl_indices.push(id1.try_into().unwrap());
                gl_indices.push(id2.try_into().unwrap());
                gl_indices.push(id3.try_into().unwrap());
            }
        }
        self.gl_mesh = Some(GLMesh::new(gl_verts, gl_indices));
    }
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

    pub fn get_adjacent_verts(&self) -> AdjacentVerts {
        assert!(self.edges.len() >= 3);
        let mut adjacent_verts: AdjacentVerts = Vec::new();
        for edge_weak in &self.edges {
            let edge_refcell = edge_weak.upgrade().unwrap();
            let edge = edge_refcell.borrow();
            let (vert_1, vert_2) = edge.verts.as_ref().unwrap();
            add_as_set(&mut adjacent_verts, vert_1);
            add_as_set(&mut adjacent_verts, vert_2);
        }
        return adjacent_verts;
    }

    pub fn get_adjacent_edges(&self) -> AdjacentEdges {
        return self.edges.clone();
    }

    /// Sort edges based on their winding
    pub fn sort_edges(&mut self) {
        let edges_len = self.edges.len();
        assert!(edges_len >= 3);
        let mut edges_ordered = Vec::new();
        let mut remaining_edges = self.edges.clone();
        let edge_weak = remaining_edges.remove(0);
        edges_ordered.push(edge_weak.clone());
        let edge_rc = edge_weak.upgrade().unwrap();
        let edge = edge_rc.borrow();
        let vert_weak = &edge.verts.as_ref().unwrap().0;
        let mut vert_other_weak = edge_rc.borrow().get_other_vert(&vert_weak).unwrap();
        fn find_edge_in_remaining(
            remaining_edges: &Vec<Weak<RefCell<Edge>>>,
            vert: &RefToVert,
        ) -> Option<(usize, bool)> {
            for (i, edge_weak) in remaining_edges.iter().enumerate() {
                let edge_rc = edge_weak.upgrade().unwrap();
                let edge = edge_rc.borrow();
                let vert_1_weak = &edge.verts.as_ref().unwrap().0;
                let vert_2_weak = &edge.verts.as_ref().unwrap().1;
                if vert_1_weak.ptr_eq(vert) {
                    return Some((i, false));
                }
                if vert_2_weak.ptr_eq(vert) {
                    return Some((i, true));
                }
            }
            return None;
        }

        let loop_run_time = remaining_edges.len();
        for _ in 0..loop_run_time {
            let (edge_index, swap) =
                find_edge_in_remaining(&remaining_edges, &vert_other_weak).unwrap();
            let edge_weak = remaining_edges.remove(edge_index);
            edges_ordered.push(edge_weak.clone());
            let edge_rc = edge_weak.upgrade().unwrap();
            let mut edge_mut = edge_rc.borrow_mut();
            if swap {
                edge_mut.swap_verts();
            }
            let vert_weak = &edge_mut.verts.as_ref().unwrap().0;
            vert_other_weak = edge_mut.get_other_vert(&vert_weak).unwrap();
        }

        drop(edge);

        let rough_norm = self.get_rough_normal_from_vert_normal();
        let normal = self.calculate_face_normal();
        if normal.dot(&rough_norm) > 0.0 {
            self.normal = Some(normal);
        } else {
            // print!("before_reverse: ");
            // _debug_edges_print_order(&edges_ordered);
            edges_ordered.reverse();
            for edge_weak in &edges_ordered {
                let edge_rc = edge_weak.upgrade().unwrap();
                let mut edge = edge_rc.borrow_mut();
                edge.swap_verts();
            }
            // print!("after_reverse: ");
            // _debug_edges_print_order(&edges_ordered);
            self.normal = Some(-normal);
        }

        // print!("unordered: ");
        // _debug_edges_print_order(&self.edges);
        // print!("ordered: ");
        // _debug_edges_print_order(&edges_ordered);

        self.edges = edges_ordered;
    }

    fn get_rough_normal_from_vert_normal(&self) -> glm::DVec3 {
        let verts = self.get_adjacent_verts();
        assert!(verts.len() >= 3);
        let vert_1_weak = &verts[0];
        let vert_1_rc = vert_1_weak.upgrade().unwrap();
        let vert_1 = vert_1_rc.borrow();
        let node_1_weak = &vert_1.node;
        let node_1_rc = node_1_weak.upgrade().unwrap();
        let node_1 = node_1_rc.borrow();
        let vert_2_weak = &verts[1];
        let vert_2_rc = vert_2_weak.upgrade().unwrap();
        let vert_2 = vert_2_rc.borrow();
        let node_2_weak = &vert_2.node;
        let node_2_rc = node_2_weak.upgrade().unwrap();
        let node_2 = node_2_rc.borrow();
        let vert_3_weak = &verts[2];
        let vert_3_rc = vert_3_weak.upgrade().unwrap();
        let vert_3 = vert_3_rc.borrow();
        let node_3_weak = &vert_3.node;
        let node_3_rc = node_3_weak.upgrade().unwrap();
        let node_3 = node_3_rc.borrow();

        let rough_norm =
            (node_1.normal.unwrap() + node_2.normal.unwrap() + node_3.normal.unwrap()).normalize();
        return rough_norm;
    }

    fn calculate_face_normal(&self) -> glm::DVec3 {
        let verts = self.get_adjacent_verts();
        assert!(verts.len() >= 3);
        let vert_1_weak = &verts[0];
        let vert_1_rc = vert_1_weak.upgrade().unwrap();
        let vert_1 = vert_1_rc.borrow();
        let node_1_weak = &vert_1.node;
        let node_1_rc = node_1_weak.upgrade().unwrap();
        let node_1 = node_1_rc.borrow();
        let vert_2_weak = &verts[1];
        let vert_2_rc = vert_2_weak.upgrade().unwrap();
        let vert_2 = vert_2_rc.borrow();
        let node_2_weak = &vert_2.node;
        let node_2_rc = node_2_weak.upgrade().unwrap();
        let node_2 = node_2_rc.borrow();
        let vert_3_weak = &verts[2];
        let vert_3_rc = vert_3_weak.upgrade().unwrap();
        let vert_3 = vert_3_rc.borrow();
        let node_3_weak = &vert_3.node;
        let node_3_rc = node_3_weak.upgrade().unwrap();
        let node_3 = node_3_rc.borrow();

        return glm::triangle_normal(&node_1.pos, &node_2.pos, &node_3.pos);
    }
}

impl Edge {
    pub fn new() -> Edge {
        return Edge {
            verts: None,
            faces: Vec::new(),
        };
    }

    pub fn get_other_vert(&self, vert: &RefToVert) -> Option<RefToVert> {
        match &self.verts {
            None => return None,
            Some((v1, v2)) => {
                if v1.ptr_eq(&vert) {
                    return Some(v2.clone());
                } else if v2.ptr_eq(&vert) {
                    return Some(v1.clone());
                } else {
                    return None;
                }
            }
        }
    }

    pub fn has_face(&self, face: &RefToFace) -> bool {
        for face_weak in &self.faces {
            if face_weak.ptr_eq(face) {
                return true;
            }
        }
        return false;
    }

    pub fn swap_verts(&mut self) {
        self.verts = Some((
            self.verts.as_ref().unwrap().1.clone(),
            self.verts.as_ref().unwrap().0.clone(),
        ));
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
            normal: None,

            verts: Vec::new(),
        };
    }

    pub fn set_normal(&mut self, normal: glm::DVec3) {
        self.normal = Some(normal);
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

fn _debug_edges_print_order(edges: &Vec<Weak<RefCell<Edge>>>) {
    for (i, edge_weak) in edges.iter().enumerate() {
        let edge_rc = edge_weak.upgrade().unwrap();
        let edge = edge_rc.borrow();
        if i == 0 {
            print!(
                "{}->{}",
                edge.verts
                    .as_ref()
                    .unwrap()
                    .0
                    .upgrade()
                    .unwrap()
                    .borrow()
                    .id,
                edge.verts
                    .as_ref()
                    .unwrap()
                    .1
                    .upgrade()
                    .unwrap()
                    .borrow()
                    .id
            );
        } else {
            print!(
                "->{}",
                edge.verts
                    .as_ref()
                    .unwrap()
                    .1
                    .upgrade()
                    .unwrap()
                    .borrow()
                    .id
            );
        }
    }
    println!("");
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
        assert_eq!(mesh.edges.len(), 7);
        for edge in &mesh.edges {
            let len = edge.borrow().faces.len();
            assert!(len == 1 || len == 0);
            match edge.borrow().verts {
                Some(_) => assert!(true),
                None => assert!(false),
            }
        }
        assert_eq!(mesh.verts.len(), 7);
        for vert in &mesh.verts {
            let len = vert.borrow().edges.len();
            assert!(len == 1 || len == 2 || len == 3);
        }
        assert_eq!(mesh.nodes.len(), 5);
        for node in &mesh.nodes {
            let len = node.borrow().verts.len();
            assert!(len == 0 || len == 1 || len == 2);
        }
    }

    #[test]
    fn mesh_no_uv() {
        let mut mesh = Mesh::new();
        match mesh.read(&Path::new("tests/obj_test_05_square_no_uv.obj")) {
            Err(err) => match err {
                MeshError::NoUV => (),
                _ => assert!(false),
            },
            _ => assert!(false),
        }
    }
}
