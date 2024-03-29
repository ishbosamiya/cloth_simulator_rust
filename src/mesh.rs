use generational_arena::{Arena, Index};
use itertools::Itertools;
use nalgebra_glm as glm;

use std::convert::TryInto;
use std::path::Path;

use crate::bvh::{BVHTree, AABB};
use crate::drawable::Drawable;
use crate::gl_mesh::{GLMesh, GLVert};
use crate::gpu_immediate::*;
use crate::meshreader::{MeshReader, MeshReaderError};
use crate::shader::Shader;

/// Node stores the world (3D) space coordinates
///
/// Each Node also optionally stores 3D space normal information
/// (commonly referred to as Vertex Normals)
///
/// Each Node can be referred to by many Verts
pub struct Node<T> {
    self_index: NodeIndex,
    pub pos: glm::DVec3,
    pub normal: Option<glm::DVec3>,
    pub extra_data: Option<T>,

    verts: IncidentVerts,
}

/// Vert stores the uv space coordinates
///
/// A Vert can only have one Node but this Node can be shared by many Verts
///
/// Each Vert can be referred to by many Edges
pub struct Vert<T> {
    self_index: VertIndex,
    pub uv: Option<glm::DVec2>,
    pub extra_data: Option<T>,

    node: Option<NodeIndex>,
    edges: IncidentEdges,
}

/// Edge stores the information gap between faces and vertices to allow for faster access of adjacent face information
///
/// Each Edge has a pair of Verts (Made as Option because it may not
/// have this information when it first is created)
pub struct Edge<T> {
    self_index: EdgeIndex,
    pub extra_data: Option<T>,

    verts: Option<(VertIndex, VertIndex)>,
    faces: IncidentFaces,
}

/// Face stores the vertices in order that form that face, this is done instead of storing edges to prevent winding/orientation problems with the mesh.
///
/// Each Face also stores the face normal optionally
pub struct Face<T> {
    self_index: FaceIndex,
    pub normal: Option<glm::DVec3>,
    pub extra_data: Option<T>,

    verts: AdjacentVerts,
}

/// Mesh stores the Node, Vert, Edge, Face data in an Arena
///
/// Mesh optionally stores a renderable mesh, GLMesh
pub struct Mesh<END, EVD, EED, EFD> {
    nodes: Arena<Node<END>>,
    verts: Arena<Vert<EVD>>,
    edges: Arena<Edge<EED>>,
    faces: Arena<Face<EFD>>,

    gl_mesh: Option<GLMesh>,
    bvh: Option<BVHTree<FaceIndex>>,
}

/// Index of Node in Mesh.nodes
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NodeIndex(pub Index);
/// Index of Vert in Mesh.nodes
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct VertIndex(pub Index);
/// Index of Edge in Mesh.nodes
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct EdgeIndex(pub Index);
/// Index of Face in Mesh.nodes
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct FaceIndex(pub Index);

type IncidentVerts = Vec<VertIndex>;
type IncidentEdges = Vec<EdgeIndex>;
type IncidentFaces = Vec<FaceIndex>;
type AdjacentVerts = IncidentVerts;

/// Errors during operations on Mesh
#[derive(Debug)]
pub enum MeshError {
    MeshReader(MeshReaderError),
    NoUV,
}

impl From<MeshReaderError> for MeshError {
    fn from(err: MeshReaderError) -> MeshError {
        MeshError::MeshReader(err)
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

impl<END, EVD, EED, EFD> Default for Mesh<END, EVD, EED, EFD> {
    fn default() -> Self {
        Mesh::new()
    }
}

impl<END, EVD, EED, EFD> Mesh<END, EVD, EED, EFD> {
    pub fn new() -> Mesh<END, EVD, EED, EFD> {
        Mesh {
            nodes: Arena::new(),
            verts: Arena::new(),
            edges: Arena::new(),
            faces: Arena::new(),

            gl_mesh: None,
            bvh: None,
        }
    }

    pub fn get_faces(&self) -> &Arena<Face<EFD>> {
        &self.faces
    }

    pub fn get_edges(&self) -> &Arena<Edge<EED>> {
        &self.edges
    }

    pub fn get_verts(&self) -> &Arena<Vert<EVD>> {
        &self.verts
    }

    pub fn get_nodes(&self) -> &Arena<Node<END>> {
        &self.nodes
    }

    pub fn get_faces_mut(&mut self) -> &mut Arena<Face<EFD>> {
        &mut self.faces
    }

    pub fn get_edges_mut(&mut self) -> &mut Arena<Edge<EED>> {
        &mut self.edges
    }

    pub fn get_verts_mut(&mut self) -> &mut Arena<Vert<EVD>> {
        &mut self.verts
    }

    pub fn get_nodes_mut(&mut self) -> &mut Arena<Node<END>> {
        &mut self.nodes
    }

    pub fn get_face(&self, index: FaceIndex) -> Option<&Face<EFD>> {
        self.faces.get(index.0)
    }

    pub fn get_edge(&self, index: EdgeIndex) -> Option<&Edge<EED>> {
        self.edges.get(index.0)
    }

    pub fn get_vert(&self, index: VertIndex) -> Option<&Vert<EVD>> {
        self.verts.get(index.0)
    }

    pub fn get_node(&self, index: NodeIndex) -> Option<&Node<END>> {
        self.nodes.get(index.0)
    }

    pub fn get_face_mut(&mut self, index: FaceIndex) -> Option<&mut Face<EFD>> {
        self.faces.get_mut(index.0)
    }

    pub fn get_edge_mut(&mut self, index: EdgeIndex) -> Option<&mut Edge<EED>> {
        self.edges.get_mut(index.0)
    }

    pub fn get_vert_mut(&mut self, index: VertIndex) -> Option<&mut Vert<EVD>> {
        self.verts.get_mut(index.0)
    }

    pub fn get_node_mut(&mut self, index: NodeIndex) -> Option<&mut Node<END>> {
        self.nodes.get_mut(index.0)
    }

    pub fn get_bvh(&self) -> &Option<BVHTree<FaceIndex>> {
        &self.bvh
    }

    /// Adds an empty Node and gives back mutable reference to it
    ///
    /// Use with caution
    fn add_empty_node(&mut self, pos: glm::DVec3) -> &mut Node<END> {
        let node_index = self
            .nodes
            .insert_with(|self_index| Node::new(NodeIndex(self_index), pos));
        &mut self.nodes[node_index]
    }

    /// Adds an empty Vert and gives back mutable reference to it
    ///
    /// Use with caution
    fn add_empty_vert(&mut self) -> &mut Vert<EVD> {
        let vert_index = self
            .verts
            .insert_with(|self_index| Vert::new(VertIndex(self_index)));
        &mut self.verts[vert_index]
    }

    /// Adds an empty Vert and gives index of it
    ///
    /// Use with caution
    fn add_empty_vert_index(&mut self) -> VertIndex {
        let vert_index = self
            .verts
            .insert_with(|self_index| Vert::new(VertIndex(self_index)));
        VertIndex(vert_index)
    }

    /// Adds an empty Edge and gives index of it
    ///
    /// Use with caution
    fn add_empty_edge_index(&mut self) -> EdgeIndex {
        let edge_index = self
            .edges
            .insert_with(|self_index| Edge::new(EdgeIndex(self_index)));
        EdgeIndex(edge_index)
    }

    /// Adds an empty Face and gives index of it
    ///
    /// Use with caution
    fn add_empty_face_index(&mut self) -> FaceIndex {
        let face_index = self
            .faces
            .insert_with(|self_index| Face::new(FaceIndex(self_index)));
        FaceIndex(face_index)
    }

    /// Gives the connecting edge index if there exists one
    pub fn get_connecting_edge_index(
        &self,
        vert_1_index: VertIndex,
        vert_2_index: VertIndex,
    ) -> Option<EdgeIndex> {
        for edge_index in &self.verts.get(vert_1_index.0)?.edges {
            let edge = self.edges.get(edge_index.0)?;
            if edge.has_vert(vert_2_index) {
                return Some(*edge_index);
            }
        }

        None
    }

    pub fn read(&mut self, path: &Path) -> Result<(), MeshError> {
        let data = MeshReader::read(path)?;

        if data.uvs.is_empty() || !data.face_has_uv {
            return Err(MeshError::NoUV);
        }

        // Create all the nodes
        for pos in data.positions {
            self.add_empty_node(pos);
        }

        // Create all the verts
        for uv in data.uvs {
            let vert = self.add_empty_vert();
            vert.uv = Some(uv);
        }

        // Work with the face indices that have been read to form the edges and faces
        for face_i in data.face_indices {
            // Update verts and nodes
            for (pos_index, uv_index, normal_index) in &face_i {
                let vert = self.verts.get_unknown_gen_mut(*uv_index).unwrap().0;
                let node = self.nodes.get_unknown_gen_mut(*pos_index).unwrap().0;

                // Update vert with node
                vert.node = Some(node.self_index);

                // Update node with vert
                node.verts.push(vert.self_index);
                // If MeshReader has found "vertex normal" information, store it in the Node
                if data.face_has_normal && !data.normals.is_empty() {
                    node.set_normal(data.normals[*normal_index]);
                }
            }

            let mut face_edges = Vec::new();
            let mut face_verts = Vec::new();

            // Update edges
            for ((_, vert_1_index, _), (_, vert_2_index, _)) in
                face_i.into_iter().circular_tuple_windows()
            {
                let vert_1_index = self.verts.get_unknown_gen_mut(vert_1_index).unwrap().1;
                let vert_2_index = self.verts.get_unknown_gen_mut(vert_2_index).unwrap().1;
                match self
                    .get_connecting_edge_index(VertIndex(vert_1_index), VertIndex(vert_2_index))
                {
                    Some(edge_index) => {
                        let edge = self.edges.get(edge_index.0).unwrap();
                        face_edges.push(edge.self_index);
                    }
                    None => {
                        let edge_index = self.add_empty_edge_index();
                        let edge = self.edges.get_mut(edge_index.0).unwrap();
                        // Update edge with vert
                        edge.verts = Some((VertIndex(vert_1_index), VertIndex(vert_2_index)));
                        // Update vert with edge
                        let vert_1 = self.verts.get_mut(vert_1_index).unwrap();
                        vert_1.edges.push(edge.self_index);
                        let vert_2 = self.verts.get_mut(vert_2_index).unwrap();
                        vert_2.edges.push(edge.self_index);
                        face_edges.push(edge.self_index);
                    }
                }

                face_verts.push(VertIndex(vert_1_index));
            }

            // Update faces
            {
                let face_index = self.add_empty_face_index();
                let face = self.faces.get_mut(face_index.0).unwrap();
                // Update face with verts
                face.verts = face_verts;

                // Update edges with face
                for edge_index in &face_edges {
                    let edge = self.edges.get_mut(edge_index.0).unwrap();
                    edge.faces.push(face.self_index);
                }
            }
        }

        // Any node without a vert gets a new vert without uv
        let mut loose_nodes = Vec::new();
        self.nodes
            .iter()
            .filter(|(_, node)| node.verts.is_empty())
            .for_each(|(_, node)| {
                loose_nodes.push(node.self_index);
            });
        for node_index in loose_nodes {
            let vert_index = self.add_empty_vert_index();
            let vert = self.verts.get_mut(vert_index.0).unwrap();
            let node = self.nodes.get_mut(node_index.0).unwrap();
            vert.node = Some(node.self_index);
            node.verts.push(vert.self_index);
        }

        // Add lines to the mesh
        for line in data.line_indices {
            for (node_index_1, node_index_2) in line.iter().tuple_windows() {
                // Since lines don't store the UV information, we take
                // the nodes' first vert to create the edge
                let edge_index = self.add_empty_edge_index();
                let edge = self.edges.get_mut(edge_index.0).unwrap();

                let node_1 = self.nodes.get_unknown_gen(*node_index_1).unwrap().0;
                let node_2 = self.nodes.get_unknown_gen(*node_index_2).unwrap().0;

                let vert_1 = self.verts.get(node_1.verts[0].0).unwrap();
                let vert_2 = self.verts.get(node_2.verts[0].0).unwrap();

                // Update edge with verts
                edge.verts = Some((vert_1.self_index, vert_2.self_index));

                // Update verts with edge
                let vert_1 = self.verts.get_mut(node_1.verts[0].0).unwrap();
                vert_1.edges.push(edge.self_index);
                let vert_2 = self.verts.get_mut(node_2.verts[0].0).unwrap();
                vert_2.edges.push(edge.self_index);
            }
        }

        Ok(())
    }

    pub fn generate_gl_mesh(&mut self, use_face_normal: bool) {
        #[inline]
        fn store_in_gl_vert<END, EVD, EFD>(
            gl_verts: &mut Vec<GLVert>,
            vert: &Vert<EVD>,
            node: &Node<END>,
            face: &Face<EFD>,
            use_face_normal: bool,
        ) {
            match vert.uv {
                Some(uv) => {
                    if use_face_normal {
                        match face.normal {
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

        let mut gl_verts: Vec<GLVert> = Vec::with_capacity(self.verts.len());
        let mut gl_indices: Vec<gl::types::GLuint> = Vec::with_capacity(self.faces.len() * 3);

        for (_, face) in &self.faces {
            let verts = &face.verts;

            let vert_1_index = verts[0];
            let vert_1 = self.verts.get(vert_1_index.0).unwrap();
            let node_1 = self.nodes.get(vert_1.node.unwrap().0).unwrap();

            let id1 = gl_verts.len();
            store_in_gl_vert(&mut gl_verts, vert_1, node_1, face, use_face_normal);

            for (vert_2_index, vert_3_index) in verts.iter().skip(1).tuple_windows() {
                let vert_2 = self.verts.get(vert_2_index.0).unwrap();
                let vert_3 = self.verts.get(vert_3_index.0).unwrap();

                let node_2 = self.nodes.get(vert_2.node.unwrap().0).unwrap();
                let node_3 = self.nodes.get(vert_3.node.unwrap().0).unwrap();

                let id2 = gl_verts.len();
                store_in_gl_vert(&mut gl_verts, vert_2, node_2, face, use_face_normal);
                let id3 = gl_verts.len();
                store_in_gl_vert(&mut gl_verts, vert_3, node_3, face, use_face_normal);

                gl_indices.push(id1.try_into().unwrap());
                gl_indices.push(id2.try_into().unwrap());
                gl_indices.push(id3.try_into().unwrap());
            }
        }

        self.gl_mesh = Some(GLMesh::new(gl_verts, gl_indices));
    }

    pub fn build_bvh(&mut self, epsilon: f64) {
        let mut bvh = BVHTree::new(self.faces.len(), epsilon, 4, 8);

        for (face_index, face) in self.get_faces() {
            let mut co = Vec::new();
            for vert_index in face.get_verts() {
                let vert = self.get_vert(*vert_index).unwrap();
                let node = self.get_node(vert.node.unwrap()).unwrap();
                co.push(node.pos);
            }
            bvh.insert(FaceIndex(face_index), co);
        }

        bvh.balance();

        self.bvh = Some(bvh);
    }

    /// This only works if the mesh connectivity remains the same,
    /// there isn't a way to know the mesh connectivity change easily
    /// as of right now so there will be some unexpected errors if
    /// there has been change in connectivity
    pub fn update_bvh(&mut self) {
        for (i, (_, face)) in self.faces.iter().enumerate() {
            let mut co = Vec::new();
            for vert_index in face.get_verts() {
                let vert = self.get_vert(*vert_index).unwrap();
                let node = self.get_node(vert.node.unwrap()).unwrap();
                co.push(node.pos);
            }
            let bvh = self.bvh.as_mut().unwrap();
            bvh.update_node(i, co, Vec::new()).unwrap();
        }

        let bvh = self.bvh.as_mut().unwrap();
        bvh.update_tree();
    }
}

#[derive(Debug, Copy, Clone)]
pub enum MeshDrawError {
    GenerateGLMeshFirst,
    ErrorWhileDrawing,
}

pub struct MeshDrawData<'a> {
    imm: &'a mut GPUImmediate,
    shader: &'a Shader,
}

impl<'a> MeshDrawData<'a> {
    pub fn new(imm: &'a mut GPUImmediate, shader: &'a Shader) -> Self {
        MeshDrawData { imm, shader }
    }
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
        MeshDrawError::ErrorWhileDrawing
    }
}

impl<END, EVD, EED, EFD> Drawable<MeshDrawData<'_>, MeshDrawError> for Mesh<END, EVD, EED, EFD> {
    fn draw(&self, draw_data: &mut MeshDrawData) -> Result<(), MeshDrawError> {
        let imm = &mut draw_data.imm;
        let shader = &draw_data.shader;
        shader.use_shader();

        let format = imm.get_cleared_vertex_format();
        let pos_attr = format.add_attribute(
            "in_pos\0".to_string(),
            GPUVertCompType::F32,
            3,
            GPUVertFetchMode::Float,
        );
        // let uv_attr = format.add_attribute(
        //     "in_uv\0".to_string(),
        //     GPUVertCompType::F32,
        //     2,
        //     GPUVertFetchMode::Float,
        // );
        let normal_attr = format.add_attribute(
            "in_normal\0".to_string(),
            GPUVertCompType::F32,
            3,
            GPUVertFetchMode::Float,
        );

        imm.begin_at_most(GPUPrimType::Tris, self.faces.len() * 10, shader);

        for (_, face) in &self.faces {
            let verts = &face.verts;
            let vert_1_index = verts[0];
            let vert_1 = self.verts.get(vert_1_index.0).unwrap();
            let node_1 = self.nodes.get(vert_1.node.unwrap().0).unwrap();
            for (vert_2_index, vert_3_index) in verts.iter().skip(1).tuple_windows() {
                let vert_2 = self.verts.get(vert_2_index.0).unwrap();
                let vert_3 = self.verts.get(vert_3_index.0).unwrap();

                let node_2 = self.nodes.get(vert_2.node.unwrap().0).unwrap();
                let node_3 = self.nodes.get(vert_3.node.unwrap().0).unwrap();

                let node_1_normal: glm::Vec3 = glm::convert(node_1.normal.unwrap());
                imm.attr_3f(
                    normal_attr,
                    node_1_normal[0],
                    node_1_normal[1],
                    node_1_normal[2],
                );
                // imm.attr_2f(uv_attr, 0.0, 0.0);
                let node_1_pos: glm::Vec3 = glm::convert(node_1.pos);
                imm.vertex_3f(pos_attr, node_1_pos[0], node_1_pos[1], node_1_pos[2]);

                let node_2_normal: glm::Vec3 = glm::convert(node_2.normal.unwrap());
                imm.attr_3f(
                    normal_attr,
                    node_2_normal[0],
                    node_2_normal[1],
                    node_2_normal[2],
                );
                // imm.attr_2f(uv_attr, 0.0, 0.0);
                let node_2_pos: glm::Vec3 = glm::convert(node_2.pos);
                imm.vertex_3f(pos_attr, node_2_pos[0], node_2_pos[1], node_2_pos[2]);

                let node_3_normal: glm::Vec3 = glm::convert(node_3.normal.unwrap());
                imm.attr_3f(
                    normal_attr,
                    node_3_normal[0],
                    node_3_normal[1],
                    node_3_normal[2],
                );
                // imm.attr_2f(uv_attr, 0.0, 0.0);
                let node_3_pos: glm::Vec3 = glm::convert(node_3.pos);
                imm.vertex_3f(pos_attr, node_3_pos[0], node_3_pos[1], node_3_pos[2]);
            }
        }

        imm.end();

        Ok(())
    }

    fn draw_wireframe(&self, draw_data: &mut MeshDrawData) -> Result<(), MeshDrawError> {
        let imm = &mut draw_data.imm;
        let shader = &draw_data.shader;
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

        imm.begin(GPUPrimType::Lines, self.edges.len() * 2, shader);

        for (_, edge) in &self.edges {
            let (vert_1_index, vert_2_index) = edge.get_verts().unwrap();
            let vert_1 = self.verts.get(vert_1_index.0).unwrap();
            let vert_2 = self.verts.get(vert_2_index.0).unwrap();
            let node_1_index = vert_1.node.unwrap();
            let node_2_index = vert_2.node.unwrap();
            let node_1 = self.nodes.get(node_1_index.0).unwrap();
            let node_2 = self.nodes.get(node_2_index.0).unwrap();
            let node_1_pos: glm::Vec3 = glm::convert(node_1.pos);
            let node_2_pos: glm::Vec3 = glm::convert(node_2.pos);

            imm.attr_4f(color_attr, 0.8, 0.8, 0.8, 1.0);
            imm.vertex_3f(pos_attr, node_1_pos[0], node_1_pos[1], node_1_pos[2]);
            imm.attr_4f(color_attr, 1.0, 1.0, 1.0, 1.0);
            imm.vertex_3f(pos_attr, node_2_pos[0], node_2_pos[1], node_2_pos[2]);
        }

        imm.end();

        Ok(())
    }
}

impl<END, EVD, EED, EFD> AABB for Mesh<END, EVD, EED, EFD> {
    type ElementIndex = FaceIndex;

    fn give_aabb(&self, face_index: Self::ElementIndex) -> Vec<f64> {
        let epsilon = 0.01;
        let mut bv = vec![
            f64::MAX,
            -f64::MAX,
            f64::MAX,
            -f64::MAX,
            f64::MAX,
            -f64::MAX,
        ];
        let face = self.get_face(face_index).unwrap();
        for vert_index in face.get_verts() {
            let vert = self.get_vert(*vert_index).unwrap();
            let node = self.get_node(vert.node.unwrap()).unwrap();
            let pos = &node.pos;

            for axis_iter in 0..3 {
                let new_min_max = pos[axis_iter];
                if new_min_max < bv[(2 * axis_iter)] {
                    bv[(2 * axis_iter)] = new_min_max - epsilon;
                }
                if new_min_max > bv[(2 * axis_iter) + 1] {
                    bv[(2 * axis_iter) + 1] = new_min_max + epsilon;
                }
            }
        }

        bv
    }
}

impl<T> Face<T> {
    pub fn new(self_index: FaceIndex) -> Face<T> {
        Face {
            self_index,
            normal: None,
            extra_data: None,

            verts: Vec::new(),
        }
    }

    pub fn get_verts(&self) -> &AdjacentVerts {
        &self.verts
    }
}

impl<T> Edge<T> {
    pub fn new(self_index: EdgeIndex) -> Edge<T> {
        Edge {
            self_index,
            extra_data: None,

            verts: None,
            faces: Vec::new(),
        }
    }

    pub fn get_self_index(&self) -> EdgeIndex {
        self.self_index
    }

    pub fn get_verts(&self) -> &Option<(VertIndex, VertIndex)> {
        &self.verts
    }

    /// Checks if self has the vert specified via VertIndex
    pub fn has_vert(&self, vert_index: VertIndex) -> bool {
        match self.verts {
            Some((v1_index, v2_index)) => {
                if v1_index == vert_index {
                    true
                } else {
                    v2_index == vert_index
                }
            }
            None => false,
        }
    }

    /// Returns the other vert's index given that a valid index (an
    /// index part of self.verts) otherwise returns None
    pub fn get_other_vert_index(&self, vert_index: VertIndex) -> Option<VertIndex> {
        match self.verts {
            Some((v1_index, v2_index)) => {
                if v1_index == vert_index {
                    Some(v2_index)
                } else if v2_index == vert_index {
                    Some(v1_index)
                } else {
                    None
                }
            }
            None => None,
        }
    }

    /// Swaps the ordering of the vert indices in self.verts if it exists
    pub fn swap_verts(&mut self) {
        if let Some((v1_index, v2_index)) = self.verts {
            self.verts = Some((v2_index, v1_index));
        }
    }
}

impl<T> Vert<T> {
    pub fn new(self_index: VertIndex) -> Vert<T> {
        Vert {
            self_index,
            uv: None,
            extra_data: None,

            node: None,
            edges: Vec::new(),
        }
    }

    pub fn get_node_index(&self) -> Option<NodeIndex> {
        self.node
    }
}

impl<T> Node<T> {
    pub fn new(self_index: NodeIndex, pos: glm::DVec3) -> Node<T> {
        Node {
            self_index,
            pos,
            normal: None,
            extra_data: None,

            verts: Vec::new(),
        }
    }

    pub fn set_normal(&mut self, normal: glm::DVec3) {
        self.normal = Some(normal);
    }
}

pub mod simple {
    pub type Node = super::Node<()>;
    pub type Vert = super::Vert<()>;
    pub type Edge = super::Edge<()>;
    pub type Face = super::Face<()>;
    pub type Mesh = super::Mesh<(), (), (), ()>;
}

fn _add_as_set<T>(vec: &mut Vec<T>, val: T)
where
    T: PartialEq,
{
    if vec.contains(&val) {
        return;
    }
    vec.push(val);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mesh_read_test() {
        // TODO(ish): add more comprehensive relation tests
        let mut mesh = simple::Mesh::new();
        mesh.read(&Path::new("tests/obj_test_01.obj")).unwrap();
        assert_eq!(mesh.faces.len(), 2);
        for (_, face) in &mesh.faces {
            assert_eq!(face.verts.len(), 3);
        }
        assert_eq!(mesh.edges.len(), 7);
        for (_, edge) in &mesh.edges {
            assert!(edge.verts.is_some());
        }
        assert_eq!(mesh.verts.len(), 7);
        for (_, vert) in &mesh.verts {
            let len = vert.edges.len();
            assert!(len == 1 || len == 2 || len == 3);
        }
        assert_eq!(mesh.nodes.len(), 5);
        for (_, node) in &mesh.nodes {
            let len = node.verts.len();
            assert!(len == 0 || len == 1 || len == 2);
        }
    }

    #[test]
    fn mesh_no_uv() {
        let mut mesh = simple::Mesh::new();
        let res = mesh.read(&Path::new("tests/obj_test_05_square_no_uv.obj"));
        if let Err(err) = res {
            match err {
                MeshError::NoUV => {}
                _ => unreachable!(),
            }
        } else {
            unreachable!()
        }
    }
}
