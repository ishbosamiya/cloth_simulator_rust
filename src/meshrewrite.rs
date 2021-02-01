use generational_arena::{Arena, Index};
use itertools::Itertools;
use nalgebra_glm as glm;

use std::path::Path;

use crate::gl_mesh::GLMesh;
use crate::meshreader::{MeshReader, MeshReaderError};

/// Node stores the world (3D) space coordinates
///
/// Each Node also optionally stores 3D space normal information
/// (commonly referred to as Vertex Normals)
///
/// Each Node can be referred to by many Verts
pub struct Node {
    self_index: NodeIndex,
    pub pos: glm::DVec3,
    pub normal: Option<glm::DVec3>,

    verts: IncidentVerts,
}

/// Vert stores the uv space coordinates
///
/// A Vert can only have one Node but this Node can be shared by many Verts
///
/// Each Vert can be referred to by many Edges
pub struct Vert {
    self_index: VertIndex,
    pub uv: Option<glm::DVec2>,

    node: Option<NodeIndex>,
    edges: IncidentEdges,
}

/// Edge stores the information gap between faces and indices to allow for faster access of adjacent face information
///
/// Each Edge has a pair of Verts (Made as Option because it may not
/// have this information when it first is created)
///
/// Each Edge also stores the Face that made that Edge, no two Faces
/// share an Edge to ensure correct winding of the Mesh can be
/// done. The verts stored will be the same but since their ordering
/// can differ based on the winding, unique Edges need to be created.
pub struct Edge {
    self_index: EdgeIndex,

    verts: Option<(VertIndex, VertIndex)>,
    face: IncidentFace,
}

/// Face stores the edges that form the Face. These edges are unique
/// (refer to documentation for Edge for further information)
///
/// Each Face also stores the face normal optionally
pub struct Face {
    self_index: FaceIndex,
    pub normal: Option<glm::DVec3>,

    edges: AdjacentEdges,
}

/// Mesh stores the Node, Vert, Edge, Face data in an Arena
///
/// Mesh optionally stores a renderable mesh, GLMesh
pub struct Mesh {
    nodes: Arena<Node>,
    verts: Arena<Vert>,
    edges: Arena<Edge>,
    faces: Arena<Face>,

    gl_mesh: Option<GLMesh>,
}

/// Index of Node in Mesh.nodes
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NodeIndex(Index);
/// Index of Vert in Mesh.nodes
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct VertIndex(Index);
/// Index of Edge in Mesh.nodes
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct EdgeIndex(Index);
/// Index of Face in Mesh.nodes
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct FaceIndex(Index);

type IncidentVerts = Vec<VertIndex>;
type IncidentEdges = Vec<EdgeIndex>;
type IncidentFace = Option<FaceIndex>;
type AdjacentEdges = IncidentEdges;
type AdjacentVerts = IncidentVerts;

/// Errors during operations on Mesh
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

impl Mesh {
    pub fn new() -> Mesh {
        return Mesh {
            nodes: Arena::new(),
            verts: Arena::new(),
            edges: Arena::new(),
            faces: Arena::new(),

            gl_mesh: None,
        };
    }

    /// Adds an empty Node and gives back mutable reference to it
    ///
    /// Use with caution
    fn add_empty_node(&mut self, pos: glm::DVec3) -> &mut Node {
        let node_index = self.nodes.insert_with(|self_index| {
            return Node::new(NodeIndex(self_index), pos);
        });
        return &mut self.nodes[node_index];
    }

    /// Adds an empty Vert and gives back mutable reference to it
    ///
    /// Use with caution
    fn add_empty_vert(&mut self) -> &mut Vert {
        let vert_index = self.verts.insert_with(|self_index| {
            return Vert::new(VertIndex(self_index));
        });
        return &mut self.verts[vert_index];
    }

    /// Adds an empty Vert and gives index of it
    ///
    /// Use with caution
    fn add_empty_vert_index(&mut self) -> VertIndex {
        let vert_index = self.verts.insert_with(|self_index| {
            return Vert::new(VertIndex(self_index));
        });
        return VertIndex(vert_index);
    }

    /// Adds an empty Edge and gives index of it
    ///
    /// Use with caution
    fn add_empty_edge_index(&mut self) -> EdgeIndex {
        let edge_index = self.edges.insert_with(|self_index| {
            return Edge::new(EdgeIndex(self_index));
        });
        return EdgeIndex(edge_index);
    }

    /// Adds an empty Face and gives index of it
    ///
    /// Use with caution
    fn add_empty_face_index(&mut self) -> FaceIndex {
        let face_index = self.faces.insert_with(|self_index| {
            return Face::new(FaceIndex(self_index));
        });
        return FaceIndex(face_index);
    }

    /// Gives set of vert indices that are adjacent to the face
    pub fn get_adjacent_vert_indices(&self, face: &Face) -> Option<AdjacentVerts> {
        assert!(self.edges.len() > 2);
        let mut adjacent_verts = Vec::new();

        for edge_index in &face.edges {
            let edge = self.edges.get(edge_index.0)?;
            let (vert_1, vert_2) = edge.verts.as_ref()?;
            add_as_set(&mut adjacent_verts, *vert_1);
            add_as_set(&mut adjacent_verts, *vert_2);
        }

        return Some(adjacent_verts);
    }

    /// Gives the list of connecting edges indices if there exists at least one
    pub fn get_connecting_edges_indices(
        &self,
        vert_1_index: VertIndex,
        vert_2_index: VertIndex,
    ) -> Option<IncidentEdges> {
        let mut edges = Vec::new();

        for edge_index in &self.verts.get(vert_1_index.0)?.edges {
            let edge = self.edges.get(edge_index.0)?;
            if edge.has_vert(vert_2_index) {
                edges.push(*edge_index);
            }
        }

        if edges.len() > 0 {
            return Some(edges);
        }
        return None;
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
                if data.face_has_normal && data.normals.len() > 0 {
                    node.set_normal(data.normals[*normal_index]);
                }
            }

            let mut face_edges = Vec::new();

            // Update edges
            for ((_, vert_1_index, _), (_, vert_2_index, _)) in
                face_i.into_iter().circular_tuple_windows()
            {
                // Always create a new edge to ensure that the mesh's faces
                // can be oriented, otherwise there will be mesh winding cannot be made consistent
                let edge_index = self.add_empty_edge_index();
                let edge = self.edges.get_mut(edge_index.0).unwrap();
                let vert_1_index = self.verts.get_unknown_gen_mut(vert_1_index).unwrap().1;
                let vert_2_index = self.verts.get_unknown_gen_mut(vert_2_index).unwrap().1;

                // Update edge with vert
                edge.verts = Some((VertIndex(vert_1_index), VertIndex(vert_2_index)));

                // Update vert with edge
                let vert_1 = self.verts.get_mut(vert_1_index).unwrap();
                vert_1.edges.push(edge.self_index);
                let vert_2 = self.verts.get_mut(vert_2_index).unwrap();
                vert_2.edges.push(edge.self_index);

                face_edges.push(edge.self_index);
            }

            // Update faces
            {
                let face_index = self.add_empty_face_index();
                let face = self.faces.get_mut(face_index.0).unwrap();
                // Update face with edge
                face.edges = face_edges;

                // Update edges with face
                for edge_index in &face.edges {
                    let edge = self.edges.get_mut(edge_index.0).unwrap();
                    edge.face = Some(face.self_index);
                }
            }
        }

        // Any node without a vert gets a new vert without uv
        let mut loose_nodes = Vec::new();
        self.nodes
            .iter()
            .filter(|(_, node)| node.verts.len() == 0)
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

        return Ok(());
    }
}

impl Face {
    pub fn new(self_index: FaceIndex) -> Face {
        return Face {
            self_index,
            normal: None,

            edges: Vec::new(),
        };
    }
}

impl Edge {
    pub fn new(self_index: EdgeIndex) -> Edge {
        return Edge {
            self_index,

            verts: None,
            face: None,
        };
    }

    /// Checks if self has the vert specified via VertIndex
    pub fn has_vert(&self, vert_index: VertIndex) -> bool {
        match self.verts {
            Some((v1_index, v2_index)) => {
                if v1_index == vert_index {
                    return true;
                } else if v2_index == vert_index {
                    return true;
                } else {
                    return false;
                }
            }
            None => {
                return false;
            }
        }
    }

    /// Returns the other vert's index given that a valid index (an
    /// index part of self.verts) otherwise returns None
    pub fn get_other_vert_index(&self, vert_index: VertIndex) -> Option<VertIndex> {
        match self.verts {
            Some((v1_index, v2_index)) => {
                if v1_index == vert_index {
                    return Some(v2_index);
                } else if v2_index == vert_index {
                    return Some(v1_index);
                } else {
                    return None;
                }
            }
            None => return None,
        }
    }

    /// Swaps the ordering of the vert indices in self.verts if it exists
    pub fn swap_verts(&mut self) {
        match self.verts {
            Some((v1_index, v2_index)) => {
                self.verts = Some((v2_index, v1_index));
            }
            _ => (),
        }
    }
}

impl Vert {
    pub fn new(self_index: VertIndex) -> Vert {
        return Vert {
            self_index,
            uv: None,

            node: None,
            edges: Vec::new(),
        };
    }
}

impl Node {
    pub fn new(self_index: NodeIndex, pos: glm::DVec3) -> Node {
        return Node {
            self_index,
            pos,
            normal: None,

            verts: Vec::new(),
        };
    }

    pub fn set_normal(&mut self, normal: glm::DVec3) {
        self.normal = Some(normal);
    }
}

fn add_as_set<T>(vec: &mut Vec<T>, val: T)
where
    T: PartialEq,
{
    if vec.contains(&val) == false {
        vec.push(val);
    }
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
        for (_, face) in &mesh.faces {
            assert_eq!(face.edges.len(), 3);
        }
        assert_eq!(mesh.edges.len(), 7);
        for (_, edge) in &mesh.edges {
            match edge.verts {
                Some(_) => assert!(true),
                None => assert!(false),
            }
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
