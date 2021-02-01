use generational_arena::{Arena, Index};
use nalgebra_glm as glm;

use crate::gl_mesh::GLMesh;
use crate::meshreader::MeshReaderError;

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

    verts: AdjacentVerts,
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
pub struct NodeIndex(Index);
/// Index of Vert in Mesh.nodes
pub struct VertIndex(Index);
/// Index of Edge in Mesh.nodes
pub struct EdgeIndex(Index);
/// Index of Face in Mesh.nodes
pub struct FaceIndex(Index);

type IncidentVerts = Vec<VertIndex>;
type IncidentEdges = Vec<EdgeIndex>;
type IncidentFace = Option<FaceIndex>;
type AdjacentEdges = IncidentEdges;
type AdjacentVerts = Option<(VertIndex, VertIndex)>;

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

    pub fn add_empty_node(&mut self, pos: glm::DVec3) -> &mut Node {
        let node_index = self.nodes.insert_with(|self_index| {
            return Node::new(NodeIndex(self_index), pos);
        });
        return &mut self.nodes[node_index];
    }

    pub fn add_empty_vert(&mut self) -> &mut Vert {
        let vert_index = self.verts.insert_with(|self_index| {
            return Vert::new(VertIndex(self_index));
        });
        return &mut self.verts[vert_index];
    }

    pub fn add_empty_edge(&mut self) -> &mut Edge {
        let edge_index = self.edges.insert_with(|self_index| {
            return Edge::new(EdgeIndex(self_index));
        });
        return &mut self.edges[edge_index];
    }

    pub fn add_empty_face(&mut self) -> &mut Face {
        let face_index = self.faces.insert_with(|self_index| {
            return Face::new(FaceIndex(self_index));
        });
        return &mut self.faces[face_index];
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
}
