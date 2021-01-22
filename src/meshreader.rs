use nalgebra_glm as glm;

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

pub struct MeshReader {
    pub positions: Vec<glm::DVec3>,
    pub uvs: Vec<glm::DVec2>,
    pub normals: Vec<glm::DVec3>,
    pub face_position_indices: Vec<Vec<usize>>,
    pub face_uv_indices: Vec<Vec<usize>>,
    pub face_normal_indices: Vec<Vec<usize>>,
}

#[derive(Debug)]
pub enum MeshReaderError {
    Io(std::io::Error),
    InvalidFile,
    Unknown,
}

impl From<std::io::Error> for MeshReaderError {
    fn from(err: std::io::Error) -> MeshReaderError {
        return MeshReaderError::Io(err);
    }
}

impl std::fmt::Display for MeshReaderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MeshReaderError::Io(error) => write!(f, "io error {}", error),
            MeshReaderError::InvalidFile => write!(f, "invalid file"),
            MeshReaderError::Unknown => write!(f, "unknown error"),
        }
    }
}

impl std::error::Error for MeshReaderError {}

impl MeshReader {
    pub fn read(path: &Path) -> Result<MeshReader, MeshReaderError> {
        match path.extension() {
            Some(extension) => match extension.to_str().unwrap() {
                "obj" => MeshReader::read_obj(path),
                _ => return Err(MeshReaderError::Unknown),
            },
            None => return Err(MeshReaderError::Unknown),
        }
    }

    fn read_obj(path: &Path) -> Result<MeshReader, MeshReaderError> {
        let fin = File::open(path)?;
        let mut positions = Vec::new();
        let mut uvs = Vec::new();
        let mut normals = Vec::new();
        let mut face_position_indices = Vec::new();
        let mut face_uv_indices = Vec::new();
        let mut face_normal_indices = Vec::new();

        let reader = BufReader::new(fin);

        for line in reader.lines() {
            let line = line?;
            if line.starts_with("#") {
                continue;
            }
            let vals: Vec<&str> = line.split(" ").collect();
            assert!(vals.len() > 0);
            match vals[0] {
                "v" => {
                    // Don't currently support positions with 4 or more coordinates
                    assert!(vals.len() == 4);
                    let x: f64 = vals[1].parse().unwrap();
                    let y: f64 = vals[2].parse().unwrap();
                    let z: f64 = vals[3].parse().unwrap();
                    positions.push(glm::vec3(x, y, z));
                }
                "vn" => {
                    // Don't currently support positions with 4 or more coordinates
                    assert!(vals.len() == 4);
                    let x: f64 = vals[1].parse().unwrap();
                    let y: f64 = vals[2].parse().unwrap();
                    let z: f64 = vals[3].parse().unwrap();
                    normals.push(glm::vec3(x, y, z));
                }
                "vt" => {
                    // Don't currently support texture coordinates with 3 or more coordinates
                    assert!(vals.len() == 3);
                    let u: f64 = vals[1].parse().unwrap();
                    let v: f64 = vals[2].parse().unwrap();
                    uvs.push(glm::vec2(u, v));
                }
                "f" => {
                    // Don't currently support face with 2 or lesser verts
                    assert!(vals.len() >= 4);
                    let mut pos_indices: Vec<usize> = Vec::new();
                    let mut uv_indices: Vec<usize> = Vec::new();
                    let mut normal_indices: Vec<usize> = Vec::new();
                    for val in vals.iter().skip(1) {
                        let indices: Vec<&str> = val.split('/').collect();
                        match indices.len() {
                            // only positions
                            1 => {
                                let pos_index: usize = indices[0].parse().unwrap();
                                pos_indices.push(pos_index - 1);
                            }
                            // positions and texture coordinates
                            2 => {
                                let pos_index: usize = indices[0].parse().unwrap();
                                let uv_index: usize = indices[1].parse().unwrap();
                                pos_indices.push(pos_index - 1);
                                uv_indices.push(uv_index - 1);
                            }
                            3 => {
                                let pos_index: usize = indices[0].parse().unwrap();
                                let uv_index: usize = indices[1].parse().unwrap();
                                let normal_index: usize = indices[2].parse().unwrap();
                                pos_indices.push(pos_index - 1);
                                uv_indices.push(uv_index - 1);
                                normal_indices.push(normal_index - 1);
                            }
                            _ => {
                                return Err(MeshReaderError::InvalidFile);
                            }
                        }
                    }
                    assert!(pos_indices.len() != 0);
                    face_position_indices.push(pos_indices);
                    if uv_indices.len() != 0 {
                        face_uv_indices.push(uv_indices);
                    }
                    if normal_indices.len() != 0 {
                        face_normal_indices.push(normal_indices);
                    }
                }
                _ => {
                    continue;
                }
            }
        }

        return Ok(MeshReader {
            positions,
            uvs,
            normals,
            face_position_indices,
            face_uv_indices,
            face_normal_indices,
        });
    }
}
