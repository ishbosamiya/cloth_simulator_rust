use nalgebra_glm as glm;

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

pub struct MeshReader {
    pub positions: Vec<glm::DVec3>,
    pub uvs: Vec<glm::DVec2>,
    pub normals: Vec<glm::DVec3>,
    pub face_indices: Vec<Vec<(usize, usize, usize)>>,
    pub face_has_uv: bool,
    pub face_has_normal: bool,
    pub line_indices: Vec<Vec<usize>>,
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
        let mut face_indices = Vec::new();
        let mut face_has_uv = false;
        let mut face_has_normal = false;
        let mut line_indices = Vec::new();

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
                    let mut face_i: Vec<(usize, usize, usize)> = Vec::new();
                    for val in vals.iter().skip(1) {
                        let indices: Vec<&str> = val.split('/').collect();
                        match indices.len() {
                            // only positions
                            1 => {
                                let pos_index: usize = indices[0].parse().unwrap();
                                face_i.push((pos_index - 1, usize::MAX, usize::MAX));
                            }
                            // positions and texture coordinates
                            2 => {
                                let pos_index: usize = indices[0].parse().unwrap();
                                let uv_index: usize = indices[1].parse().unwrap();
                                face_i.push((pos_index - 1, uv_index - 1, usize::MAX));
                                face_has_uv = true;
                            }
                            // positions, texture coordinates and normals
                            3 => {
                                let pos_index: usize = indices[0].parse().unwrap();
                                let uv_index: usize;
                                if indices[1] != "" {
                                    uv_index = indices[1].parse().unwrap();
                                } else {
                                    uv_index = usize::MAX;
                                }
                                let normal_index: usize = indices[2].parse().unwrap();
                                if uv_index == usize::MAX {
                                    face_i.push((pos_index - 1, uv_index, normal_index - 1));
                                } else {
                                    face_i.push((pos_index - 1, uv_index - 1, normal_index - 1));
                                }
                                face_has_uv = true;
                                face_has_normal = true;
                            }
                            _ => {
                                return Err(MeshReaderError::InvalidFile);
                            }
                        }
                    }
                    assert!(face_i.len() != 0);
                    face_indices.push(face_i);
                }
                "l" => {
                    assert!(vals.len() >= 3);
                    let mut indices: Vec<usize> = Vec::new();
                    for val in vals.iter().skip(1) {
                        indices.push(val.parse().unwrap());
                    }
                    line_indices.push(indices);
                }
                _ => {
                    continue;
                }
            }
        }

        // TODO(ish): validate the indices

        return Ok(MeshReader {
            positions,
            uvs,
            normals,
            face_indices,
            face_has_uv,
            face_has_normal,
            line_indices,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn meshreader_read_obj_test_01() {
        let data = MeshReader::read_obj(&Path::new("tests/obj_test_01.obj")).unwrap();
        assert_eq!(data.positions.len(), 5);
        assert_eq!(data.uvs.len(), 6);
        assert_eq!(data.normals.len(), 2);
        assert_eq!(data.face_indices.len(), 2);
        assert_eq!(data.face_indices[0].len(), 3);
        assert_eq!(data.positions[0], glm::vec3(0.778921, 1.572047, -0.878382));
        assert_eq!(data.line_indices.len(), 1);
        assert_eq!(data.line_indices[0].len(), 2);
    }
    #[test]
    fn meshreader_read_obj_test_02() {
        match MeshReader::read_obj(&Path::new("tests/obj_test_02.obj")) {
            Err(error) => match error {
                MeshReaderError::InvalidFile => (),
                _ => panic!("Should have gotten an invalid file error"),
            },
            Ok(_) => panic!("Should have gotten an invalid file error"),
        }
    }
    #[test]
    fn meshreader_read_obj_test_03() {
        MeshReader::read_obj(&Path::new("tests/obj_test_03.obj")).unwrap();
    }
}
