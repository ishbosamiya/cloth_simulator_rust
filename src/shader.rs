use gl;
use nalgebra_glm as glm;

use std::fs::File;
use std::io::prelude::*;
use std::path::Path;

use crate::util::str_to_cstr;

pub struct Shader {
    program_id: gl::types::GLuint,
}

#[derive(Debug, Copy, Clone)]
pub enum ShaderError {
    Io,
    VertexCompile,
    FragmentCompile,
    ProgramLinker,
}

impl std::fmt::Display for ShaderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ShaderError::Io => write!(f, "io error"),
            ShaderError::VertexCompile => write!(f, "vertex_compile error"),
            ShaderError::FragmentCompile => write!(f, "fragment_compile error"),
            ShaderError::ProgramLinker => write!(f, "program_linker error"),
        }
    }
}

impl std::error::Error for ShaderError {}

impl Shader {
    pub fn new(
        vertex_shader_path: &Path,
        fragment_shader_path: &Path,
    ) -> Result<Shader, ShaderError> {
        let mut v_file = match File::open(vertex_shader_path) {
            Err(_) => return Err(ShaderError::Io),
            Ok(file) => file,
        };
        let mut f_file = match File::open(fragment_shader_path) {
            Err(_) => return Err(ShaderError::Io),
            Ok(file) => file,
        };

        let mut vertex_code = String::new();
        let mut fragment_code = String::new();
        match v_file.read_to_string(&mut vertex_code) {
            Err(_) => return Err(ShaderError::Io),
            Ok(_) => (),
        }
        match f_file.read_to_string(&mut fragment_code) {
            Err(_) => return Err(ShaderError::Io),
            Ok(_) => (),
        }

        let vertex_code = std::ffi::CString::new(vertex_code).unwrap();
        let vertex_shader: gl::types::GLuint;
        unsafe {
            vertex_shader = gl::CreateShader(gl::VERTEX_SHADER);
            gl::ShaderSource(vertex_shader, 1, &vertex_code.as_ptr(), std::ptr::null());
            gl::CompileShader(vertex_shader);
        }
        unsafe {
            let mut success: gl::types::GLint = -10;
            gl::GetShaderiv(vertex_shader, gl::COMPILE_STATUS, &mut success);
            if success != gl::TRUE.into() {
                eprintln!("vertex didn't compile");
                return Err(ShaderError::VertexCompile);
            }
        }
        let fragment_code = std::ffi::CString::new(fragment_code).unwrap();
        let fragment_shader: gl::types::GLuint;
        unsafe {
            fragment_shader = gl::CreateShader(gl::FRAGMENT_SHADER);
            gl::ShaderSource(
                fragment_shader,
                1,
                &fragment_code.as_ptr(),
                std::ptr::null(),
            );
            gl::CompileShader(fragment_shader);
        }
        unsafe {
            let mut success: gl::types::GLint = -10;
            gl::GetShaderiv(fragment_shader, gl::COMPILE_STATUS, &mut success);
            if success != gl::TRUE.into() {
                eprintln!("fragment didn't compile");
                return Err(ShaderError::FragmentCompile);
            }
        }
        let shader_program: gl::types::GLuint;
        unsafe {
            shader_program = gl::CreateProgram();
            gl::AttachShader(shader_program, vertex_shader);
            gl::AttachShader(shader_program, fragment_shader);
            gl::LinkProgram(shader_program);
        }
        unsafe {
            let mut success: gl::types::GLint = -10;
            gl::GetProgramiv(shader_program, gl::LINK_STATUS, &mut success);
            if success != gl::TRUE.into() {
                eprintln!("program not linked");
                return Err(ShaderError::ProgramLinker);
            }
        }

        unsafe {
            gl::DeleteShader(vertex_shader);
            gl::DeleteShader(fragment_shader);
        }

        return Ok(Shader {
            program_id: shader_program,
        });
    }

    pub fn use_shader(&self) {
        unsafe {
            gl::UseProgram(self.program_id);
        }
    }

    pub fn set_bool(&self, name: &str, value: bool) {
        unsafe {
            if value {
                gl::Uniform1i(
                    gl::GetUniformLocation(self.program_id, str_to_cstr(name).as_ptr()),
                    1,
                );
            } else {
                gl::Uniform1i(
                    gl::GetUniformLocation(self.program_id, str_to_cstr(name).as_ptr()),
                    0,
                );
            }
        }
    }

    pub fn set_int(&self, name: &str, value: gl::types::GLint) {
        unsafe {
            gl::Uniform1i(
                gl::GetUniformLocation(self.program_id, str_to_cstr(name).as_ptr()),
                value,
            );
        }
    }

    pub fn set_float(&self, name: &str, value: gl::types::GLfloat) {
        unsafe {
            gl::Uniform1f(
                gl::GetUniformLocation(self.program_id, str_to_cstr(name).as_ptr()),
                value,
            );
        }
    }

    pub fn set_vec2(&self, name: &str, value: &glm::Vec2) {
        unsafe {
            gl::Uniform2f(
                gl::GetUniformLocation(self.program_id, str_to_cstr(name).as_ptr()),
                value[0],
                value[1],
            );
        }
    }

    pub fn set_vec3(&self, name: &str, value: &glm::Vec3) {
        unsafe {
            gl::Uniform3f(
                gl::GetUniformLocation(self.program_id, str_to_cstr(name).as_ptr()),
                value[0],
                value[1],
                value[2],
            );
        }
    }

    pub fn set_vec4(&self, name: &str, value: &glm::Vec4) {
        unsafe {
            gl::Uniform4f(
                gl::GetUniformLocation(self.program_id, str_to_cstr(name).as_ptr()),
                value[0],
                value[1],
                value[2],
                value[3],
            );
        }
    }

    pub fn set_mat2(&self, name: &str, value: &glm::Mat2) {
        unsafe {
            gl::UniformMatrix2fv(
                gl::GetUniformLocation(self.program_id, str_to_cstr(name).as_ptr()),
                1,
                gl::FALSE,
                value.as_ptr(),
            );
        }
    }

    pub fn set_mat3(&self, name: &str, value: &glm::Mat3) {
        unsafe {
            gl::UniformMatrix3fv(
                gl::GetUniformLocation(self.program_id, str_to_cstr(name).as_ptr()),
                1,
                gl::FALSE,
                value.as_ptr(),
            );
        }
    }

    pub fn set_mat4(&self, name: &str, value: &glm::Mat4) {
        unsafe {
            gl::UniformMatrix4fv(
                gl::GetUniformLocation(self.program_id, str_to_cstr(name).as_ptr()),
                1,
                gl::FALSE,
                value.as_ptr(),
            );
        }
    }
}
