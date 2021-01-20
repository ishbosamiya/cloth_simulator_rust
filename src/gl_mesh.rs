use gl;
use nalgebra_glm as glm;
use std::convert::TryInto;

#[repr(C, packed)]
#[derive(Debug, Copy, Clone)]
pub struct GLVert {
    pub pos: glm::Vec3,
    pub uv: glm::Vec2,
    pub normal: glm::Vec3,
}

impl GLVert {
    pub fn new(pos: glm::Vec3, uv: glm::Vec2, normal: glm::Vec3) -> GLVert {
        return GLVert {
            pos,
            normal: normal,
            uv: uv,
        };
    }
}

pub struct GLMesh {
    verts: Vec<GLVert>,
    indices: Vec<gl::types::GLuint>,
    vao: Option<gl::types::GLuint>,
    vbo: Option<gl::types::GLuint>,
    ebo: Option<gl::types::GLuint>,
}

impl GLMesh {
    pub fn new(verts: Vec<GLVert>, indices: Vec<gl::types::GLuint>) -> GLMesh {
        let mut mesh = GLMesh {
            verts,
            indices,
            vao: None,
            vbo: None,
            ebo: None,
        };

        mesh.setup();

        return mesh;
    }

    pub fn draw(&self) {
        unsafe {
            gl::BindVertexArray(self.vao.unwrap());
            gl::DrawElements(
                gl::TRIANGLES,
                self.indices.len().try_into().unwrap(),
                gl::UNSIGNED_INT,
                std::ptr::null(),
            );
            gl::BindVertexArray(0);
        }
    }

    fn setup(&mut self) {
        unsafe {
            let mut vao: gl::types::GLuint = 0;
            let mut vbo: gl::types::GLuint = 0;
            let mut ebo: gl::types::GLuint = 0;
            // generate the buffers needed
            gl::GenVertexArrays(1, &mut vao);
            gl::GenBuffers(1, &mut vbo);
            gl::GenBuffers(1, &mut ebo);

            if vao == 0 || vbo == 0 || ebo == 0 {
                panic!("vao, vbo, or ebo couldn't be initialized");
            }
            self.vao = Some(vao);
            self.vbo = Some(vbo);
            self.ebo = Some(ebo);

            gl::BindVertexArray(self.vao.unwrap().try_into().unwrap());

            // bind verts array
            gl::BindBuffer(gl::ARRAY_BUFFER, self.vbo.unwrap().try_into().unwrap());
            gl::BufferData(
                gl::ARRAY_BUFFER,
                (self.verts.len() * std::mem::size_of::<GLVert>())
                    .try_into()
                    .unwrap(),
                self.verts.as_ptr() as *const gl::types::GLvoid,
                gl::STATIC_DRAW,
            );
            // {
            //     let size = self.verts.len() * std::mem::size_of::<GLVert>();
            //     let mut data: Vec<GLVert> = Vec::with_capacity(self.verts.len());
            //     for _ in 0..self.verts.len() {
            //         data.push(GLVert::new(glm::zero(), glm::zero(), glm::zero()));
            //     }
            //     gl::GetBufferSubData(
            //         gl::ARRAY_BUFFER,
            //         0,
            //         size as gl::types::GLsizeiptr,
            //         data.as_mut_ptr() as *mut gl::types::GLvoid,
            //     );
            //     println!("{:?}", data);
            // }

            // bind indices array
            gl::BindBuffer(
                gl::ELEMENT_ARRAY_BUFFER,
                self.ebo.unwrap().try_into().unwrap(),
            );
            gl::BufferData(
                gl::ELEMENT_ARRAY_BUFFER,
                (self.indices.len() * std::mem::size_of::<gl::types::GLuint>())
                    .try_into()
                    .unwrap(),
                self.indices.as_ptr() as *const gl::types::GLvoid,
                gl::STATIC_DRAW,
            );
            // {
            //     let size = self.indices.len() * std::mem::size_of::<gl::types::GLuint>();
            //     let mut data: Vec<gl::types::GLuint> = Vec::with_capacity(self.indices.len());
            //     for _ in 0..self.indices.len() {
            //         data.push(0);
            //     }
            //     gl::GetBufferSubData(
            //         gl::ELEMENT_ARRAY_BUFFER,
            //         0,
            //         size as gl::types::GLsizeiptr,
            //         data.as_mut_ptr() as *mut gl::types::GLvoid,
            //     );
            //     println!("{:?}", data);
            // }

            let offset = 0;
            // positions
            gl::EnableVertexAttribArray(0);
            gl::VertexAttribPointer(
                0,
                3,
                gl::FLOAT,
                gl::FALSE,
                std::mem::size_of::<GLVert>().try_into().unwrap(),
                offset as *const gl::types::GLvoid,
            );
            let offset = offset + std::mem::size_of::<glm::Vec3>();
            //uv
            gl::EnableVertexAttribArray(1);
            gl::VertexAttribPointer(
                1,
                2,
                gl::FLOAT,
                gl::FALSE,
                std::mem::size_of::<GLVert>().try_into().unwrap(),
                offset as *const gl::types::GLvoid,
            );
            let offset = offset + std::mem::size_of::<glm::Vec2>();
            //normals
            gl::EnableVertexAttribArray(2);
            gl::VertexAttribPointer(
                2,
                3,
                gl::FLOAT,
                gl::FALSE,
                std::mem::size_of::<GLVert>().try_into().unwrap(),
                offset as *const gl::types::GLvoid,
            );

            gl::BindVertexArray(0);
        }
    }
}
