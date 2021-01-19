extern crate gl;
extern crate glfw;

use glfw::{Action, Context, Key};
use nalgebra_glm as glm;
use rand::random;

use cloth_simulator_rust::gl_mesh::{GLMesh, GLVert};

fn main() {
    let mut glfw = glfw::init(glfw::FAIL_ON_ERRORS).unwrap();

    glfw.window_hint(glfw::WindowHint::ContextVersion(3, 3));
    glfw.window_hint(glfw::WindowHint::OpenGlProfile(
        glfw::OpenGlProfileHint::Core,
    ));
    #[cfg(target_os = "macos")]
    glfw.window_hint(glfw::WindowHint::OpenGlForwardCompat(true));
    let (mut window, events) = glfw
        .create_window(1280, 720, "Hello world", glfw::WindowMode::Windowed)
        .expect("Failed to create glfw window");

    window.set_key_polling(true);
    window.set_framebuffer_size_polling(true);
    window.make_current();

    gl::load_with(|symbol| window.get_proc_address(symbol));

    let mesh = GLMesh::new(
        vec![
            GLVert::new(glm::vec3(0.5, -0.5, 0.0), glm::zero(), glm::zero()),
            GLVert::new(glm::vec3(-0.5, -0.5, 0.0), glm::zero(), glm::zero()),
            GLVert::new(glm::vec3(0.0, 0.5, 0.0), glm::zero(), glm::zero()),
        ],
        vec![0, 1, 2],
    );

    let vertex_shader_source = r#"#version 330 core
layout (location = 0) in vec3 aPos;
void main()
{
gl_Position = vec4(aPos.x, aPos.y, aPos.z, 1.0);
}"#;
    let vertex_shader_source = std::ffi::CString::new(vertex_shader_source).unwrap();
    let vertex_shader: gl::types::GLuint;
    unsafe {
        vertex_shader = gl::CreateShader(gl::VERTEX_SHADER);
        gl::ShaderSource(
            vertex_shader,
            1,
            &vertex_shader_source.as_ptr(),
            std::ptr::null(),
        );
        gl::CompileShader(vertex_shader);
    }
    unsafe {
        let mut success: gl::types::GLint = -10;
        gl::GetShaderiv(vertex_shader, gl::COMPILE_STATUS, &mut success);
        if success != gl::TRUE.into() {
            eprintln!("vertex didn't compile");
        }
    }
    let fragment_shader_source = r#"#version 330 core
out vec4 FragColor;
void main()
{
    FragColor = vec4(1.0f, 0.5f, 0.2f, 1.0f);
}"#;
    let fragment_shader_source = std::ffi::CString::new(fragment_shader_source).unwrap();
    let fragment_shader: gl::types::GLuint;
    unsafe {
        fragment_shader = gl::CreateShader(gl::FRAGMENT_SHADER);
        gl::ShaderSource(
            fragment_shader,
            1,
            &fragment_shader_source.as_ptr(),
            std::ptr::null(),
        );
        gl::CompileShader(fragment_shader);
    }
    unsafe {
        let mut success: gl::types::GLint = -10;
        gl::GetShaderiv(fragment_shader, gl::COMPILE_STATUS, &mut success);
        if success != gl::TRUE.into() {
            eprintln!("fragment didn't compile");
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
        }
    }
    unsafe {
        gl::UseProgram(shader_program);
    }

    while !window.should_close() {
        glfw.poll_events();
        for (_, event) in glfw::flush_messages(&events) {
            handle_window_event(&mut window, event);
        }

        unsafe {
            gl::Clear(gl::COLOR_BUFFER_BIT | gl::DEPTH_BUFFER_BIT);
        }

        mesh.draw();

        window.swap_buffers();
    }

    unsafe {
        gl::DeleteShader(vertex_shader);
        gl::DeleteShader(fragment_shader);
    }
}

fn handle_window_event(window: &mut glfw::Window, event: glfw::WindowEvent) {
    match event {
        glfw::WindowEvent::Key(Key::Escape, _, Action::Press, _) => {
            window.set_should_close(true);
        }
        glfw::WindowEvent::Key(Key::A, _, Action::Press, _) => unsafe {
            gl::ClearColor(random(), random(), random(), 1.0);
            gl::Clear(gl::COLOR_BUFFER_BIT);
        },
        glfw::WindowEvent::FramebufferSize(width, height) => unsafe {
            gl::Viewport(0, 0, width, height);
        },
        _ => {}
    }
}
