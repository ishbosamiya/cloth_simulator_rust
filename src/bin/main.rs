extern crate gl;
extern crate glfw;

use glfw::{Action, Context, Key};
use nalgebra_glm as glm;
use rand::random;

use std::cell::RefCell;
use std::rc::Rc;

use cloth_simulator_rust::camera::WindowCamera;
use cloth_simulator_rust::drawable::Drawable;
use cloth_simulator_rust::gl_mesh::{GLMesh, GLVert};
use cloth_simulator_rust::shader::Shader;

fn main() {
    let mut glfw = glfw::init(glfw::FAIL_ON_ERRORS).unwrap();

    glfw.window_hint(glfw::WindowHint::ContextVersion(3, 3));
    glfw.window_hint(glfw::WindowHint::OpenGlProfile(
        glfw::OpenGlProfileHint::Core,
    ));
    #[cfg(target_os = "macos")]
    glfw.window_hint(glfw::WindowHint::OpenGlForwardCompat(true));
    let (window, events) = glfw
        .create_window(1280, 720, "Hello world", glfw::WindowMode::Windowed)
        .expect("Failed to create glfw window");

    let window = Rc::new(RefCell::new(window));

    window.borrow_mut().set_key_polling(true);
    window.borrow_mut().set_cursor_pos_polling(true);
    window.borrow_mut().set_mouse_button_polling(true);
    window.borrow_mut().set_framebuffer_size_polling(true);
    window.borrow_mut().make_current();

    gl::load_with(|symbol| window.borrow_mut().get_proc_address(symbol));

    let mesh = GLMesh::new(
        vec![
            GLVert::new(glm::vec3(0.5, -0.5, -1.0), glm::zero(), glm::zero()),
            GLVert::new(glm::vec3(-0.5, -0.5, -1.0), glm::zero(), glm::zero()),
            GLVert::new(glm::vec3(0.0, 0.5, -1.0), glm::zero(), glm::zero()),
        ],
        vec![0, 1, 2],
    );

    let default_shader = Shader::new(
        std::path::Path::new("shaders/default_shader.vert"),
        std::path::Path::new("shaders/default_shader.frag"),
    )
    .unwrap();

    let mut camera = WindowCamera::new(
        Rc::downgrade(&window),
        glm::zero(),
        glm::vec3(0.0, 1.0, 0.0),
        -90.0,
        0.0,
        45.0,
    );

    let mut last_cursor = window.borrow().get_cursor_pos();

    while !window.borrow().should_close() {
        glfw.poll_events();
        for (_, event) in glfw::flush_messages(&events) {
            handle_window_event(window.clone(), event, &mut camera, &mut last_cursor);
        }

        unsafe {
            gl::Clear(gl::COLOR_BUFFER_BIT | gl::DEPTH_BUFFER_BIT);
        }

        default_shader.use_shader();
        default_shader.set_mat4(
            "projection\0",
            &glm::convert(camera.get_projection_matrix()),
        );
        default_shader.set_mat4("view\0", &glm::convert(camera.get_view_matrix()));
        default_shader.set_mat4("model\0", &glm::identity());

        mesh.draw();

        window.borrow_mut().swap_buffers();
    }
}

fn handle_window_event(
    window: Rc<RefCell<glfw::Window>>,
    event: glfw::WindowEvent,
    camera: &mut WindowCamera,
    last_cursor: &mut (f64, f64),
) {
    let cursor = window.borrow_mut().get_cursor_pos();
    match event {
        glfw::WindowEvent::Key(Key::Escape, _, Action::Press, _) => {
            window.borrow_mut().set_should_close(true);
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
    if window.borrow().get_mouse_button(glfw::MouseButtonMiddle) == Action::Press {
        if window.borrow().get_key(glfw::Key::LeftShift) == Action::Press {
            camera.pan(last_cursor.0, last_cursor.1, cursor.0, cursor.1, 1.0);
        }
        if window.borrow().get_key(glfw::Key::LeftControl) == Action::Press {
            camera.move_forward(last_cursor.1, cursor.1);
        } else {
            camera.rotate_wrt_camera_origin(
                last_cursor.0,
                last_cursor.1,
                cursor.0,
                cursor.1,
                0.1,
                false,
            );
        }
    }
    *last_cursor = cursor;
}
