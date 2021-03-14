extern crate gl;
extern crate glfw;

use glfw::{Action, Context, Key};
use nalgebra_glm as glm;
use rand::random;

use std::cell::RefCell;
use std::rc::Rc;

use cloth_simulator_rust::camera::WindowCamera;
use cloth_simulator_rust::drawable::Drawable;
use cloth_simulator_rust::gpu_immediate::*;
use cloth_simulator_rust::mesh::{simple::Mesh, MeshDrawData};
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
    window.borrow_mut().set_scroll_polling(true);
    window.borrow_mut().make_current();

    gl::load_with(|symbol| window.borrow_mut().get_proc_address(symbol));

    glfw.set_swap_interval(glfw::SwapInterval::None);

    unsafe {
        gl::Disable(gl::CULL_FACE);
        gl::Enable(gl::DEPTH_TEST);
    }

    let mut mesh = Mesh::new();
    mesh.read(&std::path::Path::new("models/monkey_subd_01.obj"))
        .unwrap();
    // mesh.read(&std::path::Path::new("models/cube.obj")).unwrap();
    // mesh.read(&std::path::Path::new("tests/obj_test_06_winding_test.obj"))
    // .unwrap();
    mesh.generate_gl_mesh(false);

    let default_shader = Shader::new(
        std::path::Path::new("shaders/default_shader.vert"),
        std::path::Path::new("shaders/default_shader.frag"),
    )
    .unwrap();
    let directional_light_shader = Shader::new(
        std::path::Path::new("shaders/directional_light.vert"),
        std::path::Path::new("shaders/directional_light.frag"),
    )
    .unwrap();
    let face_orientation_shader = Shader::new(
        std::path::Path::new("shaders/face_orientation.vert"),
        std::path::Path::new("shaders/face_orientation.frag"),
    )
    .unwrap();

    let smooth_3d_color_shader = Shader::new(
        std::path::Path::new("shaders/shader_3D_smooth_color.vert"),
        std::path::Path::new("shaders/shader_3D_smooth_color.frag"),
    )
    .unwrap();

    println!(
        "default: uniforms: {:?} attributes: {:?}",
        default_shader.get_uniforms(),
        default_shader.get_attributes(),
    );
    println!(
        "directional_light: uniforms: {:?} attributes: {:?}",
        directional_light_shader.get_uniforms(),
        directional_light_shader.get_attributes(),
    );
    println!(
        "face_orientation: uniforms: {:?} attributes: {:?}",
        face_orientation_shader.get_uniforms(),
        face_orientation_shader.get_attributes(),
    );
    println!(
        "smooth_3d_color: uniforms: {:?} attributes: {:?}",
        smooth_3d_color_shader.get_uniforms(),
        smooth_3d_color_shader.get_attributes(),
    );

    let mut camera = WindowCamera::new(
        Rc::downgrade(&window),
        glm::vec3(0.0, 0.0, 3.0),
        glm::vec3(0.0, 1.0, 0.0),
        -90.0,
        0.0,
        45.0,
    );

    let mut imm = GPUImmediate::new();

    let mut last_cursor = window.borrow().get_cursor_pos();

    let mut fps = FPS::new();

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

        smooth_3d_color_shader.use_shader();
        smooth_3d_color_shader.set_mat4(
            "projection\0",
            &glm::convert(camera.get_projection_matrix()),
        );
        smooth_3d_color_shader.set_mat4("view\0", &glm::convert(camera.get_view_matrix()));
        smooth_3d_color_shader.set_mat4("model\0", &glm::identity());

        directional_light_shader.use_shader();
        directional_light_shader.set_mat4(
            "projection\0",
            &glm::convert(camera.get_projection_matrix()),
        );
        directional_light_shader.set_mat4("view\0", &glm::convert(camera.get_view_matrix()));
        directional_light_shader.set_mat4("model\0", &glm::identity());
        directional_light_shader.set_vec3("viewPos\0", &glm::convert(camera.get_position()));
        directional_light_shader.set_vec3("material.color\0", &glm::vec3(0.3, 0.2, 0.7));
        directional_light_shader.set_vec3("material.specular\0", &glm::vec3(0.3, 0.3, 0.3));
        directional_light_shader.set_float("material.shininess\0", 4.0);
        directional_light_shader.set_vec3("light.direction\0", &glm::vec3(-0.7, -1.0, -0.7));
        directional_light_shader.set_vec3("light.ambient\0", &glm::vec3(0.3, 0.3, 0.3));
        directional_light_shader.set_vec3("light.diffuse\0", &glm::vec3(1.0, 1.0, 1.0));
        directional_light_shader.set_vec3("light.specular\0", &glm::vec3(1.0, 1.0, 1.0));

        face_orientation_shader.use_shader();
        face_orientation_shader.set_mat4(
            "projection\0",
            &glm::convert(camera.get_projection_matrix()),
        );
        face_orientation_shader.set_mat4("view\0", &glm::convert(camera.get_view_matrix()));
        face_orientation_shader.set_mat4("model\0", &glm::identity());

        // default_shader.use_shader();
        directional_light_shader.use_shader();
        // face_orientation_shader.use_shader();
        // mesh.generate_gl_mesh(false);

        let mut draw_data = MeshDrawData::new(&mut imm, &directional_light_shader);
        // let mut draw_data = MeshDrawData::new(&mut imm, &smooth_3d_color_shader);
        mesh.draw(&mut draw_data).unwrap();

        window.borrow_mut().swap_buffers();

        fps.update_and_print(20);
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
        glfw::WindowEvent::Scroll(_, scroll_y) => {
            camera.zoom(scroll_y);
        }
        _ => {}
    }
    if window.borrow().get_mouse_button(glfw::MouseButtonMiddle) == Action::Press {
        if window.borrow().get_key(glfw::Key::LeftShift) == Action::Press {
            camera.pan(last_cursor.0, last_cursor.1, cursor.0, cursor.1, 1.0);
        } else if window.borrow().get_key(glfw::Key::LeftControl) == Action::Press {
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

struct FPS {
    previous_time: std::time::Instant,
    frames: usize,
}

impl FPS {
    fn new() -> FPS {
        return FPS {
            previous_time: std::time::Instant::now(),
            frames: 0,
        };
    }

    /// Update and print every nth frame
    fn update_and_print(&mut self, n: usize) {
        self.frames += 1;

        if self.frames % n == 0 {
            let current = std::time::Instant::now();
            let fps = self.frames as f64 / (current - self.previous_time).as_secs_f64();

            println!("fps: {:.2}", fps);

            self.previous_time = current;
            self.frames = 0;
        }
    }
}
