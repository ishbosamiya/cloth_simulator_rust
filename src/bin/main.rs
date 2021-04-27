extern crate gl;
extern crate glfw;

use glfw::{Action, Context, Key};
use nalgebra_glm as glm;
use rand::random;

use std::cell::RefCell;
use std::rc::Rc;

use cloth_simulator_rust::bvh::{BVHDrawData, BVHOverlapDrawData};
use cloth_simulator_rust::camera::WindowCamera;
use cloth_simulator_rust::drawable::Drawable;
use cloth_simulator_rust::gpu_immediate::*;
use cloth_simulator_rust::mesh::MeshDrawData;
use cloth_simulator_rust::shader::Shader;
use cloth_simulator_rust::simulation::{cloth, ConstraintDrawData, Simulation};
use cloth_simulator_rust::text::{Font, Text, TextSizePT};

fn main() {
    let mut glfw = glfw::init(glfw::FAIL_ON_ERRORS).unwrap();

    glfw.window_hint(glfw::WindowHint::ContextVersion(3, 3));
    glfw.window_hint(glfw::WindowHint::OpenGlProfile(
        glfw::OpenGlProfileHint::Core,
    ));
    glfw.window_hint(glfw::WindowHint::Samples(Some(16)));
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
        gl::Enable(gl::MULTISAMPLE);
    }

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

    let text_shader = Shader::new(
        std::path::Path::new("shaders/text.vert"),
        std::path::Path::new("shaders/text.frag"),
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
    println!(
        "text: uniforms: {:?} attributes: {:?}",
        text_shader.get_uniforms(),
        text_shader.get_attributes(),
    );

    let mut camera = WindowCamera::new(
        Rc::downgrade(&window),
        glm::vec3(0.0, 0.0, 3.0),
        glm::vec3(0.0, 1.0, 0.0),
        -90.0,
        0.0,
        45.0,
    );
    let ortho_camera = WindowCamera::new(
        Rc::downgrade(&window),
        glm::vec3(0.0, 0.0, 3.0),
        glm::vec3(0.0, 1.0, 0.0),
        -90.0,
        0.0,
        45.0,
    );

    let font_file = Font::load_font_file("/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf");
    let mut font = Font::new(&font_file);

    let mut cloth = cloth::Mesh::new();
    cloth
        .read(std::path::Path::new("models/plane_subd_02.obj"))
        .unwrap();
    cloth.build_bvh(0.01);
    let mut simulation = Simulation::new(cloth, 1.0, 1.0 / 30.0, 800.0);
    let mut run_sim = false;
    let mut draw_linear_constraints = false;
    let mut draw_wireframe = false;
    let mut bvh_draw_level = 0;
    let mut bvh_draw_self_overlap = false;

    let mut imm = GPUImmediate::new();

    let mut last_cursor = window.borrow().get_cursor_pos();

    let mut fps = FPS::new();

    let dpi = glfw.with_primary_monitor(|_, monitor| {
        let monitor = monitor.expect("error: Unable to get reference to monitor");
        let (size_x, size_y) = monitor.get_physical_size();
        let video_mode = monitor.get_video_mode().unwrap();
        let (res_x, res_y) = (video_mode.width, video_mode.height);
        let raw_dpi_x = res_x as f32 * 25.4 / size_x as f32;
        let raw_dpi_y = res_y as f32 * 25.4 / size_y as f32;
        let (scale_x, scale_y) = monitor.get_content_scale();
        let dpi_x = raw_dpi_x * scale_x as f32;
        let _dpi_y = raw_dpi_y * scale_y as f32;
        return dpi_x;
    });

    while !window.borrow().should_close() {
        glfw.poll_events();
        for (_, event) in glfw::flush_messages(&events) {
            handle_window_event(
                window.clone(),
                event,
                &mut camera,
                &mut last_cursor,
                &mut simulation,
                &mut draw_wireframe,
                &mut draw_linear_constraints,
                &mut bvh_draw_level,
                &mut bvh_draw_self_overlap,
                &mut run_sim,
            );
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

        text_shader.use_shader();
        text_shader.set_mat4(
            "projection\0",
            &glm::convert(ortho_camera.get_ortho_matrix()),
        );

        // default_shader.use_shader();
        directional_light_shader.use_shader();
        // face_orientation_shader.use_shader();

        if run_sim {
            simulation.next_step(10);
        }
        simulation.cloth.update_bvh();

        let bvh = simulation.cloth.get_bvh().as_ref().unwrap();
        let overlap = bvh.overlap(
            bvh,
            Some(&|face_1_index, face_2_index| {
                if face_1_index == face_2_index {
                    return false;
                }

                let cloth = &simulation.cloth;
                let face_1 = cloth.get_face(face_1_index).unwrap();
                let face_2 = cloth.get_face(face_2_index).unwrap();

                for face_1_vert in face_1.get_verts() {
                    for face_2_vert in face_2.get_verts() {
                        if face_1_vert == face_2_vert {
                            return false;
                        }
                    }
                }

                return true;
            }),
        );

        let mut draw_data = MeshDrawData::new(&mut imm, &directional_light_shader);
        simulation.cloth.draw(&mut draw_data).unwrap();
        if draw_wireframe {
            let mut draw_data = MeshDrawData::new(&mut imm, &smooth_3d_color_shader);
            simulation.cloth.draw_wireframe(&mut draw_data).unwrap();
        }
        let mut draw_data =
            ConstraintDrawData::new(&mut imm, &smooth_3d_color_shader, draw_linear_constraints);
        simulation.draw(&mut draw_data).unwrap();
        let mut draw_data = BVHDrawData::new(&mut imm, &smooth_3d_color_shader, bvh_draw_level);
        simulation
            .cloth
            .get_bvh()
            .as_ref()
            .unwrap()
            .draw(&mut draw_data)
            .unwrap();

        if let Some(overlap) = overlap {
            let mut draw_data = BVHOverlapDrawData::new(
                &mut imm,
                &smooth_3d_color_shader,
                &simulation.cloth,
                &simulation.cloth,
            );

            if bvh_draw_self_overlap {
                overlap.draw(&mut draw_data).unwrap();
            }
        }

        text_shader.use_shader();
        Text::render(
            "helloworld",
            &mut font,
            TextSizePT(72.0),
            &glm::vec2(40.0, 50.0),
            TextSizePT(dpi),
        );

        window.borrow_mut().swap_buffers();

        fps.update_and_print(20);
    }
}

fn handle_window_event(
    window: Rc<RefCell<glfw::Window>>,
    event: glfw::WindowEvent,
    camera: &mut WindowCamera,
    last_cursor: &mut (f64, f64),
    simulation: &mut Simulation,
    r_draw_wireframe: &mut bool,
    r_draw_linear: &mut bool,
    r_bvh_draw_level: &mut usize,
    r_bvh_draw_self_overlap: &mut bool,
    r_run_sim: &mut bool,
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
        glfw::WindowEvent::Key(Key::S, _, Action::Press, _) => *r_run_sim = !*r_run_sim,
        glfw::WindowEvent::FramebufferSize(width, height) => unsafe {
            gl::Viewport(0, 0, width, height);
        },
        glfw::WindowEvent::Scroll(_, scroll_y) => {
            camera.zoom(scroll_y);
        }
        glfw::WindowEvent::Key(Key::D, _, Action::Press, _) => {
            *r_draw_linear = !*r_draw_linear;
        }
        glfw::WindowEvent::Key(Key::W, _, Action::Press, _) => {
            *r_draw_wireframe = !*r_draw_wireframe;
        }
        glfw::WindowEvent::Key(Key::N, _, Action::Press, _) => {
            *r_bvh_draw_level += 1;
            println!("bvh_draw_level now at: {}", *r_bvh_draw_level);
        }
        glfw::WindowEvent::Key(Key::O, _, Action::Press, _) => {
            *r_bvh_draw_self_overlap = !*r_bvh_draw_self_overlap;
            println!("bvh_self_overlap: {}", *r_bvh_draw_self_overlap);
        }
        glfw::WindowEvent::Key(Key::P, _, Action::Press, _) => {
            if *r_bvh_draw_level != 0 {
                *r_bvh_draw_level -= 1;
                println!("bvh_draw_level now at: {}", *r_bvh_draw_level);
            }
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
    if window.borrow().get_mouse_button(glfw::MouseButtonLeft) == Action::Press {
        simulation.try_toggle_pin_constraint(
            &camera.get_position(),
            &camera.get_raycast_direction(cursor.0, cursor.1),
        );
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
