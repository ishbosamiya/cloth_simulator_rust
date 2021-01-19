extern crate gl;
extern crate glfw;

use glfw::{Action, Context, Key};
use rand::random;

fn main() {
    let mut glfw = glfw::init(glfw::FAIL_ON_ERRORS).unwrap();

    let (mut window, events) = glfw
        .create_window(1280, 720, "Hello world", glfw::WindowMode::Windowed)
        .expect("Failed to create glfw window");

    window.set_key_polling(true);
    window.make_current();

    gl::load_with(|symbol| window.get_proc_address(symbol));

    while !window.should_close() {
        glfw.poll_events();
        for (_, event) in glfw::flush_messages(&events) {
            handle_window_event(&mut window, event);
        }
        window.swap_buffers();
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
        _ => {}
    }
}
