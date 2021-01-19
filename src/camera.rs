use nalgebra_glm as glm;
extern crate glfw;

use std::rc::Weak;

pub struct WindowCamera {
    window: Weak<glfw::Window>,
    position: glm::DVec3,
    front: glm::DVec3,
    up: glm::DVec3,
    right: glm::DVec3,
    world_up: glm::DVec3,
    yaw: f64,
    pitch: f64,
    zoom: f64,
}

impl WindowCamera {
    pub fn new(
        window: Weak<glfw::Window>,
        position: glm::DVec3,
        up: glm::DVec3,
        yaw: f64,
        pitch: f64,
        zoom: f64,
    ) -> WindowCamera {
        let mut camera = WindowCamera {
            window,
            position,
            yaw,
            pitch,
            world_up: up,
            front: glm::zero(),
            right: glm::zero(),
            up: up,
            zoom,
        };

        camera.update_camera_vectors();

        return camera;
    }

    fn update_camera_vectors(&mut self) {
        let yaw_radians = f64::to_radians(self.yaw);
        let pitch_radians = f64::to_radians(self.pitch);
        let front: glm::DVec3 = glm::vec3(
            yaw_radians.cos() * pitch_radians.cos(),
            pitch_radians.sin(),
            yaw_radians.sin() * pitch_radians.cos(),
        );
        self.front = glm::normalize(&front);

        self.right = glm::normalize(&glm::cross(&front, &self.world_up));
        self.up = glm::normalize(&glm::cross(&self.right, &front));
    }

    pub fn get_view_matrix(&self) -> glm::DMat4 {
        return glm::look_at(&self.position, &(self.position + self.front), &self.up);
    }

    pub fn get_projection_matrix(&self) -> glm::DMat4 {
        let window = self
            .window
            .upgrade()
            .expect("Window with which camera was made is lost");
        let (width, height) = window.get_size();
        return glm::perspective(
            self.zoom.to_radians(),
            width as f64 / height as f64,
            0.1,
            1000.0,
        );
    }
}
