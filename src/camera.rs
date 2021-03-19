use nalgebra_glm as glm;
extern crate glfw;

use std::cell::RefCell;
use std::rc::Weak;

pub struct WindowCamera {
    window: Weak<RefCell<glfw::Window>>,
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
        window: Weak<RefCell<glfw::Window>>,
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
            front: glm::vec3(0.0, 0.0, -1.0),
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

    pub fn get_position(&self) -> glm::DVec3 {
        return self.position;
    }

    pub fn get_view_matrix(&self) -> glm::DMat4 {
        return glm::look_at(&self.position, &(self.position + self.front), &self.up);
    }

    pub fn get_projection_matrix(&self) -> glm::DMat4 {
        let window = self
            .window
            .upgrade()
            .expect("Window with which camera was made is lost");
        let (width, height) = window.borrow().get_size();
        return glm::perspective(
            width as f64 / height as f64,
            self.zoom.to_radians(),
            0.1,
            1000.0,
        );
    }

    pub fn pan(
        &mut self,
        mouse_start_x: f64,
        mouse_start_y: f64,
        mouse_end_x: f64,
        mouse_end_y: f64,
        len: f64,
    ) {
        if mouse_start_x == mouse_end_x && mouse_start_y == mouse_end_y {
            return;
        }
        let window = self
            .window
            .upgrade()
            .expect("Window with which camera was made is lost");
        let (width, height) = window.borrow().get_size();
        let clip_x = mouse_start_x * 2.0 / width as f64 - 1.0;
        let clip_y = 1.0 - mouse_start_y * 2.0 / height as f64;

        let clip_end_x = mouse_end_x * 2.0 / width as f64 - 1.0;
        let clip_end_y = 1.0 - mouse_end_y * 2.0 / height as f64;

        let inverse_mvp = glm::inverse(&(self.get_projection_matrix() * self.get_view_matrix()));
        let out_vector = inverse_mvp * glm::vec4(clip_x, clip_y, 0.0, 1.0);
        let world_pos = glm::vec3(
            out_vector.x / out_vector.w,
            out_vector.y / out_vector.w,
            out_vector.z / out_vector.w,
        );

        let out_end_vector = inverse_mvp * glm::vec4(clip_end_x, clip_end_y, 0.0, 1.0);
        let world_pos_2 = glm::vec3(
            out_end_vector.x / out_end_vector.w,
            out_end_vector.y / out_end_vector.w,
            out_end_vector.z / out_end_vector.w,
        );

        let dir = world_pos_2 - world_pos;

        let offset = glm::length(&dir) * glm::normalize(&dir) * self.zoom * len / 2.0;

        self.position -= offset;
    }

    pub fn rotate_wrt_camera_origin(
        &mut self,
        mouse_start_x: f64,
        mouse_start_y: f64,
        mouse_end_x: f64,
        mouse_end_y: f64,
        mouse_sensitivity: f64,
        constrain_pitch: bool,
    ) {
        let x_offset = (mouse_end_x - mouse_start_x) * mouse_sensitivity;
        let y_offset = (mouse_start_y - mouse_end_y) * mouse_sensitivity;

        self.yaw += x_offset;
        self.pitch += y_offset;

        if constrain_pitch {
            if self.pitch > 89.0 {
                self.pitch = 89.0;
            } else if self.pitch < -89.0 {
                self.pitch = -89.0;
            }
        }

        self.update_camera_vectors();
    }

    pub fn move_forward(&mut self, mouse_start_y: f64, mouse_end_y: f64) {
        let window = self
            .window
            .upgrade()
            .expect("Window with which camera was made is lost");
        let (_, height) = window.borrow().get_size();
        let clip_y = 1.0 - mouse_start_y * 2.0 / height as f64;
        let clip_end_y = 1.0 - mouse_end_y * 2.0 / height as f64;

        let move_by = clip_end_y - clip_y;

        self.position += self.front * move_by;
    }

    pub fn zoom(&mut self, scroll_y: f64) {
        let min = 1.0;
        let max = 90.0;
        if self.zoom >= min && self.zoom <= max {
            self.zoom -= scroll_y;
        }
        if self.zoom < min {
            self.zoom = min;
        }
        if self.zoom > max {
            self.zoom = max;
        }
    }

    pub fn get_raycast_direction(&self, mouse_x: f64, mouse_y: f64) -> glm::DVec3 {
        let window = self
            .window
            .upgrade()
            .expect("Window with which camera was made is lost");
        let (width, height) = window.borrow().get_size();
        let x = (2.0 * mouse_x) / width as f64 - 1.0;
        let y = 1.0 - (2.0 * mouse_y) / height as f64;

        let ray_clip = glm::vec4(x, y, -1.0, 1.0);

        let ray_eye = glm::inverse(&self.get_projection_matrix()) * ray_clip;
        let ray_eye = glm::vec4(ray_eye[0], ray_eye[1], -1.0, 0.0);

        let ray_wor = glm::inverse(&self.get_view_matrix()) * ray_eye;
        let result = glm::normalize(&glm::vec4_to_vec3(&ray_wor));
        return result;
    }
}
