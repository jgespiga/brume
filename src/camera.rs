use glam::{
    Mat4, Vec3,
    camera::lh::{proj::directx::perspective, view::look_at_mat4},
};
/// A camera that orbits around a target point.
///
/// Note that coordinate `Z` is up.
pub struct OrbitCamera {
    pub target: Vec3,
    /// Horizontal angle around the target (radians).
    pub yaw: f32,
    /// Angle above the horizon (radians).
    pub pitch: f32,
    /// Distance from the target (metres).
    pub distance: f32,
}

impl OrbitCamera {
    const MIN_PITCH: f32 = 0.05;
    // Max pitch just under 90º
    const MAX_PITCH: f32 = std::f32::consts::FRAC_PI_2 - 0.05;

    pub fn new(target: Vec3, distance: f32) -> Self {
        Self {
            target,
            yaw: 0.0,
            pitch: 0.0,
            distance,
        }
    }

    /// Camera position, from the spherical coordinates.
    pub fn eye(&self) -> Vec3 {
        let (sin_yaw, cos_yaw) = self.yaw.sin_cos();
        let (sin_pitch, cos_pitch) = self.pitch.sin_cos();
        let direction = Vec3::new(cos_pitch * sin_yaw, -cos_pitch * cos_yaw, sin_pitch);
        self.target + self.distance * direction
    }

    pub fn view_proj(&self, aspect: f32) -> Mat4 {
        let view = look_at_mat4(self.eye(), self.target, Vec3::Z);
        let projection = perspective(45f32.to_radians(), aspect, 1.0, 10_000.0);
        projection * view
    }

    /// Mouse drag in pixels.
    pub fn orbit(&mut self, dx: f32, dy: f32) {
        const RADIANS_PER_PIXEL: f32 = 0.005;
        self.yaw -= dx * RADIANS_PER_PIXEL;
        self.pitch = (self.pitch + dy * RADIANS_PER_PIXEL).clamp(Self::MIN_PITCH, Self::MAX_PITCH);
    }

    /// Scroll amount; positive zooms in.
    pub fn zoom(&mut self, scroll: f32) {
        self.distance = (self.distance * (-scroll * 0.002).exp()).clamp(5.0, 5_000.0);
    }
}
