use glam::{
    Mat4, Vec3,
    camera::lh::{proj::directx::perspective, view::look_at_mat4},
};

/// Vertical field of view of the projection (radians).
pub const FOV_Y: f32 = std::f32::consts::FRAC_PI_4;

/// Smallest radius [`OrbitCamera::fit`] works with (metres), so a cloud of a
/// single point still gets a usable distance.
const MIN_FIT_RADIUS: f32 = 1.0;

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
    /// Closest zoom (metres).
    pub min_distance: f32,
    /// Furthest zoom (metres).
    pub max_distance: f32,
}

/// Distance at which a sphere of `radius` fits the vertical field of view `fov_y`.
pub fn fit_distance(radius: f32, fov_y: f32) -> f32 {
    radius / (fov_y * 0.5).sin()
}

impl OrbitCamera {
    const MIN_PITCH: f32 = 0.05;
    // Max pitch just under 90º
    const MAX_PITCH: f32 = std::f32::consts::FRAC_PI_2 - 0.05;

    /// Creates a camera that looks at the centre of the box and sees all of it.
    ///
    /// The box is wrapped in a sphere (radius = half the diagonal), and the zoom
    /// limits scale with that radius.
    pub fn fit(bounds_min: Vec3, bounds_max: Vec3, fov_y: f32) -> Self {
        let radius = (0.5 * (bounds_max - bounds_min).length()).max(MIN_FIT_RADIUS);

        Self {
            target: 0.5 * (bounds_min + bounds_max),
            yaw: 0.0,
            pitch: 0.0,
            distance: fit_distance(radius, fov_y),
            min_distance: radius * 0.01,
            max_distance: radius * 10.0,
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
        let projection = perspective(FOV_Y, aspect, 1.0, 10_000.0);
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
        self.distance =
            (self.distance * (-scroll * 0.002).exp()).clamp(self.min_distance, self.max_distance);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fit_distance_puts_the_sphere_on_the_edge_of_the_view() {
        // The line from the eye tangent to the sphere is at half the field of view.
        let distance = fit_distance(10.0, FOV_Y);

        assert!((distance * (FOV_Y * 0.5).sin() - 10.0).abs() < 1e-4);
        // For 90º, sin(45º) = 1/√2.
        assert!((fit_distance(1.0, std::f32::consts::FRAC_PI_2) - 2f32.sqrt()).abs() < 1e-6);
    }

    #[test]
    fn fit_targets_the_centre_and_scales_with_the_box() {
        // A 1 km × 1 km × 60 m tile, like a PNOA tile in local coordinates.
        let min = Vec3::new(-500.0, -500.0, -30.0);
        let max = Vec3::new(500.0, 500.0, 30.0);
        let camera = OrbitCamera::fit(min, max, FOV_Y);
        let radius = 0.5 * (max - min).length();

        assert_eq!(camera.target, Vec3::ZERO);
        assert!((camera.distance - fit_distance(radius, FOV_Y)).abs() < 1e-3);
        assert!((camera.min_distance - radius * 0.01).abs() < 1e-3);
        assert!((camera.max_distance - radius * 10.0).abs() < 1e-3);
    }

    #[test]
    fn fit_of_a_single_point_uses_the_minimum_radius() {
        let point = Vec3::new(1.0, 2.0, 3.0);
        let camera = OrbitCamera::fit(point, point, FOV_Y);

        assert_eq!(camera.target, point);
        assert!((camera.distance - fit_distance(MIN_FIT_RADIUS, FOV_Y)).abs() < 1e-6);
    }

    #[test]
    fn zoom_stays_within_the_limits() {
        let mut camera = OrbitCamera::fit(Vec3::splat(-10.0), Vec3::splat(10.0), FOV_Y);

        camera.zoom(1e6);
        assert_eq!(camera.distance, camera.min_distance);

        camera.zoom(-1e6);
        assert_eq!(camera.distance, camera.max_distance);
    }
}
