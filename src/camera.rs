//! Z-up orbit camera and projection for points and diagnostic overlays.
//!
//! [`OrbitCamera`] stores a target and spherical offsets rather than a free
//! transform. Dragging changes yaw and pitch; scrolling scales the distance.
//! [`OrbitCamera::view_proj`] uses a left-handed view and DirectX depth range,
//! which also defines the planes extracted by [`crate::lod::Frustum`].
//!
//! ```text
//! local position -> view matrix -> projection -> clip coordinates
//!                                                    |
//!                                               divide by w
//!                                                    |
//!                                        normalized device coordinates
//!                                                    |
//!                                        viewport mapping, flip screen y
//!                                                    |
//!                                              egui position
//! ```
//!
//! [`project_to_screen`] supplies overlay positions; the point shader applies
//! the same view-projection matrix on the GPU.

use glam::{
    Mat4, Vec2, Vec3,
    camera::lh::{proj::directx::perspective, view::look_at_mat4},
};

/// Vertical field of view of the projection (radians).
pub const FOV_Y: f32 = std::f32::consts::FRAC_PI_4;

/// Smallest radius [`OrbitCamera::fit`] works with (metres), so a cloud of a
/// single point still gets a usable distance.
const MIN_FIT_RADIUS: f32 = 1.0;

/// A camera that orbits a scene-relative target with Z as the up axis.
pub struct OrbitCamera {
    /// Centre of rotation in the same coordinates as the tile points.
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

/// Where `point` lands on screen: projected with `view_proj` and mapped into the
/// rectangle at `rect_min` of `rect_size`, with screen y growing down. `None` if the
/// point is behind the camera (clip w <= 0).
pub fn project_to_screen(
    view_proj: Mat4,
    point: Vec3,
    rect_min: Vec2,
    rect_size: Vec2,
) -> Option<Vec2> {
    let clip = view_proj * point.extend(1.0);
    if clip.w <= 0.0 {
        return None;
    }

    let ndc = clip.truncate().truncate() / clip.w;
    Some(rect_min + Vec2::new(0.5 + 0.5 * ndc.x, 0.5 - 0.5 * ndc.y) * rect_size)
}

impl OrbitCamera {
    /// Lower drag limit keeps the orbit above the horizontal plane.
    const MIN_PITCH: f32 = 0.05;
    /// Avoids looking exactly along the up axis, where the view basis degenerates.
    const MAX_PITCH: f32 = std::f32::consts::FRAC_PI_2 - 0.05;

    /// Centres the camera on the box and fits its sphere to the vertical field
    /// of view. A narrow viewport can still clip the sphere horizontally.
    ///
    /// The box is wrapped in a sphere (radius = half the diagonal), and the zoom
    /// limits scale with that radius.
    ///
    /// Bounds that are not finite, or with a minimum above the maximum, frame the
    /// origin instead: a scene of empty tiles has the infinite bounds of
    /// [`crate::cloud::Stats::empty`].
    pub fn fit(bounds_min: Vec3, bounds_max: Vec3, fov_y: f32) -> Self {
        let valid =
            bounds_min.is_finite() && bounds_max.is_finite() && bounds_min.cmple(bounds_max).all();
        let (bounds_min, bounds_max) = if valid {
            (bounds_min, bounds_max)
        } else {
            (Vec3::ZERO, Vec3::ZERO)
        };

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

    /// Builds a left-handed view-projection matrix with depth in `0..=1`.
    /// Near and far planes are fixed at 1 and 10,000 metres.
    pub fn view_proj(&self, aspect: f32) -> Mat4 {
        let view = look_at_mat4(self.eye(), self.target, Vec3::Z);
        let projection = perspective(FOV_Y, aspect, 1.0, 10_000.0);
        projection * view
    }

    /// Converts drag distance to angles and clamps pitch away from the poles.
    pub fn orbit(&mut self, dx: f32, dy: f32) {
        const RADIANS_PER_PIXEL: f32 = 0.005;
        self.yaw -= dx * RADIANS_PER_PIXEL;
        self.pitch = (self.pitch + dy * RADIANS_PER_PIXEL).clamp(Self::MIN_PITCH, Self::MAX_PITCH);
    }

    /// Scales distance exponentially, so equal scroll amounts give equal zoom
    /// ratios. Positive input moves closer, within the fitted distance limits.
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
    fn the_target_projects_to_the_centre_of_the_rect() {
        let mut camera = OrbitCamera::fit(Vec3::splat(-10.0), Vec3::splat(10.0), FOV_Y);
        camera.yaw = 0.7;
        camera.pitch = 0.4;
        let (rect_min, rect_size) = (Vec2::new(300.0, 20.0), Vec2::new(800.0, 600.0));
        let view_proj = camera.view_proj(rect_size.x / rect_size.y);

        let centre = project_to_screen(view_proj, camera.target, rect_min, rect_size);

        let expected = rect_min + 0.5 * rect_size;
        assert!(centre.is_some_and(|centre| (centre - expected).length() < 1e-3));
    }

    #[test]
    fn points_behind_the_camera_are_not_projected() {
        let camera = OrbitCamera::fit(Vec3::splat(-10.0), Vec3::splat(10.0), FOV_Y);
        let behind = camera.eye() + (camera.eye() - camera.target);
        let view_proj = camera.view_proj(1.0);

        assert_eq!(
            project_to_screen(view_proj, behind, Vec2::ZERO, Vec2::ONE),
            None
        );
    }

    #[test]
    fn fit_of_empty_or_invalid_bounds_frames_the_origin() {
        let empty = (Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY));
        let nan = (Vec3::splat(f32::NAN), Vec3::ONE);

        for (min, max) in [empty, nan] {
            let camera = OrbitCamera::fit(min, max, FOV_Y);
            assert_eq!(camera.target, Vec3::ZERO);
            assert_eq!(camera.distance, fit_distance(MIN_FIT_RADIUS, FOV_Y));
            assert!(camera.eye().is_finite());
        }
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
