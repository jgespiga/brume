use std::time::Duration;

use glam::{DVec3, Vec3};

pub(crate) const NO_COLOR: [u8; 4] = [190, 190, 190, 255];

/// SoA representation of a batch of points.
///
/// Each one is usually about a million points. It is the
/// batch we get from [`las::Reader::read_points`] but restructured
/// with positions relative to the origin in f32 to avoid precission
/// issues.
///
/// We usually work with one attribute across many points, therefor for
/// that access pattern SoA is the best option for cache efficiency and
/// GPU buffers.
#[derive(Debug)]
pub(crate) struct PointChunk {
    /// [pos, pos, pos]
    pub positions: Vec<[f32; 3]>,
    pub classes: Vec<u8>,
    pub colors: Vec<[u8; 4]>,
}

impl PointChunk {
    pub fn len(&self) -> usize {
        self.positions.len()
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum ColorDepth {
    /// The point format has no RGB.
    None,
    /// Values 0-255 in 16-bit fields.
    Bits8,
    /// Values 0-65535, as the spec says.
    Bits16,
}

#[derive(Debug)]
pub(crate) struct Stats {
    pub points: u64,
    pub class_counts: [u64; 256],
    pub color_depth: ColorDepth,
    pub local_min: Vec3,
    pub local_max: Vec3,
    pub elapsed: Duration,
}

/// Representation of a cloud of points and its origin.
pub(crate) struct Cloud {
    pub origin: DVec3,
    pub chunks: Vec<PointChunk>,
}
