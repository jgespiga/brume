use glam::DVec3;

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
}

impl PointChunk {
    pub fn len(&self) -> usize {
        self.positions.len()
    }
}

/// Representation of a cloud of points and its origin.
pub(crate) struct Cloud {
    pub origin: DVec3,
    pub chunks: Vec<PointChunk>,
}

pub(crate) struct Stats {
    pub class_counts: [u64; 256],
}
