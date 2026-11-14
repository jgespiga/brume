use std::{path::PathBuf, time::Duration};

use glam::{DVec3, Vec3};

use crate::octree::{Octree, permute};

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
    pub intensity: Vec<u16>,
}

impl PointChunk {
    pub fn len(&self) -> usize {
        self.positions.len()
    }

    /// Joins `chunks` into one chunk with contiguous columns, freeing each chunk
    /// once it is copied.
    pub fn concat(chunks: Vec<PointChunk>) -> PointChunk {
        let len = chunks.iter().map(PointChunk::len).sum();
        let mut all = PointChunk {
            positions: Vec::with_capacity(len),
            classes: Vec::with_capacity(len),
            colors: Vec::with_capacity(len),
            intensity: Vec::with_capacity(len),
        };

        for chunk in chunks {
            all.positions.extend(chunk.positions);
            all.classes.extend(chunk.classes);
            all.colors.extend(chunk.colors);
            all.intensity.extend(chunk.intensity);
        }

        all
    }

    /// The points in the order given by `order` (indices into this chunk), split
    /// into chunks of `chunk_size` points; only the last one may be shorter.
    pub fn reordered(&self, order: &[u32], chunk_size: usize) -> Vec<PointChunk> {
        order
            .chunks(chunk_size.max(1))
            .map(|part| PointChunk {
                positions: permute(&self.positions, part),
                classes: permute(&self.classes, part),
                colors: permute(&self.colors, part),
                intensity: permute(&self.intensity, part),
            })
            .collect()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ColorDepth {
    /// The point format has no RGB.
    None,
    /// Values 0-255 in 16-bit fields.
    Bits8,
    /// Values 0-65535, as the spec says.
    Bits16,
    /// Combined stats of tiles whose colour depths differ.
    Mixed,
}

#[derive(Debug)]
pub(crate) struct Stats {
    pub points: u64,
    pub class_counts: [u64; 256],
    pub color_depth: ColorDepth,
    pub local_min: Vec3,
    pub local_max: Vec3,
    pub intensity_min: u16,
    pub intensity_max: u16,
    pub elapsed: Duration,
}

impl Stats {
    /// Stats of no points: merging anything into them gives that thing back.
    pub fn empty() -> Self {
        Self {
            points: 0,
            class_counts: [0; 256],
            color_depth: ColorDepth::None,
            local_min: Vec3::splat(f32::INFINITY),
            local_max: Vec3::splat(f32::NEG_INFINITY),
            intensity_min: u16::MAX,
            intensity_max: u16::MIN,
            elapsed: Duration::ZERO,
        }
    }

    /// Adds `other` to these stats: counts and load times are summed, bounds and
    /// intensity ranges are joined.
    pub fn merge(&mut self, other: &Stats) {
        self.color_depth = if self.points == 0 || self.color_depth == other.color_depth {
            other.color_depth
        } else if other.points == 0 {
            self.color_depth
        } else {
            ColorDepth::Mixed
        };

        self.points += other.points;
        for (count, other_count) in self.class_counts.iter_mut().zip(&other.class_counts) {
            *count += other_count;
        }
        self.local_min = self.local_min.min(other.local_min);
        self.local_max = self.local_max.max(other.local_max);
        self.intensity_min = self.intensity_min.min(other.intensity_min);
        self.intensity_max = self.intensity_max.max(other.intensity_max);
        self.elapsed += other.elapsed;
    }
}

/// An axis-aligned box in global (e.g. UTM) coordinates.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Bounds {
    pub min: DVec3,
    pub max: DVec3,
}

impl Bounds {
    /// The smallest box that contains both.
    pub fn union(self, other: Bounds) -> Bounds {
        Bounds {
            min: self.min.min(other.min),
            max: self.max.max(other.max),
        }
    }

    /// The centre of the box, rounded to whole metres: the origin local
    /// coordinates are measured from.
    pub fn origin(self) -> DVec3 {
        ((self.min + self.max) * 0.5).round()
    }
}

/// One LAS/LAZ file, loaded with positions relative to its scene's origin.
///
/// Immutable once loaded, and shared between scenes with an `Arc`.
pub(crate) struct Tile {
    pub path: PathBuf,
    /// The points in octree order: node by node, breadth first, so coarse levels
    /// come first. Every chunk has the loader's batch size except the last, so
    /// point `i` is in chunk `i / batch size`.
    pub chunks: Vec<PointChunk>,
    pub stats: Stats,
    /// Level-of-detail octree; its nodes index into the points above.
    pub octree: Octree,
    /// Time to build the octree and reorder the points.
    pub octree_time: Duration,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stats(points: u64, class: usize, color_depth: ColorDepth, z: [f32; 2]) -> Stats {
        let mut stats = Stats::empty();
        stats.points = points;
        stats.class_counts[class] = points;
        stats.color_depth = color_depth;
        stats.local_min = Vec3::new(0.0, 0.0, z[0]);
        stats.local_max = Vec3::new(1.0, 1.0, z[1]);
        stats.intensity_min = 100 * points as u16;
        stats.intensity_max = 1000 * points as u16;
        stats.elapsed = Duration::from_secs(points);
        stats
    }

    #[test]
    fn merge_sums_counts_and_joins_ranges() {
        let mut combined = Stats::empty();
        combined.merge(&stats(2, 2, ColorDepth::Bits16, [-5.0, 10.0]));
        combined.merge(&stats(3, 6, ColorDepth::Bits16, [-1.0, 20.0]));

        assert_eq!(combined.points, 5);
        assert_eq!(combined.class_counts[2], 2);
        assert_eq!(combined.class_counts[6], 3);
        assert_eq!(combined.color_depth, ColorDepth::Bits16);
        assert_eq!(combined.local_min, Vec3::new(0.0, 0.0, -5.0));
        assert_eq!(combined.local_max, Vec3::new(1.0, 1.0, 20.0));
        assert_eq!(combined.intensity_min, 200);
        assert_eq!(combined.intensity_max, 3000);
        assert_eq!(combined.elapsed, Duration::from_secs(5));
    }

    #[test]
    fn merge_into_empty_gives_the_same_stats() {
        let mut combined = Stats::empty();
        combined.merge(&stats(4, 3, ColorDepth::Bits8, [1.0, 2.0]));

        assert_eq!(combined.points, 4);
        assert_eq!(combined.color_depth, ColorDepth::Bits8);
        assert_eq!(combined.local_min.z, 1.0);
        assert_eq!(combined.intensity_max, 4000);
    }

    #[test]
    fn merge_of_different_colour_depths_is_mixed() {
        let mut combined = Stats::empty();
        combined.merge(&stats(1, 2, ColorDepth::Bits8, [0.0, 1.0]));
        combined.merge(&stats(1, 2, ColorDepth::Bits16, [0.0, 1.0]));

        assert_eq!(combined.color_depth, ColorDepth::Mixed);
    }

    #[test]
    fn origin_is_the_rounded_centre_of_the_union() {
        let west = Bounds {
            min: DVec3::new(419_000.0, 4_462_000.0, 560.0),
            max: DVec3::new(420_000.0, 4_463_000.0, 621.3),
        };
        let east = Bounds {
            min: DVec3::new(420_000.0, 4_462_000.0, 570.0),
            max: DVec3::new(421_000.0, 4_463_000.0, 640.9),
        };
        let union = west.union(east);

        assert_eq!(union.min, DVec3::new(419_000.0, 4_462_000.0, 560.0));
        assert_eq!(union.max, DVec3::new(421_000.0, 4_463_000.0, 640.9));
        // Centre z = 600.45, rounded to whole metres.
        assert_eq!(union.origin(), DVec3::new(420_000.0, 4_462_500.0, 600.0));
    }
}
