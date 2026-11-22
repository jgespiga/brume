//! CPU storage shared by loading, reports and rendering.
//!
//! [`Bounds`] keeps header coordinates in `f64`. The loader subtracts the scene
//! origin before converting positions to `f32`, so large global coordinates do
//! not consume the precision needed for nearby points.
//!
//! ```text
//! global position, f64 -- subtract shared origin, f64 --> offset, f64
//!                                                            |
//!                                                       cast to f32
//!                                                            |
//!                                                   PointChunk.positions
//! ```
//!
//! [`Tile`] joins the point columns with their [`Octree`] and [`Stats`]. Loaded
//! tiles are immutable; [`crate::scene::LoadPlan`] shares them through
//! [`std::sync::Arc`] when adding files to a scene.

use std::{path::PathBuf, time::Duration};

use glam::{DVec3, Vec3};

use crate::octree::{Octree, permute};

/// Opaque grey for point formats without RGB fields.
pub(crate) const NO_COLOR: [u8; 4] = [190, 190, 190, 255];

/// Point attributes in separate columns, a structure of arrays.
///
/// The same index addresses one point in every column. Loading visits one
/// attribute at a time, and rendering uploads positions and colours directly.
/// Classes and intensities combine with octree depths into a packed GPU column.
///
/// ```text
///                  point i       point i+1
/// positions     [ [x, y, z]  | [x, y, z]  | ... ]
/// colors        [ [r,g,b,a]  | [r,g,b,a]  | ... ]
/// classes       [ class      | class      | ... ]
/// intensity     [ intensity  | intensity  | ... ]
///
/// One struct per point would interleave those attributes instead:
/// [ position, color, class, intensity | position, color, ... ]
/// ```
///
/// A chunk can hold one decoded batch or an entire tile after
/// [`PointChunk::concat`]. [`PointChunk::permuted`] must reorder all columns
/// together to preserve their shared indices.
#[derive(Debug)]
pub(crate) struct PointChunk {
    /// Scene-relative positions in metres.
    pub positions: Vec<[f32; 3]>,
    /// Classification codes after removing flags from legacy formats.
    pub classes: Vec<u8>,
    /// RGBA bytes, with opaque alpha and [`NO_COLOR`] where RGB is absent.
    pub colors: Vec<[u8; 4]>,
    /// Original intensity values, before the shader maps them to grey.
    pub intensity: Vec<u16>,
}

impl PointChunk {
    /// Number of points. Every attribute column must have this length.
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

    /// Copies every column through the permutation from [`Octree::build`].
    /// The original columns remain available until the caller drops them.
    pub fn permuted(&self, order: &[u32]) -> PointChunk {
        PointChunk {
            positions: permute(&self.positions, order),
            classes: permute(&self.classes, order),
            colors: permute(&self.colors, order),
            intensity: permute(&self.intensity, order),
        }
    }
}

/// RGB interpretation inferred from the first decoded batch, or combined across
/// tiles by [`Stats::merge`].
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

/// Counts and ranges gathered during conversion, before octree construction.
/// [`Stats::merge`] combines tiles without revisiting their points.
#[derive(Debug)]
pub(crate) struct Stats {
    /// Number of converted points.
    pub points: u64,
    /// Counts indexed by the full classification byte.
    pub class_counts: [u64; 256],
    /// RGB interpretation, including mixed depths across tiles.
    pub color_depth: ColorDepth,
    /// Minimum converted position relative to the scene origin.
    pub local_min: Vec3,
    /// Maximum converted position relative to the scene origin.
    pub local_max: Vec3,
    /// Smallest observed intensity.
    pub intensity_min: u16,
    /// Largest observed intensity.
    pub intensity_max: u16,
    /// Reading and conversion time, excluding octree building and scene reports.
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

/// An axis-aligned box in global coordinates, such as UTM metres.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Bounds {
    /// Minimum global coordinate on each axis.
    pub min: DVec3,
    /// Maximum global coordinate on each axis.
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
/// Immutable once loaded, and shared between scenes with [`std::sync::Arc`].
pub(crate) struct Tile {
    /// Source path, used for labels and duplicate detection.
    pub path: PathBuf,
    /// All the tile's points, in contiguous columns, in octree order: node by node,
    /// breadth first, so coarse levels come first.
    pub points: PointChunk,
    /// Counts and ranges gathered before reordering.
    pub stats: Stats,
    /// Spatial hierarchy whose node ranges address [`Tile::points`].
    pub octree: Octree,
    /// Prefix counts from [`Octree::points_up_to_depth`]. The app uses this
    /// vector's length to determine the depth slider's upper limit.
    pub points_up_to_depth: Vec<u32>,
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
