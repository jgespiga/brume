//! Decode LAS/LAZ points, convert their attributes and build each tile's octree.
//!
//! [`load`] runs on the background thread started by [`crate::app::Brume`].
//! The LAS reader reuses its decoded batch buffer, while converted [`PointChunk`]
//! columns accumulate until the entire tile is ready for sampling.
//!
//! ```text
//! LAS/LAZ -> reused PointData batch -> local attribute columns
//!                  ^                         |
//!                  | next batch              v
//!                  +------------------ Vec<PointChunk>
//!                                            |
//!                                  concat -> Octree::build
//!                                            |
//!                                  permute all columns -> Tile
//! ```
//!
//! Batching bounds the decoded buffer, not total tile memory. Concatenation and
//! permutation allocate new columns. [`LoadControl`] carries progress and
//! cooperative cancellation without transferring partially loaded tiles.

use std::{
    path::Path,
    sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
    time::Instant,
};

use anyhow::{Context, Result, bail};
use glam::DVec3;
use las::PointData;

use crate::{
    cloud::{Bounds, ColorDepth, NO_COLOR, PointChunk, Stats, Tile},
    octree::{Octree, OctreeParams},
};

/// Points per decoded batch, `2^20`. Reusing the batch avoids allocating a
/// decoded representation of the entire file at once.
const BATCH_SIZE: u64 = 1 << 20;

/// Shared between a loader thread and the UI: progress goes out, cancellation comes in.
///
/// Relaxed atomics suffice because these fields carry independent progress hints
/// and a stop request. The result channel transfers the completed scene.
#[derive(Default)]
pub(crate) struct LoadControl {
    /// Points read so far, across every tile of the load, updated after each batch.
    pub progress: AtomicU64,
    /// Index of the tile being read, for "tile i of N".
    pub tile: AtomicUsize,
    /// Whether the octree of the current tile is being built.
    pub building_octree: AtomicBool,
    /// Set by the UI to ask the loader to stop at the next batch boundary, or the
    /// next octree node.
    pub cancel: AtomicBool,
}

/// Returns whether `path` has a `.las` or `.laz` extension, ignoring case.
pub(crate) fn is_lidar_file(path: &Path) -> bool {
    path.extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| {
            extension.eq_ignore_ascii_case("las") || extension.eq_ignore_ascii_case("laz")
        })
}

/// Infers RGB depth from the first batch. A channel above 255 selects 16-bit
/// conversion for the whole file; otherwise the loader treats RGB as 8-bit data.
fn detect_color_depth(batch: &PointData) -> ColorDepth {
    match batch.rgb() {
        None => ColorDepth::None,
        Some(rgb) => {
            let max = rgb.map(|(r, g, b)| r.max(g).max(b)).max().unwrap_or(0);
            if max > 255 {
                ColorDepth::Bits16
            } else {
                ColorDepth::Bits8
            }
        }
    }
}

/// What a file's header says, read without reading any point.
pub(crate) struct HeaderInfo {
    /// Declared count used for progress and GPU buffer checks.
    pub points: u64,
    /// Declared bounds in the file's global coordinates.
    pub bounds: Bounds,
}

/// Reads the point count and bounds from the file's header, without reading points.
pub(crate) fn read_header_info(path: &Path) -> Result<HeaderInfo> {
    let reader = las::Reader::from_path(path)
        .with_context(|| format!("could not open {}", path.display()))?;
    let header = reader.header();
    let bounds = header.bounds();

    Ok(HeaderInfo {
        points: header.number_of_points(),
        bounds: Bounds {
            min: DVec3::new(bounds.min.x, bounds.min.y, bounds.min.z),
            max: DVec3::new(bounds.max.x, bounds.max.y, bounds.max.z),
        },
    })
}

/// Reads every point of the file into a [`Tile`], with positions relative to
/// `origin`. Subtraction happens in `f64` before conversion to `f32`.
///
/// After each batch, [`LoadControl::progress`] includes the converted points.
/// [`LoadControl::cancel`] stops work at a batch boundary or before an octree
/// node. Concatenation, permutation and report calculations do not check it.
/// Read errors and cancellation return no partial [`Tile`].
pub(crate) fn load(path: &Path, origin: DVec3, control: &LoadControl) -> Result<Tile> {
    let start = Instant::now();

    let mut reader = las::Reader::from_path(path)?;
    let n_points = reader.header().number_of_points();

    let n_batches = n_points.div_ceil(BATCH_SIZE) as usize;

    let extended = reader.header().point_format().is_extended;
    // Legacy formats share the classification byte with flags. Extended formats
    // give the code a full byte, so their upper bits must survive.
    let class_mask: u8 = if extended { 0xFF } else { 0x1F };

    let mut chunks = Vec::with_capacity(n_batches);
    let mut stats = Stats::empty();

    let mut batch = reader.read_points(BATCH_SIZE)?;
    stats.color_depth = detect_color_depth(&batch);

    while !batch.is_empty() {
        let mut chunk = PointChunk {
            positions: Vec::with_capacity(batch.len()),
            classes: Vec::with_capacity(batch.len()),
            colors: Vec::with_capacity(batch.len()),
            intensity: Vec::with_capacity(batch.len()),
        };

        for ((x, y), z) in batch.x().zip(batch.y()).zip(batch.z()) {
            let local = (DVec3::new(x, y, z) - origin).as_vec3();

            stats.local_min = stats.local_min.min(local);
            stats.local_max = stats.local_max.max(local);

            chunk.positions.push(local.to_array());
        }

        for class in batch.classification() {
            let class = class & class_mask;
            stats.class_counts[usize::from(class)] += 1;
            chunk.classes.push(class);
        }

        for intensity in batch.intensity() {
            stats.intensity_min = stats.intensity_min.min(intensity);
            stats.intensity_max = stats.intensity_max.max(intensity);
            chunk.intensity.push(intensity);
        }

        match (batch.rgb(), stats.color_depth) {
            (Some(rgb), ColorDepth::Bits16) => {
                for (r, g, b) in rgb {
                    chunk
                        .colors
                        .push([(r >> 8) as u8, (g >> 8) as u8, (b >> 8) as u8, 255]);
                }
            }

            (Some(rgb), _) => {
                for (r, g, b) in rgb {
                    chunk
                        .colors
                        .push([r.min(255) as u8, g.min(255) as u8, b.min(255) as u8, 255]);
                }
            }

            (None, _) => chunk.colors.resize(batch.len(), NO_COLOR),
        }

        debug_assert_eq!(chunk.classes.len(), chunk.len());
        debug_assert_eq!(chunk.colors.len(), chunk.len());
        debug_assert_eq!(chunk.intensity.len(), chunk.len());

        chunks.push(chunk);

        stats.points += batch.len() as u64;
        control
            .progress
            .fetch_add(batch.len() as u64, Ordering::Relaxed);
        if control.cancel.load(Ordering::Relaxed) {
            bail!("load cancelled");
        }

        let read = reader.fill_points(BATCH_SIZE, &mut batch)?;
        if read == 0 {
            break;
        }
    }

    stats.elapsed = start.elapsed();

    let octree_start = Instant::now();
    let (points, octree) = build_octree(chunks, control)?;
    let octree_time = octree_start.elapsed();

    Ok(Tile {
        path: path.to_owned(),
        points,
        stats,
        points_up_to_depth: octree.points_up_to_depth(),
        octree,
        octree_time,
    })
}

/// Joins batches and applies the order from [`Octree::build`] to every column.
/// The tree indexes the reordered points, so both must leave this function
/// together. [`LoadControl::building_octree`] covers concatenation and reordering
/// as well as tree construction.
fn build_octree(chunks: Vec<PointChunk>, control: &LoadControl) -> Result<(PointChunk, Octree)> {
    control.building_octree.store(true, Ordering::Relaxed);

    // The build needs all positions in one slice.
    let points = PointChunk::concat(chunks);
    let built = Octree::build(&points.positions, &OctreeParams::default(), || {
        control.cancel.load(Ordering::Relaxed)
    });
    let Some((octree, order)) = built else {
        bail!("load cancelled");
    };
    let points = points.permuted(&order);

    control.building_octree.store(false, Ordering::Relaxed);

    Ok((points, octree))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_las_and_laz_in_any_case() {
        for name in ["tile.las", "tile.laz", "tile.LAZ", "dir/tile.LaS"] {
            assert!(is_lidar_file(Path::new(name)), "{name}");
        }
    }

    #[test]
    fn rejects_other_extensions() {
        for name in ["tile.txt", "tile.copc", "tile.las.zip", "tile", "laz"] {
            assert!(!is_lidar_file(Path::new(name)), "{name}");
        }
    }
}
