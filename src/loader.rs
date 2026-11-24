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

/// The colour depth of a file, from its largest RGB channel over every point.
///
/// `None` means the point format has no RGB. The LAS spec asks for 16-bit
/// channels, but many files store 0-255 in them: any channel above 255 means
/// the file uses the full 16-bit range, otherwise it is treated as 8-bit.
fn color_depth(max_channel: Option<u16>) -> ColorDepth {
    match max_channel {
        None => ColorDepth::None,
        Some(max) if max > 255 => ColorDepth::Bits16,
        Some(_) => ColorDepth::Bits8,
    }
}

/// Converts one RGB value stored at `depth` to the 8-bit RGBA the GPU reads:
/// 16-bit channels keep their high byte, 8-bit ones are clamped to 255.
fn to_rgba8([r, g, b]: [u16; 3], depth: ColorDepth) -> [u8; 4] {
    match depth {
        ColorDepth::Bits16 => [(r >> 8) as u8, (g >> 8) as u8, (b >> 8) as u8, 255],
        _ => [r.min(255) as u8, g.min(255) as u8, b.min(255) as u8, 255],
    }
}

/// Gives every chunk its 8-bit colours from `rgb`, the raw colours of all the
/// chunks in order, converted at `depth`. Points without a raw colour (files
/// whose format has no RGB) get [`NO_COLOR`].
fn fill_colors(chunks: &mut [PointChunk], rgb: &[[u16; 3]], depth: ColorDepth) {
    let mut rgb = rgb.iter();
    for chunk in chunks {
        chunk.colors = rgb
            .by_ref()
            .take(chunk.len())
            .map(|&color| to_rgba8(color, depth))
            .collect();
        chunk.colors.resize(chunk.len(), NO_COLOR);
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

    // RGB stays 16-bit until every batch is read: only then is the file's colour
    // depth known, so a high channel late in the file can't be clamped by a
    // decision made on the first batch. Costs 6 bytes per point until converted.
    let has_color = reader.header().point_format().has_color;
    let mut rgb: Vec<[u16; 3]> = Vec::with_capacity(if has_color { n_points as usize } else { 0 });
    let mut max_channel: Option<u16> = None;

    let mut batch = reader.read_points(BATCH_SIZE)?;

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

        if let Some(colors) = batch.rgb() {
            for (r, g, b) in colors {
                max_channel = Some(max_channel.unwrap_or(0).max(r).max(g).max(b));
                rgb.push([r, g, b]);
            }
        }

        debug_assert_eq!(chunk.classes.len(), chunk.len());
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

    stats.color_depth = color_depth(max_channel);
    fill_colors(&mut chunks, &rgb, stats.color_depth);
    drop(rgb);

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

    fn chunk(len: usize) -> PointChunk {
        PointChunk {
            positions: vec![[0.0; 3]; len],
            classes: vec![0; len],
            colors: Vec::new(),
            intensity: vec![0; len],
        }
    }

    #[test]
    fn colour_depth_comes_from_the_largest_channel() {
        assert_eq!(color_depth(None), ColorDepth::None);
        assert_eq!(color_depth(Some(255)), ColorDepth::Bits8);
        assert_eq!(color_depth(Some(256)), ColorDepth::Bits16);
    }

    #[test]
    fn a_high_channel_in_a_later_batch_sets_the_depth_of_every_batch() {
        // The first batch alone looks 8-bit; the second shows the file is 16-bit.
        let rgb = [[200, 100, 0], [65_535, 32_768, 256]];
        let mut chunks = [chunk(1), chunk(1)];
        let depth = color_depth(rgb.iter().flatten().copied().max());

        fill_colors(&mut chunks, &rgb, depth);

        assert_eq!(depth, ColorDepth::Bits16);
        assert_eq!(chunks[0].colors, vec![[0, 0, 0, 255]]);
        assert_eq!(chunks[1].colors, vec![[255, 128, 1, 255]]);
    }

    #[test]
    fn eight_bit_colours_keep_their_values_and_missing_ones_are_grey() {
        let mut chunks = [chunk(2)];
        fill_colors(&mut chunks, &[[200, 255, 7], [1, 2, 3]], ColorDepth::Bits8);
        assert_eq!(chunks[0].colors, vec![[200, 255, 7, 255], [1, 2, 3, 255]]);

        let mut chunks = [chunk(3)];
        fill_colors(&mut chunks, &[], ColorDepth::None);
        assert_eq!(chunks[0].colors, vec![NO_COLOR; 3]);
    }

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
