use std::{
    path::Path,
    sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
    time::Instant,
};

use anyhow::{Context, Result, bail};
use glam::DVec3;
use las::PointData;

use crate::cloud::{Bounds, ColorDepth, NO_COLOR, PointChunk, Stats, Tile};

/// Avoid reading the whole file at once. Instead, we
/// will read batches of aprox 1 million points.
const BATCH_SIZE: u64 = 1 << 20;

/// Shared between a loader thread and the UI: progress goes out, cancellation comes in.
#[derive(Default)]
pub(crate) struct LoadControl {
    /// Points read so far, across every tile of the load, updated after each batch.
    pub progress: AtomicU64,
    /// Index of the tile being read, for "tile i of N".
    pub tile: AtomicUsize,
    /// Set by the UI to ask the loader to stop at the next batch boundary.
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
    pub points: u64,
    /// Bounds of the points, in the file's (global) coordinates.
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
/// `origin` (global - origin, as f32).
///
/// After each batch, the batch's points are added to `control.progress`, so
/// another thread can show how far a load of one or several tiles has got. If
/// `control.cancel` is set, the load stops there and returns a "load cancelled" error.
pub(crate) fn load(path: &Path, origin: DVec3, control: &LoadControl) -> Result<Tile> {
    let start = Instant::now();

    // Create las reader.
    let mut reader = las::Reader::from_path(path)?;
    let n_points = reader.header().number_of_points();

    // Calculate the number of batches we are going to process.
    let n_batches = n_points.div_ceil(BATCH_SIZE) as usize;

    let extended = reader.header().point_format().is_extended;
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

    Ok(Tile {
        path: path.to_owned(),
        chunks,
        stats,
    })
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
