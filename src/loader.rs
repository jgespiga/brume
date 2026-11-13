use std::{
    path::Path,
    sync::atomic::{AtomicBool, AtomicU64, Ordering},
    time::{Duration, Instant},
};

use anyhow::{Context, Result, bail};
use glam::{DVec3, Vec3};
use las::PointData;

use crate::cloud::{Cloud, ColorDepth, NO_COLOR, PointChunk, Stats};

/// Avoid reading the whole file at once. Instead, we
/// will read batches of aprox 1 million points.
const BATCH_SIZE: u64 = 1 << 20;

/// Shared between a loader thread and the UI: progress goes out, cancellation comes in.
#[derive(Default)]
pub(crate) struct LoadControl {
    /// Points read so far, updated after each batch.
    pub progress: AtomicU64,
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

/// Returns the number of points declared in the file's header, without reading them.
pub(crate) fn point_count(path: &Path) -> Result<u64> {
    let reader = las::Reader::from_path(path)
        .with_context(|| format!("could not open {}", path.display()))?;

    Ok(reader.header().number_of_points())
}

/// Reads every point of the file into a [`Cloud`].
///
/// After each batch, `control.progress` holds the number of points read so far,
/// so another thread can show how far the load has got. If `control.cancel` is
/// set, the load stops there and returns a "load cancelled" error.
pub(crate) fn load(path: &Path, control: &LoadControl) -> Result<(Cloud, Stats)> {
    let start = Instant::now();

    // Create las reader.
    let mut reader = las::Reader::from_path(path)?;
    let n_points = reader.header().number_of_points();

    // Calculate the origin based on the header bounds.
    let bounds = reader.header().bounds();
    let min = DVec3::new(bounds.min.x, bounds.min.y, bounds.min.z);
    let max = DVec3::new(bounds.max.x, bounds.max.y, bounds.max.z);

    let origin = ((min + max) * 0.5).round();

    // Calculate the number of batches we are going to process.
    let n_batches = n_points.div_ceil(BATCH_SIZE) as usize;

    let extended = reader.header().point_format().is_extended;
    let class_mask: u8 = if extended { 0xFF } else { 0x1F };

    let mut cloud = Cloud {
        origin,
        chunks: Vec::with_capacity(n_batches),
    };

    let mut stats = Stats {
        points: 0,
        class_counts: [0; 256],
        color_depth: ColorDepth::None,
        local_min: Vec3::splat(f32::INFINITY),
        local_max: Vec3::splat(f32::NEG_INFINITY),
        intensity_min: u16::MAX,
        intensity_max: u16::MIN,
        elapsed: Duration::ZERO,
    };

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

        cloud.chunks.push(chunk);

        stats.points += batch.len() as u64;
        control.progress.store(stats.points, Ordering::Relaxed);
        if control.cancel.load(Ordering::Relaxed) {
            bail!("load cancelled");
        }

        let read = reader.fill_points(BATCH_SIZE, &mut batch)?;
        if read == 0 {
            break;
        }
    }

    stats.elapsed = start.elapsed();

    Ok((cloud, stats))
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
