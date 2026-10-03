use std::{
    path::Path,
    time::{Duration, Instant},
};

use anyhow::{Context, Result};
use glam::{DVec3, Vec3};
use las::PointData;

use crate::cloud::{Cloud, ColorDepth, NO_COLOR, PointChunk, Stats};

/// Avoid reading the whole file at once. Instead, we
/// will read batches of aprox 1 million points.
const BATCH_SIZE: u64 = 1 << 20;

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

pub(crate) fn load(path: &Path) -> Result<(Cloud, Stats)> {
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
        elapsed: Duration::ZERO,
    };

    let mut batch = reader.read_points(BATCH_SIZE)?;
    stats.color_depth = detect_color_depth(&batch);

    while !batch.is_empty() {
        let mut chunk = PointChunk {
            positions: Vec::with_capacity(batch.len()),
            classes: Vec::with_capacity(batch.len()),
            colors: Vec::with_capacity(batch.len()),
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

        cloud.chunks.push(chunk);

        stats.points += batch.len() as u64;

        let read = reader.fill_points(BATCH_SIZE, &mut batch)?;
        if read == 0 {
            break;
        }
    }

    stats.elapsed = start.elapsed();

    Ok((cloud, stats))
}
