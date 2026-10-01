use std::{path::Path, time::Instant};

use anyhow::{Context, Result};
use egui::epaint::tessellator;
use glam::{DVec3, Vec3};

use crate::cloud::{Cloud, PointChunk, Stats};

/// Avoid reading the whole file at once. Instead, we
/// will read batches of aprox 1 million points.
const BATCH_SIZE: u64 = 1 << 20;

fn class_name(code: u8) -> &'static str {
    match code {
        1 => "Unclassified",
        2 => "Ground",
        3 => "Low vegetation",
        4 => "Medium vegetation",
        5 => "High vegetation",
        6 => "Building",
        7 => "Noise",
        9 => "Water",
        12 => "Overlap",
        14 => "Power line",
        15 => "Transmission tower",
        17 => "Bridge",
        64 => "Vehicle",
        65 => "Wall",
        67 => "Solar panel",
        68 => "Wind turbine",
        70 => "Antenna",
        _ => "Other",
    }
}

pub(crate) fn load(path: &Path) -> Result<(Cloud, Stats)> {
    let start = Instant::now();

    // Create las reader.
    let mut reader = las::Reader::from_path(path)?;
    let n_points = reader.header().number_of_points();
    println!("Number of points -> {}", n_points);

    // Calculate the origin based on the header bounds.
    let bounds = reader.header().bounds();
    let min = DVec3::new(bounds.min.x, bounds.min.y, bounds.min.z);
    let max = DVec3::new(bounds.max.x, bounds.max.y, bounds.max.z);

    let origin = ((min + max) * 0.5).round();

    // Calculate the number of batches we are going to process.
    let n_points = reader.header().number_of_points();
    let n_batches = n_points.div_ceil(BATCH_SIZE) as usize;

    let extended = reader.header().point_format().is_extended;
    let class_mask: u8 = if extended { 0xFF } else { 0x1F };

    let mut cloud = Cloud {
        origin,
        chunks: Vec::with_capacity(n_batches),
    };

    let mut stats = Stats {
        class_counts: [0; 256],
    };

    let mut total: u64 = 0;
    let mut batches: u32 = 0;

    let mut local_min = Vec3::splat(f32::INFINITY);
    let mut local_max = Vec3::splat(f32::NEG_INFINITY);

    let mut batch = reader.read_points(BATCH_SIZE)?;

    while !batch.is_empty() {
        let mut chunk = PointChunk {
            positions: Vec::with_capacity(batch.len()),
            classes: Vec::with_capacity(batch.len()),
        };

        for ((x, y), z) in batch.x().zip(batch.y()).zip(batch.z()) {
            let local = (DVec3::new(x, y, z) - origin).as_vec3();

            local_min = local_min.min(local);
            local_max = local_max.max(local);

            chunk.positions.push(local.to_array());
        }

        for class in batch.classification() {
            let class = class & class_mask;
            stats.class_counts[usize::from(class)] += 1;
            chunk.classes.push(class);
        }

        debug_assert_eq!(chunk.classes.len(), chunk.len());

        cloud.chunks.push(chunk);

        total += batch.len() as u64;
        batches += 1;

        let read = reader.fill_points(BATCH_SIZE, &mut batch)?;
        if read == 0 {
            break;
        }
    }

    let elapsed = start.elapsed();

    println!("Time -> {:.2}", elapsed.as_secs_f64());
    println!(
        "Points read -> {} ; Number of batches -> {}",
        total, batches
    );

    println!("Origin (UTM):    {:.2?}", cloud.origin);
    println!("Local min (f32): {:.2?}", local_min);
    println!("Local max (f32): {:.2?}", local_max);

    let mut present: Vec<(u8, u64)> = stats
        .class_counts
        .iter()
        .enumerate()
        .filter(|&(_, &count)| count > 0)
        .map(|(code, &count)| (code as u8, count))
        .collect();
    present.sort_by_key(|&(_, count)| std::cmp::Reverse(count));

    let total = n_points as f64;

    for (code, count) in present {
        let name = class_name(code);
        let pct = 100.0 * count as f64 / total;
        println!("{code:>3}  {name:<20} {count:>10}  {pct:>5.1}%");
    }

    Ok((cloud, stats))
}
