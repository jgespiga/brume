use std::time::Instant;

use anyhow::{Context, Result};
use glam::{DVec3, Vec3};

use crate::cloud::{Cloud, PointChunk};

/// Avoid reading the whole file at once. Instead, we
/// will read batches of aprox 1 million points.
const BATCH_SIZE: u64 = 1 << 20;

pub(crate) fn read_file(reader: &mut las::Reader) -> Result<(Cloud)> {
    let start = Instant::now();

    let mut batch = reader.read_points(BATCH_SIZE)?;

    // Calculate the origin based on the header bounds.
    let bounds = reader.header().bounds();
    let min = DVec3::new(bounds.min.x, bounds.min.y, bounds.min.z);
    let max = DVec3::new(bounds.max.x, bounds.max.y, bounds.max.z);

    let origin = ((min + max) * 0.5).round();

    let n_points = reader.header().number_of_points();
    let n_batches = n_points.div_ceil(BATCH_SIZE) as usize;

    let mut cloud = Cloud {
        origin,
        chunks: Vec::with_capacity(n_batches),
    };

    let mut total: u64 = 0;
    let mut batches: u32 = 0;

    let mut local_min = Vec3::splat(f32::INFINITY);
    let mut local_max = Vec3::splat(f32::NEG_INFINITY);

    while !batch.is_empty() {
        let mut chunk = PointChunk {
            positions: Vec::with_capacity(batch.len()),
        };

        for ((x, y), z) in batch.x().zip(batch.y()).zip(batch.z()) {
            let local = (DVec3::new(x, y, z) - origin).as_vec3();

            local_min = local_min.min(local);
            local_max = local_max.max(local);

            chunk.positions.push(local.to_array());
        }

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

    println!("Origin (UTM):    {:.2?}", origin);
    println!("Local min (f32): {:.2?}", local_min);
    println!("Local max (f32): {:.2?}", local_max);

    Ok(cloud)
}
