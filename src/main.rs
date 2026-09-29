use std::time::Instant;

use anyhow::{Context, Result};

/// Avoid reading the whole file at once. Instead, we
/// will read batches of aprox 1 million points.
const BATCH_SIZE: u64 = 1 << 20;

fn read_file(reader: &mut las::Reader) -> Result<las::PointData> {
    let start = Instant::now();

    let mut batch = reader.read_points(BATCH_SIZE)?;
    let mut total: u64 = 0;
    let mut batches: u32 = 0;

    while !batch.is_empty() {
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

    Ok(batch)
}

fn main() -> Result<()> {
    let path = std::env::args().nth(1).context("usage: brume <file.laz>")?;

    let mut reader = las::Reader::from_path(path)?;

    let header = reader.header();

    let n_points = header.number_of_points();
    println!("{}", n_points);

    let batch = read_file(&mut reader)?;

    Ok(())
}
