use std::time::Instant;

use anyhow::{Context, Result};
use glam::DVec3;

use crate::{
    cloud::{Cloud, Stats},
    loader::load,
};

mod cloud;
mod loader;

/// Map each class code with its actual class name.
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

fn log_stats(cloud: &Cloud, stats: &Stats) {
    println!("Time: {:.2}", stats.elapsed.as_secs_f64());

    println!("Origin (UTM):    {:.2?}", cloud.origin);
    println!("Local min (f32): {:.2?}", stats.local_min);
    println!("Local max (f32): {:.2?}", stats.local_max);
    println!("Color depth: {:?}", stats.color_depth);

    let mut present: Vec<(u8, u64)> = stats
        .class_counts
        .iter()
        .enumerate()
        .filter(|&(_, &count)| count > 0)
        .map(|(code, &count)| (code as u8, count))
        .collect();
    present.sort_by_key(|&(_, count)| std::cmp::Reverse(count));

    let total = stats.points as f64;

    for (code, count) in present {
        let name = class_name(code);
        let pct = 100.0 * count as f64 / total;
        println!("{code:>3}  {name:<20} {count:>10}  {pct:>5.1}%");
    }

    let mut sum = [0u64; 3];
    let mut n: u64 = 0;

    for chunk in &cloud.chunks {
        for color in &chunk.colors {
            sum[0] += u64::from(color[0]);
            sum[1] += u64::from(color[1]);
            sum[2] += u64::from(color[2]);
            n += 1;
        }
    }

    if n > 0 {
        let avg = sum.map(|s| s / n);
        println!("Average color:   rgb({}, {}, {})", avg[0], avg[1], avg[2]);
    }
}

fn main() -> Result<()> {
    let path_str = std::env::args().nth(1).context("usage: brume <file.laz>")?;
    let path = std::path::Path::new(&path_str);

    let (cloud, stats) = load(path)?;

    log_stats(&cloud, &stats);

    Ok(())
}
