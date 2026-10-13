use std::time::Duration;

use glam::{DVec3, Vec3};

use crate::cloud::{Cloud, ColorDepth, Stats};

/// Each row of the classes table.
pub struct ClassRow {
    pub code: u8,
    pub name: &'static str,
    pub count: u64,
    pub percent: f64,
}

// A Report containing all the important info.
pub struct Report {
    pub points: u64,
    pub origin: DVec3,
    pub local_min: Vec3,
    pub local_max: Vec3,
    pub color_depth: ColorDepth,
    pub intensity: (u16, u16),
    pub average_color: Option<[u64; 3]>,
    pub classes: Vec<ClassRow>,
    pub elapsed: Duration,
}

/// Map each class code with its actual class name.
pub fn class_name(code: u8) -> &'static str {
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

/// Returns each of the classes found in the file. Most common first.
pub fn class_rows(stats: &Stats) -> Vec<ClassRow> {
    let total = stats.points.max(1) as f64;
    let mut rows: Vec<ClassRow> = stats
        .class_counts
        .iter()
        .enumerate()
        .filter(|&(_, &count)| count > 0)
        .map(|(code, &count)| ClassRow {
            code: code as u8,
            name: class_name(code as u8),
            count,
            percent: 100.0 * count as f64 / total,
        })
        .collect();

    rows.sort_by_key(|row| std::cmp::Reverse(row.count));

    rows
}

/// Returns the average_color found on the cloud.
pub fn average_color(cloud: &Cloud) -> Option<[u64; 3]> {
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

    if n <= 0 {
        return None;
    }

    let avg = sum.map(|s| s / n);
    Some([avg[0], avg[1], avg[2]])
}

impl Report {
    /// Create a new [`Report`] containing all the information.
    pub fn new(cloud: &Cloud, stats: &Stats) -> Self {
        Self {
            points: stats.points,
            origin: cloud.origin,
            local_min: stats.local_min,
            local_max: stats.local_max,
            color_depth: stats.color_depth,
            intensity: (stats.intensity_min, stats.intensity_max),
            average_color: average_color(cloud),
            classes: class_rows(stats),
            elapsed: stats.elapsed,
        }
    }
}
