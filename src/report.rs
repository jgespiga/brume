use crate::cloud::{PointChunk, Stats};

/// Each row of the classes table.
pub struct ClassRow {
    pub code: u8,
    pub name: &'static str,
    pub count: u64,
    pub percent: f64,
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

/// Maps each class code to the RGB colour it is drawn with.
pub fn class_color(code: u8) -> [u8; 3] {
    match code {
        1 => [190, 190, 190],
        2 => [170, 130, 80],
        3 => [150, 210, 110],
        4 => [80, 165, 70],
        5 => [30, 105, 50],
        6 => [220, 80, 60],
        7 => [255, 0, 255],
        9 => [50, 120, 220],
        12 => [130, 130, 130],
        14 => [250, 220, 50],
        15 => [245, 150, 30],
        17 => [150, 100, 170],
        64 => [0, 220, 220],
        65 => [240, 170, 160],
        67 => [90, 80, 220],
        68 => [235, 245, 255],
        70 => [255, 120, 170],
        _ => [90, 90, 90],
    }
}

/// Octree depth drawn at the top of the viridis ramp; deeper nodes look the same.
/// Matches `DEEPEST_COLOURED_DEPTH` in points.wgsl.
const DEEPEST_COLOURED_DEPTH: f32 = 15.0;

/// The viridis colour map at `t` (0..1: dark purple, blue, green, yellow), with the
/// same polynomial fit as `viridis` in points.wgsl.
pub fn viridis(t: f32) -> [u8; 3] {
    const C: [[f32; 3]; 7] = [
        [0.277_727_33, 0.005_407_344_5, 0.334_099_8],
        [0.105_093_04, 1.404_613_5, 1.384_590_2],
        [-0.330_861_83, 0.214_847_56, 0.095_095_16],
        [-4.634_230_5, -5.799_101, -19.332_441],
        [6.228_27, 14.179_933, 56.690_55],
        [4.776_385, -13.745_145, -65.353_03],
        [-5.435_456, 4.645_852_6, 26.312_435],
    ];

    let t = t.clamp(0.0, 1.0);
    std::array::from_fn(|channel| {
        // Horner's method, from the highest power down.
        let value = C.iter().rev().fold(0.0, |value, c| value * t + c[channel]);
        (value.clamp(0.0, 1.0) * 255.0).round() as u8
    })
}

/// The colour of octree depth `depth`, as the "Octree depth" colour mode draws it.
pub fn depth_color(depth: u8) -> [u8; 3] {
    viridis(f32::from(depth) / DEEPEST_COLOURED_DEPTH)
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

/// Returns the average colour of the points in `chunks`, which may come from
/// several tiles.
pub fn average_color<'a>(chunks: impl IntoIterator<Item = &'a PointChunk>) -> Option<[u64; 3]> {
    let mut sum = [0u64; 3];
    let mut n: u64 = 0;

    for chunk in chunks {
        for color in &chunk.colors {
            sum[0] += u64::from(color[0]);
            sum[1] += u64::from(color[1]);
            sum[2] += u64::from(color[2]);

            n += 1;
        }
    }

    if n == 0 {
        return None;
    }

    let avg = sum.map(|s| s / n);
    Some([avg[0], avg[1], avg[2]])
}

/// Returns the values at the `low` and `high` percentiles (0.0..=1.0).
/// Reorders `values` in place.
pub fn percentile_range(values: &mut [f32], low: f64, high: f64) -> [f32; 2] {
    debug_assert!(low <= high && (0.0..=1.0).contains(&low) && (0.0..=1.0).contains(&high));

    if values.is_empty() {
        return [0.0, 1.0];
    }

    let last = (values.len() - 1) as f64;

    [low, high].map(|percentile| {
        let k = (last * percentile).round() as usize;
        values.select_nth_unstable_by(k, |a, b| a.total_cmp(b));
        values[k]
    })
}

/// Returns the p1-p99 range of the heights of the points in `chunks`, which may
/// come from several tiles.
pub fn height_range<'a>(chunks: impl IntoIterator<Item = &'a PointChunk>) -> [f32; 2] {
    let mut heights = chunks
        .into_iter()
        .flat_map(|chunk| chunk.positions.iter().map(|position| position[2]))
        .collect::<Vec<f32>>();

    percentile_range(&mut heights, 0.01, 0.99)
}

/// Returns the p2-p98 range of the intensities of the points in `chunks`, which
/// may come from several tiles.
pub fn intensity_range<'a>(chunks: impl IntoIterator<Item = &'a PointChunk>) -> [f32; 2] {
    let mut intensities = chunks
        .into_iter()
        .flat_map(|chunk| chunk.intensity.iter().copied().map(f32::from))
        .collect::<Vec<f32>>();

    percentile_range(&mut intensities, 0.02, 0.98)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn viridis_goes_from_dark_purple_to_yellow() {
        // The shader's polynomial at t = 0 and t = 1, as 8-bit colours.
        assert_eq!(viridis(0.0), [71, 1, 85]);
        assert_eq!(viridis(1.0), [252, 231, 33]);
        assert_eq!(viridis(-1.0), viridis(0.0));
        assert_eq!(depth_color(15), viridis(1.0));
        assert_eq!(depth_color(40), viridis(1.0));
    }
}
