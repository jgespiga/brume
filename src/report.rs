//! Summaries and colour scales derived from loaded points.
//!
//! [`class_rows`] uses counts collected by the loader. [`height_range`] and
//! [`intensity_range`] copy one attribute across all tiles, then select
//! percentile endpoints without sorting the entire copy. The shader clamps
//! values outside those endpoints so outliers do not set the colour scale.
//!
//! ```text
//! Stats.class_counts -----------------------> class rows
//! all tile heights -> select percentiles ---> shared height range
//! all intensities  -> select percentiles ---> shared intensity range
//! class_color + depth_color ----------------> GPU palettes
//! ```
//!
//! [`crate::scene::LoadedScene`] computes reports during loading. Palette
//! functions also supply the UI legend and octree box colours.

use crate::cloud::{PointChunk, Stats};

/// One nonempty classification in the scene-wide report.
pub struct ClassRow {
    /// Classification byte after the loader removes legacy flags.
    pub code: u8,
    /// Display label from [`class_name`].
    pub name: &'static str,
    /// Points assigned to this class.
    pub count: u64,
    /// Share of all scene points, expressed as a percentage.
    pub percent: f64,
}

/// Labels the supported class codes, including survey-specific codes.
/// Unlisted codes use "Other" without changing the stored code.
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

/// Number of distinct octree depth colours; deeper levels reuse them in a cycle.
/// Matches the length of `depths` in `Palettes` in points.wgsl.
pub const DEPTH_PALETTE_LEN: usize = 8;

/// Okabe-Ito depth colours with black replaced by light grey for the dark
/// background. Adjacent entries give neighbouring depths different colours.
const DEPTH_COLORS: [[u8; 3]; DEPTH_PALETTE_LEN] = [
    [86, 180, 233],  // sky blue
    [230, 159, 0],   // orange
    [0, 158, 115],   // bluish green
    [204, 121, 167], // reddish purple
    [240, 228, 66],  // yellow
    [0, 114, 178],   // blue
    [213, 94, 0],    // vermillion
    [220, 220, 220], // light grey
];

/// Maps each octree depth to the colour it is drawn with, in the "Octree depth"
/// colour mode and the node box overlay.
pub fn depth_color(depth: u8) -> [u8; 3] {
    DEPTH_COLORS[usize::from(depth) % DEPTH_PALETTE_LEN]
}

/// Builds rows from [`Stats::class_counts`], omitting zero counts and sorting
/// the most common classes first. An empty scene produces no rows.
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
/// several tiles. Uses integer division and includes fallback colours for points
/// without RGB. Returns `None` when there are no points.
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

/// Selects rounded ranks for percentiles in `0.0..=1.0`, with `low <= high`.
/// [`slice::select_nth_unstable_by`] partitions in place without a full sort.
/// Empty input returns `[0.0, 1.0]`; constant input can return equal endpoints.
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
    fn depth_colors_are_distinct_and_cycle() {
        let first: Vec<[u8; 3]> = (0..DEPTH_PALETTE_LEN as u8).map(depth_color).collect();
        for (i, a) in first.iter().enumerate() {
            assert!(
                first[i + 1..].iter().all(|b| a != b),
                "depth {i} is repeated"
            );
        }

        assert_eq!(depth_color(8), depth_color(0));
        assert_eq!(depth_color(13), depth_color(5));
    }
}
