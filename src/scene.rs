use std::{path::Path, sync::atomic::AtomicU64};

use anyhow::Result;

use crate::{
    cloud::{Cloud, Stats},
    loader::load,
    report::{ClassRow, average_color, class_rows, height_range, intensity_range},
};

/// A loaded file with everything the viewer derives from it.
///
/// Built entirely on the loader thread, so the UI only has to upload it to the GPU.
pub struct LoadedScene {
    pub cloud: Cloud,
    pub stats: Stats,
    pub class_rows: Vec<ClassRow>,
    pub average_color: Option<[u64; 3]>,
    pub height_range: [f32; 2],
    pub intensity_range: [f32; 2],
}

/// Loads `path` and computes the report and colour ranges.
///
/// `progress` holds the number of points read so far; see [`load`].
pub fn load_scene(path: &Path, progress: &AtomicU64) -> Result<LoadedScene> {
    let (cloud, stats) = load(path, progress)?;

    Ok(LoadedScene {
        class_rows: class_rows(&stats),
        average_color: average_color(&cloud),
        height_range: height_range(&cloud),
        intensity_range: intensity_range(&cloud),
        cloud,
        stats,
    })
}

impl LoadedScene {
    /// Prints the report to stdout.
    pub fn log(&self) {
        let stats = &self.stats;

        println!("Time: {:.2}", stats.elapsed.as_secs_f64());

        println!("Origin (UTM):    {:.2?}", self.cloud.origin);
        println!("Local min (f32): {:.2?}", stats.local_min);
        println!("Local max (f32): {:.2?}", stats.local_max);
        println!("Color depth: {:?}", stats.color_depth);
        println!(
            "Intensity: {} - {}\n",
            stats.intensity_min, stats.intensity_max
        );

        for row in &self.class_rows {
            let ClassRow {
                code,
                name,
                count,
                percent,
            } = row;

            println!("{code:>3}  {name:<20} {count:>10}  {percent:>5.1}%");
        }
        if let Some(avg) = self.average_color {
            println!("\nAverage color:   rgb({}, {}, {})", avg[0], avg[1], avg[2]);
        }
    }
}
