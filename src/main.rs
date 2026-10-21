use anyhow::{Context, Result};

use crate::{
    app::Brume,
    cloud::{Cloud, Stats},
    loader::load,
    report::{ClassRow, average_color, class_rows},
};

mod app;
mod camera;
mod cloud;
mod loader;
mod render;
mod report;

fn log_stats(cloud: &Cloud, stats: &Stats) {
    println!("Time: {:.2}", stats.elapsed.as_secs_f64());

    println!("Origin (UTM):    {:.2?}", cloud.origin);
    println!("Local min (f32): {:.2?}", stats.local_min);
    println!("Local max (f32): {:.2?}", stats.local_max);
    println!("Color depth: {:?}", stats.color_depth);
    println!(
        "Intensity: {} - {}\n",
        stats.intensity_min, stats.intensity_max
    );

    for row in class_rows(stats) {
        let ClassRow {
            code,
            name,
            count,
            percent,
        } = row;

        println!("{code:>3}  {name:<20} {count:>10}  {percent:>5.1}%");
    }
    if let Some(avg) = average_color(cloud) {
        println!("\nAverage color:   rgb({}, {}, {})", avg[0], avg[1], avg[2]);
    }
}

fn main() -> Result<()> {
    let path_str = std::env::args().nth(1).context("usage: brume <file.laz>")?;
    let path = std::path::Path::new(&path_str);

    let (cloud, stats) = load(path)?;

    log_stats(&cloud, &stats);

    let options = eframe::NativeOptions {
        depth_buffer: render::DEPTH_BITS,
        ..Default::default()
    };
    eframe::run_native(
        "Brume",
        options,
        Box::new(|cc| {
            let app = Brume::new(cc, cloud, stats).ok_or("Brume needs eframe's wgpu backend")?;
            Ok(Box::new(app))
        }),
    )
    .map_err(|e| anyhow::anyhow!("viewer error: {e}"))?;

    Ok(())
}
