use anyhow::{Context, Result};
use eframe::egui_wgpu::wgpu;

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
mod perf;
mod render;
mod report;

/// Command-line arguments: `brume [--no-vsync] <file.laz>`, flag in any position.
struct Args {
    path: String,
    /// Present frames without waiting for vsync, so frame times measure Brume
    /// rather than the monitor's refresh rate. For benchmarking only.
    no_vsync: bool,
}

fn parse_args() -> Result<Args> {
    let (flags, positional): (Vec<String>, Vec<String>) = std::env::args()
        .skip(1)
        .partition(|arg| arg == "--no-vsync");

    let path = positional
        .into_iter()
        .next()
        .context("usage: brume [--no-vsync] <file.laz>")?;

    Ok(Args {
        path,
        no_vsync: !flags.is_empty(),
    })
}

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
    let args = parse_args()?;
    let path = std::path::Path::new(&args.path);

    let (cloud, stats) = load(path)?;

    log_stats(&cloud, &stats);

    let mut options = eframe::NativeOptions {
        depth_buffer: render::DEPTH_BITS,
        ..Default::default()
    };
    if args.no_vsync {
        // `glow_options.vsync` only applies to the glow backend: wgpu reads the
        // surface's present mode instead.
        options.wgpu_options.surface.present_mode = wgpu::PresentMode::AutoNoVsync;
    }
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
