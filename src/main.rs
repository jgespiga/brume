use std::path::PathBuf;

use anyhow::Result;
use eframe::egui_wgpu::wgpu;

use crate::app::Brume;

mod app;
mod camera;
mod cloud;
mod loader;
mod lod;
mod octree;
mod perf;
mod render;
mod report;
mod scene;

/// Command-line arguments: `brume [--no-vsync] [tile.laz...]`, flag in any position.
struct Args {
    /// Files to open at startup, as one scene. Without any, the window starts empty.
    paths: Vec<PathBuf>,
    /// Present frames without waiting for vsync, so frame times measure Brume
    /// rather than the monitor's refresh rate. For benchmarking only.
    no_vsync: bool,
}

fn parse_args() -> Args {
    let (flags, positional): (Vec<String>, Vec<String>) = std::env::args()
        .skip(1)
        .partition(|arg| arg == "--no-vsync");

    Args {
        paths: positional.into_iter().map(PathBuf::from).collect(),
        no_vsync: !flags.is_empty(),
    }
}

fn main() -> Result<()> {
    let args = parse_args();

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
        Box::new(move |cc| {
            let app = Brume::new(cc, args.paths).ok_or("Brume needs eframe's wgpu backend")?;
            Ok(Box::new(app))
        }),
    )
    .map_err(|e| anyhow::anyhow!("viewer error: {e}"))?;

    Ok(())
}
