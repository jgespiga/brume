//! Brume is a LAS/LAZ point cloud viewer built with egui and wgpu.
//!
//! Files share a scene origin so the loader can subtract global coordinates in
//! `f64` before storing local `f32` positions. Each tile owns an octree and point
//! columns in node order. The UI selects visible node ranges each frame; the GPU
//! draws those points as instanced squares.
//!
//! ```text
//! UI thread                         brume-loader thread
//! paths -> LoadPlan ----------------> LAS/LAZ batches
//!   header and buffer checks              |
//!                                  local point columns
//!                                        |
//!                                  octree + permutation
//!                                        |
//! Scene <------- result channel ---- LoadedScene
//!   |
//! camera + settings -> LOD selection -> merged ranges
//!                                        |
//!                              PointsCallback + tile buffers
//!                                        |
//!                                  WGSL -> pixels
//! ```
//!
//! Start at [`app::Brume`] for the application lifecycle. Follow [`scene::LoadPlan`]
//! and [`loader::load`] for input, [`octree::Octree::build`] for sampling, and
//! [`lod::select`] for per-frame detail. [`render::PointRenderer`] documents the
//! GPU layout shared with the embedded `points.wgsl` shader.

use std::path::PathBuf;

use anyhow::Result;
use eframe::egui_wgpu::wgpu;

use crate::app::Brume;

mod app;
mod camera;
mod cloud;
mod edl;
mod loader;
mod lod;
mod octree;
mod perf;
mod render;
mod report;
mod scene;
mod vnt;

/// Command-line arguments: `brume [--no-vsync] [tile.laz...]`, flag in any position.
struct Args {
    /// Files to open at startup, as one scene. Without any, the window starts empty.
    paths: Vec<PathBuf>,
    /// Present frames without waiting for vsync, so frame times measure Brume
    /// rather than the monitor's refresh rate. For benchmarking only.
    no_vsync: bool,
}

/// Recognizes `--no-vsync` anywhere in the argument list and treats every other
/// argument as a path. File validation belongs to [`Brume`].
fn parse_args() -> Args {
    let (flags, positional): (Vec<String>, Vec<String>) = std::env::args()
        .skip(1)
        .partition(|arg| arg == "--no-vsync");

    Args {
        paths: positional.into_iter().map(PathBuf::from).collect(),
        no_vsync: !flags.is_empty(),
    }
}

/// Starts eframe with the wgpu backend. No depth buffer is requested: the points
/// pass tests against its own (see [`edl`]), and egui's pass needs none.
/// [`Brume::new`] rejects a context without the wgpu render state.
fn main() -> Result<()> {
    let args = parse_args();

    let mut options = eframe::NativeOptions::default();
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
