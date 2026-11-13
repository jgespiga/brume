use std::{
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
        mpsc::{self, Receiver, TryRecvError},
    },
    thread,
    time::{Duration, Instant},
};

use anyhow::Result;
use eframe::{egui, egui_wgpu};

use crate::{
    camera::OrbitCamera,
    cloud::Cloud,
    loader::point_count,
    perf::FrameTimes,
    render::{ColorMode, PointRenderer, PointsCallback, instances_to_draw},
    scene::{LoadedScene, load_scene},
};

/// Number of recent frames the performance figures are computed over.
const FRAME_WINDOW: usize = 120;

/// How often the loading screen redraws to update the progress bar and elapsed time.
const LOADING_REFRESH: Duration = Duration::from_millis(50);

pub struct Brume {
    state: AppState,
    view: ViewSettings,
}

/// What the window is showing: Loading → Ready, or Loading → Failed.
enum AppState {
    /// A loader thread is reading the file.
    Loading {
        receiver: Receiver<Result<LoadedScene>>,
        /// Points read so far, written by the loader thread.
        progress: Arc<AtomicU64>,
        /// Points declared in the file's header.
        total: u64,
        path: PathBuf,
        started: Instant,
    },
    /// The file is loaded and uploaded to the GPU.
    Ready(Box<ReadyState>),
    /// Loading failed. The window stays open to show why.
    Failed { path: PathBuf, error: String },
}

/// A loaded file and the camera looking at it.
struct ReadyState {
    scene: LoadedScene,
    camera: OrbitCamera,
}

/// View settings and measurements that don't depend on the loaded file, so they
/// survive a reload.
struct ViewSettings {
    point_size: f32,
    color_mode: ColorMode,
    frame_times: FrameTimes,
    continuous_repaint: bool,
    /// The first frame after a measurement starts still spans the idle gap before
    /// it, so it is not recorded.
    skip_next_frame: bool,
    /// Fraction of each chunk to draw (0.01..=1.0). Benchmarking aid, see
    /// [`instances_to_draw`].
    draw_fraction: f32,
    gpu_info: String,
    /// Size of the 3D view in physical pixels, from the last frame that drew it.
    viewport_px: Option<[f32; 2]>,
}

impl Brume {
    /// Opens the window and starts loading `path` in the background.
    pub fn new(cc: &eframe::CreationContext<'_>, path: PathBuf) -> Option<Self> {
        let render_state = cc.wgpu_render_state.as_ref()?;
        let gpu_info = gpu_info(&render_state.adapter.get_info());

        Some(Self {
            state: start_loading(&cc.egui_ctx, path),
            view: ViewSettings::new(gpu_info),
        })
    }

    /// Moves on from [`AppState::Loading`] once the loader thread has sent its result.
    fn poll_loading(&mut self, frame: &eframe::Frame) {
        let AppState::Loading { receiver, path, .. } = &self.state else {
            return;
        };

        let next = match receiver.try_recv() {
            Ok(Ok(scene)) => ready(scene, frame, path),
            Ok(Err(error)) => AppState::Failed {
                path: path.clone(),
                error: format!("{error:#}"),
            },
            Err(TryRecvError::Empty) => return,
            Err(TryRecvError::Disconnected) => AppState::Failed {
                path: path.clone(),
                error: "the loader thread stopped without a result (it panicked)".to_owned(),
            },
        };

        self.state = next;
    }
}

/// Starts reading `path` on a "brume-loader" thread and returns the state that
/// waits for it.
///
/// The header is read here, on the UI thread, to know the total for the progress
/// bar. It is small, and a missing or invalid file fails right away.
fn start_loading(ctx: &egui::Context, path: PathBuf) -> AppState {
    let started = Instant::now();

    let total = match point_count(&path) {
        Ok(total) => total,
        Err(error) => {
            return AppState::Failed {
                path,
                error: format!("{error:#}"),
            };
        }
    };

    let (sender, receiver) = mpsc::channel();
    let progress = Arc::new(AtomicU64::new(0));

    let spawned = thread::Builder::new()
        .name("brume-loader".to_owned())
        .spawn({
            let ctx = ctx.clone();
            let progress = Arc::clone(&progress);
            let path = path.clone();
            move || {
                let result = load_scene(&path, &progress);
                if let Ok(scene) = &result {
                    scene.log();
                }
                // Fails only if the window was closed meanwhile: nobody is left to tell.
                let _ = sender.send(result);
                ctx.request_repaint();
            }
        });

    match spawned {
        Ok(_) => AppState::Loading {
            receiver,
            progress,
            total,
            path,
            started,
        },
        Err(error) => AppState::Failed {
            path,
            error: format!("could not start the loader thread: {error}"),
        },
    }
}

/// Uploads `scene` to the GPU and returns the Ready state, or Failed without wgpu.
fn ready(scene: LoadedScene, frame: &eframe::Frame, path: &Path) -> AppState {
    let Some(render_state) = frame.wgpu_render_state() else {
        return AppState::Failed {
            path: path.to_owned(),
            error: "Brume needs eframe's wgpu backend".to_owned(),
        };
    };

    PointRenderer::register(render_state, &scene.cloud);

    AppState::Ready(Box::new(ReadyState {
        scene,
        camera: OrbitCamera::new(glam::Vec3::ZERO, 1_500.0),
    }))
}

impl ViewSettings {
    fn new(gpu_info: String) -> Self {
        Self {
            point_size: 2.0,
            color_mode: ColorMode::Rgb,
            frame_times: FrameTimes::new(FRAME_WINDOW),
            continuous_repaint: false,
            skip_next_frame: false,
            draw_fraction: 1.0,
            gpu_info,
            viewport_px: None,
        }
    }

    /// Records the time since the previous frame, measured on the CPU. Only
    /// meaningful while repainting continuously: when idle it would include the
    /// time spent waiting for input.
    fn record_frame_time(&mut self, ui: &egui::Ui) {
        if !self.continuous_repaint {
            return;
        }

        if self.skip_next_frame {
            self.skip_next_frame = false;
        } else {
            let dt = ui.input(|input| input.unstable_dt);
            self.frame_times.push(dt * 1000.0);
        }
        ui.ctx().request_repaint();
    }

    fn controls_ui(&mut self, ui: &mut egui::Ui) {
        ui.add(egui::Slider::new(&mut self.point_size, 1.0..=10.0).text("Point size (px)"));

        ui.separator();

        ui.label("Colour");
        ui.radio_value(&mut self.color_mode, ColorMode::Rgb, "RGB");
        ui.radio_value(&mut self.color_mode, ColorMode::Height, "Height");
        ui.radio_value(
            &mut self.color_mode,
            ColorMode::Classification,
            "Classification",
        );
        ui.radio_value(&mut self.color_mode, ColorMode::Intensity, "Intensity");
    }

    fn perf_controls_ui(&mut self, ui: &mut egui::Ui) {
        let toggled = ui
            .checkbox(&mut self.continuous_repaint, "Continuous repaint")
            .changed();
        if toggled && self.continuous_repaint {
            self.frame_times.clear();
            self.skip_next_frame = true;
        }

        let mut percent = self.draw_fraction * 100.0;
        let slider = egui::Slider::new(&mut percent, 1.0..=100.0)
            .integer()
            .text("Points drawn (%)");
        if ui.add(slider).changed() {
            self.draw_fraction = percent / 100.0;
        }
    }

    /// Returns how many points [`PointsCallback`] draws at the current `draw_fraction`.
    fn points_drawn(&self, cloud: &Cloud) -> u64 {
        cloud
            .chunks
            .iter()
            .map(|chunk| u64::from(instances_to_draw(chunk.len() as u32, self.draw_fraction)))
            .sum()
    }

    fn perf_ui(&self, ui: &mut egui::Ui, scene: &LoadedScene) {
        let average = self.frame_times.average();
        let fps = average.filter(|&ms| ms > 0.0).map(|ms| 1000.0 / ms);

        ui.label(format!(
            "Frame time (avg): {}",
            format_or_dash(average, 2, " ms")
        ));
        ui.label(format!(
            "Frame time (worst of {FRAME_WINDOW}): {}",
            format_or_dash(self.frame_times.max(), 2, " ms")
        ));
        ui.label(format!("FPS: {}", format_or_dash(fps, 0, "")));
        ui.label(format!("Points: {}", scene.stats.points));
        ui.label(format!("Points drawn: {}", self.points_drawn(&scene.cloud)));
        match self.viewport_px {
            Some([width, height]) => ui.label(format!("Viewport: {width:.0} × {height:.0} px")),
            None => ui.label("Viewport: –"),
        };
        ui.label(format!("GPU: {}", self.gpu_info));
    }
}

impl ReadyState {
    fn report_ui(&self, ui: &mut egui::Ui) {
        let scene = &self.scene;

        ui.heading("Brume");
        ui.label(format!(
            "Load time: {:.2} s",
            scene.stats.elapsed.as_secs_f64()
        ));
        ui.label(format!("Origin (UTM): {:.2?}", scene.cloud.origin));
        ui.label(format!("Local min (f32): {:.2?}", scene.stats.local_min));
        ui.label(format!("Local max (f32): {:.2?}", scene.stats.local_max));
        ui.label(format!("Color depth: {:?}", scene.stats.color_depth));
        ui.label(format!(
            "Intensity: {} – {}",
            scene.stats.intensity_min, scene.stats.intensity_max
        ));
        ui.label(format!(
            "Height p1–p99: {:.1} – {:.1} m",
            scene.height_range[0], scene.height_range[1]
        ));
        ui.label(format!(
            "Intensity p2–p98: {:.0} – {:.0}",
            scene.intensity_range[0], scene.intensity_range[1]
        ));
        if let Some([r, g, b]) = scene.average_color {
            ui.label(format!("Average color: rgb({r}, {g}, {b})"));
        }

        ui.separator();

        egui::Grid::new("classes").striped(true).show(ui, |ui| {
            for row in &scene.class_rows {
                ui.label(row.code.to_string());
                ui.label(row.name);
                ui.label(row.count.to_string());
                ui.label(format!("{:.1}%", row.percent));
                ui.end_row();
            }
        });
    }

    fn ui(&mut self, ui: &mut egui::Ui, view: &mut ViewSettings) {
        // Side panel: the report, the controls, then performance.
        egui::Panel::left("stats_panel").show(ui, |ui| {
            self.report_ui(ui);

            ui.separator();

            view.controls_ui(ui);

            ui.separator();

            ui.label("Performance");
            view.perf_controls_ui(ui);
            view.perf_ui(ui, &self.scene);
        });

        // Central panel: the 3D view.
        egui::CentralPanel::default().show(ui, |ui| {
            let (rect, response) = ui.allocate_exact_size(ui.available_size(), egui::Sense::drag());

            if response.dragged() {
                let delta = response.drag_delta();
                self.camera.orbit(delta.x, delta.y);
            }
            if response.hovered() {
                let scroll = ui.input(|input| input.smooth_scroll_delta.y);
                self.camera.zoom(scroll);
            }

            if rect.width() > 0.0 && rect.height() > 0.0 {
                let view_proj = self.camera.view_proj(rect.width() / rect.height());

                let pixels_per_point = ui.ctx().pixels_per_point();
                let viewport = [
                    rect.width() * pixels_per_point,
                    rect.height() * pixels_per_point,
                ];
                view.viewport_px = Some(viewport);

                ui.painter().add(egui_wgpu::Callback::new_paint_callback(
                    rect,
                    PointsCallback {
                        view_proj,
                        viewport,
                        point_size: view.point_size,
                        color_mode: view.color_mode,
                        height_range: self.scene.height_range,
                        intensity_range: self.scene.intensity_range,
                        draw_fraction: view.draw_fraction,
                    },
                ));
            }
        });
    }
}

/// Centred progress screen shown while the loader thread runs.
fn loading_ui(ui: &mut egui::Ui, path: &Path, read: u64, total: u64, started: Instant) {
    let fraction = if total > 0 {
        read as f32 / total as f32
    } else {
        0.0
    };
    // The colour ranges are computed after the last batch, with the bar already full.
    let stage = if read < total {
        "Reading points"
    } else {
        "Computing colour ranges"
    };

    egui::CentralPanel::default().show(ui, |ui| {
        ui.vertical_centered(|ui| {
            ui.add_space(ui.available_height() * 0.4);
            ui.heading(format!("Loading {}", file_name(path)));
            ui.add(
                egui::ProgressBar::new(fraction)
                    .desired_width(400.0)
                    .text(format!(
                        "{read} / {total} points ({:.0}%)",
                        fraction * 100.0
                    )),
            );
            ui.label(format!(
                "{stage} · {:.1} s",
                started.elapsed().as_secs_f32()
            ));
        });
    });

    ui.ctx().request_repaint_after(LOADING_REFRESH);
}

/// Centred error screen shown when loading failed.
fn failed_ui(ui: &mut egui::Ui, path: &Path, error: &str) {
    egui::CentralPanel::default().show(ui, |ui| {
        ui.vertical_centered(|ui| {
            ui.add_space(ui.available_height() * 0.4);
            ui.heading("Could not load the file");
            ui.label(path.display().to_string());
            ui.colored_label(ui.visuals().error_fg_color, error);
        });
    });
}

/// Returns the last component of `path`, or the whole path if it has none.
fn file_name(path: &Path) -> String {
    path.file_name().map_or_else(
        || path.display().to_string(),
        |name| name.to_string_lossy().into_owned(),
    )
}

/// Describes the GPU as "name (backend, driver driver_info)", leaving out empty driver fields.
fn gpu_info(info: &egui_wgpu::wgpu::AdapterInfo) -> String {
    let driver = [info.driver.as_str(), info.driver_info.as_str()]
        .into_iter()
        .filter(|part| !part.is_empty())
        .collect::<Vec<_>>()
        .join(" ");

    if driver.is_empty() {
        format!("{} ({:?})", info.name, info.backend)
    } else {
        format!("{} ({:?}, {driver})", info.name, info.backend)
    }
}

/// Formats `value` with `decimals` decimals and a unit, or "–" when there is no data yet.
fn format_or_dash(value: Option<f32>, decimals: usize, unit: &str) -> String {
    match value {
        Some(value) => format!("{value:.decimals$}{unit}"),
        None => "–".to_owned(),
    }
}

impl eframe::App for Brume {
    fn ui(&mut self, ui: &mut egui::Ui, frame: &mut eframe::Frame) {
        self.view.record_frame_time(ui);
        self.poll_loading(frame);

        match &mut self.state {
            AppState::Loading {
                progress,
                total,
                path,
                started,
                ..
            } => loading_ui(ui, path, progress.load(Ordering::Relaxed), *total, *started),
            AppState::Ready(ready) => ready.ui(ui, &mut self.view),
            AppState::Failed { path, error } => failed_ui(ui, path, error),
        }
    }
}
