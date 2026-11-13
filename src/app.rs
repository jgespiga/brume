use std::{
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::Ordering,
        mpsc::{self, Receiver, TryRecvError},
    },
    thread,
    time::{Duration, Instant},
};

use anyhow::{Context, Result, anyhow};
use eframe::{egui, egui_wgpu};

use crate::{
    camera::{FOV_Y, OrbitCamera},
    cloud::Cloud,
    loader::{LoadControl, is_lidar_file, point_count},
    perf::FrameTimes,
    render::{ColorMode, PointRenderer, PointsCallback, instances_to_draw},
    scene::{LoadedScene, load_scene},
};

/// Number of recent frames the performance figures are computed over.
const FRAME_WINDOW: usize = 120;

/// How often the window redraws during a load to update the progress bar and elapsed time.
const LOADING_REFRESH: Duration = Duration::from_millis(50);

pub struct Brume {
    /// The cloud on screen, if any. It stays there while another file loads.
    scene: Option<Scene>,
    /// The file being read in the background, if any.
    pending: Option<PendingLoad>,
    /// The last load error, shown until dismissed or until a load succeeds.
    error: Option<String>,
    /// A short informational message, such as "only the first dropped file was opened".
    note: Option<String>,
    view: ViewSettings,
}

/// A loaded file and the view state that belongs to it.
struct Scene {
    data: LoadedScene,
    camera: OrbitCamera,
    path: PathBuf,
}

/// A file being read by a "brume-loader" thread.
///
/// Dropping it cancels the load: replacing or clearing [`Brume::pending`] stops
/// the old thread at its next batch, and its result is never received.
struct PendingLoad {
    receiver: Receiver<Result<LoadedScene>>,
    control: Arc<LoadControl>,
    /// Points declared in the file's header.
    total: u64,
    path: PathBuf,
    started: Instant,
}

/// View settings and measurements the user chose, independent of the loaded file,
/// so they survive a reload.
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
    /// Opens the window, and starts loading `path` in the background if given.
    pub fn new(cc: &eframe::CreationContext<'_>, path: Option<PathBuf>) -> Option<Self> {
        let render_state = cc.wgpu_render_state.as_ref()?;
        let gpu_info = gpu_info(&render_state.adapter.get_info());

        let mut app = Self {
            scene: None,
            pending: None,
            error: None,
            note: None,
            view: ViewSettings::new(gpu_info),
        };
        if let Some(path) = path {
            app.open(&cc.egui_ctx, path);
        }

        Some(app)
    }

    /// Starts loading `path`, cancelling any load in progress. The current scene
    /// stays on screen meanwhile.
    ///
    /// If the new file can't even be opened, the load in progress (if any) carries on.
    fn open(&mut self, ctx: &egui::Context, path: PathBuf) {
        if !is_lidar_file(&path) {
            self.error = Some(format!("Not a LAS/LAZ file: {}", file_name(&path)));
            return;
        }

        match PendingLoad::start(ctx, path) {
            Ok(pending) => {
                // Dropping the old `PendingLoad` cancels it.
                self.pending = Some(pending);
                self.error = None;
                self.note = None;
            }
            Err(error) => self.error = Some(format!("{error:#}")),
        }
    }

    /// Takes the result of the pending load once its thread has sent it.
    fn poll_pending(&mut self, ctx: &egui::Context, frame: &eframe::Frame) {
        let Some(pending) = &self.pending else {
            return;
        };

        let result = match pending.receiver.try_recv() {
            Ok(result) => result,
            Err(TryRecvError::Empty) => return,
            Err(TryRecvError::Disconnected) => Err(anyhow!(
                "the loader thread stopped without a result (it panicked)"
            )),
        };

        let path = pending.path.clone();
        self.pending = None;

        match result {
            Ok(data) => self.show_scene(ctx, frame, data, path),
            // The old scene, if any, stays on screen.
            Err(error) => self.error = Some(format!("{}: {error:#}", file_name(&path))),
        }
    }

    /// Uploads `data` to the GPU and puts it on screen with a fitted camera.
    fn show_scene(
        &mut self,
        ctx: &egui::Context,
        frame: &eframe::Frame,
        data: LoadedScene,
        path: PathBuf,
    ) {
        let Some(render_state) = frame.wgpu_render_state() else {
            self.error = Some("Brume needs eframe's wgpu backend".to_owned());
            return;
        };

        // Replaces the previous scene's buffers in egui's callback resources.
        PointRenderer::register(render_state, &data.cloud);
        ctx.send_viewport_cmd(egui::ViewportCommand::Title(format!(
            "Brume — {}",
            file_name(&path)
        )));

        self.scene = Some(Scene::new(data, path));
        self.error = None;
    }

    /// Opens the first dropped file that has a path. Opening several tiles at once
    /// comes in a later step.
    fn handle_dropped_files(&mut self, ctx: &egui::Context) {
        let dropped = ctx.input(|input| input.raw.dropped_files.clone());
        if dropped.is_empty() {
            return;
        }

        let Some(path) = dropped
            .iter()
            .map(|file| file.path())
            .find(|path| !path.as_os_str().is_empty())
            .map(Path::to_owned)
        else {
            self.error = Some("The dropped item is not a file on disk".to_owned());
            return;
        };

        self.open(ctx, path);
        // `open` leaves `error` empty only if the load started.
        if dropped.len() > 1 && self.error.is_none() {
            self.note = Some(format!(
                "Opened the first of {} files: multiple tiles come later.",
                dropped.len()
            ));
        }
    }
}

impl PendingLoad {
    /// Starts reading `path` on a "brume-loader" thread.
    ///
    /// The header is read here, on the UI thread, to know the total for the
    /// progress bar. It is small, and a missing or invalid file fails right away.
    fn start(ctx: &egui::Context, path: PathBuf) -> Result<Self> {
        let started = Instant::now();
        let total = point_count(&path)?;

        let (sender, receiver) = mpsc::channel();
        let control = Arc::new(LoadControl::default());

        thread::Builder::new()
            .name("brume-loader".to_owned())
            .spawn({
                let ctx = ctx.clone();
                let control = Arc::clone(&control);
                let path = path.clone();
                move || {
                    let result = load_scene(&path, &control);
                    if let Ok(scene) = &result {
                        scene.log();
                    }
                    // Fails only if the load was cancelled or the window closed:
                    // nobody is waiting for the result any more.
                    let _ = sender.send(result);
                    ctx.request_repaint();
                }
            })
            .context("could not start the loader thread")?;

        Ok(Self {
            receiver,
            control,
            total,
            path,
            started,
        })
    }

    /// Points read so far.
    fn read(&self) -> u64 {
        self.control.progress.load(Ordering::Relaxed)
    }
}

impl Drop for PendingLoad {
    fn drop(&mut self) {
        self.control.cancel.store(true, Ordering::Relaxed);
    }
}

impl Scene {
    fn new(data: LoadedScene, path: PathBuf) -> Self {
        let camera = OrbitCamera::fit(data.stats.local_min, data.stats.local_max, FOV_Y);

        Self { data, camera, path }
    }

    fn report_ui(&self, ui: &mut egui::Ui) {
        let data = &self.data;

        ui.heading(file_name(&self.path));
        ui.label(format!(
            "Load time: {:.2} s",
            data.stats.elapsed.as_secs_f64()
        ));
        ui.label(format!("Origin (UTM): {:.2?}", data.cloud.origin));
        ui.label(format!("Local min (f32): {:.2?}", data.stats.local_min));
        ui.label(format!("Local max (f32): {:.2?}", data.stats.local_max));
        ui.label(format!("Color depth: {:?}", data.stats.color_depth));
        ui.label(format!(
            "Intensity: {} – {}",
            data.stats.intensity_min, data.stats.intensity_max
        ));
        ui.label(format!(
            "Height p1–p99: {:.1} – {:.1} m",
            data.height_range[0], data.height_range[1]
        ));
        ui.label(format!(
            "Intensity p2–p98: {:.0} – {:.0}",
            data.intensity_range[0], data.intensity_range[1]
        ));
        if let Some([r, g, b]) = data.average_color {
            ui.label(format!("Average color: rgb({r}, {g}, {b})"));
        }

        ui.separator();

        egui::Grid::new("classes").striped(true).show(ui, |ui| {
            for row in &data.class_rows {
                ui.label(row.code.to_string());
                ui.label(row.name);
                ui.label(row.count.to_string());
                ui.label(format!("{:.1}%", row.percent));
                ui.end_row();
            }
        });
    }

    /// The 3D view in the central panel. Returns the panel's rect.
    fn view_ui(&mut self, ui: &mut egui::Ui, view: &mut ViewSettings) -> egui::Rect {
        let panel = egui::CentralPanel::default().show(ui, |ui| {
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
                        height_range: self.data.height_range,
                        intensity_range: self.data.intensity_range,
                        draw_fraction: view.draw_fraction,
                    },
                ));
            }
        });

        panel.response.rect
    }
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

/// Compact load progress, messages and the "Open…" button, for the section pinned
/// to the bottom of the side panel. Returns whether "Open…" was clicked.
fn status_ui(
    ui: &mut egui::Ui,
    pending: &mut Option<PendingLoad>,
    error: &mut Option<String>,
    note: &mut Option<String>,
) -> bool {
    let cancelled = pending.as_ref().is_some_and(|load| {
        ui.label(format!("Loading {}", file_name(&load.path)));
        ui.horizontal(|ui| {
            let cancel = ui.button("Cancel").clicked();
            ui.add(egui::ProgressBar::new(fraction(load.read(), load.total)).show_percentage());
            cancel
        })
        .inner
    });
    if cancelled {
        // Dropping the `PendingLoad` cancels it.
        *pending = None;
    }

    messages_ui(ui, error, note);

    ui.button("Open…").clicked()
}

/// Centred screen shown while there is no scene: load progress or a hint, then
/// messages and "Open…". Returns whether "Open…" was clicked, and the panel's rect.
fn empty_ui(
    ui: &mut egui::Ui,
    pending: &mut Option<PendingLoad>,
    error: &mut Option<String>,
    note: &mut Option<String>,
) -> (bool, egui::Rect) {
    let panel = egui::CentralPanel::default().show(ui, |ui| {
        ui.vertical_centered(|ui| {
            ui.add_space(ui.available_height() * 0.35);

            let cancelled = match pending.as_ref() {
                Some(load) => loading_ui(ui, load),
                None => {
                    ui.heading("Drop a LAS/LAZ file here or use Open…");
                    false
                }
            };
            if cancelled {
                *pending = None;
            }

            ui.add_space(8.0);
            messages_ui(ui, error, note);

            ui.button("Open…").clicked()
        })
        .inner
    });

    (panel.inner, panel.response.rect)
}

/// Full progress block for the empty screen. Returns whether "Cancel" was clicked.
fn loading_ui(ui: &mut egui::Ui, load: &PendingLoad) -> bool {
    let read = load.read();
    let fraction = fraction(read, load.total);
    // The colour ranges are computed after the last batch, with the bar already full.
    let stage = if read < load.total {
        "Reading points"
    } else {
        "Computing colour ranges"
    };

    ui.heading(format!("Loading {}", file_name(&load.path)));
    ui.add(
        egui::ProgressBar::new(fraction)
            .desired_width(400.0)
            .text(format!(
                "{read} / {} points ({:.0}%)",
                load.total,
                fraction * 100.0
            )),
    );
    ui.label(format!(
        "{stage} · {:.1} s",
        load.started.elapsed().as_secs_f32()
    ));

    ui.button("Cancel").clicked()
}

/// Shows the error in red and the note in weak text, each with a "Dismiss" button.
fn messages_ui(ui: &mut egui::Ui, error: &mut Option<String>, note: &mut Option<String>) {
    let error_color = ui.visuals().error_fg_color;
    let note_color = ui.visuals().weak_text_color();

    message_ui(ui, error, error_color);
    message_ui(ui, note, note_color);
}

/// Shows `message` in `color` with a "Dismiss" button that clears it.
fn message_ui(ui: &mut egui::Ui, message: &mut Option<String>, color: egui::Color32) {
    let Some(text) = message else {
        return;
    };

    ui.colored_label(color, text.as_str());
    if ui.small_button("Dismiss").clicked() {
        *message = None;
    }
}

/// Darkens `rect` with a hint while files are dragged over the window.
fn drop_overlay(ctx: &egui::Context, rect: egui::Rect) {
    if ctx.input(|input| input.raw.hovered_files.is_empty()) {
        return;
    }

    let layer = egui::LayerId::new(egui::Order::Foreground, egui::Id::new("drop_overlay"));
    let painter = ctx.layer_painter(layer);
    painter.rect_filled(rect, 0.0, egui::Color32::from_black_alpha(180));
    painter.text(
        rect.center(),
        egui::Align2::CENTER_CENTER,
        "Drop a LAS/LAZ file to open",
        egui::FontId::proportional(24.0),
        egui::Color32::WHITE,
    );
}

/// Asks for a file with the system dialog. Blocks the UI until the dialog closes.
fn pick_file() -> Option<PathBuf> {
    // Both cases: some dialogs match the filter case-sensitively, and PNOA tiles end in `.LAZ`.
    rfd::FileDialog::new()
        .add_filter("LiDAR", &["las", "laz", "LAS", "LAZ"])
        .pick_file()
}

/// Fraction of the points read, for a progress bar.
fn fraction(read: u64, total: u64) -> f32 {
    if total > 0 {
        read as f32 / total as f32
    } else {
        0.0
    }
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
        let ctx = ui.ctx().clone();

        self.view.record_frame_time(ui);
        self.poll_pending(&ctx, frame);
        self.handle_dropped_files(&ctx);
        if self.pending.is_some() {
            ctx.request_repaint_after(LOADING_REFRESH);
        }

        let (open_clicked, view_rect) = match &mut self.scene {
            Some(scene) => {
                // Side panel: the report, the controls and performance, scrolling
                // above a status section pinned to the bottom (load progress,
                // messages and "Open…"), so that section is always visible.
                let open = egui::Panel::left("stats_panel")
                    .show(ui, |ui| {
                        // Laid out first so it claims its space at the bottom; the
                        // scroll area then fills what is left.
                        let open = egui::Panel::bottom("load_status")
                            .resizable(false)
                            .show(ui, |ui| {
                                ui.add_space(4.0);
                                status_ui(ui, &mut self.pending, &mut self.error, &mut self.note)
                            })
                            .inner;

                        egui::ScrollArea::vertical().show(ui, |ui| {
                            scene.report_ui(ui);

                            ui.separator();

                            self.view.controls_ui(ui);

                            ui.separator();

                            ui.label("Performance");
                            self.view.perf_controls_ui(ui);
                            self.view.perf_ui(ui, &scene.data);
                        });

                        open
                    })
                    .inner;

                (open, scene.view_ui(ui, &mut self.view))
            }
            None => empty_ui(ui, &mut self.pending, &mut self.error, &mut self.note),
        };

        drop_overlay(&ctx, view_rect);

        if open_clicked && let Some(path) = pick_file() {
            self.open(&ctx, path);
        }
    }
}
