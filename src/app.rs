use eframe::{egui, egui_wgpu};

use crate::{
    camera::OrbitCamera,
    cloud::{Cloud, Stats},
    perf::FrameTimes,
    render::{ColorMode, PointRenderer, PointsCallback, instances_to_draw},
    report::{ClassRow, average_color, class_rows, height_range, intensity_range},
};

/// Number of recent frames the performance figures are computed over.
const FRAME_WINDOW: usize = 120;

pub struct Brume {
    cloud: Cloud,
    stats: Stats,
    class_rows: Vec<ClassRow>,
    average_color: Option<[u64; 3]>,
    height_range: [f32; 2],
    intensity_range: [f32; 2],
    camera: OrbitCamera,
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
    pub fn new(cc: &eframe::CreationContext<'_>, cloud: Cloud, stats: Stats) -> Option<Self> {
        let render_state = cc.wgpu_render_state.as_ref()?;
        PointRenderer::register(render_state, &cloud);

        let class_rows = class_rows(&stats);
        let average_color = average_color(&cloud);
        let height_range = height_range(&cloud);
        let intensity_range = intensity_range(&cloud);

        let gpu_info = gpu_info(&render_state.adapter.get_info());

        Some(Self {
            cloud,
            stats,
            class_rows,
            average_color,
            height_range,
            intensity_range,
            camera: OrbitCamera::new(glam::Vec3::ZERO, 1_500.0),
            point_size: 2.0,
            color_mode: ColorMode::Rgb,
            frame_times: FrameTimes::new(FRAME_WINDOW),
            continuous_repaint: false,
            skip_next_frame: false,
            draw_fraction: 1.0,
            gpu_info,
            viewport_px: None,
        })
    }

    fn report_ui(&self, ui: &mut egui::Ui) {
        ui.heading("Brume");
        ui.label(format!(
            "Load time: {:.2} s",
            self.stats.elapsed.as_secs_f64()
        ));
        ui.label(format!("Origin (UTM): {:.2?}", self.cloud.origin));
        ui.label(format!("Local min (f32): {:.2?}", self.stats.local_min));
        ui.label(format!("Local max (f32): {:.2?}", self.stats.local_max));
        ui.label(format!("Color depth: {:?}", self.stats.color_depth));
        ui.label(format!(
            "Intensity: {} – {}",
            self.stats.intensity_min, self.stats.intensity_max
        ));
        ui.label(format!(
            "Height p1–p99: {:.1} – {:.1} m",
            self.height_range[0], self.height_range[1]
        ));
        ui.label(format!(
            "Intensity p2–p98: {:.0} – {:.0}",
            self.intensity_range[0], self.intensity_range[1]
        ));
        if let Some([r, g, b]) = self.average_color {
            ui.label(format!("Average color: rgb({r}, {g}, {b})"));
        }

        ui.separator();

        egui::Grid::new("classes").striped(true).show(ui, |ui| {
            for row in &self.class_rows {
                ui.label(row.code.to_string());
                ui.label(row.name);
                ui.label(row.count.to_string());
                ui.label(format!("{:.1}%", row.percent));
                ui.end_row();
            }
        });
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
    fn points_drawn(&self) -> u64 {
        self.cloud
            .chunks
            .iter()
            .map(|chunk| u64::from(instances_to_draw(chunk.len() as u32, self.draw_fraction)))
            .sum()
    }

    fn perf_ui(&self, ui: &mut egui::Ui) {
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
        ui.label(format!("Points: {}", self.stats.points));
        ui.label(format!("Points drawn: {}", self.points_drawn()));
        match self.viewport_px {
            Some([width, height]) => ui.label(format!("Viewport: {width:.0} × {height:.0} px")),
            None => ui.label("Viewport: –"),
        };
        ui.label(format!("GPU: {}", self.gpu_info));
    }
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
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        // Time since the previous frame, measured on the CPU. Only meaningful while
        // repainting continuously: when idle it would include the time spent waiting
        // for input.
        if self.continuous_repaint {
            if self.skip_next_frame {
                self.skip_next_frame = false;
            } else {
                let dt = ui.input(|input| input.unstable_dt);
                self.frame_times.push(dt * 1000.0);
            }
            ui.ctx().request_repaint();
        }

        // Side panel: the report, the controls, then performance.
        egui::Panel::left("stats_panel").show(ui, |ui| {
            self.report_ui(ui);

            ui.separator();

            self.controls_ui(ui);

            ui.separator();

            ui.label("Performance");
            self.perf_controls_ui(ui);
            self.perf_ui(ui);
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
                self.viewport_px = Some(viewport);

                ui.painter().add(egui_wgpu::Callback::new_paint_callback(
                    rect,
                    PointsCallback {
                        view_proj,
                        viewport,
                        point_size: self.point_size,
                        color_mode: self.color_mode,
                        height_range: self.height_range,
                        intensity_range: self.intensity_range,
                        draw_fraction: self.draw_fraction,
                    },
                ));
            }
        });
    }
}
