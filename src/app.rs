use eframe::{egui, egui_wgpu};

use crate::{
    camera::OrbitCamera,
    cloud::{Cloud, Stats},
    render::{ColorMode, PointRenderer, PointsCallback},
    report::{ClassRow, average_color, class_rows, height_range, intensity_range},
};

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
}

impl Brume {
    pub fn new(cc: &eframe::CreationContext<'_>, cloud: Cloud, stats: Stats) -> Option<Self> {
        let render_state = cc.wgpu_render_state.as_ref()?;
        PointRenderer::register(render_state, &cloud);

        let class_rows = class_rows(&stats);
        let average_color = average_color(&cloud);
        let height_range = height_range(&cloud);
        let intensity_range = intensity_range(&cloud);

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
}

impl eframe::App for Brume {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        // Side panel: the report, then the controls.
        egui::Panel::left("stats_panel").show(ui, |ui| {
            self.report_ui(ui);

            ui.separator();

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

                ui.painter().add(egui_wgpu::Callback::new_paint_callback(
                    rect,
                    PointsCallback {
                        view_proj,
                        viewport,
                        point_size: self.point_size,
                        color_mode: self.color_mode,
                        height_range: self.height_range,
                        intensity_range: self.intensity_range,
                    },
                ));
            }
        });
    }
}
