use eframe::{egui, egui_wgpu};

use crate::{
    camera::OrbitCamera,
    cloud::{Cloud, Stats},
    render::{PointRenderer, PointsCallback},
    report::{ClassRow, average_color, class_rows},
};

pub struct Brume {
    cloud: Cloud,
    stats: Stats,
    class_rows: Vec<ClassRow>,
    average_color: Option<[u64; 3]>,
    camera: OrbitCamera,
}

impl Brume {
    pub fn new(cc: &eframe::CreationContext<'_>, cloud: Cloud, stats: Stats) -> Option<Self> {
        let render_state = cc.wgpu_render_state.as_ref()?;
        PointRenderer::register(render_state, &cloud);

        let class_rows = class_rows(&stats);
        let average_color = average_color(&cloud);

        Some(Self {
            cloud,
            stats,
            class_rows,
            average_color,
            camera: OrbitCamera::new(glam::Vec3::ZERO, 1_500.0),
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
        egui::Panel::left("stats_panel").show(ui, |ui| self.report_ui(ui));

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

            let view_proj = self.camera.view_proj(rect.width() / rect.height());
            ui.painter().add(egui_wgpu::Callback::new_paint_callback(
                rect,
                PointsCallback { view_proj },
            ));
        });
    }
}
