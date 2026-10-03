use eframe::egui;

use crate::{
    class_name,
    cloud::{Cloud, Stats},
};

pub struct Brume {
    cloud: Cloud,
    stats: Stats,
}

impl Brume {
    pub fn new(cloud: Cloud, stats: Stats) -> Self {
        Self { cloud, stats }
    }
}

impl eframe::App for Brume {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        egui::Panel::left("stats_panel").show(ui, |ui| {
            ui.heading("Brume");
            ui.label(format!("Time: {}", self.stats.elapsed.as_secs_f64()));
            ui.label(format!("Origin (UTM): {:.2?}", self.cloud.origin));
            ui.label(format!("Local min (f32): {:.2?}", self.stats.local_min));
            ui.label(format!("Local max (f32): {:.2?}", self.stats.local_max));
            ui.label(format!("Color depth: {:?}", self.stats.color_depth));
            ui.label(format!(
                "Intensity: {} - {}\n",
                self.stats.intensity_min, self.stats.intensity_max
            ));
            ui.end_row();

            let mut present: Vec<(u8, u64)> = self
                .stats
                .class_counts
                .iter()
                .enumerate()
                .filter(|&(_, &count)| count > 0)
                .map(|(code, &count)| (code as u8, count))
                .collect();
            present.sort_by_key(|&(_, count)| std::cmp::Reverse(count));

            egui::Grid::new("classes").striped(true).show(ui, |ui| {
                for (code, count) in &present {
                    ui.label(code.to_string());
                    // TODO: refactor the printing logic
                    ui.label(class_name(*code));
                    ui.label(count.to_string());
                    ui.end_row();
                }
            });
        });

        egui::CentralPanel::default().show(ui, |ui| {
            ui.label("The 3D view will go here");
        });
    }
}
