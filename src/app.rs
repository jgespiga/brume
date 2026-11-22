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
use glam::{Mat4, Vec2, Vec3};

use crate::{
    camera::{FOV_Y, OrbitCamera, project_to_screen},
    loader::{LoadControl, is_lidar_file},
    octree::{Node, OctreeStats},
    perf::FrameTimes,
    render::{ColorMode, PointRenderer, PointsCallback},
    report::depth_color,
    scene::{LoadPlan, LoadedScene, load_scene},
};

/// Number of recent frames the performance figures are computed over.
const FRAME_WINDOW: usize = 120;

/// Most octree node boxes drawn per frame; past it the finest levels are left out.
const MAX_NODE_BOXES: usize = 5_000;

/// The 12 edges of a cube, as pairs of corner indices. Corner bits: 1 = +x, 2 = +y,
/// 4 = +z, as for octants.
const CUBE_EDGES: [(usize, usize); 12] = [
    (0, 1),
    (2, 3),
    (4, 5),
    (6, 7),
    (0, 2),
    (1, 3),
    (4, 6),
    (5, 7),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
];

/// How often the window redraws during a load to update the progress bar and elapsed time.
const LOADING_REFRESH: Duration = Duration::from_millis(50);

pub struct Brume {
    /// The cloud on screen, if any. It stays there while another file loads.
    scene: Option<Scene>,
    /// The files being read in the background, if any.
    pending: Option<PendingLoad>,
    /// The last load error, shown until dismissed or until a load succeeds.
    error: Option<String>,
    /// A short informational message, such as "skipped files that aren't LAS/LAZ".
    note: Option<String>,
    /// The device's largest buffer, in bytes: a tile's points must fit in one.
    max_buffer_size: u64,
    view: ViewSettings,
}

/// A loaded scene and the view state that belongs to it.
struct Scene {
    data: LoadedScene,
    camera: OrbitCamera,
    /// Whether each tile is drawn, in the same order as `data.tiles`.
    visible: Vec<bool>,
}

/// Files being read, one after another, by a "brume-loader" thread.
///
/// Dropping it cancels the load: replacing or clearing [`Brume::pending`] stops
/// the old thread at its next batch, and its result is never received.
struct PendingLoad {
    receiver: Receiver<Result<LoadedScene>>,
    control: Arc<LoadControl>,
    /// Points declared in the headers of all the files.
    total: u64,
    /// Files to read, in order.
    paths: Vec<PathBuf>,
    /// Number of tiles in the scene this load adds to, or `None` for a new scene.
    adds_to: Option<usize>,
    started: Instant,
}

/// What a file button asked for.
#[derive(Clone, Copy)]
enum FileAction {
    /// Pick files for a new scene.
    Open,
    /// Pick files to add to the current scene.
    Add,
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
    /// Deepest octree level drawn, or `None` for every level, however deep the
    /// scene is.
    max_depth: Option<u8>,
    /// Whether to draw the wireframe box of each octree node over the points.
    show_nodes: bool,
    /// Node boxes drawn in the last frame, and whether [`MAX_NODE_BOXES`] cut them.
    nodes_drawn: (usize, bool),
    gpu_info: String,
    /// Size of the 3D view in physical pixels, from the last frame that drew it.
    viewport_px: Option<[f32; 2]>,
}

impl Brume {
    /// Opens the window, and starts loading `paths` in the background as one scene
    /// if there are any.
    pub fn new(cc: &eframe::CreationContext<'_>, paths: Vec<PathBuf>) -> Option<Self> {
        let render_state = cc.wgpu_render_state.as_ref()?;
        let gpu_info = gpu_info(&render_state.adapter.get_info());
        let max_buffer_size = render_state.device.limits().max_buffer_size;
        println!(
            "GPU: {gpu_info}, max buffer size {} MiB",
            max_buffer_size >> 20
        );

        let mut app = Self {
            scene: None,
            pending: None,
            error: None,
            note: None,
            max_buffer_size,
            view: ViewSettings::new(gpu_info),
        };
        if !paths.is_empty() {
            app.open(&cc.egui_ctx, paths);
        }

        Some(app)
    }

    /// Starts loading `paths` as a new scene, cancelling any load in progress. The
    /// current scene stays on screen meanwhile.
    ///
    /// If the new files can't even be opened, or aren't neighbours, the load in
    /// progress (if any) carries on.
    fn open(&mut self, ctx: &egui::Context, paths: Vec<PathBuf>) {
        let Some(paths) = self.lidar_files(paths) else {
            return;
        };

        match LoadPlan::new_scene(paths, self.max_buffer_size)
            .and_then(|plan| PendingLoad::start(ctx, plan))
        {
            Ok(pending) => {
                // Dropping the old `PendingLoad` cancels it.
                self.pending = Some(pending);
                self.error = None;
            }
            Err(error) => self.error = Some(format!("{error:#}")),
        }
    }

    /// Starts loading `paths` as tiles added to the current scene, which keeps its
    /// origin and camera. Files already in the scene are skipped with a note.
    /// Without a scene, opens them as a new one.
    fn add(&mut self, ctx: &egui::Context, paths: Vec<PathBuf>) {
        if self.scene.is_none() {
            self.open(ctx, paths);
            return;
        }
        let Some(paths) = self.lidar_files(paths) else {
            return;
        };
        let Some(scene) = &self.scene else {
            return;
        };

        let (loaded, fresh): (Vec<_>, Vec<_>) = paths
            .into_iter()
            .partition(|path| scene.data.contains(path));
        let plan =
            (!fresh.is_empty()).then(|| LoadPlan::add_to(&scene.data, fresh, self.max_buffer_size));

        if !loaded.is_empty() {
            self.add_note(format!("Already loaded: {}.", file_names(&loaded)));
        }
        let Some(plan) = plan else {
            return;
        };

        match plan.and_then(|plan| PendingLoad::start(ctx, plan)) {
            Ok(pending) => {
                // Dropping the old `PendingLoad` cancels it.
                self.pending = Some(pending);
                self.error = None;
            }
            Err(error) => self.error = Some(format!("{error:#}")),
        }
    }

    /// Adds `text` to the note, after what it already says.
    fn add_note(&mut self, text: String) {
        self.note = Some(match self.note.take() {
            Some(note) => format!("{note} {text}"),
            None => text,
        });
    }

    /// Keeps the LAS/LAZ files of `paths`, made canonical and without duplicates, and
    /// notes the others. Sets an error and returns `None` if none is left.
    fn lidar_files(&mut self, paths: Vec<PathBuf>) -> Option<Vec<PathBuf>> {
        let (lidar, other): (Vec<_>, Vec<_>) =
            paths.into_iter().partition(|path| is_lidar_file(path));

        if lidar.is_empty() {
            self.error = Some(format!("Not a LAS/LAZ file: {}", file_names(&other)));
            return None;
        }
        self.note =
            (!other.is_empty()).then(|| format!("Skipped, not LAS/LAZ: {}.", file_names(&other)));

        let mut unique = Vec::with_capacity(lidar.len());
        for path in lidar {
            // Canonical paths make the same file dropped twice, or through a
            // symlink, count once.
            let path = std::fs::canonicalize(&path).unwrap_or(path);
            if !unique.contains(&path) {
                unique.push(path);
            }
        }

        Some(unique)
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

        let adds_to = pending.adds_to;
        self.pending = None;

        match (result, adds_to) {
            (Ok(data), Some(base)) => self.extend_scene(ctx, frame, data, base),
            (Ok(data), None) => self.show_scene(ctx, frame, data),
            // The old scene, if any, stays on screen.
            (Err(error), _) => self.error = Some(format!("{error:#}")),
        }
    }

    /// Uploads `data` to the GPU and puts it on screen with a fitted camera.
    fn show_scene(&mut self, ctx: &egui::Context, frame: &eframe::Frame, data: LoadedScene) {
        let Some(render_state) = frame.wgpu_render_state() else {
            self.error = Some("Brume needs eframe's wgpu backend".to_owned());
            return;
        };

        // Replaces the previous scene's buffers in egui's callback resources.
        if let Err(error) = PointRenderer::register(render_state, &data.tiles) {
            self.error = Some(format!("{error:#}"));
            return;
        }
        set_title(ctx, &data);

        self.scene = Some(Scene::new(data));
        self.error = None;
    }

    /// Puts `data`, the current scene plus new tiles after its first `base` ones, on
    /// screen. Only the new tiles are uploaded, and the camera stays where it is.
    fn extend_scene(
        &mut self,
        ctx: &egui::Context,
        frame: &eframe::Frame,
        data: LoadedScene,
        base: usize,
    ) {
        let Some(render_state) = frame.wgpu_render_state() else {
            self.error = Some("Brume needs eframe's wgpu backend".to_owned());
            return;
        };
        let Some(scene) = &mut self.scene else {
            self.show_scene(ctx, frame, data);
            return;
        };

        // The renderer holds the current scene's tiles. `data` starts with those same
        // tiles (shared `Arc`s) unless the scene changed meanwhile; only then, or if
        // there is no renderer, is everything uploaded again.
        let same_start = scene.data.tiles.len() == base
            && scene
                .data
                .tiles
                .iter()
                .zip(&data.tiles)
                .all(|(current, loaded)| Arc::ptr_eq(current, loaded));
        let appended = match data.tiles.get(base..) {
            Some(new_tiles) if same_start => PointRenderer::append(render_state, new_tiles),
            _ => Ok(false),
        };
        let uploaded = match appended {
            Ok(true) => Ok(()),
            Ok(false) => PointRenderer::register(render_state, &data.tiles),
            Err(error) => Err(error),
        };
        if let Err(error) = uploaded {
            // The scene on screen, and its buffers, stay as they were.
            self.error = Some(format!("{error:#}"));
            return;
        }
        set_title(ctx, &data);

        scene.data = data;
        // New tiles start visible; the others keep their setting.
        scene.visible.resize(scene.data.tiles.len(), true);
        self.error = None;
    }

    /// Opens the dropped files as a new scene, or adds them to the current one if
    /// Shift is held.
    fn handle_dropped_files(&mut self, ctx: &egui::Context) {
        let (dropped, shift) =
            ctx.input(|input| (input.raw.dropped_files.clone(), input.modifiers.shift));
        if dropped.is_empty() {
            return;
        }

        let paths: Vec<PathBuf> = dropped
            .iter()
            .map(|file| file.path())
            .filter(|path| !path.as_os_str().is_empty())
            .map(Path::to_owned)
            .collect();
        if paths.is_empty() {
            self.error = Some("The dropped items are not files on disk".to_owned());
            return;
        }

        if shift {
            self.add(ctx, paths);
        } else {
            self.open(ctx, paths);
        }
    }
}

impl PendingLoad {
    /// Starts reading the files of `plan` on a "brume-loader" thread.
    ///
    /// The plan was made on the UI thread from the headers, which are small: a
    /// missing or invalid file fails before any thread starts.
    fn start(ctx: &egui::Context, plan: LoadPlan) -> Result<Self> {
        let started = Instant::now();
        let total = plan.total;
        let paths = plan.paths.clone();
        let adds_to = (!plan.existing.is_empty()).then_some(plan.existing.len());

        let (sender, receiver) = mpsc::channel();
        let control = Arc::new(LoadControl::default());

        thread::Builder::new()
            .name("brume-loader".to_owned())
            .spawn({
                let ctx = ctx.clone();
                let control = Arc::clone(&control);
                move || {
                    let result = load_scene(&plan, &control);
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
            paths,
            adds_to,
            started,
        })
    }

    /// Points read so far, across all the files.
    fn read(&self) -> u64 {
        self.control.progress.load(Ordering::Relaxed)
    }

    /// The file being read: "name" when there is one, "tile i of N: name" otherwise.
    fn label(&self) -> String {
        let count = self.paths.len();
        let index = self
            .control
            .tile
            .load(Ordering::Relaxed)
            .min(count.saturating_sub(1));
        let name = self
            .paths
            .get(index)
            .map_or_else(String::new, |path| file_name(path));

        if count == 1 {
            name
        } else {
            format!("tile {} of {count}: {name}", index + 1)
        }
    }

    /// What the loader thread is doing now.
    fn stage(&self) -> &'static str {
        if self.control.building_octree.load(Ordering::Relaxed) {
            "Building octree"
        } else if self.read() < self.total {
            "Reading points"
        } else {
            // The colour ranges are computed after the last tile, with the bar full.
            "Computing colour ranges"
        }
    }
}

impl Drop for PendingLoad {
    fn drop(&mut self) {
        self.control.cancel.store(true, Ordering::Relaxed);
    }
}

impl Scene {
    /// Puts a new scene on screen, with the camera fitted to the union of its tiles.
    fn new(data: LoadedScene) -> Self {
        let camera = OrbitCamera::fit(data.stats.local_min, data.stats.local_max, FOV_Y);

        let visible = vec![true; data.tiles.len()];

        Self {
            data,
            camera,
            visible,
        }
    }

    /// The deepest octree level among the scene's tiles.
    fn deepest(&self) -> u8 {
        let levels = self
            .data
            .tiles
            .iter()
            .map(|tile| tile.points_up_to_depth.len());
        levels.max().unwrap_or(1).saturating_sub(1) as u8
    }

    /// How many points to draw from each tile, in tile order: every level down to
    /// `max_depth` (all of them for `None`), or 0 for hidden tiles.
    fn draw_counts(&self, max_depth: Option<u8>) -> Vec<u32> {
        self.data
            .tiles
            .iter()
            .zip(&self.visible)
            .map(|(tile, &visible)| {
                let counts = &tile.points_up_to_depth;
                let up_to_depth = max_depth
                    .and_then(|depth| counts.get(usize::from(depth)))
                    .or(counts.last());
                if visible {
                    up_to_depth.copied().unwrap_or(0)
                } else {
                    0
                }
            })
            .collect()
    }

    /// One row per tile (name, points, load time, octree build time, nodes and
    /// depth), with a checkbox that shows or hides it, then each tile's points per
    /// octree level.
    fn tiles_ui(&mut self, ui: &mut egui::Ui) {
        let octrees: Vec<OctreeStats> = self
            .data
            .tiles
            .iter()
            .map(|tile| tile.octree.stats())
            .collect();

        ui.label("Tiles");
        egui::Grid::new("tiles").striped(true).show(ui, |ui| {
            for heading in ["", "Points", "Load", "Octree", "Nodes", "Depth"] {
                ui.weak(heading);
            }
            ui.end_row();

            let rows = self.data.tiles.iter().zip(&mut self.visible).zip(&octrees);
            for ((tile, visible), octree) in rows {
                ui.checkbox(visible, file_name(&tile.path));
                ui.label(tile.stats.points.to_string());
                ui.label(format!("{:.2} s", tile.stats.elapsed.as_secs_f64()));
                ui.label(format!("{:.0} ms", tile.octree_time.as_secs_f64() * 1000.0));
                ui.label(octree.nodes.to_string());
                ui.label(octree.max_depth.to_string());
                ui.end_row();
            }
        });

        let single = self.data.tiles.len() == 1;
        for (index, (tile, octree)) in self.data.tiles.iter().zip(&octrees).enumerate() {
            let title = if single {
                "Points per level".to_owned()
            } else {
                format!("Points per level: {}", file_name(&tile.path))
            };
            egui::CollapsingHeader::new(title)
                .id_salt(("points_per_level", index))
                .show(ui, |ui| levels_ui(ui, index, octree));
        }
    }

    fn report_ui(&self, ui: &mut egui::Ui) {
        let data = &self.data;

        ui.heading(scene_name(data));
        ui.label(format!(
            "Load time: {:.2} s",
            data.stats.elapsed.as_secs_f64()
        ));
        ui.label(format!("Origin (UTM): {:.2?}", data.origin));
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
    }

    /// One row per class in the scene, most common first.
    fn classes_ui(&self, ui: &mut egui::Ui) {
        egui::Grid::new("classes").striped(true).show(ui, |ui| {
            for row in &self.data.class_rows {
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
                        draw_counts: self.draw_counts(view.max_depth),
                    },
                ));

                // Painted after the callback, so the boxes are on top of the points.
                if view.show_nodes {
                    view.nodes_drawn =
                        self.node_boxes_ui(ui.painter(), view_proj, rect, view.max_depth);
                }
            }
        });

        panel.response.rect
    }

    /// Draws the wireframe cube of every node down to `max_depth` of every visible
    /// tile, coloured by depth, over the 3D view in `rect`. Coarse levels go first,
    /// so when [`MAX_NODE_BOXES`] is reached the finest ones are left out. Returns
    /// the boxes drawn and whether the cap was reached.
    fn node_boxes_ui(
        &self,
        painter: &egui::Painter,
        view_proj: Mat4,
        rect: egui::Rect,
        max_depth: Option<u8>,
    ) -> (usize, bool) {
        let mut nodes: Vec<&Node> = self
            .data
            .tiles
            .iter()
            .zip(&self.visible)
            .filter(|&(_, &visible)| visible)
            .flat_map(|(tile, _)| &tile.octree.nodes)
            .filter(|node| max_depth.is_none_or(|depth| node.depth <= depth))
            .collect();
        // Stable, so each tile's nodes keep their breadth-first order within a level.
        nodes.sort_by_key(|node| node.depth);

        let capped = nodes.len() > MAX_NODE_BOXES;
        for node in nodes.iter().take(MAX_NODE_BOXES) {
            cube_ui(painter, view_proj, rect, node);
        }

        (nodes.len().min(MAX_NODE_BOXES), capped)
    }
}

/// One row per octree depth from 0 to `deepest`: a square in the depth's colour
/// and "Depth N".
fn depth_legend_ui(ui: &mut egui::Ui, deepest: u8) {
    for depth in 0..=deepest {
        ui.horizontal(|ui| {
            let [r, g, b] = depth_color(depth);
            let (square, _) = ui.allocate_exact_size(egui::vec2(12.0, 12.0), egui::Sense::hover());
            ui.painter()
                .rect_filled(square, 2.0, egui::Color32::from_rgb(r, g, b));
            ui.label(format!("Depth {depth}"));
        });
    }
}

/// Draws the edges of `node`'s cube, projected with `view_proj` into `rect`, in the
/// colour of its depth. Edges with an end behind the camera are left out.
fn cube_ui(painter: &egui::Painter, view_proj: Mat4, rect: egui::Rect, node: &Node) {
    let [r, g, b] = depth_color(node.depth);
    let stroke = egui::Stroke::new(1.0, egui::Color32::from_rgb(r, g, b));
    let (rect_min, rect_size) = (
        Vec2::new(rect.min.x, rect.min.y),
        Vec2::new(rect.width(), rect.height()),
    );

    let corners: [Option<egui::Pos2>; 8] = std::array::from_fn(|corner| {
        let offset = Vec3::new(
            (corner & 1) as f32,
            ((corner >> 1) & 1) as f32,
            ((corner >> 2) & 1) as f32,
        );
        let point = node.bounds_min + offset * node.size;
        project_to_screen(view_proj, point, rect_min, rect_size)
            .map(|screen| egui::pos2(screen.x, screen.y))
    });

    for (start, end) in CUBE_EDGES {
        if let (Some(start), Some(end)) = (corners[start], corners[end]) {
            painter.line_segment([start, end], stroke);
        }
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
            max_depth: None,
            show_nodes: false,
            nodes_drawn: (0, false),
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

    /// Point size and colour mode. With "Octree depth", also a legend for depths
    /// 0 to `deepest`.
    fn controls_ui(&mut self, ui: &mut egui::Ui, deepest: u8) {
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
        ui.radio_value(&mut self.color_mode, ColorMode::OctreeDepth, "Octree depth");

        if self.color_mode == ColorMode::OctreeDepth {
            depth_legend_ui(ui, deepest);
        }
    }

    /// Continuous repaint, the max octree depth slider (0 to `deepest`) and the
    /// node boxes checkbox.
    fn perf_controls_ui(&mut self, ui: &mut egui::Ui, deepest: u8) {
        let toggled = ui
            .checkbox(&mut self.continuous_repaint, "Continuous repaint")
            .changed();
        if toggled && self.continuous_repaint {
            self.frame_times.clear();
            self.skip_next_frame = true;
        }

        // At the scene's deepest level the slider means "everything", so it keeps
        // drawing every level after loading a deeper scene.
        let mut depth = self.max_depth.map_or(deepest, |depth| depth.min(deepest));
        let slider = egui::Slider::new(&mut depth, 0..=deepest).text("Max octree depth");
        if ui.add(slider).changed() {
            self.max_depth = (depth < deepest).then_some(depth);
        }

        ui.checkbox(&mut self.show_nodes, "Show octree nodes");
    }

    /// Returns how many points [`PointsCallback`] draws from the visible tiles of
    /// `scene` at the current `max_depth`.
    fn points_drawn(&self, scene: &Scene) -> u64 {
        scene
            .draw_counts(self.max_depth)
            .into_iter()
            .map(u64::from)
            .sum()
    }

    fn perf_ui(&self, ui: &mut egui::Ui, scene: &Scene) {
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
        ui.label(format!("Points: {}", scene.data.stats.points));
        ui.label(format!("Points drawn: {}", self.points_drawn(scene)));
        if self.show_nodes {
            let (boxes, capped) = self.nodes_drawn;
            let capped = if capped { " (capped)" } else { "" };
            ui.label(format!("Octree nodes drawn: {boxes}{capped}"));
        }
        match self.viewport_px {
            Some([width, height]) => ui.label(format!("Viewport: {width:.0} × {height:.0} px")),
            None => ui.label("Viewport: –"),
        };
        ui.label(format!("GPU: {}", self.gpu_info));
    }
}

/// Compact load progress, messages and the file buttons, for the section pinned
/// to the bottom of the side panel. Returns the file button clicked, if any.
fn status_ui(
    ui: &mut egui::Ui,
    pending: &mut Option<PendingLoad>,
    error: &mut Option<String>,
    note: &mut Option<String>,
) -> Option<FileAction> {
    let cancelled = pending.as_ref().is_some_and(|load| {
        ui.label(format!("Loading {} · {}", load.label(), load.stage()));
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

    ui.horizontal(|ui| {
        if ui.button("Open…").clicked() {
            Some(FileAction::Open)
        } else if ui.button("Add tiles…").clicked() {
            Some(FileAction::Add)
        } else {
            None
        }
    })
    .inner
}

/// Centred screen shown while there is no scene: load progress or a hint, then
/// messages and the file buttons. Returns the file button clicked, if any, and the
/// panel's rect.
fn empty_ui(
    ui: &mut egui::Ui,
    pending: &mut Option<PendingLoad>,
    error: &mut Option<String>,
    note: &mut Option<String>,
) -> (Option<FileAction>, egui::Rect) {
    let panel = egui::CentralPanel::default().show(ui, |ui| {
        ui.vertical_centered(|ui| {
            ui.add_space(ui.available_height() * 0.35);

            let cancelled = match pending.as_ref() {
                Some(load) => loading_ui(ui, load),
                None => {
                    ui.heading("Drop LAS/LAZ files here or use Open…");
                    false
                }
            };
            if cancelled {
                *pending = None;
            }

            ui.add_space(8.0);
            messages_ui(ui, error, note);

            let open = ui.button("Open…").clicked();
            // Nothing to add to yet.
            ui.add_enabled(false, egui::Button::new("Add tiles…"));

            open.then_some(FileAction::Open)
        })
        .inner
    });

    (panel.inner, panel.response.rect)
}

/// One row per octree level of a tile: depth and points kept at that depth.
fn levels_ui(ui: &mut egui::Ui, index: usize, octree: &OctreeStats) {
    egui::Grid::new(("levels", index))
        .striped(true)
        .show(ui, |ui| {
            for (depth, points) in octree.points_per_level.iter().enumerate() {
                ui.label(depth.to_string());
                ui.label(points.to_string());
                ui.end_row();
            }
        });
}

/// Full progress block for the empty screen. Returns whether "Cancel" was clicked.
fn loading_ui(ui: &mut egui::Ui, load: &PendingLoad) -> bool {
    let read = load.read();
    let fraction = fraction(read, load.total);

    ui.heading(format!("Loading {}", load.label()));
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
        "{} · {:.1} s",
        load.stage(),
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

/// Darkens `rect` with a hint while files are dragged over the window. With a
/// scene on screen, the hint mentions adding with Shift.
fn drop_overlay(ctx: &egui::Context, rect: egui::Rect, can_add: bool) {
    if ctx.input(|input| input.raw.hovered_files.is_empty()) {
        return;
    }

    let layer = egui::LayerId::new(egui::Order::Foreground, egui::Id::unique("drop_overlay"));
    let painter = ctx.layer_painter(layer);
    painter.rect_filled(rect, 0.0, egui::Color32::from_black_alpha(180));
    painter.text(
        rect.center(),
        egui::Align2::CENTER_CENTER,
        if can_add {
            "Drop LAS/LAZ files to open them · hold Shift to add them to the scene"
        } else {
            "Drop LAS/LAZ files to open them"
        },
        egui::FontId::proportional(24.0),
        egui::Color32::WHITE,
    );
}

/// Asks for one or more files with the system dialog. Blocks the UI until the
/// dialog closes.
fn pick_files() -> Option<Vec<PathBuf>> {
    // Both cases: some dialogs match the filter case-sensitively, and PNOA tiles end in `.LAZ`.
    rfd::FileDialog::new()
        .add_filter("LiDAR", &["las", "laz", "LAS", "LAZ"])
        .pick_files()
}

/// Fraction of the points read, for a progress bar.
fn fraction(read: u64, total: u64) -> f32 {
    if total > 0 {
        read as f32 / total as f32
    } else {
        0.0
    }
}

/// The file name for a scene of one tile, "N tiles" otherwise.
fn scene_name(scene: &LoadedScene) -> String {
    match scene.tiles.as_slice() {
        [tile] => file_name(&tile.path),
        tiles => format!("{} tiles", tiles.len()),
    }
}

/// Shows the scene's name in the window title.
fn set_title(ctx: &egui::Context, scene: &LoadedScene) {
    ctx.send_viewport_cmd(egui::ViewportCommand::Title(format!(
        "Brume — {}",
        scene_name(scene)
    )));
}

/// The file names of `paths`, separated by commas.
fn file_names(paths: &[PathBuf]) -> String {
    paths
        .iter()
        .map(|path| file_name(path))
        .collect::<Vec<_>>()
        .join(", ")
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

        let (file_action, view_rect) = match &mut self.scene {
            Some(scene) => {
                // Side panel: the report, the controls and performance, scrolling
                // above a status section pinned to the bottom (load progress,
                // messages and "Open…"), so that section is always visible.
                let action = egui::Panel::left("stats_panel")
                    .show(ui, |ui| {
                        // Laid out first so it claims its space at the bottom; the
                        // scroll area then fills what is left.
                        let action = egui::Panel::bottom("load_status")
                            .resizable(false)
                            .show(ui, |ui| {
                                ui.add_space(4.0);
                                status_ui(ui, &mut self.pending, &mut self.error, &mut self.note)
                            })
                            .inner;

                        egui::ScrollArea::vertical().show(ui, |ui| {
                            scene.report_ui(ui);

                            ui.separator();

                            scene.tiles_ui(ui);

                            ui.separator();

                            scene.classes_ui(ui);

                            ui.separator();

                            self.view.controls_ui(ui, scene.deepest());

                            ui.separator();

                            ui.label("Performance");
                            self.view.perf_controls_ui(ui, scene.deepest());
                            self.view.perf_ui(ui, scene);
                        });

                        action
                    })
                    .inner;

                (action, scene.view_ui(ui, &mut self.view))
            }
            None => empty_ui(ui, &mut self.pending, &mut self.error, &mut self.note),
        };

        drop_overlay(&ctx, view_rect, self.scene.is_some());

        match file_action {
            Some(FileAction::Open) => {
                if let Some(paths) = pick_files() {
                    self.open(&ctx, paths);
                }
            }
            Some(FileAction::Add) => {
                if let Some(paths) = pick_files() {
                    self.add(&ctx, paths);
                }
            }
            None => {}
        }
    }
}
