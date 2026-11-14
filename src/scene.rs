use std::{
    path::{Path, PathBuf},
    sync::{Arc, atomic::Ordering},
};

use anyhow::{Context, Result, bail};
use glam::DVec3;

use crate::{
    cloud::{Bounds, Stats, Tile},
    loader::{LoadControl, load, read_header_info},
    report::{ClassRow, average_color, class_rows, height_range, intensity_range},
};

/// Largest extent a scene may cover in X or Y (metres). Neighbouring tiles fit
/// easily; anything wider almost certainly mixes coordinate systems.
pub const MAX_SCENE_EXTENT: f64 = 50_000.0;

/// Tiles sharing one origin, with everything the viewer derives from all of them.
///
/// Built entirely on the loader thread, so the UI only has to upload it to the GPU.
pub struct LoadedScene {
    /// Global coordinates of the local (0, 0, 0) of every tile.
    pub origin: DVec3,
    /// Union of the tiles' header bounds, in global coordinates.
    pub bounds: Bounds,
    pub tiles: Vec<Arc<Tile>>,
    /// Stats of all tiles combined.
    pub stats: Stats,
    pub class_rows: Vec<ClassRow>,
    pub average_color: Option<[u64; 3]>,
    pub height_range: [f32; 2],
    pub intensity_range: [f32; 2],
}

/// What a load will read, decided on the UI thread from the files' headers.
pub struct LoadPlan {
    pub origin: DVec3,
    /// Union of the header bounds of every tile in the resulting scene.
    pub bounds: Bounds,
    /// Tiles already loaded, which the new ones join. Empty for a new scene.
    pub existing: Vec<Arc<Tile>>,
    /// Files to read, in order.
    pub paths: Vec<PathBuf>,
    /// Points declared in the headers of `paths`.
    pub total: u64,
}

impl LoadPlan {
    /// Plans a new scene from `paths`: reads every header, checks the tiles are
    /// neighbours, and puts the origin at the centre of their union.
    pub fn new_scene(paths: Vec<PathBuf>) -> Result<Self> {
        let (bounds, total) = read_headers(&paths, None)?;
        check_extent(bounds)?;

        Ok(Self {
            origin: bounds.origin(),
            bounds,
            existing: Vec::new(),
            paths,
            total,
        })
    }

    /// Plans adding `paths` to `scene`: they share its origin, and together with its
    /// tiles must still be neighbours.
    pub fn add_to(scene: &LoadedScene, paths: Vec<PathBuf>) -> Result<Self> {
        let (bounds, total) = read_headers(&paths, Some(scene.bounds))?;
        check_extent(bounds)?;

        Ok(Self {
            origin: scene.origin,
            bounds,
            existing: scene.tiles.clone(),
            paths,
            total,
        })
    }
}

/// Reads the headers of `paths`. Returns the union of their bounds (and `start`,
/// if given) and the sum of their point counts.
fn read_headers(paths: &[PathBuf], start: Option<Bounds>) -> Result<(Bounds, u64)> {
    let mut union = start;
    let mut total = 0;

    for path in paths {
        let header = read_header_info(path)?;
        total += header.points;
        union = Some(union.map_or(header.bounds, |union| union.union(header.bounds)));
    }

    Ok((union.context("no files to load")?, total))
}

/// Fails unless `bounds` is at most [`MAX_SCENE_EXTENT`] wide in X and in Y.
pub fn check_extent(bounds: Bounds) -> Result<()> {
    let size = bounds.max - bounds.min;
    if size.x > MAX_SCENE_EXTENT || size.y > MAX_SCENE_EXTENT {
        bail!(
            "the tiles span {:.1} × {:.1} km, more than the {:.0} km allowed: they must be \
             neighbours in the same coordinate system (e.g. the same UTM zone)",
            size.x / 1000.0,
            size.y / 1000.0,
            MAX_SCENE_EXTENT / 1000.0
        );
    }

    Ok(())
}

/// Loads the tiles of `plan` one after another (decompression is already
/// parallel), all relative to the plan's origin, and combines them with the
/// plan's existing tiles, which come first and are shared, not copied.
///
/// `control` reports progress across all tiles and allows cancelling; see [`load`].
pub fn load_scene(plan: &LoadPlan, control: &LoadControl) -> Result<LoadedScene> {
    let mut tiles = plan.existing.clone();
    tiles.reserve(plan.paths.len());

    for (index, path) in plan.paths.iter().enumerate() {
        control.tile.store(index, Ordering::Relaxed);
        let tile = load(path, plan.origin, control)
            .with_context(|| format!("could not load {}", path.display()))?;
        tiles.push(Arc::new(tile));
    }

    Ok(LoadedScene::from_tiles(plan.origin, plan.bounds, tiles))
}

impl LoadedScene {
    /// Combines the stats of `tiles` and computes the report and colour ranges over
    /// all of them.
    fn from_tiles(origin: DVec3, bounds: Bounds, tiles: Vec<Arc<Tile>>) -> Self {
        let mut stats = Stats::empty();
        for tile in &tiles {
            stats.merge(&tile.stats);
        }

        let chunks = || tiles.iter().flat_map(|tile| &tile.chunks);
        let average_color = average_color(chunks());
        let height_range = height_range(chunks());
        let intensity_range = intensity_range(chunks());

        Self {
            origin,
            bounds,
            class_rows: class_rows(&stats),
            tiles,
            stats,
            average_color,
            height_range,
            intensity_range,
        }
    }

    /// Whether a tile was loaded from `path`.
    pub fn contains(&self, path: &Path) -> bool {
        self.tiles.iter().any(|tile| tile.path == path)
    }

    /// Prints the report to stdout.
    pub fn log(&self) {
        let stats = &self.stats;

        println!("Time: {:.2}", stats.elapsed.as_secs_f64());

        println!("Tiles: {}", self.tiles.len());
        for tile in &self.tiles {
            println!(
                "  {}: {} points in {:.2} s",
                tile.path.display(),
                tile.stats.points,
                tile.stats.elapsed.as_secs_f64()
            );
        }
        println!("Origin (UTM):    {:.2?}", self.origin);
        println!("Local min (f32): {:.2?}", stats.local_min);
        println!("Local max (f32): {:.2?}", stats.local_max);
        println!("Color depth: {:?}", stats.color_depth);
        println!(
            "Intensity: {} - {}\n",
            stats.intensity_min, stats.intensity_max
        );

        for row in &self.class_rows {
            let ClassRow {
                code,
                name,
                count,
                percent,
            } = row;

            println!("{code:>3}  {name:<20} {count:>10}  {percent:>5.1}%");
        }
        if let Some(avg) = self.average_color {
            println!("\nAverage color:   rgb({}, {}, {})", avg[0], avg[1], avg[2]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bounds(width: f64, height: f64) -> Bounds {
        let min = DVec3::new(400_000.0, 4_400_000.0, 500.0);
        Bounds {
            min,
            max: min + DVec3::new(width, height, 100.0),
        }
    }

    #[test]
    fn extent_up_to_50_km_is_accepted() {
        assert!(check_extent(bounds(1_000.0, 1_000.0)).is_ok());
        assert!(check_extent(bounds(MAX_SCENE_EXTENT, MAX_SCENE_EXTENT)).is_ok());
    }

    #[test]
    fn extent_over_50_km_in_x_or_y_is_rejected() {
        for (width, height) in [(50_001.0, 1_000.0), (1_000.0, 50_001.0)] {
            let error = check_extent(bounds(width, height)).unwrap_err().to_string();
            assert!(error.contains("same coordinate system"), "{error}");
        }
    }

    #[test]
    fn tiles_in_different_utm_zones_are_rejected() {
        // Easting ~ 300 km near the west edge of one zone and ~ 700 km near the
        // east edge of the next look 400 km apart in raw coordinates.
        let west = Bounds {
            min: DVec3::new(699_000.0, 4_460_000.0, 0.0),
            max: DVec3::new(700_000.0, 4_461_000.0, 100.0),
        };
        let east = Bounds {
            min: DVec3::new(300_000.0, 4_460_000.0, 0.0),
            max: DVec3::new(301_000.0, 4_461_000.0, 100.0),
        };

        assert!(check_extent(west.union(east)).is_err());
    }
}
