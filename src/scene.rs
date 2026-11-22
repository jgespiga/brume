//! Plan a scene from headers, then assemble it on the loader thread.
//!
//! [`LoadPlan`] separates early validation from point decoding. It checks each
//! tile against [`check_buffer_size`] and the union against [`check_extent`].
//! These checks happen before [`load_scene`] reads points.
//!
//! ```text
//! new files -> header bounds + point counts -> LoadPlan
//!                                               |
//! existing Arc<Tile> entries --------------------+--> load new tiles
//!                                                        |
//!                                               merge stats and ranges
//!                                                        |
//!                                                   LoadedScene
//! ```
//!
//! [`LoadPlan::new_scene`] chooses a rounded shared origin.
//! [`LoadPlan::add_to`] preserves the old origin and shares existing tiles, so
//! their local coordinates and GPU buffers remain valid. The extent check is
//! a geometric limit, not coordinate-system validation or reprojection.

use std::{
    path::{Path, PathBuf},
    sync::{Arc, atomic::Ordering},
};

use anyhow::{Context, Result, bail};
use glam::DVec3;

use crate::{
    cloud::{Bounds, Stats, Tile},
    loader::{LoadControl, load, read_header_info},
    render::check_buffer_size,
    report::{ClassRow, average_color, class_rows, height_range, intensity_range},
};

/// Largest allowed extent in X or Y, in metres. This bounds horizontal offsets
/// and rejects widely separated header bounds without decoding points.
pub const MAX_SCENE_EXTENT: f64 = 50_000.0;

/// Tiles sharing one origin, with everything the viewer derives from all of them.
///
/// Built on the loader thread, including report data and colour ranges. The UI
/// uploads new tiles and creates or updates its camera and visibility state.
pub struct LoadedScene {
    /// Global coordinates of the local (0, 0, 0) of every tile.
    pub origin: DVec3,
    /// Union of the tiles' header bounds, in global coordinates.
    pub bounds: Bounds,
    /// Existing shared tiles first, followed by newly loaded tiles.
    pub tiles: Vec<Arc<Tile>>,
    /// Stats of all tiles combined.
    pub stats: Stats,
    /// Nonempty classes sorted by count for the side panel.
    pub class_rows: Vec<ClassRow>,
    /// Mean RGB across all points, or no value for an empty scene.
    pub average_color: Option<[u64; 3]>,
    /// Local height percentiles from [`height_range`], shared by every tile.
    pub height_range: [f32; 2],
    /// Intensity percentiles from [`intensity_range`], shared by every tile.
    pub intensity_range: [f32; 2],
}

/// What a load will read, decided on the UI thread from the files' headers.
pub struct LoadPlan {
    /// Global position subtracted from every point before converting to `f32`.
    pub origin: DVec3,
    /// Union of the header bounds of every tile in the resulting scene.
    pub bounds: Bounds,
    /// Tiles already loaded, which the new ones join. Empty for a new scene.
    pub existing: Vec<Arc<Tile>>,
    /// Files to read, in order.
    pub paths: Vec<PathBuf>,
    /// Points declared in the headers of [`LoadPlan::paths`], excluding old tiles.
    pub total: u64,
}

impl LoadPlan {
    /// Reads headers and checks buffer sizes and horizontal extent before any
    /// point decoding. [`Bounds::origin`] rounds the union's centre to metres.
    pub fn new_scene(paths: Vec<PathBuf>, max_buffer_size: u64) -> Result<Self> {
        let (bounds, total) = read_headers(&paths, None, max_buffer_size)?;
        check_extent(bounds)?;

        Ok(Self {
            origin: bounds.origin(),
            bounds,
            existing: Vec::new(),
            paths,
            total,
        })
    }

    /// Validates new files against the existing scene's bounds and retains its
    /// origin. Cloning the [`Arc`] entries shares old points without copying them.
    pub fn add_to(scene: &LoadedScene, paths: Vec<PathBuf>, max_buffer_size: u64) -> Result<Self> {
        let (bounds, total) = read_headers(&paths, Some(scene.bounds), max_buffer_size)?;
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
/// if given) and the sum of their point counts. Fails if a tile has too many points
/// for GPU buffers of `max_buffer_size` bytes, before any point is read.
fn read_headers(
    paths: &[PathBuf],
    start: Option<Bounds>,
    max_buffer_size: u64,
) -> Result<(Bounds, u64)> {
    let mut union = start;
    let mut total = 0;

    for path in paths {
        let header = read_header_info(path)?;
        check_buffer_size(header.points, max_buffer_size)
            .with_context(|| format!("{} doesn't fit on the GPU", path.display()))?;
        total += header.points;
        union = Some(union.map_or(header.bounds, |union| union.union(header.bounds)));
    }

    Ok((union.context("no files to load")?, total))
}

/// Fails unless `bounds` is at most [`MAX_SCENE_EXTENT`] wide in X and in Y.
/// Does not inspect coordinate-system metadata or constrain the Z extent.
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

        let chunks = || tiles.iter().map(|tile| &tile.points);
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

    /// Matches a stored path exactly. The app canonicalizes paths when possible
    /// before using this check to skip files already in the scene.
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
