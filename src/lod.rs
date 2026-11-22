use std::{cmp::Ordering, collections::BinaryHeap, ops::Range};

use glam::{Mat4, Vec3, Vec4};

use crate::octree::{NO_CHILD, Node, Octree};

/// Closest distance used for a node's screen size (metres), so a camera inside or
/// touching a node doesn't divide by zero.
const MIN_DISTANCE: f32 = 0.01;

/// The six planes of a view frustum, each `(normal, d)` with the normal pointing
/// inwards and normalised, so `normal · p + d` is the signed distance of `p`.
#[derive(Clone, Copy, Debug)]
pub struct Frustum {
    planes: [Vec4; 6],
}

impl Frustum {
    /// Extracts the planes from `view_proj` (Gribb–Hartmann), for a projection with
    /// the WebGPU/DirectX depth range: inside means -w <= x, y <= w and 0 <= z <= w.
    pub fn from_view_proj(view_proj: Mat4) -> Self {
        let [x, y, z, w] = [0, 1, 2, 3].map(|row| view_proj.row(row));
        let planes = [w + x, w - x, w + y, w - y, z, w - z].map(|plane| {
            let length = plane.truncate().length();
            if length > 0.0 { plane / length } else { plane }
        });

        Self { planes }
    }

    /// Whether the box from `min` to `max` may be in view. Conservative: it can
    /// return `true` for some boxes just outside, never `false` for a visible one.
    pub fn intersects_aabb(&self, min: Vec3, max: Vec3) -> bool {
        self.planes.iter().all(|plane| {
            // The box corner furthest along the plane's normal (the "p-vertex"):
            // if even that one is outside, the whole box is.
            let normal = plane.truncate();
            let p_vertex = Vec3::select(normal.cmpge(Vec3::ZERO), max, min);
            normal.dot(p_vertex) + plane.w >= 0.0
        })
    }
}

/// The 8 corners of the frustum of `view_proj`, in world coordinates. Corner bits:
/// 1 = right, 2 = top, 4 = far, like octants.
pub fn frustum_corners(view_proj: Mat4) -> [Vec3; 8] {
    let inverse = view_proj.inverse();
    std::array::from_fn(|corner| {
        let ndc = Vec4::new(
            if corner & 1 != 0 { 1.0 } else { -1.0 },
            if corner & 2 != 0 { 1.0 } else { -1.0 },
            if corner & 4 != 0 { 1.0 } else { 0.0 },
            1.0,
        );
        let world = inverse * ndc;
        world.truncate() / world.w
    })
}

/// How many pixels tall something of `size` metres looks at `distance` metres, in
/// a view `viewport_height_px` pixels tall with vertical field of view `fov_y`.
pub fn screen_size_px(size: f32, distance: f32, viewport_height_px: f32, fov_y: f32) -> f32 {
    let distance = distance.max(MIN_DISTANCE);
    size / (distance * (fov_y * 0.5).tan()) * (viewport_height_px * 0.5)
}

/// Distance from `eye` to the closest point of the box from `min` to `max`: 0 when
/// the eye is inside.
pub fn distance_to_aabb(eye: Vec3, min: Vec3, max: Vec3) -> f32 {
    eye.distance(eye.clamp(min, max))
}

/// How many nodes, and how detailed, [`select`] picks each frame.
#[derive(Clone, Copy, Debug)]
pub struct LodParams {
    /// Most points drawn per frame, across all tiles.
    pub point_budget: u64,
    /// A node is refined (its children considered) only while its sampling grid
    /// cells look bigger than this many pixels.
    pub min_cell_px: f32,
    /// Nodes deeper than this are never selected.
    pub max_depth: u8,
}

/// The nodes chosen to draw this frame.
#[derive(Clone, Debug, Default)]
pub struct Selection {
    /// Selected node indices of each tile, in tile order, sorted (so in buffer
    /// order, as nodes are stored breadth first).
    pub per_tile: Vec<Vec<u32>>,
    /// Points in the selected nodes.
    pub points: u64,
    pub nodes: u32,
}

/// A node's priority: its projected size in pixels. Ordered with `total_cmp`, so it
/// can go in a `BinaryHeap`.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Priority(f32);

impl Eq for Priority {}

impl Ord for Priority {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.total_cmp(&other.0)
    }
}

impl PartialOrd for Priority {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// A node waiting in the heap; the largest priority comes out first.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct Candidate {
    priority: Priority,
    tile: usize,
    node: u32,
}

/// The node's cube, as `(min, max)`.
fn node_box(node: &Node) -> (Vec3, Vec3) {
    (node.bounds_min, node.bounds_min + Vec3::splat(node.size))
}

/// Chooses the nodes to draw this frame, across all tiles.
///
/// Nodes come out of one heap ordered by projected size in pixels, so the largest
/// on screen are drawn first. Each selected node's children are queued if they
/// exist, are in the frustum, are not deeper than `params.max_depth`, and the node's
/// own sampling is still coarser than `params.min_cell_px` (its grid cell, `size /
/// grid`, projected in pixels). Selection stops at the first node whose points
/// would exceed `params.point_budget`, so the budget is never exceeded; if even a
/// root is over budget, the selection is empty.
///
/// `tiles` pairs each tile's octree with whether the tile is visible.
pub fn select(
    tiles: &[(&Octree, bool)],
    view_proj: Mat4,
    eye: Vec3,
    viewport_height_px: f32,
    fov_y: f32,
    grid: u32,
    params: &LodParams,
) -> Selection {
    let frustum = Frustum::from_view_proj(view_proj);
    let candidate = |tile: usize, index: u32, node: &Node| {
        let (min, max) = node_box(node);
        let distance = distance_to_aabb(eye, min, max);
        Candidate {
            priority: Priority(screen_size_px(
                node.size,
                distance,
                viewport_height_px,
                fov_y,
            )),
            tile,
            node: index,
        }
    };
    let in_view = |node: &Node| {
        let (min, max) = node_box(node);
        node.depth <= params.max_depth && frustum.intersects_aabb(min, max)
    };

    let mut heap = BinaryHeap::new();
    for (tile, &(octree, visible)) in tiles.iter().enumerate() {
        if let Some(root) = octree.nodes.first()
            && visible
            && in_view(root)
        {
            heap.push(candidate(tile, 0, root));
        }
    }

    let mut selection = Selection {
        per_tile: vec![Vec::new(); tiles.len()],
        ..Selection::default()
    };
    let grid = grid.max(1) as f32;

    while let Some(Candidate {
        priority,
        tile,
        node: index,
    }) = heap.pop()
    {
        let nodes = &tiles[tile].0.nodes;
        let node = &nodes[index as usize];
        let points = selection.points + u64::from(node.point_count);
        if points > params.point_budget {
            break;
        }
        selection.points = points;
        selection.nodes += 1;
        selection.per_tile[tile].push(index);

        // Refine only while this node's grid cells still look big on screen.
        if priority.0 / grid <= params.min_cell_px {
            continue;
        }
        for &child in node.children.iter().filter(|&&child| child != NO_CHILD) {
            let child_node = &nodes[child as usize];
            if in_view(child_node) {
                heap.push(candidate(tile, child, child_node));
            }
        }
    }

    for nodes in &mut selection.per_tile {
        nodes.sort_unstable();
    }

    selection
}

/// The point ranges to draw for the sorted `nodes` of `octree`, with ranges that
/// touch (one's end is the next one's start) merged into one.
pub fn draw_ranges(octree: &Octree, nodes: &[u32]) -> Vec<Range<u32>> {
    let mut ranges: Vec<Range<u32>> = Vec::new();

    for node in nodes
        .iter()
        .filter_map(|&index| octree.nodes.get(index as usize))
    {
        let range = node.first_point..node.first_point + node.point_count;
        if range.is_empty() {
            continue;
        }
        match ranges.last_mut() {
            Some(last) if last.end == range.start => last.end = range.end,
            _ => ranges.push(range),
        }
    }

    ranges
}

#[cfg(test)]
mod tests {
    use glam::camera::lh::{proj::directx::perspective, view::look_at_mat4};

    use super::*;
    use crate::{camera::FOV_Y, octree::OctreeParams};

    /// A camera at `eye` looking at `target`, with the app's projection.
    fn view_proj(eye: Vec3, target: Vec3) -> Mat4 {
        perspective(FOV_Y, 1.0, 1.0, 10_000.0) * look_at_mat4(eye, target, Vec3::Z)
    }

    /// Deterministic pseudo-random points in a 100 × 100 × 20 m box.
    fn points(n: usize) -> Vec<[f32; 3]> {
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 40) as f32 / (1u64 << 24) as f32
        };
        (0..n)
            .map(|_| [next() * 100.0, next() * 100.0, next() * 20.0])
            .collect()
    }

    fn octree(params: &OctreeParams) -> Octree {
        Octree::build(&points(20_000), params, || false)
            .expect("not cancelled")
            .0
    }

    fn small_params() -> OctreeParams {
        OctreeParams {
            grid: 8,
            max_leaf_points: 200,
            max_depth: 16,
        }
    }

    /// A camera well above the 100 m box, looking down at its centre.
    fn above() -> (Mat4, Vec3) {
        let eye = Vec3::new(50.0, 40.0, 300.0);
        (view_proj(eye, Vec3::new(50.0, 50.0, 10.0)), eye)
    }

    #[test]
    fn boxes_behind_the_camera_are_culled() {
        let frustum = Frustum::from_view_proj(view_proj(Vec3::ZERO, Vec3::X));

        assert!(!frustum.intersects_aabb(Vec3::new(-20.0, -1.0, -1.0), Vec3::new(-10.0, 1.0, 1.0)));
        assert!(frustum.intersects_aabb(Vec3::new(10.0, -1.0, -1.0), Vec3::new(20.0, 1.0, 1.0)));
    }

    #[test]
    fn boxes_containing_the_camera_are_never_culled() {
        // Larger than the near plane distance (1 m), so part of the box is always
        // in front of the camera, whichever way it looks.
        let (min, max) = (Vec3::splat(-10.0), Vec3::splat(10.0));
        for target in [
            Vec3::X,
            -Vec3::X,
            Vec3::Y,
            -Vec3::Y,
            Vec3::new(1.0, 1.0, 1.0),
        ] {
            let frustum = Frustum::from_view_proj(view_proj(Vec3::ZERO, target));
            assert!(frustum.intersects_aabb(min, max), "looking at {target}");
        }
    }

    #[test]
    fn frustum_corners_project_to_the_clip_volume_corners() {
        let matrix = view_proj(Vec3::new(5.0, -3.0, 2.0), Vec3::new(10.0, 4.0, 0.0));

        for (corner, world) in frustum_corners(matrix).into_iter().enumerate() {
            let clip = matrix * world.extend(1.0);
            let ndc = clip.truncate() / clip.w;
            let expected_z = if corner & 4 != 0 { 1.0 } else { 0.0 };
            assert!((ndc.x.abs() - 1.0).abs() < 1e-3, "{ndc}");
            assert!((ndc.y.abs() - 1.0).abs() < 1e-3, "{ndc}");
            assert!((ndc.z - expected_z).abs() < 1e-3, "{ndc}");
        }
    }

    #[test]
    fn closer_nodes_get_a_higher_priority() {
        let eye = Vec3::ZERO;
        let near = screen_size_px(
            10.0,
            distance_to_aabb(eye, Vec3::new(20.0, 0.0, 0.0), Vec3::new(30.0, 10.0, 10.0)),
            1000.0,
            FOV_Y,
        );
        let far = screen_size_px(
            10.0,
            distance_to_aabb(
                eye,
                Vec3::new(200.0, 0.0, 0.0),
                Vec3::new(210.0, 10.0, 10.0),
            ),
            1000.0,
            FOV_Y,
        );

        assert!(Priority(near) > Priority(far));
        assert_eq!(
            distance_to_aabb(Vec3::splat(5.0), Vec3::ZERO, Vec3::splat(10.0)),
            0.0
        );
    }

    #[test]
    fn selected_nodes_always_have_their_parent_selected() {
        let tree = octree(&small_params());
        let (matrix, eye) = above();
        let params = LodParams {
            point_budget: 5_000,
            min_cell_px: 1.0,
            max_depth: u8::MAX,
        };
        let selection = select(&[(&tree, true)], matrix, eye, 800.0, FOV_Y, 8, &params);

        let mut parent = vec![None; tree.nodes.len()];
        for (index, node) in tree.nodes.iter().enumerate() {
            for &child in node.children.iter().filter(|&&child| child != NO_CHILD) {
                parent[child as usize] = Some(index as u32);
            }
        }
        let selected = &selection.per_tile[0];
        assert!(selected.len() > 1, "the test needs several levels");
        for &node in selected {
            if let Some(parent) = parent[node as usize] {
                assert!(
                    selected.binary_search(&parent).is_ok(),
                    "{node} without {parent}"
                );
            }
        }
    }

    #[test]
    fn the_budget_is_never_exceeded() {
        let tree = octree(&small_params());
        let (matrix, eye) = above();

        for budget in [0, 100, 1_000, 7_000, 19_999] {
            let params = LodParams {
                point_budget: budget,
                min_cell_px: 0.0,
                max_depth: u8::MAX,
            };
            let selection = select(&[(&tree, true)], matrix, eye, 800.0, FOV_Y, 8, &params);
            let points: u64 = selection.per_tile[0]
                .iter()
                .map(|&node| u64::from(tree.nodes[node as usize].point_count))
                .sum();

            assert!(selection.points <= budget);
            assert_eq!(points, selection.points);
        }
    }

    #[test]
    fn a_root_over_budget_gives_an_empty_selection() {
        let tree = octree(&small_params());
        let (matrix, eye) = above();
        let params = LodParams {
            point_budget: u64::from(tree.nodes[0].point_count) - 1,
            min_cell_px: 0.0,
            max_depth: u8::MAX,
        };
        let selection = select(&[(&tree, true)], matrix, eye, 800.0, FOV_Y, 8, &params);

        assert_eq!(selection.nodes, 0);
        assert!(selection.per_tile[0].is_empty());
    }

    #[test]
    fn everything_in_view_with_no_limits_selects_every_node() {
        let tree = octree(&small_params());
        let (matrix, eye) = above();
        let params = LodParams {
            point_budget: u64::MAX,
            min_cell_px: 0.0,
            max_depth: u8::MAX,
        };
        let selection = select(&[(&tree, true)], matrix, eye, 800.0, FOV_Y, 8, &params);

        assert_eq!(selection.nodes as usize, tree.nodes.len());
        assert_eq!(selection.points, 20_000);
        assert_eq!(draw_ranges(&tree, &selection.per_tile[0]), vec![0..20_000]);
    }

    #[test]
    fn hidden_tiles_and_max_depth_limit_the_selection() {
        let tree = octree(&small_params());
        let (matrix, eye) = above();
        let params = LodParams {
            point_budget: u64::MAX,
            min_cell_px: 0.0,
            max_depth: 1,
        };
        let selection = select(
            &[(&tree, false), (&tree, true)],
            matrix,
            eye,
            800.0,
            FOV_Y,
            8,
            &params,
        );

        assert!(selection.per_tile[0].is_empty());
        assert!(
            selection.per_tile[1]
                .iter()
                .all(|&node| tree.nodes[node as usize].depth <= 1)
        );
    }

    #[test]
    fn touching_ranges_are_merged() {
        let tree = octree(&small_params());
        // The root and its first child are adjacent in buffer order; a node far
        // down the list is not adjacent to them.
        let last = tree.nodes.len() as u32 - 1;
        let ranges = draw_ranges(&tree, &[0, 1, last]);

        let root = &tree.nodes[0];
        let first_child = &tree.nodes[1];
        assert_eq!(ranges.len(), 2);
        assert_eq!(ranges[0], 0..root.point_count + first_child.point_count);
    }
}
