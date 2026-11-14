use std::collections::VecDeque;

use glam::Vec3;

/// Marks a missing child in [`Node::children`].
pub const NO_CHILD: u32 = u32::MAX;

/// Added to the root cube's edge so points on the far faces of the bounds still
/// fall strictly inside the grid (metres). Also gives a cloud of one point, or of
/// identical points, a cube with a non-zero edge.
const ROOT_PADDING: f32 = 1e-3;

/// How an [`Octree`] samples and splits its points.
#[derive(Clone, Copy, Debug)]
pub struct OctreeParams {
    /// Cells per edge of the sampling grid of each node: an inner node keeps at most
    /// one point per cell, so at most `grid³` points.
    pub grid: u32,
    /// A node with at most this many points keeps all of them and has no children.
    pub max_leaf_points: u32,
    /// Nodes at this depth keep all their points, however many there are.
    pub max_depth: u8,
}

impl Default for OctreeParams {
    fn default() -> Self {
        Self {
            grid: 128,
            max_leaf_points: 50_000,
            max_depth: 16,
        }
    }
}

/// One cube of an [`Octree`].
#[derive(Clone, Debug)]
pub struct Node {
    /// Corner of the cube with the smallest coordinates.
    pub bounds_min: Vec3,
    /// Edge of the cube.
    pub size: f32,
    /// 0 for the root.
    pub depth: u8,
    /// Arena index of the child in each octant, or [`NO_CHILD`]. Octant bits: 1 = upper
    /// half in x, 2 = in y, 4 = in z.
    pub children: [u32; 8],
    /// Index of this node's first point in the reordered points.
    pub first_point: u32,
    pub point_count: u32,
}

impl Node {
    fn new(bounds_min: Vec3, size: f32, depth: u8) -> Self {
        Self {
            bounds_min,
            size,
            depth,
            children: [NO_CHILD; 8],
            first_point: 0,
            point_count: 0,
        }
    }

    pub fn is_leaf(&self) -> bool {
        self.children.iter().all(|&child| child == NO_CHILD)
    }

    /// The centre of the cube.
    pub fn centre(&self) -> Vec3 {
        self.bounds_min + Vec3::splat(self.size * 0.5)
    }

    /// The octant of the cube that `point` falls in, as an index into `children`.
    pub fn octant_of(&self, point: Vec3) -> usize {
        let centre = self.centre();
        usize::from(point.x >= centre.x)
            | usize::from(point.y >= centre.y) << 1
            | usize::from(point.z >= centre.z) << 2
    }

    /// Corner and edge of the cube of `octant`.
    pub fn octant_cube(&self, octant: usize) -> (Vec3, f32) {
        let half = self.size * 0.5;
        let centre = self.centre();
        let pick = |bit: usize, low: f32, high: f32| if octant & bit != 0 { high } else { low };
        let min = Vec3::new(
            pick(1, self.bounds_min.x, centre.x),
            pick(2, self.bounds_min.y, centre.y),
            pick(4, self.bounds_min.z, centre.z),
        );

        (min, half)
    }

    /// Index of the sampling grid cell that `point` falls in.
    fn cell_of(&self, point: Vec3, grid: u32) -> usize {
        let cells = grid as f32;
        let last = grid - 1;
        let cell = ((point - self.bounds_min) / self.size * cells).floor();
        let [x, y, z] = [cell.x, cell.y, cell.z].map(|c| (c.max(0.0) as u32).min(last) as usize);
        let grid = grid as usize;

        (z * grid + y) * grid + x
    }
}

/// A level-of-detail octree over one tile's points, stored as an arena with the
/// root at index 0 and nodes in breadth-first order.
///
/// Each inner node keeps a grid-sampled subset of its points (a coarse version of
/// its cube) and passes the rest down; drawing a node and its ancestors gives every
/// point in its cube at that level of detail.
#[derive(Clone, Debug)]
pub struct Octree {
    pub nodes: Vec<Node>,
}

/// Summary of an [`Octree`]'s shape.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OctreeStats {
    pub nodes: usize,
    pub leaves: usize,
    pub max_depth: u8,
    /// Points kept by the nodes of each depth, from the root (depth 0) down.
    pub points_per_level: Vec<u64>,
}

/// Fixed-size bitset over the cells of a sampling grid.
struct Occupancy {
    words: Vec<u64>,
}

impl Occupancy {
    fn new(cells: usize) -> Self {
        Self {
            words: vec![0; cells.div_ceil(64)],
        }
    }

    fn clear(&mut self) {
        self.words.fill(0);
    }

    /// Marks `cell` as occupied. Returns whether it was empty.
    fn insert(&mut self, cell: usize) -> bool {
        let (word, bit) = (cell / 64, 1u64 << (cell % 64));
        let was_empty = self.words[word] & bit == 0;
        self.words[word] |= bit;
        was_empty
    }
}

impl Octree {
    /// Builds the octree of `positions`.
    ///
    /// Returns the tree and the order of the points: `order[i]` is the index in
    /// `positions` of the i-th reordered point. In that order each node's points are
    /// contiguous and nodes come breadth first, so coarse levels come first.
    ///
    /// `cancelled` is called before each node; when it returns `true`, the build
    /// stops and returns `None`.
    pub fn build(
        positions: &[[f32; 3]],
        params: &OctreeParams,
        mut cancelled: impl FnMut() -> bool,
    ) -> Option<(Self, Vec<u32>)> {
        let (bounds_min, size) = root_cube(positions);
        let mut nodes = vec![Node::new(bounds_min, size, 0)];
        let mut order = Vec::with_capacity(positions.len());

        let grid = params.grid.max(1);
        let mut occupancy = Occupancy::new((grid as usize).pow(3));

        // Nodes waiting to be processed, with the indices of the points that reach
        // them. Children are queued in index order, so processing the queue front
        // to back visits nodes in arena (breadth-first) order.
        let mut queue = VecDeque::from([(0, (0..positions.len() as u32).collect::<Vec<u32>>())]);

        while let Some((index, points)) = queue.pop_front() {
            if cancelled() {
                return None;
            }

            let node = nodes[index].clone();
            nodes[index].first_point = order.len() as u32;

            let is_leaf =
                points.len() <= params.max_leaf_points as usize || node.depth >= params.max_depth;
            if is_leaf {
                nodes[index].point_count = points.len() as u32;
                order.extend(points);
                continue;
            }

            // The first point in each grid cell stays here; later ones go down to
            // the child octant they fall in.
            occupancy.clear();
            let mut kept = 0;
            let mut octants: [Vec<u32>; 8] = Default::default();
            for point in points {
                let position = Vec3::from(positions[point as usize]);
                if occupancy.insert(node.cell_of(position, grid)) {
                    order.push(point);
                    kept += 1;
                } else {
                    octants[node.octant_of(position)].push(point);
                }
            }
            nodes[index].point_count = kept;

            for (octant, points) in octants.into_iter().enumerate() {
                if points.is_empty() {
                    continue;
                }
                let (child_min, child_size) = node.octant_cube(octant);
                let child = nodes.len();
                nodes.push(Node::new(child_min, child_size, node.depth + 1));
                nodes[index].children[octant] = child as u32;
                queue.push_back((child, points));
            }
        }

        Some((Self { nodes }, order))
    }

    /// Node and leaf counts, depth, and points per level.
    pub fn stats(&self) -> OctreeStats {
        let max_depth = self.nodes.iter().map(|node| node.depth).max().unwrap_or(0);
        let mut points_per_level = vec![0; usize::from(max_depth) + 1];
        for node in &self.nodes {
            points_per_level[usize::from(node.depth)] += u64::from(node.point_count);
        }

        OctreeStats {
            nodes: self.nodes.len(),
            leaves: self.nodes.iter().filter(|node| node.is_leaf()).count(),
            max_depth,
            points_per_level,
        }
    }
}

/// The cube that encloses `positions`: its corner is their minimum and its edge
/// their largest extent, plus [`ROOT_PADDING`].
fn root_cube(positions: &[[f32; 3]]) -> (Vec3, f32) {
    let Some((min, max)) = positions
        .iter()
        .map(|&p| Vec3::from(p))
        .fold(None, |bounds, p| {
            Some(bounds.map_or((p, p), |(min, max): (Vec3, Vec3)| (min.min(p), max.max(p))))
        })
    else {
        return (Vec3::ZERO, ROOT_PADDING);
    };

    (min, (max - min).max_element() + ROOT_PADDING)
}

/// Returns `values` in the order given by `order` (indices into `values`).
pub fn permute<T: Copy>(values: &[T], order: &[u32]) -> Vec<T> {
    order.iter().map(|&index| values[index as usize]).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic pseudo-random points in a 100 × 100 × 20 m box, denser near
    /// one corner so the tree is uneven.
    fn points(n: usize) -> Vec<[f32; 3]> {
        let mut state = 0x2545_f491_4f6c_dd1du64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 40) as f32 / (1u64 << 24) as f32
        };
        (0..n)
            .map(|_| {
                let (x, y, z) = (next(), next(), next());
                [x * x * 100.0, y * 100.0, z * 20.0]
            })
            .collect()
    }

    fn small_params() -> OctreeParams {
        OctreeParams {
            grid: 8,
            max_leaf_points: 100,
            max_depth: 16,
        }
    }

    fn build(positions: &[[f32; 3]], params: &OctreeParams) -> (Octree, Vec<u32>) {
        Octree::build(positions, params, || false).expect("not cancelled")
    }

    fn node_points<'a>(
        node: &Node,
        positions: &'a [[f32; 3]],
        order: &'a [u32],
    ) -> impl Iterator<Item = Vec3> + 'a {
        let range = node.first_point as usize..(node.first_point + node.point_count) as usize;
        order[range]
            .iter()
            .map(|&index| Vec3::from(positions[index as usize]))
    }

    #[test]
    fn order_is_a_permutation_and_nodes_cover_it_in_order() {
        let positions = points(20_000);
        let (tree, order) = build(&positions, &small_params());

        let mut seen = vec![false; positions.len()];
        for &index in &order {
            assert!(!seen[index as usize], "point {index} appears twice");
            seen[index as usize] = true;
        }
        assert!(seen.iter().all(|&seen| seen), "some point is missing");

        // Nodes' ranges are contiguous, in arena (breadth-first) order.
        let mut next = 0;
        for node in &tree.nodes {
            assert_eq!(node.first_point, next);
            next += node.point_count;
        }
        assert_eq!(next as usize, positions.len());
    }

    #[test]
    fn nodes_come_breadth_first() {
        let (tree, _) = build(&points(20_000), &small_params());

        assert!(
            tree.nodes
                .windows(2)
                .all(|pair| pair[0].depth <= pair[1].depth)
        );
    }

    #[test]
    fn every_point_lies_inside_its_node() {
        let positions = points(20_000);
        let (tree, order) = build(&positions, &small_params());

        for node in &tree.nodes {
            let max = node.bounds_min + Vec3::splat(node.size);
            for point in node_points(node, &positions, &order) {
                assert!(
                    point.cmpge(node.bounds_min).all() && point.cmple(max).all(),
                    "{point} outside {:?} + {}",
                    node.bounds_min,
                    node.size
                );
            }
        }
    }

    #[test]
    fn children_are_the_octants_of_their_parent() {
        let (tree, _) = build(&points(20_000), &small_params());

        for parent in &tree.nodes {
            let half = parent.size * 0.5;
            for (octant, &child) in parent.children.iter().enumerate() {
                if child == NO_CHILD {
                    continue;
                }
                let child = &tree.nodes[child as usize];
                let offset = Vec3::new(
                    (octant & 1) as f32,
                    ((octant >> 1) & 1) as f32,
                    ((octant >> 2) & 1) as f32,
                ) * half;

                assert_eq!(child.size, half);
                assert_eq!(child.depth, parent.depth + 1);
                assert!(
                    (child.bounds_min - (parent.bounds_min + offset))
                        .abs()
                        .max_element()
                        < 1e-4
                );
            }
        }
    }

    #[test]
    fn inner_nodes_keep_at_most_one_point_per_cell() {
        let positions = points(20_000);
        let params = small_params();
        let (tree, order) = build(&positions, &params);

        let inner: Vec<&Node> = tree.nodes.iter().filter(|node| !node.is_leaf()).collect();
        assert!(
            !inner.is_empty(),
            "the test input should need several levels"
        );

        for node in inner {
            assert!(node.point_count <= params.grid.pow(3));
            let mut cells = std::collections::HashSet::new();
            for point in node_points(node, &positions, &order) {
                assert!(
                    cells.insert(node.cell_of(point, params.grid)),
                    "two points in one cell"
                );
            }
        }
    }

    #[test]
    fn identical_points_stop_at_max_depth() {
        let positions = vec![[3.0, 4.0, 5.0]; 1_000];
        let params = OctreeParams {
            grid: 4,
            max_leaf_points: 10,
            max_depth: 6,
        };
        let (tree, order) = build(&positions, &params);

        assert_eq!(order.len(), positions.len());
        let deepest = tree.nodes.iter().map(|node| node.depth).max();
        assert_eq!(deepest, Some(params.max_depth));
        // One point per level above the last, which keeps the rest.
        let last = tree.nodes.last().expect("at least the root");
        assert!(last.is_leaf());
        assert_eq!(
            last.point_count as usize,
            positions.len() - usize::from(params.max_depth)
        );
    }

    #[test]
    fn stats_count_nodes_leaves_and_points_per_level() {
        let positions = vec![[3.0, 4.0, 5.0]; 1_000];
        let params = OctreeParams {
            grid: 4,
            max_leaf_points: 10,
            max_depth: 3,
        };
        let (tree, _) = build(&positions, &params);

        assert_eq!(
            tree.stats(),
            OctreeStats {
                nodes: 4,
                leaves: 1,
                max_depth: 3,
                points_per_level: vec![1, 1, 1, 997],
            }
        );
    }

    #[test]
    fn empty_input_gives_one_empty_root() {
        let (tree, order) = build(&[], &OctreeParams::default());

        assert_eq!(tree.nodes.len(), 1);
        assert!(tree.nodes[0].is_leaf());
        assert_eq!(tree.nodes[0].point_count, 0);
        assert!(order.is_empty());
    }

    #[test]
    fn small_input_is_a_single_root_leaf() {
        let positions = points(500);
        let params = OctreeParams {
            max_leaf_points: 500,
            ..OctreeParams::default()
        };
        let (tree, order) = build(&positions, &params);

        assert_eq!(tree.nodes.len(), 1);
        assert!(tree.nodes[0].is_leaf());
        assert_eq!(tree.nodes[0].point_count, 500);
        assert_eq!(order, (0..500).collect::<Vec<u32>>());
    }

    #[test]
    fn cancelling_stops_the_build() {
        let positions = points(20_000);
        let mut calls = 0;
        let result = Octree::build(&positions, &small_params(), || {
            calls += 1;
            calls > 3
        });

        assert!(result.is_none());
    }

    #[test]
    fn permute_reorders_values() {
        assert_eq!(permute(&['a', 'b', 'c'], &[2, 0, 1]), vec!['c', 'a', 'b']);
    }
}
