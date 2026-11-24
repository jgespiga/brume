//! Spatial sampling for a tile's level of detail.
//!
//! [`Octree::build`] divides a cube into a sampling grid. The first point in
//! each cell stays in the current [`Node`]; later points go to child octants.
//! Each point belongs to exactly one node. Drawing children adds detail to the
//! parent sample, so [`crate::lod::select`] selects ancestors before descendants.
//! The scheme follows the sampling of Potree's modifiable nested octree, MNO,
//! described by Scheiblauer and Wimmer in 2011.
//!
//! The tree uses an arena, a vector whose indices replace pointers between
//! nodes. Both nodes and their point ranges follow breadth-first order.
//! This example omits empty octants:
//!
//! ```text
//! Tree                  Arena
//!       root            [root, A, B, C, D]
//!       /  \
//!      A    B
//!     / \
//!    C   D
//!
//! Reordered point columns
//! | root |  A   |  B   |  C   |  D   |
//! <------>                            depth 0
//! <-------------------->              through depth 1
//! <----------------------------------> all
//! ```
//!
//! A node's range contains its own sample, not its descendants' points.
//! [`Octree::build`] returns a permutation rather than moving point attributes.
//! [`crate::cloud::PointChunk::permuted`] applies it to every column, keeping
//! colours and other attributes aligned with positions. [`crate::lod::draw_ranges`]
//! can then merge adjacent selected ranges into a single draw.
//!
//! Start with [`Octree::build`] for the sampling algorithm, then [`Node::cell_of`]
//! for grid indexing and [`Octree::points_up_to_depth`] for prefix counts.

use std::collections::VecDeque;

use glam::Vec3;

/// Marks a missing child in [`Node::children`].
pub const NO_CHILD: u32 = u32::MAX;

/// Padding in metres for the enclosing cube. It gives coincident points a
/// nonzero edge and makes room for points on the far faces when precision allows.
/// [`Node::cell_of`] also clamps cell indices because rounding can reach an edge.
const ROOT_PADDING: f32 = 1e-3;

/// How an [`Octree`] samples and splits its points.
#[derive(Clone, Copy, Debug)]
pub struct OctreeParams {
    /// Cells per edge of each node's sampling grid. An inner node keeps at most
    /// `grid^3` points. [`Octree::build`] treats zero as one cell per edge.
    pub grid: u32,
    /// A node with at most this many points keeps all of them and has no children.
    pub max_leaf_points: u32,
    /// Nodes at this depth keep all remaining points. This stops repeated splits
    /// when coincident points keep falling into the same cell.
    pub max_depth: u8,
}

impl Default for OctreeParams {
    /// Uses 128 cells per edge and a depth limit of 16. A node that receives at
    /// most 50,000 points keeps them all and becomes a leaf.
    fn default() -> Self {
        Self {
            grid: 128,
            max_leaf_points: 50_000,
            max_depth: 16,
        }
    }
}

/// A cube and its own point range in an [`Octree`].
///
/// [`Node::children`] indexes [`Octree::nodes`], while [`Node::first_point`]
/// indexes the reordered columns in [`crate::cloud::Tile::points`]. These are
/// separate index spaces. A child owns points that its parent did not keep.
#[derive(Clone, Debug)]
pub struct Node {
    /// Corner with the smallest coordinates, relative to the scene origin.
    pub bounds_min: Vec3,
    /// Edge length in metres, shared by all axes so each split makes equal cubes.
    pub size: f32,
    /// Levels below the root; the root has depth 0.
    pub depth: u8,
    /// Arena index of each child, or [`NO_CHILD`] for an empty octant.
    /// [`Node::octant_of`] defines the octant bit layout.
    pub children: [u32; 8],
    /// Start of this node's sample in the reordered columns, not the input order.
    pub first_point: u32,
    /// Length of this node's sample, excluding points in descendants.
    pub point_count: u32,
}

impl Node {
    /// Creates an empty cube. The builder fills its range and child indices
    /// after deciding which points remain in it.
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

    /// Whether the node has no children. Sampling can also produce a leaf when
    /// every point occupies a different cell and none remain for child octants.
    pub fn is_leaf(&self) -> bool {
        self.children.iter().all(|&child| child == NO_CHILD)
    }

    /// The split position shared by all child octants.
    pub fn centre(&self) -> Vec3 {
        self.bounds_min + Vec3::splat(self.size * 0.5)
    }

    /// The child octant containing `point`, as an index into [`Node::children`].
    /// Points on a split plane go to the upper half of that axis.
    ///
    /// ```text
    /// Octant index bits
    /// +-------+-------+-------+
    /// | bit 2 | bit 1 | bit 0 |
    /// |   z   |   y   |   x   |
    /// +-------+-------+-------+
    /// 0 = below centre, 1 = at or above centre
    /// ```
    pub fn octant_of(&self, point: Vec3) -> usize {
        octant_index(self.centre(), point)
    }

    /// The minimum corner and edge length for a child octant.
    /// Uses the same axis bits as [`Node::octant_of`], with half the parent edge.
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

    /// Returns the grid cell index containing `point`, with x varying fastest.
    /// [`Occupancy`] uses this index to address the cell's bit.
    /// `grid` must be nonzero. Clamping handles rounding at the cube faces.
    ///
    /// ```text
    /// local position / cube size * grid -> [x, y, z] cell coordinates
    /// cell index = (z * grid + y) * grid + x
    /// ```
    fn cell_of(&self, point: Vec3, grid: u32) -> usize {
        let cells = grid as f32;
        let last = grid - 1;
        let cell = ((point - self.bounds_min) / self.size * cells).floor();
        let [x, y, z] = [cell.x, cell.y, cell.z].map(|c| (c.max(0.0) as u32).min(last) as usize);
        let grid = grid as usize;

        (z * grid + y) * grid + x
    }
}

/// A tile's spatial hierarchy, with a coarse sample in each inner [`Node`].
///
/// The tree stores ranges, not point data. [`crate::cloud::Tile`] owns both the
/// tree and the reordered columns those ranges address. Drawing every node
/// draws every input point once; drawing only the root gives the coarsest sample.
#[derive(Clone, Debug)]
pub struct Octree {
    /// Arena in breadth-first order, with the root at index zero.
    /// Even an empty input produces one root with an empty point range.
    pub nodes: Vec<Node>,
}

/// Summary of an [`Octree`]'s shape.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OctreeStats {
    /// Total number of nodes, including the root and leaves.
    pub nodes: usize,
    /// Nodes without children, as tested by [`Node::is_leaf`].
    pub leaves: usize,
    /// Deepest occupied node level, or zero for an empty root.
    pub max_depth: u8,
    /// Points owned at each depth, starting at zero. These are per-level counts;
    /// [`Octree::points_up_to_depth`] sums them into prefix counts.
    pub points_per_level: Vec<u64>,
}

/// One bit per grid cell, reused across nodes to avoid a fresh allocation for
/// each sample. Only occupancy matters; the first point goes straight into the
/// output permutation, so the bitset need not store point indices.
struct Occupancy {
    /// Packed cells, with the last word padded to a multiple of 64 bits.
    words: Vec<u64>,
}

impl Occupancy {
    /// Allocates enough zeroed words to address every cell.
    fn new(cells: usize) -> Self {
        Self {
            words: vec![0; cells.div_ceil(64)],
        }
    }

    /// Forgets the previous node's sample while retaining the allocation.
    fn clear(&mut self) {
        self.words.fill(0);
    }

    /// Claims a cell and returns whether this is its first point.
    /// The caller must pass a cell within the allocated grid.
    fn insert(&mut self, cell: usize) -> bool {
        let (word, bit) = (cell / 64, 1u64 << (cell % 64));
        let was_empty = self.words[word] & bit == 0;
        self.words[word] |= bit;
        was_empty
    }
}

impl Octree {
    /// Builds a hierarchy from positions relative to the scene origin.
    ///
    /// Returns the tree and a permutation. `order[i]` indexes the input position
    /// for output point `i`. Apply [`permute`] to each attribute column with the
    /// same order. Each node then owns one contiguous range, and coarse levels
    /// precede fine levels.
    ///
    /// A small node or a node at [`OctreeParams::max_depth`] keeps all its points.
    /// Otherwise, the first point in each grid cell stays; the rest go to child
    /// octants. This chooses existing points without averaging their attributes.
    ///
    /// ```text
    /// One node, viewed as a slice through its sampling grid
    /// +-----------+-----------+
    /// | a, b      | c         |   a, c, d -> this node's range
    /// +-----------+-----------+
    /// |           | d, e, f   |   b, e, f -> their child octants
    /// +-----------+-----------+
    /// Letters show arrival order; the first point in each cell stays.
    /// ```
    ///
    /// Calls `cancelled` before processing each node and returns `None` if it
    /// returns `true`. The initial bounds scan and work within a node do not
    /// check cancellation. An empty input returns one empty root.
    ///
    /// Point and node indices use `u32`; callers must keep inputs within that
    /// index space. The grid needs `grid^3` occupancy bits, so callers must also
    /// choose a grid that fits in memory.
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

        // Appending children to both the arena and the queue keeps point ranges
        // in arena order. A stack would instead put descendants before siblings.
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

            // Occupancy belongs to this node's grid, even though all nodes reuse
            // the same allocation. Child grids cover smaller cubes.
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

    /// Prefix lengths containing all levels through each depth.
    ///
    /// Breadth-first point order makes the running sum of
    /// [`OctreeStats::points_per_level`] a buffer cutoff.
    /// [`crate::cloud::Tile::points_up_to_depth`] caches these values. The app
    /// uses the length to set the range of its "Max octree depth" slider.
    /// Rendering uses the ranges chosen by [`crate::lod::select`].
    ///
    /// ```text
    /// | depth 0 | depth 1 | depth 2 |
    /// <--------->                    cutoff[0]
    /// <------------------->          cutoff[1]
    /// <-----------------------------> cutoff[2]
    /// ```
    pub fn points_up_to_depth(&self) -> Vec<u32> {
        self.stats()
            .points_per_level
            .iter()
            .scan(0u32, |total, &points| {
                *total += points as u32;
                Some(*total)
            })
            .collect()
    }

    /// Scans the arena to summarize its shape and point distribution.
    /// Counts only each node's own points, so descendants do not duplicate
    /// their parents' contributions.
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

/// The octant of a cube split at `centre` that contains `point`: bit 0 is set
/// when `x >= centre.x`, bit 1 for y and bit 2 for z. Points on a split plane go
/// to the upper half of that axis.
///
/// This is the single definition of the octant order. [`Node::children`] is
/// indexed by it, and [`crate::vnt`] walks its visible node trees with it, as does
/// the shader's `lod_level_at` in points.wgsl.
pub fn octant_index(centre: Vec3, point: Vec3) -> usize {
    usize::from(point.x >= centre.x)
        | usize::from(point.y >= centre.y) << 1
        | usize::from(point.z >= centre.z) << 2
}

/// Encloses the input in a cube anchored at the minimum coordinates.
/// The longest axis sets its edge, with [`ROOT_PADDING`] added for boundary
/// points and coincident inputs. Empty input uses a padded cube at zero.
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

/// Copies an attribute column into the order returned by [`Octree::build`].
/// Applying the same permutation to every column preserves point attributes.
/// Every index in `order` must address `values`, or indexing panics.
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

    /// Uses smaller grids and leaves so test clouds exercise several levels.
    fn small_params() -> OctreeParams {
        OctreeParams {
            grid: 8,
            max_leaf_points: 100,
            max_depth: 16,
        }
    }

    /// Builds without cancellation for tests of the tree's invariants.
    fn build(positions: &[[f32; 3]], params: &OctreeParams) -> (Octree, Vec<u32>) {
        Octree::build(positions, params, || false).expect("not cancelled")
    }

    /// Resolves a node's output range back to input positions for geometry checks.
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

        // Adjacent arena entries must own adjacent ranges without gaps.
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
    fn points_up_to_depth_are_prefix_counts_of_the_levels() {
        let positions = vec![[3.0, 4.0, 5.0]; 1_000];
        let params = OctreeParams {
            grid: 4,
            max_leaf_points: 10,
            max_depth: 3,
        };
        let (tree, _) = build(&positions, &params);

        assert_eq!(tree.points_up_to_depth(), vec![1, 2, 3, 1_000]);
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
