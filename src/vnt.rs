//! Visible node trees: which octree level is drawn at each point's location.
//!
//! [`crate::lod::select`] draws a different depth in each part of a tile. To
//! size points adaptively, the shader needs to know, for a point at position `p`,
//! the deepest selected node whose cube contains `p`. That is the level of
//! detail actually drawn there, and it sets the spacing of the points around `p`.
//!
//! A point's own node depth is not enough. Every selected node keeps its own
//! sample, so a root point sits among the root's sparse sample *and* among the
//! finer samples of any selected descendants. Sized by its own depth (the
//! root's), it would cover the detail its descendants add. Sized by the deepest
//! selected level at its location, it matches its neighbours.
//!
//! [`build`] turns a tile's selected nodes into a compact tree the shader can
//! walk, the "visible node tree" of Potree. Each [`VisibleNode`] stores which
//! octants have a selected child, as a bit mask, and where its first selected
//! child is. Children of one node are contiguous and in octant order, so child
//! `k` is found by counting the mask bits below `k`:
//!
//! ```text
//! Octree selection             Visible node tree (breadth first)
//!
//!        root                  index  child_mask   first_child
//!       /    \                   0    0b0000_1001       1        root
//!   octant 0  octant 3           1    0b0100_0000       3        octant 0
//!      |                         2    0b0000_0000       -        octant 3
//!   octant 6                     3    0b0000_0000       -        octant 0 / 6
//!
//! child k of node n = first_child(n) + popcount(child_mask(n) & ((1 << k) - 1))
//! e.g. root's octant 3 = 1 + popcount(0b1001 & 0b0111) = 1 + 1 = 2
//! ```
//!
//! [`lod_level_at`] is the reference walk; `lod_level_at` in points.wgsl does the
//! same on the GPU. Both use [`octant_index`], the octree's single definition of
//! octant order.

use bytemuck::{Pod, Zeroable};
use glam::Vec3;

use crate::octree::{NO_CHILD, Octree, octant_index};

/// Most levels a walk descends. Matches `MAX_LOD_LEVELS` in points.wgsl and the
/// default [`crate::octree::OctreeParams::max_depth`], so a walk can reach every
/// level the default octree builds.
pub const MAX_LOD_LEVELS: u8 = 16;

/// One selected octree node, laid out for the GPU as two `u32`s. Matches
/// `VisibleNode` in points.wgsl.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Pod, Zeroable)]
pub struct VisibleNode {
    /// Bit `k` is set when the child in octant `k` (see [`octant_index`]) is
    /// selected too. Only bits 0 to 7 are used.
    pub child_mask: u32,
    /// Index, in the same tree, of the first selected child. Meaningless when
    /// `child_mask` is 0.
    pub first_child: u32,
}

/// Builds the visible node tree of `octree` for the `selected` node indices.
///
/// Relies on the invariant of [`crate::lod::select`] that a selected node's
/// parent is selected too, so every selected node hangs from the root. Returns
/// an empty tree when the root is not selected (the tile draws nothing).
///
/// The output is breadth first: processing nodes in output order and appending
/// each node's selected children, in octant order, keeps siblings contiguous.
pub fn build(octree: &Octree, selected: &[u32]) -> Vec<VisibleNode> {
    let mut is_selected = vec![false; octree.nodes.len()];
    for &node in selected {
        if let Some(slot) = is_selected.get_mut(node as usize) {
            *slot = true;
        }
    }
    if !is_selected.first().copied().unwrap_or(false) {
        return Vec::new();
    }

    // Arena indices in output order; `order[i]` becomes visible node `i`.
    let mut order: Vec<u32> = vec![0];
    let mut tree = Vec::with_capacity(selected.len());

    while let Some(&arena) = order.get(tree.len()) {
        let Some(node) = octree.nodes.get(arena as usize) else {
            break;
        };

        let first_child = order.len() as u32;
        let mut child_mask = 0;
        for (octant, &child) in node.children.iter().enumerate() {
            if child != NO_CHILD && is_selected.get(child as usize).copied().unwrap_or(false) {
                child_mask |= 1 << octant;
                order.push(child);
            }
        }

        tree.push(VisibleNode {
            child_mask,
            first_child: if child_mask == 0 { 0 } else { first_child },
        });
    }

    tree
}

/// The level of detail drawn at `point`: the depth of the deepest node of `tree`
/// whose cube contains it. `root_min` and `root_size` describe the root's cube,
/// from which each step derives the child cube exactly as the octree builder did.
///
/// Returns 0 for an empty tree. The walk stops after [`MAX_LOD_LEVELS`] steps.
pub fn lod_level_at(tree: &[VisibleNode], root_min: Vec3, root_size: f32, point: Vec3) -> u8 {
    let Some(mut node) = tree.first() else {
        return 0;
    };
    let (mut min, mut size, mut level) = (root_min, root_size, 0);

    while level < MAX_LOD_LEVELS {
        // Same arithmetic as `Node::centre` and `Node::octant_cube`, so the walk
        // splits at exactly the same planes as the octree.
        let half = size * 0.5;
        let centre = min + Vec3::splat(half);
        let octant = octant_index(centre, point);
        let bit = 1u32 << octant;
        if node.child_mask & bit == 0 {
            break;
        }

        let index = node.first_child + (node.child_mask & (bit - 1)).count_ones();
        let Some(child) = tree.get(index as usize) else {
            break;
        };
        min = Vec3::select(point.cmpge(centre), centre, min);
        size = half;
        level += 1;
        node = child;
    }

    level
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::octree::{Node, OctreeParams};

    /// An 8 m root at the origin with a chain of children in octant 0 down to
    /// `depth`, and one extra child of the root in octant 7.
    fn corner_chain(depth: u8) -> Octree {
        let mut nodes = vec![node(Vec3::ZERO, 8.0, 0)];
        let far = nodes.len() as u32;
        nodes.push(node(Vec3::splat(4.0), 4.0, 1));
        nodes[0].children[7] = far;

        let mut parent = 0;
        for level in 1..=depth {
            let child = nodes.len();
            let size = 8.0 / f32::from(1u8 << level);
            nodes.push(node(Vec3::ZERO, size, level));
            nodes[parent].children[0] = child as u32;
            parent = child;
        }

        Octree { nodes }
    }

    fn node(bounds_min: Vec3, size: f32, depth: u8) -> Node {
        Node {
            bounds_min,
            size,
            depth,
            children: [NO_CHILD; 8],
            first_point: 0,
            point_count: 0,
        }
    }

    fn all(octree: &Octree) -> Vec<u32> {
        (0..octree.nodes.len() as u32).collect()
    }

    #[test]
    fn only_the_root_gives_level_zero_everywhere() {
        let octree = corner_chain(3);
        let tree = build(&octree, &[0]);

        assert_eq!(tree, vec![VisibleNode::default()]);
        for point in [Vec3::ZERO, Vec3::splat(3.9), Vec3::splat(7.9)] {
            assert_eq!(lod_level_at(&tree, Vec3::ZERO, 8.0, point), 0);
        }
    }

    #[test]
    fn a_deep_corner_is_deep_and_the_rest_shallower() {
        let octree = corner_chain(3);
        let tree = build(&octree, &all(&octree));

        // Inside the 1 m cube at the origin: depth 3.
        assert_eq!(lod_level_at(&tree, Vec3::ZERO, 8.0, Vec3::splat(0.5)), 3);
        // In the 2 m cube but outside the 1 m one: depth 2.
        assert_eq!(lod_level_at(&tree, Vec3::ZERO, 8.0, Vec3::splat(1.5)), 2);
        // The selected octant 7 child: depth 1.
        assert_eq!(lod_level_at(&tree, Vec3::ZERO, 8.0, Vec3::splat(6.0)), 1);
        // An octant with no selected child: the root's level.
        assert_eq!(
            lod_level_at(&tree, Vec3::ZERO, 8.0, Vec3::new(6.0, 1.0, 1.0)),
            0
        );
    }

    #[test]
    fn unselected_children_stop_the_walk() {
        let octree = corner_chain(3);
        // Root and its octant 0 child only.
        let tree = build(&octree, &[0, 2]);

        assert_eq!(lod_level_at(&tree, Vec3::ZERO, 8.0, Vec3::splat(0.5)), 1);
        assert_eq!(lod_level_at(&tree, Vec3::ZERO, 8.0, Vec3::splat(6.0)), 0);
    }

    #[test]
    fn siblings_are_contiguous_in_octant_order() {
        let octree = corner_chain(1);
        let tree = build(&octree, &all(&octree));

        assert_eq!(tree[0].child_mask, 0b1000_0001);
        assert_eq!(tree[0].first_child, 1);
        // Octant 0 comes before octant 7, whatever their arena order.
        assert_eq!(lod_level_at(&tree, Vec3::ZERO, 8.0, Vec3::splat(1.0)), 1);
        assert_eq!(tree.len(), 3);
    }

    #[test]
    fn an_unselected_root_gives_an_empty_tree() {
        let octree = corner_chain(2);

        assert!(build(&octree, &[]).is_empty());
        assert_eq!(lod_level_at(&[], Vec3::ZERO, 8.0, Vec3::ONE), 0);
    }

    #[test]
    fn a_point_on_a_centre_plane_goes_where_the_octree_puts_it() {
        let octree = corner_chain(1);
        let tree = build(&octree, &all(&octree));
        let on_planes = Vec3::splat(4.0);

        // The octree sends a point exactly at the centre to the upper octant (7).
        assert_eq!(octree.nodes[0].octant_of(on_planes), 7);
        assert_eq!(lod_level_at(&tree, Vec3::ZERO, 8.0, on_planes), 1);
    }

    /// Walks the real octree down to the deepest node containing `point`.
    fn octree_depth_at(octree: &Octree, point: Vec3) -> u8 {
        let mut node = &octree.nodes[0];
        loop {
            match node.children[node.octant_of(point)] {
                NO_CHILD => return node.depth,
                child => node = &octree.nodes[child as usize],
            }
        }
    }

    #[test]
    fn the_walk_matches_the_octree_children_order() {
        // A real octree with several levels. If the visible node tree used a
        // different octant order from `Node::children`, some walks would end at a
        // different depth from the octree's own walk.
        let mut state = 0x2545_f491_4f6c_dd1du64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 40) as f32 / (1u64 << 24) as f32
        };
        let positions: Vec<[f32; 3]> = (0..20_000)
            .map(|_| [next() * next() * 100.0, next() * 100.0, next() * 20.0])
            .collect();
        let params = OctreeParams {
            grid: 8,
            max_leaf_points: 100,
            max_depth: 16,
        };
        let (octree, _) = Octree::build(&positions, &params, || false).expect("not cancelled");
        let tree = build(&octree, &all(&octree));
        let root = &octree.nodes[0];

        assert!(
            octree.stats().max_depth >= 3,
            "the test needs several levels"
        );
        for node in &octree.nodes {
            for point in [
                node.centre(),
                node.bounds_min,
                node.bounds_min + Vec3::splat(node.size * 0.25),
            ] {
                assert_eq!(
                    lod_level_at(&tree, root.bounds_min, root.size, point),
                    octree_depth_at(&octree, point),
                    "at {point}"
                );
            }
        }
    }
}
