// Each instance is one point from a tile's octree-ordered buffers. Four vertex
// invocations expand its projected centre into a square: a fixed number of
// pixels, or, in adaptive mode, the projected sample spacing of the level of
// detail drawn at the point (see lod_level_at).
// Rust uploads matching layouts in render.rs; changes to bindings, field order
// or colour mode values must agree on both sides.

// Matches CameraUniform in render.rs, including column-major matrix storage.
// 128 bytes; the offsets are listed there.
struct Camera {
    view_proj: mat4x4<f32>,
    viewport: vec2<f32>,
    point_size: f32,
    color_mode: u32,
    height_range: vec2<f32>,
    intensity_range: vec2<f32>,
    // 0 fixed (point_size), 1 adaptive (the fields below).
    size_mode: u32,
    size_scale: f32,
    min_size_px: f32,
    max_size_px: f32,
    // Pixels covered by one metre at view depth 1.
    projection_factor: f32,
    // 1 / octree grid: the sample spacing of a node, as a fraction of its edge.
    spacing_factor: f32,
    // 1 draws points as circles (fs_main discards the square's corners).
    round_points: u32,
    _pad: f32,
};

@group(0) @binding(0)
var<uniform> camera: Camera;

// Uploaded once per renderer. Keep DEPTH_PALETTE_LEN in report.rs and the depth
// modulus in point_color consistent with the depth array length.
struct Palettes {
    classes: array<vec4<f32>, 256>,
    depths: array<vec4<f32>, 8>,
};

@group(0) @binding(1)
var<uniform> palettes: Palettes;

// One selected octree node of a visible node tree (vnt.rs). child_mask has bit k
// set when the child in octant k is selected; the selected children are stored
// from first_child on, in octant order.
struct VisibleNode {
    child_mask: u32,
    first_child: u32,
};

// The drawn tile, selected with a dynamic offset. Matches TileParams in render.rs;
// WGSL rounds the struct up to 32 bytes because root_min, a vec3, aligns to 16.
struct TileParams {
    root_min: vec3<f32>,
    root_size: f32,
    // Index of this tile's root in visible_nodes. first_child is relative to it.
    vnt_offset: u32,
};

// The visible node trees of all tiles, rewritten every frame.
@group(1) @binding(0)
var<storage, read> visible_nodes: array<VisibleNode>;

@group(1) @binding(1)
var<uniform> tile: TileParams;

// Most levels lod_level_at descends; matches MAX_LOD_LEVELS in vnt.rs.
const MAX_LOD_LEVELS: u32 = 16u;

// Buffers advance per instance, so all four corners read the same attributes.
// Positions are f32x3; Unorm8x4 converts colour bytes to floats before this stage.
// Packed attributes match pack_attributes in render.rs:
//
// 31                    16 15             8 7              0
// +-----------------------+----------------+----------------+
// | intensity, 16 bits    | depth, 8 bits  | class, 8 bits  |
// +-----------------------+----------------+----------------+
struct VertexIn {
    @location(0) position: vec3<f32>,
    @location(1) color: vec4<f32>,
    @location(2) attributes: u32,
};

// Every corner has the same colour and centre depth. Interpolation therefore
// leaves the square's colour constant across both triangles.
struct VertexOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) color: vec4<f32>,
    // The corner's CORNERS value, interpolated across the square: -1..1 on each
    // axis, with the point's centre at (0, 0).
    @location(1) corner: vec2<f32>,
};

// The 4 corners of a square, in triangle-strip order.
//
// 2 (-1,+1) -------- 3 (+1,+1)
//     | \           |
//     |     \       |     Triangles use corners 0,1,2 and 2,1,3.
//     |         \   |
// 0 (-1,-1) -------- 1 (+1,-1)
var<private> CORNERS: array<vec2<f32>, 4> = array<vec2<f32>, 4>(
    vec2(-1.0, -1.0),
    vec2( 1.0, -1.0),
    vec2(-1.0,  1.0),
    vec2( 1.0,  1.0),
);

// Polynomial fit of matplotlib's viridis colour map by Matt Zucker, CC0.
// Values in 0..1 progress from dark purple through blue and green to yellow.
fn viridis(t: f32) -> vec3<f32> {
    let c0 = vec3(0.2777273272234177, 0.005407344544966578, 0.3340998053353061);
    let c1 = vec3(0.1050930431085774, 1.404613529898575, 1.384590162594685);
    let c2 = vec3(-0.3308618287255563, 0.214847559468213, 0.09509516302823659);
    let c3 = vec3(-4.634230498983486, -5.799100973351585, -19.33244095627987);
    let c4 = vec3(6.228269936347081, 14.17993336680509, 56.69055260068105);
    let c5 = vec3(4.776384997670288, -13.74514537774601, -65.35303263337234);
    let c6 = vec3(-5.435455855934631, 4.645852612178535, 26.3124352495832);
    return c0 + t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * (c5 + t * c6)))));
}

// The level of detail drawn at p: the depth of the deepest selected node of the
// tile whose cube contains p. Mirrors lod_level_at in vnt.rs, which has the tests.
//
// Each step splits the current cube at its centre, picks the octant containing p
// (bit 0 = x, 1 = y, 2 = z, set when at or above the centre, as octant_index in
// octree.rs), and descends if that octant's child is selected. The child sits at
// first_child plus the number of selected octants below it.
fn lod_level_at(p: vec3<f32>) -> u32 {
    var node = visible_nodes[tile.vnt_offset];
    var cube_min = tile.root_min;
    var size = tile.root_size;
    var level = 0u;

    for (var step = 0u; step < MAX_LOD_LEVELS; step++) {
        let half = size * 0.5;
        let centre = cube_min + vec3(half);
        let upper = p >= centre;
        let octant = select(0u, 1u, upper.x) | select(0u, 2u, upper.y) | select(0u, 4u, upper.z);
        let bit = 1u << octant;
        if (node.child_mask & bit) == 0u {
            break;
        }

        node = visible_nodes[tile.vnt_offset + node.first_child + countOneBits(node.child_mask & (bit - 1u))];
        cube_min = select(cube_min, centre, upper);
        size = half;
        level++;
    }
    return level;
}

// Whether this frame needs lod_level_at: adaptive size or the "LOD level" colour.
// The walk costs up to MAX_LOD_LEVELS steps per vertex, so other modes skip it.
fn needs_lod_level() -> bool {
    return camera.size_mode == 1u || camera.color_mode == 5u;
}

// The square's edge in pixels. In adaptive mode, the points of a level `level`
// node are about root_size * spacing_factor / 2^level metres apart; projected at
// this point's view depth (clip.w), that spacing times size_scale covers the gap
// to its neighbours.
fn point_size_px(level: u32, clip_w: f32) -> f32 {
    if camera.size_mode == 0u {
        return camera.point_size;
    }

    let spacing = tile.root_size * camera.spacing_factor / exp2(f32(level));
    let size_px = spacing * camera.size_scale * camera.projection_factor / clip_w;
    return clamp(size_px, camera.min_size_px, camera.max_size_px);
}

// The colour of a point in the current colour mode: every mode is here. `level`
// is the walked level of detail, used by the "LOD level" mode.
fn point_color(in: VertexIn, level: u32) -> vec4<f32> {
    // Unpack the per-point attributes: class | depth << 8 | intensity << 16.
    let classification = in.attributes & 0xFFu;
    let depth = (in.attributes >> 8u) & 0xFFu;
    let intensity = in.attributes >> 16u;

    // Percentile endpoints come from the whole scene. Clamping maps outliers to
    // endpoint colours. report.rs widens equal endpoints, so neither denominator
    // is zero, even for a constant attribute.
    let height_t = clamp(
        (in.position.z - camera.height_range.x) / (camera.height_range.y - camera.height_range.x),
        0.0, 1.0,
    );
    let intensity_t = clamp(
        (f32(intensity) - camera.intensity_range.x) / (camera.intensity_range.y - camera.intensity_range.x),
        0.0, 1.0,
    );

    // Pick the colour by mode (0 RGB, 1 height, 2 class, 3 intensity, 4 octree
    // depth, 5 LOD level). Depth and level share the depth palette.
    var color = in.color;
    switch camera.color_mode {
        case 1u: { color = vec4(viridis(height_t), 1.0); }
        case 2u: { color = palettes.classes[classification]; }
        case 3u: { color = vec4(vec3(intensity_t), 1.0); }
        case 4u: { color = palettes.depths[depth % 8u]; }
        case 5u: { color = palettes.depths[level % 8u]; }
        default: {}
    }
    return color;
}

// Multiply the normalized-device offset by clip.w so perspective division
// cancels it again. The square keeps its pixel size as camera distance changes.
// CORNERS spans two units, so size_px / viewport gives the required half edge
// in normalized device coordinates, whose full viewport spans two units.
@vertex
fn vs_main(@builtin(vertex_index) corner_index: u32, in: VertexIn) -> VertexOut {
    var clip = camera.view_proj * vec4(in.position, 1.0);

    // Walked once, and only when needed, then shared by size and colour.
    var level = 0u;
    if needs_lod_level() {
        level = lod_level_at(in.position);
    }

    let size_px = point_size_px(level, clip.w);
    let offset = CORNERS[corner_index] * size_px / camera.viewport;
    clip = vec4(clip.xy + offset * clip.w, clip.z, clip.w);

    var out: VertexOut;
    out.clip = clip;
    out.color = point_color(in, level);
    out.corner = CORNERS[corner_index];
    return out;
}

// No lighting; every kept fragment has the point colour. With round points, the
// fragments outside the circle inscribed in the square are discarded, so
// overlapping points read as discs instead of blocks. Discarding can cost some
// early depth testing on the GPU, which is why it can be turned off.
@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    if camera.round_points != 0u && dot(in.corner, in.corner) > 1.0 {
        discard;
    }
    return in.color;
}
