// Each instance is one point from a tile's octree-ordered buffers. Four vertex
// invocations expand its projected centre into a square of fixed pixel size.
// Rust uploads matching layouts in render.rs; changes to bindings, field order
// or colour mode values must agree on both sides.

// Matches CameraUniform in render.rs, including column-major matrix storage.
struct Camera {
    view_proj: mat4x4<f32>,
    viewport: vec2<f32>,
    point_size: f32,
    color_mode: u32,
    height_range: vec2<f32>,
    intensity_range: vec2<f32>,
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

// The colour of a point in the current colour mode: every mode is here.
fn point_color(in: VertexIn) -> vec4<f32> {
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

    // Pick the colour by mode (0 RGB, 1 height, 2 class, 3 intensity, 4 octree depth).
    var color = in.color;
    switch camera.color_mode {
        case 1u: { color = vec4(viridis(height_t), 1.0); }
        case 2u: { color = palettes.classes[classification]; }
        case 3u: { color = vec4(vec3(intensity_t), 1.0); }
        case 4u: { color = palettes.depths[depth % 8u]; }
        default: {}
    }
    return color;
}

// Multiply the normalized-device offset by clip.w so perspective division
// cancels it again. The square keeps its pixel size as camera distance changes.
// CORNERS spans two units, so point_size / viewport gives the required half edge
// in normalized device coordinates, whose full viewport spans two units.
@vertex
fn vs_main(@builtin(vertex_index) corner_index: u32, in: VertexIn) -> VertexOut {
    var clip = camera.view_proj * vec4(in.position, 1.0);
    let offset = CORNERS[corner_index] * camera.point_size / camera.viewport;
    clip = vec4(clip.xy + offset * clip.w, clip.z, clip.w);

    var out: VertexOut;
    out.clip = clip;
    out.color = point_color(in);
    return out;
}

// No lighting or circular mask; every covered fragment keeps the point colour.
@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    return in.color;
}
