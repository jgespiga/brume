// Composites the offscreen point targets (edl.rs) into egui's frame, shading
// them with eye-dome lighting: each pixel darkens by how far it lies behind its
// 8 neighbours, in log2 view depth. edl.rs explains the method.
//
// egui_wgpu sets the viewport to the view rectangle, so one triangle that covers
// clip space covers the rectangle. Fragment positions are window pixels; the
// rectangle's origin is subtracted to address the offscreen textures, which are
// exactly the rectangle's size.

// Matches EdlUniform in edl.rs.
struct Edl {
    // Top-left of the view rectangle in the window, in physical pixels.
    origin_px: vec2<f32>,
    // Distance of the neighbours from the centre, in physical pixels.
    radius_px: f32,
    strength: f32,
    // Gamma-space RGBA where no point is, as egui paints the panel.
    background: vec4<f32>,
    // 1 shades, 0 copies the colour.
    enabled: u32,
};

// Scales the summed log2 depth differences into the exponent. Differences of
// log2 depth are small (a 1 m step at 100 m is 0.014), so a large factor is
// needed; 300 is the value Potree uses, and the strength slider scales it.
const EDL_SCALE: f32 = 300.0;

// The 8 neighbour directions, unit length; scaled by radius_px.
const SQRT_HALF: f32 = 0.70710678;
var<private> DIRECTIONS: array<vec2<f32>, 8> = array<vec2<f32>, 8>(
    vec2( 1.0,  0.0),
    vec2( SQRT_HALF,  SQRT_HALF),
    vec2( 0.0,  1.0),
    vec2(-SQRT_HALF,  SQRT_HALF),
    vec2(-1.0,  0.0),
    vec2(-SQRT_HALF, -SQRT_HALF),
    vec2( 0.0, -1.0),
    vec2( SQRT_HALF, -SQRT_HALF),
);

@group(0) @binding(0)
var color_texture: texture_2d<f32>;

// log2 of each pixel's view depth; 0 means no point there.
@group(0) @binding(1)
var log_depth_texture: texture_2d<f32>;

@group(0) @binding(2)
var<uniform> edl: Edl;

// A triangle with corners (-1, -1), (3, -1) and (-1, 3): it covers the whole
// [-1, 1] clip square, and the parts outside are clipped away.
//
// (-1, 3)
//    +
//    | \
//    |   \
//    +----+ (1, 1)       the square is clip space, [-1, 1] on each axis
//    |    | \
//    +----+---+
// (-1, -1)    (3, -1)
@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
    let x = f32((index << 1u) & 2u) * 2.0 - 1.0;
    let y = f32(index & 2u) * 2.0 - 1.0;
    return vec4(x, y, 0.0, 1.0);
}

// The offscreen pixel under a window position, or None-like (-1, -1) outside.
fn texture_pixel(position: vec2<f32>) -> vec2<i32> {
    let pixel = vec2<i32>(floor(position - edl.origin_px));
    let size = vec2<i32>(textureDimensions(color_texture));
    if any(pixel < vec2(0)) || any(pixel >= size) {
        return vec2(-1);
    }
    return pixel;
}

// How far, on average, the pixel lies behind its neighbours: the sum of the
// positive log2 depth differences over 8 neighbours, divided by 8. Empty and
// out-of-view neighbours count as infinitely far, which adds nothing.
fn response(pixel: vec2<i32>, depth: f32) -> f32 {
    let size = vec2<i32>(textureDimensions(log_depth_texture));
    var sum = 0.0;
    for (var i = 0u; i < 8u; i++) {
        let neighbour = pixel + vec2<i32>(round(DIRECTIONS[i] * edl.radius_px));
        if any(neighbour < vec2(0)) || any(neighbour >= size) {
            continue;
        }
        let neighbour_depth = textureLoad(log_depth_texture, neighbour, 0).r;
        if neighbour_depth == 0.0 {
            continue;
        }
        sum += max(0.0, depth - neighbour_depth);
    }
    return sum / 8.0;
}

@fragment
fn fs_main(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    let pixel = texture_pixel(position.xy);
    if pixel.x < 0 {
        return edl.background;
    }

    let depth = textureLoad(log_depth_texture, pixel, 0).r;
    if depth == 0.0 {
        return edl.background;
    }

    let color = textureLoad(color_texture, pixel, 0);
    if edl.enabled == 0u {
        return color;
    }

    let shade = exp(-response(pixel, depth) * edl.strength * EDL_SCALE);
    return vec4(color.rgb * shade, color.a);
}
