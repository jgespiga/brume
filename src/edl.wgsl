// Composites the offscreen point targets (edl.rs) into egui's frame.
//
// egui_wgpu sets the viewport to the view rectangle, so one triangle that covers
// clip space covers the rectangle. Fragment positions are window pixels; the
// rectangle's origin is subtracted to address the offscreen textures, which are
// exactly the rectangle's size.

// Matches EdlUniform in edl.rs.
struct Edl {
    // Top-left of the view rectangle in the window, in physical pixels.
    origin_px: vec2<f32>,
    // Gamma-space RGBA where no point is, as egui paints the panel.
    background: vec4<f32>,
};

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

@fragment
fn fs_main(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    let pixel = texture_pixel(position.xy);
    if pixel.x < 0 {
        return edl.background;
    }

    if textureLoad(log_depth_texture, pixel, 0).r == 0.0 {
        return edl.background;
    }
    return textureLoad(color_texture, pixel, 0);
}
