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

@group(0) @binding(1)
var<uniform> class_palette: array<vec4<f32>, 256>;

struct VertexIn {
    @location(0) position: vec3<f32>,
    @location(1) color: vec4<f32>,
    @location(2) attributes: u32,
};

struct VertexOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) color: vec4<f32>,
};

// The 4 corners of a square, in triangle-strip order.
var<private> CORNERS: array<vec2<f32>, 4> = array<vec2<f32>, 4>(
    vec2(-1.0, -1.0),
    vec2( 1.0, -1.0),
    vec2(-1.0,  1.0),
    vec2( 1.0,  1.0),
);

@vertex
fn vs_main(@builtin(vertex_index) corner_index: u32, in: VertexIn) -> VertexOut {
    // Unpack the per-point attributes.
    let classification = in.attributes & 0xFFu;
    let intensity = in.attributes >> 16u;

    // Position: same as before.
    var clip = camera.view_proj * vec4(in.position, 1.0);
    let offset = CORNERS[corner_index] * camera.point_size / camera.viewport;
    clip = vec4(clip.xy + offset * clip.w, clip.z, clip.w);

    // Normalise height and intensity to 0..1 with the ranges from the uniform.
    let height_t = clamp(
        (in.position.z - camera.height_range.x) / (camera.height_range.y - camera.height_range.x),
        0.0, 1.0,
    );
    let intensity_t = clamp(
        (f32(intensity) - camera.intensity_range.x) / (camera.intensity_range.y - camera.intensity_range.x),
        0.0, 1.0,
    );

    // Pick the colour by mode (0 RGB, 1 height, 2 class, 3 intensity).
    var color = in.color;
    switch camera.color_mode {
        case 1u: { color = vec4(vec3(height_t), 1.0); }
        case 2u: { color = class_palette[classification]; }
        case 3u: { color = vec4(vec3(intensity_t), 1.0); }
        default: {}
    }

    var out: VertexOut;
    out.clip = clip;
    out.color = color;
    return out;
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    return in.color;
}