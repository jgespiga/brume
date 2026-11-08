struct Camera {
    view_proj: mat4x4<f32>,
    // Drawing area in physical pixels.
    viewport: vec2<f32>,   
    // Square size in pixels.
    point_size: f32,  
};

@group(0) @binding(0)
var<uniform> camera: Camera;

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

    var out: VertexOut;
    out.clip = clip;
    // Temporary test: ground (class 2) in red, everything else in its RGB.
    out.color = select(in.color, vec4(1.0, 0.0, 0.0, 1.0), classification == 2u);
    return out;
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    return in.color;
}