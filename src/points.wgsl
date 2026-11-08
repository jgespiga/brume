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
    var clip = camera.view_proj * vec4(in.position, 1.0);

    // Move this corner by half the point size, in pixels.
    // One pixel is 2 / viewport in clip units, so ±1 × (size / 2) × (2 / viewport).
    let offset = CORNERS[corner_index] * camera.point_size / camera.viewport;

    // Multiply by w so the GPU's divide by w leaves the offset unchanged.
    clip = vec4(clip.xy + offset * clip.w, clip.z, clip.w);

    var out: VertexOut;
    out.clip = clip;
    out.color = in.color;
    
    return out;
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    return in.color;
}