use std::num::NonZeroU64;

use bytemuck::{Pod, Zeroable};
use eframe::egui;
use eframe::egui_wgpu::{self, wgpu};
use eframe::wgpu::util::DeviceExt;
use eframe::wgpu::{BindGroup, Buffer};
use glam::Mat4;

use crate::cloud::Cloud;

/// Bits of depth buffer requested from `eframe`. The pipeline must use the same format.
pub const DEPTH_BITS: u8 = 24;

/// GPU resources for drawing points.
///
/// Stored in egui's renderer, not in the app.
pub struct PointRenderer {
    pipeline: wgpu::RenderPipeline,
    chunks: Vec<ChunkBuffers>,
    uniform_buffer: Buffer,
    bind_group: BindGroup,
}

/// One chunk's data on the GPU.
struct ChunkBuffers {
    positions: wgpu::Buffer,
    colors: wgpu::Buffer,
    attributes: wgpu::Buffer,
    len: u32,
}

/// Must match `struct Camera` in points.wgsl byte for byte (96 bytes).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct CameraUniform {
    view_proj: [[f32; 4]; 4],
    viewport: [f32; 2],
    point_size: f32,
    color_mode: u32,
    height_range: [f32; 2],
    intensity_range: [f32; 2],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColorMode {
    Rgb,
    Height,
    Classification,
    Intensity,
}

impl PointRenderer {
    /// Creates the GPU resources once and hands them to egui's renderer.
    pub fn register(render_state: &egui_wgpu::RenderState, cloud: &Cloud) {
        let device = &render_state.device;

        // We compile the shader once from de WGSL file embedded in the binary.
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("points"),
            source: wgpu::ShaderSource::Wgsl(include_str!("points.wgsl").into()),
        });

        // How the GPU reads each buffer: bytes per point and value types.
        // The `location`'s match the `@location(n)` inputs in the shader.
        let vertex_layouts = [
            Some(wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress, // 12
                step_mode: wgpu::VertexStepMode::Instance,
                attributes: &wgpu::vertex_attr_array![0 => Float32x3],
            }),
            Some(wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<[u8; 4]>() as wgpu::BufferAddress, // 4
                step_mode: wgpu::VertexStepMode::Instance,
                attributes: &wgpu::vertex_attr_array![1 => Unorm8x4],
            }),
        ];

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("camera"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: NonZeroU64::new(std::mem::size_of::<CameraUniform>() as u64),
                },
                count: None,
            }],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("points"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("camera"),
            contents: bytemuck::bytes_of(&CameraUniform::zeroed()),
            // `COPY_DST` allows writing to the buffer from the CPU every frame which
            // `UNIFORM` alone does not.
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("camera"),
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        let depth_format = egui_wgpu::depth_format_from_bits(DEPTH_BITS, 0);

        // The pipeline: shaders + fixed settings.
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("points"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[
                    // 0: positions
                    Some(wgpu::VertexBufferLayout {
                        array_stride: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                        step_mode: wgpu::VertexStepMode::Instance,
                        attributes: &[wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x3,
                            offset: 0,
                            shader_location: 0,
                        }],
                    }),
                    // 1: colors
                    Some(wgpu::VertexBufferLayout {
                        array_stride: std::mem::size_of::<[u8; 4]>() as wgpu::BufferAddress,
                        step_mode: wgpu::VertexStepMode::Instance,
                        attributes: &[wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Unorm8x4,
                            offset: 0,
                            shader_location: 1,
                        }],
                    }),
                    // 2: attributes (class | intensity << 16)  ← NEW
                    Some(wgpu::VertexBufferLayout {
                        array_stride: std::mem::size_of::<u32>() as wgpu::BufferAddress,
                        step_mode: wgpu::VertexStepMode::Instance,
                        attributes: &[wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Uint32,
                            offset: 0,
                            shader_location: 2,
                        }],
                    }),
                ],
            },

            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(render_state.target_format.into())],
                compilation_options: Default::default(),
            }),

            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                ..Default::default()
            },
            depth_stencil: depth_format.map(|format| wgpu::DepthStencilState {
                format,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Less),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        // Upload every chunk once: two buffers each, straight from the SoA columns.
        // Upload every chunk once: three buffers each, straight from the SoA columns.
        let chunks: Vec<ChunkBuffers> = cloud
            .chunks
            .iter()
            .map(|chunk| {
                // Pack class (bits 0-7) and intensity (bits 16-31) into one u32 per point.
                let attributes: Vec<u32> = chunk
                    .classes
                    .iter()
                    .zip(&chunk.intensity)
                    .map(|(&class, &intensity)| u32::from(class) | (u32::from(intensity) << 16))
                    .collect();

                ChunkBuffers {
                    positions: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("positions"),
                        contents: bytemuck::cast_slice(&chunk.positions),
                        usage: wgpu::BufferUsages::VERTEX,
                    }),
                    colors: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("colors"),
                        contents: bytemuck::cast_slice(&chunk.colors),
                        usage: wgpu::BufferUsages::VERTEX,
                    }),
                    attributes: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("attributes"),
                        contents: bytemuck::cast_slice(&attributes),
                        usage: wgpu::BufferUsages::VERTEX,
                    }),
                    len: chunk.len() as u32,
                }
            })
            .collect();

        // Hand everything to egui's renderer, where `paint` will find it.
        render_state
            .renderer
            .write()
            .callback_resources
            .insert(PointRenderer {
                pipeline,
                chunks,
                uniform_buffer,
                bind_group,
            });
    }
}

/// Attached to a rectangle in the UI; egui calls `paint` each frame.
pub struct PointsCallback {
    pub view_proj: Mat4,
    pub viewport: [f32; 2],
    pub point_size: f32,
    pub color_mode: ColorMode,
    pub height_range: [f32; 2],
    pub intensity_range: [f32; 2],
}

impl egui_wgpu::CallbackTrait for PointsCallback {
    fn prepare(
        &self,
        _device: &wgpu::Device,
        queue: &wgpu::Queue,
        _screen_descriptor: &egui_wgpu::ScreenDescriptor,
        _egui_encoder: &mut wgpu::CommandEncoder,
        resources: &mut egui_wgpu::CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        let uniform = CameraUniform {
            view_proj: self.view_proj.to_cols_array_2d(),
            viewport: self.viewport,
            point_size: self.point_size,
            color_mode: self.color_mode as u32,
            height_range: self.height_range,
            intensity_range: self.intensity_range,
        };

        if let Some(renderer) = resources.get::<PointRenderer>() {
            queue.write_buffer(&renderer.uniform_buffer, 0, bytemuck::bytes_of(&uniform));
        }
        Vec::new()
    }

    fn paint(
        &self,
        _info: egui::PaintCallbackInfo,
        render_pass: &mut wgpu::RenderPass<'static>,
        resources: &egui_wgpu::CallbackResources,
    ) {
        let Some(renderer) = resources.get::<PointRenderer>() else {
            return;
        };

        render_pass.set_pipeline(&renderer.pipeline);
        render_pass.set_bind_group(0, &renderer.bind_group, &[]);
        for chunk in &renderer.chunks {
            render_pass.set_vertex_buffer(0, chunk.positions.slice(..));
            render_pass.set_vertex_buffer(1, chunk.colors.slice(..));
            render_pass.set_vertex_buffer(2, chunk.attributes.slice(..)); // ← NEW
            render_pass.draw(0..4, 0..chunk.len);
        }
    }
}
