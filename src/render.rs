use std::{num::NonZeroU64, sync::Arc};

use anyhow::{Context, Result, bail};
use bytemuck::{Pod, Zeroable};
use eframe::egui;
use eframe::egui_wgpu::{self, wgpu};
use eframe::wgpu::util::DeviceExt;
use eframe::wgpu::{BindGroup, Buffer};
use glam::Mat4;

use crate::cloud::Tile;
use crate::report::class_color;

/// Bits of depth buffer requested from `eframe`. The pipeline must use the same format.
pub const DEPTH_BITS: u8 = 24;

/// Bytes per point of the largest per-point buffer, the positions (3 × f32).
const MAX_BYTES_PER_POINT: u64 = std::mem::size_of::<[f32; 3]>() as u64;

/// GPU resources for drawing points.
///
/// Stored in egui's renderer, not in the app.
pub struct PointRenderer {
    pipeline: wgpu::RenderPipeline,
    /// One entry per tile, in the same order as the scene's tiles.
    tiles: Vec<TileBuffers>,
    uniform_buffer: Buffer,
    bind_group: BindGroup,
}

/// One tile's points on the GPU: one buffer per attribute, in octree order.
struct TileBuffers {
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
    /// Creates the GPU resources for a new scene and hands them to egui's renderer,
    /// replacing the previous scene's.
    ///
    /// Fails, leaving the previous scene's resources in place, if a tile doesn't fit
    /// in the GPU's largest buffer.
    pub fn register(render_state: &egui_wgpu::RenderState, tiles: &[Arc<Tile>]) -> Result<()> {
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
            label: Some("uniforms"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: NonZeroU64::new(
                            std::mem::size_of::<CameraUniform>() as u64
                        ),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: NonZeroU64::new(
                            (256 * std::mem::size_of::<[f32; 4]>()) as u64,
                        ),
                    },
                    count: None,
                },
            ],
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

        // Class palette: 256 RGBA colours, one per possible class code.
        let palette: Vec<[f32; 4]> = (0..=255u8)
            .map(|code| {
                let [r, g, b] = class_color(code);
                [
                    f32::from(r) / 255.0,
                    f32::from(g) / 255.0,
                    f32::from(b) / 255.0,
                    1.0,
                ]
            })
            .collect();

        let palette_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("class palette"),
            contents: bytemuck::cast_slice(&palette),
            usage: wgpu::BufferUsages::UNIFORM, // never changes: no COPY_DST
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("uniforms"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: palette_buffer.as_entire_binding(),
                },
            ],
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

        let tiles = tiles
            .iter()
            .map(|tile| upload_tile(device, tile))
            .collect::<Result<_>>()?;

        // Hand everything to egui's renderer, where `paint` will find it. This drops
        // the previous scene's renderer and its buffers.
        render_state
            .renderer
            .write()
            .callback_resources
            .insert(PointRenderer {
                pipeline,
                tiles,
                uniform_buffer,
                bind_group,
            });

        Ok(())
    }

    /// Uploads `tiles` and adds them after the tiles already in the renderer,
    /// which stay on the GPU untouched. Returns `false` if there is no renderer yet,
    /// and fails if a tile doesn't fit in the GPU's largest buffer.
    pub fn append(render_state: &egui_wgpu::RenderState, tiles: &[Arc<Tile>]) -> Result<bool> {
        // Upload before taking the lock, so egui's renderer isn't blocked meanwhile.
        let uploaded = tiles
            .iter()
            .map(|tile| upload_tile(&render_state.device, tile))
            .collect::<Result<Vec<TileBuffers>>>()?;

        let mut renderer = render_state.renderer.write();
        match renderer.callback_resources.get_mut::<PointRenderer>() {
            Some(points) => {
                points.tiles.extend(uploaded);
                Ok(true)
            }
            None => Ok(false),
        }
    }
}

/// Fails unless a tile of `points` points fits in GPU buffers of at most
/// `max_buffer_size` bytes (the device's `max_buffer_size` limit).
pub fn check_buffer_size(points: u64, max_buffer_size: u64) -> Result<()> {
    let bytes = points.saturating_mul(MAX_BYTES_PER_POINT);
    if bytes > max_buffer_size {
        bail!(
            "its {points} points need a {} MiB position buffer, more than this GPU's maximum \
             buffer size of {} MiB (at most {} points per tile)",
            bytes >> 20,
            max_buffer_size >> 20,
            max_buffer_size / MAX_BYTES_PER_POINT
        );
    }

    Ok(())
}

/// Uploads the points of `tile` once: one buffer per attribute, straight from the
/// SoA columns. Checks the sizes against the device's limit before creating any.
fn upload_tile(device: &wgpu::Device, tile: &Tile) -> Result<TileBuffers> {
    let points = &tile.points;
    check_buffer_size(points.len() as u64, device.limits().max_buffer_size)
        .with_context(|| format!("{} doesn't fit on the GPU", tile.path.display()))?;

    // Pack class (bits 0-7) and intensity (bits 16-31) into one u32 per point.
    let attributes: Vec<u32> = points
        .classes
        .iter()
        .zip(&points.intensity)
        .map(|(&class, &intensity)| u32::from(class) | (u32::from(intensity) << 16))
        .collect();

    Ok(TileBuffers {
        positions: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("positions"),
            contents: bytemuck::cast_slice(&points.positions),
            usage: wgpu::BufferUsages::VERTEX,
        }),
        colors: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("colors"),
            contents: bytemuck::cast_slice(&points.colors),
            usage: wgpu::BufferUsages::VERTEX,
        }),
        attributes: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("attributes"),
            contents: bytemuck::cast_slice(&attributes),
            usage: wgpu::BufferUsages::VERTEX,
        }),
        len: points.len() as u32,
    })
}

/// Attached to a rectangle in the UI; egui calls `paint` each frame.
pub struct PointsCallback {
    pub view_proj: Mat4,
    pub viewport: [f32; 2],
    pub point_size: f32,
    pub color_mode: ColorMode,
    pub height_range: [f32; 2],
    pub intensity_range: [f32; 2],
    /// Fraction of each chunk to draw (0.01..=1.0). See [`instances_to_draw`].
    pub draw_fraction: f32,
    /// Whether each tile is drawn, in the renderer's tile order. Tiles without an
    /// entry are not drawn.
    pub visible_tiles: Vec<bool>,
}

/// Number of points drawn from a chunk of `len` points when drawing `fraction` of it.
///
/// Benchmarking aid: it draws the first part of each chunk, which is file order,
/// not a spatially uniform subset, so the visible points are not representative.
pub fn instances_to_draw(len: u32, fraction: f32) -> u32 {
    ((len as f32 * fraction).ceil() as u32).min(len)
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
        let visible_tiles = renderer
            .tiles
            .iter()
            .zip(&self.visible_tiles)
            .filter(|&(_, &visible)| visible)
            .map(|(tile, _)| tile);
        for tile in visible_tiles {
            render_pass.set_vertex_buffer(0, tile.positions.slice(..));
            render_pass.set_vertex_buffer(1, tile.colors.slice(..));
            render_pass.set_vertex_buffer(2, tile.attributes.slice(..));
            let instances = instances_to_draw(tile.len, self.draw_fraction);
            render_pass.draw(0..4, 0..instances);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn buffer_size_check_allows_up_to_the_limit() {
        let max = 256 << 20;

        assert!(check_buffer_size(max / MAX_BYTES_PER_POINT, max).is_ok());
        assert!(check_buffer_size(max / MAX_BYTES_PER_POINT + 1, max).is_err());
        assert!(check_buffer_size(u64::MAX, max).is_err());
    }
}
