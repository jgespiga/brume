//! Eye-dome lighting: shading point clouds from screen-space depth alone.
//!
//! Points have no normals, so ordinary lighting can't shade them, and a flat
//! colour hides their shape. Eye-dome lighting (EDL, Boucheny 2009, as used by
//! Potree and CloudCompare) darkens each pixel by how far it lies behind its
//! neighbours on screen. Edges, steps and silhouettes in front of other points
//! get dark outlines, and surfaces read as 3D, from the depth buffer only.
//!
//! ```text
//! 8 neighbours at `radius` pixels        response = sum of max(0, d - d_n) / 8
//!                                        shade    = exp(-response × strength × 300)
//!   n   n   n
//!   n   d   n      d   = log2 view depth of the centre
//!   n   n   n      d_n = log2 view depth of a neighbour
//! ```
//!
//! The depth is stored as log2 of the view depth. A step of the same proportion
//! (a wall 1 m in front of a roof 10 m away, or 10 m in front of one 100 m away)
//! gives the same difference, so the shading looks the same at any zoom. Empty
//! neighbours (no point) and neighbours outside the view are skipped: as "very
//! far" points they would add nothing to `max(0, d - d_n)` anyway.
//!
//! Points are not drawn straight into egui's render pass. In
//! [`eframe::egui_wgpu::CallbackTrait::prepare`] they are drawn into textures owned by
//! [`EdlPass`]: the colour, and the logarithm of each pixel's view depth. In
//! [`eframe::egui_wgpu::CallbackTrait::paint`], one full-screen triangle reads those
//! textures and writes the view rectangle of egui's frame.
//!
//! ```text
//! prepare (egui's encoder)                   paint (egui's render pass)
//!
//! points pass -> colour      (target format) --+
//!             -> log2 depth  (R32Float)     --+--> composite -> egui frame
//!             -> depth test  (Depth32Float)
//! ```
//!
//! Having its own depth buffer frees Brume from eframe's: egui's pass needs no
//! depth attachment, and the composite pipeline has no depth state.

use std::num::NonZeroU64;

use bytemuck::{Pod, Zeroable};
use eframe::egui_wgpu::wgpu;
use eframe::wgpu::util::DeviceExt;

use crate::render::PixelRect;

/// Format of the log depth target: one float per pixel, 0 where no point is.
pub const LOG_DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R32Float;

/// Format of the depth buffer the points pass tests against.
pub const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

/// Eye-dome lighting settings, set in the UI.
#[derive(Debug, Clone, Copy)]
pub struct EdlSettings {
    /// Off, the composite copies the colours unchanged.
    pub enabled: bool,
    /// Multiplies the darkening; 1 is Potree's default look.
    pub strength: f32,
    /// Distance of the 8 neighbours from the centre, in physical pixels. Larger
    /// radii give thicker outlines.
    pub radius_px: f32,
}

/// Uniform of the composite pass. Matches `Edl` in edl.wgsl.
///
/// ```text
/// Bytes    Field
///  0..8    origin of the view rectangle in the window, in physical pixels
///  8..12   neighbour radius in physical pixels
/// 12..16   strength
/// 16..32   background colour (gamma-space RGBA, as egui draws it)
/// 32..36   enabled flag (1 shades, 0 copies)
/// 36..48   padding to a multiple of 16
/// ```
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct EdlUniform {
    origin_px: [f32; 2],
    radius_px: f32,
    strength: f32,
    background: [f32; 4],
    enabled: u32,
    _pad: [u32; 3],
}

const _: () = assert!(std::mem::size_of::<EdlUniform>() == 48);

/// The textures the points pass renders into, all the size of the view
/// rectangle in physical pixels, and the bind group the composite reads them with.
pub(crate) struct Offscreen {
    pub size: [u32; 2],
    pub color: wgpu::TextureView,
    pub log_depth: wgpu::TextureView,
    pub depth: wgpu::TextureView,
    bind_group: wgpu::BindGroup,
}

/// The composite pipeline, its uniform, and the offscreen targets.
pub(crate) struct EdlPass {
    pipeline: wgpu::RenderPipeline,
    layout: wgpu::BindGroupLayout,
    uniform: wgpu::Buffer,
    /// The frame's format; the colour target uses it too, so compositing needs
    /// no conversion.
    color_format: wgpu::TextureFormat,
    /// Created on the first frame and whenever the view changes size.
    offscreen: Option<Offscreen>,
}

impl EdlPass {
    /// Compiles the composite shader and creates its pipeline for frames of
    /// `color_format`. The targets come later, once the view size is known.
    pub fn new(device: &wgpu::Device, color_format: wgpu::TextureFormat) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("edl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("edl.wgsl").into()),
        });

        // Both textures are read with textureLoad, so they need no sampler and
        // R32Float, which is not filterable, can be bound.
        let texture = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("edl"),
            entries: &[
                texture(0),
                texture(1),
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: NonZeroU64::new(std::mem::size_of::<EdlUniform>() as u64),
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("edl"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });

        // One triangle that covers the viewport; no vertex buffers, no depth.
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("edl"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(color_format.into())],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        let uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("edl"),
            contents: bytemuck::bytes_of(&EdlUniform::zeroed()),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        Self {
            pipeline,
            layout,
            uniform,
            color_format,
            offscreen: None,
        }
    }

    /// The offscreen targets, if a frame has created them.
    pub fn offscreen(&self) -> Option<&Offscreen> {
        self.offscreen.as_ref()
    }

    /// Makes sure the targets are `size` pixels, recreating them (and the bind
    /// group that reads them) on the first frame and when the view is resized.
    pub fn resize(&mut self, device: &wgpu::Device, size: [u32; 2]) {
        let size = [size[0].max(1), size[1].max(1)];
        if self
            .offscreen
            .as_ref()
            .is_some_and(|offscreen| offscreen.size == size)
        {
            return;
        }

        let target = |label, format, usage| {
            device
                .create_texture(&wgpu::TextureDescriptor {
                    label: Some(label),
                    size: wgpu::Extent3d {
                        width: size[0],
                        height: size[1],
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format,
                    usage,
                    view_formats: &[],
                })
                .create_view(&wgpu::TextureViewDescriptor::default())
        };
        let readable =
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING;
        let color = target("points colour", self.color_format, readable);
        let log_depth = target("points log depth", LOG_DEPTH_FORMAT, readable);
        let depth = target(
            "points depth",
            DEPTH_FORMAT,
            wgpu::TextureUsages::RENDER_ATTACHMENT,
        );

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("edl"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&color),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&log_depth),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.uniform.as_entire_binding(),
                },
            ],
        });

        self.offscreen = Some(Offscreen {
            size,
            color,
            log_depth,
            depth,
            bind_group,
        });
    }

    /// Writes this frame's uniform: where the view sits in the window, the
    /// background drawn where no point is, and the shading settings.
    pub fn write_uniform(
        &self,
        queue: &wgpu::Queue,
        view: PixelRect,
        background: [f32; 4],
        settings: EdlSettings,
    ) {
        let uniform = EdlUniform {
            origin_px: view.origin.map(|value| value as f32),
            radius_px: settings.radius_px,
            strength: settings.strength,
            background,
            enabled: u32::from(settings.enabled),
            _pad: [0; 3],
        };
        queue.write_buffer(&self.uniform, 0, bytemuck::bytes_of(&uniform));
    }

    /// Composites the targets into `pass`, whose viewport egui_wgpu has set to the
    /// view rectangle. Does nothing before the first frame created the targets.
    pub fn paint(&self, pass: &mut wgpu::RenderPass<'_>) {
        let Some(offscreen) = &self.offscreen else {
            return;
        };
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &offscreen.bind_group, &[]);
        pass.draw(0..3, 0..1);
    }
}
