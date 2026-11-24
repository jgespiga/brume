//! GPU storage and egui callbacks for instanced point rendering.
//!
//! [`PointRenderer`] lives in egui_wgpu's callback resources so [`PointsCallback`]
//! can access it during painting. Each tile uploads once, in octree point order;
//! each frame supplies uniforms and the ranges chosen by [`crate::lod::select`].
//!
//! ```text
//! Tile.points       GPU vertex buffers       shader inputs per instance
//! positions ------> f32 x 3 ----------------> location 0
//! colors ---------> unorm8 x 4 -------------> location 1
//! classes ------+
//! intensity ----+-> packed u32 -------------> location 2
//! octree depth -+
//!
//! camera, colour ranges, point sizing -> group 0, uniform binding 0
//! class + depth palettes              -> group 0, uniform binding 1
//! visible node trees of all tiles     -> group 1, storage binding 0
//! one tile's TileParams               -> group 1, uniform binding 1 (dynamic offset)
//! ```
//!
//! [`egui_wgpu::CallbackTrait::paint`] binds a tile's columns once and draws each merged
//! range as instances of a four-vertex triangle strip. The shader expands the
//! point in screen space; depth testing keeps nearer squares in front.
//!
//! In adaptive size mode the shader also walks the tile's visible node tree
//! ([`crate::vnt`]) to find the level of detail drawn at each point, and sizes
//! the point by that level's sample spacing. Group 1 carries what that walk needs:
//!
//! ```text
//! visible_nodes  [ tile 0 tree ........ | tile 1 tree .... | ... ]   storage, rewritten each frame
//!                  ^ vnt_offset 0         ^ vnt_offset n0
//!
//! tile params    [ TileParams 0 | pad | TileParams 1 | pad | ... ]   uniform, one per stride
//!                  ^ offset 0           ^ offset stride (usually 256 bytes)
//! ```

use std::{num::NonZeroU64, ops::Range, sync::Arc};

use anyhow::{Context, Result, bail};
use bytemuck::{Pod, Zeroable};
use eframe::egui;
use eframe::egui_wgpu::{self, wgpu};
use eframe::wgpu::util::DeviceExt;
use eframe::wgpu::{BindGroup, Buffer};
use glam::Mat4;

use crate::cloud::{PointChunk, Tile};
use crate::octree::{Octree, OctreeParams};
use crate::report::{DEPTH_PALETTE_LEN, class_color, depth_color};
use crate::vnt::VisibleNode;

/// Bits of depth buffer requested from `eframe`. The pipeline must use the same format.
pub const DEPTH_BITS: u8 = 24;

/// Bytes per point of the largest per-point buffer, the positions (3 × f32).
const MAX_BYTES_PER_POINT: u64 = std::mem::size_of::<[f32; 3]>() as u64;

/// GPU resources for drawing points.
///
/// [`PointRenderer::register`] replaces the scene's resources only after all
/// uploads succeed. [`PointRenderer::append`] retains existing tile buffers.
pub struct PointRenderer {
    /// Instanced triangle strips with depth writes and a strict less-than test.
    pipeline: wgpu::RenderPipeline,
    /// One entry per tile, in the same order as the scene's tiles.
    tiles: Vec<TileBuffers>,
    /// Camera and colour settings updated by [`egui_wgpu::CallbackTrait::prepare`].
    uniform_buffer: Buffer,
    /// Camera and palette bindings shared by all tiles.
    bind_group: BindGroup,
    /// Layout of group 1, kept to rebuild [`TileBindings`] when tiles are added.
    tile_layout: wgpu::BindGroupLayout,
    /// Visible node trees and per-tile parameters, sized for the current tiles.
    tile_bindings: TileBindings,
}

/// Group 1: per-tile data for adaptive point size.
///
/// Both buffers are sized for the tiles in the renderer and rebuilt by
/// [`PointRenderer::register`] and [`PointRenderer::append`]. Their contents
/// change every frame with the selection and are written by
/// [`egui_wgpu::CallbackTrait::prepare`].
struct TileBindings {
    /// The visible node trees of all tiles, one after another. Its capacity is
    /// the total node count of all tiles, the most a selection can choose.
    visible_nodes: Buffer,
    /// One [`TileParams`] per tile, `stride` bytes apart.
    params: Buffer,
    /// Bytes between consecutive tiles' parameters: the size of [`TileParams`]
    /// rounded up to the device's `min_uniform_buffer_offset_alignment`, the
    /// granularity of dynamic offsets.
    stride: u64,
    bind_group: BindGroup,
}

/// One tile's entry in the group 1 uniform buffer, selected with a dynamic
/// offset before the tile is drawn. Matches `TileParams` in points.wgsl, which
/// rounds the struct up to 32 bytes on its own (a `vec3` aligns to 16 bytes).
///
/// ```text
/// Bytes    Field
///  0..12   root cube minimum corner
/// 12..16   root cube edge
/// 16..20   index of the tile's first visible node in the shared storage buffer
/// 20..32   padding
/// ```
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct TileParams {
    root_min: [f32; 3],
    root_size: f32,
    vnt_offset: u32,
    _pad: [u32; 3],
}

const _: () = assert!(std::mem::size_of::<TileParams>() == 32);

/// One tile's points on the GPU: one buffer per attribute, in octree order.
struct TileBuffers {
    /// Scene-relative positions in the order defined by [`Octree`].
    positions: wgpu::Buffer,
    /// RGBA bytes that the vertex format normalizes to floats.
    colors: wgpu::Buffer,
    /// Classification, depth and intensity packed by [`pack_attributes`].
    attributes: wgpu::Buffer,
    /// Instance count, used to clamp submitted draw ranges.
    len: u32,
    /// The octree root's cube, where the shader's tree walk starts.
    root_min: [f32; 3],
    root_size: f32,
    /// Octree node count: the largest visible node tree this tile can have.
    node_count: u32,
}

/// Matches `Camera` in `points.wgsl` byte for byte, with a total of 128 bytes.
/// The matrix uses column order, as returned by [`Mat4::to_cols_array_2d`].
///
/// ```text
/// Bytes      Field
///   0..64    view-projection matrix
///  64..72    viewport
///  72..76    point size (fixed mode)
///  76..80    colour mode
///  80..88    height range
///  88..96    intensity range
///  96..100   point size mode
/// 100..104   size scale (adaptive mode)
/// 104..108   minimum size in pixels (adaptive mode)
/// 108..112   maximum size in pixels (adaptive mode)
/// 112..116   projection factor: pixels per metre at view depth 1
/// 116..120   spacing factor: 1 / octree grid
/// 120..128   padding to a multiple of 16
/// ```
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct CameraUniform {
    /// Scene-relative coordinates to clip coordinates.
    view_proj: [[f32; 4]; 4],
    /// Physical viewport dimensions for clip-space square offsets.
    viewport: [f32; 2],
    /// Full square edge length in physical pixels.
    point_size: f32,
    /// Numeric selector shared with [`ColorMode`] and the shader switch.
    color_mode: u32,
    /// Local height endpoints for normalization.
    height_range: [f32; 2],
    /// Intensity endpoints for normalization.
    intensity_range: [f32; 2],
    /// Numeric selector shared with [`PointSizeMode`] and the shader.
    size_mode: u32,
    /// Multiplies the sample spacing in adaptive mode.
    size_scale: f32,
    /// Adaptive sizes are clamped to `min_size_px..=max_size_px`.
    min_size_px: f32,
    max_size_px: f32,
    /// `viewport_height_px / (2 tan(fov_y / 2))`: a length `l` at view depth `w`
    /// covers `l * projection_factor / w` pixels.
    projection_factor: f32,
    /// `1 / grid`: a node of edge `s` samples points about `s / grid` apart.
    spacing_factor: f32,
    _pad: [f32; 2],
}

const _: () = assert!(std::mem::size_of::<CameraUniform>() == 128);

/// How the shader sizes points. The discriminants are the `size_mode` values the
/// shader checks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PointSizeMode {
    /// Every point is a square of [`PointSizing::fixed_px`] pixels.
    Fixed,
    /// Each point covers the sample spacing of the level of detail drawn at its
    /// location, projected to the screen. See [`crate::vnt`].
    Adaptive,
}

/// Point size settings, set in the UI.
#[derive(Debug, Clone, Copy)]
pub struct PointSizing {
    pub mode: PointSizeMode,
    /// Square edge in physical pixels in [`PointSizeMode::Fixed`].
    pub fixed_px: f32,
    /// In [`PointSizeMode::Adaptive`], the size as a multiple of the projected
    /// sample spacing. Above 1, neighbouring squares overlap and close gaps.
    pub scale: f32,
    /// Clamp of adaptive sizes, in physical pixels.
    pub min_px: f32,
    pub max_px: f32,
}

/// How points are coloured. The discriminants are the `color_mode` values the
/// shader switches on, so new modes go at the end.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColorMode {
    /// RGB from the file, or the loader's fallback grey.
    Rgb,
    /// Local height on a viridis scale between scene percentiles.
    Height,
    /// Class code mapped through [`class_color`].
    Classification,
    /// Intensity as grey between scene percentiles.
    Intensity,
    /// Owning node's depth, using the cyclic palette from [`depth_color`].
    OctreeDepth,
    /// The level of detail drawn at the point's location, from the visible node
    /// tree ([`crate::vnt`]), in the same palette. Unlike [`ColorMode::OctreeDepth`],
    /// a coarse point inside a refined region takes the refined region's colour,
    /// which shows the levels adaptive point size uses.
    LodLevel,
}

impl PointRenderer {
    /// Creates the GPU resources for a new scene and hands them to egui's renderer,
    /// replacing the previous scene's.
    ///
    /// Fails, leaving the previous scene's resources in place, if a tile doesn't fit
    /// in the GPU's largest buffer.
    pub fn register(render_state: &egui_wgpu::RenderState, tiles: &[Arc<Tile>]) -> Result<()> {
        let device = &render_state.device;

        // Embedding the shader keeps the executable independent of source paths.
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("points"),
            source: wgpu::ShaderSource::Wgsl(include_str!("points.wgsl").into()),
        });

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
                            ((256 + DEPTH_PALETTE_LEN) * std::mem::size_of::<[f32; 4]>()) as u64,
                        ),
                    },
                    count: None,
                },
            ],
        });

        let tile_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("tiles"),
                entries: &[
                    // Visible node trees: read in the vertex stage, so the device needs
                    // vertex-stage storage buffers (any non-WebGL wgpu backend has them).
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: NonZeroU64::new(
                                std::mem::size_of::<VisibleNode>() as u64
                            ),
                        },
                        count: None,
                    },
                    // The drawn tile's parameters, chosen per tile with a dynamic offset.
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: true,
                            min_binding_size: NonZeroU64::new(
                                std::mem::size_of::<TileParams>() as u64
                            ),
                        },
                        count: None,
                    },
                ],
            });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("points"),
            bind_group_layouts: &[Some(&bind_group_layout), Some(&tile_layout)],
            immediate_size: 0,
        });

        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("camera"),
            contents: bytemuck::bytes_of(&CameraUniform::zeroed()),
            // `COPY_DST` allows writing to the buffer from the CPU every frame which
            // `UNIFORM` alone does not.
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Palettes, laid out like `Palettes` in points.wgsl: 256 class colours, one
        // per possible class code, then the octree depth colours. RGBA, one vec4 each.
        let palette: Vec<[f32; 4]> = (0..=255u8)
            .map(class_color)
            .chain((0..DEPTH_PALETTE_LEN as u8).map(depth_color))
            .map(|[r, g, b]| {
                [
                    f32::from(r) / 255.0,
                    f32::from(g) / 255.0,
                    f32::from(b) / 255.0,
                    1.0,
                ]
            })
            .collect();

        let palette_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("palettes"),
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
                    // One packed integer preserves class, depth and intensity bits.
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

        let tiles: Vec<TileBuffers> = tiles
            .iter()
            .map(|tile| upload_tile(device, tile))
            .collect::<Result<_>>()?;
        let tile_bindings = TileBindings::new(device, &tile_layout, &tiles);

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
                tile_layout,
                tile_bindings,
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
                // More tiles need room for more nodes and parameters.
                points.tile_bindings =
                    TileBindings::new(&render_state.device, &points.tile_layout, &points.tiles);
                Ok(true)
            }
            None => Ok(false),
        }
    }
}

impl TileBindings {
    /// Creates the group 1 buffers and bind group for `tiles`. The contents are
    /// written each frame; only the sizes depend on the tiles.
    fn new(device: &wgpu::Device, layout: &wgpu::BindGroupLayout, tiles: &[TileBuffers]) -> Self {
        let params_size = std::mem::size_of::<TileParams>() as u64;
        let alignment = u64::from(device.limits().min_uniform_buffer_offset_alignment);
        let stride = params_size.div_ceil(alignment) * alignment;

        // At least one entry of each, so the bindings are never empty.
        let nodes: u64 = tiles.iter().map(|tile| u64::from(tile.node_count)).sum();
        let visible_nodes = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("visible nodes"),
            size: nodes.max(1) * std::mem::size_of::<VisibleNode>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tile params"),
            size: (tiles.len() as u64).max(1) * stride,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("tiles"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: visible_nodes.as_entire_binding(),
                },
                // A window of one TileParams, moved per tile by the dynamic offset.
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &params,
                        offset: 0,
                        size: NonZeroU64::new(params_size),
                    }),
                },
            ],
        });

        Self {
            visible_nodes,
            params,
            stride,
            bind_group,
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

/// Uploads positions and colours directly, and packs the remaining attributes
/// with [`pack_attributes`]. Checks the largest column against the device limit
/// before creating buffers.
fn upload_tile(device: &wgpu::Device, tile: &Tile) -> Result<TileBuffers> {
    let points = &tile.points;
    check_buffer_size(points.len() as u64, device.limits().max_buffer_size)
        .with_context(|| format!("{} doesn't fit on the GPU", tile.path.display()))?;

    let attributes = pack_attributes(points, &tile.octree);
    let (root_min, root_size) = tile.octree.nodes.first().map_or(([0.0; 3], 1.0), |root| {
        (root.bounds_min.to_array(), root.size)
    });

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
        root_min,
        root_size,
        node_count: tile.octree.nodes.len() as u32,
    })
}

/// Packs class (bits 0-7), octree node depth (bits 8-15) and intensity (bits 16-31)
/// into one u32 per point. The depth comes from the node ranges of `octree`, whose
/// node order `points` follows.
///
/// ```text
/// 31                    16 15             8 7              0
/// +-----------------------+----------------+----------------+
/// | intensity, 16 bits    | depth, 8 bits  | class, 8 bits  |
/// +-----------------------+----------------+----------------+
/// ```
fn pack_attributes(points: &PointChunk, octree: &Octree) -> Vec<u32> {
    let mut attributes: Vec<u32> = points
        .classes
        .iter()
        .zip(&points.intensity)
        .map(|(&class, &intensity)| u32::from(class) | (u32::from(intensity) << 16))
        .collect();

    for node in &octree.nodes {
        let range = node.first_point as usize..(node.first_point + node.point_count) as usize;
        if let Some(node_attributes) = attributes.get_mut(range) {
            for attribute in node_attributes {
                *attribute |= u32::from(node.depth) << 8;
            }
        }
    }

    attributes
}

/// Per-frame commands for the view rectangle. [`egui_wgpu::CallbackTrait::prepare`]
/// writes uniforms; [`egui_wgpu::CallbackTrait::paint`] uses the stored tile buffers.
pub struct PointsCallback {
    /// Current camera matrix, even when the LOD selection is frozen.
    pub view_proj: Mat4,
    /// View rectangle size in physical pixels for screen-space point sizes.
    pub viewport: [f32; 2],
    /// Fixed or adaptive point size and its settings.
    pub point_sizing: PointSizing,
    /// `viewport_height_px / (2 tan(fov_y / 2))`, for adaptive sizes.
    pub projection_factor: f32,
    /// Shader colour selector matching [`ColorMode`].
    pub color_mode: ColorMode,
    /// Scene-wide local height endpoints for the viridis scale.
    pub height_range: [f32; 2],
    /// Scene-wide intensity endpoints for the grey scale.
    pub intensity_range: [f32; 2],
    /// The point ranges to draw from each tile's buffers, in the renderer's tile
    /// order: one per selected octree node, with touching ranges merged. Each range
    /// is one draw call; tiles without ranges are not drawn.
    pub draw_ranges: Vec<Vec<Range<u32>>>,
    /// Each tile's visible node tree, in the renderer's tile order, built from the
    /// same selection as `draw_ranges` by [`crate::vnt::build`].
    pub visible_nodes: Vec<Vec<VisibleNode>>,
}

impl egui_wgpu::CallbackTrait for PointsCallback {
    /// Updates the camera uniform, the visible node trees and the per-tile
    /// parameters before painting; no extra command buffers are needed because
    /// the queue performs the writes.
    fn prepare(
        &self,
        _device: &wgpu::Device,
        queue: &wgpu::Queue,
        _screen_descriptor: &egui_wgpu::ScreenDescriptor,
        _egui_encoder: &mut wgpu::CommandEncoder,
        resources: &mut egui_wgpu::CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        let sizing = self.point_sizing;
        let uniform = CameraUniform {
            view_proj: self.view_proj.to_cols_array_2d(),
            viewport: self.viewport,
            point_size: sizing.fixed_px,
            color_mode: self.color_mode as u32,
            height_range: self.height_range,
            intensity_range: self.intensity_range,
            size_mode: sizing.mode as u32,
            size_scale: sizing.scale,
            min_size_px: sizing.min_px,
            max_size_px: sizing.max_px,
            projection_factor: self.projection_factor,
            spacing_factor: 1.0 / OctreeParams::default().grid.max(1) as f32,
            _pad: [0.0; 2],
        };

        let Some(renderer) = resources.get::<PointRenderer>() else {
            return Vec::new();
        };
        queue.write_buffer(&renderer.uniform_buffer, 0, bytemuck::bytes_of(&uniform));

        // All tiles' trees go into one buffer; each tile's parameters record
        // where its tree starts.
        let bindings = &renderer.tile_bindings;
        let mut nodes: Vec<VisibleNode> = Vec::new();
        let mut params = vec![0u8; renderer.tiles.len().max(1) * bindings.stride as usize];
        for (index, tile) in renderer.tiles.iter().enumerate() {
            let tile_params = TileParams {
                root_min: tile.root_min,
                root_size: tile.root_size,
                vnt_offset: nodes.len() as u32,
                _pad: [0; 3],
            };
            let start = index * bindings.stride as usize;
            let bytes = bytemuck::bytes_of(&tile_params);
            params[start..start + bytes.len()].copy_from_slice(bytes);
            if let Some(tree) = self.visible_nodes.get(index) {
                nodes.extend_from_slice(tree);
            }
        }

        let node_bytes: &[u8] = bytemuck::cast_slice(&nodes);
        // A selection never has more nodes than its tiles; the check only guards
        // against a callback built for a different set of tiles.
        if !node_bytes.is_empty() && node_bytes.len() as u64 <= bindings.visible_nodes.size() {
            queue.write_buffer(&bindings.visible_nodes, 0, node_bytes);
        }
        queue.write_buffer(&bindings.params, 0, &params);

        Vec::new()
    }

    /// Draws selected instances in egui's render pass. Each tile binds its columns
    /// and its group 1 offset once, then each merged range produces one
    /// four-vertex instanced draw.
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
        // Each tile's buffers are bound once, then one draw call per range.
        let bindings = &renderer.tile_bindings;
        for (index, (tile, ranges)) in renderer.tiles.iter().zip(&self.draw_ranges).enumerate() {
            if ranges.is_empty() {
                continue;
            }
            let params_offset = index as u64 * bindings.stride;
            render_pass.set_bind_group(1, &bindings.bind_group, &[params_offset as u32]);
            render_pass.set_vertex_buffer(0, tile.positions.slice(..));
            render_pass.set_vertex_buffer(1, tile.colors.slice(..));
            render_pass.set_vertex_buffer(2, tile.attributes.slice(..));
            for range in ranges {
                let instances = range.start.min(tile.len)..range.end.min(tile.len);
                if !instances.is_empty() {
                    render_pass.draw(0..4, instances);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn attributes_pack_class_depth_and_intensity() {
        let positions = vec![[3.0, 4.0, 5.0]; 100];
        let params = crate::octree::OctreeParams {
            grid: 4,
            max_leaf_points: 10,
            max_depth: 2,
        };
        let (octree, _) = Octree::build(&positions, &params, || false).expect("not cancelled");
        let points = PointChunk {
            positions,
            classes: vec![6; 100],
            colors: vec![[0; 4]; 100],
            intensity: vec![0xABCD; 100],
        };

        let attributes = pack_attributes(&points, &octree);

        // One point at depth 0, one at depth 1, the remaining 98 at depth 2.
        let depths: Vec<u32> = attributes.iter().map(|a| (a >> 8) & 0xFF).collect();
        assert_eq!(&depths[..3], &[0, 1, 2]);
        assert!(depths[2..].iter().all(|&depth| depth == 2));
        assert!(
            attributes
                .iter()
                .all(|a| a & 0xFF == 6 && a >> 16 == 0xABCD)
        );
    }

    #[test]
    fn buffer_size_check_allows_up_to_the_limit() {
        let max = 256 << 20;

        assert!(check_buffer_size(max / MAX_BYTES_PER_POINT, max).is_ok());
        assert!(check_buffer_size(max / MAX_BYTES_PER_POINT + 1, max).is_err());
        assert!(check_buffer_size(u64::MAX, max).is_err());
    }
}
