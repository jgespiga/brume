use eframe::egui;
use eframe::egui_wgpu::{self, wgpu};

/// GPU resources for drawing points.
///
/// Stored in egui's renderer, not in the app.
pub struct PointRenderer {
    pipeline: wgpu::RenderPipeline,
}

impl PointRenderer {
    /// Creates the GPU resources once and hands them to egui's renderer.
    pub fn register(render_state: &egui_wgpu::RenderState) {
        let device = &render_state.device;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("points"),
            source: wgpu::ShaderSource::Wgsl(include_str!("points.wgsl").into()),
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("points"),
            // derived from the shader; no inputs yet
            layout: None,
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },

            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(render_state.target_format.into())],
                compilation_options: Default::default(),
            }),

            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        render_state
            .renderer
            .write()
            .callback_resources
            .insert(PointRenderer { pipeline });
    }
}

/// Attached to a rectangle in the UI; egui calls `paint` each frame.
pub struct PointsCallback;

impl egui_wgpu::CallbackTrait for PointsCallback {
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
        // 3 vertex, 1 instance
        render_pass.draw(0..3, 0..1);
    }
}
