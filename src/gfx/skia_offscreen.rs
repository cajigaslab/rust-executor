use std::ffi::c_void;

use anyhow::{anyhow, Result};
use ash::vk;
use ash::vk::Handle;
use skia_safe::gpu::vk as skia_vk;
use skia_safe::gpu::{self, direct_contexts, surfaces, BackendRenderTarget, DirectContext};
use skia_safe::{Canvas, Color4f, ColorType, Surface};

use crate::gfx::base::VulkanBase;
use crate::gfx::offscreen::OffscreenTarget;

/// Draws the subject-view scene directly into the app's existing offscreen
/// `VkImage` using Skia's Vulkan (Ganesh) backend, sharing this app's `ash`
/// instance/device/queue rather than creating a separate GPU context.
///
/// Skia tracks the image's Vulkan layout itself (via the `BackendRenderTarget`)
/// separately from any tracking the rest of `gfx/` does for its own barriers, so
/// [`Self::set_layout`]/[`Self::current_layout`] keep the two in sync around
/// whatever the caller does to the image between calls to [`Self::draw`].
pub struct SkiaOffscreen {
    context: DirectContext,
    backend_render_target: BackendRenderTarget,
    surface: Surface,
}

impl SkiaOffscreen {
    /// # Safety
    /// `base` and `offscreen` must outlive the returned `SkiaOffscreen`: Skia holds
    /// onto the raw Vulkan handles they own (instance/device/queue/image) without
    /// any reference counting of its own.
    pub unsafe fn new(base: &VulkanBase, offscreen: &OffscreenTarget) -> Result<Self> {
        let get_proc = |of: skia_vk::GetProcOf| unsafe {
            let name = of.name();
            match (base.entry.static_fn().get_instance_proc_addr)(base.instance.handle(), name.as_ptr()) {
                Some(f) => f as *const c_void,
                None => std::ptr::null(),
            }
        };

        let backend_context = unsafe {
            skia_vk::BackendContext::new_builder(
                base.instance.handle().as_raw() as skia_vk::Instance,
                base.physical_device.as_raw() as skia_vk::PhysicalDevice,
                base.device.handle().as_raw() as skia_vk::Device,
                (base.queue.as_raw() as skia_vk::Queue, base.queue_family_index as usize),
                &get_proc,
                None,
            )
            .build()
        };

        let mut context = direct_contexts::make_vulkan(&backend_context, None)
            .ok_or_else(|| anyhow!("failed to create Skia Vulkan GrDirectContext"))?;

        let backend_render_target = gpu::backend_render_targets::make_vk(
            (offscreen.extent.width as i32, offscreen.extent.height as i32),
            &unsafe {
                skia_vk::ImageInfo::new(
                    offscreen.image.as_raw() as skia_vk::Image,
                    skia_vk::Alloc::default(),
                    skia_vk::ImageTiling::OPTIMAL,
                    skia_vk::ImageLayout::UNDEFINED,
                    skia_vk::Format::B8G8R8A8_UNORM,
                    1,
                    None,
                    None,
                    None,
                    None,
                )
            },
        );

        let surface = surfaces::wrap_backend_render_target(
            &mut context,
            &backend_render_target,
            gpu::SurfaceOrigin::TopLeft,
            ColorType::BGRA8888,
            None,
            None,
        )
        .ok_or_else(|| anyhow!("failed to wrap the offscreen VkImage as a Skia render target"))?;

        Ok(Self {
            context,
            backend_render_target,
            surface,
        })
    }

    /// Tells Skia's tracking what layout the image is actually in right now,
    /// e.g. after the caller transitioned it outside of Skia. Must be called
    /// before [`Self::render`] if the layout changed since the last call.
    pub fn set_layout(&mut self, layout: vk::ImageLayout) {
        gpu::backend_render_targets::set_vk_image_layout(&mut self.backend_render_target, to_skia_layout(layout));
    }

    /// The layout Skia last left the image in (e.g. after [`Self::render`]).
    pub fn current_layout(&self) -> vk::ImageLayout {
        let info = gpu::backend_render_targets::get_vk_image_info(&self.backend_render_target)
            .expect("backend render target should still describe a valid Vulkan image");
        from_skia_layout(info.layout)
    }

    /// Clears the subject view, lets `draw` paint on top of it directly (no
    /// thread hop: called in-place, on whatever thread calls `render`, for
    /// lowest latency), then submits the Vulkan work. Deliberately does *not*
    /// wait for the GPU (`SyncCpu::No`): this app's own subsequent commands on
    /// the offscreen image are submitted to the same Vulkan queue, so their
    /// pipeline barrier (see `gfx::render_subject_frame`) is enough to order
    /// against Skia's writes without a CPU-side stall every frame.
    pub fn render(&mut self, clear_color: Color4f, draw: impl FnOnce(&Canvas)) {
        let canvas = self.surface.canvas();
        canvas.clear(clear_color);
        draw(canvas);

        self.context.flush_and_submit_surface(&mut self.surface, None);
    }
}

fn to_skia_layout(layout: vk::ImageLayout) -> skia_vk::ImageLayout {
    match layout {
        vk::ImageLayout::UNDEFINED => skia_vk::ImageLayout::UNDEFINED,
        vk::ImageLayout::GENERAL => skia_vk::ImageLayout::GENERAL,
        vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL => skia_vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        vk::ImageLayout::TRANSFER_SRC_OPTIMAL => skia_vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL => skia_vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL => skia_vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        other => unimplemented!("unhandled vk::ImageLayout {other:?} in Skia interop"),
    }
}

fn from_skia_layout(layout: skia_vk::ImageLayout) -> vk::ImageLayout {
    match layout {
        skia_vk::ImageLayout::UNDEFINED => vk::ImageLayout::UNDEFINED,
        skia_vk::ImageLayout::GENERAL => vk::ImageLayout::GENERAL,
        skia_vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL => vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        skia_vk::ImageLayout::TRANSFER_SRC_OPTIMAL => vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
        skia_vk::ImageLayout::TRANSFER_DST_OPTIMAL => vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        skia_vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL => vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        other => unimplemented!("unhandled skia_vk::ImageLayout {other:?} in Skia interop"),
    }
}
