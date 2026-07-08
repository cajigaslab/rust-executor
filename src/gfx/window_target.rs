use anyhow::{Context, Result};
use ash::vk;

use crate::gfx::base::VulkanBase;

/// Number of frames that can be in flight (recorded/submitted) at once per
/// window, letting the CPU record frame N+1 while the GPU is still working on
/// frame N instead of fully serializing on a single command buffer.
pub const MAX_FRAMES_IN_FLIGHT: usize = 2;

/// Per-frame-in-flight resources: one command buffer and its own semaphore/
/// fence, so up to `MAX_FRAMES_IN_FLIGHT` of these can be outstanding at once.
/// `render_finished` isn't here — see `SwapchainWindow::render_finished`.
struct FrameSync {
  command_buffer: vk::CommandBuffer,
  image_available: vk::Semaphore,
  in_flight: vk::Fence,
}

/// A window with its own swapchain, render pass and `MAX_FRAMES_IN_FLIGHT` sets
/// of per-frame sync objects, so the CPU can stay ahead of the GPU by that many
/// frames instead of fully serializing every frame on one command buffer.
pub struct SwapchainWindow {
  pub window: winit::window::Window,
  surface: vk::SurfaceKHR,
  format: vk::Format,
  pub extent: vk::Extent2D,
  swapchain_loader: ash::khr::swapchain::Device,
  swapchain: vk::SwapchainKHR,
  images: Vec<vk::Image>,
  image_views: Vec<vk::ImageView>,
  pub render_pass: vk::RenderPass,
  framebuffers: Vec<vk::Framebuffer>,
  command_pool: vk::CommandPool,
  frames: Vec<FrameSync>,
  current_frame: usize,
  /// The fence (if any) of whichever frame-in-flight last acquired each
  /// swapchain image, so we don't start writing to an image a differently-
  /// indexed frame-in-flight slot is still using (images and frames-in-flight
  /// aren't the same count, so acquisition order isn't 1:1 with `frames`).
  images_in_flight: Vec<vk::Fence>,
  /// One semaphore per swapchain image (not per frame-in-flight!), signaled
  /// by `end_frame`'s submit and waited on by its present, indexed by
  /// `image_index`. A binary semaphore must be unsignaled when a submit
  /// signals it again, but with `MAX_FRAMES_IN_FLIGHT` != the swapchain's
  /// image count (and more so under `PresentModeKHR::IMMEDIATE`, which
  /// doesn't serialize presents the way FIFO does), a frame-in-flight-indexed
  /// semaphore can still be pending presentation when a *different* image's
  /// submit tries to reuse it — hence indexing by image instead, per
  /// <https://docs.vulkan.org/guide/latest/swapchain_semaphore_reuse.html>.
  render_finished: Vec<vk::Semaphore>,
}

/// The state needed to record and later submit/present one frame.
pub struct FrameCtx {
  pub image_index: u32,
  pub command_buffer: vk::CommandBuffer,
  pub framebuffer: vk::Framebuffer,
}

impl SwapchainWindow {
  pub fn new(
    base: &VulkanBase,
    window: winit::window::Window,
    surface: vk::SurfaceKHR,
  ) -> Result<Self> {
    let swapchain_loader = ash::khr::swapchain::Device::new(&base.instance, &base.device);
    let format = Self::choose_format(base, surface)?;
    let render_pass = Self::create_render_pass(base, format)?;

    let command_pool = unsafe {
      base.device.create_command_pool(
        &vk::CommandPoolCreateInfo::default()
          .queue_family_index(base.queue_family_index)
          .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
        None,
      )?
    };
    let command_buffers = unsafe {
      base.device.allocate_command_buffers(
        &vk::CommandBufferAllocateInfo::default()
          .command_pool(command_pool)
          .level(vk::CommandBufferLevel::PRIMARY)
          .command_buffer_count(MAX_FRAMES_IN_FLIGHT as u32),
      )?
    };

    let frames = command_buffers
      .into_iter()
      .map(|command_buffer| -> Result<FrameSync> {
        Ok(FrameSync {
          command_buffer,
          image_available: unsafe {
            base
              .device
              .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)?
          },
          in_flight: unsafe {
            base.device.create_fence(
              &vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED),
              None,
            )?
          },
        })
      })
      .collect::<Result<Vec<_>>>()?;

    let mut target = Self {
      window,
      surface,
      format,
      extent: vk::Extent2D::default(),
      swapchain_loader,
      swapchain: vk::SwapchainKHR::null(),
      images: Vec::new(),
      image_views: Vec::new(),
      render_pass,
      framebuffers: Vec::new(),
      command_pool,
      frames,
      current_frame: 0,
      images_in_flight: Vec::new(),
      render_finished: Vec::new(),
    };
    target.recreate(base)?;
    Ok(target)
  }

  fn choose_format(base: &VulkanBase, surface: vk::SurfaceKHR) -> Result<vk::Format> {
    let formats = unsafe {
      base
        .surface_loader
        .get_physical_device_surface_formats(base.physical_device, surface)?
    };
    let chosen = formats
      .iter()
      .find(|f| {
        f.format == vk::Format::B8G8R8A8_UNORM && f.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
      })
      .or_else(|| formats.first())
      .context("surface has no formats")?;
    Ok(chosen.format)
  }

  fn create_render_pass(base: &VulkanBase, format: vk::Format) -> Result<vk::RenderPass> {
    let attachments = [vk::AttachmentDescription::default()
      .format(format)
      .samples(vk::SampleCountFlags::TYPE_1)
      .load_op(vk::AttachmentLoadOp::CLEAR)
      .store_op(vk::AttachmentStoreOp::STORE)
      .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
      .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
      .initial_layout(vk::ImageLayout::UNDEFINED)
      .final_layout(vk::ImageLayout::PRESENT_SRC_KHR)];
    let color_refs = [vk::AttachmentReference::default()
      .attachment(0)
      .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)];
    let subpasses = [vk::SubpassDescription::default()
      .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
      .color_attachments(&color_refs)];
    let dependencies = [vk::SubpassDependency::default()
      .src_subpass(vk::SUBPASS_EXTERNAL)
      .dst_subpass(0)
      .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
      .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
      .src_access_mask(vk::AccessFlags::empty())
      .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)];

    let create_info = vk::RenderPassCreateInfo::default()
      .attachments(&attachments)
      .subpasses(&subpasses)
      .dependencies(&dependencies);
    Ok(unsafe { base.device.create_render_pass(&create_info, None)? })
  }

  /// (Re)creates the swapchain and everything that depends on its images/extent.
  /// Must be called once up front and again whenever the window is resized or
  /// the swapchain is reported out of date/suboptimal.
  pub fn recreate(&mut self, base: &VulkanBase) -> Result<()> {
    unsafe { base.device.device_wait_idle()? };

    let capabilities = unsafe {
      base
        .surface_loader
        .get_physical_device_surface_capabilities(base.physical_device, self.surface)?
    };
    let physical_size = self.window.inner_size();
    let (width, height) = (physical_size.width, physical_size.height);
    let extent = if capabilities.current_extent.width != u32::MAX {
      capabilities.current_extent
    } else {
      vk::Extent2D {
        width: width.clamp(
          capabilities.min_image_extent.width,
          capabilities.max_image_extent.width,
        ),
        height: height.clamp(
          capabilities.min_image_extent.height,
          capabilities.max_image_extent.height,
        ),
      }
    };

    let mut image_count = capabilities.min_image_count + 1;
    if capabilities.max_image_count > 0 {
      image_count = image_count.min(capabilities.max_image_count);
    }

    // IMMEDIATE gives the lowest possible latency (presents as soon as a
    // frame is done, no queueing, at the cost of possible tearing). It's
    // optional in the spec (unlike FIFO), so fall back if unsupported.
    let present_modes = unsafe {
      base
        .surface_loader
        .get_physical_device_surface_present_modes(base.physical_device, self.surface)?
    };
    let present_mode = if present_modes.contains(&vk::PresentModeKHR::IMMEDIATE) {
      vk::PresentModeKHR::IMMEDIATE
    } else {
      vk::PresentModeKHR::FIFO
    };

    let old_swapchain = self.swapchain;
    let create_info = vk::SwapchainCreateInfoKHR::default()
      .surface(self.surface)
      .min_image_count(image_count)
      .image_format(self.format)
      .image_color_space(vk::ColorSpaceKHR::SRGB_NONLINEAR)
      .image_extent(extent)
      .image_array_layers(1)
      .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_DST)
      .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
      .pre_transform(capabilities.current_transform)
      .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
      .present_mode(present_mode)
      .clipped(true)
      .old_swapchain(old_swapchain);

    let swapchain = unsafe { self.swapchain_loader.create_swapchain(&create_info, None)? };

    self.destroy_swapchain_dependents(base);
    if old_swapchain != vk::SwapchainKHR::null() {
      unsafe { self.swapchain_loader.destroy_swapchain(old_swapchain, None) };
    }

    self.swapchain = swapchain;
    self.extent = extent;

    self.images = unsafe { self.swapchain_loader.get_swapchain_images(swapchain)? };
    self.image_views = self
      .images
      .iter()
      .map(|image| unsafe {
        base.device.create_image_view(
          &vk::ImageViewCreateInfo::default()
            .image(*image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(self.format)
            .subresource_range(vk::ImageSubresourceRange {
              aspect_mask: vk::ImageAspectFlags::COLOR,
              base_mip_level: 0,
              level_count: 1,
              base_array_layer: 0,
              layer_count: 1,
            }),
          None,
        )
      })
      .collect::<Result<_, _>>()?;

    self.framebuffers = self
      .image_views
      .iter()
      .map(|view| unsafe {
        let attachments = [*view];
        base.device.create_framebuffer(
          &vk::FramebufferCreateInfo::default()
            .render_pass(self.render_pass)
            .attachments(&attachments)
            .width(extent.width)
            .height(extent.height)
            .layers(1),
          None,
        )
      })
      .collect::<Result<_, _>>()?;

    self.images_in_flight = vec![vk::Fence::null(); self.images.len()];

    // One per swapchain image — see `Self::render_finished`'s doc comment
    // for why this can't just be per-frame-in-flight like `image_available`.
    self.render_finished = (0..self.images.len())
      .map(|_| unsafe {
        base
          .device
          .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)
      })
      .collect::<Result<_, _>>()?;

    Ok(())
  }

  fn destroy_swapchain_dependents(&mut self, base: &VulkanBase) {
    for semaphore in self.render_finished.drain(..) {
      unsafe { base.device.destroy_semaphore(semaphore, None) };
    }
    for framebuffer in self.framebuffers.drain(..) {
      unsafe { base.device.destroy_framebuffer(framebuffer, None) };
    }
    for view in self.image_views.drain(..) {
      unsafe { base.device.destroy_image_view(view, None) };
    }
    self.images.clear();
  }

  /// The raw swapchain image at `index` (as returned in [`FrameCtx::image_index`]).
  pub fn image(&self, index: u32) -> vk::Image {
    self.images[index as usize]
  }

  /// Waits for this frame-in-flight slot's previous command buffer to be
  /// free, acquires the next swapchain image (recreating the swapchain and
  /// retrying once if it's out of date), and begins recording.
  pub fn begin_frame(&mut self, base: &VulkanBase) -> Result<FrameCtx> {
    let in_flight = self.frames[self.current_frame].in_flight;
    let image_available = self.frames[self.current_frame].image_available;
    let command_buffer = self.frames[self.current_frame].command_buffer;

    unsafe { base.device.wait_for_fences(&[in_flight], true, u64::MAX)? };

    let image_index = loop {
      let result = unsafe {
        self.swapchain_loader.acquire_next_image(
          self.swapchain,
          u64::MAX,
          image_available,
          vk::Fence::null(),
        )
      };
      match result {
        Ok((index, false)) => break index,
        Ok((_, true)) | Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
          self.recreate(base)?;
          continue;
        }
        Err(e) => return Err(e.into()),
      }
    };

    // This image might still be in use by a different frame-in-flight slot
    // than the one we're about to reuse (image count and frames-in-flight
    // count aren't necessarily equal), so wait on it specifically too.
    let image_fence = self.images_in_flight[image_index as usize];
    if image_fence != vk::Fence::null() {
      unsafe {
        base
          .device
          .wait_for_fences(&[image_fence], true, u64::MAX)?
      };
    }
    self.images_in_flight[image_index as usize] = in_flight;

    unsafe {
      base.device.reset_fences(&[in_flight])?;
      base
        .device
        .reset_command_buffer(command_buffer, vk::CommandBufferResetFlags::empty())?;
      base
        .device
        .begin_command_buffer(command_buffer, &vk::CommandBufferBeginInfo::default())?;
    }

    Ok(FrameCtx {
      image_index,
      command_buffer,
      framebuffer: self.framebuffers[image_index as usize],
    })
  }

  /// Ends, submits and presents the frame begun by [`Self::begin_frame`].
  pub fn end_frame(&mut self, base: &VulkanBase, image_index: u32) -> Result<()> {
    let frame = &self.frames[self.current_frame];
    let (command_buffer, image_available, in_flight) =
      (frame.command_buffer, frame.image_available, frame.in_flight);
    // Indexed by the swapchain image, not the frame-in-flight slot — see
    // `Self::render_finished`'s doc comment.
    let render_finished = self.render_finished[image_index as usize];

    unsafe { base.device.end_command_buffer(command_buffer)? };

    let wait_semaphores = [image_available];
    let wait_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
    let command_buffers = [command_buffer];
    let signal_semaphores = [render_finished];
    let submit_info = vk::SubmitInfo::default()
      .wait_semaphores(&wait_semaphores)
      .wait_dst_stage_mask(&wait_stages)
      .command_buffers(&command_buffers)
      .signal_semaphores(&signal_semaphores);

    unsafe {
      base
        .device
        .queue_submit(base.queue, &[submit_info], in_flight)?
    };

    let swapchains = [self.swapchain];
    let image_indices = [image_index];
    let present_info = vk::PresentInfoKHR::default()
      .wait_semaphores(&signal_semaphores)
      .swapchains(&swapchains)
      .image_indices(&image_indices);

    let present_result = unsafe {
      self
        .swapchain_loader
        .queue_present(base.queue, &present_info)
    };
    match present_result {
      Ok(false) => {}
      Ok(true) | Err(vk::Result::ERROR_OUT_OF_DATE_KHR | vk::Result::SUBOPTIMAL_KHR) => {
        self.recreate(base)?;
      }
      Err(e) => return Err(e.into()),
    }

    self.current_frame = (self.current_frame + 1) % self.frames.len();
    Ok(())
  }

  pub fn destroy(mut self, base: &VulkanBase) {
    unsafe {
      let _ = base.device.device_wait_idle();
      self.destroy_swapchain_dependents(base);
      self
        .swapchain_loader
        .destroy_swapchain(self.swapchain, None);
      for frame in &self.frames {
        base.device.destroy_semaphore(frame.image_available, None);
        base.device.destroy_fence(frame.in_flight, None);
      }
      base.device.destroy_command_pool(self.command_pool, None);
      base.device.destroy_render_pass(self.render_pass, None);
      base.surface_loader.destroy_surface(self.surface, None);
    }
  }
}
