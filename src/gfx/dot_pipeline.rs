use std::ffi::{CString, c_void};

use anyhow::{Result, anyhow};
use ash::vk;

use crate::gfx::base::VulkanBase;
use crate::gfx::offscreen::OffscreenTarget;
use crate::gfx::window_target::MAX_FRAMES_IN_FLIGHT;
use crate::touch_screen::TRACE_CAPACITY;

/// WGSL source for the touch/gaze dot pipeline: each point is drawn as an
/// instanced quad (one instance per point; the 6 corner vertices are
/// synthesized from `vertex_index` in `corner_for`, so no vertex buffer is
/// needed for the quad geometry itself — only a per-instance center
/// position), discarded outside its own inscribed circle in the fragment
/// shader so each instance reads as a round dot rather than a square.
/// Compiled to SPIR-V at startup via `naga` (see `compile_shaders`) rather
/// than shipping a shader-compiler toolchain dependency (`glslc`/`shaderc`,
/// which would need a native library like the one that made SDL3 a poor fit
/// earlier in this project) or precompiled `.spv` files.
const SHADER_SOURCE: &str = r#"
struct PushConstants {
  viewport_size: vec2<f32>,
  radius: f32,
  padding: f32,
  color: vec4<f32>,
};

var<push_constant> pc: PushConstants;

struct VertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) uv: vec2<f32>,
};

fn corner_for(i: u32) -> vec2<f32> {
  let x = select(-1.0, 1.0, (i == 1u) || (i == 3u) || (i == 4u));
  let y = select(-1.0, 1.0, (i == 2u) || (i == 4u) || (i == 5u));
  return vec2<f32>(x, y);
}

@vertex
fn vs_main(
  @builtin(vertex_index) vertex_index: u32,
  @location(0) center: vec2<f32>,
) -> VertexOutput {
  let corner = corner_for(vertex_index);
  let pixel_pos = center + corner * pc.radius;
  // Plain Vulkan NDC (Y down), matching pixel space (Y down) directly —
  // see `compile_shaders`, which disables naga's automatic Y-adjustment
  // (meant for WGSL's own assumed coordinate convention) so this is used
  // as written.
  let ndc_x = pixel_pos.x / pc.viewport_size.x * 2.0 - 1.0;
  let ndc_y = pixel_pos.y / pc.viewport_size.y * 2.0 - 1.0;

  var out: VertexOutput;
  out.position = vec4<f32>(ndc_x, ndc_y, 0.0, 1.0);
  out.uv = corner;
  return out;
}

@fragment
fn fs_main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
  if (dot(uv, uv) > 1.0) {
    discard;
  }
  return pc.color;
}
"#;

/// Layout must match `SHADER_SOURCE`'s `PushConstants` struct exactly (WGSL's
/// default struct layout rules happen to match `repr(C)` here: vec2<f32> at
/// offset 0, two f32s at 8/12, vec4<f32> at 16, size 32).
#[repr(C)]
struct PushConstants {
  viewport_size: [f32; 2],
  radius: f32,
  _padding: f32,
  color: [f32; 4],
}

/// Compiles [`SHADER_SOURCE`] to SPIR-V, once per entry point, via `naga`.
fn compile_shaders() -> Result<(Vec<u32>, Vec<u32>)> {
  let module = naga::front::wgsl::parse_str(SHADER_SOURCE).map_err(|e| anyhow!("{e:?}"))?;
  let info = naga::valid::Validator::new(
    naga::valid::ValidationFlags::default(),
    naga::valid::Capabilities::default() | naga::valid::Capabilities::PUSH_CONSTANT,
  )
  .validate(&module)
  .map_err(|e| anyhow!("{e:?}"))?;

  let mut options = naga::back::spv::Options::default();
  // `SHADER_SOURCE` writes `@builtin(position)` directly in plain Vulkan NDC
  // (Y down) itself, rather than the WGSL-assumed convention this flag
  // otherwise corrects for.
  options
    .flags
    .remove(naga::back::spv::WriterFlags::ADJUST_COORDINATE_SPACE);

  let vertex = naga::back::spv::write_vec(
    &module,
    &info,
    &options,
    Some(&naga::back::spv::PipelineOptions {
      shader_stage: naga::ShaderStage::Vertex,
      entry_point: "vs_main".to_string(),
    }),
  )
  .map_err(|e| anyhow!("{e:?}"))?;
  let fragment = naga::back::spv::write_vec(
    &module,
    &info,
    &options,
    Some(&naga::back::spv::PipelineOptions {
      shader_stage: naga::ShaderStage::Fragment,
      entry_point: "fs_main".to_string(),
    }),
  )
  .map_err(|e| anyhow!("{e:?}"))?;

  Ok((vertex, fragment))
}

fn create_shader_module(base: &VulkanBase, words: &[u32]) -> Result<vk::ShaderModule> {
  Ok(unsafe {
    base
      .device
      .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(words), None)?
  })
}

/// One persistently-mapped, host-visible+coherent vertex buffer sized for
/// both traces (touch points at [`Self::TOUCH_OFFSET`], gaze points at
/// [`Self::GAZE_OFFSET`]), reused across frames. One of these lives per
/// frame-in-flight slot in a [`DotTarget`] — see `DotPipeline::draw`'s doc
/// comment for why overwriting the *current* slot's buffer is safe without
/// any fence of its own.
struct VertexBuffer {
  buffer: vk::Buffer,
  memory: vk::DeviceMemory,
  mapped: *mut c_void,
}

impl VertexBuffer {
  const POINT_SIZE: vk::DeviceSize = size_of::<[f32; 2]>() as vk::DeviceSize;
  const TOUCH_OFFSET: vk::DeviceSize = 0;
  const GAZE_OFFSET: vk::DeviceSize = TRACE_CAPACITY as vk::DeviceSize * Self::POINT_SIZE;
  const SIZE: vk::DeviceSize = 2 * TRACE_CAPACITY as vk::DeviceSize * Self::POINT_SIZE;

  fn new(base: &VulkanBase) -> Result<Self> {
    let buffer = unsafe {
      base.device.create_buffer(
        &vk::BufferCreateInfo::default()
          .size(Self::SIZE)
          .usage(vk::BufferUsageFlags::VERTEX_BUFFER)
          .sharing_mode(vk::SharingMode::EXCLUSIVE),
        None,
      )?
    };
    let requirements = unsafe { base.device.get_buffer_memory_requirements(buffer) };
    let memory_type_index = base.find_memory_type_index(
      requirements.memory_type_bits,
      vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    )?;
    let memory = unsafe {
      base.device.allocate_memory(
        &vk::MemoryAllocateInfo::default()
          .allocation_size(requirements.size)
          .memory_type_index(memory_type_index),
        None,
      )?
    };
    unsafe { base.device.bind_buffer_memory(buffer, memory, 0)? };
    let mapped = unsafe {
      base
        .device
        .map_memory(memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())?
    };

    Ok(Self {
      buffer,
      memory,
      mapped,
    })
  }

  /// Overwrites the buffer with `touch_points` (at [`Self::TOUCH_OFFSET`])
  /// and `gaze_points` (at [`Self::GAZE_OFFSET`]), each truncated to
  /// `TRACE_CAPACITY` points (matching `PointRingBuffer`'s own cap, so this
  /// never actually truncates in practice).
  ///
  /// # Safety
  /// The GPU must not still be reading this buffer from a previous frame —
  /// see this type's doc comment for why that's guaranteed by the caller.
  unsafe fn write(&mut self, touch_points: &[(i32, i32)], gaze_points: &[(i32, i32)]) {
    unsafe {
      Self::write_at(self.mapped, Self::TOUCH_OFFSET, touch_points);
      Self::write_at(self.mapped, Self::GAZE_OFFSET, gaze_points);
    }
  }

  unsafe fn write_at(mapped: *mut c_void, offset: vk::DeviceSize, points: &[(i32, i32)]) {
    let points = &points[..points.len().min(TRACE_CAPACITY)];
    let dst = unsafe { (mapped as *mut u8).add(offset as usize) as *mut [f32; 2] };
    for (i, &(x, y)) in points.iter().enumerate() {
      unsafe { dst.add(i).write([x as f32, y as f32]) };
    }
  }

  fn destroy(self, base: &VulkanBase) {
    unsafe {
      base.device.unmap_memory(self.memory);
      base.device.destroy_buffer(self.buffer, None);
      base.device.free_memory(self.memory, None);
    }
  }
}

/// Per-offscreen-target resources for the dot pipeline: a framebuffer
/// wrapping the offscreen image (rebuilt whenever the offscreen target
/// itself is recreated — see `gfx::Graphics::resize_offscreen_targets_if_needed`)
/// plus one [`VertexBuffer`] per frame-in-flight slot of whichever
/// `SwapchainWindow` this target's dot pass shares a command buffer with
/// (subject or operator) — see `SwapchainWindow::current_frame`.
pub struct DotTarget {
  framebuffer: vk::Framebuffer,
  vertex_buffers: Vec<VertexBuffer>,
}

impl DotTarget {
  fn new(
    base: &VulkanBase,
    render_pass: vk::RenderPass,
    offscreen: &OffscreenTarget,
  ) -> Result<Self> {
    let attachments = [offscreen.view];
    let framebuffer = unsafe {
      base.device.create_framebuffer(
        &vk::FramebufferCreateInfo::default()
          .render_pass(render_pass)
          .attachments(&attachments)
          .width(offscreen.extent.width)
          .height(offscreen.extent.height)
          .layers(1),
        None,
      )?
    };

    let vertex_buffers = (0..MAX_FRAMES_IN_FLIGHT)
      .map(|_| VertexBuffer::new(base))
      .collect::<Result<Vec<_>>>()?;

    Ok(Self {
      framebuffer,
      vertex_buffers,
    })
  }

  fn destroy(self, base: &VulkanBase) {
    unsafe { base.device.destroy_framebuffer(self.framebuffer, None) };
    for buffer in self.vertex_buffers {
      buffer.destroy(base);
    }
  }
}

/// The touch/gaze dot rendering pipeline: a render pass (drawn into an
/// offscreen image right after Skia's own flush, preserving its content via
/// `LOAD_OP_LOAD`) plus the graphics pipeline that draws each trace as
/// instanced quads, replacing the old approach of rebuilding a Skia `Path`
/// with one `add_circle` per point every frame — which got proportionally
/// slower as a trace grew, unlike this pipeline's small, constant per-frame
/// cost (bounded by `TRACE_CAPACITY`). Shared by the subject and operator
/// offscreen targets — see [`DotTarget`] for the per-target resources this
/// pipeline is used with.
pub struct DotPipeline {
  render_pass: vk::RenderPass,
  pipeline_layout: vk::PipelineLayout,
  pipeline: vk::Pipeline,
}

impl DotPipeline {
  pub fn new(base: &VulkanBase) -> Result<Self> {
    let (vertex_words, fragment_words) = compile_shaders()?;
    let vertex_module = create_shader_module(base, &vertex_words)?;
    let fragment_module = create_shader_module(base, &fragment_words)?;

    let attachments = [vk::AttachmentDescription::default()
      .format(OffscreenTarget::FORMAT)
      .samples(vk::SampleCountFlags::TYPE_1)
      .load_op(vk::AttachmentLoadOp::LOAD)
      .store_op(vk::AttachmentStoreOp::STORE)
      .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
      .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
      .initial_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
      .final_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)];
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
    let render_pass = unsafe {
      base.device.create_render_pass(
        &vk::RenderPassCreateInfo::default()
          .attachments(&attachments)
          .subpasses(&subpasses)
          .dependencies(&dependencies),
        None,
      )?
    };

    let push_constant_ranges = [vk::PushConstantRange::default()
      .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT)
      .offset(0)
      .size(size_of::<PushConstants>() as u32)];
    let pipeline_layout = unsafe {
      base.device.create_pipeline_layout(
        &vk::PipelineLayoutCreateInfo::default().push_constant_ranges(&push_constant_ranges),
        None,
      )?
    };

    let entry_vs = CString::new("vs_main").expect("no interior NUL");
    let entry_fs = CString::new("fs_main").expect("no interior NUL");
    let stages = [
      vk::PipelineShaderStageCreateInfo::default()
        .stage(vk::ShaderStageFlags::VERTEX)
        .module(vertex_module)
        .name(&entry_vs),
      vk::PipelineShaderStageCreateInfo::default()
        .stage(vk::ShaderStageFlags::FRAGMENT)
        .module(fragment_module)
        .name(&entry_fs),
    ];

    let bindings = [vk::VertexInputBindingDescription::default()
      .binding(0)
      .stride(size_of::<[f32; 2]>() as u32)
      .input_rate(vk::VertexInputRate::INSTANCE)];
    let attributes = [vk::VertexInputAttributeDescription::default()
      .location(0)
      .binding(0)
      .format(vk::Format::R32G32_SFLOAT)
      .offset(0)];
    let vertex_input = vk::PipelineVertexInputStateCreateInfo::default()
      .vertex_binding_descriptions(&bindings)
      .vertex_attribute_descriptions(&attributes);

    let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
      .topology(vk::PrimitiveTopology::TRIANGLE_LIST);

    let viewport_state = vk::PipelineViewportStateCreateInfo::default()
      .viewport_count(1)
      .scissor_count(1);

    let rasterization = vk::PipelineRasterizationStateCreateInfo::default()
      .polygon_mode(vk::PolygonMode::FILL)
      .cull_mode(vk::CullModeFlags::NONE)
      .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
      .line_width(1.0);

    let multisample = vk::PipelineMultisampleStateCreateInfo::default()
      .rasterization_samples(vk::SampleCountFlags::TYPE_1);

    let color_blend_attachments = [vk::PipelineColorBlendAttachmentState::default()
      .blend_enable(true)
      .src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
      .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
      .color_blend_op(vk::BlendOp::ADD)
      .src_alpha_blend_factor(vk::BlendFactor::ONE)
      .dst_alpha_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
      .alpha_blend_op(vk::BlendOp::ADD)
      .color_write_mask(vk::ColorComponentFlags::RGBA)];
    let color_blend =
      vk::PipelineColorBlendStateCreateInfo::default().attachments(&color_blend_attachments);

    let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
    let dynamic_state =
      vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);

    let pipeline_create_info = vk::GraphicsPipelineCreateInfo::default()
      .stages(&stages)
      .vertex_input_state(&vertex_input)
      .input_assembly_state(&input_assembly)
      .viewport_state(&viewport_state)
      .rasterization_state(&rasterization)
      .multisample_state(&multisample)
      .color_blend_state(&color_blend)
      .dynamic_state(&dynamic_state)
      .layout(pipeline_layout)
      .render_pass(render_pass)
      .subpass(0);

    let pipeline = unsafe {
      base
        .device
        .create_graphics_pipelines(vk::PipelineCache::null(), &[pipeline_create_info], None)
        .map_err(|(_, e)| anyhow!("creating dot pipeline: {e}"))?[0]
    };

    unsafe {
      base.device.destroy_shader_module(vertex_module, None);
      base.device.destroy_shader_module(fragment_module, None);
    }

    Ok(Self {
      render_pass,
      pipeline_layout,
      pipeline,
    })
  }

  /// Builds the per-offscreen-target resources this pipeline needs to draw
  /// into `offscreen` — call once up front and again whenever `offscreen` is
  /// recreated (its `view` changes), destroying the old `DotTarget` via
  /// [`Self::destroy_target`] first.
  pub fn create_target(&self, base: &VulkanBase, offscreen: &OffscreenTarget) -> Result<DotTarget> {
    DotTarget::new(base, self.render_pass, offscreen)
  }

  pub fn destroy_target(&self, base: &VulkanBase, target: DotTarget) {
    target.destroy(base);
  }

  /// Records the dot render pass onto `command_buffer`: writes
  /// `target.vertex_buffers[frame_index]` with `touch_points` and
  /// `gaze_points`, then draws each trace as one instanced-quad draw call.
  /// `frame_index` must be whichever window's (subject's or operator's)
  /// `SwapchainWindow::current_frame` this `command_buffer` belongs to —
  /// that window's own `begin_frame` having already waited on that slot's
  /// fence is what makes overwriting this same slot's vertex buffer safe
  /// without a fence of this pipeline's own.
  ///
  /// The offscreen image must already be in `COLOR_ATTACHMENT_OPTIMAL`
  /// layout (matching the render pass's `initialLayout`/`LOAD_OP_LOAD`,
  /// which preserves whatever Skia already drew) and is left in that same
  /// layout afterward (`finalLayout`) — callers transition to/from it
  /// themselves (see `gfx::render_subject_frame`/`render_operator_offscreen`).
  #[allow(clippy::too_many_arguments)]
  pub fn draw(
    &self,
    base: &VulkanBase,
    target: &mut DotTarget,
    frame_index: usize,
    command_buffer: vk::CommandBuffer,
    extent: vk::Extent2D,
    touch_points: &[(i32, i32)],
    touch_color: [f32; 4],
    gaze_points: &[(i32, i32)],
    gaze_color: [f32; 4],
  ) {
    let buffer = &mut target.vertex_buffers[frame_index];
    unsafe { buffer.write(touch_points, gaze_points) };

    unsafe {
      base.device.cmd_begin_render_pass(
        command_buffer,
        &vk::RenderPassBeginInfo::default()
          .render_pass(self.render_pass)
          .framebuffer(target.framebuffer)
          .render_area(vk::Rect2D {
            offset: vk::Offset2D::default(),
            extent,
          }),
        vk::SubpassContents::INLINE,
      );

      base.device.cmd_bind_pipeline(
        command_buffer,
        vk::PipelineBindPoint::GRAPHICS,
        self.pipeline,
      );

      let viewport = vk::Viewport {
        x: 0.0,
        y: 0.0,
        width: extent.width as f32,
        height: extent.height as f32,
        min_depth: 0.0,
        max_depth: 1.0,
      };
      base.device.cmd_set_viewport(command_buffer, 0, &[viewport]);
      let scissor = vk::Rect2D {
        offset: vk::Offset2D::default(),
        extent,
      };
      base.device.cmd_set_scissor(command_buffer, 0, &[scissor]);

      let viewport_size = [extent.width as f32, extent.height as f32];
      self.draw_trace(
        base,
        command_buffer,
        buffer.buffer,
        VertexBuffer::TOUCH_OFFSET,
        touch_points.len().min(TRACE_CAPACITY) as u32,
        viewport_size,
        touch_color,
      );
      self.draw_trace(
        base,
        command_buffer,
        buffer.buffer,
        VertexBuffer::GAZE_OFFSET,
        gaze_points.len().min(TRACE_CAPACITY) as u32,
        viewport_size,
        gaze_color,
      );

      base.device.cmd_end_render_pass(command_buffer);
    }
  }

  #[allow(clippy::too_many_arguments)]
  fn draw_trace(
    &self,
    base: &VulkanBase,
    command_buffer: vk::CommandBuffer,
    buffer: vk::Buffer,
    offset: vk::DeviceSize,
    count: u32,
    viewport_size: [f32; 2],
    color: [f32; 4],
  ) {
    if count == 0 {
      return;
    }
    let push_constants = PushConstants {
      viewport_size,
      radius: super::DOT_RADIUS,
      _padding: 0.0,
      color,
    };
    unsafe {
      base
        .device
        .cmd_bind_vertex_buffers(command_buffer, 0, &[buffer], &[offset]);
      base.device.cmd_push_constants(
        command_buffer,
        self.pipeline_layout,
        vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
        0,
        std::slice::from_raw_parts(
          (&push_constants as *const PushConstants) as *const u8,
          size_of::<PushConstants>(),
        ),
      );
      base.device.cmd_draw(command_buffer, 6, count, 0, 0);
    }
  }

  pub fn destroy(self, base: &VulkanBase) {
    unsafe {
      base.device.destroy_pipeline(self.pipeline, None);
      base
        .device
        .destroy_pipeline_layout(self.pipeline_layout, None);
      base.device.destroy_render_pass(self.render_pass, None);
    }
  }
}
