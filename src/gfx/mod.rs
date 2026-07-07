mod base;
mod offscreen;
mod skia_offscreen;
mod window_target;

use anyhow::Result;
use ash::vk;
use gpu_allocator::vulkan::{Allocator, AllocatorCreateDesc};
use imgui_rs_vulkan_renderer::vulkan::{
    create_vulkan_descriptor_pool, create_vulkan_descriptor_set, create_vulkan_descriptor_set_layout,
};
use imgui_rs_vulkan_renderer::{Options, Renderer};
use imgui_winit_support::WinitPlatform;
use skia_safe::{Color4f, PictureRecorder, Rect};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::{WindowAttributes, WindowId};

use crate::behavior_task::SharedTask;
use crate::touch_screen::SharedWindowPosition;
use base::VulkanBase;
use offscreen::OffscreenTarget;
use skia_offscreen::SkiaOffscreen;
use window_target::{SwapchainWindow, MAX_FRAMES_IN_FLIGHT};

/// Background color for both the subject view and its copy in the operator
/// window, so a blank subject scene reads as blank in both places.
const BLANK_CLEAR_COLOR: [f32; 4] = [0.05, 0.05, 0.07, 1.0];

const SUBJECT_EXTENT: vk::Extent2D = vk::Extent2D {
    width: 1280,
    height: 720,
};

/// Tracks the render loop's throughput, updating an averaged reading twice a
/// second rather than every frame (which at an uncapped framerate would just
/// flicker unreadably).
struct FpsCounter {
    frame_count: u32,
    window_start: Instant,
    fps: f32,
}

impl FpsCounter {
    const UPDATE_INTERVAL: Duration = Duration::from_millis(500);

    fn new() -> Self {
        Self {
            frame_count: 0,
            window_start: Instant::now(),
            fps: 0.0,
        }
    }

    fn tick(&mut self) {
        self.frame_count += 1;
        let elapsed = self.window_start.elapsed();
        if elapsed >= Self::UPDATE_INTERVAL {
            self.fps = self.frame_count as f32 / elapsed.as_secs_f32();
            self.frame_count = 0;
            self.window_start = Instant::now();
        }
    }
}

/// Opens the subject and operator windows and runs the render loop until either
/// window is closed. Blocks the calling thread (must be the main thread on most
/// platforms) until then.
pub fn run(
    current_task: SharedTask,
    tokio_handle: tokio::runtime::Handle,
    window_position: SharedWindowPosition,
) -> Result<()> {
    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = App {
        graphics: None,
        current_task,
        tokio_handle,
        window_position,
        result: Ok(()),
    };
    event_loop.run_app(&mut app)?;
    app.result
}

/// Everything created lazily on the first `resumed` callback, torn down
/// explicitly when the app exits.
struct Graphics {
    base: VulkanBase,
    subject: SwapchainWindow,
    operator: SwapchainWindow,
    offscreen: OffscreenTarget,
    skia: SkiaOffscreen,
    imgui: imgui::Context,
    platform: WinitPlatform,
    renderer: Renderer,
    subject_texture_id: imgui::TextureId,
    subject_descriptor_pool: vk::DescriptorPool,
    subject_descriptor_set_layout: vk::DescriptorSetLayout,
    upload_command_pool: vk::CommandPool,
    fps: FpsCounter,
}

struct App {
    graphics: Option<Graphics>,
    current_task: SharedTask,
    tokio_handle: tokio::runtime::Handle,
    window_position: SharedWindowPosition,
    result: Result<()>,
}

impl App {
    fn fail(&mut self, event_loop: &ActiveEventLoop, err: anyhow::Error) {
        self.result = Err(err);
        event_loop.exit();
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.graphics.is_some() {
            return;
        }
        match Graphics::new(event_loop) {
            Ok(graphics) => {
                // Seed the initial position: `Moved`/`Resized` only fire on
                // subsequent changes, not for the window's starting placement.
                if let Ok(position) = graphics.subject.window.inner_position() {
                    *self.window_position.lock().unwrap() = (position.x, position.y);
                }
                self.graphics = Some(graphics);
            }
            Err(e) => self.fail(event_loop, e),
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, window_id: WindowId, event: WindowEvent) {
        let Some(graphics) = &mut self.graphics else {
            return;
        };

        if matches!(event, WindowEvent::CloseRequested)
            && (window_id == graphics.subject.window.id() || window_id == graphics.operator.window.id())
        {
            event_loop.exit();
            return;
        }

        if window_id == graphics.operator.window.id() {
            let full_event: winit::event::Event<()> = winit::event::Event::WindowEvent {
                window_id,
                event: event.clone(),
            };
            graphics
                .platform
                .handle_event(graphics.imgui.io_mut(), &graphics.operator.window, &full_event);
        }

        // Kept up to date here (rather than polled every frame) so
        // `touch_screen::run` can translate touch points from screen
        // coordinates to subject-window-local ones. `Moved`'s own payload is
        // the outer position, so re-query `inner_position` for the content
        // area's actual top-left instead of using it directly; `Resized` is
        // also handled since resizing can shift it too (e.g. from a top/left
        // edge or drag).
        if window_id == graphics.subject.window.id()
            && matches!(event, WindowEvent::Moved(_) | WindowEvent::Resized(_))
        {
            if let Ok(position) = graphics.subject.window.inner_position() {
                *self.window_position.lock().unwrap() = (position.x, position.y);
            }
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        let Some(graphics) = &mut self.graphics else {
            return;
        };
        if let Err(e) = graphics.render_frame(&self.current_task, &self.tokio_handle) {
            self.fail(event_loop, e);
        }
    }

    fn exiting(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(graphics) = self.graphics.take() {
            graphics.destroy();
        }
    }
}

impl Graphics {
    fn new(event_loop: &ActiveEventLoop) -> Result<Self> {
        let subject_window = event_loop.create_window(
            WindowAttributes::default()
                .with_title("Subject")
                .with_inner_size(LogicalSize::new(SUBJECT_EXTENT.width, SUBJECT_EXTENT.height))
                .with_resizable(true),
        )?;
        let operator_window = event_loop.create_window(
            WindowAttributes::default()
                .with_title("Operator")
                .with_inner_size(LogicalSize::new(960u32, 720u32))
                .with_resizable(true),
        )?;

        let (base, mut surfaces) = VulkanBase::new(&[&subject_window, &operator_window])?;
        let operator_surface = surfaces.pop().unwrap();
        let subject_surface = surfaces.pop().unwrap();

        let subject = SwapchainWindow::new(&base, subject_window, subject_surface)?;
        let operator = SwapchainWindow::new(&base, operator_window, operator_surface)?;

        let offscreen = OffscreenTarget::new(&base, SUBJECT_EXTENT)?;
        // Safety: `base` and `offscreen` outlive `skia`, both held in this same
        // `Graphics` struct and dropped only in `Graphics::destroy`.
        let skia = unsafe { SkiaOffscreen::new(&base, &offscreen)? };

        let mut imgui = imgui::Context::create();
        imgui.set_ini_filename(None);
        let mut platform = WinitPlatform::new(&mut imgui);
        platform.attach_window(imgui.io_mut(), &operator.window, imgui_winit_support::HiDpiMode::Default);

        let allocator = Allocator::new(&AllocatorCreateDesc {
            instance: base.instance.clone(),
            device: base.device.clone(),
            physical_device: base.physical_device,
            debug_settings: Default::default(),
            buffer_device_address: false,
            allocation_sizes: Default::default(),
        })?;
        let allocator = Arc::new(Mutex::new(allocator));

        // Only used to upload the imgui font atlas during renderer creation.
        let upload_command_pool = unsafe {
            base.device.create_command_pool(
                &vk::CommandPoolCreateInfo::default().queue_family_index(base.queue_family_index),
                None,
            )?
        };

        let mut renderer = Renderer::with_gpu_allocator(
            allocator,
            base.device.clone(),
            base.queue,
            upload_command_pool,
            operator.render_pass,
            &mut imgui,
            Some(Options {
                in_flight_frames: MAX_FRAMES_IN_FLIGHT,
                ..Default::default()
            }),
        )?;

        // A texture the operator's imgui UI can display: the same offscreen image
        // the subject window presents, so both views always show the same content.
        let subject_descriptor_set_layout = create_vulkan_descriptor_set_layout(&base.device)?;
        let subject_descriptor_pool = create_vulkan_descriptor_pool(&base.device, 1)?;
        let subject_descriptor_set = create_vulkan_descriptor_set(
            &base.device,
            subject_descriptor_set_layout,
            subject_descriptor_pool,
            offscreen.view,
            offscreen.sampler,
        )?;
        let subject_texture_id = renderer.textures().insert(subject_descriptor_set);

        Ok(Self {
            base,
            subject,
            operator,
            offscreen,
            skia,
            imgui,
            platform,
            renderer,
            subject_texture_id,
            subject_descriptor_pool,
            subject_descriptor_set_layout,
            upload_command_pool,
            fps: FpsCounter::new(),
        })
    }

    fn render_frame(&mut self, current_task: &SharedTask, tokio_handle: &tokio::runtime::Handle) -> Result<()> {
        self.fps.tick();
        render_subject_frame(
            &self.base,
            &mut self.subject,
            &self.offscreen,
            &mut self.skia,
            current_task,
            tokio_handle,
        )?;

        self.platform
            .prepare_frame(self.imgui.io_mut(), &self.operator.window)?;
        let ui = self.imgui.new_frame();
        let display_size = ui.io().display_size;
        ui.window("Operator")
            .position([0.0, 0.0], imgui::Condition::Always)
            .size(display_size, imgui::Condition::Always)
            .flags(
                imgui::WindowFlags::NO_DECORATION
                    | imgui::WindowFlags::NO_MOVE
                    | imgui::WindowFlags::NO_SAVED_SETTINGS
                    | imgui::WindowFlags::NO_BRING_TO_FRONT_ON_FOCUS,
            )
            .build(|| {
                ui.text(format!("FPS: {:.0}", self.fps.fps));
                ui.text("Subject view:");

                // Fit the subject image into the remaining space (minus room for the
                // Clear button below it), preserving its aspect ratio and centering it.
                let avail = ui.content_region_avail();
                let avail = [avail[0].max(1.0), (avail[1] - ui.frame_height_with_spacing()).max(1.0)];
                let aspect_ratio =
                    self.offscreen.extent.width as f32 / self.offscreen.extent.height as f32;
                let mut image_size = [avail[0], avail[0] / aspect_ratio];
                if image_size[1] > avail[1] {
                    image_size = [avail[1] * aspect_ratio, avail[1]];
                }

                let offset_x = (avail[0] - image_size[0]) * 0.5;
                if offset_x > 0.0 {
                    let [cursor_x, cursor_y] = ui.cursor_pos();
                    ui.set_cursor_pos([cursor_x + offset_x, cursor_y]);
                }

                imgui::Image::new(self.subject_texture_id, image_size).build(ui);

                if ui.button("Clear") {
                    // Intentionally a no-op for now.
                }
            });
        self.platform.prepare_render(ui, &self.operator.window);
        let draw_data = self.imgui.render();

        let frame = self.operator.begin_frame(&self.base)?;
        unsafe {
            let clear_values = [vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: BLANK_CLEAR_COLOR,
                },
            }];
            self.base.device.cmd_begin_render_pass(
                frame.command_buffer,
                &vk::RenderPassBeginInfo::default()
                    .render_pass(self.operator.render_pass)
                    .framebuffer(frame.framebuffer)
                    .render_area(vk::Rect2D {
                        offset: vk::Offset2D::default(),
                        extent: self.operator.extent,
                    })
                    .clear_values(&clear_values),
                vk::SubpassContents::INLINE,
            );
        }
        self.renderer.cmd_draw(frame.command_buffer, draw_data)?;
        unsafe { self.base.device.cmd_end_render_pass(frame.command_buffer) };

        self.operator.end_frame(&self.base, frame.image_index)?;

        self.subject.window.request_redraw();
        self.operator.window.request_redraw();
        Ok(())
    }

    fn destroy(self) {
        unsafe {
            let _ = self.base.device.device_wait_idle();
            self.base
                .device
                .destroy_descriptor_pool(self.subject_descriptor_pool, None);
            self.base
                .device
                .destroy_descriptor_set_layout(self.subject_descriptor_set_layout, None);
            self.base
                .device
                .destroy_command_pool(self.upload_command_pool, None);
        }
        // Must drop before `base.device.destroy_device`: the renderer's and Skia
        // context's Drop impls destroy their own Vulkan objects on this device.
        drop(self.renderer);
        drop(self.skia);
        self.offscreen.destroy(&self.base);
        self.subject.destroy(&self.base);
        self.operator.destroy(&self.base);
        unsafe {
            self.base.device.destroy_device(None);
            self.base.instance.destroy_instance(None);
        }
    }
}

fn image_subresource_layers() -> vk::ImageSubresourceLayers {
    vk::ImageSubresourceLayers {
        aspect_mask: vk::ImageAspectFlags::COLOR,
        mip_level: 0,
        base_array_layer: 0,
        layer_count: 1,
    }
}

fn image_subresource_range() -> vk::ImageSubresourceRange {
    vk::ImageSubresourceRange {
        aspect_mask: vk::ImageAspectFlags::COLOR,
        base_mip_level: 0,
        level_count: 1,
        base_array_layer: 0,
        layer_count: 1,
    }
}

/// Draws the shared offscreen "subject view" target via Skia (a blank scene, or
/// the growing arc while a "simple" task is running) and blits it into the
/// subject window's swapchain image.
fn render_subject_frame(
    base: &VulkanBase,
    subject: &mut SwapchainWindow,
    offscreen: &OffscreenTarget,
    skia: &mut SkiaOffscreen,
    current_task: &SharedTask,
    tokio_handle: &tokio::runtime::Handle,
) -> Result<()> {
    let clear_color = Color4f::new(
        BLANK_CLEAR_COLOR[0],
        BLANK_CLEAR_COLOR[1],
        BLANK_CLEAR_COLOR[2],
        BLANK_CLEAR_COLOR[3],
    );

    // The current task (if any) is rendered via `spawn_blocking`, recording
    // into a `Picture` rather than drawing on Skia's live canvas directly: a
    // borrowed `&Canvas` tied to this frame's surface can't cross the
    // `spawn_blocking` closure's `'static` boundary, but an owned `Picture`
    // can (and is `Send`), so we replay it onto the real canvas back on this
    // thread afterwards.
    let task = current_task.lock().unwrap().clone();
    let (width, height) = (offscreen.extent.width as f32, offscreen.extent.height as f32);
    let join = tokio_handle.spawn_blocking(move || {
        let mut recorder = PictureRecorder::new();
        let canvas = recorder.begin_recording(Rect::from_xywh(0.0, 0.0, width, height), false);
        if let Some(task) = task.as_ref() {
            task.render(canvas);
        }
        recorder.finish_recording_as_picture(None)
    });
    let picture = tokio_handle.block_on(join)?;

    skia.render(clear_color, |canvas| {
        if let Some(picture) = picture.as_ref() {
            canvas.draw_picture(picture, None, None);
        }
    });
    let layout_after_skia = skia.current_layout();

    let frame = subject.begin_frame(base)?;
    let cb = frame.command_buffer;
    let swapchain_image = subject.image(frame.image_index);

    unsafe {
        // Skia's draw wasn't CPU-waited (no `SyncCpu`), so unlike before there
        // may still be in-flight GPU work to synchronize against here; same-queue
        // submission order plus this barrier (from a deliberately wide src
        // stage/access, since we don't know Skia's internal pipeline stages)
        // is what makes that safe without a CPU stall.
        offscreen.transition(
            base,
            cb,
            layout_after_skia,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            (vk::PipelineStageFlags::ALL_COMMANDS, vk::AccessFlags::MEMORY_WRITE),
            (vk::PipelineStageFlags::TRANSFER, vk::AccessFlags::TRANSFER_READ),
        );

        let to_transfer_dst = vk::ImageMemoryBarrier::default()
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(swapchain_image)
            .subresource_range(image_subresource_range())
            .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE);
        base.device.cmd_pipeline_barrier(
            cb,
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[to_transfer_dst],
        );

        let blit = vk::ImageBlit {
            src_subresource: image_subresource_layers(),
            src_offsets: [
                vk::Offset3D::default(),
                vk::Offset3D {
                    x: offscreen.extent.width as i32,
                    y: offscreen.extent.height as i32,
                    z: 1,
                },
            ],
            dst_subresource: image_subresource_layers(),
            dst_offsets: [
                vk::Offset3D::default(),
                vk::Offset3D {
                    x: subject.extent.width as i32,
                    y: subject.extent.height as i32,
                    z: 1,
                },
            ],
        };
        base.device.cmd_blit_image(
            cb,
            offscreen.image,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            swapchain_image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &[blit],
            vk::Filter::LINEAR,
        );

        let to_present = vk::ImageMemoryBarrier::default()
            .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(swapchain_image)
            .subresource_range(image_subresource_range())
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE);
        base.device.cmd_pipeline_barrier(
            cb,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::BOTTOM_OF_PIPE,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[to_present],
        );

        offscreen.transition(
            base,
            cb,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            (vk::PipelineStageFlags::TRANSFER, vk::AccessFlags::TRANSFER_READ),
            (vk::PipelineStageFlags::FRAGMENT_SHADER, vk::AccessFlags::SHADER_READ),
        );
    }
    // Tell Skia where the image ended up so its next `draw` transitions from
    // the correct actual layout instead of the stale one it last knew about.
    skia.set_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

    subject.end_frame(base, frame.image_index)
}
