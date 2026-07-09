mod base;
mod dot_pipeline;
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
use skia_safe::{Color4f, Picture, PictureRecorder, Rect};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::{ElementState, MouseButton, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, ModifiersState, PhysicalKey};
use winit::window::{Fullscreen, WindowAttributes, WindowId};

use crate::behavior_task::{SharedTask, TaskContext, Window};
use crate::eye_tracking::SharedGazePath;
use crate::task_controller::SharedTrialCounter;
use crate::touch_screen::{SharedTouchPath, SharedWindowPosition, SharedWindowSize};
use base::VulkanBase;
use dot_pipeline::{DotPipeline, DotTarget};
use offscreen::OffscreenTarget;
use skia_offscreen::SkiaOffscreen;
use window_target::{MAX_FRAMES_IN_FLIGHT, SwapchainWindow};

/// Background color for both views, so a blank scene reads as blank in both
/// places.
const BLANK_CLEAR_COLOR: [f32; 4] = [0.05, 0.05, 0.07, 1.0];

/// Initial resolution of both offscreen render targets — backed by
/// `crate::canvas`, the fixed canvas space touch/gaze input is rescaled
/// into. Neither stays at this size past the first frame — see
/// `Graphics::resize_offscreen_targets_if_needed` — since both are kept in
/// sync with the subject window's actual size instead.
const OFFSCREEN_EXTENT: vk::Extent2D = vk::Extent2D {
  width: crate::canvas::WIDTH,
  height: crate::canvas::HEIGHT,
};

/// Upper bound on the render loop's throughput. Without this, `ControlFlow::Poll`
/// drives the loop as fast as the GPU/CPU allow, burning a full core for no
/// visible benefit past what either window can present.
const MAX_FPS: u32 = 1000;
const MIN_FRAME_INTERVAL: Duration = Duration::from_nanos(1_000_000_000 / MAX_FPS as u64);

/// Rate limit for simulated gaze input (see `App::window_event`'s
/// right-click-drag handling): `CursorMoved` events can fire far faster than
/// any real eye tracker samples, so forwarding is throttled to this rate
/// instead of once per event.
const SIMULATED_GAZE_HZ: u32 = 120;
const SIMULATED_GAZE_INTERVAL: Duration =
  Duration::from_nanos(1_000_000_000 / SIMULATED_GAZE_HZ as u64);

/// How often the operator view's touch and gaze paths are wiped (matches
/// Thalamus's own `Canvas.__clear_periodically`).
const TOUCH_PATH_CLEAR_INTERVAL: Duration = Duration::from_secs(60);
/// Color and radius of each dot in the operator view's touch path.
const TOUCH_DOT_COLOR: [f32; 4] = [1.0, 0.0, 0.0, 1.0];
/// Color of each dot in the operator view's gaze path (matches Thalamus's own
/// `AngularScalingConfig.paint`, which fills its gaze path with blue).
const GAZE_DOT_COLOR: [f32; 4] = [0.0, 0.0, 1.0, 1.0];
const DOT_RADIUS: f32 = 6.0;

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
  context: Arc<TaskContext>,
  tokio_handle: tokio::runtime::Handle,
  window_position: SharedWindowPosition,
  window_size: SharedWindowSize,
  touch_path: SharedTouchPath,
  gaze_path: SharedGazePath,
  trial_counter: SharedTrialCounter,
) -> Result<()> {
  let event_loop = EventLoop::new()?;
  event_loop.set_control_flow(ControlFlow::Poll);

  let mut app = App {
    graphics: None,
    current_task,
    context,
    tokio_handle,
    window_position,
    window_size,
    touch_path,
    gaze_path,
    trial_counter,
    result: Ok(()),
    modifiers: ModifiersState::empty(),
    simulated_gaze_active: false,
    last_cursor_pos: None,
    last_simulated_gaze_at: None,
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
  operator_offscreen: OffscreenTarget,
  operator_skia: SkiaOffscreen,
  /// Renders the touch/gaze trace dots directly (bypassing Skia — see
  /// `dot_pipeline`'s doc comment for why) into `operator_offscreen` right
  /// after its Skia flush. Operator-view only — the subject view never
  /// shows the touch/gaze traces.
  dot_pipeline: DotPipeline,
  dot_target_operator: DotTarget,
  imgui: imgui::Context,
  platform: WinitPlatform,
  renderer: Renderer,
  operator_texture_id: imgui::TextureId,
  operator_descriptor_pool: vk::DescriptorPool,
  operator_descriptor_set_layout: vk::DescriptorSetLayout,
  upload_command_pool: vk::CommandPool,
  fps: FpsCounter,
  /// When the last frame was rendered, so `App::about_to_wait` can pace the
  /// loop to [`MAX_FPS`].
  last_frame_at: Instant,
  /// When the operator view's touch and gaze paths were last wiped, so
  /// `render_frame` can clear them every [`TOUCH_PATH_CLEAR_INTERVAL`].
  touch_path_cleared_at: Instant,
  /// The last `SharedTrialCounter` value observed, so `render_frame` can
  /// detect a trial ending (the counter changing) and auto-clear the
  /// touch/gaze traces if `auto_clear` is enabled. A counter (rather than
  /// polling `current_task` for a `Some -> None` transition) so a trial
  /// ending is never missed even if a new one starts before the next frame.
  last_trial_count: u64,
  /// Opacity (0-100) applied to touch/gaze trace dots in both views, set
  /// via the operator UI's "Opacity" slider.
  trace_opacity_percent: f32,
  /// Whether the touch/gaze traces auto-clear when a trial ends, set via
  /// the operator UI's "Auto Clear" checkbox.
  auto_clear: bool,
  /// Whether the touch/gaze traces are drawn at all, set via the operator
  /// UI's "Show Touch"/"Show Gaze" checkboxes.
  show_touch: bool,
  show_gaze: bool,
}

struct App {
  graphics: Option<Graphics>,
  current_task: SharedTask,
  context: Arc<TaskContext>,
  tokio_handle: tokio::runtime::Handle,
  window_position: SharedWindowPosition,
  window_size: SharedWindowSize,
  touch_path: SharedTouchPath,
  gaze_path: SharedGazePath,
  trial_counter: SharedTrialCounter,
  result: Result<()>,
  /// Updated from `WindowEvent::ModifiersChanged` so `Ctrl+F` can be
  /// recognized in `KeyboardInput`, which doesn't carry modifier state
  /// itself.
  modifiers: ModifiersState,
  /// Whether the right mouse button is currently held down on the subject
  /// window. While true, `CursorMoved` events there are forwarded as
  /// simulated gaze samples (see `window_event`), so gaze input can be
  /// exercised — click and drag with the right mouse button — without real
  /// eye-tracking hardware.
  simulated_gaze_active: bool,
  /// The subject window's last known cursor position (physical, window-local
  /// pixels — the same space `CursorMoved` itself reports and
  /// `TaskContext::push_gaze`/`gaze_path` expect), so a
  /// right-click-without-moving-first still has a position to forward
  /// immediately on press.
  last_cursor_pos: Option<(f64, f64)>,
  /// When the last simulated gaze sample was forwarded, so `window_event`
  /// can rate-limit `CursorMoved`-driven forwarding to
  /// [`SIMULATED_GAZE_INTERVAL`] instead of forwarding one sample per
  /// `CursorMoved` event, which can fire far faster than any real eye
  /// tracker samples. Reset to `None` between drags (button released), so a
  /// new drag's first sample always forwards immediately.
  last_simulated_gaze_at: Option<Instant>,
}

impl App {
  fn fail(&mut self, event_loop: &ActiveEventLoop, err: anyhow::Error) {
    self.result = Err(err);
    event_loop.exit();
  }
}

/// Forwards `(x, y)` as a gaze sample exactly like a real OCULOMATIC reading
/// would (see `eye_tracking::run`): pushed to `context` (see
/// `TaskContext::push_gaze`) and appended to `gaze_path` for the operator
/// view's overlay. A free function (rather than an `App` method) so callers
/// already holding a `&mut self.graphics` borrow — see `window_event` — can
/// still call it, since it only needs `context`/`gaze_path`, not all of
/// `self`.
fn forward_simulated_gaze(context: &TaskContext, gaze_path: &SharedGazePath, x: i32, y: i32) {
  context.push_gaze((x, y));
  gaze_path.lock().unwrap().push((x, y));
}

impl ApplicationHandler for App {
  fn resumed(&mut self, event_loop: &ActiveEventLoop) {
    if self.graphics.is_some() {
      return;
    }
    match Graphics::new(event_loop) {
      Ok(graphics) => {
        // Seed the initial position/size so they're not left at their
        // defaults for the first frame or two before
        // `about_to_wait`'s per-frame poll catches up.
        if let Ok(position) = graphics.subject.window.inner_position() {
          *self.window_position.lock().unwrap() = (position.x, position.y);
        }
        let size = graphics.subject.window.inner_size();
        *self.window_size.lock().unwrap() = (size.width, size.height);
        self.graphics = Some(graphics);
      }
      Err(e) => self.fail(event_loop, e),
    }
  }

  fn window_event(
    &mut self,
    event_loop: &ActiveEventLoop,
    window_id: WindowId,
    event: WindowEvent,
  ) {
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
      graphics.platform.handle_event(
        graphics.imgui.io_mut(),
        &graphics.operator.window,
        &full_event,
      );
    }

    if let WindowEvent::ModifiersChanged(modifiers) = &event {
      self.modifiers = modifiers.state();
    }

    if window_id == graphics.subject.window.id() {
      if let WindowEvent::CursorMoved { position, .. } = &event {
        self.last_cursor_pos = Some((position.x, position.y));
      }

      if let WindowEvent::MouseInput {
        state,
        button: MouseButton::Right,
        ..
      } = &event
      {
        self.simulated_gaze_active = *state == ElementState::Pressed;
        // Forces an immediate forward on the very next tick of
        // `about_to_wait`'s continuous sampling below, rather than waiting
        // out whatever's left of a stale interval from an earlier drag.
        self.last_simulated_gaze_at = None;
      }

      if let WindowEvent::KeyboardInput {
        event: key_event, ..
      } = &event
      {
        if key_event.state == ElementState::Pressed && !key_event.repeat {
          match key_event.physical_key {
            PhysicalKey::Code(KeyCode::KeyF) if self.modifiers.control_key() => {
              graphics
                .subject
                .window
                .set_fullscreen(Some(Fullscreen::Borderless(None)));
            }
            PhysicalKey::Code(KeyCode::Escape) => {
              graphics.subject.window.set_fullscreen(None);
            }
            _ => {}
          }
        }
      }
    }
  }

  fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
    let Some(graphics) = &mut self.graphics else {
      return;
    };

    // Continuously resample the cursor position at SIMULATED_GAZE_HZ while
    // the right mouse button is held, rather than only forwarding on
    // `CursorMoved` — a real eye tracker keeps sampling even when gaze isn't
    // moving. Independent of frame pacing below (120 Hz is well under even
    // the 1000 Hz frame cap), so this runs before that early-returns.
    if self.simulated_gaze_active {
      let now = Instant::now();
      let due = match self.last_simulated_gaze_at {
        Some(at) => now.duration_since(at) >= SIMULATED_GAZE_INTERVAL,
        None => true,
      };
      if due {
        if let Some((x, y)) = self.last_cursor_pos {
          self.last_simulated_gaze_at = Some(now);
          forward_simulated_gaze(
            &self.context,
            &self.gaze_path,
            x.round() as i32,
            y.round() as i32,
          );
        }
      }
    }

    let next_frame_at = graphics.last_frame_at + MIN_FRAME_INTERVAL;
    if Instant::now() < next_frame_at {
      event_loop.set_control_flow(ControlFlow::WaitUntil(next_frame_at));
      return;
    }
    event_loop.set_control_flow(ControlFlow::Poll);

    // Polled every frame (rather than relying solely on `Moved`/`Resized`
    // events) since window-manager position/size events around a
    // fullscreen transition can race with the OS's own settling of the
    // window rect, otherwise leaving these stuck at pre-transition values
    // for a stray frame (or, if the racy event is the last one
    // delivered, indefinitely) — which showed up as the touch trace being
    // shifted and scaled wrong (e.g. window decorations disappearing and
    // the window resizing to fill a differently-sized monitor on
    // entering borderless fullscreen).
    if let Ok(position) = graphics.subject.window.inner_position() {
      *self.window_position.lock().unwrap() = (position.x, position.y);
    }
    let size = graphics.subject.window.inner_size();
    *self.window_size.lock().unwrap() = (size.width, size.height);

    graphics.last_frame_at = Instant::now();
    if let Err(e) = graphics.render_frame(
      &self.current_task,
      &self.tokio_handle,
      &self.touch_path,
      &self.gaze_path,
      &self.trial_counter,
    ) {
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
        .with_inner_size(LogicalSize::new(
          OFFSCREEN_EXTENT.width,
          OFFSCREEN_EXTENT.height,
        ))
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

    let offscreen = OffscreenTarget::new(&base, OFFSCREEN_EXTENT)?;
    // Safety: `base` and `offscreen` outlive `skia`, both held in this same
    // `Graphics` struct and dropped only in `Graphics::destroy`.
    let skia = unsafe { SkiaOffscreen::new(&base, &offscreen)? };

    let operator_offscreen = OffscreenTarget::new(&base, OFFSCREEN_EXTENT)?;
    // Safety: same as `skia` above, for `operator_offscreen`.
    let operator_skia = unsafe { SkiaOffscreen::new(&base, &operator_offscreen)? };

    let dot_pipeline = DotPipeline::new(&base)?;
    let dot_target_operator = dot_pipeline.create_target(&base, &operator_offscreen)?;

    let mut imgui = imgui::Context::create();
    imgui.set_ini_filename(None);
    let mut platform = WinitPlatform::new(&mut imgui);
    platform.attach_window(
      imgui.io_mut(),
      &operator.window,
      imgui_winit_support::HiDpiMode::Default,
    );

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

    // A texture the operator's imgui UI can display: the operator-phase
    // offscreen image, rendered separately from the subject view (see
    // `Graphics::render_frame`).
    let operator_descriptor_set_layout = create_vulkan_descriptor_set_layout(&base.device)?;
    let operator_descriptor_pool = create_vulkan_descriptor_pool(&base.device, 1)?;
    let operator_descriptor_set = create_vulkan_descriptor_set(
      &base.device,
      operator_descriptor_set_layout,
      operator_descriptor_pool,
      operator_offscreen.view,
      operator_offscreen.sampler,
    )?;
    let operator_texture_id = renderer.textures().insert(operator_descriptor_set);

    Ok(Self {
      base,
      subject,
      operator,
      offscreen,
      skia,
      operator_offscreen,
      operator_skia,
      dot_pipeline,
      dot_target_operator,
      imgui,
      platform,
      renderer,
      operator_texture_id,
      operator_descriptor_pool,
      operator_descriptor_set_layout,
      upload_command_pool,
      fps: FpsCounter::new(),
      last_frame_at: Instant::now(),
      touch_path_cleared_at: Instant::now(),
      last_trial_count: 0,
      trace_opacity_percent: 100.0,
      auto_clear: false,
      show_touch: true,
      show_gaze: true,
    })
  }

  /// (Re)creates `self.offscreen`/`self.skia`, `self.operator_offscreen`/
  /// `self.operator_skia`, and `self.dot_target_operator` (whose framebuffer
  /// wraps `operator_offscreen`'s view, so it goes stale the moment that's
  /// recreated) at the subject window's current physical size whenever it's
  /// changed, so the two offscreen targets always match the subject window
  /// (and, since both are always resized together here, each other) instead
  /// of a fixed resolution — a per-frame poll (like `App::about_to_wait`'s
  /// position/size tracking) rather than reacting to `WindowEvent::Resized`,
  /// for the same reason: resize events can race with the window manager
  /// still settling the window rect. A no-op most frames, once the sizes
  /// already match.
  fn resize_offscreen_targets_if_needed(&mut self) -> Result<()> {
    let physical_size = self.subject.window.inner_size();
    if physical_size.width == 0 || physical_size.height == 0 {
      // Minimized (or not yet mapped): keep the existing targets rather
      // than creating zero-sized images.
      return Ok(());
    }
    let new_extent = vk::Extent2D {
      width: physical_size.width,
      height: physical_size.height,
    };
    if new_extent == self.offscreen.extent {
      return Ok(());
    }

    // The old offscreen images may still be read by in-flight GPU work (the
    // subject blit / operator imgui sampling from a still-outstanding
    // frame), so make sure the GPU is done with them before destroying —
    // and before overwriting the operator descriptor set below, which the
    // GPU may also still be reading from.
    unsafe { self.base.device.device_wait_idle()? };

    let new_offscreen = OffscreenTarget::new(&self.base, new_extent)?;
    // Safety: same as `Graphics::new`'s construction of `skia` — `base` and
    // `new_offscreen` both outlive it, held in this same `Graphics` struct.
    let new_skia = unsafe { SkiaOffscreen::new(&self.base, &new_offscreen)? };
    let new_operator_offscreen = OffscreenTarget::new(&self.base, new_extent)?;
    let new_operator_skia = unsafe { SkiaOffscreen::new(&self.base, &new_operator_offscreen)? };
    let new_dot_target_operator = self
      .dot_pipeline
      .create_target(&self.base, &new_operator_offscreen)?;

    // The imgui texture the operator UI displays points at the *old*
    // operator offscreen's view/sampler; repoint it at the new one in place
    // (same descriptor set, same `operator_texture_id`) rather than
    // reallocating from `operator_descriptor_pool` (sized for exactly one
    // set) or changing the id the UI code references.
    let descriptor_set = *self
      .renderer
      .textures()
      .get(self.operator_texture_id)
      .expect("operator_texture_id should always resolve to a descriptor set");
    let image_info = [vk::DescriptorImageInfo {
      sampler: new_operator_offscreen.sampler,
      image_view: new_operator_offscreen.view,
      image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
    }];
    let writes = [vk::WriteDescriptorSet::default()
      .dst_set(descriptor_set)
      .dst_binding(0)
      .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
      .image_info(&image_info)];
    unsafe { self.base.device.update_descriptor_sets(&writes, &[]) };

    let old_offscreen = std::mem::replace(&mut self.offscreen, new_offscreen);
    let old_skia = std::mem::replace(&mut self.skia, new_skia);
    let old_operator_offscreen =
      std::mem::replace(&mut self.operator_offscreen, new_operator_offscreen);
    let old_operator_skia = std::mem::replace(&mut self.operator_skia, new_operator_skia);
    let old_dot_target_operator =
      std::mem::replace(&mut self.dot_target_operator, new_dot_target_operator);
    drop(old_skia);
    drop(old_operator_skia);
    // Must destroy before `old_operator_offscreen.destroy`: its framebuffer
    // references `old_operator_offscreen.view`.
    self
      .dot_pipeline
      .destroy_target(&self.base, old_dot_target_operator);
    old_offscreen.destroy(&self.base);
    old_operator_offscreen.destroy(&self.base);

    Ok(())
  }

  fn render_frame(
    &mut self,
    current_task: &SharedTask,
    tokio_handle: &tokio::runtime::Handle,
    touch_path: &SharedTouchPath,
    gaze_path: &SharedGazePath,
    trial_counter: &SharedTrialCounter,
  ) -> Result<()> {
    self.resize_offscreen_targets_if_needed()?;
    self.fps.tick();

    // A trial just ended (see `task_controller::run`, which bumps this
    // counter once per finished trial rather than relying on a
    // Some -> None transition of `current_task`, which could be missed
    // if a new trial started before the next frame).
    let trial_count = trial_counter.load(std::sync::atomic::Ordering::Relaxed);
    if trial_count != self.last_trial_count {
      self.last_trial_count = trial_count;
      if self.auto_clear {
        touch_path.lock().unwrap().clear();
        gaze_path.lock().unwrap().clear();
      }
    }

    // Touch and gaze paths are wiped together on the same clock (see
    // `TOUCH_PATH_CLEAR_INTERVAL`), matching Thalamus's own
    // `Canvas.__clear_periodically`.
    let should_clear = self.touch_path_cleared_at.elapsed() >= TOUCH_PATH_CLEAR_INTERVAL;
    if should_clear {
      self.touch_path_cleared_at = Instant::now();
    }

    let touch_points = {
      let mut points = touch_path.lock().unwrap();
      if should_clear {
        points.clear();
      }
      points.clone()
    };
    let gaze_points = {
      let mut points = gaze_path.lock().unwrap();
      if should_clear {
        points.clear();
      }
      points.clone()
    };

    let no_points: [(i32, i32); 0] = [];
    let touch_points_shown: &[(i32, i32)] = if self.show_touch {
      touch_points.points()
    } else {
      &no_points
    };
    let gaze_points_shown: &[(i32, i32)] = if self.show_gaze {
      gaze_points.points()
    } else {
      &no_points
    };
    let trace_opacity = self.trace_opacity_percent / 100.0;

    let (subject_picture, operator_picture) =
      record_task_pictures(self.offscreen.extent, current_task, tokio_handle)?;

    render_subject_frame(
      &self.base,
      &mut self.subject,
      &self.offscreen,
      &mut self.skia,
      subject_picture,
    )?;

    // Acquired here (rather than right before the render pass below, as
    // before) so its command buffer is available to record the operator
    // offscreen's post-render layout transition ahead of the render pass
    // that samples it.
    let frame = self.operator.begin_frame(&self.base)?;
    let operator_frame_index = self.operator.current_frame();
    render_operator_offscreen(
      &self.base,
      &self.operator_offscreen,
      &mut self.operator_skia,
      operator_picture,
      &self.dot_pipeline,
      &mut self.dot_target_operator,
      operator_frame_index,
      touch_points_shown,
      gaze_points_shown,
      trace_opacity,
      frame.command_buffer,
    );

    self
      .platform
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
        ui.text("Operator view:");

        // Fit the operator image into the remaining space (minus room for the
        // controls below it), preserving its aspect ratio and centering it.
        const CONTROL_ROWS: f32 = 3.0;
        let avail = ui.content_region_avail();
        let avail = [
          avail[0].max(1.0),
          (avail[1] - CONTROL_ROWS * ui.frame_height_with_spacing()).max(1.0),
        ];
        let aspect_ratio = self.operator_offscreen.extent.width as f32
          / self.operator_offscreen.extent.height as f32;
        let mut image_size = [avail[0], avail[0] / aspect_ratio];
        if image_size[1] > avail[1] {
          image_size = [avail[1] * aspect_ratio, avail[1]];
        }

        let offset_x = (avail[0] - image_size[0]) * 0.5;
        if offset_x > 0.0 {
          let [cursor_x, cursor_y] = ui.cursor_pos();
          ui.set_cursor_pos([cursor_x + offset_x, cursor_y]);
        }

        imgui::Image::new(self.operator_texture_id, image_size).build(ui);

        if let Some(task) = current_task.lock().unwrap().as_ref() {
          task.operator_widget(ui);
        }

        if ui.button("Clear") {
          touch_path.lock().unwrap().clear();
          gaze_path.lock().unwrap().clear();
        }
        ui.same_line();
        ui.checkbox("Auto Clear", &mut self.auto_clear);

        ui.slider("Opacity", 0.0f32, 100.0f32, &mut self.trace_opacity_percent);

        ui.checkbox("Show Touch", &mut self.show_touch);
        ui.same_line();
        ui.checkbox("Show Gaze", &mut self.show_gaze);
      });
    self.platform.prepare_render(ui, &self.operator.window);
    let draw_data = self.imgui.render();

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
      self
        .base
        .device
        .destroy_descriptor_pool(self.operator_descriptor_pool, None);
      self
        .base
        .device
        .destroy_descriptor_set_layout(self.operator_descriptor_set_layout, None);
      self
        .base
        .device
        .destroy_command_pool(self.upload_command_pool, None);
    }
    // Must drop before `base.device.destroy_device`: the renderer's and Skia
    // context's Drop impls destroy their own Vulkan objects on this device.
    drop(self.renderer);
    drop(self.skia);
    drop(self.operator_skia);
    // Must drop before `operator_offscreen.destroy`: the dot target's
    // framebuffer references `operator_offscreen.view`.
    self
      .dot_pipeline
      .destroy_target(&self.base, self.dot_target_operator);
    self.offscreen.destroy(&self.base);
    self.operator_offscreen.destroy(&self.base);
    self.dot_pipeline.destroy(&self.base);
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

/// Renders the current task (if any) for both views via `spawn_blocking`,
/// recording each into a `Picture` rather than drawing on Skia's live canvas
/// directly: a borrowed `&Canvas` tied to a frame's surface can't cross the
/// `spawn_blocking` closure's `'static` boundary, but an owned `Picture` can
/// (and is `Send`), so each is replayed onto the real canvas back on this
/// thread afterwards (see `render_subject_frame`/`render_operator_offscreen`).
/// Rendered in this order — subject phase, then operator phase — per
/// [`crate::behavior_task::BehaviorTask`]'s two-phase contract. Both phases
/// are recorded at the same `extent`: the subject and operator offscreen
/// targets are always kept the same size (see
/// `Graphics::resize_offscreen_targets_if_needed`), so `canvas.base_layer_size()`
/// matches whichever target each phase is actually drawn into either way.
fn record_task_pictures(
  extent: vk::Extent2D,
  current_task: &SharedTask,
  tokio_handle: &tokio::runtime::Handle,
) -> Result<(Option<Picture>, Option<Picture>)> {
  let task = current_task.lock().unwrap().clone();
  let (width, height) = (extent.width as f32, extent.height as f32);
  let join = tokio_handle.spawn_blocking(move || {
    let record = |window: Window| {
      let mut recorder = PictureRecorder::new();
      let canvas = recorder.begin_recording(Rect::from_xywh(0.0, 0.0, width, height), false);
      if let Some(task) = task.as_ref() {
        task.render(canvas, window);
      }
      recorder.finish_recording_as_picture(None)
    };
    (record(Window::Subject), record(Window::Operator))
  });
  Ok(tokio_handle.block_on(join)?)
}

/// Draws the given (already-recorded subject-phase) picture into the subject
/// offscreen target via Skia — overlaid with the touch path (red dots) and
/// gaze path (blue dots), same as the operator view — and blits it into the
/// subject window's swapchain image.
fn render_subject_frame(
  base: &VulkanBase,
  subject: &mut SwapchainWindow,
  offscreen: &OffscreenTarget,
  skia: &mut SkiaOffscreen,
  picture: Option<Picture>,
) -> Result<()> {
  let clear_color = Color4f::new(
    BLANK_CLEAR_COLOR[0],
    BLANK_CLEAR_COLOR[1],
    BLANK_CLEAR_COLOR[2],
    BLANK_CLEAR_COLOR[3],
  );

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
      (
        vk::PipelineStageFlags::ALL_COMMANDS,
        vk::AccessFlags::MEMORY_WRITE,
      ),
      (
        vk::PipelineStageFlags::TRANSFER,
        vk::AccessFlags::TRANSFER_READ,
      ),
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
      (
        vk::PipelineStageFlags::TRANSFER,
        vk::AccessFlags::TRANSFER_READ,
      ),
      (
        vk::PipelineStageFlags::FRAGMENT_SHADER,
        vk::AccessFlags::SHADER_READ,
      ),
    );
  }
  // Tell Skia where the image ended up so its next `draw` transitions from
  // the correct actual layout instead of the stale one it last knew about.
  skia.set_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

  subject.end_frame(base, frame.image_index)
}

/// Draws the given (already-recorded operator-phase) picture into the
/// operator offscreen target via Skia, then the touch path (red dots) and
/// gaze path (blue dots) via `dot_pipeline` directly (bypassing Skia — see
/// `dot_pipeline`'s doc comment for why), overlaid on top of it. Transitions
/// the result for sampling by the operator window's imgui UI, which
/// displays it in an `imgui::Image`. Unlike the subject view, there's no
/// swapchain blit here: `command_buffer` must be one already begun (via
/// `SwapchainWindow::begin_frame`) but not yet in a render pass, since both
/// the pipeline barriers and the dot render pass this records need to not be
/// nested inside another render pass.
#[allow(clippy::too_many_arguments)]
fn render_operator_offscreen(
  base: &VulkanBase,
  offscreen: &OffscreenTarget,
  skia: &mut SkiaOffscreen,
  picture: Option<Picture>,
  dot_pipeline: &DotPipeline,
  dot_target: &mut DotTarget,
  frame_index: usize,
  touch_points: &[(i32, i32)],
  gaze_points: &[(i32, i32)],
  trace_opacity: f32,
  command_buffer: vk::CommandBuffer,
) {
  let clear_color = Color4f::new(
    BLANK_CLEAR_COLOR[0],
    BLANK_CLEAR_COLOR[1],
    BLANK_CLEAR_COLOR[2],
    BLANK_CLEAR_COLOR[3],
  );

  skia.render(clear_color, |canvas| {
    if let Some(picture) = picture.as_ref() {
      canvas.draw_picture(picture, None, None);
    }
  });
  let layout_after_skia = skia.current_layout();

  // Pre-dot-pass: land the image in COLOR_ATTACHMENT_OPTIMAL — the dot
  // render pass's expected initial layout (`LOAD_OP_LOAD` preserving Skia's
  // just-drawn content) — from whatever Skia left it in. See the equivalent
  // barrier in `render_subject_frame` for why the src stage/access is
  // deliberately wide.
  offscreen.transition(
    base,
    command_buffer,
    layout_after_skia,
    vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
    (
      vk::PipelineStageFlags::ALL_COMMANDS,
      vk::AccessFlags::MEMORY_WRITE,
    ),
    (
      vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
      vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
    ),
  );

  let touch_color = [
    TOUCH_DOT_COLOR[0],
    TOUCH_DOT_COLOR[1],
    TOUCH_DOT_COLOR[2],
    TOUCH_DOT_COLOR[3] * trace_opacity,
  ];
  let gaze_color = [
    GAZE_DOT_COLOR[0],
    GAZE_DOT_COLOR[1],
    GAZE_DOT_COLOR[2],
    GAZE_DOT_COLOR[3] * trace_opacity,
  ];
  dot_pipeline.draw(
    base,
    dot_target,
    frame_index,
    command_buffer,
    offscreen.extent,
    touch_points,
    touch_color,
    gaze_points,
    gaze_color,
  );

  // The dot render pass's `finalLayout` leaves the image in
  // COLOR_ATTACHMENT_OPTIMAL; transition it the rest of the way for imgui to
  // sample it.
  offscreen.transition(
    base,
    command_buffer,
    vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
    vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
    (
      vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
      vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
    ),
    (
      vk::PipelineStageFlags::FRAGMENT_SHADER,
      vk::AccessFlags::SHADER_READ,
    ),
  );
  skia.set_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
}
