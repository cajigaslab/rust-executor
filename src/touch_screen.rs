use std::sync::{Arc, Mutex};

use crate::analog::PointSubscription;
use crate::behavior_task::TaskContext;
use crate::pb::thalamus_grpc::NodeSelector;

const TOUCH_SCREEN_NODE_TYPE: &str = "TOUCH_SCREEN";

/// The subject window's current position (top-left of its content area, in
/// physical screen pixels), kept up to date by the graphics/render side (see
/// `gfx::App::about_to_wait`) so touch points here can be translated from
/// screen coordinates to subject-window-local coordinates.
pub type SharedWindowPosition = Arc<Mutex<(i32, i32)>>;

pub fn shared_window_position() -> SharedWindowPosition {
  Arc::new(Mutex::new((0, 0)))
}

/// The subject window's current inner size, in physical pixels — kept up to
/// date by `gfx::App::about_to_wait` (polled every frame, same as
/// [`SharedWindowPosition`]). Used by `eye_tracking::factory` (gaze reports
/// are relative to screen center, so it needs the window's current size to
/// find that center); [`factory`] doesn't need it, since touch points are
/// already in the same window-local physical-pixel space `BehaviorTask::render`
/// draws into (the offscreen targets track the subject window's actual size —
/// see `gfx::Graphics::resize_offscreen_targets_if_needed`).
pub type SharedWindowSize = Arc<Mutex<(u32, u32)>>;

pub fn shared_window_size() -> SharedWindowSize {
  Arc::new(Mutex::new((crate::canvas::WIDTH, crate::canvas::HEIGHT)))
}

/// Cap on how many points a [`PointRingBuffer`] (touch or gaze trace) holds
/// at once. Also the size `gfx::dot_pipeline`'s per-trace GPU vertex buffers
/// are allocated at, since the two must agree.
pub const TRACE_CAPACITY: usize = 3600;

/// A fixed-capacity ring buffer of points for the touch/gaze traces:
/// `push` past `TRACE_CAPACITY` overwrites the oldest slot rather than
/// growing forever, so both memory use and the per-frame rendering cost of
/// drawing the trace stay bounded regardless of how long a trial (or a
/// forgotten "Clear") runs. Order doesn't matter for rendering — each point
/// is an independent dot — so [`Self::points`] just returns however many
/// slots are currently valid, in storage order rather than oldest/newest.
#[derive(Clone)]
pub struct PointRingBuffer {
  points: [(i32, i32); TRACE_CAPACITY],
  next: usize,
  len: usize,
}

impl PointRingBuffer {
  pub fn new() -> Self {
    Self {
      points: [(0, 0); TRACE_CAPACITY],
      next: 0,
      len: 0,
    }
  }

  pub fn push(&mut self, point: (i32, i32)) {
    self.points[self.next] = point;
    self.next = (self.next + 1) % TRACE_CAPACITY;
    self.len = (self.len + 1).min(TRACE_CAPACITY);
  }

  pub fn clear(&mut self) {
    self.next = 0;
    self.len = 0;
  }

  pub fn points(&self) -> &[(i32, i32)] {
    &self.points[..self.len]
  }
}

/// Touch points received since the last periodic clear (see
/// `gfx::Graphics::render_frame`, which clears it once a second), in
/// subject-window-local coordinates. Rendered as a path of dots in the
/// operator view only.
pub type SharedTouchPath = Arc<Mutex<PointRingBuffer>>;

pub fn shared_touch_path() -> SharedTouchPath {
  Arc::new(Mutex::new(PointRingBuffer::new()))
}

/// Builds the touch feed `TaskContext::subscribe_to_touch` hands back once
/// registered via `TaskContext::set_touch_factory` (see `main::run_grpc`):
/// a direct `TOUCH_SCREEN` subscription (see `Connection::subscribe_points`,
/// reached through `context.connect` so every caller shares the one
/// underlying stream) whose points are translated from screen coordinates
/// to window-local physical pixels — the same space `BehaviorTask::render`
/// draws into, so no further rescaling is needed (see [`SharedWindowSize`]'s
/// doc comment, which explains why gaze needs the window size and touch
/// doesn't).
pub fn factory(window_position: SharedWindowPosition) -> impl Fn(&TaskContext) -> Arc<PointSubscription> {
  move |context: &TaskContext| {
    let connection = context.connect(
      NodeSelector {
        name: String::new(),
        r#type: TOUCH_SCREEN_NODE_TYPE.to_string(),
      },
      vec!["X".to_string(), "Y".to_string()],
    );
    let window_position = window_position.clone();
    connection.subscribe_points(
      "X",
      "Y",
      Some(move |x: f64, y: f64| {
        let (window_x, window_y) = *window_position.lock().unwrap();
        (x - window_x as f64, y - window_y as f64)
      }),
    )
  }
}

/// Drains `context`'s own touch feed (`TaskContext::subscribe_to_touch`,
/// itself built from [`factory`], merged with anything injected via
/// `TaskContext::inject_touch`) forever, feeding `touch_path` for the
/// operator view's touch overlay.
pub async fn run_overlay(context: Arc<TaskContext>, touch_path: SharedTouchPath) -> anyhow::Result<()> {
  let subscription = context.subscribe_to_touch();
  loop {
    let notified = subscription.notify().notified();
    for (x, y) in subscription.drain() {
      touch_path
        .lock()
        .unwrap()
        .push((x.round() as i32, y.round() as i32));
    }
    notified.await;
  }
}
