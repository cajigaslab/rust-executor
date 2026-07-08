use std::sync::{Arc, Mutex};

use tonic::transport::Channel;

use crate::analog::last_span_value;
use crate::behavior_task::SharedTask;
use crate::pb::thalamus_grpc::thalamus_client::ThalamusClient;
use crate::pb::thalamus_grpc::{AnalogRequest, NodeSelector};

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
/// [`SharedWindowPosition`]). Used by `eye_tracking::run` (gaze reports are
/// relative to screen center, so it needs the window's current size to find
/// that center); `touch_screen::run` doesn't need it, since touch points are
/// already in the same window-local physical-pixel space `BehaviorTask::render`
/// draws into (the offscreen targets track the subject window's actual size —
/// see `gfx::Graphics::resize_offscreen_targets_if_needed`).
pub type SharedWindowSize = Arc<Mutex<(u32, u32)>>;

pub fn shared_window_size() -> SharedWindowSize {
  Arc::new(Mutex::new((crate::canvas::WIDTH, crate::canvas::HEIGHT)))
}

/// Touch points received since the last periodic clear (see
/// `gfx::Graphics::render_frame`, which clears it once a second), in
/// subject-window-local coordinates. Rendered as a path of dots in the
/// operator view only.
pub type SharedTouchPath = Arc<Mutex<Vec<(i32, i32)>>>;

pub fn shared_touch_path() -> SharedTouchPath {
  Arc::new(Mutex::new(Vec::new()))
}

/// Streams touch input from Thalamus's `analog` RPC for the TOUCH_SCREEN node
/// over `client` (shared with `eye_tracking::run`, since both just hit the
/// same service's `analog` RPC for different node types) and forwards each
/// touch point — translated from screen coordinates to window-local physical
/// pixels (the same space `BehaviorTask::render` draws into — no further
/// rescaling needed, see [`SharedWindowSize`]'s doc comment) — to whichever
/// `BehaviorTask` is currently running (if any), as well as appending it to
/// `touch_path` for the operator view's touch overlay.
pub async fn run(
  mut client: ThalamusClient<Channel>,
  current_task: SharedTask,
  window_position: SharedWindowPosition,
  touch_path: SharedTouchPath,
) -> anyhow::Result<()> {
  let request = AnalogRequest {
    node: Some(NodeSelector {
      name: String::new(),
      r#type: TOUCH_SCREEN_NODE_TYPE.to_string(),
    }),
    channels: Vec::new(),
    channel_names: Vec::new(),
  };

  tracing::info!("connecting to TOUCH_SCREEN analog stream");
  let response = client.analog(request).await?;
  let mut inbound = response.into_inner();

  while let Some(analog) = inbound.message().await? {
    let (Some(x), Some(y)) = (last_span_value(&analog, "X"), last_span_value(&analog, "Y")) else {
      // Either component is missing from this message: skip it.
      continue;
    };

    let (window_x, window_y) = *window_position.lock().unwrap();
    let canvas_x = (x - window_x as f64).round() as i32;
    let canvas_y = (y - window_y as f64).round() as i32;

    if let Some(task) = current_task.lock().unwrap().as_ref() {
      task.on_touch(canvas_x, canvas_y);
    }
    touch_path.lock().unwrap().push((canvas_x, canvas_y));
  }

  Ok(())
}
