mod analog;
mod behavior_task;
mod canvas;
mod eye_tracking;
mod gfx;
mod monotonic_time;
mod pb;
mod state;
mod task_controller;
mod touch_screen;

use std::sync::Arc;

use behavior_task::{SharedTask, TaskContext};
use kira::{AudioManager, AudioManagerSettings, DefaultBackend};
use pb::thalamus_grpc::thalamus_client::ThalamusClient;
use pb::thalamus_grpc::{ObservableChange, ObservableTransaction};
use serde_json::Value;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

/// Bound on chained `observable_bridge_v2` redirects, to avoid looping forever
/// if two nodes redirect to each other.
const MAX_REDIRECTS: u32 = 8;

fn main() -> anyhow::Result<()> {
  tracing_subscriber::fmt::init();

  let addr = std::env::args()
    .nth(1)
    .unwrap_or_else(|| "http://127.0.0.1:50050".to_string());

  let current_task = behavior_task::shared_task();
  let window_position = touch_screen::shared_window_position();
  let window_size = touch_screen::shared_window_size();
  let touch_path = touch_screen::shared_touch_path();
  let gaze_path = eye_tracking::shared_gaze_path();
  let angular_scaling = eye_tracking::shared_angular_scaling();
  let trial_counter = task_controller::shared_trial_counter();

  // The Thalamus/TaskController gRPC clients run on a background thread with
  // their own Tokio runtime; the windowing/Vulkan render loop below needs the
  // main thread to itself on most platforms. The render loop invokes
  // `BehaviorTask::render` via that runtime's `spawn_blocking` + `block_on`
  // (see `gfx::render_subject_frame`), so the grpc thread sends a `Handle` to
  // it back once it's built.
  let grpc_current_task = current_task.clone();
  let grpc_window_position = window_position.clone();
  let grpc_window_size = window_size.clone();
  let grpc_touch_path = touch_path.clone();
  let grpc_gaze_path = gaze_path.clone();
  let grpc_angular_scaling = angular_scaling.clone();
  let grpc_trial_counter = trial_counter.clone();
  let (handle_tx, handle_rx) = std::sync::mpsc::sync_channel(1);
  // Sent by `run_grpc` once the session-wide `TaskContext` it builds (see
  // its doc comment) is ready — shortly after the gRPC thread's Tokio
  // runtime starts, not immediately, since building it needs an async
  // connect — so `gfx::run` below can be handed the same instance
  // `task_controller::run`/`touch_screen::run`/`eye_tracking::run` use.
  let (context_tx, context_rx) = std::sync::mpsc::sync_channel(1);
  std::thread::Builder::new()
    .name("grpc".to_string())
    .spawn(move || {
      let runtime = match tokio::runtime::Runtime::new() {
        Ok(runtime) => runtime,
        Err(e) => {
          tracing::error!("failed to start Tokio runtime: {e}");
          return;
        }
      };
      let _ = handle_tx.send(runtime.handle().clone());
      if let Err(e) = runtime.block_on(run_grpc(
        addr,
        grpc_current_task,
        context_tx,
        grpc_window_position,
        grpc_window_size,
        grpc_touch_path,
        grpc_gaze_path,
        grpc_angular_scaling,
        grpc_trial_counter,
      )) {
        tracing::error!("gRPC client task ended: {e}");
      }
    })?;

  let tokio_handle = handle_rx
    .recv()
    .map_err(|_| anyhow::anyhow!("grpc thread exited before it started its Tokio runtime"))?;
  let context = context_rx
    .recv()
    .map_err(|_| anyhow::anyhow!("grpc thread exited before it built a TaskContext"))?;

  gfx::run(
    current_task,
    context,
    tokio_handle,
    window_position,
    window_size,
    touch_path,
    gaze_path,
    trial_counter,
  )
}

async fn run_grpc(
  mut addr: String,
  current_task: SharedTask,
  context_tx: std::sync::mpsc::SyncSender<Arc<TaskContext>>,
  window_position: touch_screen::SharedWindowPosition,
  window_size: touch_screen::SharedWindowSize,
  touch_path: touch_screen::SharedTouchPath,
  gaze_path: eye_tracking::SharedGazePath,
  angular_scaling: eye_tracking::SharedAngularScaling,
  trial_counter: task_controller::SharedTrialCounter,
) -> anyhow::Result<()> {
  // The TOUCH_SCREEN/OCULOMATIC analog streams (and TaskContext's own
  // logging/inject_analog streams) connect to this original address, not
  // wherever observable_bridge_v2 ends up redirecting to below — so this
  // connects right away, ahead of resolving that redirect, rather than
  // waiting on it for no reason.
  let unredirected_addr = addr.clone();
  let analog_client = ThalamusClient::connect(unredirected_addr.clone()).await?;

  // A real Thalamus `TaskContext` is constructed once for the whole task
  // controller session and shared by every trial (`task_controller::run`)
  // and by the touch/gaze analog streams below (`touch_screen::run`/
  // `eye_tracking::run`, which push samples into it), rather than each
  // opening its own. The sound manager is likewise opened once here and
  // forwarded to it, rather than each `BehaviorTask` opening its own.
  // `context_tx` hands it back to `main`, which needs the same instance for
  // `gfx::run`.
  let audio_manager = AudioManager::<DefaultBackend>::new(AudioManagerSettings::default())
    .expect("failed to open default audio device");
  let context = Arc::new(TaskContext::new(analog_client.clone(), audio_manager));
  let _ = context_tx.send(context.clone());

  let mut app_state = Value::Object(Default::default());

  // Resolve the observable_bridge_v2 stream, following a redirect if the first
  // (and, per Thalamus's server, only ever the first) message carries one. See
  // `Service::observable_bridge_v2` in grpc_impl.cpp.
  let mut inbound = None;
  // Kept alive for the rest of this function once resolved below: dropping the
  // sender closes our half of the bidi stream, which makes the server end the
  // whole RPC (see the `while (stream->Read(&in))` loop in
  // `Service::observable_bridge_v2` in grpc_impl.cpp) even though we never send
  // anything on it.
  let mut outbound_keepalive = None;
  for _ in 0..MAX_REDIRECTS {
    let channel = tonic::transport::Channel::from_shared(addr.clone())?
      .connect()
      .await?;
    let mut thalamus = ThalamusClient::new(channel);

    let (tx, rx) = mpsc::channel::<ObservableTransaction>(8);
    let outbound = ReceiverStream::new(rx);

    tracing::info!("connecting to observable_bridge_v2 at {addr}");
    let response = thalamus.observable_bridge_v2(outbound).await?;
    let mut stream = response.into_inner();

    let Some(first) = stream.message().await? else {
      return Ok(());
    };

    if !first.redirection.is_empty() {
      addr = normalize_target(&first.redirection);
      tracing::info!("observable_bridge_v2 redirected to {addr}");
      continue;
    }

    apply_transaction(&mut app_state, &first.changes, &angular_scaling)?;
    inbound = Some(stream);
    outbound_keepalive = Some(tx);
    break;
  }

  let Some(mut inbound) = inbound else {
    anyhow::bail!("too many observable_bridge_v2 redirects (last target: {addr})");
  };
  let _outbound_keepalive = outbound_keepalive;

  // Only now that the observable_bridge_v2 address is resolved (post-redirect,
  // if any) do we connect the TaskController's execution stream, to the same
  // resolved address.
  let task_controller_addr = addr.clone();
  let task_controller_context = context.clone();
  tokio::spawn(async move {
    if let Err(e) = task_controller::run(
      task_controller_addr,
      current_task,
      trial_counter,
      task_controller_context,
    )
    .await
    {
      tracing::error!("task controller execution stream failed: {e}");
    }
  });

  // TOUCH_SCREEN and OCULOMATIC both just hit `analog_client`'s service's
  // `analog` RPC for different node types, so they share the one connection
  // opened above rather than each dialing their own.
  let touch_client = analog_client.clone();
  let touch_context = context.clone();
  tokio::spawn(async move {
    if let Err(e) = touch_screen::run(touch_client, touch_context, window_position, touch_path)
      .await
    {
      tracing::error!("touch screen analog stream failed: {e}");
    }
  });
  let gaze_angular_scaling = angular_scaling.clone();
  tokio::spawn(async move {
    if let Err(e) = eye_tracking::run(
      analog_client,
      gaze_angular_scaling,
      gaze_path,
      context,
      window_size,
    )
    .await
    {
      tracing::error!("eye tracking analog stream failed: {e}");
    }
  });

  while let Some(transaction) = inbound.message().await? {
    apply_transaction(&mut app_state, &transaction.changes, &angular_scaling)?;
  }

  Ok(())
}

fn apply_transaction(
  app_state: &mut Value,
  changes: &[ObservableChange],
  angular_scaling: &eye_tracking::SharedAngularScaling,
) -> anyhow::Result<()> {
  let mut eye_scaling_changed = false;
  for change in changes {
    if touches_eye_scaling(&change.address) {
      eye_scaling_changed = true;
    }
    let action = change.action();
    if let Err(e) = state::apply_change(app_state, &change.address, &change.value, action) {
      tracing::warn!("failed to apply change {change:?}: {e}");
    }
  }
  if eye_scaling_changed {
    eye_tracking::refresh_angular_scaling(app_state, angular_scaling);
  }
  println!("{}", serde_json::to_string_pretty(app_state)?);
  Ok(())
}

/// Whether an `observable_bridge_v2` change address falls under
/// `['eye_scaling']` (or replaces the whole tree, address `""`), meaning the
/// cached `AngularScaling` config may be stale and needs refreshing.
fn touches_eye_scaling(address: &str) -> bool {
  address.is_empty() || address.starts_with("['eye_scaling']")
}

/// Thalamus redirects use the bare `grpc::CreateChannel` target format
/// (`host:port`, no scheme); tonic's `Channel` needs a URI, so add one if missing.
fn normalize_target(target: &str) -> String {
  if target.contains("://") {
    target.to_string()
  } else {
    format!("http://{target}")
  }
}
