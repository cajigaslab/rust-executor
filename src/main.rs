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

use std::num::NonZeroUsize;
use std::sync::Arc;

use behavior_task::{SharedTask, TaskContext};
use clap::Parser;
use kira::{AudioManager, AudioManagerSettings, DefaultBackend};
use pb::thalamus_grpc::thalamus_client::ThalamusClient;
use pb::thalamus_grpc::{ObservableChange, ObservableTransaction};
use state::{Bridge, State};
use tokio::sync::mpsc;
use tokio_stream::wrappers::UnboundedReceiverStream;

/// Bound on chained `observable_bridge_v2` redirects, to avoid looping forever
/// if two nodes redirect to each other.
const MAX_REDIRECTS: u32 = 8;

/// Default TaskController/Thalamus gRPC endpoint.
const DEFAULT_ADDR: &str = "http://127.0.0.1:50050";

/// Command-line arguments.
#[derive(Parser)]
struct Args {
  /// TaskController/Thalamus gRPC endpoint.
  #[arg(default_value = DEFAULT_ADDR)]
  addr: String,

  /// Number of Tokio worker threads for the gRPC runtime. Omit for Tokio's
  /// default (roughly one per CPU core).
  #[arg(long)]
  threads: Option<NonZeroUsize>,
}

fn main() -> anyhow::Result<()> {
  tracing_subscriber::fmt::init();

  let Args { addr, threads } = Args::parse();

  let current_task = behavior_task::shared_task();
  let window_position = touch_screen::shared_window_position();
  let window_size = touch_screen::shared_window_size();
  let touch_path = touch_screen::shared_touch_path();
  let gaze_path = eye_tracking::shared_gaze_path();
  let angular_scaling = eye_tracking::shared_angular_scaling();
  let trial_counter = task_controller::shared_trial_counter();

  // The Thalamus/TaskController gRPC clients run on a background thread with
  // their own Tokio runtime; the windowing/Vulkan render loop below needs the
  // main thread to itself on most platforms.
  let grpc_current_task = current_task.clone();
  let grpc_window_position = window_position.clone();
  let grpc_window_size = window_size.clone();
  let grpc_touch_path = touch_path.clone();
  let grpc_gaze_path = gaze_path.clone();
  let grpc_angular_scaling = angular_scaling.clone();
  let grpc_trial_counter = trial_counter.clone();
  // Sent by `run_grpc` once the session-wide `TaskContext` it builds (see
  // its doc comment) is ready — shortly after the gRPC thread's Tokio
  // runtime starts, not immediately, since building it needs an async
  // connect — so `gfx::run` below can be handed the same instance
  // `task_controller::run`/`touch_screen::run`/`eye_tracking::run` use.
  let (context_tx, context_rx) = std::sync::mpsc::sync_channel(1);
  std::thread::Builder::new()
    .name("grpc".to_string())
    .spawn(move || {
      // Standard multi-threaded runtime; `--threads N` pins it to N worker
      // threads (e.g. 1, for debugging), otherwise Tokio's default is used.
      let mut builder = tokio::runtime::Builder::new_multi_thread();
      builder.enable_all();
      if let Some(threads) = threads {
        builder.worker_threads(threads.get());
      }
      let runtime = match builder.build() {
        Ok(runtime) => runtime,
        Err(e) => {
          tracing::error!("failed to start Tokio runtime: {e}");
          return;
        }
      };
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

  let context = context_rx
    .recv()
    .map_err(|_| anyhow::anyhow!("grpc thread exited before it built a TaskContext"))?;

  gfx::run(
    current_task,
    context,
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
  // and by the touch/gaze feeds set up below (`touch_screen::factory`/
  // `eye_tracking::factory`, registered via `set_touch_factory`/
  // `set_gaze_factory`), rather than each opening its own. The sound
  // manager is likewise opened once here and forwarded to it, rather than
  // each `BehaviorTask` opening its own. `context_tx` hands it back to
  // `main`, which needs the same instance for `gfx::run`.
  let audio_manager = AudioManager::<DefaultBackend>::new(AudioManagerSettings::default())
    .expect("failed to open default audio device");
  let context = Arc::new(TaskContext::new(analog_client, audio_manager, window_size));
  let _ = context_tx.send(context.clone());

  // TOUCH_SCREEN and OCULOMATIC/ANGULAR_SCALING both just hit the context's
  // Thalamus client's `analog` RPC for different node types, reached
  // through `TaskContext::connect`'s connection-sharing registry rather
  // than each factory dialing its own.
  context.set_touch_factory(touch_screen::factory(window_position));
  // OCULOMATIC + client-side angular scaling (see `eye_tracking::factory`).
  // context.set_gaze_factory(eye_tracking::factory(angular_scaling.clone(), window_size));
  // ANGULAR_SCALING node — applies the scaling itself, reports absolute
  // screen coordinates (see `eye_tracking::factory_angular_scaling`).
  context.set_gaze_factory(eye_tracking::factory_angular_scaling());

  let app_state = context.app_state();

  // Resolve the observable_bridge_v2 stream, following a redirect if the first
  // (and, per Thalamus's server, only ever the first) message carries one. See
  // `Service::observable_bridge_v2` in grpc_impl.cpp.
  let mut inbound = None;
  // Owns the sending half of the bidi stream once resolved below, and is
  // installed as `app_state`'s remote storage so local writes go to Thalamus.
  // Dropping the sender closes our half of the stream, which makes the server
  // end the whole RPC (see the `while (stream->Read(&in))` loop in
  // `Service::observable_bridge_v2` in grpc_impl.cpp).
  let mut bridge = None;
  for _ in 0..MAX_REDIRECTS {
    let channel = tonic::transport::Channel::from_shared(addr.clone())?
      .connect()
      .await?;
    let mut thalamus = ThalamusClient::new(channel);

    // Unbounded: writes to `app_state` can come from any thread, including
    // non-async ones, so `Bridge` can't wait for room in a bounded channel.
    let (tx, rx) = mpsc::unbounded_channel::<ObservableTransaction>();
    let outbound = UnboundedReceiverStream::new(rx);

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

    apply_transaction(&app_state, None, &first, &angular_scaling);
    let resolved = Bridge::new(tx);
    resolved.install(&app_state);
    inbound = Some(stream);
    bridge = Some(resolved);
    break;
  }

  let Some(mut inbound) = inbound else {
    anyhow::bail!("too many observable_bridge_v2 redirects (last target: {addr})");
  };
  let bridge = bridge.expect("set whenever inbound is");

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

  let touch_overlay_context = context.clone();
  tokio::spawn(async move {
    if let Err(e) = touch_screen::run_overlay(touch_overlay_context, touch_path).await {
      tracing::error!("touch overlay forwarder failed: {e}");
    }
  });
  let gaze_overlay_context = context.clone();
  tokio::spawn(async move {
    if let Err(e) = eye_tracking::run_overlay(gaze_overlay_context, gaze_path).await {
      tracing::error!("gaze overlay forwarder failed: {e}");
    }
  });

  let result = async {
    while let Some(transaction) = inbound.message().await? {
      apply_transaction(&app_state, Some(&bridge), &transaction, &angular_scaling);
    }
    anyhow::Ok(())
  }
  .await;

  // The stream is gone: stop routing writes to it, so they apply locally.
  app_state.set_remote_storage(None);
  result
}

/// Applies one inbound `observable_bridge_v2` transaction to `app_state` —
/// through `bridge` once it exists, which also handles acknowledgements of
/// our own writes — then refreshes the cached `AngularScaling` if the
/// transaction touched `['eye_scaling']`.
fn apply_transaction(
  app_state: &State,
  bridge: Option<&Bridge>,
  transaction: &ObservableTransaction,
  angular_scaling: &eye_tracking::SharedAngularScaling,
) {
  match bridge {
    Some(bridge) => bridge.handle(app_state, transaction),
    None => state::apply_changes(app_state, &transaction.changes),
  }
  if transaction.changes.iter().any(touches_eye_scaling) {
    let eye_scaling = app_state.get("eye_scaling").map_or(serde_json::Value::Null, |v| v.to_json());
    eye_tracking::refresh_angular_scaling(
      &serde_json::json!({ "eye_scaling": eye_scaling }),
      angular_scaling,
    );
  }
}

/// Whether an `observable_bridge_v2` change address falls under
/// `['eye_scaling']` (or replaces the whole tree, address `""`), meaning the
/// cached `AngularScaling` config may be stale and needs refreshing.
fn touches_eye_scaling(change: &ObservableChange) -> bool {
  let address = change.address.trim();
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
