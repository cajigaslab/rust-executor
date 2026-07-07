mod behavior_task;
mod gfx;
mod pb;
mod state;
mod task_controller;
mod touch_screen;

use behavior_task::SharedTask;
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

    // The Thalamus/TaskController gRPC clients run on a background thread with
    // their own Tokio runtime; the windowing/Vulkan render loop below needs the
    // main thread to itself on most platforms. The render loop invokes
    // `BehaviorTask::render` via that runtime's `spawn_blocking` + `block_on`
    // (see `gfx::render_subject_frame`), so the grpc thread sends a `Handle` to
    // it back once it's built.
    let grpc_current_task = current_task.clone();
    let grpc_window_position = window_position.clone();
    let (handle_tx, handle_rx) = std::sync::mpsc::sync_channel(1);
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
            if let Err(e) = runtime.block_on(run_grpc(addr, grpc_current_task, grpc_window_position)) {
                tracing::error!("gRPC client task ended: {e}");
            }
        })?;

    let tokio_handle = handle_rx
        .recv()
        .map_err(|_| anyhow::anyhow!("grpc thread exited before it started its Tokio runtime"))?;

    gfx::run(current_task, tokio_handle, window_position)
}

async fn run_grpc(mut addr: String, current_task: SharedTask, window_position: touch_screen::SharedWindowPosition) -> anyhow::Result<()> {
    // The TOUCH_SCREEN analog stream connects to this original address, not
    // wherever observable_bridge_v2 ends up redirecting to below.
    let unredirected_addr = addr.clone();

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

        apply_transaction(&mut app_state, &first.changes)?;
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
    let touch_current_task = current_task.clone();
    tokio::spawn(async move {
        if let Err(e) = task_controller::run(task_controller_addr, current_task).await {
            tracing::error!("task controller execution stream failed: {e}");
        }
    });
    tokio::spawn(async move {
        if let Err(e) = touch_screen::run(unredirected_addr, touch_current_task, window_position).await {
            tracing::error!("touch screen analog stream failed: {e}");
        }
    });

    while let Some(transaction) = inbound.message().await? {
        apply_transaction(&mut app_state, &transaction.changes)?;
    }

    Ok(())
}

fn apply_transaction(app_state: &mut Value, changes: &[ObservableChange]) -> anyhow::Result<()> {
    for change in changes {
        let action = change.action();
        if let Err(e) = state::apply_change(app_state, &change.address, &change.value, action) {
            tracing::warn!("failed to apply change {change:?}: {e}");
        }
    }
    println!("{}", serde_json::to_string_pretty(app_state)?);
    Ok(())
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
