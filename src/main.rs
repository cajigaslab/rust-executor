mod pb;
mod state;
mod task_controller;

use pb::thalamus_grpc::thalamus_client::ThalamusClient;
use pb::thalamus_grpc::{ObservableChange, ObservableTransaction};
use serde_json::Value;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

/// Bound on chained `observable_bridge_v2` redirects, to avoid looping forever
/// if two nodes redirect to each other.
const MAX_REDIRECTS: u32 = 8;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();

    let mut addr = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "http://127.0.0.1:50050".to_string());

    let mut app_state = Value::Object(Default::default());

    // Resolve the observable_bridge_v2 stream, following a redirect if the first
    // (and, per Thalamus's server, only ever the first) message carries one. See
    // `Service::observable_bridge_v2` in grpc_impl.cpp.
    let mut inbound = None;
    // Kept alive for the rest of `main` once resolved below: dropping the sender
    // closes our half of the bidi stream, which makes the server end the whole
    // RPC (see the `while (stream->Read(&in))` loop in `Service::observable_bridge_v2`
    // in grpc_impl.cpp) even though we never send anything on it.
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
    tokio::spawn(async move {
        if let Err(e) = task_controller::run(task_controller_addr).await {
            tracing::error!("task controller execution stream failed: {e}");
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
