use crate::pb::task_controller_grpc::task_controller_client::TaskControllerClient;
use crate::pb::task_controller_grpc::{TaskConfig, TaskResult};
use serde_json::Value;
use std::time::Duration;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

/// Drives the `execution` stream: for every `TaskConfig` the server sends, print
/// its `task_type`, sleep for a second, then report a successful `TaskResult` back
/// on the same stream.
pub async fn run(addr: String) -> anyhow::Result<()> {
    let mut client = TaskControllerClient::connect(addr.clone()).await?;

    let (tx, rx) = mpsc::channel::<TaskResult>(8);
    let outbound = ReceiverStream::new(rx);

    tracing::info!("connecting to TaskController execution stream at {addr}");
    let response = client.execution(outbound).await?;
    let mut inbound = response.into_inner();

    while let Some(config) = inbound.message().await? {
        println!("task_type: {}", task_type_of(&config));

        tokio::time::sleep(Duration::from_secs(1)).await;

        let result = TaskResult {
            success: true,
            cancelled: false,
        };
        if tx.send(result).await.is_err() {
            // Server dropped the response stream; nothing more we can do.
            break;
        }
    }

    Ok(())
}

fn task_type_of(config: &TaskConfig) -> String {
    match serde_json::from_str::<Value>(&config.body) {
        Ok(Value::Object(mut map)) => match map.remove("task_type") {
            Some(Value::String(s)) => s,
            Some(other) => other.to_string(),
            None => "<no task_type field>".to_string(),
        },
        Ok(_) => "<task config body is not a JSON object>".to_string(),
        Err(_) => format!("<unparseable task config body: {}>", config.body),
    }
}
