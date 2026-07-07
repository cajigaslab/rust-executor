use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use crate::behavior_task::{self, SharedTask};
use crate::pb::task_controller_grpc::task_controller_client::TaskControllerClient;
use crate::pb::task_controller_grpc::{TaskConfig, TaskResult};
use serde_json::Value;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

/// Incremented every time a trial (task run) finishes, so other subsystems —
/// e.g. `gfx`, which auto-clears the operator view's touch/gaze traces when
/// this changes — can detect that a trial ended without polling
/// `current_task` for a `Some -> None` transition, which can be missed if a
/// new task starts before the next poll.
pub type SharedTrialCounter = Arc<AtomicU64>;

pub fn shared_trial_counter() -> SharedTrialCounter {
    Arc::new(AtomicU64::new(0))
}

/// Drives the `execution` stream: for every `TaskConfig` the server sends,
/// look up its `task_type` in the `BehaviorTask` registry, run that task
/// (installing it as the current task so the render thread can draw its
/// progress for as long as it runs, then dropping it once done), bump
/// `trial_counter`, and report a successful `TaskResult` back on the same
/// stream.
pub async fn run(addr: String, current_task: SharedTask, trial_counter: SharedTrialCounter) -> anyhow::Result<()> {
    let registry = behavior_task::registry();

    let mut client = TaskControllerClient::connect(addr.clone()).await?;

    let (tx, rx) = mpsc::channel::<TaskResult>(8);
    let outbound = ReceiverStream::new(rx);

    tracing::info!("connecting to TaskController execution stream at {addr}");
    let response = client.execution(outbound).await?;
    let mut inbound = response.into_inner();

    while let Some(config) = inbound.message().await? {
        let task_type = task_type_of(&config);
        println!("task_type: {task_type}");

        match registry.get(&task_type) {
            Some(factory) => {
                let task = factory();
                *current_task.lock().unwrap() = Some(task.clone());
                task.run().await;
                *current_task.lock().unwrap() = None;
                trial_counter.fetch_add(1, Ordering::Relaxed);
            }
            None => {
                tracing::warn!("no BehaviorTask registered for task_type {task_type:?}");
            }
        }

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
