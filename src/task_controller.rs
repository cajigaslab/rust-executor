use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::behavior_task::{self, SharedTask, TaskContext};
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
/// `trial_counter`, and report the `TaskResult` it returned back on the same
/// stream (or a failing one, if no `BehaviorTask` is registered for the
/// type). `context` is the session-wide `TaskContext` constructed once in
/// `main::run_grpc` and shared with `touch_screen::run`/`eye_tracking::run`,
/// not created here.
pub async fn run(
  addr: String,
  current_task: SharedTask,
  trial_counter: SharedTrialCounter,
  context: Arc<TaskContext>,
) -> anyhow::Result<()> {
  let registry = behavior_task::registry();

  let mut client = TaskControllerClient::connect(addr.clone()).await?;

  let (tx, rx) = mpsc::channel::<TaskResult>(8);
  let outbound = ReceiverStream::new(rx);

  tracing::info!("connecting to TaskController execution stream at {addr}");
  let response = client.execution(outbound).await?;
  let mut inbound = response.into_inner();

  while let Some(config) = inbound.message().await? {
    let body = body_of(&config);
    let task_type = task_type_of(&body);
    println!("task_type: {task_type}");
    println!(
      "{}",
      serde_json::to_string_pretty(&body).expect("config should always serialize to JSON")
    );

    let result = match registry.get(&task_type) {
      Some(task) => {
        *current_task.lock().unwrap() = Some(task.clone());

        // Ported from task_context.py:746,748: logged around every
        // `task.run` call, not just this task type's.
        let start_message = format!(
          "TRIAL START {} {}",
          get_str(&body, "task_type"),
          get_str(&body, "name")
        );
        let finished_message = format!(
          "TRIAL FINISHED {} {}",
          get_str(&body, "task_type"),
          get_str(&body, "name")
        );

        context.begin_trial(body).await;
        context.log(&start_message).await;
        let result = task.run(context.clone()).await;
        context.log(&finished_message).await;

        *current_task.lock().unwrap() = None;
        trial_counter.fetch_add(1, Ordering::Relaxed);
        result
      }
      None => {
        tracing::warn!("no BehaviorTask registered for task_type {task_type:?}");
        TaskResult {
          success: false,
          cancelled: false,
        }
      }
    };

    if tx.send(result).await.is_err() {
      // Server dropped the response stream; nothing more we can do.
      break;
    }
  }

  Ok(())
}

/// Parses `TaskConfig.body` as a JSON object, passed to `BehaviorTask::run` as
/// that trial's config. Falls back to an empty object if it's missing or
/// isn't a JSON object.
fn body_of(config: &TaskConfig) -> Value {
  match serde_json::from_str::<Value>(&config.body) {
    Ok(Value::Object(map)) => Value::Object(map),
    _ => Value::Object(Default::default()),
  }
}

/// Panics rather than defaulting: a missing/malformed field here means this
/// task's config doesn't match what the TRIAL START/FINISHED log messages
/// need, worth failing loudly on rather than logging a bogus value.
fn get_str<'a>(body: &'a Value, key: &str) -> &'a str {
  body
    .get(key)
    .and_then(Value::as_str)
    .unwrap_or_else(|| panic!("task config missing required string field {key:?}: {body}"))
}

fn task_type_of(body: &Value) -> String {
  match body.get("task_type") {
    Some(Value::String(s)) => s.clone(),
    Some(other) => other.to_string(),
    None => "<no task_type field>".to_string(),
  }
}
