mod simple_arc;
mod task_context;
mod vcp_inhibition;
mod vcp_2afc_task;
mod converter;
mod config_util;

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use skia_safe::Canvas;
use std::time::{Duration, Instant};

use crate::pb::task_controller_grpc::TaskResult;

pub use simple_arc::SimpleArcTask;
pub use task_context::{PointSubscription, TaskContext};
pub use vcp_inhibition::VcpInhibitionTask;
pub use vcp_2afc_task::Vcp2AfcTask;

/// Which physical window a `render` call is currently producing pixels for.
/// See [`BehaviorTask`] for how the render thread drives this each frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Window {
  #[default]
  Subject,
  Operator,
}

/// A behavioral task run by the TaskController for one trial.
///
/// `run` performs the task's own timing/logic and resolves once the task is
/// done, returning the `TaskResult` `task_controller::run` reports back to
/// the TaskController's execution stream. Every frame, the render thread
/// renders the task in two phases —
/// concurrently with `run`, for as long as the task is the current one: it
/// calls `render` with `Window::Subject` to draw the subject view, then
/// calls `render` again with `Window::Operator` to draw the operator view,
/// drawing each resulting image on its corresponding window. `render`
/// implementations can branch on `window` to vary what they draw between the
/// two phases (e.g. operator-only overlays), and use
/// `canvas.base_layer_size()` for the canvas's pixel dimensions. Touch and
/// gaze input aren't pushed to task-side handlers — implementations that
/// care about them subscribe to every sample via
/// `context.subscribe_to_touch()`/`context.subscribe_to_gaze()` (see
/// `TaskContext`; it doesn't track a "latest" point itself, so a task that
/// wants one keeps its own, e.g. the most recent point off its own
/// subscription). `render` itself isn't passed the `TaskContext` `run` got
/// — implementations that need it there (e.g. to show current gaze in the
/// operator view) should stash the `Arc<TaskContext>` `run` receives in a
/// field of their own and read it back from `render`. Implementations use
/// interior mutability (all methods take `&self`) since these can run on
/// different threads at the same time.
#[async_trait]
pub trait BehaviorTask: Send + Sync {
  async fn run(&self, context: Arc<TaskContext>) -> TaskResult;
  fn render(&self, canvas: &Canvas, window: Window);

  /// Draws this task's own control panel in the operator view, in the space
  /// underneath the mirrored subject-view image (see
  /// `gfx::Graphics::render_frame`, which calls this between the image and
  /// its own Clear/Opacity/Show controls). The default draws nothing; tasks
  /// that want operator-side controls override it.
  fn operator_widget(&self, _ui: &imgui::Ui) {}
}

/// Maps `task_type` (from `TaskConfig.body`) to the factory that creates the
/// `BehaviorTask` implementing that type's behavior.
pub fn registry() -> HashMap<String, Arc<dyn BehaviorTask>> {
  let mut map: HashMap<String, Arc<dyn BehaviorTask>> = HashMap::new();
  map.insert(
    "simple".to_string(),
    Arc::new(SimpleArcTask::new()),
  );
  map.insert(
    "VCP_inhibition_task".to_string(),
    Arc::new(VcpInhibitionTask::new()),
  );
  map.insert(
    "VCP_2AFC_task".to_string(),
    Arc::new(Vcp2AfcTask::new()),
  );
  map
}

/// The task currently being run by the TaskController, if any: set/cleared by
/// the gRPC thread around each call to `run`, and read every frame by the
/// render thread to call `render`.
pub type SharedTask = Arc<Mutex<Option<Arc<dyn BehaviorTask>>>>;

pub fn shared_task() -> SharedTask {
  Arc::new(Mutex::new(None))
}

/// Ported from `wait_for` (util.py:357-366): waits — event-driven, waking
/// on `notify` — until `condition` returns true or `timeout` elapses
/// (waits indefinitely if `timeout` is `None`), then returns `condition`'s
/// value at that point. `condition` is an opaque predicate — `wait_for`
/// has no notion of gaze or touch itself; a condition that cares about
/// gaze builds it via `gaze_condition` (see its doc comment) to consume a
/// `GazeQueue` internally, and callers pass that same queue's
/// `PointSubscription::notify()` here so a wakeup isn't missed.
async fn wait_for(
  notify: &tokio::sync::Notify,
  condition: impl Fn() -> bool,
  timeout: Option<Duration>,
) -> bool {
  let deadline = timeout.map(|d| Instant::now() + d);
  loop {
    // Constructed before the check below, per `PointSubscription::notify`'s
    // doc comment, so a push racing with that check isn't missed.
    let notified = notify.notified();
    if condition() {
      return true;
    }
    match deadline {
      None => notified.await,
      Some(deadline) => {
        let Some(remaining) = deadline.checked_duration_since(Instant::now()) else {
          return condition();
        };
        tokio::select! {
          _ = notified => {}
          _ = tokio::time::sleep(remaining) => {}
        }
      }
    }
  }
}

  /// Ported from `wait_for_hold` (util.py:469-504, `include_blink = False`):
  /// waits for `condition` to hold continuously for `hold_duration`,
  /// tolerating brief lapses ("blinks") of up to `blink_duration` each (time
  /// spent blinking doesn't count against `hold_duration`) — a lapse not
  /// reacquired within `blink_duration` fails the hold. `blink_duration =
  /// None` (as Python's `fixate_func` passes) waits indefinitely to
  /// reacquire, so a lapse can never fail the hold that way.
  ///
  /// `blink_resets` is a newer `wait_for_hold` parameter (from a different
  /// machine's util.py than the one this port otherwise follows): when
  /// true, reacquiring after a blink restarts `hold_duration` from zero
  /// instead of just excluding the time spent blinking (`hold_target_func`'s
  /// saccade-trial branch passes `true`; every other call site here keeps
  /// the old behavior via `false`).
  ///
  /// Returns whether the hold succeeded.
  async fn wait_for_hold(
    notify: &tokio::sync::Notify,
    condition: impl Fn() -> bool,
    hold_duration: Duration,
    blink_duration: Option<Duration>,
    blink_resets: bool,
  ) -> bool {
    let mut start = Instant::now();
    let mut time_spent_blinking = Duration::ZERO;

    loop {
      let Some(remaining) = (hold_duration + time_spent_blinking).checked_sub(start.elapsed())
      else {
        break;
      };

      let blinked = wait_for(notify, || !condition(), Some(remaining))
        .await;
      if !blinked {
        break;
      }

      //context.log("BehavState=blink").await;

      let blink_start = Instant::now();
      let reacquired = wait_for(notify, &condition, blink_duration).await;
      if !reacquired {
        return false;
      }

      if blink_resets {
        start = Instant::now();
        time_spent_blinking = Duration::ZERO;
      } else {
        time_spent_blinking += blink_start.elapsed();
      }
    }

    true
  }
