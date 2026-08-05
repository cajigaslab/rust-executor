mod simple_arc;
mod task_context;
mod vcp_inhibition;
mod vcp_2afc_task;
mod converter;

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use skia_safe::Canvas;

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
/// care about them read `context.touch()`/`context.gaze()` (the latest
/// sample) or subscribe to every sample via
/// `context.subscribe_to_touch()`/`context.subscribe_to_gaze()` (see
/// `TaskContext`). `render` itself isn't passed the `TaskContext` `run` got
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

/// Creates a fresh [`BehaviorTask`] instance for one trial of a given
/// `task_type`.
pub type BehaviorTaskFactory = fn() -> Arc<dyn BehaviorTask>;

/// Maps `task_type` (from `TaskConfig.body`) to the factory that creates the
/// `BehaviorTask` implementing that type's behavior.
pub fn registry() -> HashMap<String, BehaviorTaskFactory> {
  let mut map: HashMap<String, BehaviorTaskFactory> = HashMap::new();
  map.insert(
    "simple".to_string(),
    (|| Arc::new(SimpleArcTask::new()) as Arc<dyn BehaviorTask>) as BehaviorTaskFactory,
  );
  map.insert(
    "VCP_inhibition_task".to_string(),
    (|| Arc::new(VcpInhibitionTask::new()) as Arc<dyn BehaviorTask>) as BehaviorTaskFactory,
  );
  map.insert(
    "VCP_2AFC_task".to_string(),
    (|| Arc::new(Vcp2AfcTask::new()) as Arc<dyn BehaviorTask>) as BehaviorTaskFactory,
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
