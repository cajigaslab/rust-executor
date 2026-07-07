mod simple_arc;

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use skia_safe::Canvas;

pub use simple_arc::SimpleArcTask;

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
/// done. Every frame, the render thread renders the task in two phases —
/// concurrently with `run`, for as long as the task is the current one: it
/// calls `render` with `Window::Subject` to draw the subject view, then
/// calls `render` again with `Window::Operator` to draw the operator view,
/// drawing each resulting image on its corresponding window. `render`
/// implementations can branch on `window` to vary what they draw between the
/// two phases (e.g. operator-only overlays), and use
/// `canvas.base_layer_size()` for the canvas's pixel dimensions. `on_touch`
/// is called (also concurrently with `run`) whenever a touch point is
/// received from the TOUCH_SCREEN analog stream while this task is current
/// (see `touch_screen::run`). Implementations use interior mutability (all
/// methods take `&self`) since these can run on different threads at the
/// same time.
#[async_trait]
pub trait BehaviorTask: Send + Sync {
    async fn run(&self);
    fn render(&self, canvas: &Canvas, window: Window);

    /// Called with the screen coordinates of a touch point. The default does
    /// nothing; tasks that care about touch input override it.
    fn on_touch(&self, _x: i32, _y: i32) {}
}

/// Creates a fresh [`BehaviorTask`] instance for one trial of a given
/// `task_type`.
pub type BehaviorTaskFactory = fn() -> Arc<dyn BehaviorTask>;

/// Maps `task_type` (from `TaskConfig.body`) to the factory that creates the
/// `BehaviorTask` implementing that type's behavior.
pub fn registry() -> HashMap<String, BehaviorTaskFactory> {
    let mut map: HashMap<String, BehaviorTaskFactory> = HashMap::new();
    map.insert("simple".to_string(), (|| {
        Arc::new(SimpleArcTask::new()) as Arc<dyn BehaviorTask>
    }) as BehaviorTaskFactory);
    map
}

/// The task currently being run by the TaskController, if any: set/cleared by
/// the gRPC thread around each call to `run`, and read every frame by the
/// render thread to call `render`.
pub type SharedTask = Arc<Mutex<Option<Arc<dyn BehaviorTask>>>>;

pub fn shared_task() -> SharedTask {
    Arc::new(Mutex::new(None))
}
