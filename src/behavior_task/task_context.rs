use std::collections::HashMap;
use std::sync::{Arc, Mutex, Weak};

use kira::AudioManager;
use kira::sound::static_sound::StaticSoundData;
use serde_json::Value;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tonic::transport::Channel;

use crate::analog::{Connection, PointBroadcast, run_connection};
pub use crate::analog::PointSubscription;
use crate::pb::thalamus_grpc::thalamus_client::ThalamusClient;
use crate::pb::thalamus_grpc::{
  AnalogResponse, InjectAnalogRequest, NodeSelector, Text, inject_analog_request,
};
use crate::pb::task_controller_grpc::{BehavResult, UsedValues};
use crate::pb::thalamus_grpc::NodeRequest;
use crate::pb::thalamus_grpc::node_response::Status as NodeResponseStatus;
use crate::pb::task_controller_grpc::task_controller_client::TaskControllerClient;
use crate::state::Dict;

/// How `TaskContext` builds the touch/gaze feed every [`Self::subscribe_to_touch`]/
/// [`Self::subscribe_to_gaze`] caller gets back — see
/// [`TaskContext::set_touch_factory`]/[`TaskContext::set_gaze_factory`].
/// Called fresh on every `subscribe_to_*` call rather than once, so a
/// factory built on top of [`TaskContext::connect`] (e.g.
/// `touch_screen::factory`/`eye_tracking::factory`) stays cheap to call
/// repeatedly: `connect` reuses an already-live connection for the same
/// node/channels instead of dialing a new one each time.
type PointFactory = Box<dyn Fn(&TaskContext) -> Arc<PointSubscription> + Send + Sync>;

/// Merges `a` and `b` into a single feed carrying everything either one
/// produces, via a background task that wakes on whichever's `notify()`
/// fires first and drains both into the returned subscription. Backs
/// [`TaskContext::subscribe_to_touch`]/[`TaskContext::subscribe_to_gaze`],
/// which merge a factory's real `analog`-sourced subscription with a
/// [`PointBroadcast`] subscription (`TaskContext::inject_touch`/
/// `inject_gaze`) so an injected point (e.g. gfx's mouse-simulated gaze)
/// reaches the same consumers a real sample would.
fn merge_point_subscriptions(
  a: Arc<PointSubscription>,
  b: Arc<PointSubscription>,
) -> Arc<PointSubscription> {
  let merged = PointSubscription::new();
  let weak_merged = Arc::downgrade(&merged);
  tokio::spawn(async move {
    loop {
      let notified_a = a.notify().notified();
      let notified_b = b.notify().notified();
      let Some(merged) = weak_merged.upgrade() else {
        return; // caller dropped the merged subscription: stop forwarding
      };
      for point in a.drain() {
        merged.push(point);
      }
      for point in b.drain() {
        merged.push(point);
      }
      drop(merged); // don't hold a strong ref across the select below
      tokio::select! {
        _ = notified_a => {},
        _ = notified_b => {},
      }
    }
  });
  merged
}

/// Identifies one shared `analog` stream — a node plus the ordered list of
/// channels read from it — for [`TaskContext::connect`]'s connection-sharing
/// registry.
#[derive(Clone, PartialEq, Eq, Hash)]
struct ConnectionKey {
  node: NodeSelector,
  channels: Vec<String>,
}

/// Everything a [`super::BehaviorTask`] needs across trials: the current
/// trial's `TaskConfig.body` (parsed as JSON), a Thalamus client it can use
/// to log back to the server, a sound manager it can use to play audio, and
/// the subscribable touch and gaze feeds — `subscribe_to_touch`/
/// `subscribe_to_gaze` merge whatever factory `main::run_grpc` registered
/// via `set_touch_factory`/`set_gaze_factory` with any point injected via
/// `inject_touch`/`inject_gaze` (used by gfx's mouse-simulated gaze).
/// `TaskContext` itself doesn't track a "latest" touch/gaze point — a task
/// that wants one keeps its own, e.g. by holding the most recent point off
/// its own `subscribe_to_touch`/`subscribe_to_gaze` feed. Mirrors Python's
/// `TaskContext`, which is constructed once for the whole task controller
/// session — this `TaskContext` is likewise created once (see
/// `main::run_grpc`) and shared for the lifetime of the process, reused for
/// every trial via [`TaskContext::begin_trial`] rather than recreated per
/// trial.
pub struct TaskContext {
  config: Mutex<Value>,
  thalamus_client: ThalamusClient<Channel>,
  /// The current trial's `log` stream sender, opened by `begin_trial` (see
  /// `reopen_log_stream`). `None` only before the first `begin_trial` call.
  log_sender: Mutex<Option<mpsc::Sender<Text>>>,
  /// Ported from `TaskContext.inject_analog_streams` (task_context.py):
  /// caches the outbound sender for each node name's `inject_analog` stream,
  /// so repeat calls for the same name reuse it rather than opening a new
  /// stream every time. Persists across trials, like the rest of
  /// `TaskContext` — `begin_trial` resends the `node` handshake on each of
  /// these (Python's `refresh_streams`) rather than reopening them.
  inject_analog_streams: Mutex<HashMap<String, mpsc::Sender<InjectAnalogRequest>>>,
  /// Shared across every task and every trial (opening an audio device is
  /// expensive, and there's only ever one output), unlike sound *data*
  /// (`StaticSoundData`), which each `BehaviorTask` loads and owns itself.
  audio_manager: Mutex<AudioManager>,
  /// Set once via [`Self::set_touch_factory`] (typically from
  /// `main::run_grpc`, before anything can call `subscribe_to_touch`).
  touch_factory: Mutex<Option<PointFactory>>,
  /// Set once via [`Self::set_gaze_factory`].
  gaze_factory: Mutex<Option<PointFactory>>,
  /// Backs [`Self::inject_touch`] — merged into every
  /// [`Self::subscribe_to_touch`] feed alongside `touch_factory`'s real one.
  injected_touch: PointBroadcast,
  /// Backs [`Self::inject_gaze`] — merged into every
  /// [`Self::subscribe_to_gaze`] feed. Fed by gfx's mouse-simulated gaze.
  injected_gaze: PointBroadcast,
  /// Live `analog` connections opened via [`Self::connect`], keyed by
  /// [`ConnectionKey`] and weakly held — see that method's doc comment.
  connections: Mutex<HashMap<ConnectionKey, Weak<Connection>>>,
  _canvas_size: Arc<Mutex<(u32, u32)>>,
  /// Thalamus' application state (its whole config), kept in sync over
  /// `observable_bridge_v2` by `main::run_grpc` — see [`Self::app_state`].
  app_state: Arc<Dict>,
  /// The TaskController's client, set by `task_controller::run` once it
  /// connects.
  task_controller: Mutex<Option<TaskControllerClient<Channel>>>,
  /// The current trial's used values, accumulated by
  /// [`Self::set_used_values`] and sent by [`Self::send_used_values`].
  used_values: Mutex<serde_json::Map<String, Value>>,
  /// The subject window's scale factor (physical pixels per logical pixel),
  /// kept current by `gfx` — see [`Self::dpi`].
  window_scale_factor: Mutex<f64>,
}

/// Qt's logical DPI at a scale factor of 1.
const BASE_LOGICAL_DPI: f64 = 96.0;

/// Whether `node_type` is a storage node's: `STORAGE`, or `STORAGE` followed
/// by an integer (`STORAGE2` etc.).
fn is_storage_type(node_type: &str) -> bool {
  node_type
    .strip_prefix("STORAGE")
    .is_some_and(|n| n.bytes().all(|b| b.is_ascii_digit()))
}

impl TaskContext {
  pub fn new(thalamus_client: ThalamusClient<Channel>, audio_manager: AudioManager, canvas_size: Arc<Mutex<(u32, u32)>>) -> Self {
    Self {
      config: Mutex::new(Value::Object(Default::default())),
      thalamus_client,
      log_sender: Mutex::new(None),
      inject_analog_streams: Mutex::new(HashMap::new()),
      audio_manager: Mutex::new(audio_manager),
      touch_factory: Mutex::new(None),
      gaze_factory: Mutex::new(None),
      injected_touch: PointBroadcast::new(),
      injected_gaze: PointBroadcast::new(),
      connections: Mutex::new(HashMap::new()),
      _canvas_size: canvas_size,
      app_state: Dict::new(),
      task_controller: Mutex::new(None),
      used_values: Mutex::new(serde_json::Map::new()),
      window_scale_factor: Mutex::new(1.0),
    }
  }

  /// Thalamus' application state (Python's `TaskContext.config`, the whole
  /// Thalamus config — not the per-trial task config, see [`Self::config`]).
  /// Mirrored from Thalamus over `observable_bridge_v2` (see
  /// `main::run_grpc`), so it starts out empty and fills in once that
  /// stream connects. Writes are sent to Thalamus and only show up here once
  /// Thalamus echoes them back; pass a callback to `Dict::set_with`
  /// etc. to find out when that's happened.
  pub fn app_state(&self) -> Arc<Dict> {
    self.app_state.clone()
  }

  /// Ported from Python's `context.config.get('dpi', None) or
  /// context.widget.logicalDpiX()`: the Thalamus config's `dpi` if it's set
  /// to anything but 0 or null, else the subject window's logical DPI. Qt's
  /// logical DPI is 96 times the screen's scale factor, and this executor
  /// draws in physical pixels, which is the same thing. An error if `dpi` is
  /// set to something other than a number.
  pub fn dpi(&self) -> anyhow::Result<f64> {
    match self.app_state.get("dpi") {
      None | Some(crate::state::Value::Null) => Ok(self.logical_dpi()),
      Some(value) => {
        let Ok(dpi) = f64::try_from(value.clone()) else {
          anyhow::bail!("the Thalamus config's dpi must be a number, got {}", value.to_json());
        };
        Ok(if dpi == 0.0 { self.logical_dpi() } else { dpi })
      }
    }
  }

  fn logical_dpi(&self) -> f64 {
    BASE_LOGICAL_DPI * *self.window_scale_factor.lock().unwrap()
  }

  /// Called by `gfx` with the subject window's current scale factor.
  pub fn set_window_scale_factor(&self, scale_factor: f64) {
    *self.window_scale_factor.lock().unwrap() = scale_factor;
  }

  /// Ported from `TaskContext.get_reward`: the current reward for `channel`,
  /// `reward_schedule.schedules[channel][index]` in the Thalamus config,
  /// clamped to at least 0. Records the unclamped value as the `reward` used
  /// value, as Python does. An error if the schedule doesn't have that
  /// entry, where Python raises. Python's task controller advances `index`
  /// after each successful trial, so this doesn't.
  pub fn get_reward(&self, channel: usize) -> anyhow::Result<f64> {
    let schedule = self
      .app_state
      .get("reward_schedule")
      .ok_or_else(|| anyhow::anyhow!("the Thalamus config has no reward_schedule"))?
      .to_json();
    let index = schedule
      .get("index")
      .and_then(Value::as_u64)
      .ok_or_else(|| anyhow::anyhow!("reward_schedule.index must be a non-negative integer: {schedule}"))?;
    let reward = schedule
      .get("schedules")
      .and_then(|s| s.get(channel))
      .and_then(|s| s.get(index as usize))
      .and_then(Value::as_f64)
      .ok_or_else(|| anyhow::anyhow!("reward_schedule has no number at schedules[{channel}][{index}]: {schedule}"))?;
    self.set_used_values(serde_json::Map::from_iter([("reward".to_string(), Value::from(reward))]));
    Ok(reward.max(0.0))
  }

  /// Called by `task_controller::run` once it's connected.
  pub fn set_task_controller(&self, client: TaskControllerClient<Channel>) {
    *self.task_controller.lock().unwrap() = Some(client);
  }

  /// Merges `values` into the running trial's used values, which the
  /// TaskController puts in the trial summary: Python tasks'
  /// `context.trial_summary_data.used_values[key] = value`. They're kept
  /// here and sent together once the task returns a result (see
  /// [`Self::send_used_values`]).
  pub fn set_used_values(&self, values: serde_json::Map<String, Value>) {
    self.used_values.lock().unwrap().extend(values);
  }

  /// Sends the trial's accumulated used values to the TaskController, if
  /// there are any. Called by `task_controller::run` after the task returns
  /// a result, before reporting it, so they're in that trial's summary. An
  /// error if the TaskController isn't connected or rejects the call.
  pub async fn send_used_values(&self) -> anyhow::Result<()> {
    let values = std::mem::take(&mut *self.used_values.lock().unwrap());
    if values.is_empty() {
      return Ok(());
    }
    let mut client = self.task_controller_client()?;
    let body = Value::Object(values).to_string();
    client
      .set_used_values(UsedValues { body: body.clone() })
      .await
      .map_err(|e| anyhow::anyhow!("the TaskController rejected used values {body}: {e}"))?;
    Ok(())
  }

  /// Replaces the running trial's behav_result, which the TaskController
  /// puts in the trial summary: Python tasks' `context.behav_result = ...`.
  /// Await it before the task returns, so it's there before the result is.
  pub async fn set_behav_result(&self, result: serde_json::Map<String, Value>) -> anyhow::Result<()> {
    let mut client = self.task_controller_client()?;
    let body = Value::Object(result).to_string();
    client
      .set_behav_result(BehavResult { body: body.clone() })
      .await
      .map_err(|e| anyhow::anyhow!("the TaskController rejected behav_result {body}: {e}"))?;
    Ok(())
  }

  /// Ported from `TaskContext.node_request`: sends `request` to the node
  /// named `name` and returns its JSON response. Uses Thalamus' unary
  /// `node_request` RPC rather than Python's per-node request streams.
  pub async fn node_request(&self, name: &str, request: &Value) -> anyhow::Result<Value> {
    let response = self
      .thalamus_client
      .clone()
      .node_request(NodeRequest {
        node: name.to_string(),
        json: request.to_string(),
        ..Default::default()
      })
      .await
      .map_err(|e| anyhow::anyhow!("node_request to {name} failed: {e}"))?
      .into_inner();
    anyhow::ensure!(
      response.status == NodeResponseStatus::Ok as i32,
      "node_request to {name} returned status {:?}",
      NodeResponseStatus::try_from(response.status).map_or_else(|_| response.status.to_string(), |s| format!("{s:?}"))
    );
    serde_json::from_str(&response.json)
      .map_err(|e| anyhow::anyhow!("node_request to {name} returned invalid JSON {:?}: {e}", response.json))
  }

  /// If a storage node (`STORAGE` or `STORAGE<integer>`) is running, saves `diff` next to the file it's
  /// recording to, as `<that file>.diff`, unless that's already there or
  /// `diff` is empty. Replaces Python's `save_task_source`: the executor's
  /// commit and remote (logged every trial) plus this diff pin down the
  /// code that ran. The path is the storage node's, so this writes to the
  /// right place only on the machine (or a share) where that path is the
  /// recording's.
  pub async fn save_uncommitted_diff(&self, diff: &str) -> anyhow::Result<()> {
    if diff.is_empty() {
      return Ok(());
    }
    let Some(nodes) = self.app_state.get("nodes") else {
      return Ok(());
    };
    let nodes = nodes.to_json();
    let storage = nodes.as_array().into_iter().flatten().find(|node| {
      node.get("type").and_then(Value::as_str).is_some_and(is_storage_type)
        && node.get("Running").and_then(Value::as_bool).unwrap_or(false)
    });
    let Some(storage) = storage else {
      return Ok(());
    };
    let name = storage
      .get("name")
      .and_then(Value::as_str)
      .ok_or_else(|| anyhow::anyhow!("the running storage node has no name: {storage}"))?;

    let response = self.node_request(name, &serde_json::json!({"type": "get_current_file"})).await?;
    tracing::debug!("current_file {response}");
    let current_file = match &response {
      // Python's `if not response: return`.
      Value::Null | Value::Bool(false) => return Ok(()),
      Value::String(file) if file.is_empty() => return Ok(()),
      Value::String(file) => file,
      other => anyhow::bail!("{name}'s get_current_file returned {other}, not a file name"),
    };

    let out_path = std::path::PathBuf::from(format!("{current_file}.diff"));
    // `create_new`, so an existing copy is never overwritten.
    match std::fs::File::create_new(&out_path) {
      Ok(mut file) => {
        use std::io::Write;
        file
          .write_all(diff.as_bytes())
          .map_err(|e| anyhow::anyhow!("failed to write {}: {e}", out_path.display()))
      }
      Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => Ok(()),
      Err(e) => Err(anyhow::anyhow!("failed to create {}: {e}", out_path.display())),
    }
  }

  fn task_controller_client(&self) -> anyhow::Result<TaskControllerClient<Channel>> {
    self
      .task_controller
      .lock()
      .unwrap()
      .clone()
      .ok_or_else(|| anyhow::anyhow!("the TaskController isn't connected"))
  }

  /// Registers `factory` as how [`Self::subscribe_to_touch`] builds the
  /// real half of the feed it hands back — typically `Self::connect` +
  /// `Connection::subscribe_points` (see `touch_screen::factory`). Called
  /// once at startup, before anything can call `subscribe_to_touch` (see
  /// `main::run_grpc`); a later call replaces the previous factory rather
  /// than stacking.
  pub fn set_touch_factory<F>(&self, factory: F)
  where
    F: Fn(&TaskContext) -> Arc<PointSubscription> + Send + Sync + 'static,
  {
    *self.touch_factory.lock().unwrap() = Some(Box::new(factory));
  }

  /// Gaze counterpart to [`Self::set_touch_factory`] (see
  /// `eye_tracking::factory`).
  pub fn set_gaze_factory<F>(&self, factory: F)
  where
    F: Fn(&TaskContext) -> Arc<PointSubscription> + Send + Sync + 'static,
  {
    *self.gaze_factory.lock().unwrap() = Some(Box::new(factory));
  }

  /// A new live feed of every touch point received from now on — the merge
  /// (see [`merge_point_subscriptions`]) of whatever factory
  /// [`Self::set_touch_factory`] registered with anything injected via
  /// [`Self::inject_touch`] since this call. Panics if no factory has been
  /// registered yet.
  pub fn subscribe_to_touch(&self) -> Arc<PointSubscription> {
    let source = {
      let factory = self.touch_factory.lock().unwrap();
      (factory.as_ref()).expect("subscribe_to_touch called before set_touch_factory")(self)
    };
    merge_point_subscriptions(source, self.injected_touch.subscribe())
  }

  /// Gaze counterpart to [`Self::subscribe_to_touch`].
  pub fn subscribe_to_gaze(&self) -> Arc<PointSubscription> {
    let source = {
      let factory = self.gaze_factory.lock().unwrap();
      (factory.as_ref()).expect("subscribe_to_gaze called before set_gaze_factory")(self)
    };
    merge_point_subscriptions(source, self.injected_gaze.subscribe())
  }

  /// Injects `point` as a synthetic touch sample, reaching every live
  /// [`Self::subscribe_to_touch`] feed exactly as a real one would (see
  /// [`PointBroadcast::publish`]).
  pub fn inject_touch(&self, point: (f64, f64)) {
    self.injected_touch.publish(point);
  }

  /// Gaze counterpart to [`Self::inject_touch`] — used by gfx's
  /// mouse-simulated gaze (`gfx::forward_simulated_gaze`).
  pub fn inject_gaze(&self, point: (f64, f64)) {
    self.injected_gaze.publish(point);
  }

  /// Opens (or reuses) a shared `analog` RPC stream for `node`'s `channels`
  /// on this context's Thalamus client — see [`Connection`]. A later
  /// `connect` call for the same `node`/`channels` (see [`ConnectionKey`])
  /// reuses the same underlying stream and background reader task instead
  /// of dialing a new one. Returns immediately: dialing happens in a
  /// spawned background task ([`run_connection`]), so a connection failure
  /// is only ever logged there, not surfaced here; every
  /// [`Connection::subscribe`] caller sees every message read from the
  /// moment they subscribe, whether or not the initial handshake has
  /// completed yet.
  pub fn connect(&self, node: NodeSelector, channels: Vec<String>) -> Arc<Connection> {
    let key = ConnectionKey {
      node: node.clone(),
      channels: channels.clone(),
    };

    let mut connections = self.connections.lock().unwrap();
    if let Some(connection) = connections.get(&key).and_then(Weak::upgrade) {
      return connection;
    }

    let connection = Connection::new();
    connections.insert(key, Arc::downgrade(&connection));
    drop(connections);

    let client = self.thalamus_client.clone();
    let reader_connection = connection.clone();
    tokio::spawn(async move {
      if let Err(e) = run_connection(client, node, channels, reader_connection).await {
        tracing::warn!("analog connection failed: {e}");
      }
    });

    connection
  }

  /// Plays `sound` through the shared audio manager.
  pub fn play_sound(&self, sound: StaticSoundData) {
    if let Err(e) = self.audio_manager.lock().unwrap().play(sound) {
      tracing::warn!("failed to play sound: {e}");
    }
  }

  /// The current trial's `TaskConfig.body`, parsed as JSON.
  pub fn config(&self) -> Value {
    self.config.lock().unwrap().clone()
  }

  /// Ported from the per-trial reset at the top of `TaskContext.run`
  /// (task_context.py:723-745): swaps in `config` for the trial about to
  /// start, clears the used values, reopens the `log` stream (Python opens `self.log_queue`/
  /// `log_coroutine` fresh every trial and closes them at trial end — see
  /// `reopen_log_stream`), and resends the `node` handshake on every
  /// already-open `inject_analog` stream (Python's `refresh_streams`).
  /// Called by `task_controller::run` before each trial, since — unlike a
  /// fresh `TaskContext` per trial — a real Thalamus `TaskContext` persists
  /// for the whole session and only swaps out what changes per trial.
  pub async fn begin_trial(&self, config: Value) {
    *self.config.lock().unwrap() = config;
    self.used_values.lock().unwrap().clear();
    self.reopen_log_stream().await;
    self.refresh_inject_analog_streams().await;
  }

  /// Opens a fresh `log` stream for the trial about to start, replacing
  /// (and thereby closing, once its last sender is dropped) any previous
  /// one.
  async fn reopen_log_stream(&self) {
    let (tx, rx) = mpsc::channel::<Text>(8);
    let outbound = ReceiverStream::new(rx);
    let mut client = self.thalamus_client.clone();
    tokio::spawn(async move {
      if let Err(e) = client.log(outbound).await {
        tracing::warn!("Thalamus log RPC failed: {e}");
      }
    });
    *self.log_sender.lock().unwrap() = Some(tx);
  }

  /// Ported from `TaskContext.refresh_streams` (task_context.py:373-379),
  /// restricted to `inject_analog_streams` — the only stream kind this port
  /// has so far.
  async fn refresh_inject_analog_streams(&self) {
    let senders: Vec<(String, mpsc::Sender<InjectAnalogRequest>)> = self
      .inject_analog_streams
      .lock()
      .unwrap()
      .iter()
      .map(|(name, sender)| (name.clone(), sender.clone()))
      .collect();
    for (name, sender) in senders {
      let _ = sender
        .send(InjectAnalogRequest {
          body: Some(inject_analog_request::Body::Node(name)),
        })
        .await;
    }
  }

  /// Logs `text` to Thalamus by sending it on the current trial's `log`
  /// stream (opened by `begin_trial`).
  pub async fn log(&self, text: &str) {
    let sender = self.log_sender.lock().unwrap().clone();
    let Some(sender) = sender else {
      tracing::warn!("TaskContext::log called before begin_trial opened a log stream");
      return;
    };
    let message = Text {
      text: text.to_string(),
      time: crate::monotonic_time::now_ns(),
      remote_time: 0,
      redirect: String::new(),
    };
    let _ = sender.send(message).await;
  }

  /// Ported from `TaskContext.get_inject_stream` (task_context.py): returns
  /// the sender for `name`'s `inject_analog` stream, opening one — with the
  /// required initial `InjectAnalogRequest{node: name}` handshake message —
  /// the first time `name` is requested. The stream itself runs as a
  /// background task for as long as its sender is held onto (Python instead
  /// leaves the streaming RPC call un-awaited, relying on grpc.aio to drive
  /// it in the background).
  async fn get_inject_stream(&self, name: &str) -> mpsc::Sender<InjectAnalogRequest> {
    if let Some(sender) = self.inject_analog_streams.lock().unwrap().get(name) {
      return sender.clone();
    }

    let (tx, rx) = mpsc::channel::<InjectAnalogRequest>(8);
    let outbound = ReceiverStream::new(rx);
    let mut client = self.thalamus_client.clone();
    tokio::spawn(async move {
      if let Err(e) = client.inject_analog(outbound).await {
        tracing::warn!("Thalamus inject_analog RPC failed: {e}");
      }
    });

    let _ = tx
      .send(InjectAnalogRequest {
        body: Some(inject_analog_request::Body::Node(name.to_string())),
      })
      .await;

    self
      .inject_analog_streams
      .lock()
      .unwrap()
      .insert(name.to_string(), tx.clone());
    tx
  }

  /// Ported from `TaskContext.inject_analog` (task_context.py): sends
  /// `payload` on `name`'s `inject_analog` stream (opening it first if
  /// needed).
  pub async fn inject_analog(&self, name: &str, payload: AnalogResponse) {
    let sender = self.get_inject_stream(name).await;
    let _ = sender
      .send(InjectAnalogRequest {
        body: Some(inject_analog_request::Body::Signal(payload)),
      })
      .await;
  }

  pub fn canvas_size(&self) -> (u32, u32) {
    *self._canvas_size.lock().unwrap()
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn storage_types_are_storage_and_an_optional_integer() {
    for t in ["STORAGE", "STORAGE2", "STORAGE10"] {
      assert!(is_storage_type(t), "{t}");
    }
    for t in ["STORAGE2X", "STORAGE-1", "storage2", "ANALOG"] {
      assert!(!is_storage_type(t), "{t}");
    }
  }
}
