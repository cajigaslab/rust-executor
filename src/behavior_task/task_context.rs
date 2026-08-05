use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex, Weak};

use kira::AudioManager;
use kira::sound::static_sound::StaticSoundData;
use serde_json::Value;
use tokio::sync::{Notify, mpsc};
use tokio_stream::wrappers::ReceiverStream;
use tonic::transport::Channel;

use crate::pb::thalamus_grpc::thalamus_client::ThalamusClient;
use crate::pb::thalamus_grpc::{AnalogResponse, InjectAnalogRequest, Text, inject_analog_request};

/// Cap on how many points a single [`PointSubscription`] buffers before
/// being drained: once full, the oldest point is dropped to make room for
/// the newest, so a subscription that's never drained (or drained too
/// slowly) doesn't grow unbounded.
const MAX_QUEUED_POINTS: usize = 3600;

/// A live feed of touch or gaze points, created by
/// [`TaskContext::subscribe_to_touch`]/[`TaskContext::subscribe_to_gaze`]:
/// every point pushed to the `TaskContext` after subscribing is buffered
/// here until [`Self::drain`] collects it. Stops receiving points (and lets
/// `TaskContext` reclaim its buffer) as soon as it's dropped — `TaskContext`
/// only holds a `Weak` reference to it.
pub struct PointSubscription {
  points: Arc<Mutex<VecDeque<(i32, i32)>>>,
}

impl PointSubscription {
  /// Drains and returns every point received since the last call to this
  /// method (or since subscribing, for the first call), oldest first.
  pub fn drain(&self) -> Vec<(i32, i32)> {
    self.points.lock().unwrap().drain(..).collect()
  }
}

/// Registers `point` with every still-alive subscription in `subscribers`,
/// dropping any whose `PointSubscription` has since gone away.
fn publish(subscribers: &Mutex<Vec<Weak<Mutex<VecDeque<(i32, i32)>>>>>, point: (i32, i32)) {
  subscribers.lock().unwrap().retain(|subscriber| {
    let Some(points) = subscriber.upgrade() else {
      return false;
    };
    let mut points = points.lock().unwrap();
    if points.len() >= MAX_QUEUED_POINTS {
      points.pop_front();
    }
    points.push_back(point);
    true
  });
}

fn subscribe(subscribers: &Mutex<Vec<Weak<Mutex<VecDeque<(i32, i32)>>>>>) -> PointSubscription {
  let points = Arc::new(Mutex::new(VecDeque::new()));
  subscribers.lock().unwrap().push(Arc::downgrade(&points));
  PointSubscription { points }
}

/// Everything a [`super::BehaviorTask`] needs across trials: the current
/// trial's `TaskConfig.body` (parsed as JSON), a Thalamus client it can use
/// to log back to the server, a sound manager it can use to play audio, and
/// the latest/subscribable touch and gaze samples (see
/// `push_touch`/`push_gaze`, called by `touch_screen::run`/`eye_tracking::run`
/// as samples arrive). Mirrors Python's `TaskContext`, which is constructed
/// once for the whole task controller session — this `TaskContext` is
/// likewise created once (see `main::run_grpc`) and shared for the lifetime
/// of the process, reused for every trial via [`TaskContext::begin_trial`]
/// rather than recreated per trial.
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
  /// The latest touch point received this session, `None` until the first
  /// sample arrives. Updated by `push_touch`.
  latest_touch: Mutex<Option<(i32, i32)>>,
  /// The latest gaze point received this session, same update path as
  /// `latest_touch`.
  latest_gaze: Mutex<Option<(i32, i32)>>,
  /// Every live [`PointSubscription`] returned by `subscribe_to_touch`,
  /// weakly held — see `publish`/`subscribe`.
  touch_subscribers: Mutex<Vec<Weak<Mutex<VecDeque<(i32, i32)>>>>>,
  /// Every live [`PointSubscription`] returned by `subscribe_to_gaze`.
  gaze_subscribers: Mutex<Vec<Weak<Mutex<VecDeque<(i32, i32)>>>>>,
  /// Notified by every `push_touch`/`push_gaze` call — see [`Self::notify`].
  /// Shared by both rather than split into a touch/gaze pair, since a
  /// waiter that cares about one can just re-check its own condition and go
  /// back to waiting on a spurious wakeup from the other.
  notify: Notify,
}

impl TaskContext {
  pub fn new(thalamus_client: ThalamusClient<Channel>, audio_manager: AudioManager) -> Self {
    Self {
      config: Mutex::new(Value::Object(Default::default())),
      thalamus_client,
      log_sender: Mutex::new(None),
      inject_analog_streams: Mutex::new(HashMap::new()),
      audio_manager: Mutex::new(audio_manager),
      latest_touch: Mutex::new(None),
      latest_gaze: Mutex::new(None),
      touch_subscribers: Mutex::new(Vec::new()),
      gaze_subscribers: Mutex::new(Vec::new()),
      notify: Notify::new(),
    }
  }

  /// Records `point` as the latest touch sample (called by `touch_screen::run`
  /// for every point received from the TOUCH_SCREEN analog stream), publishes
  /// it to every live [`PointSubscription`] from `subscribe_to_touch`, and
  /// wakes anyone waiting on [`Self::notify`].
  pub fn push_touch(&self, point: (i32, i32)) {
    *self.latest_touch.lock().unwrap() = Some(point);
    publish(&self.touch_subscribers, point);
    self.notify.notify_waiters();
  }

  /// Records `point` as the latest gaze sample (called by `eye_tracking::run`
  /// for every point received from the OCULOMATIC analog stream, and by
  /// `gfx`'s mouse-simulated gaze), publishes it to every live
  /// [`PointSubscription`] from `subscribe_to_gaze`, and wakes anyone waiting
  /// on [`Self::notify`].
  pub fn push_gaze(&self, point: (i32, i32)) {
    *self.latest_gaze.lock().unwrap() = Some(point);
    publish(&self.gaze_subscribers, point);
    self.notify.notify_waiters();
  }

  /// The most recent touch point, or `None` if none has arrived yet this
  /// session.
  //pub fn touch(&self) -> Option<(i32, i32)> {
  //  *self.latest_touch.lock().unwrap()
  //}

  /// The most recent gaze point, or `None` if none has arrived yet this
  /// session.
  pub fn gaze(&self) -> Option<(i32, i32)> {
    *self.latest_gaze.lock().unwrap()
  }

  /// Notified once for every `push_touch`/`push_gaze` call, via
  /// `Notify::notify_waiters` — so, per its semantics, only wakes a waiter
  /// that had already called `notified()` (even if not yet polled) *before*
  /// the triggering push. The safe way to use this to wait for a touch/gaze
  /// condition without missing a point that arrives in between is to
  /// construct the `Notified` future *before* checking the condition, then
  /// only await it if the condition is still unmet:
  ///
  /// ```ignore
  /// loop {
  ///   let notified = context.notify().notified();
  ///   if condition() { break; }
  ///   notified.await;
  /// }
  /// ```
  ///
  /// (see `vcp_inhibition::VcpInhibitionTask::wait_for`, which combines this
  /// with a deadline via `tokio::select!`).
  pub fn notify(&self) -> &Notify {
    &self.notify
  }

  /// Starts a new live feed of every touch point received from now on —
  /// see [`PointSubscription`]. Lets a task see every sample rather than
  /// just [`Self::touch`]'s latest one.
  //pub fn subscribe_to_touch(&self) -> PointSubscription {
  //  subscribe(&self.touch_subscribers)
  //}

  /// Starts a new live feed of every gaze point received from now on — see
  /// [`PointSubscription`]. Lets a task see every sample rather than just
  /// [`Self::gaze`]'s latest one.
  pub fn subscribe_to_gaze(&self) -> PointSubscription {
    subscribe(&self.gaze_subscribers)
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
  /// start, reopens the `log` stream (Python opens `self.log_queue`/
  /// `log_coroutine` fresh every trial and closes them at trial end — see
  /// `reopen_log_stream`), and resends the `node` handshake on every
  /// already-open `inject_analog` stream (Python's `refresh_streams`).
  /// Called by `task_controller::run` before each trial, since — unlike a
  /// fresh `TaskContext` per trial — a real Thalamus `TaskContext` persists
  /// for the whole session and only swaps out what changes per trial.
  pub async fn begin_trial(&self, config: Value) {
    *self.config.lock().unwrap() = config;
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
}
