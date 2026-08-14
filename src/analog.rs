use std::collections::VecDeque;
use std::sync::{Arc, Mutex, Weak};

use tokio::sync::Notify;
use tonic::transport::Channel;

use crate::pb::thalamus_grpc::AnalogResponse;
use crate::pb::thalamus_grpc::thalamus_client::ThalamusClient;
use crate::pb::thalamus_grpc::{AnalogRequest, NodeSelector};

/// The last raw sample in the named span (whichever of `data`/`int_data`/
/// `ulong_data` is populated). Applies the span's `scale`/`offset`
/// (`raw * scale + offset`) only if `response.is_transformed` — otherwise
/// `scale`/`offset` are unset (both `0.0`) and the raw sample is already the
/// real value, so applying them would zero it out. `None` if no span with
/// that name is present, or its range is empty.
pub fn last_span_value(response: &AnalogResponse, name: &str) -> Option<f64> {
  let span = response.spans.iter().find(|span| span.name == name)?;
  let raw = last_raw_sample(response, span.begin as usize, span.end as usize)?;
  if response.is_transformed {
    Some(raw * span.scale + span.offset)
  } else {
    Some(raw)
  }
}

fn last_raw_sample(response: &AnalogResponse, begin: usize, end: usize) -> Option<f64> {
  let last_index = end.checked_sub(1)?;
  if last_index < begin {
    return None;
  }
  if response.is_int_data {
    response.int_data.get(last_index).map(|v| *v as f64)
  } else if response.is_ulong_data {
    response.ulong_data.get(last_index).map(|v| *v as f64)
  } else {
    response.data.get(last_index).copied()
  }
}

/// Cap on how many messages/points a single [`DataSubscription`] or
/// [`PointSubscription`] buffers before being drained: once full, the
/// oldest entry is dropped to make room for the newest — mirrors
/// `behavior_task::task_context::MAX_QUEUED_POINTS` (kept separate since
/// that one caps extracted `(i32, i32)` touch/gaze points, not this
/// module's raw `AnalogResponse` messages or `(f64, f64)` samples).
const MAX_QUEUED_MESSAGES: usize = 3600;

/// A shared `analog` RPC stream for one node/channel-list combination —
/// see `TaskContext::connect`, which opens (or reuses) one over its shared
/// Thalamus client. Every [`Self::subscribe`] call gets its own
/// [`DataSubscription`] fed by the same underlying stream and background
/// reader task ([`run_connection`]), so N subscribers cost one gRPC stream
/// rather than N. The reader task holds its own `Arc<Connection>` for as
/// long as the stream stays open, so the connection — and the stream —
/// outlives any individual `subscribe` caller dropping their reference; it
/// only actually closes when the stream itself ends or errors.
pub struct Connection {
  subscribers: Mutex<Vec<Weak<DataSubscription>>>,
}

impl Connection {
  /// Used only by `TaskContext::connect`, which owns the registry deciding
  /// when a fresh `Connection` (versus reusing a live one) is needed.
  pub(crate) fn new() -> Arc<Self> {
    Arc::new(Self {
      subscribers: Mutex::new(Vec::new()),
    })
  }

  /// Starts a new live feed of every message read from this connection's
  /// stream from now on — see [`DataSubscription`]. Stops receiving (and
  /// lets the connection reclaim its buffer) as soon as the returned
  /// `Arc<DataSubscription>` is dropped: the connection only holds a `Weak`
  /// reference to it.
  pub fn subscribe(&self) -> Arc<DataSubscription> {
    let subscription = DataSubscription::new();
    self
      .subscribers
      .lock()
      .unwrap()
      .push(Arc::downgrade(&subscription));
    subscription
  }

  /// Pushes `message` to every still-alive subscriber (see
  /// [`DataSubscription::push`], which also wakes anyone waiting on that
  /// subscription's own `Notify`), dropping any whose `DataSubscription`
  /// has since gone away.
  fn publish(&self, message: AnalogResponse) {
    self.subscribers.lock().unwrap().retain(|subscriber| {
      let Some(subscription) = subscriber.upgrade() else {
        return false;
      };
      subscription.push(message.clone());
      true
    });
  }

  /// Like [`Self::subscribe`], but extracts `(x_channel, y_channel)` out of
  /// each raw message (see [`last_span_value`]) instead of handing back the
  /// raw message — skipping any message missing either channel — and
  /// passes the result through `transform` first if given (e.g. to flip an
  /// axis or re-center on screen), otherwise forwarding it as-is. Spawns a
  /// background task that forwards from a raw [`Self::subscribe`] feed into
  /// the returned [`PointSubscription`], waking on the raw feed's own
  /// `Notify` (not this connection's — there isn't one; each subscription
  /// carries its own, see [`DataSubscription`]/[`PointSubscription`]). The
  /// forwarder only holds a `Weak` reference to the `PointSubscription` it
  /// feeds (unlike [`Self::publish`]'s subscriber list, this one has just
  /// the one, so there's no separate prune-on-next-publish step to rely
  /// on): once the caller drops the returned `Arc`, the task notices on its
  /// next wakeup and exits, so a subscription actually stops costing
  /// anything once dropped, rather than being kept alive forever by its own
  /// forwarder.
  pub fn subscribe_points<F>(
    &self,
    x_channel: impl Into<String>,
    y_channel: impl Into<String>,
    transform: Option<F>,
  ) -> Arc<PointSubscription>
  where
    F: Fn(f64, f64) -> (f64, f64) + Send + 'static,
  {
    let x_channel = x_channel.into();
    let y_channel = y_channel.into();
    let raw_queue = self.subscribe();

    let subscription = PointSubscription::new();
    let weak_subscription = Arc::downgrade(&subscription);

    tokio::spawn(async move {
      loop {
        let notified = raw_queue.notify().notified();
        let Some(subscription) = weak_subscription.upgrade() else {
          return; // caller dropped the PointSubscription: stop forwarding
        };
        for message in raw_queue.drain() {
          let (Some(x), Some(y)) = (
            last_span_value(&message, &x_channel),
            last_span_value(&message, &y_channel),
          ) else {
            continue; // either channel missing from this message
          };
          let point = match &transform {
            Some(f) => f(x, y),
            None => (x, y),
          };
          subscription.push(point);
        }
        drop(subscription); // don't hold a strong ref across the await below
        notified.await;
      }
    });

    subscription
  }
}

/// A live feed of `T`s, with its own [`Notify`] fired after each push —
/// the shape shared by [`DataSubscription`] (raw `analog` messages, from
/// [`Connection::subscribe`]) and [`PointSubscription`] (transformed
/// `(f64, f64)` points, from [`Connection::subscribe_points`]; see that
/// method's doc comment for why *its* subscription needs a `Notify` of its
/// own rather than sharing its source `DataSubscription`'s: the point
/// pushed there isn't available until an extra async hop — the forwarder
/// task's drain/transform — after the raw message it came from arrives, so
/// a caller waking on some earlier, shared signal could find nothing there
/// yet).
pub struct Subscription<T> {
  points: Mutex<VecDeque<T>>,
  notify: Notify,
}

impl<T: Clone> Subscription<T> {
  pub(crate) fn new() -> Arc<Self> {
    Arc::new(Self {
      points: Mutex::new(VecDeque::new()),
      notify: Notify::new(),
    })
  }

  /// Drains and returns every item received since the last call to this
  /// method (or since subscribing, for the first call), oldest first.
  pub fn drain(&self) -> Vec<T> {
    self.points.lock().unwrap().drain(..).collect()
  }

  /// The most recently pushed item, without draining it — for callers that
  /// just want to peek the latest value (e.g. an operator-view overlay)
  /// without consuming items a drain-based consumer still needs to see.
  pub fn latest(&self) -> Option<T> {
    self.points.lock().unwrap().back().cloned()
  }

  /// Notified after every item pushed here — construct the `Notified`
  /// future before checking/draining to avoid missing an item pushed in
  /// between:
  ///
  /// ```ignore
  /// loop {
  ///   let notified = subscription.notify().notified();
  ///   if condition() { break; }
  ///   notified.await;
  /// }
  /// ```
  pub fn notify(&self) -> &Notify {
    &self.notify
  }

  pub(crate) fn push(&self, item: T) {
    let mut points = self.points.lock().unwrap();
    if points.len() >= MAX_QUEUED_MESSAGES {
      points.pop_front();
    }
    points.push_back(item);
    drop(points);
    self.notify.notify_waiters();
  }
}

/// A live feed of raw `analog` messages from a [`Connection`] — see
/// [`Connection::subscribe`]. Each message is exactly as received from the
/// stream; extracting a particular channel's value out of one is up to the
/// subscriber (see [`last_span_value`]) — this just lets many subscribers
/// share the one stream those values are read from.
pub type DataSubscription = Subscription<AnalogResponse>;

/// A live, transformed feed of `(f64, f64)` points derived from a
/// [`Connection`]'s raw messages — see [`Connection::subscribe_points`].
pub type PointSubscription = Subscription<(f64, f64)>;

/// A broadcast point source with no backing gRPC stream — the same
/// subscriber fan-out shape as [`Connection`], minus the network bits.
/// Backs `TaskContext::inject_touch`/`inject_gaze` (used by gfx's
/// mouse-simulated gaze): `TaskContext::subscribe_to_touch`/
/// `subscribe_to_gaze` merge a `PointBroadcast`'s subscription with a
/// node's real `analog`-sourced one, so an injected point reaches the same
/// consumers a real sample would, rather than needing its own separate
/// path.
pub struct PointBroadcast {
  subscribers: Mutex<Vec<Weak<PointSubscription>>>,
}

impl PointBroadcast {
  pub fn new() -> Self {
    Self {
      subscribers: Mutex::new(Vec::new()),
    }
  }

  /// Starts a new live feed of every point [`Self::publish`]ed from now on.
  pub fn subscribe(&self) -> Arc<PointSubscription> {
    let subscription = PointSubscription::new();
    self
      .subscribers
      .lock()
      .unwrap()
      .push(Arc::downgrade(&subscription));
    subscription
  }

  /// Pushes `point` to every still-alive subscriber, dropping any whose
  /// subscription has since gone away.
  pub fn publish(&self, point: (f64, f64)) {
    self.subscribers.lock().unwrap().retain(|subscriber| {
      let Some(subscription) = subscriber.upgrade() else {
        return false;
      };
      subscription.push(point);
      true
    });
  }
}

/// Opens `node`'s `analog` RPC over `client`, requesting only `channels`
/// (via `AnalogRequest.channel_names`), and publishes every inbound
/// message, as-is, to `connection`'s subscribers (see
/// [`Connection::subscribe`]/[`DataSubscription`]) — extracting a
/// particular channel's value out of one is up to the subscriber (see
/// [`last_span_value`]); this only narrows what the server sends in the
/// first place. Called by `TaskContext::connect` to back a
/// freshly-created [`Connection`]. Runs until the stream ends or errors, at
/// which point `connection` — this task's only strong reference to it,
/// once every `subscribe` caller and the owning registry's own `Weak`
/// entry are accounted for — is dropped.
pub(crate) async fn run_connection(
  mut client: ThalamusClient<Channel>,
  node: NodeSelector,
  channels: Vec<String>,
  connection: Arc<Connection>,
) -> anyhow::Result<()> {
  tracing::info!(
    "connecting to analog stream (node name={:?} type={:?}, channels={channels:?})",
    node.name,
    node.r#type
  );
  let request = AnalogRequest {
    node: Some(node),
    channels: Vec::new(),
    channel_names: channels,
  };
  let response = client.analog(request).await?;
  let mut inbound = response.into_inner();

  while let Some(message) = inbound.message().await? {
    connection.publish(message);
  }

  Ok(())
}
