//! The client half of Thalamus' `observable_bridge_v2`, ported from Python's
//! `ThalamusThread` (thread.py): routes local writes to a [`Dict`] to
//! Thalamus as `ObservableTransaction`s, applies the changes Thalamus sends
//! back directly, and calls each write's callback once
//! Thalamus acknowledges it. Thalamus echoes every change it accepts back to
//! every client, the one that sent it included, and only then acknowledges
//! it — so by the time a callback runs, the change is in the local tree.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use parking_lot::Mutex;

use serde_json::Value as Json;
use tokio::sync::mpsc::UnboundedSender;

use crate::pb::thalamus_grpc::observable_change::Action as ProtoAction;
use crate::pb::thalamus_grpc::{ObservableChange, ObservableTransaction};

use super::{Action, Callback, Dict, Value, delete_jsonpath, set_jsonpath};

pub struct Bridge {
  next_id: AtomicU64,
  /// Callbacks for sent transactions, by transaction id, until acknowledged.
  pending: Mutex<HashMap<u64, Callback>>,
  outbound: UnboundedSender<ObservableTransaction>,
}

impl Bridge {
  /// `outbound` feeds the request side of the `observable_bridge_v2` stream.
  pub fn new(outbound: UnboundedSender<ObservableTransaction>) -> Arc<Bridge> {
    Arc::new(Bridge {
      next_id: AtomicU64::new(1),
      pending: Mutex::new(HashMap::new()),
      outbound,
    })
  }

  /// Makes this bridge `root`'s remote storage, so writes anywhere under it
  /// go to Thalamus instead of being applied locally (Python's
  /// `config.set_remote_storage(self.send_change)`). If the stream has gone
  /// away, the change is applied to `root` locally instead.
  pub fn install(self: &Arc<Self>, root: &Arc<Dict>) {
    let bridge = self.clone();
    let weak_root = Arc::downgrade(root);
    root.set_remote_storage(Some(Arc::new(move |action, address, value, callback| {
      let Err(callback) = bridge.send_change(action, address, &value, callback) else {
        return;
      };
      let Some(root) = weak_root.upgrade() else {
        return;
      };
      let result = match action {
        Action::Set => set_jsonpath(&root, address, &value.to_json()),
        Action::Delete => delete_jsonpath(&root, address),
      };
      if let Err(e) = result {
        tracing::warn!("failed to apply {address} locally: {e}");
      }
      callback();
    })));
  }

  /// Python's `send_change`. If the stream has gone away, hands the callback
  /// back.
  fn send_change(&self, action: Action, address: &str, value: &Value, callback: Callback) -> Result<(), Callback> {
    let id = self.next_id.fetch_add(1, Ordering::Relaxed);
    self.pending.lock().insert(id, callback);
    let transaction = ObservableTransaction {
      changes: vec![ObservableChange {
        address: address.to_string(),
        value: value.to_json().to_string(),
        action: match action {
          Action::Set => ProtoAction::Set,
          Action::Delete => ProtoAction::Delete,
        } as i32,
        ..Default::default()
      }],
      id,
      ..Default::default()
    };
    if self.outbound.send(transaction).is_err() {
      tracing::warn!("observable_bridge_v2 stream is closed; applying {address} locally");
      return Err(self.pending.lock().remove(&id).expect("inserted above"));
    }
    Ok(())
  }

  /// Handles one transaction received from Thalamus: an acknowledgement
  /// calls the matching callback; anything else is applied to `root`.
  pub fn handle(&self, root: &Arc<Dict>, transaction: &ObservableTransaction) {
    if transaction.acknowledged != 0 {
      let callback = self.pending.lock().remove(&transaction.acknowledged);
      if let Some(callback) = callback {
        callback();
      }
      return;
    }
    apply_changes(root, &transaction.changes);
  }
}

/// Applies changes received from Thalamus to `root`, directly, so they
/// aren't sent straight back. An empty address `Set` is the full
/// state (the first message on every stream), assigned into `root` so
/// existing handles and observers stay valid.
pub fn apply_changes(root: &Arc<Dict>, changes: &[ObservableChange]) {
  for change in changes {
    let result = match change.action() {
      ProtoAction::Set => set_jsonpath(root, &change.address, &parse_value(&change.value)),
      ProtoAction::Delete if change.address.trim().is_empty() => {
        root.clear_direct();
        Ok(())
      }
      ProtoAction::Delete => delete_jsonpath(root, &change.address),
    };
    if let Err(e) = result {
      tracing::warn!("failed to apply change {change:?}: {e}");
    }
  }
}

/// Thalamus sometimes serializes string values as the raw (unquoted) string
/// rather than a JSON string literal, so plain `serde_json::from_str` fails
/// for them; treat the raw text as a string in that case.
pub(super) fn parse_value(raw: &str) -> Json {
  serde_json::from_str(raw).unwrap_or_else(|_| Json::String(raw.to_string()))
}
