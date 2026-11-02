//! Port of Thalamus' state tree: the `ObservableCollection` classes from
//! Python (`thalamus/config.py`) and C++ (`src/thalamus/state.hpp`/
//! `state.cpp`).
//!
//! The tree's nodes are [`Dict`]s and [`List`]s, held through an `Arc` and
//! safe to use from any thread; they're the public interface. A [`Value`]
//! holding a node says which kind it is, as does a [`Coll`], which
//! observers are passed. Inside this module both kinds share one
//! implementation through `Coll`'s private methods. Each method
//! call is atomic, but a sequence of calls isn't: other threads' changes can
//! land in between.
//!
//! Each mutation comes in three forms: `set` (fire and forget), `set_with`
//! (with a callback for when it's done) and `set_direct` (applied locally
//! whatever the remote storage, which is how changes from Thalamus get
//! applied). Without `_direct`, writes go to the remote storage in effect
//! ([`Dict::set_remote_storage`], see [`Bridge`]) instead of being applied
//! locally. Addresses are in the JSONPath dialect [`get_jsonpath`] etc.
//! understand (a port of the C++ grammar).
//!
//! Observers ([`Dict::connect`], [`List::connect`]) see changes to their
//! node and all of its descendants, as C++'s recursive observers do. They
//! run synchronously on whichever thread made the change, with no locks
//! held, so they may freely read or write the tree themselves.
//!
//! Locking rule, which the compiler doesn't check and which a mistake turns
//! into a silent deadlock: hold at most one node's lock at a time, and
//! release it before calling out (observers, callbacks, remote storage),
//! dropping tree values or closures, or locking any other node. (Node
//! equality compares ids, so it doesn't lock.)

// A general-purpose API mirroring the Python/C++/contrib classes; not every
// method has a caller in the executor yet.
#![allow(dead_code)]

mod bridge;
mod jsonpath;
#[cfg(test)]
mod tests;

pub use bridge::{Bridge, apply_changes};
#[allow(unused_imports)]
pub use jsonpath::{delete_jsonpath, get_jsonpath, set_jsonpath};

use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, Weak};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use parking_lot::Mutex;
use serde_json::Value as Json;

pub struct OnDrop {
  action: Option<Box<dyn FnOnce() + Send>>,
}

impl OnDrop {
  pub fn noop() -> OnDrop {
    OnDrop {
      action: Some(Box::new(|| {})),
    }
  }
  pub fn new<F: FnOnce() + Send + 'static>(f: F) -> OnDrop {
    OnDrop {
      action: Some(Box::new(f)),
    }
  }
}

impl Drop for OnDrop {
  fn drop(&mut self) {
    if let Some(action) = self.action.take() {
      action();
    }
  }
}

#[derive(Debug, PartialEq, Clone, Copy)]
pub enum Action {
  Set,
  Delete,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Key {
  Int(i64),
  String(String),
}

impl Key {
  /// This key as one bracketed address segment, the way Python writes
  /// addresses (`f'[{repr(key)}]'`): `[3]`, `['name']`.
  fn address_segment(&self) -> String {
    match self {
      Key::Int(i) => format!("[{i}]"),
      Key::String(s) => format!("[{}]", python_repr(s)),
    }
  }

  fn list_index(&self) -> Option<usize> {
    match self {
      Key::Int(i) => usize::try_from(*i).ok(),
      Key::String(_) => None,
    }
  }
}

/// Python's `repr` of a string: single-quoted unless that needs escaping
/// and double quotes don't.
fn python_repr(s: &str) -> String {
  let quote = if s.contains('\'') && !s.contains('"') { '"' } else { '\'' };
  let mut out = String::with_capacity(s.len() + 2);
  out.push(quote);
  for c in s.chars() {
    if c == quote || c == '\\' {
      out.push('\\');
    }
    out.push(c);
  }
  out.push(quote);
  out
}

#[derive(Debug, Clone, PartialEq)]
pub enum Value {
  Bool(bool),
  Dict(Arc<Dict>),
  List(Arc<List>),
  Float(f64),
  Int(i64),
  String(String),
  Null,
}

impl Value {
  /// Converts parsed JSON, building fresh, detached collections for objects
  /// and arrays.
  pub fn from_json(json: &Json) -> Value {
    match json {
      Json::Null => Value::Null,
      Json::Bool(b) => Value::Bool(*b),
      Json::Number(n) => match n.as_i64() {
        Some(i) => Value::Int(i),
        None => Value::Float(n.as_f64().unwrap_or(f64::NAN)),
      },
      Json::String(s) => Value::String(s.clone()),
      Json::Array(items) => {
        let list = List::new();
        for item in items {
          list.push_direct(Value::from_json(item), ||{});
        }
        Value::List(list)
      }
      Json::Object(map) => {
        let dict = Dict::new();
        for (key, value) in map {
          dict.set_direct(key.as_str(), Value::from_json(value), ||{});
        }
        Value::Dict(dict)
      }
    }
  }

  /// A deep conversion to plain JSON. Non-finite floats become `null`, as
  /// JSON has no representation for them.
  pub fn to_json(&self) -> Json {
    match self {
      Value::Null => Json::Null,
      Value::Bool(b) => Json::Bool(*b),
      Value::Int(i) => Json::from(*i),
      Value::Float(f) => serde_json::Number::from_f64(*f).map_or(Json::Null, Json::Number),
      Value::String(s) => Json::String(s.clone()),
      Value::Dict(c) => c.to_json(),
      Value::List(c) => c.to_json(),
    }
  }

  pub fn as_dict(&self) -> Option<&Arc<Dict>> {
    match self {
      Value::Dict(d) => Some(d),
      _ => None,
    }
  }

  pub fn as_list(&self) -> Option<&Arc<List>> {
    match self {
      Value::List(l) => Some(l),
      _ => None,
    }
  }

  fn as_coll(&self) -> Option<Coll> {
    match self {
      Value::Dict(c) => Some(Coll::Dict(c.clone())),
      Value::List(c) => Some(Coll::List(c.clone())),
      _ => None,
    }
  }
}

pub trait IntoKey {
  fn into_key(self) -> Key;
}
impl IntoKey for Key {
  fn into_key(self) -> Key { self }
}
impl IntoKey for &str {
  fn into_key(self) -> Key { Key::String(self.to_string()) }
}
impl IntoKey for String {
  fn into_key(self) -> Key { Key::String(self) }
}
impl IntoKey for i64 {   // use whatever integer type StateKey::Int holds
  fn into_key(self) -> Key { Key::Int(self) }
}
impl IntoKey for i32 {   // use whatever integer type StateKey::Int holds
  fn into_key(self) -> Key { Key::Int(self as i64) }
}
impl IntoKey for usize {
  fn into_key(self) -> Key { Key::Int(self as i64) }
}

pub trait IntoValue {
  fn into_value(self) -> Value;
}
impl IntoValue for Value {
  fn into_value(self) -> Value { self }
}
impl IntoValue for Arc<Dict> {
  fn into_value(self) -> Value { Value::Dict(self) }
}
impl IntoValue for &Arc<Dict> {
  fn into_value(self) -> Value { Value::Dict(self.clone()) }
}
impl IntoValue for Arc<List> {
  fn into_value(self) -> Value { Value::List(self) }
}
impl IntoValue for &Arc<List> {
  fn into_value(self) -> Value { Value::List(self.clone()) }
}
impl IntoValue for Json {
  fn into_value(self) -> Value { Value::from_json(&self) }
}
impl IntoValue for &str {
  fn into_value(self) -> Value { Value::String(self.to_string()) }
}
impl IntoValue for String {
  fn into_value(self) -> Value { Value::String(self) }
}
impl IntoValue for i64 {   // use whatever integer type StateKey::Int holds
  fn into_value(self) -> Value { Value::Int(self) }
}
impl IntoValue for i32 {   // use whatever integer type StateKey::Int holds
  fn into_value(self) -> Value { Value::Int(self as i64) }
}
impl IntoValue for f32 {   // use whatever integer type StateKey::Int holds
  fn into_value(self) -> Value { Value::Float(self as f64) }
}
impl IntoValue for f64 {   // use whatever integer type StateKey::Int holds
  fn into_value(self) -> Value { Value::Float(self) }
}
impl IntoValue for bool {   // use whatever integer type StateKey::Int holds
  fn into_value(self) -> Value { Value::Bool(self) }
}

impl TryFrom<Value> for f64 {
  type Error = Value;

  fn try_from(value: Value) -> Result<Self, Self::Error> {
    match value {
      Value::Float(f) => Ok(f),
      Value::Int(i) => Ok(i as f64),
      other => Err(other),
    }
  }
}

impl TryFrom<Value> for i64 {
  type Error = Value;

  fn try_from(value: Value) -> Result<Self, Self::Error> {
    match value {
      Value::Int(i) => Ok(i),
      other => Err(other),
    }
  }
}

impl TryFrom<Value> for i32 {
  type Error = Value;

  fn try_from(value: Value) -> Result<Self, Self::Error> {
    match value {
      Value::Int(i) => Ok(i as i32),
      other => Err(other),
    }
  }
}

impl TryFrom<Value> for u64 {
  type Error = Value;

  fn try_from(value: Value) -> Result<Self, Self::Error> {
    match value {
      Value::Int(i) => Ok(i as u64),
      other => Err(other),
    }
  }
}

impl TryFrom<Value> for u32 {
  type Error = Value;

  fn try_from(value: Value) -> Result<Self, Self::Error> {
    match value {
      Value::Int(i) => Ok(i as u32),
      other => Err(other),
    }
  }
}

impl TryFrom<Value> for bool {
  type Error = Value;

  fn try_from(value: Value) -> Result<Self, Self::Error> {
    match value {
      Value::Bool(b) => Ok(b),
      other => Err(other),
    }
  }
}

impl TryFrom<Value> for String {
  type Error = Value;

  fn try_from(value: Value) -> Result<Self, Self::Error> {
    match value {
      Value::String(s) => Ok(s),
      other => Err(other),
    }
  }
}

impl TryFrom<Value> for Arc<Dict> {
  type Error = Value;

  fn try_from(value: Value) -> Result<Self, Self::Error> {
    match value {
      Value::Dict(d) => Ok(d),
      other => Err(other),
    }
  }
}

impl TryFrom<Value> for Arc<List> {
  type Error = Value;

  fn try_from(value: Value) -> Result<Self, Self::Error> {
    match value {
      Value::List(l) => Ok(l),
      other => Err(other),
    }
  }
}

static NEXT_ID: AtomicU64 = AtomicU64::new(0);

fn get_id() -> u64 {
  NEXT_ID.fetch_add(1, Ordering::Relaxed)
}

type Callback = Box<dyn FnOnce() + Send>;
/// The node that changed, the action, and the key and value it applied to.
pub type ObserverFn = dyn Fn(&Coll, Action, &Key, &Value) + Send + Sync;
/// `connected` is cleared by the [`OnDrop`] from [`Dict::connect`].
/// Clearing a flag rather than removing the entry means dropping the
/// `OnDrop` never takes a lock or frees the closure itself, so it's safe
/// anywhere, including inside an observer or while the node is locked.
/// Disconnected observers are pruned, and freed, by the next notify.
#[derive(Clone)]
struct Observer {
  connected: Arc<AtomicBool>,
  callback: Arc<ObserverFn>,
}
pub type RemoteStorage = Arc<dyn Fn(Action, &str, Value, Callback) + Send + Sync>;

#[derive(Clone)]
enum WeakColl {
  Dict(Weak<Dict>),
  List(Weak<List>),
}

impl WeakColl {
  fn upgrade(&self) -> Option<Coll> {
    match self {
      WeakColl::Dict(d) => {
        if let Some(d) = d.upgrade() {
          Some(Coll::Dict(d))
        } else {
          None
        }
      },
      WeakColl::List(d) => {
        if let Some(d) = d.upgrade() {
          Some(Coll::List(d))
        } else {
          None
        }
      },
    }
  }
}

struct DictInner {
  content: HashMap<Key, Value>,
  observers: Vec<Observer>,
  remote_storage: Option<RemoteStorage>,
  parent: WeakColl,
}

struct ListInner {
  content: Vec<Value>,
  observers: Vec<Observer>,
  remote_storage: Option<RemoteStorage>,
  parent: WeakColl,
}

pub struct Dict {
  id: u64,
  inner: Mutex<DictInner>,
}

pub struct List {
  id: u64,
  inner: Mutex<ListInner>,
}

impl Dict {
  pub fn new() -> Arc<Dict> {
    Arc::new(Dict {
      id: get_id(),
      inner: Mutex::new(DictInner { 
        content: HashMap::new(), 
        observers: vec![], 
        remote_storage: None, 
        parent: WeakColl::Dict(Weak::new())
      })
    })
  }

  fn as_value(self: &Arc<Self>) -> Value {
    Coll::Dict(self.clone()).as_value()
  }

  pub fn len(self: &Arc<Self>) -> usize {
    Coll::Dict(self.clone()).len()
  }

  /// A snapshot of the contents.
  pub fn entries(self: &Arc<Self>) -> Vec<(Key, Value)> {
    Coll::Dict(self.clone()).entries()
  }

  /// A deep conversion to plain JSON.
  pub fn to_json(self: &Arc<Self>) -> Json {
    Coll::Dict(self.clone()).to_json()
  }

  fn parent(self: &Arc<Self>) -> WeakColl {
    Coll::Dict(self.clone()).parent()
  }

  fn key_of(self: &Arc<Self>, value: &Value) -> Option<Key> {
    Coll::Dict(self.clone()).key_of(value)
  }

  pub fn set_remote_storage(self: &Arc<Self>, storage: Option<RemoteStorage>) {
    Coll::Dict(self.clone()).set_remote_storage(storage)
  }

  /// Calls `callback` for every change to this node or any descendant,
  /// with the node that changed, until the returned [`OnDrop`] is dropped.
  pub fn connect<F: Fn(&Coll, Action, &Key, &Value) + Send + Sync + 'static>(
    self: &Arc<Self>,
    callback: F,
  ) -> OnDrop {
    Coll::Dict(self.clone()).connect(callback)
  }

  pub fn set(self: &Arc<Self>, key: impl IntoKey, value: impl IntoValue) {
    Coll::Dict(self.clone()).set(key, value)
  }

  pub fn set_with(self: &Arc<Self>, key: impl IntoKey, value: impl IntoValue, callback: impl FnOnce() + Send + 'static) {
    Coll::Dict(self.clone()).set_with(key, value, callback)
  }

  pub fn set_direct(self: &Arc<Self>, key: impl IntoKey, value: impl IntoValue, callback: impl FnOnce()) {
    Coll::Dict(self.clone()).set_direct(key, value, callback)
  }

  pub fn remove(self: &Arc<Self>, key: impl IntoKey) {
    Coll::Dict(self.clone()).remove(key)
  }

  /// Does nothing, the callback included, if there's nothing at `key`.
  pub fn remove_with(self: &Arc<Self>, key: impl IntoKey, callback: impl FnOnce() + Send + 'static) {
    Coll::Dict(self.clone()).remove_with(key, callback)
  }

  /// Does nothing, the callback included, if there's nothing at `key`.
  pub fn remove_direct(self: &Arc<Self>, key: impl IntoKey, callback: impl FnOnce()) {
    Coll::Dict(self.clone()).remove_direct(key, callback)
  }

  /// Removes everything, last entry first so list indices stay valid.
  pub fn clear_direct(self: &Arc<Self>) {
    Coll::Dict(self.clone()).clear_direct()
  }

  pub fn assign_json(self: &Arc<Self>, json: &Json) -> anyhow::Result<()> {
    Coll::Dict(self.clone()).assign_json(json)
  }

  fn assign_entries(self: &Arc<Self>, entries: Vec<(Key, Value)>) {
    Coll::Dict(self.clone()).assign_entries(entries)
  }

  pub fn get(self: &Arc<Self>, key: impl IntoKey) -> Option<Value> {
    Coll::Dict(self.clone()).get(key)
  }

  pub fn contains_key(self: &Arc<Self>, key: impl IntoKey) -> bool {
    Coll::Dict(self.clone()).contains_key(key)
  }
}

impl List {
  pub fn new() -> Arc<List> {
    Arc::new(List {
      id: get_id(),
      inner: Mutex::new(ListInner { 
        content: vec![], 
        observers: vec![], 
        remote_storage: None, 
        parent: WeakColl::Dict(Weak::new())
      })
    })
  }

  fn as_value(self: &Arc<Self>) -> Value {
    Coll::List(self.clone()).as_value()
  }

  pub fn len(self: &Arc<Self>) -> usize {
    Coll::List(self.clone()).len()
  }

  /// A snapshot of the contents.
  pub fn entries(self: &Arc<Self>) -> Vec<(Key, Value)> {
    Coll::List(self.clone()).entries()
  }

  /// A deep conversion to plain JSON.
  pub fn to_json(self: &Arc<Self>) -> Json {
    Coll::List(self.clone()).to_json()
  }

  fn parent(self: &Arc<Self>) -> WeakColl {
    Coll::List(self.clone()).parent()
  }

  fn key_of(self: &Arc<Self>, value: &Value) -> Option<Key> {
    Coll::List(self.clone()).key_of(value)
  }

  pub fn set_remote_storage(self: &Arc<Self>, storage: Option<RemoteStorage>) {
    Coll::List(self.clone()).set_remote_storage(storage)
  }

  /// Calls `callback` for every change to this node or any descendant,
  /// with the node that changed, until the returned [`OnDrop`] is dropped.
  pub fn connect<F: Fn(&Coll, Action, &Key, &Value) + Send + Sync + 'static>(
    self: &Arc<Self>,
    callback: F,
  ) -> OnDrop {
    Coll::List(self.clone()).connect(callback)
  }

  pub fn push(self: &Arc<Self>, value: impl IntoValue) {
    Coll::List(self.clone()).push(value)
  }

  pub fn push_with(self: &Arc<Self>, value: impl IntoValue, callback: impl FnOnce() + Send + 'static) {
    Coll::List(self.clone()).push_with(value, callback)
  }

  pub fn push_direct(self: &Arc<Self>, value: impl IntoValue, callback: impl FnOnce()) {
    Coll::List(self.clone()).push_direct(value, callback)
  }

  pub fn set(self: &Arc<Self>, key: impl IntoKey, value: impl IntoValue) {
    Coll::List(self.clone()).set(key, value)
  }

  pub fn set_with(self: &Arc<Self>, key: impl IntoKey, value: impl IntoValue, callback: impl FnOnce() + Send + 'static) {
    Coll::List(self.clone()).set_with(key, value, callback)
  }

  pub fn set_direct(self: &Arc<Self>, key: impl IntoKey, value: impl IntoValue, callback: impl FnOnce()) {
    Coll::List(self.clone()).set_direct(key, value, callback)
  }

  pub fn remove(self: &Arc<Self>, key: impl IntoKey) {
    Coll::List(self.clone()).remove(key)
  }

  /// Does nothing, the callback included, if there's nothing at `key`.
  pub fn remove_with(self: &Arc<Self>, key: impl IntoKey, callback: impl FnOnce() + Send + 'static) {
    Coll::List(self.clone()).remove_with(key, callback)
  }

  /// Does nothing, the callback included, if there's nothing at `key`.
  pub fn remove_direct(self: &Arc<Self>, key: impl IntoKey, callback: impl FnOnce()) {
    Coll::List(self.clone()).remove_direct(key, callback)
  }

  /// Removes everything, last entry first so list indices stay valid.
  pub fn clear_direct(self: &Arc<Self>) {
    Coll::List(self.clone()).clear_direct()
  }

  pub fn assign_json(self: &Arc<Self>, json: &Json) -> anyhow::Result<()> {
    Coll::List(self.clone()).assign_json(json)
  }

  fn assign_entries(self: &Arc<Self>, entries: Vec<(Key, Value)>) {
    Coll::List(self.clone()).assign_entries(entries)
  }

  pub fn get(self: &Arc<Self>, key: impl IntoKey) -> Option<Value> {
    Coll::List(self.clone()).get(key)
  }

  pub fn contains_key(self: &Arc<Self>, key: impl IntoKey) -> bool {
    Coll::List(self.clone()).contains_key(key)
  }
}

// The contents are copied out before formatting, since formatting them
// formats (and so locks) the child nodes.
impl std::fmt::Debug for List {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    let content = self.inner.lock().content.clone();
    f.debug_tuple("List").field(&self.id).field(&content).finish()
  }
}
impl PartialEq for List {
  fn eq(&self, other: &Self) -> bool {
    self.id == other.id
  }
}

impl std::fmt::Debug for Dict {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    let content = self.inner.lock().content.clone();
    f.debug_tuple("Dict").field(&self.id).field(&content).finish()
  }
}
impl PartialEq for Dict {
  fn eq(&self, other: &Self) -> bool {
    self.id == other.id
  }
}

/// Either kind of node. Outside this module it's only for matching on, to get
/// the [`Dict`] or [`List`]: its methods are private, the implementation the
/// two share. See the module docs for the locking rule.
#[derive(Clone, PartialEq, Debug)]
pub enum Coll {
  Dict(Arc<Dict>),
  List(Arc<List>),
}

macro_rules! on_content {
  ($coll:expr, $content:ident => $body:expr) => {
    match $coll {
      Coll::Dict($content) => $body,
      Coll::List($content) => $body,
    }
  };
}

impl Coll {
  fn as_value(&self) -> Value {
    match self {
      Coll::Dict(d) => Value::Dict(d.clone()),
      Coll::List(d) => Value::List(d.clone()),
    }
  }

  fn is_dict(&self) -> bool {
    matches!(self, Coll::Dict(_))
  }

  fn is_list(&self) -> bool {
    matches!(self, Coll::List(_))
  }

  fn len(&self) -> usize {
    on_content!(self, c => c.inner.lock().content.len())
  }

  fn entries(&self) -> Vec<(Key, Value)> {
    match self {
      Coll::Dict(c) => {
        c.inner.lock().content.iter().map(|(k, v)| (k.clone(), v.clone())).collect()
      },
      Coll::List(c) => {
        c.inner.lock().content.iter().enumerate().map(|(k, v)| (Key::Int(k as i64), v.clone())).collect()
      },
    }
  }

  /// A deep conversion to plain JSON. Integer dict keys become strings.
  fn to_json(&self) -> Json {
    let entries = self.entries();
    if self.is_dict() {
      Json::Object(
        entries
          .into_iter()
          .map(|(key, value)| {
            let key = match key {
              Key::String(s) => s,
              Key::Int(i) => i.to_string(),
            };
            (key, value.to_json())
          })
          .collect(),
      )
    } else {
      Json::Array(entries.into_iter().map(|(_, value)| value.to_json()).collect())
    }
  }

  fn parent(&self) -> WeakColl {
    on_content!(self, c => c.inner.lock().parent.clone())
  }

  fn key_of(&self, value: &Value) -> Option<Key> {
    for (k, v) in self.entries() {
      if v == *value {
        return Some(k);
      }
    }

    None
  }

  fn downgrade(&self) -> WeakColl {
    match self {
      Coll::Dict(d) => {
        WeakColl::Dict(Arc::downgrade(d))
      },
      Coll::List(d) => {
        WeakColl::List(Arc::downgrade(d))
      },
    }
  }

  /// Sets (or with `None`, clears) the remote storage that writes to this
  /// node and its descendants go through.
  fn set_remote_storage(&self, storage: Option<RemoteStorage>) {
    on_content!(self, c => c.inner.lock().remote_storage = storage)
  }

  /// Where a write to `key` here should go instead of being applied: the
  /// remote storage in effect (this node's own, else its nearest ancestor's)
  /// and the address of `key` relative to the node that storage is on.
  /// Found in one walk up the tree, so a node being detached concurrently
  /// can't produce a wrong address: if it's no longer in its parent, this is
  /// `None` and the write applies locally, as it would to any detached node.
  fn remote_target(&self, key: &Key) -> Option<(RemoteStorage, String)> {
    let mut segments = vec![key.address_segment()];
    let mut current = self.clone();
    loop {
      let (storage, parent) = on_content!(&current, c => {
        let inner = c.inner.lock();
        (inner.remote_storage.clone(), inner.parent.upgrade())
      });
      if let Some(storage) = storage {
        segments.reverse();
        return Some((storage, segments.concat()));
      }
      let parent = parent?;
      segments.push(parent.key_of(&current.as_value())?.address_segment());
      current = parent;
    }
  }

  fn connect<F: Fn(&Coll, Action, &Key, &Value) + Send + Sync + 'static>(
    &self,
    callback: F,
  ) -> OnDrop {
    let connected = Arc::new(AtomicBool::new(true));
    let observer = Observer {
      connected: connected.clone(),
      callback: Arc::new(callback),
    };
    on_content!(self, c => c.inner.lock().observers.push(observer));
    OnDrop::new(move || connected.store(false, Ordering::Relaxed))
  }

  /// Calls the connected observers of this node, then of each ancestor in
  /// turn, passing this node as the one that changed.
  fn notify(&self, action: Action, key: &Key, value: &Value) {
    let mut current = Some(self.clone());
    while let Some(node) = current {
      for observer in node.connected_observers() {
        // Checked again in case an earlier observer disconnected this one.
        if observer.connected.load(Ordering::Relaxed) {
          (observer.callback)(self, action, key, value);
        }
      }
      current = node.parent().upgrade();
    }
  }

  /// This node's own connected observers. Disconnected ones are removed
  /// under the same lock (so a concurrent `connect` can't be lost), but
  /// freed only after it's released, since dropping a closure runs whatever
  /// `Drop` code it captured.
  fn connected_observers(&self) -> Vec<Observer> {
    let (connected, _disconnected) = on_content!(self, c => {
      let mut c = c.inner.lock();
      let (connected, disconnected): (Vec<_>, Vec<_>) =
        std::mem::take(&mut c.observers).into_iter().partition(|o| o.connected.load(Ordering::Relaxed));
      c.observers = connected.clone();
      (connected, disconnected)
    });
    connected
  }

  /// Makes this node `child`'s parent, if `child` is a node. Panics if it's
  /// this node or one of its ancestors, as that would make a cycle: a leak,
  /// and notifications that never stop climbing.
  fn claim(&self, child: &Value) {
    let Some(child_coll) = child.as_coll() else {
      return;
    };

    let mut ancestor = Some(self.clone());
    while let Some(node) = ancestor {
      assert!(node != child_coll, "Inserting a collection into itself or one of its descendants");
      ancestor = node.parent().upgrade();
    }

    on_content!(child_coll, c => {
      let mut c = c.inner.lock();
      c.parent = self.downgrade();
      c.remote_storage = None;
    })
  }

  /// Undoes `claim` for a value that's been removed or replaced.
  fn release(child: &Value) {
    let Some(child_coll) = child.as_coll() else {
      return;
    };
    
    on_content!(child_coll, c => {
      let mut c = c.inner.lock();
      c.parent = WeakColl::Dict(Weak::new());
      c.remote_storage = None;
    })
  }

  fn push(&self, value: impl IntoValue) {
    self.push_with(value, ||{});
  }

  fn push_with(&self, value: impl IntoValue, callback: impl FnOnce() + Send + 'static) {
    if self.is_dict() {
      panic!("Cannot push into dict");
    }

    if let Some((remote_storage, address)) = self.remote_target(&Key::Int(self.len() as i64)) {
      remote_storage(Action::Set, &address, value.into_value(), Box::new(callback));
      return;
    }

    self.push_direct(value, callback);
  }

  fn push_direct(&self, value: impl IntoValue, callback: impl FnOnce()) {
    let value = value.into_value();
    self.claim(&value);

    let index = match self {
      Coll::List(c) => {
        let mut c = c.inner.lock();

        let index = Key::Int(c.content.len() as i64);
        c.content.push(value.clone());
        index
      },
      Coll::Dict(_) => panic!("Cannot push into dict"),
    };

    self.notify(Action::Set, &index, &value);
    callback();
  }

  fn set(&self, key: impl IntoKey, value: impl IntoValue) {
    self.set_with(key, value, ||{});
  }

  fn set_with(&self, key: impl IntoKey, value: impl IntoValue, callback: impl FnOnce() + Send + 'static) {
    let key = key.into_key();
    let value = value.into_value();
    if self.get(key.clone()) == Some(value.clone()) {
      callback();
      return;
    }

    if let Some((remote_storage, address)) = self.remote_target(&key) {
      remote_storage(Action::Set, &address, value, Box::new(callback));
      return;
    }

    self.set_direct(key, value, callback);
  }

  fn set_direct(&self, key: impl IntoKey, value: impl IntoValue, callback: impl FnOnce()) {
    let key = key.into_key();
    let value = value.into_value();
    if self.get(key.clone()) == Some(value.clone()) {
      callback();
      return;
    }
    self.claim(&value);

    let replaced = match self {
      Coll::List(c) => {
        let mut c = c.inner.lock();

        match key {
          Key::Int(i) => {
            let i = usize::try_from(i).unwrap();
            let len = c.content.len();
            if i == len {
              c.content.push(value.clone());
              None
            } else if i < len {
              Some(std::mem::replace(&mut c.content[i], value.clone()))
            } else {
              panic!("Index out of bounds: i = {i}, len = {len}");
            }
          },
          ref other => panic!("Index List with non integer: {:?}", other),
        }
      },
      Coll::Dict(c) => c.inner.lock().content.insert(key.clone(), value.clone()),
    };

    if let Some(replaced) = replaced {
      Self::release(&replaced);
    }
    self.notify(Action::Set, &key, &value);
    callback();
  }

  fn remove(&self, key: impl IntoKey) {
    self.remove_with(key, ||{});
  }

  /// Does nothing, the callback included, if there's nothing at `key`.
  fn remove_with(&self, key: impl IntoKey, callback: impl FnOnce() + Send + 'static) {
    let key = key.into_key();
    let Some(current) = self.get(key.clone()) else {
      return;
    };

    if let Some((remote_storage, address)) = self.remote_target(&key) {
      remote_storage(Action::Delete, &address, current, Box::new(callback));
      return;
    }

    self.remove_direct(key, callback);
  }

  /// Does nothing, the callback included, if there's nothing at `key`.
  fn remove_direct(&self, key: impl IntoKey, callback: impl FnOnce()) {
    let key = key.into_key();
    let removed = match self {
      Coll::List(c) => {
        let mut c = c.inner.lock();
        match key.list_index() {
          Some(i) if i < c.content.len() => Some(c.content.remove(i)),
          _ => None,
        }
      },
      Coll::Dict(c) => c.inner.lock().content.remove(&key),
    };
    let Some(removed) = removed else {
      return;
    };

    Self::release(&removed);
    self.notify(Action::Delete, &key, &removed);
    callback();
  }

  /// Removes everything, last entry first so list indices stay valid.
  fn clear_direct(&self) {
    for (key, _) in self.entries().into_iter().rev() {
      self.remove_direct(key, ||{});
    }
  }

  /// Makes this node's contents equal to `json`, which must be an object
  /// for a dict and an array for a list. Collections of the same kind
  /// present on both sides are assigned into recursively, keeping them and
  /// their observers; everything else is replaced or removed.
  fn assign_json(&self, json: &Json) -> anyhow::Result<()> {
    let Some(source) = Value::from_json(json).as_coll() else {
      anyhow::bail!("can't assign {json} to a collection");
    };
    if source.is_dict() != self.is_dict() {
      anyhow::bail!("can't assign {json} to a {}", if self.is_dict() { "dict" } else { "list" });
    }
    self.assign_entries(source.entries());
    Ok(())
  }

  fn assign_entries(&self, entries: Vec<(Key, Value)>) {
    let mut missing: Vec<Key> = self.entries().into_iter().map(|(key, _)| key).collect();
    for (key, value) in entries {
      missing.retain(|k| *k != key);
      match (&value, self.get(key.clone())) {
        (Value::Dict(source), Some(Value::Dict(dest))) => {
          Coll::Dict(dest).assign_entries(Coll::Dict(source.clone()).entries());
        },
        (Value::List(source), Some(Value::List(dest))) => {
          Coll::List(dest).assign_entries(Coll::List(source.clone()).entries());
        },
        _ => self.set_direct(key, value, ||{}),
      }
    }
    for key in missing.into_iter().rev() {
      self.remove_direct(key, ||{});
    }
  }

  /// `None` if there's nothing at `key`, including a key of the wrong type
  /// for this collection.
  fn get(&self, key: impl IntoKey) -> Option<Value> {
    let key = key.into_key();
    match self {
      Coll::List(c) => {
        let c = c.inner.lock();
        key.list_index().and_then(|i| c.content.get(i).cloned())
      },
      Coll::Dict(c) => c.inner.lock().content.get(&key).cloned(),
    }
  }

  fn contains_key(&self, key: impl IntoKey) -> bool {
    self.get(key).is_some()
  }
}
