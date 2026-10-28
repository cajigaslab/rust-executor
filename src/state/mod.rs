//! Port of Thalamus' state tree: the `ObservableCollection` classes from
//! Python (`thalamus/config.py`) and C++ (`src/thalamus/state.hpp`/
//! `state.cpp`), exposed through the same API as thalamus-contrib's
//! FFI-backed `State` (`thalamus-contrib/rust/src/api.rs`) — `State`,
//! `StateValue`, `StateKey`, `StateAction`, `StateEntry`, `IntoStateKey`,
//! `IntoStateValue`, `OnDrop` — so code written against one reads the same
//! against the other.
//!
//! A [`State`] is a cheap, clonable handle to one dict or list node in the
//! tree. Beyond the contrib API it carries the rest of what the Python/C++
//! classes do: change callbacks and the `from_remote` flag on every
//! mutation (`*_with` methods), remote storage ([`State::set_remote_storage`]
//! — how writes get routed to Thalamus instead of being applied locally, see
//! [`Bridge`]), addresses, plain (non-recursive) observers, observers that
//! remove themselves (`remove_when`), recap on connect, list `insert`/`pop`/
//! `remove_val`/`extend`, `update`, `clear`, `copy`, JSON conversion, and
//! JSONPath get/set/delete ([`get_jsonpath`] etc., a port of the C++
//! grammar).
//!
//! Observers run synchronously on whichever thread made the change, after
//! the tree's own locks have been released, so they may freely read or
//! write the tree themselves. An observer that (directly or indirectly)
//! triggers itself isn't re-entered: the nested notification is skipped
//! with a warning.

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

use std::cell::RefCell;
use std::collections::HashSet;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Weak};

use indexmap::IndexMap;
use parking_lot::{Mutex, ReentrantMutex};
use serde_json::Value as Json;

#[derive(Debug, PartialEq, Clone, Copy)]
pub enum StateAction {
  Set,
  Delete,
}

#[derive(Debug, PartialEq, Eq, Hash, Clone, PartialOrd, Ord)]
pub enum StateKey {
  Int(i64),
  String(String),
}

impl StateKey {
  /// This key as one bracketed address segment, the way Python writes
  /// addresses (`f'[{repr(key)}]'`): `[3]`, `['name']`.
  fn address_segment(&self) -> String {
    match self {
      StateKey::Int(i) => format!("[{i}]"),
      StateKey::String(s) => format!("[{}]", python_repr(s)),
    }
  }

  fn list_index(&self) -> Option<usize> {
    match self {
      StateKey::Int(i) => usize::try_from(*i).ok(),
      StateKey::String(_) => None,
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

/// A value stored in the tree. As in contrib, `Dict`/`List` hold a [`State`]
/// handle and compare by identity.
#[derive(Debug, PartialEq, Clone)]
pub enum StateValue {
  Bool(bool),
  Dict(State),
  Float(f64),
  Int(i64),
  List(State),
  String(String),
  Null,
}

impl StateValue {
  /// Converts parsed JSON, building fresh (detached) `State`s for objects
  /// and arrays.
  pub fn from_json(json: &Json) -> StateValue {
    match json {
      Json::Null => StateValue::Null,
      Json::Bool(b) => StateValue::Bool(*b),
      Json::Number(n) => match n.as_i64() {
        Some(i) => StateValue::Int(i),
        None => StateValue::Float(n.as_f64().unwrap_or(f64::NAN)),
      },
      Json::String(s) => StateValue::String(s.clone()),
      Json::Array(_) | Json::Object(_) => State::from_json(json)
        .expect("arrays and objects always convert")
        .as_value(),
    }
  }

  /// Python's `unwrap()`: a deep conversion to plain JSON. Non-finite floats
  /// become `null`, as JSON has no representation for them.
  pub fn to_json(&self) -> Json {
    match self {
      StateValue::Null => Json::Null,
      StateValue::Bool(b) => Json::Bool(*b),
      StateValue::Int(i) => Json::from(*i),
      StateValue::Float(f) => serde_json::Number::from_f64(*f).map_or(Json::Null, Json::Number),
      StateValue::String(s) => Json::String(s.clone()),
      StateValue::Dict(state) | StateValue::List(state) => state.to_json(),
    }
  }

  /// The `State` held by a `Dict`/`List` value.
  pub fn as_state(&self) -> Option<&State> {
    match self {
      StateValue::Dict(state) | StateValue::List(state) => Some(state),
      _ => None,
    }
  }

  pub fn is_null(&self) -> bool {
    matches!(self, StateValue::Null)
  }

  /// A deep copy: collections are copied into new, detached `State`s.
  pub fn deep_copy(&self) -> StateValue {
    match self {
      StateValue::Dict(state) | StateValue::List(state) => state.copy().as_value(),
      other => other.clone(),
    }
  }
}

impl fmt::Display for StateValue {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    write!(f, "{}", self.to_json())
  }
}

impl TryFrom<StateValue> for f64 {
  type Error = StateValue;

  fn try_from(value: StateValue) -> Result<Self, Self::Error> {
    match value {
      StateValue::Float(f) => Ok(f),
      StateValue::Int(i) => Ok(i as f64),
      other => Err(other),
    }
  }
}

impl TryFrom<StateValue> for i64 {
  type Error = StateValue;

  fn try_from(value: StateValue) -> Result<Self, Self::Error> {
    match value {
      StateValue::Int(i) => Ok(i),
      other => Err(other),
    }
  }
}

impl TryFrom<StateValue> for i32 {
  type Error = StateValue;

  fn try_from(value: StateValue) -> Result<Self, Self::Error> {
    match value {
      StateValue::Int(i) => Ok(i as i32),
      other => Err(other),
    }
  }
}

impl TryFrom<StateValue> for u64 {
  type Error = StateValue;

  fn try_from(value: StateValue) -> Result<Self, Self::Error> {
    match value {
      StateValue::Int(i) => Ok(i as u64),
      other => Err(other),
    }
  }
}

impl TryFrom<StateValue> for u32 {
  type Error = StateValue;

  fn try_from(value: StateValue) -> Result<Self, Self::Error> {
    match value {
      StateValue::Int(i) => Ok(i as u32),
      other => Err(other),
    }
  }
}

impl TryFrom<StateValue> for String {
  type Error = StateValue;

  fn try_from(value: StateValue) -> Result<Self, Self::Error> {
    match value {
      StateValue::String(s) => Ok(s),
      other => Err(other),
    }
  }
}

impl TryFrom<StateValue> for bool {
  type Error = StateValue;

  fn try_from(value: StateValue) -> Result<Self, Self::Error> {
    match value {
      StateValue::Bool(b) => Ok(b),
      other => Err(other),
    }
  }
}

impl TryFrom<StateValue> for State {
  type Error = StateValue;

  fn try_from(value: StateValue) -> Result<Self, Self::Error> {
    match value {
      StateValue::Dict(state) | StateValue::List(state) => Ok(state),
      other => Err(other),
    }
  }
}

pub trait IntoStateKey {
  fn into_state_key(self) -> StateKey;
}

impl IntoStateKey for StateKey {
  fn into_state_key(self) -> StateKey {
    self
  }
}

impl IntoStateKey for &str {
  fn into_state_key(self) -> StateKey {
    StateKey::String(self.to_string())
  }
}

impl IntoStateKey for &String {
  fn into_state_key(self) -> StateKey {
    StateKey::String(self.clone())
  }
}

impl IntoStateKey for String {
  fn into_state_key(self) -> StateKey {
    StateKey::String(self)
  }
}

impl IntoStateKey for i64 {
  fn into_state_key(self) -> StateKey {
    StateKey::Int(self)
  }
}

impl IntoStateKey for i32 {
  fn into_state_key(self) -> StateKey {
    StateKey::Int(self as i64)
  }
}

impl IntoStateKey for usize {
  fn into_state_key(self) -> StateKey {
    StateKey::Int(self as i64)
  }
}

pub trait IntoStateValue {
  fn into_state_value(self) -> StateValue;
}

impl IntoStateValue for StateValue {
  fn into_state_value(self) -> StateValue {
    self
  }
}

impl IntoStateValue for &str {
  fn into_state_value(self) -> StateValue {
    StateValue::String(self.to_string())
  }
}

impl IntoStateValue for String {
  fn into_state_value(self) -> StateValue {
    StateValue::String(self)
  }
}

impl IntoStateValue for i64 {
  fn into_state_value(self) -> StateValue {
    StateValue::Int(self)
  }
}

impl IntoStateValue for i32 {
  fn into_state_value(self) -> StateValue {
    StateValue::Int(self as i64)
  }
}

impl IntoStateValue for usize {
  fn into_state_value(self) -> StateValue {
    StateValue::Int(self as i64)
  }
}

impl IntoStateValue for f32 {
  fn into_state_value(self) -> StateValue {
    StateValue::Float(self as f64)
  }
}

impl IntoStateValue for f64 {
  fn into_state_value(self) -> StateValue {
    StateValue::Float(self)
  }
}

impl IntoStateValue for bool {
  fn into_state_value(self) -> StateValue {
    StateValue::Bool(self)
  }
}

impl IntoStateValue for State {
  fn into_state_value(self) -> StateValue {
    self.as_value()
  }
}

impl IntoStateValue for &State {
  fn into_state_value(self) -> StateValue {
    self.as_value()
  }
}

impl IntoStateValue for &Json {
  fn into_state_value(self) -> StateValue {
    StateValue::from_json(self)
  }
}

impl IntoStateValue for Json {
  fn into_state_value(self) -> StateValue {
    StateValue::from_json(&self)
  }
}

/// Called once a change has been applied — immediately for a local change,
/// or once Thalamus acknowledges it when the change went through remote
/// storage. Python/C++'s `callback` argument; `Fn` rather than `FnOnce`
/// because, as there, compound operations (`assign`, `merge`, `extend`,
/// ...) pass the same callback to every change they make.
pub type Callback = Arc<dyn Fn() + Send + Sync>;

/// Python's `remove_when`: checked before each notification, and once it
/// returns true the observer is removed instead of called.
pub type RemoveWhen = Box<dyn Fn() -> bool + Send + Sync>;

/// Python's `RemoteStorage`/C++'s `remote_storage`: given a change about to
/// be made (action, address, new or removed value, completion callback),
/// returns true if it took responsibility for the change, in which case it
/// isn't applied locally — it's expected to come back with `from_remote`
/// set, typically via [`Bridge`]. Returning false applies the change
/// locally as usual.
pub type RemoteStorage = Arc<dyn Fn(StateAction, &str, &StateValue, Callback) -> bool + Send + Sync>;

fn noop_callback() -> Callback {
  Arc::new(|| {})
}

/// Runs its action when dropped — returned by [`State::connect`]/
/// [`State::observe`], disconnecting the observer. As in contrib, keep it
/// alive for as long as the observer should stay connected.
pub struct OnDrop {
  action: Option<Box<dyn FnOnce() + Send>>,
}

impl OnDrop {
  pub fn noop() -> OnDrop {
    OnDrop { action: None }
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

pub struct StateEntry {
  pub key: StateKey,
  pub val: StateValue,
}

/// Iterates over a snapshot of a `State`'s entries taken when iteration
/// started.
pub struct StateIter {
  entries: std::vec::IntoIter<(StateKey, StateValue)>,
}

impl Iterator for StateIter {
  type Item = StateEntry;

  fn next(&mut self) -> Option<Self::Item> {
    self.entries.next().map(|(key, val)| StateEntry { key, val })
  }
}

impl IntoIterator for &State {
  type Item = StateEntry;
  type IntoIter = StateIter;

  fn into_iter(self) -> Self::IntoIter {
    StateIter {
      entries: self.entries().into_iter(),
    }
  }
}

type ChangeFn = dyn FnMut(StateAction, StateKey, StateValue) + Send;
type RecursiveChangeFn = dyn FnMut(State, StateAction, StateKey, StateValue) + Send;

struct Observer<F: ?Sized> {
  id: u64,
  /// Reentrant so an observer that triggers itself on the same thread hits
  /// the `RefCell` (and is skipped) instead of deadlocking.
  callback: Arc<ReentrantMutex<RefCell<Box<F>>>>,
  remove_when: Option<Arc<dyn Fn() -> bool + Send + Sync>>,
}

impl<F: ?Sized> Clone for Observer<F> {
  fn clone(&self) -> Self {
    Self {
      id: self.id,
      callback: self.callback.clone(),
      remove_when: self.remove_when.clone(),
    }
  }
}

impl<F: ?Sized> Observer<F> {
  /// Calls this observer, unless its `remove_when` says it's done, in which
  /// case returns false so the caller removes it.
  fn call(&self, invoke: impl FnOnce(&mut F)) -> bool {
    if self.remove_when.as_ref().is_some_and(|remove_when| remove_when()) {
      return false;
    }
    let guard = self.callback.lock();
    match guard.try_borrow_mut() {
      Ok(mut callback) => invoke(&mut **callback),
      Err(_) => tracing::warn!("state observer triggered itself; skipping the nested notification"),
    }
    true
  }
}

/// Calls every observer in `observers`, from a snapshot so they can add or
/// remove observers (themselves included) while running.
fn dispatch<F: ?Sized>(observers: &Mutex<Vec<Observer<F>>>, mut invoke: impl FnMut(&mut F)) {
  let snapshot = observers.lock().clone();
  for observer in snapshot {
    if !observer.call(&mut invoke) {
      observers.lock().retain(|o| o.id != observer.id);
    }
  }
}

static NEXT_OBSERVER_ID: AtomicU64 = AtomicU64::new(1);

enum Content {
  Dict(IndexMap<StateKey, StateValue>),
  List(Vec<StateValue>),
}

struct Node {
  is_dict: bool,
  content: Mutex<Content>,
  parent: Mutex<Weak<Node>>,
  /// C++'s `changed` signal/Python's `observers`: changes to this node's
  /// own entries.
  observers: Mutex<Vec<Observer<ChangeFn>>>,
  /// C++'s `recursive_changed`/Python's `recursive_observers`: changes
  /// anywhere in this node's subtree.
  recursive_observers: Mutex<Vec<Observer<RecursiveChangeFn>>>,
  /// Python/C++ copy this down to every descendant; here it's looked up by
  /// walking up to the nearest ancestor that has one (see
  /// [`State::remote_storage`]), which comes to the same thing.
  remote_storage: Mutex<Option<RemoteStorage>>,
}

/// A handle to one dict or list in a state tree. Clones share the same
/// node; equality and hashing are by identity, as in contrib.
#[derive(Clone)]
pub struct State(Arc<Node>);

impl PartialEq for State {
  fn eq(&self, other: &Self) -> bool {
    Arc::ptr_eq(&self.0, &other.0)
  }
}

impl Eq for State {}

impl Hash for State {
  fn hash<H: Hasher>(&self, state: &mut H) {
    Arc::as_ptr(&self.0).hash(state);
  }
}

impl fmt::Debug for State {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.debug_tuple("State").field(&self.to_json()).finish()
  }
}

impl State {
  fn with_content(content: Content) -> State {
    State(Arc::new(Node {
      is_dict: matches!(content, Content::Dict(_)),
      content: Mutex::new(content),
      parent: Mutex::new(Weak::new()),
      observers: Mutex::new(Vec::new()),
      recursive_observers: Mutex::new(Vec::new()),
      remote_storage: Mutex::new(None),
    }))
  }

  /// Creates a new, detached, empty list. Attach it elsewhere with `set()`.
  pub fn make_list() -> State {
    State::with_content(Content::List(Vec::new()))
  }

  /// Creates a new, detached, empty dict. Attach it elsewhere with `set()`.
  pub fn make_dict() -> State {
    State::with_content(Content::Dict(IndexMap::new()))
  }

  /// Builds a new, detached tree from a JSON object or array. Errors for
  /// any other JSON value, which has no `State` representation.
  pub fn from_json(json: &Json) -> anyhow::Result<State> {
    let (state, entries): (State, Vec<(StateKey, StateValue)>) = match json {
      Json::Object(map) => (
        State::make_dict(),
        map
          .iter()
          .map(|(k, v)| (StateKey::String(k.clone()), StateValue::from_json(v)))
          .collect(),
      ),
      Json::Array(items) => (
        State::make_list(),
        items
          .iter()
          .enumerate()
          .map(|(i, v)| (StateKey::Int(i as i64), StateValue::from_json(v)))
          .collect(),
      ),
      other => anyhow::bail!("only JSON objects and arrays convert to a State, got {other}"),
    };
    for (_, value) in &entries {
      state.attach(value);
    }
    match &mut *state.0.content.lock() {
      Content::Dict(map) => map.extend(entries),
      Content::List(list) => list.extend(entries.into_iter().map(|(_, v)| v)),
    }
    Ok(state)
  }

  /// Python's `unwrap()`/C++'s `to_json()`: a deep conversion to plain
  /// JSON. Integer dict keys (which only C++ produces) become strings.
  pub fn to_json(&self) -> Json {
    let entries = self.entries();
    if self.is_dict() {
      Json::Object(
        entries
          .into_iter()
          .map(|(key, value)| {
            let key = match key {
              StateKey::String(s) => s,
              StateKey::Int(i) => i.to_string(),
            };
            (key, value.to_json())
          })
          .collect(),
      )
    } else {
      Json::Array(entries.into_iter().map(|(_, value)| value.to_json()).collect())
    }
  }

  /// Python's `copy()`: a deep copy, detached from any tree and without
  /// observers or remote storage.
  pub fn copy(&self) -> State {
    let copy = if self.is_dict() { State::make_dict() } else { State::make_list() };
    let entries: Vec<(StateKey, StateValue)> = self
      .entries()
      .into_iter()
      .map(|(key, value)| (key, value.deep_copy()))
      .collect();
    for (_, value) in &entries {
      copy.attach(value);
    }
    match &mut *copy.0.content.lock() {
      Content::Dict(map) => map.extend(entries),
      Content::List(list) => list.extend(entries.into_iter().map(|(_, v)| v)),
    }
    copy
  }

  /// This state wrapped as a `StateValue::Dict`/`StateValue::List`.
  pub fn as_value(&self) -> StateValue {
    if self.is_dict() {
      StateValue::Dict(self.clone())
    } else {
      StateValue::List(self.clone())
    }
  }

  pub fn is_dict(&self) -> bool {
    self.0.is_dict
  }

  pub fn is_list(&self) -> bool {
    !self.0.is_dict
  }

  pub fn ptr_eq(&self, other: &State) -> bool {
    self == other
  }

  pub fn len(&self) -> usize {
    match &*self.0.content.lock() {
      Content::Dict(map) => map.len(),
      Content::List(list) => list.len(),
    }
  }

  pub fn is_empty(&self) -> bool {
    self.len() == 0
  }

  /// A snapshot of this state's entries, in order (insertion order for
  /// dicts, as in Python).
  pub fn entries(&self) -> Vec<(StateKey, StateValue)> {
    match &*self.0.content.lock() {
      Content::Dict(map) => map.iter().map(|(k, v)| (k.clone(), v.clone())).collect(),
      Content::List(list) => list
        .iter()
        .enumerate()
        .map(|(i, v)| (StateKey::Int(i as i64), v.clone()))
        .collect(),
    }
  }

  pub fn keys(&self) -> impl Iterator<Item = StateKey> {
    self.into_iter().map(|e| e.key)
  }

  pub fn values(&self) -> impl Iterator<Item = StateValue> {
    self.into_iter().map(|e| e.val)
  }

  pub fn get(&self, index: impl IntoStateKey) -> Option<StateValue> {
    let key = index.into_state_key();
    match &*self.0.content.lock() {
      Content::Dict(map) => map.get(&key).cloned(),
      Content::List(list) => key.list_index().and_then(|i| list.get(i).cloned()),
    }
  }

  /// Python's `get(key, default)`.
  pub fn get_or(&self, index: impl IntoStateKey, default: impl IntoStateValue) -> StateValue {
    self.get(index).unwrap_or_else(|| default.into_state_value())
  }

  pub fn contains_key(&self, val: impl IntoStateKey) -> bool {
    self.get(val).is_some()
  }

  pub fn contains_val(&self, val: impl IntoStateValue) -> bool {
    let rendered = val.into_state_value();
    self.values().any(|v| v == rendered)
  }

  pub fn parent(&self) -> Option<State> {
    self.0.parent.lock().upgrade().map(State)
  }

  /// The root of the tree this state belongs to; this state itself if it
  /// has no parent.
  pub fn root(&self) -> State {
    let mut root = self.clone();
    while let Some(parent) = root.parent() {
      root = parent;
    }
    root
  }

  /// The key under which `val` is stored in this state, if it's a direct
  /// child.
  pub fn key_of(&self, val: &State) -> Option<StateKey> {
    let is_val = |v: &StateValue| v.as_state() == Some(val);
    match &*self.0.content.lock() {
      Content::Dict(map) => map.iter().find(|(_, v)| is_val(v)).map(|(k, _)| k.clone()),
      Content::List(list) => list.iter().position(is_val).map(|i| StateKey::Int(i as i64)),
    }
  }

  /// Python's `key_in_parent()`.
  pub fn key_in_parent(&self) -> Option<StateKey> {
    self.parent()?.key_of(self)
  }

  /// Python's `is_descendent()`: whether `root` is this state or one of its
  /// ancestors.
  pub fn is_descendent(&self, root: &State) -> bool {
    let mut current = Some(self.clone());
    while let Some(state) = current {
      if &state == root {
        return true;
      }
      current = state.parent();
    }
    false
  }

  /// This state's address relative to its root, e.g. `['nodes'][2]`; empty
  /// for the root itself. The format Thalamus sends and accepts over
  /// `observable_bridge_v2`.
  pub fn address(&self) -> String {
    let mut segments = Vec::new();
    let mut current = self.clone();
    while let Some(parent) = current.parent() {
      match parent.key_of(&current) {
        Some(key) => segments.push(key.address_segment()),
        None => {
          tracing::error!("Failed to find self in parent collection");
          return String::new();
        }
      }
      current = parent;
    }
    segments.reverse();
    segments.concat()
  }

  /// Sets (or with `None`, clears) the remote storage that writes to this
  /// state and its descendants go through. See [`RemoteStorage`].
  pub fn set_remote_storage(&self, storage: Option<RemoteStorage>) {
    *self.0.remote_storage.lock() = storage;
  }

  /// The remote storage in effect here: this state's own, else its nearest
  /// ancestor's.
  fn remote_storage(&self) -> Option<RemoteStorage> {
    let mut current = Some(self.clone());
    while let Some(state) = current {
      if let Some(storage) = state.0.remote_storage.lock().clone() {
        return Some(storage);
      }
      current = state.parent();
    }
    None
  }

  /// Hands `value` to remote storage if there is any and `from_remote` is
  /// false. Returns true if remote storage took it.
  fn send_remote(
    &self,
    action: StateAction,
    key: &StateKey,
    value: &StateValue,
    callback: &Callback,
    from_remote: bool,
  ) -> bool {
    if from_remote {
      return false;
    }
    let Some(storage) = self.remote_storage() else {
      return false;
    };
    let address = self.address() + &key.address_segment();
    storage(action, &address, value, callback.clone())
  }

  /// Makes `value` (if it's a collection) a child of this state. Like
  /// Python, it inherits this tree's remote storage instead of keeping its
  /// own.
  fn attach(&self, value: &StateValue) {
    if let Some(child) = value.as_state() {
      *child.0.parent.lock() = Arc::downgrade(&self.0);
      *child.0.remote_storage.lock() = None;
    }
  }

  /// Undoes `attach` for a value that's been removed or replaced.
  fn detach(value: &StateValue) {
    if let Some(child) = value.as_state() {
      *child.0.parent.lock() = Weak::new();
      *child.0.remote_storage.lock() = None;
    }
  }

  /// Notifies this state's own observers, then the recursive observers of
  /// it and each of its ancestors in turn (C++'s order).
  fn notify(&self, action: StateAction, key: &StateKey, value: &StateValue) {
    dispatch(&self.0.observers, |f| f(action, key.clone(), value.clone()));
    let mut current = Some(self.clone());
    while let Some(state) = current {
      dispatch(&state.0.recursive_observers, |f| {
        f(self.clone(), action, key.clone(), value.clone())
      });
      current = state.parent();
    }
  }

  pub fn set(&self, index: impl IntoStateKey, raw_value: impl IntoStateValue) {
    self.set_with(index, raw_value, None, false);
  }

  /// Python's `setitem(key, value, callback, from_remote)`/C++'s
  /// `ValueWrapper::assign`. Setting a scalar to its current value is a
  /// no-op (apart from calling `callback`). On a list, setting index
  /// `len()` appends.
  pub fn set_with(
    &self,
    index: impl IntoStateKey,
    raw_value: impl IntoStateValue,
    callback: Option<Callback>,
    from_remote: bool,
  ) {
    let key = index.into_state_key();
    let value = raw_value.into_state_value();
    let callback = callback.unwrap_or_else(noop_callback);

    if value.as_state().is_none() && self.get(key.clone()).as_ref() == Some(&value) {
      callback();
      return;
    }
    if let Some(child) = value.as_state()
      && self.is_descendent(child)
    {
      tracing::error!("refusing to store a state inside itself at {}", key.address_segment());
      return;
    }
    if self.send_remote(StateAction::Set, &key, &value, &callback, from_remote) {
      return;
    }

    let old = {
      let mut content = self.0.content.lock();
      match &mut *content {
        Content::Dict(map) => Ok(map.insert(key.clone(), value.clone())),
        Content::List(list) => match key.list_index() {
          Some(i) if i < list.len() => Ok(Some(std::mem::replace(&mut list[i], value.clone()))),
          Some(i) if i == list.len() => {
            list.push(value.clone());
            Ok(None)
          }
          _ => Err(list.len()),
        },
      }
    };
    let old = match old {
      Ok(old) => old,
      Err(len) => {
        tracing::error!("can't set {} on a list of length {len}", key.address_segment());
        return;
      }
    };

    self.attach(&value);
    if let Some(old) = old
      && old.as_state().is_some_and(|o| Some(o) != value.as_state())
    {
      State::detach(&old);
    }
    self.notify(StateAction::Set, &key, &value);
    callback();
  }

  pub fn remove(&self, index: impl IntoStateKey) {
    self.remove_with(index, None, false);
  }

  /// Python's `delitem(key, callback, from_remote)`/C++'s `erase`. Does
  /// nothing if there's nothing at `index`.
  pub fn remove_with(&self, index: impl IntoStateKey, callback: Option<Callback>, from_remote: bool) {
    let key = index.into_state_key();
    let callback = callback.unwrap_or_else(noop_callback);
    let Some(current) = self.get(key.clone()) else {
      return;
    };
    if self.send_remote(StateAction::Delete, &key, &current, &callback, from_remote) {
      return;
    }

    let removed = match &mut *self.0.content.lock() {
      Content::Dict(map) => map.shift_remove(&key),
      Content::List(list) => key
        .list_index()
        .filter(|&i| i < list.len())
        .map(|i| list.remove(i)),
    };
    let Some(removed) = removed else {
      return;
    };
    State::detach(&removed);
    self.notify(StateAction::Delete, &key, &removed);
    callback();
  }

  /// Python's list `insert(i, value, callback, from_remote)`: inserts before
  /// index `i` (clamped to `len()`), notifying `Set` for the inserted item
  /// and every item it shifted. Through remote storage this is sent as a
  /// `Set` at index `i`, as Python does.
  pub fn insert_with(
    &self,
    index: usize,
    raw_value: impl IntoStateValue,
    callback: Option<Callback>,
    from_remote: bool,
  ) {
    let value = raw_value.into_state_value();
    let callback = callback.unwrap_or_else(noop_callback);
    if self.is_dict() {
      tracing::error!("insert called on a dict");
      return;
    }
    let index = index.min(self.len());
    let key = StateKey::Int(index as i64);
    if let Some(child) = value.as_state()
      && self.is_descendent(child)
    {
      tracing::error!("refusing to store a state inside itself at {}", key.address_segment());
      return;
    }
    if self.send_remote(StateAction::Set, &key, &value, &callback, from_remote) {
      return;
    }

    self.attach(&value);
    let shifted: Vec<StateValue> = match &mut *self.0.content.lock() {
      Content::List(list) => {
        let index = index.min(list.len());
        list.insert(index, value);
        list[index..].to_vec()
      }
      Content::Dict(_) => unreachable!("checked above"),
    };
    for (offset, value) in shifted.iter().enumerate() {
      self.notify(StateAction::Set, &StateKey::Int((index + offset) as i64), value);
    }
    callback();
  }

  pub fn insert(&self, index: usize, raw_value: impl IntoStateValue) {
    self.insert_with(index, raw_value, None, false);
  }

  /// Python's `append`/C++'s `push_back`.
  pub fn push_with(&self, raw_value: impl IntoStateValue, callback: Option<Callback>, from_remote: bool) {
    self.insert_with(self.len(), raw_value, callback, from_remote);
  }

  pub fn push(&self, raw_value: impl IntoStateValue) {
    self.push_with(raw_value, None, false);
  }

  /// Appends `value` to this list.
  pub fn push_state(&self, value: &State) {
    self.push(value);
  }

  pub fn push_int(&self, value: i64) {
    self.push(value);
  }

  /// Python's `extend`.
  pub fn extend_with<V: IntoStateValue>(
    &self,
    values: impl IntoIterator<Item = V>,
    callback: Option<Callback>,
    from_remote: bool,
  ) {
    for value in values {
      self.push_with(value, callback.clone(), from_remote);
    }
  }

  pub fn extend<V: IntoStateValue>(&self, values: impl IntoIterator<Item = V>) {
    self.extend_with(values, None, false);
  }

  /// Python's list `pop(i)`/C++'s `pop_back` (`index = None`): removes and
  /// returns the item at `index`, or the last item. Through remote storage
  /// the item isn't actually gone until Thalamus echoes the change back.
  pub fn pop_with(&self, index: Option<usize>, callback: Option<Callback>, from_remote: bool) -> Option<StateValue> {
    if self.is_dict() {
      tracing::error!("pop called on a dict");
      return None;
    }
    let index = match index {
      Some(index) => index,
      None => self.len().checked_sub(1)?,
    };
    let value = self.get(index)?;
    self.remove_with(index, callback, from_remote);
    Some(value)
  }

  pub fn pop(&self, index: Option<usize>) -> Option<StateValue> {
    self.pop_with(index, None, false)
  }

  /// Python's list `remove(value)`: removes the first item equal to
  /// `value`. Returns whether there was one.
  pub fn remove_val_with(&self, val: impl IntoStateValue, callback: Option<Callback>, from_remote: bool) -> bool {
    let val = val.into_state_value();
    let Some(index) = self.values().position(|v| v == val) else {
      return false;
    };
    self.remove_with(index, callback, from_remote);
    true
  }

  pub fn remove_val(&self, val: impl IntoStateValue) -> bool {
    self.remove_val_with(val, None, false)
  }

  /// Python's dict `update`.
  pub fn update<K: IntoStateKey, V: IntoStateValue>(&self, items: impl IntoIterator<Item = (K, V)>) {
    for (key, value) in items {
      self.set(key, value);
    }
  }

  /// Removes every entry, last first.
  pub fn clear_with(&self, callback: Option<Callback>, from_remote: bool) {
    let mut keys: Vec<StateKey> = self.keys().collect();
    keys.sort();
    for key in keys.into_iter().rev() {
      self.remove_with(key, callback.clone(), from_remote);
    }
  }

  pub fn clear(&self) {
    self.clear_with(None, false);
  }

  /// Python's list `+`: a new list holding copies of this list's items
  /// followed by `other`'s.
  pub fn concat(&self, other: &State) -> State {
    let result = State::make_list();
    for value in self.values().chain(other.values()) {
      result.push(value.deep_copy());
    }
    result
  }

  /// Merges `entries` into this state, recursing into collections present
  /// on both sides with the same kind (so existing nodes, and the observers
  /// on them, are kept), and replacing everything else. With
  /// `delete_missing` it also removes this state's entries that `entries`
  /// lacks, at every level. Returns the keys at this level that `entries`
  /// lacked.
  fn merge_entries(
    &self,
    entries: Vec<(StateKey, StateValue)>,
    copy: bool,
    delete_missing: bool,
    callback: &Option<Callback>,
    from_remote: bool,
  ) -> HashSet<StateKey> {
    let mut missing: HashSet<StateKey> = self.keys().collect();
    for (key, value) in entries {
      missing.remove(&key);
      match (&value, self.get(key.clone())) {
        (StateValue::Dict(source), Some(StateValue::Dict(dest)))
        | (StateValue::List(source), Some(StateValue::List(dest))) => {
          if &dest != source {
            dest.merge_entries(source.entries(), copy, delete_missing, callback, from_remote);
          }
        }
        _ => {
          let value = if copy { value.deep_copy() } else { value };
          self.set_with(key, value, callback.clone(), from_remote);
        }
      }
    }
    if delete_missing {
      let mut keys: Vec<StateKey> = missing.iter().cloned().collect();
      keys.sort();
      for key in keys.into_iter().rev() {
        self.remove_with(key, callback.clone(), from_remote);
      }
    }
    missing
  }

  /// Makes this state's contents equal to `that`'s: Python's/C++'s
  /// `assign`, recursively merging into existing nested collections and
  /// removing whatever `that` doesn't have. Collections are copied out of
  /// `that` rather than moved.
  pub fn assign(&mut self, that: &State) {
    self.merge_entries(that.entries(), true, true, &None, false);
  }

  /// Python's `merge`: like `assign`, but never removes anything. Returns
  /// the top-level keys this state has that `that` doesn't.
  pub fn merge(&mut self, that: &State) -> HashSet<StateKey> {
    self.merge_entries(that.entries(), true, false, &None, false)
  }

  /// [`Self::assign`] from JSON, with Python's `callback`/`from_remote`.
  /// `json` must be an object for a dict and an array for a list.
  pub fn assign_json_with(&self, json: &Json, callback: Option<Callback>, from_remote: bool) -> anyhow::Result<()> {
    let entries = self.json_entries(json)?;
    self.merge_entries(entries, false, true, &callback, from_remote);
    Ok(())
  }

  pub fn assign_json(&self, json: &Json) -> anyhow::Result<()> {
    self.assign_json_with(json, None, false)
  }

  /// [`Self::merge`] from JSON, with Python's `callback`/`from_remote`.
  pub fn merge_json_with(&self, json: &Json, callback: Option<Callback>, from_remote: bool) -> anyhow::Result<()> {
    let entries = self.json_entries(json)?;
    self.merge_entries(entries, false, false, &callback, from_remote);
    Ok(())
  }

  pub fn merge_json(&self, json: &Json) -> anyhow::Result<()> {
    self.merge_json_with(json, None, false)
  }

  fn json_entries(&self, json: &Json) -> anyhow::Result<Vec<(StateKey, StateValue)>> {
    match (self.is_dict(), json) {
      (true, Json::Object(map)) => Ok(
        map
          .iter()
          .map(|(k, v)| (StateKey::String(k.clone()), StateValue::from_json(v)))
          .collect(),
      ),
      (false, Json::Array(items)) => Ok(
        items
          .iter()
          .enumerate()
          .map(|(i, v)| (StateKey::Int(i as i64), StateValue::from_json(v)))
          .collect(),
      ),
      (is_dict, other) => anyhow::bail!(
        "can't assign {other} to a {}",
        if is_dict { "dict" } else { "list" }
      ),
    }
  }

  /// Notifies every observer as though each entry had just been `Set`.
  pub fn recap(&self) {
    for (key, value) in self.entries() {
      self.notify(StateAction::Set, &key, &value);
    }
  }

  /// Calls `callback` as though each entry had just been `Set`, without
  /// notifying anything else.
  pub fn recap_with<T: FnMut(State, StateAction, StateKey, StateValue)>(&self, mut callback: T) {
    for (key, value) in self.entries() {
      callback(self.clone(), StateAction::Set, key, value);
    }
  }

  /// Observes changes anywhere in this state's subtree (C++'s
  /// `recursive_changed`/Python's `add_recursive_observer`). The callback
  /// gets the state that changed, the action, the key, and the value (the
  /// new one for `Set`, the removed one for `Delete`). Disconnects when the
  /// returned `OnDrop` is dropped.
  pub fn connect<T: FnMut(State, StateAction, StateKey, StateValue) + Send + 'static>(&self, callback: T) -> OnDrop {
    self.connect_with(callback, None, false)
  }

  /// [`Self::connect`] with Python's `remove_when` and `recap` options.
  pub fn connect_with<T: FnMut(State, StateAction, StateKey, StateValue) + Send + 'static>(
    &self,
    callback: T,
    remove_when: Option<RemoveWhen>,
    recap: bool,
  ) -> OnDrop {
    let observer = Observer {
      id: NEXT_OBSERVER_ID.fetch_add(1, Ordering::Relaxed),
      callback: Arc::new(ReentrantMutex::new(RefCell::new(Box::new(callback) as Box<RecursiveChangeFn>))),
      remove_when: remove_when.map(Arc::from),
    };
    self.0.recursive_observers.lock().push(observer.clone());
    if recap {
      for (key, value) in self.entries() {
        if !observer.call(|f| f(self.clone(), StateAction::Set, key, value)) {
          self.0.recursive_observers.lock().retain(|o| o.id != observer.id);
          break;
        }
      }
    }
    let node = Arc::downgrade(&self.0);
    let id = observer.id;
    OnDrop::new(move || {
      if let Some(node) = node.upgrade() {
        node.recursive_observers.lock().retain(|o| o.id != id);
      }
    })
  }

  /// Observes changes to this state's own entries only (C++'s `changed`/
  /// Python's `add_observer`). The callback gets the action, key and value.
  /// Disconnects when the returned `OnDrop` is dropped.
  pub fn observe<T: FnMut(StateAction, StateKey, StateValue) + Send + 'static>(&self, callback: T) -> OnDrop {
    self.observe_with(callback, None, false)
  }

  /// [`Self::observe`] with Python's `remove_when` and `recap` options.
  pub fn observe_with<T: FnMut(StateAction, StateKey, StateValue) + Send + 'static>(
    &self,
    callback: T,
    remove_when: Option<RemoveWhen>,
    recap: bool,
  ) -> OnDrop {
    let observer = Observer {
      id: NEXT_OBSERVER_ID.fetch_add(1, Ordering::Relaxed),
      callback: Arc::new(ReentrantMutex::new(RefCell::new(Box::new(callback) as Box<ChangeFn>))),
      remove_when: remove_when.map(Arc::from),
    };
    self.0.observers.lock().push(observer.clone());
    if recap {
      for (key, value) in self.entries() {
        if !observer.call(|f| f(StateAction::Set, key, value)) {
          self.0.observers.lock().retain(|o| o.id != observer.id);
          break;
        }
      }
    }
    let node = Arc::downgrade(&self.0);
    let id = observer.id;
    OnDrop::new(move || {
      if let Some(node) = node.upgrade() {
        node.observers.lock().retain(|o| o.id != id);
      }
    })
  }
}
