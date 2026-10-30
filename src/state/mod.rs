//! Port of Thalamus' state tree: the `ObservableCollection` classes from
//! Python (`thalamus/config.py`) and C++ (`src/thalamus/state.hpp`/
//! `state.cpp`), exposed through the same API as thalamus-contrib's
//! FFI-backed `State` (`thalamus-contrib/rust/src/api.rs`) — `State`,
//! `Value`, `StateKey`, `StateAction`, `StateEntry`, `IntoStateKey`,
//! `IntoValue`, `OnDrop` — so code written against one reads the same
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
use parking_lot::MutexGuard;
use parking_lot::RawMutex;

use std::any::TypeId;
use std::cell::RefCell;
use std::collections::HashSet;
use std::collections::HashMap;
use std::collections::hash_map;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Weak};

use indexmap::IndexMap;
use parking_lot::{Mutex, ReentrantMutex};
use serde_json::Value as Json;

trait Content {
  fn entries(&self) -> impl Iterator<Item = (Key, &Value)>;
}

impl Content for HashMap<Key, Value> {
  fn entries(&self) -> impl Iterator<Item = (Key, &Value)> {
    self.iter().map(|(k, v)| (k.clone(), v))
  }
}

impl Content for Vec<Value> {
  fn entries(&self) -> impl Iterator<Item = (Key, &Value)> {
    self.iter().enumerate().map(|(i, v)| (Key::Int(i as i64), v))
  }
}

pub struct OnDrop {
  action: Option<Box<dyn FnOnce()>>,
}

impl OnDrop {
  pub fn noop() -> OnDrop {
    OnDrop {
      action: Some(Box::new(|| {})),
    }
  }
  pub fn new<F: FnOnce() + 'static>(f: F) -> OnDrop {
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

#[derive(Debug, Clone, PartialEq)]
pub enum Key {
  Int(i64),
  String(String),
}

#[derive(Debug, Clone, PartialEq)]
pub enum Value {
  Bool(bool),
  Dict(ObservableDict),
  Float(f64),
  Int(i64),
  List(ObservableList),
  String(String),
  Null,
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

pub trait IntoValue {
  fn into_value(self) -> Value;
}
impl IntoValue for Value {
  fn into_value(self) -> Value { self }
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

#[derive(Debug, Clone, PartialEq)]
pub enum Collection {
  Dict(ObservableDict),
  List(ObservableList),
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

impl TryFrom<Value> for String {
  type Error = Value;

  fn try_from(value: Value) -> Result<Self, Self::Error> {
    match value {
      Value::String(s) => Ok(s),
      other => Err(other),
    }
  }
}

static NEXT_ID: Mutex<u64> = Mutex::new(0 as u64);

fn get_id() -> u64 {
  let mut lock = NEXT_ID.lock();
  let id = *lock;
  *lock += 1;
  id
}

type RemoteStorage = Arc<dyn Fn(Action, &str, Value, Box<dyn FnOnce()>)>;
//type ObserverLock<'a> = MutexGuard<'a, Vec<(u64, Arc<dyn Fn(&Collection, Action, &Key, &Value)>)>>;

trait ObservableCollection {
  fn entries(&self) -> Vec<(Key, Value)>;
  fn parent(&self) -> &Weak<dyn ObservableCollection>;
  fn parent_mut(&mut self) -> &mut Weak<dyn ObservableCollection>;
  fn observers(&self) -> &Vec<(u64, Arc<dyn Fn(&Collection, Action, &Key, &Value)>)>;
  fn observers_mut(&mut self) -> &mut Vec<(u64, Arc<dyn Fn(&Collection, Action, &Key, &Value)>)>;
  fn as_value(&self) -> Value;
  fn remote_storage(&self) -> &Option<RemoteStorage>;
  fn remote_storage_mut(&mut self) -> &mut Option<RemoteStorage>;
  fn id(&self) -> u64;

  fn find_remote_storage(&self) -> Option<RemoteStorage> {
    match self.remote_storage().as_ref() {
      Some(r) => Some(r.clone()),
      None => {
        match self.parent().upgrade() {
          Some(parent) => parent.find_remote_storage(),
          None => None
        }
      }
    }
  }

  fn address(&self) -> String {
    let (prefix, end_opt) = match self.parent().upgrade() {
      Some(parent) => (parent.address(), parent.key_of(&self.as_value())),
      None => return "".to_string()
    };

    match end_opt {
      Some(Key::Int(i)) => format!("{prefix}[{i}]"),
      Some(Key::String(i)) => format!("{prefix}['{i}']"),
      None => panic!("Failed to find self in parent collection")
    }
  }

  fn key_of(&self, value: &Value) -> Option<Key> {
    for (k, v) in self.entries() {
      if v == *value {
        return Some(k);
      }
    }

    None
  }
}

fn connect<F: Fn(&Collection, Action, &Key, &Value) + 'static>(
  this: &Arc<Mutex<impl ObservableCollection + 'static>>,
  callback: F,
) -> OnDrop {
  let mut lock = this.lock();
  let id = get_id();
  lock.observers_mut().push((id, Arc::new(callback)));

  let weak = Arc::downgrade(this);
  OnDrop::new(move || {
    if let Some(this) = weak.upgrade() {
      this.lock().observers_mut().retain(|(i, _)| *i != id);
    }
  })
}

impl PartialEq for ObservableDict {
  fn eq(&self, other: &Self) -> bool {
    self.id() == other.id()
  }
}
impl Eq for ObservableDict {}

impl PartialEq for ObservableList {
  fn eq(&self, other: &Self) -> bool {
    self.id() == other.id()
  }
}
impl Eq for ObservableList {}

#[derive(Clone)]
pub struct ObservableDict {
  content: HashMap<Key, Value>,
  id: u64,
  observers: Vec<(u64, Arc<dyn Fn(&Collection, Action, &Key, &Value)>)>,
  remote_storage: Option<RemoteStorage>,
  parent: Weak<dyn ObservableCollection>,
}

impl ObservableCollection for ObservableDict {
  fn entries(&self) -> Vec<(Key, Value)> {
    self.content.entries().map(|(k,v)| (k, v.clone())).collect()
  }

  fn parent(&self) -> &Weak<dyn ObservableCollection> {
    &self.parent
  }
  fn parent_mut(&mut self) -> &mut Weak<dyn ObservableCollection> {
    &mut self.parent
  }

  fn observers(&self) -> &Vec<(u64, Arc<dyn Fn(&Collection, Action, &Key, &Value)>)> {
    &self.observers
  }
  fn observers_mut(&mut self) -> &mut Vec<(u64, Arc<dyn Fn(&Collection, Action, &Key, &Value)>)> {
    &mut self.observers
  }

  fn as_value(&self) -> Value {
    Value::Dict(self.clone())
  }

  fn remote_storage(&self) -> &Option<RemoteStorage> {
    &self.remote_storage
  }
  fn remote_storage_mut(&mut self) -> &mut Option<RemoteStorage> {
    &mut self.remote_storage
  }

  fn id(&self) -> u64 {
    self.id
  }
}

const _: () = {
  fn assert_send_sync<T: Send + Sync>() {}
  fn check() {
    assert_send_sync::<ObservableDict>();
  }
}

#[derive(Clone)]
pub struct ObservableList {
  content: Vec<Value>,
  id: u64,
  observers: Vec<(u64, Arc<dyn Fn(&Collection, Action, &Key, &Value)>)>,
  remote_storage: Option<RemoteStorage>,
  parent: Weak<dyn ObservableCollection>,
}

impl ObservableCollection for ObservableList {
  fn entries(&self) -> Vec<(Key, Value)> {
    self.content.iter().enumerate().map(|(k,v)| ((k as i64).into_key(), v.clone())).collect()
  }

  fn parent(&self) -> &Weak<dyn ObservableCollection> {
    &self.parent
  }
  fn parent_mut(&mut self) -> &mut Weak<dyn ObservableCollection> {
    &mut self.parent
  }

  fn observers(&self) -> &Vec<(u64, Arc<dyn Fn(&Collection, Action, &Key, &Value)>)> {
    &self.observers
  }
  fn observers_mut(&mut self) -> &mut Vec<(u64, Arc<dyn Fn(&Collection, Action, &Key, &Value)>)> {
    &mut self.observers
  }

  fn as_value(&self) -> Value {
    Value::List(self.clone())
  }

  fn remote_storage(&self) -> &Option<RemoteStorage> {
    &self.remote_storage
  }
  fn remote_storage_mut(&mut self) -> &mut Option<RemoteStorage> {
    &mut self.remote_storage
  }

  fn id(&self) -> u64 {
    self.id
  }
}

impl ObservableList {
  pub fn push_with(&self, value: impl IntoValue, callback: impl FnOnce() + 'static) {
    let mut lock = self.common.lock();

    if lock.remote_storage.is_some() {
      let address = lock.address(&Value::List(self.clone()));
      let index = lock.content.len();
      let total_address = format!("{address}[{index}]");
      (*lock.remote_storage.as_mut().unwrap())(Action::Set, &total_address, value.into_value(), Box::new(callback));
      return;
    }

    self.push_direct(value, callback, Some(lock))
  }

  pub fn push_direct(&self, value: impl IntoValue, callback: impl FnOnce(),
                     lock: Option<MutexGuard<ObservableCollection<Vec<Value>>>>) {
    let mut lock = match lock {
      Some(lock) => lock,
      None => self.common.lock(),
    };

    let value = value.into_value();
    match &value {
      Value::Dict(d) => {
        let mut child_lock = d.common.lock();
        child_lock.parent = Some(Collection::List(self.clone()));
        child_lock.remote_storage = None;
      },
      Value::List(d) => {
        let mut child_lock = d.common.lock();
        child_lock.parent = Some(Collection::List(self.clone()));
        child_lock.remote_storage = None;
      },
      _ => {}
    }

    let index = lock.content.len() as i64;
    lock.content.push(value.clone());
    
    let observers = lock.observers.clone();
    drop(lock);
    let source = Collection::List(self.clone());
    for (_, o) in observers {
      o(&source, Action::Set, &Key::Int(index), &value);
    }
    callback();
  }

  pub fn address(&self) -> String {
    self.common.lock().address(&Value::List(self.clone()))
  }

  pub fn key_of(&self, value: &Value) -> Option<Key> {
    self.common.lock().key_of(value)
  }
}

