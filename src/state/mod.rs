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
use std::cell::Ref;
use std::cell::RefCell;
use std::collections::HashSet;
use std::collections::HashMap;
use std::collections::hash_map;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::rc::{Rc, Weak};
use std::sync::{Arc};
use std::sync::atomic::{AtomicU64, Ordering};

use indexmap::IndexMap;
use parking_lot::{Mutex, ReentrantMutex};
use serde_json::Value as Json;
use either::Either;

use crate::state::Value::Coll;

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

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Key {
  Int(i64),
  String(String),
}

#[derive(Debug, Clone, PartialEq)]
pub enum Value {
  Bool(bool),
  Coll(Rc<Collection>),
  Float(f64),
  Int(i64),
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

type RemoteStorage = Rc<dyn Fn(Action, &str, Value, Box<dyn FnOnce()>)>;

pub struct Dict {
  content: HashMap<Key, Value>,
  id: u64,
  observers: Arc<Mutex<Vec<(u64, Rc<dyn Fn(&Rc<Collection>, Action, &Key, &Value)>)>>>,
  remote_storage: Option<RemoteStorage>,
  parent: Weak<Collection>,
}

pub struct List {
  content: Vec<Value>,
  id: u64,
  observers: Arc<Mutex<Vec<(u64, Rc<dyn Fn(&Rc<Collection>, Action, &Key, &Value)>)>>>,
  remote_storage: Option<RemoteStorage>,
  parent: Weak<Collection>,
}

pub enum Collection {
  Dict(RefCell<Dict>),
  List(RefCell<List>),
}

macro_rules! on_content {
  ($coll:expr, $content:ident => $body:expr) => {
    match $coll {
      Collection::Dict($content) => $body,
      Collection::List($content) => $body,
    }
  };
}

impl PartialEq for Collection {
  fn eq(&self, other: &Self) -> bool {
    let l = on_content!(self, c => c.borrow().id);
    let r = on_content!(other, c => c.borrow().id);
    l == r
  }
}

impl std::fmt::Debug for Collection {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    match self {
      Self::Dict(c) => {
        let c = c.borrow();
        f.debug_tuple("Dict").field(&c.id).field(&c.content).finish()
      }
      Self::List(c) => {
        let c = c.borrow();
        f.debug_tuple("List").field(&c.id).field(&c.content).finish()
      }
    }
  }
}

impl Collection {
  fn entries(&self) -> Vec<(Key, Value)> {
    match self {
      Collection::Dict(c) => { 
        c.borrow().content.iter().map(|(k, v)| (k.clone(), v.clone())).collect()
      },
      Collection::List(c) => { 
        c.borrow().content.iter().enumerate().map(|(k, v)| (Key::Int(k as i64), v.clone())).collect()
      },
    }
  }

  fn parent(&self) -> Weak<Collection> {
    on_content!(self, c => c.borrow().parent.clone())
  }

  fn address(self: &Rc<Self>) -> String {
    let (prefix, end_opt) = match self.parent().upgrade().as_ref() {
      Some(parent) => (parent.address(), parent.key_of(&Value::Coll(self.clone()))),
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

  fn find_remote_storage(self: &Rc<Self>) -> Option<RemoteStorage> {
    on_content!(&**self, c => {
      match c.borrow().remote_storage.as_ref() {
        Some(r) => {
          Some(r.clone())
        }
        None => {
          match self.parent().upgrade().as_ref() {
            Some(parent) => parent.find_remote_storage(),
            None => None
          }
        }
      }
    })
  }

  fn connect<F: Fn(&Rc<Collection>, Action, &Key, &Value) + 'static>(
    self: &Rc<Self>,
    callback: F,
  ) -> OnDrop {
    let id = get_id();
    let weak = on_content!(&**self, c => {
      let c = c.borrow();
      c.observers.lock().push((id, Rc::new(callback)));
      Arc::downgrade(&c.observers)
    });
    OnDrop::new(move || {
      if let Some(this) = weak.upgrade() {
        this.lock().retain(|(i, _)| *i != id)
      }
    })
  }

  fn claim(self: &Rc<Self>, child: &Value) {
    match child {
      Value::Coll(c) => {
        on_content!(&**c, c2 => {
          let mut c2 = c2.borrow_mut();
          c2.parent = Rc::downgrade(self);
          c2.remote_storage = None;
        })
      },
      _ => {}
    }
  }

  pub fn push(self: &Rc<Self>, value: impl IntoValue) {
    self.push_with(value, ||{});
  }

  pub fn push_with(self: &Rc<Self>, value: impl IntoValue, callback: impl FnOnce() + 'static) {
    match &**self {
      Collection::List(c) => {
        let c = c.borrow();
        if let Some(remote_storage) = c.remote_storage.as_ref() {
          let address = self.address();
          let index = c.content.len();
          let total_address = format!("{address}[{index}]");
          remote_storage(Action::Set, &total_address, value.into_value(), Box::new(callback));
          return;
        }
      },
      Collection::Dict(_) => panic!("Cannot push into dict"),
    }

    self.push_direct(value, callback);
  }

  pub fn push_direct(self: &Rc<Self>, value: impl IntoValue, callback: impl FnOnce()) {
    let value = value.into_value();
    self.claim(&value);

    let (index, observers) = match &**self {
      Collection::List(c) => {
        let mut c = c.borrow_mut();

        let index = Key::Int(c.content.len() as i64);
        c.content.push(value.clone());

        (index, c.observers.lock().clone())
      },
      Collection::Dict(_) => panic!("Cannot push into dict"),
    };

    for (_, o) in observers {
      o(self, Action::Set, &index, &value);
    }
    callback();
  }

  pub fn set(self: &Rc<Self>, key: impl IntoKey, value: impl IntoValue) {
    self.set_with(key, value, ||{});
  }

  pub fn set_with(self: &Rc<Self>, key: impl IntoKey, value: impl IntoValue, callback: impl FnOnce() + 'static) {
    let key = key.into_key();
    let value = value.into_value();
    if self.get(key.clone()) == Some(value.clone()) {
      callback();
      return;
    }

    on_content!(&**self, c => {
      let c = c.borrow();
      if let Some(remote_storage) = c.remote_storage.as_ref() {
        let address = self.address();
        let index = match &key {
          Key::Int(i) => format!("[{i}]"),
          Key::String(i) => format!("['{i}']"),
        };
        let total_address = format!("{address}{index}");
        remote_storage(Action::Set, &total_address, value.into_value(), Box::new(callback));
        return;
      }
    });

    self.set_direct(key, value, callback);
  }

  pub fn set_direct(self: &Rc<Self>, key: impl IntoKey, value: impl IntoValue, callback: impl FnOnce() + 'static) {
    let key = key.into_key();
    let value = value.into_value();
    if self.get(key.clone()) == Some(value.clone()) {
      callback();
      return;
    }
    self.claim(&value);

    let observers = match &**self {
      Collection::List(c) => {
        let mut c = c.borrow_mut();

        match key {
          Key::Int(i) => {
            let i = usize::try_from(i).unwrap();
            let len = c.content.len();
            if i == len {
              c.content.push(value.clone());
            } else if i < len {
              c.content[i] = value.clone();
            } else {
              panic!("Index out of bounds: i = {i}, len = {len}");
            }
          },
          other => panic!("Index List with non integer: {:?}", other),
        }

        c.observers.lock().clone()
      },
      Collection::Dict(c) => {
        let mut c = c.borrow_mut();

        c.content.insert(key.clone(), value.clone());

        c.observers.lock().clone()
      },
    };

    for (_, o) in observers {
      o(self, Action::Set, &key, &value);
    }
    callback();
  }

  pub fn get(&self, key: impl IntoKey) -> Option<Value> {
    let key = key.into_key();
    match self {
      Collection::List(c) => {
        let c = c.borrow_mut();
        match key {
          Key::Int(i) => {
            Some(c.content[i as usize].clone())
          },
          other => panic!("Attempted to index list with non integer {:?}", other),
        }
      },
      Collection::Dict(c) => {
        let c = c.borrow_mut();
        c.content.get(&key).map(|v| v.clone())
      },
    }
  }

  pub fn contains_key(&self, key: impl IntoKey) -> bool {
    self.get(key).is_some()
  }
}
