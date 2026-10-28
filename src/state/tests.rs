use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use parking_lot::Mutex;
use serde_json::json;
use tokio::sync::mpsc;

use super::bridge::{apply_changes, parse_value};
use super::jsonpath::parse_for_test;
use super::*;
use crate::pb::thalamus_grpc::observable_change::Action as ProtoAction;
use crate::pb::thalamus_grpc::{ObservableChange, ObservableTransaction};

fn state(json: Json) -> State {
  State::from_json(&json).unwrap()
}

fn dict(value: Option<StateValue>) -> State {
  State::try_from(value.unwrap()).unwrap()
}

type Log = Arc<Mutex<Vec<(StateAction, StateKey, Json)>>>;

fn recorder(state: &State) -> (Log, OnDrop) {
  let log: Log = Default::default();
  let sink = log.clone();
  let connection = state.observe(move |action, key, value| sink.lock().push((action, key, value.to_json())));
  (log, connection)
}

#[test]
fn json_round_trip() {
  let value = json!({"a": 1, "b": [1.5, "x", null, true], "c": {"d": {}}});
  assert_eq!(state(value.clone()).to_json(), value);
}

#[test]
fn dict_keys_keep_insertion_order() {
  let s = State::make_dict();
  s.set("z", 1);
  s.set("a", 2);
  s.set("m", 3);
  let keys: Vec<StateKey> = s.keys().collect();
  assert_eq!(
    keys,
    vec![
      StateKey::String("z".into()),
      StateKey::String("a".into()),
      StateKey::String("m".into())
    ]
  );
}

#[test]
fn get_set_remove_and_conversions() {
  let s = State::make_dict();
  s.set("int", 3);
  s.set("float", 2.5);
  s.set("text", "hi");
  s.set("flag", true);
  assert_eq!(i64::try_from(s.get("int").unwrap()), Ok(3));
  assert_eq!(f64::try_from(s.get("int").unwrap()), Ok(3.0));
  assert_eq!(f64::try_from(s.get("float").unwrap()), Ok(2.5));
  assert_eq!(String::try_from(s.get("text").unwrap()), Ok("hi".to_string()));
  assert_eq!(bool::try_from(s.get("flag").unwrap()), Ok(true));
  assert!(i64::try_from(s.get("text").unwrap()).is_err());
  assert!(s.contains_key("int"));
  assert!(s.contains_val("hi"));
  assert_eq!(s.get_or("missing", 7), StateValue::Int(7));
  s.remove("int");
  assert!(!s.contains_key("int"));
  assert_eq!(s.len(), 3);
}

#[test]
fn parents_keys_and_addresses() {
  let root = state(json!({"nodes": [{"name": "a"}, {"name": "it's"}]}));
  let nodes = dict(root.get("nodes"));
  let second = dict(nodes.get(1));
  assert_eq!(second.parent(), Some(nodes.clone()));
  assert_eq!(second.root(), root);
  assert_eq!(nodes.key_of(&second), Some(StateKey::Int(1)));
  assert_eq!(second.key_in_parent(), Some(StateKey::Int(1)));
  assert_eq!(second.address(), "['nodes'][1]");
  assert!(second.is_descendent(&root));
  assert!(!root.is_descendent(&second));

  let weird = State::make_dict();
  root.set("it's", &weird);
  assert_eq!(weird.address(), "[\"it's\"]");
}

#[test]
fn replaced_and_removed_children_are_detached() {
  let root = state(json!({"a": {"x": 1}}));
  let a = dict(root.get("a"));
  root.set("a", 5);
  assert_eq!(a.parent(), None);

  let b = State::make_dict();
  root.set("b", &b);
  assert_eq!(b.parent(), Some(root.clone()));
  root.remove("b");
  assert_eq!(b.parent(), None);
}

#[test]
fn refuses_to_create_a_cycle() {
  let root = state(json!({"a": {}}));
  let a = dict(root.get("a"));
  a.set("loop", &root);
  assert!(!a.contains_key("loop"));
}

#[test]
fn observe_reports_own_changes_only() {
  let root = state(json!({"child": {}}));
  let child = dict(root.get("child"));
  let (log, _connection) = recorder(&root);
  child.set("x", 1);
  root.set("y", 2);
  root.remove("y");
  assert_eq!(
    *log.lock(),
    vec![
      (StateAction::Set, StateKey::String("y".into()), json!(2)),
      (StateAction::Delete, StateKey::String("y".into()), json!(2)),
    ]
  );
}

#[test]
fn connect_reports_the_whole_subtree_with_its_source() {
  let root = state(json!({"child": {}}));
  let child = dict(root.get("child"));
  let seen = Arc::new(Mutex::new(Vec::new()));
  let sink = seen.clone();
  let _connection = root.connect(move |source, action, key, value| {
    sink.lock().push((source, action, key, value.to_json()));
  });
  child.set("x", 1);
  assert_eq!(
    *seen.lock(),
    vec![(child.clone(), StateAction::Set, StateKey::String("x".into()), json!(1))]
  );
}

#[test]
fn dropping_the_connection_disconnects() {
  let root = State::make_dict();
  let (log, connection) = recorder(&root);
  root.set("a", 1);
  drop(connection);
  root.set("b", 2);
  assert_eq!(log.lock().len(), 1);
}

#[test]
fn remove_when_and_recap() {
  let root = state(json!({"a": 1, "b": 2}));
  let calls = Arc::new(AtomicUsize::new(0));
  let done = Arc::new(AtomicUsize::new(0));
  let (counter, gate) = (calls.clone(), done.clone());
  let _connection = root.observe_with(
    move |_, _, _| {
      counter.fetch_add(1, Ordering::SeqCst);
    },
    Some(Box::new(move || gate.load(Ordering::SeqCst) != 0)),
    true,
  );
  assert_eq!(calls.load(Ordering::SeqCst), 2, "recap reports existing entries");
  root.set("c", 3);
  assert_eq!(calls.load(Ordering::SeqCst), 3);
  done.store(1, Ordering::SeqCst);
  root.set("d", 4);
  assert_eq!(calls.load(Ordering::SeqCst), 3);
  assert!(root.0.observers.lock().is_empty(), "removed once remove_when fired");
}

#[test]
fn recap_with_and_recap() {
  let root = state(json!([10, 20]));
  let mut seen = Vec::new();
  root.recap_with(|_, action, key, value| seen.push((action, key, value)));
  assert_eq!(
    seen,
    vec![
      (StateAction::Set, StateKey::Int(0), StateValue::Int(10)),
      (StateAction::Set, StateKey::Int(1), StateValue::Int(20)),
    ]
  );
  let (log, _connection) = recorder(&root);
  root.recap();
  assert_eq!(log.lock().len(), 2);
}

#[test]
fn setting_an_unchanged_scalar_does_not_notify_but_calls_back() {
  let root = state(json!({"a": 1}));
  let (log, _connection) = recorder(&root);
  let called = Arc::new(AtomicUsize::new(0));
  let counter = called.clone();
  root.set_with(
    "a",
    1,
    Some(Arc::new(move || {
      counter.fetch_add(1, Ordering::SeqCst);
    })),
    false,
  );
  assert!(log.lock().is_empty());
  assert_eq!(called.load(Ordering::SeqCst), 1);
}

#[test]
fn observers_may_write_to_the_tree() {
  let root = State::make_dict();
  let target = root.clone();
  let _connection = root.observe(move |_, key, _| {
    if key == StateKey::String("a".into()) {
      target.set("b", 2);
    }
  });
  root.set("a", 1);
  assert_eq!(root.get("b"), Some(StateValue::Int(2)));
}

#[test]
fn list_operations() {
  let list = state(json!([1, 2, 3]));
  let (log, _connection) = recorder(&list);

  list.insert(1, 9);
  assert_eq!(list.to_json(), json!([1, 9, 2, 3]));
  // Python notifies the inserted item and every item it shifted.
  assert_eq!(log.lock().len(), 3);

  list.push_int(4);
  list.extend([5, 6]);
  assert_eq!(list.to_json(), json!([1, 9, 2, 3, 4, 5, 6]));

  assert_eq!(list.pop(None), Some(StateValue::Int(6)));
  assert_eq!(list.pop(Some(0)), Some(StateValue::Int(1)));
  assert!(list.remove_val(9));
  assert!(!list.remove_val(42));
  assert_eq!(list.to_json(), json!([2, 3, 4, 5]));

  list.set(4, 6); // index len() appends
  list.set(9, 0); // out of range: ignored
  assert_eq!(list.to_json(), json!([2, 3, 4, 5, 6]));

  let other = state(json!([7]));
  assert_eq!(list.concat(&other).to_json(), json!([2, 3, 4, 5, 6, 7]));

  list.clear();
  assert!(list.is_empty());
}

#[test]
fn assign_keeps_nested_nodes_and_removes_missing_keys() {
  let mut root = state(json!({"keep": {"x": 1, "gone": 2}, "drop": 1, "list": [1, 2, 3]}));
  let keep = dict(root.get("keep"));
  let source = state(json!({"keep": {"x": 5}, "new": true, "list": [1]}));
  root.assign(&source);
  assert_eq!(root.to_json(), json!({"keep": {"x": 5}, "list": [1], "new": true}));
  assert_eq!(dict(root.get("keep")), keep, "nested dict assigned into, not replaced");
  assert_eq!(keep.parent(), Some(root.clone()));
  // Collections are copied out of the source, not moved.
  assert_ne!(root.get("keep"), source.get("keep"));
}

#[test]
fn merge_never_removes() {
  let mut root = state(json!({"a": {"x": 1, "y": 2}, "b": 1}));
  let missing = root.merge(&state(json!({"a": {"x": 5}, "c": 3})));
  assert_eq!(root.to_json(), json!({"a": {"x": 5, "y": 2}, "b": 1, "c": 3}));
  assert_eq!(missing, HashSet::from([StateKey::String("b".into())]));
}

#[test]
fn copy_is_deep_and_detached() {
  let root = state(json!({"a": {"b": [1]}}));
  let copy = root.copy();
  assert_eq!(copy.to_json(), root.to_json());
  dict(dict(copy.get("a")).get("b")).push(2);
  assert_eq!(root.to_json(), json!({"a": {"b": [1]}}));
}

#[test]
fn update_sets_each_pair() {
  let root = State::make_dict();
  root.update([("a", 1), ("b", 2)]);
  assert_eq!(root.to_json(), json!({"a": 1, "b": 2}));
}

/// Remote storage that records what it was asked to do and takes every
/// change.
fn recording_storage() -> (RemoteStorage, Arc<Mutex<Vec<(StateAction, String, Json, Callback)>>>) {
  let sent: Arc<Mutex<Vec<(StateAction, String, Json, Callback)>>> = Default::default();
  let sink = sent.clone();
  let storage: RemoteStorage = Arc::new(move |action, address, value, callback| {
    sink.lock().push((action, address.to_string(), value.to_json(), callback));
    true
  });
  (storage, sent)
}

#[test]
fn remote_storage_routes_writes_instead_of_applying_them() {
  let root = state(json!({"nodes": [{"name": "a"}]}));
  let (storage, sent) = recording_storage();
  root.set_remote_storage(Some(storage));
  let node = dict(dict(root.get("nodes")).get(0));

  node.set("name", "b");
  node.remove("name");
  dict(root.get("nodes")).push(json!({"name": "c"}));
  assert_eq!(root.to_json(), json!({"nodes": [{"name": "a"}]}), "nothing applied locally");
  let sent: Vec<(StateAction, String, Json)> = sent.lock().iter().map(|(a, s, v, _)| (*a, s.clone(), v.clone())).collect();
  assert_eq!(
    sent,
    vec![
      (StateAction::Set, "['nodes'][0]['name']".to_string(), json!("b")),
      (StateAction::Delete, "['nodes'][0]['name']".to_string(), json!("a")),
      (StateAction::Set, "['nodes'][1]".to_string(), json!({"name": "c"})),
    ]
  );

  node.set_with("name", "d", None, true);
  assert_eq!(node.get("name"), Some(StateValue::String("d".into())), "from_remote applies locally");
}

#[test]
fn detached_nodes_lose_remote_storage() {
  let root = state(json!({"a": {}}));
  let a = dict(root.get("a"));
  root.set_with("a", 1, None, true);
  let (storage, sent) = recording_storage();
  root.set_remote_storage(Some(storage));
  a.set("x", 1);
  assert!(sent.lock().is_empty());
  assert_eq!(a.get("x"), Some(StateValue::Int(1)));
}

#[test]
fn jsonpath_get() {
  let root = state(json!({
    "nodes": [
      {"name": "eye", "type": "OCULOMATIC", "rate": 60},
      {"name": "touch", "type": "TOUCH_SCREEN", "rate": 120.5},
    ],
    "eye_scaling": {"Models": {"Angular Scaling": {"Scale Default": 100}}},
  }));
  let get = |q| get_jsonpath(&root, q).unwrap().to_json();
  assert_eq!(get("$.nodes[1].name"), json!("touch"));
  assert_eq!(get("['nodes'][0]['name']"), json!("eye"));
  assert_eq!(get("nodes[-1].rate"), json!(120.5));
  assert_eq!(get("['eye_scaling']['Models']['Angular Scaling']['Scale Default']"), json!(100));
  assert_eq!(get("$.nodes[?@.type == 'TOUCH_SCREEN'].name"), json!("touch"));
  assert_eq!(get("$.nodes[?(@.rate > 100 || @.name == 'nope')].name"), json!("touch"));
  assert_eq!(get("$.nodes[?@.rate >= 60 && @.rate < 61].name"), json!("eye"));
  assert_eq!(get("$.nodes[?@.rate == 60.0].name"), json!("eye"));
  assert_eq!(get("$.nodes[?@.name == $.nodes[1].name].type"), json!("TOUCH_SCREEN"));
  assert_eq!(get("$.nodes[5]"), json!(null));
  assert_eq!(get(""), root.to_json());
}

#[test]
fn jsonpath_parse_errors() {
  for bad in ["nodes[", "['unterminated]", "$.nodes[?@.a]", "[1]x", "$.[0]"] {
    assert!(parse_for_test(bad).is_err(), "{bad:?} should fail to parse");
  }
  assert!(parse_for_test(r"['it\'s']").is_ok());
}

#[test]
fn jsonpath_set() {
  let root = State::make_dict();
  set_jsonpath(&root, "", &json!({"a": 1}), false).unwrap();
  assert_eq!(root.to_json(), json!({"a": 1}));

  set_jsonpath(&root, "['list']", &json!([]), false).unwrap();
  set_jsonpath(&root, "['list'][2]", &json!("x"), false).unwrap();
  assert_eq!(root.get("list").unwrap().to_json(), json!([null, null, "x"]));

  set_jsonpath(&root, "$.nodes", &json!([{"name": "a", "rate": 1}]), false).unwrap();
  let node = dict(dict(root.get("nodes")).get(0));
  set_jsonpath(&root, "$.nodes[?@.name == 'a']", &json!({"name": "a", "rate": 2}), false).unwrap();
  assert_eq!(dict(dict(root.get("nodes")).get(0)), node, "same node, assigned into");
  assert_eq!(node.to_json(), json!({"name": "a", "rate": 2}));

  assert!(set_jsonpath(&root, "$.missing.x", &json!(1), false).is_err());
  assert!(set_jsonpath(&root, "$.nodes[?@.name == 'zzz']", &json!(1), false).is_err());
}

#[test]
fn jsonpath_delete() {
  let root = state(json!({"a": 1, "list": [1, 2, 3], "nodes": [{"name": "x"}, {"name": "y"}]}));
  delete_jsonpath(&root, "['a']", false).unwrap();
  delete_jsonpath(&root, "['list'][1]", false).unwrap();
  delete_jsonpath(&root, "$.nodes[?@.name == 'x']", false).unwrap();
  delete_jsonpath(&root, "$.nothing", false).unwrap();
  assert_eq!(root.to_json(), json!({"list": [1, 3], "nodes": [{"name": "y"}]}));
  assert!(delete_jsonpath(&root, "", false).is_err());
}

fn change(address: &str, value: &str, action: ProtoAction) -> ObservableChange {
  ObservableChange {
    address: address.to_string(),
    value: value.to_string(),
    action: action as i32,
    ..Default::default()
  }
}

#[test]
fn raw_unquoted_string_values_fall_back_to_strings() {
  assert_eq!(parse_value("hello world"), json!("hello world"));
  assert_eq!(parse_value("\"quoted\""), json!("quoted"));
  assert_eq!(parse_value("3"), json!(3));
}

#[test]
fn applying_remote_changes() {
  let root = State::make_dict();
  apply_changes(&root, &[change("", r#"{"nodes": [{"name": "a"}], "x": 1}"#, ProtoAction::Set)]);
  let nodes = dict(root.get("nodes"));
  apply_changes(
    &root,
    &[
      change("['nodes'][0]['name']", "probe", ProtoAction::Set),
      change("['x']", "", ProtoAction::Delete),
      change("['nodes'][1]", r#"{"name": "b"}"#, ProtoAction::Set),
    ],
  );
  assert_eq!(root.to_json(), json!({"nodes": [{"name": "probe"}, {"name": "b"}]}));
  assert_eq!(dict(root.get("nodes")), nodes);

  // A full-state Set later on is assigned in, keeping existing nodes.
  apply_changes(&root, &[change("", r#"{"nodes": [{"name": "c"}]}"#, ProtoAction::Set)]);
  assert_eq!(dict(root.get("nodes")), nodes);
  assert_eq!(root.to_json(), json!({"nodes": [{"name": "c"}]}));
}

#[test]
fn bridge_sends_changes_and_calls_back_on_acknowledgement() {
  let root = state(json!({"a": 1}));
  let (tx, mut rx) = mpsc::unbounded_channel();
  let bridge = Bridge::new(tx);
  bridge.install(&root);

  let acknowledged = Arc::new(AtomicUsize::new(0));
  let counter = acknowledged.clone();
  root.set_with(
    "a",
    2,
    Some(Arc::new(move || {
      counter.fetch_add(1, Ordering::SeqCst);
    })),
    false,
  );
  assert_eq!(root.get("a"), Some(StateValue::Int(1)), "not applied until echoed");

  let sent = rx.try_recv().unwrap();
  assert_eq!(sent.changes.len(), 1);
  assert_eq!(sent.changes[0].address, "['a']");
  assert_eq!(sent.changes[0].value, "2");
  assert_eq!(sent.changes[0].action(), ProtoAction::Set);

  // Thalamus echoes the change, then acknowledges the transaction.
  bridge.handle(
    &root,
    &ObservableTransaction {
      changes: vec![change("['a']", "2", ProtoAction::Set)],
      ..Default::default()
    },
  );
  assert_eq!(root.get("a"), Some(StateValue::Int(2)));
  assert_eq!(acknowledged.load(Ordering::SeqCst), 0);
  bridge.handle(
    &root,
    &ObservableTransaction {
      acknowledged: sent.id,
      ..Default::default()
    },
  );
  assert_eq!(acknowledged.load(Ordering::SeqCst), 1);

  // Once the stream is gone, writes fall back to applying locally.
  drop(rx);
  root.set("a", 3);
  assert_eq!(root.get("a"), Some(StateValue::Int(3)));
}
