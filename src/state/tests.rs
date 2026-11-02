use serde_json::json;
use tokio::sync::mpsc;

use super::bridge::{apply_changes, parse_value};
use super::jsonpath::parse_for_test;
use super::*;
use crate::pb::thalamus_grpc::observable_change::Action as ProtoAction;
use crate::pb::thalamus_grpc::{ObservableChange, ObservableTransaction};

fn state(json: Json) -> Arc<Dict> {
  dict(Some(Value::from_json(&json)))
}

fn list_state(json: Json) -> Arc<List> {
  list(Some(Value::from_json(&json)))
}

fn dict(value: Option<Value>) -> Arc<Dict> {
  value.unwrap().try_into().unwrap()
}

fn list(value: Option<Value>) -> Arc<List> {
  value.unwrap().try_into().unwrap()
}

/// `node`'s parent, as a value.
fn parent(node: &Arc<Dict>) -> Option<Value> {
  node.parent().upgrade().map(|p| p.as_value())
}

type Log = Arc<Mutex<Vec<(Action, Key, Json)>>>;

fn recorder(state: &Arc<Dict>) -> (Log, OnDrop) {
  let log: Log = Default::default();
  let sink = log.clone();
  let connection = state.connect(move |_, action, key, value| {
    sink.lock().push((action, key.clone(), value.to_json()));
  });
  (log, connection)
}

fn counter() -> (Arc<Mutex<usize>>, impl FnOnce() + Send + 'static) {
  let count = Arc::new(Mutex::new(0));
  let sink = count.clone();
  (count, move || *sink.lock() += 1)
}

#[test]
fn json_round_trip() {
  let value = json!({"a": 1, "b": [1.5, "x", null, true], "c": {"d": {}}});
  assert_eq!(state(value.clone()).to_json(), value);
}

#[test]
fn get_set_remove_and_conversions() {
  let s = Dict::new();
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
  s.remove("int");
  assert!(!s.contains_key("int"));
  assert_eq!(s.len(), 3);
}

#[test]
fn values_say_which_kind_of_node_they_hold() {
  let root = state(json!({"nodes": [{"name": "a"}]}));
  let nodes = root.get("nodes").unwrap();
  assert!(nodes.as_dict().is_none());
  let nodes = nodes.as_list().unwrap();
  let node = nodes.get(0usize).unwrap().as_dict().unwrap().clone();
  node.set("name", "b");
  nodes.push(Dict::new());
  assert_eq!(root.to_json(), json!({"nodes": [{"name": "b"}, {}]}));
  assert!(<Arc<Dict>>::try_from(Value::List(nodes.clone())).is_err());
  match root.get("nodes").unwrap() {
    Value::List(l) => assert_eq!(l.len(), 2),
    other => panic!("expected a list, got {other:?}"),
  }
}

#[test]
fn reading_past_a_list_or_with_the_wrong_key_type_finds_nothing() {
  let list = list_state(json!([1]));
  assert_eq!(list.get(1), None);
  assert_eq!(list.get(-1), None);
  assert_eq!(list.get("name"), None);
}

#[test]
fn list_set_replaces_and_appends() {
  let list = list_state(json!([1, 2]));
  list.set(1, 9);
  list.set(2, 3);
  list.push(4);
  assert_eq!(list.to_json(), json!([1, 9, 3, 4]));
  list.remove(0);
  assert_eq!(list.to_json(), json!([9, 3, 4]));
}

#[test]
fn replaced_and_removed_children_are_detached() {
  let root = state(json!({"a": {"x": 1}}));
  let a = dict(root.get("a"));
  root.set("a", 5);
  assert_eq!(parent(&a), None);

  let b = Dict::new();
  root.set("b", &b);
  assert_eq!(parent(&b), Some(Value::Dict(root.clone())));
  root.remove("b");
  assert_eq!(parent(&b), None);
}

#[test]
#[should_panic(expected = "into itself")]
fn refuses_to_create_a_cycle() {
  let root = state(json!({"a": {}}));
  dict(root.get("a")).set("loop", &root);
}

#[test]
fn observers_see_their_whole_subtree_with_the_changed_node() {
  let root = state(json!({"child": {"grandchild": {}}, "other": {}}));
  let child = dict(root.get("child"));
  let grandchild = dict(child.get("grandchild"));
  let seen: Arc<Mutex<Vec<(Coll, Action, Key, Json)>>> = Default::default();
  let sink = seen.clone();
  let _connection = child.connect(move |source, action, key, value| {
    sink.lock().push((source.clone(), action, key.clone(), value.to_json()));
  });

  grandchild.set("x", 1);
  child.remove("grandchild");
  dict(root.get("other")).set("y", 2);
  root.set("z", 3);
  assert_eq!(
    *seen.lock(),
    vec![
      (Coll::Dict(grandchild.clone()), Action::Set, Key::String("x".into()), json!(1)),
      (Coll::Dict(child.clone()), Action::Delete, Key::String("grandchild".into()), json!({"x": 1})),
    ]
  );
}

#[test]
fn observers_run_from_the_changed_node_up() {
  let root = state(json!({"child": {}}));
  let child = dict(root.get("child"));
  let order: Arc<Mutex<Vec<&str>>> = Default::default();
  let root_order = order.clone();
  let _root = root.connect(move |_, _, _, _| root_order.lock().push("root"));
  let child_order = order.clone();
  let _child = child.connect(move |_, _, _, _| child_order.lock().push("child"));
  child.set("x", 1);
  assert_eq!(*order.lock(), vec!["child", "root"]);
}

#[test]
fn detached_nodes_stop_notifying_their_old_ancestors() {
  let root = state(json!({"child": {}}));
  let child = dict(root.get("child"));
  let (log, _connection) = recorder(&root);
  root.remove("child");
  assert_eq!(log.lock().len(), 1);
  child.set("x", 1);
  assert_eq!(log.lock().len(), 1);
}

#[test]
fn a_disconnected_observer_can_touch_the_tree_when_freed() {
  struct ReadsOnDrop(Arc<Dict>);
  impl Drop for ReadsOnDrop {
    fn drop(&mut self) {
      self.0.get("a");
    }
  }

  let root = Dict::new();
  let capture = ReadsOnDrop(root.clone());
  let connection = root.connect(move |_, _, _, _| {
    let _ = &capture;
  });
  drop(connection);
  root.set("a", 1);
}

#[test]
fn dropping_the_connection_disconnects() {
  let root = Dict::new();
  let (log, connection) = recorder(&root);
  root.set("a", 1);
  drop(connection);
  root.set("b", 2);
  assert_eq!(log.lock().len(), 1);
}

#[test]
fn connections_can_be_dropped_on_another_thread() {
  let root = Dict::new();
  let (log, connection) = recorder(&root);
  std::thread::spawn(move || drop(connection)).join().unwrap();
  root.set("a", 1);
  assert!(log.lock().is_empty());
  // The closure was freed by that notify.
  assert_eq!(Arc::strong_count(&log), 1);
}

#[test]
fn an_observer_can_drop_its_own_connection() {
  let root = Dict::new();
  let (log, _connection) = recorder(&root);
  let slot: Arc<Mutex<Option<OnDrop>>> = Default::default();
  let own = slot.clone();
  *slot.lock() = Some(root.connect(move |_, _, _, _| drop(own.lock().take())));
  root.set("a", 1);
  root.set("b", 2);
  assert_eq!(log.lock().len(), 2);
  assert!(slot.lock().is_none());
}

#[test]
fn concurrent_writers_from_many_threads() {
  let root = state(json!({"child": {}}));
  let child = dict(root.get("child"));
  let (log, _connection) = recorder(&root);
  let threads: Vec<_> = (0..8)
    .map(|t| {
      let (root, child) = (root.clone(), child.clone());
      std::thread::spawn(move || {
        for i in 0..100 {
          root.set(format!("{t}-{i}"), i);
          child.set(format!("{t}-{i}"), i);
          root.get("child");
        }
      })
    })
    .collect();
  for thread in threads {
    thread.join().unwrap();
  }
  assert_eq!(root.len(), 801);
  assert_eq!(child.len(), 800);
  assert_eq!(log.lock().len(), 1600);
}

#[test]
fn an_observer_can_disconnect_a_later_one() {
  let root = Dict::new();
  let (first_log, first) = recorder(&root);
  let (second_log, second) = recorder(&root);
  drop(first);
  let later = Mutex::new(Some(second));
  let _connection = root.connect(move |_, _, _, _| drop(later.lock().take()));
  root.set("a", 1);
  assert!(first_log.lock().is_empty());
  assert_eq!(second_log.lock().len(), 1, "connected when the change was made");
  root.set("b", 2);
  assert_eq!(second_log.lock().len(), 1);
}

#[test]
fn setting_an_unchanged_scalar_does_not_notify_but_calls_back() {
  let root = state(json!({"a": 1}));
  let (log, _connection) = recorder(&root);
  let (called, callback) = counter();
  root.set_with("a", 1, callback);
  assert!(log.lock().is_empty());
  assert_eq!(*called.lock(), 1);
}

#[test]
fn observers_may_write_to_the_tree() {
  let root = Dict::new();
  let target = Arc::downgrade(&root);
  let _connection = root.connect(move |_, _, key, _| {
    if *key == Key::String("a".into()) {
      target.upgrade().unwrap().set("b", 2);
    }
  });
  root.set("a", 1);
  assert_eq!(root.get("b"), Some(Value::Int(2)));
}

#[test]
fn assign_keeps_nested_nodes_and_removes_missing_keys() {
  let root = state(json!({"keep": {"x": 1, "gone": 2}, "drop": 1, "list": [1, 2, 3]}));
  let keep = dict(root.get("keep"));
  root.assign_json(&json!({"keep": {"x": 5}, "new": true, "list": [1]})).unwrap();
  assert_eq!(root.to_json(), json!({"keep": {"x": 5}, "list": [1], "new": true}));
  assert_eq!(dict(root.get("keep")), keep, "nested dict assigned into, not replaced");
  assert_eq!(parent(&keep), Some(Value::Dict(root.clone())));
  assert!(root.assign_json(&json!([1])).is_err());
}

/// Remote storage that records what it was asked to do.
fn recording_storage() -> (RemoteStorage, Arc<Mutex<Vec<(Action, String, Json)>>>) {
  let sent: Arc<Mutex<Vec<(Action, String, Json)>>> = Default::default();
  let sink = sent.clone();
  let storage: RemoteStorage = Arc::new(move |action, address, value, _callback| {
    sink.lock().push((action, address.to_string(), value.to_json()));
  });
  (storage, sent)
}

#[test]
fn remote_addresses() {
  let root = state(json!({"nodes": [{"name": "a"}, {"name": "b"}]}));
  let (storage, sent) = recording_storage();
  root.set_remote_storage(Some(storage));
  dict(list(root.get("nodes")).get(1)).set("name", "c");
  let weird = Dict::new();
  root.set_direct("it's", &weird, ||{});
  weird.set("x", 1);
  root.set("top", 1);
  assert_eq!(
    *sent.lock(),
    vec![
      (Action::Set, "['nodes'][1]['name']".to_string(), json!("c")),
      (Action::Set, "[\"it's\"]['x']".to_string(), json!(1)),
      (Action::Set, "['top']".to_string(), json!(1)),
    ]
  );
}

#[test]
fn remote_storage_routes_writes_from_anywhere_in_the_tree() {
  let root = state(json!({"nodes": [{"name": "a"}]}));
  let (storage, sent) = recording_storage();
  root.set_remote_storage(Some(storage));
  let node = dict(list(root.get("nodes")).get(0));

  node.set("name", "b");
  node.remove("name");
  list(root.get("nodes")).push(json!({"name": "c"}));
  assert_eq!(root.to_json(), json!({"nodes": [{"name": "a"}]}), "nothing applied locally");
  assert_eq!(
    *sent.lock(),
    vec![
      (Action::Set, "['nodes'][0]['name']".to_string(), json!("b")),
      (Action::Delete, "['nodes'][0]['name']".to_string(), json!("a")),
      (Action::Set, "['nodes'][1]".to_string(), json!({"name": "c"})),
    ]
  );

  node.set_direct("name", "d", ||{});
  assert_eq!(node.get("name"), Some(Value::String("d".into())), "direct applies locally");
}

#[test]
fn detached_nodes_lose_remote_storage() {
  let root = state(json!({"a": {}}));
  let a = dict(root.get("a"));
  root.set_direct("a", 1, ||{});
  let (storage, sent) = recording_storage();
  root.set_remote_storage(Some(storage));
  a.set("x", 1);
  assert!(sent.lock().is_empty());
  assert_eq!(a.get("x"), Some(Value::Int(1)));
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
fn jsonpath_compares_large_ints_exactly() {
  // 2^53 + 1 and 2^53 are the same f64.
  let root = state(json!({"ids": [
    {"id": 9007199254740992i64, "name": "a"},
    {"id": 9007199254740993i64, "name": "b"},
  ]}));
  let get = |q| get_jsonpath(&root, q).unwrap().to_json();
  assert_eq!(get("$.ids[?@.id == 9007199254740993].name"), json!("b"));
  assert_eq!(get("$.ids[?@.id != 9007199254740992].name"), json!("b"));
  assert_eq!(get("$.ids[?@.id > 9007199254740992].name"), json!("b"));
  assert_eq!(get("$.ids[?@.id < 9007199254740993].name"), json!("a"));
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
  let root = Dict::new();
  set_jsonpath(&root, "", &json!({"a": 1})).unwrap();
  assert_eq!(root.to_json(), json!({"a": 1}));

  set_jsonpath(&root, "['list']", &json!([])).unwrap();
  set_jsonpath(&root, "['list'][2]", &json!("x")).unwrap();
  assert_eq!(root.get("list").unwrap().to_json(), json!([null, null, "x"]));

  set_jsonpath(&root, "$.nodes", &json!([{"name": "a", "rate": 1}])).unwrap();
  let node = dict(list(root.get("nodes")).get(0));
  set_jsonpath(&root, "$.nodes[?@.name == 'a']", &json!({"name": "a", "rate": 2})).unwrap();
  assert_eq!(dict(list(root.get("nodes")).get(0)), node, "same node, assigned into");
  assert_eq!(node.to_json(), json!({"name": "a", "rate": 2}));

  assert!(set_jsonpath(&root, "$.missing.x", &json!(1)).is_err());
  assert!(set_jsonpath(&root, "$.nodes[?@.name == 'zzz']", &json!(1)).is_err());
}

#[test]
fn jsonpath_delete() {
  let root = state(json!({"a": 1, "list": [1, 2, 3], "nodes": [{"name": "x"}, {"name": "y"}]}));
  delete_jsonpath(&root, "['a']").unwrap();
  delete_jsonpath(&root, "['list'][1]").unwrap();
  delete_jsonpath(&root, "$.nodes[?@.name == 'x']").unwrap();
  delete_jsonpath(&root, "$.nothing").unwrap();
  assert_eq!(root.to_json(), json!({"list": [1, 3], "nodes": [{"name": "y"}]}));
  assert!(delete_jsonpath(&root, "").is_err());
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
  let root = Dict::new();
  apply_changes(&root, &[change("", r#"{"nodes": [{"name": "a"}], "x": 1}"#, ProtoAction::Set)]);
  let nodes = list(root.get("nodes"));
  apply_changes(
    &root,
    &[
      change("['nodes'][0]['name']", "probe", ProtoAction::Set),
      change("['x']", "", ProtoAction::Delete),
      change("['nodes'][1]", r#"{"name": "b"}"#, ProtoAction::Set),
    ],
  );
  assert_eq!(root.to_json(), json!({"nodes": [{"name": "probe"}, {"name": "b"}]}));
  assert_eq!(list(root.get("nodes")), nodes);

  // A full-state Set later on is assigned in, keeping existing nodes.
  apply_changes(&root, &[change("", r#"{"nodes": [{"name": "c"}]}"#, ProtoAction::Set)]);
  assert_eq!(list(root.get("nodes")), nodes);
  assert_eq!(root.to_json(), json!({"nodes": [{"name": "c"}]}));

  // A bad address from Thalamus is logged, not a crash.
  apply_changes(&root, &[change("['nodes']['name']", "1", ProtoAction::Set)]);
  apply_changes(&root, &[change("", "", ProtoAction::Delete)]);
  assert_eq!(root.to_json(), json!({}));
}

#[test]
fn bridge_sends_changes_and_calls_back_on_acknowledgement() {
  let root = state(json!({"a": 1}));
  let (tx, mut rx) = mpsc::unbounded_channel();
  let bridge = Bridge::new(tx);
  bridge.install(&root);

  let (acknowledged, callback) = counter();
  root.set_with("a", 2, callback);
  assert_eq!(root.get("a"), Some(Value::Int(1)), "not applied until echoed");

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
  assert_eq!(root.get("a"), Some(Value::Int(2)));
  assert_eq!(*acknowledged.lock(), 0);
  bridge.handle(
    &root,
    &ObservableTransaction {
      acknowledged: sent.id,
      ..Default::default()
    },
  );
  assert_eq!(*acknowledged.lock(), 1);

  // Once the stream is gone, writes fall back to applying locally.
  drop(rx);
  let (called, callback) = counter();
  root.set_with("a", 3, callback);
  assert_eq!(root.get("a"), Some(Value::Int(3)));
  assert_eq!(*called.lock(), 1);
  root.remove("a");
  assert_eq!(root.to_json(), json!({}));
}
