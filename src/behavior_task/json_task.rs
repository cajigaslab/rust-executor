use std::{collections::HashMap, fs::File, sync::Arc};
use crate::{pb::task_controller_grpc::TaskResult};

use async_trait::async_trait;
use super::{BehaviorTask, PointSubscription, TaskContext, Window};
use skia_safe::Canvas;

pub struct JsonTask {
}

impl JsonTask {
  pub fn new() -> JsonTask{
    JsonTask {}
  }
}

fn to_object_array(value: &serde_json::Value) -> Vec<serde_json::Map<String, serde_json::Value>> {
  value.as_array().unwrap().iter().map(|v| v.as_object().unwrap().clone()).collect()
}

fn to_location(json: &serde_json::Value, root: &serde_json::Value, context: &TaskContext) -> (f64, f64) {
  let unit = json["unit"].as_str().unwrap_or("pixel");

  let scale = match unit {
    "pixel" => 1.0,
    "arc" => {
      let screen_distance_m = root["screen_distance_m"].as_f64().unwrap();
      let screen_dpi = root["screen_dpi"].as_f64().unwrap();
      //pixels/radian ~ (pixels/inch)*(inch/cm)*(cm/m)*(distance_m) for small angles (x ~ tan(x))
      screen_dpi/2.54*100.0*screen_distance_m
    }
    _ => {
      panic!("Unknown units: {unit}")
    }
  };

  let center = context.canvas_size();
  if let (Some(x), Some(y)) = (json["x"].as_f64(), json["y"].as_f64()) {
    let origin = json["unit"].as_str().unwrap_or("ul");
    match origin {
      "ul" => { (scale*x, scale*y) },
      "center" => { 
        (scale*x + center.0 as f64, scale*y + center.1 as f64)
      },
      _ => { panic!("Unexpected location origin: {origin}") }
    }
  } else if let (Some(r), Some(rad)) = (json["radius"].as_f64(), json["radians"].as_f64()) {
    let (x, y) = rad.sin_cos();
    (r*x + center.0 as f64, r*y + center.1 as f64)
  } else if let (Some(r), Some(deg)) = (json["radius"].as_f64(), json["degrees"].as_f64()) {
    let (x, y) = (deg*std::f64::consts::PI/180.0).sin_cos();
    (r*x + center.0 as f64, r*y + center.1 as f64)
  } else {
    panic!("Failed to interpret location {}", json.to_string());
  }
}

enum LocationPoolImpl {
  List(std::collections::VecDeque<(f64, f64)>)
}

struct LocationPool<'a> {
  root: &'a serde_json::Value,
  json: &'a serde_json::Value,
  _impl: LocationPoolImpl
}

impl<'a> LocationPool<'a> {
  fn new(json: &'a serde_json::Value, root: &'a serde_json::Value, context: &TaskContext) -> LocationPool<'a> {
    let pool_type = json["pool_type"].as_str().unwrap();
    let _impl = match pool_type {
      "list" => {
        LocationPoolImpl::List(json["values"].as_array().unwrap().iter().map(|v| to_location(v, root, context)).collect())
      },
      name => { panic!("Unsupported location pool type: {name}") }
    };

    let result = LocationPool {
      json, root, _impl
    };
    result
  }
  fn pop(&mut self) -> (f64, f64) {
    match &mut self._impl {
      LocationPoolImpl::List(v) => {
        v.pop_front().unwrap()
      }
    }
  }
}

struct ShapePool {

}

enum TaskConfigObject<'a> {
  LocationPoolObject(LocationPool<'a>),
  ShapePoolObject(ShapePool),
}

struct TaskConfig<'a> {
  root: &'a serde_json::Value,
  cache: HashMap<String, TaskConfigObject<'a>>,
  context: &'a TaskContext
}

impl<'a> TaskConfig<'a> {
  fn new(root: &'a serde_json::Value, context: &'a TaskContext) -> TaskConfig<'a> {
    TaskConfig { root, cache: HashMap::<String, TaskConfigObject<'a>>::new(), context }
  }

  fn get<'b>(&'b mut self, name: &str) -> &'b mut TaskConfigObject<'a> {

    if !self.cache.contains_key(name) {
      let json_value = &self.root[name];
      let json_type = json_value["type"].as_str().unwrap();
      let object = match json_type {
        "location_pool" => {
          TaskConfigObject::LocationPoolObject(LocationPool::new(json_value, self.root, self.context))
        },
        _ => {
          panic!("Unexpected type, {json_type}");
        }
      };
      self.cache.insert(name.to_string(), object);
    }

    self.cache.get_mut(name).unwrap()
  }
}

struct Target {

}

impl Target {
  fn new<'a, 'b>(target_config: &serde_json::Value, task_config: &'b mut TaskConfig<'a>) -> Target {
    let location_config = &target_config["location"];
    let location = if location_config["type"].as_str().unwrap() == "pool" {
      let pool_name = location_config["name"].as_str().unwrap();
      let TaskConfigObject::LocationPoolObject(pool) = task_config.get(pool_name) else {
        panic!("Not a location pool: {pool_name}")
      };
      pool.pop()
    } else {
      panic!("ERROR")
    };
    Target{}
  }
}

#[async_trait]
impl BehaviorTask for JsonTask {
  async fn run(&self, context: Arc<TaskContext>) -> TaskResult {
    let config = &context.config();
    let jsonpath = config["file"].as_str().unwrap();
    let jsonreader = File::open(jsonpath).unwrap();
    let json: serde_json::Value = serde_json::from_reader(jsonreader).unwrap();
    let mut task_config = TaskConfig::new(&json, &*context);

    let screen_distance_m = json["screen_distance_m"].as_f64();
    let screen_dpi = json["screen_dpi"].as_f64();

    let shapes = to_object_array(&json["shapes"]);
    let locations = to_object_array(&json["choice_location_pool"]);

    let locations = json["targets"]
      .as_array()
      .unwrap()
      .iter()
      .map(|v| Target::new(v, &mut task_config));

    let states: Vec<_> = json["states"].as_array().unwrap().iter().map(|v| v.as_object().unwrap()).collect();

    let mut init_state: Option<serde_json::Map<String, serde_json::Value>> = None;
    for state in states {
      if state["init"].as_bool().unwrap() {
        init_state = Some(state.clone());
      }
    }
    

    let current_state = init_state.expect("No init state found");



    TaskResult { success: true, cancelled: false }
  }

  fn render(&self, canvas: &Canvas, window: Window) {

  }
}