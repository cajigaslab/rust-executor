#![allow(unused)]

use std::{collections::HashMap, fs::File, sync::Arc};
use crate::{pb::task_controller_grpc::TaskResult};

use async_trait::async_trait;
use super::{BehaviorTask, PointSubscription, TaskContext, Window};
use skia_safe::Canvas;
use std::collections::{VecDeque};

use serde::{Serialize, Deserialize};

use rand::RngExt;
use rand::seq::{IndexedRandom, SliceRandom};

#[derive(Serialize, Deserialize, Clone)]
pub enum Origin {
  UL,
  UR,
  LL,
  LR,
  Center,
}

impl Origin {
  fn render(&self, context: &TaskContext) -> (f64, f64) {
    let size = context.canvas_size();
    match self {
      Self::UL => { (0.0, 0.0) }
      Self::UR => { (size.0 as f64, 0.0) }
      Self::LL => { (0.0, size.1 as f64) }
      Self::LR => { (size.0 as f64, size.1 as f64) }
      Self::Center => { ((size.0/2) as f64, (size.1/2) as f64) }
    }
  }
}

#[derive(Serialize, Deserialize, Clone)]
pub enum Number {
  Literal(f64),
  Param(String, String),
}

fn get_nested_f64(config: &serde_json::Value, key: &str, sub_key: &str) -> f64 {
  config
    .get(key)
    .and_then(|v| v.get(sub_key))
    .and_then(serde_json::Value::as_f64)
    .unwrap_or_else(|| {
      panic!("task config missing required number field {key:?}.{sub_key:?}: {config}")
    })
}

impl Number {
  fn render(&self, context: &TaskContext) -> f64 {
    let params = &context.config();
    match self {
      Self::Literal(value) => { *value },
      Self::Param(key, expr) => {
        let expr: meval::Expr = expr.parse().unwrap();
        let op = expr.bind("x").unwrap();
        let value_config = params
          .get(key)
          .unwrap_or_else(|| panic!("task config missing required field {key:?}: {params}"));
        let value = match value_config {
          serde_json::Value::Number(n) => {
            let tmp = n
            .as_f64()
            .unwrap_or_else(|| panic!("task config field {key:?} is not a valid number: {params}"));
            op(tmp)
          },
          serde_json::Value::Object(_) => {
            let min = get_nested_f64(params, key, "min");
            let max = get_nested_f64(params, key, "max");
            op(rand::rng().random_range(min..=max))
          }
          other => {
            panic!("task config field {key:?} should be a number or {{min,max}} object, got {other}")
          }
        };
        value
      },
    }
  }
}

#[derive(Serialize, Deserialize, Clone)]
pub enum Length {
  Arcs{value: Number},
  Pixels{value: Number},
}

impl Length {
  fn render(&self, context: &TaskContext) -> f64 {
    let params = &context.config();
    match self {
      Self::Pixels { value } => { value.render(context) }
      Self::Arcs { value } => {
        let screen_distance_m = params["screen_distance_m"].as_f64().unwrap();
        let screen_dpi = params["screen_dpi"].as_f64().unwrap();
        //pixels/radian ~ (pixels/inch)*(inch/cm)*(cm/m)*(distance_m) for small angles (x ~ tan(x))
        let scale = screen_dpi/2.54*100.0*screen_distance_m;
        value.render(context)*scale
      }
    }
  }
}

#[derive(Serialize, Deserialize, Clone)]
pub enum Angle {
  Radians{value: Number},
  Degrees{value: Number},
}

impl Angle {
  fn render(&self, context: &TaskContext) -> f64 {
    match self {
      Self::Radians { value } => { value.render(context) }
      Self::Degrees { value } => { std::f64::consts::PI/180.0*value.render(context) }
    }
  }
}

#[derive(Serialize, Deserialize, Clone)]
pub enum Location {
  Polar{radius: Length, angle: Angle, origin: Origin},
  Cartesian{x: Length, y: Length, origin: Origin},
  Pool(String),
}

trait Named {
  fn name(&self) -> &str;
}

impl Named for LocationPool {
  fn name(&self) -> &str {
    self.name.as_str()
  }
}

impl Named for TargetPool {
  fn name(&self) -> &str {
    self.name.as_str()
  }
}

fn get_named<'a, T: Named>(list: &'a mut Vec<T>, name: &str) -> &'a mut T {
  for l in list.iter_mut() {
    if l.name() == name {
      return l;
    }
  }
  panic!("Not found: {name}");
}

impl Location {
  fn render(&self, root: &mut DeclTask, context: &TaskContext) -> (f64, f64) {
    match self {
      Self::Polar { radius, angle, origin } => {
        let radius = radius.render(context);
        let angle = angle.render(context);
        let x = radius*angle.cos();
        let y = radius*angle.sin();
        let (ox, oy) = origin.render(context);
        (ox + x, oy + y)
      }
      Self::Cartesian { x, y, origin } => {
        let x = x.render(context);
        let y = y.render(context);
        let (ox, oy) = origin.render(context);
        (ox + x, oy + y)
      },
      Self::Pool(pool) => {
        let loc = get_named(&mut root.locations, pool).pop();  // borrow of self.locations ends here
        loc.render(root, context)                                  // self is free again
      }
    }
  }
}

#[derive(Serialize, Deserialize, Clone)]
pub enum ImagePoolImp {
  List(Vec<Image>),
}

#[derive(Serialize, Deserialize, Clone)]
pub struct ImagePool {
  name: String,
  imp: ImagePoolImp
}

impl ImagePool {
  fn pop(&mut self) -> Image {
    match &mut self.imp {
      ImagePoolImp::List(images) => {
        let index: usize = rand::rng().random_range(..images.len());
        images.remove(index)
      }
    }
  }
}

#[derive(Serialize, Deserialize, Clone)]
pub enum LocationPoolImp {
  List(Vec<Location>),
}

#[derive(Serialize, Deserialize, Clone)]
pub struct LocationPool {
  name: String,
  imp: LocationPoolImp,
}

impl LocationPool {
  fn pop(&mut self) -> Location {
    match &mut self.imp {
      LocationPoolImp::List(locations) => {
        let index: usize = rand::rng().random_range(..locations.len());
        locations.remove(index)
      }
    }
  }
}

#[derive(Serialize, Deserialize, Clone)]
pub enum PathOp {
  MoveTo(Location),
  LineTo(Location),
}

#[derive(Serialize, Deserialize, Clone)]
pub enum Image {
  Path(Vec<PathOp>),
  Pool(String),
}

pub struct ImageRender {
  path: skia_safe::Path,
}

impl ImageRender {
  fn draw(&self, canvas: &skia_safe::Canvas, paint: &skia_safe::Paint) {
    canvas.draw_path(&self.path, paint);
  }
}

impl Image {
  fn render(&self, root: &mut DeclTask, context: &TaskContext) -> ImageRender {
    match self {
      Self::Path(path) => {
        let mut builder = skia_safe::PathBuilder::new();
        for op in path {
          match op {
            PathOp::MoveTo(location) => {
              let point = location.render(root, context);
              builder.move_to((point.0 as f32, point.1 as f32));
            },
            PathOp::LineTo(location) => {
              let point = location.render(root, context);
              builder.line_to((point.0 as f32, point.1 as f32));
            }
          }
        }
        ImageRender { path: builder.detach() }
      }
      Self::Pool(pool) => {
        let image = get_named(&mut root.images, pool).pop();  // borrow of self.locations ends here
        image.render(root, context)                                  // self is free again
      }
    }
  }
}

#[derive(Serialize, Deserialize, Clone)]
pub struct Target {
  location: Location,
  image: Image,
}

pub struct TargetRender {
  location: (f64, f64),
  image: ImageRender,
}

impl Target {
  fn render(&self, root: &mut DeclTask, context: &TaskContext) -> Target {
    let location = self.location.render(root, context);
    let image = self.image.render(root, context);
    Target { location, image }
  }
}

#[derive(Serialize, Deserialize, Clone)]
pub enum TargetPoolImp {
  List(Vec<Target>),
}

#[derive(Serialize, Deserialize, Clone)]
pub struct TargetPool {
  name: String,
  imp: TargetPoolImp,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct State {
  init: bool,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct DeclTask {
  screen_distance_m: Number,
  screen_dpi: Number,
  images: Vec<ImagePool>,
  locations: Vec<LocationPool>,
  targets: Vec<TargetPool>,
  states: Vec<State>,
}

impl DeclTask {
}
//
//pub struct LocationPool<'a> {
//  pool: &'a mut LocationPoolConfig,
//}
//
//impl<'a> LocationPool<'a> {
//  pub fn new(pool: &'a mut LocationPoolConfig) -> LocationPool<'a> {
//    LocationPool {
//      pool
//    }
//  }
//
//  pub fn pop(&mut self, root: &mut DeclTask<'a>, context: &TaskContext) -> (f64, f64) {
//    match &mut self.pool.imp {
//      LocationPoolImpConfig::List { locations } => {
//        assert!(locations.len() > 0, "Location Pool exhausted: {}", self.pool.name);
//        let index: usize = rand::rng().random_range(..locations.len());
//        locations.remove(index).render(root, context)
//      }
//    }
//  }
//}
//
//pub struct TargetPool<'a> {
//  pool: &'a mut TargetPoolConfig,
//}
//
//impl<'a> TargetPool<'a> {
//  pub fn new(pool: &'a mut TargetPoolConfig) -> TargetPool<'a> {
//    TargetPool {
//      pool
//    }
//  }
//
//  pub fn pop(&mut self, root: &DeclTask<'a>, context: &TaskContext) -> Target {
//    match &mut self.pool.imp {
//      TargetPoolImpConfig::List(targets) => {
//        assert!(targets.len() > 0, "Target Pool exhausted: {}", self.pool.name);
//        let index: usize = rand::rng().random_range(..targets.len());
//        targets.remove(index).render(root, context)
//      }
//    }
//  }
//}

pub struct JsonTask {
  config: DeclTask,
}

impl JsonTask {
  pub fn new(config: DeclTask) -> JsonTask {
    JsonTask {config}
  }
}

pub const BASIC_DECL_TASK: DeclTask = DeclTask {
  screen_distance_m: Number::Literal(0.0),
  screen_dpi: Number::Literal(0.0),
  images: vec![],
  locations: vec![],
  targets: vec![],
  states: vec![],
};

#[async_trait]
impl BehaviorTask for JsonTask {
  async fn run(&self, context: Arc<TaskContext>) -> TaskResult {
    //let params = &context.config();
    let mut config = self.config.clone();
    //let decl_task = DeclTask::new(&mut config, &context);

    TaskResult { success: true, cancelled: false }
  }

  fn render(&self, canvas: &Canvas, window: Window) {

  }
}
