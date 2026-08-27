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
  Arcs(Angle),
  Pixels(Number),
}

impl Length {
  fn render(&self, context: &TaskContext) -> f64 {
    let params = &context.config();
    match self {
      Self::Pixels(value) => { value.render(context) }
      Self::Arcs (value) => {
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
  Radians(Number),
  Degrees(Number),
}

impl Angle {
  fn render(&self, context: &TaskContext) -> f64 {
    match self {
      Self::Radians(value) => { value.render(context) }
      Self::Degrees(value) => { std::f64::consts::PI/180.0*value.render(context) }
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

impl Named for ImagePool {
  fn name(&self) -> &str {
    self.name.as_str()
  }
}

impl Named for TargetPool {
  fn name(&self) -> &str {
    self.name.as_str()
  }
}

impl Named for LocationPoolExecutor {
  fn name(&self) -> &str {
    self.config.name.as_str()
  }
}

impl Named for ImagePoolExecutor {
  fn name(&self) -> &str {
    self.config.name.as_str()
  }
}

impl Named for TargetPoolExecutor {
  fn name(&self) -> &str {
    self.config.name.as_str()
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
  fn render(&self, root: &mut DeclTaskExecutor, context: &TaskContext) -> (f64, f64) {
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
pub enum ImageImp {
  Path(Vec<PathOp>),
  Pool(String),
}

#[derive(Serialize, Deserialize, Clone)]
pub struct Image {
  name: String,
  imp: ImageImp,
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
  fn render(&self, root: &mut DeclTaskExecutor, context: &TaskContext) -> ImageRender {
    match &self.imp {
      ImageImp::Path(path) => {
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
      ImageImp::Pool(pool) => {
        let image = get_named(&mut root.images, pool.as_str()).pop();  // borrow of self.locations ends here
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
  fn render(&self, root: &mut DeclTaskExecutor, context: &TaskContext) -> TargetRender {
    let location = self.location.render(root, context);
    let image = self.image.render(root, context);
    TargetRender { location, image }
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

pub struct LocationPoolExecutor {
  config: LocationPool,
  sequence: Vec<usize>,
  index: usize
}

impl LocationPoolExecutor {
  fn new(config: LocationPool) -> LocationPoolExecutor {
    let sequence = match &config.imp {
      LocationPoolImp::List(locations) => {
        (0..locations.len()).collect()
      },
      _ => vec![],
    };
    LocationPoolExecutor { config, sequence, index: 0 }
  }

  fn pop(&mut self) -> Location {
    match &self.config.imp {
      LocationPoolImp::List(locations) => {
        let location_index: usize = self.sequence[self.index];
        self.index += 1;
        locations[location_index].clone()
      }
    }
  }

  fn reset(&mut self) {
    self.index = 0;
  }
}

pub struct ImagePoolExecutor {
  config: ImagePool,
  sequence: Vec<usize>,
  index: usize
}

impl ImagePoolExecutor {
  fn new(config: ImagePool) -> ImagePoolExecutor {
    let sequence = match &config.imp {
      ImagePoolImp::List(images) => {
        (0..images.len()).collect()
      },
      _ => vec![],
    };
    ImagePoolExecutor { config, sequence, index: 0 }
  }

  fn pop(&mut self) -> Image {
    match &self.config.imp {
      ImagePoolImp::List(images) => {
        let image_index: usize = self.sequence[self.index];
        self.index += 1;
        images[image_index].clone()
      }
    }
  }

  fn reset(&mut self) {
    self.index = 0;
  }
}

pub struct TargetPoolExecutor {
  config: TargetPool,
  sequence: Vec<usize>,
  index: usize
}

impl TargetPoolExecutor {
  fn new(config: TargetPool) -> TargetPoolExecutor {
    let sequence = match &config.imp {
      TargetPoolImp::List(targets) => {
        (0..targets.len()).collect()
      },
      _ => vec![],
    };
    TargetPoolExecutor { config, sequence, index: 0 }
  }

  fn pop(&mut self) -> Target {
    match &self.config.imp {
      TargetPoolImp::List(targets) => {
        let target_index: usize = self.sequence[self.index];
        self.index += 1;
        targets[target_index].clone()
      }
    }
  }

  fn reset(&mut self) {
    self.index = 0;
  }
}

pub struct StateExecutor {}

pub struct DeclTaskExecutor {
  config: DeclTask,
  images: Vec<ImagePoolExecutor>,
  locations: Vec<LocationPoolExecutor>,
  targets: Vec<TargetPoolExecutor>,
  states: Vec<StateExecutor>,
}

impl DeclTaskExecutor {
  fn new(config: DeclTask) -> DeclTaskExecutor {
    let images = config.images.iter().cloned()
      .map(|c| ImagePoolExecutor::new(c)).collect();
    let locations = config.locations.iter().cloned()
      .map(|c| LocationPoolExecutor::new(c)).collect();
    let targets = config.targets.iter().cloned()
      .map(|c| TargetPoolExecutor::new(c)).collect();
    DeclTaskExecutor { config, images, locations, targets, states: vec![] }
  }
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

pub fn basic_decl_task() -> DeclTask {
  DeclTask {
    screen_distance_m: Number::Literal(0.0),
    screen_dpi: Number::Literal(0.0),
    images: vec![ImagePool{
      name: "Images".to_string(), 
      imp: ImagePoolImp::List(vec![
        Image {
          name: "Square".to_string(),
          imp: ImageImp::Path(vec![
                 PathOp::MoveTo(Location::Cartesian {
                   x: Length::Arcs(Angle::Degrees(Number::Param("sample_size_deg".to_string(), "-x/2.0".to_string()))),
                   y: Length::Arcs(Angle::Degrees(Number::Param("sample_size_deg".to_string(), "-x/2.0".to_string()))),
                   origin: Origin::Center }),
                 PathOp::MoveTo(Location::Cartesian {
                   x: Length::Arcs(Angle::Degrees(Number::Param("sample_size_deg".to_string(), "x/2.0".to_string()))),
                   y: Length::Arcs(Angle::Degrees(Number::Param("sample_size_deg".to_string(), "-x/2.0".to_string()))),
                   origin: Origin::Center }),
                 PathOp::MoveTo(Location::Cartesian {
                   x: Length::Arcs(Angle::Degrees(Number::Param("sample_size_deg".to_string(), "x/2.0".to_string()))),
                   y: Length::Arcs(Angle::Degrees(Number::Param("sample_size_deg".to_string(), "x/2.0".to_string()))),
                   origin: Origin::Center }),
                 PathOp::MoveTo(Location::Cartesian {
                   x: Length::Arcs(Angle::Degrees(Number::Param("sample_size_deg".to_string(), "-x/2.0".to_string()))),
                   y: Length::Arcs(Angle::Degrees(Number::Param("sample_size_deg".to_string(), "x/2.0".to_string()))),
                   origin: Origin::Center })
                 ])
        }
        
      ])}],
    locations: vec![LocationPool{
      name: "Choices".to_string(),
      imp: LocationPoolImp::List(vec![
        Location::Polar {
          radius: Length::Arcs(Angle::Degrees(Number::Literal(10.0))),
          angle: Angle::Degrees(Number::Literal(0.0)),
          origin: Origin::Center },
        //Location::Polar {
        //  radius: Length::Arcs(Angle::Degrees(Number::Literal(10.0))),
        //  angle: Angle::Degrees(Number::Literal(180.0)),
        //  origin: Origin::Center }
      ])
    }],
    targets: vec![TargetPool{
      name: "Choices".to_string(),
      imp: TargetPoolImp::List(vec![
        Target {
          location: Location::Pool("Choices".to_string()),
          image: Image::Pool("Images".to_string()),
        }
      ])
    }],
    states: vec![],
  }
}

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
