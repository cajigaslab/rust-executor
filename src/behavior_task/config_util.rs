use skia_safe::{
  Color4f
};

use serde_json;
use itertools;

use rand::seq::{IndexedRandom, SliceRandom};

pub fn get_color(value: &serde_json::Value) -> Color4f {
  let r = value.as_f64().unwrap()/255.0;
  let g = value.as_f64().unwrap()/255.0;
  let b = value.as_f64().unwrap()/255.0;
  Color4f::new(r as f32, g as f32, b as f32, 1.0)
}

pub fn get_f64(value: &serde_json::Value) -> f64 {
  match value {
    serde_json::Value::Object(dict) => {
      let min = dict["min"].as_f64().unwrap();
      let max = dict["max"].as_f64().unwrap();
      rand::random_range(min..=max)
    }
    serde_json::Value::Number(num) => {
      num.as_f64().unwrap()
    },
    _ => panic!("Number field must either be an object or number.")
  }
}

pub fn get_f64_with_step(value: &serde_json::Value, step: f64) -> f64 {
  match value {
    serde_json::Value::Object(dict) => {
      let min = dict["min"].as_f64().unwrap();
      let max = dict["max"].as_f64().unwrap();
      let slice: Vec<f64> = itertools::iterate(min, |&x| x+step).take_while(|&x| x < max+step).collect();
      *slice.choose(&mut rand::rng()).unwrap()
    }
    serde_json::Value::Number(num) => {
      num.as_f64().unwrap()
    },
    _ => panic!("Number field must either be an object or number.")
  }
}
