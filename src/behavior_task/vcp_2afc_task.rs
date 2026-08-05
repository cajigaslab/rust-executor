use core::num;
use std::sync::{Arc, Mutex, OnceLock};

use ndarray;
use itertools::{iproduct};

use async_trait::async_trait;
use kira::sound::static_sound::StaticSoundData;
use super::converter::{Converter, rad_to_deg, deg_to_rad};
use rand::seq::{IndexedRandom, SliceRandom};

use crate::behavior_task::converter;
use crate::pb::task_controller_grpc::TaskResult;
use crate::pb::thalamus_grpc::{AnalogResponse, Span};

use super::{BehaviorTask, PointSubscription, TaskContext, Window};

use skia_safe::{
  Canvas, Color4f, Font, FontMgr, Paint, PaintStyle, Path, PathBuilder, Rect, Shader, TileMode, PathDirection
};

pub struct Vcp2AfcTask {

}

struct Static {
  success_sound: StaticSoundData,
  abort_sound: StaticSoundData,
  failure_sound: StaticSoundData,
  loc_rand_pos: Vec::<(i32, i32)>,
  loc_rand_pos_i: i32,
  _loc_eccentric_circle_num: i32,
  _loc_polar_step_deg: i32,
  _loc_ecc_pix_min: f64,
  _loc_ecc_pix_max: f64,
  _loc_sector1_min: i32,
  _loc_sector1_max: i32,
  _loc_sector2_min: i32,
  _loc_sector2_max: i32,
  sample_pos_pix: (i32, i32),
  converter: Converter,
  loc_correct_idx: i32,
}
static STATIC: OnceLock<Mutex<Static>> = OnceLock::new();
static photodiode_blinking_square: Color4f = Color4f::new(1.0, 1.0, 1.0, 1.0);
static photodiode_static_square: Color4f = Color4f::new(0.0, 0.0, 0.0, 1.0);

static task_groups: &[&str] = &["Shapes", "Locations"];
static shapes: &[&str] = &["square", "circle", "triangle"];

fn make_converter(config: &serde_json::Value) -> converter::Converter {
  let monitorsubj_w_pix: i32 = config["monitorsubj_W_pix"].as_i64().unwrap().try_into().unwrap();
  let monitorsubj_h_pix: i32 = config["monitorsubj_H_pix"].as_i64().unwrap().try_into().unwrap();
  let monitorsubj_dist_m = config["monitorsubj_dist_m"].as_f64().unwrap();
  let monitorsubj_width_m = config["monitorsubj_width_m"].as_f64().unwrap();
  Converter::new(
    (monitorsubj_w_pix, monitorsubj_h_pix),
    monitorsubj_width_m,
    monitorsubj_dist_m,
  )
}

impl Static {
  fn new(config: &serde_json::Value) -> Self {
    let success_sound =
      StaticSoundData::from_file(r"C:\Thalamus-Extensions\seokhee\success_clip.wav").unwrap();
    let abort_sound =
      StaticSoundData::from_file(r"C:\Thalamus-Extensions\seokhee\failure_clip.wav").unwrap();
    let failure_sound =
      StaticSoundData::from_file(r"C:\Thalamus-Extensions\seokhee\timeout_failure.wav").unwrap();

    let mut result = Static {
      success_sound,
      abort_sound,
      failure_sound,
      loc_rand_pos: Vec::<(i32, i32)>::new(),
      loc_rand_pos_i: 0,
      _loc_eccentric_circle_num: 0,
      _loc_polar_step_deg: 0,
      _loc_ecc_pix_min: 0.0,
      _loc_ecc_pix_max: 0.0,
      _loc_sector1_min: 0,
      _loc_sector1_max: 0,
      _loc_sector2_min: 0,
      _loc_sector2_max: 0,
      sample_pos_pix: (0, 0),
      converter: Converter::from_config(config),
      loc_correct_idx: 0,
    };
    result.build_loc_rand_pos(config);

    result
  }

  fn build_loc_rand_pos(&mut self, config: &serde_json::Value) {
    let converter = &self.converter;
    self._loc_eccentric_circle_num = config["loc_eccentric_circle_num"].as_i64().unwrap().try_into().unwrap();
    self._loc_polar_step_deg       = config["loc_polar_step_deg"].as_i64().unwrap().try_into().unwrap();
    self._loc_ecc_pix_min          = converter.deg_to_pixel_rel(config["loc_eccentricity_deg"]["min"].as_f64().unwrap());
    self._loc_ecc_pix_max          = converter.deg_to_pixel_rel(config["loc_eccentricity_deg"]["max"].as_f64().unwrap());
    self._loc_sector1_min          = config["loc_angle_sector1_deg"]["min"].as_i64().unwrap().try_into().unwrap();
    self._loc_sector1_max          = config["loc_angle_sector1_deg"]["max"].as_i64().unwrap().try_into().unwrap();
    self._loc_sector2_min          = config["loc_angle_sector2_deg"]["min"].as_i64().unwrap().try_into().unwrap();
    self._loc_sector2_max          = config["loc_angle_sector2_deg"]["max"].as_i64().unwrap().try_into().unwrap();
    let radii = ndarray::linspace(self._loc_ecc_pix_min, self._loc_ecc_pix_max,
      self._loc_eccentric_circle_num.try_into().unwrap());
    let angles_rad = get_valid_angles_loc(
      self._loc_polar_step_deg.try_into().unwrap(),
      self._loc_sector1_min.try_into().unwrap(), self._loc_sector1_max.try_into().unwrap(),
      self._loc_sector2_min.try_into().unwrap(), self._loc_sector2_max.try_into().unwrap());
    let angles_deg = angles_rad.iter().copied().map(f64::from).map(rad_to_deg);
    let center = converter.center;

    self.loc_rand_pos = iproduct!(radii, angles_deg)
      .map(|(r, a)| (
        center.0 + ((r + a.cos()) as i32), 
        center.1 + ((r + a.sin()) as i32)))
      .collect();
    self.loc_rand_pos.shuffle(&mut rand::rng());
    self.loc_rand_pos_i = 0
  }

  fn setup_sample_and_choices(&mut self, num_choices: i32) -> (&str, i32) {
    //global sample_shape, choice_shapes, choice_pos, rand_pos, sample_pos_pix, loc_rand_pos_i

    //# 1) sample shape
    let sample_shape = *shapes.choose(&mut rand::rng()).unwrap();

    //# 2) sample position from the same location pool as Locations mode
    self.sample_pos_pix = self.loc_rand_pos[usize::try_from(self.loc_rand_pos_i).unwrap()];
    self.loc_rand_pos_i += 1;

    //# 3) assign shapes – one of them must be the sample
    let mut choice_shapes: Vec<Option<&str>> = vec![None; num_choices.try_into().unwrap()];

    //# randomly choose which index will be correct (the sample)
    let correct_idx = rand::random_range(..usize::try_from(num_choices).unwrap());
    choice_shapes[correct_idx] = Some(sample_shape);

    //# pool of distractors 
    let mut distractors: Vec<&str> = shapes.iter().filter(|s| **s != sample_shape).copied().collect();
    if distractors.is_empty() {
      distractors = vec![sample_shape];
    }

    choice_shapes = choice_shapes.iter().map(|choice_shape| {
      match choice_shape {
        None => {
          let choice = *shapes.choose(&mut rand::rng()).unwrap();
          Some(choice)
        },
        some => *some,
      }
    }).collect();

    return (sample_shape, correct_idx.try_into().unwrap())
  }

  fn setup_locations_trial(&mut self, config: &serde_json::Value) -> (i32, i32) {
    let converter = &self.converter;
    let center = converter.center;

    self.sample_pos_pix = self.loc_rand_pos[usize::try_from(self.loc_rand_pos_i).unwrap()];
    self.loc_rand_pos_i += 1;

    let ecc_pix = converter.deg_to_pixel_rel(config["choice_eccentricity"].as_f64().unwrap()) as i32;
    let loc_left_pos  = (center.0 - ecc_pix, center.1);
    let loc_right_pos = (center.0 + ecc_pix, center.1);

    if self.sample_pos_pix.0 < center.0 {
        self.loc_correct_idx = 0   // left
    } else if self.sample_pos_pix.0 > center.0 {
        self.loc_correct_idx = 1   // right
    } else {
        self.loc_correct_idx = rand::random_range(0..=1);
    }

    if self.loc_correct_idx == 0 {
      loc_left_pos
    } else {
      loc_right_pos
    }
  }
  //  """
  //  For this trial (Locations mode):
  //    - Pick the next sample position from loc_rand_pos.
  //    - Determine correct choice side (left/right) based on sample's X vs center.
  //    - Set loc_left_pos, loc_right_pos (always horizontal at choice_eccentricity).
  //  Returns targetpos_pix (the correct choice position).
  //  """
  //  global sample_pos_pix, loc_correct_idx, loc_left_pos, loc_right_pos, loc_rand_pos_i
  //  sp = loc_rand_pos[loc_rand_pos_i]
  //  sample_pos_pix = QPoint(int(sp[0]), int(sp[1]))
  //  loc_rand_pos_i += 1
//
  //  ecc_pix = converter.deg_to_pixel_rel(config['choice_eccentricity'])
  //  loc_left_pos  = QPoint(int(center.x() - ecc_pix), center.y())
  //  loc_right_pos = QPoint(int(center.x() + ecc_pix), center.y())
//
  //  if sample_pos_pix.x() < center.x():
  //      loc_correct_idx = 0   # left
  //  elif sample_pos_pix.x() > center.x():
  //      loc_correct_idx = 1   # right
  //  else:
  //      loc_correct_idx = random.randint(0, 1)
//
  //  return loc_left_pos if loc_correct_idx == 0 else loc_right_pos

  fn sync_config(&mut self, config: &serde_json::Value) {
    let converter = make_converter(config);
    let _loc_eccentric_circle_num = config["loc_eccentric_circle_num"].as_i64().unwrap().try_into().unwrap();
    let _loc_polar_step_deg       = config["loc_polar_step_deg"].as_i64().unwrap().try_into().unwrap();
    let _loc_ecc_pix_min          = converter.deg_to_pixel_rel(config["loc_eccentricity_deg"]["min"].as_f64().unwrap());
    let _loc_ecc_pix_max          = converter.deg_to_pixel_rel(config["loc_eccentricity_deg"]["max"].as_f64().unwrap());
    let _loc_sector1_min          = config["loc_angle_sector1_deg"]["min"].as_i64().unwrap().try_into().unwrap();
    let _loc_sector1_max          = config["loc_angle_sector1_deg"]["max"].as_i64().unwrap().try_into().unwrap();
    let _loc_sector2_min          = config["loc_angle_sector2_deg"]["min"].as_i64().unwrap().try_into().unwrap();
    let _loc_sector2_max          = config["loc_angle_sector2_deg"]["max"].as_i64().unwrap().try_into().unwrap();

    let new_config = (
      _loc_eccentric_circle_num,
      _loc_polar_step_deg,
      _loc_ecc_pix_min as i32,
      _loc_ecc_pix_max as i32,
      _loc_sector1_min,
      _loc_sector1_max,
      _loc_sector2_min,
      _loc_sector2_max,
    );
    let old_config = (
      self._loc_eccentric_circle_num,
      self._loc_polar_step_deg,
      self._loc_ecc_pix_min as i32,
      self._loc_ecc_pix_max as i32,
      self._loc_sector1_min,
      self._loc_sector1_max,
      self._loc_sector2_min,
      self._loc_sector2_max,
    );
    if old_config != new_config {
      self.build_loc_rand_pos(config);
    }
    if self.loc_rand_pos_i >= self.loc_rand_pos.len().try_into().unwrap() {
      self.loc_rand_pos.shuffle(&mut rand::rng());
      self.loc_rand_pos_i = 0;
    }
  }
}

impl Vcp2AfcTask {
  pub fn new() -> Vcp2AfcTask {
    Vcp2AfcTask {}
  }
}

fn angle_in_sector(angle: i32, sector_min: i32, sector_max: i32) -> bool {
  let angle_mod = angle % 360;
  let sector_min_mod = sector_min % 360;
  let sector_max_mod = sector_max % 360;
  if sector_min_mod < sector_max_mod {
      return sector_min_mod <= angle_mod && angle_mod < sector_max_mod
  } else if sector_min_mod > sector_max_mod {
      return angle_mod >= sector_min_mod || angle_mod < sector_max_mod;
  } else {
    return true
  }
}

fn get_valid_angles(step: i32, sector1_min: i32, sector1_max: i32) -> Vec<i32> {
  let ustep: usize = step.try_into().unwrap();
  let mut angles:Vec<i32> = (0..360).step_by(ustep+1)
  .filter(|angle| angle_in_sector(*angle, sector1_min, sector1_max))
  .collect();

  angles.sort();
  angles.dedup();
  angles.pop();
  angles
}

fn get_valid_angles_loc(step_deg: i32, sector1_min: i32, sector1_max: i32, sector2_min: i32, sector2_max: i32) -> Vec<i32> {
  let ustep: usize = step_deg.try_into().unwrap();
  let mut angles:Vec<i32> = (0..360).step_by(ustep)
  .filter(|angle| {
    angle_in_sector(*angle, sector1_min, sector1_max)
    || angle_in_sector(*angle, sector2_min, sector2_max)
    && !(*angle == 90 || *angle == 270)
  })
  .collect();

  angles.sort();
  angles.dedup();
  angles
}

#[async_trait]
impl BehaviorTask for Vcp2AfcTask {
  async fn run(&self, context: Arc<TaskContext>) -> TaskResult {
    let config = &context.config();
    let converter = Converter::from_config(config);

    let task_group = config["task_group"].as_str().unwrap();
    let num_choices = config.get("num_choices").unwrap().as_i64().unwrap().try_into().unwrap();
    let choice_eccentricity = config.get("num_choices").unwrap().as_f64().unwrap();
    let rand_pos: Vec<(f64, f64)> = get_valid_angles(num_choices, 0, 360).iter()
    .map(|ang_deg| {
      let ang_rad = deg_to_rad(*ang_deg as f64);
      let x_deg = choice_eccentricity * ang_rad.cos();
      let y_deg = choice_eccentricity * ang_rad.sin();
      converter.deg_to_pixel_abs(x_deg, y_deg)
    }).collect();

    let _static = STATIC.get_or_init(|| Mutex::new(Static::new(config)));
    _static.lock().unwrap().sync_config(config);

    let mut choice_pos: Option<Vec<(f64, f64)>> = None;
    let sample_and_choices = if task_group == "Shapes" {
      let (sample_shape, correct_idx) = _static.lock().unwrap().setup_sample_and_choices(num_choices);
      let sample_pos_pix = _static.lock().unwrap().sample_pos_pix;
      let targetpos_pix   = rand_pos[usize::try_from(correct_idx).unwrap()].unwrap();
      //_static.
      //choice_pos = Some(rand_pos);
      
      context
        .log(&format!(
          "trial_summary_data.used_values targetposX_pix={}",
          targetpos_pix.0
        ))
        .await;
      context
        .log(&format!(
          "trial_summary_data.used_values targetposY_pix={}",
          targetpos_pix.1
        ))
        .await;

      context
        .log(&format!(
          "trial_summary_data.used_values sample_pos_x_pix={}",
          sample_pos_pix.0
        ))
        .await;
      context
        .log(&format!(
          "trial_summary_data.used_values sample_pos_y_pix={}",
          sample_pos_pix.1
        ))
        .await;
      (Some(sample_shape), Some(correct_idx), sample_pos_pix, targetpos_pix)
    } else {
      let targetpos_pix = _static.lock().unwrap().setup_locations_trial(config);
      let sample_pos_pix = _static.lock().unwrap().sample_pos_pix;
      context
        .log(&format!(
          "trial_summary_data.used_values targetposX_pix={}",
          targetpos_pix.0
        ))
        .await;
      context
        .log(&format!(
          "trial_summary_data.used_values targetposY_pix={}",
          targetpos_pix.1
        ))
        .await;

      context
        .log(&format!(
          "trial_summary_data.used_values sample_pos_x_pix={}",
          sample_pos_pix.0
        ))
        .await;
      context
        .log(&format!(
          "trial_summary_data.used_values sample_pos_y_pix={}",
          sample_pos_pix.1
        ))
        .await;
      (None, None, sample_pos_pix, targetpos_pix)
    };

    let cross_scale = 1.0;//get_f64(config, "cross_scale");
    const VERTICES_DEG: [(f64, f64); 4] = [(-0.25, 0.0), (0.25, 0.0), (0.0, -0.25), (0.0, 0.25)];
    let vertices: Vec<(f64, f64)> = VERTICES_DEG
      .iter()
      .map(|&(x, y)| converter.deg_to_pixel_abs(cross_scale*x, cross_scale*y))
      .collect();

    let mut cross_builder = PathBuilder::new();
    cross_builder.move_to((vertices[0].0 as f32, vertices[0].1 as f32));
    cross_builder.line_to((vertices[1].0 as f32, vertices[1].1 as f32));
    cross_builder.move_to((vertices[2].0 as f32, vertices[2].1 as f32));
    cross_builder.line_to((vertices[3].0 as f32, vertices[3].1 as f32));
    let cross: Path = cross_builder.detach();

    // Build shape paths — size controlled by sample_size_deg (half-width in degrees)
    let s = config["sample_size_deg"].as_f64().unwrap() / 2.0;
  
    const square_deg: [(f64, f64); 4] = [(-s/2, -s/2), (s/2, -s/2), (s/2, s/2), (-s/2, s/2)];
    let square_vertices = square_deg.iter().copied().map(|(x, y)|converter.deg_to_pixel_abs(x, y)).collect();
    let mut square_builder = PathBuilder::new();
    square_builder.move_to((square_vertices[0][0], square_vertices[0][1]));
    square_builder.line_to((square_vertices[1][0], square_vertices[1][1]));
    square_builder.line_to((square_vertices[2][0], square_vertices[2][1]));
    square_builder.line_to((square_vertices[3][0], square_vertices[3][1]));
    square_builder.close();
    let square = square_builder.detach();
  
    const triangle_deg: [(f64, f64); 3] = [(0.0, -s/2), (s/2, s/2), (-s/2, s/2)];
    let triangle_vertices = triangle_deg.iter().copied().map(|(x, y)|converter.deg_to_pixel_abs(x, y)).collect();
    let mut triangle_builder = PathBuilder::new();
    triangle_builder.move_to((triangle_vertices[0][0], triangle_vertices[0][1]));
    triangle_builder.line_to((triangle_vertices[1][0], triangle_vertices[1][1]));
    triangle_builder.line_to((triangle_vertices[2][0], triangle_vertices[2][1]));
    triangle_builder.close();
    let triangle = triangle_builder.detach();
  
    let (x0, y0) = square_vertices[0];
    let (x2, y2) = square_vertices[2];
    let circle_builder = PathBuilder::new();
    circle_builder.add_oval(Rect::from_ltrb(x0, y0, x2, y2), PathDirection::CW, 0);
    let circle = circle_builder.detach();

    TaskResult { success: true, cancelled: false }
  }

  fn render(&self,canvas: &Canvas,window: Window) {
    todo!()
  }
}