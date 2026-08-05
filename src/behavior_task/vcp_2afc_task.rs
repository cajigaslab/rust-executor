use std::sync::{Arc, Mutex, OnceLock};

use ndarray;
use itertools::{iproduct};
use std::time::{Duration};

use async_trait::async_trait;
use kira::sound::static_sound::StaticSoundData;
use super::converter::{Converter, deg_to_rad};
use super::config_util::{get_f64, get_color, get_f64_with_step};
use rand::seq::{IndexedRandom, SliceRandom};

use crate::pb::task_controller_grpc::TaskResult;
#[allow(unused)]
use crate::pb::thalamus_grpc::{AnalogResponse, Span};

#[allow(unused)]
use super::{BehaviorTask, PointSubscription, TaskContext, Window, wait_for, wait_for_hold};

#[allow(unused)]
use skia_safe::{
  Canvas, Color4f, Font, FontMgr, Paint, PaintStyle, Path, PathBuilder, Rect, Shader, TileMode, PathDirection, Point
};

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
enum State {
  Null,
  AcquireFixation,
  Fixate,
  PresentTarget,
  #[allow(dead_code)] // matches the Python source's own unused HOLD_TARGET0
  Delay,
  GoCue,
  AcquireTarget,
  HoldTarget,
  Success,
  FailureSaccade,
  FailureHold,
  AbortFixation,
  AbortTarget,
  AbortDelay,
}

pub struct Vcp2AfcTask {
  state: Mutex<State>
}

#[allow(unused)]
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
  choice_shapes: Vec<Option<&'static str>>,
  trial_num: i32,
  trial_success_count: i32,
  trial_abort_count: i32,
  trial_failure_count: i32,
}
static STATIC: OnceLock<Mutex<Static>> = OnceLock::new();
#[allow(unused)]
static PHOTODIODE_BLINKING_SQUARE: Color4f = Color4f::new(1.0, 1.0, 1.0, 1.0);
#[allow(unused)]
static PHOTODIODE_STATIC_SQUARE: Color4f = Color4f::new(0.0, 0.0, 0.0, 1.0);

static SHAPES: &[&str] = &["square", "circle", "triangle"];

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
      choice_shapes: Vec::<Option<&str>>::new(),
      trial_num: 0,
      trial_success_count: 0,
      trial_abort_count: 0,
      trial_failure_count: 0,
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

    let angles_deg = get_valid_angles_loc(
      self._loc_polar_step_deg.try_into().unwrap(),
      self._loc_sector1_min.try_into().unwrap(), self._loc_sector1_max.try_into().unwrap(),
      self._loc_sector2_min.try_into().unwrap(), self._loc_sector2_max.try_into().unwrap());
    let angles_rad = angles_deg.iter().copied().map(f64::from).map(deg_to_rad);
    let center = converter.center;

    self.loc_rand_pos = iproduct!(radii, angles_rad)
      .map(|(r, a)| (
        center.0 + ((r + a.cos()) as i32), 
        center.1 + ((r + a.sin()) as i32)))
      .collect();
    self.loc_rand_pos.shuffle(&mut rand::rng());
    self.loc_rand_pos_i = 0
  }

  fn setup_sample_and_choices(&mut self, num_choices: i32) -> (&'static str, i32) {
    //global sample_shape, choice_shapes, choice_pos, rand_pos, sample_pos_pix, loc_rand_pos_i

    //# 1) sample shape
    let sample_shape = *SHAPES.choose(&mut rand::rng()).unwrap();

    //# 2) sample position from the same location pool as Locations mode
    self.sample_pos_pix = self.loc_rand_pos[usize::try_from(self.loc_rand_pos_i).unwrap()];
    self.loc_rand_pos_i += 1;

    //# 3) assign shapes – one of them must be the sample
    let unum_choices = usize::try_from(num_choices).unwrap();
    let mut choice_shapes: Vec<Option<&str>> = vec![None; unum_choices];

    //# randomly choose which index will be correct (the sample)
    let correct_idx = rand::random_range(..unum_choices);
    choice_shapes[correct_idx] = Some(sample_shape);

    //# pool of distractors 
    let mut distractors: Vec<&str> = SHAPES.iter().copied().filter(|s| *s != sample_shape).collect();
    if distractors.is_empty() {
      distractors = vec![sample_shape];
    }

    self.choice_shapes = choice_shapes.iter().map(|choice_shape| {
      match choice_shape {
        None => {
          let choice = *distractors.choose(&mut rand::rng()).unwrap();
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

  fn sync_config(&mut self, config: &serde_json::Value) {
    let converter = Converter::from_config(config);
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
    if usize::try_from(self.loc_rand_pos_i).unwrap() >= self.loc_rand_pos.len() {
      self.loc_rand_pos.shuffle(&mut rand::rng());
      self.loc_rand_pos_i = 0;
    }
  }
}

fn gaze_valid(x: i32, y: i32, monitorsubj_w_pix: i32, monitorsubj_h_pix: i32) -> (i32, i32) {
  if x < 0 || x > monitorsubj_w_pix || y < 0 || y > monitorsubj_h_pix {
    (99999, 99999)
  } else {
    (x, y)
  }
}

fn distance(a: (i32, i32), b: (i32, i32)) -> f64 {
  let dx = (a.0 - b.0) as f64;
  let dy = (a.1 - b.1) as f64;
  (dx * dx + dy * dy).sqrt()
}

impl Vcp2AfcTask {
  pub fn new() -> Vcp2AfcTask {
    Vcp2AfcTask {
      state: Mutex::new(State::Null),
    }
  }
}

fn angle_in_sector(angle: i32, sector_min: i32, sector_max: i32) -> bool {
  let angle_mod = angle % 360;
  let sector_min_mod = sector_min % 360;
  let sector_max_mod = sector_max % 360;
  if sector_min_mod < sector_max_mod {
      sector_min_mod <= angle_mod && angle_mod < sector_max_mod
  } else if sector_min_mod > sector_max_mod {
      angle_mod >= sector_min_mod || angle_mod < sector_max_mod
  } else {
    true
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

macro_rules! log {
  ($context:expr, $($arg:tt)*) => {
    $context.log(&format!($($arg)*)).await
  };
}

fn point_condition<'a>(
  point_queue: &'a PointSubscription,
  last_point_mutex: &'a Mutex<(i32, i32)>,
  within: impl Fn((i32, i32)) -> bool + 'a,
) -> impl Fn() -> bool + 'a {
  move || {
    let mut satisfied = false;
    let mut last_point = last_point_mutex.lock().unwrap();
    for point in point_queue.drain() {
      *last_point = point;
      if within(point) {
        satisfied = true;
      }
    }
    satisfied || within(*last_point)
  }
}

#[async_trait]
impl BehaviorTask for Vcp2AfcTask {
  async fn run(&self, context: Arc<TaskContext>) -> TaskResult {
    let config = &context.config();
    let converter = Converter::from_config(config);
    let monitorsubj_w_pix: i32 = config["monitorsubj_W_pix"].as_i64().unwrap().try_into().unwrap();
    let monitorsubj_h_pix: i32 = config["monitorsubj_H_pix"].as_i64().unwrap().try_into().unwrap();
    let center = converter.center;

    let task_group = config["task_group"].as_str().unwrap();
    let num_choices = config.get("num_choices").unwrap().as_i64().unwrap().try_into().unwrap();
    let choice_eccentricity = config.get("num_choices").unwrap().as_f64().unwrap();
    let rand_pos: Vec<(i32, i32)> = get_valid_angles(num_choices, 0, 360).iter()
    .map(|ang_deg| {
      let ang_rad = deg_to_rad(*ang_deg as f64);
      let x_deg = choice_eccentricity * ang_rad.cos();
      let y_deg = choice_eccentricity * ang_rad.sin();
      let f = converter.deg_to_pixel_abs(x_deg, y_deg);
      (f.0 as i32, f.1 as i32)
    }).collect();

    let _static = STATIC.get_or_init(|| Mutex::new(Static::new(config)));
    _static.lock().unwrap().sync_config(config);

    let sample_and_choices = if task_group == "Shapes" {
      let (sample_shape, correct_idx) = _static.lock().unwrap().setup_sample_and_choices(num_choices);
      let sample_pos_pix = _static.lock().unwrap().sample_pos_pix;
      let targetpos_pix   = rand_pos[usize::try_from(correct_idx).unwrap()];
      //_static.
      //choice_pos = Some(rand_pos);
      
      log!(context, "trial_summary_data.used_values targetposX_pix={}", targetpos_pix.0);
      log!(context, "trial_summary_data.used_values targetposY_pix={}", targetpos_pix.1);
      log!(context, "trial_summary_data.used_values sample_pos_x_pix={}", sample_pos_pix.0);
      log!(context, "trial_summary_data.used_values sample_pos_y_pix={}", sample_pos_pix.1);
      (Some(sample_shape), Some(correct_idx), sample_pos_pix, targetpos_pix)
    } else {
      let sample_pos_pix = _static.lock().unwrap().sample_pos_pix;
      let targetpos_pix = _static.lock().unwrap().setup_locations_trial(config);
      log!(context, "trial_summary_data.used_values targetposX_pix={}", targetpos_pix.0);
      log!(context, "trial_summary_data.used_values targetposY_pix={}", targetpos_pix.1);
      log!(context, "trial_summary_data.used_values sample_pos_x_pix={}", sample_pos_pix.0);
      log!(context, "trial_summary_data.used_values sample_pos_y_pix={}", sample_pos_pix.1);
      (None, None, sample_pos_pix, targetpos_pix)
    };

    #[allow(unused)]
    let (sample_shape, correct_idx, sample_pos_pix, targetpos_pix) = sample_and_choices;

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
    let _cross: Path = cross_builder.detach();

    // Build shape paths — size controlled by sample_size_deg (half-width in degrees)
    let s = config["sample_size_deg"].as_f64().unwrap() / 2.0;
  
    let square_deg = vec![(-s/2.0, -s/2.0), (s/2.0, -s/2.0), (s/2.0, s/2.0), (-s/2.0, s/2.0)];
    let square_vertices: Vec<(f64, f64)> = square_deg.iter().copied().map(|(x, y)|converter.deg_to_pixel_abs(x, y)).collect();
    let mut square_builder = PathBuilder::new();
    square_builder.move_to((square_vertices[0].0 as f32, square_vertices[0].1 as f32));
    square_builder.line_to((square_vertices[1].0 as f32, square_vertices[1].1 as f32));
    square_builder.line_to((square_vertices[2].0 as f32, square_vertices[2].1 as f32));
    square_builder.line_to((square_vertices[3].0 as f32, square_vertices[3].1 as f32));
    square_builder.close();
    let _square = square_builder.detach();
  
    let triangle_deg = vec![(0.0, -s/2.0), (s/2.0, s/2.0), (-s/2.0, s/2.0)];
    let triangle_vertices: Vec<(f64, f64)> = triangle_deg.iter().copied().map(|(x, y)|converter.deg_to_pixel_abs(x, y)).collect();
    let mut triangle_builder = PathBuilder::new();
    triangle_builder.move_to((triangle_vertices[0].0 as f32, triangle_vertices[0].1 as f32));
    triangle_builder.line_to((triangle_vertices[1].0 as f32, triangle_vertices[1].1 as f32));
    triangle_builder.line_to((triangle_vertices[2].0 as f32, triangle_vertices[2].1 as f32));
    triangle_builder.close();
    let _triangle = triangle_builder.detach();
  
    let (x0, y0) = square_vertices[0];
    let (x2, y2) = square_vertices[2];
    let mut circle_builder = PathBuilder::new();
    circle_builder.add_oval(Rect::from_ltrb(x0 as f32, y0 as f32, x2 as f32, y2 as f32), PathDirection::CW, 0);
    let _circle = circle_builder.detach();

    let accpt_fix_radius_deg = config["accpt_fix_radius_deg"].as_i64().unwrap();
    let accpt_fix_radius_pix = converter.deg_to_pixel_rel(accpt_fix_radius_deg as f64);
    let accpt_gaze_radius_deg = config["accpt_gaze_radius_deg"].as_i64().unwrap();
    let accpt_gaze_radius_pix = converter.deg_to_pixel_rel(accpt_gaze_radius_deg as f64);
    let choice_eccentricity = converter.deg_to_pixel_rel(choice_eccentricity);
    let is_height_locked = config["is_height_locked"].as_bool().unwrap();
    let paint_all_targets = config["paint_all_targets"].as_bool().unwrap();
    let target_color_rgb = get_color(&config["target_color"]);
    let background_color = get_color(&config["background_color"]);
    let penalty_delay = Duration::from_millis(config["penalty_delay"].as_i64().unwrap() as u64);
    
    //# Get various timeouts from the context (user GUI)
    let decision_timeout = Duration::from_millis(get_f64(&config["decision_timeout"]) as u64); //# dividing by 1000x to convert from ms to s
    let fix_duration = Duration::from_millis(get_f64(&config["fix_duration"]) as u64);
    let sample_present_duration = Duration::from_millis(get_f64(&config["sample_present_duration"]) as u64);
    let del_duration = Duration::from_millis(get_f64(&config["del_duration"]) as u64);
    let choice_present_duration = Duration::from_millis(get_f64(&config["choice_present_duration"]) as u64);
    let choice_hold_duration = Duration::from_millis(get_f64(&config["choice_hold_duration"]) as u64);
    let blink_duration = Duration::from_millis(get_f64(&config["blink_duration"]) as u64);
    let start_duration = Duration::from_millis(get_f64(&config["start_duration"]) as u64);

    let reward_per_trial = get_f64(&config["reward_per_trial"]); // return a uniform random number
                                                                    //
    let luminance_targ_per = get_f64_with_step(&config["luminance_targ_per"], config["luminance_targ_step"].as_f64().unwrap());
    log!(context, "trial_summary_data.used_values luminance_targ_per={}", luminance_targ_per);

    let orientation_targ_ran = get_f64_with_step(&config["orientation_targ_ran"], config["orientation_targ_step"].as_f64().unwrap());
    log!(context, "trial_summary_data.used_values orientation_targ_ran={}", orientation_targ_ran);

    let width_targ_deg = get_f64_with_step(&config["width_targ_deg"], config["widthtargdeg_step"].as_f64().unwrap());
    let width_targ_pix = converter.deg_to_pixel_rel(width_targ_deg);
    log!(context, "trial_summary_data.used_values width_targ_pix={}", width_targ_pix);

    let height_targ_pix = if is_height_locked {
      width_targ_pix
    } else {
      let deg = get_f64_with_step(&config["height_targ_deg"], config["heighttargdeg_step"].as_f64().unwrap());
      converter.deg_to_pixel_rel(deg)
    };
    log!(context, "trial_summary_data.used_values height_targ_pix={}", height_targ_pix);
    log!(context, "{}", config.to_string());

    let rates = {
      let lock = _static.lock().unwrap();
      if lock.trial_num == 0 {
        (0.0, 0.0, 0.0)
      } else {
        ((lock.trial_success_count as f64)/(lock.trial_num as f64)*100.0,
         (lock.trial_abort_count as f64)/(lock.trial_num as f64)*100.0,
         (lock.trial_failure_count as f64)/(lock.trial_num as f64)*100.0)
      }
    };
    let (trial_success_rate, trial_abort_rate, trial_failure_rate) = rates;

    let gaze_queue = context.subscribe_to_gaze();
    let last_gaze = Mutex::new((99999, 99999));

    log!(context, "BehavState=ACQUIRE_FIXATION_post-drawing");
    {
      let mut state = self.state.lock().unwrap();
      *state = State::AcquireFixation;
      println!("{:?}", *state);
    }
    wait_for_hold(
      &context,
      point_condition(&gaze_queue, &last_gaze, |point| {
        let valid_gaze = gaze_valid(point.0, point.1, monitorsubj_w_pix, monitorsubj_h_pix);
        distance(valid_gaze, center) < accpt_fix_radius_pix
      }),
      start_duration,
      None,
      false,
    )
    .await;

    TaskResult { success: true, cancelled: false }
  }

  fn render(&self, _canvas: &Canvas, _window: Window) {
    todo!()
  }
}

