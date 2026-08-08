use std::sync::{Arc, OnceLock};

use ndarray;
use itertools::{iproduct};
use std::time::{Duration};
use parking_lot;

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
  Canvas, Color4f, Font, FontMgr, Paint, PaintStyle, Path, PathBuilder, Rect, Shader, TileMode, PathDirection, Point,
  Color
};

use skia_safe::gradient::{
  Colors as GradientColors, Gradient, Interpolation, shaders as gradient_shaders,
};

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
enum State {
  Null,
  AcquireFixation,
  Fixate,
  SamplePresentation,
  Delay,
  ChoicePresentation,
  AcquireChoice,
  HoldChoice,
  Success,
  Abort,
}

pub struct Vcp2AfcTask {
  inner: parking_lot::Mutex<Inner>,
}

#[allow(unused)]
struct Inner {
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
  loc_correct_idx: i32,
  choice_shapes: Vec<Option<&'static str>>,
  trial_num: i32,
  trial_success_count: i32,
  trial_abort_count: i32,
  trial_failure_count: i32,
  state: State,
  gaze_failure_store: Vec<((i32, i32), Color4f)>,
  gaze_success_store: Vec<((i32, i32), Color4f)>,
  loc_left_pos: (i32, i32),
  loc_right_pos: (i32, i32),
  reward_total_released_ms: i32,
  background_color: Color4f,
  paint_all_targets: bool,
  target_color_rgb: Color4f,
  cross: Path,
  square: Path,
  triangle: Path,
  circle: Path,
  task_group: String,
  sample_shape: String,
  gaussian: Option<Shader>,
  orientation_targ_ran: f64,
  width_targ_pix: f64,
  height_targ_pix: f64,
  rand_pos: Vec<(i32, i32)>,
  accpt_gaze_radius_pix: f64,
  targetpos_pix: (i32, i32),
  accpt_fix_radius_pix: f64,
  gaze: (i32, i32),
}

#[allow(unused)]
static PHOTODIODE_BLINKING_SQUARE: Color4f = Color4f::new(1.0, 1.0, 1.0, 1.0);
#[allow(unused)]
static PHOTODIODE_STATIC_SQUARE: Color4f = Color4f::new(0.0, 0.0, 0.0, 1.0);

static SHAPES: &[&str] = &["square", "circle", "triangle"];

fn reward_pulse(duration_ms: u64) -> AnalogResponse {
  AnalogResponse {
    data: vec![5.0, 0.0], // 5 = HIGH, 0 = LOW voltages
    spans: vec![Span {
      begin: 0,
      end: 2,
      name: "Reward".to_string(),
      ..Default::default()
    }],
    sample_intervals: vec![1_000_000 * duration_ms], // ns
    ..Default::default()
  }
}

impl Inner {
  fn build_loc_rand_pos(&mut self, config: &serde_json::Value, converter: &Converter) {
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

  fn setup_locations_trial(&mut self, config: &serde_json::Value, converter: &Converter) -> (i32, i32) {
    let center = converter.center;

    self.sample_pos_pix = self.loc_rand_pos[usize::try_from(self.loc_rand_pos_i).unwrap()];
    self.loc_rand_pos_i += 1;

    let ecc_pix = converter.deg_to_pixel_rel(config["choice_eccentricity"].as_f64().unwrap()) as i32;
    self.loc_left_pos  = (center.0 - ecc_pix, center.1);
    self.loc_right_pos = (center.0 + ecc_pix, center.1);

    if self.sample_pos_pix.0 < center.0 {
        self.loc_correct_idx = 0   // left
    } else if self.sample_pos_pix.0 > center.0 {
        self.loc_correct_idx = 1   // right
    } else {
        self.loc_correct_idx = rand::random_range(0..=1);
    }

    if self.loc_correct_idx == 0 {
      self.loc_left_pos
    } else {
      self.loc_right_pos
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
      self.build_loc_rand_pos(config, &converter);
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
    let success_sound =
      StaticSoundData::from_file(r"C:\Thalamus-Extensions\seokhee\success_clip.wav").unwrap();
    let abort_sound =
      StaticSoundData::from_file(r"C:\Thalamus-Extensions\seokhee\failure_clip.wav").unwrap();
    let failure_sound =
      StaticSoundData::from_file(r"C:\Thalamus-Extensions\seokhee\timeout_failure.wav").unwrap();

    Vcp2AfcTask {
      inner: parking_lot::Mutex::new(Inner {
        success_sound,
        abort_sound,
        failure_sound,
        loc_rand_pos: vec![],
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
        loc_correct_idx: 0,
        choice_shapes: vec![],
        trial_num: 0,
        trial_success_count: 0,
        trial_abort_count: 0,
        trial_failure_count: 0,
        state: State::Null,
        gaze_failure_store: vec![],
        gaze_success_store: vec![],
        loc_left_pos: (0, 0),
        loc_right_pos: (0, 0),
        reward_total_released_ms: 0,
        background_color: Color4f::new(0.0, 0.0, 0.0, 0.0),
        paint_all_targets: false,
        target_color_rgb: Color4f::new(0.0, 0.0, 0.0, 0.0),
        task_group: "".to_string(),
        circle: Path::new(),
        square: Path::new(),
        cross: Path::new(),
        triangle: Path::new(),
        sample_shape: "".to_string(),
        gaussian: None,
        orientation_targ_ran: 0.0,
        width_targ_pix: 0.0,
        height_targ_pix: 0.0,
        rand_pos: vec![],
        accpt_gaze_radius_pix: 0.0,
        targetpos_pix: (0, 0),
        accpt_fix_radius_pix: 0.0,
        gaze: (0, 0),
      }),
    }
  }

  async fn set_state(&self, context: &TaskContext, text: &str, state: State) {
    context.log(text).await;
    self.inner.lock().state = state;
    println!("{:?}", state);
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

fn point_condition<'a, 'b>(
  point_queue: &'a PointSubscription,
  last_point_mutex: impl Fn() -> parking_lot::MappedMutexGuard<'b, (i32, i32)> + 'a,
  within: impl Fn((i32, i32)) -> bool + 'a,
) -> impl Fn() -> bool + 'a {
  move || {
    let mut satisfied = false;
    let mut last_point = last_point_mutex();
    for point in point_queue.drain() {
      *last_point = point;
      if within(point) {
        satisfied = true;
      }
    }
    satisfied || within(*last_point)
  }
}

fn gaussian_gradient_shader(
  background_color: Color4f,
  radius: f32,
  deviations: f32,
  brightness_in: f32,
  luminance_percent: f32,
) -> Shader {
  const RESOLUTION: usize = 1000;
  let bg_r = background_color.r * 255.0;
  let bg_g = background_color.g * 255.0;
  let bg_b = background_color.b * 255.0;
  let brightness = (brightness_in - bg_r) * luminance_percent / 100.0 + bg_r;

  let mut colors: Vec<Color4f> = (0..RESOLUTION)
    .map(|i| {
      let t = deviations * i as f32 / RESOLUTION as f32;
      let level = if bg_r == 0.0 && bg_g == 0.0 && bg_b == 0.0 {
        brightness * (-(t * t) / 2.0).exp()
      } else {
        bg_r + (brightness - bg_r) * (-(t * t) / 2.0).exp()
      };
      let level = level.trunc() / 255.0;
      Color4f::new(level, level, level, 1.0)
    })
    .collect();
  colors.push(Color4f::new(
    background_color.r,
    background_color.g,
    background_color.b,
    0.0,
  ));

  let mut positions: Vec<f32> = (0..RESOLUTION)
    .map(|i| i as f32 / RESOLUTION as f32)
    .collect();
  positions.push(1.0);

  let gradient_colors = GradientColors::new(&colors, Some(&positions), TileMode::Clamp, None);
  let gradient = Gradient::new(gradient_colors, Interpolation::default());
  gradient_shaders::radial_gradient(((0.0, 0.0), radius), &gradient, None)
    .expect("failed to build gaussian gradient shader")
}

static RENDER_FONT_DATA: &[u8] = include_bytes!("../../assets/DejaVuSans.ttf");

fn render_font() -> &'static Font {
  static FONT: OnceLock<Font> = OnceLock::new();
  FONT.get_or_init(|| {
    let typeface = FontMgr::new()
      .new_from_data(RENDER_FONT_DATA, None)
      .expect("bundled DejaVuSans.ttf should parse as a valid font");
    Font::from_typeface(typeface, 18.0)
  })
}

fn draw_text(canvas: &Canvas, text: &str, x: f32, y: f32, background_color_qt: Color4f) {
  let is_black =
    background_color_qt.r == 0.0 && background_color_qt.g == 0.0 && background_color_qt.b == 0.0;
  let (foreground, background) = if is_black {
    (
      Color4f::new(1.0, 1.0, 1.0, 1.0),
      Color4f::new(0.0, 0.0, 0.0, 1.0),
    )
  } else {
    (
      Color4f::new(0.0, 0.0, 0.0, 1.0),
      Color4f::new(1.0, 1.0, 1.0, 1.0),
    )
  };

  canvas.draw_rect(
    Rect::from_xywh(x, y, 320.0, 26.0),
    &Paint::new(background, None),
  );

  let mut foreground_paint = Paint::new(foreground, None);
  foreground_paint.set_anti_alias(true);
  canvas.draw_str(text, (x + 4.0, y + 19.0), render_font(), &foreground_paint);
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

    self.inner.lock().sync_config(config);

    let sample_and_choices = if task_group == "Shapes" {
      let (sample_shape, correct_idx) = self.inner.lock().setup_sample_and_choices(num_choices);
      let sample_pos_pix = self.inner.lock().sample_pos_pix;
      let targetpos_pix   = rand_pos[usize::try_from(correct_idx).unwrap()];
      //_static.
      //choice_pos = Some(rand_pos);
      
      context.log(&format!("trial_summary_data.used_values targetposX_pix={}", targetpos_pix.0)).await;
      context.log(&format!("trial_summary_data.used_values targetposY_pix={}", targetpos_pix.1)).await;
      context.log(&format!("trial_summary_data.used_values sample_pos_x_pix={}", sample_pos_pix.0)).await;
      context.log(&format!("trial_summary_data.used_values sample_pos_y_pix={}", sample_pos_pix.1)).await;
      (sample_shape, correct_idx, sample_pos_pix, targetpos_pix)
    } else {
      let sample_pos_pix = self.inner.lock().sample_pos_pix;
      let targetpos_pix = self.inner.lock().setup_locations_trial(config, &converter);
      context.log(&format!("trial_summary_data.used_values targetposX_pix={}", targetpos_pix.0)).await;
      context.log(&format!("trial_summary_data.used_values targetposY_pix={}", targetpos_pix.1)).await;
      context.log(&format!("trial_summary_data.used_values sample_pos_x_pix={}", sample_pos_pix.0)).await;
      context.log(&format!("trial_summary_data.used_values sample_pos_y_pix={}", sample_pos_pix.1)).await;
      ("", self.inner.lock().loc_correct_idx, sample_pos_pix, targetpos_pix)
    };

    #[allow(unused)]
    let (sample_shape, choice_idx, sample_pos_pix, targetpos_pix) = sample_and_choices;

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
  
    let square_deg = vec![(-s/2.0, -s/2.0), (s/2.0, -s/2.0), (s/2.0, s/2.0), (-s/2.0, s/2.0)];
    let square_vertices: Vec<(f64, f64)> = square_deg.iter().copied().map(|(x, y)|converter.deg_to_pixel_abs(x, y)).collect();
    let mut square_builder = PathBuilder::new();
    square_builder.move_to((square_vertices[0].0 as f32, square_vertices[0].1 as f32));
    square_builder.line_to((square_vertices[1].0 as f32, square_vertices[1].1 as f32));
    square_builder.line_to((square_vertices[2].0 as f32, square_vertices[2].1 as f32));
    square_builder.line_to((square_vertices[3].0 as f32, square_vertices[3].1 as f32));
    square_builder.close();
    let square = square_builder.detach();
  
    let triangle_deg = vec![(0.0, -s/2.0), (s/2.0, s/2.0), (-s/2.0, s/2.0)];
    let triangle_vertices: Vec<(f64, f64)> = triangle_deg.iter().copied().map(|(x, y)|converter.deg_to_pixel_abs(x, y)).collect();
    let mut triangle_builder = PathBuilder::new();
    triangle_builder.move_to((triangle_vertices[0].0 as f32, triangle_vertices[0].1 as f32));
    triangle_builder.line_to((triangle_vertices[1].0 as f32, triangle_vertices[1].1 as f32));
    triangle_builder.line_to((triangle_vertices[2].0 as f32, triangle_vertices[2].1 as f32));
    triangle_builder.close();
    let triangle = triangle_builder.detach();
  
    let (x0, y0) = square_vertices[0];
    let (x2, y2) = square_vertices[2];
    let mut circle_builder = PathBuilder::new();
    circle_builder.add_oval(Rect::from_ltrb(x0 as f32, y0 as f32, x2 as f32, y2 as f32), PathDirection::CW, 0);
    let circle = circle_builder.detach();

    let accpt_fix_radius_deg = config["accpt_fix_radius_deg"].as_i64().unwrap();
    let accpt_fix_radius_pix = converter.deg_to_pixel_rel(accpt_fix_radius_deg as f64);
    let accpt_gaze_radius_deg = config["accpt_gaze_radius_deg"].as_i64().unwrap();
    let accpt_gaze_radius_pix = converter.deg_to_pixel_rel(accpt_gaze_radius_deg as f64);
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

    let reward_per_trial = get_f64(&config["reward_per_trial"]); // return a uniform random number

    {
      let mut lock = self.inner.lock();
      lock.background_color = background_color;
      lock.paint_all_targets = paint_all_targets;
      lock.target_color_rgb = target_color_rgb;
      lock.cross = cross;
      lock.square = square;
      lock.triangle = triangle;
      lock.circle = circle;
      lock.task_group = task_group.to_string();
      lock.sample_shape = sample_shape.to_string();
      lock.rand_pos = rand_pos.clone();
      lock.accpt_gaze_radius_pix = accpt_gaze_radius_pix;
      lock.targetpos_pix = targetpos_pix;
      lock.accpt_fix_radius_pix = accpt_fix_radius_pix;
    }

    let luminance_targ_per = get_f64_with_step(&config["luminance_targ_per"], config["luminance_targ_step"].as_f64().unwrap());
    context.log(&format!("trial_summary_data.used_values luminance_targ_per={}", luminance_targ_per)).await;

    let orientation_targ_ran = get_f64_with_step(&config["orientation_targ_ran"], config["orientation_targ_step"].as_f64().unwrap());
    context.log(&format!("trial_summary_data.used_values orientation_targ_ran={}", orientation_targ_ran)).await;

    let width_targ_deg = get_f64_with_step(&config["width_targ_deg"], config["widthtargdeg_step"].as_f64().unwrap());
    let width_targ_pix = converter.deg_to_pixel_rel(width_targ_deg);
    context.log(&format!("trial_summary_data.used_values width_targ_pix={}", width_targ_pix)).await;

    let height_targ_pix = if is_height_locked {
      width_targ_pix
    } else {
      let deg = get_f64_with_step(&config["height_targ_deg"], config["heighttargdeg_step"].as_f64().unwrap());
      converter.deg_to_pixel_rel(deg)
    };
    context.log(&format!("trial_summary_data.used_values height_targ_pix={}", height_targ_pix)).await;

    let gaussian = gaussian_gradient_shader(background_color, (width_targ_pix/2.0) as f32,
                   3.0, 255.0, luminance_targ_per as f32);

    {
      let mut lock = self.inner.lock();
      lock.gaussian = Some(gaussian);
      lock.orientation_targ_ran = orientation_targ_ran;
    }

    context.log(&format!("{}", config.to_string())).await;

    let rates = {
      let lock = self.inner.lock();
      if lock.trial_num == 0 {
        (0.0, 0.0, 0.0)
      } else {
        ((lock.trial_success_count as f64)/(lock.trial_num as f64)*100.0,
         (lock.trial_abort_count as f64)/(lock.trial_num as f64)*100.0,
         (lock.trial_failure_count as f64)/(lock.trial_num as f64)*100.0)
      }
    };
    let (trial_success_rate, _trial_abort_rate, trial_failure_rate) = rates;

    let gaze_queue = context.subscribe_to_gaze();
    let get_gaze = || {
      parking_lot::MutexGuard::map(self.inner.lock(), |v| {
        &mut v.gaze
      })
    };

    self.set_state(&context, "BehavState=ACQUIRE_FIXATION_post-drawing", State::AcquireFixation).await;
    wait_for(
      &context,
      point_condition(&gaze_queue, get_gaze, |point| {
        let valid_gaze = gaze_valid(point.0, point.1, monitorsubj_w_pix, monitorsubj_h_pix);
        distance(valid_gaze, center) < accpt_fix_radius_pix
      }),
      None,
    )
    .await;

    self.set_state(&context, "BehavState=FIXATE_post-drawing", State::Fixate).await;
    wait_for_hold(
        &context,
        point_condition(&gaze_queue, get_gaze, |point| {
          let valid_gaze = gaze_valid(point.0, point.1, monitorsubj_w_pix, monitorsubj_h_pix);
          distance(valid_gaze, center) < accpt_fix_radius_pix
        }),
        fix_duration,
        Some(blink_duration),
        false,
      )
      .await;

    {
      let gaze = self.inner.lock().gaze;
      let temp_gaze = gaze_valid(gaze.0, gaze.1, monitorsubj_w_pix, monitorsubj_h_pix);
      context.log(&format!("Gaze[X,Y]_pix-abs_after-FIXATE=({}, {})", temp_gaze.0, temp_gaze.1)).await;
      let temp_gaze_deg = converter.relpix_to_absdeg(temp_gaze.0 as f64, temp_gaze.1 as f64);
      context.log(&format!("Gaze[X,Y]_deg-abs_after-FIXATE=({}, {})", temp_gaze_deg.0, temp_gaze_deg.1)).await;
    }

    self.set_state(&context, "BehavState=SAMPLE_PRESENTATION_post-drawing_PHOTODIODE-SQUARE", State::SamplePresentation).await;
    let present_success = wait_for_hold(
        &context,
        point_condition(&gaze_queue, get_gaze, |point| {
          let valid_gaze = gaze_valid(point.0, point.1, monitorsubj_w_pix, monitorsubj_h_pix);
          distance(valid_gaze, center) < accpt_fix_radius_pix
        }),
        sample_present_duration,
        Some(blink_duration),
        false,
      )
      .await;

    let trial_num = {
      let mut inner = self.inner.lock();
      inner.trial_num += 1;
      inner.trial_num
    };
    println!("Started trial # {trial_num}");
    context.log(&format!("StartedTRIAL_NUM={trial_num}")).await; // saving any variables / data from code

    if !present_success {
      self.set_state(&context, "TrialResult=ABORT", State::Abort).await;

      let (new_trial_abort_rate, trial_success_count) = {
        let mut lock = self.inner.lock();
        context.play_sound(lock.abort_sound.clone());
        lock.trial_abort_count += 1;
        let new_trial_abort_rate = (lock.trial_abort_count as f64)/(trial_num as f64)*100.0;
        let trial_success_count = lock.trial_success_count;
        (new_trial_abort_rate, trial_success_count)
      };
      context.log(&format!("TRIAL_NUM={trial_num}, SUCCESS_COUNT={trial_success_count} \
                            SUCCESS_RATE={trial_success_rate}, ABORT_RATE={new_trial_abort_rate}, FAILURE_RATE={trial_failure_rate}")).await;
      
      tokio::time::sleep(penalty_delay).await;
      return TaskResult { success: false, cancelled: false };
    }
    
    self.set_state(&context, "BehavState=DELAY", State::Delay).await;
    wait_for_hold(
        &context,
        point_condition(&gaze_queue, get_gaze, |point| {
          let valid_gaze = gaze_valid(point.0, point.1, monitorsubj_w_pix, monitorsubj_h_pix);
          distance(valid_gaze, center) < accpt_fix_radius_pix
        }),
        del_duration,
        Some(blink_duration),
        false,
      )
      .await;

    self.set_state(&context, "BehavState=CHOICE_PRESENTATION_post-drawing_PHOTODIODE-SQUARE", State::ChoicePresentation).await;
    let choice_success = wait_for_hold(
        &context,
        point_condition(&gaze_queue, get_gaze, |point| {
          let valid_gaze = gaze_valid(point.0, point.1, monitorsubj_w_pix, monitorsubj_h_pix);
          distance(valid_gaze, center) < accpt_fix_radius_pix
        }),
        choice_present_duration,
        Some(blink_duration),
        false,
      )
      .await;

    if !choice_success {
      self.set_state(&context, "TrialResult=ABORT", State::Abort).await;
      let (new_trial_abort_rate, trial_success_count) = {
        let mut lock = self.inner.lock();
        context.play_sound(lock.abort_sound.clone());
        lock.trial_abort_count += 1;
        let new_trial_abort_rate = (lock.trial_abort_count as f64)/(trial_num as f64)*100.0;
        let trial_success_count = lock.trial_success_count;
        (new_trial_abort_rate, trial_success_count)
      };
      context.log(&format!("TRIAL_NUM={trial_num}, SUCCESS_COUNT={trial_success_count} \
                            SUCCESS_RATE={trial_success_rate}, ABORT_RATE={new_trial_abort_rate}, FAILURE_RATE={trial_failure_rate}")).await;
      
      tokio::time::sleep(penalty_delay).await;
      return TaskResult { success: false, cancelled: false };
    }

    let wrong_positions = if task_group == "Shapes" {
      rand_pos.iter().enumerate().filter_map(|(i, p)| {
        if i as i32 != choice_idx { Some(*p) } else { None }
      }).collect()
    } else if task_group == "Locations" {
      let lock = self.inner.lock();
      if choice_idx == 0 { vec![lock.loc_right_pos] } else { vec![lock.loc_left_pos] }
    } else {
      Vec::<(i32, i32)>::new()
    };

    self.set_state(&context, "BehavState=ACQUIRE_CHOICE_start", State::AcquireChoice).await;
    let acquire_success = wait_for(
        &context,
        point_condition(&gaze_queue, get_gaze, |point| {
          let valid_gaze = gaze_valid(point.0, point.1, monitorsubj_w_pix, monitorsubj_h_pix);
          for pos in &wrong_positions {
            if distance(valid_gaze, *pos) < accpt_gaze_radius_pix {
              return true;
            }
          }
          if distance(valid_gaze, targetpos_pix) < accpt_gaze_radius_pix {
            return true;
          }
          false
        }),
        Some(decision_timeout)
      )
      .await;
    let correct_selection = {
      let point = self.inner.lock().gaze;
      let valid_gaze = gaze_valid(point.0, point.1, monitorsubj_w_pix, monitorsubj_h_pix);
      distance(valid_gaze, targetpos_pix) < accpt_gaze_radius_pix
    };

    let acquire_gaze = self.inner.lock().gaze;
    {
      let temp_gaze = gaze_valid(acquire_gaze.0, acquire_gaze.1, monitorsubj_w_pix, monitorsubj_h_pix);
      context.log(&format!("Gaze[X,Y]_pix-abs_after-acquiring-target={temp_gaze:?}")).await;
      let temp_gaze_deg = converter.relpix_to_absdeg(temp_gaze.0 as f64, temp_gaze.1 as f64);
      context.log(&format!("Gaze[X,Y]_deg-abs_after-acquiring-target={temp_gaze_deg:?}")).await;
    }

    if !(acquire_success && correct_selection) {
      self.set_state(&context, "TrialResult=FAILURE", State::Abort).await;
      let (new_trial_failure_rate, trial_success_count) = {
        let mut lock = self.inner.lock();
        lock.gaze_failure_store.push(
          (gaze_valid(acquire_gaze.0, acquire_gaze.1, monitorsubj_w_pix, monitorsubj_h_pix), Color4f::new(255.0/255.0, 69.0/255.0, 0.0/255.0, 128.0/255.0)));
        context.play_sound(lock.failure_sound.clone());
        lock.trial_failure_count += 1;
        let new_trial_failure_rate = (lock.trial_failure_count as f64)/(trial_num as f64)*100.0;
        let trial_success_count = lock.trial_success_count;
        (new_trial_failure_rate, trial_success_count)
      };
      context.log(&format!("TRIAL_NUM={trial_num}, SUCCESS_COUNT={trial_success_count} \
                            SUCCESS_RATE={trial_success_rate}, ABORT_RATE={new_trial_failure_rate}, FAILURE_RATE={trial_failure_rate}")).await;
      
      tokio::time::sleep(penalty_delay).await;
      return TaskResult { success: false, cancelled: false };
    }

    self.set_state(&context, "BehavState=HOLD_CHOICE_start", State::HoldChoice).await;
    let choice_success = wait_for_hold(
        &context,
        point_condition(&gaze_queue, get_gaze, |point| {
          let valid_gaze = gaze_valid(point.0, point.1, monitorsubj_w_pix, monitorsubj_h_pix);
          distance(valid_gaze, center) < accpt_gaze_radius_pix
        }),
        choice_hold_duration,
        Some(blink_duration),
        false,
      )
      .await;

    if !choice_success {
      self.set_state(&context, "TrialResult=FAILURE", State::Abort).await;

      let (new_trial_failure_rate, trial_success_count) = {
        let mut lock = self.inner.lock();
        let g = self.inner.lock().gaze;
        lock.gaze_failure_store.push(
          (gaze_valid(g.0, g.1, monitorsubj_w_pix, monitorsubj_h_pix), Color4f::new(255.0/255.0, 69.0/255.0, 0.0/255.0, 128.0/255.0)));
        context.play_sound(lock.failure_sound.clone());
        lock.trial_failure_count += 1;
        let new_trial_failure_rate = (lock.trial_failure_count as f64)/(trial_num as f64)*100.0;
        let trial_success_count = lock.trial_success_count;
        (new_trial_failure_rate, trial_success_count)
      };
      context.log(&format!("TRIAL_NUM={trial_num}, SUCCESS_COUNT={trial_success_count} \
                            SUCCESS_RATE={trial_success_rate}, ABORT_RATE={new_trial_failure_rate}, FAILURE_RATE={trial_failure_rate}")).await;
      
      tokio::time::sleep(penalty_delay).await;
      return TaskResult { success: false, cancelled: false };
    }

    let gaze = self.inner.lock().gaze;
    let temp_gaze = gaze_valid(gaze.0, gaze.1, monitorsubj_w_pix, monitorsubj_h_pix);
    context.log(&format!("Gaze[X,Y]_pix-abs_after-holding-target={temp_gaze:?}")).await;
    let temp_gaze_deg = converter.relpix_to_absdeg(temp_gaze.0 as f64, temp_gaze.1 as f64);
    context.log(&format!("Gaze[X,Y]_deg-abs_after-holding-target={temp_gaze_deg:?}")).await;

    self.inner.lock().gaze_success_store.push((temp_gaze, Color4f::new(255.0/255.0, 69.0/255.0, 0.0/255.0, 128.0/255.0)));
    self.set_state(&context, "TrialResult=SUCCESS", State::Success).await;
    context.play_sound(self.inner.lock().success_sound.clone());
    
    tokio::time::sleep(Duration::from_secs(1)).await;
    self.inner.lock().reward_total_released_ms += reward_per_trial as i32;
    context.log(&format!("starting_reward_release_of = {} ms, total_released = {} ms", self.inner.lock().reward_total_released_ms, reward_per_trial)).await;

    context
      .inject_analog("reward_in", reward_pulse(reward_per_trial as u64))
      .await;

    self.inner.lock().trial_success_count += 1;
    let trial_success_count = self.inner.lock().trial_success_count;
    let new_trial_success_rate = (self.inner.lock().trial_success_count as f64)/(trial_num as f64)*100.0;
    context.log(&format!("TRIAL_NUM={trial_num}, SUCCESS_COUNT={trial_success_count} \
                          SUCCESS_RATE={new_trial_success_rate}, ABORT_RATE={trial_failure_rate}, FAILURE_RATE={trial_failure_rate}")).await;

    TaskResult { success: true, cancelled: false }
  }

  fn render(&self, canvas: &Canvas, window: Window) {
    let (
      state,
      background_color,
      cross,
      square,
      circle,
      triangle,
      task_group,
      sample_shape,
      sample_pos_pix,
      gaussian,
      orientation_targ_ran,
      width_targ_pix,
      height_targ_pix,
      rand_pos,
      choice_shapes,
      loc_left_pos,
      loc_right_pos,
      accpt_gaze_radius_pix,
      targetpos_pix,
      accpt_fix_radius_pix,
      gaze,
      trial_num,
      trial_success_count,
      trial_abort_count,
      trial_failure_count,
      reward_total_released_ms,
    ) = {
      let lock = self.inner.lock();
      (lock.state,
       lock.background_color,
       lock.cross.clone(),
       lock.square.clone(),
       lock.circle.clone(),
       lock.triangle.clone(),
       lock.task_group.clone(),
       lock.sample_shape.clone(),
       lock.sample_pos_pix,
       lock.gaussian.clone(),
       lock.orientation_targ_ran,
       lock.width_targ_pix,
       lock.height_targ_pix,
       lock.rand_pos.clone(),
       lock.choice_shapes.clone(),
       lock.loc_left_pos,
       lock.loc_right_pos,
       lock.accpt_gaze_radius_pix,
       lock.targetpos_pix,
       lock.accpt_fix_radius_pix,
       lock.gaze,
       lock.trial_num,
       lock.trial_success_count,
       lock.trial_abort_count,
       lock.trial_failure_count,
       lock.reward_total_released_ms,
       )
    };
    canvas.draw_rect(Rect::from_xywh(0.0, 0.0, 4000.0, 4000.0), &Paint::new(background_color, None));
    let canvas_size = canvas.base_layer_size();
    let canvas_center = (canvas_size.width/2, canvas_size.height/2);
    canvas.draw_rect(Rect::from_xywh(0.0, 0.0, 4000.0, 4000.0), &Paint::new(background_color, None));

    let mut current_photodiode_static_square = PHOTODIODE_STATIC_SQUARE;

    let gaussian_ref = &gaussian;
    let draw_gaussian = move |pos: (i32, i32)| {
      canvas.save();

      canvas.translate(pos);
      canvas.rotate(orientation_targ_ran as f32, None);
      canvas.scale((1.0, (height_targ_pix / width_targ_pix) as f32));

      let mut paint = Paint::default();
      paint.set_shader(gaussian_ref.clone());
      let (width, height) = (canvas_size.width as f32, canvas_size.height as f32);
      canvas.draw_rect(
        Rect::from_xywh(-width / 2.0, -height / 2.0, width, height),
        &paint,
      );

      canvas.restore();
    };

    match state {
      State::Null => {}
      State::AcquireFixation | State::Fixate => {
        let mut pen = Paint::new(Color4f::new(0.5, 0.0, 0.5, 1.0), None);
        pen.set_style(PaintStyle::Stroke);
        pen.set_stroke_width(2.0);
        pen.set_anti_alias(true);
        canvas.draw_path(&cross, &pen);
      },
      State::SamplePresentation => {
        let mut pen = Paint::new(Color4f::new(0.5, 0.0, 0.5, 1.0), None);
        pen.set_style(PaintStyle::Stroke);
        pen.set_stroke_width(2.0);
        pen.set_anti_alias(true);
        canvas.draw_path(&cross, &pen);
        if task_group == "Shapes" {
          pen.set_color4f(Color4f::new(1.0, 1.0, 1.0, 1.0), None);
          canvas.save();
          canvas.translate(sample_pos_pix);
          match sample_shape.as_str() {
            "triangle" => {
              canvas.draw_path(&triangle, &pen);
            }
            "circle" => {
              canvas.draw_path(&circle, &pen);
            }
            "squaure" => {
              canvas.draw_path(&square, &pen);
            }
            _ => {}
          };
          canvas.restore();
        } else {
          if gaussian.is_some() {
            draw_gaussian(sample_pos_pix);
          }
        }
        current_photodiode_static_square = PHOTODIODE_BLINKING_SQUARE;
      },
      State::Delay => {
        let mut pen = Paint::new(Color4f::new(0.5, 0.0, 0.5, 1.0), None);
        pen.set_style(PaintStyle::Stroke);
        pen.set_stroke_width(2.0);
        pen.set_anti_alias(true);
        canvas.draw_path(&cross, &pen);
      },
      State::ChoicePresentation => {
        let mut pen = Paint::new(Color4f::new(0.5, 0.0, 0.5, 1.0), None);
        pen.set_style(PaintStyle::Stroke);
        pen.set_stroke_width(2.0);
        pen.set_anti_alias(true);
        canvas.draw_path(&cross, &pen);
        if task_group == "Shapes" {
          pen.set_color4f(Color4f::new(1.0, 1.0, 1.0, 1.0), None);

          for (shape, pos) in choice_shapes.iter().zip(rand_pos) {
            canvas.save();
            canvas.translate((pos.0-canvas_center.0, pos.1-canvas_center.1));
            match shape {
              Some("triangle") => {
                canvas.draw_path(&triangle, &pen);
              }
              Some("circle") => {
                canvas.draw_path(&circle, &pen);
              }
              Some("squaure") => {
                canvas.draw_path(&square, &pen);
              }
              _ => {}
            };
            canvas.restore();
          }
        } else {
          draw_gaussian(loc_left_pos);
          draw_gaussian(loc_right_pos);
        }
      },
      State::HoldChoice | State::AcquireChoice => {
        let mut pen = Paint::new(Color4f::new(1.0, 1.0, 1.0, 1.0), None);
        pen.set_style(PaintStyle::Stroke);
        pen.set_stroke_width(2.0);
        pen.set_anti_alias(true);
        if task_group == "Shapes" {
          for (shape, pos) in choice_shapes.iter().zip(rand_pos) {
            canvas.save();
            canvas.translate((pos.0-canvas_center.0, pos.1-canvas_center.1));
            match shape {
              Some("triangle") => {
                canvas.draw_path(&triangle, &pen);
              }
              Some("circle") => {
                canvas.draw_path(&circle, &pen);
              }
              Some("squaure") => {
                canvas.draw_path(&square, &pen);
              }
              _ => {}
            };
            canvas.restore();
          }
        } else {
          draw_gaussian(loc_left_pos);
          draw_gaussian(loc_right_pos);
        }
      },
      _ => {}
    }

    canvas.draw_rect(
      Rect::from_xywh(
        canvas_size.width as f32 - 50.0,
        canvas_size.height as f32 - 50.0,
        500.0,
        500.0,
      ),
      &Paint::new(current_photodiode_static_square, None),
    );

    if window == Window::Operator {
      let shading_paint = Paint::new(Color4f::new(1.0, 1.0, 1.0, 80.0 / 255.0), None);
      canvas.draw_rect(Rect::from_xywh(0.0, 0.0, 465.0, 230.0), &Paint::new(Color4f::new(1.0, 1.0, 1.0, 1.0), None));

      if task_group == "Locations" && loc_left_pos != (0, 0) {
        canvas.draw_circle(
          (loc_left_pos.0 as f32, loc_left_pos.1 as f32),
          accpt_gaze_radius_pix as f32,
          &shading_paint,
        );
        canvas.draw_circle(
          (loc_right_pos.0 as f32, loc_right_pos.1 as f32),
          accpt_gaze_radius_pix as f32,
          &shading_paint,
        );
        canvas.draw_circle(
          (targetpos_pix.0 as f32, targetpos_pix.1 as f32),
          accpt_gaze_radius_pix as f32,
          &Paint::new(Color4f::new(0.0, 1.0, 0.0, 100.0 / 255.0), None),
        );
      } else {
        canvas.draw_circle(
          (targetpos_pix.0 as f32, targetpos_pix.1 as f32),
          accpt_gaze_radius_pix as f32,
          &Paint::new(Color4f::new(1.0, 1.0, 1.0, 128.0 / 255.0), None),
        );
      }
      canvas.draw_circle(
        (canvas_center.0 as f32, canvas_center.1 as f32),
        accpt_fix_radius_pix as f32,
        &Paint::new(Color4f::new(1.0, 1.0, 1.0, 128.0 / 255.0), None),
      );

      canvas.draw_circle(
        gaze,
        12.0,
        &Paint::new(Color4f::new(138.0 / 255.0, 43.0 / 255.0, 226.0 / 255.0, 1.0), None),
      );

      draw_text(canvas, &format!("{state:?}"), 0.0, 10.0, background_color); // Draw the text message
      draw_text(canvas, &format!("Trial_num = {trial_num}"), 0.0, 40.0, background_color); // Draw the text message
      draw_text(canvas, &format!("Result = {trial_success_count} / {trial_num}"), 0.0, 70.0, background_color); // Draw the text message
      draw_text(canvas, &format!("Abort_rate = {trial_abort_count} / {trial_num}"), 0.0, 100.0, background_color); // Draw the text message
      draw_text(canvas, &format!("Failure_rate = {trial_failure_count} / {trial_num}"), 0.0, 130.0, background_color); // Draw the text message
      draw_text(canvas, &format!("Total_reward = {reward_total_released_ms} ms"), 0.0, 160.0, background_color); // Draw the text message
      let (temp_gaze_x, temp_gaze_y) = gaze_valid(gaze.0, gaze.1, canvas_size.width, canvas_size.height);
      let drawn_text = &format!("({temp_gaze_x}, {temp_gaze_y})");
      draw_text(canvas, drawn_text, temp_gaze_x as f32, temp_gaze_y as f32, background_color); // Draw the text message
      draw_text(canvas, "Gaze (pix): x = {temp_gaze_x}, y = {temp_gaze_y}", 0.0, 190.0, background_color);
    }
  }
}
//  def draw_gaze(painter, gaze_qpoint, color_rgba):
//    path = QPainterPath()
//    gaze_f = QPointF(gaze_qpoint)
//    path.addEllipse(gaze_f, 12, 12)
//    painter.fillPath(path, color_rgba) 


  //Null,
  //SamplePresentation,
  //Delay,
  //ChoicePresentation,
  //AcquireChoice,
  //HoldChoice,
  //HoldTarget,
  //Success,
  //FailureSaccade,
  //FailureHold,
  //Abort,
