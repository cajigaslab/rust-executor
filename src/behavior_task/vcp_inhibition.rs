use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

use async_trait::async_trait;
use kira::sound::static_sound::StaticSoundData;
use rand::RngExt;
use rand::seq::{IndexedRandom, SliceRandom};
use serde_json::Value;
use skia_safe::gradient::{
  Colors as GradientColors, Gradient, Interpolation, shaders as gradient_shaders,
};
use skia_safe::{
  Canvas, Color4f, Font, FontMgr, Paint, PaintStyle, Path, PathBuilder, Rect, Shader, TileMode,
};

use crate::pb::task_controller_grpc::TaskResult;
use crate::pb::thalamus_grpc::{AnalogResponse, Span};

use super::{BehaviorTask, PointSubscription, TaskContext, Window};

const SUCCESS_SOUND_PATH: &str = r"C:\Thalamus-Extensions\seokhee\success_clip.wav";
const ABORT_SOUND_PATH: &str = r"C:\Thalamus-Extensions\seokhee\failure_clip.wav";
const FAILURE_SOUND_PATH: &str = r"C:\Thalamus-Extensions\seokhee\timeout_failure.wav";

/// Degrees/pixels/meters conversions for the subject monitor. Ported from the
/// Python task's `Converter` class — only the pieces this port actually uses:
/// `deg_to_pixel_rel` (single-value form), and its two-value form plus
/// `deg_to_pixel_abs`.
struct Converter {
  deg_per_pixel: f64,
  screen_pixels: (f64, f64),
}

impl Converter {
  fn new(screen_pixels: (u32, u32), screen_width_m: f64, screen_distance_m: f64) -> Self {
    let screen_width_rad = 2.0 * (screen_width_m / 2.0).atan2(screen_distance_m);
    let rad_per_pixel = screen_width_rad / screen_pixels.0 as f64;
    let deg_per_pixel = 180.0 / std::f64::consts::PI * rad_per_pixel;
    Self {
      deg_per_pixel,
      screen_pixels: (screen_pixels.0 as f64, screen_pixels.1 as f64),
    }
  }

  fn deg_to_pixel_rel(&self, deg: f64) -> f64 {
    deg / self.deg_per_pixel
  }

  fn deg_to_pixel_rel_xy(&self, x_deg: f64, y_deg: f64) -> (f64, f64) {
    (x_deg / self.deg_per_pixel, y_deg / self.deg_per_pixel)
  }

  /// Converts degrees relative to screen center into absolute pixel
  /// coordinates (top-left origin).
  fn deg_to_pixel_abs(&self, x_deg: f64, y_deg: f64) -> (f64, f64) {
    let (x, y) = self.deg_to_pixel_rel_xy(x_deg, y_deg);
    (
      x + self.screen_pixels.0 / 2.0,
      y + self.screen_pixels.1 / 2.0,
    )
  }

  /// Ported from `Converter.relpix_to_absdeg` (two-value form): converts
  /// relative pixels to absolute degrees (relative to the screen corner
  /// rather than its center).
  fn relpix_to_absdeg(&self, x_pix: f64, y_pix: f64) -> (f64, f64) {
    (
      (x_pix + self.screen_pixels.0 / 2.0) * self.deg_per_pixel,
      (y_pix + self.screen_pixels.1 / 2.0) * self.deg_per_pixel,
    )
  }
}

/// The target-location parameters read from config, and the positions
/// derived from them — everything computed by `compute_target_positions`.
/// Cached so `VcpSetup` can tell, on each later trial, whether any of these
/// parameters changed and the positions need regenerating.
#[allow(dead_code)]
struct TargetPositions {
  polar_step_deg: i64,
  sector1_min: i64,
  sector1_max: i64,
  sector2_min: i64,
  sector2_max: i64,
  eccentricity_pix_min: f64,
  eccentricity_pix_max: f64,
  eccentric_circle_num: i64,
  circle_radii: Vec<f64>,
  rand_pos_i: usize,
  rand_pos: Vec<(f64, f64)>,
}

/// Ported from the Python task's `State` (line 128), an `IntEnum` of task
/// states. Declared in the same order as the Python source, since `render`
/// relies on that ordering (`state > State::Delay`, line 1169) matching
/// Python's `IntEnum` comparison — hence deriving `Ord` here.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
enum State {
  AcquireFixation,
  Fixate,
  PresentTarget,
  #[allow(dead_code)] // matches the Python source's own unused HOLD_TARGET0
  HoldTarget0,
  Delay,
  GoCue,
  AcquireTarget,
  HoldTarget,
  Success,
  FailureSaccade,
  FailureHold,
  AbortTarget,
  AbortDelay,
}

/// Mirrors the Python task module's global state (`converter`, `center`,
/// `rand_pos`, etc.): initialized once ever, the first time this task type
/// runs in the process (the `if converter is None:` guard in the Python
/// source), and reused by every trial after that — unlike `VcpInhibitionTask`
/// itself, which `task_controller::run` recreates fresh for each trial.
/// `target_positions` is further re-derived on any later trial whose config
/// changes the target-location parameters (the Python source's "Regenerate
/// target positions if user changes any of the parameters" block).
#[allow(dead_code)]
struct VcpSetup {
  converter: Converter,
  center: (i32, i32),
  center_f: (f64, f64),
  trial_num: AtomicU64,
  trial_saccade_count: AtomicU64,
  trial_saccade_success_count: AtomicU64,
  trial_saccade_abort_count: AtomicU64,
  trial_catch_count: AtomicU64,
  trial_catch_success_count: AtomicU64,
  photodiode_blinking_square: Color4f,
  photodiode_static_square: Color4f,
  success_sound: StaticSoundData,
  abort_sound: StaticSoundData,
  failure_sound: StaticSoundData,
  target_positions: Mutex<TargetPositions>,
  /// Ported from the Python task's global `gaze_failure_store` list: gaze
  /// points (and their marker color) recorded on trial failure, for the
  /// operator view's failure-trace overlay (`draw_gaze`, not yet ported).
  gaze_failure_store: Mutex<Vec<((i32, i32), Color4f)>>,
  /// Ported from the Python task's global `gaze_success_store` list: the
  /// same, but for trial successes.
  gaze_success_store: Mutex<Vec<((i32, i32), Color4f)>>,
  /// Ported from the Python task's global `reward_total_released_ms`:
  /// cumulative reward duration released across the whole session.
  reward_total_released_ms: Mutex<f64>,
}

static STATE: OnceLock<VcpSetup> = OnceLock::new();

/// Panics rather than defaulting: a missing/malformed field here means the
/// task config doesn't match what this task requires to run at all, which is
/// worth failing loudly on rather than silently proceeding with a bogus 0.
/// Reads a JSON number as `i64`, accepting a float-encoded whole number
/// (e.g. `20.0`) as well as a plain integer — some config values arrive
/// float-encoded depending on how they were authored.
fn value_as_i64(value: &Value) -> Option<i64> {
  value
    .as_i64()
    .or_else(|| value.as_f64().map(|f| f.round() as i64))
}

fn get_i64(config: &Value, key: &str) -> i64 {
  config
    .get(key)
    .and_then(value_as_i64)
    .unwrap_or_else(|| panic!("task config missing required integer field {key:?}: {config}"))
}

fn get_f64(config: &Value, key: &str) -> f64 {
  config
    .get(key)
    .and_then(Value::as_f64)
    .unwrap_or_else(|| panic!("task config missing required number field {key:?}: {config}"))
}

fn get_nested_i64(config: &Value, key: &str, sub_key: &str) -> i64 {
  config
    .get(key)
    .and_then(|v| v.get(sub_key))
    .and_then(value_as_i64)
    .unwrap_or_else(|| {
      panic!("task config missing required integer field {key:?}.{sub_key:?}: {config}")
    })
}

fn get_nested_f64(config: &Value, key: &str, sub_key: &str) -> f64 {
  config
    .get(key)
    .and_then(|v| v.get(sub_key))
    .and_then(Value::as_f64)
    .unwrap_or_else(|| {
      panic!("task config missing required number field {key:?}.{sub_key:?}: {config}")
    })
}

fn get_bool(config: &Value, key: &str) -> bool {
  config
    .get(key)
    .and_then(Value::as_bool)
    .unwrap_or_else(|| panic!("task config missing required boolean field {key:?}: {config}"))
}

/// Reads a 3-element `[r, g, b]` array field, as used by `target_color` and
/// `background_color`.
fn get_rgb(config: &Value, key: &str) -> (f64, f64, f64) {
  let component = |index: usize| {
    config
      .get(key)
      .and_then(|v| v.get(index))
      .and_then(Value::as_f64)
      .unwrap_or_else(|| {
        panic!("task config field {key:?}[{index}] missing or not a number: {config}")
      })
  };
  (component(0), component(1), component(2))
}

/// Ported from `TaskContext.get_value` (task_context.py): a plain
/// number/bool field is returned as-is; a `{min, max}` object yields a
/// uniform random sample in `[min, max]`. Either way, logs it as
/// `trial_summary_data.used_values`, mirroring `get_value`'s own internal
/// bookkeeping (task_context.py:519,526) — the same convention used for the
/// literal `context.trial_summary_data.used_values[key] = value` assignments
/// in the task file itself.
async fn get_value(config: &Value, context: &TaskContext, key: &str) -> f64 {
  let value_config = config
    .get(key)
    .unwrap_or_else(|| panic!("task config missing required field {key:?}: {config}"));
  let value = match value_config {
    Value::Number(n) => n
      .as_f64()
      .unwrap_or_else(|| panic!("task config field {key:?} is not a valid number: {config}")),
    Value::Bool(b) => {
      if *b {
        1.0
      } else {
        0.0
      }
    }
    Value::Object(_) => {
      let min = get_nested_f64(config, key, "min");
      let max = get_nested_f64(config, key, "max");
      rand::rng().random_range(min..=max)
    }
    other => {
      panic!("task config field {key:?} should be a number or {{min,max}} object, got {other}")
    }
  };
  context
    .log(&format!("trial_summary_data.used_values {key}={value}"))
    .await;
  value
}

/// Ported from the Python task's `pick_random_value`, a nested closure there
/// — promoted to a free function per this port's convention of avoiding
/// closures where a plain function will do (see `get_valid_angles`).
/// Mirrors `numpy.arange(min_val, max_val + step, step)` followed by
/// `random.choice`.
fn pick_random_value(min_val: f64, max_val: f64, step: f64) -> f64 {
  let count = ((max_val + step - min_val) / step).ceil().max(0.0) as usize;
  let possible_values: Vec<f64> = (0..count).map(|i| min_val + step * i as f64).collect();
  *possible_values
    .choose(&mut rand::rng())
    .expect("pick_random_value: empty range of possible values")
}

/// Ported from the Python task's `gaussian_gradient` (lines 49-63): builds a
/// radial "Gaussian" falloff shader — grayscale brightness decaying from the
/// center toward `background_color`'s red channel (matching the Python
/// source's use of only the red channel for brightness), fully transparent
/// at the outer edge.
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

/// Ported from `numpy.linspace`, as used by the Python task to lay out
/// `circle_radii`.
fn linspace(start: f64, stop: f64, num: usize) -> Vec<f64> {
  if num == 0 {
    return Vec::new();
  }
  if num == 1 {
    return vec![start];
  }
  let step = (stop - start) / (num - 1) as f64;
  (0..num).map(|i| start + step * i as f64).collect()
}

/// Ported from the Python task's `get_valid_angles`.
fn get_valid_angles(
  step_deg: i64,
  sector1_min: i64,
  sector1_max: i64,
  sector2_min: i64,
  sector2_max: i64,
) -> Vec<i64> {
  let step = step_deg as usize;
  let mut valid_angles: Vec<i64> = (sector1_min..sector1_max)
    .step_by(step)
    .chain((sector2_min..sector2_max).step_by(step))
    .collect();
  valid_angles.sort_unstable();
  valid_angles.dedup();
  valid_angles
}

/// Derives `target_positions` from `config`: shared by the Python source's
/// one-time `if converter is None:` setup (lines 702-728) and its "Regenerate
/// target positions if user changes any of the parameters" block (lines
/// 791-816) — the two are otherwise identical, differing only in when they
/// run.
async fn compute_target_positions(
  config: &Value,
  converter: &Converter,
  center_f: (f64, f64),
  context: &TaskContext,
) -> TargetPositions {
  let polar_step_deg = get_i64(config, "target_loc_polar_step_deg");
  let sector1_min = get_nested_i64(config, "target_loc_angle_sector1_deg", "min");
  let sector1_max = get_nested_i64(config, "target_loc_angle_sector1_deg", "max");
  let sector2_min = get_nested_i64(config, "target_loc_angle_sector2_deg", "min");
  let sector2_max = get_nested_i64(config, "target_loc_angle_sector2_deg", "max");
  let eccentricity_deg_min = get_nested_f64(config, "target_loc_eccentricity_deg", "min");
  let eccentricity_deg_max = get_nested_f64(config, "target_loc_eccentricity_deg", "max");
  let eccentric_circle_num = get_i64(config, "target_loc_eccentric_circle_num");

  let eccentricity_pix_min = converter.deg_to_pixel_rel(eccentricity_deg_min);
  let eccentricity_pix_max = converter.deg_to_pixel_rel(eccentricity_deg_max);

  let circle_radii = linspace(
    eccentricity_pix_min,
    eccentricity_pix_max,
    eccentric_circle_num.max(0) as usize,
  );

  let valid_angles_deg = get_valid_angles(
    polar_step_deg,
    sector1_min,
    sector1_max,
    sector2_min,
    sector2_max,
  );
  context
    .log(&format!("valid_angles_deg={valid_angles_deg:?}"))
    .await;

  let valid_angles_rad: Vec<f64> = valid_angles_deg
    .iter()
    .map(|&a| (a as f64).to_radians())
    .collect();

  let mut rand_pos: Vec<(f64, f64)> = circle_radii
    .iter()
    .flat_map(|&radius| {
      valid_angles_rad.iter().map(move |&angle| {
        (
          center_f.0 + radius * angle.cos(),
          center_f.1 - radius * angle.sin(),
        )
      })
    })
    .collect();
  rand_pos.shuffle(&mut rand::rng());

  TargetPositions {
    polar_step_deg,
    sector1_min,
    sector1_max,
    sector2_min,
    sector2_max,
    eccentricity_pix_min,
    eccentricity_pix_max,
    eccentric_circle_num,
    circle_radii,
    rand_pos_i: 0,
    rand_pos,
  }
}

/// Whether any target-location parameter in `config` differs from the
/// `target_positions` they were last derived from — ported from the Python
/// source's "Regenerate target positions if user changes any of the
/// parameters that affect target positions" condition (lines 767-774), which
/// truncates every comparison to `int`, matched here with `as i64`.
fn target_positions_stale(
  config: &Value,
  converter: &Converter,
  target_positions: &TargetPositions,
) -> bool {
  let eccentricity_deg_min = get_nested_f64(config, "target_loc_eccentricity_deg", "min");
  let eccentricity_deg_max = get_nested_f64(config, "target_loc_eccentricity_deg", "max");

  target_positions.eccentric_circle_num != get_i64(config, "target_loc_eccentric_circle_num")
    || target_positions.polar_step_deg != get_i64(config, "target_loc_polar_step_deg")
    || target_positions.eccentricity_pix_min as i64
      != converter.deg_to_pixel_rel(eccentricity_deg_min) as i64
    || target_positions.eccentricity_pix_max as i64
      != converter.deg_to_pixel_rel(eccentricity_deg_max) as i64
    || target_positions.sector1_min != get_nested_i64(config, "target_loc_angle_sector1_deg", "min")
    || target_positions.sector1_max != get_nested_i64(config, "target_loc_angle_sector1_deg", "max")
    || target_positions.sector2_min != get_nested_i64(config, "target_loc_angle_sector2_deg", "min")
    || target_positions.sector2_max != get_nested_i64(config, "target_loc_angle_sector2_deg", "max")
}

/// Ported from the Python task's `gaze_valid` (lines 65-72): clamps
/// out-of-bounds gaze points to a `(99999, 99999)` sentinel.
fn gaze_valid(x: i32, y: i32, monitorsubj_w_pix: i32, monitorsubj_h_pix: i32) -> (i32, i32) {
  if x < 0 || x > monitorsubj_w_pix || y < 0 || y > monitorsubj_h_pix {
    (99999, 99999)
  } else {
    (x, y)
  }
}

/// Ported from the repeated `abs(QPoint.dotProduct(v, v)) ** .5` pattern in
/// the Python task's `*_func` state-condition lambdas (e.g.
/// `acquire_fixation_func`): Euclidean distance between two pixel points.
fn distance(a: (i32, i32), b: (i32, i32)) -> f64 {
  let dx = (a.0 - b.0) as f64;
  let dy = (a.1 - b.1) as f64;
  (dx * dx + dy * dy).sqrt()
}

/// Builds a `wait_for`/`wait_for_hold` condition (an opaque `Fn() -> bool`,
/// per their doc comments — neither knows this closure has anything to do
/// with gaze) that's satisfied if `within` holds for any point drained from
/// `gaze_queue` since the previous call, so a transient in-then-out
/// excursion between two checks still gets caught however briefly it
/// lasted, rather than only ever seeing whatever the latest sample happens
/// to be at check time. `last_gaze` — a plain `Mutex` local the caller keeps
/// alive for the whole trial (not a `VcpInhibitionTask` field; a `Cell`
/// would be simpler, but isn't `Sync`, and this closure is held across
/// `.await` points in an `async_trait` `Send` future) — is updated with
/// each drained point and is what's checked once the queue has nothing new
/// (e.g. a condition already satisfied by an earlier call's draining, or
/// before any gaze has arrived at all).
fn gaze_condition<'a>(
  gaze_queue: &'a PointSubscription,
  last_gaze: &'a Mutex<(i32, i32)>,
  within: impl Fn((i32, i32)) -> bool + 'a,
) -> impl Fn() -> bool + 'a {
  move || {
    let mut satisfied = false;
    for point in gaze_queue.drain() {
      *last_gaze.lock().unwrap() = point;
      if within(point) {
        satisfied = true;
      }
    }
    satisfied || within(*last_gaze.lock().unwrap())
  }
}

/// Builds the `state_in` "digital pulse" payload used throughout the
/// Python task's state machine's `inject_analog` calls (e.g. lines
/// 1242-1246): a HIGH-then-LOW step, `duration_ms` milliseconds apart.
fn state_in_pulse(duration_ms: u64) -> AnalogResponse {
  AnalogResponse {
    data: vec![5.0, 0.0], // 5 = HIGH, 0 = LOW voltages
    spans: vec![Span {
      begin: 0,
      end: 2,
      name: "Time".to_string(),
      ..Default::default()
    }],
    sample_intervals: vec![1_000_000 * duration_ms], // ns
    ..Default::default()
  }
}

/// Builds the `reward_in` pulse payload (line 1425-1429): a HIGH-then-LOW
/// step, `duration_ms` milliseconds apart, on a `Reward` span (`state_in`
/// pulses use a `Time` span instead — see `state_in_pulse`).
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

/// Bundled directly rather than relying on the system font manager — see
/// `simple_arc`'s identical rationale (`assets/DejaVuSans-LICENSE.txt`) for
/// why. Used by `draw_text`, this port's version of the Python task's
/// `drawText` (lines 998-1010).
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

/// Ported from the Python task's `drawText` (lines 998-1010): draws `text`
/// at `(x, y)` over a background rect, colored to contrast with
/// `background_color_qt` (white-on-black if the trial background is pure
/// black, black-on-white otherwise). Simplified from the Python version,
/// which sizes its background rect to the full operator viewport width
/// rather than to the text.
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

/// Ported from the Python task's `draw_gaze`: a filled circle at `point`.
fn draw_gaze(canvas: &Canvas, point: (i32, i32), color: Color4f) {
  let mut paint = Paint::new(color, None);
  paint.set_anti_alias(true);
  canvas.draw_circle((point.0 as f32, point.1 as f32), 12.0, &paint);
}

/// Ported from the Python task's `trial_type` string (`"catch_trial"` /
/// `"saccade_trial"`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum TrialType {
  Saccade,
  Catch,
}

impl TrialType {
  fn as_str(self) -> &'static str {
    match self {
      TrialType::Saccade => "saccade_trial",
      TrialType::Catch => "catch_trial",
    }
  }
}

/// Per-trial state `render` (and `draw_gaussian`) need, captured at the end
/// of `run` — the Python source instead has `renderer`/`draw_gaussian` close
/// over `run`'s local variables directly.
struct TrialState {
  background_color_qt: Color4f,
  trial_type: TrialType,
  targetpos_pix: (i32, i32),
  targetpos_pix_f: (f64, f64),
  orientation_targ_ran: f64,
  width_targ_pix: f64,
  height_targ_pix: f64,
  shader: Shader,
  /// Ported from lines 842-853: the fixation cross path, in pixel space.
  cross: Path,
  off_opacity: f64,
  accpt_fix_radius_pix: f64,
  accpt_gaze_radius_pix: f64,
  /// `render` uses these instead of the live `VcpSetup` counters for the
  /// rate fields (`saccade_trial_success_rate` etc.), matching Python's
  /// `renderer` closure, which captures `run`'s locals as they were at
  /// definition time, not the (possibly-since-changed) live globals.
  stats: TrialStats,
  show_target: bool,
  luminance_targ_per: f64,
  hide_during_hold: bool
}

/// Trial-level counters/rates, computed once early in `run` (ported lines
/// 1216-1227) and bundled here for `abort_trial`'s summary log message
/// (lines 1284-1287, 1315-1318) and `TrialState.stats`.
#[derive(Clone, Copy)]
struct TrialStats {
  trial_num: u64,
  trial_saccade_success_count: u64,
  trial_saccade_count: u64,
  saccade_trial_success_rate: f64,
  trial_catch_success_count: u64,
  trial_catch_count: u64,
  catch_success_rate: f64,
  saccade_trial_abort_rate: f64,
}

/// The "VCP_inhibition_task" task's behavior.
pub struct VcpInhibitionTask {
  trial: Mutex<Option<TrialState>>,
  /// Ported from the Python task's `state` global (line 1046 `global
  /// state`/set at the start of each state-transition handler, e.g.
  /// `handle_acquire_fixation`). Stored per-trial here, since each
  /// `VcpInhibitionTask` instance already is one trial.
  state: Mutex<Option<State>>,
  /// The session-wide `TaskContext` passed to `run`, stashed here so
  /// `render` — which (per `BehaviorTask::render`'s contract) isn't itself
  /// passed a `TaskContext` — can still read the latest gaze point via
  /// `TaskContext::gaze` for the operator view's gaze marker/readout. `None`
  /// until `run` is first called.
  context: Mutex<Option<Arc<TaskContext>>>,
}

impl VcpInhibitionTask {
  pub fn new() -> Self {
    Self {
      trial: Mutex::new(None),
      state: Mutex::new(None),
      context: Mutex::new(None),
    }
  }

  /// Ported from `wait_for` (util.py:357-366): waits — event-driven, waking
  /// on `context.notify()` — until `condition` returns true or `timeout`
  /// elapses (waits indefinitely if `timeout` is `None`), then returns
  /// `condition`'s value at that point. `condition` is an opaque predicate —
  /// `wait_for` has no notion of gaze or touch itself; a condition that
  /// cares about gaze builds it via `gaze_condition` (see its doc comment)
  /// to consume a `PointSubscription` internally.
  async fn wait_for(
    &self,
    context: &TaskContext,
    condition: impl Fn() -> bool,
    timeout: Option<Duration>,
  ) -> bool {
    let deadline = timeout.map(|d| Instant::now() + d);
    loop {
      // Constructed before the check below, per `TaskContext::notify`'s doc
      // comment, so a push racing with that check isn't missed.
      let notified = context.notify().notified();
      if condition() {
        return true;
      }
      match deadline {
        None => notified.await,
        Some(deadline) => {
          let Some(remaining) = deadline.checked_duration_since(Instant::now()) else {
            return condition();
          };
          tokio::select! {
            _ = notified => {}
            _ = tokio::time::sleep(remaining) => {}
          }
        }
      }
    }
  }

  /// Ported from `wait_for_hold` (util.py:469-504, `include_blink = False`):
  /// waits for `condition` to hold continuously for `hold_duration`,
  /// tolerating brief lapses ("blinks") of up to `blink_duration` each (time
  /// spent blinking doesn't count against `hold_duration`) — a lapse not
  /// reacquired within `blink_duration` fails the hold. `blink_duration =
  /// None` (as Python's `fixate_func` passes) waits indefinitely to
  /// reacquire, so a lapse can never fail the hold that way.
  ///
  /// `blink_resets` is a newer `wait_for_hold` parameter (from a different
  /// machine's util.py than the one this port otherwise follows): when
  /// true, reacquiring after a blink restarts `hold_duration` from zero
  /// instead of just excluding the time spent blinking (`hold_target_func`'s
  /// saccade-trial branch passes `true`; every other call site here keeps
  /// the old behavior via `false`).
  ///
  /// Returns whether the hold succeeded.
  async fn wait_for_hold(
    &self,
    context: &TaskContext,
    condition: impl Fn() -> bool,
    hold_duration: Duration,
    blink_duration: Option<Duration>,
    blink_resets: bool,
  ) -> bool {
    let mut start = Instant::now();
    let mut time_spent_blinking = Duration::ZERO;

    loop {
      let Some(remaining) = (hold_duration + time_spent_blinking).checked_sub(start.elapsed())
      else {
        break;
      };

      let blinked = self
        .wait_for(context, || !condition(), Some(remaining))
        .await;
      if !blinked {
        break;
      }

      //context.log("BehavState=blink").await;

      let blink_start = Instant::now();
      let reacquired = self.wait_for(context, &condition, blink_duration).await;
      if !reacquired {
        return false;
      }

      if blink_resets {
        start = Instant::now();
        time_spent_blinking = Duration::ZERO;
      } else {
        time_spent_blinking += blink_start.elapsed();
      }
    }

    true
  }

  /// Ported from the abort/failure block repeated after PRESENT_TARGET
  /// (lines 1272-1290) and DELAY (lines 1301-1321): records the failed
  /// gaze, logs the abort, plays `abort_sound`, bumps
  /// `trial_saccade_abort_count`, logs the trial summary, sleeps
  /// `penalty_delay`, and returns the failing `TaskResult` (Python's
  /// `return TaskResult(False)`).
  #[allow(clippy::too_many_arguments)]
  async fn abort_trial(
    &self,
    context: &TaskContext,
    state: &VcpSetup,
    gaze: (i32, i32),
    abort_state: State,
    abort_pulse_ms: u64,
    monitorsubj_w_pix: i32,
    monitorsubj_h_pix: i32,
    stats: &TrialStats,
    penalty_delay: f64,
  ) -> TaskResult {
    let valid_gaze = gaze_valid(gaze.0, gaze.1, monitorsubj_w_pix, monitorsubj_h_pix);
    state.gaze_failure_store.lock().unwrap().push((
      valid_gaze,
      Color4f::new(1.0, 69.0 / 255.0, 0.0, 128.0 / 255.0),
    ));

    context.log("TrialResult=ABORT").await;
    *self.state.lock().unwrap() = Some(abort_state);
    context
      .inject_analog("state_in", state_in_pulse(abort_pulse_ms))
      .await;
    println!("{abort_state:?}");

    context.play_sound(state.abort_sound.clone());

    state
      .trial_saccade_abort_count
      .fetch_add(1, Ordering::Relaxed);
    context
      .log(&format!(
        "TRIAL_NUM={}, SACCADE_TRIAL_SUCCESS_COUNT={}, SACCADE_TRIAL_NUM={}, SACCADE_SUCCESS_RATE={:.2}%, CATCH_TRIAL_SUCCESS_COUNT={}, CATCH_TRIAL_NUM={}, CATCH_SUCCESS_RATE={:.2}%, ABORT_RATE={:.2}%",
        stats.trial_num,
        stats.trial_saccade_success_count,
        stats.trial_saccade_count,
        stats.saccade_trial_success_rate,
        stats.trial_catch_success_count,
        stats.trial_catch_count,
        stats.catch_success_rate,
        stats.saccade_trial_abort_rate
      ))
      .await;

    tokio::time::sleep(Duration::from_secs_f64(penalty_delay)).await;

    TaskResult {
      success: false,
      cancelled: false,
    }
  }

  /// Ported from `handle_acquire_fixation`/`acquire_fixation_func` (lines
  /// 454-475, 330-340): sets the behavioral state to `AcquireFixation`, logs
  /// it, then waits — via `wait_for`, no timeout — until the gaze is within
  /// `accpt_fix_radius_pix` of `center`.
  async fn handle_acquire_fixation(
    &self,
    context: &TaskContext,
    gaze_queue: &PointSubscription,
    last_gaze: &Mutex<(i32, i32)>,
    center: (i32, i32),
    accpt_fix_radius_pix: f64,
    monitorsubj_w_pix: i32,
    monitorsubj_h_pix: i32,
  ) {
    *self.state.lock().unwrap() = Some(State::AcquireFixation);
    context.log("BehavState=ACQUIRE_FIXATION").await;
    println!("{:?}", State::AcquireFixation);

    self
      .wait_for(
        context,
        gaze_condition(gaze_queue, last_gaze, |point| {
          let valid_gaze = gaze_valid(point.0, point.1, monitorsubj_w_pix, monitorsubj_h_pix);
          distance(valid_gaze, center) < accpt_fix_radius_pix
        }),
        None,
      )
      .await;
  }

  /// Ported from `handle_present_target`/`present_target_func` (lines
  /// 502-526, 373-386): sets state to `PresentTarget`, logs it, then waits —
  /// via `wait_for_hold`, tolerating one blink up to `blink_dur` — for the
  /// gaze to hold within `accpt_fix_radius_pix` of `center` (not
  /// `targetpos_pix`) for `present_target_duration`. `accpt_gaze_radius_pix`
  /// and `targetpos_pix` are threaded through only for parity with the
  /// Python signature — `present_target_func`'s condition doesn't actually
  /// use them (checks distance to `center` against `accpt_fix_radius_pix`,
  /// not `targetpos_pix`/`accpt_gaze_radius_pix`).
  #[allow(clippy::too_many_arguments)]
  async fn handle_present_target(
    &self,
    context: &TaskContext,
    gaze_queue: &PointSubscription,
    last_gaze: &Mutex<(i32, i32)>,
    center: (i32, i32),
    accpt_fix_radius_pix: f64,
    _accpt_gaze_radius_pix: f64,
    _targetpos_pix: (i32, i32),
    present_target_duration: Duration,
    blink_dur: Duration,
    monitorsubj_w_pix: i32,
    monitorsubj_h_pix: i32,
    converter: &Converter,
  ) -> bool {
    *self.state.lock().unwrap() = Some(State::PresentTarget);
    context.log("BehavState=PRESENT_TARGET").await;
    println!("{:?}", State::PresentTarget);

    let success = self
      .wait_for_hold(
        context,
        gaze_condition(gaze_queue, last_gaze, |point| {
          let valid_gaze = gaze_valid(point.0, point.1, monitorsubj_w_pix, monitorsubj_h_pix);
          distance(valid_gaze, center) < accpt_fix_radius_pix
        }),
        present_target_duration,
        Some(blink_dur),
        false,
      )
      .await;

    let gaze = *last_gaze.lock().unwrap();
    let temp_gaze = gaze_valid(gaze.0, gaze.1, monitorsubj_w_pix, monitorsubj_h_pix);
    context
      .log(&format!(
        "Gaze[X,Y]_pix-abs_after-presenting-target=({}, {})",
        temp_gaze.0, temp_gaze.1
      ))
      .await;
    let temp_gaze_deg = converter.relpix_to_absdeg(temp_gaze.0 as f64, temp_gaze.1 as f64);
    context
      .log(&format!(
        "Gaze[X,Y]_deg-abs_after-presenting-target=({}, {})",
        temp_gaze_deg.0, temp_gaze_deg.1
      ))
      .await;

    success
  }

  /// Ported from `handle_fixate`/`fixate_func` (lines 478-499, 359-371):
  /// sets state to `Fixate`, logs it, then waits — via `wait_for_hold`, with
  /// no blink timeout (Python passes `blink_duration=None`, so a lapse
  /// waits indefinitely to be reacquired rather than failing the hold) —
  /// for the gaze to hold within `accpt_fix_radius_pix` of `center` for
  /// `fix_duration`.
  #[allow(clippy::too_many_arguments)]
  async fn handle_fixate(
    &self,
    context: &TaskContext,
    gaze_queue: &PointSubscription,
    last_gaze: &Mutex<(i32, i32)>,
    center: (i32, i32),
    accpt_fix_radius_pix: f64,
    fix_duration: Duration,
    monitorsubj_w_pix: i32,
    monitorsubj_h_pix: i32,
    converter: &Converter,
  ) {
    *self.state.lock().unwrap() = Some(State::Fixate);
    context.log("BehavState=FIXATE").await;
    println!("{:?}", State::Fixate);

    self
      .wait_for_hold(
        context,
        gaze_condition(gaze_queue, last_gaze, |point| {
          let valid_gaze = gaze_valid(point.0, point.1, monitorsubj_w_pix, monitorsubj_h_pix);
          distance(valid_gaze, center) < accpt_fix_radius_pix
        }),
        fix_duration,
        None,
        false,
      )
      .await;

    let gaze = *last_gaze.lock().unwrap();
    let temp_gaze = gaze_valid(gaze.0, gaze.1, monitorsubj_w_pix, monitorsubj_h_pix);
    context
      .log(&format!(
        "Gaze[X,Y]_pix-abs_after-FIXATE=({}, {})",
        temp_gaze.0, temp_gaze.1
      ))
      .await;
    let temp_gaze_deg = converter.relpix_to_absdeg(temp_gaze.0 as f64, temp_gaze.1 as f64);
    context
      .log(&format!(
        "Gaze[X,Y]_deg-abs_after-FIXATE=({}, {})",
        temp_gaze_deg.0, temp_gaze_deg.1
      ))
      .await;
  }

  /// Ported from `handle_delay`/`delay_func` (lines 529-549, 388-399): sets
  /// state to `Delay`, logs it, then waits — via `wait_for_hold`, tolerating
  /// one blink up to `blink_dur` — for the gaze to hold within
  /// `accpt_fix_radius_pix` of `center` for `del_duration`, then logs the
  /// post-delay gaze in pixels/degrees. Returns whether the hold succeeded.
  #[allow(clippy::too_many_arguments)]
  async fn handle_delay(
    &self,
    context: &TaskContext,
    gaze_queue: &PointSubscription,
    last_gaze: &Mutex<(i32, i32)>,
    center: (i32, i32),
    accpt_fix_radius_pix: f64,
    del_duration: Duration,
    blink_dur: Duration,
    monitorsubj_w_pix: i32,
    monitorsubj_h_pix: i32,
    converter: &Converter,
  ) -> bool {
    *self.state.lock().unwrap() = Some(State::Delay);
    context.log("BehavState=DELAY").await;
    println!("{:?}", State::Delay);

    let now = Instant::now();
    let mut success = self
      .wait_for_hold(
        context,
        gaze_condition(gaze_queue, last_gaze, |point| {
          let valid_gaze = gaze_valid(point.0, point.1, monitorsubj_w_pix, monitorsubj_h_pix);
          distance(valid_gaze, center) < accpt_fix_radius_pix
        }),
        del_duration,
        Some(blink_dur),
        false,
      )
      .await;
    if now.elapsed() >= del_duration {
      success = true;
    }

    let gaze = *last_gaze.lock().unwrap();
    let temp_gaze = gaze_valid(gaze.0, gaze.1, monitorsubj_w_pix, monitorsubj_h_pix);
    context
      .log(&format!(
        "Gaze[X,Y]_pix-abs_after-delay=({}, {})",
        temp_gaze.0, temp_gaze.1
      ))
      .await;
    let temp_gaze_deg = converter.relpix_to_absdeg(temp_gaze.0 as f64, temp_gaze.1 as f64);
    context
      .log(&format!(
        "Gaze[X,Y]_deg-abs_after-delay=({}, {})",
        temp_gaze_deg.0, temp_gaze_deg.1
      ))
      .await;

    success
  }

  /// Ported from `handle_go_cue_reaction` (lines 552-575): sets state to
  /// `GoCue`, logs it, then waits — via `wait_for_hold`, tolerating one
  /// blink — for the gaze to hold within `go_cue_radius_pix` of `center`
  /// for a fixed 50ms, succeeding regardless of the hold's own result if
  /// those 50ms actually elapsed (Python: `if time.perf_counter() - now >
  /// duration_s: success = True`), then logs the post-reaction gaze in
  /// pixels/degrees. Returns whether the check succeeded.
  #[allow(clippy::too_many_arguments)]
  async fn handle_go_cue_reaction(
    &self,
    context: &TaskContext,
    gaze_queue: &PointSubscription,
    last_gaze: &Mutex<(i32, i32)>,
    center: (i32, i32),
    go_cue_radius_pix: f64,
    blink_dur: Duration,
    monitorsubj_w_pix: i32,
    monitorsubj_h_pix: i32,
    converter: &Converter,
  ) -> bool {
    *self.state.lock().unwrap() = Some(State::GoCue);
    context.log("BehavState=GO_CUE").await;
    println!("{:?}", State::GoCue);

    const DURATION: Duration = Duration::from_millis(50);
    let now = Instant::now();
    let mut success = self
      .wait_for_hold(
        context,
        gaze_condition(gaze_queue, last_gaze, |point| {
          let valid_gaze = gaze_valid(point.0, point.1, monitorsubj_w_pix, monitorsubj_h_pix);
          distance(valid_gaze, center) < go_cue_radius_pix
        }),
        DURATION,
        Some(blink_dur),
        false,
      )
      .await;
    if now.elapsed() > DURATION {
      success = true;
    }

    let gaze = *last_gaze.lock().unwrap();
    let temp_gaze = gaze_valid(gaze.0, gaze.1, monitorsubj_w_pix, monitorsubj_h_pix);
    context
      .log(&format!(
        "Gaze[X,Y]_pix-abs_after-go-cue-reaction=({}, {})",
        temp_gaze.0, temp_gaze.1
      ))
      .await;
    let temp_gaze_deg = converter.relpix_to_absdeg(temp_gaze.0 as f64, temp_gaze.1 as f64);
    context
      .log(&format!(
        "Gaze[X,Y]_deg-abs_after-go-cue-reaction=({}, {})",
        temp_gaze_deg.0, temp_gaze_deg.1
      ))
      .await;

    success
  }

  /// Ported from `handle_acquire_target`/`acquire_target_func` (lines
  /// 579-604, 401-424): sets state to `AcquireTarget`, logs it, then waits
  /// for the trial-type-appropriate condition — catch trials: gaze holds
  /// within `accpt_gaze_radius_pix` of `center` for `catch_duration`
  /// (`wait_for_hold`, tolerating one blink); saccade trials: gaze reaches
  /// within `accpt_gaze_radius_pix` of `targetpos_pix` within
  /// `decision_timeout` (`wait_for`, no hold/blink tolerance) — then logs
  /// the post-acquisition gaze in pixels/degrees. `accpt_fix_radius_pix` is
  /// threaded through only for parity with the Python signature —
  /// `acquire_target_func`'s conditions don't use it (both branches check
  /// against `accpt_gaze_radius_pix`).
  #[allow(clippy::too_many_arguments)]
  async fn handle_acquire_target(
    &self,
    context: &TaskContext,
    gaze_queue: &PointSubscription,
    last_gaze: &Mutex<(i32, i32)>,
    trial_type: TrialType,
    center: (i32, i32),
    targetpos_pix: (i32, i32),
    _accpt_fix_radius_pix: f64,
    accpt_gaze_radius_pix: f64,
    decision_timeout: Duration,
    blink_dur: Duration,
    catch_duration: Duration,
    monitorsubj_w_pix: i32,
    monitorsubj_h_pix: i32,
    converter: &Converter,
  ) -> bool {
    *self.state.lock().unwrap() = Some(State::AcquireTarget);
    context.log("BehavState=ACQUIRE_TARGET").await;
    println!("{:?}", State::AcquireTarget);

    let success = match trial_type {
      TrialType::Catch => {
        self
          .wait_for_hold(
            context,
            gaze_condition(gaze_queue, last_gaze, |point| {
              let valid_gaze = gaze_valid(point.0, point.1, monitorsubj_w_pix, monitorsubj_h_pix);
              distance(valid_gaze, center) < accpt_gaze_radius_pix
            }),
            catch_duration,
            Some(blink_dur),
            false,
          )
          .await
      }
      TrialType::Saccade => {
        self
          .wait_for(
            context,
            gaze_condition(gaze_queue, last_gaze, |point| {
              let valid_gaze = gaze_valid(point.0, point.1, monitorsubj_w_pix, monitorsubj_h_pix);
              distance(valid_gaze, targetpos_pix) < accpt_gaze_radius_pix
            }),
            Some(decision_timeout),
          )
          .await
      }
    };

    let gaze = *last_gaze.lock().unwrap();
    let temp_gaze = gaze_valid(gaze.0, gaze.1, monitorsubj_w_pix, monitorsubj_h_pix);
    context
      .log(&format!(
        "Gaze[X,Y]_pix-abs_after-acquiring-target=({}, {})",
        temp_gaze.0, temp_gaze.1
      ))
      .await;
    let temp_gaze_deg = converter.relpix_to_absdeg(temp_gaze.0 as f64, temp_gaze.1 as f64);
    context
      .log(&format!(
        "Gaze[X,Y]_deg-abs_after-acquiring-target=({}, {})",
        temp_gaze_deg.0, temp_gaze_deg.1
      ))
      .await;

    success
  }

  /// Ported from `handle_hold_target`/`hold_target_func` (lines 608-633,
  /// 426-450): sets state to `HoldTarget`, logs it, then waits for the
  /// trial-type-appropriate hold — catch trials: gaze holds within
  /// `accpt_gaze_radius_pix` of `center` for `hold_target_duration`
  /// (tolerating one blink up to `blink_dur`, `blink_resets = false`);
  /// saccade trials: gaze holds within `accpt_gaze_radius_pix` of
  /// `targetpos_pix` for `hold_target_duration` (tolerating one blink up to
  /// `decision_timeout` — not `blink_dur` — with `blink_resets = true`, so
  /// any blink restarts the hold from zero) — then logs the post-hold gaze
  /// in pixels/degrees. `accpt_fix_radius_pix` is threaded through only for
  /// parity with the Python signature — `hold_target_func`'s conditions
  /// don't use it. Returns whether the hold succeeded.
  #[allow(clippy::too_many_arguments)]
  async fn handle_hold_target(
    &self,
    context: &TaskContext,
    gaze_queue: &PointSubscription,
    last_gaze: &Mutex<(i32, i32)>,
    trial_type: TrialType,
    center: (i32, i32),
    targetpos_pix: (i32, i32),
    _accpt_fix_radius_pix: f64,
    accpt_gaze_radius_pix: f64,
    hold_target_duration: Duration,
    blink_dur: Duration,
    decision_timeout: Duration,
    monitorsubj_w_pix: i32,
    monitorsubj_h_pix: i32,
    converter: &Converter,
  ) -> bool {
    *self.state.lock().unwrap() = Some(State::HoldTarget);
    context.log("BehavState=HOLD_TARGET").await;
    println!("{:?}", State::HoldTarget);

    let success = match trial_type {
      TrialType::Catch => {
        self
          .wait_for_hold(
            context,
            gaze_condition(gaze_queue, last_gaze, |point| {
              let valid_gaze = gaze_valid(point.0, point.1, monitorsubj_w_pix, monitorsubj_h_pix);
              distance(valid_gaze, center) < accpt_gaze_radius_pix
            }),
            hold_target_duration,
            Some(blink_dur),
            false,
          )
          .await
      }
      TrialType::Saccade => {
        self
          .wait_for_hold(
            context,
            gaze_condition(gaze_queue, last_gaze, |point| {
              let valid_gaze = gaze_valid(point.0, point.1, monitorsubj_w_pix, monitorsubj_h_pix);
              distance(valid_gaze, targetpos_pix) < accpt_gaze_radius_pix
            }),
            hold_target_duration,
            Some(decision_timeout),
            true,
          )
          .await
      }
    };

    let gaze = *last_gaze.lock().unwrap();
    let temp_gaze = gaze_valid(gaze.0, gaze.1, monitorsubj_w_pix, monitorsubj_h_pix);
    context
      .log(&format!(
        "Gaze[X,Y]_pix-abs_after-holding-target=({}, {})",
        temp_gaze.0, temp_gaze.1
      ))
      .await;
    let temp_gaze_deg = converter.relpix_to_absdeg(temp_gaze.0 as f64, temp_gaze.1 as f64);
    context
      .log(&format!(
        "Gaze[X,Y]_deg-abs_after-holding-target=({}, {})",
        temp_gaze_deg.0, temp_gaze_deg.1
      ))
      .await;

    success
  }

  /// Ported from the Python task's `draw_gaussian` (line 944): draws this
  /// trial's gaussian target at `targetpos_pix`, rotated/scaled per
  /// `orientation_targ_ran`/`width_targ_pix`/`height_targ_pix`. Only the
  /// Python source's `else` branch (single current gaussian) is ported —
  /// its `if paint_all_targets:` branch is not. `alpha` has no Python
  /// equivalent; it's an added parameter (multiplied into the paint's alpha)
  /// so callers can fade the target in/out.
  #[allow(dead_code)]
  fn draw_gaussian(&self, canvas: &Canvas, alpha: f64) {
    let Some(trial) = self.trial.lock().unwrap().as_ref().map(|t| {
      (
        t.targetpos_pix,
        t.orientation_targ_ran,
        t.width_targ_pix,
        t.height_targ_pix,
        t.shader.clone(),
      )
    }) else {
      return;
    };
    let (targetpos_pix, orientation_targ_ran, width_targ_pix, height_targ_pix, shader) = trial;

    canvas.save();
    canvas.translate((targetpos_pix.0 as f32, targetpos_pix.1 as f32));
    canvas.rotate(orientation_targ_ran as f32, None);
    canvas.scale((1.0, (height_targ_pix / width_targ_pix) as f32));

    let mut paint = Paint::default();
    paint.set_shader(shader);
    paint.set_alpha_f(alpha as f32);
    let size = canvas.base_layer_size();
    let (width, height) = (size.width as f32, size.height as f32);
    canvas.draw_rect(
      Rect::from_xywh(-width / 2.0, -height / 2.0, width, height),
      &paint,
    );

    canvas.restore();
  }
}

#[async_trait]
impl BehaviorTask for VcpInhibitionTask {
  async fn run(&self, context: Arc<TaskContext>) -> TaskResult {
    // Stashed so `render` — which isn't itself passed a `TaskContext` (see
    // `BehaviorTask::render`'s doc comment) — can still show the latest
    // gaze point in the operator view.
    *self.context.lock().unwrap() = Some(context.clone());

    // A subscription covering this whole trial: every `handle_*` call below
    // threads `&gaze_queue`/`&last_gaze` through so its `wait_for`/
    // `wait_for_hold` condition (built via `gaze_condition`) sees every gaze
    // point received since the last check, not just whatever's latest —
    // see `gaze_condition`'s doc comment.
    let gaze_queue = context.subscribe_to_gaze();
    let last_gaze = Mutex::new((99999, 99999));

    let config = &context.config();
    let monitorsubj_w_pix = get_i64(config, "monitorsubj_W_pix");
    let monitorsubj_h_pix = get_i64(config, "monitorsubj_H_pix");
    let monitorsubj_dist_m = get_i64(config, "monitorsubj_dist_m");
    let monitorsubj_width_m = get_i64(config, "monitorsubj_width_m");

    // Ported from the Python task's `if converter is None:` block (lines
    // 688-741 of VCP_inhibition_task.py): setup that runs only the first
    // time this task type executes in the process, and is reused by every
    // trial after that (see `VcpSetup`/`STATE` above). The on_change/
    // WATCHING observer registration that follows it in the Python source
    // is not ported.
    if STATE.get().is_none() {
      let converter = Converter::new(
        (monitorsubj_w_pix as u32, monitorsubj_h_pix as u32),
        monitorsubj_width_m as f64,
        monitorsubj_dist_m as f64,
      );
      let center = (monitorsubj_w_pix as i32 / 2, monitorsubj_h_pix as i32 / 2);
      let center_f = (center.0 as f64, center.1 as f64);

      let success_sound =
        StaticSoundData::from_file(SUCCESS_SOUND_PATH).expect("failed to load success_clip.wav");
      let abort_sound =
        StaticSoundData::from_file(ABORT_SOUND_PATH).expect("failed to load failure_clip.wav");
      let failure_sound =
        StaticSoundData::from_file(FAILURE_SOUND_PATH).expect("failed to load timeout_failure.wav");

      let photodiode_blinking_square = Color4f::new(1.0, 1.0, 1.0, 1.0);
      let photodiode_static_square = Color4f::new(0.0, 0.0, 0.0, 1.0);

      let target_positions = compute_target_positions(config, &converter, center_f, &context).await;

      let _ = STATE.set(VcpSetup {
        converter,
        center,
        center_f,
        trial_num: AtomicU64::new(0),
        trial_saccade_count: AtomicU64::new(0),
        trial_saccade_success_count: AtomicU64::new(0),
        trial_saccade_abort_count: AtomicU64::new(0),
        trial_catch_count: AtomicU64::new(0),
        trial_catch_success_count: AtomicU64::new(0),
        photodiode_blinking_square,
        photodiode_static_square,
        success_sound,
        abort_sound,
        failure_sound,
        target_positions: Mutex::new(target_positions),
        gaze_failure_store: Mutex::new(Vec::new()),
        gaze_success_store: Mutex::new(Vec::new()),
        reward_total_released_ms: Mutex::new(0.0),
      });
    }

    // Ported from the Python task's "Regenerate target positions if user
    // changes any of the parameters that affect target positions" block
    // (lines 767-816): re-derives `target_positions` (via the same
    // `compute_target_positions` used above) whenever this trial's config
    // no longer matches the parameters they were last computed from.
    if let Some(state) = STATE.get() {
      let stale = target_positions_stale(
        config,
        &state.converter,
        &state.target_positions.lock().unwrap(),
      );
      if stale {
        let new_positions =
          compute_target_positions(config, &state.converter, state.center_f, &context).await;
        *state.target_positions.lock().unwrap() = new_positions;
      }
    }

    // Ported from lines 818-839: pick this trial's type, advance through
    // `rand_pos` (reshuffling once every position has been used), and derive
    // `targetpos_pix`/`targetpos_pix_f` accordingly. `current_rand_pos_i`
    // (line 834) has no equivalent in this port yet, so isn't ported; every
    // `context.trial_summary_data.used_values[key] = value` (lines 830-831)
    // becomes a `context.log` call instead, since there's no `trial_summary_data`
    // here — this is the convention to use for every such assignment.
    let state = STATE
      .get()
      .expect("VcpSetup should have been initialized above");

    let catch_trial_rate = get_f64(config, "catch_trial_rate");
    let trial_type = if rand::rng().random::<f64>() < catch_trial_rate {
      TrialType::Catch
    } else {
      TrialType::Saccade
    };

    let (targetpos_pix, targetpos_pix_f) = {
      let mut target_positions = state.target_positions.lock().unwrap();
      if target_positions.rand_pos_i == target_positions.rand_pos.len() {
        target_positions.rand_pos.shuffle(&mut rand::rng());
        target_positions.rand_pos_i = 0;
      }

      match trial_type {
        TrialType::Saccade => {
          let (x, y) = target_positions.rand_pos[target_positions.rand_pos_i];
          let targetpos_pix = (x as i32, y as i32);
          let targetpos_pix_f = (targetpos_pix.0 as f64, targetpos_pix.1 as f64);
          println!("Current target position (pix): {targetpos_pix:?}");
          target_positions.rand_pos_i += 1;
          (targetpos_pix, targetpos_pix_f)
        }
        TrialType::Catch => (state.center, (state.center.0 as f64, state.center.1 as f64)),
      }
    };

    if trial_type == TrialType::Saccade {
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
    }

    // Ported from lines 842-853: build the fixation cross path (two
    // perpendicular strokes centered on the screen) in pixel space.
    const VERTICES_DEG: [(f64, f64); 4] = [(-0.25, 0.0), (0.25, 0.0), (0.0, -0.25), (0.0, 0.25)];
    let vertices: Vec<(f64, f64)> = VERTICES_DEG
      .iter()
      .map(|&(x, y)| state.converter.deg_to_pixel_abs(x, y))
      .collect();

    let mut cross_builder = PathBuilder::new();
    cross_builder.move_to((vertices[0].0 as f32, vertices[0].1 as f32));
    cross_builder.line_to((vertices[1].0 as f32, vertices[1].1 as f32));
    cross_builder.move_to((vertices[2].0 as f32, vertices[2].1 as f32));
    cross_builder.line_to((vertices[3].0 as f32, vertices[3].1 as f32));
    let cross: Path = cross_builder.detach();

    // Ported from line 687 (before the one-time setup block, so read every
    // trial): controls the alpha of the fixation cross / catch-trial cue in
    // several `render` states.
    let off_opacity = get_f64(config, "off_opacity");

    // Ported from lines 860-901: read the remaining per-trial config
    // (acceptance radii, target/background color, timeouts), then randomly
    // sample this trial's target luminance/orientation/size. Every
    // `context.trial_summary_data.used_values[key] = value` (lines 892, 894,
    // 896, 901) becomes a `context.log` call, per the established
    // convention; `context.get_value(key)` (lines 871-875, 878, 890) does the
    // same internally (see `get_value` above).
    let accpt_fix_radius_deg = get_f64(config, "accpt_fix_radius_deg");
    let accpt_fix_radius_pix = state.converter.deg_to_pixel_rel(accpt_fix_radius_deg);
    let accpt_gaze_radius_deg = get_f64(config, "accpt_gaze_radius_deg");
    let accpt_gaze_radius_pix = state.converter.deg_to_pixel_rel(accpt_gaze_radius_deg);
    let is_height_locked = get_bool(config, "is_height_locked");
    let paint_all_targets = get_bool(config, "paint_all_targets");
    let hide_during_hold = get_bool(config, "hide_during_hold");
    let target_color_rgb = get_rgb(config, "target_color");
    let background_color = get_rgb(config, "background_color");
    let background_color_qt = Color4f::new(
      background_color.0 as f32 / 255.0,
      background_color.1 as f32 / 255.0,
      background_color.2 as f32 / 255.0,
      1.0,
    );

    let decision_timeout = get_value(config, &context, "decision_timeout").await / 1000.0;
    let fix_duration = get_value(config, &context, "fix_duration").await / 1000.0;
    let del_duration = get_value(config, &context, "del_duration").await / 1000.0;
    let present_target_duration =
      get_value(config, &context, "present_target_duration").await / 1000.0;
    let hold_target_duration = get_value(config, &context, "hold_target_duration").await / 1000.0;
    let penalty_delay = get_f64(config, "penalty_delay") / 1000.0;
    let blink_dur = get_f64(config, "blink_dur") / 1000.0;
    let catch_duration = get_value(config, &context, "catch_duration").await / 1000.0;

    println!("{fix_duration} {present_target_duration} {del_duration}");

    let reward_per_trial = get_value(config, &context, "reward_per_trial").await;

    let luminance_targ_per = pick_random_value(
      get_nested_f64(config, "luminance_targ_per", "min"),
      get_nested_f64(config, "luminance_targ_per", "max"),
      get_f64(config, "luminance_targ_step"),
    );
    context
      .log(&format!(
        "trial_summary_data.used_values luminance_targ_per={luminance_targ_per}"
      ))
      .await;

    let orientation_targ_ran = pick_random_value(
      get_nested_f64(config, "orientation_targ_ran", "min"),
      get_nested_f64(config, "orientation_targ_ran", "max"),
      get_f64(config, "orientation_targ_step"),
    );
    context
      .log(&format!(
        "trial_summary_data.used_values orientation_targ_ran={orientation_targ_ran}"
      ))
      .await;

    let width_targ_pix = state.converter.deg_to_pixel_rel(pick_random_value(
      get_nested_f64(config, "width_targ_deg", "min"),
      get_nested_f64(config, "width_targ_deg", "max"),
      get_f64(config, "widthtargdeg_step"),
    ));
    context
      .log(&format!(
        "trial_summary_data.used_values width_targ_pix={width_targ_pix}"
      ))
      .await;

    let height_targ_pix = if is_height_locked {
      width_targ_pix
    } else {
      state.converter.deg_to_pixel_rel(pick_random_value(
        get_nested_f64(config, "height_targ_deg", "min"),
        get_nested_f64(config, "height_targ_deg", "max"),
        get_f64(config, "heighttargdeg_step"),
      ))
    };
    context
      .log(&format!(
        "trial_summary_data.used_values height_targ_pix={height_targ_pix}"
      ))
      .await;

    // Ported from line 943: builds this trial's gaussian target shader (used
    // by `draw_gaussian` above), capturing everything it needs since — unlike
    // the Python source's nested closure — it can't just close over `run`'s
    // locals. `paint_all_targets` only affects `draw_gaussian`'s unported
    // branch, so it isn't threaded through here.
    let shader = gaussian_gradient_shader(
      background_color_qt,
      width_targ_pix as f32 / 2.0,
      3.0,
      255.0,
      100.0,
    );
    println!(
      "gaussian target: pos={targetpos_pix:?} width_pix={width_targ_pix:.1} height_pix={height_targ_pix:.1} luminance_pct={luminance_targ_per:.1} orientation_deg={orientation_targ_ran:.1} background_rgb=({}, {}, {})",
      (background_color_qt.r * 255.0) as u8,
      (background_color_qt.g * 255.0) as u8,
      (background_color_qt.b * 255.0) as u8,
    );

    let _ = (
      paint_all_targets,
      target_color_rgb,
      decision_timeout,
      hold_target_duration,
      penalty_delay,
      catch_duration,
      reward_per_trial,
    );

    // Ported from lines 1214-1238. Python logs `context.config` here — the
    // full observable app-config tree, not `context.task_config` (this
    // file's `config`) — but that tree isn't threaded into `TaskContext`
    // (only the per-trial task config is), so this logs `config` instead;
    // flag if the full app-config tree turns out to be needed here.
    context
      .log(&serde_json::to_string(config).expect("config should always serialize to JSON"))
      .await;

    let trial_saccade_count = state.trial_saccade_count.load(Ordering::Relaxed);
    let trial_saccade_success_count = state.trial_saccade_success_count.load(Ordering::Relaxed);
    let (saccade_trial_success_rate, saccade_trial_abort_rate) = if trial_saccade_count == 0 {
      (0.0, 0.0)
    } else {
      let trial_saccade_abort_count = state.trial_saccade_abort_count.load(Ordering::Relaxed);
      (
        trial_saccade_success_count as f64 / trial_saccade_count as f64 * 100.0,
        trial_saccade_abort_count as f64 / trial_saccade_count as f64 * 100.0,
      )
    };
    let trial_catch_count = state.trial_catch_count.load(Ordering::Relaxed);
    let trial_catch_success_count = state.trial_catch_success_count.load(Ordering::Relaxed);
    let catch_success_rate = if trial_catch_count == 0 {
      0.0
    } else {
      trial_catch_success_count as f64 / trial_catch_count as f64 * 100.0
    };

    let trial_num = state.trial_num.fetch_add(1, Ordering::Relaxed) + 1;
    // Bundles the counters/rates above (ported lines 1216-1227) for
    // `abort_trial`'s summary log message (lines 1284-1287, 1315-1318).
    // Built here, before `trial_saccade_count`/`trial_catch_count` below get
    // incremented for this trial, matching the Python source's own
    // staleness (it logs these same already-computed locals from any abort
    // branch, not fresh reloads).
    let trial_stats = TrialStats {
      trial_num,
      trial_saccade_success_count,
      trial_saccade_count,
      saccade_trial_success_rate,
      trial_catch_success_count,
      trial_catch_count,
      catch_success_rate,
      saccade_trial_abort_rate,
    };

    *self.trial.lock().unwrap() = Some(TrialState {
      background_color_qt,
      trial_type,
      targetpos_pix,
      targetpos_pix_f,
      orientation_targ_ran,
      width_targ_pix,
      height_targ_pix,
      shader,
      cross,
      off_opacity,
      accpt_fix_radius_pix,
      accpt_gaze_radius_pix,
      stats: trial_stats,
      show_target: false,
      luminance_targ_per,
      hide_during_hold
    });

    match trial_type {
      TrialType::Catch => {
        state.trial_catch_count.fetch_add(1, Ordering::Relaxed);
      }
      TrialType::Saccade => {
        state.trial_saccade_count.fetch_add(1, Ordering::Relaxed);
      }
    }
    println!(
      "Started trial # {trial_num}, trial type = {}",
      trial_type.as_str()
    );
    context
      .log(&format!(
        "Started TRIAL_NUM={trial_num}, TRIAL_TYPE={}",
        trial_type.as_str()
      ))
      .await;
    context
      .log(&format!(
        "targetpos_pixX={}, targetpos_pixY={}, luminance_targ_per={luminance_targ_per}, orientation_targ_ran={orientation_targ_ran}, width_targ_pix={width_targ_pix}, height_targ_pix={height_targ_pix}",
        targetpos_pix.0, targetpos_pix.1
      ))
      .await;

    // Ported from line 1241 (`handle_acquire_fixation` call): waits for
    // fixation on the screen center before continuing.
    self
      .handle_acquire_fixation(
        &context,
        &gaze_queue,
        &last_gaze,
        state.center,
        accpt_fix_radius_pix,
        monitorsubj_w_pix as i32,
        monitorsubj_h_pix as i32,
      )
      .await;

    // Ported from lines 1242-1246: signals the ACQUIRE_FIXATION state on the
    // 'state_in' analog node. Python fires this via
    // `create_task_with_exc_handling` (background, errors just logged); this
    // just awaits it directly instead.
    context.inject_analog("state_in", state_in_pulse(20)).await;

    // Ported from lines 1248-1258: waits for center fixation to hold for
    // 'fix_duration', then signals FIXATE on 'state_in'.
    self
      .handle_fixate(
        &context,
        &gaze_queue,
        &last_gaze,
        state.center,
        accpt_fix_radius_pix,
        Duration::from_secs_f64(fix_duration),
        monitorsubj_w_pix as i32,
        monitorsubj_h_pix as i32,
        &state.converter,
      )
      .await;
    context.inject_analog("state_in", state_in_pulse(20)).await;

    // Ported from lines 1260-1267: waits for continued center fixation
    // through target presentation, then signals PRESENT_TARGET on
    // 'state_in'.
    let present_target_success = self
      .handle_present_target(
        &context,
        &gaze_queue,
        &last_gaze,
        state.center,
        accpt_fix_radius_pix,
        accpt_gaze_radius_pix,
        targetpos_pix,
        Duration::from_secs_f64(present_target_duration),
        Duration::from_secs_f64(blink_dur),
        monitorsubj_w_pix as i32,
        monitorsubj_h_pix as i32,
        &state.converter,
      )
      .await;
    context.inject_analog("state_in", state_in_pulse(20)).await;

    // Ported from lines 1272-1290: on failure to hold fixation through
    // target presentation, aborts the trial — Python's `return
    // TaskResult(False)`.
    if !present_target_success {
      let gaze = *last_gaze.lock().unwrap();
      return self
        .abort_trial(
          &context,
          state,
          gaze,
          State::AbortTarget,
          150,
          monitorsubj_w_pix as i32,
          monitorsubj_h_pix as i32,
          &trial_stats,
          penalty_delay,
        )
        .await;
    }

    // Ported from lines 1292-1300: waits for continued center fixation
    // through the delay period, then signals DELAY on 'state_in'.
    let delay_start = Instant::now();
    let delay_success = self
      .handle_delay(
        &context,
        &gaze_queue,
        &last_gaze,
        state.center,
        accpt_fix_radius_pix,
        Duration::from_secs_f64(del_duration),
        Duration::from_secs_f64(blink_dur),
        monitorsubj_w_pix as i32,
        monitorsubj_h_pix as i32,
        &state.converter,
      )
      .await;
    context.inject_analog("state_in", state_in_pulse(20)).await;

    // Ported from lines 1301-1321: on failure to hold fixation through the
    // delay, aborts the trial (as above, but tagged ABORT_DELAY with a
    // 300ms pulse instead of ABORT_TARGET's 150ms).
    if !delay_success {
      let gaze = *last_gaze.lock().unwrap();
      return self
        .abort_trial(
          &context,
          state,
          gaze,
          State::AbortDelay,
          300,
          monitorsubj_w_pix as i32,
          monitorsubj_h_pix as i32,
          &trial_stats,
          penalty_delay,
        )
        .await;
    }

    // Ported from lines 1322-1325 (the `else` branch): logs how long the
    // delay actually took.
    let delay_dur_ms = delay_start.elapsed().as_secs_f64() * 1000.0;
    context.log(&format!("Duration={delay_dur_ms:.2}")).await;

    // Ported from lines 1330-1336: checks for a premature saccade during
    // the reaction window right after the delay/go cue. No 'state_in' pulse
    // here — Python doesn't send one for GO_CUE either.
    let go_cue_radius_pix = if trial_type == TrialType::Saccade {
      accpt_fix_radius_pix
    } else {
      accpt_gaze_radius_pix
    };
    let go_cue_success = self
      .handle_go_cue_reaction(
        &context,
        &gaze_queue,
        &last_gaze,
        state.center,
        go_cue_radius_pix,
        Duration::from_secs_f64(blink_dur),
        monitorsubj_w_pix as i32,
        monitorsubj_h_pix as i32,
        &state.converter,
      )
      .await;

    // Ported from lines 1338-1358: on failure, aborts the trial exactly as
    // the DELAY failure above does — same ABORT_DELAY tag/pulse, and
    // (Python's `# await context.log('Duration=...')` there is commented
    // out, unlike DELAY's own) no duration log.
    if !go_cue_success {
      let gaze = *last_gaze.lock().unwrap();
      return self
        .abort_trial(
          &context,
          state,
          gaze,
          State::AbortDelay,
          300,
          monitorsubj_w_pix as i32,
          monitorsubj_h_pix as i32,
          &trial_stats,
          penalty_delay,
        )
        .await;
    }

    // Ported from lines 1359-1362 (the `else` branch): logs the elapsed
    // time again, still measured from `delay_start` — Python reuses the
    // same `delay_start`/`delay_dur` here rather than timing the go-cue
    // reaction separately.
    let delay_dur_ms = delay_start.elapsed().as_secs_f64() * 1000.0;
    context.log(&format!("Duration={delay_dur_ms:.2}")).await;

    // Ported from lines 1364-1370: waits for target acquisition (saccade
    // trials) or continued center fixation (catch trials).
    let acquire_target_success = self
      .handle_acquire_target(
        &context,
        &gaze_queue,
        &last_gaze,
        trial_type,
        state.center,
        targetpos_pix,
        accpt_fix_radius_pix,
        accpt_gaze_radius_pix,
        Duration::from_secs_f64(decision_timeout),
        Duration::from_secs_f64(blink_dur),
        Duration::from_secs_f64(catch_duration),
        monitorsubj_w_pix as i32,
        monitorsubj_h_pix as i32,
        &state.converter,
      )
      .await;
    context.inject_analog("state_in", state_in_pulse(20)).await;

    // Ported from lines 1372-1392: on failure to acquire the target in
    // time, fails the trial — distinct from `abort_trial`'s ABORT_* cases:
    // tagged FAILURE_SACCADE, plays `failure_sound` (not `abort_sound`),
    // doesn't bump `trial_saccade_abort_count`, and splits the penalty
    // sleep in two around a `show_target` toggle (see `render`'s
    // `FailureSaccade` branch), so not reused via `abort_trial`.
    if !acquire_target_success {
      let gaze = *last_gaze.lock().unwrap();
      let valid_gaze = gaze_valid(
        gaze.0,
        gaze.1,
        monitorsubj_w_pix as i32,
        monitorsubj_h_pix as i32,
      );
      state.gaze_failure_store.lock().unwrap().push((
        valid_gaze,
        Color4f::new(1.0, 69.0 / 255.0, 0.0, 128.0 / 255.0),
      ));

      context.log("TrialResult=FAILURE_SACCADE").await;
      *self.state.lock().unwrap() = Some(State::FailureSaccade);
      println!("{:?}", State::FailureSaccade);
      context.inject_analog("state_in", state_in_pulse(450)).await;

      context.play_sound(state.failure_sound.clone());

      context
        .log(&format!(
          "TRIAL_NUM={}, SACCADE_TRIAL_SUCCESS_COUNT={}, SACCADE_TRIAL_NUM={}, SACCADE_SUCCESS_RATE={:.2}%, CATCH_TRIAL_SUCCESS_COUNT={}, CATCH_TRIAL_NUM={}, CATCH_SUCCESS_RATE={:.2}%, ABORT_RATE={:.2}%",
          trial_stats.trial_num,
          trial_stats.trial_saccade_success_count,
          trial_stats.trial_saccade_count,
          trial_stats.saccade_trial_success_rate,
          trial_stats.trial_catch_success_count,
          trial_stats.trial_catch_count,
          trial_stats.catch_success_rate,
          trial_stats.saccade_trial_abort_rate
        ))
        .await;

      // Ported from lines 1388-1391: hides the target for the first half of
      // the penalty period, then shows it for the second half.
      self.trial.lock().unwrap().as_mut().unwrap().show_target = false;
      tokio::time::sleep(Duration::from_secs_f64(penalty_delay / 2.0)).await;
      self.trial.lock().unwrap().as_mut().unwrap().show_target = true;
      tokio::time::sleep(Duration::from_secs_f64(penalty_delay / 2.0)).await;

      return TaskResult {
        success: false,
        cancelled: false,
      };
    }

    // Ported from lines 1394-1395: waits for the target hold, no 'state_in'
    // pulse afterward — Python doesn't send one for HOLD_TARGET.
    let hold_target_success = self
      .handle_hold_target(
        &context,
        &gaze_queue,
        &last_gaze,
        trial_type,
        state.center,
        targetpos_pix,
        accpt_fix_radius_pix,
        accpt_gaze_radius_pix,
        Duration::from_secs_f64(hold_target_duration),
        Duration::from_secs_f64(blink_dur),
        Duration::from_secs_f64(decision_timeout),
        monitorsubj_w_pix as i32,
        monitorsubj_h_pix as i32,
        &state.converter,
      )
      .await;

    // Ported from lines 1397-1410: on failure to hold the target, fails the
    // trial — tagged FAILURE_HOLD, plays `failure_sound`, no 'state_in'
    // pulse and no split sleep this time (unlike FAILURE_SACCADE), doesn't
    // bump `trial_saccade_abort_count`.
    if !hold_target_success {
      let gaze = *last_gaze.lock().unwrap();
      let valid_gaze = gaze_valid(
        gaze.0,
        gaze.1,
        monitorsubj_w_pix as i32,
        monitorsubj_h_pix as i32,
      );
      state.gaze_failure_store.lock().unwrap().push((
        valid_gaze,
        Color4f::new(1.0, 69.0 / 255.0, 0.0, 128.0 / 255.0),
      ));

      context.log("TrialResult=FAILURE_HOLD").await;
      *self.state.lock().unwrap() = Some(State::FailureHold);
      println!("{:?}", State::FailureHold);

      context.play_sound(state.failure_sound.clone());

      context
        .log(&format!(
          "TRIAL_NUM={}, SACCADE_TRIAL_SUCCESS_COUNT={}, SACCADE_TRIAL_NUM={}, SACCADE_SUCCESS_RATE={:.2}%, CATCH_TRIAL_SUCCESS_COUNT={}, CATCH_TRIAL_NUM={}, CATCH_SUCCESS_RATE={:.2}%, ABORT_RATE={:.2}%",
          trial_stats.trial_num,
          trial_stats.trial_saccade_success_count,
          trial_stats.trial_saccade_count,
          trial_stats.saccade_trial_success_rate,
          trial_stats.trial_catch_success_count,
          trial_stats.trial_catch_count,
          trial_stats.catch_success_rate,
          trial_stats.saccade_trial_abort_rate
        ))
        .await;

      tokio::time::sleep(Duration::from_secs_f64(penalty_delay)).await;

      return TaskResult {
        success: false,
        cancelled: false,
      };
    }

    // Ported from lines 1412-1450: the trial succeeded — plays
    // `success_sound`, releases the reward pulse, bumps the
    // trial-type-appropriate success counter, and logs/prints the same
    // summary shape as the abort/failure branches above.
    let gaze = *last_gaze.lock().unwrap();
    let valid_gaze = gaze_valid(
      gaze.0,
      gaze.1,
      monitorsubj_w_pix as i32,
      monitorsubj_h_pix as i32,
    );
    state
      .gaze_success_store
      .lock()
      .unwrap()
      .push((valid_gaze, Color4f::new(0.0, 1.0, 0.0, 128.0 / 255.0)));

    context.log("TrialResult=SUCCESS").await;
    *self.state.lock().unwrap() = Some(State::Success);
    println!("{:?}", State::Success);

    context.play_sound(state.success_sound.clone());
    // 1s delay to allow playing the sound; it doesn't play without this.
    tokio::time::sleep(Duration::from_secs(1)).await;

    let reward_total_released_ms = {
      let mut total = state.reward_total_released_ms.lock().unwrap();
      *total += reward_per_trial;
      *total
    };
    context
      .log(&format!(
        "starting_reward_release_of = {} ms, total_released = {} ms",
        reward_per_trial as i64, reward_total_released_ms as i64
      ))
      .await;
    context
      .inject_analog("reward_in", reward_pulse(reward_per_trial as u64))
      .await;

    // `trial_stats.trial_saccade_success_count`/`trial_catch_success_count`
    // were snapshotted earlier in `run`, before this trial could have
    // changed them; Python instead reads the live global at this point, so
    // the count for whichever branch below actually fires needs the fresh
    // post-increment value — the other one is unaffected by this trial, so
    // the snapshot is still accurate for it.
    let (trial_saccade_success_count, trial_catch_success_count) = match trial_type {
      TrialType::Saccade => (
        state
          .trial_saccade_success_count
          .fetch_add(1, Ordering::Relaxed)
          + 1,
        trial_stats.trial_catch_success_count,
      ),
      TrialType::Catch => (
        trial_stats.trial_saccade_success_count,
        state
          .trial_catch_success_count
          .fetch_add(1, Ordering::Relaxed)
          + 1,
      ),
    };

    let summary = format!(
      "TRIAL_NUM={}, SACCADE_TRIAL_SUCCESS_COUNT={}, SACCADE_TRIAL_NUM={}, SACCADE_SUCCESS_RATE={:.2}%, CATCH_TRIAL_SUCCESS_COUNT={}, CATCH_TRIAL_NUM={}, CATCH_SUCCESS_RATE={:.2}%, ABORT_RATE={:.2}%",
      trial_stats.trial_num,
      trial_saccade_success_count,
      trial_stats.trial_saccade_count,
      trial_stats.saccade_trial_success_rate,
      trial_catch_success_count,
      trial_stats.trial_catch_count,
      trial_stats.catch_success_rate,
      trial_stats.saccade_trial_abort_rate
    );
    context.log(&summary).await;
    println!("{summary}");

    // Ported from lines 1448-1450: Python always returns `TaskResult(False)`
    // here, regardless of the trial's outcome — per its own comment, a
    // `False` result keeps the trial queued instead of removing it, so this
    // is the intended behavior, not a bug to "fix" into `success: true`.
    TaskResult {
      success: false,
      cancelled: false,
    }
  }

  /// Ported from the Python task's `renderer` (lines 1045-1209). Not
  /// ported: the `paint_all_targets` sector-grid overlay (lines 1071-1101),
  /// `draw_gaussian_target`'s own drawn_objects bookkeeping (PRESENT_TARGET
  /// draws via `draw_gaussian` instead — see below), the `do_clear`
  /// button-clear (lines 1052-1056, no operator "Clear" button wiring yet),
  /// and the `show_target`-gated FAILURE_SACCADE draw (`show_target` isn't
  /// tracked in this port — see `handle_acquire_target`'s doc comment).
  fn render(&self, canvas: &Canvas, window: Window) {
    let Some((
      background_color_qt,
      trial_type,
      targetpos_pix_f,
      off_opacity,
      accpt_fix_radius_pix,
      accpt_gaze_radius_pix,
      cross,
      stats,
      show_target,
      luminance_targ_per,
      hide_during_hold,
    )) = self.trial.lock().unwrap().as_ref().map(|t| {
      (
        t.background_color_qt,
        t.trial_type,
        t.targetpos_pix_f,
        t.off_opacity,
        t.accpt_fix_radius_pix,
        t.accpt_gaze_radius_pix,
        t.cross.clone(),
        t.stats,
        t.show_target,
        t.luminance_targ_per,
        t.hide_during_hold,
      )
    })
    else {
      return;
    };
    let Some(setup) = STATE.get() else {
      return;
    };
    let Some(current_state) = *self.state.lock().unwrap() else {
      return;
    };

    let (monitorsubj_w_pix, monitorsubj_h_pix) = setup.converter.screen_pixels;
    let gaze = self
      .context
      .lock()
      .unwrap()
      .as_ref()
      .and_then(|context| context.gaze())
      .unwrap_or((99999, 99999));
    let valid_gaze = gaze_valid(
      gaze.0,
      gaze.1,
      monitorsubj_w_pix as i32,
      monitorsubj_h_pix as i32,
    );

    // Line 1048.
    canvas.draw_rect(
      Rect::from_xywh(0.0, 0.0, 4000.0, 4000.0),
      &Paint::new(background_color_qt, None),
    );
    let off_luminance = luminance_targ_per/100.0;

    // Line 1049 default; overridden to white in the GO_CUE/PRESENT_TARGET
    // branches below (lines 1115, 1127).
    let mut current_photodiode_static_square = Color4f::new(0.0, 0.0, 0.0, 1.0);

    // Lines 1061-1150: draw the fixation cross and/or gaussian target for
    // the current behavioral state.
    match current_state {
      State::AcquireFixation | State::Fixate => {
        let mut pen = Paint::new(Color4f::new(1.0, 0.0, 0.0, 1.0), None);
        pen.set_style(PaintStyle::Stroke);
        pen.set_stroke_width(2.0);
        pen.set_anti_alias(true);
        canvas.draw_path(&cross, &pen);
      }
      State::Delay => {
        let mut pen = Paint::new(Color4f::new(1.0, 0.0, 0.0, 1.0), None);
        pen.set_style(PaintStyle::Stroke);
        pen.set_stroke_width(2.0);
        pen.set_anti_alias(true);
        canvas.draw_path(&cross, &pen);
        if trial_type == TrialType::Saccade {
          self.draw_gaussian(canvas, off_luminance);
        }
      }
      State::GoCue => {
        current_photodiode_static_square = Color4f::new(1.0, 1.0, 1.0, 1.0);
        if trial_type == TrialType::Catch {
          let mut pen = Paint::new(Color4f::new(1.0, 0.0, 0.0, off_opacity as f32), None);
          pen.set_style(PaintStyle::Stroke);
          pen.set_stroke_width(2.0);
          pen.set_anti_alias(true);
          canvas.draw_path(&cross, &pen);
        }
        if trial_type == TrialType::Saccade {
          self.draw_gaussian(canvas, off_luminance);
        }
      }
      State::PresentTarget => {
        current_photodiode_static_square = Color4f::new(1.0, 1.0, 1.0, 1.0);
        let mut pen = Paint::new(Color4f::new(1.0, 0.0, 0.0, 1.0), None);
        pen.set_style(PaintStyle::Stroke);
        pen.set_stroke_width(2.0);
        pen.set_anti_alias(true);
        canvas.draw_path(&cross, &pen);
        if trial_type == TrialType::Saccade {
          // `draw_gaussian_target` (built with a fixed luminance of 100)
          // isn't ported; `draw_gaussian` (built from this trial's actual
          // `luminance_targ_per`) stands in for it.
          self.draw_gaussian(canvas, 1.0);
        }
      }
      State::HoldTarget | State::AcquireTarget => {
        if trial_type == TrialType::Catch {
          let mut pen = Paint::new(Color4f::new(1.0, 0.0, 0.0, off_opacity as f32), None);
          pen.set_style(PaintStyle::Stroke);
          pen.set_stroke_width(2.0);
          pen.set_anti_alias(true);
          canvas.draw_path(&cross, &pen);
        } else {
          if current_state == State::HoldTarget {
            if !hide_during_hold {
              self.draw_gaussian(canvas, off_luminance);
            }
          } else {
            self.draw_gaussian(canvas, off_luminance);
          }
        }
      }
      State::FailureSaccade => {
        // Ported from line 1148: only during the second half of the
        // ACQUIRE_TARGET failure penalty (`show_target` set back to `true`
        // partway through — see `run`) does the target reappear.
        if show_target && trial_type == TrialType::Saccade {
          // Same `draw_gaussian_target` substitution as PRESENT_TARGET's.
          self.draw_gaussian(canvas, 1.0);
        }
      }
      _ => {}
    }

    // Line 1151.
    let canvas_size = canvas.base_layer_size();
    canvas.draw_rect(
      Rect::from_xywh(
        canvas_size.width as f32 - 50.0,
        canvas_size.height as f32 - 50.0,
        500.0,
        500.0,
      ),
      &Paint::new(current_photodiode_static_square, None),
    );

    // Lines 1157-1209 (`with painter.masked(RenderOutput.OPERATOR):`):
    // drawn only in the operator view, via this render pass's `window`.
    if window == Window::Operator {
      canvas.draw_rect(
        Rect::from_xywh(0.0, 0.0, 465.0, 230.0),
        &Paint::new(Color4f::new(1.0, 1.0, 1.0, 1.0), None),
      );

      let mut shading_paint = Paint::new(Color4f::new(1.0, 1.0, 1.0, 128.0 / 255.0), None);
      shading_paint.set_anti_alias(true);
      if trial_type == TrialType::Saccade {
        canvas.draw_circle(
          (targetpos_pix_f.0 as f32, targetpos_pix_f.1 as f32),
          accpt_gaze_radius_pix as f32,
          &shading_paint,
        );
      }
      let (center_x, center_y) = setup.center_f;
      let center_radius = if trial_type == TrialType::Catch && current_state > State::Delay {
        accpt_gaze_radius_pix
      } else {
        accpt_fix_radius_pix
      };
      canvas.draw_circle(
        (center_x as f32, center_y as f32),
        center_radius as f32,
        &shading_paint,
      );

      draw_text(
        canvas,
        &format!("Trial_type={}", trial_type.as_str()),
        0.0,
        10.0,
        background_color_qt,
      );
      draw_text(
        canvas,
        &format!("{current_state:?}"),
        0.0,
        40.0,
        background_color_qt,
      );
      draw_text(
        canvas,
        &format!("Trial_num = {}", stats.trial_num),
        0.0,
        70.0,
        background_color_qt,
      );
      draw_text(
        canvas,
        &format!(
          "Catch_trial = {} / {} ({:.1}%)",
          stats.trial_catch_success_count, stats.trial_catch_count, stats.catch_success_rate
        ),
        0.0,
        100.0,
        background_color_qt,
      );
      draw_text(
        canvas,
        &format!(
          "Saccade_trial = {} / {} ({:.1}%)",
          stats.trial_saccade_success_count,
          stats.trial_saccade_count,
          stats.saccade_trial_success_rate
        ),
        0.0,
        130.0,
        background_color_qt,
      );
      draw_text(
        canvas,
        &format!("Saccade_abort_rate= {:.1}%", stats.saccade_trial_abort_rate),
        0.0,
        160.0,
        background_color_qt,
      );
      draw_text(
        canvas,
        &format!("{current_state:?}"),
        valid_gaze.0 as f32,
        valid_gaze.1 as f32,
        background_color_qt,
      );
      draw_text(
        canvas,
        &format!("Gaze (pix): x = {}, y = {}", valid_gaze.0, valid_gaze.1),
        0.0,
        190.0,
        background_color_qt,
      );

      draw_gaze(
        canvas,
        valid_gaze,
        Color4f::new(138.0 / 255.0, 43.0 / 255.0, 226.0 / 255.0, 1.0),
      );

      for &(point, color) in setup.gaze_success_store.lock().unwrap().iter() {
        draw_gaze(canvas, point, color);
      }
    }
  }

  /// A "Clear VCP" button that clears the persistent
  /// `gaze_success_store`/`gaze_failure_store` traces (see `VcpSetup`) drawn
  /// in the operator view.
  fn operator_widget(&self, ui: &imgui::Ui) {
    if ui.button("Clear VCP") {
      if let Some(state) = STATE.get() {
        state.gaze_success_store.lock().unwrap().clear();
        state.gaze_failure_store.lock().unwrap().clear();
      }
    }
  }
}
