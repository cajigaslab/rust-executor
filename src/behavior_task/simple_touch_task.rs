use std::io::Cursor;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, anyhow, bail, ensure};
use async_trait::async_trait;
use kira::sound::static_sound::StaticSoundData;
use serde_json::{Map, Value, json};
use skia_safe::{Canvas, Color4f, Paint, Rect};

use crate::pb::task_controller_grpc::TaskResult;
use crate::pb::thalamus_grpc::{AnalogResponse, Span};

use super::{BehaviorTask, PointSubscription, TaskContext, Window, wait_for, wait_for_hold};

/// Bundled rather than loaded from a path at runtime: tasks are constructed
/// once at startup (see `registry`), so a missing file would take the whole
/// executor down. Copied from Thalamus' task_controller directory.
static SUCCESS_SOUND_DATA: &[u8] = include_bytes!("../../assets/success_clip.wav");
static FAILURE_SOUND_DATA: &[u8] = include_bytes!("../../assets/failure_clip.wav");

/// Approximate eye-to-screen distance used by Python's `ecc_to_px`.
const VIEWING_DISTANCE_M: f64 = 0.4;
const STATE_INDICATOR_WIDTH: f32 = 70.0;

/// Ported from `ecc_to_px`: converts degrees of eccentricity to pixels
/// relative to the optical center.
fn ecc_to_px(ecc: f64, dpi: f64) -> f64 {
  let x_m = VIEWING_DISTANCE_M * ecc.to_radians().tan();
  x_m / 0.0254 * dpi
}

fn distance(a: (f64, f64), b: (f64, f64)) -> f64 {
  ((a.0 - b.0).powi(2) + (a.1 - b.1).powi(2)).sqrt()
}

fn reward_pulse(duration_ms: u64) -> AnalogResponse {
  AnalogResponse {
    data: vec![5.0, 0.0],
    spans: vec![Span {
      begin: 0,
      end: 2,
      name: "Reward".to_string(),
      ..Default::default()
    }],
    sample_intervals: vec![1_000_000 * duration_ms],
    ..Default::default()
  }
}

/// The trial's used values, sent to the TaskController for its summary.
type UsedValues = Map<String, Value>;

// The readers below port `TaskContext.get_value`/`get_target_value`/
// `get_target_color`: each records what it read under `name` in the used
// values, and is an error where Python raises, rather than guessing.

/// `config[key]`, else `default`: a number, or an object whose `min` and
/// `max` (each falling back to `default`'s) bound a uniform sample.
fn number(config: &Value, key: &str, default: Option<&Value>, name: &str, used: &mut UsedValues) -> anyhow::Result<f64> {
  let value = config.get(key).or(default).with_context(|| format!("{name} is missing"))?;
  let result = match value {
    Value::Number(_) => value.clone(),
    Value::Object(range) => {
      let bound = |b: &str| {
        range
          .get(b)
          .or_else(|| default.and_then(|d| d.get(b)))
          .and_then(Value::as_f64)
          .with_context(|| format!("{name} needs a numeric {b:?}, got {value}"))
      };
      let (lower, upper) = (bound("min")?, bound("max")?);
      // Python's `random.uniform`, which also allows `min > max`.
      Value::from(lower + (upper - lower) * rand::random::<f64>())
    }
    _ => bail!("{name} must be a number or {{min, max}}, got {value}"),
  };
  used.insert(name.to_string(), result.clone());
  Ok(result.as_f64().expect("a number"))
}

fn boolean(config: &Value, key: &str, name: &str, used: &mut UsedValues) -> anyhow::Result<bool> {
  let value = config
    .get(key)
    .and_then(Value::as_bool)
    .with_context(|| format!("{name} must be true or false, got {:?}", config.get(key)))?;
  used.insert(name.to_string(), Value::Bool(value));
  Ok(value)
}

fn string(config: &Value, key: &str, name: &str, used: &mut UsedValues) -> anyhow::Result<String> {
  let value = config
    .get(key)
    .and_then(Value::as_str)
    .with_context(|| format!("{name} must be a string, got {:?}", config.get(key)))?;
  used.insert(name.to_string(), Value::from(value));
  Ok(value.to_string())
}

/// An `[r, g, b]` list of 0-255 values, white if absent. Python truncates
/// each with `int`.
fn color(config: &Value, key: &str, name: &str, used: &mut UsedValues) -> anyhow::Result<Color4f> {
  let rgb = config.get(key).cloned().unwrap_or(json!([255, 255, 255]));
  let channel = |i: usize| {
    rgb
      .get(i)
      .and_then(Value::as_f64)
      .filter(|c| (0.0..256.0).contains(c))
      .map(|c| c.trunc() as f32 / 255.0)
      .ok_or_else(|| anyhow!("{name} must be [r, g, b] with each in 0-255, got {rgb}"))
  };
  let color = Color4f::new(channel(0)?, channel(1)?, channel(2)?, 1.0);
  used.insert(name.to_string(), rgb);
  Ok(color)
}

/// Python's `int(task_config[key])`, which isn't a used value.
fn required_int(config: &Value, key: &str) -> anyhow::Result<i32> {
  let value = config
    .get(key)
    .and_then(Value::as_f64)
    .with_context(|| format!("{key} must be a number, got {:?}", config.get(key)))?;
  Ok(value as i32)
}

struct Target {
  rect: Rect,
  center: (f64, f64),
  window: f64,
  color: Color4f,
  reward_channel: usize,
}

/// State touched by Python's `touch_handler`.
#[derive(Default)]
struct Touch {
  blank_space_touched: bool,
  presented_targ_acquired: bool,
  last_selected_target: Option<usize>,
}

#[derive(Default)]
struct Inner {
  targets: Vec<Target>,
  is_choice: bool,
  presented: usize,
  show_presented_target: bool,
  state_brightness: f32,
  state_indicator: (f32, f32),
  touch: Touch,
}

impl Inner {
  /// Ported from `touch_handler`.
  fn handle_touch(&mut self, cursor: (f64, f64)) {
    let targets = &self.targets;
    let in_window = |t: &Target| distance(t.center, cursor) < t.window;

    let any_in_window = targets.iter().any(in_window);
    self.touch.blank_space_touched |= !any_in_window && cursor.0 >= 0.0;

    let selected = if self.is_choice {
      targets.iter().position(in_window)
    } else {
      Some(self.presented).filter(|&i| in_window(&targets[i]))
    };
    self.touch.presented_targ_acquired = selected.is_some();
    if selected.is_some() {
      self.touch.last_selected_target = selected;
    }
  }
}

/// Ported from Thalamus' `simple_touch_task.py`: shows a target (or, in
/// choice mode, every target), waits for the subject to touch it — touching
/// blank space restarts from the intertrial interval — then requires the
/// touch to be held for `hold_timeout` (tolerating lapses up to
/// `blink_timeout`) before rewarding.
pub struct SimpleTouchTask {
  success_sound: StaticSoundData,
  failure_sound: StaticSoundData,
  inner: parking_lot::Mutex<Inner>,
}

impl SimpleTouchTask {
  pub fn new() -> Self {
    Self {
      success_sound: StaticSoundData::from_cursor(Cursor::new(SUCCESS_SOUND_DATA))
        .expect("bundled success_clip.wav should decode"),
      failure_sound: StaticSoundData::from_cursor(Cursor::new(FAILURE_SOUND_DATA))
        .expect("bundled failure_clip.wav should decode"),
      inner: parking_lot::Mutex::new(Inner::default()),
    }
  }

  /// Runs every touch that has arrived since the last call through
  /// `Inner::handle_touch`, then evaluates `predicate` on the result.
  fn touch_condition<'a>(
    &'a self,
    touches: &'a PointSubscription,
    predicate: impl Fn(&Touch) -> bool + 'a,
  ) -> impl Fn() -> bool + 'a {
    move || {
      let mut inner = self.inner.lock();
      for point in touches.drain() {
        inner.handle_touch(point);
      }
      predicate(&inner.touch)
    }
  }

  /// Sleeps for `duration` while still processing touches, as Python's
  /// `touch_handler` would.
  async fn sleep(&self, touches: &PointSubscription, duration: Duration) {
    wait_for(touches.notify(), self.touch_condition(touches, |_| false), Some(duration)).await;
  }
}

#[async_trait]
impl BehaviorTask for SimpleTouchTask {
  async fn run(&self, context: Arc<TaskContext>) -> anyhow::Result<TaskResult> {
    let config = context.config();
    let mut used = UsedValues::new();

    // Python's `RANDOM_DEFAULT`.
    let random_default = json!({"min": 1, "max": 1});
    let mut timeout = |key: &str| {
      let seconds = number(&config, key, Some(&random_default), key, &mut used)?;
      Duration::try_from_secs_f64(seconds)
        .map_err(|_| anyhow!("{key} must be a non-negative number of seconds, got {seconds}"))
    };
    let intertrial_timeout = timeout("intertrial_timeout")?;
    let start_timeout = timeout("start_timeout")?;
    let hold_timeout = timeout("hold_timeout")?;
    let blink_timeout = timeout("blink_timeout")?;
    let fail_timeout = timeout("fail_timeout")?;
    let success_timeout = timeout("success_timeout")?;
    let is_choice = boolean(&config, "is_choice", "is_choice", &mut used)?;
    let state_indicator = (
      required_int(&config, "state_indicator_x")? as f32,
      required_int(&config, "state_indicator_y")? as f32,
    );

    let dpi = context.dpi()?;
    let (canvas_width, canvas_height) = context.canvas_size();
    let canvas_center = (canvas_width as f64 / 2.0, canvas_height as f64 / 2.0);

    // Ported from `get_target_rectangles` plus the per-target lists built
    // alongside it in `run`.
    let target_configs = config
      .get("targets")
      .and_then(Value::as_array)
      .with_context(|| format!("targets must be a list, got {:?}", config.get("targets")))?;
    // Where Python's `np.random.randint(0)` raises.
    ensure!(!target_configs.is_empty(), "no targets configured");
    let mut targets = Vec::with_capacity(target_configs.len());
    for (i, target_config) in target_configs.iter().enumerate() {
      let name = |key: &str| format!("targ{i}_{key}");
      let mut value = |key: &str| number(target_config, key, None, &name(key), &mut used);
      let radius = value("radius")?;
      let angle = value("angle")?.to_radians();
      let width = ecc_to_px(value("width")?, dpi);
      let height = ecc_to_px(value("height")?, dpi);
      let window = ecc_to_px(value("window_size")?, dpi);
      let reward_channel = value("reward_channel")?;
      // Read only to be recorded, as in Python.
      value("on_luminance")?;
      value("off_luminance")?;
      ensure!(reward_channel >= 0.0, "target {i}'s reward_channel must not be negative, got {reward_channel}");
      let color = color(target_config, "color", &name("color"), &mut used)?;
      string(target_config, "name", &name("name"), &mut used)?;
      let stl_file = string(target_config, "stl_file", &name("stl_file"), &mut used)?;

      // Python applies `ecc_to_px` to each cartesian component, then flips y.
      let center = (
        canvas_center.0 + ecc_to_px(radius * angle.cos(), dpi),
        canvas_center.1 - ecc_to_px(radius * angle.sin(), dpi),
      );
      let rect = Rect::from_xywh(
        (center.0 - width / 2.0) as i32 as f32,
        (center.1 - height / 2.0) as i32 as f32,
        width as i32 as f32,
        height as i32 as f32,
      );

      if !stl_file.is_empty() {
        tracing::warn!("simple_touch_task: STL targets aren't supported, drawing target {i} as a rectangle");
      }

      targets.push(Target { rect, center, window, color, reward_channel: reward_channel as usize });
    }

    context.set_used_values(used);

    let presented = rand::random_range(0..targets.len());
    let mut behav_result = Map::new();
    behav_result.insert(
      "presented_target_ids".to_string(),
      json!(if is_choice { (0..targets.len()).collect() } else { vec![presented] }),
    );

    *self.inner.lock() = Inner {
      targets,
      is_choice,
      presented,
      state_indicator,
      ..Default::default()
    };

    let touches = context.subscribe_to_touch();

    loop {
      context.log("BehavState=intertrial").await;
      self.inner.lock().state_brightness = 0.0;
      self.sleep(&touches, intertrial_timeout).await;

      self.inner.lock().touch.blank_space_touched = false;
      context.log("BehavState=start_on").await;
      {
        let mut inner = self.inner.lock();
        inner.state_brightness = 1.0;
        inner.show_presented_target = true;
      }
      let acquired = wait_for(
        touches.notify(),
        self.touch_condition(&touches, |t| t.presented_targ_acquired || t.blank_space_touched),
        Some(start_timeout),
      )
      .await;

      let mut inner = self.inner.lock();
      if acquired && !inner.touch.blank_space_touched {
        break;
      }
      inner.show_presented_target = false;
    }

    // state: startacq
    let selected = self.inner.lock().touch.last_selected_target;
    behav_result.insert("selected_target_id".to_string(), json!(selected));
    context.log("BehavState=start_acq").await;
    let success = wait_for_hold(
      touches.notify(),
      self.touch_condition(&touches, |t| t.presented_targ_acquired),
      hold_timeout,
      Some(blink_timeout),
      false,
    )
    .await;

    if !success {
      context.log("BehavState=fail").await;
      context.set_behav_result(behav_result).await?;
      // Python's `fail_trial` assigns these without `nonlocal`, so there they
      // never take effect and the target stays up through `fail_timeout`;
      // this does what that code evidently meant to.
      {
        let mut inner = self.inner.lock();
        inner.show_presented_target = false;
        inner.state_brightness = 0.0;
      }
      context.play_sound(self.failure_sound.clone());
      self.sleep(&touches, fail_timeout).await;
      return Ok(TaskResult { success: false, ..Default::default() });
    }

    self.inner.lock().show_presented_target = false;
    context.log("BehavState=success").await;
    self.inner.lock().state_brightness = 0.0;

    // `selected` is always set here: the start_on loop only exits once a
    // target has been acquired.
    let selected = selected.expect("start_on exits only after a target is acquired");
    let reward_channel = self.inner.lock().targets[selected].reward_channel;
    let reward = context.get_reward(reward_channel)?;
    context.inject_analog("Reward", reward_pulse(reward as u64)).await;

    context.play_sound(self.success_sound.clone());
    self.sleep(&touches, success_timeout).await;

    context.set_behav_result(behav_result).await?;
    Ok(TaskResult { success: true, ..Default::default() })
  }

  fn render(&self, canvas: &Canvas, window: Window) {
    let inner = self.inner.lock();

    if inner.show_presented_target {
      let shown: Vec<&Target> = if inner.is_choice {
        inner.targets.iter().collect()
      } else {
        inner.targets.get(inner.presented).into_iter().collect()
      };
      for target in shown {
        canvas.draw_rect(target.rect, &Paint::new(target.color, None));
      }
    }

    // Python draws these under `painter.masked(RenderOutput.OPERATOR)`,
    // all with the first target's window size; each target's own size is
    // what `touch_handler` actually checks against, so that's drawn here.
    if window == Window::Operator {
      let mut paint = Paint::new(Color4f::new(1.0, 1.0, 1.0, 0.5), None);
      paint.set_anti_alias(true);
      for target in &inner.targets {
        canvas.draw_circle(
          (target.center.0 as f32, target.center.1 as f32),
          target.window as f32,
          &paint,
        );
      }
    }

    let b = inner.state_brightness;
    let (x, y) = inner.state_indicator;
    canvas.draw_rect(
      Rect::from_xywh(x, y, STATE_INDICATOR_WIDTH, STATE_INDICATOR_WIDTH),
      &Paint::new(Color4f::new(b, b, b, 1.0), None),
    );
  }
}
