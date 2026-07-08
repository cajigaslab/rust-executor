use std::sync::{Arc, Mutex};

use serde_json::Value;
use tonic::transport::Channel;

use crate::analog::last_span_value;
use crate::behavior_task::SharedTask;
use crate::pb::thalamus_grpc::thalamus_client::ThalamusClient;
use crate::pb::thalamus_grpc::{AnalogRequest, NodeSelector};
use crate::touch_screen::{PointRingBuffer, SharedWindowSize};

const OCULOMATIC_NODE_TYPE: &str = "OCULOMATIC";

/// Gaze points received since the last periodic clear (see
/// `gfx::Graphics::render_frame`, which clears it alongside the touch path),
/// in operator-canvas-local coordinates. Rendered as a path of dots in the
/// operator view only. Bounded the same way as the touch path — see
/// [`PointRingBuffer`].
pub type SharedGazePath = Arc<Mutex<PointRingBuffer>>;

pub fn shared_gaze_path() -> SharedGazePath {
  Arc::new(Mutex::new(PointRingBuffer::new()))
}

/// One calibration pin from the "Angular Scaling" eye-tracking model: an eye
/// angle/screen rotation pair, plus the eye-magnitude -> screen-magnitude
/// notches used to interpolate scale at that angle. Mirrors
/// `AngularScalingConfig` in Thalamus's `task_controller/canvas.py`.
struct Pin {
  /// Eye angle, radians (`pin_angles[:, 0]` in the Python source).
  eye_angle: f64,
  /// Screen rotation, radians (`pin_angles[:, 1]`).
  rotation: f64,
  /// (eye magnitude, screen magnitude) pairs, ascending by eye magnitude,
  /// always starting with `(0.0, 0.0)` (`pin_notches` in the Python source).
  notches: Vec<(f64, f64)>,
}

/// The parsed "Angular Scaling" model: calibration pins plus the flat
/// fallback scale used when no pins are configured.
pub struct AngularScaling {
  pins: Vec<Pin>,
  scale_default: f64,
}

/// Cached, already-parsed `['eye_scaling']['Models']['Angular Scaling']`
/// config, refreshed only when an `observable_bridge_v2` change touches that
/// subtree (see `main::apply_transaction`) rather than re-parsed on every
/// gaze sample.
pub type SharedAngularScaling = Arc<Mutex<AngularScaling>>;

pub fn shared_angular_scaling() -> SharedAngularScaling {
  Arc::new(Mutex::new(AngularScaling {
    pins: Vec::new(),
    scale_default: 100.0,
  }))
}

/// Re-parses `['eye_scaling']['Models']['Angular Scaling']` out of `state`
/// (the full observable config tree) and stores it in `cache`. Called from
/// `main::apply_transaction` whenever a change touches that subtree.
pub fn refresh_angular_scaling(state: &Value, cache: &SharedAngularScaling) {
  let model = &state["eye_scaling"]["Models"]["Angular Scaling"];
  let scale_default = model["Scale Default"].as_f64().unwrap_or(100.0);

  let mut pins = Vec::new();
  if let Some(pin_values) = model["Pins"].as_array() {
    for pin in pin_values {
      let mut notches = vec![(0.0, 0.0)];
      if let Some(notch_values) = pin["Notches"].as_array() {
        for notch in notch_values {
          notches.push((
            notch["Eye"].as_f64().unwrap_or(0.0),
            notch["Screen"].as_f64().unwrap_or(0.0),
          ));
        }
      }
      pins.push(Pin {
        eye_angle: pin["Angle"].as_f64().unwrap_or(0.0),
        rotation: pin["Rotation"].as_f64().unwrap_or(0.0),
        notches,
      });
    }
  }
  *cache.lock().unwrap() = AngularScaling {
    pins,
    scale_default,
  };
}

/// Piecewise-linear interpolation matching `numpy.interp`: `xp` must be
/// sorted ascending; queries outside `xp`'s range clamp to the nearest
/// endpoint's `fp` value.
fn interp(x: f64, xp: &[f64], fp: &[f64]) -> f64 {
  if x <= xp[0] {
    return fp[0];
  }
  if x >= *xp.last().unwrap() {
    return *fp.last().unwrap();
  }
  for i in 0..xp.len() - 1 {
    if x <= xp[i + 1] {
      let t = (x - xp[i]) / (xp[i + 1] - xp[i]);
      return fp[i] + t * (fp[i + 1] - fp[i]);
    }
  }
  *fp.last().unwrap()
}

/// Interpolates the eye-magnitude -> screen-magnitude scale for `mag` from
/// one pin's notches, extrapolating past the last notch using the slope of
/// its last two points (as `AngularScalingConfig.process` does) rather than
/// clamping to it.
fn notch_scale(notches: &[(f64, f64)], mag: f64) -> f64 {
  let (last_eye, last_screen) = notches[notches.len() - 1];
  if mag > last_eye {
    let (prev_eye, prev_screen) = notches[notches.len() - 2];
    (mag - last_eye) * (last_screen - prev_screen) / (last_eye - prev_eye) + last_screen
  } else {
    let eyes: Vec<f64> = notches.iter().map(|n| n.0).collect();
    let screens: Vec<f64> = notches.iter().map(|n| n.1).collect();
    interp(mag, &eyes, &screens)
  }
}

/// Ports `AngularScalingConfig.process` (see `task_controller/canvas.py` in
/// the Thalamus repo): converts a raw `(x, y)` gaze reading into a point
/// relative to screen center, in pixels. With no calibration pins configured,
/// falls back to a flat `scale_default` multiply.
fn angular_scaling_process(x: f64, y: f64, pins: &[Pin], scale_default: f64) -> (f64, f64) {
  if pins.is_empty() {
    return (scale_default * x, scale_default * y);
  }

  let two_pi = 2.0 * std::f64::consts::PI;
  let mut val = y.atan2(x);
  if val < 0.0 {
    val += two_pi;
  }

  let mag = (x * x + y * y).sqrt();
  if mag == 0.0 {
    return (0.0, 0.0);
  }

  let eye_angles: Vec<f64> = pins.iter().map(|p| p.eye_angle).collect();
  let screen_angles: Vec<f64> = pins.iter().map(|p| p.rotation).collect();
  // `bisect_left`: leftmost index at which `val` could be inserted to keep
  // `eye_angles` sorted.
  let i = eye_angles.partition_point(|&angle| angle < val);

  let (lower_i, upper_i, eye_angle_lower) = if i == pins.len() || i == 0 {
    if val > std::f64::consts::PI {
      val -= two_pi;
    }
    (pins.len() - 1, 0, eye_angles[pins.len() - 1] - two_pi)
  } else if pins.len() == 1 {
    (0, 0, eye_angles[0])
  } else {
    (i - 1, i, eye_angles[i - 1])
  };

  let lower_scale = notch_scale(&pins[lower_i].notches, mag);
  let upper_scale = notch_scale(&pins[upper_i].notches, mag);

  let scale = interp(
    val,
    &[eye_angle_lower, eye_angles[upper_i]],
    &[lower_scale, upper_scale],
  );
  let mut rotation = interp(
    val,
    &[eye_angle_lower, eye_angles[upper_i]],
    &[screen_angles[lower_i], screen_angles[upper_i]],
  );
  if rotation > std::f64::consts::PI {
    rotation -= two_pi;
  } else if rotation < -std::f64::consts::PI {
    rotation += two_pi;
  }

  let (cos, sin) = (rotation.cos(), rotation.sin());
  let new_x = scale * (x * cos - y * sin) / mag;
  let new_y = scale * (x * sin + y * cos) / mag;
  (new_x, new_y)
}

/// Streams eye-tracking input from Thalamus's `analog` RPC for the
/// OCULOMATIC node over `client` (shared with `touch_screen::run`, since both
/// just hit the same service's `analog` RPC for different node types),
/// converts each raw reading to window-local pixel coordinates via
/// `angular_scaling_process` (the same algorithm as Thalamus's own
/// `AngularScalingConfig.process`, which reports gaze relative to screen
/// center) plus `window_size` (to locate that center in the subject window's
/// current — possibly since-resized — size), appends it to `gaze_path` for
/// the operator view's gaze overlay, and forwards it to whichever
/// `BehaviorTask` is currently running (if any), mirroring `touch_screen::run`.
pub async fn run(
  mut client: ThalamusClient<Channel>,
  angular_scaling: SharedAngularScaling,
  gaze_path: SharedGazePath,
  current_task: SharedTask,
  window_size: SharedWindowSize,
) -> anyhow::Result<()> {
  let request = AnalogRequest {
    node: Some(NodeSelector {
      name: String::new(),
      r#type: OCULOMATIC_NODE_TYPE.to_string(),
    }),
    channels: Vec::new(),
    channel_names: Vec::new(),
  };

  tracing::info!("connecting to OCULOMATIC analog stream");
  let response = client.analog(request).await?;
  let mut inbound = response.into_inner();

  while let Some(analog) = inbound.message().await? {
    let (Some(x), Some(y)) = (last_span_value(&analog, "X"), last_span_value(&analog, "Y")) else {
      // Either component is missing from this message: skip it.
      continue;
    };

    // Thalamus flips Y before scaling (see `Canvas.on_ros_gaze`).
    let (scaled_x, scaled_y) = {
      let cache = angular_scaling.lock().unwrap();
      angular_scaling_process(x, -y, &cache.pins, cache.scale_default)
    };

    let (window_width, window_height) = *window_size.lock().unwrap();
    let local_x = (scaled_x + window_width as f64 / 2.0).round() as i32;
    let local_y = (scaled_y + window_height as f64 / 2.0).round() as i32;

    if let Some(task) = current_task.lock().unwrap().as_ref() {
      task.on_gaze(local_x, local_y);
    }
    gaze_path.lock().unwrap().push((local_x, local_y));
  }

  Ok(())
}
