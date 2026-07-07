use std::sync::{Arc, Mutex};

use crate::behavior_task::SharedTask;
use crate::pb::thalamus_grpc::thalamus_client::ThalamusClient;
use crate::pb::thalamus_grpc::{AnalogRequest, AnalogResponse, NodeSelector};

const TOUCH_SCREEN_NODE_TYPE: &str = "TOUCH_SCREEN";

/// The subject window's current position (top-left of its content area, in
/// physical screen pixels), kept up to date by the graphics/render side (see
/// `gfx::Graphics::render_frame`) so touch points here can be translated from
/// screen coordinates to subject-window-local coordinates.
pub type SharedWindowPosition = Arc<Mutex<(i32, i32)>>;

pub fn shared_window_position() -> SharedWindowPosition {
    Arc::new(Mutex::new((0, 0)))
}

/// Streams touch input from Thalamus's `analog` RPC for the TOUCH_SCREEN node
/// and forwards each touch point — translated from screen coordinates to
/// coordinates relative to the subject window — to whichever `BehaviorTask` is
/// currently running (if any).
pub async fn run(addr: String, current_task: SharedTask, window_position: SharedWindowPosition) -> anyhow::Result<()> {
    let mut client = ThalamusClient::connect(addr.clone()).await?;

    let request = AnalogRequest {
        node: Some(NodeSelector {
            name: String::new(),
            r#type: TOUCH_SCREEN_NODE_TYPE.to_string(),
        }),
        channels: Vec::new(),
        channel_names: Vec::new(),
    };

    tracing::info!("connecting to TOUCH_SCREEN analog stream at {addr}");
    let response = client.analog(request).await?;
    let mut inbound = response.into_inner();

    while let Some(analog) = inbound.message().await? {
        let (Some(x), Some(y)) = (last_span_value(&analog, "X"), last_span_value(&analog, "Y")) else {
            // Either component is missing from this message: skip it.
            continue;
        };

        let (window_x, window_y) = *window_position.lock().unwrap();
        let local_x = x.round() as i32 - window_x;
        let local_y = y.round() as i32 - window_y;

        if let Some(task) = current_task.lock().unwrap().as_ref() {
            task.on_touch(local_x, local_y);
        }
    }

    Ok(())
}

/// The last raw sample in the named span (whichever of `data`/`int_data`/
/// `ulong_data` is populated), transformed by that span's `scale`/`offset`
/// (`raw * scale + offset`). `None` if no span with that name is present, or
/// its range is empty.
fn last_span_value(response: &AnalogResponse, name: &str) -> Option<f64> {
    let span = response.spans.iter().find(|span| span.name == name)?;
    let raw = last_raw_sample(response, span.begin as usize, span.end as usize)?;
    Some(raw * span.scale + span.offset)
}

fn last_raw_sample(response: &AnalogResponse, begin: usize, end: usize) -> Option<f64> {
    let last_index = end.checked_sub(1)?;
    if last_index < begin {
        return None;
    }
    if response.is_int_data {
        response.int_data.get(last_index).map(|v| *v as f64)
    } else if response.is_ulong_data {
        response.ulong_data.get(last_index).map(|v| *v as f64)
    } else {
        response.data.get(last_index).copied()
    }
}
