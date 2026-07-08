/// The subject window's initial size — both the size it's actually created
/// at (see `gfx::Graphics::new`) and the placeholder
/// `touch_screen::SharedWindowSize` holds before the first real size is
/// polled. `BehaviorTask::render` no longer draws into a fixed canvas of
/// this size — the offscreen targets (and so `canvas.base_layer_size()`,
/// and the space touch/gaze input is reported in) track the subject
/// window's actual current size instead (see
/// `gfx::Graphics::resize_offscreen_targets_if_needed`).
pub const WIDTH: u32 = 1280;
pub const HEIGHT: u32 = 720;
