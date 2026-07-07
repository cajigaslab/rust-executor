/// The fixed canvas size `BehaviorTask::render` draws into (via
/// `canvas.base_layer_size()`, backed by `gfx::OFFSCREEN_EXTENT`) and that
/// touch/gaze input is rescaled into from window/device-native coordinates,
/// so tasks can treat input and rendering as the same coordinate space
/// regardless of the subject window's actual current size (e.g. windowed vs.
/// fullscreen on a differently-sized monitor).
pub const WIDTH: u32 = 1280;
pub const HEIGHT: u32 = 720;
