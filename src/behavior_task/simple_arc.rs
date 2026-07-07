use std::sync::Mutex;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use skia_safe::{Canvas, Color4f, Paint, PaintStyle, Path, PathBuilder, Rect};

use super::BehaviorTask;

/// How long the arc takes to grow from empty to a full circle.
const GROW_DURATION: Duration = Duration::from_secs(1);

/// The "simple" task's behavior: draws a filled arc in the subject view that
/// grows from empty to a full circle over one second, then reports success.
pub struct SimpleArcTask {
    started_at: Mutex<Option<Instant>>,
}

impl SimpleArcTask {
    pub fn new() -> Self {
        Self {
            started_at: Mutex::new(None),
        }
    }

    fn progress(&self) -> f32 {
        match *self.started_at.lock().unwrap() {
            None => 0.0,
            Some(started_at) => {
                (started_at.elapsed().as_secs_f32() / GROW_DURATION.as_secs_f32()).min(1.0)
            }
        }
    }
}

#[async_trait]
impl BehaviorTask for SimpleArcTask {
    async fn run(&self) {
        *self.started_at.lock().unwrap() = Some(Instant::now());
        tokio::time::sleep(GROW_DURATION).await;
    }

    fn render(&self, canvas: &Canvas) {
        let progress = self.progress();
        if progress <= 0.0 {
            return;
        }

        let size = canvas.base_layer_size();
        let (width, height) = (size.width as f32, size.height as f32);
        let center = (width / 2.0, height / 2.0);
        let radius = width.min(height) * 0.5 * 0.8;

        let mut paint = Paint::new(Color4f::new(90.0 / 255.0, 170.0 / 255.0, 250.0 / 255.0, 1.0), None);
        paint.set_anti_alias(true);
        paint.set_style(PaintStyle::Fill);

        let path: Path = if progress >= 1.0 {
            Path::circle(center, radius, None)
        } else {
            let oval = Rect::from_xywh(center.0 - radius, center.1 - radius, radius * 2.0, radius * 2.0);
            let mut builder = PathBuilder::new();
            builder.move_to(center);
            builder.arc_to(oval, -90.0, progress * 360.0, false);
            builder.close();
            builder.detach()
        };

        canvas.draw_path(&path, &paint);
    }
}
