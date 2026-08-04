use std::sync::{Arc, OnceLock};

use async_trait::async_trait;
use kira::sound::static_sound::StaticSoundData;

use crate::pb::task_controller_grpc::TaskResult;
use crate::pb::thalamus_grpc::{AnalogResponse, Span};

use super::{BehaviorTask, PointSubscription, TaskContext, Window};

use skia_safe::{
  Canvas, Color4f, Font, FontMgr, Paint, PaintStyle, Path, PathBuilder, Rect, Shader, TileMode,
};

pub struct Vcp2AfcTask {

}

struct Static {
  success_sound: StaticSoundData,
  abort_sound: StaticSoundData,
  failure_sound: StaticSoundData,

}
static STATIC: OnceLock<Static> = OnceLock::new();

impl Static {
  fn new() -> Self {
    let success_sound =
      StaticSoundData::from_file(r"C:\Thalamus-Extensions\seokhee\success_clip.wav").unwrap();
    let abort_sound =
      StaticSoundData::from_file(r"C:\Thalamus-Extensions\seokhee\failure_clip.wav").unwrap();
    let failure_sound =
      StaticSoundData::from_file(r"C:\Thalamus-Extensions\seokhee\timeout_failure.wav").unwrap();

    Static {
      success_sound, abort_sound, failure_sound
    }
  }
}

impl BehaviorTask for Vcp2AfcTask {
  async fn run(&self, context: Arc<TaskContext>) -> TaskResult {
    //*self.context.lock().unwrap() = Some(context.clone());
    let config = &context.config();
    let monitorsubj_w_pix = config.get("monitorsubj_W_pix").unwrap().as_i64().unwrap();
    let monitorsubj_h_pix = config.get("monitorsubj_H_pix").unwrap().as_i64().unwrap();
    let monitorsubj_dist_m = config.get("monitorsubj_dist_m").unwrap().as_i64().unwrap();
    let monitorsubj_width_m = config.get("monitorsubj_width_m").unwrap().as_i64().unwrap();
    let center = (monitorsubj_w_pix as i32 / 2, monitorsubj_h_pix as i32 / 2);

    let num_choices = config.get("num_choices").unwrap().as_i64().unwrap();

    let _static = STATIC.get_or_init(Static::new);



    //context.config()

    TaskResult { success: true, cancelled: false }
  }

  fn render(&self,canvas: &Canvas,window: Window) {
    todo!()
  }
}