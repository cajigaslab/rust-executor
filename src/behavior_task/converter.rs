use num_traits::{Float, FloatConst};

//pub fn rad_to_deg<F: Float + FloatConst>(rad: F) -> F {
//  rad * F::from(180.0).unwrap() / F::PI()
//}

pub fn deg_to_rad<F: Float + FloatConst>(rad: F) -> F {
  rad * F::PI() / F::from(180.0).unwrap()
}

pub struct Converter {
  deg_per_pixel: f64,
  pub screen_pixels: (f64, f64),
  pub center: (i32, i32)
}

impl Converter {
  pub fn new(screen_pixels: (i32, i32), screen_width_m: f64, screen_distance_m: f64) -> Self {
    let screen_width_rad = 2.0 * (screen_width_m / 2.0).atan2(screen_distance_m);
    let rad_per_pixel = screen_width_rad / screen_pixels.0 as f64;
    let deg_per_pixel = 180.0 / std::f64::consts::PI * rad_per_pixel;
    Self {
      deg_per_pixel,
      screen_pixels: (screen_pixels.0 as f64, screen_pixels.1 as f64),
      center: (screen_pixels.0/2, screen_pixels.1/2)
    }
  }

  pub fn from_config(config: &serde_json::Value) -> Self {
    let monitorsubj_w_pix: i32 = config["monitorsubj_W_pix"].as_i64().unwrap().try_into().unwrap();
    let monitorsubj_h_pix: i32 = config["monitorsubj_H_pix"].as_i64().unwrap().try_into().unwrap();
    let monitorsubj_dist_m = config["monitorsubj_dist_m"].as_f64().unwrap();
    let monitorsubj_width_m = config["monitorsubj_width_m"].as_f64().unwrap();
    Converter::new(
      (monitorsubj_w_pix, monitorsubj_h_pix),
      monitorsubj_width_m,
      monitorsubj_dist_m,
    )
  }

  pub fn deg_to_pixel_rel(&self, deg: f64) -> f64 {
    deg / self.deg_per_pixel
  }

  pub fn deg_to_pixel_rel_xy(&self, x_deg: f64, y_deg: f64) -> (f64, f64) {
    (x_deg / self.deg_per_pixel, y_deg / self.deg_per_pixel)
  }

  pub fn deg_to_pixel_abs(&self, x_deg: f64, y_deg: f64) -> (f64, f64) {
    let (x, y) = self.deg_to_pixel_rel_xy(x_deg, y_deg);
    (
      x + self.screen_pixels.0 / 2.0,
      y + self.screen_pixels.1 / 2.0,
    )
  }

  //pub fn relpix_to_absdeg(&self, x_pix: f64, y_pix: f64) -> (f64, f64) {
  //  (
  //    (x_pix + self.screen_pixels.0 / 2.0) * self.deg_per_pixel,
  //    (y_pix + self.screen_pixels.1 / 2.0) * self.deg_per_pixel,
  //  )
  //}

  //pub fn relpix_to_reldeg(&self, x_pix: f64, y_pix: f64) -> (f64, f64) {
  //  (
  //    (x_pix - self.screen_pixels.0 / 2.0) * self.deg_per_pixel,
  //    (y_pix - self.screen_pixels.1 / 2.0) * self.deg_per_pixel,
  //  )
  //}
}
