use crate::pb::thalamus_grpc::AnalogResponse;

/// The last raw sample in the named span (whichever of `data`/`int_data`/
/// `ulong_data` is populated). Applies the span's `scale`/`offset`
/// (`raw * scale + offset`) only if `response.is_transformed` — otherwise
/// `scale`/`offset` are unset (both `0.0`) and the raw sample is already the
/// real value, so applying them would zero it out. `None` if no span with
/// that name is present, or its range is empty.
pub fn last_span_value(response: &AnalogResponse, name: &str) -> Option<f64> {
  let span = response.spans.iter().find(|span| span.name == name)?;
  let raw = last_raw_sample(response, span.begin as usize, span.end as usize)?;
  if response.is_transformed {
    Some(raw * span.scale + span.offset)
  } else {
    Some(raw)
  }
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
