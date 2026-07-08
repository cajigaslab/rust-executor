//! Nanosecond timestamps on the same monotonic clock the Thalamus server
//! reads via `std::chrono::steady_clock::now().time_since_epoch()` — QPC on
//! Windows (MSVC's `steady_clock`), `CLOCK_MONOTONIC` on Linux (libstdc++'s
//! `steady_clock`) — so values sent to it are directly comparable to its own.
//! `std::time::Instant` uses the same underlying primitives but doesn't
//! expose the raw counter, hence the small platform-specific reads below.

#[cfg(windows)]
pub fn now_ns() -> u64 {
  unsafe {
    let mut counter: i64 = 0;
    let mut frequency: i64 = 0;
    QueryPerformanceCounter(&mut counter);
    QueryPerformanceFrequency(&mut frequency);
    (counter as u128 * 1_000_000_000 / frequency as u128) as u64
  }
}

#[cfg(windows)]
unsafe extern "system" {
  fn QueryPerformanceCounter(count: *mut i64) -> i32;
  fn QueryPerformanceFrequency(frequency: *mut i64) -> i32;
}

#[cfg(unix)]
pub fn now_ns() -> u64 {
  unsafe {
    let mut ts: libc::timespec = std::mem::zeroed();
    libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut ts);
    ts.tv_sec as u64 * 1_000_000_000 + ts.tv_nsec as u64
  }
}
