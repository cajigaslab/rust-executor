use std::process::Command;

/// `git` with `args`, its output if it succeeded.
fn git_raw(args: &[&str]) -> Option<String> {
  let output = Command::new("git").args(args).output().ok()?;
  output.status.success().then(|| String::from_utf8_lossy(&output.stdout).into_owned())
}

/// `git` with `args`, its trimmed output if it succeeded.
fn git(args: &[&str]) -> Option<String> {
  git_raw(args).map(|s| s.trim().to_string())
}

/// `url` without any credentials in it (`https://user:token@host/...`), so
/// they aren't logged.
fn without_credentials(url: &str) -> String {
  let Some((scheme, rest)) = url.split_once("://") else {
    return url.to_string();
  };
  let authority_end = rest.find('/').unwrap_or(rest.len());
  match rest[..authority_end].rfind('@') {
    Some(at) => format!("{scheme}://{}", &rest[at + 1..]),
    None => url.to_string(),
  }
}

/// Embeds what's being built, for `build_info`:
/// - `EXECUTOR_COMMIT`: the commit, with `-dirty` if tracked files have
///   uncommitted changes, or `unknown` without git.
/// - `EXECUTOR_REMOTE`: the `origin` remote's URL, or `unknown` if there
///   isn't one.
/// - `$OUT_DIR/uncommitted.diff`: `git diff HEAD`, the uncommitted changes to
///   tracked files, empty if there are none.
fn embed_git_info() {
  let commit = git(&["rev-parse", "HEAD"]).unwrap_or_else(|| "unknown".to_string());
  let dirty = git(&["status", "--porcelain", "--untracked-files=no"]).is_some_and(|s| !s.is_empty());
  println!("cargo:rustc-env=EXECUTOR_COMMIT={commit}{}", if dirty { "-dirty" } else { "" });

  let remote = git(&["remote", "get-url", "origin"])
    .map(|url| without_credentials(&url))
    .unwrap_or_else(|| "unknown".to_string());
  println!("cargo:rustc-env=EXECUTOR_REMOTE={remote}");

  let diff = if dirty { git_raw(&["diff", "HEAD"]).unwrap_or_default() } else { String::new() };
  let out_dir = std::env::var("OUT_DIR").expect("cargo sets OUT_DIR");
  std::fs::write(format!("{out_dir}/uncommitted.diff"), diff).expect("failed to write uncommitted.diff");

  // Rerun when the commit, the remote or any tracked file changes.
  if let Some(git_dir) = git(&["rev-parse", "--absolute-git-dir"]) {
    for file in ["HEAD", "index", "refs", "packed-refs", "config"] {
      println!("cargo:rerun-if-changed={git_dir}/{file}");
    }
  }
  // Not deleted ones: cargo reruns every build for a missing file.
  let tracked = git(&["ls-files"]).unwrap_or_default();
  for file in tracked.lines().filter(|file| std::path::Path::new(file).exists()) {
    println!("cargo:rerun-if-changed={file}");
  }
}

fn main() {
  embed_git_info();

  let proto_dir = "proto";
  let protos = [
    format!("{proto_dir}/thalamus.proto"),
    format!("{proto_dir}/task_controller.proto"),
  ];

  println!("cargo:rerun-if-changed={proto_dir}/thalamus.proto");
  println!("cargo:rerun-if-changed={proto_dir}/task_controller.proto");
  println!("cargo:rerun-if-changed={proto_dir}/util.proto");

  let fds = protox::compile(protos, [proto_dir]).expect("failed to compile proto files");

  tonic_prost_build::configure()
    .build_server(false)
    .compile_fds(fds)
    .expect("failed to generate tonic client code");
}
