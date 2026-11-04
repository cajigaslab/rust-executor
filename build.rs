use std::process::Command;

/// `git` with `args`, its trimmed output if it succeeded.
fn git(args: &[&str]) -> Option<String> {
  let output = Command::new("git").args(args).output().ok()?;
  output.status.success().then(|| String::from_utf8_lossy(&output.stdout).trim().to_string())
}

/// Embeds the commit being built as `EXECUTOR_COMMIT`, with `-dirty` if
/// tracked files have uncommitted changes, or `unknown` without git.
fn embed_commit() {
  let commit = git(&["rev-parse", "HEAD"]).unwrap_or_else(|| "unknown".to_string());
  let dirty = git(&["status", "--porcelain", "--untracked-files=no"]).is_some_and(|s| !s.is_empty());
  println!("cargo:rustc-env=EXECUTOR_COMMIT={commit}{}", if dirty { "-dirty" } else { "" });

  // Rerun when the commit or the working tree changes.
  if let Some(git_dir) = git(&["rev-parse", "--absolute-git-dir"]) {
    for file in ["HEAD", "index", "refs", "packed-refs"] {
      println!("cargo:rerun-if-changed={git_dir}/{file}");
    }
  }
  println!("cargo:rerun-if-changed=src");
  println!("cargo:rerun-if-changed=Cargo.toml");
}

fn main() {
  embed_commit();

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
