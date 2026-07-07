fn main() {
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
