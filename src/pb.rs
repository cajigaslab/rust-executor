#[allow(warnings)]
pub mod util_grpc {
  tonic::include_proto!("util_grpc");
}

#[allow(warnings)]
pub mod thalamus_grpc {
  tonic::include_proto!("thalamus_grpc");
}

#[allow(warnings)]
pub mod task_controller_grpc {
  tonic::include_proto!("task_controller_grpc");
}
