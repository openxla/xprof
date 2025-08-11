#ifndef THIRD_PARTY_XPROF_PLUGIN_TENSORBOARD_PLUGIN_PROFILE_WORKER_GRPC_UTILS_H_
#define THIRD_PARTY_XPROF_PLUGIN_TENSORBOARD_PLUGIN_PROFILE_WORKER_GRPC_UTILS_H_

#include "absl/status/status.h"
#include "grpcpp/support/status.h"

namespace xprof {
namespace profiler {

// Converts a grpc::Status to an absl::Status.
absl::Status ToAbslStatus(const grpc::Status& grpc_status);
grpc::Status ToGrpcStatus(const absl::Status& absl_status);

}  // namespace profiler
}  // namespace xprof

#endif  // THIRD_PARTY_XPROF_PLUGIN_TENSORBOARD_PLUGIN_PROFILE_WORKER_GRPC_UTILS_H_
