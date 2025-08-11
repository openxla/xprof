#include "plugin/xprof/worker/grpc_utils.h"

#include <string>

#include "absl/status/status.h"
#include "grpcpp/support/status.h"

namespace xprof {
namespace profiler {

absl::Status ToAbslStatus(const grpc::Status& grpc_status) {
  return absl::Status(static_cast<absl::StatusCode>(grpc_status.error_code()),
                      grpc_status.error_message());
}

grpc::Status ToGrpcStatus(const absl::Status& absl_status) {
  return grpc::Status(static_cast<grpc::StatusCode>(absl_status.code()),
                      std::string(absl_status.message()));
}

}  // namespace profiler
}  // namespace xprof
