/* Copyright 2025 The OpenXLA Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

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
