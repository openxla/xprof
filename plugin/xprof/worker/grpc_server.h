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

#ifndef THIRD_PARTY_XPROF_PLUGIN_TENSORBOARD_PLUGIN_PROFILE_WORKER_GRPC_SERVER_H_
#define THIRD_PARTY_XPROF_PLUGIN_TENSORBOARD_PLUGIN_PROFILE_WORKER_GRPC_SERVER_H_

#include <optional>
#include <string>

#include "absl/strings/string_view.h"

namespace xprof {
namespace profiler {

namespace internal {
std::string FormatGrpcServerAddress(absl::string_view host, int port);
// Starts (or replaces) the process-wide gRPC worker server and returns the
// bound listening port (> 0 on success, -1 if server startup fails). When
// `host_override` is set, uses it instead of reading `XPROF_WORKER_HOST`.
int InitializeGrpcServerForTesting(
    int port, int max_concurrent_requests, absl::string_view logdir = "",
    std::optional<absl::string_view> host_override = std::nullopt);
void ShutdownGrpcServerForTesting();
}  // namespace internal

// Starts (or replaces) the process-wide gRPC worker server on `port` with at
// most `max_concurrent_requests` concurrent `GetProfileData` executions.
// Binds to `127.0.0.1` by default, or to the host specified by the
// `XPROF_WORKER_HOST` environment variable (trimming ASCII whitespace and
// wrapping unbracketed IPv6 literals in square brackets). When `logdir` is
// non-empty, restricts worker profile requests to `logdir` (taking precedence
// over `XPROF_WORKER_LOGDIR`); when both `logdir` and `XPROF_WORKER_LOGDIR` are
// empty or unset, requests fall back to directories registered via
// `AddWorkerAllowedRootDir`.
// TODO: b/442301153 - Add ShutdownGrpcServer() as well.
void InitializeGrpcServer(int port, int max_concurrent_requests,
                          absl::string_view logdir = "");

}  // namespace profiler
}  // namespace xprof

#endif  // THIRD_PARTY_XPROF_PLUGIN_TENSORBOARD_PLUGIN_PROFILE_WORKER_GRPC_SERVER_H_
