/* Copyright 2026 The OpenXLA Authors. All Rights Reserved.

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

#ifndef XPROF_CONVERT_GRPC_METRICS_XPLANE_TO_GRPC_METRICS_H_
#define XPROF_CONVERT_GRPC_METRICS_XPLANE_TO_GRPC_METRICS_H_

#include <optional>
#include <string>

#include "absl/status/statusor.h"
#include "tsl/profiler/protobuf/xplane.pb.h"
#include "xprof/convert/repository.h"
#include "plugin/xprof/protobuf/topology.pb.h"

namespace xprof::grpc_metrics {

// Returns the system topology stored in the Task Environment plane of
// `xspace`, or std::nullopt if the profile does not have one.
std::optional<tensorflow::profiler::Topology> GetSystemTopology(
    const tensorflow::profiler::XSpace& xspace);

// Returns true if a TensorCore plane of `xspace` says that the programs use
// megacore.
bool UsesMegacore(const tensorflow::profiler::XSpace& xspace);

// Builds the JSON of the DCN Analyzer tool (grpc_metrics) for a session: reads
// the XSpace of each host, finishes each host, merges the hosts and prints the
// result. A host whose XSpace cannot be read is skipped.
absl::StatusOr<std::string> ConvertMultiXSpacesToGrpcMetricsJson(
    const tensorflow::profiler::SessionSnapshot& session_snapshot);

}  // namespace xprof::grpc_metrics

#endif  // XPROF_CONVERT_GRPC_METRICS_XPLANE_TO_GRPC_METRICS_H_
