/* Copyright 2024 The OpenXLA Authors. All Rights Reserved.

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

#ifndef THIRD_PARTY_XPROF_CONVERT_XPLANE_TO_UTILIZATION_VIEWER_H_
#define THIRD_PARTY_XPROF_CONVERT_XPLANE_TO_UTILIZATION_VIEWER_H_

#include <string>

#include "absl/status/statusor.h"
#include "tsl/profiler/protobuf/xplane.pb.h"
#include "xprof/convert/tool_options.h"

namespace xprof {

// Converts an XSpace with performance counters to a JSON string formatted for
// the Utilization Viewer DataTable.
absl::StatusOr<std::string> ConvertXSpaceToUtilizationViewer(
    const tensorflow::profiler::XSpace& space);

// Converts an XSpace proto into a structured Kernel Utilization JSON report.
// Extracts hardware performance counters and computes subsystem utilizations,
// MXU precision breakdowns (BF16, Int8, Int4, FP8), and anomaly diagnostics.
// Supports microbenchmark duration normalization and per-kernel filtering.
absl::StatusOr<std::string> ConvertXSpaceToKernelUtilization(
    const tensorflow::profiler::XSpace& space,
    const tensorflow::profiler::ToolOptions& options = {});

}  // namespace xprof

#endif  // THIRD_PARTY_XPROF_CONVERT_XPLANE_TO_UTILIZATION_VIEWER_H_
