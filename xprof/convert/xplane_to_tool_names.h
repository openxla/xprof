/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#ifndef XPROF_CONVERT_XPLANE_TO_TOOL_NAMES_H_
#define XPROF_CONVERT_XPLANE_TO_TOOL_NAMES_H_

#include <string>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xprof/convert/repository.h"
#include "xprof/convert/tool_options.h"

namespace tensorflow {
namespace profiler {

// Bool tool option that lists the DCN Analyzer tool (grpc_metrics) when the
// profile has its events. Off by default.
inline constexpr absl::string_view kEnableDcnAnalyzerOption =
    "enable_dcn_analyzer";

// Gets the names of the available tools given a session snapshot.
// Returns a comma separated list of tool names.
absl::StatusOr<std::string> GetAvailableToolNames(
    const SessionSnapshot& session_snapshot);

// Same as above, but `options` can turn on optional tools (see
// kEnableDcnAnalyzerOption).
absl::StatusOr<std::string> GetAvailableToolNames(
    const SessionSnapshot& session_snapshot, const ToolOptions& options);

}  // namespace profiler
}  // namespace tensorflow

#endif  // XPROF_CONVERT_XPLANE_TO_TOOL_NAMES_H_
