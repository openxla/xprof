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

#include "xprof/convert/base_hlo_processor.h"

#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/platform/statusor.h"
#include "xprof/convert/repository.h"
#include "xprof/convert/tool_options.h"
#include "xprof/convert/unified_session_snapshot.h"
#include "xprof/convert/xplane_to_hlo.h"

namespace xprof {

absl::StatusOr<std::string> BaseHloProcessor::Map(
    const XprofSessionSnapshot& session_snapshot, absl::string_view hostname,
    const tensorflow::profiler::XSpace& xspace) {
  return absl::UnimplementedError("Map not implemented for BaseHloProcessor");
}

absl::Status BaseHloProcessor::Reduce(
    const XprofSessionSnapshot& session_snapshot,
    const std::vector<std::string>& map_output_files) {
  return absl::UnimplementedError(
      "Reduce not implemented for BaseHloProcessor");
}

absl::Status BaseHloProcessor::ProcessSession(
    const XprofSessionSnapshot& session_snapshot,
    const tensorflow::profiler::ToolOptions& options) {
  const tensorflow::profiler::SessionSnapshot* profiler_session_snapshot =
      dynamic_cast<const tensorflow::profiler::SessionSnapshot*>(
          &session_snapshot);
  if (!profiler_session_snapshot) {
    return absl::InvalidArgumentError(
        "session_snapshot is not a tensorflow::profiler::SessionSnapshot");
  }

  TF_ASSIGN_OR_RETURN(xla::HloProto hlo_proto,
                      tensorflow::profiler::GetHloProtoByOptions(
                          *profiler_session_snapshot, options));

  return ProcessHlo(session_snapshot, hlo_proto, options);
}

}  // namespace xprof
