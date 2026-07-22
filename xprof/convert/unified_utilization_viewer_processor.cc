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

#include "xprof/convert/unified_utilization_viewer_processor.h"

#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "google/protobuf/arena.h"
#include "tsl/profiler/protobuf/xplane.pb.h"
#include "xprof/convert/tool_options.h"
#include "xprof/convert/unified_session_snapshot.h"
#include "xprof/convert/xplane_to_utilization_viewer.h"

namespace xprof {

absl::Status UnifiedUtilizationViewerProcessor::ProcessSession(
    const XprofSessionSnapshot& session_snapshot,
    const tensorflow::profiler::ToolOptions& options) {
  // TODO(b/something): Support multiple hosts properly if needed,
  // or unify the data aggregation logic between 1P and 3P here.
  // For now, process the first host's XSpace to establish the structure.
  if (session_snapshot.XSpaceSize() == 0) {
    return absl::NotFoundError("No XSpace found in the session.");
  }

  google::protobuf::Arena arena;
  absl::StatusOr<tensorflow::profiler::XSpace*> xspace =
      session_snapshot.GetXSpace(0, &arena);
  if (!xspace.ok()) {
    return xspace.status();
  }

  absl::StatusOr<std::string> json_output =
      ConvertXSpaceToUtilizationViewer(**xspace);
  if (!json_output.ok()) {
    return json_output.status();
  }

  SetOutput(*json_output, "application/json");
  return absl::OkStatus();
}

}  // namespace xprof
