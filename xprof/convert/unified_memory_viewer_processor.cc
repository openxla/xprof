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

#include "xprof/convert/unified_memory_viewer_processor.h"

#include <optional>
#include <string>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/numbers.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/platform/statusor.h"
#include "xprof/convert/hlo_proto_to_memory_visualization_utils.h"
#include "xprof/convert/hlo_to_tools_data.h"
#include "xprof/convert/tool_options.h"
#include "xprof/convert/unified_session_snapshot.h"

namespace xprof {

absl::Status UnifiedMemoryViewerProcessor::ProcessHlo(
    const XprofSessionSnapshot& session_snapshot,
    const xla::HloProto& hlo_proto,
    const tensorflow::profiler::ToolOptions& options) {
  LOG(INFO) << "Processing memory viewer for HLO module: "
            << hlo_proto.hlo_module().name();

  int memory_space_color = 0;
  if (!absl::SimpleAtoi(
          tensorflow::profiler::GetParamWithDefault(
              options, std::string(tensorflow::profiler::kMemorySpaceOption),
              std::string("0")),
          &memory_space_color)) {
    memory_space_color = 0;
  }

  tensorflow::profiler::MemoryViewerOption memory_viewer_option;
  memory_viewer_option.memory_color = memory_space_color;
  auto get_bool_param = [&](absl::string_view key) {
    if (std::optional<bool> value =
            tensorflow::profiler::GetParam<bool>(options, key);
        value.has_value()) {
      return *value;
    }
    if (std::optional<int> value =
            tensorflow::profiler::GetParam<int>(options, key);
        value.has_value()) {
      return *value != 0;
    }
    return false;
  };

  memory_viewer_option.timeline_option.render_timeline =
      get_bool_param("view_memory_allocation_timeline") ||
      get_bool_param("timeline");
  memory_viewer_option.timeline_option.timeline_noise =
      get_bool_param("timeline_noise");
  memory_viewer_option.small_buffer_size =
      tensorflow::profiler::kSmallBufferSize;

  std::string memory_viewer_json;

  TF_ASSIGN_OR_RETURN(memory_viewer_json,
                      tensorflow::profiler::ConvertHloProtoToMemoryViewer(
                          hlo_proto, memory_viewer_option));

  std::string content_type = "application/json";
  if (memory_viewer_option.timeline_option.render_timeline) {
    content_type = "text/html";
  }
  SetOutput(memory_viewer_json, content_type);
  return absl::OkStatus();
}


}  // namespace xprof
