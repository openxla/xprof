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

#include "xprof/convert/memory_viewer_processor.h"

#include <optional>
#include <string>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/numbers.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "tsl/profiler/protobuf/xplane.pb.h"
#include "xprof/convert/hlo_proto_to_memory_visualization_utils.h"
#include "xprof/convert/hlo_to_tools_data.h"
#include "xprof/convert/profile_processor_factory.h"
#include "xprof/convert/repository.h"
#include "xprof/convert/tool_options.h"
#include "xprof/convert/xplane_to_hlo.h"
#include "plugin/xprof/protobuf/dcn_slack_analysis.pb.h"
#include "plugin/xprof/protobuf/hardware_types.pb.h"
#include "plugin/xprof/protobuf/inference_stats.pb.h"
#include "plugin/xprof/protobuf/input_pipeline.pb.h"
#include "plugin/xprof/protobuf/kernel_stats.pb.h"
#include "plugin/xprof/protobuf/op_profile.pb.h"
#include "plugin/xprof/protobuf/op_stats.pb.h"
#include "plugin/xprof/protobuf/overview_page.pb.h"
#include "plugin/xprof/protobuf/roofline_model.pb.h"
#include "plugin/xprof/protobuf/tf_data_stats.pb.h"
#include "plugin/xprof/protobuf/tf_stats.pb.h"

namespace xprof {

using ::tensorflow::profiler::GetHloProtoByModuleName;
using ::tensorflow::profiler::GetParam;
using ::tensorflow::profiler::GetParamWithDefault;
using ::tensorflow::profiler::SessionSnapshot;
using ::tensorflow::profiler::ToolOptions;

constexpr absl::string_view kModuleNameOption = "module_name";
constexpr absl::string_view kMemorySpaceOption = "memory_space";

absl::Status MemoryViewerProcessor::ProcessSession(
    const SessionSnapshot& session_snapshot, const ToolOptions& options) {
  std::optional<std::string> hlo_module_name =
      GetParam<std::string>(options, std::string(kModuleNameOption));
  if (!hlo_module_name.has_value() || hlo_module_name->empty()) {
    return absl::InvalidArgumentError(
        "Can not find HLO module name from options.");
  }
  LOG(INFO) << "Processing memory viewer for HLO module: " << *hlo_module_name;

  // Load HLO module from file.
  TF_ASSIGN_OR_RETURN(
      xla::HloProto hlo_proto,
      GetHloProtoByModuleName(session_snapshot, *hlo_module_name));

  // Convert from HLO proto to tools data.
  int memory_space_color = 0;
  if (!absl::SimpleAtoi(
          GetParamWithDefault(options, std::string(kMemorySpaceOption),
                              std::string("0")),
          &memory_space_color)) {
    memory_space_color = 0;
  }

  tensorflow::profiler::MemoryViewerOption memory_viewer_option;
  memory_viewer_option.memory_color = memory_space_color;
  memory_viewer_option.timeline_option.render_timeline =
      !!GetParamWithDefault(options, "view_memory_allocation_timeline", 0);
  memory_viewer_option.timeline_option.timeline_noise =
      !!GetParamWithDefault(options, "timeline_noise", 0);
  memory_viewer_option.small_buffer_size =
      tensorflow::profiler::kSmallBufferSize;

  std::string memory_viewer_json;

  TF_ASSIGN_OR_RETURN(memory_viewer_json,
                      tensorflow::profiler::ConvertHloProtoToMemoryViewer(
                          hlo_proto, memory_viewer_option));

  std::string content_type = "application/json";
  if (memory_viewer_option.timeline_option.render_timeline) {
    // Return html for timeline graph.
    content_type = "text/html";
  }
  SetOutput(memory_viewer_json, content_type);
  return absl::OkStatus();
}

REGISTER_PROFILE_PROCESSOR("memory_viewer", MemoryViewerProcessor);

}  // namespace xprof
