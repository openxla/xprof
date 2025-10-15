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

#include "xprof/convert/hlo_to_tools_data.h"

#include <optional>
#include <string>
#include <utility>

#include "absl/status/statusor.h"
#include "absl/strings/numbers.h"
#include "absl/strings/string_view.h"
#include "xla/service/hlo.pb.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "tsl/platform/protobuf.h"
#include "xprof/convert/hlo_proto_to_graph_view.h"
#include "xprof/convert/hlo_proto_to_memory_visualization_utils.h"
#include "xprof/convert/repository.h"
#include "xprof/convert/tool_options.h"
#include "xprof/convert/xplane_to_hlo.h"
#include "plugin/xprof/protobuf/memory_viewer_preprocess.pb.h"

namespace tensorflow {
namespace profiler {

namespace {

absl::StatusOr<PreprocessResult> GetMemoryViewerPreprocessResult(
    const xla::HloProto& hlo_proto, const MemoryViewerOption& option) {
  auto result_or = ConvertHloProtoToPreprocessResult(hlo_proto, option);
  if (!result_or.ok()) {
    return tsl::errors::Internal(
        "Failed to convert HLO proto to memory viewer result: ",
        result_or.status().message());
  }
  return result_or;
}

absl::StatusOr<std::string> ConvertHloProtoToGraphViewer(
    const xla::HloProto& hlo_proto, const ToolOptions& options) {
  TF_ASSIGN_OR_RETURN(GraphViewerParams params,
                      ParseGraphViewerParams(options));
  if (params.type == kGraphTypeName) {
    return ConvertHloProtoToGraph(hlo_proto, params.node_name,
                                  params.graph_width, params.render_options,
                                  params.format);
  } else if (params.type == kAdjacentNodes) {
    return GetAdjacentNodes(hlo_proto, params.node_name);
  } else {
    // All other types are string view types
    return ConvertHloProtoToStringView(hlo_proto, params.type, params.verbose,
                                       params.show_metadata);
  }
}

}  // namespace

absl::StatusOr<std::string> ConvertHloProtoToMemoryViewer(
    const xla::HloProto& hlo_proto, const MemoryViewerOption& option) {
  return ConvertHloProtoToPreprocessResultJson(hlo_proto, option);
}

absl::StatusOr<std::string> ConvertHloProtoToToolData(
    const SessionSnapshot& session_snapshot, const absl::string_view tool_name,
    const ToolOptions& options) {
  // <options> must provide a hlo module_name field to identify the HLO module.
  std::optional<std::string> hlo_module_name =
      GetParam<std::string>(options, "module_name");
  if (!hlo_module_name.has_value() || hlo_module_name->empty()) {
    return tsl::errors::InvalidArgument(
        "Can not find HLO module name from options.");
  }

  // Load HLO module from file.
  TF_ASSIGN_OR_RETURN(
      xla::HloProto hlo_proto,
      GetHloProtoByModuleName(session_snapshot, *hlo_module_name));

  // Convert from HLO proto to tools data.
  int memory_space_color = 0;
  if (!absl::SimpleAtoi(
          GetParamWithDefault(options, "memory_space", std::string("0")),
          &memory_space_color)) {
    memory_space_color = 0;
  }

  if (tool_name == "memory_viewer") {
    MemoryViewerOption memory_viewer_option;
    memory_viewer_option.memory_color = memory_space_color;
    memory_viewer_option.timeline_option.render_timeline =
        !!GetParamWithDefault(options, "view_memory_allocation_timeline", 0);
    memory_viewer_option.timeline_option.timeline_noise =
        !!GetParamWithDefault(options, "timeline_noise", 0);
    memory_viewer_option.small_buffer_size = kSmallBufferSize;
    return ConvertHloProtoToMemoryViewer(hlo_proto, memory_viewer_option);
  } else if (tool_name == "graph_viewer") {
    return ConvertHloProtoToGraphViewer(hlo_proto, options);
  } else {
    return tsl::errors::InvalidArgument(
        "Can not find tool: ", tool_name,
        ". Please update to the latest version of Tensorflow.");
  }
}

}  // namespace profiler
}  // namespace tensorflow
