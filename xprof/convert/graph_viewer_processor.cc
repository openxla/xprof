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

#include "xprof/convert/graph_viewer_processor.h"

#include <memory>
#include <optional>
#include <string>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/tsl/platform/statusor.h"
#include "tsl/profiler/protobuf/xplane.pb.h"
#include "xprof/convert/hlo_proto_to_graph_view.h"
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
#include "xprof/utils/custom_call_utils.h"
#include "xprof/utils/hlo_module_utils.h"
#include "xprof/utils/hlo_proto_to_module.h"

namespace xprof {

using ::tensorflow::profiler::ConvertHloProtoToGraph;
using ::tensorflow::profiler::ConvertHloProtoToStringView;
using ::tensorflow::profiler::GetAdjacentNodes;
using ::tensorflow::profiler::GetHloProtoByModuleName;
using ::tensorflow::profiler::GetParam;
using ::tensorflow::profiler::GraphViewerParams;
using ::tensorflow::profiler::kAdjacentNodes;
using ::tensorflow::profiler::kCustomCallGraphTypeName;
using ::tensorflow::profiler::kGraphTypeName;
using ::tensorflow::profiler::ParseGraphViewerParams;
using ::tensorflow::profiler::SessionSnapshot;
using ::tensorflow::profiler::ToolOptions;

constexpr absl::string_view kModuleNameOption = "module_name";

absl::StatusOr<std::string> ConvertHloProtoToGraphViewer(
    const xla::HloProto& hlo_proto, const ToolOptions& options) {
  TF_ASSIGN_OR_RETURN(GraphViewerParams params,
                      ParseGraphViewerParams(options));
  if (params.type == kGraphTypeName) {
    return ConvertHloProtoToGraph(hlo_proto, params.node_name,
                                  params.graph_width, params.render_options,
                                  params.format);
  } else if (params.type == kCustomCallGraphTypeName) {
    TF_ASSIGN_OR_RETURN(
        std::unique_ptr<xla::HloModule> hlo_module,
        tensorflow::profiler::ConvertHloProtoToModule(hlo_proto));
    const xla::HloInstruction* hlo_instruction =
        tensorflow::profiler::FindInstruction(*hlo_module, params.node_name);
    if (hlo_instruction == nullptr) {
      return absl::InvalidArgumentError("Hlo Instruction not found.");
    }
    return GetCustomCallText(*hlo_instruction);
  } else if (params.type == kAdjacentNodes) {
    return GetAdjacentNodes(hlo_proto, params.node_name);
  } else {
    // All other types are string view types
    return ConvertHloProtoToStringView(hlo_proto, params.type, params.verbose,
                                       params.show_metadata);
  }
}

absl::Status GraphViewerProcessor::ProcessSession(
    const SessionSnapshot& session_snapshot, const ToolOptions& options) {
  std::optional<std::string> hlo_module_name =
      GetParam<std::string>(options, std::string(kModuleNameOption));
  if (!hlo_module_name.has_value() || hlo_module_name->empty()) {
    return absl::InvalidArgumentError(
        "Can not find HLO module name from options.");
  }

  LOG(INFO) << "Processing graph viewer for  hlo module: " << *hlo_module_name;

  // Load HLO module from file.
  TF_ASSIGN_OR_RETURN(
      xla::HloProto hlo_proto,
      GetHloProtoByModuleName(session_snapshot, *hlo_module_name));

  std::string graph_viewer_json;

  TF_ASSIGN_OR_RETURN(graph_viewer_json,
                      ConvertHloProtoToGraphViewer(hlo_proto, options));

  SetOutput(graph_viewer_json, "application/json");
  return absl::OkStatus();
}

REGISTER_PROFILE_PROCESSOR("graph_viewer", GraphViewerProcessor);

}  // namespace xprof
