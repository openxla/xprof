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

#include "xprof/convert/unified_graph_viewer_processor.h"

#include <memory>
#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/hlo.pb.h"
#include "xla/tsl/platform/statusor.h"
#include "xprof/convert/hlo_proto_to_graph_view.h"
#include "xprof/convert/repository.h"
#include "xprof/convert/tool_options.h"
#include "xprof/convert/unified_session_snapshot.h"
#include "xprof/convert/xplane_to_hlo.h"
#include "xprof/utils/custom_call_utils.h"
#include "xprof/utils/hlo_module_utils.h"
#include "xprof/utils/hlo_proto_to_module.h"

namespace xprof {
namespace {

using ::tensorflow::profiler::GraphViewerParams;
using ::tensorflow::profiler::ParseGraphViewerParams;
using ::tensorflow::profiler::ToolOptions;
using ::tensorflow::profiler::kGraphTypeName;
using ::tensorflow::profiler::ConvertHloProtoToGraph;
using ::tensorflow::profiler::kCustomCallGraphTypeName;
using ::tensorflow::profiler::ConvertHloProtoToModule;
using ::tensorflow::profiler::FindInstruction;
using ::tensorflow::profiler::kAdjacentNodes;
using ::tensorflow::profiler::GetAdjacentNodes;
using ::tensorflow::profiler::ConvertHloProtoToStringView;

absl::StatusOr<xla::HloProto> GetHloProto(
    const tensorflow::profiler::SessionSnapshot& session_snapshot,
    const tensorflow::profiler::ToolOptions& options) {
  absl::StatusOr<xla::HloProto> hlo_proto =
      tensorflow::profiler::GetHloProtoByOptions(session_snapshot, options);
  if (hlo_proto.ok()) return hlo_proto;

  // Fallback: If module not found/provided, try searching by node name.
  absl::StatusOr<GraphViewerParams> params = ParseGraphViewerParams(options);
  if (params.ok() && !params->node_name.empty()) {
    hlo_proto = tensorflow::profiler::GetHloProtoByNodeName(
        session_snapshot, params->node_name);
  }
  return hlo_proto;
}

}  // namespace
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
        ConvertHloProtoToModule(hlo_proto));
    const xla::HloInstruction* hlo_instruction =
        FindInstruction(*hlo_module, params.node_name);
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


absl::Status UnifiedGraphViewerProcessor::ProcessSession(
    const XprofSessionSnapshot& session_snapshot,
    const tensorflow::profiler::ToolOptions& options) {
  const tensorflow::profiler::SessionSnapshot* profiler_session_snapshot =
      dynamic_cast<const tensorflow::profiler::SessionSnapshot*>(
          &session_snapshot);
  if (!profiler_session_snapshot) {
    return absl::InvalidArgumentError(
        "session_snapshot is not a tensorflow::profiler::SessionSnapshot");
  }

  // Get HLO Proto.
  TF_ASSIGN_OR_RETURN(xla::HloProto hlo_proto,
                      GetHloProto(*profiler_session_snapshot, options));

  TF_ASSIGN_OR_RETURN(std::string data,
                      ConvertHloProtoToGraphViewer(hlo_proto, options));
  SetOutput(data, "application/json");
  return absl::OkStatus();
}

}  // namespace xprof
