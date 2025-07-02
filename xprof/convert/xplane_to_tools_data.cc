/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "xprof/convert/xplane_to_tools_data.h"

#include <string>

#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "tsl/profiler/protobuf/xplane.pb.h"
#include "xprof/convert/hlo_to_tools_data.h"
#include "xprof/convert/repository.h"
#include "xprof/convert/tool_options.h"
#include "xprof/convert/xplane_to_hlo.h"
#include "xprof/convert/xplane_to_tool_names.h"
#include "xprof/convert/tool_converters.h"
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

namespace tensorflow {
namespace profiler {

absl::StatusOr<std::string> ConvertMultiXSpacesToToolData(
    const SessionSnapshot& session_snapshot, const absl::string_view tool_name,
    const ToolOptions& options) {
  LOG(INFO) << "serving tool: " << tool_name
            << " with options: " << DebugString(options);
  if (tool_name == "trace_viewer" || tool_name == "trace_viewer@") {
    return ConvertXSpaceToTraceEvents(session_snapshot, tool_name, options);
  } else if (tool_name == "overview_page") {
    return ConvertMultiXSpacesToOverviewPage(session_snapshot);
  } else if (tool_name == "input_pipeline_analyzer") {
    return ConvertMultiXSpacesToInputPipeline(session_snapshot);
  } else if (tool_name == "framework_op_stats") {
    return ConvertMultiXSpacesToTfStats(session_snapshot);
  } else if (tool_name == "kernel_stats") {
    return ConvertMultiXSpacesToKernelStats(session_snapshot);
  } else if (tool_name == "memory_profile") {
    return ConvertXSpaceToMemoryProfile(session_snapshot);
  } else if (tool_name == "pod_viewer") {
    return ConvertMultiXSpacesToPodViewer(session_snapshot);
  } else if (tool_name == "op_profile") {
    return ConvertMultiXSpacesToOpProfileViewer(session_snapshot);
  } else if (tool_name == "hlo_stats") {
    return ConvertMultiXSpacesToHloStats(session_snapshot);
  } else if (tool_name == "roofline_model") {
    return ConvertMultiXSpacesToRooflineModel(session_snapshot);
  } else if (tool_name == "memory_viewer" || tool_name == "graph_viewer") {
    return ConvertHloProtoToToolData(session_snapshot, tool_name, options);
  } else if (tool_name == "megascale_stats") {
    return ConvertDcnCollectiveStatsToToolData(session_snapshot, options);
  } else if (tool_name == "tool_names") {
    // Generate the proto cache for hlo_proto tool.
    // This is needed for getting the module list.
    // TODO - b/378923777: Create only when needed.
    TF_ASSIGN_OR_RETURN(bool hlo_proto_status,
                        ConvertMultiXSpaceToHloProto(session_snapshot));
    LOG_IF(WARNING, !hlo_proto_status)
        << "No HLO proto found in XSpace.";
    return GetAvailableToolNames(session_snapshot);
  } else if (tool_name == "_xplane.pb") {  // internal test only.
    return PreprocessXSpace(session_snapshot);
  } else if (tool_name == "inference_profile") {
    return ConvertMultiXSpacesToInferenceStats(session_snapshot, options);
  } else {
    return tsl::errors::InvalidArgument(
        "Can not find tool: ", tool_name,
        ". Please update to the latest version of Tensorflow.");
  }
}

}  // namespace profiler
}  // namespace tensorflow
