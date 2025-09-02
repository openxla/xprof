#include "xprof/convert/processors/graph_viewer_processor.h"

#include <optional>
#include <string>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/platform/statusor.h"
#include "tsl/profiler/protobuf/xplane.pb.h"
#include "xprof/convert/hlo_proto_to_graph_view.h"
#include "xprof/convert/processors/profile_processor_factory.h"
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

using ::tensorflow::profiler::ConvertHloProtoToGraph;
using ::tensorflow::profiler::ConvertHloProtoToStringView;
using ::tensorflow::profiler::GetAdjacentNodes;
using ::tensorflow::profiler::GetHloProtoByModuleName;
using ::tensorflow::profiler::GetParam;
using ::tensorflow::profiler::GraphViewerParams;
using ::tensorflow::profiler::kAdjacentNodes;
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
