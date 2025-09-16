#include "xprof/convert/memory_viewer_processor.h"

#include <optional>
#include <string>
#include <utility>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/numbers.h"
#include "absl/strings/string_view.h"
// #include "google/protobuf/json/json.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "tsl/platform/protobuf.h"
#include "tsl/profiler/protobuf/xplane.pb.h"
#include "xprof/convert/graphviz_helper.h"
#include "xprof/convert/hlo_proto_to_memory_visualization_utils.h"
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

using ::tensorflow::profiler::ConvertHloProtoToPreprocessResult;
using ::tensorflow::profiler::GetHloProtoByModuleName;
using ::tensorflow::profiler::GetParam;
using ::tensorflow::profiler::GetParamWithDefault;
using ::tensorflow::profiler::PreprocessResult;
using ::tensorflow::profiler::SessionSnapshot;
using ::tensorflow::profiler::ToolOptions;
using ::tensorflow::profiler::WrapDotInHtml;

constexpr absl::string_view kDotLayoutEngine = "neato";

constexpr absl::string_view kModuleNameOption = "module_name";

constexpr absl::string_view kMemorySpaceOption = "memory_space";

constexpr absl::string_view kOptionViewMemoryAllocationTimeline =
    "view_memory_allocation_timeline";

absl::StatusOr<PreprocessResult> GetMemoryViewerPreprocessResult(
    const xla::HloProto& hlo_proto, int memory_space_color) {
  static constexpr int kSmallBufferSize = 16 * 1024;  // 16KB

  auto result_or = ConvertHloProtoToPreprocessResult(
      hlo_proto, kSmallBufferSize, memory_space_color);
  if (!result_or.ok()) {
    return tsl::errors::Internal(
        "Failed to convert HLO proto to memory viewer result: ",
        result_or.status().message());
  }
  return result_or;
}

absl::StatusOr<std::string> ConvertHloProtoToAllocationTimeline(
    const xla::HloProto& hlo_proto, int memory_space_color) {
  auto result_or =
      GetMemoryViewerPreprocessResult(hlo_proto, memory_space_color);
  if (!result_or.ok()) {
    return result_or.status();
  }

  return WrapDotInHtml(std::move(result_or.value().allocation_timeline()),
                       kDotLayoutEngine);
}

absl::StatusOr<std::string> ConvertHloProtoToMemoryViewer(
    const xla::HloProto& hlo_proto, int memory_space_color) {
  auto result_or =
      GetMemoryViewerPreprocessResult(hlo_proto, memory_space_color);
  if (!result_or.ok()) {
    return result_or.status();
  }

  std::string json_output;
  tsl::protobuf::util::JsonPrintOptions options;
  options.always_print_fields_with_no_presence = true;
  auto encoded_status = tsl::protobuf::util::MessageToJsonString(
      result_or.value(), &json_output, options);
  if (!encoded_status.ok()) {
    const auto& error_message = encoded_status.message();
    return tsl::errors::Internal(
        "Failed to convert memory viewer result to JSON format: ",
        absl::string_view(error_message.data(), error_message.length()));
  }

  return json_output;
}

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

  std::string memory_viewer_json;

  if (GetParamWithDefault(
          options, std::string(kOptionViewMemoryAllocationTimeline), 0)) {
    TF_ASSIGN_OR_RETURN(memory_viewer_json, ConvertHloProtoToAllocationTimeline(
                                                hlo_proto, memory_space_color));
  } else {
    TF_ASSIGN_OR_RETURN(memory_viewer_json, ConvertHloProtoToMemoryViewer(
                                                hlo_proto, memory_space_color));
  }
  SetOutput(memory_viewer_json, "application/json");
  return absl::OkStatus();
}

REGISTER_PROFILE_PROCESSOR("memory_viewer", MemoryViewerProcessor);

}  // namespace xprof
