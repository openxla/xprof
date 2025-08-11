#include "xprof/convert/megascale_stats_processor.h"

#include <optional>
#include <string>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/platform/statusor.h"
#include "tsl/profiler/protobuf/xplane.pb.h"
#include "xprof/convert/profile_processor_factory.h"
#include "xprof/convert/process_megascale_dcn.h"
#include "xprof/convert/repository.h"
#include "xprof/convert/tool_options.h"
#include "xprof/convert/xplane_to_dcn_collective_stats.h"
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

using ::tensorflow::profiler::DcnSlackAnalysis;
using ::tensorflow::profiler::GetParam;
using ::tensorflow::profiler::SessionSnapshot;
using ::tensorflow::profiler::ToolOptions;

absl::Status MegascaleStatsProcessor::ProcessSession(
    const SessionSnapshot& session_snapshot, const ToolOptions& options) {
  std::optional<std::string> hostname =
      GetParam<std::string>(options, "host_name");
  if (!hostname.has_value() || hostname->empty()) {
    return absl::InvalidArgumentError(
        "Cannot find host_name from options for megascale_stats tool.");
  }

  TF_ASSIGN_OR_RETURN(
      DcnSlackAnalysis dcnSlackAnalysis,
      GetDcnSlackAnalysisByHostName(session_snapshot, hostname.value()));

  std::string megascale_stats_json = GenerateMegaScaleJson(dcnSlackAnalysis);
  SetOutput(megascale_stats_json, "application/json");
  return absl::OkStatus();
}

REGISTER_PROFILE_PROCESSOR("megascale_stats", MegascaleStatsProcessor);

}  // namespace xprof
