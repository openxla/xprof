#include "xprof/convert/processors/inference_stats_processor.h"

#include <string>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/platform/errors.h"
#include "tsl/profiler/protobuf/xplane.pb.h"
#include "xprof/convert/multi_xspace_to_inference_stats.h"
#include "xprof/convert/processors/profile_processor_factory.h"
#include "xprof/convert/repository.h"
#include "xprof/convert/tool_options.h"

namespace xprof {

using ::tensorflow::profiler::GetParamWithDefault;
using ::tensorflow::profiler::SessionSnapshot;
using ::tensorflow::profiler::ToolOptions;
using ::tensorflow::profiler::InferenceStats;

absl::Status InferenceStatsProcessor::ProcessSession(
    const SessionSnapshot& session_snapshot, const ToolOptions& options) {

  LOG(INFO) << "Processing inference stats for host: "
            << session_snapshot.GetHostname(0);

  InferenceStats inference_stats;
  std::string request_column =
      GetParamWithDefault<std::string>(options, "request_column", "");
  std::string batch_column =
      GetParamWithDefault<std::string>(options, "batch_column", "");
  TF_RETURN_IF_ERROR(ConvertMultiXSpaceToInferenceStats(
      session_snapshot, request_column, batch_column, &inference_stats));

  std::string inference_stats_json;
  inference_stats_json = InferenceStatsToDataTableJson(inference_stats);
  SetOutput(inference_stats_json, "application/json");
  return absl::OkStatus();
}

REGISTER_PROFILE_PROCESSOR("inference_profile", InferenceStatsProcessor);

}  // namespace xprof
