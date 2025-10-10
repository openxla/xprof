#include "xprof/convert/streaming_trace_viewer_processor.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/numbers.h"
#include "absl/strings/string_view.h"
#include "google/protobuf/arena.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/file_system.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/profiler/utils/timespan.h"
#include "tsl/profiler/protobuf/xplane.pb.h"
#include "xprof/convert/preprocess_single_host_xplane.h"
#include "xprof/convert/process_megascale_dcn.h"
#include "xprof/convert/profile_processor_factory.h"
#include "xprof/convert/repository.h"
#include "xprof/convert/tool_options.h"
#include "xprof/convert/trace_viewer/trace_events.h"
#include "xprof/convert/trace_viewer/trace_events_to_json.h"
#include "xprof/convert/trace_viewer/trace_options.h"
#include "xprof/convert/trace_viewer/trace_viewer_visibility.h"
#include "xprof/convert/xplane_to_trace_container.h"

namespace xprof {

using ::tensorflow::profiler::GetParamWithDefault;
using ::tensorflow::profiler::IOBufferAdapter;
using ::tensorflow::profiler::JsonTraceOptions;
using ::tensorflow::profiler::RawData;
using ::tensorflow::profiler::SessionSnapshot;
using ::tensorflow::profiler::ToolOptions;
using ::tensorflow::profiler::TraceDeviceType;
using ::tensorflow::profiler::TraceEventsContainer;
using ::tensorflow::profiler::TraceEventsLevelDbFilePaths;
using ::tensorflow::profiler::TraceOptionsFromToolOptions;
using ::tensorflow::profiler::TraceVisibilityFilter;
using ::tensorflow::profiler::XSpace;

struct TraceViewOption {
  uint64_t resolution = 0;
  double start_time_ms = 0.0;
  double end_time_ms = 0.0;
};

absl::StatusOr<TraceViewOption> GetTraceViewOption(const ToolOptions& options) {
  TraceViewOption trace_options;
  auto start_time_ms_opt =
      GetParamWithDefault<std::string>(options, "start_time_ms", "0.0");
  auto end_time_ms_opt =
      GetParamWithDefault<std::string>(options, "end_time_ms", "0.0");
  auto resolution_opt =
      GetParamWithDefault<std::string>(options, "resolution", "0");

  if (!absl::SimpleAtoi(resolution_opt, &trace_options.resolution) ||
      !absl::SimpleAtod(start_time_ms_opt, &trace_options.start_time_ms) ||
      !absl::SimpleAtod(end_time_ms_opt, &trace_options.end_time_ms)) {
    return tsl::errors::InvalidArgument("wrong arguments");
  }
  return trace_options;
}

absl::Status StreamingTraceViewerProcessor::ProcessSession(
    const SessionSnapshot& session_snapshot, const ToolOptions& options) {
  TraceEventsContainer merged_trace_container;
  std::string tool_name = "trace_viewer@";

  TF_ASSIGN_OR_RETURN(TraceViewOption trace_option,
                      GetTraceViewOption(options));
  tensorflow::profiler::TraceOptions profiler_trace_options =
      TraceOptionsFromToolOptions(options);

  // TODO: b/452217676 - Optimize this to process hosts in parallel.
  for (int i = 0; i < session_snapshot.XSpaceSize(); ++i) {
    int host_id = i+1;
    google::protobuf::Arena arena;
    TF_ASSIGN_OR_RETURN(XSpace * xspace, session_snapshot.GetXSpace(i, &arena));
    PreprocessSingleHostXSpace(xspace, /*step_grouping=*/true,
                               /*derived_timeline=*/true);

    std::string host_name = session_snapshot.GetHostname(i);
    auto sstable_path = session_snapshot.GetFilePath(tool_name, host_name);
    if (!sstable_path) {
      return tsl::errors::Unimplemented(
          "streaming trace viewer hasn't been supported in Cloud AI");
    }
    if (!tsl::Env::Default()->FileExists(*sstable_path).ok()) {
      ProcessMegascaleDcn(xspace);
      TraceEventsContainer trace_container;
      ConvertXSpaceToTraceEventsContainer(host_name, *xspace,
                                          &trace_container);
      std::unique_ptr<tsl::WritableFile> file;
      TF_RETURN_IF_ERROR(
          tsl::Env::Default()->NewWritableFile(*sstable_path, &file));
      TF_RETURN_IF_ERROR(trace_container.StoreAsLevelDbTable(std::move(file)));
    }

    auto visibility_filter = std::make_unique<TraceVisibilityFilter>(
        tsl::profiler::MilliSpan(trace_option.start_time_ms,
                                 trace_option.end_time_ms),
        trace_option.resolution, profiler_trace_options);
    TraceEventsContainer trace_container;
    // Trace smaller than threshold will be disabled from streaming.
    constexpr int64_t kDisableStreamingThreshold = 500000;
    auto trace_events_filter =
        CreateTraceEventsFilterFromTraceOptions(profiler_trace_options);
    TraceEventsLevelDbFilePaths file_paths;
    file_paths.trace_events_file_path = *sstable_path;
    TF_RETURN_IF_ERROR(trace_container.LoadFromLevelDbTable(
        file_paths, std::move(trace_events_filter),
        std::move(visibility_filter), kDisableStreamingThreshold));
    merged_trace_container.Merge(std::move(trace_container), host_id);
  }

  std::string trace_viewer_json;
  JsonTraceOptions json_trace_options;

  tensorflow::profiler::TraceDeviceType device_type =
      tensorflow::profiler::TraceDeviceType::kUnknownDevice;
  if (IsTpuTrace(merged_trace_container.trace())) {
    device_type = TraceDeviceType::kTpu;
  }
  json_trace_options.details =
      TraceOptionsToDetails(device_type, profiler_trace_options);
  IOBufferAdapter adapter(&trace_viewer_json);
  TraceEventsToJson<IOBufferAdapter, TraceEventsContainer, RawData>(
      json_trace_options, merged_trace_container, &adapter);

  SetOutput(trace_viewer_json, "application/json");
  return absl::OkStatus();
}

// NOTE: We use "trace_viewer@" to distinguish from the non-streaming
// trace_viewer. The "@" suffix is used to indicate that this tool
// supports streaming.
REGISTER_PROFILE_PROCESSOR("trace_viewer@", StreamingTraceViewerProcessor);

}  // namespace xprof
