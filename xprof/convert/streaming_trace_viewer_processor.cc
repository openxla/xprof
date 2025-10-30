#include "xprof/convert/streaming_trace_viewer_processor.h"

#include <cmath>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>

#include "absl/container/flat_hash_map.h"
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
  std::string event_name = "";
  std::string search_prefix = "";
  double duration_ms = 0.0;
  uint64_t unique_id = 0;
};

absl::StatusOr<TraceViewOption> GetTraceViewOption(const ToolOptions& options) {
  TraceViewOption trace_options;
  auto start_time_ms_opt =
      GetParamWithDefault<std::string>(options, "start_time_ms", "0.0");
  auto end_time_ms_opt =
      GetParamWithDefault<std::string>(options, "end_time_ms", "0.0");
  auto resolution_opt =
      GetParamWithDefault<std::string>(options, "resolution", "0");
  trace_options.event_name =
      GetParamWithDefault<std::string>(options, "event_name", "");
  trace_options.search_prefix =
      GetParamWithDefault<std::string>(options, "search_prefix", "");
  auto duration_ms_opt =
      GetParamWithDefault<std::string>(options, "duration_ms", "0.0");
  auto unique_id_opt =
      GetParamWithDefault<std::string>(options, "unique_id", "0");


  if (!absl::SimpleAtoi(resolution_opt, &trace_options.resolution) ||
      !absl::SimpleAtod(start_time_ms_opt, &trace_options.start_time_ms) ||
      !absl::SimpleAtod(end_time_ms_opt, &trace_options.end_time_ms) ||
      !absl::SimpleAtoi(unique_id_opt, &trace_options.unique_id) ||
      !absl::SimpleAtod(duration_ms_opt, &trace_options.duration_ms)) {
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

  absl::flat_hash_map<std::string, int> host_to_id_map;
  if (auto all_hosts = session_snapshot.GetAllHosts()) {
    for (int i = 0; i < all_hosts->size(); ++i) {
      host_to_id_map[(*all_hosts)[i]] = i;
    }
  }

  // TODO: b/452217676 - Optimize this to process hosts in parallel.
  for (int i = 0; i < session_snapshot.XSpaceSize(); ++i) {
    int host_id = host_to_id_map[session_snapshot.GetHostname(i)];
    google::protobuf::Arena arena;
    TF_ASSIGN_OR_RETURN(XSpace * xspace, session_snapshot.GetXSpace(i, &arena));
    PreprocessSingleHostXSpace(xspace, /*step_grouping=*/true,
                               /*derived_timeline=*/true);

    std::string host_name = session_snapshot.GetHostname(i);
    auto trace_events_sstable_path = session_snapshot.MakeHostDataFilePath(
        tensorflow::profiler::StoredDataType::TRACE_LEVELDB, host_name);
    auto trace_events_metadata_sstable_path =
        session_snapshot.MakeHostDataFilePath(
            tensorflow::profiler::StoredDataType::TRACE_EVENTS_METADATA_LEVELDB,
            host_name);
    auto trace_events_prefix_trie_sstable_path =
        session_snapshot.MakeHostDataFilePath(
            tensorflow::profiler::StoredDataType::
                TRACE_EVENTS_PREFIX_TRIE_LEVELDB,
            host_name);
    if (!trace_events_sstable_path || !trace_events_metadata_sstable_path ||
        !trace_events_prefix_trie_sstable_path) {
      return tsl::errors::Unimplemented(
          "streaming trace viewer hasn't been supported in Cloud AI");
    }
    if (!tsl::Env::Default()->FileExists(*trace_events_sstable_path).ok()) {
      ProcessMegascaleDcn(xspace);
      TraceEventsContainer trace_container;
      ConvertXSpaceToTraceEventsContainer(host_name, host_id, *xspace,
                                          &trace_container);
      std::unique_ptr<tsl::WritableFile> trace_events_file;
      TF_RETURN_IF_ERROR(tsl::Env::Default()->NewWritableFile(
          *trace_events_sstable_path, &trace_events_file));
      std::unique_ptr<tsl::WritableFile> trace_events_metadata_file;
      TF_RETURN_IF_ERROR(tsl::Env::Default()->NewWritableFile(
          *trace_events_metadata_sstable_path, &trace_events_metadata_file));
      std::unique_ptr<tsl::WritableFile> trace_events_prefix_trie_file;
      TF_RETURN_IF_ERROR(tsl::Env::Default()->NewWritableFile(
          *trace_events_prefix_trie_sstable_path,
          &trace_events_prefix_trie_file));
      TF_RETURN_IF_ERROR(trace_container.StoreAsLevelDbTables(
          std::move(trace_events_file),
          std::move(trace_events_metadata_file),
          std::move(trace_events_prefix_trie_file)
      ));
    }

    TraceEventsLevelDbFilePaths file_paths;
    file_paths.trace_events_file_path = *trace_events_sstable_path;
    file_paths.trace_events_metadata_file_path =
        *trace_events_metadata_sstable_path;
    file_paths.trace_events_prefix_trie_file_path =
        *trace_events_prefix_trie_sstable_path;

    TraceEventsContainer trace_container;
    if (!trace_option.event_name.empty()) {
      TF_RETURN_IF_ERROR(trace_container.ReadFullEventFromLevelDbTable(
          *trace_events_metadata_sstable_path, *trace_events_sstable_path,
          trace_option.event_name,
          static_cast<uint64_t>(std::round(trace_option.start_time_ms * 1E9)),
          static_cast<uint64_t>(std::round(trace_option.duration_ms * 1E9)),
          trace_option.unique_id));
    } else if (!trace_option.search_prefix.empty()) {  // Search Events Request
      if (tsl::Env::Default()
              ->FileExists(*trace_events_prefix_trie_sstable_path).ok()) {
        auto trace_events_filter =
            CreateTraceEventsFilterFromTraceOptions(profiler_trace_options);
        TF_RETURN_IF_ERROR(trace_container.SearchInLevelDbTable(
            file_paths,
            trace_option.search_prefix, std::move(trace_events_filter)));
      }
    } else {
      auto visibility_filter = std::make_unique<TraceVisibilityFilter>(
          tsl::profiler::MilliSpan(trace_option.start_time_ms,
                                   trace_option.end_time_ms),
          trace_option.resolution, profiler_trace_options);
      // Trace smaller than threshold will be disabled from streaming.
      constexpr int64_t kDisableStreamingThreshold = 500000;
      auto trace_events_filter =
          CreateTraceEventsFilterFromTraceOptions(profiler_trace_options);
      TF_RETURN_IF_ERROR(trace_container.LoadFromLevelDbTable(
          file_paths, std::move(trace_events_filter),
          std::move(visibility_filter), kDisableStreamingThreshold));
    }
    merged_trace_container.Merge(std::move(trace_container));
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
