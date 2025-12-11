#include "xprof/convert/streaming_trace_viewer_processor.h"

#include <cmath>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "absl/strings/strip.h"
#include "google/protobuf/arena.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/file_system.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/profiler/utils/timespan.h"
#include "tsl/platform/path.h"
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
using internal::GetTraceViewOption;
using internal::TraceViewOption;

namespace {
// Traces with events less than threshold will be disabled from streaming.
constexpr int64_t kDisableStreamingThreshold = 500000;
}  // namespace

absl::Status StreamingTraceViewerProcessor::ProcessSession(
    const SessionSnapshot& session_snapshot, const ToolOptions& options) {
  TraceEventsContainer merged_trace_container;

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
      ConvertXSpaceToTraceEventsContainer(host_name, *xspace,
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
      auto trace_events_filter =
          CreateTraceEventsFilterFromTraceOptions(profiler_trace_options);
      TF_RETURN_IF_ERROR(trace_container.LoadFromLevelDbTable(
          file_paths, std::move(trace_events_filter),
          std::move(visibility_filter), kDisableStreamingThreshold));
    }
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

absl::StatusOr<std::string> StreamingTraceViewerProcessor::Map(
    const std::string& xspace_path) {
  std::vector<std::string> xspace_paths = {xspace_path};
  TF_ASSIGN_OR_RETURN(
      SessionSnapshot session_snapshot,
      SessionSnapshot::Create(xspace_paths, /*xspaces=*/std::nullopt));
  // get xspace from session snapshot
  std::string hostname = session_snapshot.GetHostname(0);
  google::protobuf::Arena arena;
  TF_ASSIGN_OR_RETURN(XSpace * xspace, session_snapshot.GetXSpace(0, &arena));

  return Map(session_snapshot, hostname, *xspace);
}

absl::StatusOr<std::string> StreamingTraceViewerProcessor::Map(
    const SessionSnapshot& session_snapshot, const std::string& hostname,
    const XSpace& xspace) {
  XSpace temp_xspace = xspace;
  tensorflow::profiler::PreprocessSingleHostXSpace(&temp_xspace,
                                                   /*step_grouping=*/true,
                                                   /*derived_timeline=*/true);
  tensorflow::profiler::ProcessMegascaleDcn(&temp_xspace);

  auto trace_events_sstable_path = session_snapshot.MakeHostDataFilePath(
      tensorflow::profiler::StoredDataType::TRACE_LEVELDB, hostname);
  auto trace_events_metadata_sstable_path =
      session_snapshot.MakeHostDataFilePath(
          tensorflow::profiler::StoredDataType::TRACE_EVENTS_METADATA_LEVELDB,
          hostname);
  auto trace_events_prefix_trie_sstable_path =
      session_snapshot.MakeHostDataFilePath(
          tensorflow::profiler::StoredDataType::
              TRACE_EVENTS_PREFIX_TRIE_LEVELDB,
          hostname);

  if (!trace_events_sstable_path.has_value() ||
      !trace_events_metadata_sstable_path.has_value() ||
      !trace_events_prefix_trie_sstable_path.has_value()) {
    return tsl::errors::Unimplemented(
        "streaming trace viewer hasn't been supported in Cloud AI");
  }

  if (!tsl::Env::Default()->FileExists(*trace_events_sstable_path).ok()) {
    TraceEventsContainer trace_container;
    tensorflow::profiler::ConvertXSpaceToTraceEventsContainer(
        hostname, temp_xspace, &trace_container);
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
        std::move(trace_events_file), std::move(trace_events_metadata_file),
        std::move(trace_events_prefix_trie_file)));
  }
  return *trace_events_sstable_path;
}

namespace {

absl::StatusOr<TraceEventsContainer> LoadTraceContainerForHost(
    const SessionSnapshot& session_snapshot,
    const std::string& trace_events_sstable_path,
    const TraceViewOption& trace_option,
    const tensorflow::profiler::TraceOptions& profiler_trace_options) {
  absl::string_view filename = tsl::io::Basename(trace_events_sstable_path);
  absl::ConsumeSuffix(&filename, ".SSTABLE");
  std::string hostname = std::string(filename);

  TraceEventsLevelDbFilePaths file_paths;
  file_paths.trace_events_file_path = trace_events_sstable_path;
  // These should exist as they were created in the Map phase.
  auto metadata_path = session_snapshot.MakeHostDataFilePath(
      tensorflow::profiler::StoredDataType::TRACE_EVENTS_METADATA_LEVELDB,
      hostname);
  auto trie_path = session_snapshot.MakeHostDataFilePath(
      tensorflow::profiler::StoredDataType::TRACE_EVENTS_PREFIX_TRIE_LEVELDB,
      hostname);
  if (!metadata_path.has_value() ||
      !tsl::Env::Default()->FileExists(*metadata_path).ok()) {
    return tsl::errors::Internal("Could not find metadata file for host: ",
                                 hostname, ", path: ", *metadata_path);
  }
  if (!trie_path.has_value() ||
      !tsl::Env::Default()->FileExists(*trie_path).ok()) {
    return tsl::errors::Internal("Could not find trie file for host: ",
                                 hostname, ", path: ", *trie_path);
  }
  file_paths.trace_events_metadata_file_path = *metadata_path;
  file_paths.trace_events_prefix_trie_file_path = *trie_path;

  TraceEventsContainer trace_container;
  if (!trace_option.event_name.empty()) {
    TF_RETURN_IF_ERROR(trace_container.ReadFullEventFromLevelDbTable(
        file_paths.trace_events_metadata_file_path,
        file_paths.trace_events_file_path, trace_option.event_name,
        static_cast<uint64_t>(std::round(trace_option.start_time_ms * 1E9)),
        static_cast<uint64_t>(std::round(trace_option.duration_ms * 1E9)),
        trace_option.unique_id));
  } else if (!trace_option.search_prefix.empty()) {  // Search Events Request
    if (tsl::Env::Default()
            ->FileExists(file_paths.trace_events_prefix_trie_file_path)
            .ok()) {
      auto trace_events_filter =
          CreateTraceEventsFilterFromTraceOptions(profiler_trace_options);
      TF_RETURN_IF_ERROR(trace_container.SearchInLevelDbTable(
          file_paths, trace_option.search_prefix,
          std::move(trace_events_filter)));
    }
  } else {
    auto visibility_filter = std::make_unique<TraceVisibilityFilter>(
        tsl::profiler::MilliSpan(trace_option.start_time_ms,
                                 trace_option.end_time_ms),
        trace_option.resolution, profiler_trace_options);
    auto trace_events_filter =
        CreateTraceEventsFilterFromTraceOptions(profiler_trace_options);
    TF_RETURN_IF_ERROR(trace_container.LoadFromLevelDbTable(
        file_paths, std::move(trace_events_filter),
        std::move(visibility_filter), kDisableStreamingThreshold));
  }
  return trace_container;
}

}  // namespace

absl::Status StreamingTraceViewerProcessor::Reduce(
    const SessionSnapshot& session_snapshot,
    const std::vector<std::string>& map_output_files) {
  if (map_output_files.empty()) {
    return absl::InvalidArgumentError("map_output_files cannot be empty");
  }

  TF_ASSIGN_OR_RETURN(TraceViewOption trace_option,
                      GetTraceViewOption(options_));
  tensorflow::profiler::TraceOptions profiler_trace_options =
      TraceOptionsFromToolOptions(options_);

  TraceEventsContainer merged_trace_container;

  for (int i = 0; i < map_output_files.size(); ++i) {
    const std::string& trace_events_sstable_path = map_output_files[i];
    int host_id = i + 1;

    TF_ASSIGN_OR_RETURN(
        TraceEventsContainer trace_container,
        LoadTraceContainerForHost(session_snapshot, trace_events_sstable_path,
                                  trace_option, profiler_trace_options));

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
