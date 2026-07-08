/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

#include "xprof/convert/trace_view_options.h"

#include <cmath>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/numbers.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/profiler/utils/timespan.h"
#include "xprof/convert/tool_options.h"
#include "xprof/convert/trace_viewer/trace_events_filter_interface.h"
#include "xprof/convert/trace_viewer/trace_options.h"
#include "xprof/convert/trace_viewer/trace_viewer_visibility.h"
#include "xprof/convert/xplane_to_trace_container.h"

namespace tensorflow {
namespace profiler {

absl::StatusOr<TraceViewOption> GetTraceViewOption(const ToolOptions& options) {
  TraceViewOption trace_options;
  std::string start_time_ms_opt =
      GetParamWithDefault<std::string>(options, "start_time_ms", "0.0");
  std::string end_time_ms_opt =
      GetParamWithDefault<std::string>(options, "end_time_ms", "0.0");
  std::string resolution_opt =
      GetParamWithDefault<std::string>(options, "resolution", "0");
  trace_options.event_name =
      GetParamWithDefault<std::string>(options, "event_name", "");
  trace_options.search_prefix =
      GetParamWithDefault<std::string>(options, "search_prefix", "");
  std::string duration_ms_opt =
      GetParamWithDefault<std::string>(options, "duration_ms", "0.0");
  std::string unique_id_opt =
      GetParamWithDefault<std::string>(options, "unique_id", "0");
  trace_options.format =
      GetParamWithDefault<std::string>(options, "format", "json");
  trace_options.search_metadata =
      GetParamWithDefault<bool>(options, "search_metadata", false);

  if (!absl::SimpleAtod(start_time_ms_opt, &trace_options.start_time_ms) ||
      !absl::SimpleAtod(end_time_ms_opt, &trace_options.end_time_ms) ||
      !absl::SimpleAtod(duration_ms_opt, &trace_options.duration_ms)) {
    return absl::InvalidArgumentError("wrong arguments");
  }

  if (!absl::SimpleAtoi(resolution_opt, &trace_options.resolution)) {
    double resolution_double;
    if (absl::SimpleAtod(resolution_opt, &resolution_double)) {
      trace_options.resolution = static_cast<uint64_t>(resolution_double);
    } else {
      return absl::InvalidArgumentError("resolution must be a number");
    }
  }

  if (!absl::SimpleAtoi(unique_id_opt, &trace_options.unique_id)) {
    double unique_id_double;
    if (absl::SimpleAtod(unique_id_opt, &unique_id_double)) {
      trace_options.unique_id = static_cast<uint64_t>(unique_id_double);
    } else {
      return absl::InvalidArgumentError("unique_id must be a number");
    }
  }
  return trace_options;
}

absl::Status LoadTraceEventsContainer(
    const TraceEventsLevelDbFilePaths& file_paths,
    const TraceViewOption& trace_option,
    const TraceOptions& profiler_trace_options,
    TraceEventsContainer* trace_container) {
  if (!trace_option.event_name.empty()) {
    TF_RETURN_IF_ERROR(trace_container->ReadFullEventFromLevelDbTable(
        file_paths.trace_events_metadata_file_path,
        file_paths.trace_events_file_path, trace_option.event_name,
        static_cast<uint64_t>(std::round(trace_option.start_time_ms * 1E9)),
        static_cast<uint64_t>(std::round(trace_option.duration_ms * 1E9)),
        trace_option.unique_id));
  } else if (!trace_option.search_prefix.empty()) {  // Search Events Request
    if (tsl::Env::Default()
            ->FileExists(file_paths.trace_events_prefix_trie_file_path)
            .ok()) {
      std::unique_ptr<TraceEventsFilterInterface> trace_events_filter =
          CreateTraceEventsFilterFromTraceOptions(profiler_trace_options);
      TF_RETURN_IF_ERROR(trace_container->SearchInLevelDbTable(
          file_paths, trace_option.search_prefix,
          std::move(trace_events_filter),
          {.search_metadata = trace_option.search_metadata}));
    }
  } else {
    auto visibility_filter = std::make_unique<TraceVisibilityFilter>(
        tsl::profiler::MilliSpan(trace_option.start_time_ms,
                                 trace_option.end_time_ms),
        trace_option.resolution, profiler_trace_options);
    constexpr int64_t kDisableStreamingThreshold = 500000;
    std::unique_ptr<TraceEventsFilterInterface> trace_events_filter =
        CreateTraceEventsFilterFromTraceOptions(profiler_trace_options);
    TF_RETURN_IF_ERROR(trace_container->LoadFromLevelDbTable(
        file_paths, std::move(trace_events_filter),
        std::move(visibility_filter), kDisableStreamingThreshold));
  }
  return absl::OkStatus();
}

}  // namespace profiler
}  // namespace tensorflow
