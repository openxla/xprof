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

#ifndef THIRD_PARTY_XPROF_CONVERT_TRACE_VIEW_OPTIONS_H_
#define THIRD_PARTY_XPROF_CONVERT_TRACE_VIEW_OPTIONS_H_

#include <cstdint>
#include <optional>
#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xprof/convert/tool_options.h"
#include "xprof/convert/xplane_to_trace_container.h"

namespace tensorflow {
namespace profiler {

// Options representing the parameters for viewing a trace.
struct TraceViewOption {
  uint64_t resolution = 0;
  double start_time_ms = 0.0;
  std::optional<double> end_time_ms = std::nullopt;
  std::string event_name;
  std::string search_prefix;
  double duration_ms = 0.0;
  uint64_t unique_id = 0;
  bool search_metadata = false;
  std::string format = "json";
};

// Parses the `ToolOptions` (typically from an HTTP request) into a structured
// `TraceViewOption`. Returns an `InvalidArgumentError` if required numeric
// options are malformed or if `start_time_ms` or `end_time_ms` is negative,
// non-finite, or overflows `uint64_t` picoseconds.
absl::StatusOr<TraceViewOption> GetTraceViewOption(const ToolOptions& options);

struct TraceEventsLevelDbFilePaths;
struct TraceOptions;

// Loads trace events from LevelDB tables into the provided `trace_container`
// based on the parameters in `trace_option`. When loading a time window,
// invalid bounds (negative, non-finite, or overflowing `uint64_t` picoseconds)
// or inverted bounds (`start_time_ms > *end_time_ms`) are treated as an empty
// window rather than converting out-of-range doubles to picoseconds.
absl::Status LoadTraceEventsContainer(
    const TraceEventsLevelDbFilePaths& file_paths,
    const TraceViewOption& trace_option,
    const TraceOptions& profiler_trace_options,
    TraceEventsContainer* trace_container);

}  // namespace profiler
}  // namespace tensorflow

#endif  // THIRD_PARTY_XPROF_CONVERT_TRACE_VIEW_OPTIONS_H_
