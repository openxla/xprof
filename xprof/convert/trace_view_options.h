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
  // Optional upper time bound in milliseconds. `std::nullopt` leaves the upper
  // bound open-ended (clamped to the trace end), whereas an explicit `0.0`
  // represents a closed upper bound at `0.0` ms.
  std::optional<double> end_time_ms = std::nullopt;
  std::string event_name;
  std::string search_prefix;
  double duration_ms = 0.0;
  uint64_t unique_id = 0;
  bool search_metadata = false;
  std::string format = "json";
};

// Parses the `ToolOptions` (typically from an HTTP request) into a structured
// `TraceViewOption`. Returns an `InvalidArgumentError` if provided numeric
// options are malformed, if `start_time_ms`, `end_time_ms`, or `duration_ms`
// is negative, non-finite, or overflows `uint64_t` picoseconds, if `resolution`
// is negative, non-finite, or overflows `uint64_t`, if `unique_id` is not an
// exact non-negative integer within `uint64_t`, or (when `event_name` is
// non-empty) if `start_time_ms` or `duration_ms` overflows non-negative
// `int64_t` picoseconds or `unique_id` exceeds `int64_t::max()`.
absl::StatusOr<TraceViewOption> GetTraceViewOption(const ToolOptions& options);

struct TraceEventsLevelDbFilePaths;
struct TraceOptions;

// Loads trace events from LevelDB tables into the provided `trace_container`
// based on the parameters in `trace_option`. When `event_name` is non-empty,
// returns an `InvalidArgumentError` if `start_time_ms` or `duration_ms` is
// negative, non-finite, or overflows non-negative `int64_t` picoseconds, or if
// `unique_id` exceeds `int64_t::max()`. When loading a time window, returns an
// `InvalidArgumentError` if `start_time_ms` or `end_time_ms` is negative,
// non-finite, or overflows `uint64_t` picoseconds; valid inverted bounds
// (`start_time_ms > *end_time_ms`) are normalized to an empty window and return
// `OkStatus()` with zero loaded events.
absl::Status LoadTraceEventsContainer(
    const TraceEventsLevelDbFilePaths& file_paths,
    const TraceViewOption& trace_option,
    const TraceOptions& profiler_trace_options,
    TraceEventsContainer* trace_container);

}  // namespace profiler
}  // namespace tensorflow

#endif  // THIRD_PARTY_XPROF_CONVERT_TRACE_VIEW_OPTIONS_H_
