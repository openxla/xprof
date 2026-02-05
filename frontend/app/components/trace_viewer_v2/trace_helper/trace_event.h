#ifndef THIRD_PARTY_XPROF_FRONTEND_APP_COMPONENTS_TRACE_VIEWER_V2_TRACE_HELPER_TRACE_EVENT_H_
#define THIRD_PARTY_XPROF_FRONTEND_APP_COMPONENTS_TRACE_VIEWER_V2_TRACE_HELPER_TRACE_EVENT_H_

#include <cstdint>
#include <limits>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/string_view.h"
#include "tsl/profiler/lib/context_types.h"

namespace traceviewer {

// Type aliases for clarity
using ProcessId = uint32_t;
using ThreadId = uint32_t;
// EventId is a unique identifier for an event.
// Event timestamp, duration and its name are used to generate the event ID.
// See http://shortn/_lVpPbx16ZS for more details.
using EventId = uint64_t;
// Timestamps are in microseconds, as specified in go/trace-event-format.
// An example ts: 6845940.1418570001
// We are not using absl::Duration because the data source provides timestamps
// as doubles, converted from picoseconds, which are not always integer values.
using Microseconds = double;
using Milliseconds = double;

// The phase of the event.
// More phases are defined in:
// https://source.chromium.org/chromium/chromium/src/+/main:base/trace_event/common/trace_event_common.h;l=1070-1093;drc=3874c9832e2de7ebf55eb4cad2bf9683556fb5e9
// For XProf, we are only interested in metadata and complete events.
enum class Phase : char {
  kComplete = 'X',
  kCounter = 'C',
  kMetadata = 'M',
  kFlowStart = 'b',
  kFlowEnd = 'e',

  // Represents an unknown or unspecified event phase.
  // This makes the default state more explicit and type-safe.
  kUnknown = 0,
};

// Struct to hold parsed trace event data for trace viewer. This avoids repeated
// lookups and type conversions using emscripten::val.
// See go/trace-event-format for more details.
//
// This struct differs from the TraceEvent proto
// (google3/third_party/plugin/xprof/protobuf/trace_events.proto)
// used for storage.
struct TraceEvent {
  Phase ph = Phase::kUnknown;
  EventId event_id = 0;
  ProcessId pid = 0;
  ThreadId tid = 0;
  std::string name;
  Microseconds ts = 0.0;
  Microseconds dur = 0.0;
  std::string id;
  tsl::profiler::ContextType category = tsl::profiler::ContextType::kGeneric;
  std::map<std::string, std::string> args;
};

struct CounterEvent {
  ProcessId pid = 0;
  std::string name;
  std::vector<Microseconds> timestamps;
  std::vector<double> values;
  double min_value = std::numeric_limits<double>::infinity();
  double max_value = -std::numeric_limits<double>::infinity();
};

struct ParsedTraceEvents {
  std::vector<TraceEvent> flame_events;
  std::vector<CounterEvent> counter_events;
  std::vector<TraceEvent> flow_events;
  // The full timespan of the trace, from the earliest event timestamp to the
  // latest event timestamp, in milliseconds.
  std::optional<std::pair<Milliseconds, Milliseconds>> full_timespan;
  // The initial visible range in milliseconds.
  std::optional<std::pair<Milliseconds, Milliseconds>> visible_range_from_url;

  bool mpmd_pipeline_view = false;
};

// Constants for metadata events.
inline constexpr absl::string_view kThreadName = "thread_name";
inline constexpr absl::string_view kProcessName = "process_name";
// The name of the metadata event used to sort processes (e.g., device rows
// in trace viewer).
inline constexpr absl::string_view kProcessSortIndex = "process_sort_index";
// The name of the metadata event used to sort threads within a process (e.g.,
// resource rows in trace viewer).
inline constexpr absl::string_view kThreadSortIndex = "thread_sort_index";
// The argument name for sort index in process_sort_index and
// thread_sort_index metadata events.
inline constexpr absl::string_view kSortIndex = "sort_index";
inline constexpr absl::string_view kName = "name";

// LINT.IfChange
// Constants for trace event argument keys.
inline constexpr absl::string_view kHloOp = "hlo_op";
inline constexpr absl::string_view kHloModule = "hlo_module";
inline constexpr absl::string_view kHloModuleId = "hlo_module_id";
inline constexpr absl::string_view kProgramId = "program_id";
inline constexpr absl::string_view kKernelDetails = "kernel_details";

// Constants for thread names.
inline constexpr absl::string_view kXlaOps = "XLA Ops";
inline constexpr absl::string_view kXlaModules = "XLA Modules";
inline constexpr absl::string_view kComputeUtilization =
    "Compute Utilization/Roofline Efficiency";
inline constexpr absl::string_view kDataMotionLayersUtilization =
    "Data motion layers utilization";

// Other constants.
inline constexpr absl::string_view kHloModuleDefault = "default";
inline constexpr absl::string_view kModuleRegex = "module:.*_(\\d+)";

}  // namespace traceviewer

#endif  // THIRD_PARTY_XPROF_FRONTEND_APP_COMPONENTS_TRACE_VIEWER_V2_TRACE_HELPER_TRACE_EVENT_H_
