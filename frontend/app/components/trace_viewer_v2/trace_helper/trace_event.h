#ifndef THIRD_PARTY_XPROF_FRONTEND_APP_COMPONENTS_TRACE_VIEWER_V2_TRACE_HELPER_TRACE_EVENT_H_
#define THIRD_PARTY_XPROF_FRONTEND_APP_COMPONENTS_TRACE_VIEWER_V2_TRACE_HELPER_TRACE_EVENT_H_

#include <cstdint>
#include <limits>
#include <map>
#include <string>
#include <vector>

namespace traceviewer {

// Type aliases for clarity
using ProcessId = uint32_t;
using ThreadId = uint32_t;
// Timestamps are in microseconds, as specified in go/trace-event-format.
// An example ts: 6845940.1418570001
// We are not using absl::Duration because the data source provides timestamps
// as doubles, converted from picoseconds, which are not always integer values.
using Microseconds = double;

// The phase of the event.
// More phases are defined in:
// https://source.chromium.org/chromium/chromium/src/+/main:base/trace_event/common/trace_event_common.h;l=1070-1093;drc=3874c9832e2de7ebf55eb4cad2bf9683556fb5e9
// For XProf, we are only interested in metadata and complete events.
enum class Phase : char {
  kComplete = 'X',
  kCounter = 'C',
  kMetadata = 'M',

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
  ProcessId pid = 0;
  ThreadId tid = 0;
  std::string name;
  Microseconds ts = 0.0;
  Microseconds dur = 0.0;
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
};

}  // namespace traceviewer

#endif  // THIRD_PARTY_XPROF_FRONTEND_APP_COMPONENTS_TRACE_VIEWER_V2_TRACE_HELPER_TRACE_EVENT_H_
