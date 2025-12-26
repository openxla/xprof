#include "xprof/frontend/app/components/trace_viewer_v2/trace_helper/trace_event_parser.h"

#include <emscripten/bind.h>
#include <emscripten/val.h>

#include <algorithm>

#include "xprof/frontend/app/components/trace_viewer_v2/application.h"
#include "xprof/frontend/app/components/trace_viewer_v2/timeline/data_provider.h"
#include "xprof/frontend/app/components/trace_viewer_v2/trace_helper/trace_event.h"

namespace traceviewer {

namespace {

constexpr char kFullTimespan[] = "fullTimespan";

Phase ParsePhase(const std::string& ph_str) {
  if (!ph_str.empty()) {
    switch (ph_str[0]) {
      case static_cast<char>(Phase::kComplete):
        return Phase::kComplete;
      case static_cast<char>(Phase::kCounter):
        return Phase::kCounter;
      case static_cast<char>(Phase::kMetadata):
        return Phase::kMetadata;
      default:
        return Phase::kUnknown;
    }
  }
  return Phase::kUnknown;
}

// Helper function to convert emscripten::val to TraceEvent
// Processes trace data from a JSON object.
// The JSON object is expected to have a top-level key "traceEvents", which is
// an array of event objects.
// Each event object should follow the Trace Event Format: go/trace-event-format
// Expected fields for XProf complete events:
// - "ph": Phase of the event. We are interested in:
//     - "M" (Metadata): For thread names ("thread_name").
//     - "X" (Complete Event): Represents a duration event.
// - "tid": Thread ID.
// - "pid": Process ID.
// - "ts": Timestamp in microseconds.
// - "dur": Duration in microseconds.
// - "name": Name of the event.
// - "args": Optional arguments associated with the event.
// Expected fields for XProf counter events:
// - "ph": Phase of the event. We are interested in:
//     - "C" (Counter): Represents a counter event.
// - "pid": Process ID.
// - "name": Name of the counter.
// - "entries": An array of counter entries. Each entry is an array of length 2,
//     where the first element is the timestamp in microseconds and the second
//     element is the counter value.
//     Example:
//     [
//       [1000000.0, 1.0],
//       [1000001.0, 2.0]
//     ]
void ParseAndAppend(const emscripten::val& event, ParsedTraceEvents& result) {
  if (!event.hasOwnProperty("ph")) {
    return;
  }

  std::string ph_str = event["ph"].as<std::string>();
  Phase ph = ParsePhase(ph_str);

  if (ph == Phase::kCounter) {
    if (!event.hasOwnProperty("entries")) {
      // Discard counter events without entries.
      return;
    }
    CounterEvent ev;
    if (event.hasOwnProperty("pid")) ev.pid = event["pid"].as<ProcessId>();
    if (event.hasOwnProperty("name")) ev.name = event["name"].as<std::string>();

    emscripten::val entries = event["entries"];
    // Avoid converting the entry to a vector or intermediate objects to
    // reduce memory allocation and GC pressure.
    // We can access array elements by index directly.
    const int length = entries["length"].as<int>();
    ev.timestamps.reserve(length);
    ev.values.reserve(length);
    // The points in the counter event are sorted by timestamp (this should be
    // guaranteed by the trace event producer), so we can simply append them.
    for (int i = 0; i < length; ++i) {
      emscripten::val entry = entries[i];
      // The length of the entry is expected to be 2, where the first element
      // is the timestamp and the second element is the value.
      // If the length is not 2, we discard the entry.
      if (entry["length"].as<int>() == 2) {
        Microseconds ts = entry[0].as<Microseconds>();
        double val = entry[1].as<double>();
        ev.timestamps.push_back(ts);
        ev.values.push_back(val);
        ev.min_value = std::min(ev.min_value, val);
        ev.max_value = std::max(ev.max_value, val);
      }
    }
    result.counter_events.push_back(std::move(ev));
  } else {
    // Parse non-counter events, such as complete or metadata events.
    TraceEvent ev;
    ev.ph = ph;
    if (event.hasOwnProperty("pid")) ev.pid = event["pid"].as<ProcessId>();
    if (event.hasOwnProperty("tid")) ev.tid = event["tid"].as<ThreadId>();
    if (event.hasOwnProperty("name")) ev.name = event["name"].as<std::string>();
    if (event.hasOwnProperty("ts")) ev.ts = event["ts"].as<Microseconds>();
    if (event.hasOwnProperty("dur")) ev.dur = event["dur"].as<Microseconds>();
    if (event.hasOwnProperty("args")) {
      emscripten::val args_val = event["args"];
      emscripten::val keys =
          emscripten::val::global("Object").call<emscripten::val>("keys",
                                                                  args_val);
      int length = keys["length"].as<int>();
      for (int i = 0; i < length; ++i) {
        std::string key = keys[i].as<std::string>();
        if (args_val[key].isString()) {
          ev.args[key] = args_val[key].as<std::string>();
        } else if (args_val[key].isNumber()) {
          ev.args[key] = std::to_string(args_val[key].as<double>());
        }
        // Other types such as boolean, nested objects, or arrays are currently
        // ignored as they are not required for the flame chart in trace viewer.
      }
    }
    result.flame_events.push_back(std::move(ev));
  }
}

}  // namespace

ParsedTraceEvents ParseTraceEvents(
    const emscripten::val& trace_data,
    const emscripten::val& visible_range_from_url) {
  ParsedTraceEvents result;
  if (!trace_data.hasOwnProperty("traceEvents")) {
    return result;
  }

  if (trace_data.hasOwnProperty("mpmdPipelineView")) {
    result.mpmd_pipeline_view = trace_data["mpmdPipelineView"].as<bool>();
  }

  emscripten::val events = trace_data["traceEvents"];
  const auto js_events = emscripten::vecFromJSArray<emscripten::val>(events);
  // Reserve space for the most common event type (flame events) to avoid
  // reallocations.
  // We don't reserve space for counter events as they are significantly fewer
  // in number.
  result.flame_events.reserve(js_events.size());

  for (const auto& js_event : js_events) {
    ParseAndAppend(js_event, result);
  }
  // Reclaim unused memory.
  result.flame_events.shrink_to_fit();
  // Shrink vectors for counter events to release unused memory.
  // Vectors typically double in capacity upon reallocation; shrinking ensures
  // memory usage matches the actual data size.
  result.counter_events.shrink_to_fit();

  if (trace_data.hasOwnProperty(kFullTimespan)) {
    emscripten::val span = trace_data[kFullTimespan];
    if (span["length"].as<int>() == 2) {
      Milliseconds start = span[0].as<Milliseconds>();
      Milliseconds end = span[1].as<Milliseconds>();
      if (start >= 0 && end >= 0 && start <= end) {
        result.full_timespan = std::make_pair(start, end);
      }
    }
  }

  if (!visible_range_from_url.isNull() &&
      !visible_range_from_url.isUndefined() &&
      visible_range_from_url["length"].as<int>() == 2) {
    Milliseconds start = visible_range_from_url[0].as<Milliseconds>();
    Milliseconds end = visible_range_from_url[1].as<Milliseconds>();
    result.visible_range_from_url = std::make_pair(start, end);
  }

  return result;
}

void ParseAndProcessTraceEvents(const emscripten::val& trace_data,
                                const emscripten::val& visible_range_from_url) {
  const ParsedTraceEvents parsed_events =
      ParseTraceEvents(trace_data, visible_range_from_url);

  Application::Instance().data_provider().ProcessTraceEvents(
      parsed_events, Application::Instance().timeline());

  // Reset the loading flag to allow subsequent data requests (e.g. on panning).
  // This is necessary because is_incremental_loading_ is initialized to true to
  // prevent duplicate requests during the initial load.
  Application::Instance().timeline().set_is_incremental_loading(false);
}

EMSCRIPTEN_BINDINGS(trace_event_parser) {
  // Bind std::vector<std::string>
  emscripten::register_vector<std::string>("StringVector");

  // Bind DataProvider class
  emscripten::class_<traceviewer::DataProvider>("DataProvider")
      .function("getProcessList", &traceviewer::DataProvider::GetProcessList);

  emscripten::function("processTraceEvents",
                       &traceviewer::ParseAndProcessTraceEvents);

  // Bind Application class and expose the singleton instance and data_provider
  emscripten::class_<traceviewer::Application>("Application")
      .class_function("Instance", &traceviewer::Application::Instance,
                      emscripten::return_value_policy::reference())
      .function("data_provider", &traceviewer::Application::data_provider);
}

}  // namespace traceviewer
