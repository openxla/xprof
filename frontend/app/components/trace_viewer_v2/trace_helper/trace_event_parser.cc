#include "xprof/frontend/app/components/trace_viewer_v2/trace_helper/trace_event_parser.h"

#include <emscripten/bind.h>
#include <emscripten/val.h>

#include <cstdio>

#include "xprof/frontend/app/components/trace_viewer_v2/application.h"
#include "xprof/frontend/app/components/trace_viewer_v2/timeline/data_provider.h"
#include "xprof/frontend/app/components/trace_viewer_v2/trace_helper/trace_event.h"

namespace traceviewer {

namespace {

Phase ParsePhase(const std::string& ph_str) {
  if (!ph_str.empty()) {
    switch (ph_str[0]) {
      case static_cast<char>(Phase::kComplete):
        return Phase::kComplete;
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
// Expected fields for XProf events:
// - "ph": Phase of the event. We are interested in:
//     - "M" (Metadata): For thread names ("thread_name").
//     - "X" (Complete Event): Represents a duration event.
// - "tid": Thread ID.
// - "pid": Process ID.
// - "ts": Timestamp in microseconds.
// - "dur": Duration in microseconds.
// - "name": Name of the event.
// - "args": Arguments associated with the event.
TraceEvent FromVal(const emscripten::val& event) {
  TraceEvent ev;
  if (event.hasOwnProperty("ph")) {
    std::string ph_str = event["ph"].as<std::string>();
    ev.ph = ParsePhase(ph_str);
  }
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
  return ev;
}

}  // namespace

std::vector<TraceEvent> ParseTraceEvents(const emscripten::val& trace_data) {
  std::vector<TraceEvent> event_list;
  if (!trace_data.hasOwnProperty("traceEvents")) {
    return event_list;
  }

  emscripten::val events = trace_data["traceEvents"];
  const auto js_events = emscripten::vecFromJSArray<emscripten::val>(events);

  event_list.reserve(js_events.size());
  for (const auto& js_event : js_events) {
    event_list.push_back(FromVal(js_event));
  }
  return event_list;
}

void ParseAndProcessTraceEvents(const emscripten::val& trace_data) {
  const std::vector<TraceEvent> event_list = ParseTraceEvents(trace_data);

  Application::Instance().data_provider().ProcessTraceEvents(
      event_list, Application::Instance().timeline());
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
