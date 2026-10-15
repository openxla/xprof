#include "frontend/app/components/trace_viewer_v2/trace_helper/trace_event_parser.h"

#include <emscripten/bind.h>
#include <emscripten/val.h>

#include <algorithm>
#include <cstdint>
#include <map>
#include <string>
#include <tuple>
#include <utility>

#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "tsl/profiler/lib/context_types.h"
#include "frontend/app/components/trace_viewer_v2/application.h"
#include "frontend/app/components/trace_viewer_v2/helper/time_formatter.h"
#include "frontend/app/components/trace_viewer_v2/timeline/data_provider.h"
#include "frontend/app/components/trace_viewer_v2/timeline/timeline.h"
#include "frontend/app/components/trace_viewer_v2/trace_helper/trace_event.h"
#include "frontend/app/components/trace_viewer_v2/trace_helper/trace_event_parser_core.h"

namespace traceviewer {

namespace {

constexpr char kFullTimespan[] = "fullTimespan";
// Optional top-level string key specifying the unit of the trace's x-axis.
// Missing or unrecognized values default to time.
constexpr char kTimeAxisUnit[] = "timeAxisUnit";
// Value of `kTimeAxisUnit` for a unitless sequence (e.g. scheduled bundle
// numbers) instead of time.
constexpr char kTimeAxisUnitUnitless[] = "unitless";

TimeAxisUnit ParseTimeAxisUnit(const emscripten::val& value) {
  if (value.isString() && value.as<std::string>() == kTimeAxisUnitUnitless) {
    return TimeAxisUnit::kUnitless;
  }
  return TimeAxisUnit::kTime;
}

// Optional string key of a counter event declaring how its values are shown.
// Missing or unrecognized values mean plain numbers.
constexpr char kCounterUnit[] = "unit";
constexpr absl::string_view kCounterUnitFraction = "fraction";
constexpr absl::string_view kCounterUnitBytes = "bytes";

CounterUnit ParseCounterUnit(const emscripten::val& value) {
  if (!value.isString()) return CounterUnit::kNone;
  const std::string unit = value.as<std::string>();
  if (unit == kCounterUnitFraction) return CounterUnit::kFraction;
  if (unit == kCounterUnitBytes) return CounterUnit::kBytes;
  return CounterUnit::kNone;
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
// - "unit": Optional. How the values are displayed: "fraction" (ratios in
//     [0, 1], shown as percentages) or "bytes". Anything else shows plain
//     numbers.
void ParseFlowField(const emscripten::val& event, const char* field_name,
                    const TraceEvent& ev, Phase ph,
                    std::vector<TraceEvent>& flow_events) {
  if (!event.hasOwnProperty(field_name)) return;
  emscripten::val val = event[field_name];
  if (val.isArray()) {
    const unsigned int len = val["length"].as<unsigned int>();
    for (unsigned int i = 0; i < len; ++i) {
      emscripten::val item = val[i];
      std::string flow_id;
      uint64_t parsed_flow_id = 0;
      bool has_parsed_flow_id = false;
      if (item.isString()) {
        flow_id = item.as<std::string>();
        if (!flow_id.empty()) {
          parsed_flow_id = ParseFlowIdString(flow_id);
          has_parsed_flow_id = true;
        }
      } else if (item.isNumber()) {
        parsed_flow_id =
            static_cast<uint64_t>(static_cast<int64_t>(item.as<double>()));
        flow_id = absl::StrCat(parsed_flow_id);
        has_parsed_flow_id = true;
      }
      if (has_parsed_flow_id) {
        TraceEvent flow_ev = ev;
        flow_ev.id = std::move(flow_id);
        flow_ev.flow_id = parsed_flow_id;
        flow_ev.has_flow_id = true;
        flow_ev.ph = ph;
        flow_events.push_back(std::move(flow_ev));
      }
    }
  } else if (!val.isNull() && !val.isUndefined() && val.as<bool>()) {
    if (ev.has_flow_id || ev.flow_id != 0 || !ev.id.empty()) {
      TraceEvent flow_ev = ev;
      if (flow_ev.id.empty()) {
        flow_ev.id = absl::StrCat(flow_ev.flow_id);
      }
      flow_ev.has_flow_id = true;
      flow_ev.ph = ph;
      flow_events.push_back(std::move(flow_ev));
    }
  }
}

void ParseAndAppend(
    const emscripten::val& event, ParsedTraceEvents& result,
    absl::flat_hash_map<std::tuple<ProcessId, std::string, uint64_t>,
                        TraceEvent>& open_async_events) {
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
    if (event.hasOwnProperty("pid"))
      ev.pid = static_cast<ProcessId>(event["pid"].as<double>());
    if (event.hasOwnProperty("name")) ev.name = event["name"].as<std::string>();
    if (event.hasOwnProperty("event_stats"))
      ev.event_stats = event["event_stats"].as<std::string>();
    if (event.hasOwnProperty(kCounterUnit)) {
      ev.unit = ParseCounterUnit(event[kCounterUnit]);
    }

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
    if (event.hasOwnProperty("pid"))
      ev.pid = static_cast<ProcessId>(event["pid"].as<double>());
    if (event.hasOwnProperty("tid"))
      ev.tid = static_cast<ThreadId>(event["tid"].as<double>());
    if (event.hasOwnProperty("name")) {
      ev.name = event["name"].as<std::string>();
    }
    if (event.hasOwnProperty("ts")) ev.ts = event["ts"].as<Microseconds>();
    if (event.hasOwnProperty("dur")) ev.dur = event["dur"].as<Microseconds>();
    if (event.hasOwnProperty("cat")) {
      emscripten::val cat = event["cat"];
      if (cat.isString()) {
        ev.category = GetContextTypeFromString(cat.as<std::string>());
      } else if (cat.isNumber()) {
        ev.category = tsl::profiler::GetSafeContextType(cat.as<uint32_t>());
      }
    }
    if (event.hasOwnProperty("id")) {
      emscripten::val id_val = event["id"];
      if (id_val.isString()) {
        ev.id = id_val.as<std::string>();
        if (!ev.id.empty()) {
          ev.flow_id = ParseFlowIdString(ev.id);
          ev.has_flow_id = true;
        }
      } else if (id_val.isNumber()) {
        ev.flow_id =
            static_cast<uint64_t>(static_cast<int64_t>(id_val.as<double>()));
        ev.id = absl::StrCat(ev.flow_id);
        ev.has_flow_id = true;
      }
    } else if (event.hasOwnProperty("bind_id")) {
      emscripten::val bind_id_val = event["bind_id"];
      if (bind_id_val.isString()) {
        ev.id = bind_id_val.as<std::string>();
        if (!ev.id.empty()) {
          ev.flow_id = ParseFlowIdString(ev.id);
          ev.has_flow_id = true;
        }
      } else if (bind_id_val.isNumber()) {
        ev.flow_id = static_cast<uint64_t>(
            static_cast<int64_t>(bind_id_val.as<double>()));
        ev.id = absl::StrCat(ev.flow_id);
        ev.has_flow_id = true;
      }
    }
    if (event.hasOwnProperty("args")) {
      emscripten::val args = event["args"];
      emscripten::val keys =
          emscripten::val::global("Object").call<emscripten::val>("keys", args);
      int length = keys["length"].as<int>();
      for (int i = 0; i < length; ++i) {
        std::string key = keys[i].as<std::string>();
        emscripten::val val = args[key];
        if (val.isString()) {
          ev.args[key] = val.as<std::string>();
        } else if (val.isNumber()) {
          ev.args[key] = val.call<std::string>("toString");
        } else if (val.isTrue()) {
          ev.args[key] = "true";
        } else if (val.isFalse()) {
          ev.args[key] = "false";
        } else if (key == kKernelDetails && !val.isNull() &&
                   !val.isUndefined()) {
          // Stringify specific object/array arguments. "kernel_details" can
          // contain useful information in a nested JSON structure.
          ev.args[key] = emscripten::val::global("JSON").call<std::string>(
              "stringify", val);
        }
        // Generic nested objects or arrays are ignored to prevent performance
        // issues from stringifying potentially large structures.
      }
    }
    switch (ev.ph) {
      case Phase::kAsyncBegin:
        if (ev.has_flow_id) {
          open_async_events[{ev.pid, ev.name, ev.flow_id}] = std::move(ev);
        }
        break;
      case Phase::kAsyncEnd:
        if (ev.has_flow_id) {
          auto it = open_async_events.find({ev.pid, ev.name, ev.flow_id});
          if (it != open_async_events.end()) {
            TraceEvent& begin_ev = it->second;
            begin_ev.ph = Phase::kComplete;
            begin_ev.is_async = true;
            if (ev.ts > begin_ev.ts) {
              begin_ev.dur = ev.ts - begin_ev.ts;
            }
            // Merge arguments from end event if any.
            if (!ev.args.empty()) {
              begin_ev.args.insert(ev.args.begin(), ev.args.end());
            }
            result.flame_events.push_back(std::move(begin_ev));
            open_async_events.erase(it);
          }
        }
        break;
      case Phase::kFlowStart:
      case Phase::kFlowEnd:
        if (ev.has_flow_id) {
          result.flame_events.push_back(std::move(ev));
        }
        break;
      case Phase::kComplete: {
        const bool has_flow_in = event.hasOwnProperty("flow_in");
        const bool has_flow_out = event.hasOwnProperty("flow_out");
        if (has_flow_in || has_flow_out) {
          ParseFlowField(event, "flow_in", ev, Phase::kFlowEnd,
                         result.flow_events);
          ParseFlowField(event, "flow_out", ev, Phase::kFlowStart,
                         result.flow_events);
          ev.id.clear();
          ev.flow_id = 0;
          ev.has_flow_id = false;
        }
        result.flame_events.push_back(std::move(ev));
        break;
      }
      case Phase::kInstant:
      case Phase::kInstantDeprecated:
        result.flame_events.push_back(std::move(ev));
        break;
      case Phase::kMetadata:
        result.flame_events.push_back(std::move(ev));
        break;
      default:
        break;
    }
  }
}
}  // namespace

ParsedTraceEvents ParseTraceEvents(
    const emscripten::val& trace_data,
    const emscripten::val& visible_range_from_url) {
  ParsedTraceEvents result;
  if (trace_data.isNull() || trace_data.isUndefined() ||
      !trace_data.hasOwnProperty("traceEvents")) {
    return result;
  }

  if (trace_data.hasOwnProperty("mpmdPipelineView")) {
    result.mpmd_pipeline_view = trace_data["mpmdPipelineView"].as<bool>();
  }

  if (trace_data.hasOwnProperty(kTimeAxisUnit)) {
    result.time_axis_unit = ParseTimeAxisUnit(trace_data[kTimeAxisUnit]);
  }

  emscripten::val events = trace_data["traceEvents"];
  const std::vector<emscripten::val> js_events =
      emscripten::vecFromJSArray<emscripten::val>(events);
  // Reserve space for the most common event type (flame events) to avoid
  // reallocations.
  // We don't reserve space for counter events as they are significantly fewer
  // in number.
  result.flame_events.reserve(js_events.size());

  absl::flat_hash_map<std::tuple<ProcessId, std::string, uint64_t>, TraceEvent>
      open_async_events;
  for (const auto& js_event : js_events) {
    ParseAndAppend(js_event, result, open_async_events);
  }
  // Reclaim unused memory.
  result.flame_events.shrink_to_fit();
  // Shrink vectors for counter events to release unused memory.
  // Vectors typically double in capacity upon reallocation; shrinking ensures
  // memory usage matches the actual data size.
  result.counter_events.shrink_to_fit();
  result.flow_events.shrink_to_fit();

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
  Timeline& timeline = Application::Instance().timeline();
  const bool is_initial_load =
      (timeline.last_fetch_request_range() == TimeRange::Zero());
  const ParsedTraceEvents parsed_events =
      ParseTraceEvents(trace_data, visible_range_from_url);
  const bool has_url_range = parsed_events.visible_range_from_url.has_value();
  const bool is_overview_load =
      is_initial_load && !has_url_range && timeline.incremental_fetch_enabled();
  timeline.set_full_trace_loaded(!timeline.incremental_fetch_enabled() &&
                                 !has_url_range);
  timeline.set_is_compacted(is_overview_load);
  timeline.set_window_at_full_resolution(!is_overview_load);
  Application::Instance().data_provider().ProcessTraceEvents(parsed_events,
                                                             timeline);

  // Set last_fetch_request_range_ on initial load to avoid duplicate fetches.
  // On incremental fetches, MaybeRequestData() has already set
  // last_fetch_request_range_ to the expanded fetch window.
  if (is_initial_load) {
    if (!visible_range_from_url.isNull() &&
        !visible_range_from_url.isUndefined() &&
        visible_range_from_url["length"].as<int>() == 2) {
      Milliseconds start = visible_range_from_url[0].as<Milliseconds>();
      Milliseconds end = visible_range_from_url[1].as<Milliseconds>();

      timeline.InitializeLastFetchRequestRange(
          {MillisToMicros(start), MillisToMicros(end)});
    } else {
      timeline.InitializeLastFetchRequestRange(timeline.data_time_range());
    }
  }

  // Reset the loading flag to allow subsequent data requests (e.g. on panning).
  // This is necessary because is_incremental_loading_ is initialized to true to
  // prevent duplicate requests during the initial load.
  timeline.set_is_incremental_loading(false);

  // Redraw the trace viewer to reflect the loaded data. This is necessary
  // because the trace viewer is not redrawn automatically. So every time new
  // data is processed, we need to redraw the trace viewer.
  Application::Instance().RequestRedraw();
}

void SetSearchResultsInWasm(const emscripten::val& trace_data) {
  if (!Application::Instance().IsInitialized()) {
    return;
  }
  const ParsedTraceEvents parsed_events =
      ParseTraceEvents(trace_data, emscripten::val::null());
  Application::Instance().timeline().SetSearchResults(parsed_events);
  Application::Instance().RequestRedraw();
}

emscripten::val GetAllFlowCategories() {
  emscripten::val categories = emscripten::val::array();
  for (int i = 0;
       i <= static_cast<int>(tsl::profiler::ContextType::kLastContextType);
       ++i) {
    auto type = static_cast<tsl::profiler::ContextType>(i);
    std::string name;
    // Prefer Pretty Name
    const absl::flat_hash_map<tsl::profiler::ContextType, absl::string_view>&
        pretty_map = GetPrettyNames();
    if (auto it = pretty_map.find(type); it != pretty_map.end()) {
      name = it->second;
    } else {
      // Fallback to canonical name
      const char* str = tsl::profiler::GetContextTypeString(type);
      if (str && *str) {
        name = str;
      } else {
        name = "Unknown";
      }
    }

    emscripten::val category = emscripten::val::object();
    category.set("id", i);
    category.set("name", name);
    categories.call<void>("push", category);
  }
  return categories;
}

EMSCRIPTEN_BINDINGS(trace_event_parser) {
  // Bind std::vector<std::string>
  emscripten::register_vector<std::string>("StringVector");
  emscripten::register_vector<int>("IntVector");

  emscripten::enum_<MouseMode>("MouseMode")
      .value("Select", MouseMode::kSelect)
      .value("Pan", MouseMode::kPan)
      .value("Zoom", MouseMode::kZoom)
      .value("Timing", MouseMode::kTiming);

  // Bind DataProvider class
  emscripten::class_<traceviewer::DataProvider>("DataProvider")
      .function("getFlowCategories",
                &traceviewer::DataProvider::GetFlowCategories)
      .function("reset", &traceviewer::DataProvider::Reset)
      .function(
          "getProcessMappings",
          emscripten::optional_override(
              [](const traceviewer::DataProvider& dp) {
                emscripten::val dict = emscripten::val::object();
                for (const auto& [pid, host_part] : dp.GetProcessMappings()) {
                  dict.set(static_cast<double>(pid), host_part);
                }
                return dict;
              }));

  emscripten::function("processTraceEvents",
                       &traceviewer::ParseAndProcessTraceEvents);

  emscripten::function("setSearchResultsInWasm",
                       &traceviewer::SetSearchResultsInWasm);

  emscripten::function("getAllFlowCategories",
                       &traceviewer::GetAllFlowCategories);

  // Bind Application class and expose the singleton instance and dataProvider
  emscripten::class_<traceviewer::Application>("application")
      .class_function("instance", &traceviewer::Application::Instance,
                      emscripten::return_value_policy::reference())
      .function("shutdown", &traceviewer::Application::Shutdown)
      .function("dataProvider", &traceviewer::Application::data_provider,
                emscripten::return_value_policy::reference())
      .function("getCurrentSearchResultIndex",
                &traceviewer::Application::GetCurrentSearchResultIndex)
      .function("getSearchResultsCount",
                &traceviewer::Application::GetSearchResultsCount)
      .function("navigateToNextSearchResult",
                &traceviewer::Application::NavigateToNextSearchResult)
      .function("navigateToPrevSearchResult",
                &traceviewer::Application::NavigateToPrevSearchResult)
      .function("resize", &traceviewer::Application::Resize)
      .function("setSearchQuery", &traceviewer::Application::SetSearchQuery)
      .function("setMouseMode", &traceviewer::Application::SetMouseMode)
      .function("setVisibleRange", &traceviewer::Application::SetVisibleRange)
      .function("setVisibleFlowCategory",
                &traceviewer::Application::SetVisibleFlowCategory)
      .function("setVisibleFlowCategories",
                &traceviewer::Application::SetVisibleFlowCategories)
      .function(
          "zoomIn",
          emscripten::optional_override([](traceviewer::Application& app) {
            app.timeline().Zoom(traceviewer::kButtonZoomInFactor);
          }))
      .function(
          "zoomOut",
          emscripten::optional_override([](traceviewer::Application& app) {
            app.timeline().Zoom(traceviewer::kButtonZoomOutFactor);
          }))
      .function("selectEvent",
                emscripten::optional_override(
                    [](traceviewer::Application& app, int event_index) {
                      app.timeline().ZoomEvent(event_index);
                    }));
}

}  // namespace traceviewer
