#include "xprof/frontend/app/components/trace_viewer_v2/trace_helper/trace_event_parser.h"

#include <emscripten/bind.h>
#include <emscripten/val.h>

#include <algorithm>
#include <map>

#include "absl/base/no_destructor.h"
#include "absl/container/flat_hash_map.h"
#include "absl/hash/hash.h"
#include "absl/strings/string_view.h"
#include "tsl/platform/fingerprint.h"
#include "tsl/profiler/lib/context_types.h"
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
      case static_cast<char>(Phase::kFlowStart):
        return Phase::kFlowStart;
      case static_cast<char>(Phase::kFlowEnd):
        return Phase::kFlowEnd;
      default:
        return Phase::kUnknown;
    }
  }
  return Phase::kUnknown;
}

const absl::flat_hash_map<tsl::profiler::ContextType, absl::string_view>&
GetPrettyNames() {
  static const absl::NoDestructor<
      absl::flat_hash_map<tsl::profiler::ContextType, absl::string_view>>
      kPrettyNames({
          {tsl::profiler::ContextType::kGeneric, "Generic"},
          {tsl::profiler::ContextType::kLegacy, "Legacy"},
          {tsl::profiler::ContextType::kTfExecutor, "TF Executor"},
          {tsl::profiler::ContextType::kTfrtExecutor, "TFRT Executor"},
          {tsl::profiler::ContextType::kSharedBatchScheduler,
           "Shared Batch Scheduler"},
          {tsl::profiler::ContextType::kPjRt, "PjRt"},
          {tsl::profiler::ContextType::kAdaptiveSharedBatchScheduler,
           "Adaptive Shared Batch Scheduler"},
          {tsl::profiler::ContextType::kTfrtTpuRuntime, "TFRT Tpu Runtime"},
          {tsl::profiler::ContextType::kTpuEmbeddingEngine,
           "Tpu Embedding Engine"},
          {tsl::profiler::ContextType::kGpuLaunch, "Gpu Launch"},
          {tsl::profiler::ContextType::kBatcher, "Batcher"},
          {tsl::profiler::ContextType::kTpuStream, "Tpu Stream"},
          {tsl::profiler::ContextType::kTpuLaunch, "Tpu Launch"},
          {tsl::profiler::ContextType::kPathwaysExecutor, "Pathways Executor"},
          {tsl::profiler::ContextType::kPjrtLibraryCall, "Pjrt Library Call"},
          {tsl::profiler::ContextType::kThreadpoolEvent, "Threadpool Event"},
          {tsl::profiler::ContextType::kJaxServingExecutor,
           "Jax Serving Executor"},
          {tsl::profiler::ContextType::kScOffload, "Sc Offload"},
      });
  return *kPrettyNames;
}

tsl::profiler::ContextType GetContextTypeFromString(
    absl::string_view category) {
  static const absl::NoDestructor<
      absl::flat_hash_map<absl::string_view, tsl::profiler::ContextType>>
      kCategoryMap([] {
        absl::flat_hash_map<absl::string_view, tsl::profiler::ContextType> map;

        // 1. Add pretty names
        for (const auto& [type, name] : GetPrettyNames()) {
          map[name] = type;
        }

        // 2. Add canonical names from tsl::profiler::GetContextTypeString
        for (int i = 0; i <= static_cast<int>(
                                 tsl::profiler::ContextType::kLastContextType);
             ++i) {
          auto type = static_cast<tsl::profiler::ContextType>(i);
          absl::string_view name(tsl::profiler::GetContextTypeString(type));
          if (!name.empty()) {
            map.emplace(name, type);
          }
        }
        return map;
      }());

  if (auto it = kCategoryMap->find(category); it != kCategoryMap->end()) {
    return it->second;
  }
  return tsl::profiler::ContextType::kGeneric;
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
    if (event.hasOwnProperty("cat")) {
      emscripten::val cat = event["cat"];
      if (cat.isNumber()) {
        ev.category = tsl::profiler::GetSafeContextType(cat.as<uint32_t>());
      } else if (cat.isString()) {
        ev.category = GetContextTypeFromString(cat.as<std::string>());
      }
    }
    if (event.hasOwnProperty("id")) {
      if (event["id"].isString()) {
        ev.id = event["id"].as<std::string>();
      } else {
        ev.id = std::to_string((int64_t)event["id"].as<double>());
      }
    } else if (event.hasOwnProperty("bind_id")) {
      if (event["bind_id"].isString()) {
        ev.id = event["bind_id"].as<std::string>();
      } else {
        ev.id = std::to_string((int64_t)event["bind_id"].as<double>());
      }
    }
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
    // We use Fingerprint64 for a stable event ID because absl::HashOf
    // does not guarantee stability across different executions or binaries,
    // and we need consistency for event associations.
    ev.event_id =
        tsl::Fingerprint64(absl::StrCat(ev.name, ":", ev.ts, ":", ev.dur));
    switch (ev.ph) {
      case Phase::kFlowStart:
      case Phase::kFlowEnd:
        if (!ev.id.empty()) {
          result.flow_events.push_back(std::move(ev));
        }
        break;
      case Phase::kComplete:
        if (!ev.id.empty()) {
          result.flow_events.push_back(ev);
        }
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
  if (!trace_data.hasOwnProperty("traceEvents")) {
    return result;
  }

  if (trace_data.hasOwnProperty("mpmdPipelineView")) {
    result.mpmd_pipeline_view = trace_data["mpmdPipelineView"].as<bool>();
  }

  emscripten::val events = trace_data["traceEvents"];
  const std::vector<emscripten::val> js_events =
      emscripten::vecFromJSArray<emscripten::val>(events);
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
  const ParsedTraceEvents parsed_events =
      ParseTraceEvents(trace_data, visible_range_from_url);
  Application::Instance().data_provider().ProcessTraceEvents(
      parsed_events, Application::Instance().timeline());

  // Reset the loading flag to allow subsequent data requests (e.g. on panning).
  // This is necessary because is_incremental_loading_ is initialized to true to
  // prevent duplicate requests during the initial load.
  Application::Instance().timeline().set_is_incremental_loading(false);
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

  // Bind DataProvider class
  emscripten::class_<traceviewer::DataProvider>("DataProvider")
      .function("getFlowCategories",
                &traceviewer::DataProvider::GetFlowCategories);

  emscripten::function("processTraceEvents",
                       &traceviewer::ParseAndProcessTraceEvents);
  emscripten::function("getAllFlowCategories",
                       &traceviewer::GetAllFlowCategories);

  // Bind Application class and expose the singleton instance and data_provider
  emscripten::class_<traceviewer::Application>("Application")
      .class_function("Instance", &traceviewer::Application::Instance,
                      emscripten::return_value_policy::reference())
      .function("data_provider", &traceviewer::Application::data_provider,
                emscripten::return_value_policy::reference())
      .function("setVisibleFlowCategory",
                &traceviewer::Application::SetVisibleFlowCategory)
      .function("Resize", &traceviewer::Application::Resize);
}

}  // namespace traceviewer
