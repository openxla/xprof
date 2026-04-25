#include "frontend/app/components/trace_viewer_v2/trace_helper/perfetto_proto_parser.h"

#include <emscripten/bind.h>
#include <emscripten/val.h>

#include <string>

#include "frontend/app/components/trace_viewer_v2/application.h"
#include "frontend/app/components/trace_viewer_v2/event_data.h"
#include "frontend/app/components/trace_viewer_v2/event_manager.h"
#include "frontend/app/components/trace_viewer_v2/helper/time_formatter.h"
#include "frontend/app/components/trace_viewer_v2/timeline/data_provider.h"
#include "frontend/app/components/trace_viewer_v2/timeline/timeline.h"
#include "frontend/app/components/trace_viewer_v2/trace_helper/trace_event.h"
#include "frontend/app/components/trace_viewer_v2/trace_helper/trace_lite.pb.h"

namespace traceviewer {

void ParseAndProcessPerfettoTraceEvents(
    const std::string& buffer_data,
    const emscripten::val& visible_range_from_url) {
  xprof::traceviewer::protos::Trace trace;

  if (!trace.ParseFromString(buffer_data)) {
    EventData event_data;
    event_data.try_emplace("message",
                           std::string("Failed to parse Perfetto trace"));
    EventManager::Instance().DispatchEvent("trace_parsing_error", event_data);
    return;
  }

  if (!Application::Instance().IsInitialized()) {
    EventData event_data;
    event_data.try_emplace(
        "message",
        std::string("Application is not initialized when parsing Perfetto "
                    "trace"));
    EventManager::Instance().DispatchEvent("trace_parsing_error", event_data);
    return;
  }

  ParsedTraceEvents parsed_events;

  Application::Instance().data_provider().ProcessTraceEvents(
      parsed_events, Application::Instance().timeline());

  Timeline& timeline = Application::Instance().timeline();
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

  timeline.set_is_incremental_loading(false);
  Application::Instance().RequestRedraw();
}

EMSCRIPTEN_BINDINGS(perfetto_proto_parser) {
  emscripten::function("processPerfettoTraceEvents",
                       &traceviewer::ParseAndProcessPerfettoTraceEvents);
}

}  // namespace traceviewer
