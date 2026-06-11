#include "frontend/app/components/trace_viewer_v2/trace_helper/trace_pb_event_parser.h"

#include <emscripten/bind.h>
#include <emscripten/val.h>

#include <string>
#include <utility>

#include "absl/status/statusor.h"
#include "xprof/convert/trace_viewer/delta_series/zstd_compression.h"
#include "frontend/app/components/trace_viewer_v2/application.h"
#include "frontend/app/components/trace_viewer_v2/helper/time_formatter.h"
#include "frontend/app/components/trace_viewer_v2/timeline/data_provider.h"
#include "frontend/app/components/trace_viewer_v2/timeline/timeline.h"
#include "frontend/app/components/trace_viewer_v2/trace_helper/trace_event.h"
#include "frontend/app/components/trace_viewer_v2/trace_helper/trace_event_parser_core.h"
#include "plugin/xprof/protobuf/trace_data_response.pb.h"

namespace traceviewer {

namespace {

ParsedTraceEvents ParseCompressedTraceEvents(
    const std::string& buffer_data,
    const emscripten::val& visible_range_from_url) {
  ParsedTraceEvents result;
  absl::StatusOr<std::string> decompressed_proto =
      tensorflow::profiler::ZstdCompression::Decompress(buffer_data);
  if (!decompressed_proto.ok()) {
    return result;
  }

  xprof::TraceDataResponse response;
  if (!response.ParseFromString(*decompressed_proto)) {
    return result;
  }

  if (response.details_size() > 0) {
    emscripten::val details_map = emscripten::val::global("Map").new_();
    for (const auto& detail : response.details()) {
      details_map.call<void>("set", emscripten::val(detail.name()),
                             emscripten::val(detail.value()));
    }
    emscripten::val detail_obj = emscripten::val::object();
    detail_obj.set("details", details_map);

    emscripten::val event_init = emscripten::val::object();
    event_init.set("detail", detail_obj);

    emscripten::val event =
        emscripten::val::global("CustomEvent")
            .new_(emscripten::val("details_received"), event_init);
    emscripten::val::global("window").call<void>("dispatchEvent", event);
  }
  ProcessMetadataEvents(response, result);
  ProcessCompleteEvents(response, result);
  ProcessAsyncEvents(response, result);
  ProcessCounterEvents(response, result);

  if (response.has_full_timespan_start_ps() &&
      response.has_full_timespan_end_ps()) {
    const Milliseconds start_ms = response.full_timespan_start_ps() / 1e9;
    const Milliseconds end_ms = response.full_timespan_end_ps() / 1e9;
    if (start_ms <= end_ms) {
      result.full_timespan = std::make_pair(start_ms, end_ms);
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
}  // namespace

void ParseAndProcessCompressedTraceEvents(
    const std::string& buffer_data,
    const emscripten::val& visible_range_from_url) {
  const ParsedTraceEvents parsed_events =
      ParseCompressedTraceEvents(buffer_data, visible_range_from_url);
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

void SetCompressedSearchResultsInWasm(const std::string& buffer_data) {
  if (Application::Instance().IsInitialized()) {
    const ParsedTraceEvents parsed_events =
        ParseCompressedTraceEvents(buffer_data, emscripten::val::null());
    Application::Instance().timeline().SetSearchResults(parsed_events);
    Application::Instance().RequestRedraw();
  }
}

EMSCRIPTEN_BINDINGS(trace_pb_event_parser) {
  emscripten::function("processCompressedTraceEvents",
                       &traceviewer::ParseAndProcessCompressedTraceEvents);
  emscripten::function("setCompressedSearchResultsInWasm",
                       &traceviewer::SetCompressedSearchResultsInWasm);
}

}  // namespace traceviewer
