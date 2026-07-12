#include "frontend/app/components/trace_viewer_v2/trace_helper/trace_pb_event_parser.h"

#include <emscripten/bind.h>
#include <emscripten/val.h>

#include <string>
#include <utility>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
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
    uintptr_t data_ptr, size_t data_size,
    const emscripten::val& visible_range_from_url) {
  absl::string_view buffer_data(reinterpret_cast<const char*>(data_ptr),
                                data_size);
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
  // Free decompressed proto memory early to reduce peak memory usage.
  std::string().swap(*decompressed_proto);

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
    uintptr_t data_ptr, size_t data_size,
    const emscripten::val& visible_range_from_url, DataProvider& data_provider,
    Timeline& timeline) {
  const ParsedTraceEvents parsed_events =
      ParseCompressedTraceEvents(data_ptr, data_size, visible_range_from_url);
  data_provider.ProcessTraceEvents(parsed_events, timeline);

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
  timeline.RequestRedraw();
}

void ParseAndProcessCompressedTraceEvents(
    uintptr_t data_ptr, size_t data_size,
    const emscripten::val& visible_range_from_url) {
  ParseAndProcessCompressedTraceEvents(data_ptr, data_size,
                                       visible_range_from_url,
                                       Application::Instance().data_provider(),
                                       Application::Instance().timeline());
}

void SetCompressedSearchResultsInWasm(uintptr_t data_ptr, size_t data_size) {
  if (!Application::Instance().IsInitialized()) {
    return;
  }
  // Ownership of data_ptr remains with the JS caller (see wasm_string_utils.ts).
  const ParsedTraceEvents parsed_events =
      ParseCompressedTraceEvents(data_ptr, data_size, emscripten::val::null());
  Application::Instance().timeline().SetSearchResults(parsed_events);
  Application::Instance().RequestRedraw();
}

EMSCRIPTEN_BINDINGS(trace_pb_event_parser) {
  emscripten::function(
      "processCompressedTraceEvents",
      static_cast<void (*)(uintptr_t, size_t, const emscripten::val&)>(
          &traceviewer::ParseAndProcessCompressedTraceEvents));
  emscripten::function("setCompressedSearchResultsInWasm",
                       &traceviewer::SetCompressedSearchResultsInWasm);
}

}  // namespace traceviewer
