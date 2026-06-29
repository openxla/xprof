#include "frontend/app/components/trace_viewer_v2/trace_helper/trace_pb_event_parser.h"

#include <emscripten/bind.h>
#include <emscripten/val.h>

#include <cstddef>
#include <cstdint>
#include <optional>
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

void ParseAndProcessCompressedTraceEvents(
    uintptr_t data_ptr, size_t data_size,
    const emscripten::val& visible_range_from_url, DataProvider& data_provider,
    Timeline& timeline) {
  if (data_ptr == 0 || data_size == 0) {
    return;
  }
  absl::string_view buffer_data(reinterpret_cast<const char*>(data_ptr),
                                data_size);
  absl::StatusOr<std::string> decompressed_proto =
      tensorflow::profiler::ZstdCompression::Decompress(buffer_data);
  if (!decompressed_proto.ok()) {
    return;
  }

  xprof::TraceDataResponse response;
  if (!response.ParseFromString(*decompressed_proto)) {
    return;
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

  std::optional<std::pair<Milliseconds, Milliseconds>> parsed_visible_range;
  if (!visible_range_from_url.isNull() &&
      !visible_range_from_url.isUndefined() &&
      visible_range_from_url["length"].as<int>() == 2) {
    Milliseconds start = visible_range_from_url[0].as<Milliseconds>();
    Milliseconds end = visible_range_from_url[1].as<Milliseconds>();
    parsed_visible_range = std::make_pair(start, end);
  }

  data_provider.ProcessTraceEvents(response, timeline, parsed_visible_range);

  if (parsed_visible_range.has_value()) {
    timeline.InitializeLastFetchRequestRange(
        {MillisToMicros(parsed_visible_range->first),
         MillisToMicros(parsed_visible_range->second)});
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

EMSCRIPTEN_BINDINGS(trace_pb_event_parser) {
  emscripten::function(
      "processCompressedTraceEvents",
      static_cast<void (*)(uintptr_t, size_t, const emscripten::val&)>(
          &traceviewer::ParseAndProcessCompressedTraceEvents));
}

}  // namespace traceviewer
