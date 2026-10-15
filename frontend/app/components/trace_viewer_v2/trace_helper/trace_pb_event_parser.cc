#include "frontend/app/components/trace_viewer_v2/trace_helper/trace_pb_event_parser.h"

#include <emscripten/bind.h>
#include <emscripten/val.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/no_destructor.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xprof/convert/trace_viewer/delta_series/zstd_compression.h"
#include "frontend/app/components/trace_viewer_v2/application.h"
#include "frontend/app/components/trace_viewer_v2/helper/time_formatter.h"
#include "frontend/app/components/trace_viewer_v2/timeline/data_provider.h"
#include "frontend/app/components/trace_viewer_v2/timeline/timeline.h"
#include "frontend/app/components/trace_viewer_v2/trace_helper/trace_event.h"
#include "frontend/app/components/trace_viewer_v2/trace_helper/trace_stream_ingestor.h"
#include "plugin/xprof/protobuf/trace_data_response.pb.h"

namespace traceviewer {
namespace {

// Mirrors head's flag logic in ParseAndProcessCompressedTraceEvents.
void SetCompactionFlagsForLoad(bool is_initial_load, bool has_url_range,
                               Timeline& timeline) {
  const bool is_overview_load =
      is_initial_load && !has_url_range && timeline.incremental_fetch_enabled();
  timeline.set_full_trace_loaded(!timeline.incremental_fetch_enabled() &&
                                 !has_url_range);
  timeline.set_is_compacted(is_overview_load);
  timeline.set_window_at_full_resolution(!is_overview_load);
}

StreamingTraceIngestor& GlobalStreamIngestor() {
  static absl::NoDestructor<StreamingTraceIngestor> ingestor;
  return *ingestor;
}

std::optional<std::pair<Milliseconds, Milliseconds>> ParseVisibleRangeVal(
    const emscripten::val& visible_range_from_url) {
  if (!visible_range_from_url.isNull() &&
      !visible_range_from_url.isUndefined() &&
      visible_range_from_url["length"].as<int>() == 2) {
    const Milliseconds start = visible_range_from_url[0].as<Milliseconds>();
    const Milliseconds end = visible_range_from_url[1].as<Milliseconds>();
    return std::make_pair(start, end);
  }
  return std::nullopt;
}

void DispatchDetailsReceivedEvent(
    const std::vector<std::pair<std::string, bool>>& details) {
  if (details.empty()) return;
  emscripten::val details_map = emscripten::val::global("Map").new_();
  for (const auto& [name, value] : details) {
    details_map.call<void>("set", emscripten::val(name),
                           emscripten::val(value));
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

void FinalizeSuccessfulLoad(
    bool is_initial_load,
    const std::optional<std::pair<Milliseconds, Milliseconds>>&
        parsed_visible_range,
    const std::vector<std::pair<std::string, bool>>& details,
    Timeline& timeline) {
  DispatchDetailsReceivedEvent(details);

  if (is_initial_load) {
    if (parsed_visible_range.has_value()) {
      timeline.InitializeLastFetchRequestRange(
          {MillisToMicros(parsed_visible_range->first),
           MillisToMicros(parsed_visible_range->second)});
    } else {
      timeline.InitializeLastFetchRequestRange(timeline.data_time_range());
    }
  }

  timeline.set_is_incremental_loading(false);
  timeline.RequestRedraw();
}

}  // namespace

void ParseAndProcessCompressedTraceEvents(
    uintptr_t data_ptr, size_t data_size,
    const emscripten::val& visible_range_from_url, DataProvider& data_provider,
    Timeline& timeline, bool free_data_ptr) {
  const auto free_input_if_owned = [&]() {
    if (free_data_ptr && data_ptr != 0) {
      std::free(reinterpret_cast<void*>(data_ptr));
      data_ptr = 0;
    }
  };
  const auto finish_empty = [&]() {
    free_input_if_owned();
    if (!timeline.is_incremental_loading()) {
      data_provider.ProcessTraceEvents(xprof::TraceDataResponse{}, timeline,
                                       std::nullopt);
    }
    timeline.set_is_incremental_loading(false);
    timeline.RequestRedraw();
  };
  if (data_ptr == 0 || data_size == 0) {
    finish_empty();
    return;
  }
  absl::string_view buffer_data(reinterpret_cast<const char*>(data_ptr),
                                data_size);
  absl::StatusOr<std::string> decompressed_proto =
      tensorflow::profiler::ZstdCompression::Decompress(buffer_data);
  free_input_if_owned();
  if (!decompressed_proto.ok()) {
    finish_empty();
    return;
  }

  const bool is_initial_load =
      (timeline.last_fetch_request_range() == TimeRange::Zero());
  const std::optional<std::pair<Milliseconds, Milliseconds>>
      parsed_visible_range = ParseVisibleRangeVal(visible_range_from_url);

  SetCompactionFlagsForLoad(is_initial_load, parsed_visible_range.has_value(),
                            timeline);
  std::vector<std::pair<std::string, bool>> details;
  if (!data_provider.ProcessSerializedTraceEvents(
          *std::move(decompressed_proto), timeline, parsed_visible_range,
          &details)) {
    finish_empty();
    return;
  }

  FinalizeSuccessfulLoad(is_initial_load, parsed_visible_range, details,
                         timeline);
}

void ParseAndProcessCompressedTraceEvents(
    uintptr_t data_ptr, size_t data_size,
    const emscripten::val& visible_range_from_url, bool free_data_ptr) {
  ParseAndProcessCompressedTraceEvents(
      data_ptr, data_size, visible_range_from_url,
      Application::Instance().data_provider(),
      Application::Instance().timeline(), free_data_ptr);
}

void BeginTraceStream(uint32_t stream_id) {
  GlobalStreamIngestor().Begin(stream_id);
}

bool PushTraceStreamBytes(uintptr_t data_ptr, size_t data_size) {
  if (data_size == 0) return true;
  if (data_ptr == 0) {
    GlobalStreamIngestor().Abort();
    return false;
  }
  const absl::Status status =
      GlobalStreamIngestor().PushBytes(absl::MakeConstSpan(
          reinterpret_cast<const uint8_t*>(data_ptr), data_size));
  return status.ok();
}

bool FinishTraceStream(const emscripten::val& visible_range_from_url,
                       DataProvider& data_provider, Timeline& timeline) {
  const bool is_initial_load =
      (timeline.last_fetch_request_range() == TimeRange::Zero());
  const std::optional<std::pair<Milliseconds, Milliseconds>>
      parsed_visible_range = ParseVisibleRangeVal(visible_range_from_url);

  SetCompactionFlagsForLoad(is_initial_load, parsed_visible_range.has_value(),
                            timeline);
  std::vector<std::pair<std::string, bool>> details;
  const absl::Status status = GlobalStreamIngestor().Finish(
      data_provider, timeline, parsed_visible_range, &details);
  if (!status.ok()) {
    return false;
  }

  FinalizeSuccessfulLoad(is_initial_load, parsed_visible_range, details,
                         timeline);
  return true;
}

bool FinishTraceStream(const emscripten::val& visible_range_from_url) {
  return FinishTraceStream(visible_range_from_url,
                           Application::Instance().data_provider(),
                           Application::Instance().timeline());
}

void AbortTraceStream() { GlobalStreamIngestor().Abort(); }

EMSCRIPTEN_BINDINGS(trace_pb_event_parser) {
  emscripten::function(
      "processCompressedTraceEvents",
      +[](uintptr_t data_ptr, size_t data_size,
          const emscripten::val& visible_range_from_url) {
        traceviewer::ParseAndProcessCompressedTraceEvents(
            data_ptr, data_size, visible_range_from_url,
            /*free_data_ptr=*/true);
      });
  emscripten::function("beginTraceStream", &traceviewer::BeginTraceStream);
  emscripten::function("pushTraceStreamBytes",
                       &traceviewer::PushTraceStreamBytes);
  emscripten::function(
      "finishTraceStream", +[](const emscripten::val& visible_range_from_url) {
        return traceviewer::FinishTraceStream(visible_range_from_url);
      });
  emscripten::function("abortTraceStream", &traceviewer::AbortTraceStream);
}

}  // namespace traceviewer
