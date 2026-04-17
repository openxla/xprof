#include "frontend/app/components/trace_viewer_v2/trace_helper/trace_pb_event_parser.h"

#include <emscripten/bind.h>
#include <emscripten/val.h>

#include <algorithm>
#include <cstdint>
#include <string>
#include <utility>

#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "tsl/platform/fingerprint.h"
#include "tsl/profiler/lib/context_types.h"
#include "xprof/convert/trace_viewer/delta_series/zstd_compression.h"
#include "frontend/app/components/trace_viewer_v2/application.h"
#include "frontend/app/components/trace_viewer_v2/helper/time_formatter.h"
#include "frontend/app/components/trace_viewer_v2/timeline/data_provider.h"
#include "frontend/app/components/trace_viewer_v2/timeline/timeline.h"
#include "frontend/app/components/trace_viewer_v2/trace_helper/trace_event.h"
#include "frontend/app/components/trace_viewer_v2/trace_helper/trace_event_parser.h"
#include "plugin/xprof/protobuf/trace_data_response.pb.h"

namespace traceviewer {

namespace {

void ProcessMetadataEvents(const xprof::TraceDataResponse& response,
                           ParsedTraceEvents& result) {
  for (const auto& process : response.metadata().processes()) {
    TraceEvent process_ev;
    process_ev.ph = Phase::kMetadata;
    process_ev.pid = process.id();
    process_ev.name = kProcessName;
    process_ev.args[std::string(kName)] = process.name();
    result.flame_events.push_back(std::move(process_ev));

    if (process.sort_index() != 0) {
      TraceEvent process_sort_ev;
      process_sort_ev.ph = Phase::kMetadata;
      process_sort_ev.pid = process.id();
      process_sort_ev.name = kProcessSortIndex;
      process_sort_ev.args[std::string(kSortIndex)] =
          std::to_string(process.sort_index());
      result.flame_events.push_back(std::move(process_sort_ev));
    }

    for (const auto& thread : process.threads()) {
      TraceEvent thread_ev;
      thread_ev.ph = Phase::kMetadata;
      thread_ev.pid = process.id();
      thread_ev.tid = thread.id();
      thread_ev.name = kThreadName;
      thread_ev.args[std::string(kName)] = thread.name();
      result.flame_events.push_back(std::move(thread_ev));
    }
  }
}

void ProcessCompleteEvents(const xprof::TraceDataResponse& response,
                           ParsedTraceEvents& result) {
  for (const auto& series : response.complete_events()) {
    const auto& metadata = series.metadata();
    uint64_t current_ts_ps = 0;
    for (int i = 0; i < series.deltas_size(); ++i) {
      current_ts_ps += series.deltas(i);
      TraceEvent ev;
      ev.ph = Phase::kComplete;
      ev.pid = metadata.process_id();
      ev.tid = metadata.thread_id();
      ev.ts = current_ts_ps / 1000000.0;
      ev.dur = series.durations(i) / 1000000.0;
      ev.name = response.interned_strings(series.name_refs(i));

      const auto& ev_meta = series.event_metadata(i);
      if (ev_meta.flow_id() != 0) {
        ev.id = std::to_string(ev_meta.flow_id());
        if (ev_meta.flow_category() != 0) {
          ev.category = GetContextTypeFromString(
              response.interned_strings(ev_meta.flow_category()));
        }
      }

      ev.args["uid"] = std::to_string(ev_meta.serial());
      for (const auto& [key_id, val_id] : ev_meta.args()) {
        ev.args[response.interned_strings(key_id)] =
            response.interned_strings(val_id);
      }
      ev.event_id =
          tsl::Fingerprint64(absl::StrCat(ev.name, ":", ev.ts, ":", ev.dur));

      if (!ev.id.empty()) {
        result.flow_events.push_back(ev);
      }
      result.flame_events.push_back(std::move(ev));
    }
  }
}

void ProcessAsyncEvents(const xprof::TraceDataResponse& response,
                        ParsedTraceEvents& result) {
  struct AsyncKey {
    ProcessId pid;
    std::string id;
    bool operator<(const AsyncKey& o) const {
      if (pid != o.pid) return pid < o.pid;
      return id < o.id;
    }
  };
  std::map<AsyncKey, TraceEvent> open_async_events;

  for (const auto& series : response.async_events()) {
    const auto& metadata = series.metadata();
    uint64_t current_ts_ps = 0;
    for (int i = 0; i < series.deltas_size(); ++i) {
      current_ts_ps += series.deltas(i);
      const auto& ev_meta = series.event_metadata(i);
      if (ev_meta.flow_id() != 0) {
        std::string flow_id_str = std::to_string(ev_meta.flow_id());

        TraceEvent ev;
        ev.pid = metadata.process_id();
        ev.id = flow_id_str;
        ev.ts = current_ts_ps / 1000000.0;
        ev.name = response.interned_strings(series.metadata().name_ref());

        if (ev_meta.flow_category() != 0) {
          ev.category = GetContextTypeFromString(
              response.interned_strings(ev_meta.flow_category()));
        }

        ev.args["uid"] = std::to_string(ev_meta.serial());
        if (ev_meta.group_id() != 0) {
          ev.args["group_id"] = std::to_string(ev_meta.group_id());
        }
        for (const auto& [key_id, val_id] : ev_meta.args()) {
          ev.args[response.interned_strings(key_id)] =
              response.interned_strings(val_id);
        }

        auto flow_entry_type = ev_meta.flow_entry_type();
        if (flow_entry_type == xprof::TraceEventMetadata::FLOW_START) {
          ev.ph = Phase::kAsyncBegin;
          open_async_events[{ev.pid, ev.id}] = std::move(ev);
        } else if (flow_entry_type == xprof::TraceEventMetadata::FLOW_END) {
          auto it = open_async_events.find({ev.pid, ev.id});
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
            begin_ev.event_id = tsl::Fingerprint64(absl::StrCat(
                begin_ev.name, ":", begin_ev.ts, ":", begin_ev.dur));
            result.flame_events.push_back(std::move(begin_ev));
            open_async_events.erase(it);
          }
        } else if (flow_entry_type == xprof::TraceEventMetadata::FLOW_MID ||
                   (flow_entry_type == xprof::TraceEventMetadata::FLOW_NONE &&
                    series.durations(i) > 0)) {
          ev.ph = Phase::kComplete;
          ev.is_async = true;
          ev.dur = series.durations(i) / 1000000.0;
          ev.event_id = tsl::Fingerprint64(
              absl::StrCat(ev.name, ":", ev.ts, ":", ev.dur));

          if (flow_entry_type == xprof::TraceEventMetadata::FLOW_MID) {
            // Replicate old behavior: erase pending start event with same ID.
            open_async_events.erase({ev.pid, ev.id});
          }

          result.flame_events.push_back(std::move(ev));
        }
      }
    }
  }
}

void ProcessCounterEvents(const xprof::TraceDataResponse& response,
                          ParsedTraceEvents& result) {
  for (const auto& series : response.counter_events()) {
    const auto& metadata = series.metadata();
    CounterEvent ev;
    ev.pid = metadata.process_id();
    ev.name = response.interned_strings(metadata.name_ref());
    uint64_t current_ts_ps = 0;
    for (int i = 0; i < series.deltas_size(); ++i) {
      current_ts_ps += series.deltas(i);
      ev.timestamps.push_back(current_ts_ps / 1000000.0);
      double val = 0.0;
      const auto& ev_meta = series.event_metadata(i);
      if (ev_meta.has_counter_value_double()) {
        val = ev_meta.counter_value_double();
      } else if (ev_meta.has_counter_value_uint64()) {
        val = static_cast<double>(ev_meta.counter_value_uint64());
      }
      ev.values.push_back(val);
      ev.min_value = std::min(ev.min_value, val);
      ev.max_value = std::max(ev.max_value, val);
    }
    result.counter_events.push_back(std::move(ev));
  }
}

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
  ParseAndProcessCompressedTraceEvents(buffer_data, emscripten::val::null());
}

EMSCRIPTEN_BINDINGS(trace_pb_event_parser) {
  emscripten::function("processCompressedTraceEvents",
                       &traceviewer::ParseAndProcessCompressedTraceEvents);
  emscripten::function("setCompressedSearchResultsInWasm",
                       &traceviewer::SetCompressedSearchResultsInWasm);
}

}  // namespace traceviewer
