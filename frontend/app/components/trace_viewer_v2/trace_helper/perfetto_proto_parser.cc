#include "frontend/app/components/trace_viewer_v2/trace_helper/perfetto_proto_parser.h"

#include <emscripten/bind.h>
#include <emscripten/val.h>

#include <algorithm>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "tsl/profiler/lib/context_types.h"
#include "frontend/app/components/trace_viewer_v2/application.h"
#include "frontend/app/components/trace_viewer_v2/event_data.h"
#include "frontend/app/components/trace_viewer_v2/event_manager.h"
#include "frontend/app/components/trace_viewer_v2/helper/time_formatter.h"
#include "frontend/app/components/trace_viewer_v2/timeline/data_provider.h"
#include "frontend/app/components/trace_viewer_v2/timeline/timeline.h"
#include "frontend/app/components/trace_viewer_v2/trace_helper/trace_event.h"
#include "frontend/app/components/trace_viewer_v2/trace_helper/trace_lite.pb.h"

namespace traceviewer {

ParsedTraceEvents ParsePerfettoTraceEvents(const std::string& buffer_data) {
  ParsedTraceEvents parsed_events;

  xprof::traceviewer::protos::Trace trace;

  if (!trace.ParseFromString(buffer_data)) {
    parsed_events.parsing_status = ParsingStatus::kFailed;
    return parsed_events;
  }

  std::map<uint32_t, std::map<uint64_t, std::string>> interned_strings;
  std::map<uint32_t, std::map<uint64_t, std::string>> interned_categories;
  std::map<uint64_t, xprof::traceviewer::protos::TrackDescriptor>
      track_descriptors;
  std::map<uint64_t, std::vector<TraceEvent>> slice_stacks;

  for (const auto& packet : trace.packet()) {
    uint32_t seq_id = packet.trusted_packet_sequence_id();
    if (packet.incremental_state_cleared()) {
      interned_strings[seq_id].clear();
      interned_categories[seq_id].clear();
    }

    if (packet.has_interned_data()) {
      const auto& interned = packet.interned_data();
      for (const auto& str : interned.event_names()) {
        interned_strings[seq_id][str.iid()] = str.name();
      }
      for (const auto& cat : interned.event_categories()) {
        interned_categories[seq_id][cat.iid()] = cat.name();
      }
    }

    if (packet.has_track_descriptor()) {
      const auto& desc = packet.track_descriptor();
      track_descriptors[desc.uuid()] = desc;

      if (desc.has_process()) {
        const auto& process = desc.process();

        TraceEvent event;
        event.ph = Phase::kMetadata;
        event.name = std::string(kProcessName);
        event.pid = process.pid();
        event.args["name"] = process.process_name();
        parsed_events.flame_events.push_back(event);
      }
      if (desc.has_thread()) {
        const auto& thread = desc.thread();

        TraceEvent event;
        event.ph = Phase::kMetadata;
        event.name = std::string(kThreadName);
        event.pid = thread.pid();
        event.tid = thread.tid();
        event.args["name"] = thread.thread_name();
        parsed_events.flame_events.push_back(event);
      }
    }

    if (packet.has_track_event()) {
      const auto& track_event = packet.track_event();
      uint64_t track_uuid = track_event.track_uuid();

      std::string name;
      if (track_event.has_name_iid()) {
        auto seq_it = interned_strings.find(seq_id);
        if (seq_it != interned_strings.end()) {
          auto name_it = seq_it->second.find(track_event.name_iid());
          if (name_it != seq_it->second.end()) {
            name = name_it->second;
          } else {
            name = std::string(kUnknown);
          }
        } else {
          name = std::string(kUnknown);
        }
      } else if (track_event.has_name()) {
        name = track_event.name();
      }

      // TODO: This assumes nanoseconds. While this is the default for most
      // Perfetto producers, it might be worth adding a TODO to handle
      // `ClockSnapshot` or `timestamp_clock_id` for robustness in the future.
      Microseconds ts = packet.timestamp() / 1000.0;  // ns to us

      if (track_event.type() ==
          xprof::traceviewer::protos::TrackEvent::TYPE_SLICE_BEGIN) {
        TraceEvent event;
        event.ph = Phase::kComplete;
        event.name = name;
        event.ts = ts;
        event.category = tsl::profiler::ContextType::kGeneric;

        auto it = track_descriptors.find(track_uuid);
        if (it != track_descriptors.end()) {
          const auto& desc = it->second;
          if (desc.has_thread()) {
            event.tid = desc.thread().tid();
            event.pid = desc.thread().pid();
          } else if (desc.has_process()) {
            event.pid = desc.process().pid();
          }
        }

        slice_stacks[track_uuid].push_back(event);
      } else if (track_event.type() ==
                 xprof::traceviewer::protos::TrackEvent::TYPE_SLICE_END) {
        auto& stack = slice_stacks[track_uuid];
        if (!stack.empty()) {
          TraceEvent event = stack.back();
          stack.pop_back();
          event.dur = std::max(0.0, ts - event.ts);
          parsed_events.flame_events.push_back(event);
        }
      }
    }
  }

  return parsed_events;
}

void ParseAndProcessPerfettoTraceEvents(
    const std::string& buffer_data,
    const emscripten::val& visible_range_from_url) {
  ParsedTraceEvents parsed_events = ParsePerfettoTraceEvents(buffer_data);

  if (parsed_events.parsing_status == ParsingStatus::kFailed) {
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
