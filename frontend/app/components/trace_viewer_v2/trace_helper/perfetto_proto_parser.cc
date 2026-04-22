#include "frontend/app/components/trace_viewer_v2/trace_helper/perfetto_proto_parser.h"

#include <emscripten/bind.h>
#include <emscripten/val.h>

#include <algorithm>
#include <cstddef>
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
#include "frontend/app/components/trace_viewer_v2/trace_helper/perfetto_constants.h"
#include "frontend/app/components/trace_viewer_v2/trace_helper/trace_event.h"
#include "frontend/app/components/trace_viewer_v2/trace_helper/trace_lite.pb.h"

namespace traceviewer {
namespace {

// Resolves the PID and TID for a TraceEvent based on the track descriptors.
void ResolvePidTid(
    TraceEvent& event, uint64_t track_uuid, uint32_t seq_id,
    const std::map<uint64_t, xprof::traceviewer::protos::TrackDescriptor>&
        track_descriptors,
    ProcessId fallback_pid) {
  auto it = track_descriptors.find(track_uuid);
  if (it != track_descriptors.end()) {
    const auto& desc = it->second;
    if (desc.has_thread()) {
      event.tid = desc.thread().tid();
      event.pid = desc.thread().pid();
    } else if (desc.has_process()) {
      event.pid = desc.process().pid();
    } else if (!desc.name().empty()) {
      event.tid = desc.uuid();
    }
  }

  // Fallback if pid/tid are still not resolved
  if (event.pid == 0) {
    event.pid = fallback_pid;
  }
  if (event.tid == 0) {
    event.tid = kPerfettoFallbackTidOffset + seq_id;
  }
}

}  // namespace

ParsedTraceEvents ParsePerfettoTraceEvents(const void* data, size_t size,
                                           bool normalize_timestamps) {
  ParsedTraceEvents parsed_events;

  xprof::traceviewer::protos::Trace trace;

  if (!trace.ParseFromArray(data, size)) {
    parsed_events.parsing_failed = true;
    return parsed_events;
  }

  std::map<uint32_t, std::map<uint64_t, std::string>> interned_strings;
  std::map<uint32_t, std::map<uint64_t, std::string>> interned_categories;
  std::map<uint64_t, xprof::traceviewer::protos::TrackDescriptor>
      track_descriptors;
  std::map<uint64_t, std::string> named_tracks;
  ProcessId fallback_pid = kPerfettoFallbackPid;

  // Pass 1: Collect metadata and find a fallback PID.
  Microseconds min_ts = std::numeric_limits<double>::max();
  bool found_any_timestamp = false;

  auto update_min_ts = [&](Microseconds ts) {
    if (ts > 0 && ts < min_ts) {
      min_ts = ts;
      found_any_timestamp = true;
    }
  };

  for (const auto& packet : trace.packet()) {
    if (packet.has_track_descriptor()) {
      const auto& desc = packet.track_descriptor();
      track_descriptors[desc.uuid()] = desc;

      if (desc.has_process()) {
        const auto& process = desc.process();
        if (fallback_pid == kPerfettoFallbackPid) {
          fallback_pid = process.pid();
        }

        TraceEvent event;
        event.ph = Phase::kMetadata;
        event.name = std::string(kProcessName);
        event.pid = process.pid();
        event.args["name"] = process.process_name();
        parsed_events.flame_events.push_back(event);
      } else if (desc.has_thread()) {
        // Use else if to avoid double counting if a descriptor somehow has both
        // process and thread (though unlikely in Perfetto).
        const auto& thread = desc.thread();

        TraceEvent event;
        event.ph = Phase::kMetadata;
        event.name = std::string(kThreadName);
        event.pid = thread.pid();
        event.tid = thread.tid();
        event.args["name"] = thread.thread_name();
        parsed_events.flame_events.push_back(event);
      } else if (!desc.name().empty()) {
        // Collect named tracks for later metadata emission.
        named_tracks[desc.uuid()] = desc.name();
      }
    }
  }

  // Emit metadata events for named tracks using the determined fallback_pid.
  for (const auto& pair : named_tracks) {
    TraceEvent event;
    event.ph = Phase::kMetadata;
    event.name = std::string(kThreadName);
    event.pid = fallback_pid;
    event.tid = pair.first;  // use uuid as tid
    event.args["name"] = pair.second;
    parsed_events.flame_events.push_back(event);
  }

  // Pass 2: Process track events.
  std::map<uint64_t, std::vector<TraceEvent>> slice_stacks;
  std::map<uint32_t, uint64_t> sequence_timestamps;

  for (const auto& packet : trace.packet()) {
    uint32_t seq_id = packet.trusted_packet_sequence_id();
    if (packet.incremental_state_cleared()) {
      interned_strings[seq_id].clear();
      interned_categories[seq_id].clear();
      sequence_timestamps[seq_id] = 0;
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

    if (!packet.has_track_event()) continue;

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
    uint64_t timestamp_ns = sequence_timestamps[seq_id];
    if (packet.has_timestamp()) {
      timestamp_ns = packet.timestamp();
    }

    if (track_event.has_timestamp_delta_us()) {
      timestamp_ns += track_event.timestamp_delta_us() * 1000;
    } else if (track_event.has_timestamp_absolute_us()) {
      timestamp_ns = track_event.timestamp_absolute_us() * 1000;
    }

    sequence_timestamps[seq_id] = timestamp_ns;
    Microseconds ts = timestamp_ns / 1000.0;  // ns to us

    if (track_event.type() ==
        xprof::traceviewer::protos::TrackEvent::TYPE_SLICE_BEGIN) {
      TraceEvent event;
      event.ph = Phase::kComplete;
      event.name = name.empty() ? std::string(kUnknown) : name;
      event.ts = ts;
      update_min_ts(ts);
      event.category = tsl::profiler::ContextType::kGeneric;

      ResolvePidTid(event, track_uuid, seq_id, track_descriptors, fallback_pid);

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
    } else if (track_event.type() ==
               xprof::traceviewer::protos::TrackEvent::TYPE_INSTANT) {
      TraceEvent event;
      event.ph = Phase::kInstant;
      event.name = name.empty() ? std::string(kUnknown) : name;
      event.ts = ts;
      update_min_ts(ts);
      event.dur = 0;
      event.category = tsl::profiler::ContextType::kGeneric;

      ResolvePidTid(event, track_uuid, seq_id, track_descriptors, fallback_pid);

      parsed_events.flame_events.push_back(event);
    }
  }

  // Normalize timestamps
  if (normalize_timestamps && found_any_timestamp &&
      min_ts > kPerfettoTimestampNormalizationThresholdUs) {
    // If the minimum timestamp is large (e.g., > 1 second), we assume it's an
    // absolute timestamp (like Unix epoch or boot time in microseconds).
    // We normalize it by subtracting min_ts to shift the trace to start at 0.
    // This avoids precision issues in the frontend JavaScript (which uses
    // double floats) and makes the trace easier to navigate in the UI.
    for (auto& event : parsed_events.flame_events) {
      if (event.ph != Phase::kMetadata) {
        event.ts -= min_ts;
      }
    }
    // `flow_events` and `counter_events` are not currently populated by this
    // parser, which only handles slice events. The loops below are included
    // for future compatibility but will not modify any timestamps in the
    // current implementation.
    for (auto& event : parsed_events.flow_events) {
      event.ts -= min_ts;
    }
    for (auto& event : parsed_events.counter_events) {
      for (auto& ts : event.timestamps) {
        ts -= min_ts;
      }
    }
  }

  return parsed_events;
}


void ParseAndProcessPerfettoTraceEvents(
    int ptr, int length, const emscripten::val& visible_range_from_url,
    bool normalize_timestamps) {
  ParsedTraceEvents parsed_events = ParsePerfettoTraceEvents(
      reinterpret_cast<const void*>(ptr), length, normalize_timestamps);

  if (parsed_events.parsing_failed) {
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
  emscripten::function<void, int, int, const emscripten::val&, bool>(
      "processPerfettoTraceEvents",
      &traceviewer::ParseAndProcessPerfettoTraceEvents);
}

}  // namespace traceviewer
