#include "frontend/app/components/trace_viewer_v2/trace_helper/trace_event_parser_core.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <utility>

#include "absl/base/no_destructor.h"
#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "tsl/platform/fingerprint.h"
#include "tsl/profiler/lib/context_types.h"
#include "frontend/app/components/trace_viewer_v2/trace_helper/trace_event.h"
#include "plugin/xprof/protobuf/trace_data_response.pb.h"

namespace traceviewer {

EventId GenerateEventId(absl::string_view name, Microseconds ts,
                        Microseconds dur) {
  const int64_t ts_ps = static_cast<int64_t>(std::round(ts * 1000000.0));
  const int64_t dur_ps = static_cast<int64_t>(std::round(dur * 1000000.0));
  return tsl::Fingerprint64(absl::StrCat(name, ":", ts_ps, ":", dur_ps));
}

Phase ParsePhase(absl::string_view ph_str) {
  if (!ph_str.empty()) {
    char ph_char = ph_str[0];
    switch (ph_char) {
      case static_cast<char>(Phase::kComplete):
        return Phase::kComplete;
      case static_cast<char>(Phase::kInstant):
        return Phase::kInstant;
      case static_cast<char>(Phase::kCounter):
        return Phase::kCounter;
      case static_cast<char>(Phase::kMetadata):
        return Phase::kMetadata;
      case static_cast<char>(Phase::kAsyncBegin):
        return Phase::kAsyncBegin;
      case static_cast<char>(Phase::kAsyncEnd):
        return Phase::kAsyncEnd;
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
      ev.event_id = GenerateEventId(ev.name, ev.ts, ev.dur);

      if (!ev.id.empty()) {
        result.flow_events.push_back(ev);
      }
      result.flame_events.push_back(std::move(ev));
    }
  }
}

void ProcessAsyncEvents(const xprof::TraceDataResponse& response,
                        ParsedTraceEvents& result) {
  absl::flat_hash_map<std::pair<ProcessId, std::string>, TraceEvent>
      open_async_events;

  for (const auto& series : response.async_events()) {
    const auto& metadata = series.metadata();
    uint64_t current_ts_ps = 0;
    for (int i = 0; i < series.deltas_size(); ++i) {
      current_ts_ps += series.deltas(i);
      const auto& ev_meta = series.event_metadata(i);
      if (ev_meta.flow_id() != 0) {
        std::string flow_id_str = std::to_string(ev_meta.flow_id());
        tsl::profiler::ContextType category =
            tsl::profiler::ContextType::kGeneric;
        if (ev_meta.flow_category() != 0) {
          category = GetContextTypeFromString(
              response.interned_strings(ev_meta.flow_category()));
        }

        double dur = 0.0;
        if (i < series.durations_size()) {
          dur = series.durations(i) / 1000000.0;
        }

        TraceEvent ev;
        ev.pid = metadata.process_id();
        ev.ts = current_ts_ps / 1000000.0;
        ev.name = response.interned_strings(series.metadata().name_ref());
        ev.id = flow_id_str;
        ev.category = category;
        ev.args["uid"] = std::to_string(ev_meta.serial());
        if (ev_meta.group_id() != 0) {
          ev.args["group_id"] = std::to_string(ev_meta.group_id());
        }

        if (dur > 0.0) {
          // Pre-computed duration available, treat as complete event.
          ev.dur = dur;
          ev.ph = Phase::kComplete;
          ev.is_async = true;
          ev.event_id = GenerateEventId(ev.name, ev.ts, ev.dur);
          result.flame_events.push_back(std::move(ev));
        } else {
          // No duration, assume it's part of a separate Begin/End pair.
          auto key = std::make_pair(ev.pid, ev.id);
          auto it = open_async_events.find(key);
          if (it == open_async_events.end()) {
            // Begin
            ev.ph = Phase::kAsyncBegin;
            open_async_events.try_emplace(std::move(key), std::move(ev));
          } else {
            // End
            TraceEvent& begin_ev = it->second;
            begin_ev.ph = Phase::kComplete;
            begin_ev.is_async = true;
            if (ev.ts > begin_ev.ts) {
              begin_ev.dur = ev.ts - begin_ev.ts;
            }
            if (!ev.args.empty()) {
              begin_ev.args.insert(ev.args.begin(), ev.args.end());
            }
            begin_ev.event_id =
                GenerateEventId(begin_ev.name, begin_ev.ts, begin_ev.dur);
            result.flame_events.push_back(std::move(begin_ev));
            open_async_events.erase(it);
          }
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

}  // namespace traceviewer
