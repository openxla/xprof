#include "frontend/app/components/trace_viewer_v2/trace_helper/trace_event_parser_core.h"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/no_destructor.h"
#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "tsl/platform/fingerprint.h"
#include "tsl/profiler/lib/context_types.h"
#include "frontend/app/components/trace_viewer_v2/trace_helper/trace_event.h"
#include "plugin/xprof/protobuf/trace_data_response.pb.h"

namespace traceviewer {

EventId GenerateEventId(absl::string_view name, Microseconds ts,
                        Microseconds dur) {
  const int64_t ts_ps = static_cast<int64_t>(std::round(ts * 1000000.0));
  const int64_t dur_ps = static_cast<int64_t>(std::round(dur * 1000000.0));
  char buf[512];
  if (name.size() + 44 <= sizeof(buf)) {
    std::memcpy(buf, name.data(), name.size());
    char* p = buf + name.size();
    *p++ = ':';
    p = std::to_chars(p, buf + sizeof(buf), ts_ps).ptr;
    *p++ = ':';
    p = std::to_chars(p, buf + sizeof(buf), dur_ps).ptr;
    return tsl::Fingerprint64(
        absl::string_view(buf, static_cast<size_t>(p - buf)));
  }
  return tsl::Fingerprint64(absl::StrCat(name, ":", ts_ps, ":", dur_ps));
}

Phase ParsePhase(absl::string_view ph_str) {
  if (!ph_str.empty()) {
    char ph_char = ph_str[0];
    switch (ph_char) {
      case static_cast<char>(Phase::kComplete):
        return Phase::kComplete;
      case static_cast<char>(Phase::kDurationBegin):
        return Phase::kDurationBegin;
      case static_cast<char>(Phase::kDurationEnd):
        return Phase::kDurationEnd;
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

void ProcessProcessMetadata(const xprof::Process& process,
                            ParsedTraceEvents& result) {
  TraceEvent process_ev;
  process_ev.ph = Phase::kMetadata;
  process_ev.pid = process.id();
  process_ev.name = kProcessName;
  process_ev.args[kName] = process.name();
  result.flame_events.push_back(std::move(process_ev));

  if (process.has_sort_index()) {
    TraceEvent process_sort_ev;
    process_sort_ev.ph = Phase::kMetadata;
    process_sort_ev.pid = process.id();
    process_sort_ev.name = kProcessSortIndex;
    process_sort_ev.args[kSortIndex] = absl::StrCat(process.sort_index());
    result.flame_events.push_back(std::move(process_sort_ev));
  }

  for (const xprof::Thread& thread : process.threads()) {
    TraceEvent thread_ev;
    thread_ev.ph = Phase::kMetadata;
    thread_ev.pid = process.id();
    thread_ev.tid = thread.id();
    thread_ev.name = kThreadName;
    thread_ev.args[kName] = thread.name();
    result.flame_events.push_back(std::move(thread_ev));

    if (thread.has_sort_index()) {
      TraceEvent thread_sort_ev;
      thread_sort_ev.ph = Phase::kMetadata;
      thread_sort_ev.pid = process.id();
      thread_sort_ev.tid = thread.id();
      thread_sort_ev.name = kThreadSortIndex;
      thread_sort_ev.args[kSortIndex] = absl::StrCat(thread.sort_index());
      result.flame_events.push_back(std::move(thread_sort_ev));
    }
  }
}

void ProcessMetadataEvents(const xprof::TraceDataResponse& response,
                           ParsedTraceEvents& result) {
  for (const xprof::Process& process : response.metadata().processes()) {
    ProcessProcessMetadata(process, result);
  }
}

namespace {

constexpr uint64_t kDurationPhaseBeginBit = 1ULL << 61;
constexpr uint64_t kDurationPhaseEndBit = 1ULL << 62;
constexpr uint64_t kDurationPhaseMask =
    kDurationPhaseBeginBit | kDurationPhaseEndBit;

template <typename GetInternedStringFn>
size_t ProcessCompleteEventSeriesSliceImpl(
    const xprof::TraceEventSeries& series,
    GetInternedStringFn&& get_interned_string, size_t start_idx,
    size_t max_events, uint64_t& current_ts_ps, ParsedTraceEvents& result,
    std::optional<std::pair<uint64_t, uint64_t>> filter_range_ps =
        std::nullopt,
    bool reserve_slice = true) {
  const size_t total = static_cast<size_t>(series.deltas_size());
  if (start_idx >= total || max_events == 0) {
    return 0;
  }
  const size_t count = std::min(max_events, total - start_idx);
  const size_t end_idx = start_idx + count;
  if (reserve_slice && !filter_range_ps.has_value()) {
    result.flame_events.reserve(result.flame_events.size() + count);
  }

  const auto& metadata = series.metadata();
  for (size_t idx = start_idx; idx < end_idx; ++idx) {
    const int i = static_cast<int>(idx);
    current_ts_ps += series.deltas(i);
    const uint64_t raw_dur_ps = series.durations(i);
    const uint64_t dur_ps = raw_dur_ps & ~kDurationPhaseMask;
    if (filter_range_ps.has_value()) {
      if (current_ts_ps > filter_range_ps->second) {
        return total - start_idx;
      }
      const uint64_t end_ts_ps = current_ts_ps + dur_ps;
      if (end_ts_ps < filter_range_ps->first) {
        continue;
      }
    }
    TraceEvent ev;
    if (raw_dur_ps & kDurationPhaseBeginBit) {
      ev.ph = Phase::kDurationBegin;
    } else if (raw_dur_ps & kDurationPhaseEndBit) {
      ev.ph = Phase::kDurationEnd;
    } else {
      ev.ph = Phase::kComplete;
    }
    ev.pid = metadata.process_id();
    ev.tid = metadata.thread_id();
    ev.ts = current_ts_ps / 1000000.0;
    ev.dur = dur_ps / 1000000.0;
    ev.name = std::string(get_interned_string(series.name_refs(i)));

    const auto& ev_meta = series.event_metadata(i);
    if (ev_meta.flow_id() != 0) {
      ev.id = std::to_string(ev_meta.flow_id());
      if (ev_meta.flow_category() != 0) {
        ev.category = GetContextTypeFromString(
            get_interned_string(ev_meta.flow_category()));
      }
    }

    ev.serial = ev_meta.serial();
    ev.has_serial = true;
    ev.event_id = GenerateEventId(ev.name, ev.ts, ev.dur);

    if (!ev.id.empty()) {
      result.flow_events.push_back(ev);
    }
    result.flame_events.push_back(std::move(ev));
  }
  return count;
}

}  // namespace

void ProcessCompleteEvents(const xprof::TraceDataResponse& response,
                           ParsedTraceEvents& result) {
  size_t total_deltas = 0;
  for (const auto& series : response.complete_events()) {
    total_deltas += series.deltas_size();
  }
  result.flame_events.reserve(result.flame_events.size() + total_deltas);
  for (const auto& series : response.complete_events()) {
    uint64_t current_ts_ps = 0;
    ProcessCompleteEventSeriesSliceImpl(
        series,
        [&](uint64_t ref) -> absl::string_view {
          return response.interned_strings(ref);
        },
        0, static_cast<size_t>(series.deltas_size()), current_ts_ps, result,
        /*filter_range_ps=*/std::nullopt, /*reserve_slice=*/false);
  }
}

size_t ProcessCompleteEventSeriesSlice(
    const xprof::TraceEventSeries& series,
    absl::Span<const std::string> interned_strings, size_t start_idx,
    size_t max_events, uint64_t& current_ts_ps, ParsedTraceEvents& result,
    std::optional<std::pair<uint64_t, uint64_t>> filter_range_ps) {
  return ProcessCompleteEventSeriesSliceImpl(
      series,
      [&](uint64_t ref) -> absl::string_view {
        return ref < interned_strings.size() ? interned_strings[ref]
                                             : absl::string_view();
      },
      start_idx, max_events, current_ts_ps, result, filter_range_ps);
}

namespace {

template <typename GetInternedStringFn>
void ProcessAsyncEventSeriesImpl(
    const xprof::TraceEventSeries& series,
    GetInternedStringFn&& get_interned_string,
    absl::flat_hash_map<std::pair<ProcessId, std::string>, TraceEvent>&
        open_async_events,
    ParsedTraceEvents& result) {
  const auto& metadata = series.metadata();
  uint64_t current_ts_ps = 0;
  for (int i = 0; i < series.deltas_size(); ++i) {
    current_ts_ps += series.deltas(i);
    const auto& ev_meta = series.event_metadata(i);
    std::string flow_id_str;
    tsl::profiler::ContextType category = tsl::profiler::ContextType::kGeneric;
    if (ev_meta.flow_id() != 0) {
      flow_id_str = std::to_string(ev_meta.flow_id());
      if (ev_meta.flow_category() != 0) {
        category = GetContextTypeFromString(
            get_interned_string(ev_meta.flow_category()));
      }
    }

    double dur = 0.0;
    if (i < series.durations_size()) {
      dur = series.durations(i) / 1000000.0;
    }

    TraceEvent ev;
    ev.pid = metadata.process_id();
    ev.ts = current_ts_ps / 1000000.0;
    ev.name = std::string(get_interned_string(series.metadata().name_ref()));
    ev.id = flow_id_str;
    ev.category = category;
    ev.serial = ev_meta.serial();
    ev.has_serial = true;
    if (ev_meta.group_id() != 0) {
      ev.group_id = ev_meta.group_id();
      ev.has_group_id = true;
    }

    if (dur > 0.0) {
      // Pre-computed duration available, treat as complete event.
      ev.dur = dur;
      ev.ph = Phase::kComplete;
      ev.is_async = true;
      ev.event_id = GenerateEventId(ev.name, ev.ts, ev.dur);
      if (!ev.id.empty()) {
        result.flow_events.push_back(ev);
      }
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
        if (ev.has_group_id && !begin_ev.has_group_id) {
          begin_ev.group_id = ev.group_id;
          begin_ev.has_group_id = true;
        }
        if (!ev.args.empty()) {
          begin_ev.args.insert(ev.args.begin(), ev.args.end());
        }
        begin_ev.event_id =
            GenerateEventId(begin_ev.name, begin_ev.ts, begin_ev.dur);
        if (!begin_ev.id.empty()) {
          result.flow_events.push_back(begin_ev);
        }
        result.flame_events.push_back(std::move(begin_ev));
        open_async_events.erase(it);
      }
    }
  }
}

}  // namespace

void ProcessAsyncEvents(const xprof::TraceDataResponse& response,
                        ParsedTraceEvents& result) {
  absl::flat_hash_map<std::pair<ProcessId, std::string>, TraceEvent>
      open_async_events;
  for (const auto& series : response.async_events()) {
    ProcessAsyncEventSeriesImpl(
        series,
        [&](uint64_t ref) -> absl::string_view {
          return response.interned_strings(ref);
        },
        open_async_events, result);
  }
}

void ProcessAsyncEventSeries(const xprof::TraceEventSeries& series,
                             absl::Span<const std::string> interned_strings,
                             ParsedTraceEvents& result) {
  absl::flat_hash_map<std::pair<ProcessId, std::string>, TraceEvent>
      open_async_events;
  ProcessAsyncEventSeriesImpl(
      series,
      [&](uint64_t ref) -> absl::string_view {
        return ref < interned_strings.size() ? interned_strings[ref]
                                             : absl::string_view();
      },
      open_async_events, result);
}

namespace {

template <typename GetInternedStringFn>
void ProcessCounterEventSeriesImpl(const xprof::TraceEventSeries& series,
                                   GetInternedStringFn&& get_interned_string,
                                   ParsedTraceEvents& result) {
  const auto& metadata = series.metadata();
  CounterEvent ev;
  ev.pid = metadata.process_id();
  ev.name = std::string(get_interned_string(metadata.name_ref()));
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

}  // namespace

void ProcessCounterEvents(const xprof::TraceDataResponse& response,
                          ParsedTraceEvents& result) {
  for (const auto& series : response.counter_events()) {
    ProcessCounterEventSeriesImpl(
        series,
        [&](uint64_t ref) -> absl::string_view {
          return response.interned_strings(ref);
        },
        result);
  }
}

void ProcessCounterEventSeries(const xprof::TraceEventSeries& series,
                               absl::Span<const std::string> interned_strings,
                               ParsedTraceEvents& result) {
  ProcessCounterEventSeriesImpl(
      series,
      [&](uint64_t ref) -> absl::string_view {
        return ref < interned_strings.size() ? interned_strings[ref]
                                             : absl::string_view();
      },
      result);
}

size_t ProcessOverviewSampledSeriesSlice(
    const xprof::TraceEventSeries& series,
    absl::Span<const std::string> interned_strings, size_t start_idx,
    size_t max_events, uint64_t& current_timestamp_ps,
    ParsedTraceEvents& result, bool& did_coalesce) {
  const ProcessId pid = series.metadata().process_id();
  const ThreadId tid = series.metadata().thread_id();
  const size_t total_events = static_cast<size_t>(series.deltas_size());
  if (start_idx >= total_events || max_events == 0) {
    return 0;
  }
  const size_t end_idx = std::min(total_events, start_idx + max_events);
  const size_t stride = std::clamp<size_t>(total_events / 2048, 2, 32);

  std::vector<uint64_t> parent_ends;
  parent_ends.reserve(16);

  bool in_burst = false;
  size_t burst_count = 0;
  uint64_t burst_start_ps = 0;
  uint64_t burst_end_ps = 0;
  uint64_t burst_active_dur_ps = 0;
  uint64_t burst_dom_dur_ps = 0;
  uint32_t burst_dom_name_ref = 0;
  uint32_t burst_dom_serial = 0;
  bool burst_has_serial = false;

  auto flush_burst = [&]() {
    if (!in_burst) return;
    TraceEvent ev;
    ev.ph = Phase::kComplete;
    ev.pid = pid;
    ev.tid = tid;
    ev.ts = burst_start_ps / 1000000.0;
    ev.dur = (burst_end_ps > burst_start_ps ? (burst_end_ps - burst_start_ps)
                                            : burst_active_dur_ps) /
             1000000.0;
    if (burst_dom_name_ref < interned_strings.size()) {
      ev.name = interned_strings[burst_dom_name_ref];
    }
    ev.serial = burst_dom_serial;
    ev.has_serial = burst_has_serial;
    ev.event_id = GenerateEventId(ev.name, ev.ts, ev.dur);
    result.flame_events.push_back(std::move(ev));
    if (burst_count > 1) {
      did_coalesce = true;
    }
    in_burst = false;
    burst_count = 0;
  };

  for (size_t idx = start_idx; idx < end_idx; ++idx) {
    const int i = static_cast<int>(idx);
    current_timestamp_ps += series.deltas(i);
    const uint64_t start_ps = current_timestamp_ps;
    const uint64_t raw_dur_ps = series.durations(i);
    const uint64_t dur_ps = raw_dur_ps & ~kDurationPhaseMask;
    const uint64_t end_ps = start_ps + dur_ps;
    const bool is_begin = (raw_dur_ps & kDurationPhaseBeginBit) != 0;
    const bool is_end = (raw_dur_ps & kDurationPhaseEndBit) != 0;
    const auto& ev_meta = series.event_metadata(i);
    const bool has_flow = (ev_meta.flow_id() != 0);

    bool parent_boundary_crossed = false;
    while (!parent_ends.empty() && parent_ends.back() <= start_ps) {
      parent_ends.pop_back();
      parent_boundary_crossed = true;
    }
    if (parent_boundary_crossed) {
      flush_burst();
    }

    const bool overlaps_next =
        (idx + 1 < total_events) &&
        (dur_ps > static_cast<uint64_t>(series.deltas(i + 1)));
    const bool exceeds_parent =
        !parent_ends.empty() && (end_ps > parent_ends.back());

    const bool can_coalesce_leaf = !overlaps_next && !exceeds_parent &&
                                   !is_begin && !is_end && !has_flow &&
                                   idx > start_idx && idx + 1 < end_idx;

    if (!can_coalesce_leaf) {
      flush_burst();
      if (overlaps_next && !is_begin && !is_end) {
        parent_ends.push_back(end_ps);
      }
      TraceEvent ev;
      if (is_begin) {
        ev.ph = Phase::kDurationBegin;
      } else if (is_end) {
        ev.ph = Phase::kDurationEnd;
      } else {
        ev.ph = Phase::kComplete;
      }
      ev.pid = pid;
      ev.tid = tid;
      ev.ts = start_ps / 1000000.0;
      ev.dur = dur_ps / 1000000.0;
      const uint32_t name_ref = series.name_refs(i);
      if (name_ref < interned_strings.size()) {
        ev.name = interned_strings[name_ref];
      }
      if (has_flow) {
        ev.id = std::to_string(ev_meta.flow_id());
        if (ev_meta.flow_category() != 0 &&
            ev_meta.flow_category() < interned_strings.size()) {
          ev.category = GetContextTypeFromString(
              interned_strings[ev_meta.flow_category()]);
        }
      }
      ev.serial = ev_meta.serial();
      ev.has_serial = true;
      ev.event_id = GenerateEventId(ev.name, ev.ts, ev.dur);
      if (!ev.id.empty()) {
        result.flow_events.push_back(ev);
      }
      result.flame_events.push_back(std::move(ev));
      continue;
    }

    const uint32_t name_ref = series.name_refs(i);
    const uint32_t serial = ev_meta.serial();
    if (in_burst) {
      const uint64_t gap_ps =
          start_ps > burst_end_ps ? (start_ps - burst_end_ps) : 0ULL;
      const uint64_t max_allowed_gap_ps =
          std::max<uint64_t>(burst_active_dur_ps * 2, 500000ULL);
      if (burst_count >= stride || gap_ps > max_allowed_gap_ps) {
        flush_burst();
      }
    }
    if (!in_burst) {
      in_burst = true;
      burst_count = 1;
      burst_start_ps = start_ps;
      burst_end_ps = end_ps;
      burst_active_dur_ps = dur_ps;
      burst_dom_dur_ps = dur_ps;
      burst_dom_name_ref = name_ref;
      burst_dom_serial = serial;
      burst_has_serial = true;
    } else {
      ++burst_count;
      burst_end_ps = std::max(burst_end_ps, end_ps);
      burst_active_dur_ps += dur_ps;
      if (dur_ps >= burst_dom_dur_ps) {
        burst_dom_dur_ps = dur_ps;
        burst_dom_name_ref = name_ref;
        burst_dom_serial = serial;
      }
    }
  }
  flush_burst();
  return end_idx - start_idx;
}

}  // namespace traceviewer
