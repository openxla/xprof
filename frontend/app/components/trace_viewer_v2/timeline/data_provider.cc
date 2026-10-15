#include "frontend/app/components/trace_viewer_v2/timeline/data_provider.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/no_destructor.h"
#include "absl/container/btree_map.h"
#include "absl/container/btree_set.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/strings/match.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "imgui.h"
#include "google/protobuf/arena.h"
#include "re2/re2.h"
#include "tsl/profiler/lib/context_types.h"
#include "frontend/app/components/trace_viewer_v2/color/colors.h"
#include "frontend/app/components/trace_viewer_v2/helper/time_formatter.h"
#include "frontend/app/components/trace_viewer_v2/timeline/constants.h"
#include "frontend/app/components/trace_viewer_v2/timeline/time_range.h"
#include "frontend/app/components/trace_viewer_v2/timeline/timeline.h"
#include "frontend/app/components/trace_viewer_v2/timeline/track_packer.h"
#include "frontend/app/components/trace_viewer_v2/trace_helper/proto_event_ref.h"
#include "frontend/app/components/trace_viewer_v2/trace_helper/trace_event.h"
#include "frontend/app/components/trace_viewer_v2/trace_helper/trace_event_packer.h"
#include "frontend/app/components/trace_viewer_v2/trace_helper/trace_event_tree.h"
#include "frontend/app/components/trace_viewer_v2/trace_helper/wire_trace_reader.h"
#include "plugin/xprof/protobuf/trace_data_response.pb.h"

namespace traceviewer {

namespace {

// Base offset for process sort indices of devices not ranked into an MPMD
// stage. Offsets unranked devices past stage ranks with headroom for PID
// tie-breaking.
constexpr uint32_t kMpmdUnrankedSortIndexBase = 1u << 30;

constexpr ThreadId kSyntheticOpenSpanTidBit = 0x80000000;
bool GetExpandedState(int nesting_level, absl::string_view name,
                      absl::string_view parent_name, bool default_expanded,
                      const absl::btree_map<GroupKey, bool>& expanded_states) {
  if (auto it_state = expanded_states.find(
          {nesting_level, std::string(name), std::string(parent_name)});
      it_state != expanded_states.end()) {
    return it_state->second;
  }
  return default_expanded;
}

absl::btree_map<GroupKey, bool> GetRestoredExpandedStates(
    const std::vector<Group>& groups) {
  absl::btree_map<GroupKey, bool> expanded_states;
  std::string current_process_name;
  for (const auto& group : groups) {
    if (group.nesting_level == kProcessNestingLevel) {
      current_process_name = group.name;
      expanded_states[{kProcessNestingLevel, group.name, ""}] = group.expanded;
    } else {
      const std::string parent =
          (group.parent_index != -1 &&
           group.parent_index < static_cast<int>(groups.size()))
              ? groups[group.parent_index].name
              : current_process_name;
      expanded_states[{group.nesting_level, group.name, parent}] =
          group.expanded;
    }
  }
  return expanded_states;
}

template <typename EventT>
int GetAsyncProcessPriority(ProcessId pid,
                            const TraceInformation<EventT>& trace_info) {
  absl::string_view name;
  if (auto it = trace_info.process_names.find(pid);
      it != trace_info.process_names.end()) {
    name = it->second;
    if (absl::EqualsIgnoreCase(name, kAsyncXlaOps)) return 2;
  }

  bool is_priority_1 = false;

  if (!name.empty()) {
    if (absl::StrContainsIgnoreCase(name, kDma) ||
        absl::EqualsIgnoreCase(name, kDataMotionLayersUtilization)) {
      is_priority_1 = true;
    }
  }

  for (auto it = trace_info.thread_names.lower_bound({pid, 0});
       it != trace_info.thread_names.end() && it->first.first == pid; ++it) {
    const absl::string_view thread_name = it->second;
    if (absl::EqualsIgnoreCase(thread_name, kAsyncXlaOps)) {
      return 2;
    }
    if (absl::StrContainsIgnoreCase(thread_name, kDma)) {
      is_priority_1 = true;
    }
  }

  if (is_priority_1 || trace_info.async_processes_by_events.contains(pid)) {
    return 1;
  }

  return 0;
}

std::string GetDefaultThreadName(ThreadId tid) {
  return absl::StrCat("Thread_", tid);
}

std::string GetDefaultProcessName(ProcessId pid) {
  return absl::StrCat("Process_", pid);
}

// Compares two tracks (threads or processes) for ordering in the timeline.
// Tracks are sorted hierarchically:
// 1. Sort index: Unindexed tracks default to sort index 0. Tracks are
//    ordered in ascending order of index value.
// 2. Name: Tracks with explicit names precede unnamed tracks, ordered
//    lexicographically.
// 3. ID: Ordered by ProcessId or ThreadId in ascending order as a tie-breaker.
template <typename IdType>
bool CompareTrackMetadata(std::optional<uint32_t> sort_index_a,
                          std::optional<uint32_t> sort_index_b,
                          std::optional<absl::string_view> name_a,
                          std::optional<absl::string_view> name_b, IdType id_a,
                          IdType id_b) {
  const uint32_t idx_a = sort_index_a.value_or(0);
  const uint32_t idx_b = sort_index_b.value_or(0);
  if (idx_a != idx_b) {
    return idx_a < idx_b;
  }

  if (name_a.has_value() && name_b.has_value()) {
    if (*name_a != *name_b) return *name_a < *name_b;
  } else if (name_a.has_value() != name_b.has_value()) {
    return name_a.has_value();
  }

  return id_a < id_b;
}

template <typename EventT>
bool CompareThreadsForSort(ProcessId pid,
                           const TraceInformation<EventT>& trace_info,
                           ThreadId a, ThreadId b) {
  auto get_thread_sort_index = [&](ThreadId tid) -> std::optional<uint32_t> {
    if (const auto it = trace_info.thread_sort_indices.find({pid, tid});
        it != trace_info.thread_sort_indices.end()) {
      return it->second;
    }
    return std::nullopt;
  };
  auto get_thread_name = [&](ThreadId tid) -> std::optional<absl::string_view> {
    if (const auto it = trace_info.thread_names.find({pid, tid});
        it != trace_info.thread_names.end()) {
      return it->second;
    }
    return std::nullopt;
  };

  return CompareTrackMetadata(get_thread_sort_index(a),
                              get_thread_sort_index(b), get_thread_name(a),
                              get_thread_name(b), a, b);
}

// Handles a metadata event, extracting and storing metadata such as
// thread names, process names, etc.
// An example of the JSON structure for a thread name metadata event:
// {
//   "args": {
//     "name": "Steps"
//   },
//   "name": "thread_name",
//   "ph": "M",
//   "pid": 3,
//   "tid": 1
// }
void HandleMetadataEvent(const TraceEvent& event,
                         TraceInformation<TraceEvent>& trace_info) {
  if (event.name == kThreadName) {
    if (const auto it = event.args.find(kName);
        it != event.args.end() && !it->second.empty()) {
      trace_info.thread_names[{event.pid, event.tid}] = it->second;
      if (it->second == kXlaModules) {
        trace_info.xla_modules_tids[event.pid] = event.tid;
      } else if (it->second == kXlaOps) {
        trace_info.xla_ops_tids[event.pid] = event.tid;
      }
    }
  } else if (event.name == kProcessName) {
    if (const auto it = event.args.find(kName);
        it != event.args.end() && !it->second.empty()) {
      trace_info.process_names[event.pid] = it->second;
    }
  } else if (event.name == kProcessSortIndex) {
    if (const auto it = event.args.find(kSortIndex); it != event.args.end()) {
      double sort_index_double;
      if (absl::SimpleAtod(it->second, &sort_index_double) &&
          std::isfinite(sort_index_double) && sort_index_double >= 0.0 &&
          sort_index_double <=
              static_cast<double>(std::numeric_limits<uint32_t>::max())) {
        const uint32_t sort_index = static_cast<uint32_t>(sort_index_double);
        trace_info.incoming_process_sort_indices[event.pid] = sort_index;
      }
    }
  } else if (event.name == kThreadSortIndex) {
    if (const auto it = event.args.find(kSortIndex); it != event.args.end()) {
      double sort_index_double;
      if (absl::SimpleAtod(it->second, &sort_index_double) &&
          std::isfinite(sort_index_double) && sort_index_double >= 0.0 &&
          sort_index_double <=
              static_cast<double>(std::numeric_limits<uint32_t>::max())) {
        trace_info.thread_sort_indices[{event.pid, event.tid}] =
            static_cast<uint32_t>(sort_index_double);
      }
    }
  }
}

// Counts the number of contiguous 0..K-1 MPMD stage ranks in sort_indices.
uint32_t CountContiguousMpmdRanks(
    const absl::flat_hash_map<ProcessId, uint32_t>& sort_indices) {
  absl::flat_hash_set<uint32_t> values;
  values.reserve(sort_indices.size());
  for (const auto& [pid, sort_index] : sort_indices) {
    values.insert(sort_index);
  }
  uint32_t count = 0;
  while (values.contains(count)) {
    ++count;
  }
  return count;
}

// Merges incoming process sort indices into trace_info.process_sort_indices.
// When incoming ranks provide a strictly better stage coverage, existing stage
// ranks are updated. Fallback indices (>= kMpmdUnrankedSortIndexBase) are only
// inserted if the process has no existing rank, and valid stage ranks (< Base)
// overwrite placeholder unranked indices.
template <typename EventT>
void MergeProcessSortIndices(TraceInformation<EventT>& trace_info) {
  if (trace_info.incoming_process_sort_indices.empty()) {
    return;
  }
  if (!trace_info.is_mpmd) {
    for (const auto& [pid, sort_index] :
         trace_info.incoming_process_sort_indices) {
      trace_info.process_sort_indices[pid] = sort_index;
    }
    return;
  }

  const uint32_t existing_ranks =
      CountContiguousMpmdRanks(trace_info.process_sort_indices);
  const uint32_t incoming_ranks =
      CountContiguousMpmdRanks(trace_info.incoming_process_sort_indices);

  const bool should_overwrite = incoming_ranks > existing_ranks;

  for (const auto& [pid, sort_index] :
       trace_info.incoming_process_sort_indices) {
    if (sort_index >= kMpmdUnrankedSortIndexBase) {
      trace_info.process_sort_indices.try_emplace(pid, sort_index);
    } else if (should_overwrite) {
      trace_info.process_sort_indices[pid] = sort_index;
    } else {
      auto [it, inserted] =
          trace_info.process_sort_indices.try_emplace(pid, sort_index);
      if (!inserted && it->second >= kMpmdUnrankedSortIndexBase) {
        it->second = sort_index;
      }
    }
  }
}

// Handles a complete event ('ph' == 'X'). These events represent a duration
// of activity. The function groups events by thread ID.
// An example of the JSON structure for such an event is shown below:
// {
//   "pid": 3,
//   "tid": 1,
//   "name": "0",
//   "ts": 6845940.1418570001,
//   "dur": 3208616.194286,
//   "cname": "thread_state_running",
//   "ph": "X",
//   "args": {
//     "group_id": 0,
//     "step_name": "0"
//   }
// }
void HandleCompleteEvent(const TraceEvent& event,
                         TraceInformation<TraceEvent>& trace_info) {
  trace_info.events_by_pid_tid[event.pid][event.tid].push_back(&event);
  if (event.is_async) {
    trace_info.async_processes_by_events.insert(event.pid);
  }
}

bool HasFlowId(const TraceEvent& event) {
  return event.has_flow_id || event.flow_id != 0 || !event.id.empty();
}

uint64_t GetEffectiveFlowId(const TraceEvent& event) {
  if (event.has_flow_id || event.flow_id != 0) {
    return event.flow_id;
  }
  return ParseFlowIdString(event.id);
}

void HandleFlowEvent(const TraceEvent& event,
                     TraceInformation<TraceEvent>& trace_info,
                     absl::btree_map<int, int>& category_counts) {
  if (!HasFlowId(event)) {
    return;
  }
  trace_info.flow_events_by_id[GetEffectiveFlowId(event)].push_back(&event);
  ++category_counts[static_cast<int>(event.category)];
}

// Handles a counter event ('ph' == 'C'). These events represent a counter value
// at a specific timestamp. The function groups events by process ID and counter
// name.
// An example of the JSON structure for such an event is shown below:
// {
//   "pid": 3,
//   "name": "HBM FW Power Meter PL2(W)",
//   "ph": "C",
//   "event_stats": "power",
//   "entries": [
//    {
//      "ts": 6845940.1418570001,
//      "value": 1.0
//    },
//    {
//      "ts": 6845940.1418570001,
//      "value": 5.0
//    }
//  ]
// }
// (See AddCounterEvent in google3/third_party/xprof/convert/trace_viewer/
// trace_events_to_json.h for a more detailed view of XProf counter events.)
void HandleCounterEvent(const CounterEvent& event,
                        TraceInformation<TraceEvent>& trace_info) {
  trace_info.counters_by_pid_name[event.pid][event.name].push_back(&event);
}

struct ThreadLevelInfo {
  int start_level;
  int end_level;
};

// Returns a color for a flow event category. If the category is kGeneric,
// kRed80 is returned. If the category is one of the top 5 flow categories,
// a color from top_5_colors is returned based on its rank. Otherwise, kPurple80
// is returned.
std::vector<int> GetTop5FlowCategories(
    const absl::btree_map<int, int>& flow_category_counts) {
  std::vector<std::pair<int, int>> sorted_flow_categories;
  for (const auto& [cat, count] : flow_category_counts) {
    if (cat != static_cast<int>(tsl::profiler::ContextType::kGeneric)) {
      sorted_flow_categories.push_back({cat, count});
    }
  }
  absl::c_stable_sort(sorted_flow_categories, [](const auto& a, const auto& b) {
    if (a.second != b.second) {
      return a.second > b.second;
    }
    return a.first < b.first;
  });
  std::vector<int> top_5_flow_categories;
  for (int i = 0; i < std::min<size_t>(sorted_flow_categories.size(), 5); ++i) {
    top_5_flow_categories.push_back(sorted_flow_categories[i].first);
  }
  return top_5_flow_categories;
}

ImU32 GetFlowColorForCategory(tsl::profiler::ContextType category,
                              absl::Span<const int> top_5_flow_categories,
                              const ColorPalette& palette) {
  if (category == tsl::profiler::ContextType::kGeneric) {
    return kRed80;
  }

  absl::Span<const ImU32> flow_colors = palette.GetFlowColors();
  if (flow_colors.empty()) {
    return kPurple80;  // Fallback if no flow colors are provided.
  }

  for (size_t i = 0; i < top_5_flow_categories.size(); ++i) {
    if (static_cast<int>(category) == top_5_flow_categories[i]) {
      return flow_colors[i % flow_colors.size()];
    }
  }
  return kPurple80;
}

// Returns the flame chart level of an event at (pid, tid, ts).
int GetEventFlameChartLevel(
    ProcessId pid, ThreadId tid, Microseconds ts,
    const absl::btree_map<std::pair<ProcessId, ThreadId>, ThreadLevelInfo>&
        thread_levels,
    const FlameChartTimelineData& data) {
  auto it = thread_levels.find({pid, tid});
  if (it == thread_levels.end()) return 0;
  int start = it->second.start_level;
  int end = it->second.end_level;

  // Search from deepest level up
  for (int lvl = end - 1; lvl >= start; --lvl) {
    const Group* grp = data.FindGroupForLevel(lvl);
    if (grp != nullptr && grp->tid != tid) continue;
    absl::Span<const int> indices = data.level_events(lvl);
    // Binary search for event covering ts
    // events are likely sorted by start time.
    auto it_idx =
        std::upper_bound(indices.begin(), indices.end(), ts,
                         [&](Microseconds target_ts, int idx) {
                           return target_ts < data.entry_start_times[idx];
                         });

    // it_idx points to first event starting AFTER ts.
    // Check the one before it.
    if (it_idx != indices.begin()) {
      int idx = *std::prev(it_idx);
      if (data.entry_start_times[idx] + data.entry_total_times[idx] >= ts) {
        return lvl;
      }
    }
  }
  return start;  // Default to thread top
}

template <typename EventT>
void GenerateFlowLines(const TraceInformation<EventT>& trace_info,
                       const absl::btree_map<std::pair<ProcessId, ThreadId>,
                                             ThreadLevelInfo>& thread_levels,
                       absl::Span<const int> top_5_flow_categories,
                       FlameChartTimelineData& data, TimeBounds& bounds,
                       const ColorPalette& palette) {
  for (const auto& [flow_id, flow_events] : trace_info.flow_events_by_id) {
    std::string flow_id_str;
    if constexpr (std::is_same_v<EventT, TraceEvent>) {
      flow_id_str = (!flow_events.empty() && !flow_events[0]->id.empty())
                        ? flow_events[0]->id
                        : absl::StrCat(flow_id);
      for (const TraceEvent* event : flow_events) {
        const EventId event_id =
            event->event_id != 0
                ? event->event_id
                : GenerateEventId(event->name, event->ts, event->dur);
        std::vector<std::string>& event_flow_ids =
            data.flow_ids_by_event_id[event_id];
        if (event_flow_ids.empty() || event_flow_ids.back() != flow_id_str) {
          event_flow_ids.push_back(flow_id_str);
        }
      }
    } else {
      if (flow_events.size() < 2) continue;
    }

    const uint32_t range_begin = static_cast<uint32_t>(data.flow_lines.size());
    for (size_t i = 0; i < flow_events.size() - 1; ++i) {
      const auto& u = flow_events[i];
      const auto& v = flow_events[i + 1];
      const ProcessId u_pid = GetEndpointPid(u);
      const ThreadId u_tid = GetEndpointTid(u);
      const Microseconds u_ts = GetEndpointTs(u);
      const ProcessId v_pid = GetEndpointPid(v);
      const ThreadId v_tid = GetEndpointTid(v);
      const Microseconds v_ts = GetEndpointTs(v);
      if (!thread_levels.contains({u_pid, u_tid}) ||
          !thread_levels.contains({v_pid, v_tid})) {
        continue;
      }
      const ImU32 flow_color =
          GetFlowColorForCategory(GetEndpointCategory(u), top_5_flow_categories,
                                  palette);  // Use flow category for color
      FlowLine flow_line{.source_ts = u_ts,
                         .target_ts = v_ts,
                         .source_level = GetEventFlameChartLevel(
                             u_pid, u_tid, u_ts, thread_levels, data),
                         .target_level = GetEventFlameChartLevel(
                             v_pid, v_tid, v_ts, thread_levels, data),
                         .color = flow_color,
                         .category = GetEndpointCategory(u)};
      data.flow_lines.push_back(flow_line);
      if constexpr (std::is_same_v<EventT, TraceEvent>) {
        data.flow_lines_by_flow_id[flow_id_str].push_back(flow_line);
      }
      bounds.min = std::min(bounds.min, u_ts);
      bounds.max = std::max(bounds.max, u_ts);
      bounds.min = std::min(bounds.min, v_ts);
      bounds.max = std::max(bounds.max, v_ts);
    }
    if constexpr (!std::is_same_v<EventT, TraceEvent>) {
      const uint32_t range_end = static_cast<uint32_t>(data.flow_lines.size());
      if (range_end > range_begin) {
        data.flow_line_ranges_by_flow_id.try_emplace(
            flow_id, FlameChartTimelineData::FlowLineRange{.begin = range_begin,
                                                           .end = range_end});
      }
    }
  }
}

struct TrackSpanSource {
  absl::Span<const TraceEvent* const> events;
};

TrackSpanSource ResolveTrackSpanSource(
    ProcessId pid, const absl::flat_hash_map<ProcessId, ThreadId>& tid_by_pid,
    const TraceInformation<TraceEvent>& trace_info) {
  TrackSpanSource src;
  const auto it_tid = tid_by_pid.find(pid);
  if (it_tid == tid_by_pid.end()) return src;
  if (const auto it_events = trace_info.events_by_pid_tid.find(pid);
      it_events != trace_info.events_by_pid_tid.end()) {
    if (const auto it_thread = it_events->second.find(it_tid->second);
        it_thread != it_events->second.end()) {
      src.events = it_thread->second;
    }
  }
  return src;
}

template <typename StartFn, typename DurFn, typename NameFn>
absl::string_view ScanSpansAtTimestamp(size_t count, StartFn get_start,
                                       DurFn get_dur, NameFn get_name,
                                       size_t& cursor, Microseconds& last_ts,
                                       Microseconds ts,
                                       absl::string_view default_name) {
  if (count == 0) return default_name;
  if (ts < last_ts || cursor >= count || get_start(cursor) > ts) {
    cursor = 0;
  }
  last_ts = ts;
  while (cursor < count && get_start(cursor) + get_dur(cursor) < ts) {
    ++cursor;
  }
  for (size_t k = cursor; k < count && get_start(k) <= ts; ++k) {
    if (get_start(k) + get_dur(k) >= ts) {
      return get_name(k);
    }
  }
  return default_name;
}

absl::string_view LookupTrackSpanAtTimestamp(
    const TrackSpanSource& src, size_t& cursor, Microseconds& last_ts,
    Microseconds ts, absl::string_view default_name = kHloModuleDefault) {
  if (!src.events.empty()) {
    return ScanSpansAtTimestamp(
        src.events.size(), [&](size_t k) { return src.events[k]->ts; },
        [&](size_t k) { return src.events[k]->dur; },
        [&](size_t k) -> absl::string_view { return src.events[k]->name; },
        cursor, last_ts, ts, default_name);
  }
  return default_name;
}

absl::string_view GetEventArg(const TraceEvent* event, absl::string_view key) {
  if (event->args.empty()) return "";
  if (const auto it = event->args.find(key); it != event->args.end()) {
    return it->second;
  }
  return "";
}

void AppendEventToTimelineData(
    const TraceEvent* event, int level, FlameChartTimelineData& data,
    TimeBounds& bounds, TraceInformation<TraceEvent>& trace_info,
    absl::string_view thread_name, const TrackSpanSource& module_src,
    size_t& module_cursor, Microseconds& last_module_lookup_ts,
    std::optional<Microseconds> self_time = std::nullopt) {
  static const absl::NoDestructor<std::string> kHloOpStr(kHloOp);
  static const absl::NoDestructor<std::string> kHloModuleStr(kHloModule);
  static const absl::NoDestructor<std::string> kHloModuleDefaultStr(
      kHloModuleDefault);
  static const absl::NoDestructor<std::string> kHloModuleIdStr(kHloModuleId);
  static const absl::NoDestructor<std::string> kProgramIdStr(kProgramId);
  static const absl::NoDestructor<std::string> kKernelDetailsStr(
      kKernelDetails);
  static const absl::NoDestructor<RE2> kModuleRe(kModuleRegex);

  // `level` is a cumulative row index across all process, counter, thread, and
  // async tracks (bounded by hosts * tracks_per_host * padded
  // max_observed_levels; ~256 levels for single-host medium.response_pb.gz and
  // ~2,300 levels across 8 hosts / 80 TPUs). Events at level < 0 or
  // level >= 65,535 (uint16_t max) are dropped and excluded from `data` and
  // `bounds` (leaving their enclosing Group track rows empty past 65,534).
  if (level < 0 || level >= std::numeric_limits<uint16_t>::max()) {
    ++data.dropped_level_cap_events;
    LOG_FIRST_N(WARNING, 1) << "Dropping event at level " << level
                            << ": exceeds uint16_t flame chart level range";
    return;
  }
  const size_t entry_idx = data.entry_start_times.size();
  data.entry_start_times.push_back(event->ts);
  data.entry_total_times.push_back(event->dur);
  data.entry_self_times.push_back(self_time.value_or(event->dur));
  data.entry_levels.push_back(static_cast<uint16_t>(level));
  data.entry_name_indices.push_back(
      trace_info.name_interner.Intern(event->name));
  data.entry_names.push_back(event->name);
  data.entry_event_ids.push_back(
      event->event_id != 0
          ? event->event_id
          : GenerateEventId(event->name, event->ts, event->dur));
  if (event->has_serial && event->serial != 0) {
    data.non_zero_uids_by_event_index[static_cast<int>(entry_idx)] =
        static_cast<uint32_t>(event->serial);
  }
  if (event->flow_id != 0 && !event->is_async) {
    data.flow_ids_by_event_index[static_cast<int>(entry_idx)] = event->flow_id;
  }
  if (event->has_group_id) {
    data.sparse_entry_group_ids.Append(entry_idx, event->group_id);
  }

  const absl::string_view hlo_op = GetEventArg(event, *kHloOpStr);
  const absl::string_view hlo_module = GetEventArg(event, *kHloModuleStr);
  const bool is_xla_ops_thread = thread_name == kXlaOps;
  const bool is_source_thread = thread_name == "Source code";
  const bool is_data_motion_layer = thread_name == kComputeUtilization ||
                                    thread_name == kDataMotionLayersUtilization;
  const bool has_hlo_in_args = !hlo_op.empty() && !hlo_module.empty();

  absl::string_view resolved_hlo_op = hlo_op;
  absl::string_view resolved_hlo_module = *kHloModuleDefaultStr;
  std::string formatted_hlo_module;

  if (is_xla_ops_thread || is_source_thread || is_data_motion_layer ||
      has_hlo_in_args) {
    if (is_data_motion_layer) {
      const absl::string_view name_arg = GetEventArg(event, "Name");
      if (!name_arg.empty()) {
        resolved_hlo_op = name_arg;
      }
    } else if (is_source_thread && hlo_op.empty()) {
      size_t op_cursor = 0;
      Microseconds last_op_ts = std::numeric_limits<Microseconds>::lowest();
      const absl::string_view op_name = LookupTrackSpanAtTimestamp(
          ResolveTrackSpanSource(event->pid, trace_info.xla_ops_tids,
                                 trace_info),
          op_cursor, last_op_ts, event->ts, /*default_name=*/"");
      if (!op_name.empty()) {
        resolved_hlo_op = op_name;
      }
    } else if (hlo_op.empty()) {
      resolved_hlo_op = event->name;
    }

    if (!hlo_module.empty()) {
      const absl::string_view hlo_module_id =
          GetEventArg(event, *kHloModuleIdStr);
      const absl::string_view program_id = GetEventArg(event, *kProgramIdStr);
      if (!hlo_module_id.empty()) {
        formatted_hlo_module =
            absl::StrCat(hlo_module, "(", hlo_module_id, ")");
        resolved_hlo_module = formatted_hlo_module;
      } else if (!program_id.empty()) {
        formatted_hlo_module = absl::StrCat(hlo_module, "(", program_id, ")");
        resolved_hlo_module = formatted_hlo_module;
      } else {
        const absl::string_view kernel_details =
            GetEventArg(event, *kKernelDetailsStr);
        std::string module_id;
        if (!kernel_details.empty() &&
            RE2::PartialMatch(kernel_details, *kModuleRe, &module_id)) {
          formatted_hlo_module = absl::StrCat(hlo_module, "(", module_id, ")");
          resolved_hlo_module = formatted_hlo_module;
        } else {
          resolved_hlo_module = hlo_module;
        }
      }
    } else {
      resolved_hlo_module = LookupTrackSpanAtTimestamp(
          module_src, module_cursor, last_module_lookup_ts, event->ts);
    }
  }

  data.entry_hlo_op_indices.push_back(
      trace_info.op_interner.Intern(resolved_hlo_op));
  data.entry_hlo_module_indices.push_back(
      trace_info.module_interner.Intern(resolved_hlo_module));

  if (!event->args.empty()) {
    data.sparse_entry_args[entry_idx] = event->args;
  }

  bounds.min = std::min(bounds.min, event->ts);
  bounds.max = std::max(bounds.max, event->ts + event->dur);
}

// Packs trace events for a thread track and appends them to data (inout).
// Updates max_level (inout) if any packed event exceeds it, and expands
// bounds (inout) to include the start and end timestamps of all appended
// events.
void PopulateThreadTrackEvents(absl::Span<const TraceEvent* const> events,
                               int start_level, int& max_level,
                               FlameChartTimelineData& data, TimeBounds& bounds,
                               TraceInformation<TraceEvent>& trace_info,
                               absl::string_view thread_group_name) {
  if (events.empty()) return;

  auto cmp_events = [](const TraceEvent* a, const TraceEvent* b) {
    return a->ts != b->ts ? a->ts < b->ts : a->dur > b->dur;
  };

  std::vector<const TraceEvent*> sorted_storage;
  absl::Span<const TraceEvent* const> sorted_events = events;
  for (size_t i = 0; i < events.size(); ++i) {
    if (events[i] == nullptr || (i > 0 && events[i - 1] != nullptr &&
                                 cmp_events(events[i], events[i - 1]))) {
      sorted_storage.reserve(events.size());
      for (const TraceEvent* event : events) {
        if (event != nullptr) sorted_storage.push_back(event);
      }
      std::sort(sorted_storage.begin(), sorted_storage.end(), cmp_events);
      sorted_events = sorted_storage;
      break;
    }
  }

  const size_t n = sorted_events.size();
  if (n == 0) return;

  std::vector<Microseconds> self_times(n);
  std::vector<int> relative_levels(n);
  std::vector<size_t> parent_stack;
  parent_stack.reserve(16);
  std::vector<Microseconds> row_end_times;
  row_end_times.reserve(16);

  for (size_t i = 0; i < n; ++i) {
    const TraceEvent* ev = sorted_events[i];
    const Microseconds end_time = ev->ts + ev->dur;
    self_times[i] = ev->dur;

    while (!parent_stack.empty()) {
      const size_t p_idx = parent_stack.back();
      const Microseconds p_end =
          sorted_events[p_idx]->ts + sorted_events[p_idx]->dur;
      if (ev->ts < p_end) {
        if (end_time <= p_end) {
          self_times[p_idx] = std::max(0.0, self_times[p_idx] - ev->dur);
          parent_stack.push_back(i);
        }
        break;
      }
      parent_stack.pop_back();
    }
    if (parent_stack.empty()) {
      parent_stack.push_back(i);
    }

    int assigned_row = -1;
    for (size_t r = 0; r < row_end_times.size(); ++r) {
      if (row_end_times[r] <= ev->ts) {
        assigned_row = static_cast<int>(r);
        break;
      }
    }
    if (assigned_row != -1) {
      row_end_times[assigned_row] = end_time;
    } else {
      assigned_row = static_cast<int>(row_end_times.size());
      row_end_times.push_back(end_time);
    }
    relative_levels[i] = assigned_row;
  }

  const ProcessId pid = sorted_events.front()->pid;
  const TrackSpanSource module_src =
      ResolveTrackSpanSource(pid, trace_info.xla_modules_tids, trace_info);

  const size_t new_total = data.entry_start_times.size() + n;
  data.Reserve(new_total);

  size_t module_cursor = 0;
  Microseconds last_module_lookup_ts =
      std::numeric_limits<Microseconds>::lowest();
  for (size_t i = 0; i < n; ++i) {
    const int absolute_level = start_level + relative_levels[i];
    max_level = std::max(max_level, absolute_level);
    AppendEventToTimelineData(sorted_events[i], absolute_level, data, bounds,
                              trace_info, thread_group_name, module_src,
                              module_cursor, last_module_lookup_ts,
                              self_times[i]);
  }
}

void PopulateThreadTrack(
    ProcessId pid, ThreadId tid, absl::Span<const TraceEvent* const> events,
    TraceInformation<TraceEvent>& trace_info, int& current_level,
    FlameChartTimelineData& data, TimeBounds& bounds,
    absl::btree_map<std::pair<ProcessId, ThreadId>, ThreadLevelInfo>&
        thread_levels,
    const std::string& process_group_name, bool default_expanded,
    const absl::btree_map<GroupKey, bool>& expanded_states,
    absl::flat_hash_map<GroupKey, int>& max_observed_levels,
    int parent_index = -1,
    std::optional<absl::string_view> custom_name = std::nullopt) {
  std::string thread_group_name;
  if (custom_name.has_value()) {
    thread_group_name = std::string(*custom_name);
  } else {
    const auto it = trace_info.thread_names.find({pid, tid});
    thread_group_name = it == trace_info.thread_names.end()
                            ? GetDefaultThreadName(tid)
                            : it->second;
  }

  bool expanded =
      GetExpandedState(kThreadNestingLevel, thread_group_name,
                       process_group_name, default_expanded, expanded_states);

  int child_index = static_cast<int>(data.groups.size());
  data.groups.push_back({.type = Group::Type::kFlame,
                         .name = thread_group_name,
                         .start_level = current_level,
                         .nesting_level = kThreadNestingLevel,
                         .expanded = expanded,
                         .parent_index = parent_index,
                         .pid = pid,
                         .tid = tid});

  if (parent_index != -1) {
    data.groups[parent_index].child_indices.push_back(child_index);
  }

  int start_level = current_level;
  int max_level = start_level;

  PopulateThreadTrackEvents(events, start_level, max_level, data, bounds,
                            trace_info, thread_group_name);

  const GroupKey group_key{
      .nesting_level = kThreadNestingLevel,
      .name = thread_group_name,
      .parent_name = process_group_name,
  };

  const int current_levels = events.empty() ? 1 : (max_level - start_level + 1);
  int& max_observed = max_observed_levels[group_key];
  const int level_count = std::max(current_levels, max_observed);
  max_observed = level_count;
  // NOMUTANTS(SBR): Timeline::SetTimelineData backfills level_count from level
  // difference if <= 0; removal is unobservable through Timeline public API.
  data.groups.back().level_count = level_count;

  current_level = start_level + level_count;
  auto [it, inserted] = thread_levels.try_emplace(
      {pid, tid}, ThreadLevelInfo{start_level, current_level});
  if (!inserted) {
    it->second.start_level = std::min(it->second.start_level, start_level);
    it->second.end_level = std::max(it->second.end_level, current_level);
  }

  if (max_level == start_level && !expanded_states.contains(group_key)) {
    data.groups.back().expanded = true;
  }
}

template <typename EventT>
void PopulateCounterTrack(
    ProcessId pid, const std::string& name,
    absl::Span<const CounterEvent* const> events,
    const TraceInformation<EventT>& trace_info, int& current_level,
    FlameChartTimelineData& data, TimeBounds& bounds,
    const std::string& process_group_name, bool default_expanded,
    const absl::btree_map<GroupKey, bool>& expanded_states,
    int parent_index = -1) {
  Group group;
  group.type = Group::Type::kCounter;
  group.name = name;
  group.nesting_level = kCounterNestingLevel;
  group.start_level = current_level;
  group.parent_index = parent_index;
  group.pid = pid;

  // Counters always take one level, so force them to be expanded.
  group.expanded = true;
  group.level_count = 1;

  size_t total_entries = 0;
  // The number of counter events per counter track won't be too large, so
  // it's fine to iterate twice to reserve vector capacity.
  for (const CounterEvent* event : events) {
    total_entries += event->timestamps.size();
  }

  CounterData counter_data;
  if (!events.empty()) {
    counter_data.event_stats = events.front()->event_stats;
    counter_data.unit = events.front()->unit;
  }
  counter_data.timestamps.reserve(total_entries);
  counter_data.values.reserve(total_entries);

  // Bulk insert all data first.
  for (const CounterEvent* event : events) {
    if (event->timestamps.empty()) continue;

    counter_data.timestamps.insert(counter_data.timestamps.end(),
                                   event->timestamps.begin(),
                                   event->timestamps.end());
    counter_data.values.insert(counter_data.values.end(), event->values.begin(),
                               event->values.end());

    // Use pre-calculated min/max values from the event.
    counter_data.min_value = std::min(counter_data.min_value, event->min_value);
    counter_data.max_value = std::max(counter_data.max_value, event->max_value);
  }

  if (!counter_data.values.empty()) {
    // Timestamps are sorted, so we can just look at the first and last
    // elements.
    bounds.min = std::min(bounds.min, counter_data.timestamps.front());
    bounds.max = std::max(bounds.max, counter_data.timestamps.back());
  }

  int child_index = static_cast<int>(data.groups.size());

  data.groups.push_back(std::move(group));

  if (parent_index != -1) {
    data.groups[parent_index].child_indices.push_back(child_index);
  }

  data.counter_data_by_group_index[child_index] = std::move(counter_data);

  // Increment the level by one for the next group. This will be used for binary
  // search for the visible groups.
  current_level++;
}

struct ProcessEventGroups {
  absl::btree_map<std::string, std::vector<const TraceEvent*>> async_groups;
  absl::btree_map<ThreadId, std::vector<const TraceEvent*>> sync_groups;
};

// Groups events for a process into async event groups (keyed by name) and
// sync event groups (keyed by ThreadId), ensuring threads with only async
// events do not produce duplicate sync tracks, while retaining 0-event known
// threads and async tracks.
ProcessEventGroups CollectProcessEventGroups(
    ProcessId pid, const TraceInformation<TraceEvent>& trace_info,
    const absl::btree_set<std::pair<ProcessId, ThreadId>>& known_threads,
    const absl::btree_map<ProcessId, absl::btree_set<std::string>>&
        known_async_tracks) {
  ProcessEventGroups groups;
  absl::flat_hash_set<ThreadId> async_only_tids;

  const auto it_events = trace_info.events_by_pid_tid.find(pid);
  if (it_events != trace_info.events_by_pid_tid.end()) {
    for (const auto& [tid, tid_events] : it_events->second) {
      bool has_async = false;
      bool has_sync = false;
      for (const TraceEvent* event : tid_events) {
        if (event->is_async) {
          has_async = true;
          groups.async_groups[event->name].push_back(event);
        } else {
          has_sync = true;
          groups.sync_groups[tid].push_back(event);
        }
      }
      if (has_async && !has_sync) {
        async_only_tids.insert(tid);
      }
    }
  }

  // Include known async tracks for this pid so 0-event tracks are not omitted.
  if (const auto it_async = known_async_tracks.find(pid);
      it_async != known_async_tracks.end()) {
    for (const std::string& name : it_async->second) {
      groups.async_groups.try_emplace(name);
    }
  }

  // Include threads from thread_names for this pid (excluding tids that only
  // have async events, which are already grouped into async tracks).
  for (auto it = trace_info.thread_names.lower_bound({pid, 0});
       it != trace_info.thread_names.end() && it->first.first == pid; ++it) {
    if (it->first.second < kSyntheticOpenSpanTidBit &&
        !async_only_tids.contains(it->first.second)) {
      groups.sync_groups.try_emplace(it->first.second);
    }
  }

  // Include known sync threads for this pid so 0-event threads are not omitted.
  for (auto it = known_threads.lower_bound({pid, 0});
       it != known_threads.end() && it->first == pid; ++it) {
    if (it->second < kSyntheticOpenSpanTidBit &&
        !async_only_tids.contains(it->second)) {
      groups.sync_groups.try_emplace(it->second);
    }
  }

  return groups;
}

// Populates named async tracks for a process using synthetic ThreadIds
// (starting at 0x80000000).
void PopulateAsyncProcessTracks(
    ProcessId pid,
    const absl::btree_map<std::string, std::vector<const TraceEvent*>>&
        async_groups,
    const std::string& process_group_name,
    TraceInformation<TraceEvent>& trace_info, int& current_level,
    FlameChartTimelineData& data, TimeBounds& bounds,
    absl::btree_map<std::pair<ProcessId, ThreadId>, ThreadLevelInfo>&
        thread_levels,
    const absl::btree_map<GroupKey, bool>& expanded_states,
    absl::flat_hash_map<GroupKey, int>& max_observed_levels, int parent_index) {
  for (const auto& [name, named_events] : async_groups) {
    const ThreadId tid = named_events.empty() ? 0 : named_events.front()->tid;
    PopulateThreadTrack(pid, tid, named_events, trace_info, current_level, data,
                        bounds, thread_levels, process_group_name,
                        /*default_expanded=*/true, expanded_states,
                        max_observed_levels, parent_index, name);
  }
}

// Populates standard sync thread tracks for a process, sorted according to
// metadata ordering, and filters out empty non-primary tracks for MPMD traces.
void PopulateSyncProcessTracks(
    ProcessId pid,
    const absl::btree_map<ThreadId, std::vector<const TraceEvent*>>&
        sync_groups,
    const std::string& process_group_name,
    TraceInformation<TraceEvent>& trace_info, int& current_level,
    FlameChartTimelineData& data, TimeBounds& bounds,
    absl::btree_map<std::pair<ProcessId, ThreadId>, ThreadLevelInfo>&
        thread_levels,
    const absl::btree_map<GroupKey, bool>& expanded_states,
    absl::flat_hash_map<GroupKey, int>& max_observed_levels, int parent_index) {
  std::vector<ThreadId> sorted_tids;
  sorted_tids.reserve(sync_groups.size());
  for (const auto& [tid, _] : sync_groups) {
    sorted_tids.push_back(tid);
  }

  absl::c_stable_sort(sorted_tids, [&](ThreadId a, ThreadId b) {
    return CompareThreadsForSort(pid, trace_info, a, b);
  });

  const auto it_xla_tid = trace_info.xla_modules_tids.find(pid);
  const auto it_sort = trace_info.process_sort_indices.find(pid);
  const bool is_unranked_mpmd_process =
      !trace_info.process_sort_indices.empty() &&
      (it_sort == trace_info.process_sort_indices.end() ||
       it_sort->second >= kMpmdUnrankedSortIndexBase);

  for (const ThreadId tid : sorted_tids) {
    const auto it_sync = sync_groups.find(tid);
    absl::Span<const TraceEvent* const> events;
    if (it_sync != sync_groups.end()) {
      events = it_sync->second;
    }
    if (trace_info.is_mpmd && events.empty()) {
      const auto it_name = trace_info.thread_names.find({pid, tid});
      const absl::string_view thread_name =
          (it_name != trace_info.thread_names.end())
              ? absl::string_view(it_name->second)
              : "";
      const bool is_primary_mpmd_track =
          (thread_name == kXlaModules ||
           (it_xla_tid != trace_info.xla_modules_tids.end() &&
            it_xla_tid->second == tid));
      if (!is_primary_mpmd_track || is_unranked_mpmd_process) {
        continue;
      }
    }
    PopulateThreadTrack(pid, tid, events, trace_info, current_level, data,
                        bounds, thread_levels, process_group_name,
                        /*default_expanded=*/true, expanded_states,
                        max_observed_levels, parent_index);
  }
}

// Populates counter tracks for a process, merging events from parsed counter
// events and known counter definitions.
void PopulateProcessCounterTracks(
    ProcessId pid, const std::string& process_group_name,
    const TraceInformation<TraceEvent>& trace_info, int& current_level,
    FlameChartTimelineData& data, TimeBounds& bounds,
    const absl::btree_map<GroupKey, bool>& expanded_states,
    const absl::btree_map<ProcessId, absl::btree_set<std::string>>&
        known_counters,
    int parent_index) {
  const auto it_known_counters = known_counters.find(pid);
  if (it_known_counters == known_counters.end()) {
    return;
  }
  absl::btree_map<std::string, std::vector<const CounterEvent*>>
      combined_counters;
  const auto it_counters = trace_info.counters_by_pid_name.find(pid);
  if (it_counters != trace_info.counters_by_pid_name.end() &&
      !it_counters->second.empty()) {
    combined_counters = it_counters->second;
  }
  for (const std::string& counter_name : it_known_counters->second) {
    combined_counters.try_emplace(counter_name);
  }
  for (const auto& [name, events] : combined_counters) {
    PopulateCounterTrack(pid, name, events, trace_info, current_level, data,
                         bounds, process_group_name,
                         /*default_expanded=*/true, expanded_states,
                         parent_index);
  }
}

template <typename EventT>
bool IsAsyncProcess(ProcessId pid, const TraceInformation<EventT>& trace_info) {
  return GetAsyncProcessPriority(pid, trace_info) > 0;
}

void PopulateProcessTrack(
    ProcessId pid, TraceInformation<TraceEvent>& trace_info, int& current_level,
    FlameChartTimelineData& data, TimeBounds& bounds,
    absl::btree_map<std::pair<ProcessId, ThreadId>, ThreadLevelInfo>&
        thread_levels,
    bool default_expanded,
    const absl::btree_map<GroupKey, bool>& expanded_states,
    absl::flat_hash_map<GroupKey, int>& max_observed_levels,
    const absl::btree_set<std::pair<ProcessId, ThreadId>>& known_threads,
    const absl::btree_map<ProcessId, absl::btree_set<std::string>>&
        known_counters,
    const absl::btree_map<ProcessId, absl::btree_set<std::string>>&
        known_async_tracks) {
  const auto it_events = trace_info.events_by_pid_tid.find(pid);
  const bool has_events = it_events != trace_info.events_by_pid_tid.end() &&
                          !it_events->second.empty();

  // Check if any threads exist for this PID in thread_names.
  auto it_thread_names = trace_info.thread_names.lower_bound({pid, 0});
  const bool has_named_threads =
      (it_thread_names != trace_info.thread_names.end() &&
       it_thread_names->first.first == pid);

  auto it_known_threads = known_threads.lower_bound({pid, 0});
  const bool has_known_threads = (it_known_threads != known_threads.end() &&
                                  it_known_threads->first == pid);

  const auto it_known_counters = known_counters.find(pid);
  const bool has_known_counters = it_known_counters != known_counters.end();

  const auto it_known_async = known_async_tracks.find(pid);
  const bool has_known_async = it_known_async != known_async_tracks.end();

  const bool has_thread_tracks =
      has_events || has_named_threads || has_known_threads || has_known_async;

  if (!has_thread_tracks && !has_known_counters) {
    return;
  }

  std::string process_group_name;
  if (auto it = trace_info.process_names.find(pid);
      it != trace_info.process_names.end()) {
    process_group_name = it->second;
  } else {
    process_group_name = GetDefaultProcessName(pid);
  }

  bool expanded = GetExpandedState(kProcessNestingLevel, process_group_name, "",
                                   default_expanded, expanded_states);

  std::string track_subtitle;

  const size_t separator_pos = process_group_name.find(' ');
  if (separator_pos != std::string::npos) {
    track_subtitle = process_group_name.substr(0, separator_pos);
  }

  int start_level = current_level;
  int process_index = static_cast<int>(data.groups.size());
  data.groups.push_back({.name = process_group_name,
                         .subtitle = std::move(track_subtitle),
                         .start_level = current_level,
                         .nesting_level = kProcessNestingLevel,
                         .expanded = expanded,
                         .parent_index = -1,
                         .pid = pid});

  // 1. Populate counter tracks first.
  if (has_known_counters) {
    PopulateProcessCounterTracks(pid, process_group_name, trace_info,
                                 current_level, data, bounds, expanded_states,
                                 known_counters, process_index);
  }

  // 2. Populate thread tracks (standard sync threads first, then named async
  // tracks).
  if (has_thread_tracks) {
    ProcessEventGroups groups = CollectProcessEventGroups(
        pid, trace_info, known_threads, known_async_tracks);

    // 2a. Populate standard sync thread tracks.
    PopulateSyncProcessTracks(pid, groups.sync_groups, process_group_name,
                              trace_info, current_level, data, bounds,
                              thread_levels, expanded_states,
                              max_observed_levels, process_index);

    // 2b. Populate named async tracks.
    PopulateAsyncProcessTracks(pid, groups.async_groups, process_group_name,
                               trace_info, current_level, data, bounds,
                               thread_levels, expanded_states,
                               max_observed_levels, process_index);
  }

  if (trace_info.is_mpmd && data.groups.size() == process_index + 1) {
    data.groups.pop_back();
  } else {
    data.groups[process_index].level_count = current_level - start_level;
    data.groups[process_index].has_children =
        !data.groups[process_index].child_indices.empty();
  }
}

template <typename EventT>
std::vector<ProcessId> GetSortedProcessIds(
    const TraceInformation<EventT>& trace_info,
    const absl::btree_set<std::pair<ProcessId, ThreadId>>& known_threads,
    const absl::btree_map<ProcessId, absl::btree_set<std::string>>&
        known_counters,
    const absl::btree_map<ProcessId, absl::btree_set<std::string>>&
        known_async_tracks) {
  absl::flat_hash_set<ProcessId> pid_set;
  for (const auto& [pid, _] : trace_info.process_names) {
    pid_set.insert(pid);
  }
  for (const auto& [pid, _] : trace_info.events_by_pid_tid) {
    pid_set.insert(pid);
  }
  for (const auto& [pid, _] : trace_info.counters_by_pid_name) {
    pid_set.insert(pid);
  }
  for (const auto& [key, _] : trace_info.thread_names) {
    pid_set.insert(key.first);
  }
  for (const auto& [_, events] : trace_info.flow_events_by_id) {
    for (const auto& event : events) {
      pid_set.insert(GetEndpointPid(event));
    }
  }
  for (const auto& [pid, _] : known_threads) {
    pid_set.insert(pid);
  }
  for (const auto& [pid, _] : known_counters) {
    pid_set.insert(pid);
  }
  for (const auto& [pid, _] : known_async_tracks) {
    pid_set.insert(pid);
  }

  std::vector<ProcessId> pids(pid_set.begin(), pid_set.end());

  auto get_process_sort_index = [&](ProcessId pid) -> std::optional<uint32_t> {
    if (const auto it = trace_info.process_sort_indices.find(pid);
        it != trace_info.process_sort_indices.end()) {
      return it->second;
    }
    return trace_info.is_mpmd
               ? std::make_optional(std::numeric_limits<uint32_t>::max())
               : std::nullopt;
  };
  auto get_process_name =
      [&](ProcessId pid) -> std::optional<absl::string_view> {
    if (const auto it = trace_info.process_names.find(pid);
        it != trace_info.process_names.end()) {
      return it->second;
    }
    return std::nullopt;
  };

  absl::c_stable_sort(pids, [&](ProcessId a, ProcessId b) {
    return CompareTrackMetadata(get_process_sort_index(a),
                                get_process_sort_index(b), get_process_name(a),
                                get_process_name(b), a, b);
  });
  return pids;
}

FlameChartTimelineData CreateTimelineData(
    TraceInformation<TraceEvent>& trace_info,
    absl::Span<const ProcessId> sorted_pids,
    absl::Span<const int> top_5_flow_categories, TimeBounds& bounds,
    const absl::btree_map<GroupKey, bool>& expanded_states,
    const ColorPalette& palette,
    absl::flat_hash_map<GroupKey, int>& max_observed_levels,
    const absl::btree_set<std::pair<ProcessId, ThreadId>>& known_threads,
    const absl::btree_map<ProcessId, absl::btree_set<std::string>>&
        known_counters,
    const absl::btree_map<ProcessId, absl::btree_set<std::string>>&
        known_async_tracks) {
  FlameChartTimelineData data;
  if (!trace_info.is_mpmd) {
    size_t total_events = 0;
    for (const auto& [_, events_by_tid] : trace_info.events_by_pid_tid) {
      for (const auto& [__, events] : events_by_tid) {
        total_events += events.size();
      }
    }
    if (total_events > 0) {
      data.Reserve(total_events);
    }
  }
  int current_level = 0;
  absl::btree_map<std::pair<ProcessId, ThreadId>, ThreadLevelInfo>
      thread_levels;

  for (const ProcessId pid : sorted_pids) {
    const bool default_expanded = data.groups.empty();
    PopulateProcessTrack(pid, trace_info, current_level, data, bounds,
                         thread_levels, default_expanded, expanded_states,
                         max_observed_levels, known_threads, known_counters,
                         known_async_tracks);
  }

  data.level_offsets.assign(current_level + 1, 0);
  for (size_t i = 0; i < data.entry_levels.size(); ++i) {
    ++data.level_offsets[data.entry_levels[i] + 1];
  }
  std::partial_sum(data.level_offsets.begin(), data.level_offsets.end(),
                   data.level_offsets.begin());
  data.level_event_indices.resize(data.entry_levels.size());
  std::vector<size_t> cursors = data.level_offsets;
  for (size_t i = 0; i < data.entry_levels.size(); ++i) {
    int level = data.entry_levels[i];
    data.level_event_indices[cursors[level]] = i;
    ++cursors[level];
  }
  // Events are already packed by PackTraceEvents in chronological order (start
  // time ascending, duration descending), so the CSR layout preserves this
  // order per level without needing a secondary sort pass unless incremental
  // out-of-order stitching occurred.
  auto cmp_by_start_asc_then_dur_desc = [&](int idx_a, int idx_b) {
    return data.entry_start_times[idx_a] < data.entry_start_times[idx_b] ||
           (data.entry_start_times[idx_a] == data.entry_start_times[idx_b] &&
            data.entry_total_times[idx_a] > data.entry_total_times[idx_b]);
  };
  for (int lvl = 0; lvl < current_level; ++lvl) {
    auto begin = data.level_event_indices.begin() +
                 static_cast<ptrdiff_t>(data.level_offsets[lvl]);
    auto end = data.level_event_indices.begin() +
               static_cast<ptrdiff_t>(data.level_offsets[lvl + 1]);
    if (!std::is_sorted(begin, end, cmp_by_start_asc_then_dur_desc)) {
      std::stable_sort(begin, end, cmp_by_start_asc_then_dur_desc);
    }
  }

  GenerateFlowLines(trace_info, thread_levels, top_5_flow_categories, data,
                    bounds, palette);

  data.interned_string_pool = trace_info.name_interner.TakePool();
  data.hlo_module_table = trace_info.module_interner.TakePool();
  data.hlo_op_table = trace_info.op_interner.TakePool();

  return data;
}

void SortFlowEvents(std::vector<const TraceEvent*>& events) {
  absl::c_stable_sort(events, [](const TraceEvent* a, const TraceEvent* b) {
    return std::tie(a->ts, a->pid, a->tid) < std::tie(b->ts, b->pid, b->tid);
  });
}

void SortCounterEvents(std::vector<const CounterEvent*>& events) {
  absl::c_stable_sort(events, [](const CounterEvent* a, const CounterEvent* b) {
    const auto get_ts = [](const CounterEvent* e) {
      return e->timestamps.empty() ? std::numeric_limits<Microseconds>::max()
                                   : e->timestamps.front();
    };
    return get_ts(a) < get_ts(b);
  });
}

void UpdateFlowCategories(const absl::btree_map<int, int>& flow_category_counts,
                          std::vector<int>& present_flow_categories) {
  present_flow_categories.clear();
  for (const auto& [category, _] : flow_category_counts) {
    present_flow_categories.push_back(category);
  }
}

void UpdateTimelineTimeRanges(
    Timeline& timeline, const TimeBounds& bounds, bool has_known_tracks,
    const std::optional<std::pair<Milliseconds, Milliseconds>>&
        visible_range_from_url,
    const std::optional<std::pair<Milliseconds, Milliseconds>>& full_timespan) {
  // Don't need to check for max_time because the TimeRange constructor will
  // handle any potential issues with max_time.
  if (bounds.min < std::numeric_limits<Microseconds>::max()) {
    timeline.set_fetched_data_time_range({bounds.min, bounds.max});

    // TODO: b/460265076 - Change the logic here for visible range after
    // we decided how to handle the visible range in url.
    if (visible_range_from_url.has_value()) {
      const Microseconds start = MillisToMicros(visible_range_from_url->first);
      const Microseconds end = MillisToMicros(visible_range_from_url->second);

      if (timeline.visible_range() == TimeRange::Zero()) {
        timeline.SetVisibleRange({start, end});
      }
    } else if (timeline.visible_range() == TimeRange::Zero()) {
      // If the visible range is not zero, we just keep it. This happens when
      // the incremental loading is triggered and we don't want to override the
      // current visible range.
      timeline.SetVisibleRange({bounds.min, bounds.max});
    }
  } else if (!has_known_tracks) {
    timeline.set_fetched_data_time_range(TimeRange::Zero());
    timeline.SetVisibleRange(TimeRange::Zero());
  }

  if (full_timespan.has_value()) {
    const Microseconds start = MillisToMicros(full_timespan->first);
    const Microseconds end = MillisToMicros(full_timespan->second);
    timeline.set_data_time_range({start, end});
  } else {
    timeline.set_data_time_range(timeline.fetched_data_time_range());
  }
}

}  // namespace

bool DataProvider::HasKnownTracks() const {
  return !known_threads_.empty() || !known_counters_.empty() ||
         !known_async_tracks_.empty();
}

// Processes a vector of TraceEvent structs.
// This function is independent of Emscripten types.
void DataProvider::Reset() {
  present_flow_categories_.clear();
  process_names_.clear();
  thread_names_.clear();
  process_sort_indices_.clear();
  thread_sort_indices_.clear();
  xla_modules_tids_.clear();
  max_observed_levels_.clear();
  known_threads_.clear();
  known_async_tracks_.clear();
  known_async_processes_.clear();
  known_counters_.clear();
}

// Processes a vector of TraceEvent structs.
// This function is independent of Emscripten types.
void DataProvider::ProcessTraceEvents(const ParsedTraceEvents& parsed_events,
                                      Timeline& timeline) {
  timeline.set_mpmd_pipeline_view_enabled(parsed_events.mpmd_pipeline_view);
  timeline.set_time_axis_unit(parsed_events.time_axis_unit);
  if (parsed_events.flame_events.empty() &&
      parsed_events.counter_events.empty() &&
      parsed_events.flow_events.empty() && !HasKnownTracks()) {
    FinalizeTimelineUpdate(timeline, {}, TimeBounds{},
                           parsed_events.visible_range_from_url,
                           parsed_events.full_timespan);
    return;
  }

  TraceInformation<TraceEvent> trace_info;
  trace_info.is_mpmd = parsed_events.mpmd_pipeline_view;
  trace_info.process_names = process_names_;
  trace_info.thread_names = thread_names_;
  trace_info.process_sort_indices = process_sort_indices_;
  trace_info.thread_sort_indices = thread_sort_indices_;
  trace_info.xla_modules_tids = xla_modules_tids_;
  trace_info.async_processes_by_events = known_async_processes_;

  absl::btree_map<int, int> flow_category_counts;
  for (const TraceEvent& event : parsed_events.flame_events) {
    switch (event.ph) {
      case Phase::kMetadata:
        HandleMetadataEvent(event, trace_info);
        break;
      case Phase::kComplete:
      case Phase::kInstant:
      case Phase::kInstantDeprecated:
        HandleCompleteEvent(event, trace_info);
        // TODO(b/542228691): Async flow events are skipped here until async
        // track level lookup is supported in GenerateFlowLines.
        if (HasFlowId(event) && !event.is_async) {
          HandleFlowEvent(event, trace_info, flow_category_counts);
        }
        break;
      case Phase::kFlowStart:
      case Phase::kFlowEnd:
        if (HasFlowId(event)) {
          HandleFlowEvent(event, trace_info, flow_category_counts);
        }
        break;
      default:
        // Ignore other event types.
        // TODO: b/444013042 - Check the backend to confirm if we need to handle
        // more types in the future.
        break;
    }
  }

  MergeProcessSortIndices(trace_info);
  for (const TraceEvent& event : parsed_events.flow_events) {
    if (!event.is_async) {
      HandleFlowEvent(event, trace_info, flow_category_counts);
    }
  }

  UpdateFlowCategories(flow_category_counts, present_flow_categories_);

  for (const CounterEvent& event : parsed_events.counter_events) {
    HandleCounterEvent(event, trace_info);
  }

  // Update persistent members from parsed events and metadata.
  process_names_ = trace_info.process_names;
  thread_names_ = trace_info.thread_names;
  process_sort_indices_ = trace_info.process_sort_indices;
  thread_sort_indices_ = trace_info.thread_sort_indices;
  xla_modules_tids_ = trace_info.xla_modules_tids;
  known_async_processes_ = trace_info.async_processes_by_events;

  for (const auto& [pid, events_by_tid] : trace_info.events_by_pid_tid) {
    for (const auto& [tid, events] : events_by_tid) {
      bool has_sync_events = false;
      for (const TraceEvent* event : events) {
        if (event->is_async) {
          known_async_tracks_[pid].insert(event->name);
        } else {
          has_sync_events = true;
        }
      }
      if (has_sync_events) {
        known_threads_.insert({pid, tid});
      }
    }
  }
  for (const auto& [_, events] : trace_info.flow_events_by_id) {
    for (const TraceEvent* event : events) {
      if (event->tid < kSyntheticOpenSpanTidBit) {
        known_threads_.insert({event->pid, event->tid});
      }
    }
  }
  for (const CounterEvent& event : parsed_events.counter_events) {
    known_counters_[event.pid].insert(event.name);
  }

  // Ensure all pids/tids from flow events are registered so that thread tracks
  // are created for them, which is required for level calculation.
  for (const auto& [id, events] : trace_info.flow_events_by_id) {
    for (const TraceEvent* event : events) {
      trace_info.events_by_pid_tid[event->pid].try_emplace(event->tid);
    }
  }

  // Sort events, first by timestamp (ascending), then by duration
  // (descending).
  for (auto& [pid, events_by_tid] : trace_info.events_by_pid_tid) {
    for (auto& [tid, events] : events_by_tid) {
      absl::c_stable_sort(events, [](const TraceEvent* a, const TraceEvent* b) {
        if (a->ts != b->ts) {
          return a->ts < b->ts;
        }
        return a->dur > b->dur;
      });
    }
  }

  for (auto& [id, events] : trace_info.flow_events_by_id) {
    SortFlowEvents(events);
  }

  for (auto& [pid, counters_by_name] : trace_info.counters_by_pid_name) {
    for (auto& [name, events] : counters_by_name) {
      SortCounterEvents(events);
    }
  }

  TimeBounds time_bounds;

  const absl::btree_map<GroupKey, bool> expanded_states =
      GetRestoredExpandedStates(timeline.timeline_data().groups);
  if (!parsed_events.visible_range_from_url.has_value() &&
      !timeline.stash_full_overview_on_next_update()) {
    timeline.ClearSavedFullOverview();
  }
  timeline.ReleaseTimelineEntriesPreservingState();

  const std::vector<ProcessId> sorted_pids = GetSortedProcessIds(
      trace_info, known_threads_, known_counters_, known_async_tracks_);

  FlameChartTimelineData data = CreateTimelineData(
      trace_info, sorted_pids, GetTop5FlowCategories(flow_category_counts),
      time_bounds, expanded_states, timeline.GetPalette(), max_observed_levels_,
      known_threads_, known_counters_, known_async_tracks_);

  FinalizeTimelineUpdate(timeline, std::move(data), time_bounds,
                         parsed_events.visible_range_from_url,
                         parsed_events.full_timespan);
}

void DataProvider::FinalizeTimelineUpdate(
    Timeline& timeline, FlameChartTimelineData data, const TimeBounds& bounds,
    std::optional<std::pair<Milliseconds, Milliseconds>> visible_range_from_url,
    std::optional<std::pair<Milliseconds, Milliseconds>> full_timespan) {
  const bool was_initial_overview_load =
      (timeline.visible_range() == TimeRange::Zero() &&
       !visible_range_from_url.has_value());
  if (data.dropped_level_cap_events > 0) {
    LOG(ERROR) << "Trace exceeded uint16_t flame chart level cap (65,534); "
               << "dropped " << data.dropped_level_cap_events << " events.";
  }
  timeline.SetTimelineData(std::move(data));

  UpdateTimelineTimeRanges(timeline, bounds, HasKnownTracks(),
                           visible_range_from_url, full_timespan);

  if (was_initial_overview_load ||
      (timeline.data_time_range().duration() > 0 &&
       timeline.last_fetch_request_range().duration() >=
           0.9 * timeline.data_time_range().duration())) {
    timeline.ClearSavedFullOverview();
    timeline.MarkCurrentDataAsFullOverview();
  }
}

namespace {

struct TrackCompleteSeriesSource {
  std::vector<const xprof::TraceEventSeries*> proto_series;
  std::vector<const WireSeriesView*> wire_series;

  bool empty() const { return proto_series.empty() && wire_series.empty(); }
};

struct PackedTrackRange {
  size_t entry_begin = 0;
  size_t entry_end = 0;
  int start_level = 0;
  int max_level = 0;
};

void AppendSeriesModuleIntervals(
    const xprof::TraceEventSeries& series,
    absl::FunctionRef<absl::string_view(uint32_t)> get_interned_string,
    TableInterner& module_interner,
    std::vector<ModuleInterval>& raw_module_intervals,
    uint64_t* max_end_ps_out = nullptr) {
  const xprof::TraceEventSeriesMetadata& meta = series.metadata();
  uint64_t cur_ts_ps = 0;
  uint64_t local_max_end_ps = 0;
  const int xla_deltas_count = series.deltas_size();
  for (int i = 0; i < xla_deltas_count; ++i) {
    cur_ts_ps += series.deltas(i);
    const uint32_t name_ref =
        i < series.name_refs_size() ? series.name_refs(i) : meta.name_ref();
    const uint64_t dur_ps =
        i < series.durations_size() ? series.durations(i) : 0;
    local_max_end_ps = std::max(local_max_end_ps, cur_ts_ps + dur_ps);
    const Microseconds start = cur_ts_ps / 1000000.0;
    const Microseconds dur = dur_ps / 1000000.0;
    const uint32_t mod_idx =
        module_interner.Intern(get_interned_string(name_ref));
    raw_module_intervals.push_back(
        {.start = start, .end = start + dur, .module_index = mod_idx});
  }
  if (max_end_ps_out != nullptr) {
    *max_end_ps_out = std::max(*max_end_ps_out, local_max_end_ps);
  }
}

}  // namespace

void DataProvider::ProcessTraceEvents(
    const xprof::TraceDataResponse& response, Timeline& timeline,
    std::optional<std::pair<Milliseconds, Milliseconds>>
        visible_range_from_url) {
  ProcessProtoTraceEventsInternal(
      response, /*complete_event_slices=*/{},
      /*use_wire_complete_series=*/false,
      /*async_events_already_paired=*/false, []() {}, timeline,
      visible_range_from_url);
}

bool DataProvider::ProcessSerializedTraceEvents(
    std::string&& serialized_proto, Timeline& timeline,
    std::optional<std::pair<Milliseconds, Milliseconds>> visible_range_from_url,
    std::vector<std::pair<std::string, bool>>* details_out) {
  std::vector<absl::string_view> complete_event_slices;
  std::string non_complete_wire;
  if (!SplitSerializedTraceDataResponse(serialized_proto, complete_event_slices,
                                        non_complete_wire)) {
    return false;
  }
  xprof::TraceDataResponse response;
  if (!response.ParseFromString(non_complete_wire)) {
    return false;
  }
  std::string().swap(non_complete_wire);

  std::vector<std::pair<std::string, bool>> extracted_details;
  if (details_out != nullptr) {
    extracted_details.reserve(response.details_size());
    for (const xprof::DetailItem& detail : response.details()) {
      extracted_details.emplace_back(detail.name(), detail.value());
    }
  }

  auto release_input_buffers = [&response, &serialized_proto,
                                &complete_event_slices]() {
    complete_event_slices.clear();
    complete_event_slices.shrink_to_fit();
    response.Clear();
    std::string().swap(serialized_proto);
  };

  if (!ProcessProtoTraceEventsInternal(
          response, complete_event_slices, /*use_wire_complete_series=*/true,
          /*async_events_already_paired=*/false, release_input_buffers,
          timeline, visible_range_from_url)) {
    return false;
  }

  if (details_out != nullptr) {
    *details_out = std::move(extracted_details);
  }
  return true;
}

bool DataProvider::ProcessStreamedTraceEvents(
    const xprof::TraceDataResponse& response,
    absl::Span<const absl::string_view> complete_event_slices,
    absl::FunctionRef<void()> release_input_buffers, Timeline& timeline,
    std::optional<std::pair<Milliseconds, Milliseconds>>
        visible_range_from_url) {
  return ProcessProtoTraceEventsInternal(
      response, complete_event_slices, /*use_wire_complete_series=*/true,
      /*async_events_already_paired=*/true, release_input_buffers, timeline,
      visible_range_from_url);
}

bool DataProvider::ProcessProtoTraceEventsInternal(
    const xprof::TraceDataResponse& response,
    absl::Span<const absl::string_view> complete_event_slices,
    bool use_wire_complete_series, bool async_events_already_paired,
    absl::FunctionRef<void()> release_input_buffers, Timeline& timeline,
    std::optional<std::pair<Milliseconds, Milliseconds>>
        visible_range_from_url) {
  bool mpmd_pipeline_view = false;
  for (const xprof::DetailItem& detail : response.details()) {
    if (detail.name() == "mpmd_pipeline_view") {
      mpmd_pipeline_view = detail.value();
    }
  }

  std::optional<std::pair<Milliseconds, Milliseconds>> full_timespan;
  if (response.has_full_timespan_start_ps() &&
      response.has_full_timespan_end_ps()) {
    const Milliseconds start_ms = response.full_timespan_start_ps() / 1e9;
    const Milliseconds end_ms = response.full_timespan_end_ps() / 1e9;
    if (start_ms <= end_ms) {
      full_timespan = std::make_pair(start_ms, end_ms);
    }
  }

  const uint32_t interned_strings_count =
      static_cast<uint32_t>(response.interned_strings_size());
  const auto get_interned_string =
      [&response, interned_strings_count](uint32_t ref) -> absl::string_view {
    return ref < interned_strings_count
               ? absl::string_view(response.interned_strings(ref))
               : "";
  };

  std::vector<std::string> interned_string_pool;
  interned_string_pool.reserve(interned_strings_count);
  for (const std::string& str : response.interned_strings()) {
    interned_string_pool.push_back(str);
  }

  TraceInformation<ProtoEventRef> trace_info;
  trace_info.is_mpmd = mpmd_pipeline_view;
  trace_info.process_names = process_names_;
  trace_info.thread_names = thread_names_;
  trace_info.process_sort_indices = process_sort_indices_;
  trace_info.thread_sort_indices = thread_sort_indices_;
  trace_info.xla_modules_tids = xla_modules_tids_;
  trace_info.async_processes_by_events = known_async_processes_;
  trace_info.name_interner.SetInitialPool(std::move(interned_string_pool));

  for (const xprof::Process& process : response.metadata().processes()) {
    if (!process.name().empty()) {
      trace_info.process_names[process.id()] = process.name();
    }
    if (process.has_sort_index()) {
      trace_info.incoming_process_sort_indices[process.id()] =
          process.sort_index();
    }
    for (const xprof::Thread& thread : process.threads()) {
      if (!thread.name().empty()) {
        trace_info.thread_names[{process.id(), thread.id()}] = thread.name();
        if (thread.name() == kXlaModules) {
          trace_info.xla_modules_tids[process.id()] = thread.id();
        } else if (thread.name() == kXlaOps) {
          trace_info.xla_ops_tids[process.id()] = thread.id();
        }
      }
      if (thread.has_sort_index()) {
        trace_info.thread_sort_indices[{process.id(), thread.id()}] =
            thread.sort_index();
      }
    }
  }
  MergeProcessSortIndices<ProtoEventRef>(trace_info);

  struct AsyncTrackData {
    ThreadId tid = 0;
    std::vector<TrackEvent> events;
  };
  absl::btree_map<ProcessId, absl::btree_map<std::string, AsyncTrackData>>
      async_events_by_pid_name;
  struct OpenAsyncEvent {
    TrackEvent ev;
    uint64_t ts_ps = 0;
    ProcessId pid = 0;
    ThreadId tid = 0;
    absl::string_view name;
  };
  absl::flat_hash_map<std::tuple<ProcessId, uint32_t, uint64_t>, OpenAsyncEvent>
      open_async_events;
  std::vector<OpenAsyncEvent> unclosed_async_events;
  Microseconds max_async_end_us = 0.0;

  for (const xprof::TraceEventSeries& series : response.async_events()) {
    const xprof::TraceEventSeriesMetadata& metadata = series.metadata();
    const ProcessId pid = metadata.process_id();
    const ThreadId tid = metadata.thread_id();
    trace_info.async_processes_by_events.insert(pid);

    const bool has_per_event_name_refs = series.name_refs_size() > 0;
    const absl::string_view series_name =
        get_interned_string(metadata.name_ref());
    uint64_t current_ts_ps = 0;
    const int deltas_count = series.deltas_size();
    const bool has_sparse_metadata = series.event_metadata_indices_size() > 0;
    int sparse_meta_cursor = 0;
    const int sparse_meta_count = std::min(series.event_metadata_indices_size(),
                                           series.event_metadata_size());

    const bool has_rle_group_ids =
        series.group_id_run_values_size() > 0 &&
        series.group_id_run_values_size() == series.group_id_run_lengths_size();
    int rle_idx = 0;
    uint32_t rle_remaining =
        has_rle_group_ids ? series.group_id_run_lengths(0) : 0;

    for (int i = 0; i < deltas_count; ++i) {
      current_ts_ps += series.deltas(i);
      const Microseconds ts = current_ts_ps / 1000000.0;
      const Microseconds dur =
          i < series.durations_size() ? series.durations(i) / 1000000.0 : 0.0;
      uint64_t flow_id = 0;
      int32_t group_id = -1;
      uint32_t uid = 0;
      uint32_t hlo_op_ref = 0;
      uint32_t hlo_module_ref = 0;

      if (has_rle_group_ids) {
        while (rle_remaining == 0 &&
               rle_idx + 1 < series.group_id_run_lengths_size()) {
          ++rle_idx;
          rle_remaining = series.group_id_run_lengths(rle_idx);
        }
        if (rle_remaining > 0) {
          const int64_t rle_group_id = series.group_id_run_values(rle_idx);
          if (rle_group_id >= 0) {
            group_id = static_cast<int32_t>(rle_group_id);
          }
          --rle_remaining;
        }
      }

      const xprof::TraceEventMetadata* ev_meta = nullptr;
      if (has_sparse_metadata) {
        while (sparse_meta_cursor < sparse_meta_count &&
               series.event_metadata_indices(sparse_meta_cursor) <
                   static_cast<uint32_t>(i)) {
          ++sparse_meta_cursor;
        }
        if (sparse_meta_cursor < sparse_meta_count &&
            series.event_metadata_indices(sparse_meta_cursor) ==
                static_cast<uint32_t>(i)) {
          ev_meta = &series.event_metadata(sparse_meta_cursor);
          ++sparse_meta_cursor;
        }
      } else if (i < series.event_metadata_size()) {
        ev_meta = &series.event_metadata(i);
      }

      if (ev_meta != nullptr) {
        flow_id = ev_meta->flow_id();
        if (group_id < 0 && ev_meta->group_id() != 0) {
          group_id = static_cast<int32_t>(ev_meta->group_id());
        }
        uid = ev_meta->serial();
        if (ev_meta->has_hlo_op_ref() &&
            ev_meta->hlo_op_ref() < interned_strings_count) {
          hlo_op_ref = ev_meta->hlo_op_ref();
        }
        if (ev_meta->has_hlo_module_ref() &&
            ev_meta->hlo_module_ref() < interned_strings_count) {
          hlo_module_ref = ev_meta->hlo_module_ref();
        }
      }

      const uint32_t name_ref =
          (has_per_event_name_refs && i < series.name_refs_size())
              ? series.name_refs(i)
              : metadata.name_ref();
      const absl::string_view ev_name =
          has_per_event_name_refs ? get_interned_string(name_ref) : series_name;
      const TrackEvent ev{
          .ts = ts,
          .dur = dur,
          .flow_id = flow_id,
          .group_id = group_id,
          .name_index = name_ref < interned_strings_count ? name_ref : 0,
          .uid = uid,
          .hlo_op_ref = hlo_op_ref,
          .hlo_module_ref = hlo_module_ref,
      };

      if (async_events_already_paired || dur > 0.0) {
        max_async_end_us = std::max(max_async_end_us, ts + dur);
        AsyncTrackData& track =
            async_events_by_pid_name[pid][std::string(ev_name)];
        if (track.events.empty()) {
          track.tid = tid;
        }
        track.events.push_back(ev);
      } else {
        max_async_end_us = std::max(max_async_end_us, ts);
        const std::tuple<ProcessId, uint32_t, uint64_t> key(pid, name_ref,
                                                            flow_id);
        const auto [it, inserted] = open_async_events.try_emplace(
            key, OpenAsyncEvent{.ev = ev,
                                .ts_ps = current_ts_ps,
                                .pid = pid,
                                .tid = tid,
                                .name = ev_name});
        if (!inserted) {
          OpenAsyncEvent& begin = it->second;
          if (current_ts_ps > begin.ts_ps) {
            begin.ev.dur = (current_ts_ps - begin.ts_ps) / 1000000.0;
          }
          if (begin.ev.uid == 0 && ev.uid != 0) {
            begin.ev.uid = ev.uid;
          }
          if (begin.ev.group_id < 0 && ev.group_id >= 0) {
            begin.ev.group_id = ev.group_id;
          }
          if (begin.ev.hlo_op_ref == 0 && ev.hlo_op_ref != 0) {
            begin.ev.hlo_op_ref = ev.hlo_op_ref;
          }
          if (begin.ev.hlo_module_ref == 0 && ev.hlo_module_ref != 0) {
            begin.ev.hlo_module_ref = ev.hlo_module_ref;
          }
          AsyncTrackData& track =
              async_events_by_pid_name[pid][std::string(begin.name)];
          if (track.events.empty()) {
            track.tid = begin.tid;
          }
          track.events.push_back(begin.ev);
          open_async_events.erase(it);
        }
      }
    }
    if (!has_per_event_name_refs) {
      if (const auto it = open_async_events.find({pid, metadata.name_ref(), 0});
          it != open_async_events.end()) {
        unclosed_async_events.push_back(it->second);
        open_async_events.erase(it);
      }
    } else {
      for (int i = 0; i < series.name_refs_size(); ++i) {
        if (const auto it =
                open_async_events.find({pid, series.name_refs(i), 0});
            it != open_async_events.end()) {
          unclosed_async_events.push_back(it->second);
          open_async_events.erase(it);
        }
      }
    }
  }
  if (!open_async_events.empty()) {
    std::vector<OpenAsyncEvent> remaining;
    remaining.reserve(open_async_events.size());
    for (const auto& [_, open_ev] : open_async_events) {
      remaining.push_back(open_ev);
    }
    absl::c_sort(remaining, [](const OpenAsyncEvent& a,
                               const OpenAsyncEvent& b) {
      return std::tie(a.pid, a.tid, a.ev.ts, a.ev.flow_id, a.ev.name_index) <
             std::tie(b.pid, b.tid, b.ev.ts, b.ev.flow_id, b.ev.name_index);
    });
    for (const OpenAsyncEvent& open_ev : remaining) {
      unclosed_async_events.push_back(open_ev);
    }
    open_async_events.clear();
  }

  std::vector<WireSeriesView> wire_complete_series;
  absl::btree_map<ProcessId,
                  absl::btree_map<ThreadId, TrackCompleteSeriesSource>>
      complete_series_by_pid_tid;
  size_t total_complete_events = 0;

  if (use_wire_complete_series) {
    wire_complete_series.resize(complete_event_slices.size());
    for (size_t i = 0; i < complete_event_slices.size(); ++i) {
      if (!TryParseWireSeriesView(complete_event_slices[i],
                                  wire_complete_series[i],
                                  /*max_end_ps_out=*/nullptr)) {
        return false;
      }
    }
    for (const WireSeriesView& view : wire_complete_series) {
      if (view.deltas_count == 0) continue;
      complete_series_by_pid_tid[view.process_id][view.thread_id]
          .wire_series.push_back(&view);
      known_threads_.insert({view.process_id, view.thread_id});
      total_complete_events += view.deltas_count;
    }
  } else {
    for (const xprof::TraceEventSeries& series : response.complete_events()) {
      const int deltas_count = series.deltas_size();
      if (deltas_count == 0) continue;
      const ProcessId pid = series.metadata().process_id();
      const ThreadId tid = series.metadata().thread_id();
      complete_series_by_pid_tid[pid][tid].proto_series.push_back(&series);
      known_threads_.insert({pid, tid});
      total_complete_events += static_cast<size_t>(deltas_count);
    }
  }

  timeline.set_mpmd_pipeline_view_enabled(mpmd_pipeline_view);
  known_async_processes_ = trace_info.async_processes_by_events;

  std::vector<CounterEvent> counter_events;
  counter_events.reserve(response.counter_events_size());
  uint64_t max_counter_end_ps = 0;
  for (const xprof::TraceEventSeries& series : response.counter_events()) {
    const xprof::TraceEventSeriesMetadata& metadata = series.metadata();
    CounterEvent ev;
    ev.pid = metadata.process_id();
    ev.name = std::string(get_interned_string(metadata.name_ref()));
    if (metadata.has_event_stats_ref()) {
      ev.event_stats =
          std::string(get_interned_string(metadata.event_stats_ref()));
    }
    const int deltas_count = series.deltas_size();
    ev.timestamps.reserve(deltas_count);
    ev.values.reserve(deltas_count);
    uint64_t current_ts_ps = 0;
    for (int i = 0; i < deltas_count; ++i) {
      current_ts_ps += series.deltas(i);
      ev.timestamps.push_back(current_ts_ps / 1000000.0);
      double val = 0.0;
      if (i < series.event_metadata_size()) {
        const xprof::TraceEventMetadata& ev_meta = series.event_metadata(i);
        if (ev_meta.has_counter_value_double()) {
          val = ev_meta.counter_value_double();
        } else if (ev_meta.has_counter_value_uint64()) {
          val = static_cast<double>(ev_meta.counter_value_uint64());
        }
      }
      ev.values.push_back(val);
      ev.min_value = std::min(ev.min_value, val);
      ev.max_value = std::max(ev.max_value, val);
    }
    if (deltas_count > 0) {
      max_counter_end_ps = std::max(max_counter_end_ps, current_ts_ps);
    }
    known_counters_[ev.pid].insert(ev.name);
    counter_events.push_back(std::move(ev));
  }
  absl::btree_map<
      ProcessId, absl::btree_map<std::string, std::vector<const CounterEvent*>>>
      counters_by_pid_name;
  for (const CounterEvent& ev : counter_events) {
    counters_by_pid_name[ev.pid][ev.name].push_back(&ev);
  }
  for (auto& [pid, counters_by_name] : counters_by_pid_name) {
    for (auto& [name, events] : counters_by_name) {
      absl::c_stable_sort(
          events, [](const CounterEvent* a, const CounterEvent* b) {
            const auto get_ts = [](const CounterEvent* e) {
              return e->timestamps.empty()
                         ? std::numeric_limits<Microseconds>::max()
                         : e->timestamps.front();
            };
            return get_ts(a) < get_ts(b);
          });
    }
  }

  Microseconds provisional_end_us =
      std::max(max_async_end_us, max_counter_end_ps / 1000000.0);
  if (response.has_full_timespan_end_ps()) {
    provisional_end_us = std::max(provisional_end_us,
                                  response.full_timespan_end_ps() / 1000000.0);
  }

  const auto base_async_events_by_pid_name =
      unclosed_async_events.empty() ? decltype(async_events_by_pid_name){}
                                    : async_events_by_pid_name;
  const auto populate_unclosed_async_events = [&](Microseconds trace_end_us) {
    if (unclosed_async_events.empty()) return;
    async_events_by_pid_name = base_async_events_by_pid_name;
    for (OpenAsyncEvent unclosed : unclosed_async_events) {
      unclosed.ev.dur = std::max(0.0, trace_end_us - unclosed.ev.ts);
      AsyncTrackData& track =
          async_events_by_pid_name[unclosed.pid][std::string(unclosed.name)];
      if (track.events.empty()) {
        track.tid = unclosed.tid;
      }
      track.events.push_back(unclosed.ev);
    }
  };
  populate_unclosed_async_events(provisional_end_us);

  for (const auto& [pid, tracks_by_name] : async_events_by_pid_name) {
    for (const auto& [name, _] : tracks_by_name) {
      known_async_tracks_[pid].insert(name);
    }
  }

  if (response.metadata().processes().empty() &&
      complete_series_by_pid_tid.empty() && async_events_by_pid_name.empty() &&
      counter_events.empty() && !HasKnownTracks()) {
    release_input_buffers();
    FinalizeTimelineUpdate(timeline, {}, TimeBounds{}, visible_range_from_url,
                           full_timespan);
    return true;
  }

  size_t total_events = total_complete_events;
  for (const auto& [_, name_map] : async_events_by_pid_name) {
    for (const auto& [__, track] : name_map) {
      total_events += track.events.size();
    }
  }

  const absl::btree_map<GroupKey, bool> expanded_states =
      GetRestoredExpandedStates(timeline.timeline_data().groups);
  if (!visible_range_from_url.has_value() &&
      !timeline.stash_full_overview_on_next_update()) {
    timeline.ClearSavedFullOverview();
  }
  timeline.ReleaseTimelineEntriesPreservingState();

  FlameChartTimelineData data;
  const std::vector<ProcessId> sorted_pids = GetSortedProcessIds<ProtoEventRef>(
      trace_info, known_threads_, known_counters_, known_async_tracks_);
  const uint32_t default_module_index =
      trace_info.module_interner.Intern(kHloModuleDefault);
  std::vector<uint32_t> name_ref_to_op_index;
  std::vector<uint32_t> name_ref_to_module_index;

  TimeBounds bounds;
  absl::btree_map<int, int> flow_category_counts;
  absl::btree_map<std::pair<ProcessId, ThreadId>, ThreadLevelInfo>
      thread_levels;
  int current_level = 0;
  std::vector<TrackEvent> track_buffer;
  std::vector<PackedTrackRange> packed_track_ranges;
  google::protobuf::Arena track_arena;

  const auto base_max_observed_levels = unclosed_async_events.empty()
                                            ? decltype(max_observed_levels_){}
                                            : max_observed_levels_;

  const auto pack_all_tracks = [&](uint64_t* max_end_ps_out) {
    data = FlameChartTimelineData{};
    if (total_events > 0) {
      data.Reserve(total_events, /*reserve_legacy_names_and_ids=*/false);
    }
    name_ref_to_op_index.assign(interned_strings_count,
                                std::numeric_limits<uint32_t>::max());
    name_ref_to_module_index.assign(interned_strings_count,
                                    std::numeric_limits<uint32_t>::max());
    bounds = TimeBounds{};
    flow_category_counts.clear();
    thread_levels.clear();
    current_level = 0;
    packed_track_ranges.clear();

    for (const ProcessId pid : sorted_pids) {
      const auto it_pid_complete = complete_series_by_pid_tid.find(pid);
      const bool has_complete_events =
          (it_pid_complete != complete_series_by_pid_tid.end() &&
           !it_pid_complete->second.empty());
      const auto it_thread_names =
          trace_info.thread_names.lower_bound({pid, 0});
      const bool has_named_threads =
          (it_thread_names != trace_info.thread_names.end() &&
           it_thread_names->first.first == pid);
      const auto it_known_threads = known_threads_.lower_bound({pid, 0});
      const bool has_known_threads =
          (it_known_threads != known_threads_.end() &&
           it_known_threads->first == pid);
      const bool has_async_events = async_events_by_pid_name.contains(pid);
      const bool has_known_async = known_async_tracks_.contains(pid);
      const bool has_thread_tracks = has_complete_events || has_named_threads ||
                                     has_known_threads || has_async_events ||
                                     has_known_async;
      const bool has_known_counters =
          counters_by_pid_name.contains(pid) || known_counters_.contains(pid);
      if (!has_thread_tracks && !has_known_counters) {
        continue;
      }

      std::string process_group_name;
      if (const auto it = trace_info.process_names.find(pid);
          it != trace_info.process_names.end()) {
        process_group_name = it->second;
      } else {
        process_group_name = GetDefaultProcessName(pid);
      }

      const bool process_expanded =
          GetExpandedState(kProcessNestingLevel, process_group_name, "",
                           data.groups.empty(), expanded_states);

      std::string track_subtitle;
      const size_t separator_pos = process_group_name.find(' ');
      if (separator_pos != std::string::npos) {
        track_subtitle = process_group_name.substr(0, separator_pos);
      }

      const int process_start_level = current_level;
      const int process_index = static_cast<int>(data.groups.size());
      data.groups.push_back({.name = process_group_name,
                             .subtitle = std::move(track_subtitle),
                             .start_level = current_level,
                             .nesting_level = kProcessNestingLevel,
                             .expanded = process_expanded,
                             .parent_index = -1,
                             .pid = pid});

      std::vector<ModuleInterval> module_intervals;
      const auto it_xla_tid = trace_info.xla_modules_tids.find(pid);
      if (it_xla_tid != trace_info.xla_modules_tids.end() &&
          it_pid_complete != complete_series_by_pid_tid.end()) {
        if (const auto it_xla_series =
                it_pid_complete->second.find(it_xla_tid->second);
            it_xla_series != it_pid_complete->second.end()) {
          std::vector<ModuleInterval> raw_module_intervals;
          for (const xprof::TraceEventSeries* series :
               it_xla_series->second.proto_series) {
            AppendSeriesModuleIntervals(*series, get_interned_string,
                                        trace_info.module_interner,
                                        raw_module_intervals, max_end_ps_out);
          }
          for (const WireSeriesView* view : it_xla_series->second.wire_series) {
            xprof::TraceEventSeries* parsed_series =
                google::protobuf::Arena::Create<xprof::TraceEventSeries>(&track_arena);
            if (parsed_series->ParseFromArray(
                    view->series_wire.data(),
                    static_cast<int>(view->series_wire.size()))) {
              AppendSeriesModuleIntervals(*parsed_series, get_interned_string,
                                          trace_info.module_interner,
                                          raw_module_intervals, max_end_ps_out);
            }
            track_arena.Reset();
          }
          module_intervals =
              BuildDisjointModuleSegments(std::move(raw_module_intervals));
        }
      }

      const auto it_cnt = counters_by_pid_name.find(pid);
      const auto it_kc = known_counters_.find(pid);
      if (it_cnt != counters_by_pid_name.end() ||
          it_kc != known_counters_.end()) {
        absl::btree_set<std::string> all_counter_names;
        if (it_cnt != counters_by_pid_name.end()) {
          for (const auto& [name, _] : it_cnt->second) {
            all_counter_names.insert(name);
          }
        }
        if (it_kc != known_counters_.end()) {
          for (const std::string& name : it_kc->second) {
            all_counter_names.insert(name);
          }
        }
        for (const std::string& counter_name : all_counter_names) {
          absl::Span<const CounterEvent* const> cnt_events;
          if (it_cnt != counters_by_pid_name.end()) {
            if (const auto it_e = it_cnt->second.find(counter_name);
                it_e != it_cnt->second.end()) {
              cnt_events = it_e->second;
            }
          }
          PopulateCounterTrack<ProtoEventRef>(
              pid, counter_name, cnt_events, trace_info, current_level, data,
              bounds, process_group_name, /*default_expanded=*/true,
              expanded_states, process_index);
        }
      }

      if (has_thread_tracks) {
        absl::flat_hash_set<ThreadId> tids;
        if (has_complete_events) {
          for (const auto& [tid, _] : it_pid_complete->second) {
            tids.insert(tid);
          }
        }
        for (auto it = trace_info.thread_names.lower_bound({pid, 0});
             it != trace_info.thread_names.end() && it->first.first == pid;
             ++it) {
          if (it->first.second < 0x80000000) {
            tids.insert(it->first.second);
          }
        }
        for (auto it = known_threads_.lower_bound({pid, 0});
             it != known_threads_.end() && it->first == pid; ++it) {
          if (it->second < 0x80000000) {
            tids.insert(it->second);
          }
        }

        std::vector<ThreadId> sorted_tids(tids.begin(), tids.end());
        absl::c_stable_sort(sorted_tids, [&](ThreadId a, ThreadId b) {
          return CompareThreadsForSort(pid, trace_info, a, b);
        });

        const auto it_sort = trace_info.process_sort_indices.find(pid);
        const bool is_unranked_mpmd_process =
            !trace_info.process_sort_indices.empty() &&
            (it_sort == trace_info.process_sort_indices.end() ||
             it_sort->second >= kMpmdUnrankedSortIndexBase);

        for (const ThreadId tid : sorted_tids) {
          const auto it_series =
              has_complete_events
                  ? it_pid_complete->second.find(tid)
                  : absl::btree_map<
                        ThreadId, TrackCompleteSeriesSource>::const_iterator{};
          const bool has_events = has_complete_events &&
                                  (it_series != it_pid_complete->second.end());
          std::string thread_group_name;
          if (const auto it = trace_info.thread_names.find({pid, tid});
              it != trace_info.thread_names.end()) {
            thread_group_name = it->second;
          } else {
            thread_group_name = GetDefaultThreadName(tid);
          }

          if (trace_info.is_mpmd && !has_events) {
            const bool is_primary_mpmd_track =
                (thread_group_name == kXlaModules ||
                 (it_xla_tid != trace_info.xla_modules_tids.end() &&
                  it_xla_tid->second == tid));
            if (!is_primary_mpmd_track || is_unranked_mpmd_process) {
              continue;
            }
          }

          const bool thread_expanded = GetExpandedState(
              kThreadNestingLevel, thread_group_name, process_group_name,
              /*default_expanded=*/true, expanded_states);

          const int child_index = static_cast<int>(data.groups.size());
          data.groups.push_back({.type = Group::Type::kFlame,
                                 .name = thread_group_name,
                                 .start_level = current_level,
                                 .nesting_level = kThreadNestingLevel,
                                 .expanded = thread_expanded,
                                 .parent_index = process_index,
                                 .pid = pid,
                                 .tid = tid});
          data.groups[process_index].child_indices.push_back(child_index);

          const int start_level = current_level;
          int max_level = start_level;

          if (has_events) {
            const size_t entry_begin = data.entry_levels.size();
            const TrackCompleteSeriesSource& src = it_series->second;
            if (!src.proto_series.empty()) {
              UnpackAndPackProtoTrack(
                  pid, tid, thread_group_name, src.proto_series,
                  get_interned_string, interned_strings_count, module_intervals,
                  default_module_index, start_level, max_level, data, bounds,
                  trace_info, flow_category_counts, track_buffer,
                  name_ref_to_op_index, name_ref_to_module_index,
                  kDefaultProtoTrackChunkSize, max_end_ps_out);
            } else if (!src.wire_series.empty()) {
              UnpackAndPackProtoSeriesWire(
                  pid, tid, thread_group_name, src.wire_series,
                  get_interned_string, interned_strings_count, module_intervals,
                  default_module_index, start_level, max_level, data, bounds,
                  trace_info, flow_category_counts, track_buffer,
                  name_ref_to_op_index, name_ref_to_module_index,
                  kDefaultProtoTrackChunkSize, max_end_ps_out, &track_arena);
            }
            const size_t entry_end = data.entry_levels.size();
            if (entry_end > entry_begin) {
              packed_track_ranges.push_back({.entry_begin = entry_begin,
                                             .entry_end = entry_end,
                                             .start_level = start_level,
                                             .max_level = max_level});
            }
          }

          const GroupKey group_key{
              .nesting_level = kThreadNestingLevel,
              .name = thread_group_name,
              .parent_name = process_group_name,
          };
          const int current_levels =
              !has_events ? 1 : (max_level - start_level + 1);
          int& max_observed = max_observed_levels_[group_key];
          const int level_count = std::max(current_levels, max_observed);
          max_observed = level_count;
          data.groups.back().level_count = level_count;

          current_level = start_level + level_count;
          auto [it_lvl, inserted] = thread_levels.try_emplace(
              {pid, tid}, ThreadLevelInfo{start_level, current_level});
          if (!inserted) {
            it_lvl->second.start_level =
                std::min(it_lvl->second.start_level, start_level);
            it_lvl->second.end_level =
                std::max(it_lvl->second.end_level, current_level);
          }
        }

        absl::btree_set<std::string> all_async_names;
        if (const auto it_async = async_events_by_pid_name.find(pid);
            it_async != async_events_by_pid_name.end()) {
          for (const auto& [name, _] : it_async->second) {
            all_async_names.insert(name);
          }
        }
        if (const auto it_known = known_async_tracks_.find(pid);
            it_known != known_async_tracks_.end()) {
          for (const std::string& name : it_known->second) {
            all_async_names.insert(name);
          }
        }

        for (const std::string& async_name : all_async_names) {
          absl::Span<TrackEvent> async_events;
          ThreadId tid = 0;
          if (const auto it_async = async_events_by_pid_name.find(pid);
              it_async != async_events_by_pid_name.end()) {
            if (const auto it_ev = it_async->second.find(async_name);
                it_ev != it_async->second.end()) {
              async_events = absl::MakeSpan(it_ev->second.events);
              tid = it_ev->second.tid;
            }
          }

          const bool thread_expanded = GetExpandedState(
              kThreadNestingLevel, async_name, process_group_name,
              /*default_expanded=*/true, expanded_states);

          const int child_index = static_cast<int>(data.groups.size());
          data.groups.push_back({.type = Group::Type::kFlame,
                                 .name = async_name,
                                 .start_level = current_level,
                                 .nesting_level = kThreadNestingLevel,
                                 .expanded = thread_expanded,
                                 .parent_index = process_index,
                                 .pid = pid,
                                 .tid = tid});
          data.groups[process_index].child_indices.push_back(child_index);

          const int start_level = current_level;
          int max_level = start_level;

          const size_t entry_begin = data.entry_levels.size();
          PackProtoEvents(async_events, async_name, /*is_async_track=*/true,
                          module_intervals, default_module_index, start_level,
                          max_level, data, bounds, trace_info,
                          get_interned_string, name_ref_to_op_index,
                          name_ref_to_module_index);
          const size_t entry_end = data.entry_levels.size();
          if (entry_end > entry_begin) {
            packed_track_ranges.push_back({.entry_begin = entry_begin,
                                           .entry_end = entry_end,
                                           .start_level = start_level,
                                           .max_level = max_level});
          }

          const GroupKey group_key{
              .nesting_level = kThreadNestingLevel,
              .name = async_name,
              .parent_name = process_group_name,
          };
          const int current_levels =
              async_events.empty() ? 1 : (max_level - start_level + 1);
          int& max_observed = max_observed_levels_[group_key];
          const int level_count = std::max(current_levels, max_observed);
          max_observed = level_count;
          data.groups.back().level_count = level_count;

          current_level = start_level + level_count;
          auto [it_lvl, inserted] = thread_levels.try_emplace(
              {pid, tid}, ThreadLevelInfo{start_level, current_level});
          if (!inserted) {
            it_lvl->second.start_level =
                std::min(it_lvl->second.start_level, start_level);
            it_lvl->second.end_level =
                std::max(it_lvl->second.end_level, current_level);
          }
        }
        if (unclosed_async_events.empty()) {
          async_events_by_pid_name.erase(pid);
        }
      }

      if (trace_info.is_mpmd &&
          data.groups.size() == static_cast<size_t>(process_index + 1)) {
        data.groups.pop_back();
      } else {
        data.groups[process_index].level_count =
            current_level - process_start_level;
        data.groups[process_index].has_children =
            !data.groups[process_index].child_indices.empty();
      }
    }
  };

  uint64_t max_complete_end_ps = 0;
  uint64_t* const max_end_ps_ptr =
      unclosed_async_events.empty() ? nullptr : &max_complete_end_ps;
  pack_all_tracks(max_end_ps_ptr);
  if (!unclosed_async_events.empty() &&
      max_complete_end_ps / 1000000.0 > provisional_end_us) {
    max_observed_levels_ = base_max_observed_levels;
    trace_info.module_interner = TableInterner();
    const uint32_t default_mod_idx =
        trace_info.module_interner.Intern(kHloModuleDefault);
    (void)default_mod_idx;
    trace_info.op_interner = TableInterner();
    trace_info.flow_events_by_id.clear();
    populate_unclosed_async_events(max_complete_end_ps / 1000000.0);
    pack_all_tracks(/*max_end_ps_out=*/nullptr);
  }

  // Free transient track_buffer, wire views, and serialized input buffers
  // before allocating CSR level_event_indices, flow lines, and mip pyramids.
  std::vector<TrackEvent>().swap(track_buffer);
  async_events_by_pid_name.clear();
  complete_series_by_pid_tid.clear();
  std::vector<WireSeriesView>().swap(wire_complete_series);
  release_input_buffers();

  data.level_offsets.assign(current_level + 1, 0);
  for (const PackedTrackRange& r : packed_track_ranges) {
    if (r.start_level == r.max_level && r.start_level < current_level) {
      data.level_offsets[r.start_level + 1] += (r.entry_end - r.entry_begin);
    } else {
      for (size_t i = r.entry_begin; i < r.entry_end; ++i) {
        ++data.level_offsets[data.entry_levels[i] + 1];
      }
    }
  }
  std::partial_sum(data.level_offsets.begin(), data.level_offsets.end(),
                   data.level_offsets.begin());
  data.level_event_indices.resize(data.entry_levels.size());
  std::vector<size_t> cursors = data.level_offsets;
  for (const PackedTrackRange& r : packed_track_ranges) {
    if (r.start_level == r.max_level && r.start_level < current_level) {
      const size_t count = r.entry_end - r.entry_begin;
      auto dst = data.level_event_indices.begin() +
                 static_cast<ptrdiff_t>(cursors[r.start_level]);
      std::iota(dst, dst + static_cast<ptrdiff_t>(count),
                static_cast<int>(r.entry_begin));
      cursors[r.start_level] += count;
    } else {
      for (size_t i = r.entry_begin; i < r.entry_end; ++i) {
        const int level = data.entry_levels[i];
        data.level_event_indices[cursors[level]] = static_cast<int>(i);
        ++cursors[level];
      }
    }
  }

  present_flow_categories_.clear();
  for (const auto& [category, _] : flow_category_counts) {
    present_flow_categories_.push_back(category);
  }
  for (auto& [id, events] : trace_info.flow_events_by_id) {
    absl::c_stable_sort(events,
                        [](const FlowEndpoint& a, const FlowEndpoint& b) {
                          return std::tie(a.ts, a.pid, a.tid, a.event_index) <
                                 std::tie(b.ts, b.pid, b.tid, b.event_index);
                        });
  }
  GenerateFlowLines<ProtoEventRef>(trace_info, thread_levels,
                                   GetTop5FlowCategories(flow_category_counts),
                                   data, bounds, timeline.GetPalette());

  data.interned_string_pool = trace_info.name_interner.TakePool();
  data.hlo_module_table = trace_info.module_interner.TakePool();
  data.hlo_op_table = trace_info.op_interner.TakePool();

  process_names_ = trace_info.process_names;
  thread_names_ = trace_info.thread_names;
  process_sort_indices_ = trace_info.process_sort_indices;
  thread_sort_indices_ = trace_info.thread_sort_indices;
  xla_modules_tids_ = trace_info.xla_modules_tids;
  known_async_processes_ = trace_info.async_processes_by_events;

  FinalizeTimelineUpdate(timeline, std::move(data), bounds,
                         visible_range_from_url, full_timespan);
  return true;
}

const std::vector<int>& DataProvider::GetFlowCategories() const {
  return present_flow_categories_;
}

absl::flat_hash_map<ProcessId, std::string> DataProvider::GetProcessMappings()
    const {
  absl::flat_hash_map<ProcessId, std::string> map;
  for (const auto& [pid, process_name] : process_names_) {
    if (absl::EqualsIgnoreCase(process_name, kAsyncXlaOps) ||
        absl::EqualsIgnoreCase(process_name, kDataMotionLayersUtilization) ||
        absl::StartsWithIgnoreCase(process_name, "Async ")) {
      continue;
    }
    std::string host_part = process_name.substr(0, process_name.find(' '));
    if (!host_part.empty() && !absl::StartsWith(host_part, "/")) {
      map[pid] = std::move(host_part);
    }
  }
  return map;
}

namespace internal {
void AppendTraceEventForTesting(const TraceEvent* event, int level,
                                FlameChartTimelineData& data) {
  TimeBounds bounds;
  TraceInformation<TraceEvent> trace_info;
  size_t module_cursor = 0;
  Microseconds last_module_lookup_ts =
      std::numeric_limits<Microseconds>::lowest();
  AppendEventToTimelineData(event, level, data, bounds, trace_info,
                            /*thread_name=*/"", /*module_src=*/{},
                            module_cursor, last_module_lookup_ts);
}
}  // namespace internal

}  // namespace traceviewer
