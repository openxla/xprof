#include "frontend/app/components/trace_viewer_v2/timeline/data_provider.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <deque>
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
#include "re2/re2.h"
#include "tsl/profiler/lib/context_types.h"
#include "frontend/app/components/trace_viewer_v2/color/colors.h"
#include "frontend/app/components/trace_viewer_v2/helper/time_formatter.h"
#include "frontend/app/components/trace_viewer_v2/timeline/constants.h"
#include "frontend/app/components/trace_viewer_v2/timeline/time_range.h"
#include "frontend/app/components/trace_viewer_v2/timeline/timeline.h"
#include "frontend/app/components/trace_viewer_v2/trace_helper/trace_event.h"
#include "frontend/app/components/trace_viewer_v2/trace_helper/trace_event_packer.h"
#include "frontend/app/components/trace_viewer_v2/trace_helper/trace_event_tree.h"

namespace traceviewer {

namespace {

// Base offset for process sort indices of devices not ranked into an MPMD
// stage. Offsets unranked devices past stage ranks with headroom for PID
// tie-breaking.
constexpr uint32_t kMpmdUnrankedSortIndexBase = 1u << 30;

constexpr ThreadId kSyntheticOpenSpanTidBit = 0x80000000;
constexpr size_t kDefaultMaxTrackEvents = 262144;
constexpr size_t kDefaultTargetTrackEvents = 131072;
constexpr size_t kDefaultTailKeepEvents = 2048;
constexpr size_t kMinLevelCompactionThreshold = 2048;
constexpr size_t kMinLevelBudget = 512;
constexpr size_t kMaxGlobalCachedEvents = 600000;
constexpr size_t kTargetGlobalCachedEvents = 350000;
constexpr size_t kMinTrackEventsForGlobalCompaction = 1024;
constexpr size_t kGlobalTailKeepEvents = 512;
constexpr size_t kMaxIncrementalFlowEvents = 16384;

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

struct TimeBounds {
  Microseconds min = std::numeric_limits<Microseconds>::max();
  Microseconds max = std::numeric_limits<Microseconds>::min();
};

struct CachedThreadTrack {
  // Columnar event storage where `entries.entry_levels` stores track-relative
  // levels (0..max_relative_level) and `entries.entry_names` stays empty while
  // names are interned via `unique_names` + `entry_name_ids`.
  FlameChartTimelineData entries;
  int max_relative_level = 0;
  TimeBounds bounds;
  ProcessId pid = 0;
  ThreadId tid = 0;
  std::vector<std::string> unique_names;
  absl::flat_hash_map<std::string, uint32_t> name_to_id;
  std::vector<uint32_t> entry_name_ids;
  // `entry_tids` remains empty when (in most cases) the track only has one
  // tid, in which case `tid` is used instead. `entry_tids` only materializes
  // (matching the size of events) when a different `tid` is encountered.
  std::vector<ThreadId> entry_tids;
  std::vector<Microseconds> prefix_max_end_times;

  struct OpenSpan {
    size_t entry_idx = 0;
    uint32_t name_id = 0;
    Microseconds start_ts = 0.0;
    Microseconds current_end_ts = 0.0;
    int relative_level = 0;
  };
  std::vector<OpenSpan> open_spans;
  bool was_compacted = false;

  size_t size() const { return entries.entry_start_times.size(); }
  bool empty() const { return entries.entry_start_times.empty(); }
  ThreadId tid_at(size_t i) const {
    return entry_tids.empty() ? tid : entry_tids[i];
  }
};

struct TraceInformation {
  // The TraceEvent objects pointed to must outlive this TraceInformation
  // instance.
  absl::btree_map<ProcessId,
                  absl::btree_map<ThreadId, std::vector<const TraceEvent*>>>
      events_by_pid_tid;
  absl::btree_map<
      ProcessId, absl::btree_map<std::string, std::vector<const CounterEvent*>>>
      counters_by_pid_name;
  absl::btree_map<std::pair<ProcessId, ThreadId>, std::string> thread_names;
  absl::flat_hash_map<std::pair<ProcessId, ThreadId>, uint32_t>
      thread_sort_indices;
  absl::flat_hash_map<ProcessId, std::string> process_names;
  absl::flat_hash_map<ProcessId, uint32_t> process_sort_indices;
  absl::flat_hash_map<ProcessId, uint32_t> incoming_process_sort_indices;
  absl::btree_map<std::string, std::vector<const TraceEvent*>>
      flow_events_by_id;
  absl::flat_hash_map<ProcessId, ThreadId> xla_modules_tids;
  absl::flat_hash_map<ProcessId, ThreadId> xla_ops_tids;
  absl::flat_hash_set<ProcessId> async_processes_by_events;
  bool is_mpmd = false;
  bool consume_cached_tracks = false;
  absl::flat_hash_map<std::pair<ProcessId, ThreadId>, CachedThreadTrack>*
      cached_sync_tracks = nullptr;
  absl::flat_hash_map<std::pair<ProcessId, std::string>, CachedThreadTrack>*
      cached_async_tracks = nullptr;
};

int GetAsyncProcessPriority(ProcessId pid, const TraceInformation& trace_info) {
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

bool CompareThreadsForSort(ProcessId pid, const TraceInformation& trace_info,
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
                         TraceInformation& trace_info) {
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
void MergeProcessSortIndices(TraceInformation& trace_info) {
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
                         TraceInformation& trace_info) {
  trace_info.events_by_pid_tid[event.pid][event.tid].push_back(&event);
  if (event.is_async) {
    trace_info.async_processes_by_events.insert(event.pid);
  }
}

void HandleFlowEvent(const TraceEvent& event, TraceInformation& trace_info,
                     absl::btree_map<int, int>& category_counts) {
  if (!event.id.empty()) {
    trace_info.flow_events_by_id[event.id].push_back(&event);
    category_counts[static_cast<int>(event.category)]++;
  }
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
                        TraceInformation& trace_info) {
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

// Resolves track bounds for a flow event endpoint, matching only flame tracks
// (and never counter tracks or process headers).
std::optional<ThreadLevelInfo> ResolveFlowEndpointLevels(
    const TraceEvent* e,
    const absl::btree_map<std::pair<ProcessId, ThreadId>, ThreadLevelInfo>&
        thread_levels,
    const absl::flat_hash_map<std::pair<ProcessId, absl::string_view>,
                              ThreadLevelInfo>& async_track_levels) {
  if (e->is_async) {
    if (!e->name.empty()) {
      if (auto it = async_track_levels.find({e->pid, e->name});
          it != async_track_levels.end()) {
        return it->second;
      }
    }
    return std::nullopt;
  }
  if (auto it = thread_levels.find({e->pid, e->tid});
      it != thread_levels.end()) {
    return it->second;
  }
  return std::nullopt;
}

// Returns the flame chart level of the given event within the resolved track
// levels.
int GetEventFlameChartLevel(const TraceEvent* e,
                            const ThreadLevelInfo& track_levels,
                            const FlameChartTimelineData& data) {
  const int start = track_levels.start_level;
  const int end = track_levels.end_level;
  std::optional<int> name_match_level;
  std::optional<int> candidate_level;

  // Search from deepest level up.
  for (int lvl = end - 1; lvl >= start; --lvl) {
    absl::Span<const int> indices = data.level_events(lvl);
    if (indices.empty()) continue;

    auto it_idx = std::upper_bound(indices.begin(), indices.end(), e->ts,
                                   [&](Microseconds ts, int idx) {
                                     return ts < data.entry_start_times[idx];
                                   });

    if (it_idx != indices.begin()) {
      int idx = *std::prev(it_idx);
      const Microseconds start_time = data.entry_start_times[idx];
      const Microseconds end_time = start_time + data.entry_total_times[idx];
      if (start_time <= e->ts && end_time >= e->ts) {
        if (e->event_id != 0 && data.entry_event_ids[idx] == e->event_id) {
          return lvl;
        }
        if (!e->name.empty() && data.entry_names[idx] == e->name) {
          if (!name_match_level.has_value()) {
            name_match_level = lvl;
          }
        }
        if (!candidate_level.has_value()) {
          candidate_level = lvl;
        }
      }
    }
  }

  if (name_match_level.has_value()) {
    return *name_match_level;
  }
  if (candidate_level.has_value()) {
    return *candidate_level;
  }
  return start;  // Default to flame track top
}

void GenerateFlowLines(const TraceInformation& trace_info,
                       const absl::btree_map<std::pair<ProcessId, ThreadId>,
                                             ThreadLevelInfo>& thread_levels,
                       absl::Span<const int> top_5_flow_categories,
                       FlameChartTimelineData& data, TimeBounds& bounds,
                       const ColorPalette& palette) {
  if (trace_info.flow_events_by_id.empty()) {
    return;
  }

  absl::flat_hash_map<std::pair<ProcessId, absl::string_view>, ThreadLevelInfo>
      async_track_levels;
  for (const auto& grp : data.groups) {
    if (grp.nesting_level != kProcessNestingLevel &&
        grp.type == Group::Type::kFlame) {
      async_track_levels[{grp.pid, grp.name}] =
          ThreadLevelInfo{grp.start_level, grp.start_level + grp.level_count};
    }
  }

  for (const auto& [id, flow_events] : trace_info.flow_events_by_id) {
    for (const TraceEvent* event : flow_events) {
      std::vector<std::string>& event_flow_ids =
          data.flow_ids_by_event_id[event->event_id];
      if (event_flow_ids.empty() || event_flow_ids.back() != id) {
        event_flow_ids.push_back(id);
      }
    }

    for (size_t i = 0; i < flow_events.size() - 1; ++i) {
      const TraceEvent* u = flow_events[i];
      const TraceEvent* v = flow_events[i + 1];
      std::optional<ThreadLevelInfo> u_levels =
          ResolveFlowEndpointLevels(u, thread_levels, async_track_levels);
      std::optional<ThreadLevelInfo> v_levels =
          ResolveFlowEndpointLevels(v, thread_levels, async_track_levels);
      if (!u_levels.has_value() || !v_levels.has_value()) {
        continue;
      }
      const ImU32 flow_color =
          GetFlowColorForCategory(u->category, top_5_flow_categories,
                                  palette);  // Use flow category for color
      FlowLine flow_line{
          .source_ts = u->ts,
          .target_ts = v->ts,
          .source_level = GetEventFlameChartLevel(u, *u_levels, data),
          .target_level = GetEventFlameChartLevel(v, *v_levels, data),
          .color = flow_color,
          .category = u->category};
      data.flow_lines.push_back(flow_line);
      data.flow_lines_by_flow_id[id].push_back(flow_line);
      bounds.min = std::min(bounds.min, u->ts);
      bounds.max = std::max(bounds.max, u->ts);
      bounds.min = std::min(bounds.min, v->ts);
      bounds.max = std::max(bounds.max, v->ts);
    }
  }
}

uint32_t InternHloModuleName(std::vector<std::string>& hlo_module_names,
                             absl::string_view module_name) {
  if (module_name.empty() || module_name == kHloModuleDefault) {
    return 0;
  }
  for (size_t i = hlo_module_names.size(); i > 0; --i) {
    if (hlo_module_names[i - 1] == module_name) {
      return static_cast<uint32_t>(i);
    }
  }
  hlo_module_names.emplace_back(module_name);
  return static_cast<uint32_t>(hlo_module_names.size());
}

struct TrackSpanSource {
  absl::Span<const TraceEvent* const> events;
  const CachedThreadTrack* cached_track = nullptr;
};

TrackSpanSource ResolveTrackSpanSource(
    ProcessId pid, const absl::flat_hash_map<ProcessId, ThreadId>& tid_by_pid,
    const TraceInformation& trace_info) {
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
  if (src.events.empty() && trace_info.cached_sync_tracks != nullptr) {
    if (const auto it_cached =
            trace_info.cached_sync_tracks->find({pid, it_tid->second});
        it_cached != trace_info.cached_sync_tracks->end()) {
      src.cached_track = &it_cached->second;
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
  if (src.cached_track != nullptr) {
    const CachedThreadTrack& ct = *src.cached_track;
    return ScanSpansAtTimestamp(
        ct.size(), [&](size_t k) { return ct.entries.entry_start_times[k]; },
        [&](size_t k) { return ct.entries.entry_total_times[k]; },
        [&](size_t k) -> absl::string_view {
          return ct.unique_names[ct.entry_name_ids[k]];
        },
        cursor, last_ts, ts, default_name);
  }
  return default_name;
}

// Formats `hlo_module_name` by appending `(<id>)` if `hlo_module_name` does not
// already end with that suffix.
std::string FormatHloModuleWithId(absl::string_view hlo_module_name,
                                  absl::string_view id) {
  const std::string suffix = absl::StrCat("(", id, ")");
  if (absl::EndsWith(hlo_module_name, suffix)) {
    return std::string(hlo_module_name);
  }
  return absl::StrCat(hlo_module_name, suffix);
}

void AppendEventToTimelineData(
    const TraceEvent* event, int level, FlameChartTimelineData& data,
    TimeBounds& bounds, const TraceInformation& trace_info,
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
    LOG_FIRST_N(WARNING, 1) << "Dropping event at level " << level
                            << ": exceeds uint16_t flame chart level range";
    return;
  }
  const size_t entry_idx = data.entry_start_times.size();
  data.entry_start_times.push_back(event->ts);
  data.entry_total_times.push_back(event->dur);
  data.entry_self_times.push_back(self_time.value_or(event->dur));
  data.entry_levels.push_back(static_cast<uint16_t>(level));
  data.entry_names.push_back(event->name);
  data.entry_event_ids.push_back(event->event_id);

  const bool is_xla_ops_thread = thread_name == kXlaOps;
  const bool is_source_thread = thread_name == "Source code";
  const bool is_data_motion_layer = thread_name == kComputeUtilization ||
                                    thread_name == kDataMotionLayersUtilization;

  if (event->args.empty() && !is_source_thread && !is_data_motion_layer) {
    data.entry_serials.push_back(event->serial);
    uint8_t flags = 0;
    if (event->has_serial) {
      flags |= FlameChartTimelineData::kEntryArgHasUid;
    }
    if (event->has_group_id) {
      flags |= FlameChartTimelineData::kEntryArgHasGroupId;
      data.sparse_entry_group_ids[entry_idx] = event->group_id;
    }
    if (is_xla_ops_thread) {
      flags |= FlameChartTimelineData::kEntryArgHloOpFromName;
    }
    data.entry_arg_flags.push_back(flags);

    absl::string_view hlo_module_str = *kHloModuleDefaultStr;
    if (is_xla_ops_thread) {
      hlo_module_str = LookupTrackSpanAtTimestamp(
          module_src, module_cursor, last_module_lookup_ts, event->ts);
    }
    data.entry_hlo_module_ids.push_back(
        InternHloModuleName(data.hlo_module_names, hlo_module_str));

    bounds.min = std::min(bounds.min, event->ts);
    bounds.max = std::max(bounds.max, event->ts + event->dur);
    return;
  }

  uint8_t raw_flags = 0;
  if (event->has_serial) raw_flags |= FlameChartTimelineData::kEntryArgHasUid;
  if (event->args.contains(*kHloOpStr)) {
    raw_flags |= FlameChartTimelineData::kEntryArgHadRawHloOp;
  }
  if (event->args.contains(*kHloModuleStr)) {
    raw_flags |= FlameChartTimelineData::kEntryArgHadRawHloModule;
  }
  if (event->has_group_id) {
    data.sparse_entry_group_ids[entry_idx] = event->group_id;
  }
  data.entry_serials.push_back(event->serial);
  data.entry_arg_flags.push_back(raw_flags);
  data.entry_hlo_module_ids.push_back(0);

  auto cur_args = event->args;
  if (event->has_serial) {
    cur_args.try_emplace("uid", absl::StrCat(event->serial));
  }
  if (event->has_group_id) {
    cur_args.try_emplace("group_id", absl::StrCat(event->group_id));
  }
  const bool has_hlo_in_args = event->args.count(*kHloOpStr) > 0 &&
                               event->args.count(*kHloModuleStr) > 0;
  if (is_xla_ops_thread || is_source_thread || is_data_motion_layer ||
      has_hlo_in_args) {
    if (is_data_motion_layer) {
      auto it_name = event->args.find("Name");
      if (it_name != event->args.end()) {
        cur_args[*kHloOpStr] = it_name->second;
      }
    } else if (is_source_thread && !event->args.count(*kHloOpStr)) {
      size_t op_cursor = 0;
      Microseconds last_op_ts = std::numeric_limits<Microseconds>::lowest();
      const absl::string_view op_name = LookupTrackSpanAtTimestamp(
          ResolveTrackSpanSource(event->pid, trace_info.xla_ops_tids,
                                 trace_info),
          op_cursor, last_op_ts, event->ts, /*default_name=*/"");
      if (!op_name.empty()) {
        cur_args[*kHloOpStr] = std::string(op_name);
      }
    } else if (!event->args.count(*kHloOpStr)) {
      cur_args[*kHloOpStr] = event->name;
    }
    std::string hlo_module_str = *kHloModuleDefaultStr;
    auto it_hlo_module = event->args.find(*kHloModuleStr);
    if (it_hlo_module != event->args.end()) {
      const std::string& hlo_module_name = it_hlo_module->second;
      auto it_hlo_module_id = event->args.find(*kHloModuleIdStr);
      auto it_program_id = event->args.find(*kProgramIdStr);
      if (it_hlo_module_id != event->args.end()) {
        hlo_module_str =
            FormatHloModuleWithId(hlo_module_name, it_hlo_module_id->second);
      } else if (it_program_id != event->args.end()) {
        hlo_module_str =
            FormatHloModuleWithId(hlo_module_name, it_program_id->second);
      } else {
        auto it_kernel_details = event->args.find(*kKernelDetailsStr);
        if (it_kernel_details != event->args.end()) {
          std::string module_id;
          if (RE2::PartialMatch(it_kernel_details->second, *kModuleRe,
                                &module_id)) {
            hlo_module_str = FormatHloModuleWithId(hlo_module_name, module_id);
          } else {
            hlo_module_str = hlo_module_name;
          }
        } else {
          hlo_module_str = hlo_module_name;
        }
      }
    } else {
      hlo_module_str = std::string(LookupTrackSpanAtTimestamp(
          module_src, module_cursor, last_module_lookup_ts, event->ts));
    }
    cur_args[*kHloModuleStr] = std::move(hlo_module_str);
  } else {
    cur_args[*kHloModuleStr] = *kHloModuleDefaultStr;
  }
  data.sparse_entry_args[entry_idx] = std::move(cur_args);

  bounds.min = std::min(bounds.min, event->ts);
  bounds.max = std::max(bounds.max, event->ts + event->dur);
}

struct RestoredEntryMeta {
  Microseconds self_time = 0.0;
  int relative_level = 0;
};
using RestoredMetaMap =
    absl::flat_hash_map<const TraceEvent*, RestoredEntryMeta>;

// Packs trace events for a thread track and appends them to data (inout).
// Updates max_level (inout) if any packed event exceeds it, and expands
// bounds (inout) to include the start and end timestamps of all appended
// events.
void PopulateThreadTrackEvents(
    absl::Span<const TraceEvent* const> events, int start_level, int& max_level,
    FlameChartTimelineData& data, TimeBounds& bounds,
    const TraceInformation& trace_info, absl::string_view thread_group_name,
    std::vector<CachedThreadTrack::OpenSpan>* open_spans_out = nullptr,
    absl::Span<const int> proxy_fixed_levels = {},
    std::vector<Microseconds>* proxy_self_times_out = nullptr,
    ProcessId* emitted_pid_out = nullptr,
    std::vector<ThreadId>* emitted_tids_out = nullptr,
    const RestoredMetaMap* restored_meta = nullptr) {
  if (events.empty()) return;

  const size_t num_proxies = proxy_fixed_levels.size();
  auto cmp_events = [](const TraceEvent* a, const TraceEvent* b) {
    if (a->ts != b->ts) return a->ts < b->ts;
    if (a->dur != b->dur) return a->dur > b->dur;
    return a->ph != b->ph && a->ph == Phase::kDurationEnd;
  };

  std::vector<const TraceEvent*> sorted_storage;
  absl::Span<const TraceEvent* const> sorted_events = events;
  if (num_proxies == 0) {
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
  }

  const size_t n = sorted_events.size();
  if (n == 0) return;

  std::vector<Microseconds> self_times(n);
  std::vector<int> relative_levels(n);
  std::vector<size_t> parent_stack;
  parent_stack.reserve(16);
  std::vector<Microseconds> row_end_times;
  row_end_times.reserve(16);
  for (int fixed_lvl : proxy_fixed_levels) {
    if (fixed_lvl >= static_cast<int>(row_end_times.size())) {
      row_end_times.resize(fixed_lvl + 1,
                           std::numeric_limits<Microseconds>::lowest());
    }
  }

  const bool has_restored_meta =
      restored_meta != nullptr && !restored_meta->empty();
  std::vector<const RestoredEntryMeta*> entry_meta;
  std::vector<ptrdiff_t> nearest_restored_anc;
  if (has_restored_meta) {
    entry_meta.assign(n, nullptr);
    nearest_restored_anc.assign(n, -1);
    for (size_t i = 0; i < n; ++i) {
      if (const auto it = restored_meta->find(sorted_events[i]);
          it != restored_meta->end()) {
        entry_meta[i] = &it->second;
      }
    }
  }

  for (size_t i = 0; i < n; ++i) {
    const TraceEvent* ev = sorted_events[i];
    const Microseconds end_time = ev->ts + ev->dur;
    const RestoredEntryMeta* meta = has_restored_meta ? entry_meta[i] : nullptr;
    self_times[i] = meta != nullptr ? meta->self_time : ev->dur;

    std::optional<size_t> enclosing_idx;
    while (!parent_stack.empty()) {
      const size_t p_idx = parent_stack.back();
      const Microseconds p_end =
          sorted_events[p_idx]->ts + sorted_events[p_idx]->dur;
      if (ev->ts < p_end) {
        if (end_time <= p_end) {
          enclosing_idx = p_idx;
          const RestoredEntryMeta* p_meta =
              has_restored_meta ? entry_meta[p_idx] : nullptr;
          if (p_meta == nullptr) {
            self_times[p_idx] = std::max(0.0, self_times[p_idx] - ev->dur);
            if (has_restored_meta && meta == nullptr) {
              nearest_restored_anc[i] = nearest_restored_anc[p_idx];
            }
          } else if (meta == nullptr) {
            nearest_restored_anc[i] = static_cast<ptrdiff_t>(p_idx);
          }
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
    if (i < num_proxies) {
      assigned_row = proxy_fixed_levels[i];
      if (assigned_row >= static_cast<int>(row_end_times.size())) {
        row_end_times.resize(assigned_row + 1,
                             std::numeric_limits<Microseconds>::lowest());
      }
      row_end_times[assigned_row] =
          std::max(row_end_times[assigned_row], end_time);
    } else {
      int min_row = enclosing_idx.has_value()
                        ? (relative_levels[*enclosing_idx] + 1)
                        : 0;
      if (meta != nullptr) {
        int parent_shift = 0;
        if (enclosing_idx.has_value() &&
            entry_meta[*enclosing_idx] != nullptr) {
          parent_shift =
              std::max(0, relative_levels[*enclosing_idx] -
                              entry_meta[*enclosing_idx]->relative_level);
        }
        min_row = std::max(min_row, meta->relative_level + parent_shift);
      }
      if (min_row > static_cast<int>(row_end_times.size())) {
        row_end_times.resize(min_row,
                             std::numeric_limits<Microseconds>::lowest());
      }
      for (size_t r = static_cast<size_t>(min_row); r < row_end_times.size();
           ++r) {
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
    }
    relative_levels[i] = assigned_row;
  }

  if (has_restored_meta) {
    for (size_t i = 0; i < n; ++i) {
      if (entry_meta[i] == nullptr && nearest_restored_anc[i] >= 0) {
        const size_t anc = static_cast<size_t>(nearest_restored_anc[i]);
        self_times[anc] = std::max(0.0, self_times[anc] - self_times[i]);
      }
    }
  }

  if (proxy_self_times_out != nullptr) {
    proxy_self_times_out->assign(self_times.begin(),
                                 self_times.begin() + num_proxies);
  }
  if (n <= num_proxies) return;

  const ProcessId pid = sorted_events[num_proxies]->pid;
  if (emitted_pid_out != nullptr) *emitted_pid_out = pid;
  const TrackSpanSource module_src =
      ResolveTrackSpanSource(pid, trace_info.xla_modules_tids, trace_info);

  const size_t emit_count = n - num_proxies;
  const size_t new_total = data.entry_start_times.size() + emit_count;
  data.entry_start_times.reserve(new_total);
  data.entry_total_times.reserve(new_total);
  data.entry_self_times.reserve(new_total);
  data.entry_levels.reserve(new_total);
  data.entry_names.reserve(data.entry_names.size() + emit_count);
  data.entry_event_ids.reserve(new_total);
  data.entry_serials.reserve(new_total);
  data.entry_arg_flags.reserve(new_total);
  data.entry_hlo_module_ids.reserve(new_total);
  if (emitted_tids_out != nullptr) {
    emitted_tids_out->reserve(emitted_tids_out->size() + emit_count);
  }

  size_t module_cursor = 0;
  Microseconds last_module_lookup_ts =
      std::numeric_limits<Microseconds>::lowest();
  for (size_t i = num_proxies; i < n; ++i) {
    const int absolute_level = start_level + relative_levels[i];
    max_level = std::max(max_level, absolute_level);
    const size_t prev_size = data.entry_start_times.size();
    AppendEventToTimelineData(sorted_events[i], absolute_level, data, bounds,
                              trace_info, thread_group_name, module_src,
                              module_cursor, last_module_lookup_ts,
                              self_times[i]);
    if (data.entry_start_times.size() > prev_size) {
      if (emitted_tids_out != nullptr) {
        emitted_tids_out->push_back(sorted_events[i]->tid);
      }
      if (open_spans_out != nullptr &&
          sorted_events[i]->ph == Phase::kDurationBegin) {
        open_spans_out->push_back(CachedThreadTrack::OpenSpan{
            .entry_idx = prev_size,
            .start_ts = sorted_events[i]->ts,
            .current_end_ts = sorted_events[i]->ts + sorted_events[i]->dur,
            .relative_level = relative_levels[i],
        });
      }
    }
  }
}

void AppendToCachedThreadTrack(
    CachedThreadTrack& cached, absl::Span<const TraceEvent* const> events,
    const TraceInformation& trace_info, absl::string_view thread_group_name,
    absl::Span<const int> proxy_fixed_levels = {},
    std::vector<Microseconds>* proxy_self_times_out = nullptr,
    const RestoredMetaMap* restored_meta = nullptr) {
  if (events.empty()) return;

  const size_t base_idx = cached.size();
  int max_relative_level = cached.max_relative_level;
  const size_t open_spans_base = cached.open_spans.size();
  ProcessId emitted_pid = 0;
  std::vector<ThreadId> emitted_tids;
  PopulateThreadTrackEvents(events, /*start_level=*/0, max_relative_level,
                            cached.entries, cached.bounds, trace_info,
                            thread_group_name, &cached.open_spans,
                            proxy_fixed_levels, proxy_self_times_out,
                            &emitted_pid, &emitted_tids, restored_meta);
  const size_t new_size = cached.size();
  if (new_size == base_idx) return;

  cached.max_relative_level = max_relative_level;
  if (base_idx == 0) {
    cached.pid = emitted_pid;
    cached.tid = emitted_tids.front();
  }
  bool has_per_entry_tids = !cached.entry_tids.empty();
  if (!has_per_entry_tids) {
    for (const ThreadId t : emitted_tids) {
      if (t != cached.tid) {
        cached.entry_tids.reserve(new_size);
        // Materializes `entry_tids` for all previous entries that used `tid`.
        cached.entry_tids.assign(base_idx, cached.tid);
        has_per_entry_tids = true;
        break;
      }
    }
  }
  if (has_per_entry_tids) {
    cached.entry_tids.insert(cached.entry_tids.end(), emitted_tids.begin(),
                             emitted_tids.end());
  }

  cached.entry_name_ids.reserve(new_size);
  for (std::string& name : cached.entries.entry_names) {
    auto [it, inserted] = cached.name_to_id.try_emplace(
        name, static_cast<uint32_t>(cached.unique_names.size()));
    if (inserted) {
      cached.unique_names.push_back(std::move(name));
    }
    cached.entry_name_ids.push_back(it->second);
  }
  cached.entries.entry_names.clear();

  for (size_t s = open_spans_base; s < cached.open_spans.size(); ++s) {
    cached.open_spans[s].name_id =
        cached.entry_name_ids[cached.open_spans[s].entry_idx];
  }

  cached.prefix_max_end_times.reserve(new_size);
  Microseconds running_max = cached.prefix_max_end_times.empty()
                                 ? std::numeric_limits<Microseconds>::lowest()
                                 : cached.prefix_max_end_times.back();
  for (size_t i = base_idx; i < new_size; ++i) {
    running_max =
        std::max(running_max, cached.entries.entry_start_times[i] +
                                  cached.entries.entry_total_times[i]);
    cached.prefix_max_end_times.push_back(running_max);
  }
}

template <typename T>
void ResizeVector(std::vector<T>& vec, size_t new_size, bool shrink) {
  vec.resize(new_size);
  if (shrink && vec.capacity() > new_size * 2) {
    vec.shrink_to_fit();
  }
}

void ResizeCachedTrackArrays(CachedThreadTrack& cached, size_t new_size,
                             bool shrink) {
  FlameChartTimelineData& e = cached.entries;
  ResizeVector(e.entry_start_times, new_size, shrink);
  ResizeVector(e.entry_total_times, new_size, shrink);
  ResizeVector(e.entry_self_times, new_size, shrink);
  ResizeVector(e.entry_levels, new_size, shrink);
  ResizeVector(e.entry_event_ids, new_size, shrink);
  ResizeVector(e.entry_serials, new_size, shrink);
  ResizeVector(e.entry_arg_flags, new_size, shrink);
  ResizeVector(e.entry_hlo_module_ids, new_size, shrink);
  ResizeVector(cached.entry_name_ids, new_size, shrink);
  if (!cached.entry_tids.empty()) {
    ResizeVector(cached.entry_tids, new_size, shrink);
  }
  ResizeVector(cached.prefix_max_end_times, new_size, shrink);
}

// Returns a comparator that orders entry indices in `entries` by start time
// ascending, breaking ties by total duration descending.
auto CompareEntriesByStartAscThenDurDesc(
    const FlameChartTimelineData& entries) {
  return [&entries](int idx_a, int idx_b) {
    return entries.entry_start_times[idx_a] != entries.entry_start_times[idx_b]
               ? entries.entry_start_times[idx_a] <
                     entries.entry_start_times[idx_b]
               : entries.entry_total_times[idx_a] >
                     entries.entry_total_times[idx_b];
  };
}

// Compacts `cached` in-place when its event count exceeds `max_cached_events`,
// reducing the track toward `target_compacted_events` while always preserving
// active `open_spans`, pinned events, the most recent `tail_keep_count` events
// (and any ancestor active at the last start time), and the longest-duration
// span in each time bucket per level. Returns true if any entries were dropped.
bool CompactCachedThreadTrackIfNeeded(
    CachedThreadTrack& cached,
    size_t max_cached_events = kDefaultMaxTrackEvents,
    size_t target_compacted_events = kDefaultTargetTrackEvents,
    size_t tail_keep_count = kDefaultTailKeepEvents,
    const absl::flat_hash_set<EventId>* pinned_event_ids = nullptr) {
  const size_t n = cached.size();
  if (n <= max_cached_events) return false;

  FlameChartTimelineData& entries = cached.entries;
  std::vector<uint8_t> keep(n, 0);
  // Always preserve open spans, pinned events, ancestors active at the last
  // event's start time, and the recent tail at full resolution.
  for (const auto& open_span : cached.open_spans) {
    if (open_span.entry_idx < n) keep[open_span.entry_idx] = 1;
  }
  if (pinned_event_ids != nullptr && !pinned_event_ids->empty()) {
    for (size_t i = 0; i < n; ++i) {
      if (pinned_event_ids->contains(entries.entry_event_ids[i])) keep[i] = 1;
    }
  }
  const Microseconds last_start_ts = entries.entry_start_times.back();
  const size_t tail_start = n > tail_keep_count ? (n - tail_keep_count) : 0;
  for (size_t i = 0; i < n; ++i) {
    if (i >= tail_start ||
        entries.entry_start_times[i] + entries.entry_total_times[i] >
            last_start_ts) {
      keep[i] = 1;
    }
  }

  size_t already_kept = 0;
  std::vector<std::vector<int>> events_by_level(cached.max_relative_level + 1);
  for (size_t i = 0; i < n; ++i) {
    if (keep[i]) {
      ++already_kept;
      continue;
    }
    events_by_level[entries.entry_levels[i]].push_back(static_cast<int>(i));
  }
  const size_t unkept_count = n - already_kept;
  if (unkept_count == 0) return false;
  const auto cmp_idx = CompareEntriesByStartAscThenDurDesc(entries);

  const size_t head_budget = target_compacted_events > already_kept
                                 ? (target_compacted_events - already_kept)
                                 : 0;
  const size_t non_empty_levels = std::max<size_t>(
      1, static_cast<size_t>(absl::c_count_if(
             events_by_level,
             [](const std::vector<int>& v) { return !v.empty(); })));
  const size_t per_level_cap = head_budget / non_empty_levels;
  const size_t min_level_threshold =
      std::min(kMinLevelCompactionThreshold, per_level_cap / 2);
  const size_t min_level_budget =
      std::min(kMinLevelBudget, std::max<size_t>(1, per_level_cap / 2));
  for (std::vector<int>& level_vec : events_by_level) {
    if (level_vec.empty()) continue;
    if (!absl::c_is_sorted(level_vec, cmp_idx)) {
      absl::c_stable_sort(level_vec, cmp_idx);
    }
    const size_t level_budget =
        level_vec.size() <= min_level_threshold
            ? level_vec.size()
            : std::max<size_t>(
                  min_level_budget,
                  static_cast<size_t>(static_cast<uint64_t>(head_budget) *
                                      level_vec.size() / unkept_count));
    if (level_vec.size() <= level_budget) {
      for (const int idx : level_vec) keep[idx] = 1;
      continue;
    }
    const Microseconds min_ts = entries.entry_start_times[level_vec.front()];
    const Microseconds max_start_ts =
        entries.entry_start_times[level_vec.back()];
    const Microseconds bucket_width =
        max_start_ts > min_ts
            ? (max_start_ts - min_ts) / static_cast<double>(level_budget)
            : 0.0;
    size_t current_bucket = std::numeric_limits<size_t>::max();
    int best_idx = -1;
    Microseconds best_dur = -1.0;
    for (const int idx : level_vec) {
      const Microseconds ts = entries.entry_start_times[idx];
      const Microseconds dur = entries.entry_total_times[idx];
      const size_t bucket =
          bucket_width > 0.0
              ? std::min<size_t>(level_budget - 1,
                                 static_cast<size_t>(std::max(
                                     0.0, (ts - min_ts) / bucket_width)))
              : 0;
      if (bucket != current_bucket) {
        if (best_idx >= 0) keep[best_idx] = 1;
        current_bucket = bucket;
        best_idx = idx;
        best_dur = dur;
      } else if (dur > best_dur) {
        best_dur = dur;
        best_idx = idx;
      }
    }
    if (best_idx >= 0) keep[best_idx] = 1;
  }

  constexpr uint32_t kDropped = std::numeric_limits<uint32_t>::max();
  std::vector<uint32_t> old_to_new(n, kDropped);
  size_t out_idx = 0;
  Microseconds running_max = std::numeric_limits<Microseconds>::lowest();
  for (size_t i = 0; i < n; ++i) {
    if (!keep[i]) continue;
    old_to_new[i] = static_cast<uint32_t>(out_idx);
    if (out_idx != i) {
      entries.entry_start_times[out_idx] = entries.entry_start_times[i];
      entries.entry_total_times[out_idx] = entries.entry_total_times[i];
      entries.entry_self_times[out_idx] = entries.entry_self_times[i];
      entries.entry_levels[out_idx] = entries.entry_levels[i];
      cached.entry_name_ids[out_idx] = cached.entry_name_ids[i];
      entries.entry_event_ids[out_idx] = entries.entry_event_ids[i];
      if (!cached.entry_tids.empty()) {
        cached.entry_tids[out_idx] = cached.entry_tids[i];
      }
      entries.entry_serials[out_idx] = entries.entry_serials[i];
      entries.entry_arg_flags[out_idx] = entries.entry_arg_flags[i];
      entries.entry_hlo_module_ids[out_idx] = entries.entry_hlo_module_ids[i];
    }
    running_max = std::max(running_max, entries.entry_start_times[out_idx] +
                                            entries.entry_total_times[out_idx]);
    cached.prefix_max_end_times[out_idx] = running_max;
    ++out_idx;
  }

  if (out_idx == n) return false;

  ResizeCachedTrackArrays(cached, out_idx, /*shrink=*/true);

  for (auto& open_span : cached.open_spans) {
    if (open_span.entry_idx < n &&
        old_to_new[open_span.entry_idx] != kDropped) {
      open_span.entry_idx = old_to_new[open_span.entry_idx];
    }
  }

  constexpr uint32_t kUnusedName = std::numeric_limits<uint32_t>::max();
  std::vector<uint32_t> name_remap(cached.unique_names.size(), kUnusedName);
  size_t surviving_unique_names = 0;
  for (const uint32_t name_id : cached.entry_name_ids) {
    if (name_remap[name_id] == kUnusedName) {
      name_remap[name_id] = 1;
      ++surviving_unique_names;
    }
  }
  for (const auto& open_span : cached.open_spans) {
    if (open_span.name_id < name_remap.size() &&
        name_remap[open_span.name_id] == kUnusedName) {
      name_remap[open_span.name_id] = 1;
      ++surviving_unique_names;
    }
  }
  if (surviving_unique_names < cached.unique_names.size()) {
    absl::c_fill(name_remap, kUnusedName);
    std::vector<std::string> compacted_names;
    compacted_names.reserve(surviving_unique_names);
    cached.name_to_id.clear();
    cached.name_to_id.reserve(surviving_unique_names);
    for (uint32_t& name_id : cached.entry_name_ids) {
      uint32_t& mapped = name_remap[name_id];
      if (mapped == kUnusedName) {
        mapped = static_cast<uint32_t>(compacted_names.size());
        compacted_names.push_back(std::move(cached.unique_names[name_id]));
        cached.name_to_id.emplace(compacted_names.back(), mapped);
      }
      name_id = mapped;
    }
    for (auto& open_span : cached.open_spans) {
      if (open_span.name_id < name_remap.size()) {
        uint32_t& mapped = name_remap[open_span.name_id];
        if (mapped == kUnusedName) {
          mapped = static_cast<uint32_t>(compacted_names.size());
          compacted_names.push_back(
              std::move(cached.unique_names[open_span.name_id]));
          cached.name_to_id.emplace(compacted_names.back(), mapped);
        }
        open_span.name_id = mapped;
      }
    }
    cached.unique_names = std::move(compacted_names);
  }

  auto remap_sparse_map = [&](auto& sparse_map) {
    if (sparse_map.empty()) return;
    std::decay_t<decltype(sparse_map)> remapped;
    remapped.reserve(sparse_map.size());
    for (auto& [old_i, val] : sparse_map) {
      if (old_i < n && old_to_new[old_i] != kDropped) {
        remapped[old_to_new[old_i]] = std::move(val);
      }
    }
    sparse_map = std::move(remapped);
  };
  remap_sparse_map(entries.sparse_entry_group_ids);
  remap_sparse_map(entries.sparse_entry_args);

  cached.was_compacted = true;
  return true;
}

CachedThreadTrack BuildCachedThreadTrack(
    absl::Span<const TraceEvent* const> events,
    const TraceInformation& trace_info, absl::string_view thread_group_name) {
  CachedThreadTrack cached;
  AppendToCachedThreadTrack(cached, events, trace_info, thread_group_name);
  return cached;
}

template <typename T>
void AppendVector(std::vector<T>& dst, const std::vector<T>& src) {
  dst.insert(dst.end(), src.begin(), src.end());
}

void AppendCachedThreadTrackToTimelineData(CachedThreadTrack& cached,
                                           int start_level, int& max_level,
                                           FlameChartTimelineData& data,
                                           TimeBounds& bounds,
                                           bool consume_track = false) {
  const size_t num_entries = cached.size();
  if (num_entries == 0) return;
  if (start_level < 0 || start_level + cached.max_relative_level >=
                             std::numeric_limits<uint16_t>::max()) {
    LOG_FIRST_N(WARNING, 1) << "Dropping cached track at level " << start_level
                            << ": exceeds uint16_t flame chart level range";
    return;
  }
  FlameChartTimelineData& src = cached.entries;
  const size_t base_entry_idx = data.entry_start_times.size();
  max_level = std::max(max_level, start_level + cached.max_relative_level);
  bounds.min = std::min(bounds.min, cached.bounds.min);
  bounds.max = std::max(bounds.max, cached.bounds.max);

  AppendVector(data.entry_start_times, src.entry_start_times);
  AppendVector(data.entry_total_times, src.entry_total_times);
  AppendVector(data.entry_self_times, src.entry_self_times);
  data.entry_levels.reserve(data.entry_levels.size() + num_entries);
  for (const uint16_t rel_lvl : src.entry_levels) {
    data.entry_levels.push_back(
        static_cast<uint16_t>(start_level + static_cast<int>(rel_lvl)));
  }
  data.entry_names.reserve(data.entry_names.size() + num_entries);
  for (const uint32_t name_id : cached.entry_name_ids) {
    data.entry_names.push_back(cached.unique_names[name_id]);
  }
  AppendVector(data.entry_event_ids, src.entry_event_ids);
  AppendVector(data.entry_serials, src.entry_serials);
  AppendVector(data.entry_arg_flags, src.entry_arg_flags);

  std::vector<uint32_t> module_id_map(src.hlo_module_names.size() + 1, 0);
  for (size_t i = 0; i < src.hlo_module_names.size(); ++i) {
    module_id_map[i + 1] =
        InternHloModuleName(data.hlo_module_names, src.hlo_module_names[i]);
  }
  data.entry_hlo_module_ids.reserve(data.entry_hlo_module_ids.size() +
                                    num_entries);
  for (const uint32_t local_mod_id : src.entry_hlo_module_ids) {
    data.entry_hlo_module_ids.push_back(
        local_mod_id < module_id_map.size() ? module_id_map[local_mod_id] : 0);
  }

  for (const auto& [rel_idx, group_id] : src.sparse_entry_group_ids) {
    data.sparse_entry_group_ids[base_entry_idx + rel_idx] = group_id;
  }
  if (consume_track) {
    for (auto& [rel_idx, args] : src.sparse_entry_args) {
      data.sparse_entry_args[base_entry_idx + rel_idx] = std::move(args);
    }
    src.sparse_entry_args.clear();
  } else {
    for (const auto& [rel_idx, args] : src.sparse_entry_args) {
      data.sparse_entry_args[base_entry_idx + rel_idx] = args;
    }
  }
}

// Returns the length of the longest prefix `[0, split_idx)` of `cached` that
// can be retained intact without repacking when merging a sorted batch of new
// events whose first event starts at `min_new_ts` with duration
// `first_new_dur`.
//
// Uses `cached.prefix_max_end_times` (where `prefix_max_end_times[k]` is the
// monotonically non-decreasing maximum of `ts + dur` across `0..k`) to binary
// search for the first cached entry whose end time (or an ancestor's end time)
// extends past `min_new_ts`, and then rewinds across any cached entries at
// `ts == min_new_ts` with `dur < first_new_dur` so `(ts ascending, dur
// descending)` sort order is preserved across the boundary.
size_t FindNonOverlappingPrefixSplitIndex(const CachedThreadTrack& cached,
                                          Microseconds min_new_ts,
                                          Microseconds first_new_dur) {
  if (cached.empty()) return 0;
  const auto& starts = cached.entries.entry_start_times;
  const auto& durs = cached.entries.entry_total_times;
  auto it = std::upper_bound(cached.prefix_max_end_times.begin(),
                             cached.prefix_max_end_times.end(), min_new_ts);
  size_t split_idx =
      static_cast<size_t>(it - cached.prefix_max_end_times.begin());
  while (split_idx > 0) {
    const size_t prev = split_idx - 1;
    if (starts[prev] < min_new_ts ||
        (starts[prev] == min_new_ts && durs[prev] >= first_new_dur)) {
      break;
    }
    --split_idx;
  }
  return split_idx;
}

TraceEvent ReconstructCachedTraceEvent(const CachedThreadTrack& cached,
                                       size_t i, bool is_async) {
  const FlameChartTimelineData& e = cached.entries;
  const uint8_t flags = e.entry_arg_flags[i];
  TraceEvent ev{
      .ph = Phase::kComplete,
      .event_id = e.entry_event_ids[i],
      .pid = cached.pid,
      .tid = cached.tid_at(i),
      .name = cached.unique_names[cached.entry_name_ids[i]],
      .ts = e.entry_start_times[i],
      .dur = e.entry_total_times[i],
      .serial = e.entry_serials[i],
      .has_serial = (flags & FlameChartTimelineData::kEntryArgHasUid) != 0,
      .is_async = is_async,
  };
  if (const auto it = e.sparse_entry_group_ids.find(i);
      it != e.sparse_entry_group_ids.end()) {
    ev.group_id = it->second;
    ev.has_group_id = true;
  }
  if (const auto it = e.sparse_entry_args.find(i);
      it != e.sparse_entry_args.end()) {
    ev.args = it->second;
    if (ev.has_serial) ev.args.erase("uid");
    if (ev.has_group_id) ev.args.erase("group_id");
    if ((flags & FlameChartTimelineData::kEntryArgHadRawHloOp) == 0) {
      ev.args.erase(kHloOp);
    }
    if ((flags & FlameChartTimelineData::kEntryArgHadRawHloModule) == 0) {
      ev.args.erase(kHloModule);
    }
  }
  return ev;
}

// Reconstructs `TraceEvent` objects for the suffix `[start_idx, cached.size())`
// of `cached` and truncates `cached` in-place to the non-overlapping prefix
// `[0, start_idx)`, updating `bounds` and `max_relative_level` to match the
// retained prefix.
std::vector<TraceEvent> RestoreAndTruncateCachedTrackFrom(
    CachedThreadTrack& cached, size_t start_idx, bool is_async) {
  const size_t n = cached.size();
  FlameChartTimelineData& e = cached.entries;
  std::vector<TraceEvent> restored;
  restored.reserve(n - start_idx);
  for (size_t i = start_idx; i < n; ++i) {
    restored.push_back(ReconstructCachedTraceEvent(cached, i, is_async));
    e.sparse_entry_group_ids.erase(i);
    e.sparse_entry_args.erase(i);
  }
  if (start_idx == 0) {
    cached = CachedThreadTrack{};
    return restored;
  }
  ResizeCachedTrackArrays(cached, start_idx, /*shrink=*/false);
  cached.bounds.min = e.entry_start_times.front();
  cached.bounds.max = cached.prefix_max_end_times.back();
  cached.open_spans.erase(
      std::remove_if(cached.open_spans.begin(), cached.open_spans.end(),
                     [start_idx](const CachedThreadTrack::OpenSpan& os) {
                       return os.entry_idx >= start_idx;
                     }),
      cached.open_spans.end());
  cached.max_relative_level =
      e.entry_levels.empty() ? 0 : *absl::c_max_element(e.entry_levels);
  return restored;
}

void StitchOpenSpansInCachedTrack(
    CachedThreadTrack& cached, absl::Span<const TraceEvent* const> new_events,
    const TraceInformation& trace_info, absl::string_view thread_group_name) {
  static constexpr Microseconds kTsEpsilon = 1e-7;
  const size_t num_open = cached.open_spans.size();
  std::vector<const TraceEvent*> matched_ev(num_open, nullptr);
  std::vector<const TraceEvent*> remaining_new_events;
  remaining_new_events.reserve(new_events.size());

  for (const TraceEvent* ev : new_events) {
    bool matched = false;
    if (ev->ph == Phase::kDurationBegin || ev->ph == Phase::kDurationEnd) {
      for (size_t s = 0; s < num_open; ++s) {
        if (matched_ev[s] != nullptr) continue;
        const auto& os = cached.open_spans[s];
        if (std::abs(ev->ts - os.start_ts) <= kTsEpsilon &&
            ev->name == cached.unique_names[os.name_id]) {
          matched_ev[s] = ev;
          matched = true;
          break;
        }
      }
    }
    if (!matched) {
      remaining_new_events.push_back(ev);
    }
  }

  const Microseconds min_remaining_ts =
      remaining_new_events.empty() ? std::numeric_limits<Microseconds>::max()
                                   : remaining_new_events.front()->ts;

  struct ActiveProxyInfo {
    size_t entry_idx;
    Microseconds full_dur;
    EventId full_event_id;
    Microseconds new_end_ts;
  };
  std::vector<TraceEvent> proxy_storage;
  std::vector<int> proxy_levels;
  std::vector<ActiveProxyInfo> active_proxies;
  std::vector<CachedThreadTrack::OpenSpan> surviving_open_spans;
  proxy_storage.reserve(num_open);
  proxy_levels.reserve(num_open);
  active_proxies.reserve(num_open);
  surviving_open_spans.reserve(num_open);

  FlameChartTimelineData& e = cached.entries;
  for (size_t s = 0; s < num_open; ++s) {
    auto os = cached.open_spans[s];
    const TraceEvent* ev = matched_ev[s];
    Microseconds full_dur = e.entry_total_times[os.entry_idx];
    EventId full_event_id = e.entry_event_ids[os.entry_idx];
    bool is_closed = false;
    if (ev != nullptr) {
      full_dur = std::max(full_dur, ev->dur);
      full_event_id = ev->event_id;
      is_closed = (ev->ph == Phase::kDurationEnd);
    }
    const Microseconds new_end_ts = os.start_ts + full_dur;
    Microseconds proxy_start_ts = os.current_end_ts;
    if (min_remaining_ts < proxy_start_ts && min_remaining_ts >= os.start_ts) {
      proxy_start_ts = min_remaining_ts;
    }
    if (new_end_ts > proxy_start_ts) {
      TraceEvent proxy;
      proxy.ph = is_closed ? Phase::kDurationEnd : Phase::kDurationBegin;
      proxy.pid = cached.pid;
      proxy.tid = cached.tid_at(os.entry_idx);
      proxy.ts = proxy_start_ts;
      proxy.dur = new_end_ts - proxy_start_ts;
      proxy_storage.push_back(std::move(proxy));
      proxy_levels.push_back(os.relative_level);
      active_proxies.push_back(
          ActiveProxyInfo{os.entry_idx, full_dur, full_event_id, new_end_ts});
    } else {
      e.entry_total_times[os.entry_idx] = full_dur;
      e.entry_event_ids[os.entry_idx] = full_event_id;
      cached.bounds.max = std::max(cached.bounds.max, new_end_ts);
      if (!cached.prefix_max_end_times.empty()) {
        cached.prefix_max_end_times.back() =
            std::max(cached.prefix_max_end_times.back(), new_end_ts);
      }
    }
    if (!is_closed) {
      os.current_end_ts = std::max(os.current_end_ts, new_end_ts);
      surviving_open_spans.push_back(os);
    }
  }

  cached.open_spans = std::move(surviving_open_spans);

  std::vector<const TraceEvent*> combined_events;
  combined_events.reserve(proxy_storage.size() + remaining_new_events.size());
  for (const TraceEvent& proxy : proxy_storage) {
    combined_events.push_back(&proxy);
  }
  combined_events.insert(combined_events.end(), remaining_new_events.begin(),
                         remaining_new_events.end());

  std::vector<Microseconds> proxy_self_times;
  if (!combined_events.empty()) {
    AppendToCachedThreadTrack(cached, combined_events, trace_info,
                              thread_group_name, proxy_levels,
                              &proxy_self_times);
  }
  for (size_t p = 0; p < active_proxies.size(); ++p) {
    const auto& info = active_proxies[p];
    const Microseconds self_time_in_chunk = p < proxy_self_times.size()
                                                ? proxy_self_times[p]
                                                : proxy_storage[p].dur;
    const Microseconds chunk_children_dur =
        proxy_storage[p].dur - self_time_in_chunk;
    const Microseconds delta_total_time =
        info.full_dur - e.entry_total_times[info.entry_idx];
    e.entry_total_times[info.entry_idx] = info.full_dur;
    e.entry_self_times[info.entry_idx] =
        std::max(0.0, e.entry_self_times[info.entry_idx] + delta_total_time -
                          chunk_children_dur);
    e.entry_event_ids[info.entry_idx] = info.full_event_id;
    cached.bounds.max = std::max(cached.bounds.max, info.new_end_ts);
    if (!cached.prefix_max_end_times.empty()) {
      cached.prefix_max_end_times.back() =
          std::max(cached.prefix_max_end_times.back(), info.new_end_ts);
    }
  }
  if (cached.open_spans.size() > 1) {
    absl::c_sort(cached.open_spans, [](const CachedThreadTrack::OpenSpan& a,
                                       const CachedThreadTrack::OpenSpan& b) {
      return a.relative_level < b.relative_level;
    });
  }
}

// Merges `new_events` into `cached` and, when `allow_compaction` is true,
// compacts `cached` if it exceeds the per-track budget. Returns true if
// compaction dropped any entries.
bool UpdateCachedTrackIncremental(
    CachedThreadTrack& cached, bool is_async,
    std::vector<const TraceEvent*>& new_events,
    const TraceInformation& trace_info, absl::string_view thread_group_name,
    bool allow_compaction = true,
    const absl::flat_hash_set<EventId>* pinned_event_ids = nullptr) {
  if (new_events.empty()) return false;
  auto cmp_events = [](const TraceEvent* a, const TraceEvent* b) {
    if (a->ts != b->ts) return a->ts < b->ts;
    if (a->dur != b->dur) return a->dur > b->dur;
    return a->ph != b->ph && a->ph == Phase::kDurationEnd;
  };
  if (!std::is_sorted(new_events.begin(), new_events.end(), cmp_events)) {
    absl::c_stable_sort(new_events, cmp_events);
  }
  if (new_events.size() > 1) {
    size_t write_idx = 0;
    for (const TraceEvent* ev : new_events) {
      if (ev->ph == Phase::kDurationBegin || ev->ph == Phase::kDurationEnd) {
        bool duplicate = false;
        for (size_t j = write_idx; j > 0;) {
          const TraceEvent* prev = new_events[--j];
          if (prev->ts != ev->ts) break;
          if ((prev->ph == Phase::kDurationBegin ||
               prev->ph == Phase::kDurationEnd) &&
              prev->name == ev->name) {
            duplicate = true;
            break;
          }
        }
        if (duplicate) continue;
      }
      new_events[write_idx++] = ev;
    }
    new_events.resize(write_idx);
  }
  if (!cached.open_spans.empty()) {
    StitchOpenSpansInCachedTrack(cached, new_events, trace_info,
                                 thread_group_name);
    return allow_compaction &&
           CompactCachedThreadTrackIfNeeded(
               cached, kDefaultMaxTrackEvents, kDefaultTargetTrackEvents,
               kDefaultTailKeepEvents, pinned_event_ids);
  }
  const size_t split_idx = FindNonOverlappingPrefixSplitIndex(
      cached, new_events.front()->ts, new_events.front()->dur);
  if (split_idx == cached.size()) {
    AppendToCachedThreadTrack(cached, new_events, trace_info,
                              thread_group_name);
    return allow_compaction &&
           CompactCachedThreadTrackIfNeeded(
               cached, kDefaultMaxTrackEvents, kDefaultTargetTrackEvents,
               kDefaultTailKeepEvents, pinned_event_ids);
  }
  const bool was_compacted = cached.was_compacted;
  RestoredMetaMap restored_meta;
  std::vector<RestoredEntryMeta> saved_meta;
  if (was_compacted) {
    saved_meta.reserve(cached.size() - split_idx);
    for (size_t i = split_idx; i < cached.size(); ++i) {
      saved_meta.push_back(
          {cached.entries.entry_self_times[i],
           static_cast<int>(cached.entries.entry_levels[i])});
    }
  }
  const std::vector<TraceEvent> restored =
      RestoreAndTruncateCachedTrackFrom(cached, split_idx, is_async);
  cached.was_compacted = was_compacted;
  if (was_compacted) {
    restored_meta.reserve(restored.size());
    for (size_t k = 0; k < restored.size(); ++k) {
      restored_meta.try_emplace(&restored[k], saved_meta[k]);
    }
  }
  std::vector<const TraceEvent*> merged_events;
  merged_events.reserve(restored.size() + new_events.size());
  for (const TraceEvent& ev : restored) merged_events.push_back(&ev);
  merged_events.insert(merged_events.end(), new_events.begin(),
                       new_events.end());
  std::inplace_merge(
      merged_events.begin(),
      merged_events.begin() + static_cast<ptrdiff_t>(restored.size()),
      merged_events.end(), cmp_events);
  AppendToCachedThreadTrack(
      cached, merged_events, trace_info, thread_group_name,
      /*proxy_fixed_levels=*/{}, /*proxy_self_times_out=*/nullptr,
      was_compacted ? &restored_meta : nullptr);
  return allow_compaction &&
         CompactCachedThreadTrackIfNeeded(
             cached, kDefaultMaxTrackEvents, kDefaultTargetTrackEvents,
             kDefaultTailKeepEvents, pinned_event_ids);
}

// Re-evaluates HLO module associations for cached entries in `cached` starting
// at or after `min_dirty_xla_ts` when the process's `XLA Modules` track is
// modified by a later incremental batch. Entries whose original `TraceEvent`
// explicitly specified `kHloModule` in `args` (`kEntryArgHadRawHloModule`) are
// left unchanged.
void RefreshCachedTrackHloModules(CachedThreadTrack& cached,
                                  const TraceInformation& trace_info,
                                  absl::string_view thread_name,
                                  Microseconds min_dirty_xla_ts) {
  const bool is_xla_ops_thread = thread_name == kXlaOps;
  const bool is_data_motion_layer = thread_name == kComputeUtilization ||
                                    thread_name == kDataMotionLayersUtilization;
  if (!is_xla_ops_thread && !is_data_motion_layer) return;

  const size_t n = cached.size();
  if (n == 0) return;
  FlameChartTimelineData& e = cached.entries;
  auto it_start = std::lower_bound(e.entry_start_times.begin(),
                                   e.entry_start_times.end(), min_dirty_xla_ts);
  const size_t start_i =
      static_cast<size_t>(it_start - e.entry_start_times.begin());
  if (start_i >= n) return;

  const TrackSpanSource module_src = ResolveTrackSpanSource(
      cached.pid, trace_info.xla_modules_tids, trace_info);
  size_t module_cursor = 0;
  Microseconds last_module_lookup_ts =
      std::numeric_limits<Microseconds>::lowest();
  for (size_t i = start_i; i < n; ++i) {
    const absl::string_view module_name = LookupTrackSpanAtTimestamp(
        module_src, module_cursor, last_module_lookup_ts,
        e.entry_start_times[i]);
    if (auto it_args = e.sparse_entry_args.find(i);
        it_args == e.sparse_entry_args.end()) {
      e.entry_hlo_module_ids[i] =
          InternHloModuleName(e.hlo_module_names, module_name);
    } else if ((e.entry_arg_flags[i] &
                FlameChartTimelineData::kEntryArgHadRawHloModule) == 0) {
      it_args->second[std::string(kHloModule)] = std::string(module_name);
    }
  }
}

template <typename CacheMap, typename Key>
CachedThreadTrack& GetOrBuildCachedTrack(
    CacheMap& cache_map, const Key& cache_key,
    absl::Span<const TraceEvent* const> events,
    const TraceInformation& trace_info, absl::string_view thread_group_name) {
  auto it = cache_map.find(cache_key);
  if (it == cache_map.end()) {
    it = cache_map
             .emplace(cache_key, BuildCachedThreadTrack(events, trace_info,
                                                        thread_group_name))
             .first;
  }
  return it->second;
}

void PopulateThreadTrack(
    ProcessId pid, ThreadId tid, absl::Span<const TraceEvent* const> events,
    const TraceInformation& trace_info, int& current_level,
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
  bool track_has_events = !events.empty();

  CachedThreadTrack* cached = nullptr;
  if (custom_name.has_value() && trace_info.cached_async_tracks != nullptr) {
    cached = &GetOrBuildCachedTrack(
        *trace_info.cached_async_tracks,
        std::pair<ProcessId, std::string>{pid, thread_group_name}, events,
        trace_info, thread_group_name);
  } else if (!custom_name.has_value() &&
             trace_info.cached_sync_tracks != nullptr) {
    cached = &GetOrBuildCachedTrack(*trace_info.cached_sync_tracks,
                                    std::pair<ProcessId, ThreadId>{pid, tid},
                                    events, trace_info, thread_group_name);
  }
  if (cached != nullptr) {
    track_has_events = !cached->empty();
    AppendCachedThreadTrackToTimelineData(*cached, start_level, max_level, data,
                                          bounds,
                                          trace_info.consume_cached_tracks);
  } else {
    PopulateThreadTrackEvents(events, start_level, max_level, data, bounds,
                              trace_info, thread_group_name);
  }

  const GroupKey group_key{
      .nesting_level = kThreadNestingLevel,
      .name = thread_group_name,
      .parent_name = process_group_name,
  };

  const int current_levels =
      track_has_events ? (max_level - start_level + 1) : 1;
  int& max_observed = max_observed_levels[group_key];
  const int level_count = std::max(current_levels, max_observed);
  max_observed = level_count;
  // NOMUTANTS(SBR): Timeline::SetTimelineData backfills level_count from level
  // difference if <= 0; removal is unobservable through Timeline public API.
  data.groups.back().level_count = level_count;

  current_level = start_level + level_count;

  thread_levels.try_emplace({pid, tid},
                            ThreadLevelInfo{start_level, current_level});

  if (max_level == start_level && !expanded_states.contains(group_key)) {
    data.groups.back().expanded = true;
  }
}

void PopulateCounterTrack(
    ProcessId pid, const std::string& name,
    absl::Span<const CounterEvent* const> events,
    const TraceInformation& trace_info, int& current_level,
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
    ProcessId pid, const TraceInformation& trace_info,
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

  if (trace_info.cached_async_tracks != nullptr) {
    for (const auto& [pid_name, cached] : *trace_info.cached_async_tracks) {
      if (pid_name.first != pid || cached.empty()) continue;
      const size_t count = cached.entry_tids.empty() ? 1 : cached.size();
      for (size_t i = 0; i < count; ++i) {
        if (const ThreadId t = cached.tid_at(i);
            !known_threads.contains({pid, t})) {
          async_only_tids.insert(t);
        }
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
    const std::string& process_group_name, const TraceInformation& trace_info,
    int& current_level, FlameChartTimelineData& data, TimeBounds& bounds,
    absl::btree_map<std::pair<ProcessId, ThreadId>, ThreadLevelInfo>&
        thread_levels,
    const absl::btree_map<GroupKey, bool>& expanded_states,
    absl::flat_hash_map<GroupKey, int>& max_observed_levels, int parent_index) {
  for (const auto& [name, named_events] : async_groups) {
    ThreadId tid = 0;
    if (!named_events.empty()) {
      tid = named_events.front()->tid;
    } else if (trace_info.cached_async_tracks != nullptr) {
      if (const auto it_cache =
              trace_info.cached_async_tracks->find({pid, name});
          it_cache != trace_info.cached_async_tracks->end() &&
          !it_cache->second.empty()) {
        tid = it_cache->second.tid;
      }
    }
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
    const std::string& process_group_name, const TraceInformation& trace_info,
    int& current_level, FlameChartTimelineData& data, TimeBounds& bounds,
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
    bool has_cached_events = false;
    if (trace_info.cached_sync_tracks != nullptr) {
      auto it_cache = trace_info.cached_sync_tracks->find({pid, tid});
      has_cached_events =
          it_cache != trace_info.cached_sync_tracks->end() &&
          !it_cache->second.empty();
    }

    if (trace_info.is_mpmd && events.empty() && !has_cached_events) {
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
    const TraceInformation& trace_info, int& current_level,
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

void PopulateProcessTrack(
    ProcessId pid, TraceInformation& trace_info, int& current_level,
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

std::vector<ProcessId> GetSortedProcessIds(
    const TraceInformation& trace_info,
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
    for (const auto* event : events) {
      pid_set.insert(event->pid);
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
    TraceInformation& trace_info, absl::Span<const ProcessId> sorted_pids,
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
    auto add_cached_counts = [&](const auto* tracks) {
      if (tracks == nullptr) return;
      for (const auto& [_, cached] : *tracks) total_events += cached.size();
    };
    add_cached_counts(trace_info.cached_sync_tracks);
    add_cached_counts(trace_info.cached_async_tracks);
    if (total_events > 0) {
      data.entry_start_times.reserve(total_events);
      data.entry_total_times.reserve(total_events);
      data.entry_self_times.reserve(total_events);
      data.entry_levels.reserve(total_events);
      data.entry_names.reserve(total_events);
      data.entry_event_ids.reserve(total_events);
      data.entry_serials.reserve(total_events);
      data.entry_arg_flags.reserve(total_events);
      data.entry_hlo_module_ids.reserve(total_events);
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
  const auto cmp_by_start_asc_then_dur_desc =
      CompareEntriesByStartAscThenDurDesc(data);
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
  return data;
}

template <typename TrackMap, typename SkipKeyFn>
bool CompactTrackMap(
    TrackMap& tracks, size_t compactable_events, size_t compactable_target,
    size_t compactable_tracks, SkipKeyFn skip_key,
    const absl::flat_hash_set<EventId>* pinned_event_ids = nullptr) {
  const size_t per_track_cap =
      compactable_target / std::max<size_t>(1, compactable_tracks);
  const size_t min_track_events =
      std::min(kMinTrackEventsForGlobalCompaction,
               std::max<size_t>(1, per_track_cap / 2));
  bool compacted = false;
  for (auto& [key, cached] : tracks) {
    if (skip_key(key)) continue;
    const size_t n = cached.size();
    if (n <= min_track_events) continue;
    const size_t target = std::max<size_t>(
        min_track_events,
        static_cast<size_t>(static_cast<uint64_t>(n) * compactable_target /
                            compactable_events));
    if (CompactCachedThreadTrackIfNeeded(
            cached, target, target,
            std::min(kGlobalTailKeepEvents, target / 4), pinned_event_ids)) {
      compacted = true;
    }
  }
  return compacted;
}

void SortFlowEvents(std::vector<const TraceEvent*>& events) {
  absl::c_stable_sort(events, [](const TraceEvent* a, const TraceEvent* b) {
    return a->ts < b->ts;
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

struct DataProvider::IncrementalState {
  struct EventBatch {
    absl::flat_hash_map<std::pair<ProcessId, ThreadId>,
                        std::vector<const TraceEvent*>>
        sync_events;
    absl::flat_hash_map<std::pair<ProcessId, std::string>,
                        std::vector<const TraceEvent*>>
        async_events;
    absl::flat_hash_map<ProcessId, Microseconds> dirty_xla_module_min_ts;
  };

  std::deque<TraceEvent> flow_events;
  std::deque<CounterEvent> counter_events;
  bool has_any_events = false;
  bool is_refetch = false;
  bool is_finishing = false;
  bool emitted_timeline_data = false;
  bool has_unflushed_updates = false;
  bool was_compacted = false;
  bool allow_compaction = true;
  TraceInformation trace_info;
  absl::btree_map<int, int> flow_category_counts;
  absl::flat_hash_map<std::pair<ProcessId, ThreadId>, CachedThreadTrack>
      cached_sync_tracks;
  absl::flat_hash_map<std::pair<ProcessId, std::string>, CachedThreadTrack>
      cached_async_tracks;
  absl::btree_map<GroupKey, bool> base_expanded_states;
  absl::btree_map<GroupKey, bool> last_emitted_expanded_states;
  std::optional<std::pair<Milliseconds, Milliseconds>> full_timespan;
  std::optional<std::pair<Milliseconds, Milliseconds>> visible_range_from_url;
  TimeRange initial_data_time_range = TimeRange::Zero();
  TimeRange last_auto_visible_range = TimeRange::Zero();
  TimeBounds observed_time_bounds;
  size_t total_ingested_events = 0;
  size_t last_flushed_events = 0;
  size_t last_flushed_processes = 0;

  void CopyMetadataFromProvider(const DataProvider& provider) {
    trace_info.process_names = provider.process_names_;
    trace_info.thread_names = provider.thread_names_;
    trace_info.process_sort_indices = provider.process_sort_indices_;
    trace_info.thread_sort_indices = provider.thread_sort_indices_;
    trace_info.xla_modules_tids = provider.xla_modules_tids_;
    trace_info.async_processes_by_events = provider.known_async_processes_;
  }

  void CopyMetadataToProvider(DataProvider& provider) const {
    provider.process_names_ = trace_info.process_names;
    provider.thread_names_ = trace_info.thread_names;
    provider.process_sort_indices_ = trace_info.process_sort_indices;
    provider.thread_sort_indices_ = trace_info.thread_sort_indices;
    provider.xla_modules_tids_ = trace_info.xla_modules_tids;
    provider.known_async_processes_ = trace_info.async_processes_by_events;
  }

  EventBatch PartitionFlameEvents(absl::Span<const TraceEvent> flame_events,
                                  DataProvider& provider) {
    EventBatch batch;
    std::pair<ProcessId, ThreadId> last_sync_key{0, 0};
    std::vector<const TraceEvent*>* last_sync_vec = nullptr;

    for (const TraceEvent& stored : flame_events) {
      switch (stored.ph) {
        case Phase::kMetadata: {
          const auto old_it = trace_info.xla_modules_tids.find(stored.pid);
          const std::optional<ThreadId> old_tid =
              old_it != trace_info.xla_modules_tids.end()
                  ? std::make_optional(old_it->second)
                  : std::nullopt;
          HandleMetadataEvent(stored, trace_info);
          if (const auto new_it = trace_info.xla_modules_tids.find(stored.pid);
              new_it != trace_info.xla_modules_tids.end() &&
              new_it->second != old_tid) {
            batch.dirty_xla_module_min_ts[stored.pid] =
                std::numeric_limits<Microseconds>::lowest();
          }
          break;
        }
        case Phase::kComplete:
        case Phase::kDurationBegin:
        case Phase::kDurationEnd:
        case Phase::kInstant:
        case Phase::kInstantDeprecated: {
          if (stored.is_async) {
            batch.async_events[{stored.pid, stored.name}].push_back(&stored);
            provider.known_async_tracks_[stored.pid].insert(stored.name);
            trace_info.async_processes_by_events.insert(stored.pid);
          } else {
            const std::pair<ProcessId, ThreadId> sync_key{stored.pid,
                                                          stored.tid};
            if (last_sync_vec == nullptr || sync_key != last_sync_key) {
              last_sync_key = sync_key;
              last_sync_vec = &batch.sync_events[sync_key];
            }
            last_sync_vec->push_back(&stored);
          }
          break;
        }
        default:
          break;
      }
    }

    MergeProcessSortIndices(trace_info);
    for (const auto& [sync_key, _] : batch.sync_events) {
      provider.known_threads_.insert(sync_key);
    }
    return batch;
  }

  void IngestFlowAndCounterEvents(ParsedTraceEvents& parsed_events,
                                  DataProvider& provider) {
    absl::flat_hash_set<std::string> dirty_flow_ids;
    for (TraceEvent& event : parsed_events.flow_events) {
      if (!event.is_async && event.tid < kSyntheticOpenSpanTidBit) {
        provider.known_threads_.insert({event.pid, event.tid});
      }
      if (!event.is_async) {
        trace_info.events_by_pid_tid[event.pid].try_emplace(event.tid);
      }
      if (flow_events.size() >= kMaxIncrementalFlowEvents &&
          !trace_info.flow_events_by_id.contains(event.id)) {
        ++flow_category_counts[static_cast<int>(event.category)];
        continue;
      }
      event.args.clear();
      flow_events.push_back(std::move(event));
      const TraceEvent& stored = flow_events.back();
      HandleFlowEvent(stored, trace_info, flow_category_counts);
      dirty_flow_ids.insert(stored.id);
    }
    UpdateFlowCategories(flow_category_counts,
                         provider.present_flow_categories_);
    for (const std::string& flow_id : dirty_flow_ids) {
      SortFlowEvents(trace_info.flow_events_by_id[flow_id]);
    }

    absl::flat_hash_set<std::pair<ProcessId, std::string>> dirty_counters;
    for (CounterEvent& event : parsed_events.counter_events) {
      counter_events.push_back(std::move(event));
      const CounterEvent& stored = counter_events.back();
      HandleCounterEvent(stored, trace_info);
      dirty_counters.insert({stored.pid, stored.name});
      provider.known_counters_[stored.pid].insert(stored.name);
    }
    for (const auto& [pid, name] : dirty_counters) {
      SortCounterEvents(trace_info.counters_by_pid_name[pid][name]);
    }
  }

  void UpdateCachedTracks(
      EventBatch& batch,
      const absl::flat_hash_set<EventId>* pinned_event_ids = nullptr) {
    auto get_sync_thread_name = [&](ProcessId pid, ThreadId tid) {
      const auto it = trace_info.thread_names.find({pid, tid});
      return it == trace_info.thread_names.end() ? GetDefaultThreadName(tid)
                                                 : it->second;
    };

    // Pass 1: Update XLA Modules tracks first so dependent tracks (such as XLA
    // Ops) see the newly appended XLA Module spans when looking up HLO modules.
    for (auto& [pid_tid, events] : batch.sync_events) {
      const auto xla_it = trace_info.xla_modules_tids.find(pid_tid.first);
      if (xla_it == trace_info.xla_modules_tids.end() ||
          xla_it->second != pid_tid.second || events.empty()) {
        continue;
      }
      Microseconds min_ts = events.front()->ts;
      for (const TraceEvent* ev : events) min_ts = std::min(min_ts, ev->ts);
      auto [it_min, inserted] =
          batch.dirty_xla_module_min_ts.try_emplace(pid_tid.first, min_ts);
      if (!inserted) it_min->second = std::min(it_min->second, min_ts);
      UpdateCachedTrackIncremental(
          cached_sync_tracks[pid_tid], /*is_async=*/false, events, trace_info,
          get_sync_thread_name(pid_tid.first, pid_tid.second),
          /*allow_compaction=*/false);
      events.clear();
    }

    // Refresh HLO module IDs on any existing cached entries that overlap newly
    // added or modified XLA Modules tracks.
    if (!batch.dirty_xla_module_min_ts.empty()) {
      for (auto& [pid_tid, cached] : cached_sync_tracks) {
        const auto it_min = batch.dirty_xla_module_min_ts.find(pid_tid.first);
        if (it_min == batch.dirty_xla_module_min_ts.end()) continue;
        RefreshCachedTrackHloModules(
            cached, trace_info,
            get_sync_thread_name(pid_tid.first, pid_tid.second),
            it_min->second);
      }
    }

    // Pass 2: Update all remaining sync tracks incrementally.
    for (auto& [pid_tid, events] : batch.sync_events) {
      if (events.empty()) continue;
      if (UpdateCachedTrackIncremental(
              cached_sync_tracks[pid_tid], /*is_async=*/false, events,
              trace_info, get_sync_thread_name(pid_tid.first, pid_tid.second),
              allow_compaction, pinned_event_ids)) {
        was_compacted = true;
      }
      events.clear();
    }

    // Pass 3: Update all async tracks incrementally.
    for (auto& [pid_name, events] : batch.async_events) {
      if (events.empty()) continue;
      if (UpdateCachedTrackIncremental(
              cached_async_tracks[pid_name], /*is_async=*/true, events,
              trace_info, pid_name.second, allow_compaction,
              pinned_event_ids)) {
        was_compacted = true;
      }
      events.clear();
    }

    if (!allow_compaction) return;

    // Uniform global cache budget across all tracks so full-session traces fit
    // without device-specific PID heuristics.
    const auto is_xla_modules_track =
        [&](const std::pair<ProcessId, ThreadId>& pid_tid) {
          const auto it = trace_info.xla_modules_tids.find(pid_tid.first);
          return it != trace_info.xla_modules_tids.end() &&
                 it->second == pid_tid.second;
        };
    size_t exempt_events = 0;
    size_t compactable_events = 0;
    size_t compactable_tracks = 0;
    for (const auto& [pid_tid, cached] : cached_sync_tracks) {
      if (cached.empty()) continue;
      if (is_xla_modules_track(pid_tid)) {
        exempt_events += cached.size();
      } else {
        compactable_events += cached.size();
        ++compactable_tracks;
      }
    }
    for (const auto& [_, cached] : cached_async_tracks) {
      if (cached.empty()) continue;
      compactable_events += cached.size();
      ++compactable_tracks;
    }
    if (exempt_events + compactable_events > kMaxGlobalCachedEvents &&
        compactable_events > 0) {
      const size_t compactable_target =
          kTargetGlobalCachedEvents > exempt_events
              ? (kTargetGlobalCachedEvents - exempt_events)
              : 0;
      const auto no_skip = [](const std::pair<ProcessId, std::string>&) {
        return false;
      };
      if (CompactTrackMap(cached_sync_tracks, compactable_events,
                          compactable_target, compactable_tracks,
                          is_xla_modules_track, pinned_event_ids)) {
        was_compacted = true;
      }
      if (CompactTrackMap(cached_async_tracks, compactable_events,
                          compactable_target, compactable_tracks, no_skip,
                          pinned_event_ids)) {
        was_compacted = true;
      }
    }
  }

  bool ShouldFlushTimeline(TimelineFlushMode flush_mode) const {
    if (is_finishing) return true;
    if (is_refetch || flush_mode == TimelineFlushMode::kDeferFlush) {
      return false;
    }
    const bool has_new_process =
        trace_info.process_names.size() != last_flushed_processes;
    const size_t min_events_between_flushes =
        has_new_process ? 200000 : 1500000;
    const bool should_throttle_flush =
        emitted_timeline_data && last_flushed_events > 250000 &&
        (total_ingested_events - last_flushed_events) <
            min_events_between_flushes;
    return !should_throttle_flush;
  }

  void UpdateTimeRanges(Timeline& timeline, const TimeBounds& batch_bounds,
                        bool has_known_tracks) {
    const bool has_valid_bounds =
        batch_bounds.min < std::numeric_limits<Microseconds>::max();
    if (has_valid_bounds) {
      observed_time_bounds.min =
          std::min(observed_time_bounds.min, batch_bounds.min);
      observed_time_bounds.max =
          std::max(observed_time_bounds.max, batch_bounds.max);
    }
    const TimeBounds& effective_bounds =
        has_valid_bounds ? observed_time_bounds : batch_bounds;
    const bool was_visible_zero =
        (timeline.visible_range() == TimeRange::Zero());

    UpdateTimelineTimeRanges(timeline, effective_bounds, has_known_tracks,
                             visible_range_from_url, full_timespan);

    if (has_valid_bounds) {
      if (was_visible_zero) {
        last_auto_visible_range = timeline.visible_range();
      } else if (!visible_range_from_url.has_value() &&
                 timeline.visible_range() == last_auto_visible_range) {
        const TimeRange candidate_range{effective_bounds.min,
                                        effective_bounds.max};
        const Microseconds old_dur = last_auto_visible_range.duration();
        const Microseconds new_dur = candidate_range.duration();
        if (is_finishing || old_dur < 1000000.0 || new_dur > old_dur * 1.15) {
          timeline.SetVisibleRange(candidate_range);
          last_auto_visible_range = timeline.visible_range();
        }
      }
    }
    if (!full_timespan.has_value() &&
        initial_data_time_range != TimeRange::Zero()) {
      timeline.set_data_time_range(
          {std::min(initial_data_time_range.start(),
                    timeline.fetched_data_time_range().start()),
           std::max(initial_data_time_range.end(),
                    timeline.fetched_data_time_range().end())});
    }
  }

  void FlushTimeline(Timeline& timeline, DataProvider& provider) {
    for (const auto& [key, expanded] :
         GetRestoredExpandedStates(timeline.timeline_data().groups)) {
      const auto it_last = last_emitted_expanded_states.find(key);
      if (it_last == last_emitted_expanded_states.end() ||
          it_last->second != expanded) {
        base_expanded_states[key] = expanded;
      }
    }

    TimeBounds time_bounds;
    const std::vector<ProcessId> sorted_pids = GetSortedProcessIds(
        trace_info, provider.known_threads_, provider.known_counters_,
        provider.known_async_tracks_);

    timeline.UpdateTimelineData([&]() {
      trace_info.consume_cached_tracks = is_finishing;
      FlameChartTimelineData new_timeline_data = CreateTimelineData(
          trace_info, sorted_pids, GetTop5FlowCategories(flow_category_counts),
          time_bounds, base_expanded_states, timeline.GetPalette(),
          provider.max_observed_levels_, provider.known_threads_,
          provider.known_counters_, provider.known_async_tracks_);
      if (is_finishing) {
        cached_sync_tracks = {};
        cached_async_tracks = {};
        trace_info.flow_events_by_id.clear();
        trace_info.counters_by_pid_name.clear();
        flow_events = {};
        counter_events = {};
      }
      return new_timeline_data;
    });
    emitted_timeline_data = true;
    has_unflushed_updates = false;
    last_flushed_events = total_ingested_events;
    last_flushed_processes = trace_info.process_names.size();
    last_emitted_expanded_states =
        GetRestoredExpandedStates(timeline.timeline_data().groups);

    UpdateTimeRanges(timeline, time_bounds, provider.HasKnownTracks());
  }
};

DataProvider::DataProvider() = default;
DataProvider::~DataProvider() = default;
DataProvider::DataProvider(DataProvider&&) noexcept = default;
DataProvider& DataProvider::operator=(DataProvider&&) noexcept = default;

bool DataProvider::HasKnownTracks() const {
  return !known_threads_.empty() || !known_counters_.empty() ||
         !known_async_tracks_.empty();
}

void DataProvider::Reset() {
  incremental_state_.reset();
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
  was_compacted_ = false;
  last_load_was_refetch_ = false;
}

void DataProvider::BeginIncrementalLoad(Timeline& timeline) {
  incremental_state_ = std::make_unique<IncrementalState>();
  incremental_state_->is_refetch =
      HasKnownTracks() && timeline.visible_range() != TimeRange::Zero();
  incremental_state_->allow_compaction = timeline.incremental_fetch_enabled();
  if (!incremental_state_->is_refetch) {
    timeline.ClearSavedFullOverview();
  }
  incremental_state_->initial_data_time_range = timeline.data_time_range();
  incremental_state_->CopyMetadataFromProvider(*this);
  incremental_state_->trace_info.cached_sync_tracks =
      &incremental_state_->cached_sync_tracks;
  incremental_state_->trace_info.cached_async_tracks =
      &incremental_state_->cached_async_tracks;
  incremental_state_->base_expanded_states =
      GetRestoredExpandedStates(timeline.timeline_data().groups);
  incremental_state_->last_emitted_expanded_states =
      incremental_state_->base_expanded_states;
  was_compacted_ = false;
  last_load_was_refetch_ = incremental_state_->is_refetch;
}

void DataProvider::MarkFinishingIncrementalLoad() {
  if (incremental_state_) {
    incremental_state_->is_finishing = true;
  }
}

bool DataProvider::HasPendingRefetchUpdate() const {
  return incremental_state_ && (incremental_state_->has_unflushed_updates ||
                                (incremental_state_->is_refetch &&
                                 !incremental_state_->emitted_timeline_data));
}

void DataProvider::AppendTraceEventsIncremental(ParsedTraceEvents parsed_events,
                                                Timeline& timeline,
                                                TimelineFlushMode flush_mode) {
  if (!incremental_state_) {
    BeginIncrementalLoad(timeline);
  }
  IncrementalState& state = *incremental_state_;
  if (parsed_events.mpmd_pipeline_view) {
    state.trace_info.is_mpmd = true;
  }
  timeline.set_mpmd_pipeline_view_enabled(state.trace_info.is_mpmd);
  timeline.set_time_axis_unit(parsed_events.time_axis_unit);

  if (parsed_events.full_timespan.has_value()) {
    state.full_timespan = parsed_events.full_timespan;
  }
  if (parsed_events.visible_range_from_url.has_value()) {
    state.visible_range_from_url = parsed_events.visible_range_from_url;
  }

  const size_t batch_events = parsed_events.flame_events.size() +
                              parsed_events.flow_events.size() +
                              parsed_events.counter_events.size();
  if (batch_events > 0) {
    state.has_any_events = true;
    state.has_unflushed_updates = true;
    state.total_ingested_events += batch_events;
  }

  IncrementalState::EventBatch batch =
      state.PartitionFlameEvents(parsed_events.flame_events, *this);
  state.IngestFlowAndCounterEvents(parsed_events, *this);
  state.CopyMetadataToProvider(*this);

  if (!state.has_any_events && !HasKnownTracks()) {
    timeline.SetTimelineData({});
    timeline.set_fetched_data_time_range(TimeRange::Zero());
    timeline.SetVisibleRange(TimeRange::Zero());
    return;
  }

  state.allow_compaction = timeline.incremental_fetch_enabled();
  absl::flat_hash_set<EventId> pinned_event_ids;
  if (state.allow_compaction) {
    const auto& entry_event_ids = timeline.timeline_data().entry_event_ids;
    const int sel_idx = timeline.selected_event_index();
    if (sel_idx >= 0 && static_cast<size_t>(sel_idx) < entry_event_ids.size()) {
      pinned_event_ids.insert(entry_event_ids[sel_idx]);
    }
    for (const int idx : timeline.get_matching_event_indices_for_test()) {
      if (idx >= 0 && static_cast<size_t>(idx) < entry_event_ids.size()) {
        pinned_event_ids.insert(entry_event_ids[idx]);
      }
    }
    for (const auto& sr : timeline.get_search_results_for_test()) {
      if (sr.event_id != 0) pinned_event_ids.insert(sr.event_id);
    }
    if (const auto pending_id =
            timeline.get_pending_navigation_event_id_for_test();
        pending_id.has_value()) {
      pinned_event_ids.insert(*pending_id);
    }
  }
  state.UpdateCachedTracks(batch, &pinned_event_ids);
  was_compacted_ = state.was_compacted;
  if (state.ShouldFlushTimeline(flush_mode)) {
    state.FlushTimeline(timeline, *this);
  }
}

void DataProvider::FinishIncrementalLoad() { incremental_state_.reset(); }

// Processes a vector of TraceEvent structs.
// This function is independent of Emscripten types.
void DataProvider::ProcessTraceEvents(const ParsedTraceEvents& parsed_events,
                                      Timeline& timeline) {
  incremental_state_.reset();
  was_compacted_ = false;
  last_load_was_refetch_ = false;
  timeline.set_mpmd_pipeline_view_enabled(parsed_events.mpmd_pipeline_view);
  timeline.set_time_axis_unit(parsed_events.time_axis_unit);
  if (parsed_events.flame_events.empty() &&
      parsed_events.counter_events.empty() &&
      parsed_events.flow_events.empty() && !HasKnownTracks()) {
    timeline.SetTimelineData({});
    timeline.set_fetched_data_time_range(TimeRange::Zero());
    timeline.SetVisibleRange(TimeRange::Zero());
    return;
  }

  TraceInformation trace_info;
  trace_info.is_mpmd = parsed_events.mpmd_pipeline_view;
  trace_info.process_names = process_names_;
  trace_info.thread_names = thread_names_;
  trace_info.process_sort_indices = process_sort_indices_;
  trace_info.thread_sort_indices = thread_sort_indices_;
  trace_info.xla_modules_tids = xla_modules_tids_;
  trace_info.async_processes_by_events = known_async_processes_;

  for (const TraceEvent& event : parsed_events.flame_events) {
    switch (event.ph) {
      case Phase::kMetadata:
        HandleMetadataEvent(event, trace_info);
        break;
      case Phase::kComplete:
      case Phase::kDurationBegin:
      case Phase::kDurationEnd:
      case Phase::kInstant:
      case Phase::kInstantDeprecated:
        HandleCompleteEvent(event, trace_info);
        break;
      default:
        // Ignore other event types.
        // TODO: b/444013042 - Check the backend to confirm if we need to handle
        // more types in the future.
        break;
    }
  }

  MergeProcessSortIndices(trace_info);

  absl::btree_map<int, int> flow_category_counts;
  for (const TraceEvent& event : parsed_events.flow_events) {
    HandleFlowEvent(event, trace_info, flow_category_counts);
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
      if (!event->is_async && event->tid < kSyntheticOpenSpanTidBit) {
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
      if (!event->is_async) {
        trace_info.events_by_pid_tid[event->pid].try_emplace(event->tid);
      }
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

  const std::vector<ProcessId> sorted_pids = GetSortedProcessIds(
      trace_info, known_threads_, known_counters_, known_async_tracks_);

  const bool was_initial_overview_load =
      (timeline.visible_range() == TimeRange::Zero() &&
       !parsed_events.visible_range_from_url.has_value());

  timeline.UpdateTimelineData([&]() {
    return CreateTimelineData(
        trace_info, sorted_pids, GetTop5FlowCategories(flow_category_counts),
        time_bounds, expanded_states, timeline.GetPalette(),
        max_observed_levels_, known_threads_, known_counters_,
        known_async_tracks_);
  });

  UpdateTimelineTimeRanges(timeline, time_bounds, HasKnownTracks(),
                           parsed_events.visible_range_from_url,
                           parsed_events.full_timespan);

  if (was_initial_overview_load ||
      (timeline.data_time_range().duration() > 0 &&
       timeline.last_fetch_request_range().duration() >=
           0.9 * timeline.data_time_range().duration())) {
    timeline.ClearSavedFullOverview();
    timeline.MarkCurrentDataAsFullOverview();
  }
}

const std::vector<int>& DataProvider::GetFlowCategories() const {
  return present_flow_categories_;
}

absl::flat_hash_map<ProcessId, std::string> DataProvider::GetProcessMappings()
    const {
  absl::flat_hash_map<ProcessId, std::string> map;
  for (const auto& [pid, process_name] : process_names_) {
    std::string host_part = process_name.substr(0, process_name.find(' '));
    if (!host_part.empty()) {
      map[pid] = host_part;
    }
  }
  return map;
}

namespace internal {
void AppendTraceEventForTesting(const TraceEvent* event, int level,
                                FlameChartTimelineData& data) {
  TimeBounds bounds;
  TraceInformation trace_info;
  size_t module_cursor = 0;
  Microseconds last_module_lookup_ts =
      std::numeric_limits<Microseconds>::lowest();
  AppendEventToTimelineData(event, level, data, bounds, trace_info,
                            /*thread_name=*/"", /*module_src=*/{},
                            module_cursor, last_module_lookup_ts);
}
}  // namespace internal

}  // namespace traceviewer
