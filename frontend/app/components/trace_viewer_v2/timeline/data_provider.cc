#include "frontend/app/components/trace_viewer_v2/timeline/data_provider.h"

#include <algorithm>
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
#include "frontend/app/components/trace_viewer_v2/trace_helper/proto_event_ref.h"
#include "frontend/app/components/trace_viewer_v2/trace_helper/trace_event.h"
#include "frontend/app/components/trace_viewer_v2/trace_helper/trace_event_parser_core.h"
#include "frontend/app/components/trace_viewer_v2/trace_helper/trace_event_tree.h"
#include "plugin/xprof/protobuf/trace_data_response.pb.h"

namespace traceviewer {

namespace {

struct GroupKey {
  int nesting_level;
  absl::string_view name;
  absl::string_view parent_name;

  bool operator<(const GroupKey& other) const {
    return std::tie(nesting_level, name, parent_name) <
           std::tie(other.nesting_level, other.name, other.parent_name);
  }
};

bool GetExpandedState(int nesting_level, absl::string_view name,
                      absl::string_view parent_name, bool default_expanded,
                      const absl::btree_map<GroupKey, bool>& expanded_states) {
  if (auto it_state = expanded_states.find({nesting_level, name, parent_name});
      it_state != expanded_states.end()) {
    return it_state->second;
  }
  return default_expanded;
}

absl::btree_map<GroupKey, bool> GetRestoredExpandedStates(
    const std::vector<Group>& groups) {
  absl::btree_map<GroupKey, bool> expanded_states;
  absl::string_view current_process_name;
  for (const auto& group : groups) {
    if (group.nesting_level == kProcessNestingLevel) {
      current_process_name = group.name;
      expanded_states[{kProcessNestingLevel, group.name, ""}] = group.expanded;
    } else {
      expanded_states[{group.nesting_level, group.name, current_process_name}] =
          group.expanded;
    }
  }
  return expanded_states;
}

class TableInterner {
 public:
  TableInterner() {
    pool_.push_back("");
    map_[""] = 0;
  }

  void SetInitialPool(std::vector<std::string> initial_pool) {
    if (!initial_pool.empty()) {
      if (initial_pool[0] != "") {
        pool_.clear();
        pool_.push_back("");
        for (auto& s : initial_pool) {
          pool_.push_back(std::move(s));
        }
      } else {
        pool_ = std::move(initial_pool);
      }
      map_.clear();
      for (uint32_t i = 0; i < pool_.size(); ++i) {
        map_[pool_[i]] = i;
      }
    }
  }

  uint32_t Intern(absl::string_view str) {
    if (str.empty()) return 0;
    auto it = map_.find(str);
    if (it != map_.end()) {
      return it->second;
    }
    uint32_t index = pool_.size();
    pool_.push_back(std::string(str));
    map_[pool_.back()] = index;
    return index;
  }

  std::vector<std::string> TakePool() { return std::move(pool_); }

 private:
  absl::flat_hash_map<std::string, uint32_t> map_;
  std::vector<std::string> pool_;
};

template <typename EventT>
struct TraceInformation {
  // The EventT objects pointed to must outlive this TraceInformation instance.
  absl::btree_map<ProcessId,
                  absl::btree_map<ThreadId, std::vector<const EventT*>>>
      events_by_pid_tid;
  absl::btree_map<
      ProcessId, absl::btree_map<std::string, std::vector<const CounterEvent*>>>
      counters_by_pid_name;
  absl::btree_map<std::pair<ProcessId, ThreadId>, std::string> thread_names;
  absl::flat_hash_map<ProcessId, std::string> process_names;
  absl::flat_hash_map<ProcessId, uint32_t> process_sort_indices;
  absl::btree_map<uint64_t, std::vector<const EventT*>> flow_events_by_id;
  absl::flat_hash_map<ProcessId, ThreadId> xla_modules_tids;
  absl::flat_hash_set<ProcessId> async_processes_by_events;
  TableInterner name_interner;
  TableInterner module_interner;
  TableInterner op_interner;
};

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

  if (auto it_events = trace_info.events_by_pid_tid.find(pid);
      it_events != trace_info.events_by_pid_tid.end()) {
    for (const auto& [tid, _] : it_events->second) {
      if (auto it_thread = trace_info.thread_names.find({pid, tid});
          it_thread != trace_info.thread_names.end()) {
        absl::string_view thread_name = it_thread->second;
        if (absl::EqualsIgnoreCase(thread_name, kAsyncXlaOps)) {
          return 2;
        }
        if (absl::StrContainsIgnoreCase(thread_name, kDma)) {
          is_priority_1 = true;
        }
      }
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

// Extracts the name from event.args. If not found or empty, returns the
// provided default name.
std::string GetNameWithDefault(const TraceEvent& event,
                               absl::string_view default_name) {
  const auto it = event.args.find(std::string(kName));
  if (it != event.args.end() && !it->second.empty()) {
    return it->second;
  }
  return std::string(default_name);
}

std::string GetNameWithDefault(const ProtoEventRef& event,
                               absl::string_view default_name) {
  absl::string_view name = event.metadata_name;
  if (!name.empty()) {
    return std::string(name);
  }
  name = event.GetArg(kName);
  if (!name.empty()) {
    return std::string(name);
  }
  return std::string(default_name);
}

// Handles a metadata event, extracting and storing metadata such as
// thread names, process names, etc.
// TODO: b/439791754 - Handle sort index.
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
    const std::string name =
        GetNameWithDefault(event, GetDefaultThreadName(event.tid));
    trace_info.thread_names[{event.pid, event.tid}] = name;
    if (name == kXlaModules) {
      trace_info.xla_modules_tids[event.pid] = event.tid;
    }
  } else if (event.name == kProcessName) {
    trace_info.process_names[event.pid] =
        GetNameWithDefault(event, GetDefaultProcessName(event.pid));
  } else if (event.name == kProcessSortIndex) {
    if (auto it = event.args.find(std::string(kSortIndex));
        it != event.args.end()) {
      double sort_index_double;
      if (absl::SimpleAtod(it->second, &sort_index_double)) {
        trace_info.process_sort_indices[event.pid] =
            static_cast<uint32_t>(sort_index_double);
      }
    }
  }
}

void HandleMetadataEvent(const ProtoEventRef& event,
                         TraceInformation<ProtoEventRef>& trace_info) {
  if (event.name == kThreadName) {
    const std::string name =
        GetNameWithDefault(event, GetDefaultThreadName(event.tid));
    trace_info.thread_names[{event.pid, event.tid}] = name;
    if (name == kXlaModules) {
      trace_info.xla_modules_tids[event.pid] = event.tid;
    }
  } else if (event.name == kProcessName) {
    trace_info.process_names[event.pid] =
        GetNameWithDefault(event, GetDefaultProcessName(event.pid));
  } else if (event.name == kProcessSortIndex) {
    trace_info.process_sort_indices[event.pid] = event.sort_index;
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
template <typename EventT>
void HandleCompleteEvent(const EventT& event,
                         TraceInformation<EventT>& trace_info) {
  trace_info.events_by_pid_tid[event.pid][event.tid].push_back(&event);
  if (event.is_async) {
    trace_info.async_processes_by_events.insert(event.pid);
  }
}

template <typename EventT>
void HandleFlowEvent(const EventT& event, TraceInformation<EventT>& trace_info,
                     absl::btree_map<int, int>& category_counts) {
  if (event.flow_id != 0) {
    trace_info.flow_events_by_id[event.flow_id].push_back(&event);
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
template <typename EventT>
void HandleCounterEvent(const CounterEvent& event,
                        TraceInformation<EventT>& trace_info) {
  trace_info.counters_by_pid_name[event.pid][event.name].push_back(&event);
}

struct TimeBounds {
  Microseconds min = std::numeric_limits<Microseconds>::max();
  Microseconds max = std::numeric_limits<Microseconds>::min();
};

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

template <typename EventT>
std::pair<int, int> GetEventFlameChartLevelAndIndex(
    const EventT* e,
    const absl::btree_map<std::pair<ProcessId, ThreadId>, ThreadLevelInfo>&
        thread_levels,
    const FlameChartTimelineData& data) {
  auto it = thread_levels.find({e->pid, e->tid});
  if (it == thread_levels.end()) return 0;
  int start = it->second.start_level;
  int end = it->second.end_level;

  // Search from deepest level up
  for (int lvl = end - 1; lvl >= start; --lvl) {
    absl::Span<const int> indices = data.get_level_events(lvl);
    // Binary search for event covering e->ts
    // events are likely sorted by start time.
    auto it_idx = std::upper_bound(indices.begin(), indices.end(), e->ts,
                                   [&](Microseconds ts, int idx) {
                                     return ts < data.entry_start_times[idx];
                                   });

    // it_idx points to first event starting AFTER e->ts.
    // Check the one before it.
    if (it_idx != indices.begin()) {
      int idx = *std::prev(it_idx);
      if (data.entry_start_times[idx] + data.entry_total_times[idx] >= e->ts) {
        return lvl;
      }
    }
  }
  return {start, -1};
}

template <typename EventT>
int GetEventFlameChartLevel(
    const EventT* e,
    const absl::btree_map<std::pair<ProcessId, ThreadId>, ThreadLevelInfo>&
        thread_levels,
    const FlameChartTimelineData& data) {
  return GetEventFlameChartLevelAndIndex(e, thread_levels, data).first;
}

template <typename EventT>
void GenerateFlowLines(const TraceInformation<EventT>& trace_info,
                       const absl::btree_map<std::pair<ProcessId, ThreadId>,
                                             ThreadLevelInfo>& thread_levels,
                       absl::Span<const int> top_5_flow_categories,
                       FlameChartTimelineData& data, TimeBounds& bounds,
                       const ColorPalette& palette) {
  for (const auto& [id, flow_events] : trace_info.flow_events_by_id) {
    for (const TraceEvent* event : flow_events) {
      std::vector<std::string>& event_flow_ids =
          data.flow_ids_by_event_id[event->event_id];
      if (event_flow_ids.empty() || event_flow_ids.back() != id) {
        event_flow_ids.push_back(id);
      }
    }

    for (size_t i = 0; i < flow_events.size() - 1; ++i) {
      const EventT* u = flow_events[i];
      const EventT* v = flow_events[i + 1];
      const ImU32 flow_color =
          GetFlowColorForCategory(u->category, top_5_flow_categories,
                                  palette);  // Use flow category for color
      FlowLine flow_line{
          .source_ts = u->ts,
          .target_ts = v->ts,
          .source_level = GetEventFlameChartLevel(u, thread_levels, data),
          .target_level = GetEventFlameChartLevel(v, thread_levels, data),
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

uint64_t GetEventUid(const TraceEvent* event) {
  uint64_t uid = std::numeric_limits<uint64_t>::max();
  if (auto it_uid = event->args.find("uid"); it_uid != event->args.end()) {
    if (!absl::SimpleAtoi(it_uid->second, &uid)) {
      uid = std::numeric_limits<uint64_t>::max();
    }
  }
  return uid;
}

uint64_t GetEventUid(const ProtoEventRef* event) { return event->uid; }

absl::string_view GetEventArg(const TraceEvent* event, absl::string_view key) {
  auto it = event->args.find(std::string(key));
  if (it != event->args.end()) {
    return it->second;
  }
  return "";
}

absl::string_view GetEventArg(const ProtoEventRef* event,
                              absl::string_view key) {
  return event->GetArg(key);
}

template <typename EventT>
void AppendEventToTimelineData(const EventT* event, int level,
                               Microseconds self_time,
                               FlameChartTimelineData& data, TimeBounds& bounds,
                               TraceInformation<EventT>& trace_info,
                               absl::string_view thread_name) {
  static const absl::NoDestructor<std::string> kHloOpStr(kHloOp);
  static const absl::NoDestructor<std::string> kHloModuleStr(kHloModule);
  static const absl::NoDestructor<std::string> kHloModuleDefaultStr(
      kHloModuleDefault);
  static const absl::NoDestructor<std::string> kHloModuleIdStr(kHloModuleId);
  static const absl::NoDestructor<std::string> kProgramIdStr(kProgramId);
  static const absl::NoDestructor<std::string> kKernelDetailsStr(
      kKernelDetails);
  static const absl::NoDestructor<RE2> kModuleRe(kModuleRegex);

  if (level >= std::numeric_limits<uint16_t>::max()) return;
  data.entry_start_times.push_back(event->ts);
  data.entry_total_times.push_back(event->dur);
  data.entry_self_times.push_back(self_time);
  data.entry_levels.push_back(level);
  data.entry_name_indices.push_back(event->name_index);
  data.entry_uids.push_back(GetEventUid(event));

  if (event->flow_id != 0) {
    data.flow_ids_by_event_index[data.entry_start_times.size() - 1] =
        event->flow_id;
  }

  absl::string_view hlo_op = GetEventArg(event, *kHloOpStr);
  absl::string_view hlo_module = GetEventArg(event, *kHloModuleStr);
  bool is_xla_ops_thread = thread_name == kXlaOps;
  bool is_data_motion_layer = thread_name == kComputeUtilization ||
                              thread_name == kDataMotionLayersUtilization;
  bool has_hlo_in_args = !hlo_op.empty() && !hlo_module.empty();

  std::string hlo_op_str(hlo_op);
  std::string hlo_module_str(hlo_module);

  if (is_xla_ops_thread || is_data_motion_layer || has_hlo_in_args) {
    if (is_data_motion_layer) {
      absl::string_view name_arg = GetEventArg(event, "Name");
      if (!name_arg.empty()) {
        hlo_op_str = std::string(name_arg);
      }
    } else if (hlo_op.empty()) {
      hlo_op_str = std::string(event->name);
    }
    if (!hlo_module.empty()) {
      absl::string_view hlo_module_id = GetEventArg(event, *kHloModuleIdStr);
      absl::string_view program_id = GetEventArg(event, *kProgramIdStr);
      if (!hlo_module_id.empty()) {
        hlo_module_str = absl::StrCat(hlo_module, "(", hlo_module_id, ")");
      } else if (!program_id.empty()) {
        hlo_module_str = absl::StrCat(hlo_module, "(", program_id, ")");
      } else {
        absl::string_view kernel_details =
            GetEventArg(event, *kKernelDetailsStr);
        if (!kernel_details.empty()) {
          std::string module_id;
          if (RE2::PartialMatch(kernel_details, *kModuleRe, &module_id)) {
            hlo_module_str = absl::StrCat(hlo_module, "(", module_id, ")");
          } else {
            hlo_module_str = std::string(hlo_module);
          }
        } else {
          hlo_module_str = std::string(hlo_module);
        }
      }
    } else {
      hlo_module_str = *kHloModuleDefaultStr;
      auto it_tid = trace_info.xla_modules_tids.find(event->pid);
      if (it_tid != trace_info.xla_modules_tids.end()) {
        ThreadId tid = it_tid->second;
        auto it_events = trace_info.events_by_pid_tid.find(event->pid);
        if (it_events != trace_info.events_by_pid_tid.end()) {
          auto it_thread_events = it_events->second.find(tid);
          if (it_thread_events != it_events->second.end()) {
            for (const EventT* module_event : it_thread_events->second) {
              if (module_event->ts <= event->ts &&
                  module_event->ts + module_event->dur >= event->ts) {
                hlo_module_str = std::string(module_event->name);
                break;
              }
            }
          }
        }
      }
    }
  } else {
    hlo_module_str = *kHloModuleDefaultStr;
  }
  data.entry_hlo_op_indices.push_back(
      trace_info.op_interner.Intern(hlo_op_str));
  data.entry_hlo_module_indices.push_back(
      trace_info.module_interner.Intern(hlo_module_str));

  bounds.min = std::min(bounds.min, event->ts);
  bounds.max = std::max(bounds.max, event->ts + event->dur);
}

// Appends the given nodes (an array of trees) to the data, starting at the
// given level. Returns the maximum level of the nodes.
template <typename EventT>
int AppendNodesAtLevel(
    absl::Span<const std::unique_ptr<TraceEventNodeT<EventT>>> nodes,
    int current_level, FlameChartTimelineData& data, TimeBounds& bounds,
    TraceInformation<EventT>& trace_info, absl::string_view thread_name) {
  struct StackFrame {
    absl::Span<const std::unique_ptr<TraceEventNodeT<EventT>>> nodes;
    int level;
  };

  int max_level_overall = current_level;
  std::vector<StackFrame> stack;
  if (!nodes.empty()) {
    stack.push_back({nodes, current_level});
  }

  while (!stack.empty()) {
    StackFrame frame = stack.back();
    stack.pop_back();

    int level = frame.level;
    max_level_overall = std::max(max_level_overall, level);

    for (const auto& node : frame.nodes) {
      const EventT* event = node->event;

      AppendEventToTimelineData<EventT>(event, level, node->self_time, data,
                                        bounds, trace_info, thread_name);

      if (!node->children.empty()) {
        stack.push_back({node->children, level + 1});
      }
    }
  }
  return max_level_overall;
}

template <typename EventT>
void PopulateThreadTrackWithPackedLayout(ProcessId pid, ThreadId tid,
                                         absl::Span<const EventT* const> events,
                                         int start_level, int& max_level,
                                         FlameChartTimelineData& data,
                                         TimeBounds& bounds,
                                         TraceInformation<EventT>& trace_info,
                                         const std::string& thread_group_name) {
  std::vector<const EventT*> sorted_events(events.begin(), events.end());
  absl::c_sort(sorted_events,
               [](const EventT* a, const EventT* b) { return a->ts < b->ts; });

  std::vector<Microseconds> row_end_times;
  for (const EventT* event : sorted_events) {
    const Microseconds start = event->ts;
    const Microseconds duration = event->dur;
    const Microseconds end = start + duration;

    int selected_row = -1;
    for (size_t i = 0; i < row_end_times.size(); ++i) {
      if (row_end_times[i] <= start) {
        selected_row = i;
        break;
      }
    }

    if (selected_row == -1) {
      row_end_times.push_back(end);
      selected_row = row_end_times.size() - 1;
    } else {
      row_end_times[selected_row] = end;
    }

    int absolute_level = start_level + selected_row;
    max_level = std::max(max_level, absolute_level);

    AppendEventToTimelineData<EventT>(event, absolute_level, event->dur, data,
                                      bounds, trace_info, thread_group_name);
  }
}

template <typename EventT>
void PopulateThreadTrackWithTreeLayout(ProcessId pid, ThreadId tid,
                                       absl::Span<const EventT* const> events,
                                       int current_level, int& max_level,
                                       FlameChartTimelineData& data,
                                       TimeBounds& bounds,
                                       TraceInformation<EventT>& trace_info,
                                       const std::string& thread_group_name) {
  TraceEventTreeT<EventT> event_tree = BuildTree<EventT>(events);
  max_level = AppendNodesAtLevel<EventT>(event_tree.roots, current_level, data,
                                         bounds, trace_info, thread_group_name);
}

template <typename EventT>
void PopulateThreadTrack(
    ProcessId pid, ThreadId tid, absl::Span<const EventT* const> events,
    TraceInformation<EventT>& trace_info, int& current_level,
    FlameChartTimelineData& data, TimeBounds& bounds,
    absl::btree_map<std::pair<ProcessId, ThreadId>, ThreadLevelInfo>&
        thread_levels,
    const std::string& process_group_name, bool default_expanded,
    const absl::btree_map<GroupKey, bool>& expanded_states,
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

  data.groups.push_back({.name = thread_group_name,
                         .start_level = current_level,
                         .nesting_level = kThreadNestingLevel,
                         .expanded = expanded,
                         .pid = pid,
                         .tid = tid});

  int start_level = current_level;
  int max_level = start_level;

  if (custom_name.has_value()) {
    PopulateThreadTrackWithPackedLayout<EventT>(pid, tid, events, start_level,
                                                max_level, data, bounds,
                                                trace_info, thread_group_name);
  } else {
    PopulateThreadTrackWithTreeLayout<EventT>(pid, tid, events, current_level,
                                              max_level, data, bounds,
                                              trace_info, thread_group_name);
  }

  current_level = max_level + 1;
  thread_levels[{pid, tid}] = {start_level, current_level};

  if (max_level == start_level) {
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
    const absl::btree_map<GroupKey, bool>& expanded_states) {
  Group group;
  group.type = Group::Type::kCounter;
  group.name = name;
  group.nesting_level = kCounterNestingLevel;
  group.start_level = current_level;
  group.pid = pid;

  // Counters always take one level, so force them to be expanded.
  group.expanded = true;

  size_t total_entries = 0;
  // The number of counter events per counter track won't be too large, so
  // it's fine to iterate twice to reserve vector capacity.
  for (const CounterEvent* event : events) {
    total_entries += event->timestamps.size();
  }

  CounterData counter_data;
  if (!events.empty()) {
    counter_data.event_stats = events.front()->event_stats;
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

  data.groups.push_back(std::move(group));

  data.counter_data_by_group_index[data.groups.size() - 1] =
      std::move(counter_data);

  // Increment the level by one for the next group. This will be used for binary
  // search for the visible groups.
  current_level++;
}

template <typename EventT>
void PopulateAsyncProcessTrack(
    ProcessId pid, const std::string& process_group_name,
    TraceInformation<EventT>& trace_info, int& current_level,
    FlameChartTimelineData& data, TimeBounds& bounds,
    absl::btree_map<std::pair<ProcessId, ThreadId>, ThreadLevelInfo>&
        thread_levels,
    bool default_expanded,
    const absl::btree_map<GroupKey, bool>& expanded_states) {
  absl::btree_map<std::string, std::vector<const EventT*>> async_groups;
  absl::btree_map<ThreadId, std::vector<const EventT*>> sync_groups;

  const auto it_events = trace_info.events_by_pid_tid.find(pid);
  if (it_events == trace_info.events_by_pid_tid.end()) return;

  for (const auto& [tid, tid_events] : it_events->second) {
    for (const EventT* event : tid_events) {
      if (event->is_async) {
        async_groups[std::string(event->name)].push_back(event);
      } else {
        sync_groups[tid].push_back(event);
      }
    }
  }

  // Populate named async tracks first.
  // Starting synthetic TIDs at 0x80000000 is generally safe, but if the trace
  // contains very large TIDs (e.g., from a system that uses 64-bit TIDs or
  // just very high values), there is a small risk of collision. Since these are
  // only used internally for grouping, it's likely fine.
  ThreadId next_synthetic_tid = 0x80000000;
  for (const auto& [name, named_events] : async_groups) {
    ThreadId tid =
        named_events.empty() ? next_synthetic_tid++ : named_events[0]->tid;
    PopulateThreadTrack<EventT>(
        pid, tid, absl::Span<const EventT* const>(named_events), trace_info,
        current_level, data, bounds, thread_levels, process_group_name,
        default_expanded, expanded_states, name);
    next_synthetic_tid++;
  }

  // Populate standard thread tracks.
  for (const auto& [tid, events] : sync_groups) {
    PopulateThreadTrack<EventT>(
        pid, tid, absl::Span<const EventT* const>(events), trace_info,
        current_level, data, bounds, thread_levels, process_group_name,
        default_expanded, expanded_states);
  }
}

template <typename EventT>
void PopulateSyncProcessTrack(
    ProcessId pid, const std::string& process_group_name,
    TraceInformation<EventT>& trace_info, int& current_level,
    FlameChartTimelineData& data, TimeBounds& bounds,
    absl::btree_map<std::pair<ProcessId, ThreadId>, ThreadLevelInfo>&
        thread_levels,
    bool default_expanded,
    const absl::btree_map<GroupKey, bool>& expanded_states) {
  const auto it_events = trace_info.events_by_pid_tid.find(pid);
  std::set<ThreadId> tids;
  if (it_events != trace_info.events_by_pid_tid.end()) {
    for (const auto& [tid, _] : it_events->second) {
      tids.insert(tid);
    }
  }

  // Collect tids from thread_names
  for (auto it = trace_info.thread_names.lower_bound({pid, 0});
       it != trace_info.thread_names.end() && it->first.first == pid; ++it) {
    tids.insert(it->first.second);
  }

  for (const auto tid : tids) {
    absl::Span<const EventT* const> events;
    if (it_events != trace_info.events_by_pid_tid.end()) {
      auto it = it_events->second.find(tid);
      if (it != it_events->second.end()) {
        events = absl::Span<const EventT* const>(it->second);
      }
    }
    PopulateThreadTrack<EventT>(pid, tid, events, trace_info, current_level,
                                data, bounds, thread_levels, process_group_name,
                                default_expanded, expanded_states);
  }
}

template <typename EventT>
bool IsAsyncProcess(ProcessId pid, const TraceInformation<EventT>& trace_info) {
  return GetAsyncProcessPriority<EventT>(pid, trace_info) > 0;
}

template <typename EventT>
void PopulateProcessTrack(
    ProcessId pid, TraceInformation<EventT>& trace_info, int& current_level,
    FlameChartTimelineData& data, TimeBounds& bounds,
    absl::btree_map<std::pair<ProcessId, ThreadId>, ThreadLevelInfo>&
        thread_levels,
    bool default_expanded,
    const absl::btree_map<GroupKey, bool>& expanded_states) {
  const auto it_events = trace_info.events_by_pid_tid.find(pid);
  const bool has_events = it_events != trace_info.events_by_pid_tid.end() &&
                          !it_events->second.empty();

  const auto it_counters = trace_info.counters_by_pid_name.find(pid);
  const bool has_counters =
      it_counters != trace_info.counters_by_pid_name.end() &&
      !it_counters->second.empty();

  // Check if any threads exist for this PID in thread_names.
  auto it_thread_names = trace_info.thread_names.lower_bound({pid, 0});
  bool has_named_threads = (it_thread_names != trace_info.thread_names.end() &&
                            it_thread_names->first.first == pid);

  if (!has_events && !has_counters && !has_named_threads) {
    // No events, counters, or named tracks for this process, so skip this
    // process group.
    return;
  }

  const std::string process_group_name = trace_info.process_names.at(pid);

  bool expanded = GetExpandedState(kProcessNestingLevel, process_group_name, "",
                                   default_expanded, expanded_states);

  std::string track_subtitle;

  const size_t separator_pos = process_group_name.find(' ');
  if (separator_pos != std::string::npos) {
    track_subtitle = process_group_name.substr(0, separator_pos);
  }

  data.groups.push_back({.name = process_group_name,
                         .subtitle = std::move(track_subtitle),
                         .start_level = current_level,
                         .nesting_level = kProcessNestingLevel,
                         .expanded = expanded,
                         .pid = pid});

  if (has_events || has_named_threads) {
    bool is_async_process = IsAsyncProcess<EventT>(pid, trace_info);

    if (is_async_process) {
      PopulateAsyncProcessTrack<EventT>(
          pid, process_group_name, trace_info, current_level, data, bounds,
          thread_levels, default_expanded, expanded_states);
    } else {
      PopulateSyncProcessTrack<EventT>(
          pid, process_group_name, trace_info, current_level, data, bounds,
          thread_levels, default_expanded, expanded_states);
    }
  }

  if (has_counters) {
    for (const auto& [name, events] : it_counters->second) {
      PopulateCounterTrack<EventT>(
          pid, name, absl::Span<const CounterEvent* const>(events), trace_info,
          current_level, data, bounds, process_group_name, default_expanded,
          expanded_states);
    }
  }
}

template <typename EventT>
std::vector<ProcessId> GetSortedProcessIds(
    const TraceInformation<EventT>& trace_info) {
  std::vector<ProcessId> pids;
  pids.reserve(trace_info.process_names.size());
  for (const auto& [pid, _] : trace_info.process_names) {
    pids.push_back(pid);
  }

  absl::flat_hash_map<ProcessId, int> async_process_priorities;
  for (const ProcessId pid : pids) {
    async_process_priorities[pid] =
        GetAsyncProcessPriority<EventT>(pid, trace_info);
  }

  absl::c_stable_sort(pids, [&](ProcessId a, ProcessId b) {
    int priority_a = async_process_priorities[a];
    int priority_b = async_process_priorities[b];
    if (priority_a != priority_b) return priority_a > priority_b;

    uint32_t index_a = a;
    if (auto it = trace_info.process_sort_indices.find(a);
        it != trace_info.process_sort_indices.end()) {
      index_a = it->second;
    }
    uint32_t index_b = b;
    if (auto it = trace_info.process_sort_indices.find(b);
        it != trace_info.process_sort_indices.end()) {
      index_b = it->second;
    }
    if (index_a != index_b) return index_a < index_b;
    return a < b;
  });
  return pids;
}

template <typename EventT>
FlameChartTimelineData CreateTimelineData(
    TraceInformation<EventT>& trace_info,
    absl::Span<const int> top_5_flow_categories, TimeBounds& bounds,
    const absl::btree_map<GroupKey, bool>& expanded_states,
    const ColorPalette& palette) {
  FlameChartTimelineData data;
  int current_level = 0;
  absl::btree_map<std::pair<ProcessId, ThreadId>, ThreadLevelInfo>
      thread_levels;

  bool first_process = true;
  for (const ProcessId pid : GetSortedProcessIds<EventT>(trace_info)) {
    PopulateProcessTrack<EventT>(pid, trace_info, current_level, data, bounds,
                                 thread_levels, first_process, expanded_states);
    first_process = false;
  }

  data.level_offsets.assign(current_level + 1, 0);
  for (int i = 0; i < data.entry_levels.size(); ++i) {
    ++data.level_offsets[data.entry_levels[i] + 1];
  }
  std::partial_sum(data.level_offsets.begin(), data.level_offsets.end(),
                   data.level_offsets.begin());
  data.level_event_indices.resize(data.entry_levels.size());
  std::vector<size_t> cursors = data.level_offsets;
  for (int i = 0; i < data.entry_levels.size(); ++i) {
    int level = data.entry_levels[i];
    data.level_event_indices[cursors[level]] = i;
    ++cursors[level];
  }
  for (int i = 0; i < current_level; ++i) {
    absl::Span<int> level_events = data.get_level_events(i);
    if (level_events.empty()) continue;

    // Sort by start time ascending, then duration descending.
    auto cmp_by_start_asc_then_dur_desc = [&](int idx_a, int idx_b) {
      return data.entry_start_times[idx_a] < data.entry_start_times[idx_b] ||
             (data.entry_start_times[idx_a] == data.entry_start_times[idx_b] &&
              data.entry_total_times[idx_a] > data.entry_total_times[idx_b]);
    };

    LOG_IF(WARNING,
           !absl::c_is_sorted(level_events, cmp_by_start_asc_then_dur_desc))
        << "Trace Events not sorted properly for level: " << i;
    absl::c_stable_sort(level_events, cmp_by_start_asc_then_dur_desc);
  }

  GenerateFlowLines<EventT>(trace_info, thread_levels, top_5_flow_categories,
                            data, bounds, palette);

  data.interned_string_pool = trace_info.name_interner.TakePool();
  data.hlo_module_table = trace_info.module_interner.TakePool();
  data.hlo_op_table = trace_info.op_interner.TakePool();
  return data;
}

}  // namespace

// Processes a vector of TraceEvent structs.
// This function is independent of Emscripten types.
void DataProvider::ProcessTraceEvents(const ParsedTraceEvents& parsed_events,
                                      Timeline& timeline) {
  if (parsed_events.flame_events.empty() &&
      parsed_events.counter_events.empty() &&
      parsed_events.flow_events.empty()) {
    timeline.SetTimelineData({});
    timeline.set_fetched_data_time_range(TimeRange::Zero());
    timeline.SetVisibleRange(TimeRange::Zero());
    return;
  }

  timeline.set_mpmd_pipeline_view_enabled(parsed_events.mpmd_pipeline_view);

  TraceInformation<TraceEvent> trace_info;
  trace_info.name_interner.SetInitialPool(parsed_events.interned_string_pool);

  absl::btree_map<int, int> flow_category_counts;
  for (const auto& event : parsed_events.flame_events) {
    switch (event.ph) {
      case Phase::kMetadata:
        HandleMetadataEvent(event, trace_info);
        break;
      case Phase::kComplete:
      case Phase::kInstant:
      case Phase::kInstantDeprecated:
        HandleCompleteEvent<TraceEvent>(event, trace_info);
        if (event.flow_id != 0) {
          HandleFlowEvent<TraceEvent>(event, trace_info, flow_category_counts);
        }
        break;
      case Phase::kFlowStart:
      case Phase::kFlowEnd:
        if (event.flow_id != 0) {
          HandleFlowEvent<TraceEvent>(event, trace_info, flow_category_counts);
        }
        break;
      default:
        // Ignore other event types.
        // TODO: b/444013042 - Check the backend to confirm if we need to handle
        // more types in the future.
        break;
    }
  }

  absl::btree_map<int, int> flow_category_counts;
  for (const auto& event : parsed_events.flow_events) {
    HandleFlowEvent(event, trace_info, flow_category_counts);
  }
  present_flow_categories_.clear();
  for (const auto& pair : flow_category_counts) {
    present_flow_categories_.push_back(pair.first);
  }

  for (const auto& event : parsed_events.counter_events) {
    HandleCounterEvent<TraceEvent>(event, trace_info);
  }
  for (const auto& [pid, _] : trace_info.counters_by_pid_name) {
    trace_info.process_names.try_emplace(pid, GetDefaultProcessName(pid));
  }

  // Ensure all pids from thread_names are registered.
  for (const auto& [key, _] : trace_info.thread_names) {
    trace_info.process_names.try_emplace(key.first,
                                         GetDefaultProcessName(key.first));
  }

  // Ensure all pids/tids from flow events are registered so that thread tracks
  // are created for them, which is required for level calculation.
  for (const auto& [id, events] : trace_info.flow_events_by_id) {
    for (const auto* event : events) {
      trace_info.process_names.try_emplace(event->pid,
                                           GetDefaultProcessName(event->pid));
      trace_info.events_by_pid_tid[event->pid].try_emplace(event->tid);
    }
  }

  // Sort events, first by timestamp (ascending), then by duration
  // (descending).
  // Ensure all processes have a name in process_names.
  for (const auto& [pid, _] : trace_info.events_by_pid_tid) {
    trace_info.process_names.try_emplace(pid, GetDefaultProcessName(pid));
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
    absl::c_stable_sort(events, [](const TraceEvent* a, const TraceEvent* b) {
      return a->ts < b->ts;
    });
  }

  for (auto& [pid, counters_by_name] : trace_info.counters_by_pid_name) {
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

  TimeBounds time_bounds;

  const absl::btree_map<GroupKey, bool> expanded_states =
      GetRestoredExpandedStates(timeline.timeline_data().groups);

  timeline.SetTimelineData(CreateTimelineData<TraceEvent>(
      trace_info, GetTop5FlowCategories(flow_category_counts), time_bounds,
      expanded_states, timeline.GetPalette()));

  process_names_.insert(trace_info.process_names.begin(),
                        trace_info.process_names.end());

  // Don't need to check for max_time because the TimeRange constructor will
  // handle any potential issues with max_time.
  if (time_bounds.min < std::numeric_limits<Microseconds>::max()) {
    timeline.set_fetched_data_time_range({time_bounds.min, time_bounds.max});

    // TODO: b/460265076 - Change the logic here for visible range after
    // we decided how to handle the visible range in url.
    if (parsed_events.visible_range_from_url.has_value()) {
      Microseconds start =
          MillisToMicros(parsed_events.visible_range_from_url->first);
      Microseconds end =
          MillisToMicros(parsed_events.visible_range_from_url->second);

      if (timeline.visible_range() == TimeRange::Zero()) {
        timeline.SetVisibleRange({start, end});
      }
    } else {
      // If the visible range is not zero, we just keep it. This happens when
      // the incremental loading is triggered and we don't want to override the
      // current visible range.
      if (timeline.visible_range() == TimeRange::Zero()) {
        timeline.SetVisibleRange({time_bounds.min, time_bounds.max});
      }
    }
  } else {
    timeline.set_fetched_data_time_range(TimeRange::Zero());
    timeline.SetVisibleRange(TimeRange::Zero());
  }

  if (parsed_events.full_timespan.has_value()) {
    Microseconds start = MillisToMicros(parsed_events.full_timespan->first);
    Microseconds end = MillisToMicros(parsed_events.full_timespan->second);
    timeline.set_data_time_range({start, end});
  } else {
    timeline.set_data_time_range(timeline.fetched_data_time_range());
  }
}

// Processes protobuf TraceDataResponse directly using arena-backed
// ProtoEventRef.
void DataProvider::ProcessTraceEvents(
    const xprof::TraceDataResponse& response, Timeline& timeline,
    std::optional<std::pair<Milliseconds, Milliseconds>>
        visible_range_from_url) {
  std::deque<ProtoEventRef> stable_events;
  std::vector<CounterEvent> counter_events;

  bool mpmd_pipeline_view = false;
  for (const auto& detail : response.details()) {
    if (detail.name() == "mpmd_pipeline_view") {
      mpmd_pipeline_view = detail.value();
    }
  }
  timeline.set_mpmd_pipeline_view_enabled(mpmd_pipeline_view);

  // Unmarshal metadata events
  for (const auto& process : response.metadata().processes()) {
    ProtoEventRef process_ev;
    process_ev.ph = Phase::kMetadata;
    process_ev.pid = process.id();
    process_ev.name = kProcessName;
    process_ev.metadata_name = process.name();
    stable_events.push_back(process_ev);

    if (process.sort_index() != 0) {
      ProtoEventRef process_sort_ev;
      process_sort_ev.ph = Phase::kMetadata;
      process_sort_ev.pid = process.id();
      process_sort_ev.name = kProcessSortIndex;
      process_sort_ev.sort_index = process.sort_index();
      stable_events.push_back(process_sort_ev);
    }

    for (const auto& thread : process.threads()) {
      ProtoEventRef thread_ev;
      thread_ev.ph = Phase::kMetadata;
      thread_ev.pid = process.id();
      thread_ev.tid = thread.id();
      thread_ev.name = kThreadName;
      thread_ev.metadata_name = thread.name();
      stable_events.push_back(thread_ev);
    }
  }

  // Unmarshal complete events
  for (const auto& series : response.complete_events()) {
    const auto& metadata = series.metadata();
    uint64_t current_ts_ps = 0;
    for (int i = 0; i < series.deltas_size(); ++i) {
      current_ts_ps += series.deltas(i);
      ProtoEventRef ev;
      ev.ph = Phase::kComplete;
      ev.pid = metadata.process_id();
      ev.tid = metadata.thread_id();
      ev.ts = current_ts_ps / 1000000.0;
      ev.dur = series.durations(i) / 1000000.0;
      ev.name = response.interned_strings(series.name_refs(i));
      ev.name_index = series.name_refs(i);

      if (i < series.event_metadata_size()) {
        const auto& ev_meta = series.event_metadata(i);
        if (ev_meta.flow_id() != 0) {
          ev.flow_id = ev_meta.flow_id();
          if (ev_meta.flow_category() != 0) {
            ev.category = GetContextTypeFromString(
                response.interned_strings(ev_meta.flow_category()));
          }
        }
        ev.uid = ev_meta.serial();
        ev.group_id = ev_meta.group_id();
      }
      stable_events.push_back(ev);
    }
  }

  // Unmarshal async events
  absl::flat_hash_map<std::pair<ProcessId, uint64_t>, ProtoEventRef>
      open_async_events;
  for (const auto& series : response.async_events()) {
    const auto& metadata = series.metadata();
    uint64_t current_ts_ps = 0;
    for (int i = 0; i < series.deltas_size(); ++i) {
      current_ts_ps += series.deltas(i);
      if (i < series.event_metadata_size()) {
        const auto& ev_meta = series.event_metadata(i);
        if (ev_meta.flow_id() != 0) {
          uint64_t flow_id = ev_meta.flow_id();
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

          ProtoEventRef ev;
          ev.pid = metadata.process_id();
          ev.ts = current_ts_ps / 1000000.0;
          ev.name = response.interned_strings(metadata.name_ref());
          ev.name_index = metadata.name_ref();
          ev.flow_id = flow_id;
          ev.category = category;
          ev.uid = ev_meta.serial();
          ev.group_id = ev_meta.group_id();

          if (dur > 0.0) {
            ev.dur = dur;
            ev.ph = Phase::kComplete;
            ev.is_async = true;
            stable_events.push_back(ev);
          } else {
            auto key = std::make_pair(ev.pid, ev.flow_id);
            auto it = open_async_events.find(key);
            if (it == open_async_events.end()) {
              ev.ph = Phase::kAsyncBegin;
              open_async_events.try_emplace(std::move(key), std::move(ev));
            } else {
              ProtoEventRef& begin_ev = it->second;
              begin_ev.ph = Phase::kComplete;
              begin_ev.is_async = true;
              if (ev.ts > begin_ev.ts) {
                begin_ev.dur = ev.ts - begin_ev.ts;
              }
              stable_events.push_back(begin_ev);
              open_async_events.erase(it);
            }
          }
        }
      }
    }
  }

  // Unmarshal counter events
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
      if (i < series.event_metadata_size()) {
        const auto& ev_meta = series.event_metadata(i);
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
    counter_events.push_back(std::move(ev));
  }

  if (stable_events.empty() && counter_events.empty()) {
    timeline.SetTimelineData({});
    timeline.set_fetched_data_time_range(TimeRange::Zero());
    timeline.SetVisibleRange(TimeRange::Zero());
    return;
  }

  std::vector<std::string> interned_string_pool;
  interned_string_pool.reserve(response.interned_strings_size());
  for (const auto& str : response.interned_strings()) {
    interned_string_pool.push_back(str);
  }

  TraceInformation<ProtoEventRef> trace_info;
  trace_info.name_interner.SetInitialPool(std::move(interned_string_pool));

  absl::btree_map<int, int> flow_category_counts;
  for (const auto& event : stable_events) {
    switch (event.ph) {
      case Phase::kMetadata:
        HandleMetadataEvent(event, trace_info);
        break;
      case Phase::kComplete:
      case Phase::kInstant:
      case Phase::kInstantDeprecated:
        HandleCompleteEvent<ProtoEventRef>(event, trace_info);
        if (event.flow_id != 0) {
          HandleFlowEvent<ProtoEventRef>(event, trace_info,
                                         flow_category_counts);
        }
        break;
      case Phase::kFlowStart:
      case Phase::kFlowEnd:
        if (event.flow_id != 0) {
          HandleFlowEvent<ProtoEventRef>(event, trace_info,
                                         flow_category_counts);
        }
        break;
      default:
        break;
    }
  }

  present_flow_categories_.clear();
  for (const auto& pair : flow_category_counts) {
    present_flow_categories_.push_back(pair.first);
  }

  for (const auto& event : counter_events) {
    HandleCounterEvent<ProtoEventRef>(event, trace_info);
  }
  for (const auto& [pid, _] : trace_info.counters_by_pid_name) {
    trace_info.process_names.try_emplace(pid, GetDefaultProcessName(pid));
  }
  for (const auto& [key, _] : trace_info.thread_names) {
    trace_info.process_names.try_emplace(key.first,
                                         GetDefaultProcessName(key.first));
  }
  for (const auto& [id, events] : trace_info.flow_events_by_id) {
    for (const auto* event : events) {
      trace_info.process_names.try_emplace(event->pid,
                                           GetDefaultProcessName(event->pid));
      trace_info.events_by_pid_tid[event->pid].try_emplace(event->tid);
    }
  }
  for (const auto& [pid, _] : trace_info.events_by_pid_tid) {
    trace_info.process_names.try_emplace(pid, GetDefaultProcessName(pid));
  }

  for (auto& [pid, events_by_tid] : trace_info.events_by_pid_tid) {
    for (auto& [tid, events] : events_by_tid) {
      absl::c_stable_sort(events,
                          [](const ProtoEventRef* a, const ProtoEventRef* b) {
                            if (a->ts != b->ts) {
                              return a->ts < b->ts;
                            }
                            return a->dur > b->dur;
                          });
    }
  }

  for (auto& [id, events] : trace_info.flow_events_by_id) {
    absl::c_stable_sort(events,
                        [](const ProtoEventRef* a, const ProtoEventRef* b) {
                          return a->ts < b->ts;
                        });
  }

  for (auto& [pid, counters_by_name] : trace_info.counters_by_pid_name) {
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

  TimeBounds time_bounds;
  const absl::btree_map<GroupKey, bool> expanded_states =
      GetRestoredExpandedStates(timeline.timeline_data().groups);

  timeline.SetTimelineData(CreateTimelineData<ProtoEventRef>(
      trace_info, GetTop5FlowCategories(flow_category_counts), time_bounds,
      expanded_states, timeline.GetPalette()));

  process_names_.insert(trace_info.process_names.begin(),
                        trace_info.process_names.end());

  if (time_bounds.min < std::numeric_limits<Microseconds>::max()) {
    timeline.set_fetched_data_time_range({time_bounds.min, time_bounds.max});
    if (visible_range_from_url.has_value()) {
      Microseconds start = MillisToMicros(visible_range_from_url->first);
      Microseconds end = MillisToMicros(visible_range_from_url->second);

      if (timeline.visible_range() == TimeRange::Zero()) {
        timeline.SetVisibleRange({start, end});
      }
    } else {
      if (timeline.visible_range() == TimeRange::Zero()) {
        timeline.SetVisibleRange({time_bounds.min, time_bounds.max});
      }
    }
  } else {
    timeline.set_fetched_data_time_range(TimeRange::Zero());
    timeline.SetVisibleRange(TimeRange::Zero());
  }

  if (response.has_full_timespan_start_ps() &&
      response.has_full_timespan_end_ps()) {
    const Milliseconds start_ms = response.full_timespan_start_ps() / 1e9;
    const Milliseconds end_ms = response.full_timespan_end_ps() / 1e9;
    if (start_ms <= end_ms) {
      Microseconds start = MillisToMicros(start_ms);
      Microseconds end = MillisToMicros(end_ms);
      timeline.set_data_time_range({start, end});
    } else {
      timeline.set_data_time_range(timeline.fetched_data_time_range());
    }
  } else {
    timeline.set_data_time_range(timeline.fetched_data_time_range());
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

}  // namespace traceviewer
