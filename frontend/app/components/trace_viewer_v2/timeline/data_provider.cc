#include "xprof/frontend/app/components/trace_viewer_v2/timeline/data_provider.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/btree_map.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "third_party/dear_imgui/imgui.h"
#include "re2/re2.h"
#include "tsl/profiler/lib/context_types.h"
#include "xprof/frontend/app/components/trace_viewer_v2/color/colors.h"
#include "xprof/frontend/app/components/trace_viewer_v2/helper/time_formatter.h"
#include "xprof/frontend/app/components/trace_viewer_v2/timeline/time_range.h"
#include "xprof/frontend/app/components/trace_viewer_v2/timeline/timeline.h"
#include "xprof/frontend/app/components/trace_viewer_v2/trace_helper/trace_event.h"
#include "xprof/frontend/app/components/trace_viewer_v2/trace_helper/trace_event_tree.h"
#include "util/gtl/comparator.h"

namespace traceviewer {

namespace {

// The nesting level of a process group in the flame chart.
constexpr int kProcessNestingLevel = 0;
// The nesting level of a thread group in the flame chart.
constexpr int kThreadNestingLevel = 1;

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
  absl::flat_hash_map<ProcessId, std::string> process_names;
  absl::flat_hash_map<ProcessId, uint32_t> process_sort_indices;
  absl::btree_map<std::string, std::vector<const TraceEvent*>>
      flow_events_by_id;
};

std::string GetDefaultThreadName(ThreadId tid) {
  return absl::StrCat("Thread ", tid);
}

std::string GetDefaultProcessName(ProcessId pid) {
  return absl::StrCat("Process ", pid);
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
                         TraceInformation& trace_info) {
  if (event.name == kThreadName) {
    trace_info.thread_names[{event.pid, event.tid}] =
        GetNameWithDefault(event, GetDefaultThreadName(event.tid));
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
  absl::c_stable_sort(
      sorted_flow_categories,
      gtl::ChainComparators(gtl::OrderBySecondGreater(), gtl::OrderByFirst()));
  std::vector<int> top_5_flow_categories;
  for (int i = 0; i < std::min<size_t>(sorted_flow_categories.size(), 5); ++i) {
    top_5_flow_categories.push_back(sorted_flow_categories[i].first);
  }
  return top_5_flow_categories;
}

ImU32 GetFlowColorForCategory(tsl::profiler::ContextType category,
                              const std::vector<int>& top_5_flow_categories) {
  if (category == tsl::profiler::ContextType::kGeneric) {
    return kRed80;
  }

  constexpr ImU32 top_5_colors[] = {kOrange80, kYellow80, kGreen80, kBlue80,
                                    kCyan80};
  const size_t loop_limit =
      std::min(top_5_flow_categories.size(), std::size(top_5_colors));

  for (size_t i = 0; i < loop_limit; ++i) {
    if (static_cast<int>(category) == top_5_flow_categories[i]) {
      return top_5_colors[i];
    }
  }
  return kPurple80;
}

// Returns the flame chart level of the given event.
int GetEventFlameChartLevel(
    const TraceEvent* e,
    const absl::btree_map<std::pair<ProcessId, ThreadId>, ThreadLevelInfo>&
        thread_levels,
    const FlameChartTimelineData& data) {
  auto it = thread_levels.find({e->pid, e->tid});
  if (it == thread_levels.end()) return 0;
  int start = it->second.start_level;
  int end = it->second.end_level;

  // Search from deepest level up
  for (int lvl = end - 1; lvl >= start; --lvl) {
    const auto& indices = data.events_by_level[lvl];
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
  return start;  // Default to thread top
}

void GenerateFlowLines(const TraceInformation& trace_info,
                       const absl::btree_map<std::pair<ProcessId, ThreadId>,
                                             ThreadLevelInfo>& thread_levels,
                       const std::vector<int>& top_5_flow_categories,
                       FlameChartTimelineData& data, TimeBounds& bounds) {
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
      const ImU32 flow_color = GetFlowColorForCategory(
          u->category, top_5_flow_categories);  // Use flow category for color
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

// Appends the given nodes (an array of trees) to the data, starting at the
// given level. Returns the maximum level of the nodes.
int AppendNodesAtLevel(absl::Span<const std::unique_ptr<TraceEventNode>> nodes,
                       int current_level, FlameChartTimelineData& data,
                       TimeBounds& bounds, const TraceInformation& trace_info,
                       absl::string_view thread_name) {
  int max_level = current_level;

  for (const std::unique_ptr<TraceEventNode>& node : nodes) {
    const TraceEvent* event = node->event;

    data.entry_start_times.push_back(event->ts);
    data.entry_total_times.push_back(event->dur);
    data.entry_levels.push_back(current_level);
    data.entry_names.push_back(event->name);
    data.entry_event_ids.push_back(event->event_id);
    data.entry_pids.push_back(event->pid);
    data.entry_tids.push_back(event->tid);

    auto cur_args = event->args;
    bool is_xla_ops_thread = thread_name == kXlaOps;
    bool is_data_motion_layer =
        thread_name == kComputeUtilization ||
        thread_name == kDataMotionLayersUtilization;
    bool has_hlo_in_args =
        event->args.count(std::string(kHloOp)) > 0 &&
        event->args.count(std::string(kHloModule)) > 0;
    if (is_xla_ops_thread || is_data_motion_layer || has_hlo_in_args) {
      std::string hlo_module_str = std::string(kHloModuleDefault);
      auto it_hlo_module = event->args.find(std::string(kHloModule));
      if (it_hlo_module != event->args.end()) {
        const std::string& hlo_module_name = it_hlo_module->second;
        auto it_hlo_module_id = event->args.find(std::string(kHloModuleId));
        auto it_program_id = event->args.find(std::string(kProgramId));
        if (it_hlo_module_id != event->args.end()) {
          hlo_module_str =
              absl::StrCat(hlo_module_name, "(", it_hlo_module_id->second, ")");
        } else if (it_program_id != event->args.end()) {
          hlo_module_str =
              absl::StrCat(hlo_module_name, "(", it_program_id->second, ")");
        } else {
          auto it_kernel_details =
              event->args.find(std::string(kKernelDetails));
          if (it_kernel_details != event->args.end()) {
            std::string module_id;
            if (RE2::PartialMatch(it_kernel_details->second, kModuleRegex,
                                  &module_id)) {
              hlo_module_str =
                  absl::StrCat(hlo_module_name, "(", module_id, ")");
            } else {
              hlo_module_str = hlo_module_name;
            }
          } else {
            hlo_module_str = hlo_module_name;
          }
        }
      } else {
        // search in "XLA Modules"
        bool hlo_module_found = false;
        for (auto const& [pid_tid, name] : trace_info.thread_names) {
          if (pid_tid.first == event->pid && name == kXlaModules) {
            auto it_events = trace_info.events_by_pid_tid.find(event->pid);
            if (it_events != trace_info.events_by_pid_tid.end()) {
              auto it_thread_events = it_events->second.find(pid_tid.second);
              if (it_thread_events != it_events->second.end()) {
                for (const TraceEvent* module_event :
                     it_thread_events->second) {
                  if (module_event->ts <= event->ts &&
                      module_event->ts + module_event->dur >= event->ts) {
                    hlo_module_str = module_event->name;
                    hlo_module_found = true;
                    break;
                  }
                }
              }
            }
          }
          if (hlo_module_found) break;
        }
      }
      cur_args[std::string(kHloModule)] = hlo_module_str;
    } else {
      cur_args[std::string(kHloModule)] = std::string(kHloModuleDefault);
    }
    data.entry_args.push_back(cur_args);

    bounds.min = std::min(bounds.min, event->ts);
    bounds.max = std::max(bounds.max, event->ts + event->dur);

    if (!node->children.empty()) {
      int child_max_level =
          AppendNodesAtLevel(node->children, current_level + 1, data, bounds,
                             trace_info, thread_name);
      max_level = std::max(max_level, child_max_level);
    }
  }

  return max_level;
}

void PopulateThreadTrack(ProcessId pid, ThreadId tid,
                         absl::Span<const TraceEvent* const> events,
                         const TraceInformation& trace_info, int& current_level,
                         FlameChartTimelineData& data, TimeBounds& bounds,
                         absl::btree_map<std::pair<ProcessId, ThreadId>,
                                         ThreadLevelInfo>& thread_levels,
                         bool expand_group) {
  const auto it = trace_info.thread_names.find({pid, tid});
  const std::string thread_group_name = it == trace_info.thread_names.end()
                                            ? GetDefaultThreadName(tid)
                                            : it->second;

  data.groups.push_back({.name = thread_group_name,
                         .start_level = current_level,
                         .nesting_level = kThreadNestingLevel,
                         .expanded = expand_group});

  int start_level = current_level;

  TraceEventTree event_tree = BuildTree(events);

  // Get the maximum level index used by events in this thread.
  int max_level = AppendNodesAtLevel(event_tree.roots, current_level, data,
                                     bounds, trace_info, thread_group_name);

  current_level = max_level + 1;
  thread_levels[{pid, tid}] = {start_level, current_level};
}

void PopulateCounterTrack(ProcessId pid, const std::string& name,
                          absl::Span<const CounterEvent* const> events,
                          const TraceInformation& trace_info,
                          int& current_level, FlameChartTimelineData& data,
                          TimeBounds& bounds, bool expand_group) {
  Group group;
  group.type = Group::Type::kCounter;
  group.name = name;
  group.nesting_level = kThreadNestingLevel;
  group.start_level = current_level;
  group.expanded = expand_group;

  size_t total_entries = 0;
  // The number of counter events per counter track won't be too large, so
  // it's fine to iterate twice to reserve vector capacity.
  for (const CounterEvent* event : events) {
    total_entries += event->timestamps.size();
  }

  CounterData counter_data;
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

void PopulateProcessTrack(ProcessId pid, const TraceInformation& trace_info,
                          int& current_level, FlameChartTimelineData& data,
                          TimeBounds& bounds,
                          absl::btree_map<std::pair<ProcessId, ThreadId>,
                                          ThreadLevelInfo>& thread_levels,
                          bool expand_group) {
  const auto it_events = trace_info.events_by_pid_tid.find(pid);
  const bool has_events = it_events != trace_info.events_by_pid_tid.end() &&
                          !it_events->second.empty();

  const auto it_counters = trace_info.counters_by_pid_name.find(pid);
  const bool has_counters =
      it_counters != trace_info.counters_by_pid_name.end() &&
      !it_counters->second.empty();

  if (!has_events && !has_counters) {
    // No events or counters for this process, so skip this process group.
    return;
  }

  const auto it = trace_info.process_names.find(pid);
  const std::string process_group_name = it == trace_info.process_names.end()
                                             ? GetDefaultProcessName(pid)
                                             : it->second;
  data.groups.push_back({.name = process_group_name,
                         .start_level = current_level,
                         .nesting_level = kProcessNestingLevel,
                         .expanded = expand_group});

  if (has_events) {
    for (const auto& [tid, events] : it_events->second) {
      PopulateThreadTrack(pid, tid, events, trace_info, current_level, data,
                          bounds, thread_levels, expand_group);
    }
  }

  if (has_counters) {
    for (const auto& [name, events] : it_counters->second) {
      PopulateCounterTrack(pid, name, events, trace_info, current_level, data,
                           bounds, expand_group);
    }
  }
}

std::vector<ProcessId> GetSortedProcessIds(const TraceInformation& trace_info) {
  std::vector<ProcessId> pids;
  pids.reserve(trace_info.process_names.size());
  for (const auto& [pid, _] : trace_info.process_names) {
    pids.push_back(pid);
  }

  absl::c_stable_sort(pids, [&](ProcessId a, ProcessId b) {
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

FlameChartTimelineData CreateTimelineData(
    const TraceInformation& trace_info,
    const std::vector<int>& top_5_flow_categories, TimeBounds& bounds) {
  FlameChartTimelineData data;
  int current_level = 0;
  absl::btree_map<std::pair<ProcessId, ThreadId>, ThreadLevelInfo>
      thread_levels;

  bool first_process = true;
  for (const ProcessId pid : GetSortedProcessIds(trace_info)) {
    PopulateProcessTrack(pid, trace_info, current_level, data, bounds,
                         thread_levels, first_process);
    first_process = false;
  }

  data.events_by_level.resize(current_level);
  for (int i = 0; i < data.entry_levels.size(); ++i) {
    data.events_by_level[data.entry_levels[i]].push_back(i);
  }

  GenerateFlowLines(trace_info, thread_levels, top_5_flow_categories, data,
                    bounds);
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
    timeline.set_timeline_data({});
    timeline.set_fetched_data_time_range(TimeRange::Zero());
    timeline.SetVisibleRange(TimeRange::Zero());
    return;
  }

  timeline.set_mpmd_pipeline_view_enabled(parsed_events.mpmd_pipeline_view);

  TraceInformation trace_info;
  for (const auto& event : parsed_events.flame_events) {
    switch (event.ph) {
      case Phase::kMetadata:
        HandleMetadataEvent(event, trace_info);
        break;
      case Phase::kComplete:
        HandleCompleteEvent(event, trace_info);
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
    HandleCounterEvent(event, trace_info);
  }
  for (const auto& [pid, _] : trace_info.counters_by_pid_name) {
    trace_info.process_names.try_emplace(pid, GetDefaultProcessName(pid));
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
      absl::c_stable_sort(
          events, gtl::ChainComparators(
                      gtl::OrderBy([](const TraceEvent* e) { return e->ts; }),
                      gtl::OrderBy([](const TraceEvent* e) { return e->dur; },
                                   gtl::Greater())));
    }
  }

  for (auto& [id, events] : trace_info.flow_events_by_id) {
    absl::c_stable_sort(
        events, gtl::OrderBy([](const TraceEvent* e) { return e->ts; }));
  }

  for (auto& [pid, counters_by_name] : trace_info.counters_by_pid_name) {
    for (auto& [name, events] : counters_by_name) {
      absl::c_stable_sort(
          events, gtl::OrderBy([](const CounterEvent* e) {
            return e->timestamps.empty()
                       ? std::numeric_limits<Microseconds>::max()
                       : e->timestamps.front();
          }));
    }
  }

  TimeBounds time_bounds;

  timeline.set_timeline_data(CreateTimelineData(
      trace_info, GetTop5FlowCategories(flow_category_counts), time_bounds));

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

      timeline.SetVisibleRange({start, end});
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

const std::vector<int>& DataProvider::GetFlowCategories() const {
  return present_flow_categories_;
}

}  // namespace traceviewer
