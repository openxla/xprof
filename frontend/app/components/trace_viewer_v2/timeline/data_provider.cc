#include "xprof/frontend/app/components/trace_viewer_v2/timeline/data_provider.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/btree_map.h"
#include "absl/container/flat_hash_map.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
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
  absl::btree_map<ProcessId, std::string> process_names;
  absl::flat_hash_map<ProcessId, uint32_t> process_sort_indices;
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

// Appends the given nodes (an array of trees) to the data, starting at the
// given level. Returns the maximum level of the nodes.
int AppendNodesAtLevel(absl::Span<const std::unique_ptr<TraceEventNode>> nodes,
                       int current_level, FlameChartTimelineData& data,
                       TimeBounds& bounds) {
  int max_level = current_level;

  for (const std::unique_ptr<TraceEventNode>& node : nodes) {
    const TraceEvent* event = node->event;

    data.entry_start_times.push_back(event->ts);
    data.entry_total_times.push_back(event->dur);
    data.entry_levels.push_back(current_level);
    data.entry_names.push_back(event->name);

    bounds.min = std::min(bounds.min, event->ts);
    bounds.max = std::max(bounds.max, event->ts + event->dur);

    if (!node->children.empty()) {
      int child_max_level =
          AppendNodesAtLevel(node->children, current_level + 1, data, bounds);
      max_level = std::max(max_level, child_max_level);
    }
  }

  return max_level;
}

void PopulateThreadTrack(ProcessId pid, ThreadId tid,
                         absl::Span<const TraceEvent* const> events,
                         const TraceInformation& trace_info, int& current_level,
                         FlameChartTimelineData& data, TimeBounds& bounds) {
  const auto it = trace_info.thread_names.find({pid, tid});
  const std::string thread_group_name = it == trace_info.thread_names.end()
                                            ? GetDefaultThreadName(tid)
                                            : it->second;
  data.groups.push_back({.name = thread_group_name,
                         .start_level = current_level,
                         .nesting_level = kThreadNestingLevel});

  TraceEventTree event_tree = BuildTree(events);

  // Get the maximum level index used by events in this thread.
  int max_level =
      AppendNodesAtLevel(event_tree.roots, current_level, data, bounds);

  current_level = max_level + 1;
}

void PopulateCounterTrack(ProcessId pid, const std::string& name,
                          absl::Span<const CounterEvent* const> events,
                          const TraceInformation& trace_info,
                          int& current_level, FlameChartTimelineData& data,
                          TimeBounds& bounds) {
  Group group;
  group.type = Group::Type::kCounter;
  group.name = name;
  group.nesting_level = kThreadNestingLevel;
  group.start_level = current_level;

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
                          TimeBounds& bounds) {
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
                         .nesting_level = kProcessNestingLevel});

  if (has_events) {
    for (const auto& [tid, events] : it_events->second) {
      PopulateThreadTrack(pid, tid, events, trace_info, current_level, data,
                          bounds);
    }
  }

  if (has_counters) {
    for (const auto& [name, events] : it_counters->second) {
      PopulateCounterTrack(pid, name, events, trace_info, current_level, data,
                           bounds);
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

FlameChartTimelineData CreateTimelineData(const TraceInformation& trace_info,
                                          TimeBounds& bounds) {
  FlameChartTimelineData data;
  int current_level = 0;

  std::vector<ProcessId> pids = GetSortedProcessIds(trace_info);

  for (ProcessId pid : pids) {
    PopulateProcessTrack(pid, trace_info, current_level, data, bounds);
  }

  data.events_by_level.resize(current_level);
  for (int i = 0; i < data.entry_levels.size(); ++i) {
    data.events_by_level[data.entry_levels[i]].push_back(i);
  }
  return data;
}

}  // namespace

// Processes a vector of TraceEvent structs.
// This function is independent of Emscripten types.
void DataProvider::ProcessTraceEvents(const ParsedTraceEvents& parsed_events,
                                      Timeline& timeline) {
  if (parsed_events.flame_events.empty() &&
      parsed_events.counter_events.empty()) {
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

  for (const auto& event : parsed_events.counter_events) {
    HandleCounterEvent(event, trace_info);
  }

  // Sort events, first by timestamp (ascending), then by duration
  // (descending).
  // Ensure all processes have a name in process_names.
  for (const auto& [pid, _] : trace_info.events_by_pid_tid) {
    trace_info.process_names.try_emplace(pid, GetDefaultProcessName(pid));
  }
  for (const auto& [pid, _] : trace_info.counters_by_pid_name) {
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

  // Populate process_list_ from trace_info.
  if (process_list_.empty()) {
    for (const auto& [pid, name] : trace_info.process_names) {
      process_list_.push_back(absl::StrCat(name, " (pid: ", pid, ")"));
    }
  }

  timeline.set_timeline_data(CreateTimelineData(trace_info, time_bounds));

  // Don't need to check for max_time because the TimeRange constructor will
  // handle any potential issues with max_time.
  if (time_bounds.min < std::numeric_limits<Microseconds>::max()) {
    timeline.set_fetched_data_time_range({time_bounds.min, time_bounds.max});
    if (parsed_events.visible_range_from_url.has_value()) {
      Microseconds start =
          MillisToMicros(parsed_events.visible_range_from_url->first);
      Microseconds end =
          MillisToMicros(parsed_events.visible_range_from_url->second);
      timeline.SetVisibleRange({start, end});
    } else {
      timeline.SetVisibleRange({time_bounds.min, time_bounds.max});
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

std::vector<std::string> DataProvider::GetProcessList() const {
  return process_list_;
}

}  // namespace traceviewer
