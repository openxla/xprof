#include "xprof/frontend/app/components/trace_viewer_v2/timeline/data_provider.h"

#include <algorithm>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "xprof/frontend/app/components/trace_viewer_v2/timeline/time_range.h"
#include "xprof/frontend/app/components/trace_viewer_v2/timeline/timeline.h"
#include "xprof/frontend/app/components/trace_viewer_v2/trace_helper/trace_event.h"
#include "xprof/frontend/app/components/trace_viewer_v2/trace_helper/trace_event_tree.h"
#include "absl/algorithm/container.h"
#include "absl/container/btree_map.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
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
  absl::btree_map<std::pair<ProcessId, ThreadId>, std::string> thread_names;
  absl::btree_map<ProcessId, std::string> process_names;
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
  data.groups.push_back({thread_group_name,
                         /*start_level=*/current_level,
                         /*nesting_level=*/kThreadNestingLevel});

  TraceEventTree event_tree = BuildTree(events);

  // Get the maximum level index used by events in this thread.
  int max_level =
      AppendNodesAtLevel(event_tree.roots, current_level, data, bounds);

  current_level = max_level + 1;
}

void PopulateProcessTrack(
    ProcessId pid,
    const absl::btree_map<ThreadId, std::vector<const TraceEvent*>>&
        events_by_tid,
    const TraceInformation& trace_info, int& current_level,
    FlameChartTimelineData& data, TimeBounds& bounds) {
  const auto it = trace_info.process_names.find(pid);
  const std::string process_group_name = it == trace_info.process_names.end()
                                             ? GetDefaultProcessName(pid)
                                             : it->second;
  data.groups.push_back({process_group_name, /*start_level=*/current_level,
                         /*nesting_level=*/kProcessNestingLevel});

  for (const auto& [tid, events] : events_by_tid) {
    PopulateThreadTrack(pid, tid, events, trace_info, current_level, data,
                        bounds);
  }
}

FlameChartTimelineData CreateTimelineData(const TraceInformation& trace_info,
                                          TimeBounds& bounds) {
  FlameChartTimelineData data;
  int current_level = 0;

  for (const auto& [pid, events_by_tid] : trace_info.events_by_pid_tid) {
    PopulateProcessTrack(pid, events_by_tid, trace_info, current_level, data,
                         bounds);
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
void DataProvider::ProcessTraceEvents(absl::Span<const TraceEvent> event_list,
                                      Timeline& timeline) {
  if (event_list.empty()) {
    timeline.set_timeline_data({});
    timeline.set_data_time_range(TimeRange::Zero());
    timeline.SetVisibleRange(TimeRange::Zero());
    return;
  }

  TraceInformation trace_info;
  for (const auto& event : event_list) {
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
    timeline.set_data_time_range({time_bounds.min, time_bounds.max});
    timeline.SetVisibleRange({time_bounds.min, time_bounds.max});
  } else {
    timeline.set_data_time_range(TimeRange::Zero());
    timeline.SetVisibleRange(TimeRange::Zero());
  }
}

std::vector<std::string> DataProvider::GetProcessList() const {
  return process_list_;
}

}  // namespace traceviewer
