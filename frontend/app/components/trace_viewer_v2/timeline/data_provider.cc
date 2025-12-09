#include "xprof/frontend/app/components/trace_viewer_v2/timeline/data_provider.h"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "xprof/frontend/app/components/trace_viewer_v2/timeline/time_range.h"
#include "xprof/frontend/app/components/trace_viewer_v2/timeline/timeline.h"
#include "xprof/frontend/app/components/trace_viewer_v2/trace_helper/trace_event.h"
#include "xprof/frontend/app/components/trace_viewer_v2/trace_helper/trace_event_tree.h"
#include "absl/algorithm/container.h"
#include "absl/container/btree_map.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
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
                  absl::btree_map<ThreadId, std::vector<TraceEvent*>>>
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
void HandleMetadataEvent(
    const TraceEvent& event, TraceInformation& trace_info) {
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
void HandleCompleteEvent(
    TraceEvent& event, TraceInformation& trace_info,
    absl::btree_map<ProcessId, std::string>& process_names) {
  trace_info.events_by_pid_tid[event.pid][event.tid].push_back(&event);
  if (!process_names.contains(event.pid)) {
    const auto it = trace_info.process_names.find(event.pid);
    process_names[event.pid] = it != trace_info.process_names.end()
                                   ? it->second
                                   : GetDefaultProcessName(event.pid);
  }
}

struct TimeBounds {
  Microseconds min = std::numeric_limits<Microseconds>::max();
  Microseconds max = std::numeric_limits<Microseconds>::min();
};

struct TimelineDataResult {
  FlameChartTimelineData data;
  std::vector<TraceEvent*> timeline_events;
};

// Appends the given nodes (an array of trees) to the data, starting at the
// given level. Returns the maximum level of the nodes.
int AppendNodesAtLevel(absl::Span<const std::unique_ptr<TraceEventNode>> nodes,
                       int current_level, FlameChartTimelineData& data,
                       std::vector<TraceEvent*>& timeline_events,
                       TimeBounds& bounds) {
  int max_level = current_level;

  for (const std::unique_ptr<TraceEventNode>& node : nodes) {
    TraceEvent* event = const_cast<TraceEvent*>(node->event);

    data.entry_start_times.push_back(event->ts);
    data.entry_total_times.push_back(event->dur);
    data.entry_levels.push_back(current_level);
    data.entry_names.push_back(event->name);
    timeline_events.push_back(event);

    bounds.min = std::min(bounds.min, event->ts);
    bounds.max = std::max(bounds.max, event->ts + event->dur);

    if (!node->children.empty()) {
      int child_max_level = AppendNodesAtLevel(
          node->children, current_level + 1, data, timeline_events, bounds);
      max_level = std::max(max_level, child_max_level);
    }
  }

  return max_level;
}

void PopulateThreadTrack(ProcessId pid, ThreadId tid,
                         absl::Span<TraceEvent* const> events,
                         const TraceInformation& trace_info, int& current_level,
                         FlameChartTimelineData& data,
                         std::vector<TraceEvent*>& timeline_events,
                         TimeBounds& bounds) {
  const auto it = trace_info.thread_names.find({pid, tid});
  const std::string thread_group_name = it == trace_info.thread_names.end()
                                            ? GetDefaultThreadName(tid)
                                            : it->second;
  data.groups.push_back({thread_group_name,
                         /*start_level=*/current_level,
                         /*nesting_level=*/kThreadNestingLevel});

  TraceEventTree event_tree = BuildTree(events);

  // Get the maximum level index used by events in this thread.
  int max_level = AppendNodesAtLevel(event_tree.roots, current_level, data,
                                     timeline_events, bounds);

  current_level = max_level + 1;
}

void PopulateProcessTrack(
    ProcessId pid,
    const absl::btree_map<ThreadId, std::vector<TraceEvent*>>&
        events_by_tid,
    const TraceInformation& trace_info, int& current_level,
    FlameChartTimelineData& data,
    std::vector<TraceEvent*>& timeline_events, TimeBounds& bounds) {
  const auto it = trace_info.process_names.find(pid);
  const std::string process_group_name = it == trace_info.process_names.end()
                                             ? GetDefaultProcessName(pid)
                                             : it->second;
  data.groups.push_back({process_group_name, /*start_level=*/current_level,
                         /*nesting_level=*/kProcessNestingLevel});

  for (const auto& [tid, events] : events_by_tid) {
    PopulateThreadTrack(pid, tid, events, trace_info, current_level, data,
                        timeline_events, bounds);
  }
}

TimelineDataResult CreateTimelineData(const TraceInformation& trace_info,
                                      TimeBounds& bounds) {
  TimelineDataResult result;
  int current_level = 0;

  for (const auto& [pid, events_by_tid] : trace_info.events_by_pid_tid) {
    PopulateProcessTrack(pid, events_by_tid, trace_info, current_level,
                         result.data, result.timeline_events, bounds);
  }

  result.data.events_by_level.resize(current_level);
  for (int i = 0; i < result.data.entry_levels.size(); ++i) {
    result.data.events_by_level[result.data.entry_levels[i]].push_back(i);
  }
  return result;
}

}  // namespace

void DataProvider::ParseHloModuleName(const TraceEvent& event) {
  size_t paren_pos = event.name.find('(');
  if (paren_pos != std::string::npos) {
    std::string module_name = event.name.substr(0, paren_pos);
    size_t close_paren_pos = event.name.find(')', paren_pos);
    if (close_paren_pos != std::string::npos) {
      std::string program_id =
          event.name.substr(paren_pos + 1, close_paren_pos - paren_pos - 1);
      program_id_to_hlo_module_[program_id] = module_name;
    }
  }
}

// Processes a vector of TraceEvent structs.
// This function is independent of Emscripten types.
void DataProvider::ProcessTraceEvents(std::vector<TraceEvent>& event_list,
                                      Timeline& timeline) {
  if (event_list.empty()) {
    timeline.set_timeline_data({});
    timeline.set_data_time_range(TimeRange::Zero());
    timeline.SetVisibleRange(TimeRange::Zero());
    timeline_events_.clear();
    events_.clear();
    return;
  }

  // Store pointers to the events
  events_.clear();
  events_.reserve(event_list.size());
  for (TraceEvent& event : event_list) {
    events_.push_back(&event);
  }

  program_id_to_hlo_module_.clear();
  process_names_.clear();
  xla_modules_by_pid_.clear();
  TraceInformation trace_info;
  absl::flat_hash_map<ProcessId, absl::flat_hash_set<ThreadId>>
      xla_module_threads;

  for (auto* event : events_) {
    if (event->ph == Phase::kMetadata) {
      HandleMetadataEvent(*event, trace_info);
      if (event->name == kThreadName &&
          GetNameWithDefault(*event, GetDefaultThreadName(event->tid)) ==
              "XLA Modules") {
        xla_module_threads[event->pid].insert(event->tid);
      }
    }
  }

  for (auto* event : events_) {
    ParseHloModuleName(*event);

    switch (event->ph) {
      case Phase::kMetadata:
        // Already handled in the first pass.
        break;
      case Phase::kComplete:
        HandleCompleteEvent(*event, trace_info, process_names_);
        if (xla_module_threads.contains(event->pid) &&
            xla_module_threads[event->pid].contains(event->tid)) {
          xla_modules_by_pid_[event->pid].push_back(event);
        }
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

  // Sort XLA Module events by start time.
  for (auto& [pid, modules] : xla_modules_by_pid_) {
    absl::c_stable_sort(modules, gtl::OrderBy([](const TraceEvent* e) {
      return e->ts;
    }));
  }

  // Clear timeline_events_ before populating.
  timeline_events_.clear();

  TimeBounds time_bounds;

  TimelineDataResult timeline_data_result =
      CreateTimelineData(trace_info, time_bounds);
  timeline.set_timeline_data(std::move(timeline_data_result.data));
  timeline_events_ = std::move(timeline_data_result.timeline_events);

  event_map_.clear();
  for (TraceEvent* event : timeline_events_) {
    event_map_[std::make_tuple(event->name, event->ts, event->dur)] = event;
  }

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
  std::vector<std::string> process_list;
  for (const auto& [pid, name] : process_names_) {
    process_list.push_back(absl::StrCat(name, " (pid: ", pid, ")"));
  }
  return process_list;
}

std::optional<EventMetaData> DataProvider::GetEventMetaData(
    const std::string& name, Microseconds start, Microseconds duration) const {
  auto it = event_map_.find(std::make_tuple(name, start, duration));
  if (it == event_map_.end()) {
    return std::nullopt;
  }
  const TraceEvent* event = it->second;
  EventMetaData event_data;
  event_data.name = event->name;
  event_data.start = event->ts;
  event_data.duration = event->dur;
  auto process_it = process_names_.find(event->pid);
  event_data.processName = process_it == process_names_.end()
                               ? GetDefaultProcessName(event->pid)
                               : process_it->second;
  event_data.arguments = event->args;
  return event_data;
}

std::string DataProvider::GetHloModuleForEvent(const std::string& name,
                                               Microseconds start,
                                               Microseconds duration) const {
  auto it = event_map_.find(std::make_tuple(name, start, duration));
  if (it == event_map_.end()) {
    return "";
  }
  const TraceEvent* event = it->second;
  // HLO Module events have their name in the format "module_name(program_id)".
  // We extract the module_name part.
  const size_t paren_pos = event->name.find('(');
  if (paren_pos != std::string::npos) {
    return event->name.substr(0, paren_pos);
  }

  // Fallback: Find the HLO Module by time-based enclosure.
  const auto xla_modules_it = xla_modules_by_pid_.find(event->pid);
  if (xla_modules_it != xla_modules_by_pid_.end()) {
    const std::vector<const TraceEvent*>& modules = xla_modules_it->second;
    // Binary search for a module that contains event->ts.
    auto module_it =
        absl::c_upper_bound(modules, event->ts,
                          [](Microseconds ts, const TraceEvent* module) {
                            return ts < module->ts;
                          });

    if (module_it != modules.begin()) {
      --module_it;  // Go to the module that starts at or before event->ts
      const TraceEvent* module = *module_it;
      if (event->ts >= module->ts && event->ts <= module->ts + module->dur) {
        return module->name;
      }
    }
  }

  // For HLO Op events, find the associated HLO Module using program_id.
  const auto args_it = event->args.find("program_id");
  if (args_it != event->args.end()) {
    const std::string& program_id = args_it->second;
    const auto module_it = program_id_to_hlo_module_.find(program_id);
    if (module_it != program_id_to_hlo_module_.end()) {
      return module_it->second;
    }
  }
  return "";
}

}  // namespace traceviewer
