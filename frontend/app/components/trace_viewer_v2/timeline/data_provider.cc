#include "xprof/frontend/app/components/trace_viewer_v2/timeline/data_provider.h"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "xprof/frontend/app/components/trace_viewer_v2/timeline/time_range.h"
#include "xprof/frontend/app/components/trace_viewer_v2/trace_helper/trace_event.h"
#include "xprof/frontend/app/components/trace_viewer_v2/trace_helper/trace_event_tree.h"
#include "absl/algorithm/container.h"
#include "absl/container/btree_map.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "util/gtl/comparator.h"
#include "xprof/frontend/app/components/trace_viewer_v2/timeline/timeline.h"

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
void HandleCompleteEvent(TraceEvent& event, TraceInformation& trace_info) {
  trace_info.events_by_pid_tid[event.pid][event.tid].push_back(&event);
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

// Processes a vector of TraceEvent structs.
// This function is independent of Emscripten types.
void DataProvider::ProcessTraceEvents(absl::Span<const TraceEvent> event_list,
                                      Timeline& timeline) {
  if (event_list.empty()) {
    timeline.set_timeline_data({});
    timeline.set_data_time_range(TimeRange::Zero());
    timeline.SetVisibleRange(TimeRange::Zero());
    timeline_events_.clear();
    return;
  }

  // Store the events
  events_.assign(event_list.begin(), event_list.end());

  program_id_to_hlo_module_.clear();
  process_names_.clear();
  xla_modules_by_pid_.clear();
  TraceInformation trace_info;
  absl::flat_hash_map<ProcessId, absl::flat_hash_set<ThreadId>>
      xla_module_threads;

  for (auto& event : events_) {
    if (event.ph == Phase::kMetadata) {
      HandleMetadataEvent(event, trace_info);
      if (event.name == kThreadName &&
          GetNameWithDefault(event, GetDefaultThreadName(event.tid)) ==
              "XLA Modules") {
        xla_module_threads[event.pid].insert(event.tid);
      }
      if (event.name == kProcessName) {
        process_names_[event.pid] =
            GetNameWithDefault(event, GetDefaultProcessName(event.pid));
      }
    }
  }

  for (auto& event : events_) {
    event_name_to_event_[event.name] = &event;
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

    switch (event.ph) {
      case Phase::kMetadata:
        // Already handled in the first pass.
        break;
      case Phase::kComplete:
        HandleCompleteEvent(event, trace_info);
        if (xla_module_threads.contains(event.pid) &&
            xla_module_threads[event.pid].contains(event.tid)) {
          xla_modules_by_pid_[event.pid].push_back(&event);
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

  // Populate process_list_ from trace_info.
  if (process_list_.empty()) {
    for (const auto& [pid, name] : trace_info.process_names) {
      process_list_.push_back(absl::StrCat(name, " (pid: ", pid, ")"));
    }
  }

  TimelineDataResult timeline_data_result =
      CreateTimelineData(trace_info, time_bounds);
  timeline.set_timeline_data(std::move(timeline_data_result.data));
  timeline_events_ = std::move(timeline_data_result.timeline_events);

  // Don't need to check for max_time because the TimeRange constructor will
  // handle any potential issues with max_time.
  if (time_bounds.min < std::numeric_limits<Microseconds>::max()) {
    timeline.set_data_time_range({time_bounds.min, time_bounds.max});
    // Default initialization: Set the visible range to the full data time range
    // only if the visible range hasn't been set yet.
    if (timeline.visible_range() == TimeRange::Zero()) {
      timeline.SetVisibleRange({time_bounds.min, time_bounds.max});
    }
  } else {
    timeline.set_data_time_range(TimeRange::Zero());
    timeline.SetVisibleRange(TimeRange::Zero());
  }
}

void DataProvider::SetInitialViewport(float start_time_ms, float end_time_ms,
                                      Timeline& timeline) {
  if (start_time_ms <= 0 || end_time_ms <= 0) {
    LOG(ERROR) << "Invalid initial viewport start_time_ms or end_time_ms: "
               << start_time_ms << " " << end_time_ms;
    return;
  }
  initial_viewport_was_set_ = true;

  TimeRange initial_range = {Microseconds(start_time_ms),
                             Microseconds(end_time_ms)};
  // Ensure the initial viewport is within the data time range.
  TimeRange data_range = timeline.data_time_range();
  if (initial_range.start() < data_range.start()) {
    initial_range += (data_range.start() - initial_range.start());
  }
  if (initial_range.end() > data_range.end()) {
    initial_range = initial_range - (initial_range.end() - data_range.end());
  }
  timeline.SetVisibleRange(initial_range, /*animate=*/false);
}

std::vector<std::string> DataProvider::GetProcessList() const {
  return process_list_;
}

void DataProvider::SetViewport(float start_time_ms, float end_time_ms,
                               Timeline& timeline) {
  if (start_time_ms < 0 || end_time_ms < 0) {
    LOG(ERROR) << "Invalid start_time_ms or end_time_ms: " << start_time_ms
               << " " << end_time_ms;
    return;
  }
  TimeRange new_range = {
      Microseconds(start_time_ms),
      Microseconds(end_time_ms),
  };
  timeline.SetVisibleRange(new_range, /*animate=*/true);
}

std::pair<float, float> DataProvider::GetViewport(const Timeline& timeline) {
  return {timeline.visible_range().start(),
          timeline.visible_range().end()};
}


emscripten::val DataProvider::GetEventData(std::string event_name) const {
  if (event_name.empty()) {
    return emscripten::val::null();
  }
  auto it = event_name_to_event_.find(event_name);
  if (it == event_name_to_event_.end()) {
    return emscripten::val::null();
  }
  const TraceEvent* event = it->second;
  emscripten::val event_obj = emscripten::val::object();
  // event_obj.set("eventIndex", eventIndex); // eventIndex is not available
  event_obj.set("name", event->name);
  event_obj.set("start", event->ts);
  event_obj.set("duration", event->dur);
  auto process_it = process_names_.find(event->pid);
  event_obj.set("processName", process_it == process_names_.end()
                                   ? GetDefaultProcessName(event->pid)
                                   : process_it->second);

  emscripten::val args = emscripten::val::object();
  for (const auto& pair : event->args) {
    args.set(pair.first, pair.second);
  }
  event_obj.set("arguments", args);
  return event_obj;
}

void DataProvider::UpdateEventargs(std::string event_name,
                                   const emscripten::val& args) {
  if (event_name.empty()) return;
  auto it = event_name_to_event_.find(event_name);
  if (it == event_name_to_event_.end()) {
    return;
  }
  TraceEvent* event = it->second;
  emscripten::val keys =
      emscripten::val::global("Object").call<emscripten::val>("keys", args);
  int length = keys["length"].as<int>();
  for (int i = 0; i < length; ++i) {
    std::string key = keys[i].as<std::string>();
    if (args[key].isString()) {
      event->args[key] = args[key].as<std::string>();
    } else if (args[key].isNumber()) {
      event->args[key] = std::to_string(args[key].as<double>());
    }
  }
}

std::string DataProvider::GetHloModuleForEvent(std::string event_name) const {
  if (event_name.empty()) {
    return "default";
  }
  auto it = event_name_to_event_.find(event_name);
  if (it == event_name_to_event_.end()) {
    return "default";
  }
  const TraceEvent* event = it->second;
  // HLO Module events have their name in the format "module_name(program_id)".
  // We extract the module_name part.
  size_t paren_pos = event->name.find('(');
  if (paren_pos != std::string::npos) {
    return event->name.substr(0, paren_pos);
  }

  // Fallback: Find the HLO Module by time-based enclosure.
  auto xla_modules_it = xla_modules_by_pid_.find(event->pid);
  if (xla_modules_it != xla_modules_by_pid_.end()) {
    const std::vector<const TraceEvent*>& modules = xla_modules_it->second;
    // Binary search for a module that contains event->ts.
    auto module_it = std::upper_bound(
        modules.begin(), modules.end(), event->ts,
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
  auto args_it = event->args.find("program_id");
  if (args_it != event->args.end()) {
    const std::string& program_id = args_it->second;
    auto module_it = program_id_to_hlo_module_.find(program_id);
    if (module_it != program_id_to_hlo_module_.end()) {
      return module_it->second;
    }
  }
  return "default";
}

}  // namespace traceviewer
