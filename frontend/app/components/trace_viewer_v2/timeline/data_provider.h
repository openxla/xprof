#ifndef THIRD_PARTY_XPROF_FRONTEND_APP_COMPONENTS_TRACE_VIEWER_V2_TIMELINE_DATA_PROVIDER_H_
#define THIRD_PARTY_XPROF_FRONTEND_APP_COMPONENTS_TRACE_VIEWER_V2_TIMELINE_DATA_PROVIDER_H_

#include <map>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/strings/string_view.h"
#include "xprof/frontend/app/components/trace_viewer_v2/timeline/timeline.h"
#include "xprof/frontend/app/components/trace_viewer_v2/trace_helper/trace_event.h"

namespace traceviewer {

inline constexpr absl::string_view kThreadName = "thread_name";
inline constexpr absl::string_view kProcessName = "process_name";
// The name of the metadata event used to sort processes (e.g., device rows
// in trace viewer).
inline constexpr absl::string_view kProcessSortIndex = "process_sort_index";
// The name of the metadata event used to sort threads within a process (e.g.,
// resource rows in trace viewer).
inline constexpr absl::string_view kThreadSortIndex = "thread_sort_index";
// The argument name for sort index in process_sort_index and
// thread_sort_index metadata events.
inline constexpr absl::string_view kSortIndex = "sort_index";
inline constexpr absl::string_view kName = "name";

struct EventMetaData {
  std::string name;
  Microseconds start;
  Microseconds duration;
  std::string processName;
  std::map<std::string, std::string> arguments;
};

class DataProvider {
 public:
  // Returns a list of process names.
  std::vector<std::string> GetProcessList() const;

  // Returns a list of flow categories present in the trace.
  const std::vector<int>& GetFlowCategories() const;

  // Processes vectors of TraceEvent structs.
  void ProcessTraceEvents(const ParsedTraceEvents& parsed_events,
                          Timeline& timeline,
                          std::optional<TimeRange> full_time_span);

  // Returns the HLO module name for a given event.
  std::string GetHloModuleForEvent(const std::string& name, Microseconds start,
                                 Microseconds duration) const;

 private:
  std::vector<int> present_flow_categories_;
  absl::flat_hash_map<ProcessId, std::string> process_names_;
  absl::flat_hash_map<ProcessId, absl::flat_hash_set<ThreadId>>
      xla_module_threads_;
  // Stores pointers to TraceEvent objects in the order they appear in the
  // timeline.
  std::vector<TraceEvent*> timeline_events_;
  // Stores XLA Module events, grouped by process ID and sorted by start time.
  absl::flat_hash_map<ProcessId, std::vector<const TraceEvent*>>
      xla_modules_by_pid_;
  absl::flat_hash_map<std::string, std::string> program_id_to_hlo_module_;
  absl::flat_hash_map<std::tuple<std::string, Microseconds, Microseconds>,
                      TraceEvent>
      event_map_;
};

}  // namespace traceviewer

#endif  // THIRD_PARTY_XPROF_FRONTEND_APP_COMPONENTS_TRACE_VIEWER_V2_TIMELINE_DATA_PROVIDER_H_
