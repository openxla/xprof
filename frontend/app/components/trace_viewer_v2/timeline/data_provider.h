#ifndef THIRD_PARTY_XPROF_FRONTEND_APP_COMPONENTS_TRACE_VIEWER_V2_TIMELINE_DATA_PROVIDER_H_
#define THIRD_PARTY_XPROF_FRONTEND_APP_COMPONENTS_TRACE_VIEWER_V2_TIMELINE_DATA_PROVIDER_H_

#include <map>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

#include "absl/container/btree_map.h"
#include "absl/container/flat_hash_map.h"
#include "xprof/frontend/app/components/trace_viewer_v2/timeline/timeline.h"
#include "xprof/frontend/app/components/trace_viewer_v2/trace_helper/trace_event.h"
#include "absl/strings/string_view.h"

namespace traceviewer {

inline constexpr absl::string_view kThreadName = "thread_name";
inline constexpr absl::string_view kProcessName = "process_name";
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

  // Processes a vector of TraceEvent structs.
  void ProcessTraceEvents(absl::Span<const TraceEvent> event_list,
    Timeline& timeline, std::optional<TimeRange> full_time_span = std::nullopt);

  // Returns detailed event data for a given eventIndex.
  // emscripten::val GetEventData(int eventIndex) const;
  std::optional<EventMetaData> GetEventMetaData(const std::string& name,
                                                Microseconds start,
                                                Microseconds duration) const;

  // Returns the HLO module name for a given eventIndex.
  std::string GetHloModuleForEvent(const std::string& name, Microseconds start,
                                   Microseconds duration) const;

 private:
  void ParseHloModuleName(const TraceEvent& event);
  std::vector<const TraceEvent*> events_;
  // Maps program_id to hlo_module_name.
  absl::flat_hash_map<std::string, std::string> program_id_to_hlo_module_;
  // Maps process_id to process_name.
  absl::btree_map<ProcessId, std::string> process_names_;
  // A map of (name, start time, duration) to the TraceEvent.
  absl::flat_hash_map<std::tuple<std::string, Microseconds, Microseconds>,
                      const TraceEvent*> event_map_;
  // Maps {process_id, thread_id} to thread_name.
  absl::btree_map<std::pair<ProcessId, ThreadId>, std::string>
      xla_module_threads_;
  // Stores pointers to TraceEvent objects in the order they appear in the
  // timeline.
  std::vector<TraceEvent*> timeline_events_;
  // Stores XLA Module events, grouped by process ID and sorted by start time.
  absl::flat_hash_map<ProcessId, std::vector<const TraceEvent*>>
      xla_modules_by_pid_;
};

}  // namespace traceviewer

#endif  // THIRD_PARTY_XPROF_FRONTEND_APP_COMPONENTS_TRACE_VIEWER_V2_TIMELINE_DATA_PROVIDER_H_
