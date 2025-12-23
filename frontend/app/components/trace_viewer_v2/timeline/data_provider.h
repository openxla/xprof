#ifndef THIRD_PARTY_XPROF_FRONTEND_APP_COMPONENTS_TRACE_VIEWER_V2_TIMELINE_DATA_PROVIDER_H_
#define THIRD_PARTY_XPROF_FRONTEND_APP_COMPONENTS_TRACE_VIEWER_V2_TIMELINE_DATA_PROVIDER_H_

#include <map>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

#include "absl/container/btree_map.h"
#include "absl/strings/string_view.h"
#include "xprof/frontend/app/components/trace_viewer_v2/timeline/timeline.h"
#include "xprof/frontend/app/components/trace_viewer_v2/trace_helper/trace_event.h"

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

  // Processes vectors of TraceEvent structs.
  void ProcessTraceEvents(const ParsedTraceEvents& parsed_events,
                          Timeline& timeline);

  // Returns detailed event data for a given eventIndex.
  // emscripten::val GetEventData(int eventIndex) const;
  std::optional<EventMetaData> GetEventMetaData(const std::string& name,
                                                double start_us,
                                                double duration_us) const;

 private:
  // Use btree_map to keep process tracks sorted by PID.
  absl::btree_map<ProcessId, std::string> process_names_;
  // A map of (name, start time, duration) to the TraceEvent.
  std::map<std::tuple<std::string, Microseconds, Microseconds>,
           const TraceEvent*>
      event_map_;
};

}  // namespace traceviewer

#endif  // THIRD_PARTY_XPROF_FRONTEND_APP_COMPONENTS_TRACE_VIEWER_V2_TIMELINE_DATA_PROVIDER_H_
