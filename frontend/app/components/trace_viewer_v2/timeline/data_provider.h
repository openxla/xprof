#ifndef THIRD_PARTY_XPROF_FRONTEND_APP_COMPONENTS_TRACE_VIEWER_V2_TIMELINE_DATA_PROVIDER_H_
#define THIRD_PARTY_XPROF_FRONTEND_APP_COMPONENTS_TRACE_VIEWER_V2_TIMELINE_DATA_PROVIDER_H_

#include <string>
#include <tuple>
#include <vector>

#include "xprof/frontend/app/components/trace_viewer_v2/timeline/timeline.h"
#include "xprof/frontend/app/components/trace_viewer_v2/trace_helper/trace_event.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"

#include "absl/container/flat_hash_map.h"

namespace traceviewer {

inline constexpr absl::string_view kThreadName = "thread_name";
inline constexpr absl::string_view kProcessName = "process_name";
inline constexpr absl::string_view kName = "name";

class DataProvider {
 public:
  // Returns a list of process names.
  std::vector<std::string> GetProcessList() const;

  // Processes a vector of TraceEvent structs.
  void ProcessTraceEvents(absl::Span<const TraceEvent> event_list,
                          Timeline& timeline);

 private:
  std::vector<std::string> process_list_;
  // A map of (name, start time, duration) to the TraceEvent.
  absl::flat_hash_map<std::tuple<std::string, Microseconds, Microseconds>,
                      const TraceEvent*> event_map_;
};

}  // namespace traceviewer

#endif  // THIRD_PARTY_XPROF_FRONTEND_APP_COMPONENTS_TRACE_VIEWER_V2_TIMELINE_DATA_PROVIDER_H_
