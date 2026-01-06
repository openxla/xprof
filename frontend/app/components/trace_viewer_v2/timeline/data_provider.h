#ifndef THIRD_PARTY_XPROF_FRONTEND_APP_COMPONENTS_TRACE_VIEWER_V2_TIMELINE_DATA_PROVIDER_H_
#define THIRD_PARTY_XPROF_FRONTEND_APP_COMPONENTS_TRACE_VIEWER_V2_TIMELINE_DATA_PROVIDER_H_

#include <vector>

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

class DataProvider {
 public:
  // Returns a list of process names.
  std::vector<std::string> GetProcessList() const;

  // Processes vectors of TraceEvent structs.
  void ProcessTraceEvents(const ParsedTraceEvents& parsed_events,
                          Timeline& timeline);

 private:
  std::vector<std::string> process_list_;
};

}  // namespace traceviewer

#endif  // THIRD_PARTY_XPROF_FRONTEND_APP_COMPONENTS_TRACE_VIEWER_V2_TIMELINE_DATA_PROVIDER_H_
