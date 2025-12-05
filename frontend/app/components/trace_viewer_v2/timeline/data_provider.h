#ifndef THIRD_PARTY_XPROF_FRONTEND_APP_COMPONENTS_TRACE_VIEWER_V2_TIMELINE_DATA_PROVIDER_H_
#define THIRD_PARTY_XPROF_FRONTEND_APP_COMPONENTS_TRACE_VIEWER_V2_TIMELINE_DATA_PROVIDER_H_

#include <emscripten/val.h>
#include <vector>

#include "absl/container/btree_map.h"
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

  // Returns detailed event data for a given eventIndex.
  // emscripten::val GetEventData(int eventIndex) const;
  emscripten::val GetEventData(std::string event_name) const;

  // Returns the HLO module name for a given eventIndex.
  std::string GetHloModuleForEvent(std::string event_name) const;

  // Sets the initial viewport of the timeline.
  void SetInitialViewport(float start_time_ms, float end_time_ms,
                          Timeline& timeline);

  // Sets the viewport of the timeline.
  void SetViewport(float start_time_ms, float end_time_ms, Timeline& timeline);

  // Returns the viewport of the timeline.
  std::pair<float, float> GetViewport(const Timeline& timeline);

  // Updates the arguments of the TraceEvent at eventIndex.
  void UpdateEventargs(std::string event_name, const emscripten::val& args);

 private:
  std::vector<std::string> process_list_;
  std::vector<TraceEvent> events_;
  std::vector<std::string> entry_names_;
  // Maps program_id to hlo_module_name.
  absl::flat_hash_map<std::string, std::string> program_id_to_hlo_module_;
  // Maps process_id to process_name.
  absl::btree_map<ProcessId, std::string> process_names_;
  // Maps {process_id, thread_id} to thread_name.
  absl::btree_map<std::pair<ProcessId, ThreadId>, std::string> thread_names_;
  // Stores pointers to TraceEvent objects in the order they appear in the
  // timeline.
  std::vector<TraceEvent*> timeline_events_;
  absl::btree_map<std::string, TraceEvent*> event_name_to_event_;
  // Stores XLA Module events, grouped by process ID and sorted by start time.
  absl::btree_map<ProcessId, std::vector<const TraceEvent*>>
      xla_modules_by_pid_;
  // Viewport initialized  flag
  bool initial_viewport_was_set_ = false;
};

}  // namespace traceviewer

#endif  // THIRD_PARTY_XPROF_FRONTEND_APP_COMPONENTS_TRACE_VIEWER_V2_TIMELINE_DATA_PROVIDER_H_
