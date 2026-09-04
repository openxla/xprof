#ifndef THIRD_PARTY_XPROF_FRONTEND_APP_COMPONENTS_TRACE_VIEWER_V2_TIMELINE_DATA_PROVIDER_H_
#define THIRD_PARTY_XPROF_FRONTEND_APP_COMPONENTS_TRACE_VIEWER_V2_TIMELINE_DATA_PROVIDER_H_

#include <string>
#include <utility>
#include <vector>

#include "absl/container/btree_map.h"
#include "absl/container/btree_set.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/strings/string_view.h"
#include "frontend/app/components/trace_viewer_v2/timeline/timeline.h"
#include "frontend/app/components/trace_viewer_v2/trace_helper/trace_event.h"

namespace traceviewer {

class DataProvider {
 public:
  // Returns a list of flow categories present in the trace.
  const std::vector<int>& GetFlowCategories() const;

  // Returns process mappings (pid -> hostname).
  absl::flat_hash_map<ProcessId, std::string> GetProcessMappings() const;

  // Processes vectors of TraceEvent structs.
  void ProcessTraceEvents(const ParsedTraceEvents& parsed_events,
                          Timeline& timeline);

  // Clears persistent session metadata and track registries.
  void Reset();

 private:
  // Returns true if any threads, counters, or async tracks are registered.
  bool HasKnownTracks() const;

  std::vector<int> present_flow_categories_;
  absl::flat_hash_map<ProcessId, std::string> process_names_;
  absl::btree_map<std::pair<ProcessId, ThreadId>, std::string> thread_names_;
  absl::flat_hash_map<ProcessId, uint32_t> process_sort_indices_;
  absl::flat_hash_map<std::pair<ProcessId, ThreadId>, uint32_t>
      thread_sort_indices_;
  absl::flat_hash_map<ProcessId, ThreadId> xla_modules_tids_;
  absl::flat_hash_map<GroupKey, int> max_observed_levels_;
  absl::btree_set<std::pair<ProcessId, ThreadId>> known_threads_;
  absl::btree_map<ProcessId, absl::btree_set<std::string>> known_async_tracks_;
  absl::flat_hash_set<ProcessId> known_async_processes_;
  absl::btree_map<ProcessId, absl::btree_set<std::string>> known_counters_;
};

}  // namespace traceviewer

#endif  // THIRD_PARTY_XPROF_FRONTEND_APP_COMPONENTS_TRACE_VIEWER_V2_TIMELINE_DATA_PROVIDER_H_
