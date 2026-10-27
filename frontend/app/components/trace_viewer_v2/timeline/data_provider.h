#ifndef THIRD_PARTY_XPROF_FRONTEND_APP_COMPONENTS_TRACE_VIEWER_V2_TIMELINE_DATA_PROVIDER_H_
#define THIRD_PARTY_XPROF_FRONTEND_APP_COMPONENTS_TRACE_VIEWER_V2_TIMELINE_DATA_PROVIDER_H_

#include <memory>
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

enum class TimelineFlushMode {
  kAutoFlush,
  kDeferFlush,
};

class DataProvider {
 public:
  using TimelineFlushMode = ::traceviewer::TimelineFlushMode;

  DataProvider();
  ~DataProvider();

  DataProvider(DataProvider&&) noexcept;
  DataProvider& operator=(DataProvider&&) noexcept;

  // Returns a list of flow categories present in the trace.
  const std::vector<int>& GetFlowCategories() const;

  // Returns process mappings (pid -> hostname).
  absl::flat_hash_map<ProcessId, std::string> GetProcessMappings() const;

  // Processes vectors of TraceEvent structs.
  void ProcessTraceEvents(const ParsedTraceEvents& parsed_events,
                          Timeline& timeline);

  // Prepares incremental loading state for a new fetch while keeping
  // persistent track registries across zoom/pan refetches.
  void BeginIncrementalLoad(Timeline& timeline);

  // Incrementally appends a batch of parsed trace events and updates the
  // timeline using per-track packed event caching.
  void AppendTraceEventsIncremental(
      ParsedTraceEvents parsed_events, Timeline& timeline,
      TimelineFlushMode flush_mode = TimelineFlushMode::kAutoFlush);

  // Releases temporary incremental batch buffers after all chunks of a fetch
  // have been processed.
  void FinishIncrementalLoad();

  // Marks that the incremental load is finishing so any deferred refetch
  // timeline update is flushed.
  void MarkFinishingIncrementalLoad();

  // Returns true if a refetch is still waiting to flush its timeline update.
  bool HasPendingRefetchUpdate() const;

  // Clears persistent session metadata and track registries.
  void Reset();

  // Returns true if any threads, counters, or async tracks are registered.
  bool HasKnownTracks() const;

 private:
  struct IncrementalState;

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
  std::unique_ptr<IncrementalState> incremental_state_;
};

namespace internal {
void AppendTraceEventForTesting(const TraceEvent* event, int level,
                                FlameChartTimelineData& data);

absl::btree_map<GroupKey, bool> GetRestoredExpandedStates(
    const std::vector<Group>& groups);
}  // namespace internal

}  // namespace traceviewer

#endif  // THIRD_PARTY_XPROF_FRONTEND_APP_COMPONENTS_TRACE_VIEWER_V2_TIMELINE_DATA_PROVIDER_H_
