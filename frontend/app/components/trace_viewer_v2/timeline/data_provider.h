#ifndef THIRD_PARTY_XPROF_FRONTEND_APP_COMPONENTS_TRACE_VIEWER_V2_TIMELINE_DATA_PROVIDER_H_
#define THIRD_PARTY_XPROF_FRONTEND_APP_COMPONENTS_TRACE_VIEWER_V2_TIMELINE_DATA_PROVIDER_H_

#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/btree_map.h"
#include "absl/container/btree_set.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/strings/string_view.h"
#include "frontend/app/components/trace_viewer_v2/timeline/timeline.h"
#include "frontend/app/components/trace_viewer_v2/timeline/track_packer.h"
#include "frontend/app/components/trace_viewer_v2/trace_helper/trace_event.h"
#include "plugin/xprof/protobuf/trace_data_response.pb.h"

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

  // Processes a protobuf TraceDataResponse directly into columnar timeline
  // storage in display order.
  void ProcessTraceEvents(const xprof::TraceDataResponse& response,
                          Timeline& timeline,
                          std::optional<std::pair<Milliseconds, Milliseconds>>
                              visible_range_from_url = std::nullopt);

  // Processes a serialized protobuf TraceDataResponse by indexing
  // complete_events wire slices and streaming tracks directly into columnar
  // timeline storage, freeing `serialized_proto` immediately after the track
  // loop before flow lines, colors, and mip pyramids are constructed. If
  // `details_out` is non-null, populates it with `(name, value)` pairs from
  // `response.details`. Returns false if `serialized_proto` is not a valid
  // TraceDataResponse.
  bool ProcessSerializedTraceEvents(
      std::string&& serialized_proto, Timeline& timeline,
      std::optional<std::pair<Milliseconds, Milliseconds>>
          visible_range_from_url = std::nullopt,
      std::vector<std::pair<std::string, bool>>* details_out = nullptr);

  // Processes pre-paired streamed trace events where complete_events are
  // provided as serialized TraceEventSeries wire slices and `response` carries
  // metadata, interned_strings, details, counter_events, and pre-paired
  // async_events. Releases input buffers via `release_input_buffers`
  // immediately after unpacking tracks.
  bool ProcessStreamedTraceEvents(
      const xprof::TraceDataResponse& response,
      absl::Span<const absl::string_view> complete_event_slices,
      absl::FunctionRef<void()> release_input_buffers, Timeline& timeline,
      std::optional<std::pair<Milliseconds, Milliseconds>>
          visible_range_from_url = std::nullopt);

  // Clears persistent session metadata and track registries.
  void Reset();

  // Returns true if any threads, counters, or async tracks are registered.
  bool HasKnownTracks() const;

 private:
  // Shared post-ingestion hook that updates timeline data and time ranges for
  // both TraceEvent and TraceDataResponse ingestion overloads.
  void FinalizeTimelineUpdate(
      Timeline& timeline, FlameChartTimelineData data, const TimeBounds& bounds,
      std::optional<std::pair<Milliseconds, Milliseconds>>
          visible_range_from_url,
      std::optional<std::pair<Milliseconds, Milliseconds>> full_timespan);

  bool ProcessProtoTraceEventsInternal(
      const xprof::TraceDataResponse& response,
      absl::Span<const absl::string_view> complete_event_slices,
      bool use_wire_complete_series, bool async_events_already_paired,
      absl::FunctionRef<void()> release_input_buffers, Timeline& timeline,
      std::optional<std::pair<Milliseconds, Milliseconds>>
          visible_range_from_url);

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

namespace internal {
void AppendTraceEventForTesting(const TraceEvent* event, int level,
                                FlameChartTimelineData& data);
}  // namespace internal

}  // namespace traceviewer

#endif  // THIRD_PARTY_XPROF_FRONTEND_APP_COMPONENTS_TRACE_VIEWER_V2_TIMELINE_DATA_PROVIDER_H_
