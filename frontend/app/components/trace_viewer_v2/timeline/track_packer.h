#ifndef THIRD_PARTY_XPROF_FRONTEND_APP_COMPONENTS_TRACE_VIEWER_V2_TIMELINE_TRACK_PACKER_H_
#define THIRD_PARTY_XPROF_FRONTEND_APP_COMPONENTS_TRACE_VIEWER_V2_TIMELINE_TRACK_PACKER_H_

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/container/btree_map.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/functional/function_ref.h"
#include "absl/log/check.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "google/protobuf/arena.h"
#include "tsl/profiler/lib/context_types.h"
#include "frontend/app/components/trace_viewer_v2/timeline/timeline.h"
#include "frontend/app/components/trace_viewer_v2/trace_helper/proto_event_ref.h"
#include "frontend/app/components/trace_viewer_v2/trace_helper/trace_event.h"
#include "frontend/app/components/trace_viewer_v2/trace_helper/wire_trace_reader.h"
#include "plugin/xprof/protobuf/trace_data_response.pb.h"

namespace traceviewer {

// Interns strings into a 0-based dense pool where index 0 is always "".
class TableInterner {
 public:
  TableInterner() {
    pool_.push_back("");
    map_.try_emplace("", 0);
  }

  // Initializes the pool from a pre-interned table before any Intern() calls.
  void SetInitialPool(std::vector<std::string> initial_pool) {
    DCHECK_EQ(pool_.size(), 1u);
    DCHECK(pool_[0].empty());
    if (!initial_pool.empty()) {
      DCHECK(initial_pool[0].empty());
      pool_ = std::move(initial_pool);
      map_.clear();
      map_.try_emplace("", 0);
    }
  }

  uint32_t Intern(absl::string_view str) {
    if (str.empty()) return 0;
    if (const auto it = map_.find(str); it != map_.end()) {
      return it->second;
    }
    const uint32_t index = static_cast<uint32_t>(pool_.size());
    pool_.push_back(std::string(str));
    map_.try_emplace(pool_.back(), index);
    return index;
  }

  // Appends a known-unique non-empty string directly to the pool without
  // hash-map indexing overhead.
  uint32_t AppendUnique(absl::string_view str) {
    if (str.empty()) return 0;
    const uint32_t index = static_cast<uint32_t>(pool_.size());
    pool_.emplace_back(str);
    return index;
  }

  std::vector<std::string> TakePool() { return std::move(pool_); }

 private:
  absl::flat_hash_map<std::string, uint32_t> map_;
  std::vector<std::string> pool_;
};

// Lightweight endpoint representation for flow events on the direct protobuf
// ingestion path.
struct FlowEndpoint {
  ProcessId pid = 0;
  ThreadId tid = 0;
  Microseconds ts = 0.0;
  uint32_t event_index = 0;
  tsl::profiler::ContextType category = tsl::profiler::ContextType::kGeneric;
};

inline ProcessId GetEndpointPid(const TraceEvent* e) { return e->pid; }
inline ThreadId GetEndpointTid(const TraceEvent* e) { return e->tid; }
inline Microseconds GetEndpointTs(const TraceEvent* e) { return e->ts; }
inline tsl::profiler::ContextType GetEndpointCategory(const TraceEvent* e) {
  return e->category;
}

inline ProcessId GetEndpointPid(const FlowEndpoint& e) { return e.pid; }
inline ThreadId GetEndpointTid(const FlowEndpoint& e) { return e.tid; }
inline Microseconds GetEndpointTs(const FlowEndpoint& e) { return e.ts; }
inline tsl::profiler::ContextType GetEndpointCategory(const FlowEndpoint& e) {
  return e.category;
}

// Holds per-trace metadata, interners, and grouped event structures
// parameterized by the event representation.
template <typename EventT>
struct TraceInformation {
  using FlowItem = std::conditional_t<std::is_same_v<EventT, ProtoEventRef>,
                                      FlowEndpoint, const EventT*>;

  // The EventT objects pointed to must outlive this TraceInformation instance.
  absl::btree_map<ProcessId,
                  absl::btree_map<ThreadId, std::vector<const EventT*>>>
      events_by_pid_tid;
  absl::btree_map<
      ProcessId, absl::btree_map<std::string, std::vector<const CounterEvent*>>>
      counters_by_pid_name;
  absl::btree_map<std::pair<ProcessId, ThreadId>, std::string> thread_names;
  absl::flat_hash_map<std::pair<ProcessId, ThreadId>, uint32_t>
      thread_sort_indices;
  absl::flat_hash_map<ProcessId, std::string> process_names;
  absl::flat_hash_map<ProcessId, uint32_t> process_sort_indices;
  absl::flat_hash_map<ProcessId, uint32_t> incoming_process_sort_indices;
  absl::btree_map<uint64_t, std::vector<FlowItem>> flow_events_by_id;
  absl::flat_hash_map<ProcessId, ThreadId> xla_modules_tids;
  absl::flat_hash_map<ProcessId, ThreadId> xla_ops_tids;
  absl::flat_hash_set<ProcessId> async_processes_by_events;
  TableInterner name_interner;
  TableInterner module_interner;
  TableInterner op_interner;
  bool is_mpmd = false;
};

// Accumulates minimum and maximum timestamps across packed events.
struct TimeBounds {
  Microseconds min = std::numeric_limits<Microseconds>::max();
  Microseconds max = std::numeric_limits<Microseconds>::min();
};

// Compact event representation used for per-track display-order unmarshalling
// and packing.
struct TrackEvent {
  Microseconds ts = 0.0;
  Microseconds dur = 0.0;
  uint64_t flow_id = 0;
  int32_t group_id = -1;
  uint32_t name_index = 0;
  uint32_t uid = 0;
  uint32_t hlo_op_ref = 0;
  uint32_t hlo_module_ref = 0;
};
static_assert(sizeof(TrackEvent) <= 48, "TrackEvent must be <= 48 bytes.");

// Sorts raw XLA Module intervals by (start asc, end desc) and flattens them
// into strictly disjoint half-open [start, end) segments so nested sub-modules
// never shadow outer parent modules after ending.
std::vector<ModuleInterval> BuildDisjointModuleSegments(
    std::vector<ModuleInterval> raw_intervals);

// Stateful per-track event packer that preserves active flame-chart row end
// times and nested parent stack across streamed chunks of a single track.
class ProtoTrackPacker {
 public:
  ProtoTrackPacker(
      absl::string_view thread_name, bool is_async_track,
      absl::Span<const ModuleInterval> module_intervals,
      uint32_t default_module_index, int start_level, int& max_level,
      FlameChartTimelineData& data, TimeBounds& bounds,
      TraceInformation<ProtoEventRef>& trace_info,
      absl::FunctionRef<absl::string_view(uint32_t)> get_interned_string,
      std::vector<uint32_t>& name_ref_to_op_index,
      std::vector<uint32_t>& name_ref_to_module_index);

  // Packs a chunk of track events into FlameChartTimelineData. When
  // `already_sorted` is true, skips the sorting check and sort pass.
  void PackEvents(absl::Span<TrackEvent> events, bool already_sorted = false);

 private:
  struct ActiveParent {
    Microseconds end_time = 0.0;
    Microseconds total_time = 0.0;
    size_t timeline_data_index = 0;
    uint32_t sparse_record_index = SparseSelfTimes::kNone;
  };

  bool is_async_track_;
  bool is_xla_ops_thread_;
  bool is_data_motion_layer_;
  absl::Span<const ModuleInterval> module_intervals_;
  uint32_t default_module_index_;
  int start_level_;
  int& max_level_;
  FlameChartTimelineData& data_;
  TimeBounds& bounds_;
  TraceInformation<ProtoEventRef>& trace_info_;
  absl::FunctionRef<absl::string_view(uint32_t)> get_interned_string_;
  std::vector<uint32_t>& name_ref_to_op_index_;
  std::vector<uint32_t>& name_ref_to_module_index_;
  absl::InlinedVector<Microseconds, 16> row_end_times_;
  absl::InlinedVector<ActiveParent, 16> parent_stack_;
  uint32_t module_intervals_index_ = std::numeric_limits<uint32_t>::max();
};

// Packs a single track's events into columnar FlameChartTimelineData arrays,
// assigning non-overlapping relative flame chart rows and computing fused
// self-times.
void PackProtoEvents(
    absl::Span<TrackEvent> events, absl::string_view thread_name,
    bool is_async_track, absl::Span<const ModuleInterval> module_intervals,
    uint32_t default_module_index, int start_level, int& max_level,
    FlameChartTimelineData& data, TimeBounds& bounds,
    TraceInformation<ProtoEventRef>& trace_info,
    absl::FunctionRef<absl::string_view(uint32_t)> get_interned_string,
    std::vector<uint32_t>& name_ref_to_op_index,
    std::vector<uint32_t>& name_ref_to_module_index);

inline constexpr size_t kDefaultProtoTrackChunkSize = 65536;

// Unmarshals delta-encoded protobuf event series for a single thread track into
// track_buffer and packs them into FlameChartTimelineData. Single-series tracks
// stream in chunks of `max_chunk_size` events (splitting at positive timestamp
// deltas) to bound `track_buffer` memory. If `max_end_ps_out` is non-null,
// updates `*max_end_ps_out` with the maximum event end timestamp in
// picoseconds.
void UnpackAndPackProtoTrack(
    ProcessId pid, ThreadId tid, absl::string_view thread_name,
    absl::Span<const xprof::TraceEventSeries* const> series_list,
    absl::FunctionRef<absl::string_view(uint32_t)> get_interned_string,
    uint32_t interned_strings_count,
    absl::Span<const ModuleInterval> module_intervals,
    uint32_t default_module_index, int start_level, int& max_level,
    FlameChartTimelineData& data, TimeBounds& bounds,
    TraceInformation<ProtoEventRef>& trace_info,
    absl::btree_map<int, int>& flow_category_counts,
    std::vector<TrackEvent>& track_buffer,
    std::vector<uint32_t>& name_ref_to_op_index,
    std::vector<uint32_t>& name_ref_to_module_index,
    size_t max_chunk_size = kDefaultProtoTrackChunkSize,
    uint64_t* max_end_ps_out = nullptr);

// Streams one or more WireSeriesView segments for a single track directly into
// FlameChartTimelineData in chunks of `max_chunk_size` without materializing
// protobuf TraceEventSeries messages in memory. If `max_end_ps_out` is
// non-null, updates `*max_end_ps_out` with the maximum event end timestamp in
// picoseconds. If `arena` is non-null and fallback parsing is required, uses
// and resets `arena`.
void UnpackAndPackProtoSeriesWire(
    ProcessId pid, ThreadId tid, absl::string_view thread_name,
    absl::Span<const WireSeriesView* const> series_views,
    absl::FunctionRef<absl::string_view(uint32_t)> get_interned_string,
    uint32_t interned_strings_count,
    absl::Span<const ModuleInterval> module_intervals,
    uint32_t default_module_index, int start_level, int& max_level,
    FlameChartTimelineData& data, TimeBounds& bounds,
    TraceInformation<ProtoEventRef>& trace_info,
    absl::btree_map<int, int>& flow_category_counts,
    std::vector<TrackEvent>& track_buffer,
    std::vector<uint32_t>& name_ref_to_op_index,
    std::vector<uint32_t>& name_ref_to_module_index,
    size_t max_chunk_size = kDefaultProtoTrackChunkSize,
    uint64_t* max_end_ps_out = nullptr, google::protobuf::Arena* arena = nullptr);

void UnpackAndPackProtoSeriesWire(
    ProcessId pid, ThreadId tid, absl::string_view thread_name,
    const WireSeriesView& series_view,
    absl::FunctionRef<absl::string_view(uint32_t)> get_interned_string,
    uint32_t interned_strings_count,
    absl::Span<const ModuleInterval> module_intervals,
    uint32_t default_module_index, int start_level, int& max_level,
    FlameChartTimelineData& data, TimeBounds& bounds,
    TraceInformation<ProtoEventRef>& trace_info,
    absl::btree_map<int, int>& flow_category_counts,
    std::vector<TrackEvent>& track_buffer,
    std::vector<uint32_t>& name_ref_to_op_index,
    std::vector<uint32_t>& name_ref_to_module_index,
    size_t max_chunk_size = kDefaultProtoTrackChunkSize,
    uint64_t* max_end_ps_out = nullptr, google::protobuf::Arena* arena = nullptr);

}  // namespace traceviewer

#endif  // THIRD_PARTY_XPROF_FRONTEND_APP_COMPONENTS_TRACE_VIEWER_V2_TIMELINE_TRACK_PACKER_H_
