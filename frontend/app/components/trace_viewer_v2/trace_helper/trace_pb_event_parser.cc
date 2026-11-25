#include "frontend/app/components/trace_viewer_v2/trace_helper/trace_pb_event_parser.h"

#ifdef __EMSCRIPTEN__
#include <emscripten/bind.h>
#include <emscripten/val.h>
#endif

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/no_destructor.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/functional/any_invocable.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/string_view.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "xprof/convert/trace_viewer/delta_series/zstd_compression.h"
#ifdef __EMSCRIPTEN__
#include "frontend/app/components/trace_viewer_v2/application.h"
#endif
#include "frontend/app/components/trace_viewer_v2/helper/time_formatter.h"
#include "frontend/app/components/trace_viewer_v2/timeline/data_provider.h"
#include "frontend/app/components/trace_viewer_v2/timeline/timeline.h"
#include "frontend/app/components/trace_viewer_v2/trace_helper/trace_event.h"
#include "frontend/app/components/trace_viewer_v2/trace_helper/trace_event_parser_core.h"
#include "plugin/xprof/protobuf/trace_data_response.pb.h"

namespace traceviewer {

namespace {

#ifdef __EMSCRIPTEN__
std::optional<std::pair<Milliseconds, Milliseconds>> ParseVisibleRangeFromUrl(
    const emscripten::val& visible_range_from_url) {
  if (!visible_range_from_url.isNull() &&
      !visible_range_from_url.isUndefined() &&
      visible_range_from_url["length"].as<int>() == 2) {
    const Milliseconds start = visible_range_from_url[0].as<Milliseconds>();
    const Milliseconds end = visible_range_from_url[1].as<Milliseconds>();
    return std::make_pair(start, end);
  }
  return std::nullopt;
}
#endif

void DispatchResponseDetails(const xprof::TraceDataResponse& response,
                             bool& mpmd_pipeline_view) {
  if (response.details_size() <= 0) {
    return;
  }
#ifdef __EMSCRIPTEN__
  constexpr absl::string_view kFullDmaDetailName = "full_dma";
  std::optional<bool> full_dma_value;
#endif
  constexpr absl::string_view kMpmdPipelineViewDetailName =
      "mpmd_pipeline_view";
  for (const xprof::DetailItem& detail : response.details()) {
    if (detail.name() == kMpmdPipelineViewDetailName) {
      mpmd_pipeline_view = detail.value();
    }
#ifdef __EMSCRIPTEN__
    if (detail.name() == kFullDmaDetailName) {
      full_dma_value = detail.value();
    }
#endif
  }
#ifdef __EMSCRIPTEN__
  // Only forward user-facing trace detail toggles to the host UI, matching
  // the JSON path (`maybeDispatchDetailsReceivedEvent`). Internal keys such
  // as `mpmd_pipeline_view` are consumed directly in WASM and would otherwise
  // trigger a redundant full reload.
  if (!full_dma_value.has_value()) {
    return;
  }
  emscripten::val details_map = emscripten::val::global("Map").new_();
  details_map.call<void>("set",
                         emscripten::val(std::string(kFullDmaDetailName)),
                         emscripten::val(*full_dma_value));
  emscripten::val detail_obj = emscripten::val::object();
  detail_obj.set("details", details_map);

  emscripten::val event_init = emscripten::val::object();
  event_init.set("detail", detail_obj);

  emscripten::val event =
      emscripten::val::global("CustomEvent")
          .new_(emscripten::val("details_received"), event_init);
  emscripten::val::global("window").call<void>("dispatchEvent", event);
#endif
}

ParsedTraceEvents ParseCompressedTraceEvents(
    uintptr_t data_ptr, size_t data_size,
    std::optional<std::pair<Milliseconds, Milliseconds>>
        visible_range_from_url) {
  absl::string_view buffer_data(reinterpret_cast<const char*>(data_ptr),
                                data_size);
  ParsedTraceEvents result;
  absl::StatusOr<std::string> decompressed_proto =
      tensorflow::profiler::ZstdCompression::Decompress(buffer_data);
  if (!decompressed_proto.ok()) {
    return result;
  }

  xprof::TraceDataResponse response;
  if (!response.ParseFromString(*decompressed_proto)) {
    return result;
  }
  // Free decompressed proto memory early to reduce peak memory usage.
  std::string().swap(*decompressed_proto);

  DispatchResponseDetails(response, result.mpmd_pipeline_view);
  ProcessMetadataEvents(response, result);
  ProcessCompleteEvents(response, result);
  ProcessAsyncEvents(response, result);
  ProcessCounterEvents(response, result);

  if (response.has_full_timespan_start_ps() &&
      response.has_full_timespan_end_ps()) {
    const Milliseconds start_ms = response.full_timespan_start_ps() / 1e9;
    const Milliseconds end_ms = response.full_timespan_end_ps() / 1e9;
    if (start_ms <= end_ms) {
      result.full_timespan = std::make_pair(start_ms, end_ms);
    }
  }

  result.visible_range_from_url = visible_range_from_url;

  return result;
}

struct FrameSummary {
  uint64_t min_start_ps = std::numeric_limits<uint64_t>::max();
  uint64_t max_end_ps = 0;
  // True once bounds have been recorded from complete or closing split
  // (`kDurationEnd`) events in this frame.
  bool has_non_split_bounds = false;
  bool always_process = false;
};

constexpr size_t kCachedBlockEvents = 4096;
constexpr size_t kMaxCachedPackedBytes = 512ULL * 1024 * 1024;
// Target number of visual time bins across a local cached refetch window so
// wide replays emit screen-bounded rects while deeper zooms emit raw rects.
constexpr uint64_t kTargetLocalReplayBins = 8192ULL;
// Skip sub-pixel binning on narrow (<= 100 ms) local replay windows, which
// main.ts may widen up to 100 ms for pan margin, so deep zooms always emit raw
// rects with full serial/uid detail metadata.
constexpr uint64_t kMinLocalReplayBinningWindowPs = 100ULL * 1000000000ULL;
// Bounds raw cached events decoded per step so binned wide replays stay
// responsive even when many raw events collapse into few output rects.
constexpr size_t kMaxCachedRawEventsPerStep = 262144;
constexpr absl::Duration kCachedStepWallClockBudget = absl::Milliseconds(8);
constexpr uint64_t kDurationPhaseBeginBit = 1ULL << 61;
constexpr uint64_t kDurationPhaseEndBit = 1ULL << 62;
constexpr uint64_t kDurationPhaseMask =
    kDurationPhaseBeginBit | kDurationPhaseEndBit;

inline void AppendVarint64(std::string& out, uint64_t v) {
  uint8_t buf[10];
  size_t n = 0;
  while (v >= 0x80) {
    buf[n++] = static_cast<uint8_t>(v | 0x80);
    v >>= 7;
  }
  buf[n++] = static_cast<uint8_t>(v);
  out.append(reinterpret_cast<const char*>(buf), n);
}

inline uint64_t ReadVarint64(const uint8_t*& p, const uint8_t* end) {
  uint64_t result = 0;
  int shift = 0;
  while (p < end && shift < 64) {
    const uint8_t byte = *p++;
    result |= static_cast<uint64_t>(byte & 0x7F) << shift;
    if ((byte & 0x80) == 0) {
      return result;
    }
    shift += 7;
  }
  return result;
}

inline uint64_t ZigZagEncode64(int64_t n) {
  return (static_cast<uint64_t>(n) << 1) ^ static_cast<uint64_t>(n >> 63);
}

inline int64_t ZigZagDecode64(uint64_t n) {
  return static_cast<int64_t>((n >> 1) ^ (~(n & 1) + 1));
}

struct CachedCompleteBlock {
  ProcessId pid = 0;
  ThreadId tid = 0;
  uint32_t frame_idx = 0;
  bool use_frame_strings = false;
  uint32_t event_count = 0;
  uint64_t base_ts_ps = 0;
  uint64_t min_start_ps = std::numeric_limits<uint64_t>::max();
  uint64_t max_end_ps = 0;
  bool has_split_events = false;
  std::string packed_events;
};

struct IncrementalParserSession {
  std::deque<std::pair<size_t, std::string>> pending_compressed_frames;
  std::vector<std::string> cumulative_interned_strings;
  std::vector<std::string> active_frame_interned_strings;
  bool use_active_frame_interned_strings = false;

  std::vector<std::string> cached_cumulative_interned_strings;
  std::vector<FrameSummary> cached_frame_summaries;
  std::vector<FrameSummary> building_frame_summaries;
  std::optional<std::pair<Milliseconds, Milliseconds>> cached_full_timespan;
  bool cached_mpmd_pipeline_view = false;
  size_t next_enqueued_frame_index = 0;
  size_t active_frame_index = 0;
  bool using_cached_cumulative_strings = false;
  bool recording_frame_summaries = false;

  // Compact in-memory block cache for sub-50ms zoom/pan refetches without
  // Zstd decompression or protobuf heap allocations.
  bool can_build_trace_store = true;
  bool overview_pre_compacted = false;
  size_t max_cached_packed_bytes = kMaxCachedPackedBytes;
  size_t building_packed_bytes = 0;
  std::vector<xprof::Process> building_processes;
  std::vector<TraceEvent> building_async_flame_events;
  std::vector<TraceEvent> building_async_flow_events;
  std::vector<CounterEvent> building_counter_events;
  std::vector<CachedCompleteBlock> building_complete_blocks;
  absl::flat_hash_map<size_t, std::vector<std::string>>
      building_frame_interned_strings;

  bool has_cached_trace_store = false;
  bool using_cached_trace_store = false;
  bool cached_metadata_and_globals_emitted = false;
  size_t next_cached_block_idx = 0;
  // Active enclosing parent end timestamps (ps) per (pid, tid), persisted
  // across 4,096-event blocks and split B/E spans during local replay.
  absl::flat_hash_map<std::pair<ProcessId, ThreadId>, std::vector<uint64_t>>
      replay_parent_ends_by_track;
  std::vector<xprof::Process> cached_processes;
  std::vector<TraceEvent> cached_async_flame_events;
  std::vector<TraceEvent> cached_async_flow_events;
  std::vector<CounterEvent> cached_counter_events;
  std::vector<CachedCompleteBlock> cached_complete_blocks;
  absl::flat_hash_map<size_t, std::vector<std::string>>
      cached_frame_interned_strings;

  std::unique_ptr<xprof::TraceDataResponse> active_response;
  std::vector<ProcessId> active_pids;
  size_t next_pid_idx = 0;
  bool current_pid_metadata_emitted = false;
  size_t current_complete_series_pos = 0;
  size_t current_complete_event_idx = 0;
  uint64_t current_complete_ts_ps = 0;
  std::vector<uint64_t> current_complete_parent_ends_ps;
  size_t current_async_series_pos = 0;
  size_t current_counter_series_pos = 0;
  absl::flat_hash_map<ProcessId, std::vector<int>> metadata_indices_by_pid;
  absl::flat_hash_map<ProcessId, std::vector<int>> complete_series_by_pid;
  absl::flat_hash_map<ProcessId, std::vector<int>> async_series_by_pid;
  absl::flat_hash_map<ProcessId, std::vector<int>> counter_series_by_pid;

  std::optional<std::pair<Milliseconds, Milliseconds>> visible_range_from_url;
  std::optional<std::pair<uint64_t, uint64_t>> filter_range_ps;
  std::optional<absl::flat_hash_set<std::pair<ProcessId, ThreadId>>>
      active_track_filter;
  std::optional<std::pair<Milliseconds, Milliseconds>> full_timespan;
  bool is_initial_load = true;
  bool mpmd_pipeline_view = false;
  bool appended_any_batch = false;
  // Complete, async, and counter events (excluding process/thread metadata)
  // unmarshaled so far, so metadata-only frames never flush an empty timeline.
  size_t track_events_unmarshaled = 0;
  size_t frames_flushed = 0;
  bool in_finish_drain = false;
  absl::Duration step_time_budget = absl::Milliseconds(8);
  absl::AnyInvocable<absl::Time() const> step_now_fn;
  tensorflow::profiler::ZstdDecompressorContext zstd_decompressor{
      /*initial_capacity=*/0};

  absl::Time StepNow() const {
    return step_now_fn ? step_now_fn() : absl::Now();
  }

  void ResetBuildingTraceStore() {
    can_build_trace_store = true;
    building_packed_bytes = 0;
    building_processes.clear();
    building_async_flame_events.clear();
    building_async_flow_events.clear();
    building_counter_events.clear();
    building_complete_blocks.clear();
    building_frame_interned_strings.clear();
  }

  void Reset() {
    pending_compressed_frames.clear();
    cumulative_interned_strings.clear();
    active_frame_interned_strings.clear();
    use_active_frame_interned_strings = false;
    building_frame_summaries.clear();
    overview_pre_compacted = false;
    ResetBuildingTraceStore();
    next_enqueued_frame_index = 0;
    active_frame_index = 0;
    using_cached_cumulative_strings = false;
    using_cached_trace_store = false;
    cached_metadata_and_globals_emitted = false;
    next_cached_block_idx = 0;
    replay_parent_ends_by_track.clear();
    recording_frame_summaries = false;
    ClearActiveFrame();
    visible_range_from_url = std::nullopt;
    filter_range_ps = std::nullopt;
    active_track_filter = std::nullopt;
    full_timespan = std::nullopt;
    is_initial_load = true;
    mpmd_pipeline_view = false;
    appended_any_batch = false;
    track_events_unmarshaled = 0;
    frames_flushed = 0;
    in_finish_drain = false;
    zstd_decompressor.ReleaseBuffer();
  }

  void ClearPersistentCache() {
    cached_cumulative_interned_strings.clear();
    cached_frame_summaries.clear();
    cached_full_timespan = std::nullopt;
    cached_mpmd_pipeline_view = false;
    has_cached_trace_store = false;
    using_cached_trace_store = false;
    cached_processes.clear();
    cached_async_flame_events.clear();
    cached_async_flow_events.clear();
    cached_counter_events.clear();
    cached_complete_blocks.clear();
    cached_frame_interned_strings.clear();
    zstd_decompressor.ReleaseBuffer();
  }

  void ResetCurrentPidCursors() {
    current_pid_metadata_emitted = false;
    current_complete_series_pos = 0;
    current_complete_event_idx = 0;
    current_complete_ts_ps = 0;
    current_complete_parent_ends_ps.clear();
    current_async_series_pos = 0;
    current_counter_series_pos = 0;
  }

  void ClearActiveFrame() {
    active_response.reset();
    active_pids.clear();
    next_pid_idx = 0;
    ResetCurrentPidCursors();
    metadata_indices_by_pid.clear();
    complete_series_by_pid.clear();
    async_series_by_pid.clear();
    counter_series_by_pid.clear();
  }

  absl::Span<const std::string> CurrentInternedStrings() const {
    if (use_active_frame_interned_strings) {
      return absl::MakeConstSpan(active_frame_interned_strings);
    }
    if (using_cached_cumulative_strings) {
      return absl::MakeConstSpan(cached_cumulative_interned_strings);
    }
    return absl::MakeConstSpan(cumulative_interned_strings);
  }
};

IncrementalParserSession& GetIncrementalSession() {
  static absl::NoDestructor<IncrementalParserSession> session;
  return *session;
}

void AppendCompleteSeriesSliceToBlockCache(
    IncrementalParserSession& session, const xprof::TraceEventSeries& series,
    size_t slice_start_idx, size_t slice_count, uint64_t running_ts_ps) {
  if (!session.can_build_trace_store || slice_count == 0) {
    return;
  }
  const ProcessId pid = series.metadata().process_id();
  const ThreadId tid = series.metadata().thread_id();
  const size_t slice_end_idx = slice_start_idx + slice_count;

  for (size_t b_start = slice_start_idx; b_start < slice_end_idx;
       b_start += kCachedBlockEvents) {
    const size_t b_end = std::min(b_start + kCachedBlockEvents, slice_end_idx);
    CachedCompleteBlock block;
    block.pid = pid;
    block.tid = tid;
    block.frame_idx = static_cast<uint32_t>(session.active_frame_index);
    block.use_frame_strings = session.use_active_frame_interned_strings;
    block.event_count = static_cast<uint32_t>(b_end - b_start);
    block.packed_events.reserve(block.event_count * 7);

    uint32_t prev_serial = 0;
    for (size_t idx = b_start; idx < b_end; ++idx) {
      const int i = static_cast<int>(idx);
      const uint64_t d_ts = series.deltas(i);
      running_ts_ps += d_ts;
      const uint64_t delta_ts_ps = (idx == b_start) ? 0ULL : d_ts;
      if (idx == b_start) {
        block.base_ts_ps = running_ts_ps;
      }

      const uint64_t raw_dur_ps = series.durations(i);
      const uint64_t dur_ps = raw_dur_ps & ~kDurationPhaseMask;
      const bool is_begin = (raw_dur_ps & kDurationPhaseBeginBit) != 0;
      const bool is_end = (raw_dur_ps & kDurationPhaseEndBit) != 0;
      block.min_start_ps = std::min(block.min_start_ps, running_ts_ps);
      block.max_end_ps = std::max(block.max_end_ps, running_ts_ps + dur_ps);
      if (is_begin || is_end) {
        block.has_split_events = true;
      }

      const auto& ev_meta = series.event_metadata(i);
      const bool has_flow = (ev_meta.flow_id() != 0);
      const bool ns_scaled =
          (delta_ts_ps % 1000ULL == 0) && (dur_ps % 1000ULL == 0);
      const uint64_t packed_delta_ts =
          ns_scaled ? (delta_ts_ps / 1000ULL) : delta_ts_ps;
      const uint64_t packed_dur = ns_scaled ? (dur_ps / 1000ULL) : dur_ps;
      const uint64_t dur_and_flags =
          (packed_dur << 4) | (is_begin ? 1ULL : 0ULL) |
          (is_end ? 2ULL : 0ULL) | (has_flow ? 4ULL : 0ULL) |
          (ns_scaled ? 8ULL : 0ULL);
      const uint32_t name_ref = series.name_refs(i);
      const uint32_t serial = ev_meta.serial();
      const int64_t serial_delta =
          static_cast<int64_t>(serial) - static_cast<int64_t>(prev_serial);
      prev_serial = serial;

      AppendVarint64(block.packed_events, packed_delta_ts);
      AppendVarint64(block.packed_events, dur_and_flags);
      AppendVarint64(block.packed_events, name_ref);
      AppendVarint64(block.packed_events, ZigZagEncode64(serial_delta));
      if (has_flow) {
        AppendVarint64(block.packed_events, ev_meta.flow_id());
        AppendVarint64(block.packed_events, ev_meta.flow_category());
      }
    }

    if (session.recording_frame_summaries &&
        session.active_frame_index < session.building_frame_summaries.size() &&
        block.min_start_ps <= block.max_end_ps) {
      FrameSummary& summary =
          session.building_frame_summaries[session.active_frame_index];
      if (block.has_split_events) {
        summary.always_process = true;
      } else {
        summary.min_start_ps =
            std::min(summary.min_start_ps, block.min_start_ps);
        summary.max_end_ps = std::max(summary.max_end_ps, block.max_end_ps);
        summary.has_non_split_bounds = true;
      }
    }

    session.building_packed_bytes += block.packed_events.size();
    if (session.building_packed_bytes > session.max_cached_packed_bytes) {
      session.ResetBuildingTraceStore();
      session.can_build_trace_store = false;
      return;
    }
    session.building_complete_blocks.push_back(std::move(block));
  }
}

bool BlockOverlapsFilter(const CachedCompleteBlock& block,
                         const IncrementalParserSession& session) {
  if (session.active_track_filter.has_value() &&
      !session.active_track_filter->contains({block.pid, block.tid})) {
    return false;
  }
  if (!session.filter_range_ps.has_value()) {
    return true;
  }
  if (block.min_start_ps > block.max_end_ps) {
    return false;
  }
  return !(block.max_end_ps < session.filter_range_ps->first ||
           block.min_start_ps > session.filter_range_ps->second);
}

void AppendFlameEvent(ParsedTraceEvents& batch, ProcessId pid, ThreadId tid,
                      Phase phase, uint64_t start_ps, uint64_t dur_ps,
                      uint32_t name_ref, std::optional<uint32_t> serial,
                      uint64_t flow_id, uint32_t flow_category,
                      absl::Span<const std::string> interned_strings) {
  TraceEvent ev;
  ev.ph = phase;
  ev.pid = pid;
  ev.tid = tid;
  ev.ts = start_ps / 1000000.0;
  ev.dur = dur_ps / 1000000.0;
  if (name_ref < interned_strings.size()) {
    ev.name = interned_strings[name_ref];
  }
  if (flow_id != 0) {
    ev.flow_id = flow_id;
    ev.has_flow_id = true;
    if (flow_category != 0 && flow_category < interned_strings.size()) {
      ev.category = GetContextTypeFromString(interned_strings[flow_category]);
    }
  }
  if (serial.has_value()) {
    ev.serial = *serial;
    ev.has_serial = true;
  } else {
    // Synthetic coalesced bursts do not correspond to a single LevelDB row;
    // omit uid so detail lookups do not query a mismatched (ts, dur, serial).
    ev.serial = 0;
    ev.has_serial = false;
  }
  ev.event_id = GenerateEventId(ev.name, ev.ts, ev.dur);
  // Flows are registered by DataProvider from flame events carrying
  // `has_flow_id`, so they are not duplicated into `flow_events` here.
  batch.flame_events.push_back(std::move(ev));
}

bool OverlapsSubsequentEvent(const IncrementalParserSession& session,
                             size_t block_idx, uint32_t event_idx,
                             const uint8_t* next_p, const uint8_t* end,
                             uint64_t dur_ps, uint64_t end_ps) {
  const CachedCompleteBlock& block = session.cached_complete_blocks[block_idx];
  if (event_idx + 1 < block.event_count && next_p < end) {
    const uint8_t* peek = next_p;
    const uint64_t next_packed_delta = ReadVarint64(peek, end);
    const uint64_t next_dur_and_flags = ReadVarint64(peek, end);
    const uint64_t next_delta_ps = ((next_dur_and_flags & 8ULL) != 0)
                                       ? (next_packed_delta * 1000ULL)
                                       : next_packed_delta;
    return dur_ps > next_delta_ps;
  }
  if (dur_ps > 0 && block_idx + 1 < session.cached_complete_blocks.size()) {
    const CachedCompleteBlock& next_block =
        session.cached_complete_blocks[block_idx + 1];
    if (next_block.pid == block.pid && next_block.tid == block.tid) {
      return end_ps > next_block.base_ts_ps;
    }
    // A later frame may continue this track; preserve the trailing non-zero
    // span so it cannot flatten into cross-frame children.
    return true;
  }
  return false;
}

class LeafBurstCoalescer {
 public:
  LeafBurstCoalescer(ProcessId pid, ThreadId tid, uint64_t min_bin_ps,
                     absl::Span<const std::string> interned_strings,
                     ParsedTraceEvents& batch, size_t& emitted)
      : pid_(pid),
        tid_(tid),
        min_bin_ps_(min_bin_ps),
        interned_strings_(interned_strings),
        batch_(batch),
        emitted_(emitted) {}

  void Add(uint64_t bin, uint64_t start_ps, uint64_t dur_ps, uint32_t name_ref,
           uint32_t serial) {
    const uint64_t end_ps = start_ps + dur_ps;
    uint64_t gap_ps = 0;
    if (in_burst_) {
      gap_ps = start_ps > end_ps_ ? (start_ps - end_ps_) : 0ULL;
      const uint64_t active_floor_ps =
          std::max<uint64_t>(active_dur_ps_, 500000ULL);
      const uint64_t max_allowed_gap_ps =
          std::min<uint64_t>(min_bin_ps_ / 4, active_floor_ps);
      if (bin != bin_ || gap_ps > max_allowed_gap_ps ||
          gap_ps_ + gap_ps > active_floor_ps) {
        Flush();
      }
    }
    if (!in_burst_) {
      in_burst_ = true;
      count_ = 1;
      bin_ = bin;
      start_ps_ = start_ps;
      end_ps_ = end_ps;
      active_dur_ps_ = dur_ps;
      gap_ps_ = 0;
      dom_dur_ps_ = dur_ps;
      dom_name_ref_ = name_ref;
      dom_serial_ = serial;
      return;
    }
    const uint64_t added_active_ps =
        end_ps > std::max(start_ps, end_ps_)
            ? (end_ps - std::max(start_ps, end_ps_))
            : 0ULL;
    ++count_;
    end_ps_ = std::max(end_ps_, end_ps);
    active_dur_ps_ += added_active_ps;
    gap_ps_ += gap_ps;
    if (dur_ps >= dom_dur_ps_) {
      dom_dur_ps_ = dur_ps;
      dom_name_ref_ = name_ref;
      dom_serial_ = serial;
    }
  }

  void Flush() {
    if (!in_burst_) return;
    const uint64_t dur_ps =
        count_ == 1
            ? dom_dur_ps_
            : (end_ps_ > start_ps_ ? (end_ps_ - start_ps_) : dom_dur_ps_);
    const std::optional<uint32_t> serial =
        count_ == 1 ? std::make_optional(dom_serial_) : std::nullopt;
    AppendFlameEvent(batch_, pid_, tid_, Phase::kComplete, start_ps_, dur_ps,
                     dom_name_ref_, serial, /*flow_id=*/0, /*flow_category=*/0,
                     interned_strings_);
    ++emitted_;
    in_burst_ = false;
    count_ = 0;
  }

 private:
  ProcessId pid_;
  ThreadId tid_;
  uint64_t min_bin_ps_;
  absl::Span<const std::string> interned_strings_;
  ParsedTraceEvents& batch_;
  size_t& emitted_;

  bool in_burst_ = false;
  size_t count_ = 0;
  uint64_t bin_ = 0;
  uint64_t start_ps_ = 0;
  uint64_t end_ps_ = 0;
  uint64_t active_dur_ps_ = 0;
  uint64_t gap_ps_ = 0;
  uint64_t dom_dur_ps_ = 0;
  uint32_t dom_name_ref_ = 0;
  uint32_t dom_serial_ = 0;
};

size_t DecodeCachedCompleteBlock(IncrementalParserSession& session,
                                 size_t block_idx, ParsedTraceEvents& batch,
                                 size_t& raw_events_scanned) {
  const CachedCompleteBlock& block = session.cached_complete_blocks[block_idx];
  if (!BlockOverlapsFilter(block, session)) {
    return 0;
  }
  absl::Span<const std::string> interned_strings =
      absl::MakeConstSpan(session.cached_cumulative_interned_strings);
  if (block.use_frame_strings) {
    if (const auto it =
            session.cached_frame_interned_strings.find(block.frame_idx);
        it != session.cached_frame_interned_strings.end()) {
      interned_strings = absl::MakeConstSpan(it->second);
    }
  }

  const uint8_t* p =
      reinterpret_cast<const uint8_t*>(block.packed_events.data());
  const uint8_t* end = p + block.packed_events.size();
  uint64_t current_ts_ps = block.base_ts_ps;
  uint32_t prev_serial = 0;
  size_t emitted = 0;

  uint64_t filter_start_ps = 0;
  uint64_t filter_end_ps = 0;
  uint64_t min_bin_ps = 0;
  if (session.filter_range_ps.has_value()) {
    filter_start_ps = session.filter_range_ps->first;
    filter_end_ps = session.filter_range_ps->second;
    if (filter_end_ps > filter_start_ps &&
        (filter_end_ps - filter_start_ps) > kMinLocalReplayBinningWindowPs) {
      min_bin_ps = (filter_end_ps - filter_start_ps) / kTargetLocalReplayBins;
    }
  }

  std::vector<uint64_t>* parent_ends_ptr = nullptr;
  if (min_bin_ps > 0) {
    parent_ends_ptr =
        &session.replay_parent_ends_by_track[{block.pid, block.tid}];
    if (parent_ends_ptr->capacity() < 16) {
      parent_ends_ptr->reserve(16);
    }
  }
  LeafBurstCoalescer burst_coalescer(block.pid, block.tid, min_bin_ps,
                                     interned_strings, batch, emitted);

  for (uint32_t i = 0; i < block.event_count && p < end; ++i) {
    ++raw_events_scanned;
    const uint64_t packed_delta_ts = ReadVarint64(p, end);
    const uint64_t dur_and_flags = ReadVarint64(p, end);
    const bool ns_scaled = (dur_and_flags & 8ULL) != 0;
    const uint64_t delta_ts_ps =
        ns_scaled ? (packed_delta_ts * 1000ULL) : packed_delta_ts;
    if (i > 0) {
      current_ts_ps += delta_ts_ps;
    }
    const uint32_t name_ref = static_cast<uint32_t>(ReadVarint64(p, end));
    const int64_t serial_delta = ZigZagDecode64(ReadVarint64(p, end));
    const uint32_t serial =
        static_cast<uint32_t>(static_cast<int64_t>(prev_serial) + serial_delta);
    prev_serial = serial;

    const bool has_flow = (dur_and_flags & 4ULL) != 0;
    uint64_t flow_id = 0;
    uint32_t flow_category = 0;
    if (has_flow) {
      flow_id = ReadVarint64(p, end);
      flow_category = static_cast<uint32_t>(ReadVarint64(p, end));
    }

    const uint64_t packed_dur = dur_and_flags >> 4;
    const uint64_t dur_ps = ns_scaled ? (packed_dur * 1000ULL) : packed_dur;
    if (session.filter_range_ps.has_value()) {
      if (current_ts_ps > filter_end_ps) {
        break;
      }
      if (current_ts_ps + dur_ps < filter_start_ps) {
        continue;
      }
    }

    const bool is_begin = (dur_and_flags & 1ULL) != 0;
    const bool is_end = (dur_and_flags & 2ULL) != 0;
    if (min_bin_ps > 0) {
      std::vector<uint64_t>& parent_ends = *parent_ends_ptr;
      bool parent_boundary_crossed = false;
      while (!parent_ends.empty() && parent_ends.back() <= current_ts_ps) {
        parent_ends.pop_back();
        parent_boundary_crossed = true;
      }
      if (parent_boundary_crossed) {
        burst_coalescer.Flush();
      }

      const uint64_t end_ps = current_ts_ps + dur_ps;
      const bool overlaps_next = OverlapsSubsequentEvent(
          session, block_idx, i, p, end, dur_ps, end_ps);
      const bool exceeds_parent =
          !parent_ends.empty() && end_ps > parent_ends.back();
      // Coalesce sub-pixel leaf rects within the same time bin while keeping
      // non-leaf parents, wide spans, split spans, flow endpoints, and
      // parent-crossing spans intact.
      if (dur_ps < min_bin_ps && !overlaps_next && !is_begin && !is_end &&
          !has_flow && !exceeds_parent) {
        const uint64_t bin =
            current_ts_ps >= filter_start_ps
                ? (current_ts_ps - filter_start_ps) / min_bin_ps
                : 0ULL;
        burst_coalescer.Add(bin, current_ts_ps, dur_ps, name_ref, serial);
        continue;
      }

      burst_coalescer.Flush();
      if (dur_ps > 0 && (is_begin || is_end)) {
        // Split B/E spans re-emit an updated end bound across chunks; drop any
        // narrower stale chunk boundary before pushing the extended bound.
        while (!parent_ends.empty() && parent_ends.back() < end_ps) {
          parent_ends.pop_back();
        }
        parent_ends.push_back(end_ps);
      } else if (!is_begin && !is_end &&
                 (overlaps_next || dur_ps >= min_bin_ps)) {
        parent_ends.push_back(end_ps);
      }
    }

    const Phase phase = is_begin
                            ? Phase::kDurationBegin
                            : (is_end ? Phase::kDurationEnd : Phase::kComplete);
    AppendFlameEvent(batch, block.pid, block.tid, phase, current_ts_ps, dur_ps,
                     name_ref, serial, flow_id, flow_category,
                     interned_strings);
    ++emitted;
  }
  burst_coalescer.Flush();
  return emitted;
}

bool CanSkipFrameForFilter(const IncrementalParserSession& session,
                           size_t frame_idx) {
  if (!session.filter_range_ps.has_value() ||
      !session.using_cached_cumulative_strings ||
      frame_idx >= session.cached_frame_summaries.size()) {
    return false;
  }
  const FrameSummary& summary = session.cached_frame_summaries[frame_idx];
  if (summary.always_process || !summary.has_non_split_bounds) {
    return false;
  }
  constexpr uint64_t kFrameFilterMarginPs = 25000000ULL;  // 25 ms
  const uint64_t filter_start_ps =
      session.filter_range_ps->first > kFrameFilterMarginPs
          ? (session.filter_range_ps->first - kFrameFilterMarginPs)
          : 0;
  const uint64_t filter_end_ps =
      session.filter_range_ps->second + kFrameFilterMarginPs;
  return summary.max_end_ps < filter_start_ps ||
         summary.min_start_ps > filter_end_ps;
}

bool ActivateNextCompressedFrame(IncrementalParserSession& session,
                                 bool& header_only_frame_activated) {
  header_only_frame_activated = false;
  session.ClearActiveFrame();
  while (!session.pending_compressed_frames.empty()) {
    auto [frame_idx, compressed] =
        std::move(session.pending_compressed_frames.front());
    session.pending_compressed_frames.pop_front();

    if (CanSkipFrameForFilter(session, frame_idx)) {
      continue;
    }

    absl::StatusOr<absl::string_view> decompressed_view =
        session.zstd_decompressor.Decompress(compressed);
    std::string().swap(compressed);
    std::unique_ptr<xprof::TraceDataResponse> response;
    if (decompressed_view.ok()) {
      response = std::make_unique<xprof::TraceDataResponse>();
      if (!response->ParseFromString(*decompressed_view)) {
        response.reset();
      }
    }
    if (session.zstd_decompressor.capacity() >
        tensorflow::profiler::ZstdDecompressorContext::kDefaultBufferSize) {
      session.zstd_decompressor.ReleaseBuffer();
    }
    if (response == nullptr) {
      continue;
    }

    session.active_frame_index = frame_idx;
    if (session.recording_frame_summaries) {
      if (session.building_frame_summaries.size() <= frame_idx) {
        session.building_frame_summaries.resize(frame_idx + 1);
      }
      if (response->async_events_size() > 0 ||
          response->counter_events_size() > 0 ||
          response->complete_events_size() == 0) {
        session.building_frame_summaries[frame_idx].always_process = true;
      }
    }

    DispatchResponseDetails(*response, session.mpmd_pipeline_view);

    if (response->has_full_timespan_start_ps() &&
        response->has_full_timespan_end_ps()) {
      const Milliseconds start_ms = response->full_timespan_start_ps() / 1e9;
      const Milliseconds end_ms = response->full_timespan_end_ps() / 1e9;
      if (start_ms <= end_ms) {
        session.full_timespan = std::make_pair(start_ms, end_ms);
      }
    }

    if (session.using_cached_cumulative_strings) {
      if (frame_idx > 0 && response->interned_strings_size() > 0 &&
          response->interned_strings(0).empty()) {
        session.use_active_frame_interned_strings = true;
        session.active_frame_interned_strings.clear();
        session.active_frame_interned_strings.reserve(
            response->interned_strings_size());
        for (std::string& s : *response->mutable_interned_strings()) {
          session.active_frame_interned_strings.push_back(std::move(s));
        }
      } else {
        session.use_active_frame_interned_strings = false;
      }
      response->clear_interned_strings();
    } else if (session.cumulative_interned_strings.empty() ||
               (response->interned_strings_size() > 0 &&
                !response->interned_strings(0).empty())) {
      session.use_active_frame_interned_strings = false;
      session.cumulative_interned_strings.reserve(
          session.cumulative_interned_strings.size() +
          response->interned_strings_size());
      for (std::string& s : *response->mutable_interned_strings()) {
        session.cumulative_interned_strings.push_back(std::move(s));
      }
      response->clear_interned_strings();
    } else if (response->interned_strings_size() > 0 &&
               response->interned_strings(0).empty()) {
      session.use_active_frame_interned_strings = true;
      session.active_frame_interned_strings.clear();
      session.active_frame_interned_strings.reserve(
          response->interned_strings_size());
      for (std::string& s : *response->mutable_interned_strings()) {
        session.active_frame_interned_strings.push_back(std::move(s));
      }
      response->clear_interned_strings();
    } else {
      session.use_active_frame_interned_strings = false;
    }

    if (session.recording_frame_summaries && session.can_build_trace_store &&
        session.use_active_frame_interned_strings) {
      session.building_frame_interned_strings[frame_idx] =
          session.active_frame_interned_strings;
    }

    absl::flat_hash_set<ProcessId> seen_pids;
    auto register_pid = [&](ProcessId pid) {
      if (seen_pids.insert(pid).second) {
        session.active_pids.push_back(pid);
      }
    };

    const auto& processes = response->metadata().processes();
    for (int i = 0; i < processes.size(); ++i) {
      const ProcessId pid = processes[i].id();
      register_pid(pid);
      session.metadata_indices_by_pid[pid].push_back(i);
      if (session.recording_frame_summaries && session.can_build_trace_store) {
        session.building_processes.push_back(processes[i]);
      }
    }
    for (int i = 0; i < response->complete_events_size(); ++i) {
      const ProcessId pid =
          response->complete_events(i).metadata().process_id();
      register_pid(pid);
      session.complete_series_by_pid[pid].push_back(i);
    }
    for (int i = 0; i < response->async_events_size(); ++i) {
      const ProcessId pid = response->async_events(i).metadata().process_id();
      register_pid(pid);
      session.async_series_by_pid[pid].push_back(i);
    }
    for (int i = 0; i < response->counter_events_size(); ++i) {
      const ProcessId pid = response->counter_events(i).metadata().process_id();
      register_pid(pid);
      session.counter_series_by_pid[pid].push_back(i);
    }

    if (session.active_pids.empty()) {
      header_only_frame_activated = true;
      return true;
    }

    session.active_response = std::move(response);
    session.next_pid_idx = 0;
    session.ResetCurrentPidCursors();
    return true;
  }
  return false;
}

bool UnmarshalProcessSliceBounded(IncrementalParserSession& session,
                                  ProcessId pid, size_t max_events_budget,
                                  size_t max_raw_events_budget,
                                  absl::Time step_deadline, bool& made_progress,
                                  size_t& events_unmarshaled,
                                  size_t& raw_events_scanned,
                                  ParsedTraceEvents& batch) {
  xprof::TraceDataResponse& response = *session.active_response;
  const absl::Span<const std::string> interned_strings =
      session.CurrentInternedStrings();
  const auto within_step_time_budget = [&]() {
    return !made_progress || session.StepNow() < step_deadline;
  };

  if (!session.current_pid_metadata_emitted) {
    session.current_pid_metadata_emitted = true;
    if (const auto it = session.metadata_indices_by_pid.find(pid);
        it != session.metadata_indices_by_pid.end()) {
      for (const int idx : it->second) {
        ProcessProcessMetadata(response.metadata().processes(idx), batch);
      }
    }
  }

  constexpr size_t kMaxTimedSeriesSliceEvents = 32768;
  const bool timed_step = step_deadline < absl::InfiniteFuture();
  if (const auto it = session.complete_series_by_pid.find(pid);
      it != session.complete_series_by_pid.end()) {
    const std::vector<int>& series_indices = it->second;
    while (session.current_complete_series_pos < series_indices.size() &&
           events_unmarshaled < max_events_budget &&
           raw_events_scanned < max_raw_events_budget &&
           within_step_time_budget()) {
      const int idx = series_indices[session.current_complete_series_pos];
      const auto& series = response.complete_events(idx);
      if (session.active_track_filter.has_value() &&
          !session.active_track_filter->contains(
              {pid, series.metadata().thread_id()})) {
        response.mutable_complete_events(idx)->Clear();
        ++session.current_complete_series_pos;
        session.current_complete_event_idx = 0;
        session.current_complete_ts_ps = 0;
        session.current_complete_parent_ends_ps.clear();
        made_progress = true;
        continue;
      }
      const size_t total_in_series = static_cast<size_t>(series.deltas_size());
      const size_t remaining_budget = max_events_budget - events_unmarshaled;
      const size_t raw_slice_limit =
          timed_step ? std::min<size_t>(std::max<size_t>(remaining_budget,
                                                         kCachedBlockEvents),
                                        kMaxTimedSeriesSliceEvents)
                     : (session.filter_range_ps.has_value()
                            ? std::max<size_t>(remaining_budget, 262144)
                            : std::max<size_t>(remaining_budget, 65536));
      const size_t slice_limit =
          ((raw_slice_limit + kCachedBlockEvents - 1) / kCachedBlockEvents) *
          kCachedBlockEvents;
      const size_t slice_start_idx = session.current_complete_event_idx;
      const uint64_t slice_start_ts_ps = session.current_complete_ts_ps;
      const size_t before = batch.flame_events.size();
      size_t count = 0;
      const bool can_use_overview_fast_path =
          !session.filter_range_ps.has_value() &&
          session.recording_frame_summaries && session.can_build_trace_store &&
          total_in_series > 8192;
      if (can_use_overview_fast_path) {
        const size_t planned_count =
            std::min(slice_limit, total_in_series - slice_start_idx);
        AppendCompleteSeriesSliceToBlockCache(session, series, slice_start_idx,
                                              planned_count, slice_start_ts_ps);
        if (session.can_build_trace_store) {
          bool did_coalesce = false;
          count = ProcessOverviewSampledSeriesSlice(
              series, interned_strings, session.current_complete_event_idx,
              slice_limit, session.current_complete_ts_ps, batch, did_coalesce,
              &session.current_complete_parent_ends_ps);
          if (did_coalesce) {
            session.overview_pre_compacted = true;
          }
        } else {
          count = ProcessCompleteEventSeriesSlice(
              series, interned_strings, session.current_complete_event_idx,
              slice_limit, session.current_complete_ts_ps, batch,
              session.filter_range_ps);
        }
      } else {
        count = ProcessCompleteEventSeriesSlice(
            series, interned_strings, session.current_complete_event_idx,
            slice_limit, session.current_complete_ts_ps, batch,
            session.filter_range_ps);
        if (session.recording_frame_summaries &&
            session.can_build_trace_store && count > 0) {
          AppendCompleteSeriesSliceToBlockCache(
              session, series, slice_start_idx, count, slice_start_ts_ps);
        }
      }
      if (session.recording_frame_summaries && !session.can_build_trace_store &&
          session.active_frame_index <
              session.building_frame_summaries.size()) {
        FrameSummary& summary =
            session.building_frame_summaries[session.active_frame_index];
        for (size_t i = before; i < batch.flame_events.size(); ++i) {
          const TraceEvent& ev = batch.flame_events[i];
          if (ev.ph != Phase::kComplete && ev.ph != Phase::kDurationEnd) {
            continue;
          }
          const uint64_t start_ps =
              static_cast<uint64_t>(std::max(0.0, ev.ts) * 1e6);
          const uint64_t end_ps =
              static_cast<uint64_t>(std::max(0.0, ev.ts + ev.dur) * 1e6);
          summary.min_start_ps = std::min(summary.min_start_ps, start_ps);
          summary.max_end_ps = std::max(summary.max_end_ps, end_ps);
          summary.has_non_split_bounds = true;
        }
      }
      const size_t added = batch.flame_events.size() - before;
      events_unmarshaled += added;
      const size_t filtered_scan_floor = timed_step ? 512 : 4096;
      raw_events_scanned +=
          session.filter_range_ps.has_value()
              ? std::min<size_t>(count, added + filtered_scan_floor)
              : count;
      made_progress = true;
      session.current_complete_event_idx += count;
      if (session.current_complete_event_idx >= total_in_series) {
        response.mutable_complete_events(idx)->Clear();
        ++session.current_complete_series_pos;
        session.current_complete_event_idx = 0;
        session.current_complete_ts_ps = 0;
        session.current_complete_parent_ends_ps.clear();
      }
    }
    if (session.current_complete_series_pos < series_indices.size()) {
      return false;
    }
  }

  if (const auto it = session.async_series_by_pid.find(pid);
      it != session.async_series_by_pid.end()) {
    const std::vector<int>& series_indices = it->second;
    while (session.current_async_series_pos < series_indices.size() &&
           events_unmarshaled < max_events_budget &&
           raw_events_scanned < max_raw_events_budget &&
           within_step_time_budget()) {
      const int idx = series_indices[session.current_async_series_pos++];
      const size_t raw_deltas =
          static_cast<size_t>(response.async_events(idx).deltas_size());
      const size_t before_flame = batch.flame_events.size();
      const size_t before_flow = batch.flow_events.size();
      ProcessAsyncEventSeries(response.async_events(idx), interned_strings,
                              batch);
      if (session.recording_frame_summaries && session.can_build_trace_store) {
        session.building_async_flame_events.insert(
            session.building_async_flame_events.end(),
            batch.flame_events.begin() + before_flame,
            batch.flame_events.end());
        session.building_async_flow_events.insert(
            session.building_async_flow_events.end(),
            batch.flow_events.begin() + before_flow, batch.flow_events.end());
      }
      response.mutable_async_events(idx)->Clear();
      const size_t added = batch.flame_events.size() - before_flame;
      events_unmarshaled += added;
      raw_events_scanned += std::max(added, raw_deltas);
      made_progress = true;
    }
    if (session.current_async_series_pos < series_indices.size()) {
      return false;
    }
  }

  if (const auto it = session.counter_series_by_pid.find(pid);
      it != session.counter_series_by_pid.end()) {
    const std::vector<int>& series_indices = it->second;
    while (session.current_counter_series_pos < series_indices.size() &&
           events_unmarshaled < max_events_budget &&
           raw_events_scanned < max_raw_events_budget &&
           within_step_time_budget()) {
      const int idx = series_indices[session.current_counter_series_pos++];
      const size_t raw_deltas =
          static_cast<size_t>(response.counter_events(idx).deltas_size());
      const size_t before = batch.counter_events.size();
      ProcessCounterEventSeries(response.counter_events(idx), interned_strings,
                                batch);
      if (session.recording_frame_summaries && session.can_build_trace_store) {
        session.building_counter_events.insert(
            session.building_counter_events.end(),
            batch.counter_events.begin() + before, batch.counter_events.end());
      }
      response.mutable_counter_events(idx)->Clear();
      const size_t added = batch.counter_events.size() - before;
      events_unmarshaled += added;
      raw_events_scanned += std::max(added, raw_deltas);
      made_progress = true;
    }
    if (session.current_counter_series_pos < series_indices.size()) {
      return false;
    }
  }

  return true;
}

}  // namespace

void BeginIncrementalTraceEvents(
    std::optional<std::pair<Milliseconds, Milliseconds>> visible_range_from_url,
    DataProvider& data_provider, Timeline& timeline) {
  IncrementalParserSession& session = GetIncrementalSession();
  session.Reset();
  session.is_initial_load = !data_provider.HasKnownTracks() ||
                            (timeline.visible_range() == TimeRange::Zero());
  session.visible_range_from_url = visible_range_from_url;
  if (!session.is_initial_load && session.visible_range_from_url.has_value()) {
    const Milliseconds start_ms = session.visible_range_from_url->first;
    const Milliseconds end_ms = session.visible_range_from_url->second;
    const uint64_t start_ps =
        static_cast<uint64_t>(std::max(0.0, start_ms) * 1e9);
    const uint64_t end_ps = static_cast<uint64_t>(std::max(0.0, end_ms) * 1e9);
    if (start_ps <= end_ps) {
      session.filter_range_ps = std::make_pair(start_ps, end_ps);
    }
  }
  if (session.is_initial_load) {
    session.ClearPersistentCache();
    session.recording_frame_summaries = true;
    session.can_build_trace_store = true;
  } else {
    session.can_build_trace_store = false;
    session.using_cached_trace_store = session.has_cached_trace_store;
    if (!session.cached_cumulative_interned_strings.empty()) {
      session.mpmd_pipeline_view = session.cached_mpmd_pipeline_view;
      session.full_timespan = session.cached_full_timespan;
    }
  }
  if (session.filter_range_ps.has_value()) {
    absl::flat_hash_set<std::pair<ProcessId, ThreadId>> active_tracks =
        timeline.ComputeActiveTracksForFetch(/*include_vertical_margin=*/true);
    absl::flat_hash_set<ProcessId> active_pids;
    for (const auto& [pid, _] : active_tracks) {
      active_pids.insert(pid);
    }
    for (const ProcessId pid : active_pids) {
      if (const auto xla_tid = data_provider.GetXlaModulesTid(pid);
          xla_tid.has_value()) {
        active_tracks.insert({pid, *xla_tid});
      }
    }
    if (!active_tracks.empty()) {
      timeline.SetLoadedActiveTracks(active_tracks);
      session.active_track_filter = std::move(active_tracks);
    } else {
      timeline.ClearLoadedActiveTracks();
    }
    timeline.PrepareForFilteredRefetch();
  } else {
    timeline.ClearLoadedActiveTracks();
    timeline.set_stash_full_overview_on_next_update(false);
  }
  timeline.set_full_trace_loaded(false);
  timeline.set_is_compacted(false);
  timeline.set_window_at_full_resolution(false);
  timeline.set_is_incremental_loading(true);
  data_provider.BeginIncrementalLoad(timeline);
}

#ifdef __EMSCRIPTEN__
void BeginIncrementalTraceEvents(const emscripten::val& visible_range_from_url,
                                 DataProvider& data_provider,
                                 Timeline& timeline) {
  BeginIncrementalTraceEvents(ParseVisibleRangeFromUrl(visible_range_from_url),
                              data_provider, timeline);
}

void BeginIncrementalTraceEvents(
    const emscripten::val& visible_range_from_url) {
  BeginIncrementalTraceEvents(visible_range_from_url,
                              Application::Instance().data_provider(),
                              Application::Instance().timeline());
}
#endif

bool HasCachedTraceEventsIncremental() {
  return GetIncrementalSession().has_cached_trace_store;
}

bool HasFlushedIncrementalTraceEvents() {
  return GetIncrementalSession().frames_flushed > 0;
}

void SetMaxCachedPackedBytesForTest(size_t max_bytes) {
  GetIncrementalSession().max_cached_packed_bytes = max_bytes;
}

void SetIncrementalStepTimeBudgetForTest(absl::Duration budget) {
  GetIncrementalSession().step_time_budget = budget;
}

void SetIncrementalStepClockForTest(
    absl::AnyInvocable<absl::Time() const> now_fn) {
  GetIncrementalSession().step_now_fn = std::move(now_fn);
}

size_t GetZstdDecompressorCapacityForTest() {
  return GetIncrementalSession().zstd_decompressor.capacity();
}

bool ShouldEnqueueCompressedFrameIncremental() {
  IncrementalParserSession& session = GetIncrementalSession();
  if (!session.is_initial_load &&
      !session.cached_cumulative_interned_strings.empty()) {
    session.using_cached_cumulative_strings = true;
  }
  if (CanSkipFrameForFilter(session, session.next_enqueued_frame_index)) {
    ++session.next_enqueued_frame_index;
    return false;
  }
  return true;
}

void AppendCompressedTraceEventsIncremental(uintptr_t data_ptr,
                                            size_t data_size) {
  if (data_size == 0) {
    return;
  }
  IncrementalParserSession& session = GetIncrementalSession();
  if (session.using_cached_trace_store) {
    ++session.next_enqueued_frame_index;
    return;
  }
  const char* bytes = reinterpret_cast<const char*>(data_ptr);
  if (absl::StartsWith(absl::string_view(bytes, data_size),
                       tensorflow::profiler::kXChunkMagic)) {
    size_t offset = tensorflow::profiler::kXChunkMagic.size();
    while (offset + 4 <= data_size) {
      const uint32_t frame_len =
          static_cast<uint32_t>(static_cast<uint8_t>(bytes[offset])) |
          (static_cast<uint32_t>(static_cast<uint8_t>(bytes[offset + 1]))
           << 8) |
          (static_cast<uint32_t>(static_cast<uint8_t>(bytes[offset + 2]))
           << 16) |
          (static_cast<uint32_t>(static_cast<uint8_t>(bytes[offset + 3]))
           << 24);
      offset += 4;
      if (frame_len == 0 || frame_len > data_size - offset) {
        break;
      }
      const size_t frame_idx = session.next_enqueued_frame_index++;
      session.pending_compressed_frames.emplace_back(
          frame_idx, std::string(bytes + offset, frame_len));
      offset += frame_len;
    }
  } else {
    const size_t frame_idx = session.next_enqueued_frame_index++;
    session.pending_compressed_frames.emplace_back(
        frame_idx, std::string(bytes, data_size));
  }
}

bool StepIncrementalTraceEvents(size_t max_events_per_step,
                                DataProvider& data_provider, Timeline& timeline,
                                bool enforce_time_budget) {
  if (max_events_per_step == 0) {
    max_events_per_step = 1;
  }
  IncrementalParserSession& session = GetIncrementalSession();
  if (session.filter_range_ps.has_value() &&
      timeline.is_showing_full_overview() &&
      !timeline.stash_full_overview_on_next_update()) {
    CancelIncrementalTraceEvents(data_provider, timeline);
    return false;
  }
  ParsedTraceEvents batch;
  size_t events_unmarshaled = 0;
  size_t raw_events_scanned = 0;
  bool any_processed = false;

  if (session.using_cached_trace_store) {
    if (!session.cached_metadata_and_globals_emitted) {
      session.cached_metadata_and_globals_emitted = true;
      for (const auto& proc : session.cached_processes) {
        ProcessProcessMetadata(proc, batch);
      }
      if (session.filter_range_ps.has_value()) {
        const Microseconds min_ts_us = session.filter_range_ps->first / 1e6;
        const Microseconds max_ts_us = session.filter_range_ps->second / 1e6;
        for (const TraceEvent& ev : session.cached_async_flame_events) {
          if (ev.ph == Phase::kComplete &&
              (ev.ts + ev.dur < min_ts_us || ev.ts > max_ts_us)) {
            continue;
          }
          batch.flame_events.push_back(ev);
        }
        for (const TraceEvent& ev : session.cached_async_flow_events) {
          if (ev.ph == Phase::kComplete &&
              (ev.ts + ev.dur < min_ts_us || ev.ts > max_ts_us)) {
            continue;
          }
          batch.flow_events.push_back(ev);
        }
        for (const CounterEvent& counter : session.cached_counter_events) {
          const auto first_it = std::lower_bound(
              counter.timestamps.begin(), counter.timestamps.end(), min_ts_us);
          const auto last_it =
              std::upper_bound(first_it, counter.timestamps.end(), max_ts_us);
          if (first_it == last_it) continue;
          const size_t first_idx =
              static_cast<size_t>(first_it - counter.timestamps.begin());
          const size_t last_idx =
              static_cast<size_t>(last_it - counter.timestamps.begin());
          CounterEvent sliced;
          sliced.pid = counter.pid;
          sliced.name = counter.name;
          sliced.event_stats = counter.event_stats;
          sliced.unit = counter.unit;
          sliced.timestamps.assign(first_it, last_it);
          sliced.values.assign(counter.values.begin() + first_idx,
                               counter.values.begin() + last_idx);
          for (const double val : sliced.values) {
            sliced.min_value = std::min(sliced.min_value, val);
            sliced.max_value = std::max(sliced.max_value, val);
          }
          batch.counter_events.push_back(std::move(sliced));
        }
      } else {
        batch.flame_events.insert(batch.flame_events.end(),
                                  session.cached_async_flame_events.begin(),
                                  session.cached_async_flame_events.end());
        batch.flow_events.insert(batch.flow_events.end(),
                                 session.cached_async_flow_events.begin(),
                                 session.cached_async_flow_events.end());
        batch.counter_events.insert(batch.counter_events.end(),
                                    session.cached_counter_events.begin(),
                                    session.cached_counter_events.end());
      }
      any_processed = !batch.flame_events.empty() ||
                      !batch.flow_events.empty() ||
                      !batch.counter_events.empty();
    }
    const absl::Time step_start = absl::Now();
    size_t blocks_checked = 0;
    size_t blocks_decoded = 0;
    while (session.next_cached_block_idx <
               session.cached_complete_blocks.size() &&
           events_unmarshaled < max_events_per_step &&
           raw_events_scanned < kMaxCachedRawEventsPerStep) {
      const size_t block_idx = session.next_cached_block_idx++;
      ++blocks_checked;
      size_t block_raw_scanned = 0;
      const size_t decoded = DecodeCachedCompleteBlock(
          session, block_idx, batch, block_raw_scanned);
      raw_events_scanned += block_raw_scanned;
      if (block_raw_scanned > 0) {
        ++blocks_decoded;
      }
      if (decoded > 0) {
        any_processed = true;
        events_unmarshaled += decoded;
      }
      if (!session.in_finish_drain &&
          ((blocks_decoded > 0 && (blocks_decoded & 7) == 0) ||
           (blocks_checked & 1023) == 0) &&
          (absl::Now() - step_start) >= kCachedStepWallClockBudget) {
        break;
      }
    }
    size_t skip_checked = 0;
    while (session.next_cached_block_idx <
               session.cached_complete_blocks.size() &&
           !BlockOverlapsFilter(
               session.cached_complete_blocks[session.next_cached_block_idx],
               session)) {
      ++session.next_cached_block_idx;
      ++skip_checked;
      if (!session.in_finish_drain && (skip_checked & 1023) == 0 &&
          (absl::Now() - step_start) >= kCachedStepWallClockBudget) {
        break;
      }
    }
    if (any_processed) {
      batch.mpmd_pipeline_view = session.mpmd_pipeline_view;
      batch.full_timespan = session.full_timespan;
      batch.visible_range_from_url = session.visible_range_from_url;
      data_provider.AppendTraceEventsIncremental(
          std::move(batch), timeline, TimelineFlushMode::kDeferFlush);
      session.appended_any_batch = true;
    }
    return session.next_cached_block_idx <
           session.cached_complete_blocks.size();
  }

  const absl::Time step_deadline =
      (session.in_finish_drain || !enforce_time_budget)
          ? absl::InfiniteFuture()
          : (session.StepNow() + session.step_time_budget);
  const size_t max_raw_events_per_step =
      session.filter_range_ps.has_value() ? 2097152 : 524288;
  bool made_progress = false;

  while (events_unmarshaled < max_events_per_step &&
         raw_events_scanned < max_raw_events_per_step &&
         (!made_progress || session.StepNow() < step_deadline)) {
    if (!session.active_response ||
        session.next_pid_idx >= session.active_pids.size()) {
      session.ClearActiveFrame();
      // On initial progressive load, flush at frame boundaries once we have
      // unmarshaled events so each completed frame renders cleanly. On a
      // filtered zoom refetch, continue scanning across frames up to the raw
      // event budget.
      if (events_unmarshaled > 0 &&
          (!session.filter_range_ps.has_value() ||
           raw_events_scanned >= max_raw_events_per_step / 2)) {
        break;
      }
      bool header_only = false;
      if (!ActivateNextCompressedFrame(session, header_only)) {
        break;
      }
      made_progress = true;
      if (header_only) {
        continue;
      }
      any_processed = true;
    }

    const ProcessId pid = session.active_pids[session.next_pid_idx];
    const bool pid_finished = UnmarshalProcessSliceBounded(
        session, pid, max_events_per_step, max_raw_events_per_step,
        step_deadline, made_progress, events_unmarshaled, raw_events_scanned,
        batch);
    any_processed = true;
    if (pid_finished) {
      ++session.next_pid_idx;
      session.ResetCurrentPidCursors();
    }

    if (session.next_pid_idx >= session.active_pids.size()) {
      session.ClearActiveFrame();
    }
  }

  session.track_events_unmarshaled += events_unmarshaled;
  const bool has_batch_events = !batch.flame_events.empty() ||
                                !batch.flow_events.empty() ||
                                !batch.counter_events.empty();
  const bool at_frame_boundary =
      (session.active_response == nullptr ||
       session.next_pid_idx >= session.active_pids.size());
  if (any_processed &&
      (has_batch_events ||
       (at_frame_boundary && data_provider.HasPendingRefetchUpdate()))) {
    // Metadata-only batches are still appended (deferred) so their process
    // and thread names apply once the first track events flush.
    const bool flush_timeline =
        !session.in_finish_drain && session.is_initial_load &&
        session.track_events_unmarshaled > 0 &&
        !session.filter_range_ps.has_value() && at_frame_boundary &&
        (session.frames_flushed < 3 ||
         session.pending_compressed_frames.empty());
    batch.mpmd_pipeline_view = session.mpmd_pipeline_view;
    batch.full_timespan = session.full_timespan;
    batch.visible_range_from_url = session.visible_range_from_url;
    const bool flushed = data_provider.AppendTraceEventsIncremental(
        std::move(batch), timeline,
        flush_timeline ? TimelineFlushMode::kAutoFlush
                       : TimelineFlushMode::kDeferFlush);
    session.appended_any_batch = true;
    if (flushed) {
      ++session.frames_flushed;
      timeline.RequestRedraw();
    }
  }

  const bool has_more = (session.active_response != nullptr &&
                         session.next_pid_idx < session.active_pids.size()) ||
                        !session.pending_compressed_frames.empty();
  return has_more;
}

#ifdef __EMSCRIPTEN__
bool StepIncrementalTraceEvents(size_t max_events_per_step) {
  bool enforce_time_budget = true;
  const emscripten::val document = emscripten::val::global("document");
  if (!document.isUndefined() && !document.isNull()) {
    const emscripten::val visibility = document["visibilityState"];
    if (visibility.isString() && visibility.as<std::string>() == "hidden") {
      enforce_time_budget = false;
    }
  }
  return StepIncrementalTraceEvents(
      max_events_per_step, Application::Instance().data_provider(),
      Application::Instance().timeline(), enforce_time_budget);
}
#endif

void CancelIncrementalTraceEvents(DataProvider& data_provider,
                                  Timeline& timeline) {
  data_provider.FinishIncrementalLoad();
  GetIncrementalSession().Reset();
  timeline.set_is_incremental_loading(false);
}

#ifdef __EMSCRIPTEN__
void CancelIncrementalTraceEvents() {
  CancelIncrementalTraceEvents(Application::Instance().data_provider(),
                               Application::Instance().timeline());
}
#endif

bool FinishIncrementalTraceEvents(DataProvider& data_provider,
                                  Timeline& timeline) {
  IncrementalParserSession& session = GetIncrementalSession();
  if (session.filter_range_ps.has_value() &&
      timeline.is_showing_full_overview() &&
      !timeline.stash_full_overview_on_next_update()) {
    CancelIncrementalTraceEvents(data_provider, timeline);
    return timeline.full_trace_loaded();
  }
  session.in_finish_drain = true;
  constexpr size_t kFinishMaxEventsPerStep = 131072;
  while (StepIncrementalTraceEvents(kFinishMaxEventsPerStep, data_provider,
                                    timeline)) {
  }
  if (session.filter_range_ps.has_value() &&
      timeline.is_showing_full_overview() &&
      !timeline.stash_full_overview_on_next_update()) {
    CancelIncrementalTraceEvents(data_provider, timeline);
    return timeline.full_trace_loaded();
  }
  data_provider.MarkFinishingIncrementalLoad();
  if (!session.appended_any_batch || data_provider.HasPendingRefetchUpdate()) {
    ParsedTraceEvents empty_batch;
    empty_batch.mpmd_pipeline_view = session.mpmd_pipeline_view;
    empty_batch.full_timespan = session.full_timespan;
    empty_batch.visible_range_from_url = session.visible_range_from_url;
    data_provider.AppendTraceEventsIncremental(std::move(empty_batch), timeline,
                                               TimelineFlushMode::kAutoFlush);
  }

  const bool is_unfiltered_full_load = !session.filter_range_ps.has_value();
  if (is_unfiltered_full_load) {
    timeline.InitializeLastFetchRequestRange(timeline.data_time_range());
  } else {
    timeline.set_last_fetch_request_range(
        {static_cast<Microseconds>(session.filter_range_ps->first) / 1e6,
         static_cast<Microseconds>(session.filter_range_ps->second) / 1e6});
  }
  const bool was_compacted =
      data_provider.was_compacted() || session.overview_pre_compacted ||
      (is_unfiltered_full_load && timeline.incremental_fetch_enabled() &&
       timeline.timeline_data().entry_start_times.size() >
           timeline.max_uncompacted_overview_entries());
  timeline.set_full_trace_loaded(is_unfiltered_full_load && !was_compacted);
  timeline.set_is_compacted(was_compacted);
  timeline.set_window_at_full_resolution(
      !was_compacted || data_provider.last_load_was_refetch());
  if (is_unfiltered_full_load) {
    timeline.MarkCurrentDataAsFullOverview();
  }
  const bool full_loaded = timeline.full_trace_loaded();
  if (session.recording_frame_summaries &&
      !session.cumulative_interned_strings.empty()) {
    session.cached_cumulative_interned_strings =
        std::move(session.cumulative_interned_strings);
    session.cached_frame_summaries =
        std::move(session.building_frame_summaries);
    session.cached_full_timespan = session.full_timespan;
    session.cached_mpmd_pipeline_view = session.mpmd_pipeline_view;
    if (!full_loaded && session.can_build_trace_store &&
        !session.building_complete_blocks.empty()) {
      session.cached_processes = std::move(session.building_processes);
      session.cached_async_flame_events =
          std::move(session.building_async_flame_events);
      session.cached_async_flow_events =
          std::move(session.building_async_flow_events);
      session.cached_counter_events =
          std::move(session.building_counter_events);
      session.cached_complete_blocks =
          std::move(session.building_complete_blocks);
      session.cached_frame_interned_strings =
          std::move(session.building_frame_interned_strings);
      session.has_cached_trace_store = true;
    } else {
      session.has_cached_trace_store = false;
    }
  }

  data_provider.FinishIncrementalLoad();
  session.Reset();
  timeline.set_is_incremental_loading(false);
  timeline.RequestRedraw();
  return full_loaded;
}

#ifdef __EMSCRIPTEN__
bool FinishIncrementalTraceEvents() {
  return FinishIncrementalTraceEvents(Application::Instance().data_provider(),
                                      Application::Instance().timeline());
}
#endif

bool RestoreFullTraceOverviewIfAvailable(Timeline& timeline) {
  if (timeline.RestoreSavedFullOverview()) {
    GetIncrementalSession().Reset();
    return true;
  }
  return false;
}

#ifdef __EMSCRIPTEN__
bool RestoreFullTraceOverviewIfAvailable() {
  return RestoreFullTraceOverviewIfAvailable(
      Application::Instance().timeline());
}
#endif

void ParseAndProcessCompressedTraceEvents(
    uintptr_t data_ptr, size_t data_size,
    std::optional<std::pair<Milliseconds, Milliseconds>> visible_range_from_url,
    DataProvider& data_provider, Timeline& timeline) {
  const bool is_initial_load =
      (timeline.last_fetch_request_range() == TimeRange::Zero());
  GetIncrementalSession().ClearPersistentCache();
  const ParsedTraceEvents parsed_events =
      ParseCompressedTraceEvents(data_ptr, data_size, visible_range_from_url);
  const bool has_url_range = parsed_events.visible_range_from_url.has_value();
  const bool is_overview_load =
      is_initial_load && !has_url_range && timeline.incremental_fetch_enabled();
  timeline.set_full_trace_loaded(!timeline.incremental_fetch_enabled() &&
                                 !has_url_range);
  timeline.set_is_compacted(is_overview_load);
  timeline.set_window_at_full_resolution(!is_overview_load);
  data_provider.ProcessTraceEvents(parsed_events, timeline);

  if (is_initial_load) {
    if (visible_range_from_url.has_value()) {
      const Milliseconds start = visible_range_from_url->first;
      const Milliseconds end = visible_range_from_url->second;
      timeline.InitializeLastFetchRequestRange(
          {MillisToMicros(start), MillisToMicros(end)});
    } else {
      timeline.InitializeLastFetchRequestRange(timeline.data_time_range());
    }
  }

  timeline.set_is_incremental_loading(false);
  timeline.RequestRedraw();
}

#ifdef __EMSCRIPTEN__
void ParseAndProcessCompressedTraceEvents(
    uintptr_t data_ptr, size_t data_size,
    const emscripten::val& visible_range_from_url, DataProvider& data_provider,
    Timeline& timeline) {
  ParseAndProcessCompressedTraceEvents(
      data_ptr, data_size, ParseVisibleRangeFromUrl(visible_range_from_url),
      data_provider, timeline);
}

void ParseAndProcessCompressedTraceEvents(
    uintptr_t data_ptr, size_t data_size,
    const emscripten::val& visible_range_from_url) {
  ParseAndProcessCompressedTraceEvents(data_ptr, data_size,
                                       visible_range_from_url,
                                       Application::Instance().data_provider(),
                                       Application::Instance().timeline());
}

EMSCRIPTEN_BINDINGS(trace_pb_event_parser) {
  emscripten::function(
      "processCompressedTraceEvents",
      static_cast<void (*)(uintptr_t, size_t, const emscripten::val&)>(
          &traceviewer::ParseAndProcessCompressedTraceEvents));
  emscripten::function("beginIncrementalTraceEvents",
                       static_cast<void (*)(const emscripten::val&)>(
                           &traceviewer::BeginIncrementalTraceEvents));
  emscripten::function("hasCachedTraceEventsIncremental",
                       &traceviewer::HasCachedTraceEventsIncremental);
  emscripten::function("hasFlushedIncrementalTraceEvents",
                       &traceviewer::HasFlushedIncrementalTraceEvents);
  emscripten::function("shouldEnqueueCompressedFrameIncremental",
                       &traceviewer::ShouldEnqueueCompressedFrameIncremental);
  emscripten::function("appendCompressedTraceEventsIncremental",
                       &traceviewer::AppendCompressedTraceEventsIncremental);
  emscripten::function(
      "stepIncrementalTraceEvents",
      static_cast<bool (*)(size_t)>(&traceviewer::StepIncrementalTraceEvents));
  emscripten::function(
      "finishIncrementalTraceEvents",
      static_cast<bool (*)()>(&traceviewer::FinishIncrementalTraceEvents));
  emscripten::function(
      "cancelIncrementalTraceEvents",
      static_cast<void (*)()>(&traceviewer::CancelIncrementalTraceEvents));
  emscripten::function("restoreFullTraceOverviewIfAvailable",
                       static_cast<bool (*)()>(
                           &traceviewer::RestoreFullTraceOverviewIfAvailable));
}
#endif

}  // namespace traceviewer
