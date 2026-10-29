#include "frontend/app/components/trace_viewer_v2/trace_helper/trace_pb_event_parser.h"

#include <emscripten/bind.h>
#include <emscripten/val.h>

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
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xprof/convert/trace_viewer/delta_series/zstd_compression.h"
#include "frontend/app/components/trace_viewer_v2/application.h"
#include "frontend/app/components/trace_viewer_v2/helper/time_formatter.h"
#include "frontend/app/components/trace_viewer_v2/timeline/data_provider.h"
#include "frontend/app/components/trace_viewer_v2/timeline/timeline.h"
#include "frontend/app/components/trace_viewer_v2/trace_helper/trace_event.h"
#include "frontend/app/components/trace_viewer_v2/trace_helper/trace_event_parser_core.h"
#include "plugin/xprof/protobuf/trace_data_response.pb.h"

namespace traceviewer {

namespace {

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

void DispatchResponseDetails(const xprof::TraceDataResponse& response,
                             bool& mpmd_pipeline_view) {
  if (response.details_size() <= 0) {
    return;
  }
  emscripten::val details_map = emscripten::val::global("Map").new_();
  for (const auto& detail : response.details()) {
    if (detail.name() == "mpmd_pipeline_view") {
      mpmd_pipeline_view = detail.value();
    }
    details_map.call<void>("set", emscripten::val(detail.name()),
                           emscripten::val(detail.value()));
  }
  emscripten::val detail_obj = emscripten::val::object();
  detail_obj.set("details", details_map);

  emscripten::val event_init = emscripten::val::object();
  event_init.set("detail", detail_obj);

  emscripten::val event =
      emscripten::val::global("CustomEvent")
          .new_(emscripten::val("details_received"), event_init);
  emscripten::val::global("window").call<void>("dispatchEvent", event);
}

ParsedTraceEvents ParseCompressedTraceEvents(
    uintptr_t data_ptr, size_t data_size,
    const emscripten::val& visible_range_from_url) {
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

  result.visible_range_from_url =
      ParseVisibleRangeFromUrl(visible_range_from_url);

  return result;
}

struct FrameSummary {
  uint64_t min_start_ps = std::numeric_limits<uint64_t>::max();
  uint64_t max_end_ps = 0;
  bool has_non_split_bounds = false;
  bool always_process = false;
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
  bool overview_pre_compacted = false;

  std::unique_ptr<xprof::TraceDataResponse> active_response;
  std::vector<ProcessId> active_pids;
  size_t next_pid_idx = 0;
  bool current_pid_metadata_emitted = false;
  size_t current_complete_series_pos = 0;
  size_t current_complete_event_idx = 0;
  uint64_t current_complete_ts_ps = 0;
  size_t current_async_series_pos = 0;
  size_t current_counter_series_pos = 0;
  absl::flat_hash_map<ProcessId, std::vector<int>> metadata_indices_by_pid;
  absl::flat_hash_map<ProcessId, std::vector<int>> complete_series_by_pid;
  absl::flat_hash_map<ProcessId, std::vector<int>> async_series_by_pid;
  absl::flat_hash_map<ProcessId, std::vector<int>> counter_series_by_pid;

  std::optional<std::pair<Milliseconds, Milliseconds>> visible_range_from_url;
  std::optional<std::pair<uint64_t, uint64_t>> filter_range_ps;
  std::optional<std::pair<Milliseconds, Milliseconds>> full_timespan;
  bool is_initial_load = true;
  bool mpmd_pipeline_view = false;
  bool appended_any_batch = false;
  size_t frames_flushed = 0;
  bool in_finish_drain = false;

  void Reset() {
    pending_compressed_frames.clear();
    cumulative_interned_strings.clear();
    active_frame_interned_strings.clear();
    use_active_frame_interned_strings = false;
    building_frame_summaries.clear();
    overview_pre_compacted = false;
    next_enqueued_frame_index = 0;
    active_frame_index = 0;
    using_cached_cumulative_strings = false;
    recording_frame_summaries = false;
    ClearActiveFrame();
    visible_range_from_url = std::nullopt;
    filter_range_ps = std::nullopt;
    full_timespan = std::nullopt;
    is_initial_load = true;
    mpmd_pipeline_view = false;
    appended_any_batch = false;
    frames_flushed = 0;
    in_finish_drain = false;
  }

  void ClearPersistentCache() {
    cached_cumulative_interned_strings.clear();
    cached_frame_summaries.clear();
    cached_full_timespan = std::nullopt;
    cached_mpmd_pipeline_view = false;
  }

  void ResetCurrentPidCursors() {
    current_pid_metadata_emitted = false;
    current_complete_series_pos = 0;
    current_complete_event_idx = 0;
    current_complete_ts_ps = 0;
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

    absl::StatusOr<std::string> decompressed_proto =
        tensorflow::profiler::ZstdCompression::Decompress(compressed);
    std::string().swap(compressed);
    if (!decompressed_proto.ok()) {
      continue;
    }

    auto response = std::make_unique<xprof::TraceDataResponse>();
    if (!response->ParseFromString(*decompressed_proto)) {
      continue;
    }
    std::string().swap(*decompressed_proto);

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
                                  size_t& events_unmarshaled,
                                  size_t& raw_events_scanned,
                                  ParsedTraceEvents& batch) {
  xprof::TraceDataResponse& response = *session.active_response;
  const absl::Span<const std::string> interned_strings =
      session.CurrentInternedStrings();

  if (!session.current_pid_metadata_emitted) {
    session.current_pid_metadata_emitted = true;
    if (const auto it = session.metadata_indices_by_pid.find(pid);
        it != session.metadata_indices_by_pid.end()) {
      for (const int idx : it->second) {
        ProcessProcessMetadata(response.metadata().processes(idx), batch);
      }
    }
  }

  if (const auto it = session.complete_series_by_pid.find(pid);
      it != session.complete_series_by_pid.end()) {
    const std::vector<int>& series_indices = it->second;
    while (session.current_complete_series_pos < series_indices.size() &&
           events_unmarshaled < max_events_budget &&
           raw_events_scanned < max_raw_events_budget) {
      const int idx = series_indices[session.current_complete_series_pos];
      const auto& series = response.complete_events(idx);
      const size_t total_in_series = static_cast<size_t>(series.deltas_size());
      const size_t remaining_budget = max_events_budget - events_unmarshaled;
      const size_t slice_limit =
          session.filter_range_ps.has_value()
              ? std::max<size_t>(remaining_budget, 262144)
              : std::max<size_t>(remaining_budget, 65536);
      const size_t before = batch.flame_events.size();
      size_t count = 0;
      const bool can_use_overview_fast_path =
          !session.filter_range_ps.has_value() &&
          session.recording_frame_summaries && total_in_series > 8192;
      if (can_use_overview_fast_path) {
        bool did_coalesce = false;
        count = ProcessOverviewSampledSeriesSlice(
            series, interned_strings, session.current_complete_event_idx,
            slice_limit, session.current_complete_ts_ps, batch, did_coalesce);
        if (did_coalesce) {
          session.overview_pre_compacted = true;
        }
      } else {
        count = ProcessCompleteEventSeriesSlice(
            series, interned_strings, session.current_complete_event_idx,
            slice_limit, session.current_complete_ts_ps, batch,
            session.filter_range_ps);
      }
      if (session.recording_frame_summaries &&
          session.active_frame_index <
              session.building_frame_summaries.size()) {
        FrameSummary& summary =
            session.building_frame_summaries[session.active_frame_index];
        for (size_t i = before; i < batch.flame_events.size(); ++i) {
          const TraceEvent& ev = batch.flame_events[i];
          if (ev.ph != Phase::kComplete) continue;
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
      raw_events_scanned += session.filter_range_ps.has_value()
                                ? std::min<size_t>(count, added + 4096)
                                : count;
      session.current_complete_event_idx += count;
      if (session.current_complete_event_idx >= total_in_series) {
        response.mutable_complete_events(idx)->Clear();
        ++session.current_complete_series_pos;
        session.current_complete_event_idx = 0;
        session.current_complete_ts_ps = 0;
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
           events_unmarshaled < max_events_budget) {
      const int idx = series_indices[session.current_async_series_pos++];
      const size_t before_flame = batch.flame_events.size();
      ProcessAsyncEventSeries(response.async_events(idx), interned_strings,
                              batch);
      response.mutable_async_events(idx)->Clear();
      const size_t added = batch.flame_events.size() - before_flame;
      events_unmarshaled += added;
      raw_events_scanned += added;
    }
    if (session.current_async_series_pos < series_indices.size()) {
      return false;
    }
  }

  if (const auto it = session.counter_series_by_pid.find(pid);
      it != session.counter_series_by_pid.end()) {
    const std::vector<int>& series_indices = it->second;
    while (session.current_counter_series_pos < series_indices.size() &&
           events_unmarshaled < max_events_budget) {
      const int idx = series_indices[session.current_counter_series_pos++];
      const size_t before = batch.counter_events.size();
      ProcessCounterEventSeries(response.counter_events(idx), interned_strings,
                                batch);
      response.mutable_counter_events(idx)->Clear();
      const size_t added = batch.counter_events.size() - before;
      events_unmarshaled += added;
      raw_events_scanned += added;
    }
    if (session.current_counter_series_pos < series_indices.size()) {
      return false;
    }
  }

  return true;
}

}  // namespace

void BeginIncrementalTraceEvents(const emscripten::val& visible_range_from_url,
                                 DataProvider& data_provider,
                                 Timeline& timeline) {
  IncrementalParserSession& session = GetIncrementalSession();
  session.Reset();
  session.is_initial_load = !data_provider.HasKnownTracks() ||
                            (timeline.visible_range() == TimeRange::Zero());
  session.visible_range_from_url =
      ParseVisibleRangeFromUrl(visible_range_from_url);
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
  if (session.is_initial_load ||
      session.cached_cumulative_interned_strings.empty()) {
    session.ClearPersistentCache();
    session.recording_frame_summaries = true;
  } else {
    session.using_cached_cumulative_strings = true;
    session.mpmd_pipeline_view = session.cached_mpmd_pipeline_view;
    session.full_timespan = session.cached_full_timespan;
  }
  if (session.filter_range_ps.has_value()) {
    timeline.PrepareForFilteredRefetch();
  } else {
    timeline.set_stash_full_overview_on_next_update(false);
  }
  timeline.set_full_trace_loaded(false);
  timeline.set_is_compacted(false);
  timeline.set_window_at_full_resolution(false);
  timeline.set_is_incremental_loading(true);
  data_provider.BeginIncrementalLoad(timeline);
}

void BeginIncrementalTraceEvents(
    const emscripten::val& visible_range_from_url) {
  BeginIncrementalTraceEvents(visible_range_from_url,
                              Application::Instance().data_provider(),
                              Application::Instance().timeline());
}

bool ShouldEnqueueCompressedFrameIncremental() {
  IncrementalParserSession& session = GetIncrementalSession();
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
      if (frame_len == 0 || offset + frame_len > data_size) {
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
                                DataProvider& data_provider,
                                Timeline& timeline) {
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

  const size_t max_raw_events_per_step =
      session.filter_range_ps.has_value() ? 2097152 : 524288;

  while (events_unmarshaled < max_events_per_step &&
         raw_events_scanned < max_raw_events_per_step) {
    if (!session.active_response ||
        session.next_pid_idx >= session.active_pids.size()) {
      session.ClearActiveFrame();
      if (events_unmarshaled > 0 &&
          (!session.filter_range_ps.has_value() ||
           raw_events_scanned >= max_raw_events_per_step / 2)) {
        break;
      }
      bool header_only = false;
      if (!ActivateNextCompressedFrame(session, header_only)) {
        break;
      }
      if (header_only) {
        continue;
      }
      any_processed = true;
    }

    const ProcessId pid = session.active_pids[session.next_pid_idx];
    const bool pid_finished = UnmarshalProcessSliceBounded(
        session, pid, max_events_per_step, max_raw_events_per_step,
        events_unmarshaled, raw_events_scanned, batch);
    any_processed = true;
    if (pid_finished) {
      ++session.next_pid_idx;
      session.ResetCurrentPidCursors();
    }

    if (session.next_pid_idx >= session.active_pids.size()) {
      session.ClearActiveFrame();
    }
  }

  const bool has_batch_events = !batch.flame_events.empty() ||
                                !batch.flow_events.empty() ||
                                !batch.counter_events.empty();
  if (any_processed && has_batch_events) {
    const bool at_frame_boundary =
        (session.active_response == nullptr ||
         session.next_pid_idx >= session.active_pids.size());
    const bool flush_timeline = !session.in_finish_drain &&
                                !session.filter_range_ps.has_value() &&
                                at_frame_boundary &&
                                (session.frames_flushed < 3 ||
                                 session.pending_compressed_frames.empty());
    batch.mpmd_pipeline_view = session.mpmd_pipeline_view;
    batch.full_timespan = session.full_timespan;
    batch.visible_range_from_url = session.visible_range_from_url;
    data_provider.AppendTraceEventsIncremental(
        std::move(batch), timeline,
        flush_timeline ? TimelineFlushMode::kAutoFlush
                       : TimelineFlushMode::kDeferFlush);
    session.appended_any_batch = true;
    if (flush_timeline) {
      ++session.frames_flushed;
      timeline.RequestRedraw();
    }
  }

  const bool has_more = (session.active_response != nullptr &&
                         session.next_pid_idx < session.active_pids.size()) ||
                        !session.pending_compressed_frames.empty();
  return has_more;
}

bool StepIncrementalTraceEvents(size_t max_events_per_step) {
  return StepIncrementalTraceEvents(max_events_per_step,
                                    Application::Instance().data_provider(),
                                    Application::Instance().timeline());
}

void CancelIncrementalTraceEvents(DataProvider& data_provider,
                                  Timeline& timeline) {
  data_provider.FinishIncrementalLoad();
  GetIncrementalSession().Reset();
  timeline.set_is_incremental_loading(false);
}

void CancelIncrementalTraceEvents() {
  CancelIncrementalTraceEvents(Application::Instance().data_provider(),
                               Application::Instance().timeline());
}

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
  }

  data_provider.FinishIncrementalLoad();
  session.Reset();
  timeline.set_is_incremental_loading(false);
  timeline.RequestRedraw();
  return full_loaded;
}

bool FinishIncrementalTraceEvents() {
  return FinishIncrementalTraceEvents(Application::Instance().data_provider(),
                                      Application::Instance().timeline());
}

bool RestoreFullTraceOverviewIfAvailable(Timeline& timeline) {
  if (timeline.RestoreSavedFullOverview()) {
    GetIncrementalSession().Reset();
    return true;
  }
  return false;
}

bool RestoreFullTraceOverviewIfAvailable() {
  return RestoreFullTraceOverviewIfAvailable(
      Application::Instance().timeline());
}

void ParseAndProcessCompressedTraceEvents(
    uintptr_t data_ptr, size_t data_size,
    const emscripten::val& visible_range_from_url, DataProvider& data_provider,
    Timeline& timeline) {
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
    if (!visible_range_from_url.isNull() &&
        !visible_range_from_url.isUndefined() &&
        visible_range_from_url["length"].as<int>() == 2) {
      Milliseconds start = visible_range_from_url[0].as<Milliseconds>();
      Milliseconds end = visible_range_from_url[1].as<Milliseconds>();

      timeline.InitializeLastFetchRequestRange(
          {MillisToMicros(start), MillisToMicros(end)});
    } else {
      timeline.InitializeLastFetchRequestRange(timeline.data_time_range());
    }
  }

  timeline.set_is_incremental_loading(false);
  timeline.RequestRedraw();
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

}  // namespace traceviewer
