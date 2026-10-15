#include "frontend/app/components/trace_viewer_v2/timeline/track_packer.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/btree_map.h"
#include "absl/container/inlined_vector.h"
#include "absl/functional/function_ref.h"
#include "absl/log/log.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "google/protobuf/arena.h"
#include "tsl/profiler/lib/context_types.h"
#include "frontend/app/components/trace_viewer_v2/timeline/constants.h"
#include "frontend/app/components/trace_viewer_v2/timeline/timeline.h"
#include "frontend/app/components/trace_viewer_v2/trace_helper/proto_event_ref.h"
#include "frontend/app/components/trace_viewer_v2/trace_helper/trace_event.h"
#include "frontend/app/components/trace_viewer_v2/trace_helper/trace_event_parser_core.h"
#include "frontend/app/components/trace_viewer_v2/trace_helper/wire_trace_reader.h"
#include "plugin/xprof/protobuf/trace_data_response.pb.h"

namespace traceviewer {

std::vector<ModuleInterval> BuildDisjointModuleSegments(
    std::vector<ModuleInterval> raw_intervals) {
  if (raw_intervals.empty()) return {};
  absl::c_stable_sort(raw_intervals,
                      [](const ModuleInterval& a, const ModuleInterval& b) {
                        if (a.start != b.start) return a.start < b.start;
                        return a.end > b.end;
                      });
  std::vector<ModuleInterval> disjoint;
  disjoint.reserve(raw_intervals.size());
  Microseconds max_covered_end = -std::numeric_limits<Microseconds>::infinity();
  for (const ModuleInterval& interval : raw_intervals) {
    if (interval.end <= interval.start || interval.end <= max_covered_end) {
      continue;
    }
    const Microseconds clipped_start =
        std::max(interval.start, max_covered_end);
    disjoint.push_back({
        .start = clipped_start,
        .end = interval.end,
        .module_index = interval.module_index,
    });
    max_covered_end = interval.end;
  }
  return disjoint;
}

ProtoTrackPacker::ProtoTrackPacker(
    absl::string_view thread_name, bool is_async_track,
    absl::Span<const ModuleInterval> module_intervals,
    uint32_t default_module_index, int start_level, int& max_level,
    FlameChartTimelineData& data, TimeBounds& bounds,
    TraceInformation<ProtoEventRef>& trace_info,
    absl::FunctionRef<absl::string_view(uint32_t)> get_interned_string,
    std::vector<uint32_t>& name_ref_to_op_index,
    std::vector<uint32_t>& name_ref_to_module_index)
    : is_async_track_(is_async_track),
      is_xla_ops_thread_(thread_name == kXlaOps),
      is_data_motion_layer_(thread_name == kComputeUtilization ||
                            thread_name == kDataMotionLayersUtilization),
      module_intervals_(module_intervals),
      default_module_index_(default_module_index),
      start_level_(start_level),
      max_level_(max_level),
      data_(data),
      bounds_(bounds),
      trace_info_(trace_info),
      get_interned_string_(get_interned_string),
      name_ref_to_op_index_(name_ref_to_op_index),
      name_ref_to_module_index_(name_ref_to_module_index) {}

void ProtoTrackPacker::PackEvents(absl::Span<TrackEvent> events,
                                  bool already_sorted) {
  if (events.empty()) return;

  if (!already_sorted) {
    const bool is_sorted =
        std::is_sorted(events.begin(), events.end(),
                       [](const TrackEvent& a, const TrackEvent& b) {
                         if (a.ts != b.ts) return a.ts < b.ts;
                         return a.dur > b.dur;
                       });
    if (!is_sorted) {
      absl::c_stable_sort(events, [](const TrackEvent& a, const TrackEvent& b) {
        if (a.ts != b.ts) return a.ts < b.ts;
        return a.dur > b.dur;
      });
    }
  }

  const size_t entry_begin = data_.entry_start_times.size();

  for (const TrackEvent& event : events) {
    const Microseconds start = event.ts;
    const Microseconds duration = event.dur;
    const Microseconds end = start + duration;

    std::optional<int> selected_row = std::nullopt;
    for (size_t r = 0; r < row_end_times_.size(); ++r) {
      if (row_end_times_[r] <= start) {
        selected_row = static_cast<int>(r);
        break;
      }
    }
    if (!selected_row.has_value()) {
      row_end_times_.push_back(end);
      selected_row = static_cast<int>(row_end_times_.size() - 1);
    } else {
      row_end_times_[*selected_row] = end;
    }

    const int absolute_level = start_level_ + *selected_row;
    if (absolute_level < 0 ||
        absolute_level >= std::numeric_limits<uint16_t>::max()) {
      ++data_.dropped_level_cap_events;
      LOG_FIRST_N(WARNING, 1) << "Dropping event at level " << absolute_level
                              << ": exceeds uint16_t flame chart level range";
      continue;
    }
    max_level_ = std::max(max_level_, absolute_level);

    // Mirrors PopulateThreadTrackEvents: an event that starts inside the
    // innermost open parent but ends after it is a partial overlap. It neither
    // subtracts from that parent's self time nor becomes a parent itself.
    bool is_nested = false;
    while (!parent_stack_.empty()) {
      ActiveParent& parent = parent_stack_.back();
      if (start < parent.end_time) {
        if (end <= parent.end_time) {
          is_nested = true;
          if (duration > 0.0) {
            if (parent.sparse_record_index == SparseSelfTimes::kNone) {
              const Microseconds initial_self =
                  std::max<Microseconds>(0.0, parent.total_time - duration);
              parent.sparse_record_index = data_.sparse_self_times.Append(
                  parent.timeline_data_index, initial_self);
            } else {
              data_.sparse_self_times.Subtract(parent.sparse_record_index,
                                               duration);
            }
          }
        }
        break;
      }
      parent_stack_.pop_back();
    }

    const size_t new_entry_index = data_.entry_start_times.size();
    if (is_nested || parent_stack_.empty()) {
      parent_stack_.push_back({
          .end_time = end,
          .total_time = duration,
          .timeline_data_index = new_entry_index,
          .sparse_record_index = SparseSelfTimes::kNone,
      });
    }

    data_.entry_start_times.push_back(start);
    data_.entry_total_times.push_back(duration);
    data_.entry_levels.push_back(static_cast<uint16_t>(absolute_level));
    data_.entry_name_indices.push_back(event.name_index);

    if (event.uid != 0) {
      data_.non_zero_uids_by_event_index[static_cast<int>(new_entry_index)] =
          event.uid;
    }
    if (event.group_id >= 0) {
      data_.sparse_entry_group_ids.Append(new_entry_index, event.group_id);
    }
    if (event.flow_id != 0 && !is_async_track_) {
      data_.flow_ids_by_event_index[static_cast<int>(new_entry_index)] =
          event.flow_id;
    }

    if (event.hlo_op_ref != 0 &&
        event.hlo_op_ref < name_ref_to_op_index_.size()) {
      uint32_t& cached_op = name_ref_to_op_index_[event.hlo_op_ref];
      if (cached_op == std::numeric_limits<uint32_t>::max()) {
        cached_op = trace_info_.op_interner.AppendUnique(
            get_interned_string_(event.hlo_op_ref));
      }
      if (cached_op != 0) {
        data_.sparse_entry_hlo_op_indices.Append(new_entry_index, cached_op);
      }
    }

    if (event.hlo_module_ref != 0 &&
        event.hlo_module_ref < name_ref_to_module_index_.size()) {
      uint32_t& cached_module = name_ref_to_module_index_[event.hlo_module_ref];
      if (cached_module == std::numeric_limits<uint32_t>::max()) {
        cached_module = trace_info_.module_interner.Intern(
            get_interned_string_(event.hlo_module_ref));
      }
      if (cached_module != 0) {
        data_.sparse_entry_hlo_module_indices.Append(new_entry_index,
                                                     cached_module);
      }
    }

    bounds_.min = std::min(bounds_.min, start);
    bounds_.max = std::max(bounds_.max, end);
  }

  const size_t entry_end = data_.entry_start_times.size();
  const bool use_module_intervals =
      (is_xla_ops_thread_ || is_data_motion_layer_) &&
      !module_intervals_.empty();
  if (entry_end > entry_begin && (is_xla_ops_thread_ || use_module_intervals ||
                                  default_module_index_ != 0)) {
    if (use_module_intervals &&
        module_intervals_index_ == std::numeric_limits<uint32_t>::max()) {
      if (!data_.hlo_module_intervals.empty() &&
          absl::MakeConstSpan(data_.hlo_module_intervals.back()) ==
              module_intervals_) {
        module_intervals_index_ =
            static_cast<uint32_t>(data_.hlo_module_intervals.size() - 1);
      } else {
        module_intervals_index_ =
            static_cast<uint32_t>(data_.hlo_module_intervals.size());
        data_.hlo_module_intervals.emplace_back(module_intervals_.begin(),
                                                module_intervals_.end());
      }
    }
    if (!data_.hlo_track_rules.empty() &&
        data_.hlo_track_rules.back().end_entry == entry_begin &&
        data_.hlo_track_rules.back().default_module_index ==
            default_module_index_ &&
        data_.hlo_track_rules.back().module_intervals_index ==
            module_intervals_index_ &&
        data_.hlo_track_rules.back().is_xla_ops == is_xla_ops_thread_ &&
        data_.hlo_track_rules.back().use_module_intervals ==
            use_module_intervals) {
      data_.hlo_track_rules.back().end_entry = entry_end;
    } else {
      data_.hlo_track_rules.push_back({
          .begin_entry = entry_begin,
          .end_entry = entry_end,
          .default_module_index = default_module_index_,
          .module_intervals_index = module_intervals_index_,
          .is_xla_ops = is_xla_ops_thread_,
          .use_module_intervals = use_module_intervals,
      });
    }
  }
}

void PackProtoEvents(
    absl::Span<TrackEvent> events, absl::string_view thread_name,
    bool is_async_track, absl::Span<const ModuleInterval> module_intervals,
    uint32_t default_module_index, int start_level, int& max_level,
    FlameChartTimelineData& data, TimeBounds& bounds,
    TraceInformation<ProtoEventRef>& trace_info,
    absl::FunctionRef<absl::string_view(uint32_t)> get_interned_string,
    std::vector<uint32_t>& name_ref_to_op_index,
    std::vector<uint32_t>& name_ref_to_module_index) {
  ProtoTrackPacker packer(thread_name, is_async_track, module_intervals,
                          default_module_index, start_level, max_level, data,
                          bounds, trace_info, get_interned_string,
                          name_ref_to_op_index, name_ref_to_module_index);
  packer.PackEvents(events, /*already_sorted=*/false);
}

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
    std::vector<uint32_t>& name_ref_to_module_index, size_t max_chunk_size,
    uint64_t* max_end_ps_out) {
  constexpr uint64_t kMaxExactTimestampPs = uint64_t{1} << 53;
  size_t track_event_count = 0;
  for (const xprof::TraceEventSeries* series : series_list) {
    track_event_count += series->deltas_size();
  }
  bool can_stream_chunks = (max_chunk_size > 0);
  bool verified_remaining_monotonic = false;
  const size_t target_reserve =
      can_stream_chunks ? std::min(track_event_count, max_chunk_size)
                        : track_event_count;
  track_buffer.clear();
  if (track_buffer.capacity() < target_reserve) {
    track_buffer.reserve(target_reserve);
  }

  ProtoTrackPacker packer(
      thread_name, /*is_async_track=*/false, module_intervals,
      default_module_index, start_level, max_level, data, bounds, trace_info,
      get_interned_string, name_ref_to_op_index, name_ref_to_module_index);
  bool chunk_already_sorted = true;
  uint32_t track_event_seq = 0;
  uint64_t local_max_end_ps = 0;
  uint64_t prev_series_end_ts_ps = 0;
  bool has_prev_series_event = false;

  for (size_t s = 0; s < series_list.size(); ++s) {
    const xprof::TraceEventSeries* series = series_list[s];
    const xprof::TraceEventSeriesMetadata& metadata = series->metadata();
    uint64_t current_ts_ps = 0;
    const int deltas_count = series->deltas_size();

    const bool has_sparse_metadata = series->event_metadata_indices_size() > 0;
    int sparse_meta_cursor = 0;
    const int sparse_meta_count = std::min(
        series->event_metadata_indices_size(), series->event_metadata_size());

    const bool has_rle_group_ids = series->group_id_run_values_size() > 0 &&
                                   series->group_id_run_values_size() ==
                                       series->group_id_run_lengths_size();
    int rle_idx = 0;
    uint32_t rle_remaining =
        has_rle_group_ids ? series->group_id_run_lengths(0) : 0;

    for (int i = 0; i < deltas_count; ++i) {
      const uint64_t delta_ps = series->deltas(i);
      const int64_t signed_delta_ps = static_cast<int64_t>(delta_ps);
      if (signed_delta_ps < 0) {
        can_stream_chunks = false;
        chunk_already_sorted = false;
        const uint64_t abs_delta =
            static_cast<uint64_t>(-(signed_delta_ps + 1)) + 1;
        current_ts_ps =
            (current_ts_ps >= abs_delta) ? (current_ts_ps - abs_delta) : 0;
      } else {
        current_ts_ps += delta_ps;
      }
      if (i == 0 && has_prev_series_event &&
          current_ts_ps < prev_series_end_ts_ps) {
        can_stream_chunks = false;
        chunk_already_sorted = false;
      }
      if (current_ts_ps >= kMaxExactTimestampPs) {
        can_stream_chunks = false;
        chunk_already_sorted = false;
      }
      const Microseconds ev_ts = current_ts_ps / 1000000.0;
      if (can_stream_chunks && track_buffer.size() >= max_chunk_size &&
          ev_ts > track_buffer.back().ts) {
        if (!verified_remaining_monotonic) {
          uint64_t lookahead_ts_ps = current_ts_ps;
          bool remaining_ok = true;
          for (int j = i + 1; j < deltas_count; ++j) {
            const uint64_t d = series->deltas(j);
            if (static_cast<int64_t>(d) < 0) {
              remaining_ok = false;
              break;
            }
            lookahead_ts_ps += d;
            if (lookahead_ts_ps >= kMaxExactTimestampPs) {
              remaining_ok = false;
              break;
            }
          }
          uint64_t seg_end_ts_ps = lookahead_ts_ps;
          for (size_t next_s = s + 1;
               next_s < series_list.size() && remaining_ok; ++next_s) {
            const xprof::TraceEventSeries* next_series = series_list[next_s];
            uint64_t next_ts_ps = 0;
            const int next_count = next_series->deltas_size();
            for (int j = 0; j < next_count; ++j) {
              const uint64_t d = next_series->deltas(j);
              if (static_cast<int64_t>(d) < 0) {
                remaining_ok = false;
                break;
              }
              next_ts_ps += d;
              if (j == 0 && next_ts_ps < seg_end_ts_ps) {
                remaining_ok = false;
                break;
              }
              if (next_ts_ps >= kMaxExactTimestampPs) {
                remaining_ok = false;
                break;
              }
            }
            if (next_count > 0) {
              seg_end_ts_ps = next_ts_ps;
            }
          }
          if (remaining_ok) {
            verified_remaining_monotonic = true;
          } else {
            can_stream_chunks = false;
            chunk_already_sorted = false;
            if (track_buffer.capacity() < track_event_count) {
              track_buffer.reserve(track_event_count);
            }
          }
        }
        if (can_stream_chunks) {
          packer.PackEvents(absl::MakeSpan(track_buffer), chunk_already_sorted);
          track_buffer.clear();
          chunk_already_sorted = true;
        }
      }

      const uint32_t name_ref = i < series->name_refs_size()
                                    ? series->name_refs(i)
                                    : metadata.name_ref();
      const uint64_t dur_ps =
          i < series->durations_size() ? series->durations(i) : 0;
      local_max_end_ps = std::max(local_max_end_ps, current_ts_ps + dur_ps);

      TrackEvent ev;
      ev.ts = ev_ts;
      ev.dur = dur_ps / 1000000.0;
      ev.name_index = name_ref < interned_strings_count ? name_ref : 0;

      if (has_rle_group_ids) {
        while (rle_remaining == 0 &&
               rle_idx + 1 < series->group_id_run_lengths_size()) {
          ++rle_idx;
          rle_remaining = series->group_id_run_lengths(rle_idx);
        }
        if (rle_remaining > 0) {
          const int64_t rle_group_id = series->group_id_run_values(rle_idx);
          if (rle_group_id >= 0) {
            ev.group_id = static_cast<int32_t>(rle_group_id);
          }
          --rle_remaining;
        }
      }

      const xprof::TraceEventMetadata* ev_meta = nullptr;
      if (has_sparse_metadata) {
        while (sparse_meta_cursor < sparse_meta_count &&
               series->event_metadata_indices(sparse_meta_cursor) <
                   static_cast<uint32_t>(i)) {
          ++sparse_meta_cursor;
        }
        if (sparse_meta_cursor < sparse_meta_count &&
            series->event_metadata_indices(sparse_meta_cursor) ==
                static_cast<uint32_t>(i)) {
          ev_meta = &series->event_metadata(sparse_meta_cursor);
          ++sparse_meta_cursor;
        }
      } else if (i < series->event_metadata_size()) {
        ev_meta = &series->event_metadata(i);
      }

      tsl::profiler::ContextType category =
          tsl::profiler::ContextType::kGeneric;
      if (ev_meta != nullptr) {
        if (ev_meta->flow_id() != 0) {
          ev.flow_id = ev_meta->flow_id();
          if (ev_meta->flow_category() != 0) {
            category = GetContextTypeFromString(
                get_interned_string(ev_meta->flow_category()));
          }
        }
        if (ev.group_id < 0 && ev_meta->group_id() != 0) {
          ev.group_id = static_cast<int32_t>(ev_meta->group_id());
        }
        ev.uid = ev_meta->serial();
        if (ev_meta->has_hlo_op_ref() &&
            ev_meta->hlo_op_ref() < interned_strings_count) {
          ev.hlo_op_ref = ev_meta->hlo_op_ref();
        }
        if (ev_meta->has_hlo_module_ref() &&
            ev_meta->hlo_module_ref() < interned_strings_count) {
          ev.hlo_module_ref = ev_meta->hlo_module_ref();
        }
      }
      if (ev.flow_id != 0) {
        trace_info.flow_events_by_id[ev.flow_id].push_back(FlowEndpoint{
            .pid = pid,
            .tid = tid,
            .ts = ev.ts,
            .event_index = track_event_seq,
            .category = category,
        });
        ++flow_category_counts[static_cast<int>(category)];
      }
      if (!track_buffer.empty()) {
        const TrackEvent& prev = track_buffer.back();
        if (ev.ts < prev.ts || (ev.ts == prev.ts && ev.dur > prev.dur)) {
          chunk_already_sorted = false;
        }
      }
      track_buffer.push_back(ev);
      ++track_event_seq;
    }
    if (deltas_count > 0) {
      prev_series_end_ts_ps = current_ts_ps;
      has_prev_series_event = true;
    }
  }

  if (!track_buffer.empty()) {
    packer.PackEvents(absl::MakeSpan(track_buffer), chunk_already_sorted);
  }
  if (max_end_ps_out != nullptr) {
    *max_end_ps_out = std::max(*max_end_ps_out, local_max_end_ps);
  }
}

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
    std::vector<uint32_t>& name_ref_to_module_index, size_t max_chunk_size,
    uint64_t* max_end_ps_out, google::protobuf::Arena* arena) {
  if (series_views.empty()) return;

  bool all_can_stream_directly = true;
  size_t total_deltas_count = 0;
  for (const WireSeriesView* view : series_views) {
    if (!view->can_stream_directly) {
      all_can_stream_directly = false;
    }
    total_deltas_count += view->deltas_count;
  }

  if (!all_can_stream_directly) {
    google::protobuf::Arena local_arena;
    google::protobuf::Arena* const active_arena =
        (arena != nullptr) ? arena : &local_arena;
    std::vector<const xprof::TraceEventSeries*> parsed_ptrs;
    parsed_ptrs.reserve(series_views.size());
    for (const WireSeriesView* view : series_views) {
      xprof::TraceEventSeries* parsed_series =
          google::protobuf::Arena::Create<xprof::TraceEventSeries>(active_arena);
      if (parsed_series->ParseFromArray(
              view->series_wire.data(),
              static_cast<int>(view->series_wire.size()))) {
        parsed_ptrs.push_back(parsed_series);
      }
    }
    UnpackAndPackProtoTrack(
        pid, tid, thread_name, parsed_ptrs, get_interned_string,
        interned_strings_count, module_intervals, default_module_index,
        start_level, max_level, data, bounds, trace_info, flow_category_counts,
        track_buffer, name_ref_to_op_index, name_ref_to_module_index,
        max_chunk_size, max_end_ps_out);
    if (arena != nullptr) {
      arena->Reset();
    }
    return;
  }

  constexpr uint64_t kMaxExactTimestampPs = uint64_t{1} << 53;
  bool can_stream_chunks = (max_chunk_size > 0);
  bool verified_remaining_monotonic = false;
  const size_t target_reserve =
      can_stream_chunks ? std::min(total_deltas_count, max_chunk_size)
                        : total_deltas_count;
  track_buffer.clear();
  if (track_buffer.capacity() < target_reserve) {
    track_buffer.reserve(target_reserve);
  }

  ProtoTrackPacker packer(
      thread_name, /*is_async_track=*/false, module_intervals,
      default_module_index, start_level, max_level, data, bounds, trace_info,
      get_interned_string, name_ref_to_op_index, name_ref_to_module_index);
  bool chunk_already_sorted = true;
  uint32_t track_event_seq = 0;
  uint64_t local_max_end_ps = 0;
  uint64_t prev_series_end_ts_ps = 0;
  bool has_prev_series_event = false;
  xprof::TraceEventMetadata scratch_meta;

  for (size_t s = 0; s < series_views.size(); ++s) {
    const WireSeriesView& series_view = *series_views[s];
    const size_t deltas_count = series_view.deltas_count;

    WireByteCursor deltas_cur(series_view.deltas_bytes);
    WireByteCursor durations_cur(series_view.durations_bytes);
    WireByteCursor name_refs_cur(series_view.name_refs_bytes);
    WireByteCursor meta_indices_cur(series_view.event_metadata_indices_bytes);
    WireByteCursor rle_values_cur(series_view.group_id_run_values_bytes);
    WireByteCursor rle_lengths_cur(series_view.group_id_run_lengths_bytes);
    WireByteCursor meta_msg_cur(series_view.series_wire);

    const bool has_sparse_metadata =
        series_view.event_metadata_indices_count > 0;
    size_t sparse_meta_cursor = 0;
    const size_t sparse_meta_count =
        std::min(series_view.event_metadata_indices_count,
                 series_view.event_metadata_count);
    uint32_t current_sparse_meta_idx = 0;
    if (has_sparse_metadata && sparse_meta_count > 0) {
      meta_indices_cur.ReadVarint32(current_sparse_meta_idx);
    }

    const bool has_rle_group_ids = series_view.group_id_run_values_count > 0 &&
                                   series_view.group_id_run_values_count ==
                                       series_view.group_id_run_lengths_count;
    size_t rle_idx = 0;
    uint64_t rle_val_raw = 0;
    uint32_t rle_remaining = 0;
    if (has_rle_group_ids) {
      rle_values_cur.ReadVarint64(rle_val_raw);
      rle_lengths_cur.ReadVarint32(rle_remaining);
    }

    uint64_t current_ts_ps = 0;

    for (size_t i = 0; i < deltas_count; ++i) {
      uint64_t delta_ps = 0;
      deltas_cur.ReadVarint64(delta_ps);
      const int64_t signed_delta_ps = static_cast<int64_t>(delta_ps);
      if (signed_delta_ps < 0) {
        can_stream_chunks = false;
        chunk_already_sorted = false;
        const uint64_t abs_delta =
            static_cast<uint64_t>(-(signed_delta_ps + 1)) + 1;
        current_ts_ps =
            (current_ts_ps >= abs_delta) ? (current_ts_ps - abs_delta) : 0;
      } else {
        current_ts_ps += delta_ps;
      }
      if (i == 0 && has_prev_series_event &&
          current_ts_ps < prev_series_end_ts_ps) {
        can_stream_chunks = false;
        chunk_already_sorted = false;
      }
      if (current_ts_ps >= kMaxExactTimestampPs) {
        can_stream_chunks = false;
        chunk_already_sorted = false;
      }
      const Microseconds ev_ts = current_ts_ps / 1000000.0;
      if (can_stream_chunks && track_buffer.size() >= max_chunk_size &&
          ev_ts > track_buffer.back().ts) {
        if (!verified_remaining_monotonic) {
          WireByteCursor lookahead_cur = deltas_cur;
          uint64_t lookahead_ts_ps = current_ts_ps;
          bool remaining_ok = true;
          while (!lookahead_cur.empty()) {
            uint64_t d = 0;
            if (!lookahead_cur.ReadVarint64(d) || static_cast<int64_t>(d) < 0) {
              remaining_ok = false;
              break;
            }
            lookahead_ts_ps += d;
            if (lookahead_ts_ps >= kMaxExactTimestampPs) {
              remaining_ok = false;
              break;
            }
          }
          uint64_t seg_end_ts_ps = lookahead_ts_ps;
          for (size_t next_s = s + 1;
               next_s < series_views.size() && remaining_ok; ++next_s) {
            WireByteCursor next_cur(series_views[next_s]->deltas_bytes);
            uint64_t next_ts_ps = 0;
            bool first_in_seg = true;
            while (!next_cur.empty()) {
              uint64_t d = 0;
              if (!next_cur.ReadVarint64(d) || static_cast<int64_t>(d) < 0) {
                remaining_ok = false;
                break;
              }
              next_ts_ps += d;
              if (first_in_seg) {
                if (next_ts_ps < seg_end_ts_ps) {
                  remaining_ok = false;
                  break;
                }
                first_in_seg = false;
              }
              if (next_ts_ps >= kMaxExactTimestampPs) {
                remaining_ok = false;
                break;
              }
            }
            if (!first_in_seg) {
              seg_end_ts_ps = next_ts_ps;
            }
          }
          if (remaining_ok) {
            verified_remaining_monotonic = true;
          } else {
            can_stream_chunks = false;
            chunk_already_sorted = false;
            if (track_buffer.capacity() < total_deltas_count) {
              track_buffer.reserve(total_deltas_count);
            }
          }
        }
        if (can_stream_chunks) {
          packer.PackEvents(absl::MakeSpan(track_buffer), chunk_already_sorted);
          track_buffer.clear();
          chunk_already_sorted = true;
        }
      }

      uint32_t name_ref = series_view.series_name_ref;
      if (!name_refs_cur.empty()) {
        name_refs_cur.ReadVarint32(name_ref);
      }
      uint64_t dur_ps = 0;
      if (!durations_cur.empty()) {
        durations_cur.ReadVarint64(dur_ps);
      }
      local_max_end_ps = std::max(local_max_end_ps, current_ts_ps + dur_ps);

      TrackEvent ev;
      ev.ts = ev_ts;
      ev.dur = dur_ps / 1000000.0;
      ev.name_index = name_ref < interned_strings_count ? name_ref : 0;

      if (has_rle_group_ids) {
        while (rle_remaining == 0 &&
               rle_idx + 1 < series_view.group_id_run_lengths_count) {
          ++rle_idx;
          rle_values_cur.ReadVarint64(rle_val_raw);
          rle_lengths_cur.ReadVarint32(rle_remaining);
        }
        if (rle_remaining > 0) {
          const int64_t rle_group_id = static_cast<int64_t>(rle_val_raw);
          if (rle_group_id >= 0) {
            ev.group_id = static_cast<int32_t>(rle_group_id);
          }
          --rle_remaining;
        }
      }

      const xprof::TraceEventMetadata* ev_meta = nullptr;
      if (has_sparse_metadata) {
        while (sparse_meta_cursor < sparse_meta_count &&
               current_sparse_meta_idx < static_cast<uint32_t>(i)) {
          absl::string_view skipped_payload;
          meta_msg_cur.ReadNextEventMetadataPayload(skipped_payload);
          ++sparse_meta_cursor;
          if (sparse_meta_cursor < sparse_meta_count) {
            meta_indices_cur.ReadVarint32(current_sparse_meta_idx);
          }
        }
        if (sparse_meta_cursor < sparse_meta_count &&
            current_sparse_meta_idx == static_cast<uint32_t>(i)) {
          absl::string_view meta_payload;
          if (meta_msg_cur.ReadNextEventMetadataPayload(meta_payload)) {
            scratch_meta.Clear();
            scratch_meta.ParseFromArray(meta_payload.data(),
                                        static_cast<int>(meta_payload.size()));
            ev_meta = &scratch_meta;
          }
          ++sparse_meta_cursor;
          if (sparse_meta_cursor < sparse_meta_count) {
            meta_indices_cur.ReadVarint32(current_sparse_meta_idx);
          }
        }
      } else if (i < series_view.event_metadata_count) {
        absl::string_view meta_payload;
        if (meta_msg_cur.ReadNextEventMetadataPayload(meta_payload)) {
          scratch_meta.Clear();
          scratch_meta.ParseFromArray(meta_payload.data(),
                                      static_cast<int>(meta_payload.size()));
          ev_meta = &scratch_meta;
        }
      }

      tsl::profiler::ContextType category =
          tsl::profiler::ContextType::kGeneric;
      if (ev_meta != nullptr) {
        if (ev_meta->flow_id() != 0) {
          ev.flow_id = ev_meta->flow_id();
          if (ev_meta->flow_category() != 0) {
            category = GetContextTypeFromString(
                get_interned_string(ev_meta->flow_category()));
          }
        }
        if (ev.group_id < 0 && ev_meta->group_id() != 0) {
          ev.group_id = static_cast<int32_t>(ev_meta->group_id());
        }
        ev.uid = ev_meta->serial();
        if (ev_meta->has_hlo_op_ref() &&
            ev_meta->hlo_op_ref() < interned_strings_count) {
          ev.hlo_op_ref = ev_meta->hlo_op_ref();
        }
        if (ev_meta->has_hlo_module_ref() &&
            ev_meta->hlo_module_ref() < interned_strings_count) {
          ev.hlo_module_ref = ev_meta->hlo_module_ref();
        }
      }
      if (ev.flow_id != 0) {
        trace_info.flow_events_by_id[ev.flow_id].push_back(FlowEndpoint{
            .pid = pid,
            .tid = tid,
            .ts = ev.ts,
            .event_index = track_event_seq,
            .category = category,
        });
        ++flow_category_counts[static_cast<int>(category)];
      }
      if (!track_buffer.empty()) {
        const TrackEvent& prev = track_buffer.back();
        if (ev.ts < prev.ts || (ev.ts == prev.ts && ev.dur > prev.dur)) {
          chunk_already_sorted = false;
        }
      }
      track_buffer.push_back(ev);
      ++track_event_seq;
    }
    if (deltas_count > 0) {
      prev_series_end_ts_ps = current_ts_ps;
      has_prev_series_event = true;
    }
  }

  if (!track_buffer.empty()) {
    packer.PackEvents(absl::MakeSpan(track_buffer), chunk_already_sorted);
  }
  if (max_end_ps_out != nullptr) {
    *max_end_ps_out = std::max(*max_end_ps_out, local_max_end_ps);
  }
}

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
    std::vector<uint32_t>& name_ref_to_module_index, size_t max_chunk_size,
    uint64_t* max_end_ps_out, google::protobuf::Arena* arena) {
  const WireSeriesView* const ptr = &series_view;
  UnpackAndPackProtoSeriesWire(
      pid, tid, thread_name, absl::MakeConstSpan(&ptr, 1), get_interned_string,
      interned_strings_count, module_intervals, default_module_index,
      start_level, max_level, data, bounds, trace_info, flow_category_counts,
      track_buffer, name_ref_to_op_index, name_ref_to_module_index,
      max_chunk_size, max_end_ps_out, arena);
}

}  // namespace traceviewer
