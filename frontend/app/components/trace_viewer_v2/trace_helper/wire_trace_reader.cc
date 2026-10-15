#include "frontend/app/components/trace_viewer_v2/trace_helper/wire_trace_reader.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "absl/strings/string_view.h"
#include "google/protobuf/arena.h"
#include "plugin/xprof/protobuf/trace_data_response.pb.h"

namespace traceviewer {

bool CountPackedVarints(absl::string_view bytes, size_t& count_out) {
  count_out = 0;
  if (bytes.empty()) return true;
  const uint8_t* ptr = reinterpret_cast<const uint8_t*>(bytes.data());
  const uint8_t* const end = ptr + bytes.size();
  if ((*(end - 1) & 0x80) != 0) return false;
  size_t count = 0;
  int cont_bytes = 0;
  for (; ptr < end; ++ptr) {
    if ((*ptr & 0x80) == 0) {
      if (cont_bytes == 9 && *ptr > 1) return false;
      ++count;
      cont_bytes = 0;
    } else {
      if (++cont_bytes >= 10) return false;
    }
  }
  count_out = count;
  return true;
}

bool SplitSerializedTraceDataResponse(
    absl::string_view serialized_proto,
    std::vector<absl::string_view>& complete_event_slices_out,
    std::string& non_complete_wire_out) {
  complete_event_slices_out.clear();
  non_complete_wire_out.clear();
  WireByteCursor cursor(serialized_proto);
  while (!cursor.empty()) {
    const uint8_t* const record_start = cursor.ptr;
    uint32_t tag = 0;
    if (!cursor.ReadVarint32(tag)) return false;
    const int field_num = static_cast<int>(tag >> 3);
    const int wire_type = static_cast<int>(tag & 7);
    if (field_num <= 0) return false;
    if (field_num == 1 && wire_type == 2) {
      absl::string_view payload;
      if (!cursor.ReadLengthDelimited(payload)) return false;
      complete_event_slices_out.push_back(payload);
    } else {
      if (!cursor.SkipWireValue(wire_type)) return false;
      non_complete_wire_out.append(
          reinterpret_cast<const char*>(record_start),
          static_cast<size_t>(cursor.ptr - record_start));
    }
  }
  return true;
}

bool TryParseWireSeriesView(absl::string_view series_wire, WireSeriesView& out,
                            uint64_t* max_end_ps_out) {
  out = WireSeriesView{};
  out.series_wire = series_wire;
  out.can_stream_directly = true;

  WireByteCursor cursor(series_wire);
  uint32_t seen_packed_mask = 0;
  xprof::TraceEventMetadata scratch_meta;

  while (!cursor.empty()) {
    uint32_t tag = 0;
    if (!cursor.ReadVarint32(tag)) return false;
    const int field_num = static_cast<int>(tag >> 3);
    const int wire_type = static_cast<int>(tag & 7);
    if (field_num <= 0) return false;

    if (wire_type != 2) {
      out.can_stream_directly = false;
      if (!cursor.SkipWireValue(wire_type)) return false;
      continue;
    }

    absl::string_view payload;
    if (!cursor.ReadLengthDelimited(payload)) return false;

    switch (field_num) {
      case 1: {
        xprof::TraceEventSeriesMetadata meta;
        if (!meta.ParseFromArray(payload.data(),
                                 static_cast<int>(payload.size()))) {
          return false;
        }
        out.process_id = meta.process_id();
        out.thread_id = meta.thread_id();
        out.series_name_ref = meta.name_ref();
        break;
      }
      case 2: {
        if ((seen_packed_mask & (1u << 2)) != 0) {
          out.can_stream_directly = false;
        } else {
          seen_packed_mask |= (1u << 2);
          out.deltas_bytes = payload;
          if (max_end_ps_out == nullptr &&
              !CountPackedVarints(payload, out.deltas_count)) {
            return false;
          }
        }
        break;
      }
      case 3: {
        if ((seen_packed_mask & (1u << 3)) != 0) {
          out.can_stream_directly = false;
        } else {
          seen_packed_mask |= (1u << 3);
          if (!payload.empty() &&
              (static_cast<uint8_t>(payload.back()) & 0x80) != 0) {
            return false;
          }
          out.durations_bytes = payload;
        }
        break;
      }
      case 4: {
        if ((seen_packed_mask & (1u << 4)) != 0) {
          out.can_stream_directly = false;
        } else {
          seen_packed_mask |= (1u << 4);
          if (!payload.empty() &&
              (static_cast<uint8_t>(payload.back()) & 0x80) != 0) {
            return false;
          }
          out.name_refs_bytes = payload;
        }
        break;
      }
      case 5: {
        scratch_meta.Clear();
        if (!scratch_meta.ParseFromArray(payload.data(),
                                         static_cast<int>(payload.size()))) {
          return false;
        }
        ++out.event_metadata_count;
        break;
      }
      case 6: {
        if ((seen_packed_mask & (1u << 6)) != 0) {
          out.can_stream_directly = false;
        } else {
          seen_packed_mask |= (1u << 6);
          out.event_metadata_indices_bytes = payload;
          if (!CountPackedVarints(payload, out.event_metadata_indices_count)) {
            return false;
          }
        }
        break;
      }
      case 7: {
        if ((seen_packed_mask & (1u << 7)) != 0) {
          out.can_stream_directly = false;
        } else {
          seen_packed_mask |= (1u << 7);
          out.group_id_run_values_bytes = payload;
          if (!CountPackedVarints(payload, out.group_id_run_values_count)) {
            return false;
          }
        }
        break;
      }
      case 8: {
        if ((seen_packed_mask & (1u << 8)) != 0) {
          out.can_stream_directly = false;
        } else {
          seen_packed_mask |= (1u << 8);
          out.group_id_run_lengths_bytes = payload;
          if (!CountPackedVarints(payload, out.group_id_run_lengths_count)) {
            return false;
          }
        }
        break;
      }
      default:
        out.can_stream_directly = false;
        break;
    }
  }

  if (!out.can_stream_directly) {
    google::protobuf::Arena arena;
    xprof::TraceEventSeries* fallback =
        google::protobuf::Arena::Create<xprof::TraceEventSeries>(&arena);
    if (!fallback->ParseFromArray(series_wire.data(),
                                  static_cast<int>(series_wire.size()))) {
      return false;
    }
    out.process_id = fallback->metadata().process_id();
    out.thread_id = fallback->metadata().thread_id();
    out.series_name_ref = fallback->metadata().name_ref();
    out.deltas_count = static_cast<size_t>(fallback->deltas_size());
    out.durations_count = static_cast<size_t>(fallback->durations_size());
    if (max_end_ps_out != nullptr) {
      uint64_t current_ts_ps = 0;
      const int n = fallback->deltas_size();
      for (int i = 0; i < n; ++i) {
        current_ts_ps += fallback->deltas(i);
        const uint64_t dur_ps =
            i < fallback->durations_size() ? fallback->durations(i) : 0;
        *max_end_ps_out = std::max(*max_end_ps_out, current_ts_ps + dur_ps);
      }
    }
    return true;
  }

  if (max_end_ps_out != nullptr) {
    WireByteCursor deltas_cur(out.deltas_bytes);
    WireByteCursor durations_cur(out.durations_bytes);
    uint64_t current_ts_ps = 0;
    uint64_t local_max_end_ps = 0;
    while (!deltas_cur.empty()) {
      uint64_t delta_ps = 0;
      if (!deltas_cur.ReadVarint64(delta_ps)) return false;
      current_ts_ps += delta_ps;
      ++out.deltas_count;
      uint64_t dur_ps = 0;
      if (!durations_cur.empty()) {
        if (!durations_cur.ReadVarint64(dur_ps)) return false;
        ++out.durations_count;
      }
      local_max_end_ps = std::max(local_max_end_ps, current_ts_ps + dur_ps);
    }
    while (!durations_cur.empty()) {
      uint64_t dur_ps = 0;
      if (!durations_cur.ReadVarint64(dur_ps)) return false;
      ++out.durations_count;
    }
    *max_end_ps_out = std::max(*max_end_ps_out, local_max_end_ps);
  }
  return true;
}

}  // namespace traceviewer
