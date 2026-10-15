#ifndef THIRD_PARTY_XPROF_FRONTEND_APP_COMPONENTS_TRACE_VIEWER_V2_TRACE_HELPER_WIRE_TRACE_READER_H_
#define THIRD_PARTY_XPROF_FRONTEND_APP_COMPONENTS_TRACE_VIEWER_V2_TRACE_HELPER_WIRE_TRACE_READER_H_

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include "absl/strings/string_view.h"

namespace traceviewer {

// Lightweight zero-copy protobuf wire cursor for scanning length-delimited and
// packed varint fields without materializing full protobuf message graphs.
struct WireByteCursor {
  const uint8_t* ptr = nullptr;
  const uint8_t* end = nullptr;

  explicit WireByteCursor(absl::string_view bytes)
      : ptr(reinterpret_cast<const uint8_t*>(bytes.data())),
        end(ptr + bytes.size()) {}

  bool empty() const { return ptr >= end; }

  bool ReadVarint64(uint64_t& out) {
    if (ptr >= end) return false;
    uint64_t result = 0;
    int shift = 0;
    while (ptr < end && shift < 64) {
      const uint8_t byte = *ptr++;
      if (shift == 63 && byte > 1) return false;
      result |= static_cast<uint64_t>(byte & 0x7F) << shift;
      if ((byte & 0x80) == 0) {
        out = result;
        return true;
      }
      shift += 7;
    }
    return false;
  }

  bool ReadVarint32(uint32_t& out) {
    uint64_t val = 0;
    if (!ReadVarint64(val) || val > std::numeric_limits<uint32_t>::max()) {
      return false;
    }
    out = static_cast<uint32_t>(val);
    return true;
  }

  bool ReadLengthDelimited(absl::string_view& out) {
    uint32_t len = 0;
    if (!ReadVarint32(len)) return false;
    if (static_cast<size_t>(end - ptr) < len) return false;
    out = absl::string_view(reinterpret_cast<const char*>(ptr), len);
    ptr += len;
    return true;
  }

  bool SkipWireValue(int wire_type) {
    switch (wire_type) {
      case 0: {
        uint64_t dummy = 0;
        return ReadVarint64(dummy);
      }
      case 1: {
        if (end - ptr < 8) return false;
        ptr += 8;
        return true;
      }
      case 2: {
        absl::string_view dummy;
        return ReadLengthDelimited(dummy);
      }
      case 5: {
        if (end - ptr < 4) return false;
        ptr += 4;
        return true;
      }
      default:
        return false;
    }
  }

  // Advances to the next field-5 (event_metadata) length-delimited submessage.
  bool ReadNextEventMetadataPayload(absl::string_view& out) {
    while (!empty()) {
      uint32_t tag = 0;
      if (!ReadVarint32(tag)) return false;
      const int field_num = static_cast<int>(tag >> 3);
      const int wire_type = static_cast<int>(tag & 7);
      if (field_num == 5 && wire_type == 2) {
        return ReadLengthDelimited(out);
      }
      if (!SkipWireValue(wire_type)) return false;
    }
    return false;
  }
};

// Counts the number of varints in a packed varint byte slice and validates
// varint framing.
bool CountPackedVarints(absl::string_view bytes, size_t& count_out);

// Zero-copy wire-level view of a serialized xprof::TraceEventSeries message.
struct WireSeriesView {
  uint32_t process_id = 0;
  uint32_t thread_id = 0;
  uint32_t series_name_ref = 0;
  size_t deltas_count = 0;
  size_t durations_count = 0;
  size_t event_metadata_count = 0;
  size_t event_metadata_indices_count = 0;
  size_t group_id_run_values_count = 0;
  size_t group_id_run_lengths_count = 0;
  absl::string_view deltas_bytes;
  absl::string_view durations_bytes;
  absl::string_view name_refs_bytes;
  absl::string_view event_metadata_indices_bytes;
  absl::string_view group_id_run_values_bytes;
  absl::string_view group_id_run_lengths_bytes;
  absl::string_view series_wire;
  bool can_stream_directly = true;
};

// Splits a serialized xprof::TraceDataResponse wire buffer into zero-copy
// wire slices for `complete_events` (field 1) and a serialized wire buffer
// containing all other top-level fields. Returns false if the top-level wire
// framing is invalid.
bool SplitSerializedTraceDataResponse(
    absl::string_view serialized_proto,
    std::vector<absl::string_view>& complete_event_slices_out,
    std::string& non_complete_wire_out);

// Validates a serialized xprof::TraceEventSeries wire slice and populates
// `out` with zero-copy byte spans and element counts. If `max_end_ps_out` is
// non-null, also updates `*max_end_ps_out` with the maximum event end timestamp
// in picoseconds. Returns false if the wire format is invalid.
bool TryParseWireSeriesView(absl::string_view series_wire, WireSeriesView& out,
                            uint64_t* max_end_ps_out = nullptr);

}  // namespace traceviewer

#endif  // THIRD_PARTY_XPROF_FRONTEND_APP_COMPONENTS_TRACE_VIEWER_V2_TRACE_HELPER_WIRE_TRACE_READER_H_
