#ifndef THIRD_PARTY_XPROF_CONVERT_TRACE_VIEWER_DELTA_SERIES_TRACE_DATA_TO_COMPRESSED_DELTA_SERIES_PROTO_H_
#define THIRD_PARTY_XPROF_CONVERT_TRACE_VIEWER_DELTA_SERIES_TRACE_DATA_TO_COMPRESSED_DELTA_SERIES_PROTO_H_

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/functional/any_invocable.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/platform/statusor.h"
#include "xprof/convert/trace_viewer/delta_series/zstd_compression.h"
#include "xprof/convert/trace_viewer/trace_events.h"
#include "xprof/convert/trace_viewer/trace_events_to_json.h"
#include "plugin/xprof/protobuf/trace_data_response.pb.h"
#include "plugin/xprof/protobuf/trace_events.pb.h"

namespace tensorflow {
namespace profiler {

inline constexpr absl::string_view kXChunkMagic = "XCHN";

struct DeltaSeriesProtoConversionOptions {
  bool mpmd_pipeline_view = false;
  bool mpmd_single_device_per_stage = false;
  JsonTraceOptions::Details details;
  // Device IDs whose resource threads should omit numerical sort indices,
  // enabling alphabetical sorting by resource name.
  absl::flat_hash_set<uint32_t> sort_resources_by_name;
};

// Shared string-interning table and flow/group metadata helper used by both
// one-shot (`DeltaSeriesProtoConverter`) and streaming
// (`StreamingDeltaSeriesProtoConverter`) delta-series encoders.
class DeltaSeriesStringTable {
 public:
  DeltaSeriesStringTable();

  // Interns `str` in the string table (index 0 is reserved for "") and
  // returns its 32-bit reference ID.
  uint32_t InternString(absl::string_view str);

  // Looks up a previously interned string by its reference ID.
  absl::string_view InternedString(uint32_t id) const {
    return id < interned_strings_.size()
               ? absl::string_view(interned_strings_[id])
               : absl::string_view();
  }

  // Appends any strings interned since the previous call to `response`.
  void FlushInternedStringsIncrement(xprof::TraceDataResponse* response);

  // Terminal one-shot path that moves all interned strings into `response`.
  // Must not be mixed with `FlushInternedStringsIncrement` on the same table.
  void MoveAllInternedStrings(xprof::TraceDataResponse* response) &&;

  // Populates flow and group metadata fields on `metadata`, interning any
  // non-generic flow category string.
  void PopulateFlowAndGroupMetadata(std::optional<uint64_t> flow_id,
                                    std::optional<uint32_t> flow_category,
                                    std::optional<int64_t> group_id,
                                    xprof::TraceEventMetadata* metadata);

 private:
  std::vector<std::string> interned_strings_;
  absl::flat_hash_map<std::string, uint32_t> interned_strings_map_;
  size_t num_interned_strings_flushed_ = 0;
};

// Serializes `response` and Zstd-compresses the serialized protobuf payload.
absl::StatusOr<std::string> SerializeAndCompress(
    const xprof::TraceDataResponse& response);

// Stateful helper for incremental/streaming `TraceDataResponse` encoding and
// `"XCHN"` frame emission.
class StreamingDeltaSeriesProtoConverter {
 public:
  StreamingDeltaSeriesProtoConverter() = default;

  // Interns `str` in the converter's string table (index 0 is reserved for "")
  // and returns its 32-bit reference ID.
  uint32_t InternString(absl::string_view str) {
    return string_table_.InternString(str);
  }

  // Looks up a previously interned string by its reference ID.
  absl::string_view InternedString(uint32_t id) const {
    return string_table_.InternedString(id);
  }

  // Appends any strings interned since the previous call to `response`.
  void FlushInternedStringsIncrement(xprof::TraceDataResponse* response) {
    string_table_.FlushInternedStringsIncrement(response);
  }

  // Populates flow and group metadata fields on `metadata`, interning any
  // non-generic flow category string.
  void PopulateFlowAndGroupMetadata(std::optional<uint64_t> flow_id,
                                    std::optional<uint32_t> flow_category,
                                    std::optional<int64_t> group_id,
                                    xprof::TraceEventMetadata* metadata) {
    string_table_.PopulateFlowAndGroupMetadata(flow_id, flow_category, group_id,
                                               metadata);
  }

  // Serializes `frame`, Zstd-compresses it, and appends a length-prefixed frame
  // to `out` using the `"XCHN"` streaming wire format:
  // - The 4-byte `kXChunkMagic` (`"XCHN"`) magic header appears once at the
  //   start of the stream (`include_magic_header == true`, first frame only)
  //   so clients can distinguish framed streams from a legacy single Zstd blob.
  // - It is followed by repeated
  //   `[uint32 little-endian compressed byte length][zstd(TraceDataResponse)]`
  //   data frames (`frame_len > 0`).
  // - `interned_strings` are incremental across frames: clients decode frames
  //   in order and append each frame's `interned_strings` to a running string
  //   table. If a non-first frame's `interned_strings` starts with `""` (index
  //   0), that frame carries a self-contained frame-local string table used for
  //   that frame only without replacing the cumulative table.
  // - A completed stream terminates with a 4-byte zero-length sentinel frame
  //   (`[uint32 little-endian 0]`, see `AppendEndOfStreamFrame`), allowing
  //   clients to distinguish a complete stream from one truncated mid-transfer.
  static absl::Status AppendCompressedFrame(
      const xprof::TraceDataResponse& frame, bool include_magic_header,
      std::string* out);

  // Appends the 4-byte zero-length end-of-stream sentinel frame
  // (`[uint32 little-endian 0]`) to `out`.
  static void AppendEndOfStreamFrame(std::string* out);

 private:
  DeltaSeriesStringTable string_table_;
};

// Internal class handling the stateful conversion of trace events.
// It bucketizes events by process, thread, and name, then generates columns
// and timestamp deltas prior to compression.
class DeltaSeriesProtoConverter {
 public:
  // Extracts counter values (double or uint64) from raw bytes (e.g., serialized
  // RawData) and populates the corresponding fields in TraceEventMetadata,
  // as well as optional series name.
  using CounterExtractor = absl::AnyInvocable<void(
      absl::string_view, xprof::TraceEventMetadata*, std::string* series_name)>;

  explicit DeltaSeriesProtoConverter(
      const Trace* trace, CounterExtractor counter_extractor,
      const DeltaSeriesProtoConversionOptions& options = {});

  // Generates the uncompressed TraceDataResponse protobuf.
  template <typename TraceEventsContainer>
  absl::Status GenerateResponse(const TraceEventsContainer& container,
                                xprof::TraceDataResponse* response) &&;

 private:
  xprof::TraceMetadata GetTraceMetadata() const;

  uint32_t MaybeInternString(absl::string_view str) {
    return string_table_.InternString(str);
  }

  void PopulateTraceEventMetadata(const TraceEvent& event,
                                  xprof::TraceEventMetadata* metadata);

  void AddCompleteEventTrack(uint32_t pid, uint64_t tid,
                             const TraceEventTrack& events,
                             xprof::TraceDataResponse* response);
  void AddAsyncEventTrack(uint32_t pid, absl::string_view name,
                          const TraceEventTrack& events,
                          xprof::TraceDataResponse* response);
  void AddCounterEventTrack(uint32_t pid, absl::string_view name,
                            const TraceEventTrack& events,
                            xprof::TraceDataResponse* response);

  const Trace* trace_;
  DeltaSeriesStringTable string_table_;
  CounterExtractor counter_extractor_;
  DeltaSeriesProtoConversionOptions options_;
  absl::flat_hash_map<uint32_t, uint32_t> mpmd_sort_indices_;
  absl::flat_hash_set<uint32_t> deduplicated_device_ids_;
};

// Converts TraceEventsContainer trace data into the optimized, columnar
// TraceDataResponse protobuf format, and returns the Zstd-compressed bytes.
template <typename TraceEventsContainer>
absl::StatusOr<std::string> ConvertTraceDataToCompressedDeltaSeriesProto(
    const DeltaSeriesProtoConversionOptions& options,
    const TraceEventsContainer& events) {
  typename TraceEventsContainer::RawDataType raw_data;
  auto extractor = [&raw_data](absl::string_view raw_bytes,
                               xprof::TraceEventMetadata* metadata,
                               std::string* series_name) {
    if (!raw_data.ParseFromArray(raw_bytes.data(), raw_bytes.size())) {
      LOG(ERROR) << "Failed to parse raw_data for counter event";
      return;
    }
    if (raw_data.has_args() && raw_data.args().arg_size() > 0) {
      const auto& arg = raw_data.args().arg(0);
      if (series_name != nullptr && arg.has_name()) {
        *series_name = arg.name();
      }
      if (arg.has_double_value()) {
        metadata->set_counter_value_double(arg.double_value());
      } else if (arg.has_uint_value()) {
        metadata->set_counter_value_uint64(arg.uint_value());
      }
    }
  };

  xprof::TraceDataResponse response;
  DeltaSeriesProtoConverter converter(&events.trace(), std::move(extractor),
                                      options);

  if (auto status = std::move(converter).GenerateResponse(events, &response);
      !status.ok()) {
    return status;
  }

  return SerializeAndCompress(response);
}

template <typename TraceEventsContainer>
absl::Status DeltaSeriesProtoConverter::GenerateResponse(
    const TraceEventsContainer& container,
    xprof::TraceDataResponse* response) && {
  if (options_.mpmd_pipeline_view) {
    SortMpmdDevices(container, mpmd_sort_indices_,
                    options_.mpmd_single_device_per_stage,
                    &deduplicated_device_ids_);
  }
  *response->mutable_metadata() = GetTraceMetadata();

  for (const auto& detail : options_.details) {
    auto* d = response->add_details();
    d->set_name(detail.first);
    d->set_value(detail.second);
  }

  if (trace_->has_min_timestamp_ps()) {
    response->set_full_timespan_start_ps(trace_->min_timestamp_ps());
  }
  if (trace_->has_max_timestamp_ps()) {
    response->set_full_timespan_end_ps(trace_->max_timestamp_ps());
  }

  using TidOrName = std::variant<uint64_t, absl::string_view>;

  container.ForAllTracks([this, response](uint32_t pid, TidOrName tid_or_name,
                                          const TraceEventTrack& events) {
    if (options_.mpmd_single_device_per_stage &&
        deduplicated_device_ids_.contains(pid)) {
      return true;
    }
    if (events.empty()) return true;
    if (const uint64_t* tid = std::get_if<uint64_t>(&tid_or_name)) {
      AddCompleteEventTrack(pid, *tid, events, response);
    } else {
      absl::string_view name = std::get<absl::string_view>(tid_or_name);
      const TraceEvent& first_event = *events[0];
      if (first_event.has_flow_id()) {
        AddAsyncEventTrack(pid, name, events, response);
      } else {
        AddCounterEventTrack(pid, name, events, response);
      }
    }
    return true;
  });

  std::move(string_table_).MoveAllInternedStrings(response);
  return absl::OkStatus();
}
}  // namespace profiler
}  // namespace tensorflow

#endif  // THIRD_PARTY_XPROF_CONVERT_TRACE_VIEWER_DELTA_SERIES_TRACE_DATA_TO_COMPRESSED_DELTA_SERIES_PROTO_H_
