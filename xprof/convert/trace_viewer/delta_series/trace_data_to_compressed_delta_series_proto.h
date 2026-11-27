#ifndef THIRD_PARTY_XPROF_CONVERT_TRACE_VIEWER_DELTA_SERIES_TRACE_DATA_TO_COMPRESSED_DELTA_SERIES_PROTO_H_
#define THIRD_PARTY_XPROF_CONVERT_TRACE_VIEWER_DELTA_SERIES_TRACE_DATA_TO_COMPRESSED_DELTA_SERIES_PROTO_H_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <variant>

#include "absl/base/attributes.h"
#include "absl/base/optimization.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/functional/any_invocable.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xprof/convert/trace_viewer/delta_series/zstd_compression.h"  // IWYU pragma: export
#include "xprof/convert/trace_viewer/trace_events.h"
#include "xprof/convert/trace_viewer/trace_events_to_json.h"
#include "plugin/xprof/protobuf/trace_data_response.pb.h"
#include "plugin/xprof/protobuf/trace_events.pb.h"
#include "plugin/xprof/protobuf/trace_events_raw.pb.h"

namespace tensorflow {
namespace profiler {

struct DeltaSeriesProtoConversionOptions {
  bool mpmd_pipeline_view = false;
  bool mpmd_single_device_per_stage = false;
  // When true, encodes group_id using run-length encoding and emits
  // TraceEventMetadata entries sparsely via event_metadata_indices (including
  // HLO op/module refs). Defaults to false so legacy/OSS callers continue to
  // receive dense 1:1 event_metadata.
  bool sparse_event_metadata = false;
  JsonTraceOptions::Details details;
  // Device IDs whose resource threads should omit numerical sort indices,
  // enabling alphabetical sorting by resource name.
  absl::flat_hash_set<uint32_t> sort_resources_by_name;
};

// Shared string-interning table and flow/group metadata helper used by
// `DeltaSeriesProtoConverter`.
class DeltaSeriesStringTable {
 public:
  DeltaSeriesStringTable();
  DeltaSeriesStringTable(const DeltaSeriesStringTable&) = delete;
  DeltaSeriesStringTable& operator=(const DeltaSeriesStringTable&) = delete;
  DeltaSeriesStringTable(DeltaSeriesStringTable&&) = default;
  DeltaSeriesStringTable& operator=(DeltaSeriesStringTable&&) = default;

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
  // `interned_strings_map_` stores non-owning `absl::string_view` keys that
  // point into elements of `interned_strings_`. `std::deque` keeps element
  // addresses stable across `emplace_back` and move operations. Keep
  // `interned_strings_` declared before `interned_strings_map_` so destruction
  // destroys the map before the strings. `MoveAllInternedStrings` clears the
  // map before moving strings out of the deque.
  std::deque<std::string> interned_strings_;
  absl::flat_hash_map<absl::string_view, uint32_t> interned_strings_map_;
  size_t num_interned_strings_flushed_ = 0;
};

// Run-length encoder for per-event group_id values into a TraceEventSeries.
class GroupIdRunBuilder {
 public:
  void Add(int64_t group_id, xprof::TraceEventSeries* series) {
    if (run_len_ == 0) {
      run_val_ = group_id;
      run_len_ = 1;
    } else if (group_id == run_val_) {
      ++run_len_;
    } else {
      series->add_group_id_run_values(run_val_);
      series->add_group_id_run_lengths(run_len_);
      run_val_ = group_id;
      run_len_ = 1;
    }
  }

  void Flush(xprof::TraceEventSeries* series) {
    if (run_len_ > 0) {
      DCHECK(series != nullptr);
      series->add_group_id_run_values(run_val_);
      series->add_group_id_run_lengths(run_len_);
      run_len_ = 0;
    }
  }

 private:
  int64_t run_val_ = 0;
  uint32_t run_len_ = 0;
};

// Maximum complete events per TraceEventSeries chunk.
inline constexpr size_t kMaxEventsPerSeriesChunk = 65536;

// Serializes `response` to wire format with header fields (fields 2..8)
// placed before complete_events (field 1 chunks), and returns Zstd-compressed
// bytes. If `zstd_ctx` is non-null, reuses its persistent ZSTD_CCtx state.
absl::StatusOr<std::string> SerializeTraceDataResponseHeaderFirst(
    const xprof::TraceDataResponse& response,
    ZstdCompressorContext* zstd_ctx = nullptr);

// Serializes `response` and Zstd-compresses the serialized protobuf payload.
absl::StatusOr<std::string> SerializeAndCompress(
    const xprof::TraceDataResponse& response);

namespace internal {

// Track classification controlling HLO op and module extraction rules.
enum class HloExtractionTrackKind : uint8_t {
  kRegular,
  kXlaOps,
  kDataMotionLayer,
};

// Cache mapping `TraceEvent::name_ref()` keys from one `Trace` to interned
// string IDs and resolved names, with a 1-slot MRU fast path.
// `*trace` must be non-null, outlive this cache, and keep `name_table()`
// unmodified for the lifetime of the cache. All calls on one cache instance
// must use the same string-interning table.
class TraceNameRefInternCache {
 public:
  struct CachedEntry {
    uint32_t interned_id = 0;
    // Non-owning view backed by `trace->name_table()` when `ev.has_name_ref()`
    // is true (empty if absent), or by `ev.name()` when `ev.has_name_ref()` is
    // false.
    absl::string_view resolved_name;
  };

  explicit TraceNameRefInternCache(
      const Trace* trace ABSL_ATTRIBUTE_LIFETIME_BOUND)
      : trace_(trace) {
    DCHECK(trace_ != nullptr);
  }
  TraceNameRefInternCache(const TraceNameRefInternCache&) = delete;
  TraceNameRefInternCache& operator=(const TraceNameRefInternCache&) = delete;
  TraceNameRefInternCache(TraceNameRefInternCache&&) = default;
  TraceNameRefInternCache& operator=(TraceNameRefInternCache&&) = default;

  // Resolves `ev`'s name against `*trace_` and interns it via `intern_fn`.
  // Both `*trace_` and `ev` must outlive any use of `resolved_name`.
  template <typename InternFn>
  CachedEntry ResolveAndIntern(const TraceEvent& ev
                                   ABSL_ATTRIBUTE_LIFETIME_BOUND,
                               InternFn&& intern_fn) {
    if (!ev.has_name_ref()) {
      const absl::string_view name = ev.name();
      return {intern_fn(name), name};
    }
    const uint64_t ref = ev.name_ref();
    if (has_last_ && ref == last_trace_ref_) {
      return last_entry_;
    }
    const auto cache_it = map_.find(ref);
    if (cache_it != map_.end()) {
      has_last_ = true;
      last_trace_ref_ = ref;
      last_entry_ = cache_it->second;
      return last_entry_;
    }
    const auto trace_it = trace_->name_table().find(ref);
    if (trace_it == trace_->name_table().end()) {
      // Do not insert missing keys into `map_` so `map_.size()` stays bounded
      // by `trace_->name_table().size()`.
      return {intern_fn(absl::string_view()), absl::string_view()};
    }
    const absl::string_view resolved = trace_it->second;
    const CachedEntry entry{intern_fn(resolved), resolved};
    map_.try_emplace(ref, entry);
    has_last_ = true;
    last_trace_ref_ = ref;
    last_entry_ = entry;
    return entry;
  }

  // Releases cached entries, resets MRU state, and frees map storage.
  void Clear() {
    has_last_ = false;
    last_trace_ref_ = 0;
    last_entry_ = {};
    map_ = {};
  }

 private:
  const Trace* trace_;
  bool has_last_ = false;
  uint64_t last_trace_ref_ = 0;
  CachedEntry last_entry_;
  absl::flat_hash_map<uint64_t, CachedEntry> map_;
};

}  // namespace internal

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

  // Returns parsed TraceEventArguments for a serialized RawData payload, or
  // nullptr if parsing fails or no args are present. The returned pointer
  // remains valid until the next invocation of the extractor.
  using RawDataArgsExtractor =
      absl::AnyInvocable<const TraceEventArguments*(absl::string_view)>;

  explicit DeltaSeriesProtoConverter(
      const Trace* trace, CounterExtractor counter_extractor,
      const DeltaSeriesProtoConversionOptions& options = {},
      RawDataArgsExtractor raw_data_args_extractor = nullptr);

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

  void MaybeEmitSparseEventMetadata(const TraceEvent& event,
                                    absl::string_view event_name,
                                    internal::HloExtractionTrackKind track_kind,
                                    std::optional<uint64_t> flow_id,
                                    uint32_t emitted_index,
                                    xprof::TraceEventSeries* series);

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
  internal::TraceNameRefInternCache name_cache_;
  std::string scratch_hlo_op_;
  std::string scratch_hlo_module_;
  CounterExtractor counter_extractor_;
  RawDataArgsExtractor raw_data_args_extractor_;
  DeltaSeriesProtoConversionOptions options_;
  absl::flat_hash_map<uint32_t, uint32_t> mpmd_sort_indices_;
  absl::flat_hash_set<uint32_t> deduplicated_device_ids_;
};

namespace internal {

// Maximum serialized size allowed by Protobuf's
// `SerializeWithCachedSizesToArray`
// (`GetCachedSize()` returns signed `int`).
inline constexpr size_t kMaxProtobufSerializedBytes =
    static_cast<size_t>(std::numeric_limits<int>::max());

// Returns OK if `byte_size` fits within `max_bytes` (clamped to Protobuf's
// 2 GiB signed `int` limit), otherwise returns ResourceExhaustedError.
inline absl::Status ValidateProtobufMessageSize(
    size_t byte_size, absl::string_view label,
    size_t max_bytes = kMaxProtobufSerializedBytes) {
  const size_t effective_limit =
      std::min(max_bytes, kMaxProtobufSerializedBytes);
  if (ABSL_PREDICT_FALSE(byte_size > effective_limit)) {
    return absl::ResourceExhaustedError(absl::StrCat(
        label, " serialized size (", byte_size,
        " bytes) exceeds maximum message limit (", effective_limit,
        " bytes)."));
  }
  return absl::OkStatus();
}

// Adds `delta` to `*total` with overflow checking against `max_bytes`
// (clamped to `kMaxProtobufSerializedBytes`).
inline absl::Status AddWireBytesChecked(
    size_t* total, size_t delta, absl::string_view label,
    size_t max_bytes = kMaxProtobufSerializedBytes) {
  const size_t effective_limit =
      std::min(max_bytes, kMaxProtobufSerializedBytes);
  if (ABSL_PREDICT_FALSE(delta > effective_limit ||
                         *total > effective_limit - delta)) {
    return absl::ResourceExhaustedError(
        absl::StrCat(label, " serialized size exceeds maximum message limit (",
                     effective_limit, " bytes)."));
  }
  *total += delta;
  return absl::OkStatus();
}

template <typename RawDataType>
DeltaSeriesProtoConverter::CounterExtractor MakeDefaultCounterExtractor() {
  return [raw_data = RawDataType()](absl::string_view raw_bytes,
                                    xprof::TraceEventMetadata* metadata,
                                    std::string* series_name) mutable {
    if (!raw_data.ParseFromString(raw_bytes)) {
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
}

template <typename RawDataType>
DeltaSeriesProtoConverter::RawDataArgsExtractor
MakeDefaultRawDataArgsExtractor() {
  return
      [raw_data = RawDataType()](
          absl::string_view raw_bytes) mutable -> const TraceEventArguments* {
        if (!raw_data.ParseFromString(raw_bytes) || !raw_data.has_args()) {
          return nullptr;
        }
        return &raw_data.args();
      };
}

}  // namespace internal

// Converts TraceEventsContainer trace data into the optimized, columnar
// TraceDataResponse protobuf format, and returns the Zstd-compressed bytes.
template <typename TraceEventsContainer>
absl::StatusOr<std::string> ConvertTraceDataToCompressedDeltaSeriesProto(
    const DeltaSeriesProtoConversionOptions& options,
    const TraceEventsContainer& events) {
  xprof::TraceDataResponse response;
  DeltaSeriesProtoConverter converter(
      &events.trace(),
      internal::MakeDefaultCounterExtractor<
          typename TraceEventsContainer::RawDataType>(),
      options,
      internal::MakeDefaultRawDataArgsExtractor<
          typename TraceEventsContainer::RawDataType>());

  if (absl::Status status =
          std::move(converter).GenerateResponse(events, &response);
      !status.ok()) {
    return status;
  }

  return SerializeTraceDataResponseHeaderFirst(response);
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
    xprof::DetailItem* d = response->add_details();
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

  name_cache_.Clear();
  std::string().swap(scratch_hlo_op_);
  std::string().swap(scratch_hlo_module_);
  std::move(string_table_).MoveAllInternedStrings(response);
  return absl::OkStatus();
}

}  // namespace profiler
}  // namespace tensorflow

#endif  // THIRD_PARTY_XPROF_CONVERT_TRACE_VIEWER_DELTA_SERIES_TRACE_DATA_TO_COMPRESSED_DELTA_SERIES_PROTO_H_
