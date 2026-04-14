#ifndef THIRD_PARTY_XPROF_CONVERT_TRACE_VIEWER_DELTA_SERIES_TRACE_DATA_TO_COMPRESSED_DELTA_SERIES_PROTO_H_
#define THIRD_PARTY_XPROF_CONVERT_TRACE_VIEWER_DELTA_SERIES_TRACE_DATA_TO_COMPRESSED_DELTA_SERIES_PROTO_H_

#include <cstdint>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_map.h"
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

struct DeltaSeriesProtoConversionOptions {
  bool mpmd_pipeline_view = false;
};

// Internal class handling the stateful conversion of trace events.
// It bucketizes events by process, thread, and name, then generates columns
// and timestamp deltas prior to compression.
class DeltaSeriesProtoConverter {
 public:
  // Extracts counter values (double or uint64) from raw bytes (e.g., serialized
  // RawData) and populates the corresponding fields in TraceEventMetadata.
  using CounterExtractor =
      absl::AnyInvocable<void(absl::string_view, xprof::TraceEventMetadata*)>;

  explicit DeltaSeriesProtoConverter(
      const Trace* trace, CounterExtractor counter_extractor,
      const DeltaSeriesProtoConversionOptions& options = {});

  // Generates the uncompressed TraceDataResponse protobuf.
  template <typename TraceEventsContainer>
  absl::Status GenerateResponse(const TraceEventsContainer& container,
                                xprof::TraceDataResponse* response) &&;

 private:
  xprof::TraceMetadata GetTraceMetadata() const;

  uint32_t MaybeInternString(absl::string_view str);

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

  // String interning state.
  std::vector<std::string> interned_strings_;
  absl::flat_hash_map<std::string, uint32_t> interned_strings_map_;

  CounterExtractor counter_extractor_;
  DeltaSeriesProtoConversionOptions options_;
  absl::flat_hash_map<uint32_t, uint32_t> mpmd_sort_indices_;
};

// Converts TraceEventsContainer trace data into the optimized, columnar
// TraceDataResponse protobuf format, and returns the Zstd-compressed bytes.
template <typename TraceEventsContainer>
absl::StatusOr<std::string> ConvertTraceDataToCompressedDeltaSeriesProto(
    const DeltaSeriesProtoConversionOptions& options,
    const TraceEventsContainer& events) {
  typename TraceEventsContainer::RawDataType raw_data;
  auto extractor = [&raw_data](absl::string_view raw_bytes,
                               xprof::TraceEventMetadata* metadata) {
    if (!raw_data.ParseFromArray(raw_bytes.data(), raw_bytes.size())) {
      LOG(ERROR) << "Failed to parse raw_data for counter event";
      return;
    }
    if (raw_data.has_args() && raw_data.args().arg_size() > 0) {
      const auto& arg = raw_data.args().arg(0);
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

  std::string serialized_proto;
  if (!response.SerializeToString(&serialized_proto)) {
    return absl::InternalError("Failed to serialize TraceDataResponse.");
  }

  std::string compressed_result;
  TF_ASSIGN_OR_RETURN(compressed_result,
                      ZstdCompression::Compress(serialized_proto));
  return compressed_result;
}

template <typename TraceEventsContainer>
absl::Status DeltaSeriesProtoConverter::GenerateResponse(
    const TraceEventsContainer& container,
    xprof::TraceDataResponse* response) && {
  if (options_.mpmd_pipeline_view) {
    SortMpmdDevices(container, mpmd_sort_indices_);
  }
  *response->mutable_metadata() = GetTraceMetadata();

  using TidOrName = std::variant<uint64_t, absl::string_view>;

  container.ForAllTracks([this, response](uint32_t pid, TidOrName tid_or_name,
                                          const TraceEventTrack& events) {
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

  for (std::string& str : interned_strings_) {
    response->add_interned_strings(std::move(str));
  }
  return absl::OkStatus();
}
}  // namespace profiler
}  // namespace tensorflow

#endif  // THIRD_PARTY_XPROF_CONVERT_TRACE_VIEWER_DELTA_SERIES_TRACE_DATA_TO_COMPRESSED_DELTA_SERIES_PROTO_H_
