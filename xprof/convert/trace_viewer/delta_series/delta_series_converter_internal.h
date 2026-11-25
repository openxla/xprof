#ifndef THIRD_PARTY_XPROF_CONVERT_TRACE_VIEWER_DELTA_SERIES_DELTA_SERIES_CONVERTER_INTERNAL_H_
#define THIRD_PARTY_XPROF_CONVERT_TRACE_VIEWER_DELTA_SERIES_DELTA_SERIES_CONVERTER_INTERNAL_H_

#include <cstddef>
#include <cstdint>
#include <string>

#include "absl/base/attributes.h"
#include "absl/functional/function_ref.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "tsl/profiler/lib/context_types.h"
#include "xprof/convert/trace_viewer/delta_series/trace_data_to_compressed_delta_series_proto.h"
#include "xprof/convert/trace_viewer/delta_series/zstd_compression.h"
#include "plugin/xprof/protobuf/trace_data_response.pb.h"
#include "plugin/xprof/protobuf/trace_events.pb.h"
#include "plugin/xprof/protobuf/trace_events_raw.pb.h"

namespace tensorflow {
namespace profiler {
namespace internal {

inline constexpr absl::string_view kComputeUtilizationRooflineThreadName =
    "Compute Utilization/Roofline Efficiency";
inline constexpr absl::string_view kComputeUtilizationThreadName =
    "Compute Utilization";
inline constexpr absl::string_view kDataMotionLayersLowerThreadName =
    "Data motion layers utilization";
inline constexpr absl::string_view kDataMotionLayersTitleThreadName =
    "Data Motion Layers Utilization";

// Returns true if `res_name` is a data-motion-layer / utilization thread.
inline bool IsDataMotionLayerThreadName(absl::string_view res_name) {
  return res_name == kComputeUtilizationRooflineThreadName ||
         res_name == kComputeUtilizationThreadName ||
         res_name == kDataMotionLayersLowerThreadName ||
         res_name == kDataMotionLayersTitleThreadName;
}

// Classifies a complete-event resource thread name for HLO extraction.
HloExtractionTrackKind ClassifyHloExtractionTrack(absl::string_view res_name);

// Resolves a resource's display name from `trace` by `(pid, tid)`. The
// returned view borrows from `trace` (or is empty if absent).
inline absl::string_view ResolveResourceName(const Trace& trace
                                                 ABSL_ATTRIBUTE_LIFETIME_BOUND,
                                             uint32_t pid, uint64_t tid) {
  if (const auto dev_it = trace.devices().find(pid);
      dev_it != trace.devices().end()) {
    if (const auto res_it = dev_it->second.resources().find(tid);
        res_it != dev_it->second.resources().end() &&
        res_it->second.has_name()) {
      return res_it->second.name();
    }
  }
  return "";
}

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
      series->add_group_id_run_values(run_val_);
      series->add_group_id_run_lengths(run_len_);
      run_len_ = 0;
    }
  }

 private:
  int64_t run_val_ = 0;
  uint32_t run_len_ = 0;
};

// Returns true if `event` has a non-generic, non-legacy flow category.
inline bool HasNonGenericFlowCategory(const TraceEvent& event) {
  if (!event.has_flow_category()) return false;
  const tsl::profiler::ContextType ctx =
      tsl::profiler::GetSafeContextType(event.flow_category());
  return ctx != tsl::profiler::ContextType::kGeneric &&
         ctx != tsl::profiler::ContextType::kLegacy;
}

// Populates flow, group, and serial metadata fields from `event` using
// `string_table`.
void PopulateTraceEventMetadataWithTable(const TraceEvent& event,
                                         DeltaSeriesStringTable* string_table,
                                         xprof::TraceEventMetadata* metadata);

// Extracts HLO op and module strings from `event.raw_data()` into `out_hlo_op`
// and `out_hlo_module`, using `parse_raw_data_args` to decode the container's
// concrete RawDataType. Returns true if either output string is non-empty.
bool ExtractHloStringsFromRawData(
    const Trace& trace, const TraceEvent& event,
    HloExtractionTrackKind track_kind, absl::string_view event_name,
    absl::FunctionRef<const TraceEventArguments*(absl::string_view)>
        parse_raw_data_args,
    std::string* out_hlo_op, std::string* out_hlo_module);

// Serializes `response` header-first with an explicit `max_wire_bytes` bound
// and compresses it with `zstd_ctx` (or a one-shot compressor if null).
absl::StatusOr<std::string> SerializeTraceDataResponseHeaderFirstWithLimit(
    const xprof::TraceDataResponse& response, ZstdCompressorContext* zstd_ctx,
    size_t max_wire_bytes);

}  // namespace internal
}  // namespace profiler
}  // namespace tensorflow

#endif  // THIRD_PARTY_XPROF_CONVERT_TRACE_VIEWER_DELTA_SERIES_DELTA_SERIES_CONVERTER_INTERNAL_H_
