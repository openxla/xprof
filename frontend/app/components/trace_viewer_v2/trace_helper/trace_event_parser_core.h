#ifndef THIRD_PARTY_XPROF_FRONTEND_APP_COMPONENTS_TRACE_VIEWER_V2_TRACE_HELPER_TRACE_EVENT_PARSER_CORE_H_
#define THIRD_PARTY_XPROF_FRONTEND_APP_COMPONENTS_TRACE_VIEWER_V2_TRACE_HELPER_TRACE_EVENT_PARSER_CORE_H_

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "tsl/profiler/lib/context_types.h"
#include "frontend/app/components/trace_viewer_v2/trace_helper/trace_event.h"
#include "plugin/xprof/protobuf/trace_data_response.pb.h"

namespace traceviewer {

// Converts a string category to its corresponding ContextType enum.
tsl::profiler::ContextType GetContextTypeFromString(absl::string_view category);

Phase ParsePhase(absl::string_view ph_str);

const absl::flat_hash_map<tsl::profiler::ContextType, absl::string_view>&
GetPrettyNames();

// Generates a stable event ID robust against sub-microsecond floating-point
// representation differences (e.g. string round-tripping vs integer division).
EventId GenerateEventId(absl::string_view name, Microseconds ts,
                        Microseconds dur);

// Core Processing Functions for Protobuf Unmarshalling
void ProcessProcessMetadata(const xprof::Process& process,
                            ParsedTraceEvents& result);

void ProcessMetadataEvents(const xprof::TraceDataResponse& response,
                           ParsedTraceEvents& result);

size_t ProcessCompleteEventSeriesSlice(
    const xprof::TraceEventSeries& series,
    absl::Span<const std::string> interned_strings, size_t start_idx,
    size_t max_events, uint64_t& current_ts_ps, ParsedTraceEvents& result,
    std::optional<std::pair<uint64_t, uint64_t>> filter_range_ps =
        std::nullopt);

size_t ProcessOverviewSampledSeriesSlice(
    const xprof::TraceEventSeries& series,
    absl::Span<const std::string> interned_strings, size_t start_idx,
    size_t max_events, uint64_t& current_timestamp_ps,
    ParsedTraceEvents& result, bool& did_coalesce);

void ProcessCompleteEvents(const xprof::TraceDataResponse& response,
                           ParsedTraceEvents& result);

void ProcessAsyncEventSeries(const xprof::TraceEventSeries& series,
                             absl::Span<const std::string> interned_strings,
                             ParsedTraceEvents& result);

void ProcessAsyncEvents(const xprof::TraceDataResponse& response,
                        ParsedTraceEvents& result);

void ProcessCounterEventSeries(const xprof::TraceEventSeries& series,
                               absl::Span<const std::string> interned_strings,
                               ParsedTraceEvents& result);

void ProcessCounterEvents(const xprof::TraceDataResponse& response,
                          ParsedTraceEvents& result);

}  // namespace traceviewer

#endif  // THIRD_PARTY_XPROF_FRONTEND_APP_COMPONENTS_TRACE_VIEWER_V2_TRACE_HELPER_TRACE_EVENT_PARSER_CORE_H_
