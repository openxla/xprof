#ifndef THIRD_PARTY_XPROF_FRONTEND_APP_COMPONENTS_TRACE_VIEWER_V2_TRACE_HELPER_TRACE_EVENT_PARSER_H_
#define THIRD_PARTY_XPROF_FRONTEND_APP_COMPONENTS_TRACE_VIEWER_V2_TRACE_HELPER_TRACE_EVENT_PARSER_H_

#include <emscripten/val.h>

#include "absl/strings/string_view.h"
#include "tsl/profiler/lib/context_types.h"
#include "frontend/app/components/trace_viewer_v2/trace_helper/trace_event.h"

namespace traceviewer {

class DataProvider;
class Timeline;

ParsedTraceEvents ParseTraceEvents(
    const emscripten::val& trace_data,
    const emscripten::val& visible_range_from_url);

ParsedTraceEvents ParseTraceEvents(
    const emscripten::val& trace_data,
    const emscripten::val& visible_range_from_url,
    absl::flat_hash_map<std::pair<ProcessId, std::string>, TraceEvent>&
        open_async_events);

void ParseAndProcessTraceEvents(const emscripten::val& trace_data,
                                const emscripten::val& visible_range_from_url);

void ParseAndProcessTraceEvents(const emscripten::val& trace_data,
                                const emscripten::val& visible_range_from_url,
                                DataProvider& data_provider,
                                Timeline& timeline);

// Parses trace data containing search results and updates the timeline.
void SetSearchResultsInWasm(const emscripten::val& trace_data);

// Converts a string category to its corresponding ContextType enum.
tsl::profiler::ContextType GetContextTypeFromString(absl::string_view category);

// Returns all flow categories.
emscripten::val GetAllFlowCategories();

}  // namespace traceviewer

#endif  // THIRD_PARTY_XPROF_FRONTEND_APP_COMPONENTS_TRACE_VIEWER_V2_TRACE_HELPER_TRACE_EVENT_PARSER_H_
