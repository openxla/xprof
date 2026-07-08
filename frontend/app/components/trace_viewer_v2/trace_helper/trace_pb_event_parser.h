#ifndef THIRD_PARTY_XPROF_FRONTEND_APP_COMPONENTS_TRACE_VIEWER_V2_TRACE_HELPER_TRACE_PB_EVENT_PARSER_H_
#define THIRD_PARTY_XPROF_FRONTEND_APP_COMPONENTS_TRACE_VIEWER_V2_TRACE_HELPER_TRACE_PB_EVENT_PARSER_H_

#include <emscripten/val.h>

#include <cstddef>
#include <cstdint>

namespace traceviewer {

class DataProvider;
class Timeline;

// Parses compressed trace data and updates the timeline in the Application
// instance. This is the main entry point for processing compressed trace data
// from the frontend.
void ParseAndProcessCompressedTraceEvents(
    uintptr_t data_ptr, size_t data_size,
    const emscripten::val& visible_range_from_url);

// Overload of ParseAndProcessCompressedTraceEvents that accepts explicitly
// injected dependencies (DataProvider and Timeline) to enable testability
// without relying on a global Application instance.
void ParseAndProcessCompressedTraceEvents(
    uintptr_t data_ptr, size_t data_size,
    const emscripten::val& visible_range_from_url, DataProvider& data_provider,
    Timeline& timeline);

}  // namespace traceviewer

#endif  // THIRD_PARTY_XPROF_FRONTEND_APP_COMPONENTS_TRACE_VIEWER_V2_TRACE_HELPER_TRACE_PB_EVENT_PARSER_H_
