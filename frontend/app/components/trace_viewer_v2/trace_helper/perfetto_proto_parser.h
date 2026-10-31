#ifndef THIRD_PARTY_XPROF_FRONTEND_APP_COMPONENTS_TRACE_VIEWER_V2_TRACE_HELPER_PERFETTO_PROTO_PARSER_H_
#define THIRD_PARTY_XPROF_FRONTEND_APP_COMPONENTS_TRACE_VIEWER_V2_TRACE_HELPER_PERFETTO_PROTO_PARSER_H_

#include <emscripten/val.h>


#include "frontend/app/components/trace_viewer_v2/trace_helper/trace_event.h"

namespace traceviewer {

ParsedTraceEvents ParsePerfettoTraceEvents(const void* data, size_t size,
                                           bool normalize_timestamps = true);

// Parses and processes Perfetto trace events from a memory buffer.
// data_ptr: A pointer to the memory address where Perfetto trace data is
// stored.
// data_size: The size of the Perfetto trace data in bytes.
void ParseAndProcessPerfettoTraceEvents(
    int data_ptr, int data_size, const emscripten::val& visible_range_from_url,
    bool normalize_timestamps = true);

}  // namespace traceviewer

#endif  // THIRD_PARTY_XPROF_FRONTEND_APP_COMPONENTS_TRACE_VIEWER_V2_TRACE_HELPER_PERFETTO_PROTO_PARSER_H_
