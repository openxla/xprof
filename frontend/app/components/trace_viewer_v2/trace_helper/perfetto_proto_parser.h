#ifndef THIRD_PARTY_XPROF_FRONTEND_APP_COMPONENTS_TRACE_VIEWER_V2_TRACE_HELPER_PERFETTO_PROTO_PARSER_H_
#define THIRD_PARTY_XPROF_FRONTEND_APP_COMPONENTS_TRACE_VIEWER_V2_TRACE_HELPER_PERFETTO_PROTO_PARSER_H_

#include <emscripten/val.h>

#include <string>

namespace traceviewer {

void ParseAndProcessPerfettoTraceEvents(
    const std::string& buffer_data,
    const emscripten::val& visible_range_from_url);

}  // namespace traceviewer

#endif  // THIRD_PARTY_XPROF_FRONTEND_APP_COMPONENTS_TRACE_VIEWER_V2_TRACE_HELPER_PERFETTO_PROTO_PARSER_H_
