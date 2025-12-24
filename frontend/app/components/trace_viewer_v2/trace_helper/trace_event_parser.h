#ifndef THIRD_PARTY_XPROF_FRONTEND_APP_COMPONENTS_TRACE_VIEWER_V2_TRACE_HELPER_TRACE_EVENT_PARSER_H_
#define THIRD_PARTY_XPROF_FRONTEND_APP_COMPONENTS_TRACE_VIEWER_V2_TRACE_HELPER_TRACE_EVENT_PARSER_H_

#include <emscripten/val.h>

#include "xprof/frontend/app/components/trace_viewer_v2/trace_helper/trace_event.h"

namespace traceviewer {

ParsedTraceEvents ParseTraceEvents(
    const emscripten::val& trace_data,
    const emscripten::val& visible_range_from_url);

}  // namespace traceviewer

#endif  // THIRD_PARTY_XPROF_FRONTEND_APP_COMPONENTS_TRACE_VIEWER_V2_TRACE_HELPER_TRACE_EVENT_PARSER_H_
