#ifndef XPROF_FRONTEND_APP_COMPONENTS_TRACE_VIEWER_V2_TIMELINE_DRAW_UTILS_H_
#define XPROF_FRONTEND_APP_COMPONENTS_TRACE_VIEWER_V2_TIMELINE_DRAW_UTILS_H_

#include "third_party/dear_imgui/imgui.h"

namespace traceviewer {

// Draws a loading indicator in the center of the viewport.
void DrawLoadingIndicator(const ImGuiViewport* viewport);

}  // namespace traceviewer

#endif  // XPROF_FRONTEND_APP_COMPONENTS_TRACE_VIEWER_V2_TIMELINE_DRAW_UTILS_H_
