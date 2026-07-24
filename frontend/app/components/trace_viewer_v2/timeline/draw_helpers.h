#ifndef THIRD_PARTY_XPROF_FRONTEND_APP_COMPONENTS_TRACE_VIEWER_V2_TIMELINE_DRAW_HELPERS_H_
#define THIRD_PARTY_XPROF_FRONTEND_APP_COMPONENTS_TRACE_VIEWER_V2_TIMELINE_DRAW_HELPERS_H_

#include "imgui.h"
#include "frontend/app/components/trace_viewer_v2/timeline/constants.h"

namespace traceviewer {

// Draws a pushpin icon inside a square area.
void DrawPinIcon(ImDrawList* draw_list, Pixel center_x, Pixel center_y,
                 Pixel icon_draw_size, ImU32 icon_col, bool is_pinned);

// Draws the hide/unhide icon in a square area.
void DrawHideIcon(ImDrawList* draw_list, Pixel center_x, Pixel center_y,
                  Pixel icon_draw_size, ImU32 icon_col,
                  bool is_track_hidden = false);

}  // namespace traceviewer

#endif  // THIRD_PARTY_XPROF_FRONTEND_APP_COMPONENTS_TRACE_VIEWER_V2_TIMELINE_DRAW_HELPERS_H_
