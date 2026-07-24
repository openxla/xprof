#include "frontend/app/components/trace_viewer_v2/timeline/draw_helpers.h"

#include "imgui.h"
#include "frontend/app/components/trace_viewer_v2/timeline/constants.h"

namespace traceviewer {

void DrawPinIcon(ImDrawList* draw_list, Pixel center_x, Pixel center_y,
                 Pixel icon_draw_size, ImU32 icon_col, bool is_pinned) {
  float r = icon_draw_size * 0.5f;

  // Horizontal head bar at the top
  draw_list->AddLine(ImVec2(center_x - r * 0.6f, center_y - r * 0.6f),
                     ImVec2(center_x + r * 0.6f, center_y - r * 0.6f), icon_col,
                     1.2f);
  // Head connection stem
  draw_list->AddLine(ImVec2(center_x, center_y - r * 0.6f),
                     ImVec2(center_x, center_y - r * 0.4f), icon_col, 1.2f);
  // Body center cylinder (filled if pinned, outline if unpinned)
  ImVec2 body_min(center_x - r * 0.4f, center_y - r * 0.4f);
  ImVec2 body_max(center_x + r * 0.4f, center_y + r * 0.2f);
  if (is_pinned) {
    draw_list->AddRectFilled(body_min, body_max, icon_col);
  } else {
    draw_list->AddRect(body_min, body_max, icon_col, 0.0f, 0, 1.2f);
  }
  // Pin point needle pointing down
  draw_list->AddLine(ImVec2(center_x, center_y + r * 0.2f),
                     ImVec2(center_x, center_y + r * 0.8f), icon_col, 1.2f);
}

void DrawHideIcon(ImDrawList* draw_list, Pixel center_x, Pixel center_y,
                  Pixel icon_draw_size, ImU32 icon_col, bool is_track_hidden) {
  float r = icon_draw_size * 0.5f;

  // Curve approximation using segments
  ImVec2 p0(center_x - r, center_y);
  ImVec2 p1(center_x - r * 0.5f, center_y - r * 0.45f);
  ImVec2 p2(center_x, center_y - r * 0.6f);
  ImVec2 p3(center_x + r * 0.5f, center_y - r * 0.45f);
  ImVec2 p4(center_x + r, center_y);

  // Top eye curve
  draw_list->AddLine(p0, p1, icon_col, 1.0f);
  draw_list->AddLine(p1, p2, icon_col, 1.0f);
  draw_list->AddLine(p2, p3, icon_col, 1.0f);
  draw_list->AddLine(p3, p4, icon_col, 1.0f);

  // Bottom eye curve
  ImVec2 p5(center_x - r * 0.5f, center_y + r * 0.45f);
  ImVec2 p6(center_x, center_y + r * 0.6f);
  ImVec2 p7(center_x + r * 0.5f, center_y + r * 0.45f);

  draw_list->AddLine(p0, p5, icon_col, 1.0f);
  draw_list->AddLine(p5, p6, icon_col, 1.0f);
  draw_list->AddLine(p6, p7, icon_col, 1.0f);
  draw_list->AddLine(p7, p4, icon_col, 1.0f);

  // Pupil (center)
  draw_list->AddCircleFilled(ImVec2(center_x, center_y), r * 0.25f, icon_col);

  // Slashed line for crossed eye
  if (is_track_hidden) {
    draw_list->AddLine(ImVec2(center_x - r * 0.9f, center_y - r * 0.6f),
                       ImVec2(center_x + r * 0.9f, center_y + r * 0.6f),
                       icon_col, 1.0f);
  }
}

}  // namespace traceviewer
