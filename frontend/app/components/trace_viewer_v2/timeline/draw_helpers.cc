#include "frontend/app/components/trace_viewer_v2/timeline/draw_helpers.h"

#include "imgui.h"
#include "frontend/app/components/trace_viewer_v2/timeline/constants.h"

// Define the UTF-8 representations of Unicode PUA codepoints (matching
// mapping.json).
#define ICON_PIN_BUTTON "\xee\x80\x81"      // U+E001 (decimal 57345)
#define ICON_UNPIN_BUTTON "\xee\x80\x82"    // U+E002 (decimal 57346)
#define ICON_HIDDEN_BUTTON "\xee\x80\x83"   // U+E003 (decimal 57347)
#define ICON_VISIBLE_BUTTON "\xee\x80\x84"  // U+E004 (decimal 57348)

namespace traceviewer {

void DrawPinIcon(ImDrawList* draw_list, Pixel center_x, Pixel center_y,
                 Pixel icon_draw_size, ImU32 icon_col, bool is_pinned) {
  const char* icon_str = is_pinned ? ICON_UNPIN_BUTTON : ICON_PIN_BUTTON;
  ImVec2 text_size = ImGui::CalcTextSize(icon_str);

  // Center the glyph within the button square
  ImVec2 pos(center_x - text_size.x * 0.5f, center_y - text_size.y * 0.5f);

  draw_list->AddText(pos, icon_col, icon_str);
}

void DrawHideIcon(ImDrawList* draw_list, Pixel center_x, Pixel center_y,
                  Pixel icon_draw_size, ImU32 icon_col, bool is_track_hidden) {
  const char* icon_str =
      is_track_hidden ? ICON_VISIBLE_BUTTON : ICON_HIDDEN_BUTTON;
  ImVec2 text_size = ImGui::CalcTextSize(icon_str);

  // Center the glyph within the button square
  ImVec2 pos(center_x - text_size.x * 0.5f, center_y - text_size.y * 0.5f);

  draw_list->AddText(pos, icon_col, icon_str);
}

}  // namespace traceviewer
