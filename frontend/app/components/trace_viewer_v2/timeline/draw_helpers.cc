#include "frontend/app/components/trace_viewer_v2/timeline/draw_helpers.h"

#include <cmath>

#include "imgui.h"
#include "imgui_internal.h"
#include "frontend/app/components/trace_viewer_v2/timeline/constants.h"

// Define the UTF-8 representations of Unicode PUA codepoints (matching
// mapping.json).
#define ICON_PIN_BUTTON "\xee\x80\x81"      // U+E001 (decimal 57345)
#define ICON_UNPIN_BUTTON "\xee\x80\x82"    // U+E002 (decimal 57346)
#define ICON_HIDDEN_BUTTON "\xee\x80\x83"   // U+E003 (decimal 57347)
#define ICON_VISIBLE_BUTTON "\xee\x80\x84"  // U+E004 (decimal 57348)

namespace traceviewer {
namespace {

// Draws a single-glyph UTF-8 string so that the glyph's visible bounding box
// (not its text line box, which includes ascent / descent padding) is centered
// on (center_x, center_y). Falls back to line-box centering if the glyph is
// missing from the current font.
void DrawCenteredGlyph(ImDrawList* draw_list, const char* glyph_utf8,
                       Pixel center_x, Pixel center_y, ImU32 color) {
  unsigned int codepoint = 0;
  ImTextCharFromUtf8(&codepoint, glyph_utf8, nullptr);
  ImVec2 offset;
  const ImFontGlyph* glyph = ImGui::GetFontBaked()->FindGlyphNoFallback(
      static_cast<ImWchar>(codepoint));
  if (glyph != nullptr && glyph->X1 > glyph->X0 && glyph->Y1 > glyph->Y0) {
    offset =
        ImVec2((glyph->X0 + glyph->X1) * 0.5f, (glyph->Y0 + glyph->Y1) * 0.5f);
  } else {
    const ImVec2 text_size = ImGui::CalcTextSize(glyph_utf8);
    offset = ImVec2(text_size.x * 0.5f, text_size.y * 0.5f);
  }
  // Whole-pixel positions keep the icon crisp.
  draw_list->AddText(
      ImVec2(std::round(center_x - offset.x), std::round(center_y - offset.y)),
      color, glyph_utf8);
}

}  // namespace

void DrawPinIcon(ImDrawList* draw_list, Pixel center_x, Pixel center_y,
                 Pixel icon_draw_size, ImU32 icon_col, bool is_pinned) {
  const char* icon_str = is_pinned ? ICON_UNPIN_BUTTON : ICON_PIN_BUTTON;
  DrawCenteredGlyph(draw_list, icon_str, center_x, center_y, icon_col);
}

void DrawHideIcon(ImDrawList* draw_list, Pixel center_x, Pixel center_y,
                  Pixel icon_draw_size, ImU32 icon_col, bool is_track_hidden) {
  const char* icon_str =
      is_track_hidden ? ICON_VISIBLE_BUTTON : ICON_HIDDEN_BUTTON;
  DrawCenteredGlyph(draw_list, icon_str, center_x, center_y, icon_col);
}

void DrawDragHandleIcon(ImDrawList* draw_list, Pixel center_x, Pixel center_y,
                        ImU32 icon_col) {
  // Whole-pixel edges keep the lines crisp.
  const Pixel left = std::round(center_x - kTrackDragHandleLineWidth * 0.5f);
  const Pixel right = left + kTrackDragHandleLineWidth;
  const Pixel middle_top =
      std::round(center_y - kTrackDragHandleLineThickness * 0.5f);
  for (int line = -1; line <= 1; ++line) {
    const Pixel top = middle_top + line * kTrackDragHandleLineSpacing;
    draw_list->AddRectFilled(ImVec2(left, top),
                             ImVec2(right, top + kTrackDragHandleLineThickness),
                             icon_col, kTrackDragHandleLineThickness * 0.5f);
  }
}

}  // namespace traceviewer
