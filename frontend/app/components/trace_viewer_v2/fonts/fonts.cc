#include "frontend/app/components/trace_viewer_v2/fonts/fonts.h"

#include <tuple>
#include <vector>

#include "absl/log/log.h"
#include "imgui.h"

namespace traceviewer::fonts {

ImFont* body = nullptr;
ImFont* caption = nullptr;
ImFont* label_small = nullptr;

// The font sizes correspond to the GM3 Typography Type scale tokens.
constexpr float kBodyFontSize = 16.0f;
constexpr float kLabelSmallFontSize = 11.0f;

void LoadFonts(float pixel_ratio) {
  ImGuiIO& io = ImGui::GetIO();
  io.Fonts->Clear();

  ImFontConfig config;
  // RasterizerDensity scales the font rasterization without affecting font
  // metrics. This is the correct way to handle DPI scaling for fonts without
  // changing the overall UI layout. RasterizerMultiply adjusts the
  // brightness/alpha of the rasterized glyphs. While RasterizerDensity is the
  // primary scaling factor, setting RasterizerMultiply to pixel_ratio can also
  // enhance visibility at higher resolutions by making the font appear slightly
  // bolder/brighter.

  static const ImWchar kRangesBasic[] = {
      0x0020, 0x00FF,  // Basic Latin + Latin Supplement
      0x20AC, 0x20AC,  // Euro Sign
      0x2013, 0x2013,  // en dash
      0x2026, 0x2026,  // ellipsis
      0,
  };

  io.Fonts->AddFontDefault(&config);
  io.FontDefault = body;
}

}  // namespace traceviewer::fonts
