#include "frontend/app/components/trace_viewer_v2/fonts/fonts.h"

#include <tuple>
#include <vector>

#include "absl/log/log.h"
#include "imgui.h"

namespace traceviewer::fonts {

ImFont* body_large = nullptr;
ImFont* caption = nullptr;
ImFont* label_large = nullptr;
ImFont* label_medium = nullptr;
ImFont* label_small = nullptr;
ImFont* title_small = nullptr;

// The font sizes correspond to the GM3 Typography Type scale tokens.
constexpr float kBodyLargeFontSize = 16.0f;
constexpr float kLabelLargeFontSize = 14.0f;
constexpr float kLabelMediumFontSize = 12.0f;
constexpr float kLabelSmallFontSize = 11.0f;
constexpr float kLabelSectionHeaderFontSize = 13.0f;

void LoadFonts(float pixel_ratio) {
  ImGuiIO& io = ImGui::GetIO();
  io.Fonts->Clear();

  ImFontConfig config;
  // RasterizerMultiply adjusts the brightness/alpha of the rasterized glyphs.
  // A fixed value of 1.0f preserves default font appearance.
  config.RasterizerMultiply = 1.0f;

  static const ImWchar kRangesBasic[] = {
      0x0020, 0x00FF,  // Basic Latin + Latin Supplement
      0x20AC, 0x20AC,  // Euro Sign
      0x2013, 0x2013,  // en dash
      0x2026, 0x2026,  // ellipsis
      0,
  };

  io.Fonts->AddFontDefault(&config);
  io.FontDefault = body_large;
}

}  // namespace traceviewer::fonts
