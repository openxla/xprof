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
  // RasterizerDensity scales the font rasterization without affecting font
  // metrics. This is the correct way to handle DPI scaling for fonts without
  // changing the overall UI layout.
  config.RasterizerDensity = pixel_ratio;
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
