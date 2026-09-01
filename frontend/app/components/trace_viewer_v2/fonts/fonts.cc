#include "frontend/app/components/trace_viewer_v2/fonts/fonts.h"

#include <tuple>
#include <vector>

#include "absl/log/log.h"
#include "imgui.h"
#include "frontend/app/components/trace_viewer_v2/fonts/roboto_wdthwght.h"
#include "frontend/app/components/trace_viewer_v2/fonts/trace_viewer_icons.h"

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

  const char* kFontRegular = roboto_wdthwght_compressed_data_base85;

  ImFontConfig config_large = config;
  // Typography tracking for Label Large: requires +0.1 space, but ImGui removed
  // ExtraSpacing.

  ImFontConfig config_medium = config;
  // Typography tracking for Label Medium: requires +0.5 space.

  // TODO: b/444025890 - Get the fonts and sizes from the UX design.
  auto styles = std::vector{
      std::tuple(&body_large, kBodyLargeFontSize, kFontRegular, &config),
      std::tuple(&label_large, kLabelLargeFontSize, kFontRegular,
                 &config_large),
      std::tuple(&label_medium, kLabelMediumFontSize, kFontRegular,
                 &config_medium),
      std::tuple(&label_small, kLabelSmallFontSize, kFontRegular, &config),
      std::tuple(&title_small, kLabelSectionHeaderFontSize,
                 kFontRegular, &config_medium)};

  for (const auto& [font_ptr, base_size, font_data, font_config] : styles) {
    // We don't multiply the base_size by pixel_ratio because the font sizes are
    // specified in dips (points). And we
    *(font_ptr) = io.Fonts->AddFontFromMemoryCompressedBase85TTF(
        font_data, base_size, font_config, kRangesBasic);

    if (*(font_ptr) == nullptr) {
      LOG(ERROR) << "Failed to load font size " << base_size
                 << ". Using default.";
      *(font_ptr) = io.Fonts->AddFontDefault();
      continue;
    }

    // Merge custom icon glyphs into the successfully loaded base font.
    ImFontConfig icons_config = *font_config;
    icons_config.MergeMode = true;
    icons_config.PixelSnapH = true;
    static const ImWchar icons_ranges[] = {0xe000, 0xe050, 0};
    io.Fonts->AddFontFromMemoryCompressedBase85TTF(
        trace_viewer_icons_compressed_data_base85, base_size, &icons_config,
        icons_ranges);
  }
  io.FontDefault = body_large;
}

}  // namespace traceviewer::fonts
