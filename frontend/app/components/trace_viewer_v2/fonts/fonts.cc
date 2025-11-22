#include "xprof/frontend/app/components/trace_viewer_v2/fonts/fonts.h"

#include <vector>

#include "xprof/frontend/app/components/trace_viewer_v2/fonts/roboto_light.h"
#include "xprof/frontend/app/components/trace_viewer_v2/fonts/roboto_regular.h"
#include "absl/log/log.h"
#include "third_party/dear_imgui/imgui.h"
#include "third_party/dear_imgui/misc/freetype/imgui_freetype.h"

namespace traceviewer::fonts {

ImFont* body = nullptr;
ImFont* caption = nullptr;

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
  config.RasterizerDensity = pixel_ratio;
  config.RasterizerMultiply = pixel_ratio;
  // Use light hinting to make the font look less bold.
  config.FontBuilderFlags = ImGuiFreeTypeBuilderFlags_LightHinting;

  static const ImWchar kRangesBasic[] = {
      0x0020, 0x00FF,  // Basic Latin + Latin Supplement
      0x20AC, 0x20AC,  // Euro Sign
      0x2013, 0x2013,  // en dash
      0x2026, 0x2026,  // ellipsis
      0,
  };

  const char* kFontRegular = roboto_regular_compressed_data_base85;

  // TODO: b/444025890 - Get the fonts and sizes from the UX design.
  auto styles = std::vector{std::tuple(&body, 14.0f, kFontRegular)};

  for (const auto& [font_ptr, base_size, font_data] : styles) {
    // We don't multiply the base_size by pixel_ratio because the font sizes are
    // specified in dips (points). And we
    *(font_ptr) = io.Fonts->AddFontFromMemoryCompressedBase85TTF(
        font_data, base_size, &config, kRangesBasic);

    if (*(font_ptr) == nullptr) {
      LOG(ERROR) << "Failed to load font size " << base_size
                 << ". Using default.";
      *(font_ptr) = io.Fonts->AddFontDefault();
    }
  }
  io.FontDefault = body;
}

}  // namespace traceviewer::fonts
