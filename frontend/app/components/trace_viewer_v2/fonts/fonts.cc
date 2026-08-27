#include "frontend/app/components/trace_viewer_v2/fonts/fonts.h"

#include <tuple>
#include <vector>

#include "absl/log/log.h"
#include "imgui.h"

namespace traceviewer::fonts {
namespace {
std::vector<uint8_t> roboto_font_buffer;
}

void RegisterRobotoFont(const uint8_t* data, int size) {
  roboto_font_buffer.assign(data, data + size);
}

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

  const void* font_data = nullptr;
  int font_size_bytes = 0;
  bool is_base85 = false;

  if (!roboto_font_buffer.empty()) {
    font_data = roboto_font_buffer.data();
    font_size_bytes = roboto_font_buffer.size();
    is_base85 = false;
  } else {
  }

  if (font_data == nullptr) {
    io.Fonts->AddFontDefault(&config);
    ImFont* default_font = io.Fonts->Fonts.back();
    body_large = default_font;
    caption = default_font;
    label_large = default_font;
    label_medium = default_font;
    label_small = default_font;
    title_small = default_font;
    io.FontDefault = body_large;
    return;
  }

  ImFontConfig config_large = config;
  // Typography tracking for Label Large: requires +0.1 space, but ImGui removed
  // ExtraSpacing.

  ImFontConfig config_medium = config;
  // Typography tracking for Label Medium: requires +0.5 space.

  // TODO: b/444025890 - Get the fonts and sizes from the UX design.
  auto styles = std::vector{
      std::tuple(&body_large, kBodyLargeFontSize, &config),
      std::tuple(&label_large, kLabelLargeFontSize, &config_large),
      std::tuple(&label_medium, kLabelMediumFontSize, &config_medium),
      std::tuple(&label_small, kLabelSmallFontSize, &config),
      std::tuple(&title_small, kLabelSectionHeaderFontSize, &config_medium)};

  for (const auto& [font_ptr, base_size, font_config] : styles) {
    if (is_base85) {
      *(font_ptr) = io.Fonts->AddFontFromMemoryCompressedBase85TTF(
          static_cast<const char*>(font_data), base_size, font_config,
          kRangesBasic);
    } else {
      ImFontConfig font_config_copy = *font_config;
      font_config_copy.FontDataOwnedByAtlas = false;
      *(font_ptr) = io.Fonts->AddFontFromMemoryTTF(
          const_cast<void*>(font_data), font_size_bytes, base_size,
          &font_config_copy, kRangesBasic);
    }

    if (*(font_ptr) == nullptr) {
      LOG(ERROR) << "Failed to load font size " << base_size
                 << ". Using default.";
      *(font_ptr) = io.Fonts->AddFontDefault();
    }
  }
  io.FontDefault = body_large;
}

}  // namespace traceviewer::fonts
