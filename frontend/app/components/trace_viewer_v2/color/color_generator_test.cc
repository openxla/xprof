#include "frontend/app/components/trace_viewer_v2/color/color_generator.h"

#include "gtest/gtest.h"
#include "absl/strings/string_view.h"
#include "imgui.h"
#include "frontend/app/components/trace_viewer_v2/color/colors.h"

namespace traceviewer {
namespace {

TEST(ColorGeneratorTest, ReturnsSameColorForSameId) {
  static constexpr ImU32 kColors[] = {0xFF0000FF, 0xFF00FF00, 0xFFFF0000};
  ImU32 color1 = GetColorForId("test_id", kColors);
  ImU32 color2 = GetColorForId("test_id", kColors);

  EXPECT_EQ(color1, color2);
}

TEST(ColorGeneratorTest, ReturnsDifferentColorForDifferentId) {
  static constexpr ImU32 kColors[] = {0xFF0000FF, 0xFF00FF00, 0xFFFF0000,
                                      0xFF00FFFF, 0xFFFFFF00};
  // These two strings are chosen because they hash to different values mod the
  // number of colors.
  const ImU32 color1 = GetColorForId("a", kColors);
  const ImU32 color2 = GetColorForId("aa", kColors);

  EXPECT_NE(color1, color2);
}

TEST(ColorsTest, CalculateLuminance) {
  // Black
  EXPECT_NEAR(CalculateLuminance(0xFF000000), 0.0f, 0.001f);
  // White
  EXPECT_NEAR(CalculateLuminance(0xFFFFFFFF), 1.0f, 0.001f);
  // Pure Red (0xFF0000FF in ImU32)
  EXPECT_NEAR(CalculateLuminance(0xFF0000FF), 0.2126f, 0.001f);
  // Pure Green (0xFF00FF00 in ImU32)
  EXPECT_NEAR(CalculateLuminance(0xFF00FF00), 0.7152f, 0.001f);
  // Pure Blue (0xFFFF0000 in ImU32)
  EXPECT_NEAR(CalculateLuminance(0xFFFF0000), 0.0722f, 0.001f);
}

TEST(ColorsTest, CalculateContrastRatio) {
  // White vs Black
  EXPECT_NEAR(CalculateContrastRatio(0xFFFFFFFF, 0xFF000000), 21.0f, 0.1f);
  // White vs White
  EXPECT_NEAR(CalculateContrastRatio(0xFFFFFFFF, 0xFFFFFFFF), 1.0f, 0.1f);
}

TEST(ColorsTest, GetTextColorForContrast_LightMode) {
  ImU32 on_surface = 0xFF000000;          // Black
  ImU32 inverse_on_surface = 0xFFFFFFFF;  // White

  // For a light background (white), prefer dark text
  EXPECT_EQ(GetTextColorForContrast(0xFFFFFFFF, on_surface, inverse_on_surface),
            on_surface);
  // For a dark background (black), prefer light text
  EXPECT_EQ(GetTextColorForContrast(0xFF000000, on_surface, inverse_on_surface),
            inverse_on_surface);

  // Edge case: Background just above threshold (contrast >= 4.5)
  // Color #757575 has luminance ~0.176, contrast with black is ~4.52
  ImU32 just_above_threshold = 0xFF757575;
  EXPECT_EQ(GetTextColorForContrast(just_above_threshold, on_surface,
                                    inverse_on_surface),
            on_surface);

  // Edge case: Background just below threshold (contrast < 4.5)
  // Color #747474 has luminance ~0.174, contrast with black is ~4.48
  ImU32 just_below_threshold = 0xFF747474;
  EXPECT_EQ(GetTextColorForContrast(just_below_threshold, on_surface,
                                    inverse_on_surface),
            inverse_on_surface);

  // Both meet 4.5, should prefer on_surface (Black)
  // Color #767676 has luminance ~0.180, contrast with black is ~4.6, with white
  // is ~4.56
  ImU32 both_ok_bg = 0xFF767676;
  EXPECT_EQ(GetTextColorForContrast(both_ok_bg, on_surface, inverse_on_surface),
            on_surface);
}

TEST(ColorsTest, GetTextColorForContrast_DarkMode) {
  ImU32 on_surface = 0xFFFFFFFF;          // White
  ImU32 inverse_on_surface = 0xFF000000;  // Black

  // For a dark background (black), prefer on_surface (White)
  EXPECT_EQ(GetTextColorForContrast(0xFF000000, on_surface, inverse_on_surface),
            on_surface);
  // For a light background (white), prefer inverse_on_surface (Black)
  EXPECT_EQ(GetTextColorForContrast(0xFFFFFFFF, on_surface, inverse_on_surface),
            inverse_on_surface);

  // Dark mode edge case: Background just above threshold (contrast >= 4.5)
  // Color #767676 has luminance ~0.180, contrast with white is ~4.56
  ImU32 dark_just_above = 0xFF767676;
  EXPECT_EQ(
      GetTextColorForContrast(dark_just_above, on_surface, inverse_on_surface),
      on_surface);

  // Dark mode edge case: Background just below threshold (contrast < 4.5)
  // Color #797979 has luminance ~0.190, contrast with white is ~4.37
  ImU32 dark_just_below = 0xFF797979;
  EXPECT_EQ(
      GetTextColorForContrast(dark_just_below, on_surface, inverse_on_surface),
      inverse_on_surface);

  // Both meet 4.5, should prefer on_surface (White)
  ImU32 both_ok_bg = 0xFF767676;
  EXPECT_EQ(GetTextColorForContrast(both_ok_bg, on_surface, inverse_on_surface),
            on_surface);
}

}  // namespace
}  // namespace traceviewer
