#include "frontend/app/components/trace_viewer_v2/color/color_generator.h"

#include "<gtest/gtest.h>"
#include "absl/strings/string_view.h"
#include "imgui.h"

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

}  // namespace
}  // namespace traceviewer
