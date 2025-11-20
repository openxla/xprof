#include "xprof/frontend/app/components/trace_viewer_v2/color/color_generator.h"

#include "<gtest/gtest.h>"
#include "absl/strings/string_view.h"
#include "third_party/dear_imgui/imgui.h"

namespace traceviewer {
namespace {

TEST(ColorGeneratorTest, ReturnsSameColorForSameId) {
  ImU32 color1 = GetColorForId("test_id");
  ImU32 color2 = GetColorForId("test_id");

  EXPECT_EQ(color1, color2);
}

TEST(ColorGeneratorTest, ReturnsDifferentColorForDifferentId) {
  // These two strings are chosen because they hash to different values mod the
  // number of colors.
  const ImU32 color1 = GetColorForId("a");
  const ImU32 color2 = GetColorForId("aa");

  EXPECT_NE(color1, color2);
}

}  // namespace
}  // namespace traceviewer
