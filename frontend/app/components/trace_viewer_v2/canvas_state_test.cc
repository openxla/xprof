#include "xprof/frontend/app/components/trace_viewer_v2/canvas_state.h"

#include <cstdint>

#include "testing/base/public/gmock.h"
#include "<gtest/gtest.h>"
#include "third_party/dear_imgui/imgui.h"

namespace traceviewer {

class CanvasStateTest : public ::testing::Test {
 protected:
  void SetCanvasState(float dpr, int width, int height) {
    CanvasState::instance_ = CanvasState(dpr, width, height);
    CanvasState::current_version_++;
  }

  void SetDpr(float dpr) { CanvasState::instance_.device_pixel_ratio_ = dpr; }
};

namespace {

using ::testing::FloatEq;

constexpr float kDefaultDpr = 1.0f;
constexpr float kHighDpr = 2.0f;
constexpr float kTestDpr = 1.5f;
constexpr int kDefaultWidth = 800;
constexpr int kDefaultHeight = 600;
constexpr int kAltWidth = 400;
constexpr int kAltHeight = 300;
constexpr int kTestHeight = 700;

TEST_F(CanvasStateTest, EqualityOperators) {
  SetCanvasState(kDefaultDpr, kDefaultWidth, kDefaultHeight);
  const CanvasState state1 = CanvasState::Current();
  SetCanvasState(kDefaultDpr, kDefaultWidth, kDefaultHeight);
  const CanvasState state2 = CanvasState::Current();

  EXPECT_EQ(state1, state2);

  SetCanvasState(kHighDpr, kDefaultWidth, kDefaultHeight);
  const CanvasState state3 = CanvasState::Current();

  EXPECT_NE(state1, state3);

  SetCanvasState(kDefaultDpr, kDefaultWidth, kTestHeight);
  const CanvasState state4 = CanvasState::Current();

  EXPECT_NE(state3, state4);
  EXPECT_NE(state1, state4);
}

TEST_F(CanvasStateTest, BasicGetters) {
  SetCanvasState(kTestDpr, kDefaultWidth, kDefaultHeight);

  EXPECT_THAT(CanvasState::Current().device_pixel_ratio(), FloatEq(kTestDpr));
  EXPECT_EQ(CanvasState::Current().height(), kDefaultHeight);
  EXPECT_EQ(CanvasState::Current().width(), kDefaultWidth);
}

TEST_F(CanvasStateTest, PhysicalPixelsGetter) {
  SetCanvasState(kTestDpr, kDefaultWidth, kDefaultHeight);

  const ImVec2 physical_pixels = CanvasState::Current().physical_pixels();

  EXPECT_THAT(physical_pixels.x, FloatEq(kDefaultWidth * kTestDpr));
  EXPECT_THAT(physical_pixels.y, FloatEq(kDefaultHeight * kTestDpr));
}

TEST_F(CanvasStateTest, LogicalPixelsGetter) {
  SetCanvasState(kTestDpr, kDefaultWidth, kDefaultHeight);

  const ImVec2 logical_pixels = CanvasState::Current().logical_pixels();

  EXPECT_THAT(logical_pixels.x, FloatEq(kDefaultWidth));
  EXPECT_THAT(logical_pixels.y, FloatEq(kDefaultHeight));
}

TEST_F(CanvasStateTest, SetState) {
  // Set canvas state to initial value.
  CanvasState::SetState(kHighDpr, kDefaultWidth, kDefaultHeight);
  uint8_t version_before = CanvasState::version();

  EXPECT_THAT(CanvasState::Current().device_pixel_ratio(), FloatEq(kHighDpr));
  EXPECT_EQ(CanvasState::Current().height(), kDefaultHeight);
  EXPECT_EQ(CanvasState::Current().width(), kDefaultWidth);

  // SetState with same values should not change version.
  CanvasState::SetState(kHighDpr, kDefaultWidth, kDefaultHeight);
  EXPECT_EQ(CanvasState::version(), version_before);

  // SetState with different values should change version.
  CanvasState::SetState(kDefaultDpr, kAltWidth, kAltHeight);
  EXPECT_EQ(CanvasState::version(), version_before + 1);
  EXPECT_THAT(CanvasState::Current().device_pixel_ratio(),
              FloatEq(kDefaultDpr));
  EXPECT_EQ(CanvasState::Current().height(), kAltHeight);
  EXPECT_EQ(CanvasState::Current().width(), kAltWidth);
}

TEST_F(CanvasStateTest, DprAware) {
  SetCanvasState(kDefaultDpr, kDefaultWidth, kDefaultHeight);
  const DprAware<int> dpr_aware_int(10);

  EXPECT_THAT(*dpr_aware_int, FloatEq(10.0f));

  SetCanvasState(kHighDpr, kDefaultWidth, kDefaultHeight);

  EXPECT_THAT(*dpr_aware_int, FloatEq(20.0f));

  const DprAware<float> dpr_aware_float(12.5f);

  EXPECT_THAT(*dpr_aware_float, FloatEq(25.0f));

  // check caching: change dpr but not version, should return cached value.
  const uint8_t version_before = CanvasState::version();
  SetDpr(kDefaultDpr);

  EXPECT_EQ(CanvasState::version(), version_before);
  EXPECT_THAT(*dpr_aware_float, FloatEq(25.0f));
}

}  // namespace
}  // namespace traceviewer
