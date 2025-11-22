#include "xprof/frontend/app/components/trace_viewer_v2/canvas_state.h"

#include <cstdint>

#include "testing/base/public/gmock.h"
#include "<gtest/gtest.h>"
#include "third_party/dear_imgui/imgui.h"

namespace traceviewer {

class CanvasStateTest : public ::testing::Test {
 protected:
  void SetCanvasState(float dpr, int height, int width) {
    CanvasState::instance_ = CanvasState(dpr, height, width);
    CanvasState::current_version_++;
  }

  void SetDpr(float dpr) { CanvasState::instance_.device_pixel_ratio_ = dpr; }
};

namespace {

using ::testing::FloatEq;

TEST_F(CanvasStateTest, EqualityOperators) {
  SetCanvasState(1.0f, 600, 800);
  const CanvasState state1 = CanvasState::Current();
  SetCanvasState(1.0f, 600, 800);
  const CanvasState state2 = CanvasState::Current();

  EXPECT_EQ(state1, state2);

  SetCanvasState(2.0f, 600, 800);
  const CanvasState state3 = CanvasState::Current();

  EXPECT_NE(state1, state3);

  SetCanvasState(1.0f, 700, 800);
  const CanvasState state4 = CanvasState::Current();

  EXPECT_NE(state3, state4);
  EXPECT_NE(state1, state4);
}

TEST_F(CanvasStateTest, BasicGetters) {
  SetCanvasState(1.5f, 600, 800);

  EXPECT_THAT(CanvasState::Current().device_pixel_ratio(), FloatEq(1.5f));
  EXPECT_EQ(CanvasState::Current().height(), 600);
  EXPECT_EQ(CanvasState::Current().width(), 800);
}

TEST_F(CanvasStateTest, PhysicalPixelsGetter) {
  SetCanvasState(1.5f, 600, 800);

  const ImVec2 physical_pixels = CanvasState::Current().physical_pixels();

  EXPECT_THAT(physical_pixels.x, FloatEq(800 * 1.5f));
  EXPECT_THAT(physical_pixels.y, FloatEq(600 * 1.5f));
}

TEST_F(CanvasStateTest, LogicalPixelsGetter) {
  SetCanvasState(1.5f, 600, 800);

  const ImVec2 logical_pixels = CanvasState::Current().logical_pixels();

  EXPECT_THAT(logical_pixels.x, FloatEq(800));
  EXPECT_THAT(logical_pixels.y, FloatEq(600));
}

TEST_F(CanvasStateTest, Update) {
  // Set canvas state to something different from default construction in test.
  // Default construction in test will result in dpr=1.0, height=0, width=0.
  SetCanvasState(2.0f, 600, 800);
  uint8_t version_before = CanvasState::version();

  // Update should detect a change because CanvasState() will be {1.0, 0, 0}.
  EXPECT_TRUE(CanvasState::Update());
  EXPECT_EQ(CanvasState::version(), version_before + 1);
  EXPECT_THAT(CanvasState::Current().device_pixel_ratio(), FloatEq(1.0f));
  EXPECT_EQ(CanvasState::Current().height(), 0);
  EXPECT_EQ(CanvasState::Current().width(), 0);

  // Calling update again should not detect a change.
  version_before = CanvasState::version();
  EXPECT_FALSE(CanvasState::Update());
  EXPECT_EQ(CanvasState::version(), version_before);
}

TEST_F(CanvasStateTest, DprAware) {
  SetCanvasState(1.0f, 600, 800);
  const DprAware<int> dpr_aware_int(10);

  EXPECT_THAT(*dpr_aware_int, FloatEq(10.0f));

  SetCanvasState(2.0f, 600, 800);

  EXPECT_THAT(*dpr_aware_int, FloatEq(20.0f));

  const DprAware<float> dpr_aware_float(12.5f);

  EXPECT_THAT(*dpr_aware_float, FloatEq(25.0f));

  // check caching: change dpr but not version, should return cached value.
  const uint8_t version_before = CanvasState::version();
  SetDpr(1.0f);

  EXPECT_EQ(CanvasState::version(), version_before);
  EXPECT_THAT(*dpr_aware_float, FloatEq(25.0f));
}

}  // namespace
}  // namespace traceviewer
