#include "xprof/frontend/app/components/trace_viewer_v2/animation.h"

#include <cmath>

#include "testing/base/public/gmock.h"
#include "<gtest/gtest.h>"

namespace traceviewer {
namespace {

using ::testing::FloatEq;
using ::testing::Ne;

class AnimationTest : public ::testing::Test {
 protected:
  // Helper to check if an animation is currently active.
  // An animation is active if its current value is not yet at the target value.
  bool IsAnimating(Animated<float>& anim) {
    // Check if the current value is different from the target value within a
    // tolerance.
    return std::abs(*anim - anim.target()) > 1e-9f;
  }

  void TearDown() override {
    // No need to clear, animations unregister themselves on destruction.
  }
};

TEST_F(AnimationTest, InitializationWithOneArgument) {
  // When constructed with one argument, the value is set, and the target
  // defaults to T(), which is 0.0f for floats.
  Animated<float> anim(1.0f);

  EXPECT_THAT(*anim, FloatEq(1.0f));
  EXPECT_THAT(anim.target(), FloatEq(0.0f));  // Target defaults to 0.0f
  EXPECT_TRUE(IsAnimating(anim));  // It is animating because 1.0 != 0.0
}

TEST_F(AnimationTest, InitializationWithTwoArguments) {
  // When constructed with two arguments, value and target are set accordingly.
  Animated<float> anim(0.0f, 1.0f);

  EXPECT_THAT(*anim, FloatEq(0.0f));
  EXPECT_THAT(anim.target(), FloatEq(1.0f));
  EXPECT_TRUE(IsAnimating(anim));
}

TEST_F(AnimationTest, StartAnimationWithAssignment) {
  Animated<float> anim(0.0f);

  anim = 1.0f;

  EXPECT_THAT(anim.target(), FloatEq(1.0f));
  EXPECT_TRUE(IsAnimating(anim));
}

TEST_F(AnimationTest, AnimationCompletesWithLargeDelta) {
  Animated<float> anim(0.0f);
  anim = 1.0f;

  // A large delta time should complete the animation in one step.
  Animation::UpdateAll(1.0f);

  EXPECT_THAT(*anim, FloatEq(1.0f));
  EXPECT_FALSE(IsAnimating(anim));
}

TEST_F(AnimationTest, StartAnimationWithOperator) {
  Animated<float> anim(0.0f);
  bool finished = false;

  anim(1.0f, [&](float val) { finished = true; });

  EXPECT_THAT(anim.target(), FloatEq(1.0f));
  EXPECT_TRUE(IsAnimating(anim));
  EXPECT_FALSE(finished);
}

TEST_F(AnimationTest, AnimationCompletesWithOperator) {
  Animated<float> anim(0.0f);
  bool finished = false;
  anim(1.0f, [&](float val) { finished = true; });

  Animation::UpdateAll(1.0f);

  EXPECT_THAT(*anim, FloatEq(1.0f));
  EXPECT_FALSE(IsAnimating(anim));
  EXPECT_TRUE(finished);
}

TEST_F(AnimationTest, SnapTo) {
  Animated<float> anim(0.0f, 1.0f);

  anim.snap_to(0.5f);

  EXPECT_THAT(*anim, FloatEq(0.5f));
  EXPECT_THAT(anim.target(), FloatEq(0.5f));
  EXPECT_FALSE(IsAnimating(anim));
}

TEST_F(AnimationTest, AnimationStopsWhenTargetReached) {
  Animated<float> anim(0.0f);
  anim = 1.0f;

  // Animate halfway. The speed is 12.5. So 1/12.5 = 0.08s to complete.
  // lets update for a small amount of time.
  Animation::UpdateAll(0.04f);

  EXPECT_GT(*anim, 0.0f);
  EXPECT_LT(*anim, 1.0f);
  EXPECT_TRUE(IsAnimating(anim));

  // Animate the rest of the way
  Animation::UpdateAll(1.0f);

  EXPECT_THAT(*anim, FloatEq(1.0f));
  EXPECT_FALSE(IsAnimating(anim));
}

TEST_F(AnimationTest, NoDoubleRegistration) {
  Animated<float> anim(0.0f);

  // Start an animation
  anim = 1.0f;
  Animation::UpdateAll(1e-6f);
  float val1 = *anim;

  // Setting the same target again should not reset the animation
  anim = 1.0f;
  Animation::UpdateAll(1e-6f);
  float val2 = *anim;

  EXPECT_NE(val1, val2);

  // Complete the animation
  Animation::UpdateAll(1.0f);

  // Start a new one
  anim = 0.0f;
  Animation::UpdateAll(1e-6f);

  EXPECT_THAT(*anim, Ne(1.0f));
  EXPECT_TRUE(IsAnimating(anim));
}

}  // namespace
}  // namespace traceviewer
