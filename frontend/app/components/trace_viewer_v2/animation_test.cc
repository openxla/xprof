#include "frontend/app/components/trace_viewer_v2/animation.h"

#include <cmath>
#include <limits>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

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

TEST_F(AnimationTest, ZeroDeltaTimeDoesNotSnapOrStall) {
  Animated<float> anim(0.0f, 1.0f);

  Animation::UpdateAll(0.0f);

  EXPECT_THAT(*anim, FloatEq(0.0f));
  EXPECT_TRUE(Animation::HasActiveAnimations());
}

TEST_F(AnimationTest, NegativeDeltaTimeDoesNotOvershootOrStall) {
  Animated<float> anim(0.0f, 1.0f);

  Animation::UpdateAll(-0.016f);

  EXPECT_THAT(*anim, FloatEq(0.0f));
  EXPECT_TRUE(Animation::HasActiveAnimations());
}

TEST_F(AnimationTest, SnapsWhenFloatingPointPrecisionStalls) {
  // At 1e7f, float ULP is 1.0f. A step of 2.0f * (12.5f * 1e-4f) = 2.5e-3f
  // rounds back to current_ (1e7f), triggering the stall guard.
  const float start = 1e7f;
  const float target = start + 2.0f;
  bool finished = false;
  Animated<float> anim(start, target, [&](float) { finished = true; });

  Animation::UpdateAll(1e-4f);

  EXPECT_THAT(*anim, FloatEq(target));
  EXPECT_FALSE(Animation::HasActiveAnimations());
  EXPECT_TRUE(finished);
}

TEST_F(AnimationTest,
       IgnoresNonPositiveAndNanDeltaTimeEvenWhenWithinTolerance) {
  // Start within kAnimationAbsoluteTolerance (1e-4) of target so a missing
  // delta_time guard would erroneously converge and fire on_finished.
  bool finished = false;
  Animated<float> anim(0.0f, 1e-5f, [&](float) { finished = true; });

  Animation::UpdateAll(0.0f);
  EXPECT_THAT(*anim, FloatEq(0.0f));
  EXPECT_TRUE(Animation::HasActiveAnimations());
  EXPECT_FALSE(finished);

  Animation::UpdateAll(-0.016f);
  EXPECT_THAT(*anim, FloatEq(0.0f));
  EXPECT_TRUE(Animation::HasActiveAnimations());
  EXPECT_FALSE(finished);

  Animation::UpdateAll(std::numeric_limits<float>::quiet_NaN());
  EXPECT_FALSE(std::isnan(*anim));
  EXPECT_THAT(*anim, FloatEq(0.0f));
  EXPECT_TRUE(Animation::HasActiveAnimations());
  EXPECT_FALSE(finished);
}

TEST_F(AnimationTest, CompletesAtExactTargetWhenFactorIsAtLeastOne) {
  bool finished = false;
  Animated<float> anim(0.0f, 100.0f, [&](float) { finished = true; });

  // delta_time = 0.1f produces factor = 1.25f > 1.0f; must clamp to exact
  // target without overshooting.
  Animation::UpdateAll(0.1f);

  EXPECT_THAT(*anim, FloatEq(100.0f));
  EXPECT_FALSE(Animation::HasActiveAnimations());
  EXPECT_TRUE(finished);
}

TEST_F(AnimationTest,
       AnimationConvergenceToleranceUsesRelativeAndAbsoluteFloor) {
  EXPECT_DOUBLE_EQ(AnimationConvergenceTolerance(100.0f),
                   100.0 * kAnimationRelativeTolerance);
  EXPECT_DOUBLE_EQ(AnimationConvergenceTolerance(0.0f),
                   kAnimationAbsoluteTolerance);
}

}  // namespace
}  // namespace traceviewer
