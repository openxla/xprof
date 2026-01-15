#include "xprof/frontend/app/components/trace_viewer_v2/timeline/time_range.h"

#include <cmath>

#include "<gtest/gtest.h>"
#include "xprof/frontend/app/components/trace_viewer_v2/animation.h"

namespace traceviewer {
namespace {

TEST(TimeRangeTest, DurationWithValidRange) {
  TimeRange valid_range(10.0, 20.0);

  EXPECT_EQ(valid_range.duration(), 10.0);
}

TEST(TimeRangeTest, DurationWithInvertedRange) {
  TimeRange inverted_range(20.0, 10.0);

  EXPECT_EQ(inverted_range.duration(), 0.0);
}

TEST(TimeRangeTest, DurationWithZeroRange) {
  TimeRange zero_duration_range(5.0, 5.0);

  EXPECT_EQ(zero_duration_range.duration(), 0.0);
}

TEST(TimeRangeTest, ClampsNegativeEnd) {
  TimeRange negative_end(10.0, -20.0);

  EXPECT_EQ(negative_end, TimeRange(10.0, 10.0));
}

TEST(TimeRangeTest, Center) {
  TimeRange range(10.0, 20.0);

  EXPECT_EQ(range.center(), 15.0);
}

TEST(TimeRangeTest, EncompassSmallerRange) {
  TimeRange range(10.0, 20.0);
  TimeRange other(12.0, 18.0);

  range.Encompass(other);

  EXPECT_EQ(range, TimeRange(10.0, 20.0));
}

TEST(TimeRangeTest, EncompassShouldExpandStart) {
  TimeRange range(10.0, 20.0);
  TimeRange other(5.0, 15.0);

  range.Encompass(other);

  EXPECT_EQ(range, TimeRange(5.0, 20.0));
}

TEST(TimeRangeTest, EncompassShouldExpandEnd) {
  TimeRange range(10.0, 20.0);
  TimeRange other(15.0, 25.0);

  range.Encompass(other);

  EXPECT_EQ(range, TimeRange(10.0, 25.0));
}

TEST(TimeRangeTest, ZoomOut) {
  TimeRange range(10.0, 20.0);

  range.Zoom(2.0);

  EXPECT_EQ(range, TimeRange(5.0, 25.0));
}

TEST(TimeRangeTest, ZoomIn) {
  TimeRange range(5.0, 25.0);

  range.Zoom(0.5);

  EXPECT_EQ(range, TimeRange(10.0, 20.0));
}

TEST(TimeRangeTest, ZoomClampsStartWhenNewStartIsNegativeAndStartsAtZero) {
  TimeRange range(0.0, 10.0);

  // Zooming out by a factor of 2.0.
  // current_duration = 10.0, center = 5.0, delta = 10.0
  // new_start = 5.0 - 10.0 = -5.0 (negative, so clamped)
  // new_end = 5.0 + 10.0 = 15.0
  range.Zoom(2.0);

  // Expected: start = 0.0, end = current_duration * zoom_factor = 10.0 * 2.0
  // = 20.0
  EXPECT_EQ(range, TimeRange(0.0, 20.0));
}

TEST(TimeRangeTest,
     ZoomClampsStartWhenNewStartIsNegativeAndDoesNotStartAtZero) {
  TimeRange range(2.0, 12.0);

  // Zooming out by a factor of 2.0.
  // current_duration = 10.0, center = 7.0, delta = 10.0
  // new_start = 7.0 - 10.0 = -3.0 (negative, so clamped)
  // new_end = 7.0 + 10.0 = 17.0
  range.Zoom(2.0);

  // Expected: start = 0.0, end = current_duration * zoom_factor = 10.0 * 2.0
  // = 20.0
  EXPECT_EQ(range, TimeRange(0.0, 20.0));
}

TEST(TimeRangeTest, OperatorPlusScalar) {
  TimeRange range(10.0, 20.0);

  EXPECT_EQ(range + 5.0, TimeRange(15.0, 25.0));
}

TEST(TimeRangeTest, OperatorMinusScalar) {
  TimeRange range(10.0, 20.0);

  EXPECT_EQ(range - 5.0, TimeRange(5.0, 15.0));
}

TEST(TimeRangeTest, OperatorMultiplyScalar) {
  TimeRange range(10.0, 20.0);

  EXPECT_EQ(range * 2.0, TimeRange(20.0, 40.0));
}

TEST(TimeRangeTest, OperatorPlusEqualScalar) {
  TimeRange range(10.0, 20.0);

  range += 5.0;

  EXPECT_EQ(range, TimeRange(15.0, 25.0));
}

TEST(TimeRangeTest, OperatorMinus) {
  TimeRange range1(10.0, 20.0);
  TimeRange range2(5.0, 8.0);

  EXPECT_EQ(range1 - range2, TimeRange(5.0, 12.0));
}

TEST(TimeRangeTest, Abs) {
  TimeRange range(10.0, 20.0);

  EXPECT_EQ(abs(range), 30.0);
}

TEST(TimeRangeTest, AnimatedTimeRangeBeforeUpdate) {
  Animated<TimeRange> animated_range(TimeRange(10.0, 20.0));

  animated_range = TimeRange(20.0, 30.0);

  EXPECT_EQ(*animated_range, TimeRange(10.0, 20.0));
}

TEST(TimeRangeTest, AnimatedTimeRangeAfterUpdate) {
  Animated<TimeRange> animated_range(TimeRange(10.0, 20.0));
  animated_range = TimeRange(20.0, 30.0);

  Animation::UpdateAll(0.08f);

  EXPECT_EQ(*animated_range, TimeRange(20.0, 30.0));
}

TEST(TimeRangeTest, Scale) {
  TimeRange range(10.0, 20.0);
  TimeRange scaled = range.Scale(2.0);

  EXPECT_EQ(scaled, TimeRange(5.0, 25.0));
}

TEST(TimeRangeTest, Contains) {
  TimeRange range(10.0, 20.0);

  EXPECT_TRUE(range.Contains(TimeRange(12.0, 18.0)));
  EXPECT_TRUE(range.Contains(TimeRange(10.0, 20.0)));
  EXPECT_FALSE(range.Contains(TimeRange(5.0, 15.0)));
  EXPECT_FALSE(range.Contains(TimeRange(15.0, 25.0)));
}

TEST(TimeRangeTest, Intersect) {
  TimeRange range1(10.0, 20.0);
  TimeRange range2(15.0, 25.0);

  EXPECT_EQ(range1.Intersect(range2), TimeRange(15.0, 20.0));

  TimeRange range3(5.0, 15.0);
  EXPECT_EQ(range1.Intersect(range3), TimeRange(10.0, 15.0));

  TimeRange range4(0.0, 5.0);
  TimeRange intersection = range1.Intersect(range4);
  EXPECT_EQ(intersection.start(), intersection.end());
}

TEST(TimeRangeTest, ContainsWithTolerance) {
  // Simulate double precision issues with large timestamps
  double center = 1e12;
  double duration = 100.0;

  double kPreserveRatio = 2.0;
  double kFetchRatio = 3.0;

  TimeRange visible(center - duration / 2,
                    center + duration / 2);  // [950, 1050]

  // Calculate fetch and preserve
  // TimeRange preserve = visible.Scale(kPreserveRatio); // [900, 1100]
  TimeRange fetch = visible.Scale(kFetchRatio);  // [850, 1150]

  // Assume fetched data matches fetch exactly
  TimeRange fetched = fetch;

  // Now simulate a tiny pan that causes floating point drift.
  double pan_amount = 1e-13;
  TimeRange visible_panned = visible + pan_amount;
  TimeRange preserve_panned = visible_panned.Scale(kPreserveRatio);

  EXPECT_TRUE(fetched.Contains(preserve_panned));
}

TEST(TimeRangeTest, ContainsWithTolerance_JustOutside) {
  // this: [100, 200]
  // other: [99.99999, 200.00001]
  // With tolerance 1e-4 (absolute):
  // 1e-5 is within tolerance.
  TimeRange range(100.0, 200.0);
  // Slightly wider range within tolerance
  TimeRange other(100.0 - 1e-5, 200.0 + 1e-5);

  EXPECT_TRUE(range.Contains(other));
}

TEST(TimeRangeTest, ContainsWithTolerance_WayOutside) {
  TimeRange range(100.0, 200.0);
  // Significantly wider range.
  // Tolerance is approx 0.01. 1.0 is way outside.
  TimeRange other(100.0 - 1.0, 200.0 + 1.0);
  EXPECT_FALSE(range.Contains(other));
}

TEST(TimeRangeTest, ContainsWithTolerance_Absolute) {
  // Small numbers where absolute tolerance 1e-4 dominates.
  // this: [0, 1e-5]
  // other: [-0.5e-4, 1e-5 + 0.5e-4]
  // 0.5e-4 < 1e-4 (absolute tolerance). Should be contained.
  TimeRange range(0.0, 1e-5);
  TimeRange other(-0.5e-4, 1e-5 + 0.5e-4);

  EXPECT_TRUE(range.Contains(other));

  // Outside absolute tolerance
  // other: [-2e-4, ...]
  // 2e-4 > 1e-4. Should NOT be contained.
  TimeRange other_bad(-2e-4, 1e-5);

  EXPECT_FALSE(range.Contains(other_bad));
}

TEST(TimeRangeTest, ContainsWithTolerance_NextAfter) {
  // Verify that values just outside the strict boundary by 1 ULP (Unit in the
  // Last Place) are considered contained.
  double start = 100.0;
  double end = 200.0;
  TimeRange outer(start, end);

  // inner is slightly larger than outer by 1 ULP at the boundaries.
  double inner_start = std::nextafter(start, start - 1.0);  // start - epsilon
  double inner_end = std::nextafter(end, end + 1.0);        // end + epsilon

  TimeRange inner(inner_start, inner_end);

  // Strictly speaking, inner_start < start and inner_end > end.
  EXPECT_LT(inner_start, start);
  EXPECT_GT(inner_end, end);

  // But with tolerance, it should return true.
  EXPECT_TRUE(outer.Contains(inner));
}

TEST(TimeRangeTest, ContainsWithTolerance_ZeroBoundary) {
  // Edge case at 0.0
  TimeRange outer(0.0, 100.0);

  // A tiny negative start value (within absolute tolerance 1e-4)
  double tiny_negative = -1e-10;
  TimeRange inner(tiny_negative, 100.0);

  EXPECT_TRUE(outer.Contains(inner));

  // A large negative start value (outside absolute tolerance)
  double large_negative = -1.0;
  TimeRange inner_bad(large_negative, 100.0);

  EXPECT_FALSE(outer.Contains(inner_bad));
}

}  // namespace
}  // namespace traceviewer
