#include "xprof/frontend/app/components/trace_viewer_v2/timeline/time_range.h"

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

}  // namespace
}  // namespace traceviewer
