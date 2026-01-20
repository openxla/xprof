#include "xprof/frontend/app/components/trace_viewer_v2/timeline/timeline.h"

#include <any>
#include <string>
#include <utility>

#include "testing/base/public/gmock.h"
#include "<gtest/gtest.h>"
#include "absl/strings/match.h"
#include "absl/strings/string_view.h"
#include "third_party/dear_imgui/imgui.h"
#include "third_party/dear_imgui/imgui_internal.h"
#include "tsl/profiler/lib/context_types.h"
#include "xprof/frontend/app/components/trace_viewer_v2/animation.h"
#include "xprof/frontend/app/components/trace_viewer_v2/event_data.h"
#include "xprof/frontend/app/components/trace_viewer_v2/helper/time_formatter.h"
#include "xprof/frontend/app/components/trace_viewer_v2/timeline/constants.h"
#include "xprof/frontend/app/components/trace_viewer_v2/timeline/time_range.h"

namespace traceviewer {
namespace testing {
namespace {

using ::testing::_;
using ::testing::DoubleEq;
using ::testing::ElementsAre;
using ::testing::FloatEq;
using ::testing::Return;
using ::testing::Test;

// Mock class for Timeline to mock virtual methods.
class MockTimeline : public Timeline {
 public:
  MOCK_METHOD(ImVec2, GetTextSize, (absl::string_view text), (const, override));
  MOCK_METHOD(void, Pan, (Pixel pixel_amount), (override));
  MOCK_METHOD(void, Zoom, (float zoom_factor, double pivot), (override));
  MOCK_METHOD(void, Scroll, (Pixel pixel_amount), (override));
};

TEST(TimelineTest, SetTimelineData) {
  Timeline timeline;
  FlameChartTimelineData data;
  data.groups.push_back({.name = "Group 1", .start_level = 0});
  data.entry_levels.push_back(0);
  data.entry_start_times.push_back(10.0);
  data.entry_total_times.push_back(5.0);
  data.entry_pids.push_back(1);
  data.entry_args.push_back({});

  timeline.set_timeline_data(std::move(data));

  EXPECT_THAT(timeline.timeline_data().entry_levels, ElementsAre(0));
  EXPECT_THAT(timeline.timeline_data().entry_start_times, ElementsAre(10.0));
  EXPECT_THAT(timeline.timeline_data().entry_total_times, ElementsAre(5.0));
}

TEST(TimelineTest, SetVisibleRange) {
  Timeline timeline;
  TimeRange range(10.0, 50.0);

  timeline.SetVisibleRange(range);

  EXPECT_EQ(timeline.visible_range().start(), 10.0);
  EXPECT_EQ(timeline.visible_range().end(), 50.0);
}

TEST(TimelineTest, PixelToTime) {
  Timeline timeline;
  timeline.SetVisibleRange({0.0, 100.0});
  double px_per_unit = 10.0;  // 10 pixels per time unit

  EXPECT_EQ(timeline.PixelToTime(0.0f, px_per_unit), 0.0);
  EXPECT_EQ(timeline.PixelToTime(100.0f, px_per_unit), 10.0);
  EXPECT_EQ(timeline.PixelToTime(500.0f, px_per_unit), 50.0);
  EXPECT_EQ(timeline.PixelToTime(1000.0f, px_per_unit), 100.0);

  // Test with zero px_per_unit
  EXPECT_EQ(timeline.PixelToTime(100.0f, 0.0), 0.0);

  // Test with negative px_per_unit
  EXPECT_EQ(timeline.PixelToTime(100.0f, -10.0), 0.0);
}

TEST(TimelineTest, TimeToPixel) {
  Timeline timeline;
  timeline.SetVisibleRange({0.0, 100.0});
  double px_per_unit = 10.0;  // 10 pixels per time unit

  EXPECT_EQ(timeline.TimeToPixel(0.0, px_per_unit), 0.0f);
  EXPECT_EQ(timeline.TimeToPixel(10.0, px_per_unit), 100.0f);
  EXPECT_EQ(timeline.TimeToPixel(50.0, px_per_unit), 500.0f);
  EXPECT_EQ(timeline.TimeToPixel(100.0, px_per_unit), 1000.0f);

  // Test with zero px_per_unit
  EXPECT_EQ(timeline.TimeToPixel(50.0, 0.0), 0.0f);

  // Test with negative px_per_unit
  EXPECT_EQ(timeline.TimeToPixel(50.0, -10.0), 0.0f);
}

TEST(TimelineTest, TimeToScreenX) {
  Timeline timeline;
  timeline.SetVisibleRange({0.0, 100.0});
  double px_per_unit = 10.0;  // 10 pixels per time unit
  Pixel screen_x_offset = 50.0f;

  EXPECT_EQ(timeline.TimeToScreenX(0.0, screen_x_offset, px_per_unit), 50.0f);
  EXPECT_EQ(timeline.TimeToScreenX(10.0, screen_x_offset, px_per_unit), 150.0f);
  EXPECT_EQ(timeline.TimeToScreenX(50.0, screen_x_offset, px_per_unit), 550.0f);
  EXPECT_EQ(timeline.TimeToScreenX(100.0, screen_x_offset, px_per_unit),
            1050.0f);

  // Test with zero px_per_unit
  EXPECT_EQ(timeline.TimeToScreenX(50.0, screen_x_offset, 0.0), 50.0f);

  // Test with negative px_per_unit
  EXPECT_EQ(timeline.TimeToScreenX(50.0, screen_x_offset, -10.0), 50.0f);
}

// Constants for CalculateEventRect tests
constexpr double kPxPerTimeUnit = 1.0;
constexpr Pixel kScreenXOffset = 0.0f;
constexpr Pixel kScreenYOffset = 0.0f;
constexpr int kLevelInGroup = 0;
constexpr Pixel kTimelineWidth = 100.0f;

TEST(TimelineTest, CalculateEventRect_EventFullyWithinView) {
  Timeline timeline;
  timeline.SetVisibleRange({100.0, 200.0});

  // Event time range: [110.0, 120.0].
  // Screen range before adjustments: [10.0, 20.0].
  EventRect rect = timeline.CalculateEventRect(
      /*start=*/110.0, /*end=*/120.0, kScreenXOffset, kScreenYOffset,
      kPxPerTimeUnit, kLevelInGroup, kTimelineWidth);

  EXPECT_FLOAT_EQ(rect.left, 10.0f);
  EXPECT_FLOAT_EQ(rect.right, 20.0f - kEventPaddingRight);
  EXPECT_FLOAT_EQ(rect.top, 0.0f);
  EXPECT_FLOAT_EQ(rect.bottom, kEventHeight);
}

TEST(TimelineTest, CalculateEventRect_EventPartiallyClippedLeft) {
  Timeline timeline;
  timeline.SetVisibleRange({100.0, 200.0});

  // Event time range: [90.0, 110.0].
  // Screen range after left clipping: [0.0, 10.0].
  EventRect rect = timeline.CalculateEventRect(
      /*start=*/90.0, /*end=*/110.0, kScreenXOffset, kScreenYOffset,
      kPxPerTimeUnit, kLevelInGroup, kTimelineWidth);

  EXPECT_FLOAT_EQ(rect.left, 0.0f);
  EXPECT_FLOAT_EQ(rect.right, 10.0f - kEventPaddingRight);
}

TEST(TimelineTest, CalculateEventRect_EventPartiallyClippedRight) {
  Timeline timeline;
  timeline.SetVisibleRange({100.0, 200.0});

  // Event time range: [190.0, 210.0].
  // Screen range after right clipping: [90.0, 100.0].
  EventRect rect = timeline.CalculateEventRect(
      /*start=*/190.0, /*end=*/210.0, kScreenXOffset, kScreenYOffset,
      kPxPerTimeUnit, kLevelInGroup, kTimelineWidth);

  EXPECT_FLOAT_EQ(rect.left, 90.0f);
  EXPECT_FLOAT_EQ(rect.right, 100.0f);
}

TEST(TimelineTest, CalculateEventRect_EventCompletelyOutsideLeft) {
  Timeline timeline;
  timeline.SetVisibleRange({100.0, 200.0});

  // Event time range: [80.0, 90.0].
  // Screen range will be less than time start.
  // Expected to be fully clipped to the left edge [0.0, 0.0] (padding right
  // won't effect here because the event is clipped to the left edge).
  EventRect rect = timeline.CalculateEventRect(
      /*start=*/80.0, /*end=*/90.0, kScreenXOffset, kScreenYOffset,
      kPxPerTimeUnit, kLevelInGroup, kTimelineWidth);

  EXPECT_FLOAT_EQ(rect.left, 0.0f);
  EXPECT_FLOAT_EQ(rect.right, 0.0f);
}

TEST(TimelineTest, CalculateEventRect_EventCompletelyOutsideRight) {
  Timeline timeline;
  timeline.SetVisibleRange({100.0, 200.0});

  // Event time range: [210.0, 220.0].
  // Screen range will be larger than time end.
  // Expected to be fully clipped to the right edge [100.0, 100.0] (padding
  // right won't effect here because the event is clipped to the right edge).
  EventRect rect = timeline.CalculateEventRect(
      /*start=*/210.0, /*end=*/220.0, kScreenXOffset, kScreenYOffset,
      kPxPerTimeUnit, kLevelInGroup, kTimelineWidth);

  EXPECT_FLOAT_EQ(rect.left, 100.0f);
  EXPECT_FLOAT_EQ(rect.right, 100.0f);
}

TEST(TimelineTest, CalculateEventRect_EventSmallerThanMinimumWidth) {
  Timeline timeline;
  timeline.SetVisibleRange({100.0, 200.0});

  // Event time range: [110.0, 110.1].
  // Screen width is expanded to kEventMinimumDrawWidth.
  EventRect rect = timeline.CalculateEventRect(
      /*start=*/110.0, /*end=*/110.1, kScreenXOffset, kScreenYOffset,
      kPxPerTimeUnit, kLevelInGroup, kTimelineWidth);

  EXPECT_FLOAT_EQ(rect.left, 10.0f);
  EXPECT_FLOAT_EQ(rect.right,
                  10.0f + kEventMinimumDrawWidth - kEventPaddingRight);
}

TEST(TimelineTest, CalculateEventRect_ZeroPxPerTimeUnit) {
  Timeline timeline;
  timeline.SetVisibleRange({100.0, 100.0});  // Zero duration

  // With px_per_time_unit = 0, the event width is 0, so it's expanded to
  // kEventMinimumDrawWidth.
  EventRect rect = timeline.CalculateEventRect(
      /*start=*/110.0, /*end=*/120.0, kScreenXOffset, kScreenYOffset,
      /*px_per_time_unit=*/0.0, kLevelInGroup, kTimelineWidth);

  // left becomes screen_x_offset (0), right becomes max(0, 0 +
  // kEventMinimumDrawWidth)
  EXPECT_FLOAT_EQ(rect.left, 0.0f);
  EXPECT_FLOAT_EQ(rect.right, kEventMinimumDrawWidth - kEventPaddingRight);
}

TEST(TimelineTest, CalculateEventTextRect_TextFits) {
  MockTimeline timeline;
  std::string event_name = "Test";
  EventRect event_rect = {10.0f, 0.0f, 100.0f, kEventHeight};
  ImVec2 fake_text_size = {40.0f, kEventHeight};

  EXPECT_CALL(timeline, GetTextSize(event_name))
      .WillOnce(Return(fake_text_size));

  ImVec2 text_pos = timeline.CalculateEventTextRect(event_name, event_rect);

  float event_width = event_rect.right - event_rect.left;
  float expected_left =
      event_rect.left + (event_width - fake_text_size.x) * 0.5f;
  float expected_top = (kEventHeight - fake_text_size.y) * 0.5f;

  EXPECT_FLOAT_EQ(text_pos.x, expected_left);
  EXPECT_FLOAT_EQ(text_pos.y, expected_top);
}

TEST(TimelineTest, CalculateEventTextRect_TextWiderThanRect) {
  MockTimeline timeline;
  std::string event_name = "ThisIsAVeryLongEventName";
  EventRect event_rect = {10.0f, 0.0f, 50.0f, kEventHeight};
  ImVec2 fake_text_size = {100.0f, kEventHeight};

  EXPECT_CALL(timeline, GetTextSize(event_name))
      .WillOnce(Return(fake_text_size));

  ImVec2 text_pos = timeline.CalculateEventTextRect(event_name, event_rect);

  float expected_top = (kEventHeight - fake_text_size.y) * 0.5f;

  EXPECT_FLOAT_EQ(text_pos.x, event_rect.left);
  EXPECT_FLOAT_EQ(text_pos.y, expected_top);
}

TEST(TimelineTest, CalculateEventTextRect_EmptyText) {
  MockTimeline timeline;
  std::string event_name = "";
  EventRect event_rect = {10.0f, 0.0f, 100.0f, kEventHeight};
  ImVec2 fake_text_size = {0.0f, kEventHeight};

  EXPECT_CALL(timeline, GetTextSize(event_name))
      .WillOnce(Return(fake_text_size));

  ImVec2 text_pos = timeline.CalculateEventTextRect(event_name, event_rect);

  float event_width = event_rect.right - event_rect.left;
  float expected_left = event_rect.left + event_width * 0.5f;
  float expected_top = (kEventHeight - fake_text_size.y) * 0.5f;

  EXPECT_FLOAT_EQ(text_pos.x, expected_left);
  EXPECT_FLOAT_EQ(text_pos.y, expected_top);
}

TEST(TimelineTest, CalculateEventTextRect_SpecialCharacters) {
  MockTimeline timeline;
  std::string event_name = "Test!@#$%^&*()_+";
  EventRect event_rect = {10.0f, 0.0f, 150.0f, kEventHeight};
  ImVec2 fake_text_size = {120.0f, kEventHeight};

  EXPECT_CALL(timeline, GetTextSize(event_name))
      .WillOnce(Return(fake_text_size));

  ImVec2 text_pos = timeline.CalculateEventTextRect(event_name, event_rect);

  float event_width = event_rect.right - event_rect.left;
  float expected_left =
      event_rect.left + (event_width - fake_text_size.x) * 0.5f;
  float expected_top = (kEventHeight - fake_text_size.y) * 0.5f;

  EXPECT_FLOAT_EQ(text_pos.x, expected_left);
  EXPECT_FLOAT_EQ(text_pos.y, expected_top);
}

TEST(TimelineTest, CalculateEventTextRect_NarrowEvent) {
  MockTimeline timeline;
  std::string event_name = "Name";
  EventRect event_rect = {0.0f, 0.0f, 0.0f, kEventHeight};  // 0px wide
  ImVec2 fake_text_size = {50.0f, kEventHeight};

  EXPECT_CALL(timeline, GetTextSize(event_name))
      .WillOnce(Return(fake_text_size));

  ImVec2 text_pos = timeline.CalculateEventTextRect(event_name, event_rect);

  float expected_top = (kEventHeight - fake_text_size.y) * 0.5f;

  // Text is wider than the event, so it should start at the event's left edge.
  EXPECT_FLOAT_EQ(text_pos.x, event_rect.left);
  EXPECT_FLOAT_EQ(text_pos.y, expected_top);
}

TEST(TimelineTest, GetTextForDisplayWhenTextFits) {
  MockTimeline timeline;
  const std::string text = "Test";

  EXPECT_CALL(timeline, GetTextSize(absl::string_view(text)))
      .WillOnce(Return(ImVec2{50.0f, kEventHeight}));

  // Text width 50.0 is smaller than 1000.0, so no truncation.
  EXPECT_EQ(timeline.GetTextForDisplay(text, 1000.0f), text);
}

constexpr float kCharWidth = 10.0f;
constexpr float kEllipsisWidth = 30.0f;

TEST(TimelineTest, GetTextForDisplayWhenTextTruncated) {
  MockTimeline timeline;
  const std::string text = "Long Event Name";
  // If char width is kCharWidth, ellipsis "..." width is kEllipsisWidth.
  // Available width allows "L..." (kCharWidth + kEllipsisWidth) but not "Lo..."
  // (2 * kCharWidth + kEllipsisWidth).
  const float available_width = kCharWidth + kEllipsisWidth + 5.0f;

  EXPECT_CALL(timeline, GetTextSize(_)).WillRepeatedly([](absl::string_view s) {
    if (s == "...") return ImVec2{kEllipsisWidth, kEventHeight};
    return ImVec2{s.length() * kCharWidth, kEventHeight};
  });

  EXPECT_EQ(timeline.GetTextForDisplay(text, available_width), "L...");
}

TEST(TimelineTest, GetTextForDisplayWhenTextTruncatedToEmpty) {
  MockTimeline timeline;
  const std::string text = "Long Event Name";
  // If char width is kCharWidth, ellipsis "..." width is kEllipsisWidth.
  // Available width doesn't even allow "L..." (kCharWidth + kEllipsisWidth).
  const float available_width = kCharWidth + kEllipsisWidth - 5.0f;

  EXPECT_CALL(timeline, GetTextSize(_)).WillRepeatedly([](absl::string_view s) {
    if (s == "...") return ImVec2{kEllipsisWidth, kEventHeight};
    return ImVec2{s.length() * kCharWidth, kEventHeight};
  });

  EXPECT_EQ(timeline.GetTextForDisplay(text, available_width), "");
}

TEST(TimelineTest, GetTextForDisplayWhenWidthTooSmallForEllipsis) {
  MockTimeline timeline;
  const std::string text = "Long Event Name";
  // If char width is kCharWidth, ellipsis "..." width is kEllipsisWidth.
  // Available width is smaller than ellipsis width.
  const float available_width = kEllipsisWidth - 5.0f;

  EXPECT_CALL(timeline, GetTextSize(_)).WillRepeatedly([](absl::string_view s) {
    if (s == "...") return ImVec2{kEllipsisWidth, kEventHeight};
    return ImVec2{s.length() * kCharWidth, kEventHeight};
  });

  const std::string result = timeline.GetTextForDisplay(text, available_width);

  EXPECT_EQ(result, "");
}

TEST(TimelineTest, ConstrainTimeRange_NoChange) {
  // Data Range: [===========================]
  // Range:        {----------------------}
  // Constrained:  (----------------------)
  Timeline timeline;
  timeline.set_data_time_range({0.0, 100.0});
  TimeRange range(10.0, 90.0);

  timeline.ConstrainTimeRange(range);

  EXPECT_DOUBLE_EQ(range.start(), 10.0);
  EXPECT_DOUBLE_EQ(range.end(), 90.0);
}

TEST(TimelineTest, ConstrainTimeRange_StartBeforeDataRange) {
  // Data Range:      [==========================]
  // Range:      {---------}
  // Constrained:     (---------)
  Timeline timeline;
  timeline.set_data_time_range({10.0, 100.0});
  TimeRange range(0.0, 50.0);

  timeline.ConstrainTimeRange(range);

  EXPECT_DOUBLE_EQ(range.start(), 10.0);
  EXPECT_DOUBLE_EQ(range.end(), 60.0);
}

TEST(TimelineTest, ConstrainTimeRange_StartBeforeDataRangeEndCapped) {
  // Data Range:      [========================]
  // Range:      {----------------------------}
  // Constrained:     (========================)
  Timeline timeline;
  timeline.set_data_time_range({10.0, 100.0});
  TimeRange range(0.0, 99.0);

  timeline.ConstrainTimeRange(range);

  EXPECT_DOUBLE_EQ(range.start(), 10.0);
  EXPECT_DOUBLE_EQ(range.end(), 100.0);
}

TEST(TimelineTest, ConstrainTimeRange_EndAfterDataRange) {
  // Data Range: [=====================]
  // Range:                  {--------------}
  // Constrained:       (--------------)
  Timeline timeline;
  timeline.set_data_time_range({10.0, 100.0});
  TimeRange range(60.0, 110.0);

  timeline.ConstrainTimeRange(range);

  EXPECT_DOUBLE_EQ(range.start(), 50.0);
  EXPECT_DOUBLE_EQ(range.end(), 100.0);
}

TEST(TimelineTest, ConstrainTimeRange_EndAfterDataRangeStartCapped) {
  // Data Range:  [====================]
  // Range:         {---------------------------}
  // Constrained: (====================)
  Timeline timeline;
  timeline.set_data_time_range({10.0, 100.0});
  TimeRange range(11.0, 110.0);

  timeline.ConstrainTimeRange(range);

  EXPECT_DOUBLE_EQ(range.start(), 10.0);
  EXPECT_DOUBLE_EQ(range.end(), 100.0);
}

TEST(TimelineTest, ConstrainTimeRange_RangeCoversDataRange) {
  // Data Range:      [================]
  // Range: {------------------------------}
  // Constrained:     (================)
  Timeline timeline;
  timeline.set_data_time_range({10.0, 100.0});
  TimeRange range(0.0, 120.0);

  timeline.ConstrainTimeRange(range);

  EXPECT_DOUBLE_EQ(range.start(), 10.0);
  EXPECT_DOUBLE_EQ(range.end(), 100.0);
}

TEST(TimelineTest, ConstrainTimeRange_EnforceMinDuration) {
  Timeline timeline;
  timeline.set_data_time_range({0.0, 100.0});
  TimeRange range(50.0, 50.0);

  timeline.ConstrainTimeRange(range);

  EXPECT_NEAR(range.duration(), kMinDurationMicros, 1e-14);
  EXPECT_DOUBLE_EQ(range.center(), 50.0);
  EXPECT_DOUBLE_EQ(range.start(), 50.0 - kMinDurationMicros / 2.0);
  EXPECT_DOUBLE_EQ(range.end(), 50.0 + kMinDurationMicros / 2.0);
}

TEST(TimelineTest, ConstrainTimeRange_MinDurationExpansionClampedAtStart) {
  Timeline timeline;
  timeline.set_data_time_range({10.0, 100.0});
  // This range has duration kMinDurationMicros / 2, centered at 10.0.
  TimeRange range(10.0 - kMinDurationMicros / 4, 10.0 + kMinDurationMicros / 4);

  timeline.ConstrainTimeRange(range);

  // It should be expanded to kMinDurationMicros centered around 10.0,
  // becoming {10.0 - kMinDur/2, 10.0 + kMinDur/2}.
  // The start 10.0 - kMinDur/2 is less than fetched_data_time_range_.start(),
  // so it should be clamped to {10.0, 10.0 + kMinDurationMicros}.
  EXPECT_DOUBLE_EQ(range.start(), 10.0);
  EXPECT_DOUBLE_EQ(range.end(), 10.0 + kMinDurationMicros);
  EXPECT_NEAR(range.duration(), kMinDurationMicros, 1e-14);
}

TEST(TimelineTest, ConstrainTimeRange_MinDurationExpansionClampedAtEnd) {
  Timeline timeline;
  timeline.set_data_time_range({10.0, 100.0});
  // This range has duration kMinDurationMicros / 2, centered at 100.0.
  TimeRange range(100.0 - kMinDurationMicros / 4,
                  100.0 + kMinDurationMicros / 4);

  timeline.ConstrainTimeRange(range);

  // It should be expanded to kMinDurationMicros centered around 100.0,
  // becoming {100.0 - kMinDur/2, 100.0 + kMinDur/2}.
  // The end 100.0 + kMinDur/2 is greater than fetched_data_time_range_.end(),
  // so it should be clamped to {100.0 - kMinDurationMicros, 100.0}.
  EXPECT_DOUBLE_EQ(range.start(), 100.0 - kMinDurationMicros);
  EXPECT_DOUBLE_EQ(range.end(), 100.0);
  EXPECT_NEAR(range.duration(), kMinDurationMicros, 1e-14);
}

TEST(TimelineTest, NavigateToEvent) {
  Timeline timeline;
  FlameChartTimelineData data;
  data.groups.push_back(
      {.name = "Group 1", .start_level = 0, .nesting_level = 0});
  data.events_by_level.push_back({0, 1});
  data.entry_names.push_back("event0");
  data.entry_names.push_back("event1");
  data.entry_levels.push_back(0);
  data.entry_levels.push_back(0);
  data.entry_start_times.push_back(10.0);
  data.entry_start_times.push_back(100.0);
  data.entry_total_times.push_back(10.0);
  data.entry_total_times.push_back(20.0);
  data.entry_pids.push_back(1);
  data.entry_pids.push_back(2);
  data.entry_args.push_back({});
  data.entry_args.push_back({});
  timeline.set_timeline_data(std::move(data));
  timeline.set_data_time_range({0.0, 200.0});
  timeline.SetVisibleRange({0.0, 50.0});

  timeline.NavigateToEvent(1);
  // A large delta time should complete the animation in one step.
  Animation::UpdateAll(1.0f);

  // event 1 is 100-120, center is 110.
  // duration before navigation is 50 and should not change.
  EXPECT_DOUBLE_EQ(timeline.visible_range().center(), 110.0);
  EXPECT_DOUBLE_EQ(timeline.visible_range().duration(), 50.0);
  EXPECT_DOUBLE_EQ(timeline.visible_range().start(), 110.0 - 25.0);
  EXPECT_DOUBLE_EQ(timeline.visible_range().end(), 110.0 + 25.0);
}

TEST(TimelineTest, NavigateToEventWithNegativeIndex) {
  Timeline timeline;
  FlameChartTimelineData data;
  data.entry_start_times.push_back(10.0);
  data.entry_pids.push_back(1);
  data.entry_args.push_back({});
  timeline.set_timeline_data(std::move(data));
  TimeRange initial_range(0.0, 50.0);
  timeline.SetVisibleRange(initial_range);

  timeline.NavigateToEvent(-1);

  // Visible range should not change because event index is invalid.
  EXPECT_EQ(timeline.visible_range().start(), initial_range.start());
  EXPECT_EQ(timeline.visible_range().end(), initial_range.end());
}

TEST(TimelineTest, NavigateToEventWithIndexOutOfBounds) {
  Timeline timeline;
  FlameChartTimelineData data;
  data.entry_start_times.push_back(10.0);
  data.entry_pids.push_back(1);
  data.entry_args.push_back({});
  timeline.set_timeline_data(std::move(data));
  TimeRange initial_range(0.0, 50.0);
  timeline.SetVisibleRange(initial_range);

  timeline.NavigateToEvent(1);

  // Visible range should not change because event index is out of bounds.
  EXPECT_EQ(timeline.visible_range().start(), initial_range.start());
  EXPECT_EQ(timeline.visible_range().end(), initial_range.end());
}

TEST(TimelineTest, MaybeRequestDataRefetchesWhenZoomedInDespiteRangeCoverage) {
  Timeline timeline;
  // Step 1: Initialize with full range fetch to set last_fetch_request_range_.
  // Data: [0, 10s].
  timeline.set_data_time_range({0.0, 10000000.0});

  // Simulate that we previously asked for the full range and received it.
  // This sets last_fetch_request_range_ to [0, 10s].

  // Step 2: Simulate "We have all the data now".
  timeline.set_fetched_data_time_range({0.0, 10000000.0});
  timeline.set_is_incremental_loading(false);

  // Step 3: Zoom IN heavily.
  // Visible: [5s, 5.0002s] (200us).
  // Fetch (Scale 3): 600us -> Expanded to kMinFetchDurationMicros (1000us).
  // Fetched (10s) / Fetch (1000us) = 10000 > kRefetchZoomRatio (8).
  // last_fetch ([0, 10s]) CONTAINS preserve.

  bool request_triggered = false;
  EventData received_data;
  timeline.set_event_callback(
      [&](absl::string_view type, const EventData& detail) {
        if (type == kFetchData) {
          request_triggered = true;
          received_data = detail;
        }
      });

  // Visible: 200us centered at 5.0001s.
  // Fetch: 1000us centered at 5.0001s.
  // Start: 5.0001 - 0.5 = 4.9996s.
  // End: 5.0001 + 0.5 = 5.0006s.
  timeline.SetVisibleRange({5000000.0, 5000200.0});

  ASSERT_TRUE(request_triggered);
  EXPECT_DOUBLE_EQ(
      std::any_cast<double>(received_data.at(std::string(kFetchDataStart))),
      MicrosToMillis(4999600.0));
  EXPECT_DOUBLE_EQ(
      std::any_cast<double>(received_data.at(std::string(kFetchDataEnd))),
      MicrosToMillis(5000600.0));
}

TEST(TimelineTest, MaybeRequestDataTriggeredWhenPanningOutsidePreserveRange) {
  Timeline timeline;
  // Visible range: [100, 200]. Duration: 100. Center: 150.
  // Preserve range (Scale 2.0): [50, 250].
  // Data range: [60, 300].
  // Preserve range start (50) < Data range start (60), so it should trigger
  // fetch.
  timeline.set_data_time_range({0.0, 300.0});
  timeline.set_fetched_data_time_range({60.0, 300.0});
  timeline.set_is_incremental_loading(false);

  bool request_triggered = false;
  EventData received_data;
  timeline.set_event_callback(
      [&](absl::string_view type, const EventData& detail) {
        if (type == kFetchData) {
          request_triggered = true;
          received_data = detail;
        }
      });

  // Simulate panning outside preserve range.
  timeline.SetVisibleRange({100.0, 200.0});

  ASSERT_TRUE(request_triggered);
  // Fetch range (Scale 3.0 of visible): [0, 300].
  EXPECT_DOUBLE_EQ(
      std::any_cast<double>(received_data.at(std::string(kFetchDataStart))),
      MicrosToMillis(0.0));
  EXPECT_DOUBLE_EQ(
      std::any_cast<double>(received_data.at(std::string(kFetchDataEnd))),
      MicrosToMillis(300.0));
}

TEST(TimelineTest, MaybeRequestDataNotTriggeredWhenPreserveExceedsDataRange) {
  Timeline timeline;
  // Visible range: [100, 200]. Duration: 100. Center: 150.
  // Preserve range (Scale 2.0): [50, 250].
  // Data range: [60, 300].
  // Preserve range start (50) < Data range start (60).
  // Before fix: Triggered fetch for [50, ...].
  // After fix: ConstrainTimeRange clamps preserve to [60, ...], which is
  // contained in fetched.
  const TimeRange data_range = {60.0, 300.0};
  timeline.set_data_time_range(data_range);
  timeline.set_fetched_data_time_range(data_range);

  bool request_triggered = false;
  timeline.set_event_callback(
      [&](absl::string_view type, const EventData& detail) {
        if (type == kFetchData) {
          request_triggered = true;
        }
      });

  // Simulate panning.
  timeline.SetVisibleRange({100.0, 200.0});

  EXPECT_FALSE(request_triggered);
}

TEST(TimelineTest,
     MaybeRequestDataNotTriggeredWhenPreserveExceedsDataRangeStart) {
  Timeline timeline;
  // Visible range: [100, 200]. Duration: 100. Center: 150.
  // Preserve range (Scale 2.0): [50, 250].
  // Data range: [60, 300].
  // Preserve range start (50) < Data range start (60).
  const TimeRange data_range = {60.0, 300.0};
  timeline.set_data_time_range(data_range);
  timeline.set_fetched_data_time_range(data_range);
  timeline.set_is_incremental_loading(false);

  bool request_triggered = false;
  timeline.set_event_callback(
      [&](absl::string_view type, const EventData& detail) {
        if (type == kFetchData) {
          request_triggered = true;
        }
      });

  // Simulate panning.
  timeline.SetVisibleRange({100.0, 200.0});

  EXPECT_FALSE(request_triggered);
}

TEST(TimelineTest,
     MaybeRequestDataNotTriggeredWhenPreserveExceedsDataRangeEnd) {
  Timeline timeline;
  // Visible range: [800, 900]. Duration: 100. Center: 850.
  // Preserve range (Scale 2.0): [750, 950].
  // Data range: [700, 940].
  // Preserve range end (950) > Data range end (940).
  const TimeRange data_range = {700.0, 940.0};
  timeline.set_data_time_range(data_range);
  timeline.set_fetched_data_time_range(data_range);
  timeline.set_is_incremental_loading(false);

  bool request_triggered = false;
  timeline.set_event_callback(
      [&](absl::string_view type, const EventData& detail) {
        if (type == kFetchData) {
          request_triggered = true;
        }
      });

  // Simulate panning.
  timeline.SetVisibleRange({800.0, 900.0});

  EXPECT_FALSE(request_triggered);
}

TEST(TimelineTest, MaybeRequestDataNotTriggeredWhenInsidePreserveRange) {
  Timeline timeline;
  // Visible range: [100, 200]. Duration: 100. Center: 150.
  // Preserve range (Scale 2.0): [50, 250].
  // Data range: [0, 300].
  // Preserve range is fully contained in Data range.
  timeline.set_data_time_range({0.0, 300.0});
  timeline.set_fetched_data_time_range({0.0, 300.0});
  timeline.set_is_incremental_loading(false);

  bool request_triggered = false;
  timeline.set_event_callback(
      [&](absl::string_view type, const EventData& detail) {
        if (type == kFetchData) {
          request_triggered = true;
        }
      });

  // Simulate panning inside preserve range.
  timeline.SetVisibleRange({100.0, 200.0});

  EXPECT_FALSE(request_triggered);
}

TEST(TimelineTest, MaybeRequestDataNotTriggeredWhenLoading) {
  Timeline timeline;
  timeline.set_data_time_range({60.0, 300.0});
  timeline.set_fetched_data_time_range({60.0, 300.0});
  // Simulate loading state
  timeline.set_is_incremental_loading(true);

  bool request_triggered = false;
  timeline.set_event_callback(
      [&](absl::string_view type, const EventData& detail) {
        if (type == kFetchData) {
          request_triggered = true;
        }
      });

  timeline.SetVisibleRange({100.0, 200.0});

  EXPECT_FALSE(request_triggered);
}

TEST(TimelineTest, MaybeRequestDataSetsIsLoadingToTrue) {
  Timeline timeline;
  timeline.set_data_time_range({0.0, 300.0});
  timeline.set_fetched_data_time_range({60.0, 300.0});
  timeline.set_is_incremental_loading(false);

  int request_count = 0;
  timeline.set_event_callback(
      [&](absl::string_view type, const EventData& detail) {
        if (type == kFetchData) {
          request_count++;
        }
      });

  timeline.SetVisibleRange({100.0, 200.0});
  EXPECT_EQ(request_count, 1);

  // Should not trigger again because is_incremental_loading_ should be set to
  // true inside `MaybeRequestData`.
  timeline.SetVisibleRange({100.0, 200.0});
  EXPECT_EQ(request_count, 1);
}

TEST(TimelineTest, MaybeRequestDataRefetchWhenZoomedIn) {
  Timeline timeline;
  // Scenario: Focus on zoom-in logic WITHOUT triggering MinFetchDuration
  // expansion.
  // We need Fetch duration > kMinFetchDurationMicros (1ms).
  // Let Visible = 400ms.
  // Fetch (Scale 3.0) = 1.2s > 1s. (No expansion).
  // Fetched Data must be large enough to trigger ratio > 8.
  // Fetched > 8 * 1.2s = 9.6s. Let's use 10s.

  timeline.set_data_time_range({0.0, 20000000.0});
  timeline.set_fetched_data_time_range({0.0, 10000000.0});
  timeline.set_is_incremental_loading(false);

  bool request_triggered = false;
  EventData received_data;
  timeline.set_event_callback(
      [&](absl::string_view type, const EventData& detail) {
        if (type == kFetchData) {
          request_triggered = true;
          received_data = detail;
        }
      });

  // Visible Range: [5s, 5.4s] (400ms).
  // Center: 5.2s.
  // Fetch center 5.2s. Duration 1.2s.
  // Start: 5.2 - 0.6 = 4.6s.
  // End: 5.2 + 0.6 = 5.8s.
  timeline.SetVisibleRange({5000000.0, 5400000.0});

  ASSERT_TRUE(request_triggered);
  EXPECT_DOUBLE_EQ(
      std::any_cast<double>(received_data.at(std::string(kFetchDataStart))),
      MicrosToMillis(4600000.0));
  EXPECT_DOUBLE_EQ(
      std::any_cast<double>(received_data.at(std::string(kFetchDataEnd))),
      MicrosToMillis(5800000.0));
}

TEST(TimelineTest, MaybeRequestDataExpandsToMinDuration) {
  Timeline timeline;
  // Scenario: Visible range is very small (100us).
  // Fetch calculated (Scale 3.0) would be 300us.
  // Must expand to kMinFetchDurationMicros (1000us = 1ms).
  // Data range large enough to not constrain.

  timeline.set_data_time_range({0.0, 20000000.0});
  timeline.set_fetched_data_time_range({0.0, 2000000.0});
  timeline.set_is_incremental_loading(false);

  bool request_triggered = false;
  EventData received_data;
  timeline.set_event_callback(
      [&](absl::string_view type, const EventData& detail) {
        if (type == kFetchData) {
          request_triggered = true;
          received_data = detail;
        }
      });

  // Visible: [5s, 5.0001s] (100us). Center 5.00005s.
  // Expanded Fetch: 1000us.
  // Start: 5.00005s - 500us = 4.99955s.
  // End: 5.00005s + 500us = 5.00055s.
  timeline.SetVisibleRange({5000000.0, 5000100.0});

  ASSERT_TRUE(request_triggered);
  EXPECT_DOUBLE_EQ(
      std::any_cast<double>(received_data.at(std::string(kFetchDataStart))),
      MicrosToMillis(4999550.0));
  EXPECT_DOUBLE_EQ(
      std::any_cast<double>(received_data.at(std::string(kFetchDataEnd))),
      MicrosToMillis(5000550.0));
}

TEST(TimelineTest, MaybeRequestDataFetchesConstrainedRange) {
  Timeline timeline;
  // Data range: [0, 10s] = 10,000,000.
  // Fetched range: [0, 2s] = 2,000,000.
  // Visible range: [9s, 10s] = [9,000,000, 10,000,000]. Duration 1s.
  // Fetch (Scale 3.0): Center 9.5s, Duration 3s. Range [8s, 11s].
  // MinDuration 100ms << 3s. No expansion.
  // Constrain [8s, 11s] to data range [0s, 10s].
  // Since 11s > 10s, shift left by 11s - 10s = 1s.
  // Shifted range: [8s - 1s, 11s - 1s] = [7s, 10s].
  // Result: Start 7s, End 10s.

  timeline.set_data_time_range({0.0, 10000000.0});
  timeline.set_fetched_data_time_range({0.0, 2000000.0});
  timeline.set_is_incremental_loading(false);

  bool request_triggered = false;
  EventData received_data;
  timeline.set_event_callback(
      [&](absl::string_view type, const EventData& detail) {
        if (type == kFetchData) {
          request_triggered = true;
          received_data = detail;
        }
      });

  // Move visible range to the end of the data range.
  timeline.SetVisibleRange({9000000.0, 10000000.0});

  ASSERT_TRUE(request_triggered);
  EXPECT_DOUBLE_EQ(
      std::any_cast<double>(received_data.at(std::string(kFetchDataStart))),
      MicrosToMillis(7000000.0));
  EXPECT_DOUBLE_EQ(
      std::any_cast<double>(received_data.at(std::string(kFetchDataEnd))),
      MicrosToMillis(10000000.0));
}

TEST(TimelineTest, MaybeRequestDataSkipsFetchIfRangeAlreadyRequested) {
  Timeline timeline;
  // Scenario: We requested [0, 3000].
  // We currently have [0, 2500] (maybe partial load).
  // We pan to a place that needs [2600].
  // Preserve is in [0, 3000], but NOT in [0, 2500].
  // Expect: NO refetch (because we already asked for it).

  timeline.set_data_time_range({0.0, 10000.0});

  // Simulate that we previously asked for [0, 3000].
  // We use set_fetched_data_time_range to initialize last_fetch_request_range_
  // to [0, 3000] (since it's initially empty).
  timeline.set_fetched_data_time_range({0.0, 3000.0});
  timeline.set_is_incremental_loading(false);

  // Update fetched data to simulate partial arrival.
  // last_fetch_request_range_ remains [0, 3000].
  timeline.set_fetched_data_time_range({500.0, 2500.0});

  bool request_triggered = false;
  timeline.set_event_callback(
      [&](absl::string_view type, const EventData& detail) {
        if (type == kFetchData) {
          request_triggered = true;
        }
      });

  // Step 3: Pan to right edge of what we have.
  // Visible=[2400, 2500]. Duration 100.
  // Preserve (2x) = [2350, 2550].
  // Fetched ([0, 2500]) DOES NOT contain Preserve (end 2550 > 2500).
  // normally this WOULD trigger refetch.
  // But last_fetch ([0, 3000]) DOES contain Preserve.
  timeline.SetVisibleRange({2400.0, 2500.0});

  EXPECT_FALSE(request_triggered);
}

TEST(TimelineTest, MaybeRequestDataSuppressesRedundantFetchWithExpandedRange) {
  Timeline timeline;
  // Scenario: Visible 100us -> Expanded Fetch 1000us (1ms).
  // last_fetch and fetched cover this 1000us.
  // Zoom ratio is low (fetched is small).
  // Result: NO Refetch.

  timeline.set_data_time_range({0.0, 20000000.0});
  timeline.set_is_incremental_loading(false);

  // Step 1: Trigger initial fetch to set last_fetch_request_range_.
  // We want fetched size < 8 * 1000us = 8000us.
  // Use Visible 400us. Fetch 1200us.
  timeline.set_fetched_data_time_range({-1.0, -1.0});  // Trigger

  bool request_triggered = false;
  timeline.set_event_callback(
      [&](absl::string_view type, const EventData& detail) {
        if (type == kFetchData) {
          request_triggered = true;
        }
      });

  // Visible [5.0s, 5.0004s].
  timeline.SetVisibleRange({5000000.0, 5000400.0});
  ASSERT_TRUE(request_triggered);
  // last_fetch approx [4.9996s, 5.0008s] (Duration 1200us > 1000us Min).

  // Step 2: Simulate data arrival.
  // We can just set fetched to something covering the next preserve.
  // Next preserve will be for Visible 100us -> 200us wide.
  // [5.0s, 5.0001s]. Preserve [4.99995s, 5.00015s].
  // last_fetch covers this.
  // fetched must cover preserve.
  // 4995us to 5005us.
  const double center = 5000000.0 + 200.0;
  timeline.set_fetched_data_time_range({center - 2000.0, center + 2000.0});
  timeline.set_is_incremental_loading(false);
  request_triggered = false;

  // Step 3: Zoom to 100us.
  // Visible [5.0s, 5.0001s].
  // Fetch -> Expanded 1000us.
  // Ratio: Fetched (4000us) / Fetch (1000us) = 4.0 < 8.
  timeline.SetVisibleRange({5000000.0, 5000100.0});

  EXPECT_FALSE(request_triggered);
}

// Test fixture for tests that require an ImGui context.
template <typename TimelineT>
class TimelineImGuiTestFixture : public Test {
 protected:
  void SetUp() override {
    ImGui::CreateContext();

    ImGuiIO& io = ImGui::GetIO();
    // Set dummy display size and delta time, required for ImGui to function.
    io.DisplaySize = ImVec2(1920, 1080);
    io.DeltaTime = 0.1f;
    // The font atlas must be built before ImGui::NewFrame() is called.
    io.Fonts->Build();
    timeline_.set_timeline_data(
        {{},
         {},
         {},
         {},
         {},
         {},
         {},
         {{.name = "group", .start_level = 0, .nesting_level = 0}},
        {},
        {}});
  }

  void TearDown() override { ImGui::DestroyContext(); }

  void SimulateFrame() {
    ImGui::NewFrame();
    // Draw() calls HandleKeyboard() internally, which may update animation
    // targets (e.g., via Pan/Zoom).
    timeline_.Draw();
    // Update all animations by delta time. This must be called *after* Draw()
    // to ensure animations progress towards targets set in HandleKeyboard().
    Animation::UpdateAll(ImGui::GetIO().DeltaTime);
    ImGui::EndFrame();
  }

  void SimulateKeyHeldForDuration(ImGuiKey key, float duration) {
    ImGuiIO& io = ImGui::GetIO();
    io.DeltaTime = duration;
    io.AddKeyEvent(key, true);
    // Set DownDuration to 0.0f. ImGui::NewFrame() in SimulateFrame() increments
    // DownDuration by io.DeltaTime before Draw()/HandleKeyboard() is called.
    // Setting it to 0 ensures HandleKeyboard sees io.DeltaTime as DownDuration.
    ImGui::GetKeyData(key)->DownDuration = 0.0f;
  }

  TimelineT timeline_;
};

using MockTimelineImGuiFixture = TimelineImGuiTestFixture<MockTimeline>;

TEST_F(MockTimelineImGuiFixture, PanLeftWithAKey) {
  ImGui::GetIO().AddKeyEvent(ImGuiKey_A, true);

  // No acceleration is applied because io.DeltaTime (0.1f) <
  // kAccelerateThreshold (0.25f).
  EXPECT_CALL(timeline_,
              Pan(FloatEq(-kPanningSpeed * ImGui::GetIO().DeltaTime)));

  SimulateFrame();
}

TEST_F(MockTimelineImGuiFixture, PanRightWithDKey) {
  ImGui::GetIO().AddKeyEvent(ImGuiKey_D, true);

  // No acceleration is applied because io.DeltaTime (0.1f) <
  // kAccelerateThreshold (0.25f).
  EXPECT_CALL(timeline_,
              Pan(FloatEq(kPanningSpeed * ImGui::GetIO().DeltaTime)));

  SimulateFrame();
}

TEST_F(MockTimelineImGuiFixture, ZoomInWithWKey) {
  ImGui::GetIO().AddKeyEvent(ImGuiKey_W, true);

  // No acceleration is applied because io.DeltaTime (0.1f) <
  // kAccelerateThreshold (0.25f).
  EXPECT_CALL(timeline_,
              Zoom(FloatEq(1.0f - kZoomSpeed * ImGui::GetIO().DeltaTime), _));

  SimulateFrame();
}

TEST_F(MockTimelineImGuiFixture, ZoomOutWithSKey) {
  ImGui::GetIO().AddKeyEvent(ImGuiKey_S, true);

  // No acceleration is applied because io.DeltaTime (0.1f) <
  // kAccelerateThreshold (0.25f).
  EXPECT_CALL(timeline_,
              Zoom(FloatEq(1.0f + kZoomSpeed * ImGui::GetIO().DeltaTime), _));

  SimulateFrame();
}

TEST_F(MockTimelineImGuiFixture, ZoomInWithWKey_MouseInsideTimeline) {
  // Set a visible range that results in a round number for px_per_time_unit
  // to make test calculations predictable. With a timeline width of 1669px
  // (based on 1920px window width, 250px label width, and 1px padding),
  // a duration of 166.9 gives 10px per microsecond.
  timeline_.SetVisibleRange({0.0, 166.9});

  ImGuiIO& io = ImGui::GetIO();
  // Timeline starts at label_width (250).
  // Set mouse at x=350, y=50.
  // Relative x = 350 - 250 = 100.
  // Scale = 10 px/us.
  // Pivot = 100 / 10 = 10.0 us.
  io.MousePos = ImVec2(350.0f, 50.0f);
  io.AddKeyEvent(ImGuiKey_W, true);

  // No acceleration is applied.
  EXPECT_CALL(timeline_,
              Zoom(FloatEq(1.0f - kZoomSpeed * io.DeltaTime), DoubleEq(10.0)));

  SimulateFrame();
}

TEST_F(MockTimelineImGuiFixture, ZoomInWithWKey_MouseOutsideTimeline) {
  timeline_.SetVisibleRange({0.0, 166.9});

  ImGuiIO& io = ImGui::GetIO();
  // Mouse outside timeline (x < 250).
  io.MousePos = ImVec2(100.0f, 50.0f);
  io.AddKeyEvent(ImGuiKey_W, true);

  // Pivot should be center of visible range [0, 166.9] -> 83.45.
  EXPECT_CALL(timeline_,
              Zoom(FloatEq(1.0f - kZoomSpeed * io.DeltaTime), DoubleEq(83.45)));

  SimulateFrame();
}

TEST_F(MockTimelineImGuiFixture, ScrollUpWithUpArrowKey) {
  ImGui::GetIO().AddKeyEvent(ImGuiKey_UpArrow, true);

  EXPECT_CALL(timeline_,
              Scroll(FloatEq(-kScrollSpeed * ImGui::GetIO().DeltaTime)));

  SimulateFrame();
}

TEST_F(MockTimelineImGuiFixture, ScrollDownWithDownArrowKey) {
  ImGui::GetIO().AddKeyEvent(ImGuiKey_DownArrow, true);

  EXPECT_CALL(timeline_,
              Scroll(FloatEq(kScrollSpeed * ImGui::GetIO().DeltaTime)));

  SimulateFrame();
}

TEST_F(MockTimelineImGuiFixture, PanLeftWithAKey_Accelerated) {
  SimulateKeyHeldForDuration(ImGuiKey_A, 1.0f);

  // DownDuration becomes 1.0f > kAccelerateThreshold (0.1f).
  // accelerated_time = 1.0f - 0.1f = 0.9f
  // multiplier = 0.9f * kAccelerateRate (10.0f) = 9.0f
  // total multiplier = 1.0f + std::min(9.0f, kMaxAccelerateFactor) = 10.0f
  EXPECT_CALL(timeline_,
              Pan(FloatEq(-kPanningSpeed * ImGui::GetIO().DeltaTime * 10.0f)));

  SimulateFrame();
}

TEST_F(MockTimelineImGuiFixture, ZoomInWithWKey_Accelerated) {
  SimulateKeyHeldForDuration(ImGuiKey_W, 0.5f);

  // DownDuration becomes 0.5f > kAccelerateThreshold (0.1f).
  // accelerated_time = 0.5f - 0.1f = 0.4f
  // multiplier = 0.4f * kAccelerateRate (10.0f) = 4.0f
  // total multiplier = 1.0f + std::min(4.0f, kMaxAccelerateFactor) = 5.0f
  EXPECT_CALL(
      timeline_,
      Zoom(FloatEq(1.0f - kZoomSpeed * ImGui::GetIO().DeltaTime * 5.0f), _));

  SimulateFrame();
}

TEST_F(MockTimelineImGuiFixture, PanRightWithDKey_MaxAccelerated) {
  SimulateKeyHeldForDuration(ImGuiKey_D, 6.0f);

  // DownDuration becomes 6.0f > kAccelerateThreshold (0.25f).
  // accelerated_time = 6.0f - 0.25f = 5.75f
  // multiplier = 5.75f * kAccelerateRate (10.0f) = 57.5f
  // total multiplier = 1.0f + std::min(57.5f, kMaxAccelerateFactor (30.0f))
  //                  = 1.0f + 30.0f = 31.0f
  EXPECT_CALL(timeline_,
              Pan(FloatEq(kPanningSpeed * ImGui::GetIO().DeltaTime * 31.0f)));

  SimulateFrame();
}

TEST_F(MockTimelineImGuiFixture, ScrollDownWithMouseWheel) {
  ImGui::GetIO().AddMouseWheelEvent(0.0f, 1.0f);

  EXPECT_CALL(timeline_, Scroll(FloatEq(1.0f)));

  SimulateFrame();
}

TEST_F(MockTimelineImGuiFixture, ScrollUpWithMouseWheel) {
  ImGui::GetIO().AddMouseWheelEvent(0.0f, -1.0f);

  EXPECT_CALL(timeline_, Scroll(FloatEq(-1.0f)));

  SimulateFrame();
}

TEST_F(MockTimelineImGuiFixture, ZoomOutWithMouseWheelAndCtrlKey) {
  // Set a visible range for predictable pivot calculation.
  timeline_.SetVisibleRange({0.0, 166.9});

  ImGuiIO& io = ImGui::GetIO();
  io.AddMouseWheelEvent(0.0f, 1.0f);
  io.AddKeyEvent(ImGuiMod_Ctrl, true);
  // Mouse inside timeline area. x=350 -> Relative x = 100.
  // px_per_unit = 10.0. Pivot = 10.0.
  io.MousePos = ImVec2(350.0f, 50.0f);

  const float expected_zoom_factor = 1.0f + 1.0f * kMouseWheelZoomSpeed;

  EXPECT_CALL(timeline_, Zoom(FloatEq(expected_zoom_factor), DoubleEq(10.0)));

  SimulateFrame();
}

TEST_F(MockTimelineImGuiFixture, ZoomInWithMouseWheelAndCtrlKey) {
  // Set a visible range for predictable pivot calculation.
  timeline_.SetVisibleRange({0.0, 166.9});

  ImGuiIO& io = ImGui::GetIO();
  io.AddMouseWheelEvent(0.0f, -1.0f);
  io.AddKeyEvent(ImGuiMod_Ctrl, true);
  // Mouse inside timeline area. x=350 -> Relative x = 100.
  // px_per_unit = 10.0. Pivot = 10.0.
  io.MousePos = ImVec2(350.0f, 50.0f);

  const float expected_zoom_factor = 1.0f + (-1.0f) * kMouseWheelZoomSpeed;

  EXPECT_CALL(timeline_, Zoom(FloatEq(expected_zoom_factor), DoubleEq(10.0)));

  SimulateFrame();
}

TEST_F(MockTimelineImGuiFixture, PanRightWithHorizontalMouseWheel) {
  ImGui::GetIO().AddMouseWheelEvent(1.0f, 0.0f);

  EXPECT_CALL(timeline_, Pan(FloatEq(1.0f)));

  SimulateFrame();
}

TEST_F(MockTimelineImGuiFixture, PanLeftWithHorizontalMouseWheel) {
  ImGui::GetIO().AddMouseWheelEvent(-1.0f, 0.0f);

  EXPECT_CALL(timeline_, Pan(FloatEq(-1.0f)));

  SimulateFrame();
}

TEST_F(MockTimelineImGuiFixture, PanRightWithShiftAndMouseWheel) {
  ImGuiIO& io = ImGui::GetIO();
  io.AddMouseWheelEvent(0.0f, 1.0f);
  io.AddKeyEvent(ImGuiMod_Shift, true);

  EXPECT_CALL(timeline_, Pan(FloatEq(1.0f)));

  SimulateFrame();
}

TEST_F(MockTimelineImGuiFixture, PanLeftWithShiftAndMouseWheel) {
  ImGuiIO& io = ImGui::GetIO();
  io.AddMouseWheelEvent(0.0f, -1.0f);
  io.AddKeyEvent(ImGuiMod_Shift, true);

  EXPECT_CALL(timeline_, Pan(FloatEq(-1.0f)));

  SimulateFrame();
}

TEST_F(MockTimelineImGuiFixture, PanWithMouseDrag) {
  ImGuiIO& io = ImGui::GetIO();
  // Main window pos is (0,0), content_min is (0,0), label_width is 250.
  // So timeline area starts at x=250.
  io.MousePos = ImVec2(300.0f, 50.0f);
  SimulateFrame();  // Establish initial state

  // Press mouse button without shift.
  io.AddMouseButtonEvent(0, true);

  // In the first frame of a drag, MouseDelta will be zero.
  EXPECT_CALL(timeline_, Pan(0.0f));
  EXPECT_CALL(timeline_, Scroll(0.0f));

  SimulateFrame();  // This will call HandleMouse and set is_dragging_ to true

  // Drag the mouse.
  io.AddMousePosEvent(310.0f, 60.0f);

  EXPECT_CALL(timeline_, Pan(FloatEq(-10.0f)));
  EXPECT_CALL(timeline_, Scroll(FloatEq(-10.0f)));

  SimulateFrame();

  // Release mouse button.
  io.AddMouseButtonEvent(0, false);
  SimulateFrame();
}

TEST_F(MockTimelineImGuiFixture,
       ShiftClickAndReleaseShiftMidDragContinuesSelection) {
  // Setup similar to TimelineDragSelectionTest to ensure predictable
  // coordinates.
  timeline_.SetVisibleRange({0.0, 166.9});
  timeline_.set_data_time_range({0.0, 166.9});
  ImGuiIO& io = ImGui::GetIO();

  // Start with Shift held down.
  io.AddKeyEvent(ImGuiMod_Shift, true);

  // Start drag in timeline area.
  // X=300 is safely inside the timeline (250 + padding < 300).
  io.MousePos = ImVec2(300.0f, 50.0f);
  io.AddMouseButtonEvent(0, true);
  SimulateFrame();

  // Release Shift key.
  io.AddKeyEvent(ImGuiMod_Shift, false);

  // Drag mouse to X=500.
  io.MousePos = ImVec2(500.0f, 50.0f);
  SimulateFrame();

  // End drag.
  io.AddMouseButtonEvent(0, false);
  SimulateFrame();

  // Verify that a selection was created despite Shift being released.
  ASSERT_EQ(timeline_.selected_time_ranges().size(), 1);
  const TimeRange& range = timeline_.selected_time_ranges()[0];
  // Calculate expected range based on pixel movement.
  // 10px/us assumption from TimelineDragSelectionTest.
  // 300 -> 5.0. 500 -> 25.0.
  EXPECT_NEAR(range.start(), 5.0, 1e-5);
  EXPECT_NEAR(range.end(), 25.0, 1e-5);
}

TEST_F(MockTimelineImGuiFixture, ClickAndPressShiftMidDragContinuesPanning) {
  // Setup similar to TimelineDragSelectionTest.
  timeline_.SetVisibleRange({0.0, 166.9});
  timeline_.set_fetched_data_time_range({0.0, 166.9});
  timeline_.SetVisibleRange({0.0, 166.9});
  timeline_.set_data_time_range({0.0, 166.9});
  timeline_.SetVisibleRange({0.0, 166.9});
  timeline_.set_data_time_range({0.0, 166.9});
  ImGuiIO& io = ImGui::GetIO();

  // Start without Shift.
  io.AddKeyEvent(ImGuiMod_Shift, false);

  // Start drag in timeline area.
  io.MousePos = ImVec2(300.0f, 50.0f);
  io.AddMouseButtonEvent(0, true);

  EXPECT_CALL(timeline_, Pan(0.0f));
  EXPECT_CALL(timeline_, Scroll(0.0f));
  SimulateFrame();

  // Press Shift key mid-drag.
  io.AddKeyEvent(ImGuiMod_Shift, true);

  // Drag mouse to left (simulate pan right).
  // Move from 300 to 200 (-100px).
  io.MousePos = ImVec2(200.0f, 50.0f);

  EXPECT_CALL(timeline_, Pan(FloatEq(100.0f)));
  EXPECT_CALL(timeline_, Scroll(FloatEq(0.0f)));
  SimulateFrame();

  // End drag.
  io.AddMouseButtonEvent(0, false);
  SimulateFrame();

  // Verify that NO selection was created.
  EXPECT_TRUE(timeline_.selected_time_ranges().empty());
}

TEST_F(MockTimelineImGuiFixture, DrawEventNameTextHiddenWhenTooNarrow) {
  FlameChartTimelineData data;

  data.groups.push_back(
      {.name = "Group 1", .start_level = 0, .nesting_level = 0});
  data.events_by_level.push_back({0});
  data.entry_names.push_back("event1");
  data.entry_levels.push_back(0);
  data.entry_start_times.push_back(10.0);
  data.entry_total_times.push_back(0.001);
  data.entry_pids.push_back(1);
  data.entry_args.push_back({});
  timeline_.set_timeline_data(std::move(data));
  timeline_.SetVisibleRange({0.0, 100.0});

  // The event rect width will be kEventMinimumDrawWidth = 2.0f because
  // event duration 0.001 is small.
  // kMinTextWidth is 5.0f.
  // Since 2.0f < 5.0f, DrawEventName should not draw text, so GetTextForDisplay
  // and CalculateEventTextRect won't be called, and thus GetTextSize should not
  // be called.
  EXPECT_CALL(timeline_, GetTextSize(_)).Times(0);

  SimulateFrame();
}

TEST_F(MockTimelineImGuiFixture,
       DrawEventNameTextHiddenWhenSlightlyNarrowerThanMinTextWidth) {
  FlameChartTimelineData data;

  data.groups.push_back(
      {.name = "Group 1", .start_level = 0, .nesting_level = 0});
  data.events_by_level.push_back({0});
  data.entry_names.push_back("event1");
  data.entry_levels.push_back(0);
  data.entry_start_times.push_back(10.0);
  data.entry_total_times.push_back(0.255);
  data.entry_pids.push_back(1);
  data.entry_args.push_back({});
  timeline_.set_timeline_data(std::move(data));
  timeline_.SetVisibleRange({0.0, 100.0});

  // The event rect width will be around 4.51f, which is < kMinTextWidth (5.0f).
  // DrawEventName should not draw text, so GetTextForDisplay and
  // CalculateEventTextRect won't be called, and GetTextSize should not be
  // called.
  EXPECT_CALL(timeline_, GetTextSize(_)).Times(0);

  SimulateFrame();
}

TEST_F(MockTimelineImGuiFixture, HandleWheel_DiagonalScroll) {
  // Simulate diagonal scrolling (both horizontal and vertical wheel).
  ImGui::GetIO().AddMouseWheelEvent(1.0f, 2.0f);  // X=1, Y=2

  // Expect standard behavior:
  // MouseWheelH (X) -> Pan
  // MouseWheel (Y) -> Scroll
  EXPECT_CALL(timeline_, Pan(FloatEq(1.0f)));
  EXPECT_CALL(timeline_, Scroll(FloatEq(2.0f)));

  SimulateFrame();
}

TEST_F(MockTimelineImGuiFixture, HandleWheel_Shift_DiagonalScroll) {
  // Simulate diagonal scrolling with Shift key.
  ImGui::GetIO().AddMouseWheelEvent(1.0f, 2.0f);  // X=1, Y=2

  // Manually run the frame loop to inject KeyShift after NewFrame updates the
  // IO. This avoids issues where NewFrame might reset io.KeyShift if the key
  // event isn't processed as expected in the mock.
  ImGui::NewFrame();
  ImGui::GetIO().KeyShift = true;

  // Expect swapped behavior:
  // MouseWheel (Y) -> Pan
  // MouseWheelH (X) -> Scroll
  EXPECT_CALL(timeline_, Pan(FloatEq(2.0f)));
  EXPECT_CALL(timeline_, Scroll(FloatEq(1.0f)));

  timeline_.Draw();
  Animation::UpdateAll(ImGui::GetIO().DeltaTime);
  ImGui::EndFrame();
}

using RealTimelineImGuiFixture = TimelineImGuiTestFixture<Timeline>;

// Add a sanity check that the window padding is set to zero.
// This is the presumption for all the drawing logic. And all tests below assume
// this.
TEST_F(RealTimelineImGuiFixture, DrawSetsWindowPaddingToZero) {
  ImGui::NewFrame();
  timeline_.Draw();

  ImGuiWindow* window = ImGui::FindWindowByName("Timeline viewer");
  ASSERT_NE(window, nullptr);
  EXPECT_EQ(window->WindowPadding.x, 0.0f);
  EXPECT_EQ(window->WindowPadding.y, 0.0f);

  ImGui::EndFrame();
}

TEST_F(RealTimelineImGuiFixture, ClickEventSelectsEvent) {
  FlameChartTimelineData data;

  data.groups.push_back(
      {.name = "Group 1", .start_level = 0, .nesting_level = 0});
  data.events_by_level.push_back({0});
  data.entry_names.push_back("event1");
  data.entry_levels.push_back(0);
  data.entry_start_times.push_back(0.0);
  data.entry_total_times.push_back(100.0);
  data.entry_pids.push_back(1);
  data.entry_args.push_back({});
  timeline_.set_timeline_data(std::move(data));
  timeline_.SetVisibleRange({0.0, 100.0});

  bool callback_called = false;
  std::string event_type;
  EventData event_detail;
  timeline_.set_event_callback(
      [&](absl::string_view type, const EventData& detail) {
        callback_called = true;
        event_type = type;
        event_detail = detail;
      });

  // Set a mouse position that is guaranteed to be over the event, since the
  // event spans the entire timeline.
  // y=28 is safely within the event rect (starts at 20, height 16 -> ends at
  // 36).
  ImGui::GetIO().MousePos = ImVec2(300.f, 28.f);
  ImGui::GetIO().MouseDown[0] = true;

  SimulateFrame();

  EXPECT_TRUE(callback_called);
  EXPECT_EQ(event_type, kEventSelected);
  ASSERT_TRUE(event_detail.contains(kEventSelectedIndex));
  ASSERT_TRUE(event_detail.contains(kEventSelectedName));
  EXPECT_EQ(std::any_cast<int>(event_detail.at(kEventSelectedIndex)), 0);
  EXPECT_EQ(std::any_cast<std::string>(event_detail.at(kEventSelectedName)),
            "event1");
}

TEST_F(RealTimelineImGuiFixture, ClickOutsideEventDoesNotSelectEvent) {
  FlameChartTimelineData data;

  data.groups.push_back(
      {.name = "Group 1", .start_level = 0, .nesting_level = 0});
  data.events_by_level.push_back({0});
  data.entry_names.push_back("event1");
  data.entry_levels.push_back(0);
  data.entry_start_times.push_back(0.0);
  data.entry_total_times.push_back(100.0);
  data.entry_pids.push_back(1);
  data.entry_args.push_back({});
  timeline_.set_timeline_data(std::move(data));
  timeline_.SetVisibleRange({0.0, 100.0});

  bool callback_called = false;
  timeline_.set_event_callback(
      [&](absl::string_view type, const EventData& detail) {
        callback_called = true;
      });

  // Set a mouse position that is guaranteed to be outside the event.
  ImGui::GetIO().MousePos = ImVec2(300.f, 100.f);
  ImGui::GetIO().MouseDown[0] = true;

  SimulateFrame();

  EXPECT_FALSE(callback_called);
}

TEST_F(RealTimelineImGuiFixture,
       ClickingSelectedEventAgainDoesNotFireCallback) {
  FlameChartTimelineData data;

  data.groups.push_back(
      {.name = "Group 1", .start_level = 0, .nesting_level = 0});
  data.events_by_level.push_back({0});
  data.entry_names.push_back("event1");
  data.entry_levels.push_back(0);
  data.entry_start_times.push_back(0.0);
  data.entry_total_times.push_back(100.0);
  data.entry_pids.push_back(1);
  data.entry_args.push_back({});
  timeline_.set_timeline_data(std::move(data));
  timeline_.SetVisibleRange({0.0, 100.0});

  int callback_count = 0;
  timeline_.set_event_callback(
      [&](absl::string_view type, const EventData& detail) {
        callback_count++;
      });

  ImGui::GetIO().MousePos = ImVec2(300.f, 28.f);

  // First click.
  ImGui::GetIO().MouseDown[0] = true;
  SimulateFrame();

  EXPECT_EQ(callback_count, 1);

  // Frame with mouse up.
  ImGui::GetIO().MouseDown[0] = false;
  SimulateFrame();

  // Second click on the same event.
  ImGui::GetIO().MouseDown[0] = true;
  SimulateFrame();

  // Callback count should still be 1.
  EXPECT_EQ(callback_count, 1);
}

TEST_F(RealTimelineImGuiFixture, ClickEmptyAreaDeselectsEvent) {
  FlameChartTimelineData data;

  data.groups.push_back(
      {.name = "Group 1", .start_level = 0, .nesting_level = 0});
  data.events_by_level.push_back({0});
  data.entry_names.push_back("event1");
  data.entry_levels.push_back(0);
  data.entry_start_times.push_back(0.0);
  data.entry_total_times.push_back(100.0);
  data.entry_pids.push_back(1);
  data.entry_args.push_back({});
  timeline_.set_timeline_data(std::move(data));
  timeline_.SetVisibleRange({0.0, 100.0});

  // First, select an event.
  ImGui::GetIO().MousePos = ImVec2(300.f, 28.f);  // A position over the event.
  ImGui::GetIO().MouseDown[0] = true;
  SimulateFrame();
  ImGui::GetIO().MouseDown[0] = false;  // Release the mouse.
  SimulateFrame();

  bool deselection_callback_called = false;
  timeline_.set_event_callback(
      [&](absl::string_view type, const EventData& detail) {
        if (type == kEventSelected) {
          auto it = detail.find(std::string(kEventSelectedIndex));
          if (it != detail.end()) {
            if (std::any_cast<int>(it->second) == -1) {
              deselection_callback_called = true;
            }
          }
        }
      });

  // Now, click on an empty area.
  ImGui::GetIO().MousePos =
      ImVec2(300.f, 100.f);  // A position outside the event.
  ImGui::GetIO().MouseDown[0] = true;

  SimulateFrame();

  EXPECT_TRUE(deselection_callback_called);
}

TEST_F(RealTimelineImGuiFixture, ClickEmptyAreaDeselectsOnlyOnce) {
  FlameChartTimelineData data;

  data.groups.push_back(
      {.name = "Group 1", .start_level = 0, .nesting_level = 0});
  data.events_by_level.push_back({0});
  data.entry_names.push_back("event1");
  data.entry_levels.push_back(0);
  data.entry_start_times.push_back(0.0);
  data.entry_total_times.push_back(100.0);
  data.entry_pids.push_back(1);
  data.entry_args.push_back({});
  timeline_.set_timeline_data(std::move(data));
  timeline_.SetVisibleRange({0.0, 100.0});

  // First, select an event.
  ImGui::GetIO().MousePos = ImVec2(300.f, 28.f);  // A position over the event.
  ImGui::GetIO().MouseDown[0] = true;
  SimulateFrame();
  ImGui::GetIO().MouseDown[0] = false;  // Release the mouse.
  SimulateFrame();

  int deselection_callback_count = 0;
  timeline_.set_event_callback(
      [&](absl::string_view type, const EventData& detail) {
        if (type == kEventSelected) {
          auto it = detail.find(std::string(kEventSelectedIndex));
          if (it != detail.end()) {
            if (std::any_cast<int>(it->second) == -1) {
              deselection_callback_count++;
            }
          }
        }
      });

  // Click on an empty area to deselect.
  ImGui::GetIO().MousePos =
      ImVec2(300.f, 100.f);  // A position outside the event.
  ImGui::GetIO().MouseDown[0] = true;
  SimulateFrame();
  ImGui::GetIO().MouseDown[0] = false;
  SimulateFrame();

  EXPECT_EQ(deselection_callback_count, 1);

  // Click on an empty area again.
  ImGui::GetIO().MouseDown[0] = true;
  SimulateFrame();

  // The deselection callback should not be called again.
  EXPECT_EQ(deselection_callback_count, 1);
}

TEST_F(RealTimelineImGuiFixture, ClickEmptyAreaWhenNoEventSelectedDoesNothing) {
  FlameChartTimelineData data;

  data.groups.push_back(
      {.name = "Group 1", .start_level = 0, .nesting_level = 0});
  data.events_by_level.push_back({0});
  data.entry_names.push_back("event1");
  data.entry_levels.push_back(0);
  data.entry_start_times.push_back(0.0);
  data.entry_total_times.push_back(100.0);
  data.entry_pids.push_back(1);
  data.entry_args.push_back({});
  timeline_.set_timeline_data(std::move(data));
  timeline_.SetVisibleRange({0.0, 100.0});

  bool callback_called = false;
  timeline_.set_event_callback(
      [&](absl::string_view type, const EventData& detail) {
        callback_called = true;
      });

  // Now, click on an empty area.
  ImGui::GetIO().MousePos =
      ImVec2(300.f, 100.f);  // A position outside the event.
  ImGui::GetIO().MouseDown[0] = true;

  SimulateFrame();

  EXPECT_FALSE(callback_called);
}

TEST_F(RealTimelineImGuiFixture, DrawsTimelineWindowWhenTimelineDataIsEmpty) {
  timeline_.set_timeline_data({});

  // We don't use SimulateFrame() here because we need to inspect the draw list
  // before ImGui::EndFrame() is called.
  ImGui::NewFrame();
  timeline_.Draw();

  EXPECT_NE(ImGui::FindWindowByName("Timeline viewer"), nullptr);

  ImGui::EndFrame();
}

TEST_F(RealTimelineImGuiFixture, ShiftClickEventTogglesCurtain) {
  FlameChartTimelineData data;
  data.groups.push_back(
      {.name = "Group 1", .start_level = 0, .nesting_level = 0});
  data.events_by_level.push_back({0});
  data.entry_names.push_back("event1");
  data.entry_levels.push_back(0);
  data.entry_start_times.push_back(10.0);
  data.entry_total_times.push_back(20.0);
  data.entry_pids.push_back(1);
  data.entry_args.push_back({});
  timeline_.set_timeline_data(std::move(data));
  timeline_.SetVisibleRange({0.0, 100.0});

  // Mouse is over the event
  ImGui::GetIO().MousePos = ImVec2(500.f, 28.f);
  ImGui::GetIO().AddKeyEvent(ImGuiMod_Shift, true);
  ImGui::GetIO().MouseDown[0] = true;

  // First shift-click, should add a curtain range.
  SimulateFrame();

  ASSERT_EQ(timeline_.selected_time_ranges().size(), 1);
  EXPECT_EQ(timeline_.selected_time_ranges()[0].start(), 10.0);
  EXPECT_EQ(timeline_.selected_time_ranges()[0].end(), 30.0);

  // Frame with mouse up
  ImGui::GetIO().MouseDown[0] = false;
  SimulateFrame();

  // Second shift-click on the same event, should remove the curtain.
  ImGui::GetIO().MouseDown[0] = true;
  SimulateFrame();

  EXPECT_TRUE(timeline_.selected_time_ranges().empty());

  // Reset the mouse and shift key to avoid affecting other tests.
  ImGui::GetIO().MouseDown[0] = false;
  ImGui::GetIO().AddKeyEvent(ImGuiMod_Shift, false);
}

TEST_F(RealTimelineImGuiFixture,
       ShiftClickMultipleEventsSelectsMultipleRanges) {
  FlameChartTimelineData data;
  data.groups.push_back(
      {.name = "Group 1", .start_level = 0, .nesting_level = 0});
  data.events_by_level.push_back({0, 1});  // event 0 and 1 on level 0
  data.entry_names.push_back("event1");
  data.entry_names.push_back("event2");
  data.entry_levels.push_back(0);
  data.entry_levels.push_back(0);
  data.entry_start_times.push_back(10.0);
  data.entry_start_times.push_back(50.0);
  data.entry_total_times.push_back(20.0);
  data.entry_total_times.push_back(10.0);
  data.entry_pids.push_back(1);
  data.entry_pids.push_back(2);
  data.entry_args.push_back({});
  data.entry_args.push_back({});
  timeline_.set_timeline_data(std::move(data));
  timeline_.SetVisibleRange({0.0, 100.0});

  ImGui::GetIO().AddKeyEvent(ImGuiMod_Shift, true);

  // First shift-click on event 1.
  ImGui::GetIO().MousePos = ImVec2(500.f, 28.f);  // Position over event 1.
  ImGui::GetIO().MouseDown[0] = true;
  SimulateFrame();

  ASSERT_EQ(timeline_.selected_time_ranges().size(), 1);
  EXPECT_EQ(timeline_.selected_time_ranges()[0].start(), 10.0);
  EXPECT_EQ(timeline_.selected_time_ranges()[0].end(), 30.0);

  // Frame with mouse up.
  ImGui::GetIO().MouseDown[0] = false;
  SimulateFrame();

  // Second shift-click on event 2.
  ImGui::GetIO().MousePos = ImVec2(1100.f, 28.f);  // Position over event 2.
  ImGui::GetIO().MouseDown[0] = true;
  SimulateFrame();

  ASSERT_EQ(timeline_.selected_time_ranges().size(), 2);
  EXPECT_EQ(timeline_.selected_time_ranges()[0].start(), 10.0);
  EXPECT_EQ(timeline_.selected_time_ranges()[0].end(), 30.0);
  EXPECT_EQ(timeline_.selected_time_ranges()[1].start(), 50.0);
  EXPECT_EQ(timeline_.selected_time_ranges()[1].end(), 60.0);

  // Frame with mouse up.
  ImGui::GetIO().MouseDown[0] = false;
  SimulateFrame();

  // Third shift-click on event 1 again to deselect.
  ImGui::GetIO().MousePos = ImVec2(500.f, 28.f);  // Position over event 1.
  ImGui::GetIO().MouseDown[0] = true;
  SimulateFrame();

  ASSERT_EQ(timeline_.selected_time_ranges().size(), 1);
  EXPECT_EQ(timeline_.selected_time_ranges()[0].start(), 50.0);
  EXPECT_EQ(timeline_.selected_time_ranges()[0].end(), 60.0);

  // Reset the mouse and shift key to avoid affecting other tests.
  ImGui::GetIO().MouseDown[0] = false;
  ImGui::GetIO().AddKeyEvent(ImGuiMod_Shift, false);
}

TEST_F(RealTimelineImGuiFixture, PanLeftBeyondDataRangeShouldBeConstrained) {
  timeline_.set_data_time_range({10.0, 100.0});
  timeline_.SetVisibleRange({11.0, 61.0});

  // Simulate holding 'A' (Pan Left). Panning left will attempt to move the
  // visible range before the data range start, so it should be constrained.
  SimulateKeyHeldForDuration(ImGuiKey_A, 1.0f);
  SimulateFrame();

  // The visible range end should not go below the data range start (10.0).
  EXPECT_DOUBLE_EQ(timeline_.visible_range().start(), 10.0);
}

TEST_F(RealTimelineImGuiFixture, PanRightBeyondDataRangeShouldBeConstrained) {
  timeline_.set_data_time_range({10.0, 100.0});
  timeline_.SetVisibleRange({49.0, 99.0});

  // Simulate holding 'D' (Pan Right). Panning right will attempt to move the
  // visible range beyond the data range end, so it should be constrained.
  SimulateKeyHeldForDuration(ImGuiKey_D, 1.0f);
  SimulateFrame();

  // The visible range end should not go above the data range end (100.0).
  EXPECT_DOUBLE_EQ(timeline_.visible_range().end(), 100.0);
}

TEST_F(RealTimelineImGuiFixture, ZoomInBeyondMinDurationShouldBeConstrained) {
  timeline_.set_data_time_range({0.0, 100.0});
  // set duration to be very close to kMinDurationMicros
  timeline_.SetVisibleRange(
      {50.0 - kMinDurationMicros / 2.0, 50.0 + kMinDurationMicros / 2.0});

  ASSERT_NEAR(timeline_.visible_range().duration(), kMinDurationMicros, 1e-9);

  // Zoom in, duration should decrease but be capped at kMinDurationMicros by
  // ConstrainTimeRange. Hold W for 1s to zoom in a lot.
  SimulateKeyHeldForDuration(ImGuiKey_W, 1.0f);
  SimulateFrame();

  EXPECT_NEAR(timeline_.visible_range().duration(), kMinDurationMicros, 1e-9);
  EXPECT_DOUBLE_EQ(timeline_.visible_range().center(), 50.0);
}

class TimelineDragSelectionTest : public RealTimelineImGuiFixture {
 protected:
  void SetUp() override {
    RealTimelineImGuiFixture::SetUp();
    // Set a visible range that results in a round number for px_per_time_unit
    // to make test calculations predictable. With a timeline width of 1669px
    // (based on 1920px window width, 250px label width, and 1px padding),
    // a duration of 166.9 gives 10px per microsecond.
    timeline_.SetVisibleRange({0.0, 166.9});
    timeline_.set_data_time_range({0.0, 166.9});

    ImGui::GetIO().AddKeyEvent(ImGuiMod_Shift, true);
  }

  void TearDown() override {
    ImGui::GetIO().AddKeyEvent(ImGuiMod_Shift, false);
    RealTimelineImGuiFixture::TearDown();
  }
};

TEST_F(TimelineDragSelectionTest, ShiftDragCreatesTimeSelection) {
  ImGuiIO& io = ImGui::GetIO();

  // Start drag in timeline area.
  // The label column is 250px wide, so timeline starts after that.
  io.MousePos = ImVec2(300.0f, 50.0f);
  io.AddMouseButtonEvent(0, true);
  SimulateFrame();

  // Dragging
  io.MousePos = ImVec2(500.0f, 50.0f);
  SimulateFrame();

  // End drag
  io.AddMouseButtonEvent(0, false);
  SimulateFrame();

  ASSERT_EQ(timeline_.selected_time_ranges().size(), 1);
  const TimeRange& range = timeline_.selected_time_ranges()[0];
  EXPECT_DOUBLE_EQ(range.start(), 5.0);
  EXPECT_DOUBLE_EQ(range.end(), 25.0);
}

TEST_F(TimelineDragSelectionTest, ShiftDragCreatesMultipleTimeSelections) {
  ImGuiIO& io = ImGui::GetIO();

  // First drag
  io.MousePos = ImVec2(300.0f, 50.0f);
  io.AddMouseButtonEvent(0, true);
  SimulateFrame();
  io.MousePos = ImVec2(400.0f, 50.0f);
  SimulateFrame();
  io.AddMouseButtonEvent(0, false);
  SimulateFrame();

  ASSERT_EQ(timeline_.selected_time_ranges().size(), 1);
  EXPECT_DOUBLE_EQ(timeline_.selected_time_ranges()[0].start(), 5.0);
  EXPECT_DOUBLE_EQ(timeline_.selected_time_ranges()[0].end(), 15.0);

  // Second drag
  io.MousePos = ImVec2(500.0f, 50.0f);
  io.AddMouseButtonEvent(0, true);
  SimulateFrame();
  io.MousePos = ImVec2(600.0f, 50.0f);
  SimulateFrame();
  io.AddMouseButtonEvent(0, false);
  SimulateFrame();

  ASSERT_EQ(timeline_.selected_time_ranges().size(), 2);
  EXPECT_DOUBLE_EQ(timeline_.selected_time_ranges()[0].start(), 5.0);
  EXPECT_DOUBLE_EQ(timeline_.selected_time_ranges()[0].end(), 15.0);
  EXPECT_DOUBLE_EQ(timeline_.selected_time_ranges()[1].start(), 25.0);
  EXPECT_DOUBLE_EQ(timeline_.selected_time_ranges()[1].end(), 35.0);
}

TEST_F(TimelineDragSelectionTest, DraggingUpdatesCurrentSelectedTimeRange) {
  ImGuiIO& io = ImGui::GetIO();

  // Start drag in timeline area.
  io.MousePos = ImVec2(300.0f, 50.0f);
  io.AddMouseButtonEvent(0, true);
  SimulateFrame();

  // Dragging
  io.MousePos = ImVec2(500.0f, 50.0f);
  SimulateFrame();

  // During drag, current_selected_time_range_ should be set, but
  // selected_time_ranges_ should be empty.
  ASSERT_TRUE(timeline_.current_selected_time_range().has_value());
  EXPECT_DOUBLE_EQ(timeline_.current_selected_time_range()->start(), 5.0);
  EXPECT_DOUBLE_EQ(timeline_.current_selected_time_range()->end(), 25.0);
  EXPECT_TRUE(timeline_.selected_time_ranges().empty());

  // End drag
  io.AddMouseButtonEvent(0, false);
  SimulateFrame();

  // After drag, current_selected_time_range_ should be reset, and
  // selected_time_ranges_ should contain the new range.
  EXPECT_FALSE(timeline_.current_selected_time_range().has_value());
  ASSERT_EQ(timeline_.selected_time_ranges().size(), 1);
  EXPECT_DOUBLE_EQ(timeline_.selected_time_ranges()[0].start(), 5.0);
  EXPECT_DOUBLE_EQ(timeline_.selected_time_ranges()[0].end(), 25.0);
}

TEST_F(TimelineDragSelectionTest, ClickCloseButtonRemovesSelectedTimeRange) {
  ImGuiIO& io = ImGui::GetIO();

  // Start drag in timeline area.
  io.MousePos = ImVec2(300.0f, 50.0f);
  io.AddMouseButtonEvent(0, true);
  SimulateFrame();

  // Dragging
  io.MousePos = ImVec2(500.0f, 50.0f);
  SimulateFrame();

  // End drag
  io.AddMouseButtonEvent(0, false);
  SimulateFrame();

  ASSERT_EQ(timeline_.selected_time_ranges().size(), 1);

  // Calculate button position.
  // The range is [5.0, 25.0], duration is 20.0 us.
  // FormatTime uses %.4g and non-breaking space. 20.0 becomes "20".
  const std::string text = "20\xc2\xa0us";
  const ImVec2 text_size = ImGui::CalcTextSize(text.c_str());

  // Coordinates:
  // timeline_x_start = 250.0f (label_width)
  // range_start_x = 300.0f
  // range_end_x = 500.0f
  // text_x = 300 + (200 - text_size.x) / 2
  // text_y = 1080 - text_size.y - kSelectedTimeRangeTextBottomPadding (10.0f)
  const float text_x = 300.0f + (200.0f - text_size.x) / 2.0f;
  const float text_y =
      io.DisplaySize.y - text_size.y - kSelectedTimeRangeTextBottomPadding;

  const float button_x = text_x + text_size.x + kCloseButtonPadding;
  const float button_y = text_y + (text_size.y - kCloseButtonSize) / 2.0f;

  const ImVec2 button_center(button_x + kCloseButtonSize / 2.0f,
                             button_y + kCloseButtonSize / 2.0f);

  // Move mouse to button and click.
  io.MousePos = button_center;
  // Simulate a frame to update the hover state and verify the cursor changes to
  // a hand.
  SimulateFrame();
  EXPECT_EQ(ImGui::GetMouseCursor(), ImGuiMouseCursor_Hand);

  io.AddMouseButtonEvent(0, true);
  SimulateFrame();
  io.AddMouseButtonEvent(0, false);
  SimulateFrame();

  EXPECT_TRUE(timeline_.selected_time_ranges().empty());
  // Verify that the cursor changes back to an arrow after the button is
  // removed.
  EXPECT_EQ(ImGui::GetMouseCursor(), ImGuiMouseCursor_Arrow);
}

TEST_F(TimelineDragSelectionTest, ClickingTextDoesNotRemoveSelectedTimeRange) {
  ImGuiIO& io = ImGui::GetIO();

  // Start drag in timeline area.
  io.MousePos = ImVec2(300.0f, 50.0f);
  io.AddMouseButtonEvent(0, true);
  SimulateFrame();

  // Dragging
  io.MousePos = ImVec2(500.0f, 50.0f);
  SimulateFrame();

  // End drag
  io.AddMouseButtonEvent(0, false);
  SimulateFrame();

  ASSERT_EQ(timeline_.selected_time_ranges().size(), 1);

  // FormatTime uses %.4g and non-breaking space. 20.0 becomes "20".
  const std::string text = "20\xc2\xa0us";
  const ImVec2 text_size = ImGui::CalcTextSize(text.c_str());

  const float text_x = 300.0f + (200.0f - text_size.x) / 2.0f;
  const float text_y =
      io.DisplaySize.y - text_size.y - kSelectedTimeRangeTextBottomPadding;

  // Click on the text (center of text).
  const ImVec2 text_center(text_x + text_size.x / 2.0f,
                           text_y + text_size.y / 2.0f);

  io.MousePos = text_center;
  io.AddMouseButtonEvent(0, true);
  SimulateFrame();
  io.AddMouseButtonEvent(0, false);
  SimulateFrame();

  EXPECT_EQ(timeline_.selected_time_ranges().size(), 1);
}

TEST_F(RealTimelineImGuiFixture, DrawCounterTrack) {
  FlameChartTimelineData data;
  data.groups.push_back({.type = Group::Type::kCounter,
                         .name = "Counter Group",
                         .start_level = 0,
                         .nesting_level = 0});

  CounterData counter_data;
  counter_data.timestamps = {10.0, 20.0, 30.0};
  counter_data.values = {0.0, 10.0, 5.0};
  counter_data.min_value = 0.0;
  counter_data.max_value = 10.0;
  data.counter_data_by_group_index[0] = std::move(counter_data);

  timeline_.set_timeline_data(std::move(data));
  timeline_.SetVisibleRange({0.0, 100.0});

  ImGui::NewFrame();
  timeline_.Draw();

  ImGuiWindow* counter_window = nullptr;
  // The child window name is constructed as
  // "TimelineChild_<group_name>_<group_index>". We search for a window that
  // contains this string in its name.
  const std::string child_id = "TimelineChild_Counter Group_0";
  for (ImGuiWindow* w : ImGui::GetCurrentContext()->Windows) {
    if (std::string(w->Name).find(child_id) != std::string::npos) {
      counter_window = w;
      break;
    }
  }
  ASSERT_NE(counter_window, nullptr);

  // Check if anything was drawn to this window's draw list.
  EXPECT_FALSE(counter_window->DrawList->VtxBuffer.empty());

  ImGui::EndFrame();
}

TEST_F(RealTimelineImGuiFixture, HoverCounterTrackShowsTooltip) {
  FlameChartTimelineData data;
  data.groups.push_back({.type = Group::Type::kCounter,
                         .name = "Counter Group",
                         .start_level = 0,
                         .nesting_level = 0});

  CounterData counter_data;
  counter_data.timestamps = {10.0, 20.0, 30.0};
  counter_data.values = {0.0, 10.0, 5.0};
  counter_data.min_value = 0.0;
  counter_data.max_value = 10.0;
  data.counter_data_by_group_index[0] = std::move(counter_data);

  timeline_.set_timeline_data(std::move(data));
  timeline_.SetVisibleRange({0.0, 100.0});

  // Render first frame to layout windows and find the counter track location.
  ImGui::NewFrame();
  timeline_.Draw();

  ImGuiWindow* counter_window = nullptr;
  const std::string child_id = "TimelineChild_Counter Group_0";
  for (ImGuiWindow* w : ImGui::GetCurrentContext()->Windows) {
    if (std::string(w->Name).find(child_id) != std::string::npos) {
      counter_window = w;
      break;
    }
  }
  ASSERT_NE(counter_window, nullptr);

  // Check that initially there are no black vertices (no circle outline).
  bool has_black_vertices = false;
  for (const auto& vtx : counter_window->DrawList->VtxBuffer) {
    if (vtx.col == kBlackColor) {
      has_black_vertices = true;
      break;
    }
  }
  EXPECT_FALSE(has_black_vertices);

  // Calculate a position over the track corresponding to timestamp 20.0.
  // The track handles its own X mapping using TimeToScreenX with
  // GetCursorScreenPos. We can just use the window's position and size to pick
  // a point inside. Since visible range is 0-100 and data has points at 10, 20,
  // 30, they should be roughly at 10%, 20%, 30% of the width. Let's target 20.0
  // (20% width).
  ImVec2 target_pos = counter_window->Pos;
  target_pos.x += counter_window->Size.x * 0.2f;
  target_pos.y += counter_window->Size.y * 0.5f;

  ImGui::EndFrame();

  // Next frame: Move mouse to target position.
  ImGui::GetIO().MousePos = target_pos;
  ImGui::NewFrame();
  timeline_.Draw();

  // Find window again (pointer might be unstable across frames if reallocations
  // happen, though usually stable).
  counter_window = nullptr;
  for (ImGuiWindow* w : ImGui::GetCurrentContext()->Windows) {
    if (std::string(w->Name).find(child_id) != std::string::npos) {
      counter_window = w;
      break;
    }
  }
  ASSERT_NE(counter_window, nullptr);

  // Check for black vertices (circle outline).
  has_black_vertices = false;
  for (const auto& vtx : counter_window->DrawList->VtxBuffer) {
    if (vtx.col == kBlackColor) {
      has_black_vertices = true;
      break;
    }
  }
  EXPECT_TRUE(has_black_vertices);

  ImGui::EndFrame();
}

TEST_F(RealTimelineImGuiFixture, ClickEventSetsSelectionIndices) {
  FlameChartTimelineData data;
  data.groups.push_back(
      {.name = "Group 1", .start_level = 0, .nesting_level = 0});
  data.events_by_level.push_back({0});
  data.entry_names.push_back("event1");
  data.entry_levels.push_back(0);
  data.entry_start_times.push_back(0.0);
  data.entry_total_times.push_back(100.0);
  data.entry_pids.push_back(1);
  data.entry_args.push_back({});
  timeline_.set_timeline_data(std::move(data));
  timeline_.SetVisibleRange({0.0, 100.0});

  // Set a mouse position that is guaranteed to be over the event.
  ImGui::GetIO().MousePos = ImVec2(300.f, 28.f);
  ImGui::GetIO().MouseDown[0] = true;

  SimulateFrame();

  EXPECT_EQ(timeline_.selected_event_index(), 0);
  EXPECT_EQ(timeline_.selected_group_index(), 0);
  EXPECT_EQ(timeline_.selected_counter_index(), -1);
}

TEST_F(RealTimelineImGuiFixture, ClickCounterEventSetsSelectionIndices) {
  FlameChartTimelineData data;
  data.groups.push_back({.type = Group::Type::kCounter,
                         .name = "Counter Group",
                         .start_level = 0,
                         .nesting_level = 0});

  CounterData counter_data;
  counter_data.timestamps = {10.0, 20.0, 30.0};
  counter_data.values = {0.0, 10.0, 5.0};
  counter_data.min_value = 0.0;
  counter_data.max_value = 10.0;
  data.counter_data_by_group_index[0] = std::move(counter_data);

  timeline_.set_timeline_data(std::move(data));
  timeline_.SetVisibleRange({0.0, 100.0});

  ImGui::NewFrame();
  timeline_.Draw();

  ImGuiWindow* counter_window = nullptr;
  const std::string child_id = "TimelineChild_Counter Group_0";
  for (ImGuiWindow* w : ImGui::GetCurrentContext()->Windows) {
    if (std::string(w->Name).find(child_id) != std::string::npos) {
      counter_window = w;
      break;
    }
  }
  ASSERT_NE(counter_window, nullptr);

  // Calculate a position over the track corresponding to timestamp 20.0.
  // We use 0.25f (25% of timeline width) instead of 0.2f (20%) to ensure we
  // click safely to the right of the 20.0 timestamp (which is at 20%).
  // This accounts for ImGui window padding which shifts the content origin
  // right, effectively reducing the "time" value for a given absolute X pixel.
  // With 0.2f, the calculated time might fall slightly below 20.0 due to this
  // shift, causing the selection to pick the previous interval (or none).
  ImVec2 target_pos = counter_window->Pos;
  target_pos.x += counter_window->Size.x * 0.25f;
  target_pos.y += counter_window->Size.y * 0.5f;

  ImGui::EndFrame();

  // Next frame: Move mouse to target position and click.
  ImGui::GetIO().MousePos = target_pos;
  ImGui::GetIO().MouseDown[0] = true;
  SimulateFrame();

  EXPECT_EQ(timeline_.selected_event_index(), -1);
  EXPECT_EQ(timeline_.selected_group_index(), 0);
  // Timestamp 20.0 is at index 1.
  EXPECT_EQ(timeline_.selected_counter_index(), 1);
}

TEST_F(RealTimelineImGuiFixture, SelectionMutualExclusion) {
  FlameChartTimelineData data;
  // Group 0: Flame Events
  data.groups.push_back(
      {.name = "Group 1", .start_level = 0, .nesting_level = 0});
  data.events_by_level.push_back({0});
  data.entry_names.push_back("event1");
  data.entry_levels.push_back(0);
  data.entry_start_times.push_back(0.0);
  data.entry_total_times.push_back(100.0);
  data.entry_pids.push_back(1);
  data.entry_args.push_back({});

  // Group 1: Counter Events
  data.groups.push_back({.type = Group::Type::kCounter,
                         .name = "Counter Group",
                         .start_level = 1,
                         .nesting_level = 0});
  CounterData counter_data;
  // We need at least 2 timestamps for the counter track to be drawn.
  counter_data.timestamps = {20.0, 30.0};
  counter_data.values = {5.0, 5.0};
  counter_data.min_value = 0.0;
  counter_data.max_value = 10.0;
  data.counter_data_by_group_index[1] = std::move(counter_data);

  timeline_.set_timeline_data(std::move(data));
  timeline_.SetVisibleRange({0.0, 100.0});

  // Step 1: Select Flame Event
  ImGui::GetIO().MousePos = ImVec2(300.f, 28.f);  // Over flame event
  ImGui::GetIO().MouseDown[0] = true;
  SimulateFrame();
  ImGui::GetIO().MouseDown[0] = false;
  SimulateFrame();

  EXPECT_EQ(timeline_.selected_event_index(), 0);
  EXPECT_EQ(timeline_.selected_group_index(), 0);
  EXPECT_EQ(timeline_.selected_counter_index(), -1);

  // Step 2: Select Counter Event
  ImGui::NewFrame();
  timeline_.Draw();
  ImGuiWindow* counter_window = nullptr;
  const std::string child_id = "TimelineChild_Counter Group_1";
  for (ImGuiWindow* w : ImGui::GetCurrentContext()->Windows) {
    if (std::string(w->Name).find(child_id) != std::string::npos) {
      counter_window = w;
      break;
    }
  }
  ASSERT_NE(counter_window, nullptr);
  ImVec2 counter_pos = counter_window->Pos;
  // Use 0.25f to be safe against window padding.
  counter_pos.x += counter_window->Size.x * 0.25f;  // At 20.0 (starts at 20%)
  counter_pos.y += counter_window->Size.y * 0.5f;
  ImGui::EndFrame();

  ImGui::GetIO().MousePos = counter_pos;
  ImGui::GetIO().MouseDown[0] = true;
  SimulateFrame();
  ImGui::GetIO().MouseDown[0] = false;
  SimulateFrame();

  EXPECT_EQ(timeline_.selected_event_index(), -1);
  EXPECT_EQ(timeline_.selected_group_index(), 1);
  EXPECT_EQ(timeline_.selected_counter_index(), 0);

  // Step 3: Select Flame Event Again
  ImGui::GetIO().MousePos = ImVec2(300.f, 28.f);
  ImGui::GetIO().MouseDown[0] = true;
  SimulateFrame();

  EXPECT_EQ(timeline_.selected_event_index(), 0);
  EXPECT_EQ(timeline_.selected_group_index(), 0);
  EXPECT_EQ(timeline_.selected_counter_index(), -1);
}

TEST_F(RealTimelineImGuiFixture, ClickEmptyAreaClearsSelectionIndices) {
  FlameChartTimelineData data;
  data.groups.push_back(
      {.name = "Group 1", .start_level = 0, .nesting_level = 0});
  data.events_by_level.push_back({0});
  data.entry_names.push_back("event1");
  data.entry_levels.push_back(0);
  data.entry_start_times.push_back(0.0);
  data.entry_total_times.push_back(100.0);
  data.entry_pids.push_back(1);
  data.entry_args.push_back({});
  timeline_.set_timeline_data(std::move(data));
  timeline_.SetVisibleRange({0.0, 100.0});

  // Select event
  ImGui::GetIO().MousePos = ImVec2(300.f, 28.f);
  ImGui::GetIO().MouseDown[0] = true;
  SimulateFrame();
  ImGui::GetIO().MouseDown[0] = false;
  SimulateFrame();

  EXPECT_NE(timeline_.selected_event_index(), -1);

  // Click empty area
  ImGui::GetIO().MousePos = ImVec2(300.f, 100.f);
  ImGui::GetIO().MouseDown[0] = true;
  SimulateFrame();

  EXPECT_EQ(timeline_.selected_event_index(), -1);
  EXPECT_EQ(timeline_.selected_group_index(), -1);
  EXPECT_EQ(timeline_.selected_counter_index(), -1);
}

FlameChartTimelineData GetTestFlowData() {
  FlameChartTimelineData data;
  data.groups.push_back(
      {.name = "Group 1", .start_level = 0, .nesting_level = 0});
  data.events_by_level.push_back({0, 1});
  data.entry_names = {"event0", "event1"};
  data.entry_event_ids = {1000, 2000};
  data.entry_levels = {0, 0};
  data.entry_start_times = {10.0, 50.0};
  data.entry_total_times = {5.0, 5.0};
  FlowLine flow1 = {.source_ts = 12.0,
                    .target_ts = 52.0,
                    .source_level = 0,
                    .target_level = 0,
                    .color = 0xFFFF0000,
                    .category = tsl::profiler::ContextType::kGeneric};
  FlowLine flow2 = {.source_ts = 15.0,
                    .target_ts = 55.0,
                    .source_level = 0,
                    .target_level = 0,
                    .color = 0xFF00FF00,
                    .category = tsl::profiler::ContextType::kGpuLaunch};
  data.flow_lines = {flow1, flow2};
  data.flow_ids_by_event_id = {{1000, {"1"}}, {2000, {"2"}}};
  data.flow_lines_by_flow_id = {{"1", {flow1}}, {"2", {flow2}}};
  return data;
}

TEST_F(RealTimelineImGuiFixture, DrawFlowsWithNoFilter) {
  timeline_.set_timeline_data(GetTestFlowData());
  timeline_.SetVisibleRange({0.0, 100.0});
  timeline_.SetVisibleFlowCategory(-1);  // Show all flows

  ImGui::NewFrame();
  timeline_.Draw();
  ImDrawList* draw_list = ImGui::GetForegroundDrawList();
  EXPECT_FALSE(draw_list->VtxBuffer.empty());
  ImGui::EndFrame();
}

TEST_F(RealTimelineImGuiFixture, DrawFlowsWithNoneFilter) {
  timeline_.set_timeline_data(GetTestFlowData());
  timeline_.SetVisibleRange({0.0, 100.0});
  timeline_.SetVisibleFlowCategory(-2);  // Show no flows

  ImGui::NewFrame();
  timeline_.Draw();
  ImDrawList* draw_list = ImGui::GetForegroundDrawList();
  // VtxBuffer might contain things other than flows (e.g. selection),
  // so we check size before/after.
  // The fixture does not draw selection by default.
  // When no flows are drawn, cliprect commands may be issued, but no vertices
  // should be added to buffer.
  EXPECT_TRUE(draw_list->VtxBuffer.empty());
  ImGui::EndFrame();
}

TEST_F(RealTimelineImGuiFixture, DrawFlowsWithCategoryFilter) {
  timeline_.set_timeline_data(GetTestFlowData());
  timeline_.SetVisibleRange({0.0, 100.0});
  // kGpuLaunch is category 10.
  timeline_.SetVisibleFlowCategory(
      static_cast<int>(tsl::profiler::ContextType::kGpuLaunch));

  ImGui::NewFrame();
  timeline_.Draw();
  ImDrawList* draw_list = ImGui::GetForegroundDrawList();

  ASSERT_FALSE(draw_list->VtxBuffer.empty());

  // flow1 has color 0xFFFF0000 (Red). flow2 has color 0xFF00FF00 (Green).
  // Since we filter by kGpuLaunch, only flow2 should be drawn.
  bool found_flow1_color = false;
  bool found_flow2_color = false;
  for (const auto& vtx : draw_list->VtxBuffer) {
    if (vtx.col == 0xFFFF0000) found_flow1_color = true;
    if (vtx.col == 0xFF00FF00) found_flow2_color = true;
  }
  EXPECT_FALSE(found_flow1_color);
  EXPECT_TRUE(found_flow2_color);

  ImGui::EndFrame();
}

TEST_F(RealTimelineImGuiFixture, DrawFlowsForSelectedEvent) {
  timeline_.set_timeline_data(GetTestFlowData());
  timeline_.SetVisibleRange({0.0, 100.0});
  timeline_.SetVisibleFlowCategory(-1);  // Show all flows if no event selected

  // Select event 0 (id 1000), which is part of flow "1" (flow1).
  timeline_.NavigateToEvent(0);

  ImGui::NewFrame();
  timeline_.Draw();
  ImDrawList* draw_list = ImGui::GetForegroundDrawList();

  ASSERT_FALSE(draw_list->VtxBuffer.empty());

  // Event 0 (id 1000) is associated with flow1 (Red).
  // We should see flow1's color and not flow2's color.
  bool found_flow1_color = false;
  bool found_flow2_color = false;
  for (const auto& vtx : draw_list->VtxBuffer) {
    if (vtx.col == 0xFFFF0000) found_flow1_color = true;
    if (vtx.col == 0xFF00FF00) found_flow2_color = true;
  }
  EXPECT_TRUE(found_flow1_color);
  EXPECT_FALSE(found_flow2_color);

  ImGui::EndFrame();
}

using TimelineImGuiFixture = TimelineImGuiTestFixture<Timeline>;

TEST_F(TimelineImGuiFixture, LevelYPositionsCalculation) {
  FlameChartTimelineData data;
  data.groups.push_back(
      {.name = "Group 0", .start_level = 0, .nesting_level = 0});
  data.groups.push_back(
      {.name = "Group 1", .start_level = 2, .nesting_level = 0});
  data.groups.push_back(
      {.name = "Group 2", .start_level = 3, .nesting_level = 0});
  // Level 0, 1 in Group 0
  // Level 2 in Group 1
  // Level 3, 4, 5 in Group 2
  data.events_by_level.resize(6);
  timeline_.set_timeline_data(std::move(data));

  SimulateFrame();

  const auto& y_positions = timeline_.GetLevelYPositions();
  EXPECT_EQ(y_positions.size(), 6);

  const float level_height = kEventHeight + kEventPaddingBottom;

  // We need to get the initial cursor screen pos Y to verify the absolute
  // positions. This is tricky as it depends on the ImGui window state. Let's
  // assume it's 0 for the first group's content for now and adjust if needed.
  // A more robust way would be to mock ImGui::GetCursorScreenPos(), but that's
  // not easily done without changing the Timeline class interface.

  // Instead of absolute y, we can check the relative y positions within a
  // group. However, the level_y_positions_ are absolute screen coordinates.

  // Let's run the test once to see what the actual y_positions[0] is, assuming
  // the window starts at some Y. All other positions will be relative to that.

  // For now, let's just check the difference between levels in the same group.
  if (y_positions.size() >= 2) {
    EXPECT_FLOAT_EQ(y_positions[1] - y_positions[0], level_height);
  }
  if (y_positions.size() >= 6) {
    EXPECT_FLOAT_EQ(y_positions[4] - y_positions[3], level_height);
    EXPECT_FLOAT_EQ(y_positions[5] - y_positions[4], level_height);
  }

  // The difference between the start of Group 1 (level 2) and Group 0 (level 1)
  // will depend on the height of Group 0, which is 2 * level_height.
  if (y_positions.size() >= 3) {
    // This check is not straight forward because of the child windows.
  }

  // Group 0: Levels 0, 1
  const float group0_base_y = y_positions[0];
  EXPECT_FLOAT_EQ(y_positions[1] - group0_base_y, level_height);

  // Group 1: Level 2
  // No relative checks needed within Group 1 as it has only one level.

  // Group 2: Levels 3, 4, 5
  const float group2_base_y = y_positions[3];
  EXPECT_FLOAT_EQ(y_positions[4] - group2_base_y, level_height);
  EXPECT_FLOAT_EQ(y_positions[5] - group2_base_y, 2 * level_height);
}

TEST(TimelineTest, BezierControlPointCalculation) {
  ImVec2 cp0, cp1;

  // start_x < end_x
  Timeline::CalculateBezierControlPoints(100.0f, 50.0f, 200.0f, 50.0f, cp0,
                                         cp1);
  EXPECT_FLOAT_EQ(cp0.x, 150.0f);  // 100 + (200 - 100) * 0.5
  EXPECT_FLOAT_EQ(cp0.y, 50.0f);
  EXPECT_FLOAT_EQ(cp1.x, 150.0f);  // 200 - (200 - 100) * 0.5
  EXPECT_FLOAT_EQ(cp1.y, 50.0f);

  // start_x > end_x
  Timeline::CalculateBezierControlPoints(200.0f, 50.0f, 100.0f, 50.0f, cp0,
                                         cp1);
  EXPECT_FLOAT_EQ(cp0.x, 250.0f);  // 200 + abs(100 - 200) * 0.5
  EXPECT_FLOAT_EQ(cp0.y, 50.0f);
  EXPECT_FLOAT_EQ(cp1.x, 50.0f);  // 100 - abs(100 - 200) * 0.5
  EXPECT_FLOAT_EQ(cp1.y, 50.0f);

  // start_x == end_x
  Timeline::CalculateBezierControlPoints(100.0f, 50.0f, 100.0f, 150.0f, cp0,
                                         cp1);
  EXPECT_FLOAT_EQ(cp0.x, 100.0f);
  EXPECT_FLOAT_EQ(cp0.y, 50.0f);
  EXPECT_FLOAT_EQ(cp1.x, 100.0f);
  EXPECT_FLOAT_EQ(cp1.y, 150.0f);
}

TEST_F(RealTimelineImGuiFixture, SelectionOverlayIsDrawnOnTopOfTracks) {
  // Ensure we have some data so tracks are drawn.
  FlameChartTimelineData data;
  data.groups.push_back({.name = "Group 1", .start_level = 0});
  timeline_.set_timeline_data(std::move(data));

  ImGui::NewFrame();
  timeline_.Draw();

  ImGuiWindow* timeline_window = ImGui::FindWindowByName("Timeline viewer");
  ASSERT_NE(timeline_window, nullptr);

  bool found_tracks = false;
  bool found_overlay = false;
  bool overlay_is_after_tracks = false;

  for (ImGuiWindow* child : timeline_window->DC.ChildWindows) {
    if (absl::StrContains(child->Name, "Tracks")) {
      found_tracks = true;
    } else if (absl::StrContains(child->Name, "SelectionOverlay")) {
      found_overlay = true;
      if (found_tracks) {
        overlay_is_after_tracks = true;
      }
    }
  }

  EXPECT_TRUE(found_tracks) << "Tracks child window not found";
  EXPECT_TRUE(found_overlay) << "SelectionOverlay child window not found";
  EXPECT_TRUE(overlay_is_after_tracks)
      << "SelectionOverlay should be drawn after Tracks to appear on top";

  ImGui::EndFrame();
}

}  // namespace
}  // namespace testing
}  // namespace traceviewer
