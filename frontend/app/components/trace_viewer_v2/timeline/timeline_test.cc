#include "xprof/frontend/app/components/trace_viewer_v2/timeline/timeline.h"

#include <any>
#include <functional>
#include <string>
#include <utility>
#include <vector>

#include "testing/base/public/gmock.h"
#include "<gtest/gtest.h>"
#include "absl/strings/match.h"
#include "absl/strings/string_view.h"
#include "third_party/dear_imgui/imgui.h"
#include "third_party/dear_imgui/imgui_internal.h"
#include "xprof/frontend/app/components/trace_viewer_v2/animation.h"
#include "xprof/frontend/app/components/trace_viewer_v2/event_data.h"
#include "xprof/frontend/app/components/trace_viewer_v2/timeline/constants.h"
#include "xprof/frontend/app/components/trace_viewer_v2/timeline/time_range.h"

namespace traceviewer {
namespace testing {
namespace {

using ::testing::_;
using ::testing::ElementsAre;
using ::testing::FloatEq;
using ::testing::Return;
using ::testing::Test;

// Mock class for Timeline to mock virtual methods.
class MockTimeline : public Timeline {
 public:
  MOCK_METHOD(ImVec2, GetTextSize, (absl::string_view text), (const, override));
  MOCK_METHOD(void, Pan, (Pixel pixel_amount), (override));
  MOCK_METHOD(void, Zoom, (float zoom_factor), (override));
  MOCK_METHOD(void, Scroll, (Pixel pixel_amount), (override));
};

TEST(TimelineTest, SetTimelineData) {
  Timeline timeline;
  FlameChartTimelineData data;
  data.groups.push_back({.name = "Group 1", .start_level = 0});
  data.entry_levels.push_back(0);
  data.entry_start_times.push_back(10.0);
  data.entry_total_times.push_back(5.0);

  timeline.SetTimelineData(std::move(data));

  EXPECT_THAT(timeline.timeline_data().entry_levels, ElementsAre(0));
  EXPECT_THAT(timeline.timeline_data().entry_start_times, ElementsAre(10.0));
  EXPECT_THAT(timeline.timeline_data().entry_total_times, ElementsAre(5.0));
}

TEST(TimelineTest, CalculateGroupOffsets) {
  Timeline timeline;
  FlameChartTimelineData data;

  // Group 0: Flame chart with 2 levels (0, 1).
  // Height = 2 * (kEventHeight + kEventPaddingBottom) = 2 * 17.0 = 34.0.
  data.groups.push_back(
      {.type = Group::Type::kFlame, .name = "Group 0", .start_level = 0});

  // Group 1: Counter track.
  // Height = kCounterTrackHeight = 40.0.
  data.groups.push_back(
      {.type = Group::Type::kCounter, .name = "Group 1", .start_level = 2});

  // Group 2: Empty flame chart (0 levels).
  // Height = 1 * (kEventHeight + kEventPaddingBottom) = 17.0 (min height).
  data.groups.push_back(
      {.type = Group::Type::kFlame, .name = "Group 2", .start_level = 3});

  // Simulate 3 levels in total (0, 1, 2). Group 2 starts at 3, so it's empty.
  data.events_by_level.resize(3);

  timeline.SetTimelineData(std::move(data));

  const std::vector<float>& offsets = timeline.group_offsets();

  // We expect 4 offsets: 0, height(G0), height(G0)+height(G1), total height.
  ASSERT_THAT(offsets, ::testing::SizeIs(4));

  EXPECT_THAT(offsets[0], FloatEq(0.0f));
  EXPECT_THAT(offsets[1], FloatEq(34.0f));
  EXPECT_THAT(offsets[2], FloatEq(34.0f + 40.0f));  // 74.0f
  EXPECT_THAT(offsets[3], FloatEq(74.0f + 17.0f));  // 91.0f
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
  // The start 10.0 - kMinDur/2 is less than data_time_range.start(),
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
  // The end 100.0 + kMinDur/2 is greater than data_time_range.end(),
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
  timeline.SetTimelineData(std::move(data));
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
  timeline.SetTimelineData(std::move(data));
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
  timeline.SetTimelineData(std::move(data));
  TimeRange initial_range(0.0, 50.0);
  timeline.SetVisibleRange(initial_range);

  timeline.NavigateToEvent(1);

  // Visible range should not change because event index is out of bounds.
  EXPECT_EQ(timeline.visible_range().start(), initial_range.start());
  EXPECT_EQ(timeline.visible_range().end(), initial_range.end());
}

// Specialized class for testing internal behavior and state.
class TestableTimeline : public Timeline {
 public:
  // Expose protected methods for testing.
  using Timeline::GetVisibleGroupRange;

  // Intercept CalculateGroupOffsets to verify internal state during updates.
  std::vector<float> CalculateGroupOffsets(
      const FlameChartTimelineData& data) const override {
    if (on_calculate_group_offsets) {
      on_calculate_group_offsets();
    }
    return Timeline::CalculateGroupOffsets(data);
  }

  std::function<void()> on_calculate_group_offsets;
};

TEST(TimelineTest, DataRemainsOldDuringCalculation) {
  TestableTimeline timeline;

  // 1. Set initial data
  FlameChartTimelineData old_data;
  old_data.groups.push_back({.name = "Old Group", .start_level = 0});
  timeline.SetTimelineData(old_data);  // Initial set

  ASSERT_EQ(timeline.timeline_data().groups.size(), 1);
  ASSERT_EQ(timeline.timeline_data().groups[0].name, "Old Group");

  // 2. Prepare new data
  FlameChartTimelineData new_data;
  new_data.groups.push_back({.name = "New Group", .start_level = 0});

  // 3. Setup verification hook
  bool checked = false;
  timeline.on_calculate_group_offsets = [&]() {
    checked = true;
    // CRITICAL CHECK: The timeline should still hold the old data
    EXPECT_EQ(timeline.timeline_data().groups.size(), 1);
    EXPECT_EQ(timeline.timeline_data().groups[0].name, "Old Group");

    // And obviously not the new data (though size=1 check above is weak if both
    // are size 1)
    EXPECT_NE(timeline.timeline_data().groups[0].name, "New Group");
  };

  // 4. Trigger update
  timeline.SetTimelineData(std::move(new_data));

  EXPECT_TRUE(checked);

  // 5. Verify final state
  EXPECT_EQ(timeline.timeline_data().groups.size(), 1);
  EXPECT_EQ(timeline.timeline_data().groups[0].name, "New Group");
}

class GetVisibleGroupRangeTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Setup some dummy data with predictable offsets.
    // 10 groups, each height 10.0 (simulated).
    // group_offsets_ will be {0, 10, 20, ..., 100}
    FlameChartTimelineData data;
    for (int i = 0; i < 10; ++i) {
      data.groups.push_back({.name = "Group " + std::to_string(i)});
    }
    // We can't control CalculateGroupOffsets easily to produce exact numbers
    // without mocking or setting up complex events.
    // Instead, we rely on setting up data such that CalculateGroupOffsets
    // produces known values.
    // Let's assume kCounterTrackHeight is 40.0f.
    // We'll use counter tracks for simplicity.
    data.groups.clear();
    for (int i = 0; i < 10; ++i) {
      data.groups.push_back({.type = Group::Type::kCounter,
                             .name = "Group " + std::to_string(i),
                             .start_level = 0});
    }
    // Height of each group will be kCounterTrackHeight (40.0f).
    // Offsets: 0, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400.
    timeline_.SetTimelineData(std::move(data));
  }

  TestableTimeline timeline_;
  const float kRowHeight = kCounterTrackHeight;  // 40.0f
};

TEST_F(GetVisibleGroupRangeTest, EmptyData) {
  TestableTimeline empty_timeline;
  empty_timeline.SetTimelineData({});
  auto [start, end] = empty_timeline.GetVisibleGroupRange(0.0f, 100.0f);

  // Implementation returns {0, -1} for empty offsets.
  EXPECT_EQ(start, 0);
  EXPECT_EQ(end, -1);
}

TEST_F(GetVisibleGroupRangeTest, ScrollAtTop_VisibleOneRow) {
  // Scroll 0, Visible 40 (1 row).
  // Visible: Group 0 [0, 40).
  // Start: 0.
  // End: 0.
  // Expected: [0, 0].
  auto [start, end] = timeline_.GetVisibleGroupRange(0.0f, kRowHeight);

  EXPECT_EQ(start, 0);
  EXPECT_EQ(end, 0);
}

TEST_F(GetVisibleGroupRangeTest, ScrollAtTop_VisibleTwoRows) {
  // Scroll 0, Visible 80 (2 rows).
  // Visible: Group 0 [0, 40), Group 1 [40, 80).
  // Start: 0.
  // End: 1.
  auto [start, end] = timeline_.GetVisibleGroupRange(0.0f, kRowHeight * 2);

  EXPECT_EQ(start, 0);
  EXPECT_EQ(end, 1);
}

TEST_F(GetVisibleGroupRangeTest, ScrollInMiddle_ExactAlign) {
  // Scroll 40 (start of Group 1), Visible 40.
  // Visible: Group 1 [40, 80).
  // Start: 1.
  // End: 1.
  auto [start, end] = timeline_.GetVisibleGroupRange(kRowHeight, kRowHeight);

  EXPECT_EQ(start, 1);
  EXPECT_EQ(end, 1);
}

TEST_F(GetVisibleGroupRangeTest, ScrollInMiddle_PartialAlign) {
  // Scroll 60 (middle of Group 1), Visible 40.
  // Visible Range: [60, 100).
  // Covers Group 1 [40, 80) and Group 2 [80, 120).
  // Start: Group 1.
  // End: Group 2.
  auto [start, end] =
      timeline_.GetVisibleGroupRange(kRowHeight * 1.5f, kRowHeight);

  EXPECT_EQ(start, 1);
  EXPECT_EQ(end, 2);
}

TEST_F(GetVisibleGroupRangeTest, ScrollAtBottom) {
  // Scroll 360 (start of Group 9, last one), Visible 40.
  // Visible: Group 9 [360, 400).
  // Start: 9.
  // End: 9.
  auto [start, end] =
      timeline_.GetVisibleGroupRange(kRowHeight * 9, kRowHeight);

  EXPECT_EQ(start, 9);
  EXPECT_EQ(end, 9);
}

TEST_F(GetVisibleGroupRangeTest, ScrollPastBottom) {
  // Scroll 500 (beyond 400), Visible 100.
  // Visible Range: [500, 600).
  // Start: > 400. upper_bound returns end(), distance is 11. start_index 10.
  // End: > 400. lower_bound returns end(), distance 11. end_index 10.
  // Clamped to 9.
  // Result: [10, 9]. (Empty range because start > end)
  auto [start, end] = timeline_.GetVisibleGroupRange(500.0f, 100.0f);

  EXPECT_EQ(start, 10);
  EXPECT_EQ(end, 9);
}

TEST_F(GetVisibleGroupRangeTest, VisibleHeightLargerThanContent) {
  // Scroll 0, Visible 1000.
  // Visible Range [0, 1000).
  // Start: 0.
  // End: 9 (clamped).
  auto [start, end] = timeline_.GetVisibleGroupRange(0.0f, 1000.0f);

  EXPECT_EQ(start, 0);
  EXPECT_EQ(end, 9);
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
    timeline_.SetTimelineData(
        {{},
         {},
         {},
         {},
         {{.name = "group", .start_level = 0, .nesting_level = 0}}});
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
              Zoom(FloatEq(1.0f - kZoomSpeed * ImGui::GetIO().DeltaTime)));

  SimulateFrame();
}

TEST_F(MockTimelineImGuiFixture, ZoomOutWithSKey) {
  ImGui::GetIO().AddKeyEvent(ImGuiKey_S, true);

  // No acceleration is applied because io.DeltaTime (0.1f) <
  // kAccelerateThreshold (0.25f).
  EXPECT_CALL(timeline_,
              Zoom(FloatEq(1.0f + kZoomSpeed * ImGui::GetIO().DeltaTime)));

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
      Zoom(FloatEq(1.0f - kZoomSpeed * ImGui::GetIO().DeltaTime * 5.0f)));

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
  ImGuiIO& io = ImGui::GetIO();
  io.AddMouseWheelEvent(0.0f, 1.0f);
  io.AddKeyEvent(ImGuiMod_Ctrl, true);

  const float expected_zoom_factor = 1.0f + 1.0f * kMouseWheelZoomSpeed;

  EXPECT_CALL(timeline_, Zoom(FloatEq(expected_zoom_factor)));

  SimulateFrame();
}

TEST_F(MockTimelineImGuiFixture, ZoomInWithMouseWheelAndCtrlKey) {
  ImGuiIO& io = ImGui::GetIO();
  io.AddMouseWheelEvent(0.0f, -1.0f);
  io.AddKeyEvent(ImGuiMod_Ctrl, true);

  const float expected_zoom_factor = 1.0f + (-1.0f) * kMouseWheelZoomSpeed;

  EXPECT_CALL(timeline_, Zoom(FloatEq(expected_zoom_factor)));

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
  timeline_.SetTimelineData(std::move(data));
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
  timeline_.SetTimelineData(std::move(data));
  timeline_.SetVisibleRange({0.0, 100.0});

  // The event rect width will be around 4.51f, which is < kMinTextWidth (5.0f).
  // DrawEventName should not draw text, so GetTextForDisplay and
  // CalculateEventTextRect won't be called, and GetTextSize should not be
  // called.
  EXPECT_CALL(timeline_, GetTextSize(_)).Times(0);

  SimulateFrame();
}

using RealTimelineImGuiFixture = TimelineImGuiTestFixture<Timeline>;

TEST_F(RealTimelineImGuiFixture, ClickEventSelectsEvent) {
  FlameChartTimelineData data;

  data.groups.push_back(
      {.name = "Group 1", .start_level = 0, .nesting_level = 0});
  data.events_by_level.push_back({0});
  data.entry_names.push_back("event1");
  data.entry_levels.push_back(0);
  data.entry_start_times.push_back(0.0);
  data.entry_total_times.push_back(100.0);
  timeline_.SetTimelineData(std::move(data));
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
  ImGui::GetIO().MousePos = ImVec2(300.f, 30.f);
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
  timeline_.SetTimelineData(std::move(data));
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
  timeline_.SetTimelineData(std::move(data));
  timeline_.SetVisibleRange({0.0, 100.0});

  int callback_count = 0;
  timeline_.set_event_callback(
      [&](absl::string_view type, const EventData& detail) {
        callback_count++;
      });

  ImGui::GetIO().MousePos = ImVec2(300.f, 30.f);

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
  timeline_.SetTimelineData(std::move(data));
  timeline_.SetVisibleRange({0.0, 100.0});

  // First, select an event.
  ImGui::GetIO().MousePos = ImVec2(300.f, 30.f);  // A position over the event.
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
  timeline_.SetTimelineData(std::move(data));
  timeline_.SetVisibleRange({0.0, 100.0});

  // First, select an event.
  ImGui::GetIO().MousePos = ImVec2(300.f, 30.f);  // A position over the event.
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
  timeline_.SetTimelineData(std::move(data));
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

// This is a test for making sure the wasm is not crashing when timeline data
// is empty.
TEST_F(RealTimelineImGuiFixture, DrawsTimelineWindowWhenTimelineDataIsEmpty) {
  timeline_.SetTimelineData({});

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
  timeline_.SetTimelineData(std::move(data));
  timeline_.SetVisibleRange({0.0, 100.0});

  // Mouse is over the event
  ImGui::GetIO().MousePos = ImVec2(500.f, 30.f);
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
  timeline_.SetTimelineData(std::move(data));
  timeline_.SetVisibleRange({0.0, 100.0});

  ImGui::GetIO().AddKeyEvent(ImGuiMod_Shift, true);

  // First shift-click on event 1.
  ImGui::GetIO().MousePos = ImVec2(500.f, 30.f);  // Position over event 1.
  ImGui::GetIO().MouseDown[0] = true;
  SimulateFrame();

  ASSERT_EQ(timeline_.selected_time_ranges().size(), 1);
  EXPECT_EQ(timeline_.selected_time_ranges()[0].start(), 10.0);
  EXPECT_EQ(timeline_.selected_time_ranges()[0].end(), 30.0);

  // Frame with mouse up.
  ImGui::GetIO().MouseDown[0] = false;
  SimulateFrame();

  // Second shift-click on event 2.
  ImGui::GetIO().MousePos = ImVec2(1100.f, 30.f);  // Position over event 2.
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
  ImGui::GetIO().MousePos = ImVec2(500.f, 30.f);  // Position over event 1.
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
    // (based on 1920px window width and no paddings), a duration of 166.9 gives
    // 10px per microsecond.
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

  timeline_.SetTimelineData(std::move(data));
  timeline_.SetVisibleRange({0.0, 100.0});

  ImGui::NewFrame();
  timeline_.Draw();

  ImGuiWindow* counter_window = nullptr;
  // The child window name is constructed as
  // "TimelineChild_<group_name>_<group_index>". We search for a window that
  // contains this string in its name.
  const std::string child_id = "TimelineChild_Counter Group_0";
  for (ImGuiWindow* w : ImGui::GetCurrentContext()->Windows) {
    if (absl::StrContains(w->Name, child_id)) {
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

  timeline_.SetTimelineData(std::move(data));
  timeline_.SetVisibleRange({0.0, 100.0});

  // Render first frame to layout windows and find the counter track location.
  ImGui::NewFrame();
  timeline_.Draw();

  ImGuiWindow* counter_window = nullptr;
  const std::string child_id = "TimelineChild_Counter Group_0";
  for (ImGuiWindow* w : ImGui::GetCurrentContext()->Windows) {
    if (absl::StrContains(w->Name, child_id)) {
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
    if (absl::StrContains(w->Name, child_id)) {
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
  timeline_.SetTimelineData(std::move(data));
  timeline_.SetVisibleRange({0.0, 100.0});

  // Set a mouse position that is guaranteed to be over the event.
  ImGui::GetIO().MousePos = ImVec2(300.f, 30.f);
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

  timeline_.SetTimelineData(std::move(data));
  timeline_.SetVisibleRange({0.0, 100.0});

  ImGui::NewFrame();
  timeline_.Draw();

  ImGuiWindow* counter_window = nullptr;
  const std::string child_id = "TimelineChild_Counter Group_0";
  for (ImGuiWindow* w : ImGui::GetCurrentContext()->Windows) {
    if (absl::StrContains(w->Name, child_id)) {
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

  timeline_.SetTimelineData(std::move(data));
  timeline_.SetVisibleRange({0.0, 100.0});

  // Step 1: Select Flame Event
  ImGui::GetIO().MousePos = ImVec2(300.f, 30.f);  // Over flame event
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
    if (absl::StrContains(w->Name, child_id)) {
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
  ImGui::GetIO().MousePos = ImVec2(300.f, 30.f);
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
  timeline_.SetTimelineData(std::move(data));
  timeline_.SetVisibleRange({0.0, 100.0});

  // Select event
  ImGui::GetIO().MousePos = ImVec2(300.f, 30.f);
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

TEST_F(RealTimelineImGuiFixture, DrawsOnlyVisibleGroups) {
  FlameChartTimelineData data;
  // Create 100 groups.
  for (int i = 0; i < 100; ++i) {
    data.groups.push_back(
        {.name = "Group " + std::to_string(i), .start_level = 0});
  }
  // Ensure we have enough levels so DrawGroup doesn't crash or skip?
  // Timeline::CalculateGroupOffsets ensures min height of 1 level (17px).
  // So empty groups have height 17px.

  timeline_.SetTimelineData(std::move(data));
  // Total height ~ 1700px.
  // Viewport ~1080px.

  ImGui::NewFrame();
  timeline_.Draw();

  int child_window_count = 0;
  for (ImGuiWindow* w : ImGui::GetCurrentContext()->Windows) {
    if (absl::StrContains(std::string(w->Name), "TimelineChild_")) {
      child_window_count++;
    }
  }

  // Should verify culling.
  // If visible height is 1080 (minus ruler ~20, minus labels...), say 1000.
  // 1000/17 ~ 58.
  // Plus buffer (1 before, 1 after). ~60.
  // Definitely < 100.
  EXPECT_LT(child_window_count, 90);
  // And > 0.
  EXPECT_GT(child_window_count, 10);

  ImGui::EndFrame();
}

TEST_F(RealTimelineImGuiFixture, ContentHeightMatchesTotalGroupHeight) {
  FlameChartTimelineData data;
  // Add enough groups to have some height.
  // 10 groups of height 40 (Counter). Total 400.
  for (int i = 0; i < 10; ++i) {
    data.groups.push_back({.type = Group::Type::kCounter,
                           .name = "Group " + std::to_string(i),
                           .start_level = 0});
  }
  timeline_.SetTimelineData(std::move(data));
  // Total height should be 400.0f.

  ImGui::NewFrame();
  timeline_.Draw();

  // Find the "Tracks" child window.
  ImGuiWindow* tracks_window = nullptr;
  for (ImGuiWindow* w : ImGui::GetCurrentContext()->Windows) {
    if (absl::StrContains(w->Name, "Tracks") &&
        !absl::StrContains(w->Name, "TimelineChild")) {
      tracks_window = w;
      break;
    }
  }
  ASSERT_NE(tracks_window, nullptr);

  // The expected total height is the last offset.
  const float expected_height = timeline_.group_offsets().back();

  // ContentSize is updated at the end of the frame/child. Since we are
  // calling this after Draw() (which calls EndChild()), ContentSize should be
  // valid. However, if it's 0, it might be because the table handling defers
  // size report.
  // Instead, let's look at the cursor position which tracks the layout.
  // DC.CursorMaxPos tracks the maximum position reached by the cursor (content
  // size).
  float content_height =
      tracks_window->DC.CursorMaxPos.y - tracks_window->DC.CursorStartPos.y;

  // We expect it to match exactly.
  EXPECT_FLOAT_EQ(content_height, expected_height);

  ImGui::EndFrame();
}

TEST_F(RealTimelineImGuiFixture,
       DrawGroupChildWindowHeightMatchesOffsetDifference) {
  FlameChartTimelineData data;
  data.groups.push_back(
      {.type = Group::Type::kCounter, .name = "TestGroup", .start_level = 0});
  timeline_.SetTimelineData(std::move(data));

  ImGui::NewFrame();
  timeline_.Draw();

  ImGuiWindow* group_window = nullptr;
  for (ImGuiWindow* w : ImGui::GetCurrentContext()->Windows) {
    if (absl::StrContains(w->Name, "TimelineChild_TestGroup_0")) {
      group_window = w;
      break;
    }
  }
  ASSERT_NE(group_window, nullptr);

  const float expected_height =
      timeline_.group_offsets()[1] - timeline_.group_offsets()[0];
  EXPECT_FLOAT_EQ(group_window->Size.y, expected_height);

  ImGui::EndFrame();
}

TEST_F(RealTimelineImGuiFixture, DrawsOnlyVisibleEvents) {
  FlameChartTimelineData data;
  data.groups.push_back({.name = "Group 1", .start_level = 0});
  // Level 0
  data.events_by_level.resize(1);
  std::vector<int>& events = data.events_by_level[0];

  // Create 100 events, each 10us long, spaced 20us start-to-start.
  // Events at: 0, 20, 40, 60, ..., 1980.
  // End times: 10, 30, 50, 70, ..., 1990.
  for (int i = 0; i < 100; ++i) {
    data.entry_names.push_back("event");
    data.entry_levels.push_back(0);
    data.entry_start_times.push_back(i * 20.0);
    data.entry_total_times.push_back(10.0);
    events.push_back(i);
  }

  timeline_.SetTimelineData(std::move(data));
  timeline_.set_data_time_range({0.0, 2000.0});

  // Visible range: 35.0 to 75.0.
  // Event 0: 0-10. Outside.
  // Event 1: 20-30. Outside.
  // Event 2: 40-50. Inside.
  // Event 3: 60-70. Inside.
  // Event 4: 80-90. Outside.
  // So 2 events should be drawn.
  timeline_.SetVisibleRange({35.0, 75.0});

  ImGui::NewFrame();
  timeline_.Draw();

  // Find the child window.
  ImGuiWindow* group_window = nullptr;
  // Group index 0.
  const std::string child_id = "TimelineChild_Group 1_0";
  for (ImGuiWindow* w : ImGui::GetCurrentContext()->Windows) {
    if (absl::StrContains(w->Name, child_id)) {
      group_window = w;
      break;
    }
  }
  ASSERT_NE(group_window, nullptr);

  // Check vertices count.
  // 2 events. Each is a rect (4 vertices).
  // Total ~8 vertices.
  // If all 100 events were drawn, we'd have ~400 vertices.
  // Allow some overhead for text (though "event" is short).
  // Let's expect < 100 vertices.
  EXPECT_LT(group_window->DrawList->VtxBuffer.Size, 100);
  EXPECT_GT(group_window->DrawList->VtxBuffer.Size, 0);

  ImGui::EndFrame();
}

}  // namespace
}  // namespace testing
}  // namespace traceviewer
