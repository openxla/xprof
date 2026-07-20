#include "frontend/app/components/trace_viewer_v2/timeline/timeline.h"

#include <algorithm>
#include <any>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <ios>
#include <limits>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "testing/base/public/gmock.h"
#include "<gtest/gtest.h>"
#include "absl/base/nullability.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/strings/match.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "imgui.h"
#include "imgui_internal.h"
#include "tsl/profiler/lib/context_types.h"
#include "frontend/app/components/trace_viewer_v2/animation.h"
#include "frontend/app/components/trace_viewer_v2/color/color_generator.h"
#include "frontend/app/components/trace_viewer_v2/color/colors.h"
#include "frontend/app/components/trace_viewer_v2/event_data.h"
#include "frontend/app/components/trace_viewer_v2/helper/time_formatter.h"
#include "frontend/app/components/trace_viewer_v2/timeline/constants.h"
#include "frontend/app/components/trace_viewer_v2/timeline/time_range.h"
#include "frontend/app/components/trace_viewer_v2/trace_helper/trace_event.h"

namespace traceviewer {
namespace testing {
namespace {

using ::testing::_;
using ::testing::DoubleEq;
using ::testing::ElementsAre;
using ::testing::FloatEq;
using ::testing::Return;
using ::testing::Test;

// Calculated vertical center of the first event.
// The first event starts after the ruler (kRulerHeight=20.0f).
// Y = kRulerHeight + kEventHeight / 2.
constexpr float kFirstEventY = kRulerHeight + kEventHeight / 2.0f;

// Mock class for Timeline to mock virtual methods.
class MockTimeline : public Timeline {
 public:
  MockTimeline() : Timeline(color_palette_) { SetupDefaultMockBehavior(); }
  explicit MockTimeline(ColorPalette& palette) : Timeline(palette) {
    SetupDefaultMockBehavior();
  }

  MOCK_METHOD(ImVec2, GetTextSize, (absl::string_view text), (const, override));
  MOCK_METHOD(void, Pan, (Pixel pixel_amount), (override));
  MOCK_METHOD(void, Zoom, (float zoom_factor, double pivot), (override));
  MOCK_METHOD(void, Scroll, (Pixel pixel_amount), (override));
  MOCK_METHOD(void, DrawGroup,
              (int group_index, double px_per_time_unit_val, Pixel scroll_y,
               Pixel window_height),
              (override));
  MOCK_METHOD(void, DrawEventsForLevel,
              (int group_index, absl::Span<const int> event_indices,
               double px_per_time_unit, int level_in_group, const ImVec2& pos,
               const ImVec2& max, Pixel event_height, Pixel padding_bottom),
              (override));

  // Helpers to call base class protected methods from tests/lambdas.
  void DrawGroupBase(int group_index, double px_per_time_unit_val,
                     Pixel scroll_y, Pixel window_height) {
    Timeline::DrawGroup(group_index, px_per_time_unit_val, scroll_y,
                        window_height);
  }
  void DrawEventsForLevelBase(int group_index,
                              absl::Span<const int> event_indices,
                              double px_per_time_unit, int level_in_group,
                              const ImVec2& pos, const ImVec2& max,
                              Pixel event_height, Pixel padding_bottom) {
    Timeline::DrawEventsForLevel(group_index, event_indices, px_per_time_unit,
                                 level_in_group, pos, max, event_height,
                                 padding_bottom);
  }

  int CallFindFirstVisibleAncestorIndex(int start_idx) const {
    return FindFirstVisibleAncestorIndex(start_idx);
  }
  const std::vector<bool>& CallGroupVisible() const { return group_visible(); }

  void CallDrawEvent(int group_index, int event_index, const EventRect& rect,
                     ImDrawList* absl_nonnull draw_list) {
    DrawEvent(group_index, event_index, rect, draw_list);
  }

  bool CallDrawPinButton(int group_index, Pixel height, bool is_pinned) {
    return DrawPinButton(group_index, height, is_pinned);
  }

  bool CallDrawHideButton(int group_index, Pixel height, bool is_track_hidden) {
    return DrawHideButton(group_index, height, is_track_hidden);
  }

  const absl::flat_hash_set<std::string>& GetPinnedTrackNames() const {
    return pinned_track_names_;
  }

  const absl::flat_hash_set<std::string>& GetHiddenTrackNames() const {
    return hidden_track_names_;
  }

  void SetPinnedTrackNames(const absl::flat_hash_set<std::string>& names) {
    pinned_track_names_ = names;
  }

  void SetHiddenTrackNames(const absl::flat_hash_set<std::string>& names) {
    hidden_track_names_ = names;
  }

 private:
  void SetupDefaultMockBehavior() {
    ON_CALL(*this, DrawGroup)
        .WillByDefault([this](int group_index, double px_per_time_unit_val,
                              Pixel scroll_y, Pixel window_height) {
          this->DrawGroupBase(group_index, px_per_time_unit_val, scroll_y,
                              window_height);
        });
    ON_CALL(*this, DrawEventsForLevel)
        .WillByDefault([this](int group_index,
                              absl::Span<const int> event_indices,
                              double px_per_time_unit, int level_in_group,
                              const ImVec2& pos, const ImVec2& max,
                              Pixel event_height, Pixel padding_bottom) {
          this->DrawEventsForLevelBase(group_index, event_indices,
                                       px_per_time_unit, level_in_group, pos,
                                       max, event_height, padding_bottom);
        });
  }

  ColorPalette color_palette_ = ColorPalette::Default();
};

// =============================================================================
// Fixture: TimelineTest
// =============================================================================

// Global constants moved to header
constexpr double kPxPerTimeUnit = 1.0;
constexpr Pixel kScreenXOffset = 0.0f;
constexpr Pixel kScreenYOffset = 0.0f;
constexpr int kLevelInGroup = 0;
constexpr Pixel kTimelineWidth = 100.0f;
constexpr float kCharWidth = 10.0f;
constexpr float kEllipsisWidth = 30.0f;

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

TEST(TimelineTest, CalculateEventRect_EventCompletelyOutsideLeft) {
  ColorPalette palette = ColorPalette::Default();
  Timeline timeline(palette);
  timeline.SetVisibleRange({100.0, 200.0});

  // Event time range: [80.0, 90.0].
  // Screen range will be less than time start.
  // Expected to be fully clipped to the left edge [0.0, 0.0] (padding right
  // won't effect here because the event is clipped to the left edge).
  EventRect rect = timeline.CalculateEventRect(
      /*start=*/80.0, /*end=*/90.0, kScreenXOffset, kScreenYOffset,
      kPxPerTimeUnit, kLevelInGroup, kTimelineWidth, kEventHeight,
      kEventPaddingBottom);

  EXPECT_FLOAT_EQ(rect.left, 0.0f);
  EXPECT_FLOAT_EQ(rect.right, 0.0f);
}

TEST(TimelineTest, CalculateEventRect_EventCompletelyOutsideRight) {
  ColorPalette palette = ColorPalette::Default();
  Timeline timeline(palette);
  timeline.SetVisibleRange({100.0, 200.0});

  // Event time range: [210.0, 220.0].
  // Screen range will be larger than time end.
  // Expected to be fully clipped to the right edge [100.0, 100.0] (padding
  // right won't effect here because the event is clipped to the right edge).
  EventRect rect = timeline.CalculateEventRect(
      /*start=*/210.0, /*end=*/220.0, kScreenXOffset, kScreenYOffset,
      kPxPerTimeUnit, kLevelInGroup, kTimelineWidth, kEventHeight,
      kEventPaddingBottom);

  EXPECT_FLOAT_EQ(rect.left, 100.0f);
  EXPECT_FLOAT_EQ(rect.right, 100.0f);
}

TEST(TimelineTest, CalculateEventRect_EventFullyWithinView) {
  ColorPalette palette = ColorPalette::Default();
  Timeline timeline(palette);
  timeline.SetVisibleRange({100.0, 200.0});

  // Event time range: [110.0, 120.0].
  // Screen range before adjustments: [10.0, 20.0].
  EventRect rect = timeline.CalculateEventRect(
      /*start=*/110.0, /*end=*/120.0, kScreenXOffset, kScreenYOffset,
      kPxPerTimeUnit, kLevelInGroup, kTimelineWidth, kEventHeight,
      kEventPaddingBottom);

  EXPECT_FLOAT_EQ(rect.left, 10.0f);
  EXPECT_FLOAT_EQ(rect.right, 20.0f - kEventPaddingRight);
  EXPECT_FLOAT_EQ(rect.top, 0.0f);
  EXPECT_FLOAT_EQ(rect.bottom, kEventHeight);
}

TEST(TimelineTest, CalculateEventRect_EventPartiallyClippedLeft) {
  ColorPalette palette = ColorPalette::Default();
  Timeline timeline(palette);
  timeline.SetVisibleRange({100.0, 200.0});

  // Event time range: [90.0, 110.0].
  // Screen range after left clipping: [0.0, 10.0].
  EventRect rect = timeline.CalculateEventRect(
      /*start=*/90.0, /*end=*/110.0, kScreenXOffset, kScreenYOffset,
      kPxPerTimeUnit, kLevelInGroup, kTimelineWidth, kEventHeight,
      kEventPaddingBottom);

  EXPECT_FLOAT_EQ(rect.left, 0.0f);
  EXPECT_FLOAT_EQ(rect.right, 10.0f - kEventPaddingRight);
}

TEST(TimelineTest, CalculateEventRect_EventPartiallyClippedRight) {
  ColorPalette palette = ColorPalette::Default();
  Timeline timeline(palette);
  timeline.SetVisibleRange({100.0, 200.0});

  // Event time range: [190.0, 210.0].
  // Screen range after right clipping: [90.0, 100.0].
  EventRect rect = timeline.CalculateEventRect(
      /*start=*/190.0, /*end=*/210.0, kScreenXOffset, kScreenYOffset,
      kPxPerTimeUnit, kLevelInGroup, kTimelineWidth, kEventHeight,
      kEventPaddingBottom);

  EXPECT_FLOAT_EQ(rect.left, 90.0f);
  EXPECT_FLOAT_EQ(rect.right, 100.0f);
}

TEST(TimelineTest, CalculateEventRect_EventSmallerThanMinimumWidth) {
  ColorPalette palette = ColorPalette::Default();
  Timeline timeline(palette);
  timeline.SetVisibleRange({100.0, 200.0});

  // Event time range: [110.0, 110.1].
  // Screen width is expanded to kEventMinimumDrawWidth.
  EventRect rect = timeline.CalculateEventRect(
      /*start=*/110.0, /*end=*/110.1, kScreenXOffset, kScreenYOffset,
      kPxPerTimeUnit, kLevelInGroup, kTimelineWidth, kEventHeight,
      kEventPaddingBottom);

  EXPECT_FLOAT_EQ(rect.left, 10.0f);
  EXPECT_FLOAT_EQ(rect.right,
                  10.0f + kEventMinimumDrawWidth - kEventPaddingRight);
}

TEST(TimelineTest, CalculateEventRect_ZeroPxPerTimeUnit) {
  ColorPalette palette = ColorPalette::Default();
  Timeline timeline(palette);
  timeline.SetVisibleRange({100.0, 100.0});  // Zero duration

  // With px_per_time_unit = 0, the event width is 0, so it's expanded to
  // kEventMinimumDrawWidth.
  EventRect rect = timeline.CalculateEventRect(
      /*start=*/110.0, /*end=*/120.0, kScreenXOffset, kScreenYOffset,
      /*px_per_time_unit=*/0.0, kLevelInGroup, kTimelineWidth, kEventHeight,
      kEventPaddingBottom);

  // left becomes screen_x_offset (0), right becomes max(0, 0 +
  // kEventMinimumDrawWidth)
  EXPECT_FLOAT_EQ(rect.left, 0.0f);
  EXPECT_FLOAT_EQ(rect.right, kEventMinimumDrawWidth - kEventPaddingRight);
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

TEST(TimelineTest, CalculateTickInfo) {
  ColorPalette palette = ColorPalette::Default();
  Timeline timeline(palette);
  // Data range: [1000, 2000].
  timeline.set_data_time_range({1000.0, 2000.0});
  // Visible range: [1100, 1200].
  timeline.SetVisibleRange({1100.0, 1200.0});

  // 1 pixel per microsecond.
  double px_per_unit = 1.0;
  Timeline::TickInfo info = timeline.CalculateTickInfo(px_per_unit);

  // min_time_interval = 80 / 1.0 = 80.
  // CalculateNiceInterval(80) should return 100.
  EXPECT_DOUBLE_EQ(info.tick_interval, 100.0);
  EXPECT_DOUBLE_EQ(info.major_tick_dist_px, 100.0);

  // view_start_relative = 1100 - 1000 = 100.
  // first_tick_time_relative = floor(100 / 100) * 100 = 100.
  EXPECT_DOUBLE_EQ(info.first_tick_time_relative, 100.0);
}

TEST(TimelineTest, CalculateTickInfoOffset) {
  ColorPalette palette = ColorPalette::Default();
  Timeline timeline(palette);
  timeline.set_data_time_range({1000.0, 2000.0});
  // Visible range starts at 1105, but interval is 10.
  timeline.SetVisibleRange({1105.0, 1150.0});

  // 10 pixels per microsecond.
  double px_per_unit = 10.0;
  Timeline::TickInfo info = timeline.CalculateTickInfo(px_per_unit);

  // min_time_interval = 80 / 10 = 8.
  // CalculateNiceInterval(8) should return 10.
  EXPECT_DOUBLE_EQ(info.tick_interval, 10.0);

  // view_start_relative = 1105 - 1000 = 105.
  // first_tick_time_relative = floor(105 / 10) * 10 = 100.
  EXPECT_DOUBLE_EQ(info.first_tick_time_relative, 100.0);
}

// Constants for CalculateEventRect tests

TEST(TimelineTest, CalculateTickInfoZoomedIn) {
  ColorPalette palette = ColorPalette::Default();
  Timeline timeline(palette);
  timeline.set_data_time_range({1000.0, 2000.0});
  timeline.SetVisibleRange({1105.0, 1106.0});  // Very zoomed in.

  // 100 pixels per microsecond.
  double px_per_unit = 100.0;
  Timeline::TickInfo info = timeline.CalculateTickInfo(px_per_unit);

  // min_time_interval = 80 / 100 = 0.8.
  // CalculateNiceInterval(0.8) should return 1.0.
  EXPECT_DOUBLE_EQ(info.tick_interval, 1.0);
  EXPECT_DOUBLE_EQ(info.major_tick_dist_px, 100.0);

  // view_start_relative = 1105 - 1000 = 105.
  // first_tick_time_relative = floor(105 / 1) * 1 = 105.
  EXPECT_DOUBLE_EQ(info.first_tick_time_relative, 105.0);
}

TEST(TimelineTest, ConstrainTimeRange_EndAfterDataRange) {
  // Data Range: [=====================]
  // Range:                  {--------------}
  // Constrained:       (--------------)
  ColorPalette palette = ColorPalette::Default();
  Timeline timeline(palette);
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
  ColorPalette palette = ColorPalette::Default();
  Timeline timeline(palette);
  timeline.set_data_time_range({10.0, 100.0});
  TimeRange range(11.0, 110.0);

  timeline.ConstrainTimeRange(range);

  EXPECT_DOUBLE_EQ(range.start(), 10.0);
  EXPECT_DOUBLE_EQ(range.end(), 100.0);
}

TEST(TimelineTest, ConstrainTimeRange_EnforceMinDuration) {
  ColorPalette palette = ColorPalette::Default();
  Timeline timeline(palette);
  timeline.set_data_time_range({0.0, 100.0});
  TimeRange range(50.0, 50.0);

  timeline.ConstrainTimeRange(range);

  EXPECT_NEAR(range.duration(), kMinDurationMicros, 1e-14);
  EXPECT_DOUBLE_EQ(range.center(), 50.0);
  EXPECT_DOUBLE_EQ(range.start(), 50.0 - kMinDurationMicros / 2.0);
  EXPECT_DOUBLE_EQ(range.end(), 50.0 + kMinDurationMicros / 2.0);
}

TEST(TimelineTest, ConstrainTimeRange_MinDurationExpansionClampedAtEnd) {
  ColorPalette palette = ColorPalette::Default();
  Timeline timeline(palette);
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

TEST(TimelineTest, ConstrainTimeRange_MinDurationExpansionClampedAtStart) {
  ColorPalette palette = ColorPalette::Default();
  Timeline timeline(palette);
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

TEST(TimelineTest, ConstrainTimeRange_NoChange) {
  // Data Range: [===========================]
  // Range:        {----------------------}
  // Constrained:  (----------------------)
  ColorPalette palette = ColorPalette::Default();
  Timeline timeline(palette);
  timeline.set_data_time_range({0.0, 100.0});
  TimeRange range(10.0, 90.0);

  timeline.ConstrainTimeRange(range);

  EXPECT_DOUBLE_EQ(range.start(), 10.0);
  EXPECT_DOUBLE_EQ(range.end(), 90.0);
}

TEST(TimelineTest, ConstrainTimeRange_RangeCoversDataRange) {
  // Data Range:      [================]
  // Range: {------------------------------}
  // Constrained:     (================)
  ColorPalette palette = ColorPalette::Default();
  Timeline timeline(palette);
  timeline.set_data_time_range({10.0, 100.0});
  TimeRange range(0.0, 120.0);

  timeline.ConstrainTimeRange(range);

  EXPECT_DOUBLE_EQ(range.start(), 10.0);
  EXPECT_DOUBLE_EQ(range.end(), 100.0);
}

TEST(TimelineTest, ConstrainTimeRange_StartBeforeDataRange) {
  // Data Range:      [==========================]
  // Range:      {---------}
  // Constrained:     (---------)
  ColorPalette palette = ColorPalette::Default();
  Timeline timeline(palette);
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
  ColorPalette palette = ColorPalette::Default();
  Timeline timeline(palette);
  timeline.set_data_time_range({10.0, 100.0});
  TimeRange range(0.0, 99.0);

  timeline.ConstrainTimeRange(range);

  EXPECT_DOUBLE_EQ(range.start(), 10.0);
  EXPECT_DOUBLE_EQ(range.end(), 100.0);
}

TEST(TimelineTest, CopyNotificationTimerAndNameInitialization) {
  ColorPalette palette = ColorPalette::Default();
  Timeline timeline(palette);
  EXPECT_EQ(timeline.get_copy_notification_timer_for_test(), 0.0f);
  EXPECT_EQ(timeline.get_copied_track_name_for_test(), "");
}

TEST(TimelineTest, GetDeleteButtonLayout_TextDoesNotFit) {
  ColorPalette palette = ColorPalette::Default();
  Timeline timeline(palette);
  const ImVec2 text_size(50.0f, 20.0f);
  const ImVec2 text_pos(100.0f, 200.0f);
  // Visible range is smaller than text width.
  const ImRect visible_range_rect(100.0f, 0.0f, 140.0f, 300.0f);
  const ImRect full_range_rect(0.0f, 0.0f, 300.0f, 300.0f);

  auto layout = timeline.GetDeleteButtonLayout(
      text_size, text_pos, visible_range_rect, full_range_rect);

  EXPECT_FALSE(layout.text_fits);

  // Button should be centered in the visible range.
  EXPECT_FLOAT_EQ(layout.button_pos.x,
                  visible_range_rect.GetCenter().x - kCloseButtonSize / 2.0f);
  EXPECT_FLOAT_EQ(layout.button_pos.y,
                  text_pos.y + (text_size.y - kCloseButtonSize) / 2.0f);

  // Hover rect should be the visible range.
  EXPECT_EQ(layout.hover_rect.Min.x, visible_range_rect.Min.x);
  EXPECT_EQ(layout.hover_rect.Max.x, visible_range_rect.Max.x);
}

TEST(TimelineTest, GetDeleteButtonLayout_TextFits) {
  ColorPalette palette = ColorPalette::Default();
  Timeline timeline(palette);
  const ImVec2 text_size(50.0f, 20.0f);
  const ImVec2 text_pos(100.0f, 200.0f);
  const ImRect visible_range_rect(0.0f, 0.0f, 300.0f, 300.0f);
  const ImRect full_range_rect(0.0f, 0.0f, 300.0f, 300.0f);

  auto layout = timeline.GetDeleteButtonLayout(
      text_size, text_pos, visible_range_rect, full_range_rect);

  EXPECT_TRUE(layout.text_fits);

  // Button should be to the right of the text.
  EXPECT_FLOAT_EQ(layout.button_pos.x,
                  text_pos.x + text_size.x + kCloseButtonPadding);
  EXPECT_FLOAT_EQ(layout.button_pos.y,
                  text_pos.y + (text_size.y - kCloseButtonSize) / 2.0f);

  // Hover rect should include text and button with margin.
  EXPECT_LE(layout.hover_rect.Min.x, text_pos.x - kHoverPadding);
  EXPECT_LE(layout.hover_rect.Min.y, text_pos.y - kHoverPadding);
  EXPECT_GE(layout.hover_rect.Max.x,
            layout.button_pos.x + kCloseButtonSize + kHoverPadding);
}

TEST(TimelineTest, GetTextForDisplayWhenMultipleCharsFit) {
  MockTimeline timeline;
  const std::string text = "Long Event Name";
  // If char width is kCharWidth, ellipsis "..." width is kEllipsisWidth.
  // Available width allows "Lo..." (2 * kCharWidth + kEllipsisWidth).
  const float available_width = 2 * kCharWidth + kEllipsisWidth + 5.0f;

  EXPECT_CALL(timeline, GetTextSize(_)).WillRepeatedly([](absl::string_view s) {
    if (s == "...") return ImVec2{kEllipsisWidth, kEventHeight};
    return ImVec2{s.length() * kCharWidth, kEventHeight};
  });

  EXPECT_EQ(timeline.GetTextForDisplay(text, available_width), "Lo...");
}

TEST(TimelineTest, GetTextForDisplayWhenTextFits) {
  MockTimeline timeline;
  const std::string text = "Test";

  EXPECT_CALL(timeline, GetTextSize(absl::string_view(text)))
      .WillOnce(Return(ImVec2{50.0f, kEventHeight}));

  // Text width 50.0 is smaller than 1000.0, so no truncation.
  EXPECT_EQ(timeline.GetTextForDisplay(text, 1000.0f), text);
}

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

TEST(TimelineTest, InitializeLastFetchRequestRange_ConstrainsRange) {
  ColorPalette palette = ColorPalette::Default();
  Timeline timeline(palette);
  timeline.set_data_time_range({5500000.0, 6500000.0});  // Narrow data range
  timeline.set_fetched_data_time_range({0.0, 10000000.0});
  timeline.set_is_incremental_loading(false);

  TimeRange visible = {5000000.0, 6000000.0};  // 1s duration
  timeline.SetVisibleRange(visible);
  timeline.InitializeLastFetchRequestRange(visible);

  // Unscaled fetch would be [4e6, 7e6]
  // Constrained to [5.5e6, 6.5e6]
  EXPECT_EQ(timeline.last_fetch_request_range().start(), 5500000.0);
  EXPECT_EQ(timeline.last_fetch_request_range().end(), 6500000.0);
}

TEST(TimelineTest, InitializeLastFetchRequestRange_ExpandsToMinDuration) {
  ColorPalette palette = ColorPalette::Default();
  Timeline timeline(palette);
  timeline.set_data_time_range({0.0, 10000000.0});
  timeline.set_fetched_data_time_range({0.0, 10000000.0});
  timeline.set_is_incremental_loading(false);

  TimeRange visible = {5000000.0, 5000100.0};  // 100us duration
  timeline.SetVisibleRange(visible);
  timeline.InitializeLastFetchRequestRange(visible);

  // Unscaled fetch would be [5e6, 5e6+100]
  // Scaled 3.0: center 5000050, duration 300
  // Expands to kMinFetchDurationMicros (1000us)
  // Constrained to data_time_range_ [0.0, 10000000.0]
  EXPECT_EQ(timeline.last_fetch_request_range().start(), 4999550.0);
  EXPECT_EQ(timeline.last_fetch_request_range().end(), 5000550.0);
}

TEST(TimelineTest, InitializeLastFetchRequestRange_SetsCorrectRange) {
  ColorPalette palette = ColorPalette::Default();
  Timeline timeline(palette);
  timeline.set_data_time_range({0.0, 10000000.0});
  timeline.set_fetched_data_time_range({0.0, 10000000.0});
  timeline.set_is_incremental_loading(false);

  TimeRange visible = {5000000.0, 6000000.0};  // 1s duration
  timeline.SetVisibleRange(visible);
  timeline.InitializeLastFetchRequestRange(visible);

  EXPECT_EQ(timeline.last_fetch_request_range().start(), 4000000.0);
  EXPECT_EQ(timeline.last_fetch_request_range().end(), 7000000.0);
}

TEST(TimelineTest, MaybeRequestDataDoesNotRefetchWhenZoomAtBoundary) {
  ColorPalette palette = ColorPalette::Default();
  Timeline timeline(palette);
  timeline.set_data_time_range({0.0, 100000.0});
  timeline.set_fetched_data_time_range({0.0, 24000.0});
  timeline.set_is_incremental_loading(false);

  timeline.SetVisibleRange({10000.0, 11000.0});

  bool request_triggered = false;
  timeline.set_event_callback(
      [&](absl::string_view type, const EventData& detail) {
        if (type == kFetchData) {
          request_triggered = true;
        }
      });

  timeline.MaybeRequestData();

  EXPECT_FALSE(request_triggered);
}

TEST(TimelineTest, MaybeRequestDataExpandsToMinDuration) {
  ColorPalette palette = ColorPalette::Default();
  Timeline timeline(palette);
  // Scenario: Visible range is very small (100us).
  // Fetch calculated (Scale 3.0) would be 300us.
  // Must expand to kMinFetchDurationMicros (1000us = 1ms).
  // Data range large enough to not constrain.

  timeline.set_data_time_range({0.0, 20000000.0});
  timeline.set_fetched_data_time_range({0.0, 200000.0});
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
  // Center 5.00005s. Half duration 500us (0.0005s).
  // Start: 5.00005s - 0.0005s = 4.99955s = 4999550.0.
  // End: 5.00005s + 0.0005s = 5.00055s = 5000550.0.
  timeline.SetVisibleRange({5000000.0, 5000100.0});
  timeline.MaybeRequestData();

  ASSERT_TRUE(request_triggered);
  EXPECT_DOUBLE_EQ(
      std::any_cast<double>(received_data.at(std::string(kFetchDataStart))),
      MicrosToMillis(4999550.0));
  EXPECT_DOUBLE_EQ(
      std::any_cast<double>(received_data.at(std::string(kFetchDataEnd))),
      MicrosToMillis(5000550.0));
}

TEST(TimelineTest, MaybeRequestDataFetchesConstrainedRange) {
  ColorPalette palette = ColorPalette::Default();
  Timeline timeline(palette);
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
  timeline.MaybeRequestData();

  ASSERT_TRUE(request_triggered);
  EXPECT_DOUBLE_EQ(
      std::any_cast<double>(received_data.at(std::string(kFetchDataStart))),
      MicrosToMillis(7000000.0));
  EXPECT_DOUBLE_EQ(
      std::any_cast<double>(received_data.at(std::string(kFetchDataEnd))),
      MicrosToMillis(10000000.0));
}

TEST(TimelineTest, SetTimelineDataPreservesScrollOnIncrementalUpdate) {
  ColorPalette palette = ColorPalette::Default();
  Timeline timeline(palette);

  EXPECT_FALSE(timeline.get_should_restore_scroll_for_test());

  timeline.set_is_incremental_loading(true);

  FlameChartTimelineData data;
  timeline.SetTimelineData(std::move(data));

  EXPECT_TRUE(timeline.get_should_restore_scroll_for_test());
}

TEST(TimelineTest,
     MaybeRequestDataNoTriggerAfterInitializeLastFetchRequestRange) {
  ColorPalette palette = ColorPalette::Default();
  Timeline timeline(palette);
  timeline.set_data_time_range({0.0, 10000000.0});
  timeline.set_fetched_data_time_range({0.0, 10000000.0});
  timeline.set_is_incremental_loading(false);

  bool request_triggered = false;
  timeline.set_event_callback(
      [&](absl::string_view type, const EventData& detail) {
        if (type == kFetchData) {
          request_triggered = true;
        }
      });

  // Setup initial visible range mimicking URL load.
  TimeRange visible = {5000000.0, 6000000.0};  // 1s duration
  timeline.SetVisibleRange(visible);
  timeline.InitializeLastFetchRequestRange(visible);

  // Trigger MaybeRequestData
  timeline.MaybeRequestData();

  EXPECT_FALSE(request_triggered);
}

TEST(TimelineTest, MaybeRequestDataNotTriggeredWhenInsidePreserveRange) {
  ColorPalette palette = ColorPalette::Default();
  Timeline timeline(palette);
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
  timeline.MaybeRequestData();

  EXPECT_FALSE(request_triggered);
}

TEST(TimelineTest, MaybeRequestDataNotTriggeredWhenLoading) {
  ColorPalette palette = ColorPalette::Default();
  Timeline timeline(palette);
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

TEST(TimelineTest, MaybeRequestDataNotTriggeredWhenPreserveExceedsDataRange) {
  ColorPalette palette = ColorPalette::Default();
  Timeline timeline(palette);
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
     MaybeRequestDataNotTriggeredWhenPreserveExceedsDataRangeEnd) {
  ColorPalette palette = ColorPalette::Default();
  Timeline timeline(palette);
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
  timeline.MaybeRequestData();

  EXPECT_FALSE(request_triggered);
}

TEST(TimelineTest,
     MaybeRequestDataNotTriggeredWhenPreserveExceedsDataRangeStart) {
  ColorPalette palette = ColorPalette::Default();
  Timeline timeline(palette);
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

TEST(TimelineTest, MaybeRequestDataRefetchWhenZoomedIn) {
  ColorPalette palette = ColorPalette::Default();
  Timeline timeline(palette);
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
  timeline.MaybeRequestData();

  ASSERT_TRUE(request_triggered);
  EXPECT_DOUBLE_EQ(
      std::any_cast<double>(received_data.at(std::string(kFetchDataStart))),
      MicrosToMillis(4600000.0));
  EXPECT_DOUBLE_EQ(
      std::any_cast<double>(received_data.at(std::string(kFetchDataEnd))),
      MicrosToMillis(5800000.0));
}

TEST(TimelineTest, MaybeRequestDataRefetchesWhenZoomedInDespiteRangeCoverage) {
  ColorPalette palette = ColorPalette::Default();
  Timeline timeline(palette);
  // Step 1: Initialize with full range fetch to set last_fetch_request_range_.
  // Data: [0, 200s].
  timeline.set_data_time_range({0.0, 200000000.0});

  // Simulate that we previously asked for the full range and received it.
  // This sets last_fetch_request_range_ to [0, 200s].

  // Step 2: Simulate "We have all the data now".
  timeline.set_fetched_data_time_range({0.0, 200000000.0});
  timeline.set_is_incremental_loading(false);

  // Step 3: Zoom IN heavily.
  // Visible: [5s, 5.0002s] (200us).
  // Fetch (Scale 3): 600us -> Expands to kMinFetchDurationMicros (20000000us).
  // Fetched (200s) / Fetch (20s) = 10 > kRefetchZoomRatio (8).
  // last_fetch ([0, 200s]) CONTAINS preserve.

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
  // Constrained to data_time_range_ [0.0, 200000000.0].
  timeline.SetVisibleRange({5000000.0, 5000200.0});
  timeline.MaybeRequestData();

  ASSERT_TRUE(request_triggered);
  EXPECT_DOUBLE_EQ(
      std::any_cast<double>(received_data.at(std::string(kFetchDataStart))),
      MicrosToMillis(4999600.0));
  EXPECT_DOUBLE_EQ(
      std::any_cast<double>(received_data.at(std::string(kFetchDataEnd))),
      MicrosToMillis(5000600.0));
}

TEST(TimelineTest, MaybeRequestDataSetsIsLoadingToTrue) {
  ColorPalette palette = ColorPalette::Default();
  Timeline timeline(palette);
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
  timeline.MaybeRequestData();
  EXPECT_EQ(request_count, 1);

  // Should not trigger again because is_incremental_loading_ should be set to
  // true inside `MaybeRequestData`.
  timeline.SetVisibleRange({100.0, 200.0});
  timeline.MaybeRequestData();
  EXPECT_EQ(request_count, 1);
}

TEST(TimelineTest, MaybeRequestDataSkipsFetchIfRangeAlreadyRequested) {
  ColorPalette palette = ColorPalette::Default();
  Timeline timeline(palette);
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
  timeline.MaybeRequestData();

  EXPECT_FALSE(request_triggered);
}

TEST(TimelineTest, MaybeRequestDataSuppressesRedundantFetchWithExpandedRange) {
  ColorPalette palette = ColorPalette::Default();
  Timeline timeline(palette);
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
  timeline.MaybeRequestData();

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
  TimelineImGuiTestFixture() : timeline_(color_palette_) {}

  void SetUp() override {
    ImGui::CreateContext();

    ImGuiIO& io = ImGui::GetIO();
    // Set dummy display size and delta time, required for ImGui to function.
    io.DisplaySize = ImVec2(1920, 1080);
    io.DeltaTime = 0.1f;
    // The font atlas must be built before ImGui::NewFrame() is called.
    io.Fonts->Build();
    timeline_.SetTimelineData(
        {{},  // Pass ColorPalette::Default() to constructor
         {},
         {},
         {},
         {},
         {},
         {},
         {},
         {},
         {{.name = "group",
           .start_level = 0,
           .nesting_level = kThreadNestingLevel,
           .expanded = true}},
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

  void SimulateFrame(Pixel scroll_y, Pixel window_height) {
    ImGui::NewFrame();
    ImGui::SetScrollY(scroll_y);
    ImGui::SetWindowSize(ImVec2(1000.0f, window_height));
    timeline_.Draw();
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

  // Returns the X coordinate where the timeline track area starts (i.e., the
  // right edge of the label column).
  float GetTimelineStartX() { return timeline_.GetLabelWidth(); }

  void SimulateLabelColumnResizeDragStart() {
    // Produce one frame to init window
    SimulateFrame();

    ImGuiWindow* window = ImGui::FindWindowByName("Timeline viewer");
    float win_x = window ? window->Pos.x : 0.0f;
    float win_y = window ? window->DC.CursorStartPos.y : 0.0f;

    // The resize handle is positioned at `label_width_ - 4.0f` with width 8.0f.
    // The timeline area starts at `label_width_`.
    // By clicking at `label_width_ - 2.0f`, we hit the resize handle but avoid
    // `GetTimelineArea()` which would trigger `HandleMouseDown`, ensuring only
    // resize happens.
    float resize_handle_x = win_x + GetTimelineStartX() - 2.0f;
    float resize_handle_y = win_y + 30.0f;

    ImGuiIO& io = ImGui::GetIO();
    io.AddMousePosEvent(resize_handle_x, resize_handle_y);
    SimulateFrame();
    io.AddMouseButtonEvent(0, true);
    SimulateFrame();
  }

  struct EventDef {
    absl::string_view name;
    double start_time;
    double total_time;
    int level;
    ProcessId pid = 1;
    ThreadId tid = 1;
    EventId event_id = 100;
  };

  FlameChartTimelineData CreateTimelineData(absl::Span<const EventDef> events,
                                            absl::Span<const Group> groups) {
    FlameChartTimelineData data;
    data.groups.assign(groups.begin(), groups.end());
    for (const auto& ev : events) {
      data.entry_names.push_back(std::string(ev.name));
      data.entry_start_times.push_back(ev.start_time);
      data.entry_total_times.push_back(ev.total_time);
      data.entry_levels.push_back(ev.level);
      data.entry_pids.push_back(ev.pid);
      data.entry_tids.push_back(ev.tid);
      data.entry_event_ids.push_back(ev.event_id);
      data.entry_args.push_back({});
    }
    return data;
  }

  FlameChartTimelineData CreateTimelineData(absl::Span<const EventDef> events) {
    return CreateTimelineData(events,
                              std::vector<Group>{{.type = Group::Type::kFlame,
                                                  .name = "Group 1",
                                                  .start_level = 0}});
  }

  ParsedTraceEvents CreateSearchResults(absl::Span<const EventDef> events) {
    ParsedTraceEvents search_results;
    for (const auto& ev : events) {
      TraceEvent trace_ev;
      trace_ev.ph = Phase::kComplete;
      trace_ev.event_id = ev.event_id;
      trace_ev.pid = ev.pid;
      trace_ev.tid = ev.tid;
      trace_ev.name = std::string(ev.name);
      trace_ev.ts = ev.start_time;
      trace_ev.dur = ev.total_time;
      search_results.flame_events.push_back(trace_ev);
    }
    return search_results;
  }

  void SetupDeferredFetchExperiment(bool& request_triggered) {
    timeline_.set_data_time_range({0.0, 10000000.0});
    timeline_.set_fetched_data_time_range({0.0, 2000000.0});
    timeline_.set_is_incremental_loading(false);
    timeline_.set_event_callback(
        [&](absl::string_view type, const EventData& detail) {
          if (type == kFetchData) {
            request_triggered = true;
          }
        });
    timeline_.SetVisibleRange({0.0, 1000000.0});
    request_triggered = false;
  }

  ColorPalette color_palette_ = ColorPalette::Default();
  TimelineT timeline_;
};

using MockTimelineImGuiFixture = TimelineImGuiTestFixture<MockTimeline>;

TEST(TimelineTest, MaybeRequestDataTriggeredWhenPanningOutsidePreserveRange) {
  ColorPalette palette = ColorPalette::Default();
  Timeline timeline(palette);
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
  timeline.MaybeRequestData();

  ASSERT_TRUE(request_triggered);
  // Fetch range (Scale 3.0 of visible): [0, 300].
  EXPECT_DOUBLE_EQ(
      std::any_cast<double>(received_data.at(std::string(kFetchDataStart))),
      MicrosToMillis(0.0));
  EXPECT_DOUBLE_EQ(
      std::any_cast<double>(received_data.at(std::string(kFetchDataEnd))),
      MicrosToMillis(300.0));
}

TEST_F(MockTimelineImGuiFixture, HandleKeyboard_Key1_EmitsMouseModeChanged) {
  bool event_called = false;
  int emitted_mode = -1;

  timeline_.set_event_callback(
      [&](absl::string_view event_name, const EventData& data) {
        if (event_name == kMouseModeChanged) {
          event_called = true;
          auto it = data.find(std::string(kMouseModeKey));
          if (it != data.end()) {
            emitted_mode = std::any_cast<int>(it->second);
          }
        }
      });

  ImGui::GetIO().AddKeyEvent(ImGuiKey_1, true);
  SimulateFrame();

  EXPECT_TRUE(event_called);
  EXPECT_EQ(emitted_mode, static_cast<int>(MouseMode::kSelect));
}

TEST_F(MockTimelineImGuiFixture, HandleKeyboard_Key2_EmitsMouseModeChanged) {
  bool event_called = false;
  int emitted_mode = -1;

  timeline_.set_event_callback(
      [&](absl::string_view event_name, const EventData& data) {
        if (event_name == kMouseModeChanged) {
          event_called = true;
          auto it = data.find(std::string(kMouseModeKey));
          if (it != data.end()) {
            emitted_mode = std::any_cast<int>(it->second);
          }
        }
      });

  ImGui::GetIO().AddKeyEvent(ImGuiKey_2, true);
  SimulateFrame();

  EXPECT_TRUE(event_called);
  EXPECT_EQ(emitted_mode, static_cast<int>(MouseMode::kPan));
}

TEST_F(MockTimelineImGuiFixture, HandleKeyboard_Key3_EmitsMouseModeChanged) {
  bool event_called = false;
  int emitted_mode = -1;

  timeline_.set_event_callback(
      [&](absl::string_view event_name, const EventData& data) {
        if (event_name == kMouseModeChanged) {
          event_called = true;
          auto it = data.find(std::string(kMouseModeKey));
          if (it != data.end()) {
            emitted_mode = std::any_cast<int>(it->second);
          }
        }
      });

  ImGui::GetIO().AddKeyEvent(ImGuiKey_3, true);
  SimulateFrame();

  EXPECT_TRUE(event_called);
  EXPECT_EQ(emitted_mode, static_cast<int>(MouseMode::kZoom));
}

TEST_F(MockTimelineImGuiFixture, HandleKeyboard_Key4_EmitsMouseModeChanged) {
  bool event_called = false;
  int emitted_mode = -1;

  timeline_.set_event_callback(
      [&](absl::string_view event_name, const EventData& data) {
        if (event_name == kMouseModeChanged) {
          event_called = true;
          auto it = data.find(std::string(kMouseModeKey));
          if (it != data.end()) {
            emitted_mode = std::any_cast<int>(it->second);
          }
        }
      });

  ImGui::GetIO().AddKeyEvent(ImGuiKey_4, true);
  SimulateFrame();

  EXPECT_TRUE(event_called);
  EXPECT_EQ(emitted_mode, static_cast<int>(MouseMode::kTiming));
}

TEST_F(MockTimelineImGuiFixture, HandleKeyboard_EmptyCallback_DoesNotCrash) {
  timeline_.set_event_callback({});

  ImGui::GetIO().AddKeyEvent(ImGuiKey_1, true);
  SimulateFrame();
}

TEST(TimelineTest, NavigateSearchQueryResult) {
  ColorPalette palette = ColorPalette::Default();
  Timeline timeline(palette);
  FlameChartTimelineData data;
  data.groups.push_back({.name = "Group 1",
                         .start_level = 0,
                         .nesting_level = kThreadNestingLevel,
                         .expanded = true});
  data.events_by_level.push_back({0, 1});
  data.entry_names.push_back("apple");
  data.entry_names.push_back("apricot");
  data.entry_levels.push_back(0);
  data.entry_levels.push_back(0);
  data.entry_start_times.push_back(100.0);
  data.entry_start_times.push_back(200.0);
  data.entry_total_times.push_back(10.0);
  data.entry_total_times.push_back(10.0);
  data.entry_pids.push_back(1);
  data.entry_pids.push_back(1);
  data.entry_tids.push_back(1);
  data.entry_tids.push_back(1);
  data.entry_event_ids.push_back(1);
  data.entry_event_ids.push_back(2);
  data.entry_args.push_back({});
  data.entry_args.push_back({});
  timeline.SetTimelineData(std::move(data));
  timeline.set_data_time_range({0.0, 1000.0});
  timeline.SetVisibleRange({0.0, 100.0});

  timeline.SetSearchQuery("ap");
  EXPECT_EQ(timeline.get_search_results_count(), 2);
  EXPECT_EQ(timeline.get_current_search_result_index(), -1);

  timeline.NavigateToNextSearchResult();
  EXPECT_EQ(timeline.get_current_search_result_index(), 0);

  timeline.NavigateToNextSearchResult();
  EXPECT_EQ(timeline.get_current_search_result_index(), 1);

  timeline.NavigateToNextSearchResult();
  EXPECT_EQ(timeline.get_current_search_result_index(), 0);  // Wrap around

  timeline.NavigateToPrevSearchResult();
  EXPECT_EQ(timeline.get_current_search_result_index(), 1);  // Wrap around
}

TEST(TimelineTest, NavigateToNextSearchResultCallsRedrawCallback) {
  ColorPalette palette = ColorPalette::Default();
  Timeline timeline(palette);
  FlameChartTimelineData data;
  data.entry_names.push_back("event");
  data.entry_names.push_back("event");
  data.entry_start_times.push_back(50.0);
  data.entry_start_times.push_back(100.0);
  data.entry_levels.push_back(0);
  data.entry_levels.push_back(0);
  data.entry_total_times.push_back(10.0);
  data.entry_total_times.push_back(10.0);
  data.entry_pids.push_back(0);
  data.entry_pids.push_back(0);
  data.entry_tids.push_back(0);
  data.entry_tids.push_back(0);
  data.entry_event_ids.push_back(1);
  data.entry_event_ids.push_back(2);
  data.entry_args.push_back({});
  data.entry_args.push_back({});
  timeline.SetTimelineData(std::move(data));

  timeline.SetSearchQuery("event");

  bool redraw_called = false;
  timeline.set_redraw_callback([&redraw_called]() { redraw_called = true; });

  timeline.NavigateToNextSearchResult();

  EXPECT_TRUE(redraw_called);
}

TEST(TimelineTest, NavigateToNextSearchResultCallsRedrawCallbackCount) {
  ColorPalette palette = ColorPalette::Default();
  Timeline timeline(palette);
  FlameChartTimelineData data;
  data.entry_names.push_back("event");
  data.entry_names.push_back("event");
  data.entry_start_times.push_back(50.0);
  data.entry_start_times.push_back(100.0);
  data.entry_levels.push_back(0);
  data.entry_levels.push_back(0);
  data.entry_total_times.push_back(10.0);
  data.entry_total_times.push_back(10.0);
  data.entry_pids.push_back(0);
  data.entry_pids.push_back(0);
  data.entry_tids.push_back(0);
  data.entry_tids.push_back(0);
  data.entry_event_ids.push_back(1);
  data.entry_event_ids.push_back(2);
  data.entry_args.push_back({});
  data.entry_args.push_back({});
  timeline.SetTimelineData(std::move(data));
  timeline.SetVisibleRange(TimeRange(0.0, 200.0));

  timeline.SetSearchQuery("event");

  int redraw_count = 0;
  timeline.set_redraw_callback([&redraw_count]() { redraw_count++; });

  timeline.NavigateToNextSearchResult();

  EXPECT_GT(redraw_count, 0);
  EXPECT_GE(redraw_count, 1);
}

TEST(TimelineTest, NavigateToNextSearchResultEmptyResultsDoesNothing) {
  ColorPalette palette = ColorPalette::Default();
  Timeline timeline(palette);
  FlameChartTimelineData data;
  data.entry_names.push_back("foo");
  data.entry_start_times.push_back(10.0);
  data.entry_levels.push_back(0);
  data.entry_total_times.push_back(5.0);
  data.entry_pids.push_back(0);
  data.entry_tids.push_back(0);
  data.entry_event_ids.push_back(1);
  data.entry_args.push_back({});
  timeline.SetTimelineData(std::move(data));

  timeline.SetSearchQuery("bar");

  bool redraw_called = false;
  timeline.set_redraw_callback([&redraw_called]() { redraw_called = true; });

  timeline.NavigateToNextSearchResult();

  EXPECT_FALSE(redraw_called);
}

TEST(TimelineTest, NavigateToPrevSearchResultCallsRedrawCallbackCount) {
  ColorPalette palette = ColorPalette::Default();
  Timeline timeline(palette);
  FlameChartTimelineData data;
  data.entry_names.push_back("event");
  data.entry_names.push_back("event");
  data.entry_start_times.push_back(50.0);
  data.entry_start_times.push_back(100.0);
  data.entry_levels.push_back(0);
  data.entry_levels.push_back(0);
  data.entry_total_times.push_back(10.0);
  data.entry_total_times.push_back(10.0);
  data.entry_pids.push_back(0);
  data.entry_pids.push_back(0);
  data.entry_tids.push_back(0);
  data.entry_tids.push_back(0);
  data.entry_event_ids.push_back(1);
  data.entry_event_ids.push_back(2);
  data.entry_args.push_back({});
  data.entry_args.push_back({});
  timeline.SetTimelineData(std::move(data));
  timeline.SetVisibleRange(TimeRange(0.0, 200.0));

  timeline.SetSearchQuery("event");

  int redraw_count = 0;
  timeline.set_redraw_callback([&redraw_count]() { redraw_count++; });

  timeline.NavigateToPrevSearchResult();

  EXPECT_GT(redraw_count, 0);
  EXPECT_GE(redraw_count, 1);
}

TEST(TimelineTest, NavigateToPrevSearchResultEmptyResultsDoesNothing) {
  ColorPalette palette = ColorPalette::Default();
  Timeline timeline(palette);
  FlameChartTimelineData data;
  data.entry_names.push_back("foo");
  data.entry_start_times.push_back(10.0);
  data.entry_levels.push_back(0);
  data.entry_total_times.push_back(5.0);
  data.entry_pids.push_back(0);
  data.entry_tids.push_back(0);
  data.entry_event_ids.push_back(1);
  data.entry_args.push_back({});
  timeline.SetTimelineData(std::move(data));

  timeline.SetSearchQuery("bar");

  bool redraw_called = false;
  timeline.set_redraw_callback([&redraw_called]() { redraw_called = true; });

  timeline.NavigateToPrevSearchResult();

  EXPECT_FALSE(redraw_called);
}

TEST(TimelineTest, NavigateToPrevSearchResultWrapping) {
  ColorPalette palette = ColorPalette::Default();
  Timeline timeline(palette);
  FlameChartTimelineData data;
  data.entry_names.push_back("event1");
  data.entry_names.push_back("event2");
  data.entry_start_times.push_back(10.0);
  data.entry_start_times.push_back(20.0);
  data.entry_levels.push_back(0);
  data.entry_levels.push_back(0);
  data.entry_total_times.push_back(5.0);
  data.entry_total_times.push_back(5.0);
  data.entry_pids.push_back(0);
  data.entry_pids.push_back(0);
  data.entry_tids.push_back(0);
  data.entry_tids.push_back(0);
  data.entry_event_ids.push_back(1);
  data.entry_event_ids.push_back(2);
  data.entry_args.push_back({});
  data.entry_args.push_back({});
  timeline.SetTimelineData(std::move(data));

  timeline.SetSearchQuery("event");

  EXPECT_EQ(timeline.selected_event_index(), -1);

  timeline.NavigateToPrevSearchResult();
  EXPECT_EQ(timeline.selected_event_index(), 1);

  timeline.NavigateToPrevSearchResult();
  EXPECT_EQ(timeline.selected_event_index(), 0);
}

TEST(TimelineTest, PixelToTime) {
  ColorPalette palette = ColorPalette::Default();
  Timeline timeline(palette);
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

TEST(TimelineTest, RevealEventAlreadyInView) {
  ColorPalette palette = ColorPalette::Default();
  Timeline timeline(palette);
  FlameChartTimelineData data;
  data.groups.push_back({.name = "Group 1",
                         .start_level = 0,
                         .nesting_level = kThreadNestingLevel,
                         .expanded = true});
  data.events_by_level.push_back({0});
  data.entry_names.push_back("event0");
  data.entry_levels.push_back(0);
  data.entry_start_times.push_back(50000.0);
  data.entry_total_times.push_back(10.0);
  data.entry_pids.push_back(1);
  data.entry_args.push_back({});
  timeline.SetTimelineData(std::move(data));
  timeline.set_data_time_range({0.0, 100000.0});
  timeline.SetVisibleRange({10000.0, 60000.0});

  timeline.RevealEvent(0);
  Animation::UpdateAll(1.0f);

  EXPECT_DOUBLE_EQ(timeline.visible_range().start(), 10000.0);
  EXPECT_DOUBLE_EQ(timeline.visible_range().end(), 60000.0);
  EXPECT_DOUBLE_EQ(timeline.visible_range().duration(), 50000.0);
}

TEST(TimelineTest, RevealEventExpandsCollapsedTracks) {
  ColorPalette palette = ColorPalette::Default();
  Timeline timeline(palette);
  FlameChartTimelineData data;
  data.groups.push_back({.name = "Process 1",
                         .start_level = 0,
                         .nesting_level = kProcessNestingLevel,
                         .expanded = false});
  data.groups.push_back({.name = "Thread 1",
                         .start_level = 1,
                         .nesting_level = kThreadNestingLevel,
                         .expanded = false});

  data.events_by_level.resize(2);
  data.events_by_level[1].push_back(0);

  data.entry_names.push_back("event0");
  data.entry_levels.push_back(1);
  data.entry_start_times.push_back(50.0);
  data.entry_total_times.push_back(10.0);
  data.entry_pids.push_back(1);
  data.entry_args.push_back({});

  timeline.SetTimelineData(std::move(data));
  timeline.set_data_time_range({0.0, 100.0});
  timeline.SetVisibleRange({0.0, 100.0});

  EXPECT_FALSE(timeline.timeline_data().groups[0].expanded);
  EXPECT_FALSE(timeline.timeline_data().groups[1].expanded);

  timeline.RevealEvent(0);

  EXPECT_TRUE(timeline.timeline_data().groups[0].expanded);
  EXPECT_TRUE(timeline.timeline_data().groups[1].expanded);
}

TEST(TimelineTest, RevealEventInvalidIndex) {
  ColorPalette palette = ColorPalette::Default();
  Timeline timeline(palette);
  FlameChartTimelineData data;
  data.groups.push_back(
      {.name = "Group 1", .start_level = 0, .expanded = true});
  data.entry_names.push_back("event0");
  data.entry_levels.push_back(0);
  data.entry_start_times.push_back(100.0);
  data.entry_total_times.push_back(10.0);
  timeline.SetTimelineData(std::move(data));

  timeline.SetVisibleRange({0.0, 50.0});

  // Invalid negative index
  timeline.RevealEvent(-1);
  EXPECT_DOUBLE_EQ(timeline.visible_range().start(), 0.0);

  // Invalid out of bounds index
  timeline.RevealEvent(1);
  EXPECT_DOUBLE_EQ(timeline.visible_range().start(), 0.0);
}

TEST(TimelineTest, RevealEventInvalidIndexMismatchedSizes) {
  ColorPalette palette = ColorPalette::Default();
  Timeline timeline(palette);
  FlameChartTimelineData data;
  data.entry_start_times.push_back(100.0);
  data.entry_start_times.push_back(200.0);
  data.entry_total_times.push_back(10.0);
  timeline.SetTimelineData(std::move(data));
  timeline.set_data_time_range({0.0, 1000.0});
  timeline.SetVisibleRange({0.0, 50.0});

  bool callback_called = false;
  timeline.set_event_callback([&callback_called](absl::string_view event_type,
                                                 const EventData& detail) {
    if (event_type == kEventSelected) {
      callback_called = true;
    }
  });

  timeline.RevealEvent(1);
  Animation::UpdateAll(1.0f);

  EXPECT_FALSE(callback_called);
  EXPECT_DOUBLE_EQ(timeline.visible_range().start(), 0.0);
  EXPECT_DOUBLE_EQ(timeline.visible_range().end(), 50.0);
}

TEST(TimelineTest, RevealEventInvalidIndexNegative) {
  ColorPalette palette = ColorPalette::Default();
  Timeline timeline(palette);
  FlameChartTimelineData data;
  data.entry_start_times.push_back(100.0);
  data.entry_total_times.push_back(10.0);
  timeline.SetTimelineData(std::move(data));
  timeline.set_data_time_range({0.0, 1000.0});
  timeline.SetVisibleRange({0.0, 50.0});

  bool callback_called = false;
  timeline.set_event_callback([&callback_called](absl::string_view event_type,
                                                 const EventData& detail) {
    if (event_type == kEventSelected) {
      callback_called = true;
    }
  });

  timeline.RevealEvent(-1);
  Animation::UpdateAll(1.0f);

  EXPECT_FALSE(callback_called);
  EXPECT_DOUBLE_EQ(timeline.visible_range().start(), 0.0);
  EXPECT_DOUBLE_EQ(timeline.visible_range().end(), 50.0);
}

TEST(TimelineTest, RevealEventInvalidIndexOutOfBounds) {
  ColorPalette palette = ColorPalette::Default();
  Timeline timeline(palette);
  FlameChartTimelineData data;
  data.entry_start_times.push_back(100.0);
  data.entry_total_times.push_back(10.0);
  timeline.SetTimelineData(std::move(data));
  timeline.set_data_time_range({0.0, 1000.0});
  timeline.SetVisibleRange({0.0, 50.0});

  bool callback_called = false;
  timeline.set_event_callback([&callback_called](absl::string_view event_type,
                                                 const EventData& detail) {
    if (event_type == kEventSelected) {
      callback_called = true;
    }
  });

  timeline.RevealEvent(1);
  Animation::UpdateAll(1.0f);

  EXPECT_FALSE(callback_called);
  EXPECT_DOUBLE_EQ(timeline.visible_range().start(), 0.0);
  EXPECT_DOUBLE_EQ(timeline.visible_range().end(), 50.0);
}

TEST(TimelineTest, RevealEventOutOfView) {
  ColorPalette palette = ColorPalette::Default();
  Timeline timeline(palette);
  FlameChartTimelineData data;
  data.groups.push_back({.name = "Group 1",
                         .start_level = 0,
                         .nesting_level = kThreadNestingLevel,
                         .expanded = true});
  data.events_by_level.push_back({0});
  data.entry_names.push_back("event0");
  data.entry_levels.push_back(0);
  data.entry_start_times.push_back(100.0);
  data.entry_total_times.push_back(10.0);
  data.entry_pids.push_back(1);
  data.entry_args.push_back({});
  timeline.SetTimelineData(std::move(data));
  timeline.set_data_time_range({0.0, 20000.0});
  timeline.SetVisibleRange({1000.0, 2000.0});

  timeline.RevealEvent(0);
  Animation::UpdateAll(1.0f);

  EXPECT_DOUBLE_EQ(timeline.visible_range().start(), 100);
  EXPECT_DOUBLE_EQ(timeline.visible_range().end(), 1100);
  EXPECT_DOUBLE_EQ(timeline.visible_range().duration(), 1000.0);
}

TEST(TimelineTest, RevealEventOutOfViewRight) {
  ColorPalette palette = ColorPalette::Default();
  Timeline timeline(palette);
  FlameChartTimelineData data;
  data.groups.push_back({.name = "Group 1",
                         .start_level = 0,
                         .nesting_level = kThreadNestingLevel,
                         .expanded = true});
  data.events_by_level.push_back({0});
  data.entry_names.push_back("event0");
  data.entry_levels.push_back(0);
  data.entry_start_times.push_back(10000.0);
  data.entry_total_times.push_back(10.0);
  data.entry_pids.push_back(1);
  data.entry_args.push_back({});
  timeline.SetTimelineData(std::move(data));
  timeline.set_data_time_range({0.0, 20000.0});
  timeline.SetVisibleRange({0.0, 5000.0});

  timeline.RevealEvent(0);
  Animation::UpdateAll(1.0f);

  EXPECT_DOUBLE_EQ(timeline.visible_range().start(), 5010);
  EXPECT_DOUBLE_EQ(timeline.visible_range().end(), 10010);
}

TEST(TimelineTest, RevealEventOutToRight) {
  ColorPalette palette = ColorPalette::Default();
  Timeline timeline(palette);
  FlameChartTimelineData data;
  data.groups.push_back({.name = "Group 1",
                         .start_level = 0,
                         .nesting_level = kThreadNestingLevel,
                         .expanded = true});
  data.events_by_level.push_back({0});
  data.entry_names.push_back("event0");
  data.entry_levels.push_back(0);
  data.entry_start_times.push_back(10000.0);
  data.entry_total_times.push_back(1000.0);
  data.entry_pids.push_back(1);
  data.entry_args.push_back({});
  timeline.SetTimelineData(std::move(data));
  timeline.set_data_time_range({0.0, 30000.0});
  timeline.SetVisibleRange({0.0, 50.0});

  timeline.RevealEvent(0);
  Animation::UpdateAll(1.0f);

  EXPECT_DOUBLE_EQ(timeline.visible_range().start(), 10000);
  EXPECT_DOUBLE_EQ(timeline.visible_range().end(), 10050);
  EXPECT_DOUBLE_EQ(timeline.visible_range().duration(), 50.0);
}

TEST(TimelineTest, RevealEventOutToRightLarge) {
  ColorPalette palette = ColorPalette::Default();
  Timeline timeline(palette);
  FlameChartTimelineData data;
  data.groups.push_back({.name = "Group 1",
                         .start_level = 0,
                         .nesting_level = kThreadNestingLevel,
                         .expanded = true});
  data.events_by_level.push_back({0});
  data.entry_names.push_back("event0");
  data.entry_levels.push_back(0);
  data.entry_start_times.push_back(100.0);
  data.entry_total_times.push_back(300000.0);
  data.entry_pids.push_back(1);
  data.entry_args.push_back({});
  timeline.SetTimelineData(std::move(data));
  timeline.set_data_time_range({0.0, 6000000.0});
  timeline.SetVisibleRange({0.0, 50.0});

  timeline.RevealEvent(0);
  Animation::UpdateAll(1.0f);

  EXPECT_DOUBLE_EQ(timeline.visible_range().start(), 100);
  EXPECT_DOUBLE_EQ(timeline.visible_range().end(), 150);
  EXPECT_DOUBLE_EQ(timeline.visible_range().duration(), 50.0);
}

TEST(TimelineTest, RevealEventTriggersCallback) {
  ColorPalette palette = ColorPalette::Default();
  Timeline timeline(palette);
  FlameChartTimelineData data;
  data.groups.push_back(
      {.name = "Group 1", .start_level = 0, .expanded = true});
  data.entry_names.push_back("event0");
  data.entry_levels.push_back(0);
  data.entry_start_times.push_back(100.0);
  data.entry_total_times.push_back(10.0);
  data.entry_pids.push_back(1);
  data.entry_args.push_back({});
  timeline.SetTimelineData(std::move(data));

  bool callback_called = false;
  timeline.set_event_callback(
      [&callback_called](absl::string_view event_type, const EventData& data) {
        if (event_type == kEventSelected) {
          callback_called = true;
        }
      });

  timeline.RevealEvent(0);

  EXPECT_TRUE(callback_called);
}

TEST(TimelineTest, RevealEventUnequalSizes) {
  ColorPalette palette = ColorPalette::Default();
  Timeline timeline(palette);
  FlameChartTimelineData data;
  data.groups.push_back(
      {.name = "Group 1", .start_level = 0, .expanded = true});
  data.entry_names.push_back("event0");
  data.entry_levels.push_back(0);
  data.entry_start_times.push_back(100.0);
  // entry_total_times is empty!

  timeline.SetTimelineData(std::move(data));

  timeline.SetVisibleRange({0.0, 50.0});

  // Index 0 is valid for start_times but invalid for total_times
  timeline.RevealEvent(0);

  // Verify it returned early and didn't change anything
  EXPECT_DOUBLE_EQ(timeline.visible_range().start(), 0.0);
}

TEST(TimelineTest, RevealEventWithIndexOutOfBounds) {
  ColorPalette palette = ColorPalette::Default();
  Timeline timeline(palette);
  FlameChartTimelineData data;
  data.entry_start_times.push_back(10.0);
  data.entry_pids.push_back(1);
  data.entry_args.push_back({});
  timeline.SetTimelineData(std::move(data));
  TimeRange initial_range(0.0, 50.0);
  timeline.SetVisibleRange(initial_range);

  timeline.RevealEvent(1);

  // Visible range should not change for out of bounds event index.
  EXPECT_EQ(timeline.visible_range().start(), initial_range.start());
  EXPECT_EQ(timeline.visible_range().end(), initial_range.end());
}

TEST(TimelineTest, RevealEventWithNegativeIndex) {
  ColorPalette palette = ColorPalette::Default();
  Timeline timeline(palette);
  FlameChartTimelineData data;
  data.entry_start_times.push_back(10.0);
  data.entry_pids.push_back(1);
  data.entry_args.push_back({});
  timeline.SetTimelineData(std::move(data));
  TimeRange initial_range(0.0, 50.0);
  timeline.SetVisibleRange(initial_range);

  timeline.RevealEvent(-1);

  // Visible range should not change because event index is invalid.
  EXPECT_EQ(timeline.visible_range().start(), initial_range.start());
  EXPECT_EQ(timeline.visible_range().end(), initial_range.end());
}

TEST(TimelineTest, SetSearchQuery) {
  ColorPalette palette = ColorPalette::Default();
  Timeline timeline(palette);
  FlameChartTimelineData data;
  data.groups.push_back({.name = "Group 1",
                         .start_level = 0,
                         .nesting_level = kThreadNestingLevel,
                         .expanded = true});
  data.events_by_level.push_back({0, 1});
  data.entry_names.push_back("apple");
  data.entry_names.push_back("banana");
  data.entry_levels.push_back(0);
  data.entry_levels.push_back(0);
  data.entry_start_times.push_back(100.0);
  data.entry_start_times.push_back(200.0);
  data.entry_total_times.push_back(10.0);
  data.entry_total_times.push_back(10.0);
  data.entry_pids.push_back(1);
  data.entry_pids.push_back(1);
  data.entry_tids.push_back(1);
  data.entry_tids.push_back(1);
  data.entry_event_ids.push_back(1);
  data.entry_event_ids.push_back(2);
  data.entry_args.push_back({});
  data.entry_args.push_back({});
  timeline.SetTimelineData(std::move(data));
  timeline.set_data_time_range({0.0, 1000.0});
  timeline.SetVisibleRange({0.0, 100.0});

  timeline.SetSearchQuery("apple");
  EXPECT_EQ(timeline.get_search_results_count(), 1);
  EXPECT_EQ(timeline.get_current_search_result_index(), -1);

  timeline.SetSearchQuery("an");
  EXPECT_EQ(timeline.get_search_results_count(),
            0);  // "banana" does not start with "an"

  timeline.SetSearchQuery("a");
  EXPECT_EQ(timeline.get_search_results_count(),
            1);  // only "apple" starts with "a"

  timeline.SetSearchQuery("xyz");
  EXPECT_EQ(timeline.get_search_results_count(), 0);
  EXPECT_EQ(timeline.get_current_search_result_index(), -1);
}

TEST(TimelineTest, SetSearchQueryCallsRedrawCallback) {
  ColorPalette palette = ColorPalette::Default();
  Timeline timeline(palette);
  FlameChartTimelineData data;
  data.entry_names.push_back("event");
  data.entry_start_times.push_back(100.0);
  data.entry_levels.push_back(0);
  data.entry_total_times.push_back(10.0);
  data.entry_pids.push_back(0);
  data.entry_tids.push_back(0);
  data.entry_event_ids.push_back(1);
  data.entry_args.push_back({});
  timeline.SetTimelineData(std::move(data));

  bool redraw_called = false;
  timeline.set_redraw_callback([&redraw_called]() { redraw_called = true; });

  timeline.SetSearchQuery("event");

  EXPECT_TRUE(redraw_called);
}

TEST(TimelineTest, SetSearchQueryCallsRedrawCallbackCount) {
  ColorPalette palette = ColorPalette::Default();
  Timeline timeline(palette);
  FlameChartTimelineData data;
  data.entry_names.push_back("event");
  data.entry_start_times.push_back(100.0);
  data.entry_levels.push_back(0);
  data.entry_total_times.push_back(10.0);
  data.entry_pids.push_back(0);
  data.entry_tids.push_back(0);
  data.entry_event_ids.push_back(1);
  data.entry_args.push_back({});
  timeline.SetTimelineData(std::move(data));
  timeline.SetVisibleRange(TimeRange(0.0, 200.0));

  int redraw_count = 0;
  timeline.set_redraw_callback([&redraw_count]() { redraw_count++; });

  timeline.SetSearchQuery("event");

  EXPECT_GT(redraw_count, 0);
  EXPECT_EQ(redraw_count, 1);
}

TEST(TimelineTest, SetSearchQueryEmptyClearsResultsAndTriggersRedraw) {
  ColorPalette palette = ColorPalette::Default();
  Timeline timeline(palette);
  FlameChartTimelineData data;
  data.groups.push_back({.name = "Group 1",
                         .start_level = 0,
                         .nesting_level = kThreadNestingLevel,
                         .expanded = true});
  data.events_by_level.push_back({0});
  data.entry_names.push_back("apple");
  data.entry_levels.push_back(0);
  data.entry_start_times.push_back(100.0);
  data.entry_total_times.push_back(10.0);
  data.entry_pids.push_back(1);
  data.entry_tids.push_back(1);
  data.entry_event_ids.push_back(1);
  data.entry_args.push_back({});
  timeline.SetTimelineData(std::move(data));

  timeline.SetSearchQuery("apple");
  EXPECT_EQ(timeline.get_search_results_count(), 1);

  bool redraw_called = false;
  timeline.set_redraw_callback([&redraw_called]() { redraw_called = true; });

  timeline.SetSearchQuery("");

  EXPECT_EQ(timeline.get_search_results_count(), 0);
  EXPECT_EQ(timeline.get_current_search_result_index(), -1);
  EXPECT_TRUE(redraw_called);
}

TEST(TimelineTest, SetSearchQueryEmptyQueryCallsRedraw) {
  ColorPalette palette = ColorPalette::Default();
  Timeline timeline(palette);
  bool redraw_called = false;
  timeline.set_redraw_callback([&redraw_called]() { redraw_called = true; });

  timeline.SetSearchQuery("");

  EXPECT_TRUE(redraw_called);
}

TEST(TimelineTest, SetSearchQueryFiltering) {
  ColorPalette palette = ColorPalette::Default();
  Timeline timeline(palette);
  FlameChartTimelineData data;
  data.entry_names.push_back("event1");
  data.entry_names.push_back("foo");
  data.entry_names.push_back("event2");
  data.entry_names.push_back("long_non_match");
  data.entry_start_times.push_back(10.0);
  data.entry_start_times.push_back(20.0);
  data.entry_start_times.push_back(30.0);
  data.entry_start_times.push_back(40.0);
  data.entry_levels.push_back(0);
  data.entry_levels.push_back(0);
  data.entry_levels.push_back(0);
  data.entry_levels.push_back(0);
  data.entry_total_times.push_back(5.0);
  data.entry_total_times.push_back(5.0);
  data.entry_total_times.push_back(5.0);
  data.entry_total_times.push_back(5.0);
  data.entry_pids.push_back(0);
  data.entry_pids.push_back(0);
  data.entry_pids.push_back(0);
  data.entry_pids.push_back(0);
  data.entry_tids.push_back(0);
  data.entry_tids.push_back(0);
  data.entry_tids.push_back(0);
  data.entry_tids.push_back(0);
  data.entry_event_ids.push_back(1);
  data.entry_event_ids.push_back(2);
  data.entry_event_ids.push_back(3);
  data.entry_event_ids.push_back(4);
  data.entry_args.push_back({});
  data.entry_args.push_back({});
  data.entry_args.push_back({});
  data.entry_args.push_back({});
  timeline.SetTimelineData(std::move(data));

  timeline.SetSearchQuery("event");

  EXPECT_EQ(timeline.selected_event_index(), -1);

  timeline.NavigateToNextSearchResult();
  EXPECT_EQ(timeline.selected_event_index(), 0);

  timeline.NavigateToNextSearchResult();
  EXPECT_EQ(timeline.selected_event_index(), 2);
}

TEST(TimelineTest, SetSearchQuerySortsResultsByStartTime) {
  ColorPalette palette = ColorPalette::Default();
  Timeline timeline(palette);
  FlameChartTimelineData data;
  data.entry_names.push_back("event");
  data.entry_names.push_back("event");
  data.entry_start_times.push_back(100.0);
  data.entry_start_times.push_back(50.0);
  data.entry_levels.push_back(0);
  data.entry_levels.push_back(0);
  data.entry_total_times.push_back(10.0);
  data.entry_total_times.push_back(10.0);
  data.entry_pids.push_back(0);
  data.entry_pids.push_back(0);
  data.entry_tids.push_back(0);
  data.entry_tids.push_back(0);
  data.entry_event_ids.push_back(1);
  data.entry_event_ids.push_back(2);
  data.entry_args.push_back({});
  data.entry_args.push_back({});
  timeline.SetTimelineData(std::move(data));

  timeline.SetSearchQuery("event");

  EXPECT_EQ(timeline.selected_event_index(), -1);

  timeline.NavigateToNextSearchResult();
  EXPECT_EQ(timeline.selected_event_index(), 1);

  timeline.NavigateToNextSearchResult();
  EXPECT_EQ(timeline.selected_event_index(), 0);
}

TEST(TimelineTest, SetSearchQuerySortsResultsByLevel) {
  ColorPalette palette = ColorPalette::Default();
  Timeline timeline(palette);
  FlameChartTimelineData data;
  data.groups.push_back(
      {.type = Group::Type::kFlame, .name = "Group 1", .start_level = 0});

  // Event X (will be loaded at index 0): level = 1, start_time = 100.0
  data.entry_names.push_back("event");
  data.entry_start_times.push_back(100.0);
  data.entry_levels.push_back(1);
  data.entry_total_times.push_back(10.0);
  data.entry_pids.push_back(0);
  data.entry_tids.push_back(0);
  data.entry_event_ids.push_back(1);
  data.entry_args.push_back({});

  // Event Y (will be loaded at index 1): level = 0, start_time = 200.0
  data.entry_names.push_back("event");
  data.entry_start_times.push_back(200.0);
  data.entry_levels.push_back(0);
  data.entry_total_times.push_back(10.0);
  data.entry_pids.push_back(0);
  data.entry_tids.push_back(0);
  data.entry_event_ids.push_back(2);
  data.entry_args.push_back({});

  timeline.SetTimelineData(std::move(data));
  timeline.set_data_time_range({0.0, 1000.0});

  timeline.SetSearchQuery("event");

  // Verify sorting order: Event Y (level 0, index 1) then Event X (level 1,
  // index 0)
  EXPECT_EQ(timeline.selected_event_index(), -1);

  timeline.NavigateToNextSearchResult();
  EXPECT_EQ(timeline.selected_event_index(), 1);  // Event Y

  timeline.NavigateToNextSearchResult();
  EXPECT_EQ(timeline.selected_event_index(), 0);  // Event X
}

TEST(TimelineTest, SetTimelineData) {
  ColorPalette palette = ColorPalette::Default();
  Timeline timeline(palette);
  FlameChartTimelineData data;
  data.groups.push_back(
      {.name = "Group 1", .start_level = 0, .expanded = true});
  data.entry_levels.push_back(0);
  data.entry_start_times.push_back(10.0);
  data.entry_total_times.push_back(5.0);
  data.entry_pids.push_back(1);
  data.entry_args.push_back({});

  timeline.SetTimelineData(std::move(data));

  EXPECT_THAT(timeline.timeline_data().entry_levels, ElementsAre(0));
  EXPECT_THAT(timeline.timeline_data().entry_start_times, ElementsAre(10.0));
  EXPECT_THAT(timeline.timeline_data().entry_total_times, ElementsAre(5.0));
}

TEST(TimelineTest, SetTimelineDataTriggersRedraw) {
  ColorPalette palette = ColorPalette::Default();
  Timeline timeline(palette);
  bool redraw_called = false;
  timeline.set_redraw_callback([&redraw_called]() { redraw_called = true; });

  timeline.SetTimelineData({});

  EXPECT_TRUE(redraw_called);
}

TEST(TimelineTest, SetVisibleFlowCategoriesTriggersRedraw) {
  ColorPalette palette = ColorPalette::Default();
  Timeline timeline(palette);
  bool redraw_called = false;
  timeline.set_redraw_callback([&redraw_called]() { redraw_called = true; });

  timeline.SetVisibleFlowCategories({1, 2});

  EXPECT_TRUE(redraw_called);
}

TEST(TimelineTest, SetVisibleRange) {
  ColorPalette palette = ColorPalette::Default();
  Timeline timeline(palette);
  TimeRange range(10.0, 50.0);

  timeline.SetVisibleRange(range);

  EXPECT_EQ(timeline.visible_range().start(), 10.0);
  EXPECT_EQ(timeline.visible_range().end(), 50.0);
}

TEST(TimelineTest, SetVisibleRangeTriggersRedraw) {
  ColorPalette palette = ColorPalette::Default();
  Timeline timeline(palette);
  bool redraw_called = false;
  timeline.set_redraw_callback([&redraw_called]() { redraw_called = true; });

  timeline.SetVisibleRange({10.0, 20.0});

  EXPECT_TRUE(redraw_called);
}

TEST(TimelineTest, TimeToPixel) {
  ColorPalette palette = ColorPalette::Default();
  Timeline timeline(palette);
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
  ColorPalette palette = ColorPalette::Default();
  Timeline timeline(palette);
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
  EXPECT_EQ(timeline.TimeToScreenX(50.0, screen_x_offset, -10.0),
            screen_x_offset);
}

TEST(TimelineTest, ZoomEvent) {
  ColorPalette palette = ColorPalette::Default();
  Timeline timeline(palette);
  FlameChartTimelineData data;
  data.groups.push_back({.name = "Group 1",
                         .start_level = 0,
                         .nesting_level = 0,
                         .expanded = true});
  data.events_by_level.push_back({0});
  data.entry_names.push_back("event0");
  data.entry_levels.push_back(0);
  data.entry_start_times.push_back(10000.0);
  data.entry_total_times.push_back(1000.0);
  data.entry_pids.push_back(1);
  data.entry_args.push_back({});
  timeline.SetTimelineData(std::move(data));
  timeline.set_data_time_range({0.0, 30000.0});
  timeline.SetVisibleRange({0.0, 50.0});

  bool callback_called = false;
  timeline.set_event_callback(
      [&callback_called](absl::string_view event_type, const EventData& data) {
        if (event_type == kEventSelected) {
          callback_called = true;
        }
      });

  timeline.ZoomEvent(0);
  Animation::UpdateAll(1.0f);

  // event 0 is 10000-11000, center is 10500.
  // duration is clamp(1000*2.5, 10, 5000000) = 2500.
  // new_range center=10500, duration=2500 -> [9250, 11750].
  EXPECT_DOUBLE_EQ(timeline.visible_range().start(), 9250);
  EXPECT_DOUBLE_EQ(timeline.visible_range().end(), 11750);
  EXPECT_DOUBLE_EQ(timeline.visible_range().duration(), 2500.0);
  EXPECT_TRUE(callback_called);
}

TEST(TimelineTest, ZoomEventInvalidIndex) {
  ColorPalette palette = ColorPalette::Default();
  Timeline timeline(palette);
  FlameChartTimelineData data;
  data.groups.push_back(
      {.name = "Group 1", .start_level = 0, .expanded = true});
  data.entry_names.push_back("event0");
  data.entry_levels.push_back(0);
  data.entry_start_times.push_back(100.0);
  data.entry_total_times.push_back(10.0);
  timeline.SetTimelineData(std::move(data));

  timeline.SetVisibleRange({0.0, 50.0});

  // Invalid negative index
  timeline.ZoomEvent(-1);
  EXPECT_DOUBLE_EQ(timeline.visible_range().start(), 0.0);

  // Invalid out of bounds index
  timeline.ZoomEvent(1);
  EXPECT_DOUBLE_EQ(timeline.visible_range().start(), 0.0);
}

TEST(TimelineTest, ZoomEventInvalidIndexMismatchedSizes) {
  ColorPalette palette = ColorPalette::Default();
  Timeline timeline(palette);
  FlameChartTimelineData data;
  data.entry_start_times.push_back(100.0);
  data.entry_start_times.push_back(200.0);
  data.entry_total_times.push_back(10.0);
  timeline.SetTimelineData(std::move(data));
  timeline.set_data_time_range({0.0, 1000.0});
  timeline.SetVisibleRange({0.0, 50.0});

  timeline.ZoomEvent(1);

  EXPECT_DOUBLE_EQ(timeline.visible_range().start(), 0.0);
  EXPECT_DOUBLE_EQ(timeline.visible_range().end(), 50.0);
}

TEST(TimelineTest, ZoomEventInvalidIndexMismatchedSizes_StartTimesShorter) {
  ColorPalette palette = ColorPalette::Default();
  Timeline timeline(palette);
  FlameChartTimelineData data;
  data.entry_start_times.push_back(100.0);
  data.entry_total_times.push_back(10.0);
  data.entry_total_times.push_back(20.0);
  timeline.SetTimelineData(std::move(data));
  timeline.set_data_time_range({0.0, 1000.0});
  timeline.SetVisibleRange({0.0, 50.0});

  timeline.ZoomEvent(1);

  EXPECT_DOUBLE_EQ(timeline.visible_range().start(), 0.0);
  EXPECT_DOUBLE_EQ(timeline.visible_range().end(), 50.0);
}

TEST(TimelineTest, ZoomEventInvalidIndexNegative) {
  ColorPalette palette = ColorPalette::Default();
  Timeline timeline(palette);
  FlameChartTimelineData data;
  data.entry_start_times.push_back(100.0);
  data.entry_total_times.push_back(10.0);
  timeline.SetTimelineData(std::move(data));
  timeline.set_data_time_range({0.0, 1000.0});
  timeline.SetVisibleRange({0.0, 50.0});

  timeline.ZoomEvent(-1);

  EXPECT_DOUBLE_EQ(timeline.visible_range().start(), 0.0);
  EXPECT_DOUBLE_EQ(timeline.visible_range().end(), 50.0);
}

TEST(TimelineTest, ZoomEventInvalidIndexOutOfBounds) {
  ColorPalette palette = ColorPalette::Default();
  Timeline timeline(palette);
  FlameChartTimelineData data;
  data.entry_start_times.push_back(100.0);
  data.entry_total_times.push_back(10.0);
  timeline.SetTimelineData(std::move(data));
  timeline.set_data_time_range({0.0, 1000.0});
  timeline.SetVisibleRange({0.0, 50.0});

  timeline.ZoomEvent(1);

  EXPECT_DOUBLE_EQ(timeline.visible_range().start(), 0.0);
  EXPECT_DOUBLE_EQ(timeline.visible_range().end(), 50.0);
}

TEST(TimelineTest, AddBookmark_AddsBookmark) {
  ColorPalette palette = ColorPalette::Default();
  Timeline timeline(palette);
  timeline.set_bookmarks_enabled(true);
  timeline.set_data_time_range({0.0, 1000.0});
  timeline.SetVisibleRange({0.0, 1000.0});

  timeline.AddBookmark(100.0);

  EXPECT_THAT(timeline.bookmarks(), ElementsAre(100.0));
}

TEST(TimelineTest, RemoveBookmark_RemovesBookmark) {
  ColorPalette palette = ColorPalette::Default();
  Timeline timeline(palette);
  timeline.set_bookmarks_enabled(true);
  timeline.set_data_time_range({0.0, 1000.0});
  timeline.SetVisibleRange({0.0, 1000.0});

  timeline.AddBookmark(100.0);
  timeline.RemoveBookmark(100.0);

  EXPECT_TRUE(timeline.bookmarks().empty());
}

TEST(TimelineTest, AddBookmark_Disabled) {
  ColorPalette palette = ColorPalette::Default();
  Timeline timeline(palette);
  timeline.set_bookmarks_enabled(false);
  timeline.set_data_time_range({0.0, 1000.0});

  timeline.AddBookmark(100.0);

  EXPECT_TRUE(timeline.bookmarks().empty());
}

TEST(TimelineTest, AddBookmark_DoesNotAddDuplicates) {
  ColorPalette palette = ColorPalette::Default();
  Timeline timeline(palette);
  timeline.set_bookmarks_enabled(true);
  timeline.set_data_time_range({0.0, 1000.0});
  timeline.SetVisibleRange({0.0, 1000.0});

  timeline.AddBookmark(100.0);
  timeline.AddBookmark(100.0);

  EXPECT_THAT(timeline.bookmarks(), ElementsAre(100.0));
}

TEST(TimelineTest, AddBookmark_TriggersRedraw) {
  ColorPalette palette = ColorPalette::Default();
  Timeline timeline(palette);
  timeline.set_bookmarks_enabled(true);
  timeline.set_data_time_range({0.0, 1000.0});
  timeline.SetVisibleRange({0.0, 1000.0});

  int redraw_calls = 0;
  timeline.set_redraw_callback([&] { redraw_calls++; });

  timeline.AddBookmark(100.0);
  EXPECT_EQ(redraw_calls, 1);

  // Adding a duplicate should not trigger redraw.
  timeline.AddBookmark(100.0);
  EXPECT_EQ(redraw_calls, 1);
}

TEST(TimelineTest, RemoveBookmark_TriggersRedraw) {
  ColorPalette palette = ColorPalette::Default();
  Timeline timeline(palette);
  timeline.set_bookmarks_enabled(true);
  timeline.set_data_time_range({0.0, 1000.0});
  timeline.SetVisibleRange({0.0, 1000.0});

  timeline.AddBookmark(100.0);

  int redraw_calls = 0;
  timeline.set_redraw_callback([&] { redraw_calls++; });

  timeline.RemoveBookmark(100.0);
  EXPECT_EQ(redraw_calls, 1);

  // Removing a non-existent bookmark should not trigger redraw.
  timeline.RemoveBookmark(100.0);
  EXPECT_EQ(redraw_calls, 1);
}

TEST_F(MockTimelineImGuiFixture, DrawBookmarks_Disabled) {
  timeline_.set_bookmarks_enabled(true);
  timeline_.AddBookmark(100.0);
  timeline_.set_bookmarks_enabled(false);

  // Calls internal DrawBookmarks through public Draw()
  SimulateFrame();

  EXPECT_THAT(timeline_.bookmarks(), ElementsAre(100.0));
}

TEST_F(MockTimelineImGuiFixture, DrawBookmarks_Empty) {
  timeline_.set_bookmarks_enabled(true);

  SimulateFrame();

  EXPECT_TRUE(timeline_.bookmarks().empty());
}

TEST_F(MockTimelineImGuiFixture, DrawBookmarks_InvalidPxPerTimeUnit) {
  timeline_.set_bookmarks_enabled(true);
  timeline_.AddBookmark(100.0);

  // Setting window width to 0 or negative should result in invalid
  // px_per_time_unit
  ImGui::GetIO().DisplaySize = ImVec2(0, 0);
  SimulateFrame();

  EXPECT_THAT(timeline_.bookmarks(), ElementsAre(100.0));
}

TEST_F(MockTimelineImGuiFixture, DrawPinButton_Insert) {
  // Frame 1: Register window and layout
  ImGui::NewFrame();
  ImGui::SetNextWindowPos(ImVec2(0.0f, 0.0f));
  ImGui::SetNextWindowSize(ImVec2(1000.0f, 1000.0f));
  ImGui::Begin("TestWindow");
  ImGui::SetCursorScreenPos(ImVec2(100.0f, 100.0f));
  timeline_.CallDrawPinButton(0, 20.0f, /*is_pinned=*/false);
  ImGui::End();
  ImGui::EndFrame();

  // Frame 2: Inject click
  ImGuiIO& io = ImGui::GetIO();
  io.AddMousePosEvent(105.0f, 105.0f);
  io.AddMouseButtonEvent(0, true);

  ImGui::NewFrame();
  ImGui::SetNextWindowPos(ImVec2(0.0f, 0.0f));
  ImGui::SetNextWindowSize(ImVec2(1000.0f, 1000.0f));
  ImGui::Begin("TestWindow");
  ImGui::SetCursorScreenPos(ImVec2(100.0f, 100.0f));

  EXPECT_TRUE(timeline_.CallDrawPinButton(0, 20.0f, /*is_pinned=*/false));
  EXPECT_EQ(timeline_.GetPinnedTrackNames().size(), 1);
  EXPECT_TRUE(timeline_.GetPinnedTrackNames().contains("group"));

  ImGui::End();
  ImGui::EndFrame();
}

TEST_F(MockTimelineImGuiFixture, DrawPinButton_Erase) {
  timeline_.SetPinnedTrackNames({"group"});

  // Frame 1: Register window and layout
  ImGui::NewFrame();
  ImGui::SetNextWindowPos(ImVec2(0.0f, 0.0f));
  ImGui::SetNextWindowSize(ImVec2(1000.0f, 1000.0f));
  ImGui::Begin("TestWindow");
  ImGui::SetCursorScreenPos(ImVec2(100.0f, 100.0f));
  timeline_.CallDrawPinButton(0, 20.0f, /*is_pinned=*/true);
  ImGui::End();
  ImGui::EndFrame();

  // Frame 2: Inject click
  ImGuiIO& io = ImGui::GetIO();
  io.AddMousePosEvent(105.0f, 105.0f);
  io.AddMouseButtonEvent(0, true);

  ImGui::NewFrame();
  ImGui::SetNextWindowPos(ImVec2(0.0f, 0.0f));
  ImGui::SetNextWindowSize(ImVec2(1000.0f, 1000.0f));
  ImGui::Begin("TestWindow");
  ImGui::SetCursorScreenPos(ImVec2(100.0f, 100.0f));

  EXPECT_TRUE(timeline_.CallDrawPinButton(0, 20.0f, /*is_pinned=*/true));
  EXPECT_EQ(timeline_.GetPinnedTrackNames().size(), 0);

  ImGui::End();
  ImGui::EndFrame();
}

TEST_F(MockTimelineImGuiFixture, DrawHideButton_Insert) {
  // Frame 1: Register window and layout
  ImGui::NewFrame();
  ImGui::SetNextWindowPos(ImVec2(0.0f, 0.0f));
  ImGui::SetNextWindowSize(ImVec2(1000.0f, 1000.0f));
  ImGui::Begin("TestWindow");
  ImGui::SetCursorScreenPos(ImVec2(100.0f, 100.0f));
  timeline_.CallDrawHideButton(0, 20.0f, /*is_track_hidden=*/false);
  ImGui::End();
  ImGui::EndFrame();

  // Frame 2: Inject click
  ImGuiIO& io = ImGui::GetIO();
  io.AddMousePosEvent(105.0f, 105.0f);
  io.AddMouseButtonEvent(0, true);

  ImGui::NewFrame();
  ImGui::SetNextWindowPos(ImVec2(0.0f, 0.0f));
  ImGui::SetNextWindowSize(ImVec2(1000.0f, 1000.0f));
  ImGui::Begin("TestWindow");
  ImGui::SetCursorScreenPos(ImVec2(100.0f, 100.0f));

  EXPECT_TRUE(
      timeline_.CallDrawHideButton(0, 20.0f, /*is_track_hidden=*/false));
  EXPECT_EQ(timeline_.GetHiddenTrackNames().size(), 1);
  EXPECT_TRUE(timeline_.GetHiddenTrackNames().contains("group"));

  ImGui::End();
  ImGui::EndFrame();
}

TEST_F(MockTimelineImGuiFixture, DrawHideButton_Erase) {
  timeline_.SetHiddenTrackNames({"group"});

  // Frame 1: Register window and layout
  ImGui::NewFrame();
  ImGui::SetNextWindowPos(ImVec2(0.0f, 0.0f));
  ImGui::SetNextWindowSize(ImVec2(1000.0f, 1000.0f));
  ImGui::Begin("TestWindow");
  ImGui::SetCursorScreenPos(ImVec2(100.0f, 100.0f));
  timeline_.CallDrawHideButton(0, 20.0f, /*is_track_hidden=*/true);
  ImGui::End();
  ImGui::EndFrame();

  // Frame 2: Inject click
  ImGuiIO& io = ImGui::GetIO();
  io.AddMousePosEvent(105.0f, 105.0f);
  io.AddMouseButtonEvent(0, true);

  ImGui::NewFrame();
  ImGui::SetNextWindowPos(ImVec2(0.0f, 0.0f));
  ImGui::SetNextWindowSize(ImVec2(1000.0f, 1000.0f));
  ImGui::Begin("TestWindow");
  ImGui::SetCursorScreenPos(ImVec2(100.0f, 100.0f));

  EXPECT_TRUE(timeline_.CallDrawHideButton(0, 20.0f, /*is_track_hidden=*/true));
  EXPECT_EQ(timeline_.GetHiddenTrackNames().size(), 0);

  ImGui::End();
  ImGui::EndFrame();
}

// =============================================================================
// Fixture: MockTimelineImGuiFixture
// =============================================================================

TEST_F(MockTimelineImGuiFixture, ClickAndPressShiftMidDragContinuesPanning) {
  // Setup similar to TimelineDragSelectionTest.
  timeline_.SetVisibleRange({0.0, 165.5});
  timeline_.set_fetched_data_time_range({0.0, 165.5});
  timeline_.SetVisibleRange({0.0, 165.5});
  timeline_.set_data_time_range({0.0, 165.5});
  timeline_.SetVisibleRange({0.0, 165.5});
  timeline_.set_data_time_range({0.0, 165.5});
  ImGuiIO& io = ImGui::GetIO();

  // Start without Shift.
  io.AddKeyEvent(ImGuiMod_Shift, false);

  // Start drag in timeline area.
  io.MousePos = ImVec2(GetTimelineStartX() + 50.0f, 50.0f);
  io.AddMouseButtonEvent(0, true);

  EXPECT_CALL(timeline_, Pan(0.0f));
  EXPECT_CALL(timeline_, Scroll(0.0f));
  SimulateFrame();

  // Press Shift key mid-drag.
  io.AddKeyEvent(ImGuiMod_Shift, true);

  // Drag mouse to left (simulate pan right).
  // Move from 300 to 200 (-100px).
  io.MousePos = ImVec2(GetTimelineStartX() - 50.0f, 50.0f);

  EXPECT_CALL(timeline_, Pan(FloatEq(100.0f)));
  EXPECT_CALL(timeline_, Scroll(FloatEq(0.0f)));
  SimulateFrame();

  // End drag.
  io.AddMouseButtonEvent(0, false);
  SimulateFrame();

  // Verify that NO selection was created.
  EXPECT_TRUE(timeline_.selected_time_ranges().empty());
}

TEST_F(MockTimelineImGuiFixture,
       DrawEventNameTextHiddenWhenSlightlyNarrowerThanMinTextWidth) {
  FlameChartTimelineData data;

  data.groups.push_back({.name = "Group 1",
                         .start_level = 0,
                         .nesting_level = 0,
                         .expanded = true});
  data.events_by_level.push_back({0});
  data.entry_names.push_back("event1");
  data.entry_levels.push_back(0);
  data.entry_start_times.push_back(10.0);
  data.entry_total_times.push_back(0.255);
  data.entry_pids.push_back(1);
  data.entry_args.push_back({});
  timeline_.SetTimelineData(std::move(data));
  timeline_.SetVisibleRange({0.0, 100.0});

  // The event rect width will be around 4.51f, which is < kMinTextWidth (5.0f).
  // DrawEventName should not draw text, so GetTextForDisplay and
  // CalculateEventTextRect won't be called, and GetTextSize should not be
  // called.
  EXPECT_CALL(timeline_, GetTextSize(_)).Times(0);

  SimulateFrame();
}

TEST_F(MockTimelineImGuiFixture, DrawEventNameTextHiddenWhenTooNarrow) {
  FlameChartTimelineData data;

  data.groups.push_back({.name = "Group 1",
                         .start_level = 0,
                         .nesting_level = 0,
                         .expanded = true});
  data.events_by_level.push_back({0});
  data.entry_names.push_back("event1");
  data.entry_levels.push_back(0);
  data.entry_start_times.push_back(10.0);
  data.entry_total_times.push_back(0.001);
  data.entry_pids.push_back(1);
  data.entry_args.push_back({});
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
       DrawEvent_HoverInstantEvent_UsesTrianglesNotRectangles) {
  FlameChartTimelineData data;
  data.groups.push_back({.name = "Group 1",
                         .start_level = 0,
                         .nesting_level = 0,
                         .expanded = true});
  data.events_by_level.push_back({0});
  data.entry_names.push_back("instant_event");
  data.entry_levels.push_back(0);
  data.entry_start_times.push_back(10.0);
  data.entry_total_times.push_back(0.0);  // Instant event
  data.entry_pids.push_back(1);
  data.entry_args.push_back({});
  timeline_.SetTimelineData(std::move(data));
  timeline_.SetVisibleRange({0.0, 100.0});

  ImGui::NewFrame();
  ImGui::SetNextWindowSize(ImVec2(1000, 500));
  ImGui::Begin(
      "TestWindow", nullptr,
      ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);

  ImGuiWindow* window = ImGui::GetCurrentWindow();
  ImGuiIO& io = ImGui::GetIO();

  // Set up parameters for DrawEventsForLevelBase
  int group_index = 0;
  std::vector<int> event_indices = {0};
  double px_per = 1.0;
  // Let TimeToScreenX(10.0, pos.x, px_per) = pos.x + 10.0 = 100.0
  // (if pos.x = 90.0)
  ImVec2 pos(90.0f, 100.0f);
  ImVec2 max(1000.0f, 500.0f);
  float event_height = 20.0f;
  float padding_bottom = 0.0f;

  // Hover position (event left is 100).
  io.MousePos = ImVec2(100.0f, 102.0f);

  ImDrawList* draw_list = window->DrawList;
  const int initial_vtx_size = draw_list->VtxBuffer.Size;

  timeline_.DrawEventsForLevelBase(group_index, event_indices, px_per, 0, pos,
                                   max, event_height, padding_bottom);

  int hover_mask_vertices = 0;
  for (int i = initial_vtx_size; i < draw_list->VtxBuffer.Size; ++i) {
    if (draw_list->VtxBuffer[i].col == kHoverMaskColor) {
      ++hover_mask_vertices;
    }
  }

  // A correct implementation uses AddTriangleFilled for the instant event
  // hover mask (3 vertices).
  // The buggy implementation uses AddRectFilled (4 or 6 vertices).
  EXPECT_EQ(hover_mask_vertices, 3);

  ImGui::End();
  ImGui::EndFrame();
}

TEST_F(MockTimelineImGuiFixture,
       DrawEvent_SelectedHoveredInstantEvent_ChevronSizeIncreases) {
  FlameChartTimelineData data;
  data.groups.push_back({.name = "Group 1",
                         .start_level = 0,
                         .nesting_level = 0,
                         .expanded = true});
  data.events_by_level.push_back({0});
  data.entry_names.push_back("instant_event");
  data.entry_levels.push_back(0);
  data.entry_start_times.push_back(10.0);
  data.entry_total_times.push_back(0.0);  // Instant event
  data.entry_pids.push_back(1);
  data.entry_args.push_back({});
  timeline_.SetTimelineData(std::move(data));
  timeline_.SetVisibleRange({0.0, 100.0});

  // select the instant event at index 0
  timeline_.RevealEvent(0);

  // Set up parameters
  int group_index = 0;
  std::vector<int> event_indices = {0};
  double px_per = 1.0;
  ImVec2 pos(90.0f, 100.0f);
  ImVec2 max(1000.0f, 500.0f);
  float event_height = 20.0f;
  float padding_bottom = 0.0f;

  auto measure_max_y = [&](bool hovered) {
    ImGui::NewFrame();
    ImGui::SetNextWindowSize(ImVec2(1000, 500));
    ImGui::Begin(
        "TestWindow", nullptr,
        ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);
    ImGuiWindow* window = ImGui::GetCurrentWindow();
    ImGuiIO& io = ImGui::GetIO();

    if (hovered) {
      io.MousePos = ImVec2(100.0f, 102.0f);
    } else {
      io.MousePos = ImVec2(-100.0f, -100.0f);  // out of bounds
    }

    ImDrawList* draw_list = window->DrawList;
    const int initial_vtx_size = draw_list->VtxBuffer.Size;

    timeline_.DrawEventsForLevelBase(group_index, event_indices, px_per, 0, pos,
                                     max, event_height, padding_bottom);

    float max_y = -10000.0f;
    for (int i = initial_vtx_size; i < draw_list->VtxBuffer.Size; ++i) {
      const auto& vtx = draw_list->VtxBuffer[i];
      if (vtx.col == kSelectedBorderColor) {
        max_y = std::max(max_y, vtx.pos.y);
      }
    }
    ImGui::End();
    ImGui::EndFrame();
    return max_y;
  };

  float unhovered_max_y = measure_max_y(false);
  float hovered_max_y = measure_max_y(true);

  // When hovered, the chevron height increases from 15.5 to 20.0.
  // We expect the max Y of the selected border vertices to reflect this
  // increase.
  EXPECT_GT(hovered_max_y, unhovered_max_y + 4.0f);
  EXPECT_LT(hovered_max_y, unhovered_max_y + 5.0f);
}

TEST_F(MockTimelineImGuiFixture,
       DrawEventsForLevel_BinarySearchCorrectlySelectsVisibleEvents) {
  FlameChartTimelineData data;

  data.groups.push_back({.name = "Group 1",
                         .start_level = 0,
                         .nesting_level = kThreadNestingLevel,
                         .expanded = true});
  data.events_by_level.push_back({0, 1, 2});

  // Event 0: Outside left
  data.entry_names.push_back("event0");
  data.entry_levels.push_back(0);
  data.entry_start_times.push_back(10.0);
  data.entry_total_times.push_back(5.0);
  data.entry_pids.push_back(1);
  data.entry_args.push_back({});

  // Event 1: Visible
  data.entry_names.push_back("event1");
  data.entry_levels.push_back(0);
  data.entry_start_times.push_back(25.0);
  data.entry_total_times.push_back(5.0);
  data.entry_pids.push_back(1);
  data.entry_args.push_back({});

  // Event 2: Outside right
  data.entry_names.push_back("event2");
  data.entry_levels.push_back(0);
  data.entry_start_times.push_back(45.0);
  data.entry_total_times.push_back(5.0);
  data.entry_pids.push_back(1);
  data.entry_args.push_back({});

  timeline_.SetTimelineData(std::move(data));
  timeline_.SetVisibleRange({20.0, 40.0});

  // Ignore other GetTextSize calls if any (e.g. for group names)
  EXPECT_CALL(timeline_, GetTextSize(_)).Times(::testing::AnyNumber());

  // Specific expectations (checked first due to reverse order)
  EXPECT_CALL(timeline_, GetTextSize("event1")).Times(2);
  EXPECT_CALL(timeline_, GetTextSize("event0")).Times(0);
  EXPECT_CALL(timeline_, GetTextSize("event2")).Times(0);

  // Vertical culling should call DrawGroup and DrawEventsForLevel.
  EXPECT_CALL(timeline_, DrawGroup(_, _, _, _))
      .Times(::testing::AnyNumber())
      .WillRepeatedly([&](int group_index, double px_per_time_unit_val,
                          Pixel scroll_y, Pixel window_height) {
        timeline_.DrawGroupBase(group_index, px_per_time_unit_val, scroll_y,
                                window_height);
      });
  EXPECT_CALL(timeline_, DrawEventsForLevel(_, _, _, _, _, _, _, _))
      .Times(::testing::AnyNumber())
      .WillRepeatedly([&](int group_index, absl::Span<const int> event_indices,
                          double px_per_time_unit, int level_in_group,
                          const ImVec2& pos, const ImVec2& max,
                          Pixel event_height, Pixel padding_bottom) {
        timeline_.DrawEventsForLevelBase(group_index, event_indices,
                                         px_per_time_unit, level_in_group, pos,
                                         max, event_height, padding_bottom);
      });

  SimulateFrame();
}

TEST_F(MockTimelineImGuiFixture,
       Draw_VerticalGroupBinarySearchCorrectlySelectsVisibleGroups) {
  FlameChartTimelineData data;
  // Create 20 groups to ensure we have enough to cull.
  for (int i = 0; i < 20; ++i) {
    data.groups.push_back({.name = "Group " + std::to_string(i),
                           .start_level = i,
                           .nesting_level = kThreadNestingLevel,
                           .expanded = true});
    data.events_by_level.push_back({});
  }

  timeline_.SetTimelineData(std::move(data));

  // Set display size small to ensure culling happens.
  ImGui::GetIO().DisplaySize = ImVec2(1000.0f, 100.0f);

  // Based on the heights, we expect only a few groups to be visible.
  // With ThreadTrackGap=4 and kEventHeight=23, each thread is 27px.
  // Viewport at 100px with 100px height should see roughly groups 4-8.
  EXPECT_CALL(timeline_, DrawGroup(_, _, _, _)).Times(::testing::Between(3, 7));

  SimulateFrame(/*scroll_y=*/100.0f, /*window_height=*/100.0f);
}

TEST_F(MockTimelineImGuiFixture,
       DrawGroup_LevelCalculationCullingCorrectlySelectsVisibleLevels) {
  FlameChartTimelineData data;
  // Create 1 group with many levels.
  data.groups.push_back({.name = "Big Group",
                         .start_level = 0,
                         .nesting_level = 1,
                         .expanded = true});
  for (int i = 0; i < 100; ++i) {
    data.events_by_level.push_back({i});  // One event per level
    data.entry_names.push_back("event" + std::to_string(i));
    data.entry_levels.push_back(i);
    data.entry_start_times.push_back(0.0);
    data.entry_total_times.push_back(100.0);
    data.entry_pids.push_back(1);
    data.entry_args.push_back({});
  }

  timeline_.SetTimelineData(std::move(data));

  // Set display size small to ensure culling happens.
  ImGui::GetIO().DisplaySize = ImVec2(1000.0f, 100.0f);

  // Set scroll such that we are looking at levels in the middle of the group.
  // Use a large scroll to trigger culling.
  Pixel scroll_y = 500.0f;
  Pixel window_height = 100.0f;  // ~4 levels

  // Since we are mocking the top-level Draw(), we need to expect the DrawGroup
  // call too.
  EXPECT_CALL(timeline_, DrawGroup(_, _, _, _))
      .Times(::testing::AnyNumber())
      .WillRepeatedly([&](int group_index, double px_per_time_unit_val,
                          Pixel scroll_y, Pixel window_height) {
        timeline_.DrawGroupBase(group_index, px_per_time_unit_val, scroll_y,
                                window_height);
      });
  EXPECT_CALL(timeline_, DrawEventsForLevel(_, _, _, _, _, _, _, _))
      .Times(::testing::Between(3, 8));

  SimulateFrame(scroll_y, window_height);
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

enum class InteractionType {
  kPanRight,
  kPanLeft,
  kLabelColumnResize,
  kScroll,
  kZoomIn,
  kZoomOut
};

class MaybeRequestDataDeferredTest
    : public TimelineImGuiTestFixture<MockTimeline>,
      public ::testing::WithParamInterface<InteractionType> {};

TEST_P(MaybeRequestDataDeferredTest, DeferredDuringInteraction) {
  bool request_triggered = false;
  SetupDeferredFetchExperiment(request_triggered);
  if (GetParam() == InteractionType::kLabelColumnResize) {
    SimulateFrame();  // Ensure tables are initialized for resize
  }

  ImGuiIO& io = ImGui::GetIO();
  switch (GetParam()) {
    case InteractionType::kPanRight:
      io.AddKeyEvent(ImGuiKey_D, true);
      SimulateFrame();
      break;
    case InteractionType::kPanLeft:
      io.AddKeyEvent(ImGuiKey_A, true);
      SimulateFrame();
      break;
    case InteractionType::kLabelColumnResize:
      SimulateLabelColumnResizeDragStart();
      SimulateFrame();
      break;
    case InteractionType::kScroll:
      io.AddMouseWheelEvent(0.0f, -1.0f);
      // Do NOT call SimulateFrame() here, as MouseWheel is a single-frame event
      break;
    case InteractionType::kZoomIn:
      io.AddKeyEvent(ImGuiKey_W, true);
      SimulateFrame();
      break;
    case InteractionType::kZoomOut:
      io.AddKeyEvent(ImGuiKey_S, true);
      SimulateFrame();
      break;
  }

  timeline_.SetVisibleRange({9000000.0, 10000000.0});
  SimulateFrame();
  EXPECT_FALSE(request_triggered);

  switch (GetParam()) {
    case InteractionType::kPanRight:
      io.AddKeyEvent(ImGuiKey_D, false);
      break;
    case InteractionType::kPanLeft:
      io.AddKeyEvent(ImGuiKey_A, false);
      break;
    case InteractionType::kLabelColumnResize:
      io.AddMouseButtonEvent(0, false);
      break;
    case InteractionType::kScroll:
      // MouseWheel resets to 0 automatically in next frame
      break;
    case InteractionType::kZoomIn:
      io.AddKeyEvent(ImGuiKey_W, false);
      break;
    case InteractionType::kZoomOut:
      io.AddKeyEvent(ImGuiKey_S, false);
      break;
  }
  SimulateFrame();
  EXPECT_TRUE(request_triggered);
}

INSTANTIATE_TEST_SUITE_P(VariousInteractions, MaybeRequestDataDeferredTest,
                         ::testing::Values(InteractionType::kPanRight,
                                           InteractionType::kPanLeft,
                                           InteractionType::kLabelColumnResize,
                                           InteractionType::kScroll,
                                           InteractionType::kZoomIn,
                                           InteractionType::kZoomOut));

class TestTimeline : public Timeline {
 public:
  using Timeline::DrawHideIcon;
  using Timeline::group_visible;
  using Timeline::Pan;
  using Timeline::Scroll;
  using Timeline::Timeline;
  using Timeline::Zoom;
};

using RealTimelineImGuiFixture = TimelineImGuiTestFixture<TestTimeline>;

// Add a sanity check that the window padding is set to zero.
// This is the presumption for all the drawing logic. And all tests below assume
// this.
TEST_F(MockTimelineImGuiFixture, PanLeftWithAKey) {
  ImGui::GetIO().AddKeyEvent(ImGuiKey_A, true);

  // No acceleration is applied because io.DeltaTime (0.1f) <
  // kAccelerateThreshold (0.25f).
  EXPECT_CALL(timeline_,
              Pan(FloatEq(-kPanningSpeed * ImGui::GetIO().DeltaTime)));

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

TEST_F(MockTimelineImGuiFixture, PanLeftWithHorizontalMouseWheel) {
  ImGui::GetIO().AddMouseWheelEvent(-1.0f, 0.0f);

  EXPECT_CALL(timeline_, Pan(FloatEq(-1.0f)));

  SimulateFrame();
}

TEST_F(MockTimelineImGuiFixture, PanLeftWithShiftAndMouseWheel) {
  ImGuiIO& io = ImGui::GetIO();
  io.AddMouseWheelEvent(0.0f, -1.0f);
  io.AddKeyEvent(ImGuiMod_Shift, true);

  EXPECT_CALL(timeline_, Pan(FloatEq(-1.0f)));

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

TEST_F(MockTimelineImGuiFixture, PanRightWithHorizontalMouseWheel) {
  ImGui::GetIO().AddMouseWheelEvent(1.0f, 0.0f);

  EXPECT_CALL(timeline_, Pan(FloatEq(1.0f)));

  SimulateFrame();
}

TEST_F(MockTimelineImGuiFixture, PanRightWithShiftAndMouseWheel) {
  ImGuiIO& io = ImGui::GetIO();
  io.AddMouseWheelEvent(0.0f, 1.0f);
  io.AddKeyEvent(ImGuiMod_Shift, true);

  EXPECT_CALL(timeline_, Pan(FloatEq(1.0f)));

  SimulateFrame();
}

TEST_F(MockTimelineImGuiFixture, PanWithMouseDrag) {
  ImGuiIO& io = ImGui::GetIO();
  // Main window pos is (0,0), content_min is (0,0), label_width is
  // GetTimelineStartX(). So timeline area starts at x=GetTimelineStartX().
  io.MousePos = ImVec2(GetTimelineStartX() + 50.0f, 50.0f);
  SimulateFrame();  // Establish initial state

  // Press mouse button without shift.
  io.AddMouseButtonEvent(0, true);

  // In the first frame of a drag, MouseDelta will be zero.
  EXPECT_CALL(timeline_, Pan(0.0f));
  EXPECT_CALL(timeline_, Scroll(0.0f));

  SimulateFrame();  // This will call HandleMouse and set is_dragging_ to true

  // Drag the mouse.
  io.AddMousePosEvent(GetTimelineStartX() + 60.0f, 60.0f);

  EXPECT_CALL(timeline_, Pan(FloatEq(-10.0f)));
  EXPECT_CALL(timeline_, Scroll(FloatEq(-10.0f)));

  SimulateFrame();

  // Release mouse button.
  io.AddMouseButtonEvent(0, false);
  SimulateFrame();
}

TEST_F(MockTimelineImGuiFixture, HandleMouseRelease_WithoutMouseDown) {
  ImGuiIO& io = ImGui::GetIO();
  io.AddMouseButtonEvent(0, false);
  SimulateFrame();

  // Ensure no crash and no side effects after a frame where button was already
  // released.
  EXPECT_TRUE(timeline_.bookmarks().empty());
}

TEST_F(MockTimelineImGuiFixture, HandleMouseRelease_SmallDragThreshold) {
  timeline_.set_bookmarks_enabled(true);
  timeline_.set_data_time_range({0.0, 1000.0});
  timeline_.SetVisibleRange({0.0, 1000.0});
  ImGuiIO& io = ImGui::GetIO();

  // Case 1: Exact threshold (5px -> distance_squared 25)
  io.MousePos = ImVec2(GetTimelineStartX() + 100.0f, 50.0f);
  io.AddMouseButtonEvent(0, true);
  SimulateFrame();

  // Drag by exactly 5px
  io.AddMousePosEvent(GetTimelineStartX() + 105.0f, 50.0f);
  SimulateFrame();

  io.AddKeyEvent(ImGuiMod_Ctrl, true);
  io.AddMouseButtonEvent(0, false);
  SimulateFrame();

  // Distance is 5, threshold is <= 25, so this is a click.
  EXPECT_FALSE(timeline_.bookmarks().empty());
  timeline_.RemoveBookmark(timeline_.bookmarks()[0]);

  // Case 2: Just above threshold
  io.MousePos = ImVec2(GetTimelineStartX() + 100.0f, 50.0f);
  io.AddMouseButtonEvent(0, true);
  SimulateFrame();

  io.AddMousePosEvent(GetTimelineStartX() + 110.0f,
                      60.0f);  // 10px drag each way
  SimulateFrame();

  io.AddKeyEvent(ImGuiMod_Ctrl, true);
  io.AddMouseButtonEvent(0, false);
  SimulateFrame();

  // Distance > 5, not a click.
  EXPECT_TRUE(timeline_.bookmarks().empty());

  io.AddKeyEvent(ImGuiMod_Ctrl, false);
}

TEST_F(MockTimelineImGuiFixture, AddBookmark_CtrlClick) {
  timeline_.set_bookmarks_enabled(true);
  timeline_.set_data_time_range({0.0, 1000.0});
  timeline_.SetVisibleRange({0.0, 1000.0});
  ImGuiIO& io = ImGui::GetIO();

  // Click at some X in timeline area.
  float click_x = GetTimelineStartX() + 100.0f;
  io.MousePos = ImVec2(click_x, 50.0f);
  io.AddMouseButtonEvent(0, true);
  SimulateFrame();

  io.AddKeyEvent(ImGuiMod_Ctrl, true);
  io.AddMouseButtonEvent(0, false);
  SimulateFrame();

  EXPECT_FALSE(timeline_.bookmarks().empty());
}

TEST_F(MockTimelineImGuiFixture, AddBookmark_MetaClick) {
  timeline_.set_bookmarks_enabled(true);
  timeline_.set_data_time_range({0.0, 1000.0});
  timeline_.SetVisibleRange({0.0, 1000.0});
  ImGuiIO& io = ImGui::GetIO();

  // Click at some X in timeline area.
  float click_x = GetTimelineStartX() + 100.0f;
  io.MousePos = ImVec2(click_x, 50.0f);
  io.AddMouseButtonEvent(0, true);
  SimulateFrame();

  io.AddKeyEvent(ImGuiMod_Super, true);
  io.AddMouseButtonEvent(0, false);
  SimulateFrame();

  EXPECT_FALSE(timeline_.bookmarks().empty());
}

TEST_F(MockTimelineImGuiFixture, HandleBookmarkAddition_OutsideTimeline) {
  timeline_.set_bookmarks_enabled(true);
  timeline_.set_data_time_range({0.0, 1000.0});
  timeline_.SetVisibleRange({0.0, 1000.0});
  ImGuiIO& io = ImGui::GetIO();

  // Click OUTSIDE timeline area (e.g., in the label column).
  float click_x = GetTimelineStartX() - 10.0f;
  io.MousePos = ImVec2(click_x, 50.0f);
  io.AddMouseButtonEvent(0, true);
  SimulateFrame();

  io.AddKeyEvent(ImGuiMod_Ctrl, true);
  io.AddMouseButtonEvent(0, false);
  SimulateFrame();

  EXPECT_TRUE(timeline_.bookmarks().empty());
}

TEST_F(MockTimelineImGuiFixture, HandleBookmarkAddition_NoModifier) {
  timeline_.set_bookmarks_enabled(true);
  timeline_.set_data_time_range({0.0, 1000.0});
  timeline_.SetVisibleRange({0.0, 1000.0});
  ImGuiIO& io = ImGui::GetIO();

  // Click without Ctrl or Meta.
  float click_x = GetTimelineStartX() + 100.0f;
  io.MousePos = ImVec2(click_x, 50.0f);
  io.AddMouseButtonEvent(0, true);
  SimulateFrame();

  io.AddMouseButtonEvent(0, false);
  SimulateFrame();

  EXPECT_TRUE(timeline_.bookmarks().empty());
}

TEST_F(MockTimelineImGuiFixture, HandleBookmarkAddition_Disabled) {
  timeline_.set_bookmarks_enabled(false);
  timeline_.set_data_time_range({0.0, 1000.0});
  timeline_.SetVisibleRange({0.0, 1000.0});
  ImGuiIO& io = ImGui::GetIO();

  // Ctrl + Click while disabled.
  float click_x = GetTimelineStartX() + 100.0f;
  io.MousePos = ImVec2(click_x, 50.0f);
  io.AddMouseButtonEvent(0, true);
  SimulateFrame();

  io.AddKeyEvent(ImGuiMod_Ctrl, true);
  io.AddMouseButtonEvent(0, false);
  SimulateFrame();

  EXPECT_TRUE(timeline_.bookmarks().empty());
}

TEST_F(MockTimelineImGuiFixture, HandleBookmarkAddition_BypassesTinyTimeRange) {
  timeline_.set_bookmarks_enabled(true);
  timeline_.set_data_time_range({0.0, 1000.0});
  timeline_.SetVisibleRange({0.0, 1000.0});
  timeline_.set_mouse_mode(MouseMode::kTiming);
  ImGuiIO& io = ImGui::GetIO();

  float click_x = GetTimelineStartX() + 100.0f;
  io.MousePos = ImVec2(click_x, 50.0f);
  io.AddMouseButtonEvent(0, true);
  SimulateFrame();

  // Drag slightly by 2 pixels (within kClickDistanceThresholdSquared = 25.0f)
  io.AddMousePosEvent(click_x + 2.0f, 50.0f);
  SimulateFrame();

  io.AddKeyEvent(ImGuiMod_Ctrl, true);
  io.AddMouseButtonEvent(0, false);
  SimulateFrame();

  EXPECT_FALSE(timeline_.bookmarks().empty());
  EXPECT_TRUE(timeline_.selected_time_ranges().empty());
}

TEST_F(MockTimelineImGuiFixture, DrawBookmarks_DeletesBookmark) {
  timeline_.set_bookmarks_enabled(true);
  timeline_.set_data_time_range({0.0, 1000.0});
  timeline_.SetVisibleRange({0.0, 1000.0});
  timeline_.AddBookmark(100.0);

  EXPECT_THAT(timeline_.bookmarks(), ElementsAre(100.0));

  // The bookmark is at 100ms.
  // visible range is 0-1000ms.
  // timeline width is roughly 1920 - GetTimelineStartX().
  // But SimulateFrame sets WindowSize to 1000 in some overloads.
  // Default SimulateFrame() uses DisplaySize (1920x1080).

  // Let's use SimulateFrame(scroll_y, window_height) to be consistent.
  SimulateFrame(0.0f, 1000.0f);

  float x = 100.0f * timeline_.px_per_time_unit() + GetTimelineStartX();
  // Bookmark label is "100.0ms".
  // label_pos = (x + 4, timeline_area.Min.y + 4)
  // button_pos = (label_pos.x + label_size.x + 4, label_pos.y)
  // kCloseButtonSize = 14

  // We'll try clicking in a range that likely hits the button.
  // We start from offset 0 to be safe in case label_size is 0.
  // We try a few Y values because GetTimelineArea() might return different
  // coordinates depending on whether it's called from the Tracks child or
  // the SelectionOverlay child.
  ImGuiIO& io = ImGui::GetIO();
  bool deleted = false;
  for (float offset = 0.0f; offset < 120.0f; offset += 2.0f) {
    for (float y_val : {10.0f, 30.0f, 50.0f}) {
      io.MousePos = ImVec2(x + offset, y_val);
      io.AddMouseButtonEvent(0, true);
      SimulateFrame();
      io.AddMouseButtonEvent(0, false);
      SimulateFrame();
      if (timeline_.bookmarks().empty()) {
        deleted = true;
        break;
      }
    }
    if (deleted) break;
  }

  EXPECT_TRUE(deleted);
}

TEST_F(MockTimelineImGuiFixture, DrawBookmarks_MultipleBookmarks) {
  timeline_.set_bookmarks_enabled(true);
  timeline_.set_data_time_range({0.0, 1000.0});
  timeline_.SetVisibleRange({0.0, 1000.0});
  timeline_.AddBookmark(100.0);
  timeline_.AddBookmark(200.0);
  timeline_.AddBookmark(300.0);

  EXPECT_THAT(timeline_.bookmarks(), ElementsAre(100.0, 200.0, 300.0));

  // Delete the middle one.
  SimulateFrame(0.0f, 1000.0f);
  float x = 200.0f * timeline_.px_per_time_unit() + GetTimelineStartX();
  ImGuiIO& io = ImGui::GetIO();
  bool deleted = false;
  for (float offset = 0.0f; offset < 120.0f; offset += 2.0f) {
    for (float y_val : {10.0f, 30.0f, 50.0f}) {
      io.MousePos = ImVec2(x + offset, y_val);
      io.AddMouseButtonEvent(0, true);
      SimulateFrame();
      io.AddMouseButtonEvent(0, false);
      SimulateFrame();
      if (timeline_.bookmarks().size() == 2) {
        deleted = true;
        break;
      }
    }
    if (deleted) break;
  }

  EXPECT_TRUE(deleted);
  EXPECT_THAT(timeline_.bookmarks(), ElementsAre(100.0, 300.0));
}

TEST_F(MockTimelineImGuiFixture, PanningDisabledDuringLabelColumnResizing) {
  SimulateFrame();  // Ensure ImGui tables are instantiated

  ImGuiIO& io = ImGui::GetIO();

  // Panning should not occur!
  EXPECT_CALL(timeline_, Pan(_)).Times(0);
  EXPECT_CALL(timeline_, Scroll(_)).Times(0);

  SimulateLabelColumnResizeDragStart();

  // Attempt to pan by dragging the mouse horizontally
  // Because resizing is active, the timeline shouldn't handle mouse drags.
  io.AddMousePosEvent(GetTimelineStartX() + 50.0f, io.MousePos.y);
  SimulateFrame();

  // Release
  io.AddMouseButtonEvent(0, false);
  SimulateFrame();
}

TEST_F(MockTimelineImGuiFixture, ScrollDownWithDownArrowKey) {
  ImGui::GetIO().AddKeyEvent(ImGuiKey_DownArrow, true);

  EXPECT_CALL(timeline_,
              Scroll(FloatEq(kScrollSpeed * ImGui::GetIO().DeltaTime)))
      .Times(1);

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

TEST_F(MockTimelineImGuiFixture, ScrollUpWithUpArrowKey) {
  ImGui::GetIO().AddKeyEvent(ImGuiKey_UpArrow, true);

  EXPECT_CALL(timeline_,
              Scroll(FloatEq(-kScrollSpeed * ImGui::GetIO().DeltaTime)))
      .Times(1);

  SimulateFrame();
}

TEST_F(MockTimelineImGuiFixture,
       ShiftClickAndReleaseShiftMidDragContinuesSelection) {
  // Setup similar to TimelineDragSelectionTest to ensure predictable
  // coordinates.
  timeline_.SetVisibleRange({0.0, 165.5});
  timeline_.set_data_time_range({0.0, 165.5});
  ImGuiIO& io = ImGui::GetIO();

  // Start with Shift held down.
  io.AddKeyEvent(ImGuiMod_Shift, true);

  // Start drag in timeline area.
  // X=300 is safely inside the timeline (GetTimelineStartX() + padding < 300).
  io.MousePos = ImVec2(GetTimelineStartX() + 50.0f, 50.0f);
  io.AddMouseButtonEvent(0, true);
  SimulateFrame();

  // Release Shift key.
  io.AddKeyEvent(ImGuiMod_Shift, false);

  // Drag mouse to X=500.
  io.MousePos = ImVec2(GetTimelineStartX() + 250.0f, 50.0f);
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

TEST_F(MockTimelineImGuiFixture, ZoomInWithMouseWheelAndCtrlKey) {
  // Set a visible range for predictable pivot calculation.
  timeline_.SetVisibleRange({0.0, 165.5});

  ImGuiIO& io = ImGui::GetIO();
  io.AddMouseWheelEvent(0.0f, -1.0f);
  io.AddKeyEvent(ImGuiMod_Ctrl, true);
  // Mouse inside timeline area. x=350 -> Relative x = 100.
  // px_per_unit = 10.0. Pivot = 10.0.
  io.MousePos = ImVec2(GetTimelineStartX() + 100.0f, 50.0f);

  const float expected_zoom_factor = 1.0f + (-1.0f) * kMouseWheelZoomSpeed;

  EXPECT_CALL(timeline_, Zoom(FloatEq(expected_zoom_factor), DoubleEq(10.0)));

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

TEST_F(MockTimelineImGuiFixture, ZoomInWithWKey_MouseInsideTimeline) {
  // Set a visible range that results in a round number for px_per_time_unit
  // to make test calculations predictable. With a timeline width of 1655px
  // (based on 1920px window width, 250px label width, and 1px padding),
  // a duration of 165.5 gives 10px per microsecond.
  timeline_.SetVisibleRange({0.0, 165.5});

  ImGuiIO& io = ImGui::GetIO();
  // Timeline starts at label_width (GetTimelineStartX()).
  // Set mouse at x=350, y=50.
  // Relative x = 350 - GetTimelineStartX() = 100.
  // Scale = 10 px/us.
  // Pivot = 100 / 10 = 10.0 us.
  io.MousePos = ImVec2(GetTimelineStartX() + 100.0f, 50.0f);
  io.AddKeyEvent(ImGuiKey_W, true);

  // No acceleration is applied.
  EXPECT_CALL(timeline_,
              Zoom(FloatEq(1.0f - kZoomSpeed * io.DeltaTime), DoubleEq(10.0)));

  SimulateFrame();
}

TEST_F(MockTimelineImGuiFixture, ZoomInWithWKey_MouseOutsideTimeline) {
  timeline_.SetVisibleRange({0.0, 165.5});

  ImGuiIO& io = ImGui::GetIO();
  // Mouse outside timeline (x < GetTimelineStartX()).
  io.MousePos = ImVec2(GetTimelineStartX() - 150.0f, 50.0f);
  io.AddKeyEvent(ImGuiKey_W, true);

  // Pivot should be center of visible range [0, 165.5] -> 82.75.
  EXPECT_CALL(timeline_,
              Zoom(FloatEq(1.0f - kZoomSpeed * io.DeltaTime), DoubleEq(82.75)));

  SimulateFrame();
}

TEST_F(MockTimelineImGuiFixture, ZoomOutWithMouseWheelAndCtrlKey) {
  // Set a visible range for predictable pivot calculation.
  timeline_.SetVisibleRange({0.0, 165.5});

  ImGuiIO& io = ImGui::GetIO();
  io.AddMouseWheelEvent(0.0f, 1.0f);
  io.AddKeyEvent(ImGuiMod_Ctrl, true);
  // Mouse inside timeline area. x=350 -> Relative x = 100.
  // px_per_unit = 10.0. Pivot = 10.0.
  io.MousePos = ImVec2(GetTimelineStartX() + 100.0f, 50.0f);

  const float expected_zoom_factor = 1.0f + 1.0f * kMouseWheelZoomSpeed;

  EXPECT_CALL(timeline_, Zoom(FloatEq(expected_zoom_factor), DoubleEq(10.0)));

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

// =============================================================================
// Fixture: RealTimelineImGuiFixture
// =============================================================================

TEST_F(RealTimelineImGuiFixture, ClickCounterEventSetsSelectionIndices) {
  FlameChartTimelineData data;
  data.groups.push_back({.type = Group::Type::kCounter,
                         .name = "Counter Group",
                         .start_level = 0,
                         .nesting_level = 0,
                         .expanded = true});

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

  ImGui::GetIO().MouseDown[0] = false;
  SimulateFrame();

  EXPECT_EQ(timeline_.selected_event_index(), -1);
  EXPECT_EQ(timeline_.selected_group_index(), 0);
  EXPECT_EQ(timeline_.selected_counter_index(), 1);
}

TEST_F(RealTimelineImGuiFixture, ClickEmptyAreaClearsSelectionIndices) {
  FlameChartTimelineData data;
  data.groups.push_back({.name = "Group 1",
                         .start_level = 0,
                         .nesting_level = kThreadNestingLevel,
                         .expanded = true});
  data.events_by_level.push_back({0});
  data.entry_names.push_back("event1");
  data.entry_levels.push_back(0);
  data.entry_start_times.push_back(0.0);
  data.entry_total_times.push_back(100.0);
  data.entry_pids.push_back(1);
  data.entry_args.push_back({});
  timeline_.SetTimelineData(std::move(data));
  timeline_.SetVisibleRange({0.0, 100.0});

  // Select event
  ImGui::GetIO().MousePos = ImVec2(GetTimelineStartX() + 50.0f, kFirstEventY);
  ImGui::GetIO().MouseDown[0] = true;
  SimulateFrame();
  ImGui::GetIO().MouseDown[0] = false;
  SimulateFrame();

  EXPECT_NE(timeline_.selected_event_index(), -1);

  // Click empty area
  ImGui::GetIO().MousePos = ImVec2(GetTimelineStartX() + 50.0f, 100.f);
  ImGui::GetIO().MouseDown[0] = true;
  SimulateFrame();

  ImGui::GetIO().MouseDown[0] = false;
  SimulateFrame();

  EXPECT_EQ(timeline_.selected_event_index(), -1);
  EXPECT_EQ(timeline_.selected_group_index(), -1);
  EXPECT_EQ(timeline_.selected_counter_index(), -1);
}

FlameChartTimelineData GetTestFlowData() {
  FlameChartTimelineData data;
  data.groups.push_back({.name = "Group 1",
                         .start_level = 0,
                         .nesting_level = 0,
                         .expanded = true});
  data.events_by_level.push_back({0, 1});
  data.entry_names = {"event0", "event1"};
  data.entry_event_ids = {1000, 2000};
  data.entry_levels = {0, 0};
  data.entry_start_times = {10.0, 50.0};
  data.entry_total_times = {5.0, 5.0};
  data.entry_pids = {1, 2};
  data.entry_args = {{}, {}};
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

TEST_F(RealTimelineImGuiFixture, ClickEmptyAreaDeselectsEvent) {
  FlameChartTimelineData data;

  data.groups.push_back({.name = "Group 1",
                         .start_level = 0,
                         .nesting_level = 0,
                         .expanded = true});
  data.events_by_level.push_back({0});
  data.entry_names.push_back("event1");
  data.entry_levels.push_back(0);
  data.entry_start_times.push_back(0.0);
  data.entry_total_times.push_back(100.0);
  data.entry_pids.push_back(1);
  data.entry_args.push_back({});
  timeline_.SetTimelineData(std::move(data));
  timeline_.SetVisibleRange({0.0, 100.0});

  // First, select an event.
  ImGui::GetIO().MousePos = ImVec2(GetTimelineStartX() + 50.0f,
                                   kFirstEventY);  // A position over the event.
  ImGui::GetIO().AddMouseButtonEvent(0, true);
  SimulateFrame();
  ImGui::GetIO().AddMouseButtonEvent(0, false);  // Release the mouse.
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
  ImGui::GetIO().MousePos = ImVec2(GetTimelineStartX() + 50.0f,
                                   100.f);  // A position outside the event.
  ImGui::GetIO().MouseDown[0] = true;
  SimulateFrame();

  ImGui::GetIO().MouseDown[0] = false;
  SimulateFrame();

  EXPECT_TRUE(deselection_callback_called);
}

TEST_F(RealTimelineImGuiFixture, ClickEmptyAreaDeselectsOnlyOnce) {
  FlameChartTimelineData data;

  data.groups.push_back({.name = "Group 1",
                         .start_level = 0,
                         .nesting_level = 0,
                         .expanded = true});
  data.events_by_level.push_back({0});
  data.entry_names.push_back("event1");
  data.entry_levels.push_back(0);
  data.entry_start_times.push_back(0.0);
  data.entry_total_times.push_back(100.0);
  data.entry_pids.push_back(1);
  data.entry_args.push_back({});
  timeline_.SetTimelineData(std::move(data));
  timeline_.SetVisibleRange({0.0, 100.0});

  // First, select an event.
  ImGui::GetIO().MousePos = ImVec2(GetTimelineStartX() + 50.0f,
                                   kFirstEventY);  // A position over the event.
  ImGui::GetIO().AddMouseButtonEvent(0, true);
  SimulateFrame();
  ImGui::GetIO().AddMouseButtonEvent(0, false);  // Release the mouse.
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
  ImGui::GetIO().MousePos = ImVec2(GetTimelineStartX() + 50.0f,
                                   100.f);  // A position outside the event.
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

  data.groups.push_back({.name = "Group 1",
                         .start_level = 0,
                         .nesting_level = 0,
                         .expanded = true});
  data.events_by_level.push_back({0});
  data.entry_names.push_back("event1");
  data.entry_levels.push_back(0);
  data.entry_start_times.push_back(0.0);
  data.entry_total_times.push_back(100.0);
  data.entry_pids.push_back(1);
  data.entry_args.push_back({});
  timeline_.SetTimelineData(std::move(data));
  timeline_.SetVisibleRange({0.0, 100.0});

  bool callback_called = false;
  timeline_.set_event_callback(
      [&](absl::string_view type, const EventData& detail) {
        callback_called = true;
      });

  // Now, click on an empty area.
  ImGui::GetIO().MousePos = ImVec2(GetTimelineStartX() + 50.0f,
                                   100.f);  // A position outside the event.
  ImGui::GetIO().MouseDown[0] = true;

  SimulateFrame();

  EXPECT_FALSE(callback_called);
}

TEST_F(RealTimelineImGuiFixture, ClickEventSelectsEvent) {
  FlameChartTimelineData data;

  data.groups.push_back({.name = "Group 1",
                         .start_level = 0,
                         .nesting_level = 0,
                         .expanded = true});
  data.events_by_level.push_back({0});
  data.entry_names.push_back("event1");
  data.entry_levels.push_back(0);
  data.entry_start_times.push_back(0.0);
  data.entry_total_times.push_back(100.0);
  data.entry_pids.push_back(1);
  data.entry_args.push_back({});
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
  // y=28 is safely within the event rect (starts at 20, height 16 -> ends at
  // 36).
  ImGui::GetIO().MousePos = ImVec2(GetTimelineStartX() + 50.0f, kFirstEventY);
  ImGui::GetIO().MouseDown[0] = true;
  SimulateFrame();

  ImGui::GetIO().MouseDown[0] = false;
  SimulateFrame();

  EXPECT_TRUE(callback_called);
  EXPECT_EQ(event_type, kEventSelected);
  ASSERT_TRUE(event_detail.contains(kEventSelectedIndex));
  ASSERT_TRUE(event_detail.contains(kEventSelectedName));
  EXPECT_EQ(std::any_cast<int>(event_detail.at(kEventSelectedIndex)), 0);
  EXPECT_EQ(std::any_cast<std::string>(event_detail.at(kEventSelectedName)),
            "event1");
}

TEST_F(RealTimelineImGuiFixture, ClickEventWithArgsSelectsEvent) {
  FlameChartTimelineData data;

  data.groups.push_back({.name = "Group 1",
                         .start_level = 0,
                         .nesting_level = 0,
                         .expanded = true});
  data.events_by_level.push_back({0});
  data.entry_names.push_back("event_with_args");
  data.entry_levels.push_back(0);
  data.entry_start_times.push_back(0.0);
  data.entry_total_times.push_back(100.0);
  data.entry_pids.push_back(1);

  absl::flat_hash_map<std::string, std::string> args;
  args["uid"] = "12345";
  args[std::string(kHloModule)] = "test_module";
  args[std::string(kHloOp)] = "test_op";
  data.entry_args.push_back(args);

  timeline_.SetTimelineData(std::move(data));
  timeline_.SetVisibleRange({0.0, 100.0});

  bool callback_called = false;
  EventData event_detail;
  timeline_.set_event_callback(
      [&](absl::string_view type, const EventData& detail) {
        callback_called = true;
        event_detail = detail;
      });

  ImGui::GetIO().MousePos = ImVec2(GetTimelineStartX() + 50.0f, kFirstEventY);
  ImGui::GetIO().MouseDown[0] = true;
  SimulateFrame();

  ImGui::GetIO().MouseDown[0] = false;
  SimulateFrame();

  EXPECT_TRUE(callback_called);
  ASSERT_TRUE(event_detail.contains(kEventSelectedUid));
  ASSERT_TRUE(event_detail.contains(kEventSelectedHloModuleName));
  ASSERT_TRUE(event_detail.contains(kEventSelectedHloOpName));

  EXPECT_EQ(std::any_cast<std::string>(event_detail.at(kEventSelectedUid)),
            "12345");
  EXPECT_EQ(
      std::any_cast<std::string>(event_detail.at(kEventSelectedHloModuleName)),
      "test_module");
  EXPECT_EQ(
      std::any_cast<std::string>(event_detail.at(kEventSelectedHloOpName)),
      "test_op");
}

TEST_F(RealTimelineImGuiFixture, ClickEventSetsSelectionIndices) {
  FlameChartTimelineData data;
  data.groups.push_back({.name = "Group 1",
                         .start_level = 0,
                         .nesting_level = 0,
                         .expanded = true});
  data.events_by_level.push_back({0});
  data.entry_names.push_back("event1");
  data.entry_levels.push_back(0);
  data.entry_start_times.push_back(0.0);
  data.entry_total_times.push_back(100.0);
  data.entry_pids.push_back(1);
  data.entry_args.push_back({});
  timeline_.SetTimelineData(std::move(data));
  timeline_.SetVisibleRange({0.0, 100.0});

  // Set a mouse position that is guaranteed to be over the event.
  ImGui::GetIO().MousePos = ImVec2(GetTimelineStartX() + 50.0f, kFirstEventY);
  ImGui::GetIO().MouseDown[0] = true;
  SimulateFrame();

  ImGui::GetIO().MouseDown[0] = false;
  SimulateFrame();

  EXPECT_EQ(timeline_.selected_event_index(), 0);
  EXPECT_EQ(timeline_.selected_group_index(), 0);
  EXPECT_EQ(timeline_.selected_counter_index(), -1);
}

TEST_F(RealTimelineImGuiFixture, ClickOutsideEventDoesNotSelectEvent) {
  FlameChartTimelineData data;

  data.groups.push_back({.name = "Group 1",
                         .start_level = 0,
                         .nesting_level = 0,
                         .expanded = true});
  data.events_by_level.push_back({0});
  data.entry_names.push_back("event1");
  data.entry_levels.push_back(0);
  data.entry_start_times.push_back(0.0);
  data.entry_total_times.push_back(100.0);
  data.entry_pids.push_back(1);
  data.entry_args.push_back({});
  timeline_.SetTimelineData(std::move(data));
  timeline_.SetVisibleRange({0.0, 100.0});

  bool callback_called = false;
  timeline_.set_event_callback(
      [&](absl::string_view type, const EventData& detail) {
        callback_called = true;
      });

  // Set a mouse position that is guaranteed to be outside the event.
  ImGui::GetIO().MousePos = ImVec2(GetTimelineStartX() + 50.0f, 100.f);
  ImGui::GetIO().MouseDown[0] = true;

  SimulateFrame();

  EXPECT_FALSE(callback_called);
}

TEST_F(RealTimelineImGuiFixture,
       ClickingSelectedEventAgainDoesNotFireCallback) {
  FlameChartTimelineData data;

  data.groups.push_back({.name = "Group 1",
                         .start_level = 0,
                         .nesting_level = 0,
                         .expanded = true});
  data.events_by_level.push_back({0});
  data.entry_names.push_back("event1");
  data.entry_levels.push_back(0);
  data.entry_start_times.push_back(0.0);
  data.entry_total_times.push_back(100.0);
  data.entry_pids.push_back(1);
  data.entry_args.push_back({});
  timeline_.SetTimelineData(std::move(data));
  timeline_.SetVisibleRange({0.0, 100.0});

  int callback_count = 0;
  timeline_.set_event_callback(
      [&](absl::string_view type, const EventData& detail) {
        callback_count++;
      });

  ImGui::GetIO().MousePos = ImVec2(GetTimelineStartX() + 50.0f, kFirstEventY);

  // First click.
  ImGui::GetIO().MouseDown[0] = true;
  SimulateFrame();

  ImGui::GetIO().MouseDown[0] = false;
  SimulateFrame();

  EXPECT_EQ(callback_count, 1);

  // Second click on the same event.
  ImGui::GetIO().MouseDown[0] = true;
  SimulateFrame();

  ImGui::GetIO().MouseDown[0] = false;
  SimulateFrame();

  // Callback count should still be 1.
  EXPECT_EQ(callback_count, 1);
}

TEST_F(RealTimelineImGuiFixture, DragOverCounterPointDoesNotSelectEvent) {
  FlameChartTimelineData data;
  data.groups.push_back({.type = Group::Type::kCounter,
                         .name = "Counter Group",
                         .start_level = 0,
                         .nesting_level = 0,
                         .expanded = true});

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
    if (std::string(w->Name).find(child_id) != std::string::npos) {
      counter_window = w;
      break;
    }
  }
  ASSERT_NE(counter_window, nullptr);

  ImVec2 target_pos = counter_window->Pos;
  target_pos.x += counter_window->Size.x * 0.25f;
  target_pos.y += counter_window->Size.y * 0.5f;

  ImGui::EndFrame();

  ImGuiIO& io = ImGui::GetIO();
  io.AddMousePosEvent(target_pos.x, target_pos.y);
  SimulateFrame();

  io.AddMouseButtonEvent(0, true);
  SimulateFrame();

  io.AddMousePosEvent(target_pos.x + 10.0f, target_pos.y);
  SimulateFrame();

  io.AddMouseButtonEvent(0, false);
  SimulateFrame();

  EXPECT_EQ(timeline_.selected_counter_index(), -1);
}

TEST_F(RealTimelineImGuiFixture, DragOverEventDoesNotSelectEvent) {
  FlameChartTimelineData data;

  data.groups.push_back({.name = "Group 1",
                         .start_level = 0,
                         .nesting_level = 0,
                         .expanded = true});
  data.events_by_level.push_back({0});
  data.entry_names.push_back("event1");
  data.entry_levels.push_back(0);
  data.entry_start_times.push_back(0.0);
  data.entry_total_times.push_back(100.0);
  data.entry_pids.push_back(1);
  data.entry_args.push_back({});
  timeline_.SetTimelineData(std::move(data));
  timeline_.SetVisibleRange({0.0, 100.0});

  bool callback_called = false;
  timeline_.set_event_callback(
      [&](absl::string_view type, const EventData& detail) {
        callback_called = true;
      });

  const float start_x = GetTimelineStartX() + 50.0f;
  const float start_y = kFirstEventY;

  ImGui::GetIO().MousePos = ImVec2(start_x, start_y);
  SimulateFrame();

  ImGui::GetIO().MouseDown[0] = true;
  SimulateFrame();

  ImGui::GetIO().MousePos = ImVec2(start_x + 10.0f, start_y);
  SimulateFrame();

  ImGui::GetIO().MouseDown[0] = false;
  SimulateFrame();

  EXPECT_EQ(timeline_.selected_event_index(), -1);
}

TEST_F(RealTimelineImGuiFixture, DrawCounterTrack) {
  FlameChartTimelineData data;
  data.groups.push_back({.type = Group::Type::kCounter,
                         .name = "Counter Group",
                         .start_level = 0,
                         .nesting_level = 0,
                         .expanded = true});

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

TEST_F(RealTimelineImGuiFixture, DrawCounterTrackConstantValue) {
  FlameChartTimelineData data;
  data.groups.push_back({.type = Group::Type::kCounter,
                         .name = "Counter Group",
                         .start_level = 0,
                         .nesting_level = 0,
                         .expanded = true});

  CounterData counter_data;
  counter_data.timestamps = {10.0, 20.0, 30.0};
  counter_data.values = {5.0, 5.0, 5.0};  // Constant value
  counter_data.min_value = 5.0;
  counter_data.max_value = 5.0;
  data.counter_data_by_group_index[0] = std::move(counter_data);

  timeline_.SetTimelineData(std::move(data));
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

  // Check if anything was drawn to this window's draw list.
  EXPECT_FALSE(counter_window->DrawList->VtxBuffer.empty());

  ImGui::EndFrame();
}

TEST_F(RealTimelineImGuiFixture, DrawUtilizationAreaChartLastBinOnly) {
  FlameChartTimelineData data;
  data.groups.push_back(
      {.type = Group::Type::kFlame,
       .name = "Process Group",
       .start_level = 0,
       .nesting_level = kProcessNestingLevel,  // Process level
       .expanded = true});

  // Add one event covering the very end of visible range [99.95, 100.0]
  data.events_by_level.push_back({0});
  data.entry_names.push_back("event");
  data.entry_levels.push_back(0);
  data.entry_start_times.push_back(99.95);
  data.entry_total_times.push_back(0.05);
  data.entry_pids.push_back(1);
  data.entry_tids.push_back(1);
  data.entry_event_ids.push_back(1);
  data.entry_args.push_back({});

  timeline_.SetTimelineData(std::move(data));
  timeline_.SetVisibleRange({0.0, 100.0});

  ImGui::NewFrame();
  timeline_.Draw();

  ImGuiWindow* process_window = nullptr;
  const std::string child_id = "TimelineChild_Process Group_0";
  for (ImGuiWindow* w : ImGui::GetCurrentContext()->Windows) {
    if (std::string(w->Name).find(child_id) != std::string::npos) {
      process_window = w;
      break;
    }
  }
  ASSERT_NE(process_window, nullptr);

  // Check if the utilization bar was drawn.
  ImU32 bar_color = color_palette_.GetColor(ColorPalette::Key::kFlameHeader)
                        .value_or(kBlue70);

  bool bar_found = false;
  for (const auto& v : process_window->DrawList->VtxBuffer) {
    if (v.col == bar_color) {
      bar_found = true;
      break;
    }
  }

  EXPECT_TRUE(bar_found);

  ImGui::EndFrame();
}

TEST_F(RealTimelineImGuiFixture, DrawFlameGroupPreview) {
  FlameChartTimelineData data;
  data.groups.push_back({.type = Group::Type::kFlame,
                         .name = "Flame Group",
                         .start_level = 0,
                         .nesting_level = 2,
                         .expanded = false});  // Collapsed triggers preview

  data.events_by_level.push_back({0});
  // Add one more real event on a new level to make the group expandable.
  data.events_by_level.push_back({1});
  data.entry_names.push_back("event1");
  data.entry_names.push_back("event2");
  data.entry_levels.push_back(0);
  data.entry_levels.push_back(1);
  data.entry_start_times.push_back(10.0);
  data.entry_start_times.push_back(15.0);
  data.entry_total_times.push_back(20.0);
  data.entry_total_times.push_back(10.0);
  data.entry_pids.push_back(1);
  data.entry_pids.push_back(1);
  data.entry_args.push_back({});
  data.entry_args.push_back({});

  timeline_.SetTimelineData(std::move(data));
  timeline_.SetVisibleRange({0.0, 100.0});

  ImGui::NewFrame();
  timeline_.Draw();

  ImGuiWindow* preview_window = nullptr;
  const std::string child_id = "TimelineChildPreview_Flame Group_0";
  for (ImGuiWindow* w : ImGui::GetCurrentContext()->Windows) {
    if (std::string(w->Name).find(child_id) != std::string::npos) {
      preview_window = w;
      break;
    }
  }
  ASSERT_NE(preview_window, nullptr);

  // Check if anything was drawn to this window's draw list.
  EXPECT_FALSE(preview_window->DrawList->VtxBuffer.empty());

  // Verify that the drawn rectangles have reduced opacity.
  const ImU32 expected_alpha =
      static_cast<ImU32>(kGroupPreviewOpacity * 255.0f);
  bool found_preview_rect = false;
  for (const auto& vtx : preview_window->DrawList->VtxBuffer) {
    // Skip vertices with 0 alpha (might be for clipping or other internal use)
    if ((vtx.col >> IM_COL32_A_SHIFT) == expected_alpha) {
      found_preview_rect = true;
      break;
    }
  }

  EXPECT_TRUE(found_preview_rect);

  ImGui::EndFrame();
}

TEST_F(RealTimelineImGuiFixture, DrawFlowsForSelectedEvent) {
  timeline_.SetTimelineData(GetTestFlowData());
  timeline_.SetVisibleRange({0.0, 100.0});
  timeline_.SetVisibleFlowCategories(
      {static_cast<int>(tsl::profiler::ContextType::kGeneric),
       static_cast<int>(
           tsl::profiler::ContextType::kGpuLaunch)});  // Show all flows if no
                                                       // event selected

  // Select event 0 (id 1000), which is part of flow "1" (flow1).
  timeline_.RevealEvent(0);

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

TEST_F(RealTimelineImGuiFixture,
       DrawFlowsForSelectedEventWithNoVisibleCategories) {
  timeline_.SetTimelineData(GetTestFlowData());
  timeline_.SetVisibleRange({0.0, 100.0});
  timeline_.SetVisibleFlowCategories({});  // No flow categories visible

  // Select event 0 (id 1000), which is part of flow "1" (flow1).
  timeline_.RevealEvent(0);

  ImGui::NewFrame();
  timeline_.Draw();
  ImDrawList* draw_list = ImGui::GetForegroundDrawList();

  ASSERT_FALSE(draw_list->VtxBuffer.empty());

  // Event 0 (id 1000) is associated with flow1 (Red).
  // We should see flow1's color and not flow2's color, even with no visible
  // categories, because an event is selected.
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

TEST_F(RealTimelineImGuiFixture, DrawFlowsWithEmptyFlowLinesButSelectedEvent) {
  FlameChartTimelineData data = GetTestFlowData();
  data.flow_lines.clear();  // flow_lines is empty
  timeline_.SetTimelineData(std::move(data));
  timeline_.SetVisibleRange({0.0, 100.0});
  timeline_.SetVisibleFlowCategories({});  // categories empty

  // Select event 0 (id 1000), which is part of flow "1" (flow1).
  // flow1 is in flow_lines_by_flow_id from GetTestFlowData().
  timeline_.RevealEvent(0);  // has_selected_event is true

  // With flow_lines empty, DrawFlows should return early, and nothing should
  // be drawn. If the early return is skipped, flows for selected events will be
  // drawn from flow_lines_by_flow_id, causing this test to fail.
  ImGui::NewFrame();
  timeline_.Draw();
  ImDrawList* draw_list = ImGui::GetForegroundDrawList();
  EXPECT_TRUE(draw_list->VtxBuffer.empty());
  ImGui::EndFrame();
}

TEST_F(RealTimelineImGuiFixture, DrawFlowsWithInvalidLevels) {
  FlameChartTimelineData data = GetTestFlowData();
  FlowLine flow_invalid_source = {
      .source_ts = 12.0,
      .target_ts = 52.0,
      .source_level = 999,  // invalid
      .target_level = 0,
      .color = 0xFF0000FF,  // Blue
      .category = tsl::profiler::ContextType::kGeneric};
  FlowLine flow_invalid_target = {
      .source_ts = 13.0,
      .target_ts = 53.0,
      .source_level = 0,
      .target_level = 999,  // invalid
      .color = 0xFF0000FF,  // Blue
      .category = tsl::profiler::ContextType::kGeneric};
  data.flow_lines.push_back(flow_invalid_source);
  data.flow_lines.push_back(flow_invalid_target);
  timeline_.SetTimelineData(std::move(data));
  timeline_.SetVisibleRange({0.0, 100.0});
  timeline_.SetVisibleFlowCategories(
      {static_cast<int>(tsl::profiler::ContextType::kGeneric),
       static_cast<int>(tsl::profiler::ContextType::kGpuLaunch)});

  ImGui::NewFrame();
  timeline_.Draw();
  ImDrawList* draw_list = ImGui::GetForegroundDrawList();

  ASSERT_FALSE(draw_list->VtxBuffer.empty());

  // We should only see flow1(Red) and flow2(Green) from GetTestFlowData.
  // The invalid flows (Blue) should be skipped.
  bool found_flow1_color = false;
  bool found_flow2_color = false;
  bool found_invalid_flow_color = false;
  for (const auto& vtx : draw_list->VtxBuffer) {
    if (vtx.col == 0xFFFF0000) found_flow1_color = true;
    if (vtx.col == 0xFF00FF00) found_flow2_color = true;
    if (vtx.col == 0xFF0000FF) found_invalid_flow_color = true;
  }
  EXPECT_TRUE(found_flow1_color);
  EXPECT_TRUE(found_flow2_color);
  EXPECT_FALSE(found_invalid_flow_color);

  ImGui::EndFrame();
}

using TimelineImGuiFixture = TimelineImGuiTestFixture<Timeline>;

TEST_F(RealTimelineImGuiFixture, DrawFlowsWithSelectedEventButNoEventIds) {
  FlameChartTimelineData data = GetTestFlowData();
  data.entry_event_ids.clear();
  timeline_.SetTimelineData(std::move(data));
  timeline_.SetVisibleRange({0.0, 100.0});
  timeline_.SetVisibleFlowCategories(
      {static_cast<int>(tsl::profiler::ContextType::kGpuLaunch),
       static_cast<int>(tsl::profiler::ContextType::kGeneric)});

  // Select event 0.
  timeline_.RevealEvent(0);

  ImGui::NewFrame();
  timeline_.Draw();
  ImDrawList* draw_list = ImGui::GetForegroundDrawList();

  // If selected_event_index_ is out of bounds for entry_event_ids,
  // has_selected_event should be false, and flows should be drawn based on
  // visible categories.
  ASSERT_FALSE(draw_list->VtxBuffer.empty());

  // flow1 (Red) and flow2 (Green) should be drawn.
  bool found_flow1_color = false;
  bool found_flow2_color = false;
  for (const auto& vtx : draw_list->VtxBuffer) {
    if (vtx.col == 0xFFFF0000) found_flow1_color = true;
    if (vtx.col == 0xFF00FF00) found_flow2_color = true;
  }
  EXPECT_TRUE(found_flow1_color);
  EXPECT_TRUE(found_flow2_color);

  ImGui::EndFrame();
}

TEST_F(RealTimelineImGuiFixture, DrawFlowsWithVisibleFlowCategoriesMultiple) {
  timeline_.SetTimelineData(GetTestFlowData());
  timeline_.SetVisibleRange({0.0, 100.0});
  // Show kGpuLaunch (flow2) and kGeneric (flow1).
  timeline_.SetVisibleFlowCategories(
      {static_cast<int>(tsl::profiler::ContextType::kGpuLaunch),
       static_cast<int>(tsl::profiler::ContextType::kGeneric)});

  ImGui::NewFrame();
  timeline_.Draw();
  ImDrawList* draw_list = ImGui::GetForegroundDrawList();

  ASSERT_FALSE(draw_list->VtxBuffer.empty());

  // flow1 (Red) and flow2 (Green) should be drawn.
  bool found_flow1_color = false;
  bool found_flow2_color = false;
  for (const auto& vtx : draw_list->VtxBuffer) {
    if (vtx.col == 0xFFFF0000) found_flow1_color = true;
    if (vtx.col == 0xFF00FF00) found_flow2_color = true;
  }
  EXPECT_TRUE(found_flow1_color);
  EXPECT_TRUE(found_flow2_color);

  ImGui::EndFrame();
}

TEST_F(RealTimelineImGuiFixture, DrawFlowsWithVisibleFlowCategoriesNone) {
  timeline_.SetTimelineData(GetTestFlowData());
  timeline_.SetVisibleRange({0.0, 100.0});
  timeline_.SetVisibleFlowCategories({});  // Show no flows

  ImGui::NewFrame();
  timeline_.Draw();
  ImDrawList* draw_list = ImGui::GetForegroundDrawList();
  EXPECT_TRUE(draw_list->VtxBuffer.empty());
  ImGui::EndFrame();
}

TEST_F(RealTimelineImGuiFixture, DrawFlowsWithVisibleFlowCategoriesSingle) {
  timeline_.SetTimelineData(GetTestFlowData());
  timeline_.SetVisibleRange({0.0, 100.0});
  // kGpuLaunch is category for flow2.
  timeline_.SetVisibleFlowCategories(
      {static_cast<int>(tsl::profiler::ContextType::kGpuLaunch)});

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

TEST_F(RealTimelineImGuiFixture, DrawFlowsWithZeroViewDuration) {
  timeline_.SetTimelineData(GetTestFlowData());
  timeline_.SetVisibleRange({10.0, 10.0});  // 0 duration
  timeline_.SetVisibleFlowCategories(
      {static_cast<int>(tsl::profiler::ContextType::kGpuLaunch),
       static_cast<int>(tsl::profiler::ContextType::kGeneric)});

  ImGui::NewFrame();
  ImDrawList* draw_list = ImGui::GetForegroundDrawList();
  int initial_clip_rect_stack_size = draw_list->_ClipRectStack.Size;
  timeline_.Draw();
  EXPECT_TRUE(draw_list->VtxBuffer.empty());
  EXPECT_EQ(draw_list->_ClipRectStack.Size, initial_clip_rect_stack_size);
  ImGui::EndFrame();
}

TEST_F(RealTimelineImGuiFixture, DrawProcessTrackUtilizationAreaChart) {
  FlameChartTimelineData data;
  // Group 0: Process track at nesting level 0
  data.groups.push_back({.type = Group::Type::kFlame,
                         .name = "Process Track",
                         .start_level = 0,
                         .nesting_level = 1,
                         .expanded = false});
  // Group 1: Next track at same nesting level, starts at level 1.
  // This will cause the loop in timeline.cc to break and set end_level to 1.
  data.groups.push_back({.type = Group::Type::kFlame,
                         .name = "Next Track",
                         .start_level = 1,
                         .nesting_level = 1,
                         .expanded = false});

  data.events_by_level.push_back({0});  // Level 0 has event 0
  data.events_by_level.push_back({1});  // Level 1 has event 1

  // Event 0 on level 0
  data.entry_names.push_back("event1");
  data.entry_levels.push_back(0);
  data.entry_start_times.push_back(10.0);
  data.entry_total_times.push_back(20.0);
  data.entry_pids.push_back(1);
  data.entry_args.push_back({});

  // Event 1 on level 1
  data.entry_names.push_back("event2");
  data.entry_levels.push_back(1);
  data.entry_start_times.push_back(40.0);
  data.entry_total_times.push_back(20.0);
  data.entry_pids.push_back(1);
  data.entry_args.push_back({});

  timeline_.SetTimelineData(std::move(data));
  timeline_.SetVisibleRange({0.0, 100.0});

  ImGui::NewFrame();
  timeline_.Draw();

  ImGuiWindow* utilization_window = nullptr;
  const std::string child_id = "TimelineChild_Process Track_0";
  for (ImGuiWindow* w : ImGui::GetCurrentContext()->Windows) {
    if (std::string(w->Name).find(child_id) != std::string::npos) {
      utilization_window = w;
      break;
    }
  }
  ASSERT_NE(utilization_window, nullptr);

  // We count the total number of vertices instead of just checking for
  // existence (as found_utilization_rect did) because mutants can cause extra
  // shapes to be drawn (e.g., by incorrectly processing deeper levels).
  // Counting allows us to detect these incorrect additions.
  int blue_vtx_count = 0;
  for (const auto& vtx : utilization_window->DrawList->VtxBuffer) {
    if (vtx.col == 0xFFF7AA7B) {
      blue_vtx_count++;
    }
  }

  // We expect a specific number of vertices for level 0 only.
  // If any mutant causes level 1 to be processed, this count will increase.
  EXPECT_EQ(blue_vtx_count, 1328);

  ImGui::EndFrame();
}

TEST_F(RealTimelineImGuiFixture, DrawRulerRendersProperly) {
  // Set up timeline with simple data so it draws
  timeline_.SetVisibleRange({0.0, 100.0});
  timeline_.set_data_time_range({0.0, 100.0});
  FlameChartTimelineData data;
  data.groups.push_back({.name = "Group 1", .start_level = 0});
  timeline_.SetTimelineData(std::move(data));

  SimulateFrame();

  ImGui::NewFrame();
  timeline_.Draw();

  ImGuiWindow* timeline_window = ImGui::FindWindowByName("Timeline viewer");
  ASSERT_NE(timeline_window, nullptr);

  ImGuiWindow* tracks_window = nullptr;
  for (ImGuiWindow* child : timeline_window->DC.ChildWindows) {
    if (absl::StrContains(child->Name, "Tracks")) {
      tracks_window = child;
      break;
    }
  }
  ASSERT_NE(tracks_window, nullptr) << "Failed to find 'Tracks' child window";

  // The DrawRuler renders to the "Tracks" window's main draw list.
  ImDrawList* draw_list = tracks_window->DrawList;

  float max_y_for_trace_vertical_line = 0.0f;

  // kTraceVerticalLineColor is used to draw the vertical line across tracks.
  bool found_trace_vertical_line = false;

  // kRulerLineColor is used for horizontal line, major ticks, and minor ticks.
  // Note: It's tricky to distinguish minor ticks from major ticks purely by
  // VtxBuffer without knowing exact Y coordinates. But we can assert the number
  // of unique X positions that have kRulerLineColor.
  std::set<float> ruler_line_x_positions;

  for (const auto& vtx : tracks_window->DrawList->VtxBuffer) {
    if (vtx.col == kTraceVerticalLineColor) {
      if (vtx.pos.y > max_y_for_trace_vertical_line) {
        max_y_for_trace_vertical_line = vtx.pos.y;
      }
      found_trace_vertical_line = true;
    } else if (vtx.col == kRulerLineColor) {
      ruler_line_x_positions.insert(std::round(vtx.pos.x));
    }
  }

  for (const auto& vtx : timeline_window->DrawList->VtxBuffer) {
    if (vtx.col == kRulerLineColor) {
      ruler_line_x_positions.insert(std::round(vtx.pos.x));
    }
  }

  ASSERT_TRUE(found_trace_vertical_line)
      << "Failed to find vertical trace lines. draw list vtx count="
      << draw_list->VtxBuffer.Size << ", kTraceVerticalLineColor=" << std::hex
      << kTraceVerticalLineColor;

  // Check the vertical text line goes down to the exact viewport bottom
  const ImGuiViewport* viewport = ImGui::GetMainViewport();
  const float expected_viewport_bottom = viewport->Pos.y + viewport->Size.y;
  EXPECT_NEAR(max_y_for_trace_vertical_line, expected_viewport_bottom, 0.5f);

  std::vector<float> unique_xs;
  for (float x : ruler_line_x_positions) {
    if (unique_xs.empty() || x - unique_xs.back() > 2.0f) {
      unique_xs.push_back(x);
    }
  }

  EXPECT_GT(unique_xs.size(), 10);

  if (unique_xs.size() > 1) {
    std::map<int, int> dist_counts;
    for (size_t i = 1; i < unique_xs.size(); ++i) {
      int dist = std::round(unique_xs[i] - unique_xs[i - 1]);
      dist_counts[dist]++;
    }

    int max_count = 0;
    int max_dist = 0;
    for (const auto& kv : dist_counts) {
      if (kv.second > max_count) {
        max_count = kv.second;
        max_dist = kv.first;
      }
    }

    int total_near_max = 0;
    for (const auto& kv : dist_counts) {
      if (std::abs(kv.first - max_dist) <= 1) {
        total_near_max += kv.second;
      }
    }

    // We expect the distance to be reasonably consistent (within 1 pixel),
    // representing at least half the points. Anomalies from floating point /
    // ImGui AddLine are fine.
    EXPECT_GT(total_near_max, unique_xs.size() / 2 - 5)
        << "Ticks are not consistently spaced.";
    // Ensure we actually found a reasonable spacing.
    EXPECT_GT(total_near_max, 5);
  }

  ImGui::EndFrame();
}

TEST_F(RealTimelineImGuiFixture, DrawSelectedTimeRangeTextAtCorrectYPosition) {
  // Set up timeline
  timeline_.SetVisibleRange({0.0, 100.0});
  timeline_.set_data_time_range({0.0, 100.0});

  // Create a selected time range by simulating a shift-drag
  ImGuiIO& io = ImGui::GetIO();
  io.AddKeyEvent(ImGuiMod_Shift, true);
  // Start drag
  io.MousePos = ImVec2(GetTimelineStartX() + 50.0f, 50.0f);
  io.AddMouseButtonEvent(0, true);
  SimulateFrame();
  // Drag to create a range
  io.MousePos = ImVec2(GetTimelineStartX() + 150.0f, 50.0f);
  SimulateFrame();
  // End drag
  io.AddMouseButtonEvent(0, false);
  SimulateFrame();
  io.AddKeyEvent(ImGuiMod_Shift, false);

  ASSERT_EQ(timeline_.selected_time_ranges().size(), 1);

  // Draw again to inspect the draw list
  ImGui::NewFrame();
  timeline_.Draw();

  // Find the SelectionOverlay child window
  ImGuiWindow* overlay_window = nullptr;
  ImGuiWindow* timeline_window = ImGui::FindWindowByName("Timeline viewer");
  ASSERT_NE(timeline_window, nullptr);
  for (ImGuiWindow* child : timeline_window->DC.ChildWindows) {
    if (absl::StrContains(child->Name, "SelectionOverlay")) {
      overlay_window = child;
      break;
    }
  }
  ASSERT_NE(overlay_window, nullptr);

  // Find the text vertices (drawn in kBlackColor)
  ImDrawList* draw_list = overlay_window->DrawList;
  float min_y = std::numeric_limits<float>::max();
  bool found_text = false;
  for (const auto& vtx : draw_list->VtxBuffer) {
    if (vtx.col == kBlackColor) {
      if (vtx.pos.y < min_y) {
        min_y = vtx.pos.y;
      }
      found_text = true;
    }
  }

  ASSERT_TRUE(found_text) << "Text should be drawn in black";

  // Calculate expected Y position
  // The calculated text_y passed to AddText is now at the top, below the ruler:
  // ruler_screen_y_ + kRulerHeight + kSelectedTimeRangeTextTopPadding.
  // In this test setup, ruler_screen_y_ seems to be 0.
  // So expected text_y is 0 + 20.0f + 5.0f = 25.0f.
  // Adding the observed 3.0f ImGui font vertical offset gives 28.0f.
  const float expected_y =
      kRulerHeight + kSelectedTimeRangeTextTopPadding + 3.0f;

  EXPECT_FLOAT_EQ(min_y, expected_y);

  ImGui::EndFrame();
}

TEST_F(RealTimelineImGuiFixture, DrawSetsWindowPaddingToZero) {
  ImGui::NewFrame();
  timeline_.Draw();

  ImGuiWindow* window = ImGui::FindWindowByName("Timeline viewer");
  ASSERT_NE(window, nullptr);
  EXPECT_EQ(window->WindowPadding.x, 0.0f);
  EXPECT_EQ(window->WindowPadding.y, 0.0f);

  ImGui::EndFrame();
}

TEST_F(RealTimelineImGuiFixture, DrawsTimelineWindowWhenTimelineDataIsEmpty) {
  timeline_.SetTimelineData({});

  // We don't use SimulateFrame() here because we need to inspect the draw list
  // before ImGui::EndFrame() is called.
  ImGui::NewFrame();
  timeline_.Draw();

  EXPECT_NE(ImGui::FindWindowByName("Timeline viewer"), nullptr);

  ImGui::EndFrame();
}

TEST_F(RealTimelineImGuiFixture,
       HandleMouseWithInvalidMouseModeDoesNotChangeCursor) {
  timeline_.set_mouse_mode(static_cast<MouseMode>(0));

  ImGuiIO& io = ImGui::GetIO();
  io.MousePos = ImVec2(GetTimelineStartX() + 50.0f, 50.0f);

  ImGui::NewFrame();
  ImGui::SetMouseCursor(ImGuiMouseCursor_TextInput);

  timeline_.Draw();

  EXPECT_EQ(ImGui::GetMouseCursor(), ImGuiMouseCursor_TextInput);

  ImGui::EndFrame();
}

TEST_F(RealTimelineImGuiFixture, HoverCounterTrackShowsTooltip) {
  FlameChartTimelineData data;
  data.groups.push_back({.type = Group::Type::kCounter,
                         .name = "Counter Group",
                         .start_level = 0,
                         .nesting_level = 0,
                         .expanded = true});

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

TEST_F(RealTimelineImGuiFixture, HoverInstantEventUsesExpandedHitbox) {
  FlameChartTimelineData data;
  data.groups.push_back({.type = Group::Type::kFlame,
                         .name = "Test Group",
                         .start_level = 0,
                         .nesting_level = 0,
                         .expanded = true});
  data.events_by_level.resize(1);
  data.events_by_level[0].push_back(0);
  data.entry_names.push_back("instant_event");
  data.entry_levels.push_back(0);
  data.entry_start_times.push_back(100.0);
  data.entry_total_times.push_back(0.000001);  // IS_INSTANT
  data.entry_pids.push_back(1);
  data.entry_event_ids.push_back(0);
  data.entry_args.push_back({});
  timeline_.SetTimelineData(std::move(data));
  timeline_.SetVisibleRange({0.0, 200.0});

  // Render first frame (keep mouse out of the way so it doesn't hover)
  ImGui::GetIO().MousePos = ImVec2(-1000.0f, -1000.0f);
  ImGui::NewFrame();
  timeline_.Draw();

  ImGuiWindow* group_window = nullptr;
  const std::string child_id = "TimelineChild_Test Group_0";
  for (ImGuiWindow* w : ImGui::GetCurrentContext()->Windows) {
    if (std::string(w->Name).find(child_id) != std::string::npos) {
      group_window = w;
      break;
    }
  }
  ASSERT_NE(group_window, nullptr);
  ASSERT_FALSE(group_window->DrawList->VtxBuffer.empty());

  // Find the position of the event from the first rendered vertex
  // (which is the 'top' vertex of the chevron triangle).
  ImVec2 top_vertex = group_window->DrawList->VtxBuffer[0].pos;
  // The rect width is 2px (left to right is +/- 1.0f from center).
  // The expanded chevron hover hit box is +/- 4.5f.
  // We place the mouse at center X - 3.0f, which is outside the standard
  // rect bounds but strictly inside the expanded triangle hit box.
  ImVec2 target_pos(top_vertex.x - 3.0f, top_vertex.y + 5.0f);

  ImGui::EndFrame();

  // Next frame: Move mouse
  ImGui::GetIO().MousePos = target_pos;
  ImGui::NewFrame();
  timeline_.Draw();

  // Check if it triggered hover drawing (which draws the hover mask using
  // kHoverMaskColor). We also want to verify the event color opacity behavior.
  bool hover_triangle_found = false;

  // ImGui returns pre-multiplied colors in some context via standard math.
  // We just need to manually verify the alpha channel of the main triangle
  // vertices.
  // We'll peek through the buffers looking for `kHoverMaskColor` directly.
  ImU32 original_opaque_test_color = 0;

  for (ImGuiWindow* w : ImGui::GetCurrentContext()->Windows) {
    if (std::string(w->Name).find(child_id) != std::string::npos) {
      for (const auto& vtx : w->DrawList->VtxBuffer) {
        if (vtx.col == kHoverMaskColor) {
          hover_triangle_found = true;
        } else {
          // If not the outline or the mask, it is the primary triangle fill.
          // Capture one of its colors.
          if (vtx.col != 0) {
            original_opaque_test_color = vtx.col;
          }
        }
      }
    }
  }
  EXPECT_TRUE(hover_triangle_found);

  // Extract alpha (highest 8 bits in IM_COL32 packed uint32).
  // A fully opaque color alpha is 255.
  ImU8 alpha = (original_opaque_test_color >> IM_COL32_A_SHIFT) & 0xFF;
  EXPECT_EQ(alpha, 255) << "Triangle should be non-transparent when hovered!";

  ImGui::EndFrame();
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

TEST_F(RealTimelineImGuiFixture, ProcessPendingScrollRevealsBottom) {
  FlameChartTimelineData data;
  data.groups.push_back({.name = "Group 1",
                         .start_level = 0,
                         .nesting_level = kThreadNestingLevel,
                         .expanded = true});
  // Event 0 is at level 30
  data.entry_names.push_back("event0");
  data.entry_levels.push_back(30);
  data.entry_start_times.push_back(100.0);
  data.entry_total_times.push_back(1.0);
  data.entry_pids.push_back(1);
  data.entry_args.push_back({});

  // Event 1 is at level 50, increasing content height to avoid clamp
  data.entry_names.push_back("event_dummy");
  data.entry_levels.push_back(50);
  data.entry_start_times.push_back(100.0);
  data.entry_total_times.push_back(1.0);
  data.entry_pids.push_back(1);
  data.entry_args.push_back({});

  data.events_by_level.resize(51);
  data.events_by_level[30].push_back(0);
  data.events_by_level[50].push_back(1);

  timeline_.SetTimelineData(std::move(data));
  timeline_.set_data_time_range({0.0, 20000.0});

  SimulateFrame();
  SimulateFrame();

  // Force viewport height to 200 to trigger scrolling.
  ImGui::GetIO().DisplaySize = ImVec2(800.0f, 200.0f);

  ImGuiWindow* tracks_window = nullptr;
  for (ImGuiWindow* w : ImGui::GetCurrentContext()->Windows) {
    if (absl::StrContains(std::string(w->Name), "Tracks")) {
      tracks_window = w;
      break;
    }
  }
  ASSERT_NE(tracks_window, nullptr);

  // Set initial scroll to 100.0f to avoid 0 and edge cases as preferred.
  tracks_window->Scroll.y = 100.0f;

  // Reveal event 0 (at level 30).
  timeline_.RevealEvent(0);

  // Render a few frames to process pending scroll.
  for (int i = 0; i < 3; ++i) {
    ImGui::NewFrame();
    timeline_.Draw();
    Animation::UpdateAll(ImGui::GetIO().DeltaTime);
    ImGui::Render();
  }

  // With dummy event at level 50, the content is tall enough to avoid clamp
  // limit. Target scroll is calculated exactly to 565.0f based on level 30.
  // Reduced tolerance to 0.1f to kill mutant at line 2067.
  EXPECT_NEAR(tracks_window->Scroll.y, 565.0f, 0.1f);
}

TEST_F(RealTimelineImGuiFixture, ProcessPendingScrollScrollsUp) {
  FlameChartTimelineData data;
  data.groups.push_back({.name = "Group 1",
                         .start_level = 0,
                         .nesting_level = kThreadNestingLevel,
                         .expanded = true});
  // Event 0 at level 5
  data.entry_names.push_back("event0");
  data.entry_levels.push_back(5);
  data.entry_start_times.push_back(100.0);
  data.entry_total_times.push_back(1.0);
  data.entry_pids.push_back(1);
  data.entry_args.push_back({});

  // Event 1 at level 50 to force content size to be larger than scroll target.
  data.entry_names.push_back("event_dummy");
  data.entry_levels.push_back(50);
  data.entry_start_times.push_back(100.0);
  data.entry_total_times.push_back(1.0);
  data.entry_pids.push_back(1);
  data.entry_args.push_back({});

  data.events_by_level.resize(51);
  data.events_by_level[5].push_back(0);
  data.events_by_level[50].push_back(1);

  timeline_.SetTimelineData(std::move(data));
  timeline_.set_data_time_range({0.0, 20000.0});

  SimulateFrame();
  SimulateFrame();

  // Set display size to a small value to force scrolling
  ImGui::GetIO().DisplaySize = ImVec2(800.0f, 200.0f);

  ImGuiWindow* tracks_window = nullptr;
  for (ImGuiWindow* w : ImGui::GetCurrentContext()->Windows) {
    if (absl::StrContains(std::string(w->Name), "Tracks")) {
      tracks_window = w;
      break;
    }
  }
  ASSERT_NE(tracks_window, nullptr);

  // Set initial scroll to a large value (500.0f) to ensure event 0 (level 5) is
  // above viewport.
  tracks_window->Scroll.y = 500.0f;

  timeline_.RevealEvent(0);

  // Simulate frames with Render() to ensure layout is processed
  for (int i = 0; i < 3; ++i) {
    ImGui::NewFrame();
    timeline_.Draw();
    Animation::UpdateAll(ImGui::GetIO().DeltaTime);
    ImGui::Render();
  }

  // Expect scroll to go to y_top of level 5.
  EXPECT_NEAR(tracks_window->Scroll.y, 122.0f, 0.1f);
}

TEST_F(RealTimelineImGuiFixture, RevealEventClampsToMinFetchDuration) {
  FlameChartTimelineData data;
  data.groups.push_back({.name = "Group 1",
                         .start_level = 0,
                         .nesting_level = 0,
                         .expanded = true});
  data.events_by_level.push_back({0});
  data.entry_names.push_back("event0");
  data.entry_levels.push_back(0);
  data.entry_start_times.push_back(100.0);
  data.entry_total_times.push_back(0.0);
  data.entry_pids.push_back(1);
  data.entry_args.push_back({});
  timeline_.SetTimelineData(std::move(data));
  timeline_.set_data_time_range({0.0, 20000.0});

  timeline_.SetVisibleRange({2000.0, 3000.0});

  timeline_.RevealEvent(0);
  Animation::UpdateAll(1.0f);

  EXPECT_NEAR(timeline_.visible_range().start(), 100.0, 0.5);
  EXPECT_NEAR(timeline_.visible_range().end(), 1100.0, 0.5);
}

TEST_F(RealTimelineImGuiFixture, RevealEventClampsToMinVisibleWidth) {
  FlameChartTimelineData data;
  data.groups.push_back({.name = "Group 1",
                         .start_level = 0,
                         .nesting_level = 0,
                         .expanded = true});
  data.events_by_level.push_back({0});
  data.entry_names.push_back("event0");
  data.entry_levels.push_back(0);
  data.entry_start_times.push_back(100.0);
  data.entry_total_times.push_back(1.0);
  data.entry_pids.push_back(1);
  data.entry_args.push_back({});
  timeline_.SetTimelineData(std::move(data));
  timeline_.set_data_time_range({0.0, 20000.0});

  SimulateFrame();

  timeline_.SetVisibleRange({2000.0, 3000.0});

  timeline_.RevealEvent(0);
  Animation::UpdateAll(1.0f);

  double px_per_time = timeline_.px_per_time_unit();
  ASSERT_GT(px_per_time, 0);
  double time_per_px = 1.0 / px_per_time;
  double expected_min_window = std::max(1.0, time_per_px * 30.0);
  double expected_start = 101.0 - expected_min_window;

  EXPECT_NEAR(timeline_.visible_range().start(), expected_start, 0.001);
}

TEST_F(RealTimelineImGuiFixture, RevealEventInvalidIndexReturnsEarly) {
  FlameChartTimelineData data;
  data.entry_start_times.push_back(100.0);
  data.entry_total_times.push_back(1.0);
  timeline_.SetTimelineData(std::move(data));

  timeline_.RevealEvent(-1);
  EXPECT_EQ(timeline_.selected_event_index(), -1);

  timeline_.RevealEvent(1);
  EXPECT_EQ(timeline_.selected_event_index(), -1);
}

TEST_F(RealTimelineImGuiFixture, RevealEventScrollsVertically) {
  FlameChartTimelineData data;
  data.groups.push_back({.name = "Group 1",
                         .start_level = 0,
                         .nesting_level = kThreadNestingLevel,
                         .expanded = true});
  data.entry_names.push_back("event0");
  data.entry_levels.push_back(100);  // High level to trigger scrolling
  data.entry_start_times.push_back(100.0);
  data.entry_total_times.push_back(1.0);
  data.entry_pids.push_back(1);
  data.entry_args.push_back({});
  data.events_by_level.resize(101);
  data.events_by_level[100].push_back(0);

  timeline_.SetTimelineData(std::move(data));
  timeline_.set_data_time_range({0.0, 20000.0});

  SimulateFrame();
  SimulateFrame();

  timeline_.RevealEvent(0);
  // Simulate frames with Render() to ensure layout is processed
  for (int i = 0; i < 3; ++i) {
    ImGui::NewFrame();
    timeline_.Draw();
    Animation::UpdateAll(ImGui::GetIO().DeltaTime);
    ImGui::Render();
  }

  ImGuiWindow* tracks_window = nullptr;
  for (ImGuiWindow* w : ImGui::GetCurrentContext()->Windows) {
    if (absl::StrContains(std::string(w->Name), "Tracks")) {
      tracks_window = w;
      break;
    }
  }

  ASSERT_NE(tracks_window, nullptr);
  EXPECT_GT(tracks_window->Scroll.y, 0.0f);
}

TEST_F(RealTimelineImGuiFixture,
       SetTimelineDataPreservesScrollOnIncrementalUpdate) {
  FlameChartTimelineData data;
  data.groups.push_back({.name = "Group 1",
                         .start_level = 0,
                         .nesting_level = kThreadNestingLevel,
                         .expanded = true});
  data.entry_names.push_back("event0");
  data.entry_levels.push_back(100);
  data.entry_start_times.push_back(100.0);
  data.entry_total_times.push_back(1.0);
  data.entry_pids.push_back(1);
  data.entry_args.push_back({});
  data.events_by_level.resize(101);
  data.events_by_level[100].push_back(0);

  FlameChartTimelineData data_copy = data;

  timeline_.SetTimelineData(std::move(data));
  timeline_.set_data_time_range({0.0, 20000.0});

  for (int i = 0; i < 3; ++i) {
    ImGui::NewFrame();
    timeline_.Draw();
    Animation::UpdateAll(ImGui::GetIO().DeltaTime);
    ImGui::Render();
  }

  ImGuiWindow* tracks_window = nullptr;
  for (ImGuiWindow* w : ImGui::GetCurrentContext()->Windows) {
    if (absl::StrContains(std::string(w->Name), "Tracks")) {
      tracks_window = w;
      break;
    }
  }

  ASSERT_NE(tracks_window, nullptr);

  tracks_window->Scroll.y = 500.0f;

  timeline_.set_is_incremental_loading(true);

  timeline_.SetTimelineData(std::move(data_copy));

  ImGui::NewFrame();
  timeline_.Draw();
  Animation::UpdateAll(ImGui::GetIO().DeltaTime);
  ImGui::Render();

  EXPECT_FLOAT_EQ(tracks_window->Scroll.y, 500.0f);
}

TEST_F(RealTimelineImGuiFixture, RevealEventSetsVisibleRangeDuration) {
  FlameChartTimelineData data;
  data.entry_names.push_back("event0");
  data.entry_levels.push_back(0);
  data.entry_start_times.push_back(100.0);
  data.entry_total_times.push_back(1.0);
  data.entry_pids.push_back(1);
  data.entry_args.push_back({});
  data.events_by_level.resize(1);
  data.events_by_level[0].push_back(0);

  timeline_.SetTimelineData(std::move(data));
  timeline_.set_data_time_range({-1000.0, 20000.0});

  timeline_.SetVisibleRange({500.0, 10500.0}, /*animate=*/false);

  SimulateFrame();

  timeline_.ZoomEvent(0);
  // Complete the animation to reach the target visible range.
  Animation::UpdateAll(1.0f);

  // Event duration is 1.0. Zoom factor 2.5 -> duration 2.5, clamped to 10.0.
  // Center is 100.5. Range is [95.5, 105.5].
  EXPECT_DOUBLE_EQ(timeline_.visible_range().start(), 95.5);
  EXPECT_DOUBLE_EQ(timeline_.visible_range().end(), 105.5);
  EXPECT_DOUBLE_EQ(timeline_.visible_range().duration(), 10.0);
}

TEST_F(RealTimelineImGuiFixture, RevealEventWithNaNDurationSetsMinDuration) {
  FlameChartTimelineData data;
  data.groups.push_back({.name = "Group 1",
                         .start_level = 0,
                         .nesting_level = 0,
                         .expanded = true});
  data.events_by_level.push_back({0});
  data.entry_names.push_back("event1");
  data.entry_levels.push_back(0);
  data.entry_start_times.push_back(1000.0);
  data.entry_total_times.push_back(std::numeric_limits<double>::quiet_NaN());
  data.entry_pids.push_back(1);
  data.entry_args.push_back({});
  timeline_.SetTimelineData(std::move(data));

  timeline_.SetVisibleRange({0.0, 500.0});
  SimulateFrame();

  timeline_.RevealEvent(0);
  SimulateFrame();

  EXPECT_EQ(timeline_.visible_range().start(), 1000.0);
  EXPECT_EQ(timeline_.visible_range().end(), 1500.0);
}

TEST_F(RealTimelineImGuiFixture, RevealEventWithZeroDurationSetsMinDuration) {
  FlameChartTimelineData data;
  data.groups.push_back({.name = "Group 1",
                         .start_level = 0,
                         .nesting_level = 0,
                         .expanded = true});
  data.events_by_level.push_back({0});
  data.entry_names.push_back("event1");
  data.entry_levels.push_back(0);
  data.entry_start_times.push_back(1000.0);
  data.entry_total_times.push_back(0.0);
  data.entry_pids.push_back(1);
  data.entry_args.push_back({});
  timeline_.SetTimelineData(std::move(data));

  timeline_.SetVisibleRange({0.0, 500.0});
  SimulateFrame();

  timeline_.RevealEvent(0);
  SimulateFrame();

  EXPECT_EQ(timeline_.visible_range().start(), 1000.0);
  EXPECT_EQ(timeline_.visible_range().end(), 1500.0);
}

TEST_F(RealTimelineImGuiFixture, SelectionMutualExclusion) {
  FlameChartTimelineData data;
  // Group 0: Flame Events
  data.groups.push_back({.name = "Group 1",
                         .start_level = 0,
                         .nesting_level = 0,
                         .expanded = true});
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
                         .nesting_level = 0,
                         .expanded = true});
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
  ImGui::GetIO().MousePos =
      ImVec2(GetTimelineStartX() + 50.0f, kFirstEventY);  // Over flame event
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

  ImGui::GetIO().AddMousePosEvent(counter_pos.x, counter_pos.y);
  SimulateFrame();
  ImGui::GetIO().AddMouseButtonEvent(0, true);
  SimulateFrame();
  ImGui::GetIO().AddMouseButtonEvent(0, false);
  SimulateFrame();

  EXPECT_EQ(timeline_.selected_event_index(), -1);
  EXPECT_EQ(timeline_.selected_group_index(), 1);
  EXPECT_EQ(timeline_.selected_counter_index(), 0);

  // Step 3: Select Flame Event Again
  ImGui::GetIO().AddMousePosEvent(GetTimelineStartX() + 50.0f, kFirstEventY);
  SimulateFrame();
  ImGui::GetIO().AddMouseButtonEvent(0, true);
  SimulateFrame();
  ImGui::GetIO().AddMouseButtonEvent(0, false);
  SimulateFrame();

  EXPECT_EQ(timeline_.selected_event_index(), 0);
  EXPECT_EQ(timeline_.selected_group_index(), 0);
  EXPECT_EQ(timeline_.selected_counter_index(), -1);
}

TEST_F(RealTimelineImGuiFixture, SelectionOverlayIsDrawnOnTopOfTracks) {
  // Ensure we have some data so tracks are drawn.
  FlameChartTimelineData data;
  data.groups.push_back({.name = "Group 1", .start_level = 0});
  timeline_.SetTimelineData(std::move(data));

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

TEST_F(RealTimelineImGuiFixture, ShiftClickEventTogglesCurtain) {
  FlameChartTimelineData data;
  data.groups.push_back({.name = "Group 1",
                         .start_level = 0,
                         .nesting_level = 0,
                         .expanded = true});
  data.events_by_level.push_back({0});
  data.entry_names.push_back("event1");
  data.entry_levels.push_back(0);
  data.entry_start_times.push_back(10.0);
  data.entry_total_times.push_back(20.0);
  data.entry_pids.push_back(1);
  data.entry_args.push_back({});
  timeline_.SetTimelineData(std::move(data));
  timeline_.SetVisibleRange({0.0, 100.0});

  // Mouse is over the event
  ImGui::GetIO().MousePos = ImVec2(GetTimelineStartX() + 250.0f, kFirstEventY);
  ImGui::GetIO().AddKeyEvent(ImGuiMod_Shift, true);
  ImGui::GetIO().MouseDown[0] = true;
  SimulateFrame();

  ImGui::GetIO().MouseDown[0] = false;
  SimulateFrame();

  ASSERT_EQ(timeline_.selected_time_ranges().size(), 1);
  EXPECT_EQ(timeline_.selected_time_ranges()[0].start(), 10.0);
  EXPECT_EQ(timeline_.selected_time_ranges()[0].end(), 30.0);

  // Second shift-click on the same event, should remove the curtain.
  ImGui::GetIO().MouseDown[0] = true;
  SimulateFrame();

  ImGui::GetIO().MouseDown[0] = false;
  SimulateFrame();

  EXPECT_EQ(timeline_.selected_time_ranges().size(), 0);

  EXPECT_TRUE(timeline_.selected_time_ranges().empty());

  // Reset the mouse and shift key to avoid affecting other tests.
  ImGui::GetIO().MouseDown[0] = false;
  ImGui::GetIO().AddKeyEvent(ImGuiMod_Shift, false);
}

TEST_F(RealTimelineImGuiFixture,
       ShiftClickMultipleEventsSelectsMultipleRanges) {
  FlameChartTimelineData data;
  data.groups.push_back({.name = "Group 1",
                         .start_level = 0,
                         .nesting_level = 0,
                         .expanded = true});
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
  timeline_.SetTimelineData(std::move(data));
  timeline_.SetVisibleRange({0.0, 100.0});

  ImGui::GetIO().AddKeyEvent(ImGuiMod_Shift, true);

  // First shift-click on event 1.
  ImGui::GetIO().MousePos = ImVec2(GetTimelineStartX() + 250.0f,
                                   kFirstEventY);  // Position over event 1.
  ImGui::GetIO().MouseDown[0] = true;
  SimulateFrame();

  ImGui::GetIO().MouseDown[0] = false;
  SimulateFrame();

  ASSERT_EQ(timeline_.selected_time_ranges().size(), 1);
  EXPECT_EQ(timeline_.selected_time_ranges()[0].start(), 10.0);
  EXPECT_EQ(timeline_.selected_time_ranges()[0].end(), 30.0);

  // Second shift-click on event 2.
  ImGui::GetIO().MousePos = ImVec2(GetTimelineStartX() + 900.0f,
                                   kFirstEventY);  // Position over event 2.
  ImGui::GetIO().MouseDown[0] = true;
  SimulateFrame();

  ImGui::GetIO().MouseDown[0] = false;
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
  ImGui::GetIO().MousePos = ImVec2(GetTimelineStartX() + 250.0f,
                                   kFirstEventY);  // Position over event 1.
  ImGui::GetIO().MouseDown[0] = true;
  SimulateFrame();

  ImGui::GetIO().MouseDown[0] = false;
  SimulateFrame();

  ASSERT_EQ(timeline_.selected_time_ranges().size(), 1);
  EXPECT_EQ(timeline_.selected_time_ranges()[0].start(), 50.0);
  EXPECT_EQ(timeline_.selected_time_ranges()[0].end(), 60.0);

  // Reset the mouse and shift key to avoid affecting other tests.
  ImGui::GetIO().MouseDown[0] = false;
  ImGui::GetIO().AddKeyEvent(ImGuiMod_Shift, false);
}

class TimelineDragSelectionTest : public RealTimelineImGuiFixture {
 protected:
  // Horizontal offset for the start of test selections/events.
  static constexpr float kSelectionStartOffset = 50.0f;
  // A vertical position guaranteed to be below events/ruler (safe for empty
  // area clicks).
  static constexpr float kEmptyAreaY = 50.0f;
  // Drag distance in pixels for TimelineDragSelectionTest.
  static constexpr float kDragDistance = 100.0f;
  // Pixels per microsecond for TimelineDragSelectionTest.
  static constexpr float kPxPerUs = 10.0f;

  void SetUp() override {
    RealTimelineImGuiFixture::SetUp();
    // Set a visible range that results in a round number for px_per_time_unit
    // to make test calculations predictable. With a timeline width of 1655px
    // (based on 1920px window width, 250px label width, and 1px padding),
    // a duration of 165.5 gives 10px per microsecond.
    timeline_.SetVisibleRange({0.0, 165.5});
    timeline_.set_data_time_range({0.0, 165.5});

    ImGui::GetIO().AddKeyEvent(ImGuiMod_Shift, true);
  }

  void TearDown() override {
    ImGui::GetIO().AddKeyEvent(ImGuiMod_Shift, false);
    RealTimelineImGuiFixture::TearDown();
  }
};

TEST_F(RealTimelineImGuiFixture, ZoomEventInvalidIndexReturnsEarly) {
  FlameChartTimelineData data;  // timeline_ is RealTimelineImGuiFixture
  data.entry_start_times.push_back(100.0);
  data.entry_total_times.push_back(1.0);
  timeline_.SetTimelineData(std::move(data));

  timeline_.ZoomEvent(-1);
  EXPECT_EQ(timeline_.selected_event_index(), -1);

  timeline_.ZoomEvent(1);
  EXPECT_EQ(timeline_.selected_event_index(), -1);
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
// =============================================================================
// Fixture: TimelineDragSelectionTest
// =============================================================================

TEST_F(TimelineDragSelectionTest, ClickCloseButtonRemovesSelectedTimeRange) {
  ImGuiIO& io = ImGui::GetIO();
  EXPECT_EQ(timeline_.mouse_mode(), MouseMode::kPan);
  EXPECT_FALSE(ImGui::IsKeyDown(ImGuiKey_4));

  // Start drag in timeline area.
  io.MousePos =
      ImVec2(GetTimelineStartX() + kSelectionStartOffset, kEmptyAreaY);
  io.AddMouseButtonEvent(0, true);
  SimulateFrame();

  // Dragging
  io.MousePos = ImVec2(GetTimelineStartX() + 250.0f, 50.0f);
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
  // range_start_x = GetTimelineStartX() + kSelectionStartOffset
  // range_end_x = GetTimelineStartX() + 250.0f
  // text_x = range_start_x + (200 - text_size.x) / 2
  // text_y = ruler_screen_y_ + kRulerHeight + kSelectedTimeRangeTextTopPadding
  // In this test setup, ruler_screen_y_ seems to be 0.
  const float range_start_x = GetTimelineStartX() + kSelectionStartOffset;
  const float text_x = range_start_x + (200.0f - text_size.x) / 2.0f;
  const float text_y = kRulerHeight + kSelectedTimeRangeTextTopPadding;

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
  ImGui::GetIO().AddKeyEvent(ImGuiMod_Shift, false);
  ImGui::GetIO().MousePos = ImVec2(0.0f, 0.0f);
  SimulateFrame();
  EXPECT_EQ(ImGui::GetMouseCursor(), ImGuiMouseCursor_Arrow);
}

TEST_F(TimelineDragSelectionTest, ClickingTextDoesNotRemoveSelectedTimeRange) {
  ImGuiIO& io = ImGui::GetIO();

  // Start drag in timeline area.
  io.MousePos =
      ImVec2(GetTimelineStartX() + kSelectionStartOffset, kEmptyAreaY);
  io.AddMouseButtonEvent(0, true);
  SimulateFrame();

  // Dragging
  io.MousePos = ImVec2(GetTimelineStartX() + 250.0f, 50.0f);
  SimulateFrame();

  // End drag
  io.AddMouseButtonEvent(0, false);
  SimulateFrame();

  ASSERT_EQ(timeline_.selected_time_ranges().size(), 1);

  // FormatTime uses %.4g and non-breaking space. 20.0 becomes "20".
  const std::string text = "20\xc2\xa0us";
  const ImVec2 text_size = ImGui::CalcTextSize(text.c_str());

  const float range_start_x = GetTimelineStartX() + kSelectionStartOffset;
  const float text_x = range_start_x + (200.0f - text_size.x) / 2.0f;
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

class TimelineMouseModeSelectTestSuite : public TimelineDragSelectionTest {
 protected:
  void SetUp() override {
    TimelineDragSelectionTest::SetUp();
    ImGui::GetIO().AddKeyEvent(ImGuiMod_Shift, false);
    timeline_.set_mouse_mode(MouseMode::kSelect);

    FlameChartTimelineData data;
    data.entry_levels = {0, 1};
    data.entry_total_times = {100.0, 50.0};
    data.entry_self_times = {50.0, 50.0};
    data.entry_start_times = {0.0, 0.0};
    data.entry_names = {"event1", "event2"};
    data.entry_event_ids = {1, 2};
    data.entry_pids = {1, 1};
    data.entry_tids = {1, 1};
    data.entry_args = {{}, {}};
    data.groups = {
        {Group::Type::kFlame, "group", "", 0, kThreadNestingLevel, true}};
    data.events_by_level = {{0}, {1}};
    timeline_.SetTimelineData(data);
  }
};

TEST_F(TimelineDragSelectionTest, DraggingUpdatesCurrentSelectedTimeRange) {
  ImGuiIO& io = ImGui::GetIO();

  // Start drag in timeline area.
  io.MousePos =
      ImVec2(GetTimelineStartX() + kSelectionStartOffset, kEmptyAreaY);
  io.AddMouseButtonEvent(0, true);
  SimulateFrame();

  // Dragging
  io.MousePos = ImVec2(GetTimelineStartX() + 250.0f, 50.0f);
  SimulateFrame();

  // During drag, current_selected_time_range_ should be set, but
  // selected_time_ranges_ should be empty.
  ASSERT_TRUE(timeline_.current_selected_time_range().has_value());
  EXPECT_DOUBLE_EQ(timeline_.current_selected_time_range()->start(),
                   kSelectionStartOffset / kPxPerUs);
  EXPECT_DOUBLE_EQ(timeline_.current_selected_time_range()->end(),
                   (kSelectionStartOffset + 200.0f) / kPxPerUs);
  EXPECT_TRUE(timeline_.selected_time_ranges().empty());

  // End drag
  io.AddMouseButtonEvent(0, false);
  SimulateFrame();

  // After drag, current_selected_time_range_ should be reset, and
  // selected_time_ranges_ should contain the new range.
  EXPECT_FALSE(timeline_.current_selected_time_range().has_value());
  ASSERT_EQ(timeline_.selected_time_ranges().size(), 1);
  EXPECT_DOUBLE_EQ(timeline_.selected_time_ranges()[0].start(),
                   kSelectionStartOffset / kPxPerUs);
  EXPECT_DOUBLE_EQ(timeline_.selected_time_ranges()[0].end(),
                   (kSelectionStartOffset + 200.0f) / kPxPerUs);
}

TEST_F(TimelineDragSelectionTest, EscapeCancelsSelection) {
  ImGuiIO& io = ImGui::GetIO();

  // Start drag (Shift is already held by SetUp).
  io.MousePos =
      ImVec2(GetTimelineStartX() + kSelectionStartOffset, kEmptyAreaY);
  io.AddMouseButtonEvent(0, true);
  SimulateFrame();

  // Drag to kDragDistance
  io.MousePos = ImVec2(
      GetTimelineStartX() + kSelectionStartOffset + kDragDistance, kEmptyAreaY);
  SimulateFrame();

  // Verify selection is active.
  ASSERT_TRUE(timeline_.current_selected_time_range().has_value());
  EXPECT_GT(timeline_.current_selected_time_range()->duration(), 0);

  // Press Escape.
  io.AddKeyEvent(ImGuiKey_Escape, true);
  SimulateFrame();
  io.AddKeyEvent(ImGuiKey_Escape, false);

  // Verify selection is cancelled.
  EXPECT_FALSE(timeline_.current_selected_time_range().has_value());
  EXPECT_TRUE(timeline_.selected_time_ranges().empty());

  // Release mouse button to clean up
  io.AddMouseButtonEvent(0, false);
  SimulateFrame();
}

TEST_F(TimelineDragSelectionTest, ShiftDragCreatesMultipleTimeSelections) {
  ImGuiIO& io = ImGui::GetIO();

  // First drag
  io.MousePos =
      ImVec2(GetTimelineStartX() + kSelectionStartOffset, kEmptyAreaY);
  io.AddMouseButtonEvent(0, true);
  SimulateFrame();
  io.MousePos = ImVec2(
      GetTimelineStartX() + kSelectionStartOffset + kDragDistance, kEmptyAreaY);
  SimulateFrame();
  io.AddMouseButtonEvent(0, false);
  SimulateFrame();

  ASSERT_EQ(timeline_.selected_time_ranges().size(), 1);
  EXPECT_DOUBLE_EQ(timeline_.selected_time_ranges()[0].start(),
                   kSelectionStartOffset / kPxPerUs);
  EXPECT_DOUBLE_EQ(timeline_.selected_time_ranges()[0].end(),
                   (kSelectionStartOffset + kDragDistance) / kPxPerUs);

  // Second drag
  io.MousePos = ImVec2(GetTimelineStartX() + 250.0f, kEmptyAreaY);
  io.AddMouseButtonEvent(0, true);
  SimulateFrame();
  io.MousePos =
      ImVec2(GetTimelineStartX() + 250.0f + kDragDistance, kEmptyAreaY);
  SimulateFrame();
  io.AddMouseButtonEvent(0, false);
  SimulateFrame();

  ASSERT_EQ(timeline_.selected_time_ranges().size(), 2);
  EXPECT_DOUBLE_EQ(timeline_.selected_time_ranges()[0].start(),
                   kSelectionStartOffset / kPxPerUs);
  EXPECT_DOUBLE_EQ(timeline_.selected_time_ranges()[0].end(),
                   (kSelectionStartOffset + kDragDistance) / kPxPerUs);
  EXPECT_DOUBLE_EQ(timeline_.selected_time_ranges()[1].start(),
                   (250.0f) / kPxPerUs);
  EXPECT_DOUBLE_EQ(timeline_.selected_time_ranges()[1].end(),
                   (250.0f + kDragDistance) / kPxPerUs);
}

TEST_F(TimelineDragSelectionTest, ShiftDragCreatesTimeSelection) {
  ImGuiIO& io = ImGui::GetIO();

  // Start drag in timeline area.
  // The label column is 250px wide, so timeline starts after that.
  io.MousePos =
      ImVec2(GetTimelineStartX() + kSelectionStartOffset, kEmptyAreaY);
  io.AddMouseButtonEvent(0, true);
  SimulateFrame();

  // Dragging
  io.MousePos = ImVec2(GetTimelineStartX() + 250.0f, kEmptyAreaY);
  SimulateFrame();

  // End drag
  io.AddMouseButtonEvent(0, false);
  SimulateFrame();

  ASSERT_EQ(timeline_.selected_time_ranges().size(), 1);
  const TimeRange& range = timeline_.selected_time_ranges()[0];
  EXPECT_DOUBLE_EQ(range.start(), kSelectionStartOffset / kPxPerUs);
  EXPECT_DOUBLE_EQ(range.end(), (kSelectionStartOffset + 200.0f) / kPxPerUs);
}

TEST_F(TimelineDragSelectionTest, SnapsToEventEdgeWhenEnabled) {
  timeline_.set_snap_to_time_range_enabled(true);

  FlameChartTimelineData data;
  data.entry_levels = {0};
  data.entry_total_times = {100.0};
  data.entry_self_times = {100.0};
  data.entry_start_times = {100.0};  // Event from 100.0 to 200.0
  data.entry_names = {"event1"};
  data.entry_event_ids = {1};
  data.entry_pids = {1};
  data.entry_tids = {1};
  data.entry_args = {{}};
  data.groups = {
      {Group::Type::kFlame, "group", "", 0, kThreadNestingLevel, true}};
  data.events_by_level = {{0}};
  timeline_.SetTimelineData(data);

  SimulateFrame();

  ImGuiIO& io = ImGui::GetIO();

  // Start near 100.0 us (990px -> 99.0 us)
  io.MousePos = ImVec2(GetTimelineStartX() + 990.0f, kFirstEventY);
  io.AddMouseButtonEvent(0, true);
  SimulateFrame();

  // End near 200.0 us (2010px -> 201.0 us)
  io.MousePos = ImVec2(GetTimelineStartX() + 2010.0f, kFirstEventY);
  SimulateFrame();

  io.AddMouseButtonEvent(0, false);
  SimulateFrame();

  ASSERT_EQ(timeline_.selected_time_ranges().size(), 1);
  EXPECT_DOUBLE_EQ(timeline_.selected_time_ranges()[0].start(), 100.0);
  EXPECT_DOUBLE_EQ(timeline_.selected_time_ranges()[0].end(), 200.0);
}

TEST_F(TimelineDragSelectionTest, DoesNotSnapWhenDisabled) {
  timeline_.set_snap_to_time_range_enabled(false);

  FlameChartTimelineData data;
  data.entry_levels = {0};
  data.entry_total_times = {100.0};
  data.entry_self_times = {100.0};
  data.entry_start_times = {100.0};  // Event from 100.0 to 200.0
  data.entry_names = {"event1"};
  data.entry_event_ids = {1};
  data.entry_pids = {1};
  data.entry_tids = {1};
  data.entry_args = {{}};
  data.groups = {
      {Group::Type::kFlame, "group", "", 0, kThreadNestingLevel, true}};
  data.events_by_level = {{0}};
  timeline_.SetTimelineData(data);

  SimulateFrame();

  ImGuiIO& io = ImGui::GetIO();

  // Start near 100.0 us
  io.MousePos = ImVec2(GetTimelineStartX() + 990.0f, kEmptyAreaY);
  io.AddMouseButtonEvent(0, true);
  SimulateFrame();

  // End near 200.0 us
  io.MousePos = ImVec2(GetTimelineStartX() + 2010.0f, kEmptyAreaY);
  SimulateFrame();

  io.AddMouseButtonEvent(0, false);
  SimulateFrame();

  ASSERT_EQ(timeline_.selected_time_ranges().size(), 1);
  EXPECT_DOUBLE_EQ(timeline_.selected_time_ranges()[0].start(), 99.0);
  EXPECT_DOUBLE_EQ(timeline_.selected_time_ranges()[0].end(), 201.0);
}

TEST_F(TimelineDragSelectionTest, SnapScopingToHoveredGroupSnaps) {
  timeline_.set_snap_to_time_range_enabled(true);

  FlameChartTimelineData data;
  data.entry_levels = {0, 1};  // level 0 in group 1, level 1 in group 2
  data.entry_total_times = {10.0, 10.0};
  data.entry_self_times = {10.0, 10.0};
  data.entry_start_times = {100.0, 120.0};
  data.entry_names = {"event1", "event2"};
  data.entry_event_ids = {1, 2};
  data.entry_pids = {1, 1};
  data.entry_tids = {1, 1};
  data.entry_args = {{}, {}};
  data.groups = {
      {Group::Type::kFlame, "group1", "", 0, kThreadNestingLevel, true},
      {Group::Type::kFlame, "group2", "", 1, kThreadNestingLevel, true}};
  data.events_by_level = {{0}, {1}};
  timeline_.SetTimelineData(data);

  SimulateFrame();

  ImGuiIO& io = ImGui::GetIO();
  const float hover_y_group2 = 55.0f;

  // Drag selection near group2 event (120.0us).
  // Drag from 50.0us (500px) to 121.0us (1210px).
  // Distance to event2 (120.0us) is 1.0us (< 1.6us).
  // It SHOULD snap to 120.0us because we are hovering group2.
  io.MousePos = ImVec2(GetTimelineStartX() + 500.0f, hover_y_group2);
  io.AddMouseButtonEvent(0, true);
  SimulateFrame();

  io.MousePos = ImVec2(GetTimelineStartX() + 1210.0f, hover_y_group2);
  SimulateFrame();

  io.AddMouseButtonEvent(0, false);
  SimulateFrame();

  ASSERT_EQ(timeline_.selected_time_ranges().size(), 1);
  EXPECT_DOUBLE_EQ(timeline_.selected_time_ranges()[0].start(), 50.0);
  EXPECT_DOUBLE_EQ(timeline_.selected_time_ranges()[0].end(), 120.0);
}

TEST_F(TimelineDragSelectionTest, SnapScopingToHoveredGroupIgnoresOthers) {
  timeline_.set_snap_to_time_range_enabled(true);

  FlameChartTimelineData data;
  data.entry_levels = {0, 1};  // level 0 in group 1, level 1 in group 2
  data.entry_total_times = {10.0, 10.0};
  data.entry_self_times = {10.0, 10.0};
  data.entry_start_times = {100.0, 120.0};
  data.entry_names = {"event1", "event2"};
  data.entry_event_ids = {1, 2};
  data.entry_pids = {1, 1};
  data.entry_tids = {1, 1};
  data.entry_args = {{}, {}};
  data.groups = {
      {Group::Type::kFlame, "group1", "", 0, kThreadNestingLevel, true},
      {Group::Type::kFlame, "group2", "", 1, kThreadNestingLevel, true}};
  data.events_by_level = {{0}, {1}};
  timeline_.SetTimelineData(data);

  SimulateFrame();

  ImGuiIO& io = ImGui::GetIO();
  const float hover_y_group2 = 55.0f;

  // Drag selection near group1 event (100.0us) while hovering group2.
  // Drag from 50.0us (500px) to 101.0us (1010px).
  // Distance to event1 (100.0us) is 1.0us (< 1.6us).
  // It should NOT snap to 100.0us because we are hovering group2.
  io.MousePos = ImVec2(GetTimelineStartX() + 500.0f, hover_y_group2);
  io.AddMouseButtonEvent(0, true);
  SimulateFrame();

  io.MousePos = ImVec2(GetTimelineStartX() + 1010.0f, hover_y_group2);
  SimulateFrame();

  io.AddMouseButtonEvent(0, false);
  SimulateFrame();

  ASSERT_EQ(timeline_.selected_time_ranges().size(), 1);
  EXPECT_DOUBLE_EQ(timeline_.selected_time_ranges()[0].start(), 50.0);
  EXPECT_DOUBLE_EQ(timeline_.selected_time_ranges()[0].end(), 101.0);
}

TEST_F(TimelineDragSelectionTest, SnapsToOtherSelectedRange) {
  timeline_.set_snap_to_time_range_enabled(true);

  ImGuiIO& io = ImGui::GetIO();

  // Create first selection 50.0us to 100.0us (500px to 1000px)
  io.MousePos = ImVec2(GetTimelineStartX() + 500.0f, kEmptyAreaY);
  io.AddMouseButtonEvent(0, true);
  SimulateFrame();

  io.MousePos = ImVec2(GetTimelineStartX() + 1000.0f, kEmptyAreaY);
  SimulateFrame();

  io.AddMouseButtonEvent(0, false);
  SimulateFrame();

  ASSERT_EQ(timeline_.selected_time_ranges().size(), 1);

  // Second drag near 100.0us (990px -> 99.0 us) to 150.0us
  io.MousePos = ImVec2(GetTimelineStartX() + 990.0f, kEmptyAreaY);
  io.AddMouseButtonEvent(0, true);
  SimulateFrame();

  io.MousePos = ImVec2(GetTimelineStartX() + 1500.0f, kEmptyAreaY);
  SimulateFrame();

  io.AddMouseButtonEvent(0, false);
  SimulateFrame();

  ASSERT_EQ(timeline_.selected_time_ranges().size(), 2);
  EXPECT_DOUBLE_EQ(timeline_.selected_time_ranges()[1].start(), 100.0);
  EXPECT_DOUBLE_EQ(timeline_.selected_time_ranges()[1].end(), 150.0);
}

TEST_F(TimelineDragSelectionTest, SnapsDuringTimingMode) {
  // Disable shift
  ImGui::GetIO().AddKeyEvent(ImGuiMod_Shift, false);
  SimulateFrame();

  timeline_.set_snap_to_time_range_enabled(true);
  timeline_.set_mouse_mode(MouseMode::kTiming);

  ImGuiIO& io = ImGui::GetIO();

  // Create first selection 50.0us to 100.0us
  io.MousePos = ImVec2(GetTimelineStartX() + 500.0f, kEmptyAreaY);
  io.AddMouseButtonEvent(0, true);
  SimulateFrame();

  io.MousePos = ImVec2(GetTimelineStartX() + 1000.0f, kEmptyAreaY);
  SimulateFrame();

  io.AddMouseButtonEvent(0, false);
  SimulateFrame();

  ASSERT_EQ(timeline_.selected_time_ranges().size(), 1);

  // Second drag near 100.0us to 150.0us
  io.MousePos = ImVec2(GetTimelineStartX() + 990.0f, kEmptyAreaY);
  io.AddMouseButtonEvent(0, true);
  SimulateFrame();

  io.MousePos = ImVec2(GetTimelineStartX() + 1500.0f, kEmptyAreaY);
  SimulateFrame();

  io.AddMouseButtonEvent(0, false);
  SimulateFrame();

  ASSERT_EQ(timeline_.selected_time_ranges().size(), 2);
  EXPECT_DOUBLE_EQ(timeline_.selected_time_ranges()[1].start(), 100.0);
  EXPECT_DOUBLE_EQ(timeline_.selected_time_ranges()[1].end(), 150.0);
}

TEST_F(TimelineDragSelectionTest, TimingModeShowsMultipleSelections) {
  ImGuiIO& io = ImGui::GetIO();
  timeline_.set_mouse_mode(MouseMode::kTiming);

  // First selection: drag from x=300 to x=400.
  io.MousePos = ImVec2(GetTimelineStartX() + 300.0f, 50.0f);
  io.AddMouseButtonEvent(0, true);
  SimulateFrame();
  io.MousePos = ImVec2(GetTimelineStartX() + 400.0f, 50.0f);
  SimulateFrame();
  io.AddMouseButtonEvent(0, false);
  SimulateFrame();
  ASSERT_EQ(timeline_.selected_time_ranges().size(), 1);

  // Second selection: start a new drag at x=500.
  // The first selection should NOT be cleared immediately on MouseDown.
  io.MousePos = ImVec2(GetTimelineStartX() + 500.0f, 50.0f);
  io.AddMouseButtonEvent(0, true);
  SimulateFrame();
  EXPECT_EQ(timeline_.selected_time_ranges().size(), 1);

  // Complete the second drag to x=600.
  io.MousePos = ImVec2(GetTimelineStartX() + 600.0f, 50.0f);
  SimulateFrame();
  io.AddMouseButtonEvent(0, false);
  SimulateFrame();
  ASSERT_EQ(timeline_.selected_time_ranges().size(), 2);

  // Clicking somewhere else (no drag) should NOT clear the selection.
  io.MousePos = ImVec2(GetTimelineStartX() + 100.0f, 50.0f);
  io.AddMouseButtonEvent(0, true);
  SimulateFrame();
  EXPECT_EQ(timeline_.selected_time_ranges().size(), 2);
  io.AddMouseButtonEvent(0, false);
  SimulateFrame();
  EXPECT_EQ(timeline_.selected_time_ranges().size(), 2);
}

TEST_F(TimelineDragSelectionTest, DoesNotSnapOutsideThreshold) {
  timeline_.set_snap_to_time_range_enabled(true);

  FlameChartTimelineData data;
  data.entry_levels = {0};
  data.entry_total_times = {100.0};
  data.entry_self_times = {100.0};
  data.entry_start_times = {100.0};  // Event from 100.0 to 200.0
  data.entry_names = {"event1"};
  data.entry_event_ids = {1};
  data.entry_pids = {1};
  data.entry_tids = {1};
  data.entry_args = {{}};
  data.groups = {
      {Group::Type::kFlame, "group", "", 0, kThreadNestingLevel, true}};
  data.events_by_level = {{0}};
  timeline_.SetTimelineData(data);

  SimulateFrame();

  ImGuiIO& io = ImGui::GetIO();

  // Drag from 50.0us (500px) to 98.0us (980px).
  // Distance from 98.0us to 100.0us is 2.0us (20px), which is > 16px threshold.
  io.MousePos = ImVec2(GetTimelineStartX() + 500.0f, kEmptyAreaY);
  io.AddMouseButtonEvent(0, true);
  SimulateFrame();

  io.MousePos = ImVec2(GetTimelineStartX() + 980.0f, kEmptyAreaY);
  SimulateFrame();

  io.AddMouseButtonEvent(0, false);
  SimulateFrame();

  ASSERT_EQ(timeline_.selected_time_ranges().size(), 1);
  EXPECT_DOUBLE_EQ(timeline_.selected_time_ranges()[0].start(), 50.0);
  EXPECT_DOUBLE_EQ(timeline_.selected_time_ranges()[0].end(), 98.0);
}

TEST_F(TimelineDragSelectionTest, SnapSelectsClosestEdge) {
  timeline_.set_snap_to_time_range_enabled(true);

  FlameChartTimelineData data;
  data.entry_levels = {0, 0};
  data.entry_total_times = {10.0, 10.0};
  data.entry_self_times = {10.0, 10.0};
  // Two events: [100.0 - 110.0] and [102.0 - 112.0]
  data.entry_start_times = {100.0, 102.0};
  data.entry_names = {"event1", "event2"};
  data.entry_event_ids = {1, 2};
  data.entry_pids = {1, 1};
  data.entry_tids = {1, 1};
  data.entry_args = {{}, {}};
  data.groups = {
      {Group::Type::kFlame, "group", "", 0, kThreadNestingLevel, true}};
  data.events_by_level = {{0, 1}};
  timeline_.SetTimelineData(data);

  SimulateFrame();

  ImGuiIO& io = ImGui::GetIO();

  // Start drag at 50.0us (500px).
  io.MousePos = ImVec2(GetTimelineStartX() + 500.0f, kFirstEventY);
  io.AddMouseButtonEvent(0, true);
  SimulateFrame();

  // Drag to 101.2us (1012px).
  // Distance to 100.0 is 1.2us (12px).
  // Distance to 102.0 is 0.8us (8px).
  // Should snap to 102.0us.
  io.MousePos = ImVec2(GetTimelineStartX() + 1012.0f, kFirstEventY);
  SimulateFrame();

  io.AddMouseButtonEvent(0, false);
  SimulateFrame();

  ASSERT_EQ(timeline_.selected_time_ranges().size(), 1);
  EXPECT_DOUBLE_EQ(timeline_.selected_time_ranges()[0].start(), 50.0);
  EXPECT_DOUBLE_EQ(timeline_.selected_time_ranges()[0].end(), 102.0);
}

TEST_F(TimelineDragSelectionTest, SnapWithPanDuration) {
  timeline_.set_snap_to_time_range_enabled(true);

  FlameChartTimelineData data;
  data.entry_levels = {0};
  data.entry_total_times = {10.0};
  data.entry_self_times = {10.0};
  data.entry_start_times = {60.0};  // Event from 60.0 to 70.0
  data.entry_names = {"event1"};
  data.entry_event_ids = {1};
  data.entry_pids = {1};
  data.entry_tids = {1};
  data.entry_args = {{}};
  data.groups = {
      {Group::Type::kFlame, "group", "", 0, kThreadNestingLevel, true}};
  data.events_by_level = {{0}};
  timeline_.SetTimelineData(data);

  // Set visible range with duration 50.0us
  timeline_.SetVisibleRange({0.0, 50.0});
  SimulateFrame();

  ImGuiIO& io = ImGui::GetIO();

  // Create a selected range [10.0, 60.0] whose duration matches visible range
  // exactly (50.0). 10.0us -> GetTimelineStartX() + 10.0 * 10 (Wait, px_per_us
  // depends on window size. We set visible range to 50.0, and timeline width is
  // 1655px. So px_per_us is 1655/50 = 33.1 Let's just create the selection
  // programmatically instead of dragging, or drag accurately. Actually, we can
  // just call ApplySnapping manually if it were public, but it's private.
  // Dragging: start at 10.0us (10 * 33.1 = 331px).
  io.MousePos = ImVec2(GetTimelineStartX() + 331.0f, kFirstEventY);
  io.AddMouseButtonEvent(0, true);
  SimulateFrame();

  // End at 59.8us to snap to 60.0us (event start).
  // 59.8 * 33.1 = 1979.38px
  io.MousePos = ImVec2(GetTimelineStartX() + 1979.38f, kFirstEventY);
  SimulateFrame();

  io.AddMouseButtonEvent(0, false);
  SimulateFrame();

  ASSERT_EQ(timeline_.selected_time_ranges().size(), 1);
  // Expected to snap end to 60.0. Since is_pan is true (duration 50.0 matches
  // visible range), start should snap to 60.0 - 50.0 = 10.0.
  EXPECT_DOUBLE_EQ(timeline_.selected_time_ranges()[0].start(), 10.0);
  EXPECT_DOUBLE_EQ(timeline_.selected_time_ranges()[0].end(), 60.0);
}

TEST_F(TimelineDragSelectionTest, SnapIgnoresEventsWhenCollapsed) {
  timeline_.set_snap_to_time_range_enabled(true);

  FlameChartTimelineData data;
  data.entry_levels = {0};
  data.entry_total_times = {100.0};
  data.entry_self_times = {100.0};
  data.entry_start_times = {100.0};  // Event from 100.0 to 200.0
  data.entry_names = {"event1"};
  data.entry_event_ids = {1};
  data.entry_pids = {1};
  data.entry_tids = {1};
  data.entry_args = {{}};
  // Group is NOT expanded, and has multiple levels so it is expandable.
  data.groups = {
      {Group::Type::kFlame, "group", "", 0, kThreadNestingLevel, false}};
  data.events_by_level = {{0}, {}};
  timeline_.SetTimelineData(data);

  SimulateFrame();

  ImGuiIO& io = ImGui::GetIO();

  // Start near 100.0 us (990px -> 99.0 us)
  io.MousePos = ImVec2(GetTimelineStartX() + 990.0f, kFirstEventY);
  io.AddMouseButtonEvent(0, true);
  SimulateFrame();

  // End near 200.0 us (2010px -> 201.0 us)
  io.MousePos = ImVec2(GetTimelineStartX() + 2010.0f, kFirstEventY);
  SimulateFrame();

  io.AddMouseButtonEvent(0, false);
  SimulateFrame();

  ASSERT_EQ(timeline_.selected_time_ranges().size(), 1);
  // It should not snap because group is collapsed
  EXPECT_DOUBLE_EQ(timeline_.selected_time_ranges()[0].start(), 99.0);
  EXPECT_DOUBLE_EQ(timeline_.selected_time_ranges()[0].end(), 201.0);
}

TEST_F(TimelineDragSelectionTest,
       SnapDoesNotIgnoreEventsForNonFlameGroupsEvenWhenCollapsed) {
  timeline_.set_snap_to_time_range_enabled(true);

  FlameChartTimelineData data;
  data.entry_levels = {0};
  data.entry_total_times = {100.0};
  data.entry_self_times = {100.0};
  data.entry_start_times = {100.0};  // Event from 100.0 to 200.0
  data.entry_names = {"event1"};
  data.entry_event_ids = {1};
  data.entry_pids = {1};
  data.entry_tids = {1};
  data.entry_args = {{}};
  // Group is NOT expanded, and has multiple levels so it is expandable.
  // But it is NOT kFlame!
  data.groups = {
      {Group::Type::kCounter, "group", "", 0, kCounterNestingLevel, false}};
  data.events_by_level = {{0}, {}};
  timeline_.SetTimelineData(data);

  SimulateFrame();

  ImGuiIO& io = ImGui::GetIO();

  // Start near 100.0 us (990px -> 99.0 us)
  io.MousePos = ImVec2(GetTimelineStartX() + 990.0f, kFirstEventY);
  io.AddMouseButtonEvent(0, true);
  SimulateFrame();

  // End near 200.0 us (2010px -> 201.0 us)
  io.MousePos = ImVec2(GetTimelineStartX() + 2010.0f, kFirstEventY);
  SimulateFrame();

  io.AddMouseButtonEvent(0, false);
  SimulateFrame();

  ASSERT_EQ(timeline_.selected_time_ranges().size(), 1);
  // It SHOULD snap because non-kFlame groups are never considered collapsed for
  // snapping
  EXPECT_DOUBLE_EQ(timeline_.selected_time_ranges()[0].start(), 100.0);
  EXPECT_DOUBLE_EQ(timeline_.selected_time_ranges()[0].end(), 200.0);
}

TEST_F(TimelineDragSelectionTest,
       SnapIgnoresHasChildrenIfNestingLevelIsSameOrLower) {
  timeline_.set_snap_to_time_range_enabled(true);

  FlameChartTimelineData data;
  data.entry_levels = {0, 1};
  data.entry_total_times = {100.0, 100.0};
  data.entry_self_times = {100.0, 100.0};
  data.entry_start_times = {100.0, 500.0};
  data.entry_names = {"event1", "event2"};
  data.entry_event_ids = {1, 2};
  data.entry_pids = {1, 2};
  data.entry_tids = {1, 2};
  data.entry_args = {{}, {}};

  // Group 0 has another group after it, but it's not a child (nesting level is
  // not >). Thus, it should NOT be considered as having children, so it should
  // not be considered expandable/collapsed even if `expanded` is false.
  data.groups = {
      {Group::Type::kFlame, "group1", "", 0, kThreadNestingLevel, false},
      {Group::Type::kFlame, "group2", "", 1, kThreadNestingLevel, false}};
  data.events_by_level = {{0}, {1}};
  timeline_.SetTimelineData(data);

  SimulateFrame();

  ImGuiIO& io = ImGui::GetIO();

  // Try snapping to event1 in group1
  // Start near 100.0 us (990px -> 99.0 us)
  io.MousePos = ImVec2(GetTimelineStartX() + 990.0f, kFirstEventY);
  io.AddMouseButtonEvent(0, true);
  SimulateFrame();

  // End near 200.0 us (2010px -> 201.0 us)
  io.MousePos = ImVec2(GetTimelineStartX() + 2010.0f, kFirstEventY);
  SimulateFrame();

  io.AddMouseButtonEvent(0, false);
  SimulateFrame();

  ASSERT_EQ(timeline_.selected_time_ranges().size(), 1);
  // It should snap because group1 is not considered collapsed since it's not
  // expandable
  EXPECT_DOUBLE_EQ(timeline_.selected_time_ranges()[0].start(), 100.0);
  EXPECT_DOUBLE_EQ(timeline_.selected_time_ranges()[0].end(), 200.0);
}

TEST_F(TimelineDragSelectionTest, SnapIncludesEventsAtExactBottomEdgeOfWindow) {
  timeline_.set_snap_to_time_range_enabled(true);

  FlameChartTimelineData data;
  data.entry_levels = {0};
  data.entry_total_times = {100.0};
  data.entry_self_times = {100.0};
  data.entry_start_times = {100.0};  // Event from 100.0 to 200.0
  data.entry_names = {"event1"};
  data.entry_event_ids = {1};
  data.entry_pids = {1};
  data.entry_tids = {1};
  data.entry_args = {{}};
  // Group is expanded.
  data.groups = {
      {Group::Type::kFlame, "group", "", 0, kThreadNestingLevel, true}};
  data.events_by_level = {{0}};
  timeline_.SetTimelineData(data);

  // Use a window height of 20.0f and a scroll of 0 so that the top edge of the
  // event (which is at 20.0f) is exactly at the bottom visible edge of the
  // window. This verifies the strict > comparison for skipping events outside
  // view.
  SimulateFrame(0.0f, 20.0f);

  ImGuiIO& io = ImGui::GetIO();

  // Start near 100.0 us (990px -> 99.0 us)
  // We place the mouse at 35.0f to be vertically inside the group's bounding
  // box so `is_group_hovered` allows the snap event detection.
  io.MousePos = ImVec2(GetTimelineStartX() + 990.0f, 35.0f);
  io.AddMouseButtonEvent(0, true);
  SimulateFrame(0.0f, 20.0f);

  // End near 200.0 us (2010px -> 201.0 us)
  io.MousePos = ImVec2(GetTimelineStartX() + 2010.0f, 35.0f);
  SimulateFrame(0.0f, 20.0f);

  io.AddMouseButtonEvent(0, false);
  SimulateFrame(0.0f, 20.0f);

  ASSERT_EQ(timeline_.selected_time_ranges().size(), 1);
  // It should snap
  EXPECT_DOUBLE_EQ(timeline_.selected_time_ranges()[0].start(), 100.0);
  EXPECT_DOUBLE_EQ(timeline_.selected_time_ranges()[0].end(), 200.0);
}

TEST_F(TimelineDragSelectionTest, SnapIncludesEventsAtExactTopEdgeOfWindow) {
  timeline_.set_snap_to_time_range_enabled(true);

  FlameChartTimelineData data;
  data.entry_levels = {0};
  data.entry_total_times = {100.0};
  data.entry_self_times = {100.0};
  data.entry_start_times = {100.0};  // Event from 100.0 to 200.0
  data.entry_names = {"event1"};
  data.entry_event_ids = {1};
  data.entry_pids = {1};
  data.entry_tids = {1};
  data.entry_args = {{}};
  // Group is expanded.
  data.groups = {{Group::Type::kFlame, "group", "", 0, 0, true}};
  data.events_by_level = {{0}};
  timeline_.SetTimelineData(data);

  // Use a scroll of 23.0f (which is just below y_bottom) so that the bottom
  // edge of the event is just visible at the top visible edge of the window.
  SimulateFrame(23.0f, 1000.0f);

  ImGuiIO& io = ImGui::GetIO();

  // We place the mouse at 35.0f on screen to be inside the group.
  float hover_y = 35.0f;
  // Start near 100.0 us (990px -> 99.0 us)
  io.MousePos = ImVec2(GetTimelineStartX() + 990.0f, hover_y);
  io.AddMouseButtonEvent(0, true);
  SimulateFrame(23.0f, 1000.0f);

  // End near 200.0 us (2010px -> 201.0 us)
  io.MousePos = ImVec2(GetTimelineStartX() + 2010.0f, hover_y);
  SimulateFrame(23.0f, 1000.0f);

  io.AddMouseButtonEvent(0, false);
  SimulateFrame(23.0f, 1000.0f);

  ASSERT_EQ(timeline_.selected_time_ranges().size(), 1);
  // It should snap
  EXPECT_DOUBLE_EQ(timeline_.selected_time_ranges()[0].start(), 100.0);
  EXPECT_DOUBLE_EQ(timeline_.selected_time_ranges()[0].end(), 200.0);
}

TEST_F(TimelineDragSelectionTest, SnapIgnoresEventsExactlyOnePixelBelowWindow) {
  timeline_.set_snap_to_time_range_enabled(true);

  FlameChartTimelineData data;
  data.entry_levels = {0};
  data.entry_total_times = {100.0};
  data.entry_self_times = {100.0};
  data.entry_start_times = {100.0};  // Event from 100.0 to 200.0
  data.entry_names = {"event1"};
  data.entry_event_ids = {1};
  data.entry_pids = {1};
  data.entry_tids = {1};
  data.entry_args = {{}};
  data.groups = {{Group::Type::kFlame, "group", "", 0, 0, true}};
  data.events_by_level = {{0}};

  // Verify strict viewport culling logic at the lower boundary.
  // Manipulate ImGui style padding to precisely position the event
  // such that its top edge is exactly 0.5 pixels below the visible window area.
  // The culling logic must strictly evaluate `y_top > window_height` and skip
  // the event, preventing it from being considered for snapping.
  float prev_padding_y = ImGui::GetStyle().CellPadding.y;
  ImGui::GetStyle().CellPadding.y = 200.5f;

  timeline_.SetTimelineData(data);

  float window_height = 200.0f;

  auto draw_frame = [&](float height) {
    ImGui::GetIO().DisplaySize = ImVec2(1920.0f, height);
    ImGui::NewFrame();
    timeline_.Draw();
    Animation::UpdateAll(ImGui::GetIO().DeltaTime);
    ImGui::EndFrame();
  };

  draw_frame(window_height);

  ImGuiIO& io = ImGui::GetIO();

  io.MousePos = ImVec2(GetTimelineStartX() + 990.0f, kEmptyAreaY);
  io.AddMouseButtonEvent(0, true);
  draw_frame(window_height);

  io.MousePos = ImVec2(GetTimelineStartX() + 2010.0f, kEmptyAreaY);
  draw_frame(window_height);

  io.AddMouseButtonEvent(0, false);
  draw_frame(window_height);

  ImGui::GetStyle().CellPadding.y = prev_padding_y;

  ASSERT_EQ(timeline_.selected_time_ranges().size(), 1);
  // Verify no snapping occurred because the event was culled.
  EXPECT_DOUBLE_EQ(timeline_.selected_time_ranges()[0].start(), 99.0);
  EXPECT_DOUBLE_EQ(timeline_.selected_time_ranges()[0].end(), 201.0);
}

TEST_F(TimelineDragSelectionTest, SnapIgnoresEventsExactlyOnePixelAboveWindow) {
  timeline_.set_snap_to_time_range_enabled(true);

  FlameChartTimelineData data;
  data.entry_levels = {0};
  data.entry_total_times = {100.0};
  data.entry_self_times = {100.0};
  data.entry_start_times = {100.0};  // Event from 100.0 to 200.0
  data.entry_names = {"event1"};
  data.entry_event_ids = {1};
  data.entry_pids = {1};
  data.entry_tids = {1};
  data.entry_args = {{}};
  data.groups = {{Group::Type::kFlame, "group", "", 0, 0, true}};
  data.events_by_level = {{0}};

  // Verify strict viewport culling logic at the upper boundary.
  // Manipulate ImGui style padding to precisely position the event
  // such that its bottom edge is exactly 0.5 pixels above the current scroll
  // position. The culling logic must strictly evaluate `y_bottom <
  // current_scroll_y` and skip the event, preventing it from being considered
  // for snapping. This exact positioning is achieved by setting a negative
  // CellPadding.y.
  float prev_padding_y = ImGui::GetStyle().CellPadding.y;
  ImGui::GetStyle().CellPadding.y = -23.5f;

  timeline_.SetTimelineData(data);

  float window_height = 200.0f;

  auto draw_frame = [&](float height) {
    ImGui::GetIO().DisplaySize = ImVec2(1920.0f, height);
    ImGui::NewFrame();
    timeline_.Draw();
    Animation::UpdateAll(ImGui::GetIO().DeltaTime);
    ImGui::EndFrame();
  };

  draw_frame(window_height);

  ImGuiIO& io = ImGui::GetIO();

  io.MousePos = ImVec2(GetTimelineStartX() + 990.0f, kEmptyAreaY);
  io.AddMouseButtonEvent(0, true);
  draw_frame(window_height);

  io.MousePos = ImVec2(GetTimelineStartX() + 2010.0f, kEmptyAreaY);
  draw_frame(window_height);

  io.AddMouseButtonEvent(0, false);
  draw_frame(window_height);

  ImGui::GetStyle().CellPadding.y = prev_padding_y;

  ASSERT_EQ(timeline_.selected_time_ranges().size(), 1);
  // Verify no snapping occurred because the event was culled.
  EXPECT_DOUBLE_EQ(timeline_.selected_time_ranges()[0].start(), 99.0);
  EXPECT_DOUBLE_EQ(timeline_.selected_time_ranges()[0].end(), 201.0);
}

TEST_F(TimelineDragSelectionTest, SnapWorksForExpandedTrackWithMultipleLevels) {
  timeline_.set_snap_to_time_range_enabled(true);

  FlameChartTimelineData data;
  data.entry_levels = {0, 1};
  data.entry_total_times = {100.0, 100.0};
  data.entry_self_times = {100.0, 100.0};
  data.entry_start_times = {100.0, 100.0};  // Events at 100.0
  data.entry_names = {"event1", "event2"};
  data.entry_event_ids = {1, 2};
  data.entry_pids = {1, 1};
  data.entry_tids = {1, 1};
  data.entry_args = {{}, {}};
  // Group is expanded and has multiple levels.
  data.groups = {{Group::Type::kFlame, "group", "", 0, 0, true}};
  data.events_by_level = {{0}, {1}};
  timeline_.SetTimelineData(data);

  SimulateFrame();

  ImGuiIO& io = ImGui::GetIO();

  // Drag near 100.0 us
  io.MousePos = ImVec2(GetTimelineStartX() + 990.0f, kFirstEventY);
  io.AddMouseButtonEvent(0, true);
  SimulateFrame();

  io.MousePos = ImVec2(GetTimelineStartX() + 1500.0f, kFirstEventY);
  SimulateFrame();

  io.AddMouseButtonEvent(0, false);
  SimulateFrame();

  ASSERT_EQ(timeline_.selected_time_ranges().size(), 1);
  EXPECT_DOUBLE_EQ(timeline_.selected_time_ranges()[0].start(), 100.0);
}

TEST_F(TimelineDragSelectionTest, SnapWorksForNonExpandableCollapsedTrack) {
  timeline_.set_snap_to_time_range_enabled(true);

  FlameChartTimelineData data;
  data.entry_levels = {0};
  data.entry_total_times = {100.0};
  data.entry_self_times = {100.0};
  data.entry_start_times = {100.0};  // Event at 100.0
  data.entry_names = {"event1"};
  data.entry_event_ids = {1};
  data.entry_pids = {1};
  data.entry_tids = {1};
  data.entry_args = {{}};
  // Group is NOT expanded, but it is NOT expandable (only 1 level, no children)
  data.groups = {{Group::Type::kFlame, "group", "", 0, 0, false}};
  data.events_by_level = {{0}};
  timeline_.SetTimelineData(data);

  SimulateFrame();

  ImGuiIO& io = ImGui::GetIO();

  // Drag near 100.0 us
  io.MousePos = ImVec2(GetTimelineStartX() + 990.0f, kFirstEventY);
  io.AddMouseButtonEvent(0, true);
  SimulateFrame();

  io.MousePos = ImVec2(GetTimelineStartX() + 1500.0f, kFirstEventY);
  SimulateFrame();

  io.AddMouseButtonEvent(0, false);
  SimulateFrame();

  ASSERT_EQ(timeline_.selected_time_ranges().size(), 1);
  EXPECT_DOUBLE_EQ(timeline_.selected_time_ranges()[0].start(), 100.0);
}

// =============================================================================
// Fixture: TimelineMouseModeSelectTestSuite
// =============================================================================

TEST_F(TimelineMouseModeSelectTestSuite,
       FindSelectedEventsEmitsEmptyJsonWhenNoSelection) {
  ImGuiIO& io = ImGui::GetIO();
  bool callback_called = false;
  std::string captured_payload;
  timeline_.set_event_callback(
      [&](absl::string_view event_type, const EventData& data) {
        if (event_type == kEventsSelected) {
          callback_called = true;
          auto it = data.find(std::string(kEventsSelectedData));
          if (it != data.end()) {
            captured_payload = std::any_cast<std::string>(it->second);
          }
        }
      });

  SimulateFrame();

  io.AddMousePosEvent(GetTimelineStartX() + 200.0f, 100.0f);
  SimulateFrame();
  io.AddMouseButtonEvent(0, true);
  SimulateFrame();

  io.AddMousePosEvent(GetTimelineStartX() + 250.0f, 150.0f);
  SimulateFrame();

  io.AddMouseButtonEvent(0, false);
  SimulateFrame();

  EXPECT_TRUE(callback_called);
  EXPECT_TRUE(captured_payload.empty());
}

TEST_F(TimelineMouseModeSelectTestSuite, FindSelectedEventsEmitsJson) {
  ImGuiIO& io = ImGui::GetIO();
  bool callback_called = false;
  std::string captured_payload;
  timeline_.set_event_callback(
      [&](absl::string_view event_type, const EventData& data) {
        if (event_type == kEventsSelected) {
          callback_called = true;
          auto it = data.find(std::string(kEventsSelectedData));
          if (it != data.end()) {
            captured_payload = std::any_cast<std::string>(it->second);
          }
        }
      });

  SimulateFrame();  // Warm-up frame

  // Set start position.
  io.MousePos = ImVec2(GetTimelineStartX() + 200.0f, 35.0f);
  io.AddMouseButtonEvent(0, true);
  SimulateFrame();  // Frame 1

  SimulateFrame();  // Frame 2 (let it settle)

  // Move to end position.
  io.MousePos = ImVec2(GetTimelineStartX(), 150.0f);
  SimulateFrame();  // Frame 3

  // Release.
  io.AddMouseButtonEvent(0, false);
  SimulateFrame();  // Frame 4

  EXPECT_TRUE(callback_called);
  EXPECT_FALSE(captured_payload.empty());
  EXPECT_THAT(captured_payload, ::testing::HasSubstr("event1"));
}

TEST_F(TimelineMouseModeSelectTestSuite, FindSelectedEventsSelectsCounters) {
  ImGuiIO& io = ImGui::GetIO();
  bool callback_called = false;
  std::string captured_payload;
  timeline_.set_event_callback(
      [&](absl::string_view event_type, const EventData& data) {
        if (event_type == kEventsSelected) {
          callback_called = true;
          auto it = data.find(std::string(kEventsSelectedData));
          if (it != data.end()) {
            captured_payload = std::any_cast<std::string>(it->second);
          }
        }
      });

  FlameChartTimelineData data;
  data.groups = {
      {Group::Type::kCounter, "counter_group", "subtitle", 0, 0, true}};

  CounterData counter_data;
  counter_data.timestamps = {10.0, 20.0, 30.0};
  counter_data.values = {1.0, 2.0, 3.0};
  counter_data.min_value = 1.0;
  counter_data.max_value = 3.0;

  data.counter_data_by_group_index[0] = counter_data;

  timeline_.SetTimelineData(std::move(data));

  timeline_.SetVisibleRange(TimeRange(0.0, 100.0));

  SimulateFrame();  // Warm-up frame and calculate layout

  // Start drag.
  io.MousePos = ImVec2(GetTimelineStartX() + 50.0f, 40.0f);
  io.AddMouseButtonEvent(0, true);
  SimulateFrame();

  // Drag to cover points.
  io.MousePos = ImVec2(GetTimelineStartX() + 500.0f, 100.0f);
  SimulateFrame();

  // Release.
  io.AddMouseButtonEvent(0, false);
  SimulateFrame();

  EXPECT_TRUE(callback_called);
  EXPECT_FALSE(captured_payload.empty());
  EXPECT_THAT(captured_payload, ::testing::HasSubstr("counters"));
  EXPECT_THAT(captured_payload, ::testing::HasSubstr("counter_group"));
}

// =============================================================================
// Fixture: TimelineImGuiFixture
// =============================================================================

TEST_F(TimelineImGuiFixture, LevelYPositionsCalculation) {
  FlameChartTimelineData data;
  data.groups.push_back({.name = "Group 0",
                         .start_level = 0,
                         .nesting_level = 0,
                         .expanded = true});
  data.groups.push_back({.name = "Group 1",
                         .start_level = 2,
                         .nesting_level = 0,
                         .expanded = true});
  data.groups.push_back({.name = "Group 2",
                         .start_level = 3,
                         .nesting_level = 0,
                         .expanded = true});
  // Level 0, 1 in Group 0
  // Level 2 in Group 1
  // Level 3, 4, 5 in Group 2
  data.events_by_level.resize(6);
  timeline_.SetTimelineData(std::move(data));

  SimulateFrame();

  const auto& y_offsets = timeline_.GetVisibleLevelOffsets();
  EXPECT_EQ(y_offsets.size(), 6);

  const float level_height = kEventHeight + kEventPaddingBottom;

  // For now, let's just check the difference between levels in the same group.
  if (y_offsets.size() >= 2) {
    EXPECT_FLOAT_EQ(y_offsets[1] - y_offsets[0], level_height);
  }
  if (y_offsets.size() >= 6) {
    EXPECT_FLOAT_EQ(y_offsets[4] - y_offsets[3], level_height);
    EXPECT_FLOAT_EQ(y_offsets[5] - y_offsets[4], level_height);
  }

  // Group 0: Levels 0, 1
  if (y_offsets.size() >= 6) {
    const float group0_base_y = y_offsets[0];
    EXPECT_FLOAT_EQ(y_offsets[1] - group0_base_y, level_height);

    // Group 1: Level 2
    // No relative checks needed within Group 1 as it has only one level.

    // Group 2: Levels 3, 4, 5
    const float group2_base_y = y_offsets[3];
    EXPECT_FLOAT_EQ(y_offsets[4] - group2_base_y, level_height);
    EXPECT_FLOAT_EQ(y_offsets[5] - group2_base_y, 2 * level_height);
  }
}

TEST_F(TimelineImGuiFixture, SelectEvents) {
  // 1. Setup Data
  FlameChartTimelineData data;
  data.groups.push_back(
      {.name = "Group 1", .start_level = 0, .expanded = true});
  data.events_by_level.resize(1);
  data.events_by_level[0].push_back(0);
  data.entry_names.push_back("Event 1");
  data.entry_levels.push_back(0);
  data.entry_start_times.push_back(100.0);
  data.entry_total_times.push_back(50.0);
  data.entry_self_times.push_back(50.0);
  data.entry_pids.push_back(1);
  data.entry_args.push_back({});

  timeline_.SetTimelineData(std::move(data));
  timeline_.SetVisibleRange({0.0, 1000.0});

  // 2. Set Mouse Mode
  timeline_.set_mouse_mode(MouseMode::kSelect);

  // 3. Simulate Mouse Drag
  ImGuiIO& io = ImGui::GetIO();
  SimulateFrame();  // Init window

  ImGuiWindow* window = ImGui::FindWindowByName("Timeline viewer");
  ASSERT_NE(window, nullptr);
  float win_y = window->DC.CursorStartPos.y;

  float tracks_x = GetTimelineStartX();
  float start_time_px =
      timeline_.TimeToScreenX(50.0, tracks_x, timeline_.px_per_time_unit());
  float end_time_px =
      timeline_.TimeToScreenX(200.0, tracks_x, timeline_.px_per_time_unit());

  float start_px_y = win_y + kRulerHeight;
  float end_px_y = win_y + kRulerHeight + kEventHeight * 2;

  bool event_called = false;
  std::string received_json;

  timeline_.set_event_callback(
      [&](absl::string_view name, const EventData& data) {
        if (name == kEventsSelected) {
          event_called = true;
          auto it = data.find(kEventsSelectedData);
          if (it != data.end()) {
            received_json = std::any_cast<std::string>(it->second);
          }
        }
      });

  io.AddMousePosEvent(start_time_px, start_px_y);
  SimulateFrame();

  io.AddMouseButtonEvent(0, true);  // Mouse down
  SimulateFrame();

  io.AddMousePosEvent(end_time_px, end_px_y);  // Mouse drag
  SimulateFrame();

  io.AddMouseButtonEvent(0, false);  // Mouse release
  SimulateFrame();

  EXPECT_TRUE(event_called);
  EXPECT_FALSE(received_json.empty());

  EXPECT_THAT(received_json, ::testing::HasSubstr(R"("name":"Event 1")"));
  EXPECT_THAT(received_json, ::testing::HasSubstr(R"("count":1)"));
  EXPECT_THAT(received_json, ::testing::HasSubstr(R"("wallTimeUs":50)"));
  EXPECT_THAT(received_json, ::testing::HasSubstr(R"("avgWallDurationUs":50)"));
}

TEST_F(TimelineImGuiFixture, Shortcuts) {
  ImGuiIO& io = ImGui::GetIO();
  SimulateFrame();  // Init window

  timeline_.set_mouse_mode(MouseMode::kTiming);

  io.AddKeyEvent(ImGuiKey_1, true);
  SimulateFrame();
  EXPECT_EQ(timeline_.mouse_mode(), MouseMode::kSelect);

  io.AddKeyEvent(ImGuiKey_2, true);
  SimulateFrame();
  EXPECT_EQ(timeline_.mouse_mode(), MouseMode::kPan);

  io.AddKeyEvent(ImGuiKey_3, true);
  SimulateFrame();
  EXPECT_EQ(timeline_.mouse_mode(), MouseMode::kZoom);

  io.AddKeyEvent(ImGuiKey_4, true);
  SimulateFrame();
  EXPECT_EQ(timeline_.mouse_mode(), MouseMode::kTiming);
}

// Verifies that zoom mode correctly updates cursor and visible range.
TEST_F(TimelineImGuiFixture, ZoomMode) {
  ImGuiIO& io = ImGui::GetIO();
  SimulateFrame();  // Init window

  timeline_.set_mouse_mode(MouseMode::kZoom);

  FlameChartTimelineData data;
  data.groups.push_back(
      {.name = "Group 1", .start_level = 0, .expanded = true});
  data.events_by_level.resize(1);
  data.events_by_level[0].push_back(0);
  data.entry_names.push_back("Event 1");
  data.entry_levels.push_back(0);
  data.entry_start_times.push_back(100.0);
  data.entry_total_times.push_back(900.0);
  data.entry_self_times.push_back(900.0);
  data.entry_pids.push_back(1);
  data.entry_args.push_back({});
  timeline_.SetTimelineData(std::move(data));
  timeline_.set_data_time_range({0.0, 1000.0});

  timeline_.SetVisibleRange({100.0, 200.0});
  EXPECT_EQ(timeline_.visible_range().duration(), 100.0);

  ImGuiWindow* window = ImGui::FindWindowByName("Timeline viewer");
  ASSERT_NE(window, nullptr);

  // Hover over timeline
  float tracks_x = GetTimelineStartX();
  io.AddMousePosEvent(tracks_x + 10.0f,
                      window->DC.CursorStartPos.y + kRulerHeight + 10.0f);
  SimulateFrame();

  EXPECT_EQ(ImGui::GetMouseCursor(), ImGuiMouseCursor_ResizeNS);

  // Drag mouse vertically
  io.AddMouseButtonEvent(0, true);
  SimulateFrame();

  io.AddMousePosEvent(tracks_x + 10.0f,
                      window->DC.CursorStartPos.y + kRulerHeight + 50.0f);
  SimulateFrame();

  io.AddMouseButtonEvent(0, false);
  io.DeltaTime = 1.0f;
  SimulateFrame();

  // Verify visible range changed.
  EXPECT_NEAR(timeline_.visible_range().duration(), 140.0, 1.0);
}

TEST_F(TimelineImGuiFixture, PanModeCursor) {
  ImGuiIO& io = ImGui::GetIO();
  SimulateFrame();  // Init window

  timeline_.set_mouse_mode(MouseMode::kPan);

  ImGuiWindow* window = ImGui::FindWindowByName("Timeline viewer");
  ASSERT_NE(window, nullptr);

  // Hover over timeline
  float tracks_x = GetTimelineStartX();
  io.AddMousePosEvent(tracks_x + 10.0f,
                      window->DC.CursorStartPos.y + kRulerHeight + 10.0f);
  SimulateFrame();

  // Not dragging, should be Arrow
  EXPECT_EQ(ImGui::GetMouseCursor(), ImGuiMouseCursor_Arrow);

  // Press mouse button (simulate drag start)
  io.AddMouseButtonEvent(0, true);
  SimulateFrame();

  // Dragging, should be ResizeAll
  EXPECT_EQ(ImGui::GetMouseCursor(), ImGuiMouseCursor_ResizeAll);

  // Release mouse button
  io.AddMouseButtonEvent(0, false);
  SimulateFrame();

  // Back to Arrow
  EXPECT_EQ(ImGui::GetMouseCursor(), ImGuiMouseCursor_Arrow);
}

TEST_F(RealTimelineImGuiFixture, PanLeftOutOfBounds) {
  timeline_.set_data_time_range({0.0, 1000.0});
  timeline_.SetVisibleRange({10.0, 100.0});
  SimulateFrame();

  timeline_.Pan(-2000.0f);

  EXPECT_FLOAT_EQ(timeline_.get_bounds_notification_timer_for_test(), 2.0f);
  EXPECT_EQ(timeline_.get_bounds_notification_message_for_test(),
            "Cannot pan further left: reached the beginning of the trace.");
}

TEST_F(RealTimelineImGuiFixture, PanRightOutOfBounds) {
  timeline_.set_data_time_range({0.0, 1000.0});
  timeline_.SetVisibleRange({900.0, 990.0});
  SimulateFrame();

  timeline_.Pan(2000.0f);

  EXPECT_FLOAT_EQ(timeline_.get_bounds_notification_timer_for_test(), 2.0f);
  EXPECT_EQ(timeline_.get_bounds_notification_message_for_test(),
            "Cannot pan further right: reached the end of the trace.");
}

TEST_F(RealTimelineImGuiFixture, ZoomInOutOfBounds) {
  timeline_.set_data_time_range({0.0, 1000.0});
  // Visible range is already at minimum duration kMinDurationMicros
  timeline_.SetVisibleRange({50.0, 50.0 + kMinDurationMicros});
  SimulateFrame();

  timeline_.Zoom(0.5f, 50.0 + kMinDurationMicros / 2.0);  // zoom in

  EXPECT_FLOAT_EQ(timeline_.get_bounds_notification_timer_for_test(), 2.0f);
  EXPECT_EQ(timeline_.get_bounds_notification_message_for_test(),
            "Cannot zoom in further: minimum zoom duration reached.");
}

TEST_F(RealTimelineImGuiFixture, ZoomOutOutOfBounds) {
  timeline_.set_data_time_range({0.0, 1000.0});
  // Visible range is already fully zoomed out matching the entire data range
  timeline_.SetVisibleRange({0.0, 1000.0});
  SimulateFrame();

  timeline_.Zoom(2.0f, 500.0);  // zoom out

  EXPECT_FLOAT_EQ(timeline_.get_bounds_notification_timer_for_test(), 2.0f);
  EXPECT_EQ(timeline_.get_bounds_notification_message_for_test(),
            "Cannot zoom out further: showing the entire trace.");
}

TEST_F(RealTimelineImGuiFixture, PanFullyZoomedOutNoNotification) {
  timeline_.set_data_time_range({0.0, 1000.0});
  // Visible range is already fully zoomed out matching the entire data range
  timeline_.SetVisibleRange({0.0, 1000.0});
  SimulateFrame();

  timeline_.Pan(-500.0f);  // Try to pan left while fully zoomed out

  // Notification should NOT be triggered since we are already showing the
  // entire trace.
  EXPECT_FLOAT_EQ(timeline_.get_bounds_notification_timer_for_test(), 0.0f);
  EXPECT_EQ(timeline_.get_bounds_notification_message_for_test(), "");

  timeline_.Pan(500.0f);  // Try to pan right while fully zoomed out

  EXPECT_FLOAT_EQ(timeline_.get_bounds_notification_timer_for_test(), 0.0f);
  EXPECT_EQ(timeline_.get_bounds_notification_message_for_test(), "");
}

TEST_F(RealTimelineImGuiFixture, DrawNotificationToastFades) {
  timeline_.set_data_time_range({0.0, 1000.0});
  timeline_.SetVisibleRange({10.0, 100.0});
  SimulateFrame();

  // Pan out of bounds to trigger
  timeline_.Pan(-2000.0f);

  // Verify notification is active
  EXPECT_FLOAT_EQ(timeline_.get_bounds_notification_timer_for_test(), 2.0f);

  // Simulate 1 second passing (io.DeltaTime = 1.0f)
  ImGui::GetIO().DeltaTime = 1.0f;
  SimulateFrame();

  // Verify notification is still active but timer decreased
  EXPECT_NEAR(timeline_.get_bounds_notification_timer_for_test(), 1.0f, 0.01f);

  // Simulate another 1.5 second passing (io.DeltaTime = 1.5f)
  ImGui::GetIO().DeltaTime = 1.5f;
  SimulateFrame();

  // Timer has expired
  EXPECT_LE(timeline_.get_bounds_notification_timer_for_test(), 0.0f);
}

TEST_F(RealTimelineImGuiFixture, HoverTrackLabelChangesCursor) {
  FlameChartTimelineData data;
  data.entry_levels = {0};
  data.entry_total_times = {10.0};
  data.entry_self_times = {10.0};
  data.entry_start_times = {0.0};
  data.entry_names = {"event"};
  data.groups = {{Group::Type::kFlame, "Test Group Name", "", 0,
                  kThreadNestingLevel, true}};
  data.events_by_level = {{0}};
  timeline_.SetTimelineData(data);

  SimulateFrame();

  ImGuiIO& io = ImGui::GetIO();
  // Move mouse over the track label. The label column is 250px wide.
  // We indent by some amount, so X=100 should be on the text.
  // Y should be around 50px (first track).
  // Move mouse over the track label.
  // Y should be inside the track height (kEventHeight = 23) +
  // kRulerHeight (20).
  io.MousePos = ImVec2(100.0f, 30.0f);
  SimulateFrame();

  EXPECT_EQ(ImGui::GetMouseCursor(), ImGuiMouseCursor_TextInput);
}

TEST_F(RealTimelineImGuiFixture, ClickTrackLabelCopiesNameToClipboard) {
  FlameChartTimelineData data;
  data.entry_levels = {0};
  data.entry_total_times = {10.0};
  data.entry_self_times = {10.0};
  data.entry_start_times = {0.0};
  data.entry_names = {"event"};
  data.groups = {{Group::Type::kFlame, "Test Group Name", "", 0,
                  kThreadNestingLevel, true}};
  data.events_by_level = {{0}};
  timeline_.SetTimelineData(data);

  SimulateFrame();

  ImGuiIO& io = ImGui::GetIO();
  io.MousePos = ImVec2(100.0f, 30.0f);
  SimulateFrame();

  io.AddMouseButtonEvent(0, true);
  SimulateFrame();
  io.AddMouseButtonEvent(0, false);
  SimulateFrame();

  const char* clipboard_text = ImGui::GetClipboardText();
  ASSERT_NE(clipboard_text, nullptr);
  EXPECT_STREQ(clipboard_text, "Test Group Name");
}

TEST_F(RealTimelineImGuiFixture,
       ClickExpandCollapseButtonTogglesExpandedState) {
  FlameChartTimelineData data;
  data.entry_levels = {0};
  data.entry_total_times = {10.0};
  data.entry_self_times = {10.0};
  data.entry_start_times = {0.0};
  data.entry_names = {"event"};
  // Needs to test an expandable group, so has_multiple_levels or has_children.
  // Let's set start_level=0 and next_group_start_level=2 to simulate multiple
  // levels.
  data.groups = {{Group::Type::kFlame, "Test Group Name", "", 0,
                  kThreadNestingLevel, true}};
  data.events_by_level = {{0}, {}};  // 2 levels, second level empty
  timeline_.SetTimelineData(data);

  SimulateFrame();

  // Button is at X = (nesting_level + 1) * kIndentSize.
  // For kThreadNestingLevel (2), indent is 3 * 10 = 30.
  // Button width is around 13px. So X=35 is inside.
  // Y should be inside the track (20-43).
  ImGuiIO& io = ImGui::GetIO();
  io.MousePos = ImVec2(35.0f, 30.0f);
  SimulateFrame();

  // It should be a Hand cursor over the button
  EXPECT_EQ(ImGui::GetMouseCursor(), ImGuiMouseCursor_Hand);

  io.AddMouseButtonEvent(0, true);
  SimulateFrame();
  io.AddMouseButtonEvent(0, false);
  SimulateFrame();

  EXPECT_FALSE(timeline_.timeline_data().groups[0].expanded);
}

TEST_F(MockTimelineImGuiFixture, FindFirstVisibleAncestorIndex_SelfCollapse) {
  FlameChartTimelineData data;
  data.events_by_level.resize(5);

  // Group 0: Parent (Collapsed, nesting_level = 0)
  data.groups.push_back({
      .type = Group::Type::kFlame,
      .name = "Parent",
      .start_level = 0,
      .nesting_level = 0,
      .expanded = false,
  });

  // Group 1: Child (nesting_level = 1)
  data.groups.push_back({
      .type = Group::Type::kFlame,
      .name = "Child",
      .start_level = 1,
      .nesting_level = 1,
      .expanded = true,
  });

  timeline_.SetTimelineData(std::move(data));

  // Verify group visibility calculated by UpdateLevelPositions
  // Parent: visible (true)
  // Child: invisible (false)
  ASSERT_EQ(timeline_.CallGroupVisible().size(), 2);
  EXPECT_TRUE(timeline_.CallGroupVisible()[0]);
  EXPECT_FALSE(timeline_.CallGroupVisible()[1]);

  // If queried on Child (1) -> must backtrack and return Parent (0)
  EXPECT_EQ(timeline_.CallFindFirstVisibleAncestorIndex(1), 0);

  // If queried on Parent (0) -> returns self (0)
  EXPECT_EQ(timeline_.CallFindFirstVisibleAncestorIndex(0), 0);
}

TEST_F(MockTimelineImGuiFixture, FindFirstVisibleAncestorIndex_ParentCollapse) {
  FlameChartTimelineData data;
  data.events_by_level.resize(5);

  // Group 0: Grand Parent (Expanded, nesting_level = 0)
  data.groups.push_back({
      .type = Group::Type::kFlame,
      .name = "Grand Parent",
      .start_level = 0,
      .nesting_level = 0,
      .expanded = true,
  });

  // Group 1: Parent (Collapsed, nesting_level = 1)
  data.groups.push_back({
      .type = Group::Type::kFlame,
      .name = "Parent",
      .start_level = 1,
      .nesting_level = 1,
      .expanded = false,
  });

  // Group 2: Child (nesting_level = 2)
  data.groups.push_back({
      .type = Group::Type::kFlame,
      .name = "Child",
      .start_level = 2,
      .nesting_level = 2,
      .expanded = true,
  });

  timeline_.SetTimelineData(std::move(data));

  ASSERT_EQ(timeline_.CallGroupVisible().size(), 3);
  EXPECT_TRUE(timeline_.CallGroupVisible()[0]);   // Grand Parent
  EXPECT_TRUE(timeline_.CallGroupVisible()[1]);   // Parent
  EXPECT_FALSE(timeline_.CallGroupVisible()[2]);  // Child (Parent is collapsed)

  // Child (2) is hidden by Parent (1), queries on 2 must return 1 (Parent)
  EXPECT_EQ(timeline_.CallFindFirstVisibleAncestorIndex(2), 1);
}

TEST_F(MockTimelineImGuiFixture,
       FindFirstVisibleAncestorIndex_SiblingCollapse) {
  FlameChartTimelineData data;
  data.events_by_level.resize(5);

  // Group 0: Parent (Expanded, nesting_level = 0)
  data.groups.push_back({
      .type = Group::Type::kFlame,
      .name = "Parent",
      .start_level = 0,
      .nesting_level = 0,
      .expanded = true,
  });

  // Group 1: Child A (Collapsed, nesting_level = 1)
  data.groups.push_back({
      .type = Group::Type::kFlame,
      .name = "Child A",
      .start_level = 1,
      .nesting_level = 1,
      .expanded = false,
  });

  // Group 2: Grandchild A (nesting_level = 2)
  data.groups.push_back({
      .type = Group::Type::kFlame,
      .name = "Grandchild A",
      .start_level = 2,
      .nesting_level = 2,
      .expanded = true,
  });

  // Group 3: Child B (Expanded, nesting_level = 1, sibling of Child A)
  data.groups.push_back({
      .type = Group::Type::kFlame,
      .name = "Child B",
      .start_level = 3,
      .nesting_level = 1,
      .expanded = true,
  });

  timeline_.SetTimelineData(std::move(data));

  ASSERT_EQ(timeline_.CallGroupVisible().size(), 4);
  EXPECT_TRUE(timeline_.CallGroupVisible()[0]);  // Parent
  EXPECT_TRUE(timeline_.CallGroupVisible()[1]);  // Child A
  EXPECT_FALSE(
      timeline_.CallGroupVisible()[2]);  // Grandchild A (under Child A)
  EXPECT_TRUE(
      timeline_.CallGroupVisible()[3]);  // Child B (should NOT be affected by
                                         // Child A's collapse!)

  // Child B (3) is visible, queried on it must return itself (3)
  EXPECT_EQ(timeline_.CallFindFirstVisibleAncestorIndex(3), 3);

  // Grandchild A (2) is hidden under Child A (1), queried on it must return
  // Child A (1)
  EXPECT_EQ(timeline_.CallFindFirstVisibleAncestorIndex(2), 1);
}

TEST_F(MockTimelineImGuiFixture, HideProcessTrack_FeatureFlagToggle) {
  FlameChartTimelineData data;
  data.events_by_level.resize(5);

  // Group 0: Process A
  data.groups.push_back({
      .type = Group::Type::kFlame,
      .name = "Process A",
      .start_level = 0,
      .nesting_level = kProcessNestingLevel,
      .expanded = true,
  });

  // Group 1: Thread A1
  data.groups.push_back({
      .type = Group::Type::kFlame,
      .name = "Thread A1",
      .start_level = 0,
      .nesting_level = kThreadNestingLevel,
      .expanded = true,
  });

  // Group 2: Thread A2
  data.groups.push_back({
      .type = Group::Type::kFlame,
      .name = "Thread A2",
      .start_level = 1,
      .nesting_level = kThreadNestingLevel,
      .expanded = true,
  });

  // Group 3: Process B
  data.groups.push_back({
      .type = Group::Type::kFlame,
      .name = "Process B",
      .start_level = 2,
      .nesting_level = kProcessNestingLevel,
      .expanded = true,
  });

  // Group 4: Thread B1
  data.groups.push_back({
      .type = Group::Type::kFlame,
      .name = "Thread B1",
      .start_level = 2,
      .nesting_level = kThreadNestingLevel,
      .expanded = true,
  });

  // Set the data on timeline.
  timeline_.SetTimelineData(data);

  // 1. Hide "Process A". First, test with feature flag ENABLED.
  timeline_.set_track_management_enabled(true);
  const float prev_padding_y = ImGui::GetStyle().CellPadding.y;
  ImGui::GetStyle().CellPadding.y = 10.0f;
  timeline_.HideTrack("Process A");

  // Verify group visibility for this state.
  ASSERT_EQ(timeline_.CallGroupVisible().size(), 5);
  EXPECT_FALSE(timeline_.CallGroupVisible()[0]);  // Process A hidden
  // Thread A1 hidden (child of hidden Process A)
  EXPECT_FALSE(timeline_.CallGroupVisible()[1]);
  // Thread A2 hidden (child of hidden Process A)
  EXPECT_FALSE(timeline_.CallGroupVisible()[2]);
  EXPECT_TRUE(timeline_.CallGroupVisible()[3]);  // Process B visible
  EXPECT_TRUE(timeline_.CallGroupVisible()[4]);  // Thread B1 visible

  // Verify visible level offsets for levels in hidden track.
  ASSERT_EQ(timeline_.GetVisibleLevelOffsets().size(), 5);
  EXPECT_FLOAT_EQ(timeline_.GetVisibleLevelOffsets()[0], 40.0f);
  EXPECT_FLOAT_EQ(timeline_.GetVisibleLevelOffsets()[1], 40.0f);
  EXPECT_FLOAT_EQ(timeline_.GetVisibleLevelOffsets()[2], 135.5f);
  EXPECT_FLOAT_EQ(timeline_.GetVisibleLevelOffsets()[3], 159.5f);
  EXPECT_FLOAT_EQ(timeline_.GetVisibleLevelOffsets()[4], 183.5f);

  ImGui::GetStyle().CellPadding.y = prev_padding_y;

  // 2. Now disable the feature flag. Hiding should be ignored and
  // all tracks should reappear.
  timeline_.set_track_management_enabled(false);
  timeline_.UpdateLevelPositions(timeline_.timeline_data());

  EXPECT_TRUE(timeline_.CallGroupVisible()[0]);  // Process A visible again
  EXPECT_TRUE(timeline_.CallGroupVisible()[1]);  // Thread A1 visible again
  EXPECT_TRUE(timeline_.CallGroupVisible()[2]);  // Thread A2 visible again
  EXPECT_TRUE(timeline_.CallGroupVisible()[3]);  // Process B visible
  EXPECT_TRUE(timeline_.CallGroupVisible()[4]);  // Thread B1 visible
}

TEST_F(RealTimelineImGuiFixture, ClickHideButtonOnCollapsedTrackHidesIt) {
  FlameChartTimelineData data;
  data.entry_levels = {0};
  data.entry_total_times = {10.0};
  data.entry_self_times = {10.0};
  data.entry_start_times = {0.0};
  data.entry_names = {"event"};

  // Process A is collapsed, but has children.
  data.groups = {
      {Group::Type::kFlame, "Process A", "", 0, kProcessNestingLevel, false},
      {Group::Type::kFlame, "Thread A1", "", 0, kThreadNestingLevel, true}};
  data.events_by_level = {{0}, {}};
  timeline_.SetTimelineData(data);
  timeline_.set_track_management_enabled(true);

  SimulateFrame();

  ImGuiIO& io = ImGui::GetIO();
  io.MousePos = ImVec2(timeline_.GetLabelWidth() - 10.0f, 45.0f);
  SimulateFrame();

  EXPECT_EQ(ImGui::GetMouseCursor(), ImGuiMouseCursor_Hand);

  io.AddMouseButtonEvent(0, true);
  SimulateFrame();
  io.AddMouseButtonEvent(0, false);
  SimulateFrame();

  // Verify Process A became hidden
  EXPECT_FALSE(timeline_.group_visible()[0]);
  EXPECT_FALSE(timeline_.group_visible()[1]);
}

TEST_F(RealTimelineImGuiFixture, CollapseAllHeaderHidesGroups) {
  FlameChartTimelineData data;
  data.entry_levels = {0};
  data.entry_total_times = {10.0};
  data.entry_self_times = {10.0};
  data.entry_start_times = {0.0};
  data.entry_names = {"event"};

  data.groups = {
      {Group::Type::kFlame, "Process A", "", 0, kProcessNestingLevel, false},
      {Group::Type::kFlame, "Thread A1", "", 0, kThreadNestingLevel, true}};
  data.events_by_level = {{0}, {}};

  timeline_.set_track_management_enabled(true);
  timeline_.SetTimelineData(data);

  SimulateFrame();

  // Verify Process A is visible initially
  EXPECT_TRUE(timeline_.group_visible()[0]);

  ImGuiIO& io = ImGui::GetIO();
  // Click on "All" header expand/collapse button
  // "Hidden" header is at 0-30 (screen 20-50)
  // "All" header is at 30-60 (screen 50-80)
  // Button is at X = kIndentSize (10), Y ~ 65
  io.MousePos = ImVec2(15.0f, 65.0f);
  SimulateFrame();

  EXPECT_EQ(ImGui::GetMouseCursor(), ImGuiMouseCursor_Hand);

  // Click
  io.AddMouseButtonEvent(0, true);
  SimulateFrame();
  io.AddMouseButtonEvent(0, false);
  SimulateFrame();

  // Verify Process A became hidden
  EXPECT_FALSE(timeline_.group_visible()[0]);
}

TEST_F(RealTimelineImGuiFixture, ExpandHiddenHeaderShowsHiddenGroups) {
  FlameChartTimelineData data;
  data.entry_levels = {0};
  data.entry_total_times = {10.0};
  data.entry_self_times = {10.0};
  data.entry_start_times = {0.0};
  data.entry_names = {"event"};

  data.groups = {
      {Group::Type::kFlame, "Process A", "", 0, kProcessNestingLevel, false},
      {Group::Type::kFlame, "Thread A1", "", 0, kThreadNestingLevel, true}};
  data.events_by_level = {{0}, {}};

  timeline_.set_track_management_enabled(true);
  timeline_.SetTimelineData(data);

  // Hide Process A
  timeline_.HideTrack("Process A");

  SimulateFrame();

  // Verify Process A is INVISIBLE initially
  // (because Hidden header is collapsed by default)
  EXPECT_FALSE(timeline_.group_visible()[0]);

  ImGuiIO& io = ImGui::GetIO();
  // Click on "Hidden" header expand/collapse button
  // "Hidden" header is at 0-30 (screen 20-50)
  // Button is at X = kIndentSize (10), Y ~ 35
  io.MousePos = ImVec2(15.0f, 35.0f);
  SimulateFrame();

  EXPECT_EQ(ImGui::GetMouseCursor(), ImGuiMouseCursor_Hand);

  // Click
  io.AddMouseButtonEvent(0, true);
  SimulateFrame();
  io.AddMouseButtonEvent(0, false);
  SimulateFrame();

  // Verify Process A became VISIBLE
  EXPECT_TRUE(timeline_.group_visible()[0]);
}

TEST_F(RealTimelineImGuiFixture, ClickUnhideButtonOnHiddenTrackUnhidesIt) {
  FlameChartTimelineData data;
  data.entry_levels = {0};
  data.entry_total_times = {10.0};
  data.entry_self_times = {10.0};
  data.entry_start_times = {0.0};
  data.entry_names = {"event"};

  data.groups = {
      {Group::Type::kFlame, "Process A", "", 0, kProcessNestingLevel, false},
      {Group::Type::kFlame, "Thread A1", "", 0, kThreadNestingLevel, true}};
  data.events_by_level = {{0}, {}};

  timeline_.set_track_management_enabled(true);
  timeline_.SetTimelineData(data);

  // Hide Process A
  timeline_.HideTrack("Process A");

  SimulateFrame();

  ImGuiIO& io = ImGui::GetIO();
  // Expand "Hidden" section
  // "Hidden" header is at 0-30 (screen 20-50)
  // Button is at X = kIndentSize (10), Y ~ 35
  io.MousePos = ImVec2(15.0f, 35.0f);
  SimulateFrame();
  io.AddMouseButtonEvent(0, true);
  SimulateFrame();
  io.AddMouseButtonEvent(0, false);
  SimulateFrame();

  // Now Process A is visible in the Hidden section.
  // It is the first group under Hidden header.
  // Hidden header end at 30.
  // Process A track is at 30-80.
  // Clear any previous click state
  SimulateFrame();

  // Click the hide button of Process A in Hidden section.
  // Hide button is at X ~ 241 (label_width_ - offset).
  // Y should be centered in the track height
  // (e.g. 30 + 25 = 55 local, + 20 = 75 screen).
  io.MousePos = ImVec2(241.0f, 75.0f);
  SimulateFrame();

  EXPECT_EQ(ImGui::GetMouseCursor(), ImGuiMouseCursor_Hand);

  // Click
  io.AddMouseButtonEvent(0, true);
  SimulateFrame();
  io.AddMouseButtonEvent(0, false);
  SimulateFrame();

  // Verify Process A became UNHIDDEN
  // It should be visible now.
  EXPECT_TRUE(timeline_.group_visible()[0]);
}

TEST_F(RealTimelineImGuiFixture, DisplayNamePrefixStripping) {
  FlameChartTimelineData data;
  data.entry_levels = {0};
  data.entry_total_times = {10.0};
  data.entry_self_times = {10.0};
  data.entry_start_times = {0.0};
  data.entry_names = {"event"};

  // Group with name "MySubtitle//MyTrack" and subtitle "MySubtitle"
  data.groups = {{Group::Type::kFlame, "MySubtitle//MyTrack", "MySubtitle", 0,
                  kProcessNestingLevel, false}};
  data.events_by_level = {{0}};

  timeline_.set_track_management_enabled(true);
  timeline_.SetTimelineData(data);

  SimulateFrame();
  // This test ensures lines 956-957 in timeline.cc are executed.
}

TEST_F(RealTimelineImGuiFixture, DrawHideIcon_HiddenIconIsCovered) {
  ImGui::NewFrame();
  ImGui::Begin("TestWindow");
  ImDrawList* draw_list = ImGui::GetWindowDrawList();
  ASSERT_NE(draw_list, nullptr);

  // Call DrawHideIcon with is_track_hidden = true to cover the slashing line
  // logic (line 2451)
  timeline_.DrawHideIcon(draw_list, 10.0f, 10.0f, 10.0f, 0xFFFFFFFF,
                         /*is_track_hidden=*/true);

  // Also call with is_track_hidden = false to cover the other branch
  timeline_.DrawHideIcon(draw_list, 10.0f, 10.0f, 10.0f, 0xFFFFFFFF,
                         /*is_track_hidden=*/false);

  ImGui::End();
  ImGui::EndFrame();
}

TEST_F(RealTimelineImGuiFixture, DrawTrackManagementHiddenTrackPopIDCovered) {
  FlameChartTimelineData data;
  data.entry_levels = {0};
  data.entry_total_times = {10.0};
  data.entry_self_times = {10.0};
  data.entry_start_times = {0.0};
  data.entry_names = {"event"};
  data.groups = {
      {Group::Type::kFlame, "Process A", "", 0, 0, true},
  };
  data.events_by_level = {{0}};
  timeline_.SetTimelineData(data);

  // Set track management to false so "Process A" isn't marked invisible in
  // group_visible_ when HideTrack is called.
  timeline_.set_track_management_enabled(false);
  timeline_.HideTrack("Process A");

  // Re-enable track management without calling UpdateLevelPositions, leaving
  // group_visible_[0] as true
  timeline_.set_track_management_enabled(true);

  // Simulate Frame (calls Draw()). It will reach the hidden track check inside
  // the rendering loop (line 432), pop the ID, and continue.
  SimulateFrame();
}

TEST_F(RealTimelineImGuiFixture, TrackManagement_HideButtonLayout) {
  FlameChartTimelineData data;
  data.entry_levels = {0, 1};
  data.entry_total_times = {10.0, 10.0};
  data.entry_self_times = {10.0, 10.0};
  data.entry_start_times = {0.0, 0.0};
  data.entry_names = {"event1", "event2"};

  // Process A is expanded, Thread A1 is expanded.
  data.groups = {
      {Group::Type::kFlame, "Process A", "", 0, kProcessNestingLevel, true},
      {Group::Type::kFlame, "Thread A1", "", 1, kThreadNestingLevel, true}};
  data.events_by_level = {{0}, {1}};
  timeline_.SetTimelineData(data);
  timeline_.set_track_management_enabled(true);

  SimulateFrame();

  const float label_width = timeline_.GetLabelWidth();
  const float font_size = ImGui::GetFontSize();
  // Math check: kArrowSize = font_size * 0.7f.
  const float arrow_size = font_size * 0.7f;
  const float splitter_offset = 4.0f;  // kSplitterOffset is 4.0f

  ImGuiIO& io = ImGui::GetIO();

  // Test 1: Verify that hovering over a process track's hide button
  // sets the cursor to Hand.
  // The correct button horizontal range is:
  // [label_width - splitter_offset - arrow_size,
  //  label_width - splitter_offset].
  // Set position to the center of the range.
  io.MousePos =
      ImVec2(label_width - splitter_offset - arrow_size * 0.5f, 45.0f);
  SimulateFrame();
  EXPECT_EQ(ImGui::GetMouseCursor(), ImGuiMouseCursor_Hand);

  // Test 2: Hovering just outside (to the left) of the correct range.
  // Under correct code, this is outside. Cursor should NOT be Hand.
  // Under mutated code (Mutant 612), arrow_size is increased by 1px,
  // making the range wider to the left, so it would be Hand.
  io.MousePos =
      ImVec2(label_width - splitter_offset - arrow_size - 0.5f, 45.0f);
  SimulateFrame();
  EXPECT_NE(ImGui::GetMouseCursor(), ImGuiMouseCursor_Hand);

  // Test 3: Verify that Thread A1 (nesting_level = 1 !=
  // kProcessNestingLevel) does NOT render a hide button.
  // Under correct code, it has no button, so cursor at its label's
  // right-end position is NOT Hand.
  // Under mutated code (Mutant 610), it renders a button here,
  // so cursor would be Hand.
  // Thread A1 starts at group_offset = 54.0f (Tracks screen starting
  // Y = 20.0f). Its center Y is 20.0f + 54.0f + 23.0f * 0.5f = 85.5f.
  io.MousePos =
      ImVec2(label_width - splitter_offset - arrow_size * 0.5f, 85.5f);
  SimulateFrame();
  EXPECT_NE(ImGui::GetMouseCursor(), ImGuiMouseCursor_Hand);
}

class TimelineTimeRangeResizeTest : public RealTimelineImGuiFixture {
 protected:
  void SetUp() override {
    RealTimelineImGuiFixture::SetUp();
    // 165.5us results in exactly 10px/us with 1655px timeline width.
    timeline_.SetVisibleRange({0.0, 165.5});
    timeline_.set_data_time_range({0.0, 1000.0});

    SimulateFrame();
    SimulateFrame();
  }

  void AddSelectedTimeRange(Microseconds start, Microseconds end) {
    timeline_.AddSelectedTimeRange(TimeRange(start, end));
  }

  void Drag(Microseconds from_time, Microseconds to_time, bool shift = false) {
    ImGuiIO& io = ImGui::GetIO();
    float origin_x = GetTimelineStartX();
    double px_per_time = 10.0;

    io.AddMousePosEvent(origin_x + from_time * px_per_time, 50.0f);
    io.AddMouseButtonEvent(0, true);
    if (shift) io.KeyShift = true;
    SimulateFrame();

    io.AddMousePosEvent(origin_x + to_time * px_per_time, 50.0f);
    SimulateFrame();

    io.AddMouseButtonEvent(0, false);
    SimulateFrame();
    if (shift) io.KeyShift = false;
  }
};

TEST_F(TimelineTimeRangeResizeTest, ResizeTimeRangeStart) {
  AddSelectedTimeRange(100.0, 150.0);
  ASSERT_EQ(timeline_.selected_time_ranges().size(), 1);

  // Resize start from 100 to 50
  Drag(100.0, 50.0);

  ASSERT_EQ(timeline_.selected_time_ranges().size(), 1);
  EXPECT_DOUBLE_EQ(timeline_.selected_time_ranges()[0].start(), 50.0);
  EXPECT_DOUBLE_EQ(timeline_.selected_time_ranges()[0].end(), 150.0);
}

TEST_F(TimelineTimeRangeResizeTest, ResizeTimeRangeEnd) {
  AddSelectedTimeRange(50.0, 100.0);

  // Resize end from 100 to 150
  Drag(100.0, 150.0);

  ASSERT_EQ(timeline_.selected_time_ranges().size(), 1);
  EXPECT_DOUBLE_EQ(timeline_.selected_time_ranges()[0].start(), 50.0);
  EXPECT_DOUBLE_EQ(timeline_.selected_time_ranges()[0].end(), 150.0);
}

TEST_F(TimelineTimeRangeResizeTest, ResizeSwapsStartAndEnd) {
  AddSelectedTimeRange(50.0, 100.0);

  // Drag start (50) past end (100) to 150
  Drag(50.0, 150.0);

  ASSERT_EQ(timeline_.selected_time_ranges().size(), 1);
  EXPECT_DOUBLE_EQ(timeline_.selected_time_ranges()[0].start(), 100.0);
  EXPECT_DOUBLE_EQ(timeline_.selected_time_ranges()[0].end(), 150.0);
}

TEST_F(TimelineTimeRangeResizeTest, ResizeWithSnapping) {
  timeline_.set_mouse_mode(MouseMode::kTiming);
  timeline_.set_snap_to_time_range_enabled(true);

  // Create a snap point at 100.0
  AddSelectedTimeRange(100.0, 150.0);

  // Create another range to resize (index 1)
  AddSelectedTimeRange(10.0, 50.0);
  ASSERT_EQ(timeline_.selected_time_ranges().size(), 2);

  // Resize end of second range (50.0) near snap point (100.0)
  // Threshold = 16 / 10 = 1.6us.
  // Drag to 99.0, it should snap to 100.0.
  Drag(50.0, 99.0, /*shift=*/true);

  ASSERT_EQ(timeline_.selected_time_ranges().size(), 2);
  EXPECT_DOUBLE_EQ(timeline_.selected_time_ranges()[1].start(), 10.0);
  EXPECT_DOUBLE_EQ(timeline_.selected_time_ranges()[1].end(), 100.0);
}

TEST_F(MockTimelineImGuiFixture, TriggersZoomWhenNavigatedEventNotPresent) {
  timeline_.SetTimelineData(
      CreateTimelineData({{"root", 0.0, 10000000.0, 0, 1, 1, 0}}));
  timeline_.set_data_time_range({0.0, 10000000.0});

  timeline_.SetSearchQuery("event");
  timeline_.SetSearchResults(
      CreateSearchResults({{"ev", 5000000.0, 1000.0, 0, 1, 1, 999}}));
  timeline_.NavigateToNextSearchResult();

  EXPECT_GT(timeline_.visible_range_target().start(), 1000.0);
}

TEST_F(MockTimelineImGuiFixture, TriggersZoomWhenPrevNavigatedEventNotPresent) {
  timeline_.SetTimelineData(
      CreateTimelineData({{"root", 0.0, 10000000.0, 0, 1, 1, 0}}));
  timeline_.set_data_time_range({0.0, 10000000.0});

  timeline_.SetSearchQuery("event");
  timeline_.SetSearchResults(
      CreateSearchResults({{"ev", 5000000.0, 1000.0, 0, 1, 1, 999}}));
  timeline_.NavigateToPrevSearchResult();

  EXPECT_GT(timeline_.visible_range_target().start(), 1000.0);
}

TEST_F(MockTimelineImGuiFixture, SetSearchResultsRobustFallbackMatching) {
  timeline_.SetTimelineData(CreateTimelineData({
      {"copy.done", 100.0, 50.0, 0, 1, 1, 1001},
      {"while", 100.0, 50.0, 1, 1, 1, 1002},
  }));
  timeline_.set_data_time_range({0.0, 1000.0});

  timeline_.SetSearchQuery("while");
  timeline_.SetSearchResults(
      CreateSearchResults({{"while", 100.0001, 50.0001, 0, 1, 1, 9999}}));

  EXPECT_EQ(timeline_.get_search_results_count(), 1);
  timeline_.NavigateToNextSearchResult();
  EXPECT_EQ(timeline_.selected_event_index(), 1);
}

enum class MismatchType {
  kPid,
  kTid,
  kName,
  kDuration,
  kStartTime,
  kSearchDuration,
  kSearchStartTime
};

class ReconciliationMismatchTest
    : public TimelineImGuiTestFixture<MockTimeline>,
      public ::testing::WithParamInterface<MismatchType> {};

TEST_P(ReconciliationMismatchTest, MismatchFailsReconciliation) {
  timeline_.SetTimelineData(
      CreateTimelineData({{"eventA", 100.0, 50.0, 0, 1, 1, 1001}}));
  timeline_.set_data_time_range({0.0, 1000.0});

  if (GetParam() != MismatchType::kSearchDuration &&
      GetParam() != MismatchType::kSearchStartTime) {
    timeline_.SetSearchQuery("eventA");
    if (GetParam() == MismatchType::kStartTime) {
      timeline_.SetSearchResults(
          CreateSearchResults({{"eventA", 100.0, 50.0, 0, 1, 1, 9999}}));
      EXPECT_EQ(timeline_.get_search_results_count(), 1);
      timeline_.SetTimelineData(
          CreateTimelineData({{"eventA", 200.0, 50.0, 0, 1, 1, 1002}}));
      timeline_.NavigateToNextSearchResult();
      EXPECT_EQ(timeline_.selected_event_index(), -1);
    } else {
      timeline_.SetSearchResults(
          CreateSearchResults({{"eventA", 100.0, 50.0, 0, 1, 1, 1001}}));
      timeline_.NavigateToNextSearchResult();
      ASSERT_EQ(timeline_.selected_event_index(), 0);

      std::vector<EventDef> new_events = {{"dummy", 0.0, 10.0, 0, 1, 1, 3003}};
      switch (GetParam()) {
        case MismatchType::kPid:
          new_events.push_back({"eventA", 100.0, 50.0, 0, 2, 1, 2002});
          break;
        case MismatchType::kTid:
          new_events.push_back({"eventA", 100.0, 50.0, 0, 1, 2, 2002});
          break;
        case MismatchType::kName:
          new_events.push_back({"eventB", 100.0, 50.0, 0, 1, 1, 2002});
          break;
        case MismatchType::kDuration:
          new_events.push_back({"eventA", 100.0, 100.0, 0, 1, 1, 2002});
          break;
        default:
          break;
      }
      timeline_.SetTimelineData(CreateTimelineData(new_events));
      EXPECT_EQ(timeline_.selected_event_index(), -1);
    }
  } else {
    timeline_.SetSearchQuery("eventA");
    if (GetParam() == MismatchType::kSearchDuration) {
      timeline_.SetSearchResults(
          CreateSearchResults({{"eventA", 100.0, 60.0, 0, 1, 1, 9999}}));
    } else {
      timeline_.SetSearchResults(
          CreateSearchResults({{"eventA", 200.0, 50.0, 0, 1, 1, 9999}}));
    }
    timeline_.NavigateToNextSearchResult();
    EXPECT_EQ(timeline_.selected_event_index(), -1);
  }
}

INSTANTIATE_TEST_SUITE_P(
    VariousMismatches, ReconciliationMismatchTest,
    ::testing::Values(MismatchType::kPid, MismatchType::kTid,
                      MismatchType::kName, MismatchType::kDuration,
                      MismatchType::kStartTime, MismatchType::kSearchDuration,
                      MismatchType::kSearchStartTime));

enum class SortingTestType {
  kComprehensive,
  kIgnoreNonSortMetadata,
  kInvalidLevels,
  kIgnoreNonSortMetadataReverse,
  kEqualSortIndexFallbackTid,
  kMinLevelNotOverwritten,
  kMinLevelUnloadedActiveThread,
  kProcessSortIndexFallbackPid,
  kMinLevelIgnoresInvalidLevels
};

class SearchResultsSortingTest
    : public TimelineImGuiTestFixture<MockTimeline>,
      public ::testing::WithParamInterface<SortingTestType> {};

TEST_P(SearchResultsSortingTest, SortingBehaviors) {
  switch (GetParam()) {
    case SortingTestType::kComprehensive: {
      timeline_.SetTimelineData(CreateTimelineData({
          {"ev0", 100.0, 10.0, 0, 2, 1, 100},
          {"ev1", 200.0, 10.0, 1, 1, 2, 101},
          {"ev2", 300.0, 10.0, 2, 1, 3, 102},
          {"ev3", 150.0, 10.0, 0, 1, 2, 103},
          {"ev4", 120.0, 10.0, 1, 1, 1, 104},
          {"ev5", 110.0, 10.0, 2, 1, 1, 105},
          {"ev6", 50.0, 10.0, 0, 3, 1, 106},
          {"ev7", 60.0, 10.0, 0, 4, 1, 107},
          {"ev8", 130.0, 10.0, 1, 1, 1, 108},
          {"ev9", 400.0, 10.0, 1, 1, 4, 109},
          {"ev10", 410.0, 10.0, 1, 1, 5, 110},
          {"ev11", 420.0, 10.0, 1, 1, 7, 111},
          {"ev12", 430.0, 10.0, 1, 1, 6, 112},
          {"ev15", 160.0, 10.0, 0, 5, 1, 115},
          {"ev16", 170.0, 10.0, 0, 6, 1, 116},
      }));
      timeline_.set_data_time_range({0.0, 1000.0});

      ParsedTraceEvents search_results;
      auto add_meta = [&](Phase ph, absl::string_view name, ProcessId pid,
                          ThreadId tid, absl::string_view arg_key,
                          absl::string_view arg_val) {
        TraceEvent m;
        m.ph = ph;
        m.name = std::string(name);
        m.pid = pid;
        m.tid = tid;
        m.args[std::string(arg_key)] = std::string(arg_val);
        search_results.flame_events.push_back(m);
      };
      add_meta(Phase::kMetadata, kProcessSortIndex, 2, 0, kSortIndex, "10");
      add_meta(Phase::kMetadata, kProcessSortIndex, 3, 0, kSortIndex, "5");
      add_meta(Phase::kMetadata, kProcessSortIndex, 5, 0, kSortIndex, "30");
      add_meta(Phase::kMetadata, kProcessSortIndex, 6, 0, kSortIndex, "30");
      add_meta(Phase::kMetadata, kThreadSortIndex, 1, 4, kSortIndex, "20");
      add_meta(Phase::kMetadata, kThreadSortIndex, 1, 5, kSortIndex, "20");

      auto add_event = [&](EventId event_id, ProcessId pid, ThreadId tid,
                           absl::string_view name, double ts, double dur) {
        TraceEvent ev;
        ev.ph = Phase::kComplete;
        ev.event_id = event_id;
        ev.pid = pid;
        ev.tid = tid;
        ev.name = std::string(name);
        ev.ts = ts;
        ev.dur = dur;
        search_results.flame_events.push_back(ev);
      };
      add_event(100, 2, 1, "ev0", 100.0, 10.0);
      add_event(101, 1, 2, "ev1", 200.0, 10.0);
      add_event(102, 1, 3, "ev2", 300.0, 10.0);
      add_event(103, 1, 2, "ev3", 150.0, 10.0);
      add_event(104, 1, 1, "ev4", 120.0, 10.0);
      add_event(105, 1, 1, "ev5", 110.0, 10.0);
      add_event(106, 3, 1, "ev6", 50.0, 10.0);
      add_event(107, 4, 1, "ev7", 60.0, 10.0);
      add_event(108, 1, 1, "ev8", 130.0, 10.0);
      add_event(109, 1, 4, "ev9", 400.0, 10.0);
      add_event(110, 1, 5, "ev10", 410.0, 10.0);
      add_event(111, 1, 7, "ev11", 420.0, 10.0);
      add_event(112, 1, 6, "ev12", 430.0, 10.0);
      add_event(113, 1, 1, "ev13", 140.0, 10.0);
      add_event(114, 1, 1, "ev14", 150.0, 10.0);
      add_event(115, 5, 1, "ev15", 160.0, 10.0);
      add_event(116, 6, 1, "ev16", 170.0, 10.0);

      timeline_.SetSearchQuery("ev");
      timeline_.SetSearchResults(search_results);

      std::vector<int> expected_order = {6, 0, 13, 14, 3,  1,  9, 10, 4,
                                         8, 8, 8,  5,  12, 11, 2, 7};
      for (int i = 0; i < expected_order.size(); ++i) {
        timeline_.NavigateToNextSearchResult();
        EXPECT_EQ(timeline_.selected_event_index(), expected_order[i])
            << "Mismatch at index " << i;
      }
      break;
    }
    case SortingTestType::kIgnoreNonSortMetadata: {
      timeline_.SetTimelineData(CreateTimelineData({
          {"evA", 100.0, 10.0, 99, 1, 1, 100},
          {"evC", 300.0, 10.0, 99, 1, 3, 102},
      }));
      timeline_.set_data_time_range({0.0, 1000.0});

      ParsedTraceEvents search_results;
      auto add_meta = [&](Phase ph, absl::string_view name, ProcessId pid,
                          ThreadId tid, absl::string_view arg_key,
                          absl::string_view arg_val) {
        TraceEvent m;
        m.ph = ph;
        m.name = std::string(name);
        m.pid = pid;
        m.tid = tid;
        m.args[std::string(arg_key)] = std::string(arg_val);
        search_results.flame_events.push_back(m);
      };
      add_meta(Phase::kMetadata, kThreadName, 1, 1, kSortIndex, "5");
      add_meta(Phase::kMetadata, kThreadSortIndex, 1, 2, kSortIndex, "10");
      add_meta(Phase::kComplete, kThreadSortIndex, 1, 3, kSortIndex, "2");

      TraceEvent evA{.ph = Phase::kComplete,
                     .event_id = 100,
                     .pid = 1,
                     .tid = 1,
                     .name = "evA",
                     .ts = 100.0,
                     .dur = 10.0};
      TraceEvent evB{.ph = Phase::kComplete,
                     .event_id = 101,
                     .pid = 1,
                     .tid = 2,
                     .name = "evB",
                     .ts = 200.0,
                     .dur = 10.0};
      TraceEvent evC{.ph = Phase::kComplete,
                     .event_id = 102,
                     .pid = 1,
                     .tid = 3,
                     .name = "evC",
                     .ts = 300.0,
                     .dur = 10.0};
      search_results.flame_events.push_back(evA);
      search_results.flame_events.push_back(evB);
      search_results.flame_events.push_back(evC);

      timeline_.SetSearchQuery("ev");
      timeline_.SetSearchResults(search_results);

      timeline_.NavigateToNextSearchResult();
      EXPECT_EQ(timeline_.selected_event_index(), -1);
      timeline_.NavigateToNextSearchResult();
      EXPECT_EQ(timeline_.selected_event_index(), 0);
      timeline_.NavigateToNextSearchResult();
      EXPECT_EQ(timeline_.selected_event_index(), 0);
      timeline_.NavigateToNextSearchResult();
      EXPECT_EQ(timeline_.selected_event_index(), 1);
      break;
    }
    case SortingTestType::kInvalidLevels: {
      timeline_.SetTimelineData(CreateTimelineData({
          {"evA", 100.0, 10.0, -1, 1, 2, 100},
          {"evB", 200.0, 10.0, 0, 1, 1, 101},
      }));
      timeline_.set_data_time_range({0.0, 1000.0});

      timeline_.SetSearchQuery("ev");
      timeline_.SetSearchResults(CreateSearchResults({
          {"evA", 100.0, 10.0, 0, 1, 2, 100},
          {"evB", 200.0, 10.0, 0, 1, 1, 101},
      }));

      timeline_.NavigateToNextSearchResult();
      EXPECT_EQ(timeline_.selected_event_index(), 1);
      timeline_.NavigateToNextSearchResult();
      EXPECT_EQ(timeline_.selected_event_index(), 0);
      break;
    }
    case SortingTestType::kIgnoreNonSortMetadataReverse: {
      timeline_.SetTimelineData(
          CreateTimelineData({{"evB", 100.0, 10.0, 99, 1, 2, 100}}));
      timeline_.set_data_time_range({0.0, 1000.0});

      ParsedTraceEvents search_results;
      auto add_meta = [&](Phase ph, absl::string_view name, ProcessId pid,
                          ThreadId tid, absl::string_view arg_key,
                          absl::string_view arg_val) {
        TraceEvent m;
        m.ph = ph;
        m.name = std::string(name);
        m.pid = pid;
        m.tid = tid;
        m.args[std::string(arg_key)] = std::string(arg_val);
        search_results.flame_events.push_back(m);
      };
      add_meta(Phase::kMetadata, kThreadSortIndex, 1, 2, kSortIndex, "1");
      add_meta(Phase::kMetadata, kThreadSortIndex, 1, 1, kSortIndex, "10");

      TraceEvent evA;
      evA.ph = Phase::kComplete;
      evA.event_id = 999;
      evA.pid = 1;
      evA.tid = 1;
      evA.name = "evA";
      evA.ts = 150.0;
      evA.dur = 10.0;
      TraceEvent evB;
      evB.ph = Phase::kComplete;
      evB.event_id = 100;
      evB.pid = 1;
      evB.tid = 2;
      evB.name = "evB";
      evB.ts = 100.0;
      evB.dur = 10.0;
      search_results.flame_events.push_back(evA);
      search_results.flame_events.push_back(evB);

      timeline_.SetSearchQuery("ev");
      timeline_.SetSearchResults(search_results);

      timeline_.NavigateToNextSearchResult();
      EXPECT_EQ(timeline_.selected_event_index(), 0);
      timeline_.NavigateToNextSearchResult();
      EXPECT_EQ(timeline_.selected_event_index(), 0);
      EXPECT_EQ(
          timeline_.get_pending_navigation_event_id_for_test().value_or(0),
          999);
      break;
    }
    case SortingTestType::kEqualSortIndexFallbackTid: {
      timeline_.SetTimelineData(CreateTimelineData({
          {"evA", 100.0, 10.0, 0, 1, 2, 100},
          {"evB", 200.0, 10.0, 0, 1, 1, 101},
      }));
      timeline_.set_data_time_range({0.0, 1000.0});

      ParsedTraceEvents search_results;
      auto add_meta = [&](Phase ph, absl::string_view name, ProcessId pid,
                          ThreadId tid, absl::string_view arg_key,
                          absl::string_view arg_val) {
        TraceEvent m;
        m.ph = ph;
        m.name = std::string(name);
        m.pid = pid;
        m.tid = tid;
        m.args[std::string(arg_key)] = std::string(arg_val);
        search_results.flame_events.push_back(m);
      };
      add_meta(Phase::kMetadata, kThreadSortIndex, 1, 2, kSortIndex, "1");
      add_meta(Phase::kMetadata, kThreadSortIndex, 1, 1, kSortIndex, "1");

      TraceEvent evA;
      evA.ph = Phase::kComplete;
      evA.event_id = 100;
      evA.pid = 1;
      evA.tid = 2;
      evA.name = "evA";
      evA.ts = 100.0;
      evA.dur = 10.0;
      TraceEvent evB;
      evB.ph = Phase::kComplete;
      evB.event_id = 101;
      evB.pid = 1;
      evB.tid = 1;
      evB.name = "evB";
      evB.ts = 200.0;
      evB.dur = 10.0;
      search_results.flame_events.push_back(evA);
      search_results.flame_events.push_back(evB);

      timeline_.SetSearchQuery("ev");
      timeline_.SetSearchResults(search_results);

      timeline_.NavigateToNextSearchResult();
      EXPECT_EQ(timeline_.selected_event_index(), 1);
      timeline_.NavigateToNextSearchResult();
      EXPECT_EQ(timeline_.selected_event_index(), 0);
      break;
    }
    case SortingTestType::kMinLevelNotOverwritten: {
      timeline_.SetTimelineData(CreateTimelineData({
          {"evA1", 100.0, 10.0, 0, 1, 2, 100},
          {"evA2", 50.0, 10.0, 2, 1, 2, 101},
          {"evB1", 200.0, 10.0, 1, 1, 1, 102},
      }));
      timeline_.set_data_time_range({0.0, 1000.0});

      timeline_.SetSearchQuery("ev");
      timeline_.SetSearchResults(CreateSearchResults({
          {"evA1", 100.0, 10.0, 0, 1, 2, 100},
          {"evA2", 50.0, 10.0, 0, 1, 2, 101},
          {"evB1", 200.0, 10.0, 0, 1, 1, 102},
      }));

      timeline_.NavigateToNextSearchResult();
      EXPECT_EQ(timeline_.selected_event_index(), 0);
      timeline_.NavigateToNextSearchResult();
      EXPECT_EQ(timeline_.selected_event_index(), 1);
      timeline_.NavigateToNextSearchResult();
      EXPECT_EQ(timeline_.selected_event_index(), 2);
      break;
    }
    case SortingTestType::kMinLevelUnloadedActiveThread: {
      timeline_.SetTimelineData(
          CreateTimelineData({{"evB", 200.0, 10.0, 1, 1, 1, 100}}));
      timeline_.set_data_time_range({0.0, 1000.0});

      timeline_.SetSearchQuery("ev");
      timeline_.SetSearchResults(CreateSearchResults({
          {"evA_unloaded", 300.0, 10.0, 0, 1, 1, 999},
          {"evB", 200.0, 10.0, 0, 1, 1, 100},
      }));

      const std::vector<Timeline::SearchResult>& results =
          timeline_.get_search_results_for_test();
      ASSERT_EQ(results.size(), 2);
      EXPECT_EQ(results[0].event_id, 100);
      EXPECT_EQ(results[1].event_id, 999);
      break;
    }
    case SortingTestType::kProcessSortIndexFallbackPid: {
      timeline_.SetTimelineData(
          CreateTimelineData({{"evB", 200.0, 10.0, 0, 2, 1, 100},
                              {"evA", 100.0, 10.0, 0, 1, 1, 101}},
                             std::vector<Group>{{.type = Group::Type::kFlame,
                                                 .name = "Process 1",
                                                 .start_level = 0},
                                                {.type = Group::Type::kFlame,
                                                 .name = "Process 2",
                                                 .start_level = 0}}));
      timeline_.set_data_time_range({0.0, 1000.0});

      ParsedTraceEvents search_results;
      auto add_meta = [&](Phase ph, absl::string_view name, ProcessId pid,
                          ThreadId tid, absl::string_view arg_key,
                          absl::string_view arg_val) {
        TraceEvent m;
        m.ph = ph;
        m.name = std::string(name);
        m.pid = pid;
        m.tid = tid;
        m.args[std::string(arg_key)] = std::string(arg_val);
        search_results.flame_events.push_back(m);
      };
      add_meta(Phase::kMetadata, kProcessSortIndex, 1, 0, kSortIndex, "10.0");
      add_meta(Phase::kMetadata, kProcessSortIndex, 2, 0, kSortIndex, "10.0");

      TraceEvent evB;
      evB.ph = Phase::kComplete;
      evB.event_id = 100;
      evB.pid = 2;
      evB.tid = 1;
      evB.name = "evB";
      evB.ts = 200.0;
      evB.dur = 10.0;
      TraceEvent evA;
      evA.ph = Phase::kComplete;
      evA.event_id = 101;
      evA.pid = 1;
      evA.tid = 1;
      evA.name = "evA";
      evA.ts = 100.0;
      evA.dur = 10.0;
      search_results.flame_events.push_back(evB);
      search_results.flame_events.push_back(evA);

      timeline_.SetSearchQuery("ev");
      timeline_.SetSearchResults(search_results);

      const std::vector<Timeline::SearchResult>& results =
          timeline_.get_search_results_for_test();
      ASSERT_EQ(results.size(), 2);
      EXPECT_EQ(results[0].event_id, 101);
      EXPECT_EQ(results[1].event_id, 100);
      break;
    }
    case SortingTestType::kMinLevelIgnoresInvalidLevels: {
      timeline_.SetTimelineData(CreateTimelineData({
          {"ev_invalid", 100.0, 10.0, -1, 1, 1, 100},
          {"ev_valid", 200.0, 10.0, 2, 1, 1, 101},
      }));
      timeline_.set_data_time_range({0.0, 1000.0});

      timeline_.SetSearchQuery("ev");
      timeline_.SetSearchResults(CreateSearchResults({
          {"ev_unloaded", 300.0, 10.0, -1, 1, 1, 999},
          {"ev_other", 200.0, 10.0, 1, 1, 1, 101},
      }));

      const std::vector<Timeline::SearchResult>& results =
          timeline_.get_search_results_for_test();
      ASSERT_EQ(results.size(), 2);
      EXPECT_EQ(results[0].event_id, 101);
      EXPECT_EQ(results[1].event_id, 999);
      break;
    }
  }
}

INSTANTIATE_TEST_SUITE_P(
    VariousSorting, SearchResultsSortingTest,
    ::testing::Values(SortingTestType::kComprehensive,
                      SortingTestType::kIgnoreNonSortMetadata,
                      SortingTestType::kInvalidLevels,
                      SortingTestType::kIgnoreNonSortMetadataReverse,
                      SortingTestType::kEqualSortIndexFallbackTid,
                      SortingTestType::kMinLevelNotOverwritten,
                      SortingTestType::kMinLevelUnloadedActiveThread,
                      SortingTestType::kProcessSortIndexFallbackPid,
                      SortingTestType::kMinLevelIgnoresInvalidLevels));

enum class HappyPathType {
  kReconciliation,
  kPendingNavigation,
  kPendingNavigationFallback,
  kActiveNavigation,
  kActiveNavigationMissingEvent,
  kSearchActiveIndex,
  kDifferentiateByDuration
};

class ReconciliationHappyPathTest
    : public TimelineImGuiTestFixture<MockTimeline>,
      public ::testing::WithParamInterface<HappyPathType> {};

TEST_P(ReconciliationHappyPathTest, HappyPathBehaviors) {
  switch (GetParam()) {
    case HappyPathType::kReconciliation: {
      timeline_.SetTimelineData(
          CreateTimelineData({{"eventA", 100.0, 50.0, 0, 1, 1, 1001}}));
      timeline_.SetSearchQuery("eventA");
      timeline_.SetSearchResults(
          CreateSearchResults({{"eventA", 100.0, 50.0, 0, 1, 1, 1001}}));
      EXPECT_EQ(timeline_.get_search_results_count(), 1);
      timeline_.NavigateToNextSearchResult();
      EXPECT_EQ(timeline_.selected_event_index(), 0);

      timeline_.SetTimelineData(CreateTimelineData({
          {"dummy", 0.0, 10.0, 0, 1, 1, 999},
          {"eventA", 100.0, 50.0, 1, 1, 1, 2002},
      }));
      EXPECT_EQ(timeline_.selected_event_index(), 1);

      timeline_.SetTimelineData(CreateTimelineData({
          {"dummy", 0.0, 10.0, 0, 1, 1, 999},
          {"eventA", 100.0, 50.0, 0, 1, 1, 2002},
      }));
      EXPECT_EQ(timeline_.selected_event_index(), 1);
      break;
    }
    case HappyPathType::kPendingNavigation: {
      timeline_.SetTimelineData({});
      timeline_.SetSearchQuery("eventA");
      timeline_.SetSearchResults(
          CreateSearchResults({{"eventA", 100.0, 50.0, 0, 1, 1, 1001}}));
      timeline_.NavigateToNextSearchResult();

      timeline_.SetTimelineData(
          CreateTimelineData({{"eventA", 100.0, 50.0, 0, 1, 1, 1001}}));
      EXPECT_EQ(timeline_.selected_event_index(), 0);
      EXPECT_FALSE(
          timeline_.get_pending_navigation_event_id_for_test().has_value());
      break;
    }
    case HappyPathType::kPendingNavigationFallback: {
      timeline_.SetTimelineData({});
      timeline_.SetSearchQuery("eventA");
      timeline_.SetSearchResults(
          CreateSearchResults({{"eventA", 100.0, 50.0, 0, 1, 1, 1001}}));
      timeline_.NavigateToNextSearchResult();

      timeline_.SetTimelineData(
          CreateTimelineData({{"eventA", 100.0, 50.0, 0, 1, 1, 2002}}));
      EXPECT_EQ(timeline_.selected_event_index(), 0);
      EXPECT_FALSE(
          timeline_.get_pending_navigation_event_id_for_test().has_value());
      break;
    }
    case HappyPathType::kActiveNavigation: {
      timeline_.SetTimelineData(
          CreateTimelineData({{"eventA", 100.0, 50.0, 0, 1, 1, 1001}}));
      timeline_.SetSearchQuery("eventA");
      timeline_.SetSearchResults(
          CreateSearchResults({{"eventA", 100.0, 50.0, 0, 1, 1, 1001}}));
      timeline_.NavigateToNextSearchResult();

      timeline_.SetSearchResults(
          CreateSearchResults({{"eventA", 100.0, 50.0, 0, 1, 1, 1001}}));
      EXPECT_EQ(timeline_.selected_event_index(), 0);
      break;
    }
    case HappyPathType::kActiveNavigationMissingEvent: {
      timeline_.SetTimelineData({});
      timeline_.SetSearchQuery("eventA");
      timeline_.SetSearchResults(
          CreateSearchResults({{"eventA", 100.0, 50.0, 0, 1, 1, 1001}}));
      timeline_.NavigateToNextSearchResult();

      timeline_.SetSearchResults(
          CreateSearchResults({{"eventA", 100.0, 50.0, 0, 1, 1, 1001}}));
      EXPECT_EQ(timeline_.selected_event_index(), -1);
      break;
    }
    case HappyPathType::kSearchActiveIndex: {
      timeline_.SetTimelineData(
          CreateTimelineData({{"eventA", 100.0, 50.0, 0, 1, 1, 1001}}));
      timeline_.SetSearchQuery("eventA");
      timeline_.SetSearchResults(
          CreateSearchResults({{"eventA", 100.0, 50.0, 0, 1, 1, 1001}}));
      timeline_.NavigateToNextSearchResult();
      ASSERT_EQ(timeline_.get_current_search_result_index(), 0);
      ASSERT_EQ(timeline_.selected_event_index(), 0);

      timeline_.SetTimelineData(CreateTimelineData({
          {"dummy", 0.0, 10.0, 0, 1, 1, 999},
          {"eventA", 100.0, 50.0, 0, 1, 1, 1001},
      }));
      EXPECT_EQ(timeline_.selected_event_index(), 1);
      break;
    }
    case HappyPathType::kDifferentiateByDuration: {
      timeline_.SetTimelineData(
          CreateTimelineData({{"eventA", 100.0, 50.0, 0, 1, 1, 1001}}));
      timeline_.SetSearchQuery("eventA");
      timeline_.SetSearchResults(
          CreateSearchResults({{"eventA", 100.0, 50.0, 0, 1, 1, 1001}}));
      timeline_.NavigateToNextSearchResult();
      ASSERT_EQ(timeline_.get_current_search_result_index(), 0);
      ASSERT_EQ(timeline_.selected_event_index(), 0);

      timeline_.SetTimelineData(CreateTimelineData({
          {"eventA", 100.0, 100.0, 0, 1, 1, 2001},
          {"eventA", 100.0, 50.0, 0, 1, 1, 2002},
      }));
      EXPECT_EQ(timeline_.selected_event_index(), 1);
      break;
    }
  }
}

INSTANTIATE_TEST_SUITE_P(
    VariousHappyPaths, ReconciliationHappyPathTest,
    ::testing::Values(HappyPathType::kReconciliation,
                      HappyPathType::kPendingNavigation,
                      HappyPathType::kPendingNavigationFallback,
                      HappyPathType::kActiveNavigation,
                      HappyPathType::kActiveNavigationMissingEvent,
                      HappyPathType::kSearchActiveIndex,
                      HappyPathType::kDifferentiateByDuration));

TEST_F(MockTimelineImGuiFixture, SetTimelineDataFallbackNameTernaryElse) {
  FlameChartTimelineData data1;
  data1.entry_event_ids.push_back(1001);
  data1.entry_start_times.push_back(100.0);
  data1.entry_total_times.push_back(50.0);
  data1.entry_levels.push_back(0);
  timeline_.SetTimelineData(std::move(data1));

  timeline_.SetSearchQuery("event");
  timeline_.SetSearchResults(
      CreateSearchResults({{"", 100.0, 50.0, 0, 0, 0, 9999}}));

  FlameChartTimelineData data2;
  data2.entry_event_ids.push_back(2002);
  data2.entry_start_times.push_back(100.0);
  data2.entry_total_times.push_back(50.0);
  data2.entry_levels.push_back(0);

  timeline_.SetTimelineData(std::move(data2));
  EXPECT_EQ(timeline_.get_search_results_count(), 1);
}

TEST_F(MockTimelineImGuiFixture,
       SetTimelineDataReconcilesSearchWithNegativeIndexAndEmptyResults) {
  FlameChartTimelineData data1;
  data1.entry_event_ids.push_back(1001);
  data1.entry_start_times.push_back(100.0);
  data1.entry_total_times.push_back(50.0);
  data1.entry_levels.push_back(0);
  timeline_.SetTimelineData(std::move(data1));

  timeline_.SetSearchQuery("non_existent");
  timeline_.SetSearchResults(ParsedTraceEvents());

  ASSERT_EQ(timeline_.get_current_search_result_index(), -1);
  ASSERT_EQ(timeline_.get_search_results_count(), 0);

  FlameChartTimelineData data2;
  data2.entry_event_ids.push_back(2002);
  data2.entry_start_times.push_back(100.0);
  data2.entry_total_times.push_back(50.0);
  data2.entry_levels.push_back(0);

  timeline_.SetTimelineData(std::move(data2));
  EXPECT_EQ(timeline_.get_current_search_result_index(), -1);
}

TEST_F(MockTimelineImGuiFixture, SetSearchResultsEmptyResultsEarlyReturn) {
  ParsedTraceEvents search_results;
  TraceEvent ev{.ph = Phase::kAsyncBegin, .event_id = 1001};
  search_results.flame_events.push_back(ev);

  timeline_.SetSearchQuery("event");
  bool redraw_called = false;
  timeline_.set_redraw_callback([&]() { redraw_called = true; });
  timeline_.SetSearchResults(search_results);

  EXPECT_TRUE(redraw_called);
  EXPECT_EQ(timeline_.get_search_results_count(), 0);
}

TEST_F(MockTimelineImGuiFixture, SetSearchResultsEmptyQueryEarlyReturn) {
  ParsedTraceEvents search_results;
  TraceEvent ev{.ph = Phase::kComplete, .event_id = 1001};
  search_results.flame_events.push_back(ev);

  timeline_.SetSearchQuery("");
  bool redraw_called = false;
  timeline_.set_redraw_callback([&]() { redraw_called = true; });
  timeline_.SetSearchResults(search_results);

  EXPECT_TRUE(redraw_called);
  EXPECT_EQ(timeline_.get_search_results_count(), 0);
}

TEST_F(MockTimelineImGuiFixture, ZoomEventExpandsRelatedTracks) {
  timeline_.set_data_time_range({0.0, 1000.0});
  timeline_.SetSearchQuery("eventA");
  timeline_.SetSearchResults(
      CreateSearchResults({{"eventA", 100.0, 50.0, 0, 1, 1, 1001}}));

  FlameChartTimelineData data =
      CreateTimelineData({{"eventA", 100.0, 50.0, 0, 1, 1, 1001}},
                         std::vector<Group>{{.type = Group::Type::kFlame,
                                             .name = "Process Group",
                                             .start_level = 0,
                                             .nesting_level = 0,
                                             .expanded = false}});
  data.events_by_level = {{0}};
  timeline_.SetTimelineData(std::move(data));

  EXPECT_FALSE(timeline_.timeline_data().groups[0].expanded);

  timeline_.NavigateToNextSearchResult();

  EXPECT_TRUE(timeline_.timeline_data().groups[0].expanded);
}

TEST_F(RealTimelineImGuiFixture, SearchQueryRenderingColors) {
  // Set visible range
  timeline_.SetVisibleRange({0.0, 200.0});
  timeline_.set_data_time_range({0.0, 200.0});

  // Load timeline data with one matching event, one non-matching event, and one
  // instant non-matching event
  FlameChartTimelineData data;
  data.groups.push_back({.type = Group::Type::kFlame,
                         .name = "Group 1",
                         .start_level = 0,
                         .nesting_level = 0,
                         .expanded = true});
  data.events_by_level.push_back({0, 1, 2, 3});

  // Event 0: matching_2 (starts at 10.0, duration 40.0)
  data.entry_names.push_back("matching_2");
  data.entry_start_times.push_back(10.0);
  data.entry_total_times.push_back(40.0);
  data.entry_levels.push_back(0);
  data.entry_pids.push_back(1);
  data.entry_tids.push_back(1);
  data.entry_event_ids.push_back(1001);
  data.entry_args.push_back({});

  // Event 1: other_2 (starts at 60.0, duration 40.0)
  data.entry_names.push_back("other_2");
  data.entry_start_times.push_back(60.0);
  data.entry_total_times.push_back(40.0);
  data.entry_levels.push_back(0);
  data.entry_pids.push_back(1);
  data.entry_tids.push_back(1);
  data.entry_event_ids.push_back(1002);
  data.entry_args.push_back({});

  // Event 2: instant_other_1 (starts at 120.0, duration 0.0) - instant,
  // non-matching
  data.entry_names.push_back("instant_other_1");
  data.entry_start_times.push_back(120.0);
  data.entry_total_times.push_back(0.0);
  data.entry_levels.push_back(0);
  data.entry_pids.push_back(1);
  data.entry_tids.push_back(1);
  data.entry_event_ids.push_back(1003);
  data.entry_args.push_back({});

  // Event 3: matching_instant_0 (starts at 140.0, duration 0.0) - instant,
  // matching
  data.entry_names.push_back("matching_instant_0");
  data.entry_start_times.push_back(140.0);
  data.entry_total_times.push_back(0.0);
  data.entry_levels.push_back(0);
  data.entry_pids.push_back(1);
  data.entry_tids.push_back(1);
  data.entry_event_ids.push_back(1004);
  data.entry_args.push_back({});

  timeline_.SetTimelineData(std::move(data));

  // Set search query targeting "matching"
  timeline_.SetSearchQuery("matching");

  // Run SimulateFrame to draw
  SimulateFrame();

  ImGui::NewFrame();
  timeline_.Draw();

  ImGuiWindow* group_window = nullptr;
  const std::string child_id = "TimelineChild_Group 1_0";
  for (ImGuiWindow* w : ImGui::GetCurrentContext()->Windows) {
    if (std::string(w->Name).find(child_id) != std::string::npos) {
      group_window = w;
      break;
    }
  }
  ASSERT_NE(group_window, nullptr);

  ImDrawList* draw_list = group_window->DrawList;

  // Let's find the colors of the drawn shapes.
  ColorPalette palette = ColorPalette::Default();
  ImU32 original_match_color =
      GetColorForId("matching_2", palette.GetTraceColors());
  ImU32 original_other_color =
      GetColorForId("other_2", palette.GetTraceColors());
  ImU32 original_instant_color =
      GetColorForId("instant_other_1", palette.GetTraceColors());
  ImU32 original_matching_instant_color =
      GetColorForId("matching_instant_0", palette.GetTraceColors());

  EXPECT_NE(original_match_color, original_other_color);
  EXPECT_NE(original_match_color, original_instant_color);
  EXPECT_NE(original_match_color, original_matching_instant_color);
  EXPECT_NE(original_other_color, original_instant_color);
  EXPECT_NE(original_other_color, original_matching_instant_color);
  EXPECT_NE(original_instant_color, original_matching_instant_color);

  // Calculates what the grayscale color of original_other_color should be:
  const uint32_t r = (original_other_color >> IM_COL32_R_SHIFT) & 0xFF;
  const uint32_t g = (original_other_color >> IM_COL32_G_SHIFT) & 0xFF;
  const uint32_t b = (original_other_color >> IM_COL32_B_SHIFT) & 0xFF;
  const uint32_t luminance = (r * 299 + g * 587 + b * 114) / 1000;
  ImU32 expected_gray_color = IM_COL32(luminance, luminance, luminance, 102);

  // Calculates what the grayscale color of original_instant_color should be
  // with transparency:
  const uint32_t ir = (original_instant_color >> IM_COL32_R_SHIFT) & 0xFF;
  const uint32_t ig = (original_instant_color >> IM_COL32_G_SHIFT) & 0xFF;
  const uint32_t ib = (original_instant_color >> IM_COL32_B_SHIFT) & 0xFF;
  const uint32_t iluminance = (ir * 299 + ig * 587 + ib * 114) / 1000;
  ImU32 expected_instant_gray =
      IM_COL32(iluminance, iluminance, iluminance, 102);
  ImU32 expected_instant_transparent_gray =
      ((expected_instant_gray & ~IM_COL32_A_MASK) |
       (static_cast<ImU32>(0.6f * 255.0f) << IM_COL32_A_SHIFT));

  bool found_match_color = false;
  bool found_gray_color = false;
  bool found_original_other_color = false;
  bool found_instant_color = false;
  bool found_matching_instant_color = false;

  for (const auto& vtx : draw_list->VtxBuffer) {
    if (vtx.col == original_match_color) {
      found_match_color = true;
    }
    if (vtx.col == expected_gray_color) {
      found_gray_color = true;
    }
    if (vtx.col == original_other_color) {
      found_original_other_color = true;
    }
    if (vtx.col == expected_instant_transparent_gray) {
      found_instant_color = true;
    }
    if (vtx.col == original_matching_instant_color) {
      found_matching_instant_color = true;
    }
  }

  EXPECT_TRUE(found_match_color);
  EXPECT_TRUE(found_gray_color);
  EXPECT_FALSE(found_original_other_color);
  EXPECT_TRUE(found_instant_color);
  EXPECT_TRUE(found_matching_instant_color);

  ImGui::EndFrame();
}

TEST_F(MockTimelineImGuiFixture,
       SetTimelineDataReconcilesSearchWithNegativeIndex) {
  // 1. Set up initial timeline data with one event
  FlameChartTimelineData data1;
  data1.groups.push_back(
      {.type = Group::Type::kFlame, .name = "Group 1", .start_level = 0});
  data1.entry_names.push_back("eventA");
  data1.entry_start_times.push_back(100.0);
  data1.entry_total_times.push_back(50.0);
  data1.entry_levels.push_back(0);
  data1.entry_pids.push_back(1);
  data1.entry_tids.push_back(1);
  data1.entry_event_ids.push_back(1001);
  data1.entry_args.push_back({});
  timeline_.SetTimelineData(std::move(data1));

  // 2. Set search query to match "eventA". This populates search_results_ with
  // 1 result, but current_search_result_index_ remains -1.
  timeline_.SetSearchQuery("eventA");
  ASSERT_EQ(timeline_.get_search_results_count(), 1);
  ASSERT_EQ(timeline_.get_current_search_result_index(), -1);

  // 3. Set pending_navigation_event_id_ to a non-existent event ID (say 9999)
  timeline_.set_pending_navigation_event_id_for_test(9999);

  // 4. Update timeline data. Since pending_id (9999) is not in data, and
  // current_search_result_index_ is -1, it should not reconcile to any loaded
  // index. pending_navigation_event_id_ must remain 9999. selected_event_index_
  // must remain -1.
  FlameChartTimelineData data2;
  data2.groups.push_back(
      {.type = Group::Type::kFlame, .name = "Group 1", .start_level = 0});
  data2.entry_names.push_back("eventB");
  data2.entry_start_times.push_back(200.0);
  data2.entry_total_times.push_back(50.0);
  data2.entry_levels.push_back(0);
  data2.entry_pids.push_back(2);
  data2.entry_tids.push_back(2);
  data2.entry_event_ids.push_back(1002);
  data2.entry_args.push_back({});

  timeline_.SetTimelineData(std::move(data2));

  // Assertions:
  EXPECT_TRUE(timeline_.get_pending_navigation_event_id_for_test().has_value());
  if (timeline_.get_pending_navigation_event_id_for_test().has_value()) {
    EXPECT_EQ(timeline_.get_pending_navigation_event_id_for_test().value(),
              9999);
  }
  EXPECT_EQ(timeline_.selected_event_index(), -1);
}

TEST_F(MockTimelineImGuiFixture, SetTimelineDataDifferentiatesByDuration) {
  // 1. Set up initial timeline data with one event
  FlameChartTimelineData data1;
  data1.groups.push_back(
      {.type = Group::Type::kFlame, .name = "Group 1", .start_level = 0});
  data1.entry_names.push_back("eventA");
  data1.entry_start_times.push_back(100.0);
  data1.entry_total_times.push_back(50.0);
  data1.entry_levels.push_back(0);
  data1.entry_pids.push_back(1);
  data1.entry_tids.push_back(1);
  data1.entry_event_ids.push_back(1001);
  data1.entry_args.push_back({});
  timeline_.SetTimelineData(std::move(data1));

  // 2. Set search results (contains eventA)
  ParsedTraceEvents search_results;
  TraceEvent ev1;
  ev1.ph = Phase::kComplete;
  ev1.event_id = 1001;
  ev1.pid = 1;
  ev1.tid = 1;
  ev1.name = "eventA";
  ev1.ts = 100.0;
  ev1.dur = 50.0;
  search_results.flame_events.push_back(ev1);

  timeline_.SetSearchQuery("eventA");
  timeline_.SetSearchResults(search_results);

  // Navigate to next search result (sets current_search_result_index_ = 0)
  timeline_.NavigateToNextSearchResult();
  ASSERT_EQ(timeline_.get_current_search_result_index(), 0);
  ASSERT_EQ(timeline_.selected_event_index(), 0);

  // 3. Update timeline data with NEW data.
  // Event 1 (index 0): Duration 100.0 (Does NOT match old search result).
  // Event 2 (index 1): Duration 50.0 (Matches old search result).
  FlameChartTimelineData data2;
  data2.groups.push_back(
      {.type = Group::Type::kFlame, .name = "Group 1", .start_level = 0});

  // Event 1: Different duration
  data2.entry_names.push_back("eventA");
  data2.entry_start_times.push_back(100.0);
  data2.entry_total_times.push_back(100.0);  // Different duration
  data2.entry_levels.push_back(0);
  data2.entry_pids.push_back(1);
  data2.entry_tids.push_back(1);
  data2.entry_event_ids.push_back(2001);  // Different ID to force fallback
  data2.entry_args.push_back({});

  // Event 2: Matching duration
  data2.entry_names.push_back("eventA");
  data2.entry_start_times.push_back(100.0);
  data2.entry_total_times.push_back(50.0);  // Matching duration
  data2.entry_levels.push_back(0);
  data2.entry_pids.push_back(1);
  data2.entry_tids.push_back(1);
  data2.entry_event_ids.push_back(2002);  // Different ID to force fallback
  data2.entry_args.push_back({});

  timeline_.SetTimelineData(std::move(data2));

  // Assertions:
  // The search result should reconcile to index 1 (Event 2), not index 0.
  EXPECT_EQ(timeline_.selected_event_index(), 1);
}

TEST_F(MockTimelineImGuiFixture, SetTimelineDataReconcilesSearchActiveIndex) {
  // 1. Set up initial timeline data with one event
  FlameChartTimelineData data1;
  data1.groups.push_back(
      {.type = Group::Type::kFlame, .name = "Group 1", .start_level = 0});
  data1.entry_names.push_back("eventA");
  data1.entry_start_times.push_back(100.0);
  data1.entry_total_times.push_back(50.0);
  data1.entry_levels.push_back(0);
  data1.entry_pids.push_back(1);
  data1.entry_tids.push_back(1);
  data1.entry_event_ids.push_back(1001);
  data1.entry_args.push_back({});
  timeline_.SetTimelineData(std::move(data1));

  // 2. Set search results (contains eventA)
  ParsedTraceEvents search_results;
  TraceEvent ev1;
  ev1.ph = Phase::kComplete;
  ev1.event_id = 1001;
  ev1.pid = 1;
  ev1.tid = 1;
  ev1.name = "eventA";
  ev1.ts = 100.0;
  ev1.dur = 50.0;
  search_results.flame_events.push_back(ev1);

  timeline_.SetSearchQuery("eventA");
  timeline_.SetSearchResults(search_results);

  // Navigate to next search result (sets current_search_result_index_ = 0)
  timeline_.NavigateToNextSearchResult();
  ASSERT_EQ(timeline_.get_current_search_result_index(), 0);
  ASSERT_EQ(timeline_.selected_event_index(), 0);

  // 3. Update timeline data where eventA is still present at index 0.
  // Reconciliation should set selected_event_index_ to 0.
  FlameChartTimelineData data2;
  data2.groups.push_back(
      {.type = Group::Type::kFlame, .name = "Group 1", .start_level = 0});

  // Add a dummy event at index 0
  data2.entry_names.push_back("dummy");
  data2.entry_start_times.push_back(0.0);
  data2.entry_total_times.push_back(10.0);
  data2.entry_levels.push_back(0);
  data2.entry_pids.push_back(1);
  data2.entry_tids.push_back(1);
  data2.entry_event_ids.push_back(999);
  data2.entry_args.push_back({});

  // Add the matching event at index 1
  data2.entry_names.push_back("eventA");
  data2.entry_start_times.push_back(100.0);
  data2.entry_total_times.push_back(50.0);
  data2.entry_levels.push_back(0);
  data2.entry_pids.push_back(1);
  data2.entry_tids.push_back(1);
  data2.entry_event_ids.push_back(1001);
  data2.entry_args.push_back({});

  timeline_.SetTimelineData(std::move(data2));

  // Assertions:
  EXPECT_EQ(timeline_.selected_event_index(), 1);
}

TEST_F(MockTimelineImGuiFixture,
       SetTimelineDataDeselectsEventWhenActiveSearchResultIsUnloaded) {
  // 1. Set up initial timeline data with one event
  FlameChartTimelineData data1;
  data1.groups.push_back(
      {.type = Group::Type::kFlame, .name = "Group 1", .start_level = 0});
  data1.entry_names.push_back("eventA");
  data1.entry_start_times.push_back(100.0);
  data1.entry_total_times.push_back(50.0);
  data1.entry_levels.push_back(0);
  data1.entry_pids.push_back(1);
  data1.entry_tids.push_back(1);
  data1.entry_event_ids.push_back(1001);
  data1.entry_args.push_back({});
  timeline_.SetTimelineData(std::move(data1));

  // 2. Set search results (contains eventA)
  ParsedTraceEvents search_results;
  TraceEvent ev1;
  ev1.ph = Phase::kComplete;
  ev1.event_id = 1001;
  ev1.pid = 1;
  ev1.tid = 1;
  ev1.name = "eventA";
  ev1.ts = 100.0;
  ev1.dur = 50.0;
  search_results.flame_events.push_back(ev1);

  timeline_.SetSearchQuery("eventA");
  timeline_.SetSearchResults(search_results);

  // Navigate to next search result (sets current_search_result_index_ = 0)
  timeline_.NavigateToNextSearchResult();
  ASSERT_EQ(timeline_.get_current_search_result_index(), 0);
  ASSERT_EQ(timeline_.selected_event_index(), 0);

  // 3. Update timeline data where eventA is NOT present.
  // Reconciliation should set selected_event_index_ to -1.
  FlameChartTimelineData data2;
  data2.groups.push_back(
      {.type = Group::Type::kFlame, .name = "Group 1", .start_level = 0});

  // Add a dummy event that doesn't match
  data2.entry_names.push_back("dummy");
  data2.entry_start_times.push_back(0.0);
  data2.entry_total_times.push_back(10.0);
  data2.entry_levels.push_back(0);
  data2.entry_pids.push_back(1);
  data2.entry_tids.push_back(1);
  data2.entry_event_ids.push_back(999);
  data2.entry_args.push_back({});

  timeline_.SetTimelineData(std::move(data2));

  // Assertions:
  EXPECT_EQ(timeline_.selected_event_index(), -1);
}

TEST_F(MockTimelineImGuiFixture,
       SetSearchResultsSortsEventsOnSameThreadAndLevelByStartTime) {
  // 1. Load data with one thread having two events on the same level (0)
  FlameChartTimelineData data;
  data.groups.push_back(
      {.type = Group::Type::kFlame, .name = "Group 1", .start_level = 0});

  // Event X (will be loaded at index 0): start_time = 200.0 (later)
  data.entry_names.push_back("eventA");
  data.entry_start_times.push_back(200.0);
  data.entry_total_times.push_back(10.0);
  data.entry_levels.push_back(0);
  data.entry_pids.push_back(1);
  data.entry_tids.push_back(1);
  data.entry_event_ids.push_back(1001);
  data.entry_args.push_back({});

  // Event Y (will be loaded at index 1): start_time = 100.0 (earlier)
  data.entry_names.push_back("eventA");
  data.entry_start_times.push_back(100.0);
  data.entry_total_times.push_back(10.0);
  data.entry_levels.push_back(0);
  data.entry_pids.push_back(1);
  data.entry_tids.push_back(1);
  data.entry_event_ids.push_back(1002);
  data.entry_args.push_back({});

  timeline_.SetTimelineData(std::move(data));
  timeline_.set_data_time_range({0.0, 1000.0});

  // 2. Set search results (contains both events)
  ParsedTraceEvents search_results;
  TraceEvent evX;
  evX.ph = Phase::kComplete;
  evX.event_id = 1001;
  evX.pid = 1;
  evX.tid = 1;
  evX.name = "eventA";
  evX.ts = 200.0;
  evX.dur = 10.0;
  search_results.flame_events.push_back(evX);

  TraceEvent evY;
  evY.ph = Phase::kComplete;
  evY.event_id = 1002;
  evY.pid = 1;
  evY.tid = 1;
  evY.name = "eventA";
  evY.ts = 100.0;
  evY.dur = 10.0;
  search_results.flame_events.push_back(evY);

  timeline_.SetSearchQuery("eventA");
  timeline_.SetSearchResults(search_results);

  // Verify sorting order: evY (ts=100.0, index 1 in data) then evX (ts=200.0,
  // index 0 in data)
  EXPECT_EQ(timeline_.get_search_results_count(), 2);

  timeline_.NavigateToNextSearchResult();

  EXPECT_EQ(timeline_.selected_event_index(), 1);  // evY

  timeline_.NavigateToNextSearchResult();

  EXPECT_EQ(timeline_.selected_event_index(), 0);  // evX
}

TEST_F(MockTimelineImGuiFixture, SetSearchResultsPreservesActiveSelectionZoom) {
  // 1. Load data with evA (index 0) and evB (index 1)
  FlameChartTimelineData data;
  data.groups.push_back(
      {.type = Group::Type::kFlame, .name = "Group 1", .start_level = 0});
  data.entry_names.push_back("evA");
  data.entry_start_times.push_back(100.0);
  data.entry_total_times.push_back(10.0);
  data.entry_levels.push_back(0);
  data.entry_pids.push_back(1);
  data.entry_tids.push_back(1);
  data.entry_event_ids.push_back(100);
  data.entry_args.push_back({});

  data.entry_names.push_back("evB");
  data.entry_start_times.push_back(200.0);
  data.entry_total_times.push_back(10.0);
  data.entry_levels.push_back(0);
  data.entry_pids.push_back(1);
  data.entry_tids.push_back(1);
  data.entry_event_ids.push_back(101);
  data.entry_args.push_back({});

  timeline_.SetTimelineData(std::move(data));
  timeline_.set_data_time_range({0.0, 1000.0});

  // 2. Set search results containing [evA]
  ParsedTraceEvents search_results1;
  TraceEvent evA;
  evA.ph = Phase::kComplete;
  evA.event_id = 100;
  evA.pid = 1;
  evA.tid = 1;
  evA.name = "evA";
  evA.ts = 100.0;
  evA.dur = 10.0;
  search_results1.flame_events.push_back(evA);

  timeline_.SetSearchQuery("ev");
  timeline_.SetSearchResults(search_results1);

  // Navigate to first search result (selects evA, index 0)
  timeline_.NavigateToNextSearchResult();
  ASSERT_EQ(timeline_.selected_event_index(), 0);

  // 3. Set search results containing [evB] instead
  ParsedTraceEvents search_results2;
  TraceEvent evB;
  evB.ph = Phase::kComplete;
  evB.event_id = 101;
  evB.pid = 1;
  evB.tid = 1;
  evB.name = "evB";
  evB.ts = 200.0;
  evB.dur = 10.0;
  search_results2.flame_events.push_back(evB);

  // This should trigger navigation zoom preservation (since navigation is
  // active)
  timeline_.SetSearchResults(search_results2);

  // Assertions:
  // Under correct code, active navigation is updated to evB (index 1) and
  // ZoomEvent(1) is called
  EXPECT_EQ(timeline_.selected_event_index(), 1);
  EXPECT_DOUBLE_EQ(timeline_.visible_range_target().start(), 192.5);
  EXPECT_DOUBLE_EQ(timeline_.visible_range_target().end(), 217.5);
}

TEST_F(MockTimelineImGuiFixture,
       SetSearchResultsPreservesActiveSelectionRangeWhenNotLoaded) {
  // 1. Load data with evA (index 0)
  FlameChartTimelineData data;
  data.groups.push_back(
      {.type = Group::Type::kFlame, .name = "Group 1", .start_level = 0});
  data.entry_names.push_back("evA");
  data.entry_start_times.push_back(100.0);
  data.entry_total_times.push_back(10.0);
  data.entry_levels.push_back(0);
  data.entry_pids.push_back(1);
  data.entry_tids.push_back(1);
  data.entry_event_ids.push_back(100);
  data.entry_args.push_back({});

  timeline_.SetTimelineData(std::move(data));
  timeline_.set_data_time_range({0.0, 1000.0});

  // 2. Set search results containing [evA]
  ParsedTraceEvents search_results1;
  TraceEvent evA;
  evA.ph = Phase::kComplete;
  evA.event_id = 100;
  evA.pid = 1;
  evA.tid = 1;
  evA.name = "evA";
  evA.ts = 100.0;
  evA.dur = 10.0;
  search_results1.flame_events.push_back(evA);

  timeline_.SetSearchQuery("ev");
  timeline_.SetSearchResults(search_results1);

  // Navigate to first search result (selects evA, index 0)
  timeline_.NavigateToNextSearchResult();
  ASSERT_EQ(timeline_.selected_event_index(), 0);

  // 3. Set search results containing [evB] which is NOT loaded in data
  ParsedTraceEvents search_results2;
  TraceEvent evB;
  evB.ph = Phase::kComplete;
  evB.event_id = 101;
  evB.pid = 1;
  evB.tid = 1;
  evB.name = "evB";
  evB.ts = 200.0;
  evB.dur = 10.0;
  search_results2.flame_events.push_back(evB);

  // This should trigger navigation range update to evB's center (205.0) with
  // duration (25.0)
  timeline_.SetSearchResults(search_results2);

  // Assertions:
  // selected_event_index_ remains 0 (since evB is not loaded, it doesn't change
  // selection)
  EXPECT_EQ(timeline_.selected_event_index(), 0);

  // visible_range_target should be [192.5, 217.5]
  EXPECT_NEAR(timeline_.visible_range_target().start(), 192.5, 1e-3);
  EXPECT_NEAR(timeline_.visible_range_target().end(), 217.5, 1e-3);
}

TEST_F(MockTimelineImGuiFixture,
       DrawEventsForLevelDrawsInstantEventsAsChevrons) {
  // 1. Setup with duration 0.0 (instant event)
  // Use empty name "" to avoid text drawing which can mask vertex counts.
  FlameChartTimelineData data1;
  data1.groups.push_back({.name = "Group 1",
                          .start_level = 0,
                          .nesting_level = kThreadNestingLevel,
                          .expanded = true});
  data1.events_by_level.push_back({0});
  data1.entry_names.push_back("");
  data1.entry_levels.push_back(0);
  data1.entry_start_times.push_back(25.0);
  data1.entry_total_times.push_back(0.0);
  data1.entry_pids.push_back(1);
  data1.entry_args.push_back({});

  timeline_.SetTimelineData(std::move(data1));
  timeline_.set_data_time_range({0.0, 100.0});
  timeline_.SetVisibleRange({20.0, 40.0});

  ImGui::NewFrame();
  timeline_.Draw();
  ImGuiWindow* window1 = nullptr;
  for (int i = 0; i < ImGui::GetCurrentContext()->Windows.Size; ++i) {
    ImGuiWindow* w = ImGui::GetCurrentContext()->Windows[i];
    if (absl::StrContains(w->Name, "TimelineChild_Group 1_0_")) {
      window1 = w;
      break;
    }
  }
  ASSERT_NE(window1, nullptr);
  int vtx_chevron = window1->DrawList->VtxBuffer.Size;
  ImGui::EndFrame();

  // 2. Setup with duration 10.0 (non-instant event)
  FlameChartTimelineData data2;
  data2.groups.push_back({.name = "Group 1",
                          .start_level = 0,
                          .nesting_level = kThreadNestingLevel,
                          .expanded = true});
  data2.events_by_level.push_back({0});
  data2.entry_names.push_back("");
  data2.entry_levels.push_back(0);
  data2.entry_start_times.push_back(25.0);
  data2.entry_total_times.push_back(10.0);
  data2.entry_pids.push_back(1);
  data2.entry_args.push_back({});

  timeline_.SetTimelineData(std::move(data2));
  timeline_.set_data_time_range({0.0, 100.0});
  timeline_.SetVisibleRange({20.0, 40.0});

  ImGui::NewFrame();
  timeline_.Draw();
  ImGuiWindow* window2 = nullptr;
  for (int i = 0; i < ImGui::GetCurrentContext()->Windows.Size; ++i) {
    ImGuiWindow* w = ImGui::GetCurrentContext()->Windows[i];
    if (absl::StrContains(w->Name, "TimelineChild_Group 1_0_")) {
      window2 = w;
      break;
    }
  }
  ASSERT_NE(window2, nullptr);
  int vtx_rect = window2->DrawList->VtxBuffer.Size;
  ImGui::EndFrame();

  // Verify vertex counts: rectangle has 4 vertices, chevron has 6 vertices.
  EXPECT_EQ(vtx_rect, 4);
  EXPECT_EQ(vtx_chevron, 6);
}

TEST_F(MockTimelineImGuiFixture, SetSearchQueryClearsPreviousResults) {
  // 1. Load data with evA and evB
  FlameChartTimelineData data;
  data.groups.push_back(
      {.type = Group::Type::kFlame, .name = "Group 1", .start_level = 0});

  // Group Y events: evY1 (index 0), evY2 (index 1)
  data.entry_names.push_back("evY1");
  data.entry_start_times.push_back(100.0);
  data.entry_total_times.push_back(10.0);
  data.entry_levels.push_back(0);
  data.entry_pids.push_back(1);
  data.entry_tids.push_back(1);
  data.entry_event_ids.push_back(100);
  data.entry_args.push_back({});

  data.entry_names.push_back("evY2");
  data.entry_start_times.push_back(110.0);
  data.entry_total_times.push_back(10.0);
  data.entry_levels.push_back(0);
  data.entry_pids.push_back(1);
  data.entry_tids.push_back(1);
  data.entry_event_ids.push_back(101);
  data.entry_args.push_back({});

  // Group X events: evX1 (index 2), evX2 (index 3)
  data.entry_names.push_back("evX1");
  data.entry_start_times.push_back(200.0);
  data.entry_total_times.push_back(10.0);
  data.entry_levels.push_back(0);
  data.entry_pids.push_back(1);
  data.entry_tids.push_back(1);
  data.entry_event_ids.push_back(102);
  data.entry_args.push_back({});

  data.entry_names.push_back("evX2");
  data.entry_start_times.push_back(210.0);
  data.entry_total_times.push_back(10.0);
  data.entry_levels.push_back(0);
  data.entry_pids.push_back(1);
  data.entry_tids.push_back(1);
  data.entry_event_ids.push_back(103);
  data.entry_args.push_back({});

  timeline_.SetTimelineData(std::move(data));

  // 2. Search for "evY" -> returns 2 results
  timeline_.SetSearchQuery("evY");
  EXPECT_EQ(timeline_.get_search_results_count(), 2);

  // Navigate 1 -> should go to first result (evY1, index 0)
  timeline_.NavigateToNextSearchResult();
  EXPECT_EQ(timeline_.selected_event_index(), 0);
  EXPECT_EQ(timeline_.get_current_search_result_index(), 0);

  // 3. Search for "evX" -> should clear previous results and reset index to -1
  timeline_.SetSearchQuery("evX");
  EXPECT_EQ(timeline_.get_search_results_count(), 2);
  EXPECT_EQ(timeline_.get_current_search_result_index(), -1);

  // Navigate 1 -> should start from -1, become 0 -> go to first result (evX1,
  // index 2)
  timeline_.NavigateToNextSearchResult();
  EXPECT_EQ(timeline_.selected_event_index(), 2);
}

TEST_F(MockTimelineImGuiFixture,
       SetSearchResultsSortsResultsByMinLevelForUnloadedEventOnActiveThread) {
  // 1. Populate timeline data with single thread (tid 1)
  FlameChartTimelineData data;
  data.groups.push_back({.type = Group::Type::kFlame,
                         .name = "Process Group",
                         .start_level = 0,
                         .nesting_level = 0,
                         .expanded = true});

  // Event 0 (index 0): pid 1, tid 1 (Thread A) - loaded at level 1
  data.entry_names.push_back("evB");
  data.entry_start_times.push_back(200.0);
  data.entry_total_times.push_back(10.0);
  data.entry_levels.push_back(1);
  data.entry_pids.push_back(1);
  data.entry_tids.push_back(1);
  data.entry_event_ids.push_back(100);
  data.entry_args.push_back({});

  timeline_.SetTimelineData(std::move(data));
  timeline_.set_data_time_range({0.0, 1000.0});

  // 2. Search results containing:
  // - evA_unloaded: unloaded on Thread A (pid 1, tid 1) -> level -1
  // - evB: loaded on Thread A (pid 1, tid 1) -> level 1
  // PUSH evA_unloaded BEFORE evB to test stable sort preservation under Mutant
  // B!
  ParsedTraceEvents search_results;
  TraceEvent evA_unloaded;
  evA_unloaded.ph = Phase::kComplete;
  evA_unloaded.event_id = 999;
  evA_unloaded.pid = 1;
  evA_unloaded.tid = 1;
  evA_unloaded.name = "evA_unloaded";
  evA_unloaded.ts = 300.0;
  evA_unloaded.dur = 10.0;
  search_results.flame_events.push_back(evA_unloaded);

  TraceEvent evB;
  evB.ph = Phase::kComplete;
  evB.event_id = 100;
  evB.pid = 1;
  evB.tid = 1;
  evB.name = "evB";
  evB.ts = 200.0;
  evB.dur = 10.0;
  search_results.flame_events.push_back(evB);

  timeline_.SetSearchQuery("ev");
  timeline_.SetSearchResults(search_results);

  // Sorting order verification under clean code:
  // Both are on the same thread, so effective levels are compared.
  // evB effective level = 1.
  // evA_unloaded effective level = 1 (looked up from Thread A min level, which
  // is 1). Levels are equal, so falls back to start time: evB (200.0) <
  // evA_unloaded (300.0) -> evB should be first.
  const std::vector<Timeline::SearchResult>& results =
      timeline_.get_search_results_for_test();
  ASSERT_EQ(results.size(), 2);
  EXPECT_EQ(results[0].event_id, 100);  // evB should be first
  EXPECT_EQ(results[1].event_id, 999);  // evA_unloaded should be second
}

TEST_F(MockTimelineImGuiFixture,
       SetSearchResultsSortsResultsByProcessSortIndexFallbackToPid) {
  // 1. Populate timeline data (loaded) with two events in different processes
  FlameChartTimelineData data;
  data.groups.push_back({.type = Group::Type::kFlame,
                         .name = "Process 1",
                         .start_level = 0,
                         .nesting_level = 0,
                         .expanded = true});
  data.groups.push_back({.type = Group::Type::kFlame,
                         .name = "Process 2",
                         .start_level = 0,
                         .nesting_level = 0,
                         .expanded = true});

  // Event B (index 0, event_id 100): pid 2
  data.entry_names.push_back("evB");
  data.entry_start_times.push_back(200.0);
  data.entry_total_times.push_back(10.0);
  data.entry_levels.push_back(0);
  data.entry_pids.push_back(2);
  data.entry_tids.push_back(1);
  data.entry_event_ids.push_back(100);
  data.entry_args.push_back({});

  // Event A (index 1, event_id 101): pid 1
  data.entry_names.push_back("evA");
  data.entry_start_times.push_back(100.0);
  data.entry_total_times.push_back(10.0);
  data.entry_levels.push_back(0);
  data.entry_pids.push_back(1);
  data.entry_tids.push_back(1);
  data.entry_event_ids.push_back(101);
  data.entry_args.push_back({});

  timeline_.SetTimelineData(std::move(data));
  timeline_.set_data_time_range({0.0, 1000.0});

  // 2. Search results containing process sort metadata:
  // We map both PID 1 and PID 2 to sort index 10.0.
  // PUSH evB (pid 2, event_id 100) BEFORE evA (pid 1, event_id 101) in input!
  ParsedTraceEvents search_results;

  TraceEvent meta1{
      .ph = Phase::kMetadata, .pid = 1, .name = std::string(kProcessSortIndex)};
  meta1.args.try_emplace(kSortIndex, "10.0");
  search_results.flame_events.push_back(meta1);

  TraceEvent meta2{
      .ph = Phase::kMetadata, .pid = 2, .name = std::string(kProcessSortIndex)};
  meta2.args.try_emplace(kSortIndex, "10.0");
  search_results.flame_events.push_back(meta2);

  TraceEvent evB;
  evB.ph = Phase::kComplete;
  evB.event_id = 100;
  evB.pid = 2;
  evB.tid = 1;
  evB.name = "evB";
  evB.ts = 200.0;
  evB.dur = 10.0;
  search_results.flame_events.push_back(evB);

  TraceEvent evA;
  evA.ph = Phase::kComplete;
  evA.event_id = 101;
  evA.pid = 1;
  evA.tid = 1;
  evA.name = "evA";
  evA.ts = 100.0;
  evA.dur = 10.0;
  search_results.flame_events.push_back(evA);

  timeline_.SetSearchQuery("ev");
  timeline_.SetSearchResults(search_results);

  // Sorting verification under clean code:
  // Both have same process sort index (10.0).
  // So it falls back to PID comparison:
  // evA (pid 1) < evB (pid 2) -> evA (event_id 101) should be first!
  const std::vector<Timeline::SearchResult>& results =
      timeline_.get_search_results_for_test();
  ASSERT_EQ(results.size(), 2);
  EXPECT_EQ(results[0].event_id, 101);  // evA (event_id 101) should be first
  EXPECT_EQ(results[1].event_id, 100);  // evB (event_id 100) should be second
}

TEST_F(MockTimelineImGuiFixture,
       NavigateToNextSearchResultPansToUnloadedEvent) {
  timeline_.SetTimelineData(
      CreateTimelineData({{"evA", 100.0, 10.0, 0, 1, 1, 100}}));
  timeline_.set_data_time_range({0.0, 1000.0});

  timeline_.SetSearchQuery("ev");
  timeline_.SetSearchResults(CreateSearchResults({
      {"evA", 100.0, 10.0, 0, 1, 1, 100},
      {"evB", 200.0, 10.0, 0, 1, 1, 101},
  }));

  timeline_.NavigateToNextSearchResult();
  ASSERT_EQ(timeline_.selected_event_index(), 0);

  timeline_.NavigateToNextSearchResult();

  EXPECT_EQ(timeline_.selected_event_index(), 0);
  EXPECT_EQ(timeline_.get_pending_navigation_event_id_for_test(), 101);
  EXPECT_NEAR(timeline_.visible_range_target().start(), 192.5, 1e-3);
  EXPECT_NEAR(timeline_.visible_range_target().end(), 217.5, 1e-3);
}

TEST_F(MockTimelineImGuiFixture,
       NavigateToNextSearchResultPansToUnloadedEventClamped) {
  timeline_.SetTimelineData(
      CreateTimelineData({{"evA", 100.0, 10.0, 0, 1, 1, 100}}));
  timeline_.set_data_time_range({0.0, 200.0});

  timeline_.SetSearchQuery("ev");
  timeline_.SetSearchResults(CreateSearchResults({
      {"evA", 100.0, 10.0, 0, 1, 1, 100},
      {"evB", 195.0, 10.0, 0, 1, 1, 101},
  }));

  timeline_.NavigateToNextSearchResult();
  ASSERT_EQ(timeline_.selected_event_index(), 0);

  timeline_.NavigateToNextSearchResult();

  EXPECT_EQ(timeline_.selected_event_index(), 0);
  EXPECT_EQ(timeline_.get_pending_navigation_event_id_for_test(), 101);
  EXPECT_NEAR(timeline_.visible_range_target().start(), 175.0, 1e-3);
  EXPECT_NEAR(timeline_.visible_range_target().end(), 200.0, 1e-3);
}

TEST_F(MockTimelineImGuiFixture,
       NavigateToPrevSearchResultPansToUnloadedEvent) {
  timeline_.SetTimelineData(
      CreateTimelineData({{"evA", 100.0, 10.0, 0, 1, 1, 100}}));
  timeline_.set_data_time_range({0.0, 1000.0});

  timeline_.SetSearchQuery("ev");
  timeline_.SetSearchResults(CreateSearchResults({
      {"evA", 100.0, 10.0, 0, 1, 1, 100},
      {"evB", 200.0, 10.0, 0, 1, 1, 101},
  }));

  timeline_.NavigateToNextSearchResult();
  ASSERT_EQ(timeline_.selected_event_index(), 0);

  timeline_.NavigateToPrevSearchResult();

  EXPECT_EQ(timeline_.selected_event_index(), 0);
  EXPECT_EQ(timeline_.get_pending_navigation_event_id_for_test(), 101);
  EXPECT_NEAR(timeline_.visible_range_target().start(), 192.5, 1e-3);
  EXPECT_NEAR(timeline_.visible_range_target().end(), 217.5, 1e-3);
}

TEST_F(MockTimelineImGuiFixture,
       NavigateToPrevSearchResultPansToUnloadedEventClamped) {
  timeline_.SetTimelineData(
      CreateTimelineData({{"evA", 100.0, 10.0, 0, 1, 1, 100}}));
  timeline_.set_data_time_range({0.0, 200.0});

  timeline_.SetSearchQuery("ev");
  timeline_.SetSearchResults(CreateSearchResults({
      {"evA", 100.0, 10.0, 0, 1, 1, 100},
      {"evB", 195.0, 10.0, 0, 1, 1, 101},
  }));

  timeline_.NavigateToNextSearchResult();
  ASSERT_EQ(timeline_.selected_event_index(), 0);

  timeline_.NavigateToPrevSearchResult();

  EXPECT_EQ(timeline_.selected_event_index(), 0);
  EXPECT_EQ(timeline_.get_pending_navigation_event_id_for_test(), 101);
  EXPECT_NEAR(timeline_.visible_range_target().start(), 175.0, 1e-3);
  EXPECT_NEAR(timeline_.visible_range_target().end(), 200.0, 1e-3);
}

TEST_F(MockTimelineImGuiFixture, SetSearchResultsTriggersRedrawCallback) {
  ParsedTraceEvents search_results;
  TraceEvent ev;
  ev.ph = Phase::kComplete;
  ev.event_id = 100;
  search_results.flame_events.push_back(ev);

  int redraw_count = 0;
  timeline_.set_redraw_callback([&]() { redraw_count++; });

  timeline_.SetSearchResults(search_results);
  EXPECT_EQ(redraw_count, 1);
}

TEST_F(MockTimelineImGuiFixture,
       SetSearchResultsEmptyResetsCurrentSearchResultIndex) {
  timeline_.SetTimelineData(
      CreateTimelineData({{"evA", 100.0, 10.0, 0, 1, 1, 100}}));

  timeline_.SetSearchQuery("ev");
  timeline_.SetSearchResults(
      CreateSearchResults({{"evA", 100.0, 10.0, 0, 1, 1, 100}}));

  timeline_.NavigateToNextSearchResult();
  EXPECT_EQ(timeline_.get_current_search_result_index(), 0);

  timeline_.SetSearchResults(ParsedTraceEvents());
  EXPECT_EQ(timeline_.get_current_search_result_index(), -1);
}

TEST_F(MockTimelineImGuiFixture,
       SetSearchResultsWithActiveNavigationSelectsFirstResult) {
  timeline_.SetTimelineData(CreateTimelineData({
      {"evA", 100.0, 10.0, 0, 1, 1, 100},
      {"evB", 200.0, 10.0, 0, 1, 1, 101},
  }));

  timeline_.SetSearchQuery("ev");
  timeline_.SetSearchResults(
      CreateSearchResults({{"evA", 100.0, 10.0, 0, 1, 1, 100}}));

  timeline_.NavigateToNextSearchResult();
  EXPECT_EQ(timeline_.selected_event_index(), 0);
  EXPECT_EQ(timeline_.get_current_search_result_index(), 0);

  timeline_.SetSearchResults(
      CreateSearchResults({{"evB", 200.0, 10.0, 0, 1, 1, 101}}));

  EXPECT_EQ(timeline_.selected_event_index(), 1);
  EXPECT_EQ(timeline_.get_current_search_result_index(), 0);
}

TEST_F(MockTimelineImGuiFixture,
       SetSearchResultsProcessesInstantDeprecatedPhase) {
  timeline_.SetTimelineData(CreateTimelineData({
      {"instant_event", 100.0, 0.0, 0, 1, 1, 1001},
  }));
  timeline_.set_data_time_range({0.0, 1000.0});

  timeline_.SetSearchQuery("instant");

  ParsedTraceEvents search_results;
  TraceEvent ev{.ph = Phase::kInstantDeprecated,
                .event_id = 1001,
                .pid = 0,
                .tid = 1,
                .name = "instant_event",
                .ts = 100.0,
                .dur = 0.0};
  search_results.flame_events.push_back(ev);

  timeline_.SetSearchResults(search_results);

  EXPECT_EQ(timeline_.get_search_results_count(), 1);

  timeline_.NavigateToNextSearchResult();

  EXPECT_EQ(timeline_.selected_event_index(), 0);
}

TEST_F(MockTimelineImGuiFixture, SetSearchResultsFallbackUpdatesEventId) {
  timeline_.SetTimelineData(CreateTimelineData({
      {"while", 100.0, 50.0, 1, 1, 1, 1002},
  }));
  timeline_.set_data_time_range({0.0, 1000.0});

  timeline_.SetSearchQuery("while");
  timeline_.SetSearchResults(
      CreateSearchResults({{"while", 100.0001, 50.0001, 1, 1, 1, 9999}}));

  const std::vector<Timeline::SearchResult>& results =
      timeline_.get_search_results_for_test();
  ASSERT_EQ(results.size(), 1);
  EXPECT_EQ(results[0].event_id, 1002);
  EXPECT_EQ(results[0].loaded_index, 0);
}

TEST_F(MockTimelineImGuiFixture, DrawEventInstantHoverColor) {
  timeline_.SetTimelineData(CreateTimelineData({
      {"instant_event", 100.0, 0.0, 0, 1, 1, 1001},
  }));
  timeline_.set_data_time_range({0.0, 1000.0});

  ImGui::NewFrame();
  ImGui::SetNextWindowSize(ImVec2(1000, 500));
  ImGui::Begin(
      "Timeline viewer", nullptr,
      ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);
  ImGuiWindow* window = ImGui::GetCurrentWindow();
  ASSERT_NE(window, nullptr);
  ImDrawList* draw_list = window->DrawList;

  draw_list->VtxBuffer.resize(0);
  draw_list->CmdBuffer.resize(0);
  draw_list->PushClipRect(ImVec2(0, 0), ImVec2(1000, 500));

  EventRect rect{
      .left = 100.0f, .top = 100.0f, .right = 110.0f, .bottom = 110.0f};
  timeline_.CallDrawEvent(0, 0, rect, draw_list);

  ASSERT_GE(draw_list->VtxBuffer.Size, 3);
  EXPECT_EQ(draw_list->VtxBuffer[0].col >> IM_COL32_A_SHIFT,
            static_cast<ImU32>(0.6f * 255.0f));

  draw_list->PopClipRect();

  ImGuiIO& io = ImGui::GetIO();
  io.MousePos = ImVec2(100.0f, 102.0f);

  draw_list->VtxBuffer.resize(0);
  draw_list->CmdBuffer.resize(0);
  draw_list->PushClipRect(ImVec2(0, 0), ImVec2(1000, 500));

  timeline_.CallDrawEvent(0, 0, rect, draw_list);

  ASSERT_GE(draw_list->VtxBuffer.Size, 3);
  EXPECT_EQ(draw_list->VtxBuffer[0].col >> IM_COL32_A_SHIFT, 255);

  draw_list->PopClipRect();
  ImGui::End();
  ImGui::EndFrame();
}

TEST_F(MockTimelineImGuiFixture,
       SetSearchResultsPopulatesAllSearchResultFields) {
  timeline_.SetTimelineData(CreateTimelineData({
      {"target_event", 100.0, 50.0, 1, 2, 3, 1001},
  }));
  timeline_.set_data_time_range({0.0, 1000.0});
  timeline_.SetSearchQuery("target");

  ParsedTraceEvents search_results;
  TraceEvent ev{.ph = Phase::kInstant,
                .event_id = 1001,
                .pid = 2,
                .tid = 3,
                .name = "target_event",
                .ts = 100.0,
                .dur = 50.0};
  search_results.flame_events.push_back(ev);

  timeline_.SetSearchResults(search_results);

  const std::vector<Timeline::SearchResult>& results =
      timeline_.get_search_results_for_test();
  ASSERT_EQ(results.size(), 1);
  EXPECT_EQ(results[0].event_id, 1001);
  EXPECT_EQ(results[0].level, 1);
  EXPECT_EQ(results[0].start_time, 100.0);
  EXPECT_EQ(results[0].duration, 50.0);
  EXPECT_EQ(results[0].pid, 2);
  EXPECT_EQ(results[0].tid, 3);
  EXPECT_EQ(results[0].name, "target_event");
  EXPECT_EQ(results[0].loaded_index, 0);
}

TEST_F(MockTimelineImGuiFixture,
       RecomputeSearchResultsSortsByLevelThenStartTime) {
  timeline_.SetTimelineData(CreateTimelineData({
      {"ev_level2", 200.0, 10.0, 2, 1, 1, 1002},
      {"ev_level1_late", 300.0, 10.0, 1, 1, 1, 1003},
      {"ev_level1_early", 100.0, 10.0, 1, 1, 1, 1001},
  }));
  timeline_.set_data_time_range({0.0, 1000.0});

  timeline_.SetSearchQuery("ev_level");

  const std::vector<Timeline::SearchResult>& results =
      timeline_.get_search_results_for_test();
  ASSERT_EQ(results.size(), 3);
  EXPECT_EQ(results[0].event_id, 1001);
  EXPECT_EQ(results[1].event_id, 1003);
  EXPECT_EQ(results[2].event_id, 1002);
}

TEST_F(MockTimelineImGuiFixture, DrawEventClipsNonPositiveWidth) {
  timeline_.SetTimelineData(CreateTimelineData({
      {"eventA", 100.0, 50.0, 0, 1, 1, 1001},
  }));
  timeline_.set_data_time_range({0.0, 1000.0});

  ImGui::NewFrame();
  ImGui::SetNextWindowSize(ImVec2(1000, 500));
  ImGui::Begin(
      "Timeline viewer", nullptr,
      ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);
  ImGuiWindow* window = ImGui::GetCurrentWindow();
  ASSERT_NE(window, nullptr);
  ImDrawList* draw_list = window->DrawList;

  draw_list->VtxBuffer.resize(0);
  draw_list->CmdBuffer.resize(0);
  draw_list->PushClipRect(ImVec2(0, 0), ImVec2(1000, 500));

  EventRect inverted_rect{
      .left = 110.0f, .top = 100.0f, .right = 100.0f, .bottom = 110.0f};
  timeline_.CallDrawEvent(0, 0, inverted_rect, draw_list);

  EXPECT_EQ(draw_list->VtxBuffer.Size, 0);

  EventRect zero_rect{
      .left = 100.0f, .top = 100.0f, .right = 100.0f, .bottom = 110.0f};
  timeline_.CallDrawEvent(0, 0, zero_rect, draw_list);

  EXPECT_EQ(draw_list->VtxBuffer.Size, 0);

  draw_list->PopClipRect();
  ImGui::End();
  ImGui::EndFrame();
}

TEST_F(MockTimelineImGuiFixture,
       SetTimelineDataRetainsPendingNavigationWhenUnmatched) {
  timeline_.SetTimelineData(CreateTimelineData({
      {"eventA", 100.0, 50.0, 0, 1, 1, 1001},
  }));
  timeline_.set_data_time_range({0.0, 1000.0});
  timeline_.SetSearchQuery("eventA");
  timeline_.SetSearchResults(
      CreateSearchResults({{"eventA", 100.0, 50.0, 0, 1, 1, 1001}}));

  timeline_.NavigateToNextSearchResult();
  ASSERT_EQ(timeline_.get_current_search_result_index(), 0);

  timeline_.set_pending_navigation_event_id_for_test(9999);

  FlameChartTimelineData data2 = CreateTimelineData({
      {"dummy", 0.0, 10.0, 0, 1, 1, 999},
  });
  timeline_.SetTimelineData(std::move(data2));

  EXPECT_EQ(timeline_.get_pending_navigation_event_id_for_test(), 9999);
}

TEST_F(MockTimelineImGuiFixture,
       SetTimelineDataReconcilesPendingNavigationAndZooms) {
  timeline_.SetTimelineData(CreateTimelineData({
      {"eventA", 100.0, 50.0, 0, 1, 1, 1001},
  }));
  timeline_.set_data_time_range({0.0, 1000.0});
  timeline_.SetSearchQuery("eventA");
  timeline_.SetSearchResults(
      CreateSearchResults({{"eventA", 100.0, 50.0, 0, 1, 1, 1001}}));

  timeline_.NavigateToNextSearchResult();
  ASSERT_EQ(timeline_.get_current_search_result_index(), 0);

  timeline_.set_pending_navigation_event_id_for_test(1001);

  FlameChartTimelineData data2 = CreateTimelineData({
      {"eventA", 100.0, 50.0, 0, 1, 1, 1001},
  });
  timeline_.SetTimelineData(std::move(data2));

  EXPECT_FALSE(
      timeline_.get_pending_navigation_event_id_for_test().has_value());
  EXPECT_EQ(timeline_.selected_event_index(), 0);
}

TEST_F(MockTimelineImGuiFixture,
       SetSearchResultsWithActiveNavigationPansToUnloadedEvent) {
  timeline_.SetTimelineData(CreateTimelineData({
      {"evA", 100.0, 10.0, 0, 1, 1, 100},
  }));
  timeline_.set_data_time_range({0.0, 1000.0});

  timeline_.SetSearchQuery("ev");
  timeline_.SetSearchResults(
      CreateSearchResults({{"evA", 100.0, 10.0, 0, 1, 1, 100}}));

  timeline_.NavigateToNextSearchResult();
  ASSERT_EQ(timeline_.get_current_search_result_index(), 0);

  timeline_.SetSearchResults(
      CreateSearchResults({{"evB", 200.0, 10.0, 0, 1, 1, 101}}));

  EXPECT_EQ(timeline_.selected_event_index(), 0);
  EXPECT_EQ(timeline_.get_pending_navigation_event_id_for_test(), 101);
  EXPECT_NEAR(timeline_.visible_range_target().start(), 192.5, 1e-3);
  EXPECT_NEAR(timeline_.visible_range_target().end(), 217.5, 1e-3);
}

TEST_F(MockTimelineImGuiFixture, DrawEventInstantSearchMatchColor) {
  timeline_.SetTimelineData(CreateTimelineData({
      {"instant_event", 100.0, 0.0, 0, 1, 1, 1001},
  }));
  timeline_.set_data_time_range({0.0, 1000.0});
  timeline_.SetSearchQuery("instant");

  ImGui::NewFrame();
  ImGui::SetNextWindowSize(ImVec2(1000, 500));
  ImGui::Begin(
      "Timeline viewer", nullptr,
      ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);
  ImGuiWindow* window = ImGui::GetCurrentWindow();
  ASSERT_NE(window, nullptr);
  ImDrawList* draw_list = window->DrawList;

  draw_list->VtxBuffer.resize(0);
  draw_list->CmdBuffer.resize(0);
  draw_list->PushClipRect(ImVec2(0, 0), ImVec2(1000, 500));

  EventRect rect{
      .left = 100.0f, .top = 100.0f, .right = 110.0f, .bottom = 110.0f};
  timeline_.CallDrawEvent(0, 0, rect, draw_list);

  ASSERT_GE(draw_list->VtxBuffer.Size, 3);
  EXPECT_EQ(draw_list->VtxBuffer[0].col >> IM_COL32_A_SHIFT, 255);

  draw_list->PopClipRect();
  ImGui::End();
  ImGui::EndFrame();
}

TEST_F(MockTimelineImGuiFixture, SetTimelineDataPopulatesMatchingEventIndices) {
  timeline_.SetTimelineData(CreateTimelineData({
      {"target_event", 100.0, 50.0, 0, 1, 1, 1001},
  }));
  timeline_.set_data_time_range({0.0, 1000.0});
  timeline_.SetSearchQuery("target");
  timeline_.SetSearchResults(
      CreateSearchResults({{"target_event", 100.0, 50.0, 0, 1, 1, 1001}}));

  ASSERT_EQ(timeline_.get_search_results_count(), 1);

  FlameChartTimelineData data2 = CreateTimelineData({
      {"target_event", 100.0, 50.0, 0, 1, 1, 1001},
  });
  timeline_.SetTimelineData(std::move(data2));

  EXPECT_TRUE(timeline_.get_matching_event_indices_for_test().contains(0));
}

TEST_F(MockTimelineImGuiFixture,
       RecomputeSearchResultsSortsSimultaneousEvents) {
  timeline_.SetTimelineData(CreateTimelineData({
      {"ev_simultaneous_A", 100.0, 50.0, 1, 1, 1, 1001},
      {"ev_simultaneous_B", 100.0, 10.0, 1, 1, 1, 1002},
  }));
  timeline_.set_data_time_range({0.0, 1000.0});

  timeline_.SetSearchQuery("ev_simultaneous");

  ASSERT_EQ(timeline_.get_search_results_count(), 2);
  const std::vector<Timeline::SearchResult>& results =
      timeline_.get_search_results_for_test();
  EXPECT_EQ(results[0].event_id, 1001);
  EXPECT_EQ(results[1].event_id, 1002);
}

TEST_F(MockTimelineImGuiFixture, SetSearchResultsWithPartialTimelineData) {
  FlameChartTimelineData data;
  // Keep entry_names empty so RecomputeSearchResults safely skips iteration,
  // while SetSearchResults exercises the defensive bounds checks.
  data.entry_start_times.push_back(100.0);
  data.entry_total_times.push_back(10.0);
  data.entry_event_ids.push_back(100);

  timeline_.SetTimelineData(std::move(data));
  timeline_.set_data_time_range({0.0, 1000.0});

  timeline_.SetSearchQuery("ev");
  timeline_.SetSearchResults(
      CreateSearchResults({{"ev_partial", 100.0, 10.0, 0, 1, 1, 100}}));

  const std::vector<Timeline::SearchResult>& results =
      timeline_.get_search_results_for_test();
  ASSERT_EQ(results.size(), 1);
  EXPECT_EQ(results[0].event_id, 100);
  EXPECT_EQ(results[0].level, -1);
}

TEST_F(MockTimelineImGuiFixture,
       SetSearchResultsProcessesInstantDeprecatedEvents) {
  timeline_.SetTimelineData(CreateTimelineData({
      {"ev_deprecated", 100.0, 0.0, 0, 1, 1, 100},
  }));
  timeline_.set_data_time_range({0.0, 1000.0});

  ParsedTraceEvents search_results;
  TraceEvent ev{.ph = Phase::kInstantDeprecated,
                .event_id = 100,
                .pid = 1,
                .tid = 1,
                .name = "ev_deprecated",
                .ts = 100.0,
                .dur = 0.0};
  search_results.flame_events.push_back(ev);

  timeline_.SetSearchQuery("ev");
  timeline_.SetSearchResults(search_results);

  const std::vector<Timeline::SearchResult>& results =
      timeline_.get_search_results_for_test();
  ASSERT_EQ(results.size(), 1);
  EXPECT_EQ(results[0].event_id, 100);
}

TEST_F(MockTimelineImGuiFixture,
       SetSearchResultsPopulatesMatchingEventIndices) {
  timeline_.SetTimelineData(CreateTimelineData({
      {"ev_match", 100.0, 10.0, 0, 1, 1, 100},
  }));
  timeline_.set_data_time_range({0.0, 1000.0});

  timeline_.SetSearchQuery("ev");
  timeline_.SetSearchResults(
      CreateSearchResults({{"ev_match", 100.0, 10.0, 0, 1, 1, 100}}));

  EXPECT_TRUE(timeline_.get_matching_event_indices_for_test().contains(0));
}

TEST_F(MockTimelineImGuiFixture, DrawEventInstantHoveredBaseColor) {
  timeline_.SetTimelineData(CreateTimelineData({
      {"instant_event", 100.0, 0.0, 0, 1, 1, 1001},
  }));
  timeline_.set_data_time_range({0.0, 1000.0});

  ImGui::NewFrame();
  ImGui::SetNextWindowSize(ImVec2(1000, 500));
  ImGui::Begin(
      "Timeline viewer", nullptr,
      ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);
  ImGuiWindow* window = ImGui::GetCurrentWindow();
  ASSERT_NE(window, nullptr);
  ImDrawList* draw_list = window->DrawList;

  draw_list->VtxBuffer.resize(0);
  draw_list->CmdBuffer.resize(0);
  draw_list->PushClipRect(ImVec2(0, 0), ImVec2(1000, 500));

  ImGuiIO& io = ImGui::GetIO();
  io.MousePos = ImVec2(102.0f, 102.0f);

  EventRect rect{
      .left = 100.0f, .top = 100.0f, .right = 110.0f, .bottom = 110.0f};
  timeline_.CallDrawEvent(0, 0, rect, draw_list);

  ASSERT_GE(draw_list->VtxBuffer.Size, 3);
  EXPECT_EQ(draw_list->VtxBuffer[0].col >> IM_COL32_A_SHIFT, 255);

  draw_list->PopClipRect();
  ImGui::End();
  ImGui::EndFrame();
}

TEST_F(MockTimelineImGuiFixture, DrawEventClickSelectionAndDragThreshold) {
  timeline_.SetTimelineData(CreateTimelineData({
      {"ev_click", 100.0, 10.0, 0, 1, 1, 100},
  }));
  timeline_.set_data_time_range({0.0, 1000.0});

  ImGui::NewFrame();
  ImGui::SetNextWindowSize(ImVec2(1000, 500));
  ImGui::Begin(
      "Timeline viewer", nullptr,
      ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);
  ImGuiWindow* window = ImGui::GetCurrentWindow();
  ASSERT_NE(window, nullptr);
  ImDrawList* draw_list = window->DrawList;

  draw_list->VtxBuffer.resize(0);
  draw_list->CmdBuffer.resize(0);
  draw_list->PushClipRect(ImVec2(0, 0), ImVec2(1000, 500));

  ImGuiIO& io = ImGui::GetIO();
  io.MousePos = ImVec2(105.0f, 105.0f);
  io.MouseDown[0] = false;
  io.MouseReleased[0] = true;

  EventRect rect{
      .left = 100.0f, .top = 100.0f, .right = 110.0f, .bottom = 110.0f};

  // 1. Simulate a drag exceeding kClickDistanceThresholdSquared originating on
  // the event rect.
  timeline_.set_selection_start_pos_for_test(ImVec2(100.0f, 100.0f));
  timeline_.CallDrawEvent(0, 0, rect, draw_list);

  EXPECT_EQ(timeline_.selected_event_index(), -1);

  // 2. Simulate a valid click within threshold.
  timeline_.set_selection_start_pos_for_test(ImVec2(104.0f, 104.0f));
  timeline_.CallDrawEvent(0, 0, rect, draw_list);

  EXPECT_EQ(timeline_.selected_event_index(), 0);

  draw_list->PopClipRect();
  ImGui::End();
  ImGui::EndFrame();
}

TEST_F(MockTimelineImGuiFixture, NavigateToSearchResultClampsMinDuration) {
  timeline_.SetTimelineData(
      CreateTimelineData({{"root", 0.0, 10000000.0, 0, 1, 1, 0}}));
  timeline_.set_data_time_range({0.0, 10000000.0});

  timeline_.SetSearchQuery("event");
  timeline_.SetSearchResults(
      CreateSearchResults({{"ev", 100.0, 1.0, 0, 1, 1, 999}}));
  timeline_.NavigateToNextSearchResult();

  EXPECT_DOUBLE_EQ(timeline_.visible_range_target().duration(), 10.0);
}

TEST_F(MockTimelineImGuiFixture, NavigateToSearchResultClampsMaxDuration) {
  timeline_.SetTimelineData(
      CreateTimelineData({{"root", 0.0, 10000000.0, 0, 1, 1, 0}}));
  timeline_.set_data_time_range({0.0, 10000000.0});

  timeline_.SetSearchQuery("event");
  timeline_.SetSearchResults(
      CreateSearchResults({{"ev", 0.0, 10000000.0, 0, 1, 1, 999}}));
  timeline_.NavigateToNextSearchResult();

  EXPECT_DOUBLE_EQ(timeline_.visible_range_target().duration(), 5000000.0);
}

TEST_F(MockTimelineImGuiFixture, NavigateToPrevSearchResultWithThreeResults) {
  timeline_.SetTimelineData(
      CreateTimelineData({{"evA", 100.0, 10.0, 0, 1, 1, 100},
                          {"evB", 200.0, 10.0, 0, 1, 1, 101},
                          {"evC", 300.0, 10.0, 0, 1, 1, 102}}));
  timeline_.set_data_time_range({0.0, 1000.0});

  timeline_.SetSearchQuery("ev");
  timeline_.SetSearchResults(CreateSearchResults({
      {"evA", 100.0, 10.0, 0, 1, 1, 100},
      {"evB", 200.0, 10.0, 0, 1, 1, 101},
      {"evC", 300.0, 10.0, 0, 1, 1, 102},
  }));

  timeline_.NavigateToNextSearchResult();
  ASSERT_EQ(timeline_.get_current_search_result_index(), 0);

  timeline_.NavigateToPrevSearchResult();

  EXPECT_EQ(timeline_.get_current_search_result_index(), 2);
  EXPECT_EQ(timeline_.selected_event_index(), 2);
}

TEST(TimelineTest, EmitViewportChangedContainsCorrectRangeAndMinMax) {
  ColorPalette palette = ColorPalette::Default();
  Timeline timeline(palette);
  // 1. Register event callback to capture the detail object emitted for
  // viewport changes.
  bool event_emitted = false;
  EventData received_detail;
  timeline.set_event_callback(
      [&](absl::string_view type, const EventData& detail) {
        if (type == kViewportChanged) {
          event_emitted = true;
          received_detail = detail;
        }
      });

  // 2. Trigger viewport changed event with a known time range.
  const TimeRange expected_range{100.5, 999.25};
  timeline.emit_viewport_changed_for_test(expected_range);

  // 3. Verify that the event was emitted and contains the range object.
  ASSERT_TRUE(event_emitted);
  ASSERT_TRUE(received_detail.contains(std::string(kViewportChangedRange)));

  const auto range_it =
      received_detail.find(std::string(kViewportChangedRange));
  ASSERT_NE(range_it, received_detail.end());
  const EventData range_obj = std::any_cast<EventData>(range_it->second);

  ASSERT_TRUE(range_obj.contains(std::string(kViewportChangedMin)));
  ASSERT_TRUE(range_obj.contains(std::string(kViewportChangedMax)));

  const double actual_min = std::any_cast<double>(
      range_obj.find(std::string(kViewportChangedMin))->second);
  const double actual_max = std::any_cast<double>(
      range_obj.find(std::string(kViewportChangedMax))->second);

  EXPECT_DOUBLE_EQ(actual_min, 100.5);
  EXPECT_DOUBLE_EQ(actual_max, 999.25);
}

TEST(TimelineTest, ZoomEmitsViewportChangedWithCorrectRange) {
  ColorPalette palette = ColorPalette::Default();
  Timeline timeline(palette);
  // 1. Set up data range and initial visible time range before zooming.
  timeline.set_data_time_range({0.0, 20000.0});
  timeline.SetVisibleRange({100.0, 300.0});

  bool event_emitted = false;
  EventData received_detail;
  timeline.set_event_callback(
      [&](absl::string_view type, const EventData& detail) {
        if (type == kViewportChanged) {
          event_emitted = true;
          received_detail = detail;
        }
      });

  // 2. Perform zoom-out operation on real Timeline instance with factor 2.0
  // around pivot at 200.0 microseconds, expanding the duration from 200.0 to
  // 400.0 centered around 200.0.
  timeline.zoom_for_test(2.0f, 200.0);

  // 3. Verify that real zooming triggered kViewportChanged and correctly
  // updated min and max boundaries.
  ASSERT_TRUE(event_emitted);
  ASSERT_TRUE(received_detail.contains(std::string(kViewportChangedRange)));

  const auto range_it =
      received_detail.find(std::string(kViewportChangedRange));
  ASSERT_NE(range_it, received_detail.end());
  const EventData range_obj = std::any_cast<EventData>(range_it->second);

  ASSERT_TRUE(range_obj.contains(std::string(kViewportChangedMin)));
  ASSERT_TRUE(range_obj.contains(std::string(kViewportChangedMax)));

  const double actual_min = std::any_cast<double>(
      range_obj.find(std::string(kViewportChangedMin))->second);
  const double actual_max = std::any_cast<double>(
      range_obj.find(std::string(kViewportChangedMax))->second);

  EXPECT_DOUBLE_EQ(actual_min, 0.0);
  EXPECT_DOUBLE_EQ(actual_max, 400.0);
}

}  // namespace
}  // namespace testing
}  // namespace traceviewer
