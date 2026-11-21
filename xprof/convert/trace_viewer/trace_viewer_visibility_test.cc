/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include "xprof/convert/trace_viewer/trace_viewer_visibility.h"

#include <cstdint>
#include <limits>
#include <optional>

#include <gtest/gtest.h>
#include "xla/tsl/profiler/utils/timespan.h"
#include "xprof/convert/trace_viewer/trace_options.h"
#include "plugin/xprof/protobuf/trace_events.pb.h"

namespace tensorflow {
namespace profiler {
namespace {

using tsl::profiler::Timespan;

constexpr uint32_t kDeviceId = 10;
constexpr uint32_t kResourceId = 1;
constexpr uint32_t kSrcResourceId = 2;
constexpr uint32_t kDstResourceId = 4;

TraceEvent Complete(Timespan span, uint32_t resource_id = kResourceId) {
  TraceEvent event;
  event.set_device_id(kDeviceId);
  event.set_resource_id(resource_id);
  event.set_timestamp_ps(span.begin_ps());
  event.set_duration_ps(span.duration_ps());
  return event;
}

TraceEvent Counter(uint64_t time_ps) {
  TraceEvent event;
  event.set_device_id(kDeviceId);
  event.set_timestamp_ps(time_ps);
  return event;
}

TraceEvent Flow(Timespan span, uint64_t flow_id, uint32_t resource_id) {
  TraceEvent event;
  event.set_flow_id(flow_id);
  event.set_device_id(kDeviceId);
  event.set_resource_id(resource_id);
  event.set_timestamp_ps(span.begin_ps());
  event.set_duration_ps(span.duration_ps());
  return event;
}

TEST(TraceViewerVisibilityTest, VisibilityNoDownsampling) {
  TraceViewerVisibility v(Timespan(1000, 1000));

  // Instant events.
  EXPECT_FALSE(v.Visible(Complete(Timespan(999))));
  EXPECT_TRUE(v.Visible(Complete(Timespan(1000))));
  EXPECT_TRUE(v.Visible(Complete(Timespan(1500))));
  EXPECT_TRUE(v.Visible(Complete(Timespan(2000))));
  EXPECT_FALSE(v.Visible(Complete(Timespan(2001))));

  // Complete events.
  EXPECT_FALSE(v.Visible(Complete(Timespan(900, 99))));
  EXPECT_TRUE(v.Visible(Complete(Timespan(900, 100))));
  EXPECT_TRUE(v.Visible(Complete(Timespan(1450, 100))));
  EXPECT_TRUE(v.Visible(Complete(Timespan(2000, 50))));
  EXPECT_FALSE(v.Visible(Complete(Timespan(2001, 50))));
}

// TODO(b/218368708): Counter events are currently always visible.
TEST(TraceViewerVisibilityTest, DISABLED_CounterEventsDownsampling) {
  TraceViewerVisibility v(Timespan(1000, 1000), 100);

  // A counter event within the visible span is visible if its distance from the
  // previous event is >= resolution_ps.
  EXPECT_FALSE(v.Visible(Counter(999)));
  EXPECT_TRUE(v.Visible(Counter(1000)));
  EXPECT_FALSE(v.Visible(Counter(1099)));
  EXPECT_TRUE(v.Visible(Counter(1100)));
  EXPECT_TRUE(v.Visible(Counter(2000)));
  EXPECT_FALSE(v.Visible(Counter(2001)));
}

TEST(TraceViewerVisibilityTest, CompleteEventsDownsampling) {
  TraceViewerVisibility v(Timespan(1000, 1000), 100);

  // First event is always visible.
  EXPECT_TRUE(v.Visible(Complete(Timespan(950, 50))));
  // Next visible event must have duration_ps >= resolution_ps or its distance
  // from the previous event must be >= resolution_ps.
  EXPECT_FALSE(v.Visible(Complete(Timespan(1050, 50))));
  EXPECT_TRUE(v.Visible(Complete(Timespan(1055, 200))));
  EXPECT_TRUE(v.Visible(Complete(Timespan(1355, 50))));
}

TEST(TraceViewerVisibilityTest, CompleteNestedEventsDownsampling) {
  TraceViewerVisibility v(Timespan(1000, 1000), 100);

  // First event is always visible.
  EXPECT_TRUE(v.Visible(Complete(Timespan(1000, 200))));
  // Nested events are visible when increasing depth.
  EXPECT_TRUE(v.Visible(Complete(Timespan(1200, 190))));
  EXPECT_TRUE(v.Visible(Complete(Timespan(1250, 20))));
  // Next visible event must have duration_ps >= resolution_ps or its distance
  // from the previous event must be >= resolution_ps.
  EXPECT_FALSE(v.Visible(Complete(Timespan(1270, 20))));
  EXPECT_TRUE(v.Visible(Complete(Timespan(1290, 100))));
}

TEST(TraceViewerVisibilityTest, FlowEventsDownsampling) {
  TraceViewerVisibility v(Timespan(1000, 1000), 100);

  // First event in the flow determines the visibility of the full flow.
  // First flow event in a row is always visible.
  EXPECT_TRUE(v.Visible(Flow(Timespan(1000, 50), 1, kSrcResourceId)));
  // Distance between arrow points must be >= resolution_ps.
  EXPECT_FALSE(v.Visible(Flow(Timespan(1050, 50), 2, kSrcResourceId)));
  EXPECT_TRUE(v.Visible(Flow(Timespan(1100, 50), 3, kSrcResourceId)));

  // Other events in the flow have the same visibility as the first event in
  // the flow.
  EXPECT_TRUE(v.Visible(Flow(Timespan(1100, 50), 1, kDstResourceId)));
  EXPECT_FALSE(v.Visible(Flow(Timespan(1200, 52), 2, kDstResourceId)));
  EXPECT_TRUE(v.Visible(Flow(Timespan(1252, 10), 3, kDstResourceId)));

  // Sanity check for first events: complete events with same distance between
  // events, the third event is not visible because the distance between
  // rectangles is not enough, unlike the distance between arrows.
  EXPECT_TRUE(v.Visible(Complete(Timespan(1300, 50))));
  EXPECT_FALSE(v.Visible(Complete(Timespan(1350, 50))));
  EXPECT_FALSE(v.Visible(Complete(Timespan(1400, 50))));

  // Sanity check for other events: if only the distance between arrows is
  // considered, the second flow would be visible and the third would be
  // invisible.
  EXPECT_TRUE(v.Visible(Flow(Timespan(1600, 50), 4, kResourceId)));
  EXPECT_TRUE(v.Visible(Flow(Timespan(1700, 52), 5, kResourceId)));
  EXPECT_FALSE(v.Visible(Flow(Timespan(1752, 10), 6, kResourceId)));
}

TEST(TraceViewerVisibilityTest, TestTraceVisibilityFilter) {
  Trace trace;
  trace.set_min_timestamp_ps(1000);
  trace.set_max_timestamp_ps(2000);

  Device& device = (*trace.mutable_devices())[kDeviceId];
  device.set_device_id(kDeviceId);
  device.set_name("/device:TPU:0");
  TraceOptions options;
  options.full_dma = false;

  TraceVisibilityFilter filter(Timespan(1000, 1000), 100, options);
  filter.SetUp(trace);

  TraceEvent event = Complete(Timespan(1000, 100));
  EXPECT_FALSE(filter.Filter(event));

  TraceEvent event2 = Complete(Timespan(2000, 100));
  EXPECT_FALSE(filter.Filter(event2));

  TraceEvent event3 = Complete(Timespan(900, 50));
  EXPECT_TRUE(filter.Filter(event3));

  TraceEvent event4 = Flow(Timespan(1000, 100), 1, kSrcResourceId);
  event4.set_flow_entry_type(TraceEvent::FLOW_MID);
  EXPECT_TRUE(filter.Filter(event4));

  options.full_dma = true;
  TraceVisibilityFilter filter2(Timespan(1000, 1000), 100, options);
  filter2.SetUp(trace);
  EXPECT_FALSE(filter2.Filter(event4));

  device.set_name("GPU");
  filter2.SetUp(trace);
  EXPECT_FALSE(filter2.Filter(event4));
  EXPECT_TRUE(filter2.Filter(event3));

  device.set_name("#Chip TPU Non-Core HBM");
  filter.SetUp(trace);
  EXPECT_TRUE(filter.Filter(event4));
}

Trace MakeTrace(uint64_t min_timestamp_ps, uint64_t max_timestamp_ps) {
  Trace trace;
  trace.set_min_timestamp_ps(min_timestamp_ps);
  trace.set_max_timestamp_ps(max_timestamp_ps);
  return trace;
}

TEST(TraceViewerVisibilityTest, UnconstrainedWindowMakesAllEventsVisible) {
  for (TraceViewerVisibility visibility :
       {TraceViewerVisibility(TraceViewerVisibility::Window::Unconstrained()),
        TraceViewerVisibility(Timespan(0, 0))}) {
    EXPECT_TRUE(visibility.VisibleWindow().IsUnconstrained());
    EXPECT_FALSE(visibility.VisibleWindow().IsEmpty());
    EXPECT_EQ(visibility.VisibleWindow().BeginPs(), 0);
    EXPECT_EQ(visibility.VisibleWindow().EndPs(), std::nullopt);
    EXPECT_TRUE(visibility.Visible(Complete(Timespan(1000, 100))));
    EXPECT_TRUE(visibility.Visible(Complete(Timespan(1450, 100))));
    EXPECT_TRUE(visibility.Visible(Complete(Timespan(1500, 0))));
    EXPECT_TRUE(visibility.Visible(Complete(Timespan(2000, 100))));
  }
}

TEST(TraceViewerVisibilityTest,
     UnconstrainedWindowAppliesExplicitResolutionDownsampling) {
  TraceViewerVisibility visibility(
      TraceViewerVisibility::Window::Unconstrained(), /*resolution_ps=*/100);

  EXPECT_TRUE(visibility.VisibleWindow().IsUnconstrained());
  EXPECT_EQ(visibility.ResolutionPs(), 100);
  // First short event is always visible regardless of timestamp.
  EXPECT_TRUE(visibility.Visible(Complete(Timespan(100, 50))));
  // Second short event within resolution_ps (160 - 150 < 100) is downsampled.
  EXPECT_FALSE(visibility.Visible(Complete(Timespan(160, 50))));
  // Short event >= resolution_ps past previous visible end (250 - 150 == 100).
  EXPECT_TRUE(visibility.Visible(Complete(Timespan(250, 50))));
  // Long event (duration >= resolution_ps) is visible even immediately after.
  EXPECT_TRUE(visibility.Visible(Complete(Timespan(300, 120))));
}

TEST(TraceViewerVisibilityTest,
     ConstrainedInstantSpanFiltersNonOverlappingEvents) {
  TraceViewerVisibility visibility(
      TraceViewerVisibility::Window::FromEndPoints(1500, 1500),
      /*resolution_ps=*/0);

  EXPECT_FALSE(visibility.VisibleWindow().IsUnconstrained());
  EXPECT_EQ(visibility.VisibleWindow().BeginPs(), 1500);
  EXPECT_EQ(visibility.VisibleWindow().EndPs(), 1500);
  EXPECT_FALSE(visibility.Visible(Complete(Timespan(1000, 100))));
  EXPECT_TRUE(visibility.Visible(Complete(Timespan(1450, 100))));
  EXPECT_TRUE(visibility.Visible(Complete(Timespan(1500, 0))));
  EXPECT_FALSE(visibility.Visible(Complete(Timespan(2000, 100))));
}

TEST(TraceViewerVisibilityTest,
     OpenEndedInstantSpanFiltersOnlyEventsEndingBeforeStart) {
  for (TraceViewerVisibility visibility :
       {TraceViewerVisibility(
            TraceViewerVisibility::Window::FromEndPoints(1500, std::nullopt),
            /*resolution_ps=*/0),
        TraceViewerVisibility(Timespan(1500, 0), /*resolution_ps=*/0)}) {
    EXPECT_FALSE(visibility.VisibleWindow().IsUnconstrained());
    EXPECT_FALSE(visibility.VisibleWindow().IsEmpty());
    EXPECT_EQ(visibility.VisibleWindow().BeginPs(), 1500);
    EXPECT_EQ(visibility.VisibleWindow().EndPs(), std::nullopt);
    EXPECT_FALSE(visibility.Visible(Complete(Timespan(1000, 100))));
    EXPECT_TRUE(visibility.Visible(Complete(Timespan(1450, 100))));
    EXPECT_TRUE(visibility.Visible(Complete(Timespan(1500, 0))));
    EXPECT_TRUE(visibility.Visible(Complete(Timespan(2000, 100))));
  }
}

TEST(TraceViewerVisibilityTest,
     DefaultOpenEndedStartTimeClampsEndTimeToTraceMax) {
  Trace trace = MakeTrace(1000, 2000);
  TraceVisibilityFilter filter(Timespan(1500, 0), 100, TraceOptions());

  filter.SetUp(trace);

  EXPECT_FALSE(filter.VisibleWindow().IsEmpty());
  EXPECT_EQ(filter.VisibleWindow().BeginPs(), 1500);
  EXPECT_EQ(filter.VisibleWindow().EndPs(), 2000);
}

TEST(TraceViewerVisibilityTest, OpenEndedStartTimeClampsEndTimeToTraceMax) {
  Trace trace = MakeTrace(1000, 2000);
  TraceVisibilityFilter filter(
      TraceViewerVisibility::Window::FromEndPoints(1500, std::nullopt), 100);

  filter.SetUp(trace);

  EXPECT_FALSE(filter.VisibleWindow().IsEmpty());
  EXPECT_EQ(filter.VisibleWindow().BeginPs(), 1500);
  EXPECT_EQ(filter.VisibleWindow().EndPs(), 2000);
}

TEST(TraceViewerVisibilityTest,
     OpenEndedStartTimeFiltersCorrectlyBeforeSetUpAndWithoutTraceMax) {
  // Filtering before SetUp() treats [1500, +inf) as open-ended.
  TraceVisibilityFilter pre_setup(
      TraceViewerVisibility::Window::FromEndPoints(1500, std::nullopt), 0.0);
  EXPECT_FALSE(pre_setup.VisibleWindow().IsEmpty());
  EXPECT_FALSE(pre_setup.VisibleWindow().IsUnconstrained());
  EXPECT_TRUE(pre_setup.Filter(Complete(Timespan(1000, 100))));
  EXPECT_FALSE(pre_setup.Filter(Complete(Timespan(1450, 100))));
  EXPECT_FALSE(pre_setup.Filter(Complete(Timespan(1500, 0))));
  EXPECT_FALSE(pre_setup.Filter(Complete(Timespan(2000, 100))));

  // Filtering after SetUp(Trace()) with no min/max timestamps remains
  // open-ended and keeps events starting after start_ps.
  TraceVisibilityFilter no_bounds(
      TraceViewerVisibility::Window::FromEndPoints(1500, std::nullopt), 0.0);
  no_bounds.SetUp(Trace());
  EXPECT_FALSE(no_bounds.VisibleWindow().IsEmpty());
  EXPECT_FALSE(no_bounds.VisibleWindow().IsUnconstrained());
  EXPECT_TRUE(no_bounds.Filter(Complete(Timespan(1000, 100))));
  EXPECT_FALSE(no_bounds.Filter(Complete(Timespan(1450, 100))));
  EXPECT_FALSE(no_bounds.Filter(Complete(Timespan(1500, 0))));
  EXPECT_FALSE(no_bounds.Filter(Complete(Timespan(2000, 100))));

  // SetUp() with only min_timestamp_ps clamps the lower bound without marking
  // the open-ended filter empty.
  Trace min_only_trace;
  min_only_trace.set_min_timestamp_ps(1200);
  TraceVisibilityFilter min_only(
      TraceViewerVisibility::Window::FromEndPoints(0, std::nullopt), 0.0);
  min_only.SetUp(min_only_trace);
  EXPECT_FALSE(min_only.VisibleWindow().IsEmpty());
  EXPECT_FALSE(min_only.VisibleWindow().IsUnconstrained());
  EXPECT_EQ(min_only.VisibleWindow().BeginPs(), 1200);
  EXPECT_EQ(min_only.VisibleWindow().EndPs(), std::nullopt);
  EXPECT_TRUE(min_only.Filter(Complete(Timespan(1000, 100))));
  EXPECT_FALSE(min_only.Filter(Complete(Timespan(1500, 100))));
}

TEST(TraceViewerVisibilityTest,
     ClampToTraceWithOnlyMaxTimestampClampsUpperBound) {
  Trace max_only_trace;
  max_only_trace.set_max_timestamp_ps(2000);

  // Unconstrained clamps to [0, 2000].
  {
    TraceVisibilityFilter filter(TraceViewerVisibility::Window::Unconstrained(),
                                 0.0);
    filter.SetUp(max_only_trace);
    EXPECT_FALSE(filter.VisibleWindow().IsEmpty());
    EXPECT_FALSE(filter.VisibleWindow().IsUnconstrained());
    EXPECT_EQ(filter.VisibleWindow().BeginPs(), 0);
    EXPECT_EQ(filter.VisibleWindow().EndPs(), 2000);
    EXPECT_FALSE(filter.Filter(Complete(Timespan(0, 100))));
    EXPECT_FALSE(filter.Filter(Complete(Timespan(2000, 100))));
    EXPECT_TRUE(filter.Filter(Complete(Timespan(2100, 100))));
  }

  // Open-ended start <= max clamps to [1500, 2000].
  {
    TraceVisibilityFilter filter(
        TraceViewerVisibility::Window::FromEndPoints(1500, std::nullopt), 0.0);
    filter.SetUp(max_only_trace);
    EXPECT_FALSE(filter.VisibleWindow().IsEmpty());
    EXPECT_EQ(filter.VisibleWindow().BeginPs(), 1500);
    EXPECT_EQ(filter.VisibleWindow().EndPs(), 2000);
    EXPECT_TRUE(filter.Filter(Complete(Timespan(1000, 100))));
    EXPECT_FALSE(filter.Filter(Complete(Timespan(1800, 100))));
    EXPECT_TRUE(filter.Filter(Complete(Timespan(2100, 100))));
  }

  // Open-ended start > max becomes Empty.
  {
    TraceVisibilityFilter filter(
        TraceViewerVisibility::Window::FromEndPoints(2500, std::nullopt), 0.0);
    filter.SetUp(max_only_trace);
    EXPECT_TRUE(filter.VisibleWindow().IsEmpty());
    EXPECT_EQ(filter.VisibleWindow().BeginPs(), 0);
    EXPECT_EQ(filter.VisibleWindow().EndPs(), std::nullopt);
    EXPECT_TRUE(filter.Filter(Complete(Timespan(1500, 100))));
  }

  // Bounded inside [0, 2000] remains unchanged.
  {
    TraceVisibilityFilter filter(
        TraceViewerVisibility::Window::FromEndPoints(500, 1500), 0.0);
    filter.SetUp(max_only_trace);
    EXPECT_FALSE(filter.VisibleWindow().IsEmpty());
    EXPECT_EQ(filter.VisibleWindow().BeginPs(), 500);
    EXPECT_EQ(filter.VisibleWindow().EndPs(), 1500);
    EXPECT_FALSE(filter.Filter(Complete(Timespan(1000, 100))));
    EXPECT_TRUE(filter.Filter(Complete(Timespan(1600, 100))));
  }

  // Bounded with end > max clamps upper bound to 2000.
  {
    TraceVisibilityFilter filter(
        TraceViewerVisibility::Window::FromEndPoints(500, 2500), 0.0);
    filter.SetUp(max_only_trace);
    EXPECT_FALSE(filter.VisibleWindow().IsEmpty());
    EXPECT_EQ(filter.VisibleWindow().BeginPs(), 500);
    EXPECT_EQ(filter.VisibleWindow().EndPs(), 2000);
    EXPECT_FALSE(filter.Filter(Complete(Timespan(1900, 100))));
    EXPECT_TRUE(filter.Filter(Complete(Timespan(2100, 100))));
  }

  // Bounded with start > max becomes Empty.
  {
    TraceVisibilityFilter filter(
        TraceViewerVisibility::Window::FromEndPoints(2500, 3000), 0.0);
    filter.SetUp(max_only_trace);
    EXPECT_TRUE(filter.VisibleWindow().IsEmpty());
    EXPECT_EQ(filter.VisibleWindow().BeginPs(), 0);
    EXPECT_EQ(filter.VisibleWindow().EndPs(), std::nullopt);
    EXPECT_TRUE(filter.Filter(Complete(Timespan(1500, 100))));
  }
}

TEST(TraceViewerVisibilityTest,
     OpenEndedStartTimeAtTraceMaxFiltersEarlierEvents) {
  Trace trace = MakeTrace(1000, 2000);
  TraceVisibilityFilter filter(
      TraceViewerVisibility::Window::FromEndPoints(2000, std::nullopt), 100);

  filter.SetUp(trace);

  EXPECT_FALSE(filter.VisibleWindow().IsEmpty());
  EXPECT_EQ(filter.VisibleWindow().BeginPs(), 2000);
  EXPECT_EQ(filter.VisibleWindow().EndPs(), 2000);
  EXPECT_TRUE(filter.Filter(Complete(Timespan(1000, 100))));
  EXPECT_FALSE(filter.Filter(Complete(Timespan(2000, 100))));
  EXPECT_TRUE(filter.Filter(Complete(Timespan(2100, 100))));
}

TEST(TraceViewerVisibilityTest,
     OpenEndedStartTimePastTraceMaxFiltersAllEvents) {
  Trace trace = MakeTrace(1000, 2000);
  TraceVisibilityFilter filter(
      TraceViewerVisibility::Window::FromEndPoints(2500, std::nullopt), 100);

  filter.SetUp(trace);

  EXPECT_TRUE(filter.VisibleWindow().IsEmpty());
  EXPECT_EQ(filter.VisibleWindow().BeginPs(), 0);
  EXPECT_EQ(filter.VisibleWindow().EndPs(), std::nullopt);
  EXPECT_TRUE(filter.Filter(Complete(Timespan(1000, 100))));
  EXPECT_TRUE(filter.Filter(Complete(Timespan(2000, 100))));
  EXPECT_TRUE(filter.Filter(Complete(Timespan(2500, 0))));
}

TEST(TraceViewerVisibilityTest,
     ExplicitInstantSpanFiltersNonOverlappingEvents) {
  Trace trace = MakeTrace(1000, 2000);
  TraceVisibilityFilter filter(
      TraceViewerVisibility::Window::FromEndPoints(1500, 1500), 100);

  filter.SetUp(trace);

  EXPECT_FALSE(filter.VisibleWindow().IsEmpty());
  EXPECT_EQ(filter.VisibleWindow().BeginPs(), 1500);
  EXPECT_EQ(filter.VisibleWindow().EndPs(), 1500);
  EXPECT_TRUE(filter.Filter(Complete(Timespan(1000, 100))));
  EXPECT_TRUE(filter.Filter(Complete(Timespan(2000, 100))));
  EXPECT_FALSE(filter.Filter(Complete(Timespan(1450, 100))));
  EXPECT_FALSE(filter.Filter(Complete(Timespan(1500, 0))));
}

TEST(TraceViewerVisibilityTest,
     ExplicitZeroEndTimeAtOriginFiltersEventsAfterZero) {
  Trace trace = MakeTrace(0, 2000);
  TraceVisibilityFilter filter(
      TraceViewerVisibility::Window::FromEndPoints(0, 0), 100);

  filter.SetUp(trace);

  EXPECT_FALSE(filter.VisibleWindow().IsEmpty());
  EXPECT_FALSE(filter.VisibleWindow().IsUnconstrained());
  EXPECT_EQ(filter.VisibleWindow().BeginPs(), 0);
  EXPECT_EQ(filter.VisibleWindow().EndPs(), 0);
  EXPECT_FALSE(filter.Filter(Complete(Timespan(0, 0))));
  EXPECT_FALSE(filter.Filter(Complete(Timespan(0, 100))));
  EXPECT_TRUE(filter.Filter(Complete(Timespan(100, 100))));
}

TEST(TraceViewerVisibilityTest,
     ExplicitZeroEndTimeBeforeTraceMinFiltersAllEvents) {
  Trace trace = MakeTrace(1000, 2000);
  TraceVisibilityFilter filter(
      TraceViewerVisibility::Window::FromEndPoints(0, 0), 100);

  filter.SetUp(trace);

  EXPECT_TRUE(filter.VisibleWindow().IsEmpty());
  EXPECT_EQ(filter.VisibleWindow().BeginPs(), 0);
  EXPECT_EQ(filter.VisibleWindow().EndPs(), std::nullopt);
  EXPECT_TRUE(filter.Filter(Complete(Timespan(1200, 100))));
}

TEST(TraceViewerVisibilityTest, DefaultEmptySpanExpandsToTraceBounds) {
  Trace trace = MakeTrace(1000, 2000);
  TraceVisibilityFilter filter(
      TraceViewerVisibility::Window::FromEndPoints(0, std::nullopt), 100);

  filter.SetUp(trace);

  EXPECT_FALSE(filter.VisibleWindow().IsEmpty());
  EXPECT_EQ(filter.VisibleWindow().BeginPs(), 1000);
  EXPECT_EQ(filter.VisibleWindow().EndPs(), 2000);
  EXPECT_FALSE(filter.Filter(Complete(Timespan(1200, 100))));
}

TEST(TraceViewerVisibilityTest, SpanBeforeTraceMinFiltersAllEvents) {
  Trace trace = MakeTrace(1000, 2000);
  TraceVisibilityFilter filter(
      TraceViewerVisibility::Window::FromEndPoints(500, 800), 100);

  filter.SetUp(trace);

  EXPECT_TRUE(filter.VisibleWindow().IsEmpty());
  EXPECT_EQ(filter.VisibleWindow().BeginPs(), 0);
  EXPECT_EQ(filter.VisibleWindow().EndPs(), std::nullopt);
  EXPECT_TRUE(filter.Filter(Complete(Timespan(1000, 100))));
  EXPECT_TRUE(filter.Filter(Complete(Timespan(2000, 100))));
}

TEST(TraceViewerVisibilityTest,
     EmptyAndInvertedFactoriesNormalizeAndFilterAllEvents) {
  Trace trace = MakeTrace(1000, 2000);
  TraceVisibilityFilter empty_filter(TraceViewerVisibility::Window::Empty(),
                                     100);

  EXPECT_TRUE(empty_filter.VisibleWindow().IsEmpty());
  EXPECT_FALSE(empty_filter.VisibleWindow().IsUnconstrained());
  EXPECT_EQ(empty_filter.VisibleWindow().BeginPs(), 0);
  EXPECT_EQ(empty_filter.VisibleWindow().EndPs(), std::nullopt);
  EXPECT_FALSE(empty_filter.VisibleWindow().Overlaps(Timespan(1000, 100)));

  empty_filter.SetUp(trace);

  EXPECT_TRUE(empty_filter.VisibleWindow().IsEmpty());
  EXPECT_FALSE(empty_filter.VisibleWindow().IsUnconstrained());
  EXPECT_EQ(empty_filter.VisibleWindow().BeginPs(), 0);
  EXPECT_EQ(empty_filter.VisibleWindow().EndPs(), std::nullopt);
  EXPECT_TRUE(empty_filter.Filter(Complete(Timespan(1000, 100))));
  EXPECT_TRUE(empty_filter.Filter(Complete(Timespan(2000, 100))));

  TraceVisibilityFilter inverted_filter(
      TraceViewerVisibility::Window::FromEndPoints(1800, 1200), 100);
  EXPECT_TRUE(inverted_filter.VisibleWindow().IsEmpty());
  EXPECT_FALSE(inverted_filter.VisibleWindow().IsUnconstrained());
  EXPECT_EQ(inverted_filter.VisibleWindow().BeginPs(), 0);
  EXPECT_EQ(inverted_filter.VisibleWindow().EndPs(), std::nullopt);
  EXPECT_TRUE(inverted_filter.Filter(Complete(Timespan(1500, 100))));
  inverted_filter.SetUp(trace);
  EXPECT_TRUE(inverted_filter.VisibleWindow().IsEmpty());
  EXPECT_TRUE(inverted_filter.Filter(Complete(Timespan(1500, 100))));
}

TEST(TraceViewerVisibilityTest,
     VisibleWindowReflectsUnconstrainedAndEmptyStates) {
  // TraceViewerVisibility directly.
  EXPECT_TRUE(
      TraceViewerVisibility(Timespan(0, 0)).VisibleWindow().IsUnconstrained());
  EXPECT_TRUE(
      TraceViewerVisibility(TraceViewerVisibility::Window::Unconstrained())
          .VisibleWindow()
          .IsUnconstrained());
  EXPECT_FALSE(TraceViewerVisibility(Timespan(1500, 0))
                   .VisibleWindow()
                   .IsUnconstrained());
  EXPECT_FALSE(TraceViewerVisibility(
                   TraceViewerVisibility::Window::FromEndPoints(1500, 1500),
                   /*resolution_ps=*/0)
                   .VisibleWindow()
                   .IsUnconstrained());
  EXPECT_FALSE(
      TraceViewerVisibility(
          TraceViewerVisibility::Window::FromEndPoints(1500, std::nullopt),
          /*resolution_ps=*/0)
          .VisibleWindow()
          .IsUnconstrained());
  EXPECT_FALSE(TraceViewerVisibility(Timespan(1000, 1000))
                   .VisibleWindow()
                   .IsUnconstrained());

  // Default unconstrained TraceVisibilityFilter before and after SetUp.
  TraceVisibilityFilter default_filter(Timespan(0, 0), 0.0, TraceOptions());
  EXPECT_FALSE(default_filter.VisibleWindow().IsEmpty());
  EXPECT_TRUE(default_filter.VisibleWindow().IsUnconstrained());
  default_filter.SetUp(Trace());
  EXPECT_FALSE(default_filter.VisibleWindow().IsEmpty());
  EXPECT_TRUE(default_filter.VisibleWindow().IsUnconstrained());
  default_filter.SetUp(MakeTrace(1000, 2000));
  EXPECT_FALSE(default_filter.VisibleWindow().IsEmpty());
  EXPECT_FALSE(default_filter.VisibleWindow().IsUnconstrained());

  // Explicit exact-instant filter is never unconstrained.
  TraceVisibilityFilter exact_instant_filter(
      TraceViewerVisibility::Window::FromEndPoints(1500, 1500), 0.0);
  EXPECT_FALSE(exact_instant_filter.VisibleWindow().IsEmpty());
  EXPECT_FALSE(exact_instant_filter.VisibleWindow().IsUnconstrained());
  exact_instant_filter.SetUp(MakeTrace(1000, 2000));
  EXPECT_FALSE(exact_instant_filter.VisibleWindow().IsEmpty());
  EXPECT_FALSE(exact_instant_filter.VisibleWindow().IsUnconstrained());

  // Empty filter is never unconstrained.
  TraceVisibilityFilter explicit_empty_filter(
      TraceViewerVisibility::Window::Empty(), 0.0);
  EXPECT_TRUE(explicit_empty_filter.VisibleWindow().IsEmpty());
  EXPECT_FALSE(explicit_empty_filter.VisibleWindow().IsUnconstrained());
  explicit_empty_filter.SetUp(MakeTrace(1000, 2000));
  EXPECT_TRUE(explicit_empty_filter.VisibleWindow().IsEmpty());
  EXPECT_FALSE(explicit_empty_filter.VisibleWindow().IsUnconstrained());

  // Post-SetUp out-of-range empty window flips IsEmpty() to true and keeps
  // IsUnconstrained() false with normalized bounds.
  TraceVisibilityFilter past_end_filter(
      TraceViewerVisibility::Window::FromEndPoints(2500, std::nullopt), 0.0);
  EXPECT_FALSE(past_end_filter.VisibleWindow().IsEmpty());
  EXPECT_FALSE(past_end_filter.VisibleWindow().IsUnconstrained());
  past_end_filter.SetUp(MakeTrace(1000, 2000));
  EXPECT_TRUE(past_end_filter.VisibleWindow().IsEmpty());
  EXPECT_FALSE(past_end_filter.VisibleWindow().IsUnconstrained());
  EXPECT_EQ(past_end_filter.VisibleWindow().BeginPs(), 0);
  EXPECT_EQ(past_end_filter.VisibleWindow().EndPs(), std::nullopt);

  TraceVisibilityFilter before_start_filter(
      TraceViewerVisibility::Window::FromEndPoints(500, 800), 0.0);
  EXPECT_FALSE(before_start_filter.VisibleWindow().IsEmpty());
  before_start_filter.SetUp(MakeTrace(1000, 2000));
  EXPECT_TRUE(before_start_filter.VisibleWindow().IsEmpty());
  EXPECT_FALSE(before_start_filter.VisibleWindow().IsUnconstrained());
}

TEST(TraceViewerVisibilityTest,
     UpdateVisibilityPreservesWindowPolicyAcrossSpanKinds) {
  // Post-SetUp exact-instant at 1500.
  {
    TraceVisibilityFilter filter(
        TraceViewerVisibility::Window::FromEndPoints(1500, 1500), 100.0);
    filter.SetUp(MakeTrace(1000, 2000));
    for (double resolution : {0.0, 100.0}) {
      SCOPED_TRACE(testing::Message() << "resolution=" << resolution);
      filter.UpdateVisibility(resolution);
      EXPECT_FALSE(filter.VisibleWindow().IsEmpty());
      EXPECT_FALSE(filter.VisibleWindow().IsUnconstrained());
      EXPECT_TRUE(filter.Filter(Complete(Timespan(1000, 100))));
      EXPECT_FALSE(filter.Filter(Complete(Timespan(1450, 100))));
      EXPECT_FALSE(filter.Filter(Complete(Timespan(1500, 0))));
      EXPECT_TRUE(filter.Filter(Complete(Timespan(2000, 100))));
    }
  }

  // Post-SetUp exact-instant at explicit 0 end time.
  {
    TraceVisibilityFilter filter(
        TraceViewerVisibility::Window::FromEndPoints(0, 0), 100.0);
    filter.SetUp(MakeTrace(0, 2000));
    for (double resolution : {0.0, 100.0}) {
      SCOPED_TRACE(testing::Message() << "resolution=" << resolution);
      filter.UpdateVisibility(resolution);
      EXPECT_FALSE(filter.VisibleWindow().IsEmpty());
      EXPECT_FALSE(filter.VisibleWindow().IsUnconstrained());
      EXPECT_FALSE(filter.Filter(Complete(Timespan(0, 0))));
      EXPECT_FALSE(filter.Filter(Complete(Timespan(0, 100))));
      EXPECT_TRUE(filter.Filter(Complete(Timespan(100, 100))));
    }
  }

  // Post-SetUp open-ended without trace bounds keeps open-ended semantics.
  {
    TraceVisibilityFilter filter(
        TraceViewerVisibility::Window::FromEndPoints(1500, std::nullopt),
        100.0);
    filter.SetUp(Trace());
    for (double resolution : {0.0, 100.0}) {
      SCOPED_TRACE(testing::Message() << "resolution=" << resolution);
      filter.UpdateVisibility(resolution);
      EXPECT_FALSE(filter.VisibleWindow().IsEmpty());
      EXPECT_FALSE(filter.VisibleWindow().IsUnconstrained());
      EXPECT_TRUE(filter.Filter(Complete(Timespan(1000, 100))));
      EXPECT_FALSE(filter.Filter(Complete(Timespan(1450, 100))));
      EXPECT_FALSE(filter.Filter(Complete(Timespan(2000, 100))));
    }
  }

  // Post-SetUp open-ended clamped to exact instant at trace max keeps exact
  // instant semantics.
  {
    TraceVisibilityFilter filter(
        TraceViewerVisibility::Window::FromEndPoints(2000, std::nullopt),
        100.0);
    filter.SetUp(MakeTrace(1000, 2000));
    for (double resolution : {0.0, 100.0}) {
      SCOPED_TRACE(testing::Message() << "resolution=" << resolution);
      filter.UpdateVisibility(resolution);
      EXPECT_FALSE(filter.VisibleWindow().IsEmpty());
      EXPECT_FALSE(filter.VisibleWindow().IsUnconstrained());
      EXPECT_TRUE(filter.Filter(Complete(Timespan(1000, 100))));
      EXPECT_FALSE(filter.Filter(Complete(Timespan(2000, 100))));
      EXPECT_TRUE(filter.Filter(Complete(Timespan(2100, 100))));
    }
  }

  // Post-SetUp empty window stays empty across UpdateVisibility calls.
  {
    TraceVisibilityFilter filter(
        TraceViewerVisibility::Window::FromEndPoints(2500, std::nullopt),
        100.0);
    filter.SetUp(MakeTrace(1000, 2000));
    for (double resolution : {0.0, 100.0}) {
      SCOPED_TRACE(testing::Message() << "resolution=" << resolution);
      filter.UpdateVisibility(resolution);
      EXPECT_TRUE(filter.VisibleWindow().IsEmpty());
      EXPECT_FALSE(filter.VisibleWindow().IsUnconstrained());
      EXPECT_TRUE(filter.Filter(Complete(Timespan(0, 100))));
      EXPECT_TRUE(filter.Filter(Complete(Timespan(1500, 100))));
    }
  }

  // Positive-width clamped window updates ResolutionPs() against clamped width.
  {
    TraceVisibilityFilter filter(
        TraceViewerVisibility::Window::FromEndPoints(1000, 3000), 100.0);
    filter.SetUp(MakeTrace(1000, 2000));
    EXPECT_EQ(filter.ResolutionPs(), 10);
    filter.UpdateVisibility(0);
    EXPECT_EQ(filter.ResolutionPs(), 0);
    filter.UpdateVisibility(50);
    EXPECT_EQ(filter.ResolutionPs(), 20);
    EXPECT_EQ(filter.VisibleWindow().EndPs(), 2000);
  }
}

TEST(TraceViewerVisibilityTest,
     ResolutionPsHandlesLargeDurationsAndInvalidResolutionsWithoutOverflow) {
  constexpr uint64_t kTwoTo63 = uint64_t{1} << 63;
  constexpr uint64_t kLargeDurationPs = kTwoTo63 + 2048;

  // Bounded window computes duration from endpoints; unbounded, empty, and
  // exact-instant windows disable downsampling (ResolutionPs == 0).
  EXPECT_EQ(TraceVisibilityFilter(
                TraceViewerVisibility::Window::FromEndPoints(1000, 2000), 100.0)
                .ResolutionPs(),
            10);
  EXPECT_EQ(TraceVisibilityFilter(TraceViewerVisibility::Window::Empty(), 100.0)
                .ResolutionPs(),
            0);
  EXPECT_EQ(TraceVisibilityFilter(
                TraceViewerVisibility::Window::Unconstrained(), 100.0)
                .ResolutionPs(),
            0);
  EXPECT_EQ(TraceVisibilityFilter(TraceViewerVisibility::Window::FromEndPoints(
                                      1500, std::nullopt),
                                  100.0)
                .ResolutionPs(),
            0);
  EXPECT_EQ(TraceVisibilityFilter(
                TraceViewerVisibility::Window::FromEndPoints(1500, 1500), 100.0)
                .ResolutionPs(),
            0);

  // Duration in [2^63, 2^64 - 1) with resolution = 1.0 stays exact in uint64_t.
  TraceVisibilityFilter unit_res_filter(Timespan(0, kLargeDurationPs), 1.0,
                                        TraceOptions());
  EXPECT_EQ(unit_res_filter.ResolutionPs(), kLargeDurationPs);

  // Duration above 2^53 (not exactly representable in double) with
  // resolution = 1.0 preserves exact integer precision.
  constexpr uint64_t kAboveDoubleExactInt = (uint64_t{1} << 53) + 1;
  EXPECT_EQ(TraceVisibilityFilter(Timespan(0, kAboveDoubleExactInt), 1.0,
                                  TraceOptions())
                .ResolutionPs(),
            kAboveDoubleExactInt);

  // Duration in [2^63, 2^64 - 1) with resolution = 0.5 clamps to uint64_t::max.
  TraceVisibilityFilter half_res_filter(Timespan(0, kTwoTo63), 0.5,
                                        TraceOptions());
  EXPECT_EQ(half_res_filter.ResolutionPs(),
            std::numeric_limits<uint64_t>::max());

  // Tiny positive resolution overflowing double to +inf clamps to
  // uint64_t::max.
  TraceVisibilityFilter tiny_res_filter(Timespan(0, kTwoTo63), 1e-308,
                                        TraceOptions());
  EXPECT_EQ(tiny_res_filter.ResolutionPs(),
            std::numeric_limits<uint64_t>::max());

  // uint64_t::max() duration with resolution = 1.0 clamps to uint64_t::max.
  TraceVisibilityFilter max_dur_filter(
      Timespan(0, std::numeric_limits<uint64_t>::max()), 1.0, TraceOptions());
  EXPECT_EQ(max_dur_filter.ResolutionPs(),
            std::numeric_limits<uint64_t>::max());

  // Zero, negative, NaN, and +/-inf resolutions disable downsampling.
  EXPECT_EQ(
      TraceVisibilityFilter(Timespan(0, kLargeDurationPs), 0.0, TraceOptions())
          .ResolutionPs(),
      0);
  EXPECT_EQ(
      TraceVisibilityFilter(Timespan(0, kLargeDurationPs), -1.0, TraceOptions())
          .ResolutionPs(),
      0);
  EXPECT_EQ(TraceVisibilityFilter(Timespan(0, kLargeDurationPs),
                                  std::numeric_limits<double>::quiet_NaN(),
                                  TraceOptions())
                .ResolutionPs(),
            0);
  EXPECT_EQ(TraceVisibilityFilter(Timespan(0, kLargeDurationPs),
                                  std::numeric_limits<double>::infinity(),
                                  TraceOptions())
                .ResolutionPs(),
            0);
  EXPECT_EQ(TraceVisibilityFilter(Timespan(0, kLargeDurationPs),
                                  -std::numeric_limits<double>::infinity(),
                                  TraceOptions())
                .ResolutionPs(),
            0);
}

TEST(TraceViewerVisibilityTest,
     EmptyWindowAndBoundsAccessorsReflectWindowState) {
  // TraceViewerVisibility with Window::Empty() normalizes bounds and
  // resolution.
  TraceViewerVisibility empty_vis(TraceViewerVisibility::Window::Empty(),
                                  /*resolution_ps=*/100);
  EXPECT_TRUE(empty_vis.VisibleWindow().IsEmpty());
  EXPECT_FALSE(empty_vis.VisibleWindow().IsUnconstrained());
  EXPECT_EQ(empty_vis.VisibleWindow().BeginPs(), 0);
  EXPECT_EQ(empty_vis.VisibleWindow().EndPs(), std::nullopt);
  EXPECT_EQ(empty_vis.ResolutionPs(), 0);
  EXPECT_FALSE(empty_vis.VisibleWindow().Overlaps(Timespan(0, 0)));
  EXPECT_FALSE(empty_vis.Visible(Complete(Timespan(0, 0))));

  // Inverted Window::FromEndPoints() normalizes to Window::Empty().
  TraceViewerVisibility::Window inverted =
      TraceViewerVisibility::Window::FromEndPoints(2000, 1000);
  EXPECT_TRUE(inverted.IsEmpty());
  EXPECT_EQ(inverted.BeginPs(), 0);
  EXPECT_EQ(inverted.EndPs(), std::nullopt);
  EXPECT_FALSE(inverted.Overlaps(Timespan(1500, 100)));

  // VisibleWindow() BeginPs() and EndPs() across all window states.
  TraceVisibilityFilter unconstrained(
      TraceViewerVisibility::Window::FromEndPoints(0, std::nullopt), 0.0);
  EXPECT_EQ(unconstrained.VisibleWindow().BeginPs(), 0);
  EXPECT_EQ(unconstrained.VisibleWindow().EndPs(), std::nullopt);
  unconstrained.SetUp(MakeTrace(1000, 2000));
  EXPECT_EQ(unconstrained.VisibleWindow().BeginPs(), 1000);
  EXPECT_EQ(unconstrained.VisibleWindow().EndPs(), 2000);

  TraceVisibilityFilter open_ended(
      TraceViewerVisibility::Window::FromEndPoints(1500, std::nullopt), 0.0);
  EXPECT_EQ(open_ended.VisibleWindow().BeginPs(), 1500);
  EXPECT_EQ(open_ended.VisibleWindow().EndPs(), std::nullopt);
  open_ended.SetUp(Trace());
  EXPECT_EQ(open_ended.VisibleWindow().BeginPs(), 1500);
  EXPECT_EQ(open_ended.VisibleWindow().EndPs(), std::nullopt);

  TraceVisibilityFilter bounded(
      TraceViewerVisibility::Window::FromEndPoints(1000, 1500), 0.0);
  EXPECT_EQ(bounded.VisibleWindow().BeginPs(), 1000);
  EXPECT_EQ(bounded.VisibleWindow().EndPs(), 1500);

  TraceVisibilityFilter empty(TraceViewerVisibility::Window::Empty(), 0.0);
  EXPECT_EQ(empty.VisibleWindow().BeginPs(), 0);
  EXPECT_EQ(empty.VisibleWindow().EndPs(), std::nullopt);
}

TEST(TraceViewerVisibilityTest,
     FullDmaTpuFlowMidRespectsVisibleSpanAndEmptyWindow) {
  Trace trace = MakeTrace(1000, 2000);
  Device& device = (*trace.mutable_devices())[kDeviceId];
  device.set_device_id(kDeviceId);
  device.set_name("/device:TPU:0");

  TraceOptions full_dma_opts;
  full_dma_opts.full_dma = true;

  TraceEvent flow_mid_before = Flow(Timespan(500, 50), 1, kSrcResourceId);
  flow_mid_before.set_flow_entry_type(TraceEvent::FLOW_MID);
  TraceEvent flow_mid_inside = Flow(Timespan(1200, 50), 2, kSrcResourceId);
  flow_mid_inside.set_flow_entry_type(TraceEvent::FLOW_MID);
  TraceEvent flow_mid_after = Flow(Timespan(1800, 50), 3, kSrcResourceId);
  flow_mid_after.set_flow_entry_type(TraceEvent::FLOW_MID);

  // Bounded window [1100, 1500]: full_dma keeps only the overlapping FLOW_MID.
  TraceVisibilityFilter bounded(
      TraceViewerVisibility::Window::FromEndPoints(1100, 1500), 0.0,
      full_dma_opts);
  bounded.SetUp(trace);
  EXPECT_TRUE(bounded.Filter(flow_mid_before));
  EXPECT_FALSE(bounded.Filter(flow_mid_inside));
  EXPECT_TRUE(bounded.Filter(flow_mid_after));

  // Unclamped open-ended [1600, +inf): drops earlier FLOW_MID, keeps later one.
  Trace unclamped_tpu_trace;
  (*unclamped_tpu_trace.mutable_devices())[kDeviceId] = device;
  TraceVisibilityFilter open_ended(
      TraceViewerVisibility::Window::FromEndPoints(1600, std::nullopt), 0.0,
      full_dma_opts);
  open_ended.SetUp(unclamped_tpu_trace);
  EXPECT_TRUE(open_ended.Filter(flow_mid_before));
  EXPECT_TRUE(open_ended.Filter(flow_mid_inside));
  EXPECT_FALSE(open_ended.Filter(flow_mid_after));

  // Empty window filters out all TPU FLOW_MID events even with full_dma = true.
  TraceVisibilityFilter empty(TraceViewerVisibility::Window::Empty(), 0.0,
                              full_dma_opts);
  empty.SetUp(trace);
  EXPECT_TRUE(empty.Filter(flow_mid_inside));
}

TEST(TraceViewerVisibilityTest,
     RepeatedSetUpRecomputesFromConfiguredBoundsAndClearsDevices) {
  // 1. Unconstrained filter called with [1000, 2000] then [100, 500] must not
  // latch min=1000 and turn empty.
  {
    TraceVisibilityFilter filter(Timespan(0, 0), 0.0, TraceOptions());
    filter.SetUp(MakeTrace(1000, 2000));
    EXPECT_FALSE(filter.VisibleWindow().IsEmpty());
    EXPECT_EQ(filter.VisibleWindow().BeginPs(), 1000);
    EXPECT_EQ(filter.VisibleWindow().EndPs(), 2000);

    filter.SetUp(MakeTrace(100, 500));
    EXPECT_FALSE(filter.VisibleWindow().IsEmpty());
    EXPECT_EQ(filter.VisibleWindow().BeginPs(), 100);
    EXPECT_EQ(filter.VisibleWindow().EndPs(), 500);
    EXPECT_FALSE(filter.Filter(Complete(Timespan(200, 50))));
    EXPECT_TRUE(filter.Filter(Complete(Timespan(1200, 50))));
  }

  // 2. Configured [1500, 2500] emptied by a disjoint trace recovers when SetUp
  // is subsequently called with an overlapping trace.
  {
    TraceVisibilityFilter filter(
        TraceViewerVisibility::Window::FromEndPoints(1500, 2500), 0.0);
    filter.SetUp(MakeTrace(100, 500));
    EXPECT_TRUE(filter.VisibleWindow().IsEmpty());
    EXPECT_TRUE(filter.Filter(Complete(Timespan(1600, 50))));

    filter.SetUp(MakeTrace(1000, 3000));
    EXPECT_FALSE(filter.VisibleWindow().IsEmpty());
    EXPECT_EQ(filter.VisibleWindow().BeginPs(), 1500);
    EXPECT_EQ(filter.VisibleWindow().EndPs(), 2500);
    EXPECT_FALSE(filter.Filter(Complete(Timespan(1600, 50))));
  }

  // 3. Reclassifying a device from TPU to GPU on a subsequent SetUp clears
  // stale TPU device sets so FLOW_MID is no longer filtered as TPU DMA.
  {
    TraceOptions opts;
    opts.full_dma = false;
    TraceVisibilityFilter filter(
        TraceViewerVisibility::Window::FromEndPoints(1000, 2000), 0.0, opts);

    Trace tpu_trace = MakeTrace(1000, 2000);
    Device& tpu_dev = (*tpu_trace.mutable_devices())[kDeviceId];
    tpu_dev.set_device_id(kDeviceId);
    tpu_dev.set_name("/device:TPU:0");
    filter.SetUp(tpu_trace);

    TraceEvent flow_mid = Flow(Timespan(1200, 50), 1, kSrcResourceId);
    flow_mid.set_flow_entry_type(TraceEvent::FLOW_MID);
    EXPECT_TRUE(filter.Filter(flow_mid));

    Trace gpu_trace = MakeTrace(1000, 2000);
    Device& gpu_dev = (*gpu_trace.mutable_devices())[kDeviceId];
    gpu_dev.set_device_id(kDeviceId);
    gpu_dev.set_name("GPU");
    filter.SetUp(gpu_trace);
    EXPECT_FALSE(filter.Filter(flow_mid));
  }
}

}  // namespace
}  // namespace profiler
}  // namespace tensorflow
