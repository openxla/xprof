// Tests for the HLO dependency lookup and the arrow routing in
// timeline_hlo_dependencies.cc.

#include "frontend/app/components/trace_viewer_v2/timeline/timeline_hlo_dependencies.h"

#include <any>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "frontend/app/components/trace_viewer_v2/color/colors.h"
#include "frontend/app/components/trace_viewer_v2/event_data.h"
#include "frontend/app/components/trace_viewer_v2/timeline/constants.h"
#include "frontend/app/components/trace_viewer_v2/timeline/timeline.h"
#include "frontend/app/components/trace_viewer_v2/trace_helper/trace_event.h"

namespace traceviewer {
namespace testing {
namespace {

using ::testing::ElementsAre;
using ::testing::IsEmpty;

// Compact multi-event builder specification.
struct EventSpec {
  std::string name = "Event";
  double start = 100.0;
  double dur = 50.0;
  uint16_t level = 0;
  EventId id = 1;
};

// Compactly appends events to timeline data parallel vectors.
inline void AddEvents(FlameChartTimelineData& data,
                      std::initializer_list<EventSpec> specs) {
  for (const EventSpec& s : specs) {
    data.entry_names.push_back(s.name);
    data.entry_start_times.push_back(s.start);
    data.entry_total_times.push_back(s.dur);
    data.entry_levels.push_back(s.level);
    data.entry_event_ids.push_back(s.id);
    data.entry_args.push_back({});
  }
}

// Helpers for concise timeline test data construction and verification.
Group MakeProcessGroup(absl::string_view name, int start_level = 0,
                       int level_count = 1, bool expanded = true,
                       ProcessId pid = 0) {
  return {.type = Group::Type::kFlame,
          .name = std::string(name),
          .start_level = start_level,
          .nesting_level = kProcessNestingLevel,
          .expanded = expanded,
          .level_count = level_count,
          .pid = pid};
}

Group MakeThreadGroup(absl::string_view name, int parent_index,
                      int start_level = 0, int level_count = 1,
                      bool expanded = true, ProcessId pid = 0,
                      ThreadId tid = 0) {
  return {.type = Group::Type::kFlame,
          .name = std::string(name),
          .start_level = start_level,
          .nesting_level = kThreadNestingLevel,
          .expanded = expanded,
          .parent_index = parent_index,
          .level_count = level_count,
          .pid = pid,
          .tid = tid};
}

// Builds data for the HLO dependency tests: an "XLA Ops" track with levels 0
// and 1, followed by an "Other" track with level 2. `events` must be sorted by
// level, then by start time.
FlameChartTimelineData MakeDependencyTestData(
    absl::Span<const EventSpec> events) {
  FlameChartTimelineData data;
  data.groups = {MakeThreadGroup("XLA Ops", /*parent_index=*/-1,
                                 /*start_level=*/0, /*level_count=*/2),
                 MakeThreadGroup("Other", /*parent_index=*/-1,
                                 /*start_level=*/2, /*level_count=*/1)};
  for (const EventSpec& event : events) AddEvents(data, {event});
  data.level_offsets.assign(/*total_levels=*/3 + 1, 0);
  for (const uint16_t level : data.entry_levels) {
    ++data.level_offsets[level + 1];
  }
  std::partial_sum(data.level_offsets.begin(), data.level_offsets.end(),
                   data.level_offsets.begin());
  data.level_event_indices.resize(data.entry_levels.size());
  std::iota(data.level_event_indices.begin(), data.level_event_indices.end(),
            0);
  return data;
}

TEST(TimelineTest, SelectedEventDependenciesResolveToClosestEvents) {
  ColorPalette palette = ColorPalette::Default();
  Timeline timeline(palette);
  timeline.set_hlo_dependency_arrows_enabled(true);
  // level 0:  operand   operand   selected   consumer   consumer
  //                     ^ closest            ^ closest
  timeline.SetTimelineData(MakeDependencyTestData({
      {.name = "operand", .start = 0.0, .dur = 10.0, .level = 0, .id = 1},
      {.name = "operand", .start = 20.0, .dur = 10.0, .level = 0, .id = 2},
      {.name = "selected", .start = 40.0, .dur = 10.0, .level = 0, .id = 3},
      {.name = "consumer", .start = 60.0, .dur = 10.0, .level = 0, .id = 4},
      {.name = "consumer", .start = 80.0, .dur = 10.0, .level = 0, .id = 5},
  }));
  timeline.set_selected_event_index_for_test(2);

  timeline.SetSelectedEventDependencies({"operand"}, {"consumer"});

  EXPECT_THAT(timeline.producer_event_indices(), ElementsAre(1));
  EXPECT_THAT(timeline.consumer_event_indices(), ElementsAre(3));
}

TEST(TimelineTest, SelectedEventDependenciesResolveAcrossLevelsOfTrack) {
  ColorPalette palette = ColorPalette::Default();
  Timeline timeline(palette);
  timeline.set_hlo_dependency_arrows_enabled(true);
  // level 0:  operand             selected             consumer
  // level 1:            operand               consumer
  //                     ^ closest             ^ closest
  timeline.SetTimelineData(MakeDependencyTestData({
      {.name = "operand", .start = 10.0, .dur = 10.0, .level = 0, .id = 1},
      {.name = "selected", .start = 40.0, .dur = 10.0, .level = 0, .id = 2},
      {.name = "consumer", .start = 70.0, .dur = 10.0, .level = 0, .id = 3},
      {.name = "operand", .start = 20.0, .dur = 10.0, .level = 1, .id = 4},
      {.name = "consumer", .start = 60.0, .dur = 10.0, .level = 1, .id = 5},
  }));
  timeline.set_selected_event_index_for_test(1);

  timeline.SetSelectedEventDependencies({"operand"}, {"consumer"});

  EXPECT_THAT(timeline.producer_event_indices(), ElementsAre(3));
  EXPECT_THAT(timeline.consumer_event_indices(), ElementsAre(4));
}

TEST(TimelineTest, SelectedEventDependenciesIgnoreOtherTracks) {
  ColorPalette palette = ColorPalette::Default();
  Timeline timeline(palette);
  timeline.set_hlo_dependency_arrows_enabled(true);
  // XLA Ops, level 0:  operand             selected
  // Other,   level 2:            operand             consumer   (not searched)
  timeline.SetTimelineData(MakeDependencyTestData({
      {.name = "operand", .start = 0.0, .dur = 10.0, .level = 0, .id = 1},
      {.name = "selected", .start = 40.0, .dur = 10.0, .level = 0, .id = 2},
      {.name = "operand", .start = 30.0, .dur = 10.0, .level = 2, .id = 3},
      {.name = "consumer", .start = 60.0, .dur = 10.0, .level = 2, .id = 4},
  }));
  timeline.set_selected_event_index_for_test(1);

  timeline.SetSelectedEventDependencies({"operand"}, {"consumer"});

  EXPECT_THAT(timeline.producer_event_indices(), ElementsAre(0));
  EXPECT_THAT(timeline.consumer_event_indices(), IsEmpty());
}

TEST(TimelineTest, SelectedEventDependenciesSkipUnknownAndRepeatedNames) {
  ColorPalette palette = ColorPalette::Default();
  Timeline timeline(palette);
  timeline.set_hlo_dependency_arrows_enabled(true);
  timeline.SetTimelineData(MakeDependencyTestData({
      {.name = "a", .start = 0.0, .dur = 10.0, .level = 0, .id = 1},
      {.name = "b", .start = 10.0, .dur = 10.0, .level = 0, .id = 2},
      {.name = "selected", .start = 40.0, .dur = 10.0, .level = 0, .id = 3},
  }));
  timeline.set_selected_event_index_for_test(2);

  timeline.SetSelectedEventDependencies({"b", "missing", "a", "b"},
                                        {"missing"});

  EXPECT_THAT(timeline.producer_event_indices(), ElementsAre(1, 0));
  EXPECT_THAT(timeline.consumer_event_indices(), IsEmpty());
}

TEST(TimelineTest, SelectedEventDependenciesExcludeEventsStartingAtSameTime) {
  ColorPalette palette = ColorPalette::Default();
  Timeline timeline(palette);
  timeline.set_hlo_dependency_arrows_enabled(true);
  // level 0:  selected
  // level 1:  child      (same start: neither before nor after)
  timeline.SetTimelineData(MakeDependencyTestData({
      {.name = "selected", .start = 40.0, .dur = 10.0, .level = 0, .id = 1},
      {.name = "child", .start = 40.0, .dur = 5.0, .level = 1, .id = 2},
  }));
  timeline.set_selected_event_index_for_test(0);

  timeline.SetSelectedEventDependencies({"child"}, {"child"});

  EXPECT_THAT(timeline.producer_event_indices(), IsEmpty());
  EXPECT_THAT(timeline.consumer_event_indices(), IsEmpty());
}

TEST(TimelineTest, SelectedEventDependenciesMatchNamesExactly) {
  ColorPalette palette = ColorPalette::Default();
  Timeline timeline(palette);
  timeline.set_hlo_dependency_arrows_enabled(true);
  timeline.SetTimelineData(MakeDependencyTestData({
      {.name = "fusion.1", .start = 0.0, .dur = 10.0, .level = 0, .id = 1},
      {.name = "fusion.10", .start = 20.0, .dur = 10.0, .level = 0, .id = 2},
      {.name = "selected", .start = 40.0, .dur = 10.0, .level = 0, .id = 3},
  }));
  timeline.set_selected_event_index_for_test(2);

  timeline.SetSelectedEventDependencies({"fusion.1"}, {});

  EXPECT_THAT(timeline.producer_event_indices(), ElementsAre(0));
}

TEST(TimelineTest, SelectedEventDependenciesResolveFromNestedLevel) {
  ColorPalette palette = ColorPalette::Default();
  Timeline timeline(palette);
  timeline.set_hlo_dependency_arrows_enabled(true);
  // level 0:  operand   [ parent          ]   consumer
  // level 1:                selected
  timeline.SetTimelineData(MakeDependencyTestData({
      {.name = "operand", .start = 0.0, .dur = 10.0, .level = 0, .id = 1},
      {.name = "parent", .start = 30.0, .dur = 30.0, .level = 0, .id = 2},
      {.name = "consumer", .start = 70.0, .dur = 10.0, .level = 0, .id = 3},
      {.name = "selected", .start = 40.0, .dur = 10.0, .level = 1, .id = 4},
  }));
  timeline.set_selected_event_index_for_test(3);

  timeline.SetSelectedEventDependencies({"operand"}, {"consumer"});

  EXPECT_THAT(timeline.producer_event_indices(), ElementsAre(0));
  EXPECT_THAT(timeline.consumer_event_indices(), ElementsAre(2));
}

TEST(TimelineTest, SelectedEventDependenciesIgnoreTracksAboveSelectedEvent) {
  ColorPalette palette = ColorPalette::Default();
  Timeline timeline(palette);
  timeline.set_hlo_dependency_arrows_enabled(true);
  // XLA Ops, level 0:            operand                   (not searched)
  // XLA Ops, level 1:                               consumer   (not searched)
  // Other,   level 2:  operand          selected
  timeline.SetTimelineData(MakeDependencyTestData({
      {.name = "operand", .start = 30.0, .dur = 5.0, .level = 0, .id = 1},
      {.name = "consumer", .start = 60.0, .dur = 5.0, .level = 1, .id = 2},
      {.name = "operand", .start = 0.0, .dur = 10.0, .level = 2, .id = 3},
      {.name = "selected", .start = 40.0, .dur = 10.0, .level = 2, .id = 4},
  }));
  timeline.set_selected_event_index_for_test(3);

  timeline.SetSelectedEventDependencies({"operand"}, {"consumer"});

  // Only the "Other" track of the selected event is searched, so the closer
  // `operand` and the `consumer` on the "XLA Ops" track are not found.
  EXPECT_THAT(timeline.producer_event_indices(), ElementsAre(2));
  EXPECT_THAT(timeline.consumer_event_indices(), IsEmpty());
}

TEST(TimelineTest, SelectedEventDependenciesSkipEventsBeyondSearchLimit) {
  ColorPalette palette = ColorPalette::Default();
  Timeline timeline(palette);
  timeline.set_hlo_dependency_arrows_enabled(true);
  // At most 10,000 events per level are checked on each side of the selected
  // event. With 9,999 other events in between, each `near_*` event is the
  // 10,000th event on its side and is found, and each `far_*` event is the
  // 10,001st and is not.
  //
  // level 0:  far_operand  near_operand  [other x 9,999]  selected
  //           [other x 9,999]  near_consumer  far_consumer
  constexpr int kEventsInBetween = 9999;
  std::vector<EventSpec> events;
  const auto add_event = [&events](absl::string_view name) {
    const size_t index = events.size();
    events.push_back({.name = std::string(name),
                      .start = static_cast<double>(index),
                      .dur = 0.5,
                      .id = index + 1});
  };
  add_event("far_operand");
  add_event("near_operand");
  for (int i = 0; i < kEventsInBetween; ++i) add_event("other");
  add_event("selected");
  for (int i = 0; i < kEventsInBetween; ++i) add_event("other");
  add_event("near_consumer");
  add_event("far_consumer");
  timeline.SetTimelineData(MakeDependencyTestData(events));
  timeline.set_selected_event_index_for_test(kEventsInBetween + 2);

  timeline.SetSelectedEventDependencies({"far_operand", "near_operand"},
                                        {"near_consumer", "far_consumer"});

  EXPECT_THAT(timeline.producer_event_indices(), ElementsAre(1));
  EXPECT_THAT(timeline.consumer_event_indices(),
              ElementsAre(2 * kEventsInBetween + 3));
}

TEST(TimelineTest, SelectedEventDependenciesAreEmptyForOutOfRangeSelection) {
  ColorPalette palette = ColorPalette::Default();
  Timeline timeline(palette);
  timeline.set_hlo_dependency_arrows_enabled(true);
  timeline.SetTimelineData(MakeDependencyTestData({
      {.name = "operand", .start = 0.0, .dur = 10.0, .level = 0, .id = 1},
      {.name = "consumer", .start = 60.0, .dur = 10.0, .level = 0, .id = 2},
  }));
  timeline.set_selected_event_index_for_test(2);

  timeline.SetSelectedEventDependencies({"operand"}, {"consumer"});

  EXPECT_THAT(timeline.producer_event_indices(), IsEmpty());
  EXPECT_THAT(timeline.consumer_event_indices(), IsEmpty());
}

TEST(TimelineTest, SelectedEventDependenciesAreReplacedByLaterCall) {
  ColorPalette palette = ColorPalette::Default();
  Timeline timeline(palette);
  timeline.set_hlo_dependency_arrows_enabled(true);
  timeline.SetTimelineData(MakeDependencyTestData({
      {.name = "old_operand", .start = 0.0, .dur = 10.0, .level = 0, .id = 1},
      {.name = "new_operand", .start = 20.0, .dur = 10.0, .level = 0, .id = 2},
      {.name = "selected", .start = 40.0, .dur = 10.0, .level = 0, .id = 3},
      {.name = "old_consumer", .start = 60.0, .dur = 10.0, .level = 0, .id = 4},
  }));
  timeline.set_selected_event_index_for_test(2);
  timeline.SetSelectedEventDependencies({"old_operand"}, {"old_consumer"});

  timeline.SetSelectedEventDependencies({"new_operand"}, {});

  EXPECT_THAT(timeline.producer_event_indices(), ElementsAre(1));
  EXPECT_THAT(timeline.consumer_event_indices(), IsEmpty());
}

TEST(TimelineTest, SelectedEventDependenciesAreIgnoredWhenFlagIsDisabled) {
  ColorPalette palette = ColorPalette::Default();
  Timeline timeline(palette);
  timeline.SetTimelineData(MakeDependencyTestData({
      {.name = "operand", .start = 0.0, .dur = 10.0, .level = 0, .id = 1},
      {.name = "selected", .start = 40.0, .dur = 10.0, .level = 0, .id = 2},
      {.name = "consumer", .start = 60.0, .dur = 10.0, .level = 0, .id = 3},
  }));
  timeline.set_selected_event_index_for_test(1);

  timeline.SetSelectedEventDependencies({"operand"}, {"consumer"});

  EXPECT_THAT(timeline.producer_event_indices(), IsEmpty());
  EXPECT_THAT(timeline.consumer_event_indices(), IsEmpty());
}

TEST(TimelineTest, SelectedEventDependenciesAreEmptyWithoutSelection) {
  ColorPalette palette = ColorPalette::Default();
  Timeline timeline(palette);
  timeline.set_hlo_dependency_arrows_enabled(true);
  timeline.SetTimelineData(MakeDependencyTestData({
      {.name = "operand", .start = 0.0, .dur = 10.0, .level = 0, .id = 1},
      {.name = "consumer", .start = 60.0, .dur = 10.0, .level = 0, .id = 2},
  }));

  timeline.SetSelectedEventDependencies({"operand"}, {"consumer"});

  EXPECT_THAT(timeline.producer_event_indices(), IsEmpty());
  EXPECT_THAT(timeline.consumer_event_indices(), IsEmpty());
}

TEST(TimelineTest, SelectedEventDependenciesAreHiddenAfterSelectionChange) {
  ColorPalette palette = ColorPalette::Default();
  Timeline timeline(palette);
  timeline.set_hlo_dependency_arrows_enabled(true);
  timeline.SetTimelineData(MakeDependencyTestData({
      {.name = "operand", .start = 0.0, .dur = 10.0, .level = 0, .id = 1},
      {.name = "selected", .start = 40.0, .dur = 10.0, .level = 0, .id = 2},
      {.name = "consumer", .start = 60.0, .dur = 10.0, .level = 0, .id = 3},
  }));
  timeline.set_selected_event_index_for_test(1);
  timeline.SetSelectedEventDependencies({"operand"}, {"consumer"});

  timeline.set_selected_event_index_for_test(0);

  EXPECT_THAT(timeline.producer_event_indices(), IsEmpty());
  EXPECT_THAT(timeline.consumer_event_indices(), IsEmpty());
}

TEST(TimelineTest, SelectedEventDependenciesAreResolvedAgainBySetTimelineData) {
  ColorPalette palette = ColorPalette::Default();
  Timeline timeline(palette);
  timeline.set_hlo_dependency_arrows_enabled(true);
  // before:  operand   selected   consumer             (indices 0, 1, 2)
  // after:   earlier   operand    selected   consumer  (indices 0, 1, 2, 3)
  timeline.SetTimelineData(MakeDependencyTestData({
      {.name = "operand", .start = 10.0, .dur = 10.0, .level = 0, .id = 1},
      {.name = "selected", .start = 40.0, .dur = 10.0, .level = 0, .id = 2},
      {.name = "consumer", .start = 60.0, .dur = 10.0, .level = 0, .id = 3},
  }));
  timeline.set_selected_event_index_for_test(1);
  timeline.SetSelectedEventDependencies({"operand"}, {"consumer"});

  // The new data has an earlier event, which shifts all event indices.
  timeline.SetTimelineData(MakeDependencyTestData({
      {.name = "earlier", .start = 0.0, .dur = 5.0, .level = 0, .id = 4},
      {.name = "operand", .start = 10.0, .dur = 10.0, .level = 0, .id = 1},
      {.name = "selected", .start = 40.0, .dur = 10.0, .level = 0, .id = 2},
      {.name = "consumer", .start = 60.0, .dur = 10.0, .level = 0, .id = 3},
  }));

  EXPECT_EQ(timeline.selected_event_index(), 2);
  EXPECT_THAT(timeline.producer_event_indices(), ElementsAre(1));
  EXPECT_THAT(timeline.consumer_event_indices(), ElementsAre(3));
}

TEST(TimelineTest, SelectedEventDependenciesPickUpEventsLoadedByRefetch) {
  ColorPalette palette = ColorPalette::Default();
  Timeline timeline(palette);
  timeline.set_hlo_dependency_arrows_enabled(true);
  // before:  operand             selected
  // after:   operand   operand   selected   consumer
  //                    ^ closer now         ^ new
  timeline.SetTimelineData(MakeDependencyTestData({
      {.name = "operand", .start = 0.0, .dur = 10.0, .level = 0, .id = 1},
      {.name = "selected", .start = 40.0, .dur = 10.0, .level = 0, .id = 2},
  }));
  timeline.set_selected_event_index_for_test(1);
  timeline.SetSelectedEventDependencies({"operand"}, {"consumer"});

  // The refetched data has a closer `operand` and a `consumer` that were not
  // loaded before, e.g. after zooming in.
  timeline.SetTimelineData(MakeDependencyTestData({
      {.name = "operand", .start = 0.0, .dur = 10.0, .level = 0, .id = 1},
      {.name = "operand", .start = 20.0, .dur = 10.0, .level = 0, .id = 3},
      {.name = "selected", .start = 40.0, .dur = 10.0, .level = 0, .id = 2},
      {.name = "consumer", .start = 60.0, .dur = 10.0, .level = 0, .id = 4},
  }));

  EXPECT_EQ(timeline.selected_event_index(), 2);
  EXPECT_THAT(timeline.producer_event_indices(), ElementsAre(1));
  EXPECT_THAT(timeline.consumer_event_indices(), ElementsAre(3));
}

TEST(TimelineTest, SelectedEventDependenciesDropEventsMissingAfterRefetch) {
  ColorPalette palette = ColorPalette::Default();
  Timeline timeline(palette);
  timeline.set_hlo_dependency_arrows_enabled(true);
  // before:  operand   selected   consumer
  // after:             selected   consumer
  timeline.SetTimelineData(MakeDependencyTestData({
      {.name = "operand", .start = 0.0, .dur = 10.0, .level = 0, .id = 1},
      {.name = "selected", .start = 40.0, .dur = 10.0, .level = 0, .id = 2},
      {.name = "consumer", .start = 60.0, .dur = 10.0, .level = 0, .id = 3},
  }));
  timeline.set_selected_event_index_for_test(1);
  timeline.SetSelectedEventDependencies({"operand"}, {"consumer"});

  // The refetched data no longer has the `operand`, e.g. at a coarser
  // resolution.
  timeline.SetTimelineData(MakeDependencyTestData({
      {.name = "selected", .start = 40.0, .dur = 10.0, .level = 0, .id = 2},
      {.name = "consumer", .start = 60.0, .dur = 10.0, .level = 0, .id = 3},
  }));

  EXPECT_EQ(timeline.selected_event_index(), 0);
  EXPECT_THAT(timeline.producer_event_indices(), IsEmpty());
  EXPECT_THAT(timeline.consumer_event_indices(), ElementsAre(1));
}

TEST(TimelineTest, SelectedEventDependenciesSurviveRepeatedRefetches) {
  ColorPalette palette = ColorPalette::Default();
  Timeline timeline(palette);
  timeline.set_hlo_dependency_arrows_enabled(true);
  const FlameChartTimelineData data = MakeDependencyTestData({
      {.name = "operand", .start = 0.0, .dur = 10.0, .level = 0, .id = 1},
      {.name = "selected", .start = 40.0, .dur = 10.0, .level = 0, .id = 2},
      {.name = "consumer", .start = 60.0, .dur = 10.0, .level = 0, .id = 3},
  });
  timeline.SetTimelineData(data);
  timeline.set_selected_event_index_for_test(1);
  timeline.SetSelectedEventDependencies({"operand"}, {"consumer"});

  timeline.SetTimelineData(data);
  timeline.SetTimelineData(data);

  EXPECT_EQ(timeline.selected_event_index(), 1);
  EXPECT_THAT(timeline.producer_event_indices(), ElementsAre(0));
  EXPECT_THAT(timeline.consumer_event_indices(), ElementsAre(2));
}

TEST(TimelineTest,
     SelectedEventDependenciesOfPreviousSelectionAreNotResolvedAgain) {
  ColorPalette palette = ColorPalette::Default();
  Timeline timeline(palette);
  timeline.set_hlo_dependency_arrows_enabled(true);
  const FlameChartTimelineData data = MakeDependencyTestData({
      {.name = "operand", .start = 0.0, .dur = 10.0, .level = 0, .id = 1},
      {.name = "selected", .start = 40.0, .dur = 10.0, .level = 0, .id = 2},
      {.name = "consumer", .start = 60.0, .dur = 10.0, .level = 0, .id = 3},
  });
  timeline.SetTimelineData(data);
  timeline.set_selected_event_index_for_test(1);
  timeline.SetSelectedEventDependencies({"operand"}, {"consumer"});
  timeline.set_selected_event_index_for_test(2);

  timeline.SetTimelineData(data);

  EXPECT_EQ(timeline.selected_event_index(), 2);
  EXPECT_THAT(timeline.producer_event_indices(), IsEmpty());
  EXPECT_THAT(timeline.consumer_event_indices(), IsEmpty());
}

TEST(TimelineTest, SelectedEventDependenciesAreNotResolvedAgainForAnotherOp) {
  ColorPalette palette = ColorPalette::Default();
  Timeline timeline(palette);
  timeline.set_hlo_dependency_arrows_enabled(true);
  const FlameChartTimelineData data = MakeDependencyTestData({
      {.name = "operand", .start = 0.0, .dur = 10.0, .level = 0, .id = 1},
      {.name = "selected", .start = 40.0, .dur = 10.0, .level = 0, .id = 2},
      {.name = "consumer", .start = 60.0, .dur = 10.0, .level = 0, .id = 3},
  });
  timeline.SetTimelineData(data);
  timeline.set_selected_event_index_for_test(1);
  timeline.SetSelectedEventDependencies({"operand"}, {"consumer"});
  // Makes `SetTimelineData` select the active search result instead.
  timeline.set_search_results_for_test({{.event_id = 3, .name = "consumer"}});
  timeline.set_current_search_result_index_for_test(0);

  timeline.SetTimelineData(data);

  EXPECT_EQ(timeline.selected_event_index(), 2);
  EXPECT_THAT(timeline.producer_event_indices(), IsEmpty());
  EXPECT_THAT(timeline.consumer_event_indices(), IsEmpty());
}

TEST(TimelineTest,
     SelectedEventDependenciesAreClearedWhenSelectedEventIsNotRefetched) {
  ColorPalette palette = ColorPalette::Default();
  Timeline timeline(palette);
  timeline.set_hlo_dependency_arrows_enabled(true);
  // before:  operand   selected   consumer
  // after:   operand              consumer
  timeline.SetTimelineData(MakeDependencyTestData({
      {.name = "operand", .start = 0.0, .dur = 10.0, .level = 0, .id = 1},
      {.name = "selected", .start = 40.0, .dur = 10.0, .level = 0, .id = 2},
      {.name = "consumer", .start = 60.0, .dur = 10.0, .level = 0, .id = 3},
  }));
  timeline.set_selected_event_index_for_test(1);
  timeline.SetSelectedEventDependencies({"operand"}, {"consumer"});

  timeline.SetTimelineData(MakeDependencyTestData({
      {.name = "operand", .start = 0.0, .dur = 10.0, .level = 0, .id = 1},
      {.name = "consumer", .start = 60.0, .dur = 10.0, .level = 0, .id = 3},
  }));

  EXPECT_EQ(timeline.selected_event_index(), -1);
  EXPECT_THAT(timeline.producer_event_indices(), IsEmpty());
  EXPECT_THAT(timeline.consumer_event_indices(), IsEmpty());
}

TEST(TimelineTest, SelectedEventDependenciesSetDuringRefetchAreKept) {
  ColorPalette palette = ColorPalette::Default();
  Timeline timeline(palette);
  timeline.set_hlo_dependency_arrows_enabled(true);
  // level 0:  operand   selected   new_operand   selected
  //                     ^ before                 ^ navigated to in the refetch
  const FlameChartTimelineData data = MakeDependencyTestData({
      {.name = "operand", .start = 0.0, .dur = 10.0, .level = 0, .id = 1},
      {.name = "selected", .start = 20.0, .dur = 10.0, .level = 0, .id = 2},
      {.name = "new_operand", .start = 40.0, .dur = 10.0, .level = 0, .id = 3},
      {.name = "selected", .start = 60.0, .dur = 10.0, .level = 0, .id = 4},
  });
  timeline.SetTimelineData(data);
  timeline.set_data_time_range({0.0, 100.0});
  timeline.set_selected_event_index_for_test(1);
  timeline.SetSelectedEventDependencies({"operand"}, {});
  // Makes `SetTimelineData` navigate to another event with the same name,
  // whose selection handler sets new dependencies.
  timeline.set_search_results_for_test({{.event_id = 4, .name = "selected"}});
  timeline.set_current_search_result_index_for_test(0);
  timeline.set_pending_navigation_event_id_for_test(4);
  timeline.set_event_callback(
      [&timeline](absl::string_view type, const EventData&) {
        if (type == kEventSelected) {
          timeline.SetSelectedEventDependencies({"new_operand"}, {});
        }
      });

  timeline.SetTimelineData(data);

  EXPECT_EQ(timeline.selected_event_index(), 3);
  EXPECT_THAT(timeline.producer_event_indices(), ElementsAre(2));
}

// Builds data for the module run tests: a process with an "XLA Modules" track
// at level 0 and an "XLA Ops" track at level 1. `events` must be sorted by
// level, then by start time.
FlameChartTimelineData MakeModuleRunTestData(
    std::initializer_list<EventSpec> events) {
  FlameChartTimelineData data = MakeDependencyTestData(events);
  Group process = MakeProcessGroup("TPU:0");
  process.child_indices = {1, 2};
  data.groups = {std::move(process),
                 MakeThreadGroup("XLA Modules", /*parent_index=*/0,
                                 /*start_level=*/0, /*level_count=*/1),
                 MakeThreadGroup("XLA Ops", /*parent_index=*/0,
                                 /*start_level=*/1, /*level_count=*/1)};
  return data;
}

TEST(TimelineTest,
     SelectedEventDependenciesAreLimitedToModuleRunOfSelectedEvent) {
  ColorPalette palette = ColorPalette::Default();
  Timeline timeline(palette);
  timeline.set_hlo_dependency_arrows_enabled(true);
  // XLA Modules:  [ module     ]  [ module                      ]  [ module  ]
  // XLA Ops:        earlier_op      operand   selected   consumer    later_op
  //                 (other run)     ^ picked              ^ picked  (other run)
  timeline.SetTimelineData(MakeModuleRunTestData({
      {.name = "module", .start = 0.0, .dur = 40.0, .level = 0, .id = 1},
      {.name = "module", .start = 50.0, .dur = 40.0, .level = 0, .id = 2},
      {.name = "module", .start = 100.0, .dur = 40.0, .level = 0, .id = 3},
      {.name = "earlier_op", .start = 10.0, .dur = 5.0, .level = 1, .id = 4},
      {.name = "operand", .start = 55.0, .dur = 5.0, .level = 1, .id = 5},
      {.name = "selected", .start = 65.0, .dur = 5.0, .level = 1, .id = 6},
      {.name = "consumer", .start = 75.0, .dur = 5.0, .level = 1, .id = 7},
      {.name = "later_op", .start = 110.0, .dur = 5.0, .level = 1, .id = 8},
  }));
  timeline.set_selected_event_index_for_test(5);

  // `earlier_op` and `later_op` are in other module runs.
  timeline.SetSelectedEventDependencies({"earlier_op", "operand"},
                                        {"consumer", "later_op"});

  EXPECT_THAT(timeline.producer_event_indices(), ElementsAre(4));
  EXPECT_THAT(timeline.consumer_event_indices(), ElementsAre(6));
}

TEST(TimelineTest, SelectedEventDependenciesAreNotLimitedOutsideModuleRuns) {
  ColorPalette palette = ColorPalette::Default();
  Timeline timeline(palette);
  timeline.set_hlo_dependency_arrows_enabled(true);
  // XLA Modules:  [ module    ]                         [ module    ]
  // XLA Ops:        operand            selected            consumer
  //                                    ^ in no module run: not limited
  timeline.SetTimelineData(MakeModuleRunTestData({
      {.name = "module", .start = 0.0, .dur = 40.0, .level = 0, .id = 1},
      {.name = "module", .start = 100.0, .dur = 40.0, .level = 0, .id = 2},
      {.name = "operand", .start = 10.0, .dur = 5.0, .level = 1, .id = 3},
      {.name = "selected", .start = 60.0, .dur = 5.0, .level = 1, .id = 4},
      {.name = "consumer", .start = 110.0, .dur = 5.0, .level = 1, .id = 5},
  }));
  timeline.set_selected_event_index_for_test(3);

  timeline.SetSelectedEventDependencies({"operand"}, {"consumer"});

  EXPECT_THAT(timeline.producer_event_indices(), ElementsAre(2));
  EXPECT_THAT(timeline.consumer_event_indices(), ElementsAre(4));
}

TEST(TimelineTest,
     SelectedEventDependenciesIncludeEventsAtModuleRunBoundaries) {
  ColorPalette palette = ColorPalette::Default();
  Timeline timeline(palette);
  timeline.set_hlo_dependency_arrows_enabled(true);
  // XLA Modules:            [ module                           ]
  // XLA Ops:   out_operand  edge_operand  selected  edge_consumer  out_consumer
  //                         ^ starts at 50          ^ starts at 90, the end
  timeline.SetTimelineData(MakeModuleRunTestData({
      {.name = "module", .start = 50.0, .dur = 40.0, .level = 0, .id = 1},
      {.name = "out_operand", .start = 45.0, .dur = 4.0, .level = 1, .id = 2},
      {.name = "edge_operand", .start = 50.0, .dur = 4.0, .level = 1, .id = 3},
      {.name = "selected", .start = 60.0, .dur = 5.0, .level = 1, .id = 4},
      {.name = "edge_consumer", .start = 90.0, .dur = 4.0, .level = 1, .id = 5},
      {.name = "out_consumer", .start = 95.0, .dur = 4.0, .level = 1, .id = 6},
  }));
  timeline.set_selected_event_index_for_test(3);

  // `edge_*` start exactly at the start and end of the module run.
  timeline.SetSelectedEventDependencies({"out_operand", "edge_operand"},
                                        {"edge_consumer", "out_consumer"});

  EXPECT_THAT(timeline.producer_event_indices(), ElementsAre(2));
  EXPECT_THAT(timeline.consumer_event_indices(), ElementsAre(4));
}

TEST(TimelineTest, SelectedEventDependenciesAreLimitedOnlyByXlaModulesTrack) {
  ColorPalette palette = ColorPalette::Default();
  Timeline timeline(palette);
  timeline.set_hlo_dependency_arrows_enabled(true);
  // Steps:                 [ step           ]       (not "XLA Modules")
  // XLA Ops:    operand        selected          consumer
  FlameChartTimelineData data = MakeModuleRunTestData({
      {.name = "step", .start = 50.0, .dur = 40.0, .level = 0, .id = 1},
      {.name = "operand", .start = 10.0, .dur = 5.0, .level = 1, .id = 2},
      {.name = "selected", .start = 60.0, .dur = 5.0, .level = 1, .id = 3},
      {.name = "consumer", .start = 110.0, .dur = 5.0, .level = 1, .id = 4},
  });
  // Another track of the same process, not "XLA Modules", covers the selected
  // event.
  data.groups[1].name = "Steps";
  timeline.SetTimelineData(std::move(data));
  timeline.set_selected_event_index_for_test(2);

  timeline.SetSelectedEventDependencies({"operand"}, {"consumer"});

  EXPECT_THAT(timeline.producer_event_indices(), ElementsAre(1));
  EXPECT_THAT(timeline.consumer_event_indices(), ElementsAre(3));
}

// The route of an arrow between two bars.

constexpr Pixel kRowY = 100.0f;
constexpr Pixel kHalfBarHeight = kEventHeight * 0.5f;

TEST(RouteDependencyArrowTest, RunsOperandArrowsAboveTheRow) {
  //  operand: [100     150]    selected: [200       260]
  const DependencyArrowRoute route = RouteDependencyArrow(
      {.left = 100.0f, .right = 150.0f, .row_y = kRowY},
      {.left = 200.0f, .right = 260.0f, .row_y = kRowY}, /*above=*/true,
      /*slot=*/0);

  // From the middle of the operand's top edge, along a lane above the row, to
  // the top edge of the selected op 10 px inside its left end.
  EXPECT_FLOAT_EQ(route.start.x, 125.0f);
  EXPECT_FLOAT_EQ(route.start.y, kRowY - kHalfBarHeight);
  EXPECT_FLOAT_EQ(route.lane_y,
                  kRowY - kHalfBarHeight - kHloDependencyArrowLaneGap);
  EXPECT_FLOAT_EQ(route.end.x, 210.0f);
  EXPECT_FLOAT_EQ(route.end.y, kRowY - kHalfBarHeight);
  EXPECT_FLOAT_EQ(route.corner_radius, kHloDependencyArrowCornerRadius);
  EXPECT_TRUE(route.with_head);
}

TEST(RouteDependencyArrowTest, RunsConsumerArrowsBelowTheRow) {
  //  selected: [100     150]    consumer: [200       260]
  const DependencyArrowRoute route = RouteDependencyArrow(
      {.left = 100.0f, .right = 150.0f, .row_y = kRowY},
      {.left = 200.0f, .right = 260.0f, .row_y = kRowY}, /*above=*/false,
      /*slot=*/0);

  // From the bottom edge of the selected op 10 px inside its right end, along
  // a lane below the row, to the middle of the consumer's bottom edge.
  EXPECT_FLOAT_EQ(route.start.x, 140.0f);
  EXPECT_FLOAT_EQ(route.start.y, kRowY + kHalfBarHeight);
  EXPECT_FLOAT_EQ(route.lane_y,
                  kRowY + kHalfBarHeight + kHloDependencyArrowLaneGap);
  EXPECT_FLOAT_EQ(route.end.x, 230.0f);
  EXPECT_FLOAT_EQ(route.end.y, kRowY + kHalfBarHeight);
  EXPECT_TRUE(route.with_head);
}

TEST(RouteDependencyArrowTest, PutsParallelArrowsIntoSeparateLanes) {
  const DependencyArrowRoute first = RouteDependencyArrow(
      {.left = 100.0f, .right = 150.0f, .row_y = kRowY},
      {.left = 200.0f, .right = 260.0f, .row_y = kRowY}, /*above=*/true,
      /*slot=*/0);
  const DependencyArrowRoute third = RouteDependencyArrow(
      {.left = 0.0f, .right = 50.0f, .row_y = kRowY},
      {.left = 200.0f, .right = 260.0f, .row_y = kRowY}, /*above=*/true,
      /*slot=*/2);

  EXPECT_FLOAT_EQ(third.lane_y, first.lane_y - 2 * kHloDependencyArrowLaneStep);
}

TEST(RouteDependencyArrowTest, RunsAroundBothRowsWhenTheBarsDiffer) {
  // The operand is on the row below the selected op: the lane lies above the
  // upper row, and the arrow leaves the operand through its own top edge.
  const DependencyArrowRoute route = RouteDependencyArrow(
      {.left = 100.0f, .right = 150.0f, .row_y = kRowY + 20.0f},
      {.left = 200.0f, .right = 260.0f, .row_y = kRowY}, /*above=*/true,
      /*slot=*/0);

  EXPECT_FLOAT_EQ(route.start.y, kRowY + 20.0f - kHalfBarHeight);
  EXPECT_FLOAT_EQ(route.lane_y,
                  kRowY - kHalfBarHeight - kHloDependencyArrowLaneGap);
  EXPECT_FLOAT_EQ(route.end.y, kRowY - kHalfBarHeight);
}

TEST(RouteDependencyArrowTest, KeepsTheInsetWithinAQuarterOfTheBar) {
  // A 12 px wide selected op: the arrow ends 3 px inside it, not 10 px.
  const DependencyArrowRoute route = RouteDependencyArrow(
      {.left = 100.0f, .right = 150.0f, .row_y = kRowY},
      {.left = 200.0f, .right = 212.0f, .row_y = kRowY}, /*above=*/true,
      /*slot=*/0);

  EXPECT_FLOAT_EQ(route.end.x, 203.0f);
}

TEST(RouteDependencyArrowTest, DropsTheHeadAndTightensTheCornersWhenClose) {
  // Two 4 px wide bars right next to each other: the route is 4 px wide, so
  // it gets 2 px corners and no arrowhead.
  const DependencyArrowRoute route = RouteDependencyArrow(
      {.left = 100.0f, .right = 104.0f, .row_y = kRowY},
      {.left = 105.0f, .right = 109.0f, .row_y = kRowY}, /*above=*/true,
      /*slot=*/0);

  EXPECT_FLOAT_EQ(route.start.x, 102.0f);
  EXPECT_FLOAT_EQ(route.end.x, 106.0f);
  EXPECT_FLOAT_EQ(route.corner_radius, 2.0f);
  EXPECT_FALSE(route.with_head);
}

TEST(RouteDependencyArrowTest, DropsTheHeadForACloseConsumerToo) {
  //  selected: [100 104]  consumer: [105 109]
  const DependencyArrowRoute route = RouteDependencyArrow(
      {.left = 100.0f, .right = 104.0f, .row_y = kRowY},
      {.left = 105.0f, .right = 109.0f, .row_y = kRowY}, /*above=*/false,
      /*slot=*/0);

  EXPECT_FLOAT_EQ(route.start.x, 103.0f);
  EXPECT_FLOAT_EQ(route.end.x, 107.0f);
  EXPECT_FLOAT_EQ(route.corner_radius, 2.0f);
  EXPECT_FALSE(route.with_head);
}

TEST(RouteDependencyArrowTest, KeepsTheHeadBetweenTouchingWideBars) {
  // Two 40 px bars with no gap between them, the usual case on a busy row:
  // the arrow still spans half of the operand plus the inset into the
  // selected op, 30 px, so it keeps its head and full corners.
  const DependencyArrowRoute from_operand = RouteDependencyArrow(
      {.left = 100.0f, .right = 140.0f, .row_y = kRowY},
      {.left = 140.0f, .right = 180.0f, .row_y = kRowY}, /*above=*/true,
      /*slot=*/0);
  const DependencyArrowRoute to_consumer = RouteDependencyArrow(
      {.left = 100.0f, .right = 140.0f, .row_y = kRowY},
      {.left = 140.0f, .right = 180.0f, .row_y = kRowY}, /*above=*/false,
      /*slot=*/0);

  EXPECT_FLOAT_EQ(from_operand.start.x, 120.0f);
  EXPECT_FLOAT_EQ(from_operand.end.x, 150.0f);
  EXPECT_FLOAT_EQ(from_operand.corner_radius, kHloDependencyArrowCornerRadius);
  EXPECT_TRUE(from_operand.with_head);
  EXPECT_FLOAT_EQ(to_consumer.start.x, 130.0f);
  EXPECT_FLOAT_EQ(to_consumer.end.x, 160.0f);
  EXPECT_FLOAT_EQ(to_consumer.corner_radius, kHloDependencyArrowCornerRadius);
  EXPECT_TRUE(to_consumer.with_head);
}

TEST(RouteDependencyArrowTest, GrowsTheHeadFromTheMinimumSpan) {
  // Touching bars of the same width span three quarters of that width: half
  // of the first bar plus the quarter-width inset into the second. 20 px bars
  // span 15 px and get no head; 22 px bars span 16.5 px and do.
  const DependencyArrowRoute just_too_close = RouteDependencyArrow(
      {.left = 100.0f, .right = 120.0f, .row_y = kRowY},
      {.left = 120.0f, .right = 140.0f, .row_y = kRowY}, /*above=*/true,
      /*slot=*/0);
  const DependencyArrowRoute far_enough = RouteDependencyArrow(
      {.left = 100.0f, .right = 122.0f, .row_y = kRowY},
      {.left = 122.0f, .right = 144.0f, .row_y = kRowY}, /*above=*/true,
      /*slot=*/0);

  EXPECT_FLOAT_EQ(just_too_close.end.x - just_too_close.start.x, 15.0f);
  EXPECT_LT(just_too_close.end.x - just_too_close.start.x,
            kHloDependencyArrowMinSpanForHead);
  EXPECT_FALSE(just_too_close.with_head);
  EXPECT_FLOAT_EQ(far_enough.end.x - far_enough.start.x, 16.5f);
  EXPECT_GE(far_enough.end.x - far_enough.start.x,
            kHloDependencyArrowMinSpanForHead);
  EXPECT_TRUE(far_enough.with_head);
}

TEST(RouteDependencyArrowTest, RunsBackwardsWhenTheOperandReachesPastTheOp) {
  // A long operand on the row below overlaps the selected op in time, as an
  // asynchronous op can: its middle lies to the right of where the arrow
  // enters the selected op, so the lane runs leftwards. The span counts either
  // way, and the route keeps its head and corners.
  const DependencyArrowRoute route = RouteDependencyArrow(
      {.left = 100.0f, .right = 300.0f, .row_y = kRowY + 20.0f},
      {.left = 150.0f, .right = 400.0f, .row_y = kRowY}, /*above=*/true,
      /*slot=*/0);

  EXPECT_FLOAT_EQ(route.start.x, 200.0f);
  EXPECT_FLOAT_EQ(route.end.x, 160.0f);
  EXPECT_FLOAT_EQ(route.corner_radius, kHloDependencyArrowCornerRadius);
  EXPECT_TRUE(route.with_head);
}

TEST(RouteDependencyArrowTest, BracketsNarrowBarsThatAreCloseTogether) {
  // 8 px bars 2 px apart, as on a dense row zoomed out. The arrow from the
  // neighbour spans 8 px (half of its bar, the gap and the 2 px inset), so it
  // is a bracket with 4 px corners and no head; the arrow from the bar before
  // it spans 18 px and gets its head back, one lane further out.
  const BarGeometry selected{.left = 120.0f, .right = 128.0f, .row_y = kRowY};
  const DependencyArrowRoute from_neighbour = RouteDependencyArrow(
      {.left = 110.0f, .right = 118.0f, .row_y = kRowY}, selected,
      /*above=*/true, /*slot=*/0);
  const DependencyArrowRoute from_next = RouteDependencyArrow(
      {.left = 100.0f, .right = 108.0f, .row_y = kRowY}, selected,
      /*above=*/true, /*slot=*/1);

  EXPECT_FLOAT_EQ(from_neighbour.start.x, 114.0f);
  EXPECT_FLOAT_EQ(from_neighbour.end.x, 122.0f);
  EXPECT_FLOAT_EQ(from_neighbour.corner_radius, 4.0f);
  EXPECT_FALSE(from_neighbour.with_head);
  EXPECT_FLOAT_EQ(from_next.start.x, 104.0f);
  EXPECT_FLOAT_EQ(from_next.end.x, 122.0f);
  EXPECT_FLOAT_EQ(from_next.corner_radius, kHloDependencyArrowCornerRadius);
  EXPECT_TRUE(from_next.with_head);
  EXPECT_FLOAT_EQ(from_next.lane_y,
                  from_neighbour.lane_y - kHloDependencyArrowLaneStep);
}

// Events much shorter than the time of a pixel at the current zoom map to
// hairline bars (the timeline draws them at a minimum width, see
// kEventMinimumDrawWidth).

TEST(RouteDependencyArrowTest, StaysOnHairlineBars) {
  // A 1 px wide selected op between a 1 px operand and a 1 px consumer, far
  // enough apart for arrowheads: the arrows still touch the hairline bars, a
  // quarter pixel inside the ends of the selected op and in the middle of the
  // others.
  const BarGeometry operand{.left = 100.0f, .right = 101.0f, .row_y = kRowY};
  const BarGeometry selected{.left = 300.0f, .right = 301.0f, .row_y = kRowY};
  const BarGeometry consumer{.left = 500.0f, .right = 501.0f, .row_y = kRowY};

  const DependencyArrowRoute from_operand =
      RouteDependencyArrow(operand, selected, /*above=*/true, /*slot=*/0);
  const DependencyArrowRoute to_consumer =
      RouteDependencyArrow(selected, consumer, /*above=*/false, /*slot=*/0);

  EXPECT_FLOAT_EQ(from_operand.start.x, 100.5f);
  EXPECT_FLOAT_EQ(from_operand.end.x, 300.25f);
  EXPECT_TRUE(from_operand.with_head);
  EXPECT_FLOAT_EQ(to_consumer.start.x, 300.75f);
  EXPECT_FLOAT_EQ(to_consumer.end.x, 500.5f);
  EXPECT_TRUE(to_consumer.with_head);
}

TEST(RouteDependencyArrowTest, RoutesZeroWidthBars) {
  // Bars of zero width: the arrow starts and ends on their pixel, and keeps
  // its corners and head.
  const DependencyArrowRoute route = RouteDependencyArrow(
      {.left = 100.0f, .right = 100.0f, .row_y = kRowY},
      {.left = 300.0f, .right = 300.0f, .row_y = kRowY}, /*above=*/true,
      /*slot=*/0);

  EXPECT_FLOAT_EQ(route.start.x, 100.0f);
  EXPECT_FLOAT_EQ(route.end.x, 300.0f);
  EXPECT_FLOAT_EQ(route.corner_radius, kHloDependencyArrowCornerRadius);
  EXPECT_TRUE(route.with_head);
}

TEST(RouteDependencyArrowTest, CollapsesToATickBetweenTouchingHairlineBars) {
  // Two 1 px bars next to each other, as on a dense row zoomed far out: the
  // route is under a pixel wide, so it has no head and almost no corners, but
  // still rises the full lane gap above the row, where it reads as a tick
  // between the two bars.
  const DependencyArrowRoute route = RouteDependencyArrow(
      {.left = 100.0f, .right = 101.0f, .row_y = kRowY},
      {.left = 101.0f, .right = 102.0f, .row_y = kRowY}, /*above=*/true,
      /*slot=*/0);

  EXPECT_FLOAT_EQ(route.start.x, 100.5f);
  EXPECT_FLOAT_EQ(route.end.x, 101.25f);
  EXPECT_FLOAT_EQ(route.corner_radius, 0.375f);
  EXPECT_FALSE(route.with_head);
  EXPECT_FLOAT_EQ(route.lane_y,
                  kRowY - kHalfBarHeight - kHloDependencyArrowLaneGap);
}

TEST(RouteDependencyArrowTest, CollapsesToAVerticalTickOnASharedPixel) {
  // Zoomed out far enough, an op and its operand land on the same pixel. The
  // route goes straight up to the lane and back down: no corners, no head.
  const DependencyArrowRoute route = RouteDependencyArrow(
      {.left = 100.0f, .right = 100.0f, .row_y = kRowY},
      {.left = 100.0f, .right = 100.0f, .row_y = kRowY}, /*above=*/true,
      /*slot=*/0);

  EXPECT_FLOAT_EQ(route.start.x, 100.0f);
  EXPECT_FLOAT_EQ(route.end.x, 100.0f);
  EXPECT_FLOAT_EQ(route.corner_radius, 0.0f);
  EXPECT_FALSE(route.with_head);
}

TEST(TimelineTest, SelectAdjacentEventFocusesClosestOperand) {
  ColorPalette palette = ColorPalette::Default();
  Timeline timeline(palette);
  // Works even when HLO dependency arrows are disabled.
  timeline.SetTimelineData(MakeDependencyTestData({
      {.name = "operand", .start = 0.0, .dur = 10.0, .level = 0, .id = 1},
      {.name = "operand", .start = 20.0, .dur = 10.0, .level = 0, .id = 2},
      {.name = "selected", .start = 40.0, .dur = 10.0, .level = 0, .id = 3},
  }));
  timeline.set_data_time_range({0.0, 100.0});
  timeline.set_selected_event_index_for_test(2);
  int emitted_index = -1;
  timeline.set_event_callback(
      [&emitted_index](absl::string_view type, const EventData& detail) {
        if (type == kEventSelected) {
          emitted_index = std::any_cast<int>(detail.at(kEventSelectedIndex));
        }
      });

  timeline.SelectAdjacentEvent("%operand", /*is_operand=*/true);

  EXPECT_EQ(timeline.selected_event_index(), 1);
  EXPECT_EQ(emitted_index, 1);
}

TEST(TimelineTest, SelectAdjacentEventFocusesClosestConsumer) {
  ColorPalette palette = ColorPalette::Default();
  Timeline timeline(palette);
  timeline.SetTimelineData(MakeDependencyTestData({
      {.name = "selected", .start = 40.0, .dur = 10.0, .level = 0, .id = 1},
      {.name = "consumer", .start = 60.0, .dur = 10.0, .level = 0, .id = 2},
      {.name = "consumer", .start = 80.0, .dur = 10.0, .level = 0, .id = 3},
  }));
  timeline.set_data_time_range({0.0, 100.0});
  timeline.set_selected_event_index_for_test(0);
  int emitted_index = -1;
  timeline.set_event_callback(
      [&emitted_index](absl::string_view type, const EventData& detail) {
        if (type == kEventSelected) {
          emitted_index = std::any_cast<int>(detail.at(kEventSelectedIndex));
        }
      });

  timeline.SelectAdjacentEvent("consumer", /*is_operand=*/false);

  EXPECT_EQ(timeline.selected_event_index(), 1);
  EXPECT_EQ(emitted_index, 1);
}

TEST(TimelineTest, SelectAdjacentEventNoOpWhenNotFound) {
  ColorPalette palette = ColorPalette::Default();
  Timeline timeline(palette);
  timeline.SetTimelineData(MakeDependencyTestData({
      {.name = "selected", .start = 40.0, .dur = 10.0, .level = 0, .id = 1},
  }));
  timeline.set_selected_event_index_for_test(0);

  timeline.SelectAdjacentEvent("missing", /*is_operand=*/true);

  EXPECT_EQ(timeline.selected_event_index(), 0);
}

TEST(TimelineTest,
     SelectedEventDependenciesEmitResolvedNamesWhenArrowsDisabled) {
  ColorPalette palette = ColorPalette::Default();
  Timeline timeline(palette);
  timeline.SetTimelineData(MakeDependencyTestData({
      {.name = "operand", .start = 0.0, .dur = 10.0, .level = 0, .id = 1},
      {.name = "selected", .start = 40.0, .dur = 10.0, .level = 0, .id = 2},
      {.name = "consumer", .start = 60.0, .dur = 10.0, .level = 0, .id = 3},
  }));
  timeline.set_selected_event_index_for_test(1);

  int emitted_index = -1;
  std::vector<std::string> emitted_operands;
  std::vector<std::string> emitted_consumers;
  timeline.set_event_callback(
      [&](absl::string_view type, const EventData& detail) {
        if (type == kSelectedEventDependencies) {
          emitted_index = std::any_cast<int>(detail.at(kEventSelectedIndex));
          emitted_operands = std::any_cast<std::vector<std::string>>(
              detail.at(kSelectedEventDependenciesOperands));
          emitted_consumers = std::any_cast<std::vector<std::string>>(
              detail.at(kSelectedEventDependenciesConsumers));
        }
      });

  timeline.SetSelectedEventDependencies({"%operand", "%bitcast.1"},
                                        {"%consumer", "%tuple.2"});

  EXPECT_EQ(emitted_index, 1);
  EXPECT_THAT(emitted_operands, ElementsAre("operand"));
  EXPECT_THAT(emitted_consumers, ElementsAre("consumer"));
  EXPECT_THAT(timeline.producer_event_indices(), IsEmpty());
  EXPECT_THAT(timeline.consumer_event_indices(), IsEmpty());
}

TEST(TimelineTest, SelectedEventDependenciesEmitUpdatedNamesOnRefetch) {
  ColorPalette palette = ColorPalette::Default();
  Timeline timeline(palette);
  timeline.SetTimelineData(MakeDependencyTestData({
      {.name = "operand", .start = 0.0, .dur = 10.0, .level = 0, .id = 1},
      {.name = "selected", .start = 40.0, .dur = 10.0, .level = 0, .id = 2},
  }));
  timeline.set_selected_event_index_for_test(1);

  int emitted_index = -1;
  std::vector<std::string> emitted_operands;
  std::vector<std::string> emitted_consumers;
  int emit_count = 0;
  timeline.set_event_callback(
      [&](absl::string_view type, const EventData& detail) {
        if (type == kSelectedEventDependencies) {
          ++emit_count;
          emitted_index = std::any_cast<int>(detail.at(kEventSelectedIndex));
          emitted_operands = std::any_cast<std::vector<std::string>>(
              detail.at(kSelectedEventDependenciesOperands));
          emitted_consumers = std::any_cast<std::vector<std::string>>(
              detail.at(kSelectedEventDependenciesConsumers));
        }
      });

  timeline.SetSelectedEventDependencies({"operand"}, {"consumer"});
  EXPECT_EQ(emit_count, 1);
  EXPECT_EQ(emitted_index, 1);
  EXPECT_THAT(emitted_operands, ElementsAre("operand"));
  EXPECT_THAT(emitted_consumers, IsEmpty());

  // Refetch drops operand and brings consumer into the loaded timeline.
  timeline.SetTimelineData(MakeDependencyTestData({
      {.name = "selected", .start = 40.0, .dur = 10.0, .level = 0, .id = 2},
      {.name = "consumer", .start = 60.0, .dur = 10.0, .level = 0, .id = 3},
  }));
  EXPECT_EQ(emit_count, 2);
  EXPECT_EQ(emitted_index, 0);
  EXPECT_THAT(emitted_operands, IsEmpty());
  EXPECT_THAT(emitted_consumers, ElementsAre("consumer"));

  // Refetch where selected event itself is no longer present.
  timeline.SetTimelineData(MakeDependencyTestData({
      {.name = "consumer", .start = 60.0, .dur = 10.0, .level = 0, .id = 3},
  }));
  EXPECT_EQ(emit_count, 3);
  EXPECT_EQ(emitted_index, -1);
  EXPECT_THAT(emitted_operands, IsEmpty());
  EXPECT_THAT(emitted_consumers, IsEmpty());
}

TEST(TimelineTest,
     SelectedEventDependenciesDoNotEmitNegativeOneWhenRefetchSelectsAnotherOp) {
  ColorPalette palette = ColorPalette::Default();
  Timeline timeline(palette);
  const FlameChartTimelineData data = MakeDependencyTestData({
      {.name = "operand", .start = 0.0, .dur = 10.0, .level = 0, .id = 1},
      {.name = "selected", .start = 40.0, .dur = 10.0, .level = 0, .id = 2},
      {.name = "consumer", .start = 60.0, .dur = 10.0, .level = 0, .id = 3},
  });
  timeline.SetTimelineData(data);
  timeline.set_selected_event_index_for_test(1);

  int emit_count = 0;
  int emitted_index = -1;
  timeline.set_event_callback(
      [&](absl::string_view type, const EventData& detail) {
        if (type == kSelectedEventDependencies) {
          ++emit_count;
          emitted_index = std::any_cast<int>(detail.at(kEventSelectedIndex));
        }
      });

  timeline.SetSelectedEventDependencies({"operand"}, {"consumer"});
  EXPECT_EQ(emit_count, 1);
  EXPECT_EQ(emitted_index, 1);

  // Refetch switches `selected_event_index_` to another event ("consumer")
  // via active search result. `kSelectedEventDependencies` must not emit `-1`.
  timeline.set_search_results_for_test({{.event_id = 3, .name = "consumer"}});
  timeline.set_current_search_result_index_for_test(0);

  timeline.SetTimelineData(data);

  EXPECT_EQ(timeline.selected_event_index(), 2);
  EXPECT_EQ(emit_count, 1);
  EXPECT_EQ(emitted_index, 1);
}
}  // namespace
}  // namespace testing
}  // namespace traceviewer
