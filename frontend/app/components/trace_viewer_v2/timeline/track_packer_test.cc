#include "frontend/app/components/trace_viewer_v2/timeline/track_packer.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/btree_map.h"
#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "google/protobuf/arena.h"
#include "tsl/profiler/lib/context_types.h"
#include "frontend/app/components/trace_viewer_v2/timeline/timeline.h"
#include "frontend/app/components/trace_viewer_v2/trace_helper/proto_event_ref.h"
#include "frontend/app/components/trace_viewer_v2/trace_helper/trace_event.h"
#include "frontend/app/components/trace_viewer_v2/trace_helper/wire_trace_reader.h"
#include "plugin/xprof/protobuf/trace_data_response.pb.h"

namespace traceviewer {
namespace {

using ::testing::AllOf;
using ::testing::DoubleEq;
using ::testing::ElementsAre;
using ::testing::Eq;
using ::testing::Field;
using ::testing::IsEmpty;
using ::testing::Pair;
using ::testing::UnorderedElementsAre;

TEST(TrackPackerTest,
     BuildDisjointModuleSegmentsFlattensNestedAndAdjacentHalfOpenIntervals) {
  EXPECT_THAT(BuildDisjointModuleSegments({}), IsEmpty());

  const std::vector<ModuleInterval> raw = {
      {.start = 100.0, .end = 100.0, .module_index = 9},
      {.start = 200.0, .end = 150.0, .module_index = 8},
      {.start = 100.0, .end = 500.0, .module_index = 2},
      {.start = 50.0, .end = 250.0, .module_index = 1},
      {.start = 200.0, .end = 300.0, .module_index = 3},
      {.start = 500.0, .end = 700.0, .module_index = 4},
  };

  const std::vector<ModuleInterval> segments = BuildDisjointModuleSegments(raw);

  EXPECT_THAT(segments,
              ElementsAre(AllOf(Field(&ModuleInterval::start, DoubleEq(50.0)),
                                Field(&ModuleInterval::end, DoubleEq(250.0)),
                                Field(&ModuleInterval::module_index, Eq(1u))),
                          AllOf(Field(&ModuleInterval::start, DoubleEq(250.0)),
                                Field(&ModuleInterval::end, DoubleEq(500.0)),
                                Field(&ModuleInterval::module_index, Eq(2u))),
                          AllOf(Field(&ModuleInterval::start, DoubleEq(500.0)),
                                Field(&ModuleInterval::end, DoubleEq(700.0)),
                                Field(&ModuleInterval::module_index, Eq(4u)))));
}

TEST(TrackPackerTest, PackProtoEventsAssignsHalfOpenModulesAtExactBoundary) {
  const std::vector<std::string> pool = {"", "op_before", "op_boundary_zero",
                                         "op_boundary_pos", "op_at_end"};
  const auto get_str = [&pool](uint32_t ref) -> absl::string_view {
    return ref < pool.size() ? absl::string_view(pool[ref]) : "";
  };

  const std::vector<ModuleInterval> modules = BuildDisjointModuleSegments({
      {.start = 50.0, .end = 250.0, .module_index = 1},
      {.start = 100.0, .end = 500.0, .module_index = 2},
  });

  std::vector<TrackEvent> events = {
      {.ts = 249.0, .dur = 1.0, .name_index = 1},
      {.ts = 250.0, .dur = 0.0, .name_index = 2},
      {.ts = 250.0, .dur = 10.0, .name_index = 3},
      {.ts = 500.0, .dur = 5.0, .name_index = 4},
  };

  FlameChartTimelineData data;
  TimeBounds bounds;
  TraceInformation<ProtoEventRef> trace_info;
  std::vector<uint32_t> name_ref_to_op(pool.size(),
                                       std::numeric_limits<uint32_t>::max());
  std::vector<uint32_t> name_ref_to_module(
      pool.size(), std::numeric_limits<uint32_t>::max());
  int max_level = 0;

  PackProtoEvents(absl::MakeSpan(events), kXlaOps, /*is_async_track=*/false,
                  modules, /*default_module_index=*/0, /*start_level=*/0,
                  max_level, data, bounds, trace_info, get_str, name_ref_to_op,
                  name_ref_to_module);

  data.interned_string_pool = pool;
  EXPECT_THAT(data.entry_hlo_module_indices, IsEmpty());
  EXPECT_THAT(data.entry_hlo_op_indices, IsEmpty());
  EXPECT_THAT(data.sparse_entry_hlo_module_indices.runs(), IsEmpty());
  EXPECT_THAT(data.sparse_entry_hlo_op_indices.runs(), IsEmpty());
  EXPECT_THAT(
      std::vector<uint32_t>(
          {data.GetEntryHloModuleIndex(0), data.GetEntryHloModuleIndex(1),
           data.GetEntryHloModuleIndex(2), data.GetEntryHloModuleIndex(3)}),
      ElementsAre(1u, 2u, 2u, 0u));
  // At ts = 250.0, PackEvents sorts dur = 10.0 (name_index 3) before dur = 0.0
  // (name_index 2).
  EXPECT_THAT(std::vector<uint32_t>(
                  {data.GetEntryHloOpIndex(0), data.GetEntryHloOpIndex(1),
                   data.GetEntryHloOpIndex(2), data.GetEntryHloOpIndex(3)}),
              ElementsAre(1u, 3u, 2u, 4u));
  EXPECT_THAT(std::vector<absl::string_view>(
                  {data.GetEntryHloOp(0), data.GetEntryHloOp(1),
                   data.GetEntryHloOp(2), data.GetEntryHloOp(3)}),
              ElementsAre("op_before", "op_boundary_pos", "op_boundary_zero",
                          "op_at_end"));
}

TEST(TrackPackerTest, PackProtoEventsAssignsLevelsAndComputesFusedSelfTime) {
  const std::vector<std::string> pool = {"", "parent", "nested_child",
                                         "partial_overlap", "child_of_overlap"};
  const auto get_str = [&pool](uint32_t ref) -> absl::string_view {
    return ref < pool.size() ? absl::string_view(pool[ref]) : "";
  };

  std::vector<TrackEvent> events = {
      {.ts = 100.0, .dur = 200.0, .name_index = 1},
      {.ts = 120.0, .dur = 50.0, .name_index = 2},
      {.ts = 200.0, .dur = 150.0, .name_index = 3},
      {.ts = 300.0, .dur = 10.0, .name_index = 4},
  };

  FlameChartTimelineData data;
  TimeBounds bounds;
  TraceInformation<ProtoEventRef> trace_info;
  std::vector<uint32_t> name_ref_to_op(pool.size(),
                                       std::numeric_limits<uint32_t>::max());
  std::vector<uint32_t> name_ref_to_module(
      pool.size(), std::numeric_limits<uint32_t>::max());
  int max_level = 3;

  PackProtoEvents(absl::MakeSpan(events), "Stream #0",
                  /*is_async_track=*/false, /*module_intervals=*/{},
                  /*default_module_index=*/0, /*start_level=*/3, max_level,
                  data, bounds, trace_info, get_str, name_ref_to_op,
                  name_ref_to_module);

  EXPECT_THAT(max_level, Eq(4));
  EXPECT_THAT(data.entry_levels, ElementsAre(3u, 4u, 4u, 3u));
  EXPECT_THAT(data.entry_self_times, IsEmpty());
  // Only the non-leaf parent (index 0) allocates a sparse self-time record.
  // "partial_overlap" starts inside "parent" but ends after it, so, as in the
  // JSON path's PopulateThreadTrackEvents, it neither subtracts from "parent"
  // nor becomes a parent itself. "child_of_overlap" therefore has no open
  // parent ("parent" has ended at ts 300) and subtracts from nothing.
  EXPECT_THAT(data.sparse_self_times.size(), Eq(1u));
  EXPECT_THAT(std::vector<Microseconds>(
                  {data.GetEntrySelfTime(0), data.GetEntrySelfTime(1),
                   data.GetEntrySelfTime(2), data.GetEntrySelfTime(3)}),
              ElementsAre(DoubleEq(150.0), DoubleEq(50.0), DoubleEq(150.0),
                          DoubleEq(10.0)));
  EXPECT_THAT(bounds.min, DoubleEq(100.0));
  EXPECT_THAT(bounds.max, DoubleEq(350.0));
}

TEST(TrackPackerTest, PackProtoEventsDropsLevelsExceedingUint16Range) {
  const auto get_str = [](uint32_t) -> absl::string_view { return "ev"; };

  const int last_valid_level =
      static_cast<int>(std::numeric_limits<uint16_t>::max()) - 1;
  std::vector<TrackEvent> events = {
      {.ts = 10.0, .dur = 20.0, .name_index = 1},
      {.ts = 15.0, .dur = 10.0, .name_index = 1},
  };

  FlameChartTimelineData data;
  TimeBounds bounds;
  TraceInformation<ProtoEventRef> trace_info;
  std::vector<uint32_t> name_ref_to_op(2, std::numeric_limits<uint32_t>::max());
  std::vector<uint32_t> name_ref_to_module(
      2, std::numeric_limits<uint32_t>::max());
  int max_level = last_valid_level;

  PackProtoEvents(absl::MakeSpan(events), "T0", /*is_async_track=*/false,
                  /*module_intervals=*/{}, /*default_module_index=*/0,
                  /*start_level=*/last_valid_level, max_level, data, bounds,
                  trace_info, get_str, name_ref_to_op, name_ref_to_module);

  EXPECT_THAT(max_level, Eq(last_valid_level));
  EXPECT_THAT(data.entry_levels,
              ElementsAre(static_cast<uint16_t>(last_valid_level)));
  EXPECT_THAT(data.entry_start_times, ElementsAre(DoubleEq(10.0)));
  EXPECT_THAT(data.dropped_level_cap_events, Eq(1u));
}

TEST(TrackPackerTest,
     UnpackAndPackProtoTrackMergesMultiSeriesAndRecordsFlowEndpoints) {
  const std::vector<std::string> pool = {"", "late_ev", "early_ev",
                                         "tpu_launch"};
  const auto get_str = [&pool](uint32_t ref) -> absl::string_view {
    return ref < pool.size() ? absl::string_view(pool[ref]) : "";
  };

  xprof::TraceEventSeries series_a;
  series_a.mutable_metadata()->set_process_id(7);
  series_a.mutable_metadata()->set_thread_id(2);
  series_a.mutable_metadata()->set_name_ref(1);
  series_a.add_deltas(200ULL * 1000000ULL);
  series_a.add_durations(30ULL * 1000000ULL);
  xprof::TraceEventMetadata* meta_a = series_a.add_event_metadata();
  meta_a->set_serial(42);
  meta_a->set_flow_id(99);
  meta_a->set_flow_category(3);
  meta_a->set_group_id(7);

  xprof::TraceEventSeries series_b;
  series_b.mutable_metadata()->set_process_id(7);
  series_b.mutable_metadata()->set_thread_id(2);
  series_b.add_deltas(100ULL * 1000000ULL);
  series_b.add_durations(50ULL * 1000000ULL);
  series_b.add_name_refs(2);

  const std::vector<const xprof::TraceEventSeries*> series_list = {&series_a,
                                                                   &series_b};

  FlameChartTimelineData data;
  TimeBounds bounds;
  TraceInformation<ProtoEventRef> trace_info;
  trace_info.name_interner.SetInitialPool(pool);
  absl::btree_map<int, int> flow_category_counts;
  std::vector<TrackEvent> track_buffer;
  std::vector<uint32_t> name_ref_to_op(pool.size(),
                                       std::numeric_limits<uint32_t>::max());
  std::vector<uint32_t> name_ref_to_module(
      pool.size(), std::numeric_limits<uint32_t>::max());
  int max_level = 0;

  UnpackAndPackProtoTrack(
      /*pid=*/7, /*tid=*/2, "Thread_2", series_list, get_str,
      static_cast<uint32_t>(pool.size()), /*module_intervals=*/{},
      /*default_module_index=*/0, /*start_level=*/0, max_level, data, bounds,
      trace_info, flow_category_counts, track_buffer, name_ref_to_op,
      name_ref_to_module);

  EXPECT_THAT(data.entry_start_times,
              ElementsAre(DoubleEq(100.0), DoubleEq(200.0)));
  EXPECT_THAT(data.entry_name_indices, ElementsAre(2u, 1u));
  EXPECT_THAT(data.non_zero_uids_by_event_index,
              UnorderedElementsAre(Pair(1, 42u)));
  EXPECT_THAT(
      data.sparse_entry_group_ids.runs(),
      ElementsAre(GroupIdRuns::Run{.begin = 1, .end = 2, .group_id = 7}));
  EXPECT_THAT(data.flow_ids_by_event_index,
              UnorderedElementsAre(Pair(1, 99ULL)));
  EXPECT_THAT(trace_info.name_interner.TakePool(),
              ElementsAre("", "late_ev", "early_ev", "tpu_launch"));
  const auto it_flow = trace_info.flow_events_by_id.find(99ULL);
  ASSERT_NE(it_flow, trace_info.flow_events_by_id.end());
  EXPECT_THAT(
      it_flow->second,
      ElementsAre(AllOf(Field(&FlowEndpoint::pid, Eq(7u)),
                        Field(&FlowEndpoint::tid, Eq(2u)),
                        Field(&FlowEndpoint::ts, DoubleEq(200.0)),
                        Field(&FlowEndpoint::category,
                              Eq(tsl::profiler::ContextType::kTpuLaunch)))));
}

TEST(TrackPackerTest,
     UnpackAndPackProtoTrackDecodesSparseMetadataRleGroupIdsAndCachesHloRefs) {
  const std::vector<std::string> pool = {"",   "k0",       "k1",
                                         "k2", "fusion.1", "module_main(1)"};
  int get_str_calls_for_module = 0;
  const auto get_str =
      [&pool, &get_str_calls_for_module](uint32_t ref) -> absl::string_view {
    if (ref == 5) {
      ++get_str_calls_for_module;
    }
    return ref < pool.size() ? absl::string_view(pool[ref]) : "";
  };

  xprof::TraceEventSeries series;
  series.mutable_metadata()->set_process_id(1);
  series.mutable_metadata()->set_thread_id(1);
  // Event 0: no sparse metadata, RLE group_id = -1 (none)
  series.add_deltas(10ULL * 1000000ULL);
  series.add_durations(5ULL * 1000000ULL);
  series.add_name_refs(1);
  // Event 1: sparse metadata at index 1 (hlo_op_ref=4, hlo_module_ref=5),
  // RLE group_id = 12
  series.add_deltas(10ULL * 1000000ULL);
  series.add_durations(5ULL * 1000000ULL);
  series.add_name_refs(2);
  series.add_event_metadata_indices(1);
  xprof::TraceEventMetadata* m1 = series.add_event_metadata();
  m1->set_serial(101);
  m1->set_hlo_op_ref(4);
  m1->set_hlo_module_ref(5);
  // Event 2: sparse metadata at index 2 (same hlo_op_ref=4, hlo_module_ref=5),
  // RLE group_id = 12
  series.add_deltas(10ULL * 1000000ULL);
  series.add_durations(5ULL * 1000000ULL);
  series.add_name_refs(3);
  series.add_event_metadata_indices(2);
  xprof::TraceEventMetadata* m2 = series.add_event_metadata();
  m2->set_hlo_op_ref(4);
  m2->set_hlo_module_ref(5);

  series.add_group_id_run_values(-1);
  series.add_group_id_run_lengths(1);
  series.add_group_id_run_values(12);
  series.add_group_id_run_lengths(2);

  const std::vector<const xprof::TraceEventSeries*> series_list = {&series};

  FlameChartTimelineData data;
  TimeBounds bounds;
  TraceInformation<ProtoEventRef> trace_info;
  absl::btree_map<int, int> flow_category_counts;
  std::vector<TrackEvent> track_buffer;
  std::vector<uint32_t> name_ref_to_op(pool.size(),
                                       std::numeric_limits<uint32_t>::max());
  std::vector<uint32_t> name_ref_to_module(
      pool.size(), std::numeric_limits<uint32_t>::max());
  int max_level = 0;

  UnpackAndPackProtoTrack(
      /*pid=*/1, /*tid=*/1, "Stream #0", series_list, get_str,
      static_cast<uint32_t>(pool.size()), /*module_intervals=*/{},
      /*default_module_index=*/0, /*start_level=*/0, max_level, data, bounds,
      trace_info, flow_category_counts, track_buffer, name_ref_to_op,
      name_ref_to_module);

  EXPECT_THAT(
      data.sparse_entry_group_ids.runs(),
      ElementsAre(GroupIdRuns::Run{.begin = 1, .end = 3, .group_id = 12}));
  EXPECT_THAT(data.non_zero_uids_by_event_index,
              UnorderedElementsAre(Pair(1, 101u)));
  EXPECT_THAT(std::vector<uint32_t>({data.GetEntryHloOpIndex(0),
                                     data.GetEntryHloOpIndex(1),
                                     data.GetEntryHloOpIndex(2)}),
              ElementsAre(0u, 1u, 1u));
  EXPECT_THAT(std::vector<uint32_t>({data.GetEntryHloModuleIndex(0),
                                     data.GetEntryHloModuleIndex(1),
                                     data.GetEntryHloModuleIndex(2)}),
              ElementsAre(0u, 1u, 1u));
  // Module interning is cached by name_ref (L5-3), so get_str(5) runs once.
  EXPECT_THAT(get_str_calls_for_module, Eq(1));
}

TEST(TrackPackerTest,
     PackProtoEventsDroppedLevelExceedingUint16PreservesNeighborGroupIds) {
  const auto get_str = [](uint32_t) -> absl::string_view { return "ev"; };

  const int last_valid_level =
      static_cast<int>(std::numeric_limits<uint16_t>::max()) - 1;
  std::vector<TrackEvent> events = {
      {.ts = 10.0, .dur = 20.0, .group_id = 5, .name_index = 1},
      {.ts = 15.0, .dur = 5.0, .group_id = 99, .name_index = 1},
      {.ts = 40.0, .dur = 10.0, .group_id = 5, .name_index = 1},
  };

  FlameChartTimelineData data;
  data.interned_string_pool = {"", "ev"};
  TimeBounds bounds;
  TraceInformation<ProtoEventRef> trace_info;
  std::vector<uint32_t> name_ref_to_op(2, std::numeric_limits<uint32_t>::max());
  std::vector<uint32_t> name_ref_to_module(
      2, std::numeric_limits<uint32_t>::max());
  int max_level = last_valid_level;

  PackProtoEvents(absl::MakeSpan(events), "T0", /*is_async_track=*/false,
                  /*module_intervals=*/{}, /*default_module_index=*/0,
                  /*start_level=*/last_valid_level, max_level, data, bounds,
                  trace_info, get_str, name_ref_to_op, name_ref_to_module);

  ASSERT_THAT(data.entry_start_times,
              ElementsAre(DoubleEq(10.0), DoubleEq(40.0)));
  const absl::flat_hash_map<std::string, std::string> args0 =
      data.GetEntryArgs(0);
  const absl::flat_hash_map<std::string, std::string> args1 =
      data.GetEntryArgs(1);
  EXPECT_THAT(args0.at("group_id"), Eq("5"));
  EXPECT_THAT(args1.at("group_id"), Eq("5"));
  EXPECT_THAT(
      data.sparse_entry_group_ids.runs(),
      ElementsAre(GroupIdRuns::Run{.begin = 0, .end = 2, .group_id = 5}));
}

TEST(TrackPackerTest,
     PackProtoEventsPreservesEqualGroupIdsAcrossTrackBoundary) {
  const auto get_str = [](uint32_t) -> absl::string_view { return "ev"; };

  std::vector<TrackEvent> track0_events = {
      {.ts = 10.0, .dur = 5.0, .group_id = 11, .name_index = 1},
  };
  std::vector<TrackEvent> track1_events = {
      {.ts = 20.0, .dur = 5.0, .group_id = 11, .name_index = 1},
  };

  FlameChartTimelineData data;
  data.interned_string_pool = {"", "ev"};
  TimeBounds bounds;
  TraceInformation<ProtoEventRef> trace_info;
  std::vector<uint32_t> name_ref_to_op(2, std::numeric_limits<uint32_t>::max());
  std::vector<uint32_t> name_ref_to_module(
      2, std::numeric_limits<uint32_t>::max());
  int max_level0 = 0;
  int max_level1 = 1;

  PackProtoEvents(absl::MakeSpan(track0_events), "T0", /*is_async_track=*/false,
                  /*module_intervals=*/{}, /*default_module_index=*/0,
                  /*start_level=*/0, max_level0, data, bounds, trace_info,
                  get_str, name_ref_to_op, name_ref_to_module);
  PackProtoEvents(absl::MakeSpan(track1_events), "T1", /*is_async_track=*/false,
                  /*module_intervals=*/{}, /*default_module_index=*/0,
                  /*start_level=*/1, max_level1, data, bounds, trace_info,
                  get_str, name_ref_to_op, name_ref_to_module);

  ASSERT_THAT(data.entry_start_times,
              ElementsAre(DoubleEq(10.0), DoubleEq(20.0)));
  const absl::flat_hash_map<std::string, std::string> args0 =
      data.GetEntryArgs(0);
  const absl::flat_hash_map<std::string, std::string> args1 =
      data.GetEntryArgs(1);
  EXPECT_THAT(args0.at("group_id"), Eq("11"));
  EXPECT_THAT(args1.at("group_id"), Eq("11"));
  EXPECT_THAT(
      data.sparse_entry_group_ids.runs(),
      ElementsAre(GroupIdRuns::Run{.begin = 0, .end = 2, .group_id = 11}));
}

TEST(TrackPackerTest,
     ZeroDurationChildDoesNotAllocateSparseSelfTimeRecordForParent) {
  const auto get_str = [](uint32_t) -> absl::string_view { return "ev"; };
  std::vector<TrackEvent> events = {
      {.ts = 100.0, .dur = 50.0, .name_index = 1},
      {.ts = 120.0, .dur = 0.0, .name_index = 1},
  };

  FlameChartTimelineData data;
  TimeBounds bounds;
  TraceInformation<ProtoEventRef> trace_info;
  std::vector<uint32_t> name_ref_to_op(2, std::numeric_limits<uint32_t>::max());
  std::vector<uint32_t> name_ref_to_module(
      2, std::numeric_limits<uint32_t>::max());
  int max_level = 0;

  PackProtoEvents(absl::MakeSpan(events), "T0", /*is_async_track=*/false,
                  /*module_intervals=*/{}, /*default_module_index=*/0,
                  /*start_level=*/0, max_level, data, bounds, trace_info,
                  get_str, name_ref_to_op, name_ref_to_module);

  EXPECT_THAT(data.sparse_self_times.size(), Eq(0u));
  EXPECT_THAT(data.GetEntrySelfTime(0), DoubleEq(50.0));
  EXPECT_THAT(data.GetEntrySelfTime(1), DoubleEq(0.0));
}

TEST(TrackPackerTest,
     ChunkedSingleSeriesUnpackAndWireStreamingMatchMonolithicPacking) {
  const std::vector<std::string> pool = {"",          "outer_step", "inner_a",
                                         "tie_short", "tie_long",   "tail"};
  const auto get_str = [&pool](uint32_t ref) -> absl::string_view {
    return ref < pool.size() ? absl::string_view(pool[ref]) : "";
  };

  xprof::TraceEventSeries series;
  series.mutable_metadata()->set_process_id(1);
  series.mutable_metadata()->set_thread_id(5);
  // Event 0: [100, 500] outer_step
  series.add_deltas(100ULL * 1000000ULL);
  series.add_durations(400ULL * 1000000ULL);
  series.add_name_refs(1);
  // Event 1: [150, 200] inner_a
  series.add_deltas(50ULL * 1000000ULL);
  series.add_durations(50ULL * 1000000ULL);
  series.add_name_refs(2);
  // Event 2 & 3 at ts=250 with tie_short (dur=10) before tie_long (dur=80):
  // chunk boundary at max_chunk_size=2 falls between Event 2 and Event 3,
  // which must NOT split because delta==0!
  series.add_deltas(100ULL * 1000000ULL);
  series.add_durations(10ULL * 1000000ULL);
  series.add_name_refs(3);
  series.add_deltas(0ULL);
  series.add_durations(80ULL * 1000000ULL);
  series.add_name_refs(4);
  // Event 4: [400, 420] tail
  series.add_deltas(150ULL * 1000000ULL);
  series.add_durations(20ULL * 1000000ULL);
  series.add_name_refs(5);

  const std::vector<const xprof::TraceEventSeries*> series_list = {&series};

  auto run_monolithic = [&]() {
    FlameChartTimelineData d;
    TimeBounds b;
    TraceInformation<ProtoEventRef> ti;
    absl::btree_map<int, int> fc;
    std::vector<TrackEvent> buf;
    std::vector<uint32_t> op(pool.size(), std::numeric_limits<uint32_t>::max());
    std::vector<uint32_t> mod(pool.size(),
                              std::numeric_limits<uint32_t>::max());
    int max_lvl = 0;
    UnpackAndPackProtoTrack(1, 5, kXlaOps, series_list, get_str,
                            static_cast<uint32_t>(pool.size()), {}, 0, 0,
                            max_lvl, d, b, ti, fc, buf, op, mod,
                            /*max_chunk_size=*/65536);
    return d;
  };

  auto run_chunked = [&]() {
    FlameChartTimelineData d;
    TimeBounds b;
    TraceInformation<ProtoEventRef> ti;
    absl::btree_map<int, int> fc;
    std::vector<TrackEvent> buf;
    std::vector<uint32_t> op(pool.size(), std::numeric_limits<uint32_t>::max());
    std::vector<uint32_t> mod(pool.size(),
                              std::numeric_limits<uint32_t>::max());
    int max_lvl = 0;
    UnpackAndPackProtoTrack(1, 5, kXlaOps, series_list, get_str,
                            static_cast<uint32_t>(pool.size()), {}, 0, 0,
                            max_lvl, d, b, ti, fc, buf, op, mod,
                            /*max_chunk_size=*/2);
    return d;
  };

  auto run_wire = [&]() {
    std::string wire_bytes;
    EXPECT_TRUE(series.SerializeToString(&wire_bytes));
    WireSeriesView view;
    uint64_t max_end_ps = 0;
    EXPECT_TRUE(TryParseWireSeriesView(wire_bytes, view, &max_end_ps));
    EXPECT_THAT(max_end_ps, Eq(500ULL * 1000000ULL));

    FlameChartTimelineData d;
    TimeBounds b;
    TraceInformation<ProtoEventRef> ti;
    absl::btree_map<int, int> fc;
    std::vector<TrackEvent> buf;
    std::vector<uint32_t> op(pool.size(), std::numeric_limits<uint32_t>::max());
    std::vector<uint32_t> mod(pool.size(),
                              std::numeric_limits<uint32_t>::max());
    int max_lvl = 0;
    UnpackAndPackProtoSeriesWire(1, 5, kXlaOps, view, get_str,
                                 static_cast<uint32_t>(pool.size()), {}, 0, 0,
                                 max_lvl, d, b, ti, fc, buf, op, mod,
                                 /*max_chunk_size=*/2);
    return d;
  };

  const FlameChartTimelineData mono = run_monolithic();
  const FlameChartTimelineData chunked = run_chunked();
  const FlameChartTimelineData wire = run_wire();

  EXPECT_THAT(chunked.entry_start_times, Eq(mono.entry_start_times));
  EXPECT_THAT(chunked.entry_total_times, Eq(mono.entry_total_times));
  EXPECT_THAT(chunked.entry_levels, Eq(mono.entry_levels));
  EXPECT_THAT(chunked.entry_name_indices, Eq(mono.entry_name_indices));

  EXPECT_THAT(wire.entry_start_times, Eq(mono.entry_start_times));
  EXPECT_THAT(wire.entry_total_times, Eq(mono.entry_total_times));
  EXPECT_THAT(wire.entry_levels, Eq(mono.entry_levels));
  EXPECT_THAT(wire.entry_name_indices, Eq(mono.entry_name_indices));

  for (size_t i = 0; i < mono.entry_start_times.size(); ++i) {
    EXPECT_THAT(chunked.GetEntrySelfTime(i),
                DoubleEq(mono.GetEntrySelfTime(i)));
    EXPECT_THAT(wire.GetEntrySelfTime(i), DoubleEq(mono.GetEntrySelfTime(i)));
    EXPECT_THAT(chunked.GetEntryHloOpIndex(i), Eq(mono.GetEntryHloOpIndex(i)));
    EXPECT_THAT(wire.GetEntryHloOpIndex(i), Eq(mono.GetEntryHloOpIndex(i)));
  }
}

TEST(TrackPackerTest,
     CrossChunkNonMonotonicSeriesFallsBackToFullTrackSortAndFusesMaxEndPs) {
  const std::vector<std::string> pool = {"", "ev_100", "ev_200", "ev_50",
                                         "ev_120"};
  const auto get_str = [&pool](uint32_t ref) -> absl::string_view {
    return ref < pool.size() ? absl::string_view(pool[ref]) : "";
  };

  // Chunk 0 (events 0, 1): ts = 100us (dur 200us), ts = 200us (dur 20us).
  // Chunk 1 (events 2, 3): negative delta -150us -> ts = 50us (dur 10us),
  //                        +70us -> ts = 120us (dur 30us, child of ev_100).
  xprof::TraceEventSeries series;
  series.mutable_metadata()->set_process_id(1);
  series.mutable_metadata()->set_thread_id(5);
  series.add_deltas(100ULL * 1000000ULL);
  series.add_durations(200ULL * 1000000ULL);
  series.add_name_refs(1);
  series.add_deltas(100ULL * 1000000ULL);
  series.add_durations(20ULL * 1000000ULL);
  series.add_name_refs(2);
  series.add_deltas(static_cast<uint64_t>(int64_t{-150LL * 1000000LL}));
  series.add_durations(10ULL * 1000000ULL);
  series.add_name_refs(3);
  series.add_deltas(70ULL * 1000000ULL);
  series.add_durations(30ULL * 1000000ULL);
  series.add_name_refs(4);

  const std::vector<const xprof::TraceEventSeries*> series_list = {&series};

  FlameChartTimelineData proto_data;
  TimeBounds proto_bounds;
  TraceInformation<ProtoEventRef> proto_ti;
  absl::btree_map<int, int> proto_fc;
  std::vector<TrackEvent> proto_buf;
  std::vector<uint32_t> proto_op(pool.size(),
                                 std::numeric_limits<uint32_t>::max());
  std::vector<uint32_t> proto_mod(pool.size(),
                                  std::numeric_limits<uint32_t>::max());
  int proto_max_lvl = 0;
  uint64_t proto_max_end_ps = 0;
  UnpackAndPackProtoTrack(1, 5, "T5", series_list, get_str,
                          static_cast<uint32_t>(pool.size()), {}, 0, 0,
                          proto_max_lvl, proto_data, proto_bounds, proto_ti,
                          proto_fc, proto_buf, proto_op, proto_mod,
                          /*max_chunk_size=*/2, &proto_max_end_ps);

  std::string wire_bytes;
  ASSERT_TRUE(series.SerializeToString(&wire_bytes));
  WireSeriesView view;
  ASSERT_TRUE(TryParseWireSeriesView(wire_bytes, view));
  EXPECT_THAT(view.deltas_count, Eq(4u));

  FlameChartTimelineData wire_data;
  TimeBounds wire_bounds;
  TraceInformation<ProtoEventRef> wire_ti;
  absl::btree_map<int, int> wire_fc;
  std::vector<TrackEvent> wire_buf;
  std::vector<uint32_t> wire_op(pool.size(),
                                std::numeric_limits<uint32_t>::max());
  std::vector<uint32_t> wire_mod(pool.size(),
                                 std::numeric_limits<uint32_t>::max());
  int wire_max_lvl = 0;
  uint64_t wire_max_end_ps = 0;
  google::protobuf::Arena arena;
  UnpackAndPackProtoSeriesWire(1, 5, "T5", view, get_str,
                               static_cast<uint32_t>(pool.size()), {}, 0, 0,
                               wire_max_lvl, wire_data, wire_bounds, wire_ti,
                               wire_fc, wire_buf, wire_op, wire_mod,
                               /*max_chunk_size=*/2, &wire_max_end_ps, &arena);

  EXPECT_THAT(proto_max_end_ps, Eq(300ULL * 1000000ULL));
  EXPECT_THAT(wire_max_end_ps, Eq(300ULL * 1000000ULL));
  EXPECT_THAT(proto_data.entry_start_times,
              ElementsAre(DoubleEq(50.0), DoubleEq(100.0), DoubleEq(120.0),
                          DoubleEq(200.0)));
  EXPECT_THAT(wire_data.entry_start_times, Eq(proto_data.entry_start_times));
  EXPECT_THAT(wire_data.entry_levels, Eq(proto_data.entry_levels));
  // ev_100 (at index 1, total 200us) has two nested children (ev_120: 30us,
  // ev_200: 20us), so its self-time must be 150us.
  EXPECT_THAT(proto_data.GetEntrySelfTime(1), DoubleEq(150.0));
  EXPECT_THAT(wire_data.GetEntrySelfTime(1), DoubleEq(150.0));
}

TEST(TrackPackerTest,
     CrossChunkEqualTimestampLongerDurationAndDelayedNegativeDeltaMatchSort) {
  const std::vector<std::string> pool = {"",         "ev_0",     "short_200",
                                         "long_200", "late_250", "early_80"};
  const auto get_str = [&pool](uint32_t ref) -> absl::string_view {
    return ref < pool.size() ? absl::string_view(pool[ref]) : "";
  };

  // Case A: Equal timestamp at chunk boundary (index 2 has delta=0, ts=200us,
  // dur=100us > index 1's dur=20us).
  xprof::TraceEventSeries eq_series;
  eq_series.mutable_metadata()->set_process_id(1);
  eq_series.mutable_metadata()->set_thread_id(5);
  eq_series.add_deltas(100ULL * 1000000ULL);
  eq_series.add_durations(50ULL * 1000000ULL);
  eq_series.add_name_refs(1);
  eq_series.add_deltas(100ULL * 1000000ULL);
  eq_series.add_durations(20ULL * 1000000ULL);
  eq_series.add_name_refs(2);
  eq_series.add_deltas(0ULL);
  eq_series.add_durations(100ULL * 1000000ULL);
  eq_series.add_name_refs(3);
  eq_series.add_deltas(50ULL * 1000000ULL);
  eq_series.add_durations(10ULL * 1000000ULL);
  eq_series.add_name_refs(4);

  std::string eq_wire;
  ASSERT_TRUE(eq_series.SerializeToString(&eq_wire));
  WireSeriesView eq_view;
  ASSERT_TRUE(TryParseWireSeriesView(eq_wire, eq_view));

  FlameChartTimelineData eq_data;
  TimeBounds eq_bounds;
  TraceInformation<ProtoEventRef> eq_ti;
  absl::btree_map<int, int> eq_fc;
  std::vector<TrackEvent> eq_buf;
  std::vector<uint32_t> eq_op(pool.size(),
                              std::numeric_limits<uint32_t>::max());
  std::vector<uint32_t> eq_mod(pool.size(),
                               std::numeric_limits<uint32_t>::max());
  int eq_max_lvl = 0;
  UnpackAndPackProtoSeriesWire(1, 5, "T5", eq_view, get_str,
                               static_cast<uint32_t>(pool.size()), {}, 0, 0,
                               eq_max_lvl, eq_data, eq_bounds, eq_ti, eq_fc,
                               eq_buf, eq_op, eq_mod, /*max_chunk_size=*/2);
  // long_200 (name_ref 3, dur 100) must precede short_200 (name_ref 2, dur 20)
  // and have self-time 100 - 20 - 10 = 70us.
  EXPECT_THAT(eq_data.entry_name_indices, ElementsAre(1u, 3u, 2u, 4u));
  EXPECT_THAT(eq_data.GetEntrySelfTime(1), DoubleEq(70.0));

  // Case B: Chunk 1 starts with a positive delta (+50us -> 250us), then has a
  // negative delta at index 3 (-170us -> 80us). Lookahead must prevent early
  // commit of Chunk 0.
  xprof::TraceEventSeries delayed_neg;
  delayed_neg.mutable_metadata()->set_process_id(1);
  delayed_neg.mutable_metadata()->set_thread_id(5);
  delayed_neg.add_deltas(100ULL * 1000000ULL);
  delayed_neg.add_durations(200ULL * 1000000ULL);
  delayed_neg.add_name_refs(1);
  delayed_neg.add_deltas(100ULL * 1000000ULL);
  delayed_neg.add_durations(20ULL * 1000000ULL);
  delayed_neg.add_name_refs(2);
  delayed_neg.add_deltas(50ULL * 1000000ULL);
  delayed_neg.add_durations(10ULL * 1000000ULL);
  delayed_neg.add_name_refs(4);
  delayed_neg.add_deltas(static_cast<uint64_t>(int64_t{-170LL * 1000000LL}));
  delayed_neg.add_durations(10ULL * 1000000ULL);
  delayed_neg.add_name_refs(5);

  std::string dn_wire;
  ASSERT_TRUE(delayed_neg.SerializeToString(&dn_wire));
  WireSeriesView dn_view;
  ASSERT_TRUE(TryParseWireSeriesView(dn_wire, dn_view));

  FlameChartTimelineData dn_data;
  TimeBounds dn_bounds;
  TraceInformation<ProtoEventRef> dn_ti;
  absl::btree_map<int, int> dn_fc;
  std::vector<TrackEvent> dn_buf;
  std::vector<uint32_t> dn_op(pool.size(),
                              std::numeric_limits<uint32_t>::max());
  std::vector<uint32_t> dn_mod(pool.size(),
                               std::numeric_limits<uint32_t>::max());
  int dn_max_lvl = 0;
  UnpackAndPackProtoSeriesWire(1, 5, "T5", dn_view, get_str,
                               static_cast<uint32_t>(pool.size()), {}, 0, 0,
                               dn_max_lvl, dn_data, dn_bounds, dn_ti, dn_fc,
                               dn_buf, dn_op, dn_mod, /*max_chunk_size=*/2);
  EXPECT_THAT(dn_data.entry_start_times,
              ElementsAre(DoubleEq(80.0), DoubleEq(100.0), DoubleEq(200.0),
                          DoubleEq(250.0)));
}

TEST(TrackPackerTest,
     MultiSegmentWireSeriesStreamsChunksAndPreservesCrossSegmentHierarchy) {
  const std::vector<std::string> pool = {"", "parent_seg0", "child_seg0",
                                         "child_seg1", "tail_seg1"};
  const auto get_str = [&pool](uint32_t ref) -> absl::string_view {
    return ref < pool.size() ? absl::string_view(pool[ref]) : "";
  };

  xprof::TraceEventSeries seg0;
  seg0.mutable_metadata()->set_process_id(1);
  seg0.mutable_metadata()->set_thread_id(5);
  seg0.add_deltas(100ULL * 1000000ULL);
  seg0.add_durations(300ULL * 1000000ULL);
  seg0.add_name_refs(1);
  seg0.add_deltas(20ULL * 1000000ULL);
  seg0.add_durations(30ULL * 1000000ULL);
  seg0.add_name_refs(2);

  xprof::TraceEventSeries seg1;
  seg1.mutable_metadata()->set_process_id(1);
  seg1.mutable_metadata()->set_thread_id(5);
  seg1.add_deltas(200ULL * 1000000ULL);
  seg1.add_durations(50ULL * 1000000ULL);
  seg1.add_name_refs(3);
  seg1.add_deltas(300ULL * 1000000ULL);
  seg1.add_durations(20ULL * 1000000ULL);
  seg1.add_name_refs(4);

  std::string wire0, wire1;
  ASSERT_TRUE(seg0.SerializeToString(&wire0));
  ASSERT_TRUE(seg1.SerializeToString(&wire1));
  WireSeriesView v0, v1;
  ASSERT_TRUE(TryParseWireSeriesView(wire0, v0));
  ASSERT_TRUE(TryParseWireSeriesView(wire1, v1));
  const std::vector<const WireSeriesView*> views = {&v0, &v1};

  FlameChartTimelineData data;
  TimeBounds bounds;
  TraceInformation<ProtoEventRef> ti;
  absl::btree_map<int, int> fc;
  std::vector<TrackEvent> buf;
  std::vector<uint32_t> op(pool.size(), std::numeric_limits<uint32_t>::max());
  std::vector<uint32_t> mod(pool.size(), std::numeric_limits<uint32_t>::max());
  int max_lvl = 0;
  UnpackAndPackProtoSeriesWire(1, 5, "T5", views, get_str,
                               static_cast<uint32_t>(pool.size()), {}, 0, 0,
                               max_lvl, data, bounds, ti, fc, buf, op, mod,
                               /*max_chunk_size=*/2);

  // Buffer capacity is bounded by max_chunk_size (2) rather than total track
  // events (4), and parent_seg0 (300us) subtracts both child_seg0 (30us) and
  // child_seg1 (50us) across segment boundaries -> 220us self-time.
  EXPECT_THAT(buf.capacity(), Eq(2u));
  EXPECT_THAT(data.entry_levels, ElementsAre(0u, 1u, 1u, 0u));
  EXPECT_THAT(data.GetEntrySelfTime(0), DoubleEq(220.0));
}

}  // namespace
}  // namespace traceviewer
