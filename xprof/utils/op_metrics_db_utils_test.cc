/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "xprof/utils/op_metrics_db_utils.h"

#include <cstdint>
#include <string>
#include <vector>

#include "net/proto2/contrib/parse_proto/parse_text_proto.h"
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "xla/tsl/profiler/utils/tf_xplane_visitor.h"
#include "xla/tsl/profiler/utils/xplane_builder.h"
#include "xla/tsl/profiler/utils/xplane_schema.h"
#include "xla/tsl/profiler/utils/xplane_visitor.h"
#include "tsl/profiler/protobuf/xplane.pb.h"
#include "plugin/xprof/protobuf/op_metrics.pb.h"
#include "plugin/xprof/protobuf/source_info.pb.h"

namespace tensorflow {
namespace profiler {
namespace {
#if defined(PLATFORM_GOOGLE)
using ::testing::EqualsProto;
using ::testing::proto::IgnoringRepeatedFieldOrdering;
#endif
using ::google::protobuf::contrib::parse_proto::ParseTextProtoOrDie;
using ::tsl::profiler::StatType;
using ::tsl::profiler::XEventBuilder;
using ::tsl::profiler::XEventMetadata;
using ::tsl::profiler::XLineBuilder;
using ::tsl::profiler::XPlaneBuilder;
using ::tsl::profiler::XStatsBuilder;

constexpr double kMaxError = 1E-10;

TEST(OpMetricsDbTest, IdleTimeRatio) {
  OpMetricsDb metrics_db_0;
  metrics_db_0.set_total_time_ps(100000000);
  metrics_db_0.set_total_op_time_ps(60000000);
  EXPECT_NEAR(0.4, IdleTimeRatio(metrics_db_0), kMaxError);

  OpMetricsDb metrics_db_1;
  metrics_db_1.set_total_time_ps(200000000);
  metrics_db_1.set_total_op_time_ps(150000000);
  EXPECT_NEAR(0.25, IdleTimeRatio(metrics_db_1), kMaxError);

  OpMetricsDb metrics_db_2;
  metrics_db_1.set_total_time_ps(0);
  metrics_db_1.set_total_op_time_ps(0);
  EXPECT_NEAR(1.0, IdleTimeRatio(metrics_db_2), kMaxError);
}

TEST(OpMetricsDbTest, FromXEventHandlesMissingOccurrences) {
  XPlane raw_plane;
  XPlaneBuilder plane(&raw_plane);
  XLineBuilder line = plane.GetOrCreateLine(0);
  XEventMetadata* event_metadata = plane.GetOrCreateEventMetadata("metadata");
  event_metadata->set_display_name("display_name");
  XStatsBuilder<XEventMetadata> stats(event_metadata, &plane);
  stats.AddStatValue(
      *plane.GetOrCreateStatMetadata(GetStatTypeStr(StatType::kProgramId)), 1);
  stats.AddStatValue(
      *plane.GetOrCreateStatMetadata(GetStatTypeStr(StatType::kSymbolId)), 2);
  stats.AddStatValue(*plane.GetOrCreateStatMetadata(
                         GetStatTypeStr(StatType::kDeduplicatedName)),
                     "deduplicated_name");
  stats.AddStatValue(
      *plane.GetOrCreateStatMetadata(GetStatTypeStr(StatType::kTfOp)), "tf_op");
  stats.AddStatValue(
      *plane.GetOrCreateStatMetadata(GetStatTypeStr(StatType::kHloCategory)),
      "tf_op_category");
  stats.AddStatValue(
      *plane.GetOrCreateStatMetadata(GetStatTypeStr(StatType::kFlops)), 3);
  stats.AddStatValue(
      *plane.GetOrCreateStatMetadata(GetStatTypeStr(StatType::kModelFlops)), 4);
  stats.AddStatValue(
      *plane.GetOrCreateStatMetadata(GetStatTypeStr(StatType::kBytesAccessed)),
      5);
  stats.AddStatValue(
      *plane.GetOrCreateStatMetadata(GetStatTypeStr(StatType::kSourceInfo)),
      "my_file.py:123");
  stats.AddStatValue(
      *plane.GetOrCreateStatMetadata(GetStatTypeStr(StatType::kSourceStack)),
      "my_file.py:123\nmy_other_file.py:456");
  XEventBuilder event = line.AddEvent(*event_metadata);
  event.SetOffsetPs(0);
  event.SetDurationPs(100);
  tsl::profiler::XPlaneVisitor plane_visitor =
      tsl::profiler::CreateTfXPlaneVisitor(&raw_plane);
  tsl::profiler::XEventVisitor event_visitor(
      &plane_visitor, &raw_plane.lines(0), &raw_plane.lines(0).events(0));
  OpMetrics op_metrics = FromXEvent(event_visitor);

#if defined(PLATFORM_GOOGLE)
  EXPECT_THAT(op_metrics, EqualsProto(R"pb(
                occurrences: 1
                time_ps: 100
                self_time_ps: 100
                dma_stall_ps: 0
                hlo_module_id: 1
                flops: 3
                flops_v2: 3
                model_flops: 4
                model_flops_v2: 4
                bytes_accessed: 5
                name: "display_name"
                long_name: "metadata"
                deduplicated_name: "deduplicated_name"
                category: "tf_op_category"
                provenance: "tf_op"
                min_time_ps: 100
                num_cores: 1
                source_info {
                  file_name: "my_file.py"
                  line_number: 123
                  stack_frame: "my_file.py:123\nmy_other_file.py:456"
                }
              )pb"));
#endif
}

TEST(OpMetricsDbTest, FromXEventExtractsVddEnergy) {
  XPlane raw_plane;
  XPlaneBuilder plane(&raw_plane);
  XLineBuilder line = plane.GetOrCreateLine(0);
  XEventMetadata* event_metadata = plane.GetOrCreateEventMetadata("metadata");
  XEventBuilder event = line.AddEvent(*event_metadata);
  event.SetOffsetPs(0);
  event.SetDurationPs(100);

  XStatMetadata* energy_stat_metadata =
      plane.GetOrCreateStatMetadata("vdd_energy_j");
  event.AddStatValue(*energy_stat_metadata, 42.0);

  tsl::profiler::XPlaneVisitor plane_visitor =
      tsl::profiler::CreateTfXPlaneVisitor(&raw_plane);
  tsl::profiler::XEventVisitor event_visitor(
      &plane_visitor, &raw_plane.lines(0), &raw_plane.lines(0).events(0));

  OpMetrics op_metrics = FromXEvent(event_visitor);

  EXPECT_DOUBLE_EQ(op_metrics.vdd_energy_j(), 42.0);
}

TEST(OpMetricsDbTest, GetOpKeyFromXEvent) {
  XPlane raw_plane;
  XPlaneBuilder plane(&raw_plane);
  XEventMetadata* event_metadata = plane.GetOrCreateEventMetadata("metadata");
  event_metadata->set_display_name("display_name");
  XStatsBuilder<XEventMetadata> stats(event_metadata, &plane);
  stats.AddStatValue(
      *plane.GetOrCreateStatMetadata(GetStatTypeStr(StatType::kProgramId)), 1);
  stats.AddStatValue(
      *plane.GetOrCreateStatMetadata(GetStatTypeStr(StatType::kSymbolId)), 2);
  XLineBuilder line = plane.GetOrCreateLine(0);
  XEventBuilder event = line.AddEvent(*event_metadata);
  event.SetOffsetPs(0);
  event.SetDurationPs(100);
  tsl::profiler::XPlaneVisitor plane_visitor =
      tsl::profiler::CreateTfXPlaneVisitor(&raw_plane);
  tsl::profiler::XEventVisitor event_visitor(
      &plane_visitor, &raw_plane.lines(0), &raw_plane.lines(0).events(0));
  XEventsOpMetricsDbBuilder::OpKey op_key = GetOpKeyFromXEvent(event_visitor);
  EXPECT_EQ(op_key.program_id, 1);
  EXPECT_EQ(op_key.symbol_id, 2);
}

TEST(OpMetricsDbTest, AddOpMetric) {
  XPlane raw_plane;
  XPlaneBuilder plane(&raw_plane);
  XLineBuilder line = plane.GetOrCreateLine(0);
  {
    XEventMetadata* event_metadata = plane.GetOrCreateEventMetadata("m1");
    event_metadata->set_display_name("display_name1");
    XStatsBuilder<XEventMetadata> stats(event_metadata, &plane);
    stats.AddStatValue(
        *plane.GetOrCreateStatMetadata(GetStatTypeStr(StatType::kProgramId)),
        1);
    stats.AddStatValue(
        *plane.GetOrCreateStatMetadata(GetStatTypeStr(StatType::kSymbolId)), 1);
    XStatMetadata* time_scale_multiplier_stat = plane.GetOrCreateStatMetadata(
        GetStatTypeStr(StatType::kTimeScaleMultiplier));
    XStatMetadata* energy_stat_metadata =
        plane.GetOrCreateStatMetadata("vdd_energy_j");
    XStatMetadata* estimated_flops_stat = plane.GetOrCreateStatMetadata(
        GetStatTypeStr(StatType::kEstimatedFlops));
    XStatMetadata* estimated_bytes_accessed_stat =
        plane.GetOrCreateStatMetadata(
            GetStatTypeStr(StatType::kEstimatedBytesAccessed));
    XEventBuilder event = line.AddEvent(*event_metadata);
    event.SetOffsetPs(0);
    event.SetDurationPs(100);
    event.AddStatValue(*time_scale_multiplier_stat, 0.5);
    event.AddStatValue(*energy_stat_metadata, 10.0);
    event.AddStatValue(*estimated_flops_stat, 300);
    event.AddStatValue(*estimated_bytes_accessed_stat, 500);
    XEventBuilder event2 = line.AddEvent(*event_metadata);
    event2.SetOffsetPs(100);
    event2.SetDurationPs(100);
    event2.AddStatValue(*time_scale_multiplier_stat, 1.0);
    event2.AddStatValue(*energy_stat_metadata, 20.0);
    event2.AddStatValue(*estimated_flops_stat, 200);
    event2.AddStatValue(*estimated_bytes_accessed_stat, 100);
  }
  {
    XEventMetadata* event_metadata = plane.GetOrCreateEventMetadata("m2");
    event_metadata->set_display_name("display_name2");
    XStatsBuilder<XEventMetadata> stats(event_metadata, &plane);
    stats.AddStatValue(
        *plane.GetOrCreateStatMetadata(GetStatTypeStr(StatType::kProgramId)),
        1);
    stats.AddStatValue(
        *plane.GetOrCreateStatMetadata(GetStatTypeStr(StatType::kSymbolId)), 2);
    XStatMetadata* time_scale_multiplier_stat = plane.GetOrCreateStatMetadata(
        GetStatTypeStr(StatType::kTimeScaleMultiplier));
    XEventBuilder event = line.AddEvent(*event_metadata);
    event.SetOffsetPs(0);
    event.SetDurationPs(100);
    event.AddStatValue(*time_scale_multiplier_stat, 2.0);
  }
  {
    XEventMetadata* event_metadata = plane.GetOrCreateEventMetadata("m3");
    event_metadata->set_display_name("display_name3");
    XStatsBuilder<XEventMetadata> stats(event_metadata, &plane);
    stats.AddStatValue(
        *plane.GetOrCreateStatMetadata(GetStatTypeStr(StatType::kSymbolId)), 1);
    XEventBuilder event = line.AddEvent(*event_metadata);
    event.SetOffsetPs(0);
    event.SetDurationPs(100);
  }

  XEventsOpMetricsDbBuilder builder;
  tsl::profiler::XPlaneVisitor plane_visitor =
      tsl::profiler::CreateTfXPlaneVisitor(&raw_plane);
  plane_visitor.ForEachLine([&](const tsl::profiler::XLineVisitor& line) {
    line.ForEachEvent([&](const tsl::profiler::XEventVisitor& event) {
      builder.AddOpMetric(FromXEvent(event), GetOpKeyFromXEvent(event));
    });
  });
#if defined(PLATFORM_GOOGLE)
  OpMetricsDb db = builder.Finalize();
  EXPECT_THAT(db, IgnoringRepeatedFieldOrdering(EqualsProto(R"pb(
                metrics_db {
                  hlo_module_id: 1
                  self_time_ps: 200
                  occurrences: 2
                  name: "display_name1"
                  long_name: "m1"
                  time_ps: 200
                  normalized_time_ps: 150
                  min_time_ps: 100
                  num_cores: 1
                  vdd_energy_j: 30
                  estimated_flops: 500
                  estimated_bytes_accessed: 600
                }
                metrics_db {
                  hlo_module_id: 1
                  self_time_ps: 100
                  occurrences: 1
                  name: "display_name2"
                  long_name: "m2"
                  time_ps: 100
                  min_time_ps: 100
                  normalized_time_ps: 200
                  num_cores: 1
                }
                total_op_time_ps: 300
                normalized_total_op_time_ps: 350
              )pb")));
#endif
}

// TODO(bhupendradubey): Re-enable this test once we bring on demand provenance
// grouping.
TEST(OpMetricsDbTest, DISABLED_ParseProvenanceTest) {
  // Test case 1: Empty provenance string.
  std::string provenance_str_1 = "";
  std::vector<std::string> result_1 = ParseProvenance(provenance_str_1);
  EXPECT_TRUE(result_1.empty());

  // Test case 2: A provenance string with a single part.
  std::string provenance_str_2 = "my_op";
  std::vector<std::string> result_2 = ParseProvenance(provenance_str_2);
  ASSERT_EQ(result_2.size(), 1);
  EXPECT_EQ(result_2[0], "my_op");

  // Test case 3: A provenance string with multiple parts.
  std::string provenance_str_3 = "my_op1/my_op2/my_op3:xyz";
  std::vector<std::string> result_3 = ParseProvenance(provenance_str_3);
  ASSERT_EQ(result_3.size(), 3);
  EXPECT_EQ(result_3[0], "my_op1");
  EXPECT_EQ(result_3[1], "my_op2");
  EXPECT_EQ(result_3[2], "my_op3");
}

TEST(OpMetricsDbTest, GetRooflineModelRecordFromOpMetrics) {
  OpMetricsDb op_metrics_db = ParseTextProtoOrDie(R"pb(
    metrics_db {
      hlo_module_id: 1
      name: "root"
      occurrences: 1
      self_time_ps: 2
      time_ps: 4
      flops: 8
      flops_v2: 8
      source_info { stack_frame: "file.py:1" }
      children {
        metrics_db {
          hlo_module_id: 1
          name: "child"
          occurrences: 1
          self_time_ps: 4
          time_ps: 8
          flops: 2
          flops_v2: 2
          source_info { stack_frame: "file.py:1" }
          children {
            metrics_db {
              hlo_module_id: 1
              name: "descendant"
              occurrences: 1
              self_time_ps: 4
              time_ps: 8
              flops: 1
              flops_v2: 1
              source_info { stack_frame: "file.py:1" }
            }
          }
        }
      }
    }
  )pb");
  EXPECT_THAT(GetRootOpMetricsFromDb(op_metrics_db).size(), 1);
  EXPECT_THAT(GetRootOpMetricsFromDb(op_metrics_db)[0]->name(), "root");
}

TEST(OpMetricsDbTest, IsSyncWaitDoneOpDistinguishesSyncWaitFromCollectives) {
  EXPECT_TRUE(IsSyncWaitDoneOp("call-done"));
  EXPECT_TRUE(IsSyncWaitDoneOp("call-done.12"));
  EXPECT_TRUE(IsSyncWaitDoneOp("sparse-core-call.cloned.1.call-done"));
  EXPECT_TRUE(IsSyncWaitDoneOp("async-done"));
  EXPECT_TRUE(IsSyncWaitDoneOp("async-done.5"));
  EXPECT_TRUE(IsSyncWaitDoneOp("sync-wait-done.1"));
  EXPECT_TRUE(IsSyncWaitDoneOp("host-send-done"));
  EXPECT_TRUE(IsSyncWaitDoneOp("host-recv-done"));
  EXPECT_TRUE(IsSyncWaitDoneOp("Sync Wait:5"));
  EXPECT_TRUE(IsSyncWaitDoneOp("Barrier Wait:3"));
  EXPECT_TRUE(IsSyncWaitDoneOp("SFence"));

  EXPECT_FALSE(IsSyncWaitDoneOp("all-reduce-done"));
  EXPECT_FALSE(IsSyncWaitDoneOp("all-gather-done"));
  EXPECT_FALSE(IsSyncWaitDoneOp("reduce-scatter-done"));
  EXPECT_FALSE(IsSyncWaitDoneOp("collective-permute-done"));
  EXPECT_FALSE(IsSyncWaitDoneOp("fusion.1"));
}

TEST(OpMetricsDbTest, XEventsOpMetricsDbBuilderInPlaceAddOpMetricDetail) {
  XPlane raw_plane;
  XPlaneBuilder plane(&raw_plane);
  XLineBuilder line = plane.GetOrCreateLine(0);
  XEventMetadata* event_metadata = plane.GetOrCreateEventMetadata("metadata");
  event_metadata->set_display_name("display_name");
  XStatsBuilder<XEventMetadata> stats(event_metadata, &plane);
  stats.AddStatValue(
      *plane.GetOrCreateStatMetadata(GetStatTypeStr(StatType::kProgramId)), 1);
  stats.AddStatValue(
      *plane.GetOrCreateStatMetadata(GetStatTypeStr(StatType::kSymbolId)), 2);
  stats.AddStatValue(
      *plane.GetOrCreateStatMetadata(GetStatTypeStr(StatType::kSourceInfo)),
      "my_file.py:123");
  stats.AddStatValue(
      *plane.GetOrCreateStatMetadata(GetStatTypeStr(StatType::kSourceStack)),
      "my_file.py:123\nmy_other_file.py:456");
  XEventMetadata* child_metadata = plane.CreateEventMetadata();
  child_metadata->set_name("child_op");
  event_metadata->add_child_id(child_metadata->id());

  XEventBuilder event = line.AddEvent(*event_metadata);
  event.SetOffsetPs(0);
  event.SetDurationPs(100);
  tsl::profiler::XPlaneVisitor plane_visitor =
      tsl::profiler::CreateTfXPlaneVisitor(&raw_plane);
  tsl::profiler::XEventVisitor event_visitor(
      &plane_visitor, &raw_plane.lines(0), &raw_plane.lines(0).events(0));
  XEventsOpMetricsDbBuilder::OpKey key{/*program_id=*/1, /*symbol_id=*/2};

  XEventsOpMetricsDbBuilder skip_builder;
  skip_builder.AddOpMetric(event_visitor, key, /*total_time_ps=*/100,
                           /*self_time_ps=*/100,
                           OpMetadataDetail::kSkipSourceInfoAndChildren);
  OpMetricsDb skip_db = skip_builder.Finalize();
  ASSERT_EQ(skip_db.metrics_db_size(), 1);
  const OpMetrics& skip_metrics = skip_db.metrics_db(0);
  EXPECT_EQ(skip_metrics.name(), "display_name");
  EXPECT_FALSE(skip_metrics.has_source_info());
  EXPECT_EQ(skip_metrics.children().metrics_db_size(), 0);

  XEventsOpMetricsDbBuilder full_builder;
  full_builder.AddOpMetric(event_visitor, key, /*total_time_ps=*/100,
                           /*self_time_ps=*/100, OpMetadataDetail::kFull);
  OpMetricsDb full_db = full_builder.Finalize();
  ASSERT_EQ(full_db.metrics_db_size(), 1);
  const OpMetrics& full_metrics = full_db.metrics_db(0);
  EXPECT_EQ(full_metrics.name(), "display_name");
  EXPECT_EQ(full_metrics.source_info().file_name(), "my_file.py");
  EXPECT_EQ(full_metrics.source_info().line_number(), 123);
  EXPECT_EQ(full_metrics.source_info().stack_frame(),
            "my_file.py:123\nmy_other_file.py:456");
  ASSERT_EQ(full_metrics.children().metrics_db_size(), 1);
  EXPECT_EQ(full_metrics.children().metrics_db(0).name(), "child_op");
}

TEST(OpMetricsDbTest,
     XEventsOpMetricsDbBuilderInPlaceAddOpMetricKeepsCategory) {
  XPlane raw_plane;
  XPlaneBuilder plane(&raw_plane);
  XLineBuilder line = plane.GetOrCreateLine(0);
  XEventMetadata* event_metadata = plane.GetOrCreateEventMetadata(
      "%call-done.3 = f32[8]{0} call-done(f32[8]{0} %call-start.3)");
  event_metadata->set_display_name("call-done.3");
  XStatsBuilder<XEventMetadata> stats(event_metadata, &plane);
  stats.AddStatValue(
      *plane.GetOrCreateStatMetadata(GetStatTypeStr(StatType::kProgramId)), 1);
  stats.AddStatValue(
      *plane.GetOrCreateStatMetadata(GetStatTypeStr(StatType::kSymbolId)), 2);
  stats.AddStatValue(
      *plane.GetOrCreateStatMetadata(GetStatTypeStr(StatType::kHloCategory)),
      "custom-call");
  XEventBuilder event = line.AddEvent(*event_metadata);
  event.SetOffsetPs(0);
  event.SetDurationPs(100);
  tsl::profiler::XPlaneVisitor plane_visitor =
      tsl::profiler::CreateTfXPlaneVisitor(&raw_plane);
  tsl::profiler::XEventVisitor event_visitor(
      &plane_visitor, &raw_plane.lines(0), &raw_plane.lines(0).events(0));

  XEventsOpMetricsDbBuilder builder;
  builder.AddOpMetric(event_visitor,
                      XEventsOpMetricsDbBuilder::OpKey{/*program_id=*/1,
                                                       /*symbol_id=*/2},
                      /*total_time_ps=*/100, /*self_time_ps=*/100,
                      OpMetadataDetail::kSkipSourceInfoAndChildren);

  OpMetricsDb db = builder.Finalize();
  ASSERT_EQ(db.metrics_db_size(), 1);
  // Recategorizing sync wait/done ops is left to the caller.
  EXPECT_EQ(db.metrics_db(0).category(), "custom-call");
}

TEST(OpMetricsDbTest, XEventsOpMetricsDbBuilderInPlaceAddOpMetric) {
  XPlane raw_plane;
  XPlaneBuilder plane(&raw_plane);
  XLineBuilder line = plane.GetOrCreateLine(0);
  XEventMetadata* event_metadata =
      plane.GetOrCreateEventMetadata("op_long_name");
  event_metadata->set_display_name("op_name");
  XStatsBuilder<XEventMetadata> stats(event_metadata, &plane);
  stats.AddStatValue(
      *plane.GetOrCreateStatMetadata(GetStatTypeStr(StatType::kProgramId)), 10);
  stats.AddStatValue(
      *plane.GetOrCreateStatMetadata(GetStatTypeStr(StatType::kSymbolId)), 20);
  stats.AddStatValue(
      *plane.GetOrCreateStatMetadata(GetStatTypeStr(StatType::kSourceInfo)),
      "my_file.py:123");
  stats.AddStatValue(
      *plane.GetOrCreateStatMetadata(GetStatTypeStr(StatType::kFlops)), 100);
  stats.AddStatValue(
      *plane.GetOrCreateStatMetadata(GetStatTypeStr(StatType::kModelFlops)),
      80);
  stats.AddStatValue(
      *plane.GetOrCreateStatMetadata(GetStatTypeStr(StatType::kBytesAccessed)),
      200);
  MemoryAccessBreakdown breakdown;
  OpMetrics::MemoryAccessed* mem_access = breakdown.add_memory_accessed();
  mem_access->set_memory_space(1);
  mem_access->set_bytes_accessed(200);
  const MemoryAccessBreakdown& const_breakdown = breakdown;
  stats.AddStatValue(*plane.GetOrCreateStatMetadata(
                         GetStatTypeStr(StatType::kMemoryAccessBreakdown)),
                     const_breakdown);

  XStatMetadata* estimated_flops_stat =
      plane.GetOrCreateStatMetadata(GetStatTypeStr(StatType::kEstimatedFlops));
  XStatMetadata* estimated_bytes_accessed_stat = plane.GetOrCreateStatMetadata(
      GetStatTypeStr(StatType::kEstimatedBytesAccessed));

  XEventBuilder event1 = line.AddEvent(*event_metadata);
  event1.SetOffsetPs(0);
  event1.SetDurationPs(1000);
  event1.AddStatValue(*estimated_flops_stat, 15);
  event1.AddStatValue(*estimated_bytes_accessed_stat, 35);

  XEventBuilder event2 = line.AddEvent(*event_metadata);
  event2.SetOffsetPs(2000);
  event2.SetDurationPs(2000);
  event2.SetNumOccurrences(2);
  event2.AddStatValue(*estimated_flops_stat, 25);
  event2.AddStatValue(*estimated_bytes_accessed_stat, 45);

  tsl::profiler::XPlaneVisitor plane_visitor =
      tsl::profiler::CreateTfXPlaneVisitor(&raw_plane);
  tsl::profiler::XEventVisitor ev1(&plane_visitor, &raw_plane.lines(0),
                                   &raw_plane.lines(0).events(0));
  tsl::profiler::XEventVisitor ev2(&plane_visitor, &raw_plane.lines(0),
                                   &raw_plane.lines(0).events(1));

  XEventsOpMetricsDbBuilder builder;
  XEventsOpMetricsDbBuilder::OpKey key{/*program_id=*/10, /*symbol_id=*/20};

  builder.AddOpMetric(ev1, key, /*total_time_ps=*/1000, /*self_time_ps=*/700,
                      OpMetadataDetail::kSkipSourceInfoAndChildren);
  builder.AddOpMetric(ev2, key, /*total_time_ps=*/2000, /*self_time_ps=*/1500,
                      OpMetadataDetail::kSkipSourceInfoAndChildren);

  OpMetricsDb db = builder.Finalize();
  ASSERT_EQ(db.metrics_db_size(), 1);
  const OpMetrics& metric = db.metrics_db(0);
  EXPECT_EQ(metric.name(), "op_name");
  EXPECT_EQ(metric.occurrences(), 3);
  EXPECT_EQ(metric.time_ps(), 3000);
  EXPECT_EQ(metric.self_time_ps(), 2200);
  EXPECT_EQ(metric.min_time_ps(), 1000);
  EXPECT_EQ(metric.flops(), 300);
  EXPECT_DOUBLE_EQ(metric.flops_v2(), 300.0);
  EXPECT_EQ(metric.model_flops(), 240);
  EXPECT_DOUBLE_EQ(metric.model_flops_v2(), 240.0);
  EXPECT_EQ(metric.bytes_accessed(), 600);
  ASSERT_EQ(metric.memory_accessed_breakdown_size(), 1);
  EXPECT_EQ(metric.memory_accessed_breakdown(0).memory_space(), 1);
  EXPECT_EQ(metric.memory_accessed_breakdown(0).bytes_accessed(), 600);
  EXPECT_DOUBLE_EQ(metric.estimated_flops(), 40.0);
  EXPECT_DOUBLE_EQ(metric.estimated_bytes_accessed(), 80.0);
  EXPECT_FALSE(metric.has_source_info());
  EXPECT_TRUE(metric.source_info().file_name().empty());
}

TEST(OpMetricsDbTest,
     XEventsOpMetricsDbBuilderInPlaceAddOpMetricMatchesFromXEvent) {
  XPlane raw_plane;
  XPlaneBuilder plane(&raw_plane);
  XLineBuilder line = plane.GetOrCreateLine(0);
  const XStatMetadata& program_id_stat =
      *plane.GetOrCreateStatMetadata(GetStatTypeStr(StatType::kProgramId));
  const XStatMetadata& symbol_id_stat =
      *plane.GetOrCreateStatMetadata(GetStatTypeStr(StatType::kSymbolId));
  const XStatMetadata& min_duration_stat =
      *plane.GetOrCreateStatMetadata(GetStatTypeStr(StatType::kMinDurationPs));
  const XStatMetadata& self_duration_stat =
      *plane.GetOrCreateStatMetadata(GetStatTypeStr(StatType::kSelfDurationPs));
  const XStatMetadata& dma_stall_stat = *plane.GetOrCreateStatMetadata(
      GetStatTypeStr(StatType::kDmaStallDurationPs));
  const XStatMetadata& time_scale_stat = *plane.GetOrCreateStatMetadata(
      GetStatTypeStr(StatType::kTimeScaleMultiplier));
  const XStatMetadata& device_offset_stat =
      *plane.GetOrCreateStatMetadata(GetStatTypeStr(StatType::kDeviceOffsetPs));
  const XStatMetadata& device_duration_stat = *plane.GetOrCreateStatMetadata(
      GetStatTypeStr(StatType::kDeviceDurationPs));
  const XStatMetadata& vdd_energy_stat =
      *plane.GetOrCreateStatMetadata("vdd_energy_j");

  XEventMetadata* op_a_metadata = plane.GetOrCreateEventMetadata("op_a");
  {
    XStatsBuilder<XEventMetadata> stats(op_a_metadata, &plane);
    stats.AddStatValue(program_id_stat, 1);
    stats.AddStatValue(symbol_id_stat, 2);
    stats.AddStatValue(
        *plane.GetOrCreateStatMetadata(GetStatTypeStr(StatType::kFlops)), 10);
  }
  XEventMetadata* op_b_metadata = plane.GetOrCreateEventMetadata("op_b");
  {
    XStatsBuilder<XEventMetadata> stats(op_b_metadata, &plane);
    stats.AddStatValue(program_id_stat, 1);
    stats.AddStatValue(symbol_id_stat, 3);
  }
  {
    XEventBuilder event = line.AddEvent(*op_a_metadata);
    event.SetOffsetPs(0);
    event.SetDurationPs(1000);
    event.SetNumOccurrences(3);
    event.AddStatValue(min_duration_stat, 200);
    event.AddStatValue(self_duration_stat, 400);
    event.AddStatValue(dma_stall_stat, 50);
    event.AddStatValue(time_scale_stat, 1.5);
    event.AddStatValue(device_offset_stat, 10);
    event.AddStatValue(device_duration_stat, 900);
    event.AddStatValue(vdd_energy_stat, 0.25);
  }
  {
    XEventBuilder event = line.AddEvent(*op_a_metadata);
    event.SetOffsetPs(2000);
    event.SetDurationPs(500);
    event.AddStatValue(min_duration_stat, 100);
    event.AddStatValue(dma_stall_stat, 20);
    event.AddStatValue(time_scale_stat, 2.0);
    event.AddStatValue(device_offset_stat, 2010);
    event.AddStatValue(device_duration_stat, 450);
    event.AddStatValue(vdd_energy_stat, 0.5);
  }
  {
    XEventBuilder event = line.AddEvent(*op_b_metadata);
    event.SetOffsetPs(3000);
    event.SetDurationPs(300);
    event.AddStatValue(device_offset_stat, 3010);
    event.AddStatValue(device_duration_stat, 250);
  }
  // Device total and self times passed by the caller for each event.
  constexpr uint64_t kTotalTimesPs[] = {900, 450, 250};
  constexpr uint64_t kSelfTimesPs[] = {600, 450, 250};

  tsl::profiler::XPlaneVisitor plane_visitor =
      tsl::profiler::CreateTfXPlaneVisitor(&raw_plane);
  XEventsOpMetricsDbBuilder from_xevent_builder;
  XEventsOpMetricsDbBuilder in_place_builder;
  for (int i = 0; i < raw_plane.lines(0).events_size(); ++i) {
    tsl::profiler::XEventVisitor event(&plane_visitor, &raw_plane.lines(0),
                                       &raw_plane.lines(0).events(i));
    XEventsOpMetricsDbBuilder::OpKey key = GetOpKeyFromXEvent(event);
    OpMetrics op_metrics = FromXEvent(event);
    op_metrics.set_time_ps(kTotalTimesPs[i]);
    op_metrics.set_self_time_ps(kSelfTimesPs[i]);
    from_xevent_builder.AddOpMetric(op_metrics, key);
    in_place_builder.AddOpMetric(event, key, kTotalTimesPs[i], kSelfTimesPs[i],
                                 OpMetadataDetail::kFull);
  }

  OpMetricsDb from_xevent_db = from_xevent_builder.Finalize();
  OpMetricsDb in_place_db = in_place_builder.Finalize();
#if defined(PLATFORM_GOOGLE)
  EXPECT_THAT(in_place_db,
              IgnoringRepeatedFieldOrdering(EqualsProto(from_xevent_db)));
#endif
  ASSERT_EQ(in_place_db.metrics_db_size(), 2);
  const bool op_a_first = in_place_db.metrics_db(0).name() == "op_a";
  const OpMetrics& op_a = in_place_db.metrics_db(op_a_first ? 0 : 1);
  const OpMetrics& op_b = in_place_db.metrics_db(op_a_first ? 1 : 0);
  EXPECT_EQ(op_a.occurrences(), 4);
  EXPECT_EQ(op_a.time_ps(), 1350);
  EXPECT_EQ(op_a.self_time_ps(), 1050);
  EXPECT_EQ(op_a.min_time_ps(), 100);
  // Normalized time is based on the event durations (1000 * 1.5 + 500 * 2).
  EXPECT_EQ(op_a.normalized_time_ps(), 2500);
  EXPECT_EQ(op_a.dma_stall_ps(), 70);
  EXPECT_EQ(op_a.flops(), 40);
  EXPECT_DOUBLE_EQ(op_a.vdd_energy_j(), 0.75);
  EXPECT_EQ(op_b.name(), "op_b");
  EXPECT_EQ(op_b.min_time_ps(), 300);
  EXPECT_FALSE(op_b.has_vdd_energy_j());
}

TEST(OpMetricsDbTest, XEventsOpMetricsDbBuilderFinalizeResetsBuilder) {
  OpMetrics op_metrics;
  op_metrics.set_name("op");
  op_metrics.set_occurrences(1);
  op_metrics.set_time_ps(100);
  XEventsOpMetricsDbBuilder builder;
  builder.AddOpMetric(op_metrics, XEventsOpMetricsDbBuilder::OpKey{
                                      /*program_id=*/1, /*symbol_id=*/2});

  EXPECT_EQ(builder.Finalize().metrics_db_size(), 1);
  EXPECT_EQ(builder.Finalize().metrics_db_size(), 0);
}

}  // namespace
}  // namespace profiler
}  // namespace tensorflow
