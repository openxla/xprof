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

#include <string>
#include <vector>

#include "net/proto2/contrib/parse_proto/parse_text_proto.h"
#include "testing/base/public/gmock.h"
#include "<gtest/gtest.h>"
#include "xla/tsl/profiler/utils/tf_xplane_visitor.h"
#include "xla/tsl/profiler/utils/xplane_builder.h"
#include "xla/tsl/profiler/utils/xplane_schema.h"
#include "xla/tsl/profiler/utils/xplane_visitor.h"
#include "tsl/profiler/protobuf/xplane.pb.h"
#include "plugin/xprof/protobuf/op_metrics.pb.h"

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
                model_flops: 4
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
    XEventBuilder event = line.AddEvent(*event_metadata);
    event.SetOffsetPs(0);
    event.SetDurationPs(100);
    event.AddStatValue(*time_scale_multiplier_stat, 0.5);
    XEventBuilder event2 = line.AddEvent(*event_metadata);
    event2.SetOffsetPs(100);
    event2.SetDurationPs(100);
    event2.AddStatValue(*time_scale_multiplier_stat, 1.0);
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
      source_info { stack_frame: "file.py:1" }
      children {
        metrics_db {
          hlo_module_id: 1
          name: "child"
          occurrences: 1
          self_time_ps: 4
          time_ps: 8
          flops: 2
          source_info { stack_frame: "file.py:1" }
          children {
            metrics_db {
              hlo_module_id: 1
              name: "descendant"
              occurrences: 1
              self_time_ps: 4
              time_ps: 8
              flops: 1
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

}  // namespace
}  // namespace profiler
}  // namespace tensorflow
