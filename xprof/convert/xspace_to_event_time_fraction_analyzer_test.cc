#include "xprof/convert/xspace_to_event_time_fraction_analyzer.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "<gtest/gtest.h>"
#include "absl/status/statusor.h"
#include "xla/tsl/profiler/utils/xplane_builder.h"
#include "xla/tsl/profiler/utils/xplane_schema.h"
#include "tsl/profiler/protobuf/xplane.pb.h"
#include "xprof/convert/repository.h"
#include "plugin/xprof/protobuf/event_time_fraction_analyzer.pb.h"

namespace tensorflow {
namespace profiler {
namespace {

using ::tsl::profiler::XEventBuilder;
using ::tsl::profiler::XLineBuilder;
using ::tsl::profiler::XPlaneBuilder;
using ::tsl::profiler::XStatsBuilder;
using ::tsl::profiler::StatType;

TEST(ConvertXSpaceToEventTimeFractionAnalyzerResult, BasicTest) {
  XSpace xspace;
  XPlane* plane = xspace.add_planes();
  XPlaneBuilder plane_builder(plane);
  plane_builder.SetId(0);
  plane_builder.SetName("/device:TPU:0");

  // Create Step line
  XLineBuilder step_line = plane_builder.GetOrCreateLine(0);
  step_line.SetName(tsl::profiler::kStepLineName);

  XEventMetadata* step_metadata =
    plane_builder.GetOrCreateEventMetadata("step");
  XStatsBuilder<XEventMetadata> step_stats(step_metadata, &plane_builder);
  const XStatMetadata& group_id_stat = *plane_builder.GetOrCreateStatMetadata(
      GetStatTypeStr(StatType::kGroupId));

  {
    XEventBuilder event = step_line.AddEvent(*step_metadata);
    event.SetOffsetPs(0);
    event.SetDurationPs(1000);
    event.AddStatValue(group_id_stat, 1);  // Step 1
  }

  // Create Op line
  XLineBuilder op_line = plane_builder.GetOrCreateLine(1);
  op_line.SetName(tsl::profiler::kXlaOpLineName);

  XEventMetadata* op_metadata =
    plane_builder.GetOrCreateEventMetadata("matmul");
  XStatsBuilder<XEventMetadata> op_stats(op_metadata, &plane_builder);

  const XStatMetadata& duration_stat = *plane_builder.GetOrCreateStatMetadata(
      GetStatTypeStr(StatType::kDeviceDurationPs));

  {
    XEventBuilder event = op_line.AddEvent(*op_metadata);
    event.SetOffsetPs(100);
    event.SetDurationPs(200);
    event.AddStatValue(group_id_stat, 1);
    event.AddStatValue(duration_stat, static_cast<uint64_t>(200));
  }
  {
    XEventBuilder event = op_line.AddEvent(*op_metadata);
    event.SetOffsetPs(400);
    event.SetDurationPs(100);
    event.AddStatValue(group_id_stat, 1);
    event.AddStatValue(duration_stat, static_cast<uint64_t>(100));
  }

  auto result_or =
    ConvertXSpaceToEventTimeFractionAnalyzerResult(xspace, "matmul");
  ASSERT_TRUE(result_or.ok());
  const auto& result = result_or.value();

  ASSERT_EQ(result.chip_event_time_fractions().size(), 1);
  ASSERT_TRUE(result.chip_event_time_fractions().contains("/device:TPU:0"));
  const auto& fractions =
    result.chip_event_time_fractions().at("/device:TPU:0");
  ASSERT_EQ(fractions.event_time_fractions_size(), 1);
  // (200 + 100) / 1000 = 0.3
  EXPECT_NEAR(fractions.event_time_fractions(0), 0.3, 1e-6);
}

TEST(ConvertXSpaceToEventTimeFractionAnalyzerResult, BarrierCoresFiltering) {
  XSpace xspace;
  XPlane* plane = xspace.add_planes();
  XPlaneBuilder plane_builder(plane);
  plane_builder.SetId(0);
  plane_builder.SetName("/device:TPU:0");

  XLineBuilder step_line = plane_builder.GetOrCreateLine(0);
  step_line.SetName(tsl::profiler::kStepLineName);

  XEventMetadata* step_metadata =
    plane_builder.GetOrCreateEventMetadata("step");
  const XStatMetadata& group_id_stat = *plane_builder.GetOrCreateStatMetadata(
      GetStatTypeStr(StatType::kGroupId));

  {
    XEventBuilder event = step_line.AddEvent(*step_metadata);
    event.SetOffsetPs(0);
    event.SetDurationPs(10000);
    event.AddStatValue(group_id_stat, 1);
  }

  XLineBuilder op_line = plane_builder.GetOrCreateLine(1);
  op_line.SetName(tsl::profiler::kXlaOpLineName);

  XEventMetadata* op_metadata =
    plane_builder.GetOrCreateEventMetadata("barrier-cores");
  const XStatMetadata& duration_stat = *plane_builder.GetOrCreateStatMetadata(
      GetStatTypeStr(StatType::kDeviceDurationPs));

  {
    // Should be skipped (duration 0)
    XEventBuilder event = op_line.AddEvent(*op_metadata);
    event.SetOffsetPs(100);
    event.SetDurationPs(0);
    event.AddStatValue(group_id_stat, 1);
    event.AddStatValue(duration_stat, static_cast<uint64_t>(0));
  }
  {
    // Should be skipped (duration 1250)
    XEventBuilder event = op_line.AddEvent(*op_metadata);
    event.SetOffsetPs(200);
    event.SetDurationPs(1250);
    event.AddStatValue(group_id_stat, 1);
    event.AddStatValue(duration_stat, static_cast<uint64_t>(1250));
  }
  {
    // Should be counted
    XEventBuilder event = op_line.AddEvent(*op_metadata);
    event.SetOffsetPs(300);
    event.SetDurationPs(5000);
    event.AddStatValue(group_id_stat, 1);
    event.AddStatValue(duration_stat, static_cast<uint64_t>(5000));
  }

  auto result_or =
    ConvertXSpaceToEventTimeFractionAnalyzerResult(xspace, "barrier-cores");
  ASSERT_TRUE(result_or.ok());
  const auto& result = result_or.value();

  ASSERT_EQ(result.chip_event_time_fractions().size(), 1);
  ASSERT_TRUE(result.chip_event_time_fractions().contains("/device:TPU:0"));
  const auto& fractions =
    result.chip_event_time_fractions().at("/device:TPU:0");
  ASSERT_EQ(fractions.event_time_fractions_size(), 1);
  // 5000 / 10000 = 0.5
  EXPECT_NEAR(fractions.event_time_fractions(0), 0.5, 1e-6);
}

TEST(ConvertMultiXSpacesToEventTimeFractionAnalyzerResult, MultiXSpaceTest) {
  std::vector<std::unique_ptr<XSpace>> xspaces;
  std::vector<std::string> xspace_paths;

  // XSpace 1
  {
    auto xspace = std::make_unique<XSpace>();
    xspace->add_hostnames("host1");
    XPlane* plane = xspace->add_planes();
    XPlaneBuilder plane_builder(plane);
    plane_builder.SetId(0);
    plane_builder.SetName("/device:TPU:0");

    XLineBuilder step_line = plane_builder.GetOrCreateLine(0);
    step_line.SetName(tsl::profiler::kStepLineName);
    XEventMetadata* step_metadata =
      plane_builder.GetOrCreateEventMetadata("step");
    const XStatMetadata& group_id_stat =
        *plane_builder.GetOrCreateStatMetadata(
          GetStatTypeStr(StatType::kGroupId));

    {
      XEventBuilder event = step_line.AddEvent(*step_metadata);
      event.SetOffsetPs(0);
      event.SetDurationPs(1000);
      event.AddStatValue(group_id_stat, 1);
    }

    XLineBuilder op_line = plane_builder.GetOrCreateLine(1);
    op_line.SetName(tsl::profiler::kXlaOpLineName);
    XEventMetadata* op_metadata =
    plane_builder.GetOrCreateEventMetadata("matmul");
    const XStatMetadata& duration_stat =
        *plane_builder.GetOrCreateStatMetadata(
          GetStatTypeStr(StatType::kDeviceDurationPs));

    {
      XEventBuilder event = op_line.AddEvent(*op_metadata);
      event.SetOffsetPs(100);
      event.SetDurationPs(200);
      event.AddStatValue(group_id_stat, 1);
      event.AddStatValue(duration_stat, static_cast<uint64_t>(200));
    }

    xspaces.push_back(std::move(xspace));
    xspace_paths.push_back("/tmp/host1.xplane.pb");
  }

  // XSpace 2
  {
    auto xspace = std::make_unique<XSpace>();
    xspace->add_hostnames("host2");
    XPlane* plane = xspace->add_planes();
    XPlaneBuilder plane_builder(plane);
    plane_builder.SetId(0);
    plane_builder.SetName("/device:TPU:1");

    XLineBuilder step_line = plane_builder.GetOrCreateLine(0);
    step_line.SetName(tsl::profiler::kStepLineName);
    XEventMetadata* step_metadata =
      plane_builder.GetOrCreateEventMetadata("step");
    const XStatMetadata& group_id_stat =
        *plane_builder.GetOrCreateStatMetadata(
          GetStatTypeStr(StatType::kGroupId));

    {
      XEventBuilder event = step_line.AddEvent(*step_metadata);
      event.SetOffsetPs(0);
      event.SetDurationPs(2000);
      event.AddStatValue(group_id_stat, 1);
    }

    XLineBuilder op_line = plane_builder.GetOrCreateLine(1);
    op_line.SetName(tsl::profiler::kXlaOpLineName);
    XEventMetadata* op_metadata =
      plane_builder.GetOrCreateEventMetadata("matmul");
    const XStatMetadata& duration_stat =
        *plane_builder.GetOrCreateStatMetadata(
          GetStatTypeStr(StatType::kDeviceDurationPs));

    {
      XEventBuilder event = op_line.AddEvent(*op_metadata);
      event.SetOffsetPs(100);
      event.SetDurationPs(1000);
      event.AddStatValue(group_id_stat, 1);
      event.AddStatValue(duration_stat, static_cast<uint64_t>(1000));
    }

    xspaces.push_back(std::move(xspace));
    xspace_paths.push_back("/tmp/host2.xplane.pb");
  }

  auto session_snapshot_or = SessionSnapshot::Create(
    xspace_paths, std::move(xspaces));
  ASSERT_TRUE(session_snapshot_or.ok()) << session_snapshot_or.status();
  const auto& session_snapshot = session_snapshot_or.value();

  auto result_or = ConvertMultiXSpacesToEventTimeFractionAnalyzerResult(
    session_snapshot, "matmul");
  ASSERT_TRUE(result_or.ok());
  const auto& result = result_or.value();

  ASSERT_EQ(result.chip_event_time_fractions().size(), 2);
  ASSERT_TRUE(result.chip_event_time_fractions().contains("/device:TPU:0"));
  ASSERT_TRUE(result.chip_event_time_fractions().contains("/device:TPU:1"));

  // Check TPU:0
  EXPECT_NEAR(result.chip_event_time_fractions()
                  .at("/device:TPU:0")
                  .event_time_fractions(0),
              0.2, 1e-6);
  // Check TPU:1
  EXPECT_NEAR(result.chip_event_time_fractions()
                  .at("/device:TPU:1")
                  .event_time_fractions(0),
              0.5, 1e-6);

  ASSERT_EQ(result.host_event_time_fractions().size(), 2);
  ASSERT_TRUE(result.host_event_time_fractions().contains("host1"));
  ASSERT_TRUE(result.host_event_time_fractions().contains("host2"));

  EXPECT_NEAR(result.host_event_time_fractions()
                  .at("host1")
                  .event_time_fractions(0),
              0.2, 1e-6);
  EXPECT_NEAR(result.host_event_time_fractions()
                  .at("host2")
                  .event_time_fractions(0),
              0.5, 1e-6);
}

}  // namespace
}  // namespace profiler
}  // namespace tensorflow
