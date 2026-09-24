#include "xprof/convert/multi_xspace_to_inference_stats.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "xla/tsl/profiler/utils/device_utils.h"
#include "xla/tsl/profiler/utils/group_events.h"
#include "xla/tsl/profiler/utils/xplane_builder.h"
#include "xla/tsl/profiler/utils/xplane_schema.h"
#include "tsl/profiler/protobuf/xplane.pb.h"
#include "xprof/convert/data_table_utils.h"
#include "xprof/convert/inference_stats.h"
#include "xprof/convert/repository.h"
#include "xprof/utils/event_span.h"

namespace tensorflow {
namespace profiler {
namespace {

using ::testing::AllOf;
using ::testing::Contains;
using ::testing::ElementsAre;
using ::testing::Field;
using ::testing::IsEmpty;
using ::testing::Not;
using ::testing::Pair;
using ::testing::Property;

using ::tsl::profiler::DeviceType;
using ::tsl::profiler::GetHostEventTypeStr;
using ::tsl::profiler::GetStatTypeStr;
using ::tsl::profiler::GroupMetadata;
using ::tsl::profiler::GroupMetadataMap;
using ::tsl::profiler::HostEventType;
using ::tsl::profiler::kHostThreadsPlaneName;
using ::tsl::profiler::kTpuPlanePrefix;
using ::tsl::profiler::kXlaModuleLineName;
using ::tsl::profiler::StatType;
using ::tsl::profiler::XEventBuilder;
using ::tsl::profiler::XEventMetadata;
using ::tsl::profiler::XLineBuilder;
using ::tsl::profiler::XPlaneBuilder;
using ::tsl::profiler::XStatsBuilder;

class ConvertMultiXSpaceToInferenceStatsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Set up mock XSpace data here.
    xspace_ = std::make_unique<XSpace>();
    XPlane* plane = xspace_->add_planes();
    plane->set_name(kHostThreadsPlaneName);
    // Add more lines and events to simulate real data
    XLine* line = plane->add_lines();
    line->set_name("MyThread");
    XEvent* event = line->add_events();
    event->set_offset_ps(1000);
    event->set_duration_ps(2000);
    // Add stats to the event
    XStat* stat = event->add_stats();
    stat->set_int64_value(12345);
  }

  std::unique_ptr<XSpace> xspace_;
};

TEST_F(ConvertMultiXSpaceToInferenceStatsTest, TestWithMultipleXSpaces) {
  std::string test_name =
      ::testing::UnitTest::GetInstance()->current_test_info()->name();
  std::string path = absl::StrCat("ram://", test_name, "/");
  std::vector<std::string> paths = {absl::StrCat(path, "hostname1.xplane.pb"),
                                    absl::StrCat(path, "hostname2.xplane.pb")};

  std::vector<std::unique_ptr<XSpace>> xspaces;
  xspaces.push_back(std::make_unique<XSpace>());
  xspaces.push_back(std::make_unique<XSpace>());

  absl::StatusOr<SessionSnapshot> session_snapshot_status =
      SessionSnapshot::Create(paths, std::move(xspaces));
  SessionSnapshot session_snapshot = std::move(session_snapshot_status.value());

  InferenceStats inference_stats;
  absl::Status status = ConvertMultiXSpaceToInferenceStats(
      session_snapshot, "request", "batch", &inference_stats);

  EXPECT_OK(status);
}

TEST_F(ConvertMultiXSpaceToInferenceStatsTest,
       PopulatesProgramIdFromTpuModuleMetadata) {
  XSpace xspace;

  // 1. Set up TPU device plane (/device:TPU:0)
  XPlane* device_plane = xspace.add_planes();
  device_plane->set_name(absl::StrCat(kTpuPlanePrefix, "0"));
  XPlaneBuilder device_builder(device_plane);
  XLineBuilder xla_line = device_builder.GetOrCreateLine(0);
  xla_line.SetName(kXlaModuleLineName);

  // Store program_id on XEventMetadata
  constexpr uint64_t kExpectedProgramId = 9876543210ULL;
  constexpr int64_t kGroupId = 42;
  XEventMetadata* event_metadata =
      device_builder.GetOrCreateEventMetadata("my_hlo_module");
  XStatsBuilder<XEventMetadata> metadata_stats(event_metadata, &device_builder);
  metadata_stats.AddStatValue(*device_builder.GetOrCreateStatMetadata(
                                  GetStatTypeStr(StatType::kProgramId)),
                              kExpectedProgramId);

  // Create TPU device event referencing the metadata and carrying kGroupId
  XEventBuilder compute_event = xla_line.AddEvent(*event_metadata);
  compute_event.SetTimestampNs(1000);
  compute_event.SetDurationNs(2000);
  compute_event.AddStatValue(*device_builder.GetOrCreateStatMetadata(
                                 GetStatTypeStr(StatType::kGroupId)),
                             kGroupId);

  // 2. Set up Host CPU plane (/host:CPU) with ProcessBatch event
  XPlane* host_plane = xspace.add_planes();
  host_plane->set_name(kHostThreadsPlaneName);
  XPlaneBuilder host_builder(host_plane);
  XLineBuilder host_line = host_builder.GetOrCreateLine(0);
  host_line.SetName("BatchThread");

  XEventMetadata* batch_metadata = host_builder.GetOrCreateEventMetadata(
      GetHostEventTypeStr(HostEventType::kProcessBatch));
  XEventBuilder batch_event = host_line.AddEvent(*batch_metadata);
  batch_event.SetTimestampNs(500);
  batch_event.SetDurationNs(3000);
  batch_event.AddStatValue(
      *host_builder.GetOrCreateStatMetadata(GetStatTypeStr(StatType::kGroupId)),
      kGroupId);

  // 3. Populate group metadata
  GroupMetadataMap group_metadata_map;
  group_metadata_map[kGroupId] = GroupMetadata();

  // 4. Run GenerateInferenceStats
  std::vector<XPlane*> device_traces = {device_plane};
  StepEvents nonoverlapped_step_events;
  InferenceStats inference_stats;
  GenerateInferenceStats(device_traces, nonoverlapped_step_events,
                         group_metadata_map, xspace, DeviceType::kTpu,
                         /*host_id=*/0, &inference_stats);

  // 5. Assert batch_details has program_id populated!
  EXPECT_THAT(
      inference_stats.inference_stats_per_host(),
      Contains(Pair(
          0,
          Property(&PerHostInferenceStats::batch_details,
                   Contains(AllOf(Property(&BatchDetail::batch_id, kGroupId),
                                  Property(&BatchDetail::program_ids,
                                           Contains(kExpectedProgramId))))))));

  // 6. Verify data table generation exports Program ID(s) column
  bool has_batching = true;
  bool has_tensor_pattern = false;
  std::vector<std::string> sorted_model_ids = {"my_model"};
  std::vector<DataTable> tables;
  SampledInferenceStatsProto sampled_stats;
  SampledPerModelInferenceStatsProto per_model_sampled;
  const auto& batch_detail =
      inference_stats.inference_stats_per_host().at(0).batch_details(0);
  *per_model_sampled.add_sampled_batches() = batch_detail;
  sampled_stats.mutable_sampled_inference_stats_per_model()->insert(
      {0, per_model_sampled});
  inference_stats.mutable_inference_stats_per_model()->insert(
      {0, PerModelInferenceStats()});
  inference_stats.mutable_model_id_db()->mutable_id_to_index()->insert(
      {"my_model", 0});
  GeneratePerModelInferenceDataTables(
      inference_stats, sampled_stats, sorted_model_ids, tables, has_batching,
      has_tensor_pattern, "test_session", /*is_tpu=*/true);
  EXPECT_THAT(
      tables,
      ElementsAre(
          testing::_,
          AllOf(Property(&DataTable::GetColumns,
                         Contains(AllOf(
                             Field(&TableColumn::type, "string"),
                             Field(&TableColumn::label, "Program ID(s)")))),
                Property(&DataTable::GetRows, Not(IsEmpty())))));
}

TEST_F(ConvertMultiXSpaceToInferenceStatsTest,
       ExtractsProgramIdFromEventNameIfStatMissing) {
  XSpace xspace;

  // 1. Set up TPU device plane (/device:TPU:0)
  XPlane* device_plane = xspace.add_planes();
  device_plane->set_name(absl::StrCat(kTpuPlanePrefix, "0"));
  XPlaneBuilder device_builder(device_plane);
  XLineBuilder xla_line = device_builder.GetOrCreateLine(0);
  xla_line.SetName(kXlaModuleLineName);

  // Expected program ID matches the 12345 in the string
  constexpr uint64_t kExpectedProgramId = 12345ULL;
  constexpr int64_t kGroupId = 42;

  // Note: kProgramId stat is NOT added; the program ID string format is used
  // instead.
  XEventMetadata* event_metadata =
      device_builder.GetOrCreateEventMetadata("my_hlo_module(12345)");

  // Create TPU device event referencing the metadata and carrying kGroupId
  XEventBuilder compute_event = xla_line.AddEvent(*event_metadata);
  compute_event.SetTimestampNs(1000);
  compute_event.SetDurationNs(2000);
  compute_event.AddStatValue(*device_builder.GetOrCreateStatMetadata(
                                 GetStatTypeStr(StatType::kGroupId)),
                             kGroupId);

  // 2. Set up Host CPU plane (/host:CPU) with ProcessBatch event
  XPlane* host_plane = xspace.add_planes();
  host_plane->set_name(kHostThreadsPlaneName);
  XPlaneBuilder host_builder(host_plane);
  XLineBuilder host_line = host_builder.GetOrCreateLine(0);
  host_line.SetName("BatchThread");

  XEventMetadata* batch_metadata = host_builder.GetOrCreateEventMetadata(
      GetHostEventTypeStr(HostEventType::kProcessBatch));
  XEventBuilder batch_event = host_line.AddEvent(*batch_metadata);
  batch_event.SetTimestampNs(500);
  batch_event.SetDurationNs(3000);
  batch_event.AddStatValue(
      *host_builder.GetOrCreateStatMetadata(GetStatTypeStr(StatType::kGroupId)),
      kGroupId);

  // 3. Populate group metadata
  GroupMetadataMap group_metadata_map;
  group_metadata_map[kGroupId] = GroupMetadata();

  // 4. Run GenerateInferenceStats
  std::vector<XPlane*> device_traces = {device_plane};
  StepEvents nonoverlapped_step_events;
  InferenceStats inference_stats;
  GenerateInferenceStats(device_traces, nonoverlapped_step_events,
                         group_metadata_map, xspace, DeviceType::kTpu,
                         /*host_id=*/0, &inference_stats);

  // 5. Assert batch_details has program_id populated!
  EXPECT_THAT(
      inference_stats.inference_stats_per_host(),
      Contains(Pair(
          0,
          Property(&PerHostInferenceStats::batch_details,
                   Contains(AllOf(Property(&BatchDetail::batch_id, kGroupId),
                                  Property(&BatchDetail::program_ids,
                                           Contains(kExpectedProgramId))))))));
}

TEST_F(ConvertMultiXSpaceToInferenceStatsTest,
       AppendsProgramIdOnBatchCollision) {
  XSpace xspace;

  // 1. Set up TPU device plane with two different program_ids for the same
  // group_id
  XPlane* device_plane = xspace.add_planes();
  device_plane->set_name(absl::StrCat(kTpuPlanePrefix, "0"));
  XPlaneBuilder device_builder(device_plane);
  XLineBuilder xla_line = device_builder.GetOrCreateLine(0);
  xla_line.SetName(kXlaModuleLineName);

  constexpr uint64_t kProgramId1 = 11111ULL;
  constexpr uint64_t kProgramId2 = 22222ULL;
  constexpr int64_t kGroupId = 42;

  // Event 1 with kProgramId1
  XEventMetadata* meta1 =
      device_builder.GetOrCreateEventMetadata("hlo_module_1");
  XStatsBuilder<XEventMetadata>(meta1, &device_builder)
      .AddStatValue(*device_builder.GetOrCreateStatMetadata(
                        GetStatTypeStr(StatType::kProgramId)),
                    kProgramId1);
  XEventBuilder event1 = xla_line.AddEvent(*meta1);
  event1.SetTimestampNs(1000);
  event1.SetDurationNs(1000);
  event1.AddStatValue(*device_builder.GetOrCreateStatMetadata(
                          GetStatTypeStr(StatType::kGroupId)),
                      kGroupId);

  // Event 2 with kProgramId2 on the same group_id (collision)
  XEventMetadata* meta2 =
      device_builder.GetOrCreateEventMetadata("hlo_module_2");
  XStatsBuilder<XEventMetadata>(meta2, &device_builder)
      .AddStatValue(*device_builder.GetOrCreateStatMetadata(
                        GetStatTypeStr(StatType::kProgramId)),
                    kProgramId2);
  XEventBuilder event2 = xla_line.AddEvent(*meta2);
  event2.SetTimestampNs(2000);
  event2.SetDurationNs(1000);
  event2.AddStatValue(*device_builder.GetOrCreateStatMetadata(
                          GetStatTypeStr(StatType::kGroupId)),
                      kGroupId);

  // 2. Set up Host CPU plane with ProcessBatch event
  XPlane* host_plane = xspace.add_planes();
  host_plane->set_name(kHostThreadsPlaneName);
  XPlaneBuilder host_builder(host_plane);
  XLineBuilder host_line = host_builder.GetOrCreateLine(0);
  host_line.SetName("BatchThread");
  XEventMetadata* batch_metadata = host_builder.GetOrCreateEventMetadata(
      GetHostEventTypeStr(HostEventType::kProcessBatch));
  XEventBuilder batch_event = host_line.AddEvent(*batch_metadata);
  batch_event.SetTimestampNs(500);
  batch_event.SetDurationNs(3000);
  batch_event.AddStatValue(
      *host_builder.GetOrCreateStatMetadata(GetStatTypeStr(StatType::kGroupId)),
      kGroupId);

  // 3. Populate group metadata
  GroupMetadataMap group_metadata_map;
  group_metadata_map[kGroupId] = GroupMetadata();

  // 4. Run GenerateInferenceStats
  std::vector<XPlane*> device_traces = {device_plane};
  StepEvents nonoverlapped_step_events;
  InferenceStats inference_stats;
  GenerateInferenceStats(device_traces, nonoverlapped_step_events,
                         group_metadata_map, xspace, DeviceType::kTpu,
                         /*host_id=*/0, &inference_stats);

  // 5. Verify both program IDs are present due to collision
  EXPECT_THAT(
      inference_stats.inference_stats_per_host(),
      Contains(Pair(0, Property(&PerHostInferenceStats::batch_details,
                                Contains(Property(
                                    &BatchDetail::program_ids,
                                    ElementsAre(kProgramId1, kProgramId2)))))));
}

}  // namespace
}  // namespace profiler
}  // namespace tensorflow
