#include "xprof/convert/multi_xspace_to_inference_stats.h"

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

namespace tensorflow {
namespace profiler {
namespace {

class ConvertMultiXSpaceToInferenceStatsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Set up mock XSpace data here.
    xspace_ = std::make_unique<XSpace>();
    XPlane* plane = xspace_->add_planes();
    plane->set_name(tsl::profiler::kHostThreadsPlaneName);
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
  device_plane->set_name(absl::StrCat(tsl::profiler::kTpuPlanePrefix, "0"));
  tsl::profiler::XPlaneBuilder device_builder(device_plane);
  tsl::profiler::XLineBuilder xla_line = device_builder.GetOrCreateLine(0);
  xla_line.SetName(tsl::profiler::kXlaModuleLineName);

  // Store program_id on XEventMetadata
  constexpr uint64_t kExpectedProgramId = 9876543210ULL;
  constexpr int64_t kGroupId = 42;
  tsl::profiler::XEventMetadata* event_metadata =
      device_builder.GetOrCreateEventMetadata("my_hlo_module");
  tsl::profiler::XStatsBuilder<tsl::profiler::XEventMetadata> metadata_stats(
      event_metadata, &device_builder);
  metadata_stats.AddStatValue(
      *device_builder.GetOrCreateStatMetadata(
          tsl::profiler::GetStatTypeStr(tsl::profiler::StatType::kProgramId)),
      kExpectedProgramId);

  // Create TPU device event referencing the metadata and carrying kGroupId
  tsl::profiler::XEventBuilder compute_event =
      xla_line.AddEvent(*event_metadata);
  compute_event.SetTimestampNs(1000);
  compute_event.SetDurationNs(2000);
  compute_event.AddStatValue(
      *device_builder.GetOrCreateStatMetadata(
          tsl::profiler::GetStatTypeStr(tsl::profiler::StatType::kGroupId)),
      kGroupId);

  // 2. Set up Host CPU plane (/host:CPU) with ProcessBatch event
  XPlane* host_plane = xspace.add_planes();
  host_plane->set_name(tsl::profiler::kHostThreadsPlaneName);
  tsl::profiler::XPlaneBuilder host_builder(host_plane);
  tsl::profiler::XLineBuilder host_line = host_builder.GetOrCreateLine(0);
  host_line.SetName("BatchThread");

  tsl::profiler::XEventMetadata* batch_metadata =
      host_builder.GetOrCreateEventMetadata(tsl::profiler::GetHostEventTypeStr(
          tsl::profiler::HostEventType::kProcessBatch));
  tsl::profiler::XEventBuilder batch_event =
      host_line.AddEvent(*batch_metadata);
  batch_event.SetTimestampNs(500);
  batch_event.SetDurationNs(3000);
  batch_event.AddStatValue(
      *host_builder.GetOrCreateStatMetadata(
          tsl::profiler::GetStatTypeStr(tsl::profiler::StatType::kGroupId)),
      kGroupId);

  // 3. Populate group metadata
  tsl::profiler::GroupMetadataMap group_metadata_map;
  group_metadata_map[kGroupId] = tsl::profiler::GroupMetadata();

  // 4. Run GenerateInferenceStats
  std::vector<tsl::profiler::XPlane*> device_traces = {device_plane};
  StepEvents nonoverlapped_step_events;
  InferenceStats inference_stats;
  GenerateInferenceStats(
      device_traces, nonoverlapped_step_events, group_metadata_map, xspace,
      tsl::profiler::DeviceType::kTpu, /*host_id=*/0, &inference_stats);

  // 5. Assert batch_details has program_id populated!
  EXPECT_THAT(
      inference_stats.inference_stats_per_host(),
      testing::Contains(testing::Pair(
          0, testing::Property(
                 &PerHostInferenceStats::batch_details,
                 testing::Contains(testing::AllOf(
                     testing::Property(&BatchDetail::batch_id, kGroupId),
                     testing::Property(
                         &BatchDetail::program_ids,
                         testing::Contains(kExpectedProgramId))))))));

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
      testing::ElementsAre(
          testing::_,
          testing::AllOf(
              testing::Property(
                  &DataTable::GetColumns,
                  testing::Contains(testing::AllOf(
                      testing::Field(&TableColumn::type, "string"),
                      testing::Field(&TableColumn::label, "Program ID(s)")))),
              testing::Property(&DataTable::GetRows,
                                testing::Not(testing::IsEmpty())))));
}

TEST_F(ConvertMultiXSpaceToInferenceStatsTest,
       AppendsProgramIdOnBatchCollision) {
  XSpace xspace;

  // 1. Set up TPU device plane with two different program_ids for the same
  // group_id
  XPlane* device_plane = xspace.add_planes();
  device_plane->set_name(absl::StrCat(tsl::profiler::kTpuPlanePrefix, "0"));
  tsl::profiler::XPlaneBuilder device_builder(device_plane);
  tsl::profiler::XLineBuilder xla_line = device_builder.GetOrCreateLine(0);
  xla_line.SetName(tsl::profiler::kXlaModuleLineName);

  constexpr uint64_t kProgramId1 = 11111ULL;
  constexpr uint64_t kProgramId2 = 22222ULL;
  constexpr int64_t kGroupId = 42;

  // Event 1 with kProgramId1
  tsl::profiler::XEventMetadata* meta1 =
      device_builder.GetOrCreateEventMetadata("hlo_module_1");
  tsl::profiler::XStatsBuilder<tsl::profiler::XEventMetadata>(meta1,
                                                              &device_builder)
      .AddStatValue(
          *device_builder.GetOrCreateStatMetadata(tsl::profiler::GetStatTypeStr(
              tsl::profiler::StatType::kProgramId)),
          kProgramId1);
  tsl::profiler::XEventBuilder event1 = xla_line.AddEvent(*meta1);
  event1.SetTimestampNs(1000);
  event1.SetDurationNs(1000);
  event1.AddStatValue(
      *device_builder.GetOrCreateStatMetadata(
          tsl::profiler::GetStatTypeStr(tsl::profiler::StatType::kGroupId)),
      kGroupId);

  // Event 2 with kProgramId2 on the same group_id (collision)
  tsl::profiler::XEventMetadata* meta2 =
      device_builder.GetOrCreateEventMetadata("hlo_module_2");
  tsl::profiler::XStatsBuilder<tsl::profiler::XEventMetadata>(meta2,
                                                              &device_builder)
      .AddStatValue(
          *device_builder.GetOrCreateStatMetadata(tsl::profiler::GetStatTypeStr(
              tsl::profiler::StatType::kProgramId)),
          kProgramId2);
  tsl::profiler::XEventBuilder event2 = xla_line.AddEvent(*meta2);
  event2.SetTimestampNs(2000);
  event2.SetDurationNs(1000);
  event2.AddStatValue(
      *device_builder.GetOrCreateStatMetadata(
          tsl::profiler::GetStatTypeStr(tsl::profiler::StatType::kGroupId)),
      kGroupId);

  // 2. Set up Host CPU plane with ProcessBatch event
  XPlane* host_plane = xspace.add_planes();
  host_plane->set_name(tsl::profiler::kHostThreadsPlaneName);
  tsl::profiler::XPlaneBuilder host_builder(host_plane);
  tsl::profiler::XLineBuilder host_line = host_builder.GetOrCreateLine(0);
  host_line.SetName("BatchThread");
  tsl::profiler::XEventMetadata* batch_metadata =
      host_builder.GetOrCreateEventMetadata(tsl::profiler::GetHostEventTypeStr(
          tsl::profiler::HostEventType::kProcessBatch));
  tsl::profiler::XEventBuilder batch_event =
      host_line.AddEvent(*batch_metadata);
  batch_event.SetTimestampNs(500);
  batch_event.SetDurationNs(3000);
  batch_event.AddStatValue(
      *host_builder.GetOrCreateStatMetadata(
          tsl::profiler::GetStatTypeStr(tsl::profiler::StatType::kGroupId)),
      kGroupId);

  // 3. Populate group metadata
  tsl::profiler::GroupMetadataMap group_metadata_map;
  group_metadata_map[kGroupId] = tsl::profiler::GroupMetadata();

  // 4. Run GenerateInferenceStats
  std::vector<tsl::profiler::XPlane*> device_traces = {device_plane};
  StepEvents nonoverlapped_step_events;
  InferenceStats inference_stats;
  GenerateInferenceStats(
      device_traces, nonoverlapped_step_events, group_metadata_map, xspace,
      tsl::profiler::DeviceType::kTpu, /*host_id=*/0, &inference_stats);

  // 5. Verify both program IDs are present due to collision
  EXPECT_THAT(
      inference_stats.inference_stats_per_host(),
      testing::Contains(testing::Pair(
          0, testing::Property(
                 &PerHostInferenceStats::batch_details,
                 testing::Contains(testing::Property(
                     &BatchDetail::program_ids,
                     testing::ElementsAre(kProgramId1, kProgramId2)))))));
}

}  // namespace
}  // namespace profiler
}  // namespace tensorflow
