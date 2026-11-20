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

#include "xprof/convert/xplane_to_dcn_collective_stats.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/service/hlo.pb.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/file_system.h"
#include "xla/tsl/platform/status.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/profiler/utils/math_utils.h"
#include "xla/tsl/profiler/utils/xplane_builder.h"
#include "xla/tsl/profiler/utils/xplane_schema.h"
#include "xla/tsl/profiler/utils/xplane_utils.h"
#include "tsl/profiler/protobuf/xplane.pb.h"
#include "xprof/convert/repository.h"
#include "plugin/xprof/protobuf/dcn_slack_analysis.pb.h"
#include "plugin/xprof/protobuf/topology.pb.h"

namespace tensorflow {
namespace profiler {
namespace {

DcnSlackAnalysis CreateDcnSlackAnalysisProto() {
  DcnSlackAnalysis dcn_slack_analysis;
  DcnSlackSummary* dcn_slack_summary =
      dcn_slack_analysis.add_dcn_slack_summary();
  dcn_slack_summary->set_rendezvous("collective");
  dcn_slack_summary->set_recv_op_name("recv-done");
  dcn_slack_summary->set_send_op_name("send");
  dcn_slack_summary->set_slack_us(2);
  dcn_slack_summary->set_observed_duration_us(12);
  dcn_slack_summary->set_stall_duration_us(5);
  dcn_slack_summary->set_occurrences(4);
  dcn_slack_summary->set_bytes_transmitted_over_network(819200);
  return dcn_slack_analysis;
}

SessionSnapshot CreateSessionSnapshot(bool create_cache_file,
                                      bool has_dcn_collective_stats) {
  std::string test_name =
      ::testing::UnitTest::GetInstance()->current_test_info()->name();
  std::string path = absl::StrCat("ram://", test_name, "/");
  std::unique_ptr<tsl::WritableFile> xplane_file;
  std::vector<std::string> paths = {absl::StrCat(path, "hostname.xplane.pb")};

  auto xspace = std::make_unique<XSpace>();
  XPlane* xplane =
      tsl::profiler::FindOrAddMutablePlaneWithName(xspace.get(), "/host:CPU");
  if (has_dcn_collective_stats) {
    tsl::profiler::XPlaneBuilder xplane_builder(xplane);
    xplane_builder.GetOrCreateEventMetadata("MegaScale:");
  }

  if (create_cache_file) {
    if (has_dcn_collective_stats) {
      tsl::Env::Default()
          ->NewAppendableFile(
              absl::StrCat(path, "hostname.dcn_collective_stats.pb"),
              &xplane_file)
          .IgnoreError();
      tsl::Env::Default()
          ->NewAppendableFile(
              absl::StrCat(path, "ALL_HOSTS.dcn_collective_stats.pb"),
              &xplane_file)
          .IgnoreError();
    } else {
      tsl::Env::Default()
          ->NewAppendableFile(
              absl::StrCat(path, "NO_HOST.dcn_collective_stats.pb"),
              &xplane_file)
          .IgnoreError();
    }
  }

  std::vector<std::unique_ptr<XSpace>> xspaces;
  xspaces.push_back(std::move(xspace));

  absl::StatusOr<SessionSnapshot> session_snapshot_status =
      SessionSnapshot::Create(paths, std::move(xspaces));
  TF_CHECK_OK(session_snapshot_status.status());
  SessionSnapshot session_snapshot = std::move(session_snapshot_status.value());
  if (has_dcn_collective_stats) {
    DcnSlackAnalysis dcn_slack_analysis = CreateDcnSlackAnalysisProto();
    TF_CHECK_OK(session_snapshot.WriteBinaryProto(
        DCN_COLLECTIVE_STATS, "hostname", dcn_slack_analysis));
    TF_CHECK_OK(session_snapshot.WriteBinaryProto(
        DCN_COLLECTIVE_STATS, kAllHostsIdentifier, dcn_slack_analysis));
  }
  return session_snapshot;
}

TEST(ConvertXplaneToDcnCollectiveStats,
     HasAllHostsDcnCollectiveStatsCacheFile) {
  SessionSnapshot session_snapshot = CreateSessionSnapshot(true, true);

  absl::StatusOr<bool> status =
      HasDcnCollectiveStatsInMultiXSpace(session_snapshot);
  EXPECT_EQ(status.value(), true);
}

TEST(ConvertXplaneToDcnCollectiveStats, HasNoHostDcnCollectiveStatsCacheFile) {
  SessionSnapshot session_snapshot = CreateSessionSnapshot(true, false);

  absl::StatusOr<bool> status =
      HasDcnCollectiveStatsInMultiXSpace(session_snapshot);
  EXPECT_EQ(status.value(), false);
}

TEST(ConvertXplaneToDcnCollectiveStats,
     NoCacheFileButTraceHasDcnCollectiveStats) {
  SessionSnapshot session_snapshot = CreateSessionSnapshot(false, true);

  absl::StatusOr<bool> status =
      HasDcnCollectiveStatsInMultiXSpace(session_snapshot);
  EXPECT_EQ(status.value(), true);
}

TEST(ConvertXplaneToDcnCollectiveStats,
     NoCacheFileNoDcnCollectiveStatsPresent) {
  SessionSnapshot session_snapshot = CreateSessionSnapshot(false, false);

  absl::StatusOr<bool> status =
      HasDcnCollectiveStatsInMultiXSpace(session_snapshot);
  EXPECT_EQ(status.value(), false);
}

TEST(ConvertXplaneToDcnCollectiveStats,
     ConvertXSpaceToDcnCollectiveStatsWhenStatsPresent) {
  SessionSnapshot session_snapshot = CreateSessionSnapshot(false, true);

  absl::StatusOr<bool> status =
      ConvertMultiXSpaceToDcnCollectiveStats(session_snapshot);
  absl::StatusOr<std::optional<std::string>> all_hosts_filepath =
      session_snapshot.GetHostDataFilePath(StoredDataType::DCN_COLLECTIVE_STATS,
                                           kAllHostsIdentifier);
  absl::StatusOr<std::optional<std::string>> host_filepath =
      session_snapshot.GetHostDataFilePath(StoredDataType::DCN_COLLECTIVE_STATS,
                                           "hostname");

  EXPECT_EQ(status.value(), true);
  TF_EXPECT_OK(all_hosts_filepath.status());
  EXPECT_TRUE(all_hosts_filepath.value().has_value());
  EXPECT_FALSE(all_hosts_filepath.value().value().empty());
  TF_EXPECT_OK(host_filepath.status());
  EXPECT_TRUE(host_filepath.value().has_value());
  EXPECT_FALSE(host_filepath.value().value().empty());
}

TEST(ConvertXplaneToDcnCollectiveStats,
     ConvertXSpaceToDcnCollectiveStatsWhenStatsNotPresent) {
  SessionSnapshot session_snapshot = CreateSessionSnapshot(false, false);

  absl::StatusOr<bool> status =
      ConvertMultiXSpaceToDcnCollectiveStats(session_snapshot);
  absl::StatusOr<std::optional<std::string>> filepath =
      session_snapshot.GetHostDataFilePath(StoredDataType::DCN_COLLECTIVE_STATS,
                                           kNoHostIdentifier);

  EXPECT_EQ(status.value(), false);
  TF_EXPECT_OK(filepath.status());
  EXPECT_TRUE(filepath.value().has_value());
  EXPECT_FALSE(filepath.value().value().empty());
}

TEST(ConvertXplaneToDcnCollectiveStats,
     GetHostDcnSlackAnalysisWhenStatsNotPresent) {
  SessionSnapshot session_snapshot = CreateSessionSnapshot(false, false);

  absl::StatusOr<DcnSlackAnalysis> host_dcn_slack_analysis =
      GetDcnSlackAnalysisByHostName(session_snapshot, "hostname");

  TF_EXPECT_OK(host_dcn_slack_analysis.status());
  EXPECT_EQ(host_dcn_slack_analysis.value().dcn_slack_summary_size(), 0);
}

TEST(ConvertXplaneToDcnCollectiveStats,
     GetHostDcnSlackAnalysisWhenStatsPresent) {
  SessionSnapshot session_snapshot = CreateSessionSnapshot(true, true);

  absl::StatusOr<DcnSlackAnalysis> host_dcn_slack_analysis =
      GetDcnSlackAnalysisByHostName(session_snapshot, "hostname");

  TF_EXPECT_OK(host_dcn_slack_analysis.status());
  EXPECT_EQ(host_dcn_slack_analysis.value().dcn_slack_summary_size(), 1);
}

TEST(ConvertXplaneToDcnCollectiveStats,
     HasDcnCollectiveStatsWithMegascaleTracePlaneOnly) {
  XSpace xspace;
  XPlane* plane = tsl::profiler::FindOrAddMutablePlaneWithName(
      &xspace, "/device:CUSTOM:Megascale Trace");
  tsl::profiler::XPlaneBuilder plane_builder(plane);
  tsl::profiler::XLineBuilder line = plane_builder.GetOrCreateLine(0);
  line.AddEvent(*plane_builder.GetOrCreateEventMetadata("megascale_op"));
  EXPECT_TRUE(HasDcnCollectiveStatsInXSpace(xspace));
}

TEST(ConvertXplaneToDcnCollectiveStats,
     HasNoDcnCollectiveStatsForEmptyMegascaleTracePlane) {
  XSpace xspace;
  tsl::profiler::FindOrAddMutablePlaneWithName(
      &xspace, "/device:CUSTOM:Megascale Trace");
  EXPECT_FALSE(HasDcnCollectiveStatsInXSpace(xspace));
}

TEST(ConvertXplaneToDcnCollectiveStats,
     ConvertMultiXSpaceExtractsMegascaleTraceAndTopology) {
  std::string test_name =
      ::testing::UnitTest::GetInstance()->current_test_info()->name();
  std::string path = absl::StrCat("ram://", test_name, "/");
  std::vector<std::string> paths = {absl::StrCat(path, "host0.xplane.pb")};

  auto xspace = std::make_unique<XSpace>();

  // Metadata plane with valid HLO proto containing send & recv-done.
  constexpr absl::string_view kHloText = R"(
HloModule test_module
ENTRY main {
  p0 = f32[4] parameter(0)
  after-all.1 = token[] after-all()
  send.1 = (f32[4], u32[], token[]) send(p0, after-all.1), channel_id=1, is_host_transfer=true, frontend_attributes={_xla_host_transfer_rendezvous="all-reduce-1",_xla_megascale_transfer_type="ALL_REDUCE"}
  send-done.1 = token[] send-done(send.1), channel_id=1, is_host_transfer=true
  recv.1 = (f32[4], u32[], token[]) recv(after-all.1), channel_id=2, is_host_transfer=true, frontend_attributes={_xla_host_transfer_rendezvous="all-reduce-1",_xla_megascale_transfer_type="ALL_REDUCE"}
  ROOT recv-done.1 = (f32[4], token[]) recv-done(recv.1), channel_id=2, is_host_transfer=true
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto parsed_module,
                          xla::ParseAndReturnUnverifiedModule(kHloText));
  xla::HloProto hlo_proto;
  *hlo_proto.mutable_hlo_module() = parsed_module->ToProto();

  XPlane* metadata_plane = tsl::profiler::FindOrAddMutablePlaneWithName(
      xspace.get(), tsl::profiler::kMetadataPlaneName);
  tsl::profiler::XPlaneBuilder metadata_builder(metadata_plane);
  auto* hlo_stat_meta = metadata_builder.GetOrCreateStatMetadata(
      tsl::profiler::GetStatTypeStr(tsl::profiler::StatType::kHloProto));
  auto* hlo_meta = metadata_builder.GetOrCreateEventMetadata("test_module");
  XStat* hlo_stat = hlo_meta->add_stats();
  hlo_stat->set_metadata_id(hlo_stat_meta->id());
  hlo_stat->set_bytes_value(hlo_proto.SerializeAsString());

  // TPU device plane with XLA Modules and XLA Ops lines.
  XPlane* tpu_plane = tsl::profiler::FindOrAddMutablePlaneWithName(
      xspace.get(), tsl::profiler::TpuPlaneName(0));
  tsl::profiler::XPlaneBuilder tpu_builder(tpu_plane);
  tpu_builder.SetId(0);
  auto mod_line = tpu_builder.GetOrCreateLine(0);
  mod_line.SetName(tsl::profiler::kXlaModuleLineName);
  mod_line.SetTimestampNs(0);
  auto mod_event = mod_line.AddEvent(*tpu_builder.GetOrCreateEventMetadata(
      absl::StrCat("test_module(", hlo_meta->id(), ")")));
  mod_event.SetOffsetPs(0);
  mod_event.SetDurationPs(tsl::profiler::NanoToPico(15000));

  auto op_line = tpu_builder.GetOrCreateLine(1);
  op_line.SetName(tsl::profiler::kXlaOpLineName);
  op_line.SetTimestampNs(0);
  const auto& hlo_cat_stat = *tpu_builder.GetOrCreateStatMetadata(
      tsl::profiler::GetStatTypeStr(tsl::profiler::StatType::kHloCategory));

  auto add_op = [&](absl::string_view name, absl::string_view cat,
                    int64_t offset_ns, int64_t duration_ns) {
    auto* meta = tpu_builder.GetOrCreateEventMetadata(name);
    meta->set_display_name(std::string(name));
    tsl::profiler::XStatsBuilder<XEventMetadata> stats(meta, &tpu_builder);
    stats.AddStatValue(hlo_cat_stat, cat);
    auto ev = op_line.AddEvent(*meta);
    ev.SetOffsetPs(tsl::profiler::NanoToPico(offset_ns));
    ev.SetDurationPs(tsl::profiler::NanoToPico(duration_ns));
  };
  add_op("send.1", "host send", 1000, 1000);
  add_op("send-done.1", "host send-done", 2000, 1000);
  add_op("recv.1", "host recv", 8000, 1000);
  add_op("recv-done.1", "host recv-done", 9000, 1000);

  // Megascale Trace plane with an overlapping Host Stall event.
  XPlane* megascale_plane = tsl::profiler::FindOrAddMutablePlaneWithName(
      xspace.get(), "/device:CUSTOM:Megascale Trace");
  tsl::profiler::XPlaneBuilder ms_builder(megascale_plane);
  auto ms_line = ms_builder.GetOrCreateLine(0);
  ms_line.SetTimestampNs(0);
  auto* host_meta =
      ms_builder.GetOrCreateEventMetadata("device_000004_gid_all-reduce-1");
  auto host_event = ms_line.AddEvent(*host_meta);
  host_event.SetOffsetPs(tsl::profiler::NanoToPico(3000));
  host_event.SetDurationPs(tsl::profiler::NanoToPico(3000));

  // Task Environment plane with a non-matching stat and system_topology stat.
  XPlane* task_env = tsl::profiler::FindOrAddMutablePlaneWithName(
      xspace.get(), tsl::profiler::kTaskEnvPlaneName);
  tsl::profiler::XPlaneBuilder env_builder(task_env);
  auto* other_meta = env_builder.GetOrCreateStatMetadata("other_stat");
  XStat* other_stat = task_env->add_stats();
  other_stat->set_metadata_id(other_meta->id());
  other_stat->set_str_value("ignored");
  Topology topology;
  auto* loc = topology.add_mesh_location();
  loc->set_x(1);
  loc->set_global_id(4);
  loc->set_index_on_host(0);
  auto* topo_meta = env_builder.GetOrCreateStatMetadata("system_topology");
  XStat* topo_stat = task_env->add_stats();
  topo_stat->set_metadata_id(topo_meta->id());
  topo_stat->set_bytes_value(topology.SerializeAsString());

  std::vector<std::unique_ptr<XSpace>> xspaces;
  xspaces.push_back(std::move(xspace));
  TF_ASSERT_OK_AND_ASSIGN(SessionSnapshot session_snapshot,
                          SessionSnapshot::Create(paths, std::move(xspaces)));

  TF_ASSERT_OK_AND_ASSIGN(
      bool converted, ConvertMultiXSpaceToDcnCollectiveStats(session_snapshot));
  EXPECT_TRUE(converted);

  TF_ASSERT_OK_AND_ASSIGN(
      DcnSlackAnalysis analysis,
      GetDcnSlackAnalysisByHostName(session_snapshot, "host0"));
  ASSERT_EQ(analysis.dcn_slack_summary_size(), 1);
  EXPECT_EQ(analysis.dcn_slack_summary(0).host_events_count(), 1);
  EXPECT_EQ(analysis.dcn_slack_summary(0).recv_duration_us(), 1);
}

}  // namespace
}  // namespace profiler
}  // namespace tensorflow
