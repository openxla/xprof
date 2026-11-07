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

#include "xprof/convert/xplane_to_op_stats.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "xla/hlo/ir/hlo_opcode.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/profiler/convert/xla_op_utils.h"
#include "xla/tsl/profiler/utils/math_utils.h"
#include "xla/tsl/profiler/utils/tf_xplane_visitor.h"
#include "xla/tsl/profiler/utils/xplane_builder.h"
#include "xla/tsl/profiler/utils/xplane_schema.h"
#include "xla/tsl/profiler/utils/xplane_test_utils.h"
#include "tsl/profiler/protobuf/xplane.pb.h"
#include "xprof/convert/duty_cycle_tracker.h"
#include "xprof/convert/multi_xplanes_to_op_stats.h"
#include "xprof/convert/op_stats_to_input_pipeline_analysis.h"
#include "xprof/convert/repository.h"
#include "xprof/convert/step_events_to_steps_db.h"
#include "plugin/xprof/protobuf/diagnostics.pb.h"
#include "plugin/xprof/protobuf/op_metrics.pb.h"
#include "plugin/xprof/protobuf/op_stats.pb.h"
#include "plugin/xprof/protobuf/steps_db.pb.h"
#include "plugin/xprof/protobuf/tf_function.pb.h"
#include "plugin/xprof/protobuf/tpu_input_pipeline.pb.h"
#include "xprof/utils/hlo_proto_map.h"
#include "xprof/utils/op_metrics_db_utils.h"

namespace tensorflow {
namespace profiler {
namespace {

using ::testing::Property;
using ::testing::UnorderedElementsAre;
using ::tsl::profiler::GetOrCreateGpuXPlane;
using ::tsl::profiler::GetOrCreateHostXPlane;
using ::tsl::profiler::GetOrCreateTpuXPlane;
using ::tsl::profiler::HostEventType;
using ::tsl::profiler::kDeviceVendorAMD;
using ::tsl::profiler::kDeviceVendorNvidia;
using ::tsl::profiler::kSparseCoreModuleLineName;
using ::tsl::profiler::kSparseCoreOpLineName;
using ::tsl::profiler::kXlaModuleLineName;
using ::tsl::profiler::kXlaOpLineName;
using ::tsl::profiler::StatType;
using ::tsl::profiler::XEventBuilder;
using ::tsl::profiler::XLineBuilder;
using ::tsl::profiler::XPlaneBuilder;
using ::tsl::profiler::XStatsBuilder;

TEST(ConvertXPlaneToOpStats, GpuPerfEnv) {
  auto space = std::make_unique<XSpace>();
  constexpr double kMaxError = 0.01;
  constexpr int kClockRateKHz = 1530000;
  constexpr int kCoreCount = 80;
  constexpr uint64_t kMemoryBandwidthBytesPerSecond =
      uint64_t{900} * 1000 * 1000 * 1000;
  // Volta.
  constexpr int kComputeCapMajor = 7;
  constexpr int kComputeCapMinor = 0;

  XPlaneBuilder device_plane(
      GetOrCreateGpuXPlane(space.get(), /*device_ordinal=*/0));
  device_plane.AddStatValue(*device_plane.GetOrCreateStatMetadata(
                                GetStatTypeStr(StatType::kDevVendor)),
                            kDeviceVendorNvidia);
  device_plane.AddStatValue(*device_plane.GetOrCreateStatMetadata("clock_rate"),
                            kClockRateKHz);
  device_plane.AddStatValue(*device_plane.GetOrCreateStatMetadata("core_count"),
                            kCoreCount);
  device_plane.AddStatValue(
      *device_plane.GetOrCreateStatMetadata("memory_bandwidth"),
      kMemoryBandwidthBytesPerSecond);
  device_plane.AddStatValue(
      *device_plane.GetOrCreateStatMetadata("compute_cap_major"),
      kComputeCapMajor);
  device_plane.AddStatValue(
      *device_plane.GetOrCreateStatMetadata("compute_cap_minor"),
      kComputeCapMinor);

  std::vector<std::unique_ptr<XSpace>> xspaces;
  xspaces.push_back(std::move(space));
  auto session_snapshot_or =
      SessionSnapshot::Create({"test_xspace"}, std::move(xspaces));
  ASSERT_OK(session_snapshot_or.status());
  OpStatsOptions options;
  options.generate_op_metrics_db = true;
  OpStats op_stats;
  ASSERT_OK(ConvertMultiXSpacesToCombinedOpStats(session_snapshot_or.value(),
                                                 options, &op_stats));
  const PerfEnv& perf_env = op_stats.perf_env();
  // Change to lower flops number that we do not use sum of the tensor core peak
  // flops and the cuda core peak flops together as peak flops. Only use the
  // tensor core peak flops as all those white papers are using.
  EXPECT_NEAR(125.34, perf_env.peak_tera_flops_per_second(), kMaxError);
  EXPECT_NEAR(
      900,
      perf_env.peak_bws_giga_bytes_per_second(MemBwType::MEM_BW_TYPE_HBM_RW),
      kMaxError);
  // Ridge point changed accordingly from above peak flops change.
  EXPECT_NEAR(139.26, perf_env.ridge_point(), kMaxError);
}

TEST(ConvertXPlaneToOpStats, GpuPerfEnvPrefersBackendPeaks) {
  constexpr double kMaxError = 0.01;
  XSpace space;
  // MI300X: 304 CU at 2.1 GHz, 5.3 TB/s HBM.
  XPlane* device_plane = GetOrCreateGpuXPlane(&space, /*device_ordinal=*/0);
  XPlaneBuilder device_plane_builder(device_plane);
  device_plane_builder.AddStatValue(
      *device_plane_builder.GetOrCreateStatMetadata(
          GetStatTypeStr(StatType::kDevVendor)),
      kDeviceVendorAMD);
  device_plane_builder.AddStatValue(
      *device_plane_builder.GetOrCreateStatMetadata(
          GetStatTypeStr(StatType::kGpuDeviceName)),
      "gfx942");
  device_plane_builder.AddStatValue(
      *device_plane_builder.GetOrCreateStatMetadata("clock_rate"), 2100000);
  device_plane_builder.AddStatValue(
      *device_plane_builder.GetOrCreateStatMetadata("core_count"), 304);
  device_plane_builder.AddStatValue(
      *device_plane_builder.GetOrCreateStatMetadata("memory_bandwidth"),
      uint64_t{5300} * 1000 * 1000 * 1000);
  device_plane_builder.AddStatValue(
      *device_plane_builder.GetOrCreateStatMetadata(
          GetStatTypeStr(StatType::kDevCapPeakTeraflopsPerSecond)),
      1250.0);
  device_plane_builder.AddStatValue(
      *device_plane_builder.GetOrCreateStatMetadata(
          GetStatTypeStr(StatType::kDevCapPeakSramRdBwGigabytesPerSecond)),
      65000.0);
  device_plane_builder.AddStatValue(
      *device_plane_builder.GetOrCreateStatMetadata(
          GetStatTypeStr(StatType::kDevCapPeakSramWrBwGigabytesPerSecond)),
      48000.0);

  PerfEnv perf_env = GetPerfEnvFromXPlane(*device_plane);
  // Reported over the 1307.44 TFLOP/s and 81715.2 GB/s the tables derive.
  EXPECT_NEAR(1250.0, perf_env.peak_tera_flops_per_second(), kMaxError);
  EXPECT_NEAR(
      65000.0,
      perf_env.peak_bws_giga_bytes_per_second(MemBwType::MEM_BW_TYPE_SRAM_RD),
      kMaxError);
  EXPECT_NEAR(
      48000.0,
      perf_env.peak_bws_giga_bytes_per_second(MemBwType::MEM_BW_TYPE_SRAM_WR),
      kMaxError);
  // HBM has no backend stat on the GPU path, so it still comes from the caps.
  EXPECT_NEAR(
      5300.0,
      perf_env.peak_bws_giga_bytes_per_second(MemBwType::MEM_BW_TYPE_HBM_RW),
      kMaxError);
}

TEST(ConvertXPlaneToOpStats, GpuPerfEnvDerivesPeaksWithoutBackendStats) {
  constexpr double kMaxError = 0.01;
  XSpace space;
  // MI300X: 304 CU at 2.1 GHz, 5.3 TB/s HBM.
  XPlane* device_plane = GetOrCreateGpuXPlane(&space, /*device_ordinal=*/0);
  XPlaneBuilder device_plane_builder(device_plane);
  device_plane_builder.AddStatValue(
      *device_plane_builder.GetOrCreateStatMetadata(
          GetStatTypeStr(StatType::kDevVendor)),
      kDeviceVendorAMD);
  device_plane_builder.AddStatValue(
      *device_plane_builder.GetOrCreateStatMetadata(
          GetStatTypeStr(StatType::kGpuDeviceName)),
      "gfx942");
  device_plane_builder.AddStatValue(
      *device_plane_builder.GetOrCreateStatMetadata("clock_rate"), 2100000);
  device_plane_builder.AddStatValue(
      *device_plane_builder.GetOrCreateStatMetadata("core_count"), 304);
  device_plane_builder.AddStatValue(
      *device_plane_builder.GetOrCreateStatMetadata("memory_bandwidth"),
      uint64_t{5300} * 1000 * 1000 * 1000);

  PerfEnv perf_env = GetPerfEnvFromXPlane(*device_plane);
  // 304 CU x 2048 FLOP/clock, and 304 CU x 128 LDS B/clock, at 2.1 GHz.
  EXPECT_NEAR(1307.44, perf_env.peak_tera_flops_per_second(), kMaxError);
  // Read and write both fall back to the one derived LDS figure.
  EXPECT_NEAR(
      81715.2,
      perf_env.peak_bws_giga_bytes_per_second(MemBwType::MEM_BW_TYPE_SRAM_RD),
      kMaxError);
  EXPECT_NEAR(
      81715.2,
      perf_env.peak_bws_giga_bytes_per_second(MemBwType::MEM_BW_TYPE_SRAM_WR),
      kMaxError);
}

TEST(ConvertXPlaneToOpStats, GpuPerfEnvDerivesPeaksWhenBackendReportsZero) {
  constexpr double kMaxError = 0.01;
  XSpace space;
  // MI300X: 304 CU at 2.1 GHz, 5.3 TB/s HBM.
  XPlane* device_plane = GetOrCreateGpuXPlane(&space, /*device_ordinal=*/0);
  XPlaneBuilder device_plane_builder(device_plane);
  device_plane_builder.AddStatValue(
      *device_plane_builder.GetOrCreateStatMetadata(
          GetStatTypeStr(StatType::kDevVendor)),
      kDeviceVendorAMD);
  device_plane_builder.AddStatValue(
      *device_plane_builder.GetOrCreateStatMetadata(
          GetStatTypeStr(StatType::kGpuDeviceName)),
      "gfx942");
  device_plane_builder.AddStatValue(
      *device_plane_builder.GetOrCreateStatMetadata("clock_rate"), 2100000);
  device_plane_builder.AddStatValue(
      *device_plane_builder.GetOrCreateStatMetadata("core_count"), 304);
  device_plane_builder.AddStatValue(
      *device_plane_builder.GetOrCreateStatMetadata("memory_bandwidth"),
      uint64_t{5300} * 1000 * 1000 * 1000);
  // A stat that is present but zero must fall back just as an absent one does.
  device_plane_builder.AddStatValue(
      *device_plane_builder.GetOrCreateStatMetadata(
          GetStatTypeStr(StatType::kDevCapPeakTeraflopsPerSecond)),
      0.0);
  device_plane_builder.AddStatValue(
      *device_plane_builder.GetOrCreateStatMetadata(
          GetStatTypeStr(StatType::kDevCapPeakSramRdBwGigabytesPerSecond)),
      0.0);
  device_plane_builder.AddStatValue(
      *device_plane_builder.GetOrCreateStatMetadata(
          GetStatTypeStr(StatType::kDevCapPeakSramWrBwGigabytesPerSecond)),
      0.0);

  PerfEnv perf_env = GetPerfEnvFromXPlane(*device_plane);
  EXPECT_NEAR(1307.44, perf_env.peak_tera_flops_per_second(), kMaxError);
  EXPECT_NEAR(
      81715.2,
      perf_env.peak_bws_giga_bytes_per_second(MemBwType::MEM_BW_TYPE_SRAM_RD),
      kMaxError);
  EXPECT_NEAR(
      81715.2,
      perf_env.peak_bws_giga_bytes_per_second(MemBwType::MEM_BW_TYPE_SRAM_WR),
      kMaxError);
}

TEST(ConvertXPlaneToOpStats, GpuRunEnvironment) {
  auto space = std::make_unique<XSpace>();
  XPlaneBuilder device_plane1(
      GetOrCreateGpuXPlane(space.get(), /*device_ordinal=*/0));
  device_plane1.AddStatValue(*device_plane1.GetOrCreateStatMetadata(
                                 GetStatTypeStr(StatType::kDevVendor)),
                             kDeviceVendorNvidia);
  XPlaneBuilder device_plane2(
      GetOrCreateGpuXPlane(space.get(), /*device_ordinal=*/1));
  device_plane2.AddStatValue(*device_plane2.GetOrCreateStatMetadata(
                                 GetStatTypeStr(StatType::kDevVendor)),
                             kDeviceVendorNvidia);

  std::vector<std::unique_ptr<XSpace>> xspaces;
  xspaces.push_back(std::move(space));
  auto session_snapshot_or =
      SessionSnapshot::Create({"test_xspace"}, std::move(xspaces));
  ASSERT_OK(session_snapshot_or.status());
  OpStats op_stats;
  ASSERT_OK(ConvertMultiXSpacesToCombinedOpStats(session_snapshot_or.value(),
                                                 OpStatsOptions(), &op_stats));
  const RunEnvironment& run_env = op_stats.run_environment();

  EXPECT_EQ("Nvidia GPU", run_env.device_type());
  EXPECT_EQ(1, run_env.host_count());
  EXPECT_EQ(1, run_env.task_count());
  EXPECT_EQ(2, run_env.device_core_count());
}

TEST(ConvertXPlaneToOpStats, CpuOnlyStepDbTest) {
  constexpr int64_t kStepNum = 123;
  constexpr int64_t kStepId = 0;

  auto space = std::make_unique<XSpace>();
  XPlaneBuilder host_plane_builder(GetOrCreateHostXPlane(space.get()));
  host_plane_builder.ReserveLines(2);

  auto main_thread = host_plane_builder.GetOrCreateLine(0);
  CreateXEvent(&host_plane_builder, &main_thread, HostEventType::kTraceContext,
               0, 100, {{StatType::kStepNum, kStepNum}});
  CreateXEvent(&host_plane_builder, &main_thread, HostEventType::kFunctionRun,
               10, 90,
               {{StatType::kStepId, kStepId},
                {StatType::kProducerType, int64_t{1}},
                {StatType::kProducerId, kStepId}});

  auto tf_executor_thread = host_plane_builder.GetOrCreateLine(1);
  CreateXEvent(&host_plane_builder, &tf_executor_thread,
               HostEventType::kExecutorStateProcess, 20, 80,
               {{StatType::kStepId, kStepId},
                {StatType::kConsumerType, int64_t{1}},
                {StatType::kConsumerId, kStepId}});
  CreateXEvent(&host_plane_builder, &tf_executor_thread, "matmul", 30, 70);

  OpStatsOptions options;
  options.generate_op_metrics_db = true;
  options.generate_step_db = true;
  std::vector<std::unique_ptr<XSpace>> xspaces;
  xspaces.push_back(std::move(space));
  auto session_snapshot_or =
      SessionSnapshot::Create({"test_xspace"}, std::move(xspaces));
  ASSERT_OK(session_snapshot_or.status());
  OpStats op_stats;
  ASSERT_OK(ConvertMultiXSpacesToCombinedOpStats(session_snapshot_or.value(),
                                                 options, &op_stats));
  const StepDatabaseResult& step_db = op_stats.step_db();

  EXPECT_EQ(step_db.step_sequence_size(), 1);
}

TEST(ConvertXPlaneToOpStats, GpuStepDbTest) {
  constexpr int64_t kStepNum = 123;
  constexpr int64_t kStepId = 0;
  constexpr int64_t kCorrelationId = 100;

  auto space = std::make_unique<XSpace>();
  XPlaneBuilder host_plane_builder(GetOrCreateHostXPlane(space.get()));
  host_plane_builder.ReserveLines(2);

  auto main_thread = host_plane_builder.GetOrCreateLine(0);
  CreateXEvent(&host_plane_builder, &main_thread, HostEventType::kTraceContext,
               0, 100, {{StatType::kStepNum, kStepNum}});
  CreateXEvent(&host_plane_builder, &main_thread, HostEventType::kFunctionRun,
               10, 90,
               {{StatType::kStepId, kStepId},
                {StatType::kProducerType, int64_t{1}},
                {StatType::kProducerId, kStepId}});

  auto tf_executor_thread = host_plane_builder.GetOrCreateLine(1);
  CreateXEvent(&host_plane_builder, &tf_executor_thread,
               HostEventType::kExecutorStateProcess, 20, 20,
               {{StatType::kStepId, kStepId},
                {StatType::kConsumerType, int64_t{1}},
                {StatType::kConsumerId, kStepId}});
  CreateXEvent(&host_plane_builder, &tf_executor_thread, "matmul", 30, 10,
               {{StatType::kCorrelationId, kCorrelationId}});

  XPlaneBuilder device_plane_builder(
      GetOrCreateGpuXPlane(space.get(), /*device_ordinal=*/0));
  device_plane_builder.ReserveLines(1);

  auto stream = device_plane_builder.GetOrCreateLine(0);
  CreateXEvent(&device_plane_builder, &stream, "matmul", 50, 40,
               {{StatType::kCorrelationId, kCorrelationId}});

  OpStatsOptions options;
  options.generate_op_metrics_db = true;
  options.generate_step_db = true;
  std::vector<std::unique_ptr<XSpace>> xspaces;
  xspaces.push_back(std::move(space));
  auto session_snapshot_or =
      SessionSnapshot::Create({"test_xspace"}, std::move(xspaces));
  ASSERT_OK(session_snapshot_or.status());
  OpStats op_stats;
  ASSERT_OK(ConvertMultiXSpacesToCombinedOpStats(session_snapshot_or.value(),
                                                 options, &op_stats));
  const StepDatabaseResult& step_db = op_stats.step_db();

  EXPECT_EQ(step_db.step_sequence_size(), 1);

  PrecisionStats precision_stats =
      op_stats.device_op_metrics_db().precision_stats();
  EXPECT_EQ(precision_stats.compute_16bit_ps(), 0);
  EXPECT_EQ(precision_stats.compute_32bit_ps(), 40);
}

TEST(ConvertXPlaneToOpStats, PropagateAndDedupErrors) {
  XSpace space;
  static constexpr char kError[] = "host: error";
  *space.add_errors() = kError;
  *space.add_errors() = kError;

  ASSERT_OK_AND_ASSIGN(OpStats op_stats,
                       ConvertXSpaceToOpStats(space, OpStatsOptions()));

  EXPECT_EQ(1, op_stats.diagnostics().errors_size());
  EXPECT_EQ(kError, op_stats.diagnostics().errors(/*index=*/0));
}

TEST(ConvertXPlaneToOpStats, Hostnames) {
  XSpace space;
  static constexpr char kHost[] = "host1";
  *space.add_hostnames() = kHost;

  ASSERT_OK_AND_ASSIGN(OpStats op_stats,
                       ConvertXSpaceToOpStats(space, OpStatsOptions()));
  EXPECT_EQ(
      kHost,
      op_stats.core_id_to_details().at(kDefaultGpuLocalCoreId).hostname());
}

void BuildXSpaceForTest(XSpace& xspace, absl::string_view hostname) {
  constexpr int64_t kStepNum = 123;
  constexpr int64_t kStepId = 456;
  // Create a host only XSpace for test.
  XPlaneBuilder host_plane_builder(GetOrCreateHostXPlane(&xspace));
  host_plane_builder.ReserveLines(2);

  auto main_thread = host_plane_builder.GetOrCreateLine(0);
  CreateXEvent(&host_plane_builder, &main_thread, HostEventType::kTraceContext,
               0, 100, {{StatType::kStepNum, kStepNum}});
  CreateXEvent(&host_plane_builder, &main_thread, HostEventType::kFunctionRun,
               10, 90,
               {{StatType::kStepId, kStepId},
                {StatType::kProducerType, int64_t{1}},
                {StatType::kProducerId, kStepId}});

  auto executor_thread = host_plane_builder.GetOrCreateLine(1);
  CreateXEvent(&host_plane_builder, &executor_thread,
               HostEventType::kExecutorStateProcess, 20, 80,
               {{StatType::kStepId, kStepId},
                {StatType::kConsumerType, int64_t{1}},
                {StatType::kConsumerId, kStepId}});
  // Create a TensorFlow op that runs for 70 ps.
  CreateXEvent(&host_plane_builder, &executor_thread, "aaa:bbb", 30, 70);
  xspace.add_hostnames(std::string(hostname));
}

TEST(ConvertXPlaneToOpStats, TestConvertMultiXSpacesToCombinedOpStats) {
  static constexpr char kHost1[] = "host1";
  static constexpr char kHost2[] = "host2";

  auto xspace1 = std::make_unique<XSpace>();
  auto xspace2 = std::make_unique<XSpace>();

  BuildXSpaceForTest(*xspace1, kHost1);
  BuildXSpaceForTest(*xspace2, kHost2);

  std::vector<std::string> xspace_paths;
  xspace_paths.push_back("host1.pb");
  xspace_paths.push_back("host2.pb");

  std::vector<std::unique_ptr<XSpace>> xspaces;
  xspaces.push_back(std::move(xspace1));
  xspaces.push_back(std::move(xspace2));

  auto session_snapshot_or =
      SessionSnapshot::Create(std::move(xspace_paths), std::move(xspaces));
  ASSERT_OK(session_snapshot_or.status());

  OpStatsOptions options;
  options.generate_op_metrics_db = true;
  options.generate_step_db = true;
  OpStats combined_op_stats;

  ASSERT_OK(ConvertMultiXSpacesToCombinedOpStats(session_snapshot_or.value(),
                                                 options, &combined_op_stats))
      << "Failed to convert multi XSpace to OpStats";

  // Result OpStats has 2 Host Ops, "IDLE" and "aaa:bbb".
  ASSERT_EQ(combined_op_stats.host_op_metrics_db().metrics_db_size(), 2);
  const auto& metric = combined_op_stats.host_op_metrics_db().metrics_db(1);
  EXPECT_EQ(metric.name(), "aaa");
  EXPECT_EQ(metric.category(), "bbb");
  // Each host has the HostOp "aaa:bbb" running for 70 ps, so the combined
  // OpStats has "aaa:bbb" running for 140 ps in total.
  EXPECT_EQ(metric.self_time_ps(), 140);

  // Result OpStats has 1 step, 2 cores.
  ASSERT_EQ(combined_op_stats.step_db().step_sequence_size(), 1);
  ASSERT_EQ(
      combined_op_stats.step_db().step_sequence(0).step_info_per_core_size(),
      2);
  const auto& step_info_per_core =
      combined_op_stats.step_db().step_sequence(0).step_info_per_core();
  // global_core_id is computed using: 1000 * host_id + local_core_id.
  EXPECT_TRUE(step_info_per_core.contains(kDefaultGpuLocalCoreId));
  EXPECT_TRUE(step_info_per_core.contains(1000 + kDefaultGpuLocalCoreId));

  const auto& core_details_map = combined_op_stats.core_id_to_details();
  EXPECT_EQ(kHost1, core_details_map.at(kDefaultGpuLocalCoreId).hostname());
  EXPECT_EQ(kHost2,
            core_details_map.at(1000 + kDefaultGpuLocalCoreId).hostname());
}

TEST(ConvertXPlaneToOpStats, RunEnvironmentExtractedFromTpuPlane) {
  XSpace xspace;
  for (int i : {0, 1, 2, 3}) {
    GetOrCreateTpuXPlane(&xspace, i, "TPU V4", 0, 0);
  }

  ASSERT_OK_AND_ASSIGN(OpStats op_stats,
                       ConvertXSpaceToOpStats(xspace, OpStatsOptions()));

  EXPECT_EQ(op_stats.run_environment().device_type(), "TPU V4");
  EXPECT_EQ(op_stats.run_environment().device_core_count(), 4);
}

TEST(ConvertXPlaneToOpStats, TpuPerfEnv) {
  auto space = std::make_unique<XSpace>();
  constexpr double kMaxError = 0.01;
  constexpr int kClockRateKHz = 1530000;
  constexpr int kCoreCount = 80;
  constexpr uint64_t kMemoryBandwidthBytesPerSecond =
      uint64_t{900} * 1000 * 1000 * 1000;
  // Volta.
  constexpr int kComputeCapMajor = 7;
  constexpr int kComputeCapMinor = 0;
  constexpr double kDevCapPeakTeraflopsPerSecond = 141.0;
  constexpr double kDevCapPeakHbmBwGigabytesPerSecond = 900.0;
  constexpr double kDevCapPeakSramRdBwGigabytesPerSecond = 101.0;
  constexpr double kDevCapPeakSramWrBwGigabytesPerSecond = 102.0;
  constexpr double kDevCapPeakCmemRdBwGigabytesPerSecond = 101.0;
  constexpr double kDevCapPeakCmemWrBwGigabytesPerSecond = 102.0;
  constexpr double kDevCapPeakVmemRdBwGigabytesPerSecond = 201.0;
  constexpr double kDevCapPeakVmemWrBwGigabytesPerSecond = 202.0;

  XPlaneBuilder device_plane(GetOrCreateTpuXPlane(
      space.get(), /*device_ordinal=*/0, "TPU V4",
      kDevCapPeakTeraflopsPerSecond, kDevCapPeakHbmBwGigabytesPerSecond));
  /*device_plane.AddStatValue(*device_plane.GetOrCreateStatMetadata(
                            GetStatTypeStr(StatType::kDevVendor)),
                        kDeviceVendorNvidia); // "Google, Inc.");*/
  device_plane.AddStatValue(*device_plane.GetOrCreateStatMetadata("clock_rate"),
                            kClockRateKHz);
  device_plane.AddStatValue(*device_plane.GetOrCreateStatMetadata("core_count"),
                            kCoreCount);
  device_plane.AddStatValue(
      *device_plane.GetOrCreateStatMetadata("memory_bandwidth"),
      kMemoryBandwidthBytesPerSecond);
  device_plane.AddStatValue(
      *device_plane.GetOrCreateStatMetadata("compute_cap_major"),
      kComputeCapMajor);
  device_plane.AddStatValue(
      *device_plane.GetOrCreateStatMetadata("compute_cap_minor"),
      kComputeCapMinor);
  device_plane.AddStatValue(*device_plane.GetOrCreateStatMetadata(
                                "peak_sram_rd_bw_gigabytes_per_second"),
                            kDevCapPeakSramRdBwGigabytesPerSecond);
  device_plane.AddStatValue(*device_plane.GetOrCreateStatMetadata(
                                "peak_sram_wr_bw_gigabytes_per_second"),
                            kDevCapPeakSramWrBwGigabytesPerSecond);
  device_plane.AddStatValue(*device_plane.GetOrCreateStatMetadata(
                                "peak_cmem_rd_bw_gigabytes_per_second"),
                            kDevCapPeakCmemRdBwGigabytesPerSecond);
  device_plane.AddStatValue(*device_plane.GetOrCreateStatMetadata(
                                "peak_cmem_wr_bw_gigabytes_per_second"),
                            kDevCapPeakCmemWrBwGigabytesPerSecond);
  device_plane.AddStatValue(*device_plane.GetOrCreateStatMetadata(
                                "peak_vmem_rd_bw_gigabytes_per_second"),
                            kDevCapPeakVmemRdBwGigabytesPerSecond);
  device_plane.AddStatValue(*device_plane.GetOrCreateStatMetadata(
                                "peak_vmem_wr_bw_gigabytes_per_second"),
                            kDevCapPeakVmemWrBwGigabytesPerSecond);
  device_plane.AddStatValue(
      *device_plane.GetOrCreateStatMetadata("peak_sc_teraflops_per_second"),
      50.0);
  device_plane.AddStatValue(
      *device_plane.GetOrCreateStatMetadata("num_sparse_core_tiles"), 16);

  OpStatsOptions options;
  options.generate_op_metrics_db = true;
  std::vector<std::unique_ptr<XSpace>> xspaces;
  xspaces.push_back(std::move(space));
  auto session_snapshot_or =
      SessionSnapshot::Create({"test_xspace"}, std::move(xspaces));
  ASSERT_OK(session_snapshot_or.status());
  OpStats op_stats;
  ASSERT_OK(ConvertMultiXSpacesToCombinedOpStats(session_snapshot_or.value(),
                                                 options, &op_stats));
  const PerfEnv& perf_env = op_stats.perf_env();
  EXPECT_NEAR(kDevCapPeakTeraflopsPerSecond,
              perf_env.peak_tera_flops_per_second(), kMaxError);
  EXPECT_NEAR(50.0, perf_env.peak_sc_tera_flops_per_second(), kMaxError);
  EXPECT_EQ(16, perf_env.num_sparse_core_tiles());
  EXPECT_NEAR(
      kDevCapPeakHbmBwGigabytesPerSecond,
      perf_env.peak_bws_giga_bytes_per_second(MemBwType::MEM_BW_TYPE_HBM_RW),
      kMaxError);
  EXPECT_NEAR(
      kDevCapPeakSramRdBwGigabytesPerSecond,
      perf_env.peak_bws_giga_bytes_per_second(MemBwType::MEM_BW_TYPE_SRAM_RD),
      kMaxError);
  EXPECT_NEAR(
      kDevCapPeakSramWrBwGigabytesPerSecond,
      perf_env.peak_bws_giga_bytes_per_second(MemBwType::MEM_BW_TYPE_SRAM_WR),
      kMaxError);
  EXPECT_NEAR(
      kDevCapPeakCmemRdBwGigabytesPerSecond,
      perf_env.peak_bws_giga_bytes_per_second(MemBwType::MEM_BW_TYPE_CMEM_RD),
      kMaxError);
  EXPECT_NEAR(
      kDevCapPeakCmemWrBwGigabytesPerSecond,
      perf_env.peak_bws_giga_bytes_per_second(MemBwType::MEM_BW_TYPE_CMEM_WR),
      kMaxError);
  EXPECT_NEAR(
      kDevCapPeakVmemRdBwGigabytesPerSecond,
      perf_env.peak_bws_giga_bytes_per_second(MemBwType::MEM_BW_TYPE_VMEM_RD),
      kMaxError);
  EXPECT_NEAR(
      kDevCapPeakVmemWrBwGigabytesPerSecond,
      perf_env.peak_bws_giga_bytes_per_second(MemBwType::MEM_BW_TYPE_VMEM_WR),
      kMaxError);
  EXPECT_NEAR(156.67, perf_env.ridge_point(), kMaxError);
}

TEST(ConvertXPlaneToOpStats, TpuRunEnvironment) {
  auto space = std::make_unique<XSpace>();
  XPlaneBuilder device_plane1(
      GetOrCreateTpuXPlane(space.get(), /*device_ordinal=*/0, "TPU V4", 0, 0));
  XPlaneBuilder device_plane2(
      GetOrCreateTpuXPlane(space.get(), /*device_ordinal=*/1, "TPU V4", 0, 0));

  std::vector<std::unique_ptr<XSpace>> xspaces;
  xspaces.push_back(std::move(space));
  auto session_snapshot_or =
      SessionSnapshot::Create({"test_xspace"}, std::move(xspaces));
  ASSERT_OK(session_snapshot_or.status());
  OpStats op_stats;
  ASSERT_OK(ConvertMultiXSpacesToCombinedOpStats(session_snapshot_or.value(),
                                                 OpStatsOptions(), &op_stats));
  const RunEnvironment& run_env = op_stats.run_environment();

  EXPECT_EQ("TPU V4", run_env.device_type());
  EXPECT_EQ(1, run_env.host_count());
  EXPECT_EQ(1, run_env.task_count());
  EXPECT_EQ(2, run_env.device_core_count());
}

TEST(ConvertXPlaneToOpStats, TpuDeviceTraceToStepDb) {
  auto space = std::make_unique<XSpace>();
  constexpr double kDevCapPeakTeraflopsPerSecond = 141.0;
  constexpr double kDevCapPeakHbmBwGigabytesPerSecond = 1000.0;
  XPlaneBuilder xplane_builder(GetOrCreateTpuXPlane(
      space.get(), /*device_ordinal=*/0, "TPU V4",
      kDevCapPeakTeraflopsPerSecond, kDevCapPeakHbmBwGigabytesPerSecond));

  XEventMetadata* event_metadata = xplane_builder.GetOrCreateEventMetadata(1);
  event_metadata->set_name("op_name");
  XStatsBuilder<XEventMetadata> stats(event_metadata, &xplane_builder);

  stats.AddStatValue(*xplane_builder.GetOrCreateStatMetadata(
                         GetStatTypeStr(StatType::kProgramId)),
                     1);
  stats.AddStatValue(*xplane_builder.GetOrCreateStatMetadata(
                         GetStatTypeStr(StatType::kSymbolId)),
                     1);
  stats.AddStatValue(*xplane_builder.GetOrCreateStatMetadata(
                         GetStatTypeStr(StatType::kSelfDurationPs)),
                     10);
  stats.AddStatValue(
      *xplane_builder.GetOrCreateStatMetadata(GetStatTypeStr(StatType::kTfOp)),
      "tf_op_name");
  stats.AddStatValue(*xplane_builder.GetOrCreateStatMetadata(
                         GetStatTypeStr(StatType::kHloCategory)),
                     "category");
  XLineBuilder line = xplane_builder.GetOrCreateLine(1);
  line.SetName(tsl::profiler::kTensorFlowOpLineName);
  XEventBuilder event = line.AddEvent(*event_metadata);
  event.SetOffsetNs(0);
  event.SetDurationNs(10);

  OpStatsOptions options;
  options.generate_op_metrics_db = true;
  std::vector<std::unique_ptr<XSpace>> xspaces;
  xspaces.push_back(std::move(space));
  auto session_snapshot_or =
      SessionSnapshot::Create({"test_xspace"}, std::move(xspaces));
  ASSERT_OK(session_snapshot_or.status());
  OpStats op_stats;
  ASSERT_OK(ConvertMultiXSpacesToCombinedOpStats(session_snapshot_or.value(),
                                                 options, &op_stats));
  EXPECT_THAT(op_stats.device_op_metrics_db().metrics_db(),
              UnorderedElementsAre(Property(&OpMetrics::name, "op_name"),
                                   Property(&OpMetrics::name, "IDLE")));
}

// Verifies that the step db is generated correctly by intersecting for
// multi-device TPU.
TEST(ConvertXPlaneToOpStats, TpuMultiDeviceStepDbTest) {
  auto space = std::make_unique<XSpace>();

  XPlaneBuilder device_plane_builder1(
      GetOrCreateTpuXPlane(space.get(), /*device_ordinal=*/0, "TPU V4", 0, 0));
  XPlaneBuilder device_plane_builder2(
      GetOrCreateTpuXPlane(space.get(), /*device_ordinal=*/1, "TPU V4", 0, 0));
  device_plane_builder1.ReserveLines(1);
  device_plane_builder2.ReserveLines(1);

  // Create 1 step in xplane in TPU ordinal 0.
  XLineBuilder step_line = device_plane_builder1.GetOrCreateLine(0);
  step_line.SetName(tsl::profiler::kStepLineName);
  CreateXEvent(&device_plane_builder1, &step_line, "Step 1", /*offset_ps=*/100,
               /*duration_ps=*/100000, {{StatType::kGroupId, 1}});

  XLineBuilder op_line = device_plane_builder1.GetOrCreateLine(1);
  op_line.SetName(kXlaOpLineName);
  CreateXEvent(&device_plane_builder1, &op_line, "op.1", /*offset_ps=*/110,
               /*duration_ps=*/10,
               {{StatType::kHloCategory, tsl::profiler::kHloInfeed},
                {StatType::kGroupId, 1}});

  // Create 2 steps in xplane in TPU ordinal 1.
  step_line = device_plane_builder2.GetOrCreateLine(0);
  step_line.SetName(tsl::profiler::kStepLineName);
  CreateXEvent(&device_plane_builder2, &step_line, "Step 1", /*offset_ps=*/300,
               /*duration_ps=*/100, {{StatType::kGroupId, 1}});
  CreateXEvent(&device_plane_builder2, &step_line, "Step 2", /*offset_ps=*/300,
               /*duration_ps=*/100, {{StatType::kGroupId, 2}});
  op_line = device_plane_builder2.GetOrCreateLine(1);
  op_line.SetName(kXlaOpLineName);
  CreateXEvent(&device_plane_builder2, &op_line, "op.1", /*offset_ps=*/310,
               /*duration_ps=*/10,
               {{StatType::kHloCategory, tsl::profiler::kHloInfeed},
                {StatType::kGroupId, 1}});
  CreateXEvent(&device_plane_builder2, &op_line, "op.2", /*offset_ps=*/310,
               /*duration_ps=*/10,
               {{StatType::kHloCategory, tsl::profiler::kHloInfeed},
                {StatType::kGroupId, 2}});

  OpStatsOptions options;
  options.generate_op_metrics_db = true;
  options.generate_step_db = true;
  ASSERT_OK_AND_ASSIGN(OpStats op_stats,
                       ConvertXSpaceToOpStats(*space, options));
  const StepDatabaseResult& step_db = op_stats.step_db();
  // For TPU step events, we intersect the step events by step num across
  // different TPU devices.
  EXPECT_EQ(step_db.step_sequence_size(), 1);
}

TEST(ConvertXPlaneToOpStats, TpuTCAndSCStepDbTest) {
  auto space = std::make_unique<XSpace>();
  XPlaneBuilder tc_plane_builder(
      GetOrCreateTpuXPlane(space.get(), /*device_ordinal=*/0, "TPU V4", 0, 0));
  int64_t tc_core_id = 1;
  tc_plane_builder.SetId(tc_core_id);
  tc_plane_builder.ReserveLines(2);
  XLineBuilder tc_step_line = tc_plane_builder.GetOrCreateLine(0);
  tc_step_line.SetName(tsl::profiler::kStepLineName);
  CreateXEvent(&tc_plane_builder, &tc_step_line, "Step 1", /*offset_ps=*/100,
               /*duration_ps=*/100000, {{StatType::kGroupId, 1}});
  XLineBuilder tc_op_line = tc_plane_builder.GetOrCreateLine(1);
  tc_op_line.SetName(kXlaOpLineName);
  CreateXEvent(&tc_plane_builder, &tc_op_line, "op.1", /*offset_ps=*/110,
               /*duration_ps=*/10,
               {{StatType::kHloCategory, tsl::profiler::kHloInfeed},
                {StatType::kGroupId, 1}});

  XPlaneBuilder sc_plane_builder(
      GetOrCreateTpuXPlane(space.get(), /*device_ordinal=*/1, "TPU V4", 0, 0));
  int64_t sc_core_id = 2;
  sc_plane_builder.SetId(sc_core_id);
  sc_plane_builder.SetName("/device:TPU:0 SparseCore 0");
  sc_plane_builder.ReserveLines(2);
  XLineBuilder sc_step_line = sc_plane_builder.GetOrCreateLine(0);
  sc_step_line.SetName(tsl::profiler::kSparseCoreStepLineName);
  CreateXEvent(&sc_plane_builder, &sc_step_line, "Step 1", /*offset_ps=*/1000,
               /*duration_ps=*/10000,
               // TODO(b/397774568): Remove this once the SparseCore OpMetricsDb
               // is implemented.
               {{StatType::kGroupId, 1}, {StatType::kStepIdleTimePs, 9000}});
  XLineBuilder sc_op_line = sc_plane_builder.GetOrCreateLine(1);
  sc_op_line.SetName(kSparseCoreOpLineName);
  CreateXEvent(
      &sc_plane_builder, &sc_op_line, "op.2", /*offset_ps=*/1010,
      /*duration_ps=*/1000,
      {{StatType::kHloCategory, "sparse_core_op"}, {StatType::kGroupId, 1}});

  OpStatsOptions options;
  options.generate_op_metrics_db = true;
  options.generate_step_db = true;
  ASSERT_OK_AND_ASSIGN(OpStats op_stats,
                       ConvertXSpaceToOpStats(*space, options));
  const StepDatabaseResult& step_db = op_stats.step_db();
  EXPECT_EQ(step_db.step_sequence_size(), 1);
  EXPECT_EQ(step_db.step_sequence(0).step_info_per_core_size(), 2);
  auto step_info_per_core = step_db.step_sequence(0).step_info_per_core();
  auto tc_core_step_info = step_info_per_core[tc_core_id];
  EXPECT_EQ(tc_core_step_info.duration_ps(), 100000);
  EXPECT_EQ(tc_core_step_info.begin_ps(), 100);
  auto sc_core_step_info =
      step_info_per_core[kSparseCoreIndexStart + sc_core_id];
  EXPECT_EQ(sc_core_step_info.duration_ps(), 10000);
  EXPECT_EQ(sc_core_step_info.begin_ps(), 1000);
}

TEST(ConvertXPlaneToOpStats, ConstructDutyCycleTrackerFromXlaOps) {
  XSpace space;
  XPlane* device_plane = GetOrCreateTpuXPlane(
      &space, /*device_ordinal=*/0, /*device_type=*/"TPU v4",
      /*peak_tera_flops_per_second=*/0,
      /*peak_hbm_bw_gigabytes_per_second=*/0);
  XPlaneBuilder device_plane_builder(device_plane);
  XLineBuilder op_line = device_plane_builder.GetOrCreateLine(0);
  op_line.SetName(kXlaOpLineName);
  CreateXEventMetadata(&device_plane_builder, "op.1",
                       {{StatType::kHloCategory, tsl::profiler::kHloInfeed}});
  CreateXEvent(&device_plane_builder, &op_line, "op.1", /*offset_ps=*/10,
               /*duration_ps=*/10);
  CreateXEventMetadata(&device_plane_builder, "op.2",
                       {{StatType::kHloCategory, tsl::profiler::kHloCall}});
  CreateXEvent(&device_plane_builder, &op_line, "op.2", /*offset_ps=*/20,
               /*duration_ps=*/10);
  CreateXEventMetadata(&device_plane_builder, "op.3",
                       {{StatType::kHloCategory, tsl::profiler::kHloCall}});
  CreateXEvent(&device_plane_builder, &op_line, "op.3", /*offset_ps=*/30,
               /*duration_ps=*/10);
  CreateXEventMetadata(&device_plane_builder, "op.4",
                       {{StatType::kHloCategory, tsl::profiler::kHloOutfeed}});
  CreateXEvent(&device_plane_builder, &op_line, "op.4", /*offset_ps=*/40,
               /*duration_ps=*/10);
  XLineBuilder xla_module_line = device_plane_builder.GetOrCreateLine(1);
  xla_module_line.SetName(kXlaModuleLineName);
  CreateXEvent(&device_plane_builder, &xla_module_line, "module.1",
               /*offset_ps=*/5,
               /*duration_ps=*/50);

  XPlaneVisitor visitor = tsl::profiler::CreateTfXPlaneVisitor(device_plane);
  DutyCycleTracker tracker = ConstructDutyCycleTracker(visitor);
  EXPECT_EQ(tracker.GetActiveTimePs(), 20);
  EXPECT_EQ(tracker.GetIdleTimePs(), 30);
}

TEST(ConvertXPlaneToOpStats, ConstructDutyCycleTrackerFromSparseCore) {
  XSpace space;
  XPlane* sc_plane = GetOrCreateTpuXPlane(
      &space, /*device_ordinal=*/0, /*device_type=*/"TPU v4",
      /*peak_tera_flops_per_second=*/0,
      /*peak_hbm_bw_gigabytes_per_second=*/0);
  XPlaneBuilder sc_plane_builder(sc_plane);
  XLineBuilder op_line = sc_plane_builder.GetOrCreateLine(0);
  op_line.SetName(kSparseCoreOpLineName);
  CreateXEvent(&sc_plane_builder, &op_line, "op.1", /*offset_ps=*/10,
               /*duration_ps=*/10);
  CreateXEvent(&sc_plane_builder, &op_line, "op.2", /*offset_ps=*/20,
               /*duration_ps=*/10);
  CreateXEvent(&sc_plane_builder, &op_line, "op.3", /*offset_ps=*/30,
               /*duration_ps=*/10);
  CreateXEvent(&sc_plane_builder, &op_line, "op.4", /*offset_ps=*/40,
               /*duration_ps=*/10);
  XLineBuilder module_line = sc_plane_builder.GetOrCreateLine(1);
  module_line.SetName(kSparseCoreModuleLineName);
  CreateXEvent(&sc_plane_builder, &module_line, "module.1", /*offset_ps=*/5,
               /*duration_ps=*/50);

  XPlaneVisitor visitor = tsl::profiler::CreateTfXPlaneVisitor(sc_plane);
  DutyCycleTracker tracker = ConstructDutyCycleTracker(visitor);
  EXPECT_EQ(tracker.GetActiveTimePs(), 40);
  EXPECT_EQ(tracker.GetIdleTimePs(), 10);
}

TEST(ConvertXPlaneToOpStats, DISABLED_ConstructDutyCycleTrackerFromCustomCall) {
  XSpace space;
  XPlane* device_plane = GetOrCreateTpuXPlane(
      &space, /*device_ordinal=*/0, /*device_type=*/"TPU v4",
      /*peak_tera_flops_per_second=*/0,
      /*peak_hbm_bw_gigabytes_per_second=*/0);
  XPlaneBuilder device_plane_builder(device_plane);
  XLineBuilder op_line = device_plane_builder.GetOrCreateLine(0);
  op_line.SetName(kXlaOpLineName);

  // CustomCall with flops = 0 -> off duty
  CreateXEvent(&device_plane_builder, &op_line, "custom_call_0_flops",
               /*offset_ps=*/10,
               /*duration_ps=*/10,
               {{StatType::kHloCategory,
                 xla::HloOpcodeString(xla::HloOpcode::kCustomCall)},
                {StatType::kFlops, int64_t{0}}});

  // CustomCall with flops > 0 -> on duty
  CreateXEvent(&device_plane_builder, &op_line, "custom_call_with_flops",
               /*offset_ps=*/20,
               /*duration_ps=*/10,
               {{StatType::kHloCategory,
                 xla::HloOpcodeString(xla::HloOpcode::kCustomCall)},
                {StatType::kFlops, int64_t{5}}});

  // CustomCall without flops stat -> off duty
  CreateXEvent(&device_plane_builder, &op_line, "custom_call_no_flops",
               /*offset_ps=*/30,
               /*duration_ps=*/10,
               {{StatType::kHloCategory,
                 xla::HloOpcodeString(xla::HloOpcode::kCustomCall)}});

  XLineBuilder xla_module_line = device_plane_builder.GetOrCreateLine(1);
  xla_module_line.SetName(kXlaModuleLineName);
  CreateXEvent(&device_plane_builder, &xla_module_line, "module.1",
               /*offset_ps=*/5,
               /*duration_ps=*/50);

  XPlaneVisitor visitor = tsl::profiler::CreateTfXPlaneVisitor(device_plane);
  DutyCycleTracker tracker = ConstructDutyCycleTracker(visitor);

  // active time is op 2 (10ps)
  EXPECT_EQ(tracker.GetActiveTimePs(), 10);
  // idle time is 50ps (module) - 10ps (active) = 40ps
  EXPECT_EQ(tracker.GetIdleTimePs(), 40);
}

TEST(ConvertXPlaneToOpStats,
     DISABLED_ConstructDutyCycleTrackerFromCustomCallWithIci) {
  XSpace space;
  XPlane* device_plane = GetOrCreateTpuXPlane(
      &space, /*device_ordinal=*/0, /*device_type=*/"TPU v4",
      /*peak_tera_flops_per_second=*/0,
      /*peak_hbm_bw_gigabytes_per_second=*/0);
  XPlaneBuilder device_plane_builder(device_plane);
  XLineBuilder op_line = device_plane_builder.GetOrCreateLine(0);
  op_line.SetName(kXlaOpLineName);

  // CustomCall with uses_ici stat = 1 -> on duty (offset 10-20)
  CreateXEvent(&device_plane_builder, &op_line, "custom_call_with_ici",
               /*offset_ps=*/10,
               /*duration_ps=*/10,
               {{StatType::kHloCategory,
                 xla::HloOpcodeString(xla::HloOpcode::kCustomCall)},
                {StatType::kUsesIci, int64_t{1}}});

  // CustomCall with uses_ici = 0, no flops, no model flops -> off duty (offset
  // 20-30)
  CreateXEvent(&device_plane_builder, &op_line, "custom_call_no_ici_no_flops",
               /*offset_ps=*/20,
               /*duration_ps=*/10,
               {{StatType::kHloCategory,
                 xla::HloOpcodeString(xla::HloOpcode::kCustomCall)},
                {StatType::kUsesIci, int64_t{0}}});

  // CustomCall with uses_ici = 1, with flops -> on duty (offset 30-40)
  CreateXEvent(&device_plane_builder, &op_line,
               "custom_call_with_ici_and_flops",
               /*offset_ps=*/30,
               /*duration_ps=*/10,
               {{StatType::kHloCategory,
                 xla::HloOpcodeString(xla::HloOpcode::kCustomCall)},
                {StatType::kUsesIci, int64_t{1}},
                {StatType::kFlops, int64_t{5}}});

  // Regular CustomCall, no ici, no flops -> off duty (offset 40-50)
  CreateXEvent(&device_plane_builder, &op_line, "custom_call_no_ici",
               /*offset_ps=*/40,
               /*duration_ps=*/10,
               {{StatType::kHloCategory,
                 xla::HloOpcodeString(xla::HloOpcode::kCustomCall)}});

  // CustomCall with uses_ici = 0, but with model_flops -> on duty (offset
  // 50-60)
  CreateXEvent(&device_plane_builder, &op_line, "custom_call_with_model_flops",
               /*offset_ps=*/50,
               /*duration_ps=*/10,
               {{StatType::kHloCategory,
                 xla::HloOpcodeString(xla::HloOpcode::kCustomCall)},
                {StatType::kUsesIci, int64_t{0}},
                {StatType::kModelFlops, int64_t{5}}});

  // Megacore event overlapping with on duty op -> should not double count
  // (offset 15-25, overlaps with 10-20 and 20-30)
  CreateXEvent(&device_plane_builder, &op_line, "megacore_op",
               /*offset_ps=*/15,
               /*duration_ps=*/10, {{StatType::kHloCategory, "megacore"}});

  // Non-custom call with uses_ici = 1 -> uses_ici shouldn't matter if it has
  // flops (offset 60-70)
  CreateXEvent(&device_plane_builder, &op_line, "dot",
               /*offset_ps=*/60,
               /*duration_ps=*/10,
               {{StatType::kHloCategory, "dot"},
                {StatType::kUsesIci, int64_t{1}},
                {StatType::kFlops, int64_t{10}}});

  // Non-custom call with uses_ici = 1, no flops -> it's active based on
  // category unless IsOffDutyOp (offset 70-80)
  CreateXEvent(
      &device_plane_builder, &op_line, "add",
      /*offset_ps=*/70,
      /*duration_ps=*/10,
      {{StatType::kHloCategory, "add"}, {StatType::kUsesIci, int64_t{1}}});

  XLineBuilder xla_module_line = device_plane_builder.GetOrCreateLine(1);
  xla_module_line.SetName(kXlaModuleLineName);
  CreateXEvent(&device_plane_builder, &xla_module_line, "module.1",
               /*offset_ps=*/5,
               /*duration_ps=*/90);  // 5 to 95 (total length 90)

  // Active times:
  // 10-25 (custom call with ici from 10-20, extended to 25 by megacore_op)
  // 30-40 (custom call with ici + flops)
  // 50-60 (custom call with model flops)
  // 60-70 (dot with flops)
  // 70-80 (add)
  // total active time = 15 + 10 + 10 + 10 + 10 = 55
  // Module time = 90.
  // Idle time = 90 - 55 = 35.

  XPlaneVisitor visitor = tsl::profiler::CreateTfXPlaneVisitor(device_plane);
  DutyCycleTracker tracker = ConstructDutyCycleTracker(visitor);

  EXPECT_EQ(tracker.GetActiveTimePs(), 55);
  EXPECT_EQ(tracker.GetIdleTimePs(), 35);
}

TEST(ConvertXPlaneToOpStats, MultiCoreChipBusyAndIdleTimeTest) {
  XSpace space;
  CoreDetails tc_core_details;
  tc_core_details.set_local_chip_id(0);
  CoreDetails sc_core_details;
  sc_core_details.set_local_chip_id(0);
  XPlane* tc_plane = GetOrCreateTpuXPlane(
      &space, /*device_ordinal=*/0, /*device_type=*/"TPU v4",
      /*peak_tera_flops_per_second=*/0,
      /*peak_hbm_bw_gigabytes_per_second=*/0);
  XPlaneBuilder tc_plane_builder(tc_plane);
  tc_plane_builder.AddStatValue(*tc_plane_builder.GetOrCreateStatMetadata(
                                    GetStatTypeStr(StatType::kCoreDetails)),
                                tc_core_details);
  XLineBuilder xla_op_line = tc_plane_builder.GetOrCreateLine(0);
  xla_op_line.SetName(kXlaOpLineName);
  CreateXEvent(&tc_plane_builder, &xla_op_line, "op.1", /*offset_ps=*/10,
               /*duration_ps=*/10,
               {{StatType::kHloCategory, tsl::profiler::kHloInfeed}});
  CreateXEvent(&tc_plane_builder, &xla_op_line, "op.2", /*offset_ps=*/20,
               /*duration_ps=*/10,
               {{StatType::kHloCategory, tsl::profiler::kHloCall}});
  CreateXEvent(&tc_plane_builder, &xla_op_line, "op.3", /*offset_ps=*/30,
               /*duration_ps=*/10);
  CreateXEvent(&tc_plane_builder, &xla_op_line, "op.4", /*offset_ps=*/40,
               /*duration_ps=*/10,
               {{StatType::kHloCategory, tsl::profiler::kHloOutfeed}});
  XLineBuilder xla_module_line = tc_plane_builder.GetOrCreateLine(1);
  xla_module_line.SetName(kXlaModuleLineName);
  CreateXEvent(&tc_plane_builder, &xla_module_line, "module.1", /*offset_ps=*/5,
               /*duration_ps=*/50);

  XPlane* sc_plane = GetOrCreateTpuXPlane(
      &space, /*device_ordinal=*/1, /*device_type=*/"TPU v4",
      /*peak_tera_flops_per_second=*/0,
      /*peak_hbm_bw_gigabytes_per_second=*/0);
  XPlaneBuilder sc_plane_builder(sc_plane);
  sc_plane_builder.AddStatValue(*sc_plane_builder.GetOrCreateStatMetadata(
                                    GetStatTypeStr(StatType::kCoreDetails)),
                                sc_core_details);
  XLineBuilder sc_op_line = sc_plane_builder.GetOrCreateLine(0);
  sc_op_line.SetName(kSparseCoreOpLineName);
  CreateXEvent(&sc_plane_builder, &sc_op_line, "op.1", /*offset_ps=*/10,
               /*duration_ps=*/10);
  CreateXEvent(&sc_plane_builder, &sc_op_line, "op.2", /*offset_ps=*/20,
               /*duration_ps=*/10);
  CreateXEvent(&sc_plane_builder, &sc_op_line, "op.3", /*offset_ps=*/30,
               /*duration_ps=*/10);
  CreateXEvent(&sc_plane_builder, &sc_op_line, "op.4", /*offset_ps=*/40,
               /*duration_ps=*/10);
  XLineBuilder sc_module_line = sc_plane_builder.GetOrCreateLine(1);
  sc_module_line.SetName(kSparseCoreModuleLineName);
  CreateXEvent(&sc_plane_builder, &sc_module_line, "module.1", /*offset_ps=*/5,
               /*duration_ps=*/50);

  ASSERT_OK_AND_ASSIGN(OpStats op_stats,
                       ConvertXSpaceToOpStats(space, OpStatsOptions()));
  EXPECT_EQ(op_stats.device_op_metrics_db().idle_time_ps(), 10);
  EXPECT_EQ(op_stats.device_op_metrics_db().busy_time_ps(), 40);
}

TEST(ConvertXPlaneToOpStats, ConvertXSpaceToFlatOpMetricsDbTest) {
  XSpace space;

  auto add_module_event = [&](XPlaneBuilder& plane_builder, XLineBuilder* line,
                              const std::string& name, int64_t start_ns,
                              int64_t duration_ns, uint64_t offload_core_id,
                              uint64_t tc_start_id) {
    XEventBuilder builder =
        line->AddEvent(*plane_builder.GetOrCreateEventMetadata(name));
    builder.SetTimestampNs(start_ns);
    builder.SetDurationNs(duration_ns);

    builder.AddStatValue(*plane_builder.GetOrCreateStatMetadata(
                             GetStatTypeStr(StatType::kDeviceOffsetPs)),
                         start_ns * 1000);
    builder.AddStatValue(*plane_builder.GetOrCreateStatMetadata(
                             GetStatTypeStr(StatType::kDeviceDurationPs)),
                         duration_ns * 1000);

    builder.AddStatValue(*plane_builder.GetOrCreateStatMetadata(
                             GetStatTypeStr(StatType::kOffloadCoreId)),
                         offload_core_id);
    builder.AddStatValue(*plane_builder.GetOrCreateStatMetadata(
                             GetStatTypeStr(StatType::kTcOffloadStartId)),
                         tc_start_id);
    return builder;
  };

  // TC Plane 0
  XPlaneBuilder tc_plane_builder0(GetOrCreateTpuXPlane(
      &space, /*device_ordinal=*/0, "TPU V4", 1000.0, 2000.0));
  tc_plane_builder0.SetId(1);

  XLineBuilder tc_step_line0 = tc_plane_builder0.GetOrCreateLine(0);
  tc_step_line0.SetName(tsl::profiler::kStepLineName);
  XEventBuilder tc_step_event0 = tc_step_line0.AddEvent(
      *tc_plane_builder0.GetOrCreateEventMetadata("Step 1"));
  tc_step_event0.SetTimestampNs(0);
  tc_step_event0.SetDurationNs(10);  // 10 ns = 10000 ps
  tc_step_event0.AddStatValue(*tc_plane_builder0.GetOrCreateStatMetadata(
                                  GetStatTypeStr(StatType::kDeviceOffsetPs)),
                              int64_t{0});
  tc_step_event0.AddStatValue(*tc_plane_builder0.GetOrCreateStatMetadata(
                                  GetStatTypeStr(StatType::kDeviceDurationPs)),
                              int64_t{10000});

  XLineBuilder tc_op_line0 = tc_plane_builder0.GetOrCreateLine(1);
  tc_op_line0.SetName(kXlaOpLineName);

  XEventMetadata* tc_event_metadata =
      tc_plane_builder0.GetOrCreateEventMetadata("tc_op_1");
  {
    XStat* stat = tc_event_metadata->add_stats();
    stat->set_metadata_id(
        tc_plane_builder0
            .GetOrCreateStatMetadata(GetStatTypeStr(StatType::kProgramId))
            ->id());
    stat->set_uint64_value(1);
  }
  {
    XStat* stat = tc_event_metadata->add_stats();
    stat->set_metadata_id(
        tc_plane_builder0
            .GetOrCreateStatMetadata(GetStatTypeStr(StatType::kSymbolId))
            ->id());
    stat->set_uint64_value(1);
  }
  XEventBuilder tc_event = tc_op_line0.AddEvent(*tc_event_metadata);
  tc_event.SetTimestampNs(1);  // 1 ns = 1000 ps
  tc_event.SetDurationNs(1);   // 1 ns = 1000 ps
  tc_event.AddStatValue(*tc_plane_builder0.GetOrCreateStatMetadata(
                            GetStatTypeStr(StatType::kOffloadCoreId)),
                        uint64_t{2});
  tc_event.AddStatValue(*tc_plane_builder0.GetOrCreateStatMetadata(
                            GetStatTypeStr(StatType::kTcOffloadStartId)),
                        int64_t{10});
  tc_event.AddStatValue(*tc_plane_builder0.GetOrCreateStatMetadata(
                            GetStatTypeStr(StatType::kDeviceOffsetPs)),
                        int64_t{1000});
  tc_event.AddStatValue(*tc_plane_builder0.GetOrCreateStatMetadata(
                            GetStatTypeStr(StatType::kDeviceDurationPs)),
                        int64_t{1000});

  // Add another event to TC Plane 0
  XEventMetadata* tc_event_metadata2 =
      tc_plane_builder0.GetOrCreateEventMetadata("tc_op_2");
  {
    XStat* stat = tc_event_metadata2->add_stats();
    stat->set_metadata_id(
        tc_plane_builder0
            .GetOrCreateStatMetadata(GetStatTypeStr(StatType::kProgramId))
            ->id());
    stat->set_uint64_value(1);
  }
  {
    XStat* stat = tc_event_metadata2->add_stats();
    stat->set_metadata_id(
        tc_plane_builder0
            .GetOrCreateStatMetadata(GetStatTypeStr(StatType::kSymbolId))
            ->id());
    stat->set_uint64_value(2);
  }
  XEventBuilder tc_event2 = tc_op_line0.AddEvent(*tc_event_metadata2);
  tc_event2.SetTimestampNs(3);  // 3 ns = 3000 ps
  tc_event2.SetDurationNs(1);   // 1 ns = 1000 ps
  tc_event2.AddStatValue(*tc_plane_builder0.GetOrCreateStatMetadata(
                             GetStatTypeStr(StatType::kDeviceOffsetPs)),
                         int64_t{3000});
  tc_event2.AddStatValue(*tc_plane_builder0.GetOrCreateStatMetadata(
                             GetStatTypeStr(StatType::kDeviceDurationPs)),
                         int64_t{1000});

  // SC Plane 0 (linked to TC Plane 0, event 1)
  XPlaneBuilder sc_plane_builder0(
      GetOrCreateTpuXPlane(&space, /*device_ordinal=*/1, "TPU V4", 0, 0));
  sc_plane_builder0.SetId(2);
  sc_plane_builder0.SetName("/device:TPU:0 SparseCore 0");
  XLineBuilder sc_op_line0 = sc_plane_builder0.GetOrCreateLine(0);
  sc_op_line0.SetName(kSparseCoreOpLineName);

  XLineBuilder sc_module_line0 = sc_plane_builder0.GetOrCreateLine(1);
  sc_module_line0.SetName(tsl::profiler::kSparseCoreModuleLineName);
  add_module_event(sc_plane_builder0, &sc_module_line0, "Module1_sc0", 0, 5, 2,
                   10);

  XEventMetadata* sc_event_metadata =
      sc_plane_builder0.GetOrCreateEventMetadata("sc_op_1");
  {
    XStat* stat = sc_event_metadata->add_stats();
    stat->set_metadata_id(
        sc_plane_builder0
            .GetOrCreateStatMetadata(GetStatTypeStr(StatType::kProgramId))
            ->id());
    stat->set_uint64_value(2);
  }
  {
    XStat* stat = sc_event_metadata->add_stats();
    stat->set_metadata_id(
        sc_plane_builder0
            .GetOrCreateStatMetadata(GetStatTypeStr(StatType::kSymbolId))
            ->id());
    stat->set_uint64_value(1);
  }
  XEventBuilder sc_event = sc_op_line0.AddEvent(*sc_event_metadata);
  sc_event.SetTimestampNs(1);  // 1 ns = 1000 ps
  sc_event.SetDurationNs(1);   // 1 ns = 1000 ps
  sc_event.AddStatValue(*sc_plane_builder0.GetOrCreateStatMetadata(
                            GetStatTypeStr(StatType::kOffloadCoreId)),
                        int64_t{2});
  sc_event.AddStatValue(*sc_plane_builder0.GetOrCreateStatMetadata(
                            GetStatTypeStr(StatType::kTcOffloadStartId)),
                        int64_t{10});
  sc_event.AddStatValue(*sc_plane_builder0.GetOrCreateStatMetadata(
                            GetStatTypeStr(StatType::kDeviceOffsetPs)),
                        int64_t{1000});
  sc_event.AddStatValue(*sc_plane_builder0.GetOrCreateStatMetadata(
                            GetStatTypeStr(StatType::kDeviceDurationPs)),
                        int64_t{1000});

  // TC Plane 1
  XPlaneBuilder tc_plane_builder1(
      GetOrCreateTpuXPlane(&space, /*device_ordinal=*/2, "TPU V4", 0, 0));
  tc_plane_builder1.SetId(3);

  XLineBuilder tc_step_line1 = tc_plane_builder1.GetOrCreateLine(0);
  tc_step_line1.SetName(tsl::profiler::kStepLineName);
  XEventBuilder tc_step_event1 = tc_step_line1.AddEvent(
      *tc_plane_builder1.GetOrCreateEventMetadata("Step 1"));
  tc_step_event1.SetTimestampNs(0);
  tc_step_event1.SetDurationNs(10);
  tc_step_event1.AddStatValue(*tc_plane_builder1.GetOrCreateStatMetadata(
                                  GetStatTypeStr(StatType::kDeviceOffsetPs)),
                              int64_t{0});
  tc_step_event1.AddStatValue(*tc_plane_builder1.GetOrCreateStatMetadata(
                                  GetStatTypeStr(StatType::kDeviceDurationPs)),
                              int64_t{10000});

  XLineBuilder tc_op_line1 = tc_plane_builder1.GetOrCreateLine(1);
  tc_op_line1.SetName(kXlaOpLineName);

  XEventMetadata* tc_event_metadata3 =
      tc_plane_builder1.GetOrCreateEventMetadata("tc_op_3");
  {
    XStat* stat = tc_event_metadata3->add_stats();
    stat->set_metadata_id(
        tc_plane_builder1
            .GetOrCreateStatMetadata(GetStatTypeStr(StatType::kProgramId))
            ->id());
    stat->set_uint64_value(3);
  }
  {
    XStat* stat = tc_event_metadata3->add_stats();
    stat->set_metadata_id(
        tc_plane_builder1
            .GetOrCreateStatMetadata(GetStatTypeStr(StatType::kSymbolId))
            ->id());
    stat->set_uint64_value(1);
  }
  XEventBuilder tc_event3 = tc_op_line1.AddEvent(*tc_event_metadata3);
  tc_event3.SetTimestampNs(2);  // 2 ns = 2000 ps
  tc_event3.SetDurationNs(2);   // 2 ns = 2000 ps
  tc_event3.AddStatValue(*tc_plane_builder1.GetOrCreateStatMetadata(
                             GetStatTypeStr(StatType::kOffloadCoreId)),
                         uint64_t{4});
  tc_event3.AddStatValue(*tc_plane_builder1.GetOrCreateStatMetadata(
                             GetStatTypeStr(StatType::kTcOffloadStartId)),
                         int64_t{20});
  tc_event3.AddStatValue(*tc_plane_builder1.GetOrCreateStatMetadata(
                             GetStatTypeStr(StatType::kDeviceOffsetPs)),
                         int64_t{2000});
  tc_event3.AddStatValue(*tc_plane_builder1.GetOrCreateStatMetadata(
                             GetStatTypeStr(StatType::kDeviceDurationPs)),
                         int64_t{2000});

  // SC Plane 1 (linked to TC Plane 1, event 3)
  XPlaneBuilder sc_plane_builder1(
      GetOrCreateTpuXPlane(&space, /*device_ordinal=*/3, "TPU V4", 0, 0));
  sc_plane_builder1.SetId(4);
  sc_plane_builder1.SetName("/device:TPU:1 SparseCore 0");
  XLineBuilder sc_op_line1 = sc_plane_builder1.GetOrCreateLine(0);
  sc_op_line1.SetName(kSparseCoreOpLineName);

  XLineBuilder sc_module_line1 = sc_plane_builder1.GetOrCreateLine(1);
  sc_module_line1.SetName(tsl::profiler::kSparseCoreModuleLineName);
  add_module_event(sc_plane_builder1, &sc_module_line1, "Module1_sc1", 0, 5, 4,
                   20);

  XEventMetadata* sc_event_metadata2 =
      sc_plane_builder1.GetOrCreateEventMetadata("sc_op_2");
  {
    XStat* stat = sc_event_metadata2->add_stats();
    stat->set_metadata_id(
        sc_plane_builder1
            .GetOrCreateStatMetadata(GetStatTypeStr(StatType::kProgramId))
            ->id());
    stat->set_uint64_value(4);
  }
  {
    XStat* stat = sc_event_metadata2->add_stats();
    stat->set_metadata_id(
        sc_plane_builder1
            .GetOrCreateStatMetadata(GetStatTypeStr(StatType::kSymbolId))
            ->id());
    stat->set_uint64_value(1);
  }
  XEventBuilder sc_event2 = sc_op_line1.AddEvent(*sc_event_metadata2);
  sc_event2.SetTimestampNs(2);  // 2 ns = 2000 ps
  sc_event2.SetDurationNs(2);   // 2 ns = 2000 ps
  sc_event2.AddStatValue(*sc_plane_builder1.GetOrCreateStatMetadata(
                             GetStatTypeStr(StatType::kOffloadCoreId)),
                         int64_t{4});
  sc_event2.AddStatValue(*sc_plane_builder1.GetOrCreateStatMetadata(
                             GetStatTypeStr(StatType::kTcOffloadStartId)),
                         int64_t{20});
  sc_event2.AddStatValue(*sc_plane_builder1.GetOrCreateStatMetadata(
                             GetStatTypeStr(StatType::kDeviceOffsetPs)),
                         int64_t{2000});
  sc_event2.AddStatValue(*sc_plane_builder1.GetOrCreateStatMetadata(
                             GetStatTypeStr(StatType::kDeviceDurationPs)),
                         int64_t{2000});

  OpStatsOptions options;
  options.generate_op_metrics_db = true;
  options.generate_step_db = true;
  options.use_flat_op_metrics_db = true;
  auto flat_op_metrics_db_or = ConvertXSpaceToOpStats(space, options);

  ASSERT_TRUE(flat_op_metrics_db_or.ok());
  OpStats op_stats = *flat_op_metrics_db_or;
  const FlatOpMetricsDb& flat_op_metrics_db =
      op_stats.flat_device_op_metrics_db();

  // Verify field on OpStats instead of FlatOpMetricsDb
  EXPECT_EQ(op_stats.run_environment().device_type(), "TPU V4");

  // Verify PerfEnv on OpStats instead of FlatOpMetricsDb
  EXPECT_EQ(op_stats.perf_env().peak_tera_flops_per_second(), 1000.0);
  EXPECT_EQ(op_stats.perf_env().peak_bws_giga_bytes_per_second(0), 2000.0);

  // We expect at least 5 instances
  EXPECT_GE(flat_op_metrics_db.op_instances_size(), 5);
  EXPECT_TRUE(flat_op_metrics_db.has_precision_stats());

  // Verify specific instances
  bool found_tc_op_1 = false;
  bool found_tc_op_2 = false;
  bool found_tc_op_3 = false;
  bool found_sc_op_1 = false;
  bool found_sc_op_2 = false;

  for (const auto& op : flat_op_metrics_db.op_instances()) {
    if (op.hlo_name() == "tc_op_1") {
      found_tc_op_1 = true;
      EXPECT_EQ(op.time_ps(), 1000);
    } else if (op.hlo_name() == "tc_op_2") {
      found_tc_op_2 = true;
      EXPECT_EQ(op.time_ps(), 1000);
    } else if (op.hlo_name() == "tc_op_3") {
      found_tc_op_3 = true;
      EXPECT_EQ(op.time_ps(), 2000);
    } else if (op.hlo_name() == "sc_op_1") {
      found_sc_op_1 = true;
      EXPECT_EQ(op.time_ps(), 1000);
    } else if (op.hlo_name() == "sc_op_2") {
      found_sc_op_2 = true;
      EXPECT_EQ(op.time_ps(), 2000);
    }
  }

  EXPECT_TRUE(found_tc_op_1);
  EXPECT_TRUE(found_tc_op_2);
  EXPECT_TRUE(found_tc_op_3);
  EXPECT_TRUE(found_sc_op_1);
  EXPECT_TRUE(found_sc_op_2);
}

TEST(ConvertXPlaneToOpStats, HandleSparseCoreBusyOpMetrics) {
  XSpace space;
  XPlane* tc_plane = GetOrCreateTpuXPlane(
      &space, /*device_ordinal=*/0, /*device_type=*/"TPU v4",
      /*peak_tera_flops_per_second=*/0,
      /*peak_hbm_bw_gigabytes_per_second=*/0);
  XPlaneBuilder tc_plane_builder(tc_plane);
  tc_plane_builder.SetId(0);
  XLineBuilder tc_step_line = tc_plane_builder.GetOrCreateLine(0);
  tc_step_line.SetName(tsl::profiler::kStepLineName);
  CreateXEvent(&tc_plane_builder, &tc_step_line, "step.1", /*offset_ps=*/10,
               /*duration_ps=*/10, {{StatType::kGroupId, int64_t{1}}});
  CreateXEvent(&tc_plane_builder, &tc_step_line, "step.2", /*offset_ps=*/20,
               /*duration_ps=*/10, {{StatType::kGroupId, int64_t{2}}});
  CreateXEvent(&tc_plane_builder, &tc_step_line, "step.3", /*offset_ps=*/30,
               /*duration_ps=*/10, {{StatType::kGroupId, int64_t{3}}});
  CreateXEvent(&tc_plane_builder, &tc_step_line, "step.4", /*offset_ps=*/40,
               /*duration_ps=*/10, {{StatType::kGroupId, int64_t{4}}});
  XLineBuilder tc_module_line = tc_plane_builder.GetOrCreateLine(1);
  tc_module_line.SetName(kXlaModuleLineName);
  CreateXEvent(&tc_plane_builder, &tc_module_line, "module.1", /*offset_ps=*/10,
               /*duration_ps=*/10, {{StatType::kGroupId, int64_t{1}}});
  CreateXEvent(&tc_plane_builder, &tc_module_line, "module.2", /*offset_ps=*/20,
               /*duration_ps=*/10, {{StatType::kGroupId, int64_t{2}}});
  CreateXEvent(&tc_plane_builder, &tc_module_line, "module.3", /*offset_ps=*/30,
               /*duration_ps=*/10, {{StatType::kGroupId, int64_t{3}}});
  CreateXEvent(&tc_plane_builder, &tc_module_line, "module.4", /*offset_ps=*/40,
               /*duration_ps=*/10, {{StatType::kGroupId, int64_t{4}}});
  XLineBuilder tc_op_line = tc_plane_builder.GetOrCreateLine(2);
  tc_op_line.SetName(kXlaOpLineName);
  auto& program_id_stat = *tc_plane_builder.GetOrCreateStatMetadata(
      GetStatTypeStr(StatType::kProgramId));
  auto& symbol_id_stat = *tc_plane_builder.GetOrCreateStatMetadata(
      GetStatTypeStr(StatType::kSymbolId));
  XStatsBuilder<XEventMetadata> op1_stats(
      tc_plane_builder.GetOrCreateEventMetadata("op.1"), &tc_plane_builder);
  op1_stats.AddStatValue(program_id_stat, 1);
  op1_stats.AddStatValue(symbol_id_stat, 1);
  XStatsBuilder<XEventMetadata> op2_stats(
      tc_plane_builder.GetOrCreateEventMetadata("op.2"), &tc_plane_builder);
  op2_stats.AddStatValue(program_id_stat, 1);
  op2_stats.AddStatValue(symbol_id_stat, 2);
  XStatsBuilder<XEventMetadata> op3_stats(
      tc_plane_builder.GetOrCreateEventMetadata("op.3"), &tc_plane_builder);
  op3_stats.AddStatValue(program_id_stat, 1);
  op3_stats.AddStatValue(symbol_id_stat, 3);
  XStatsBuilder<XEventMetadata> op4_stats(
      tc_plane_builder.GetOrCreateEventMetadata("op.4"), &tc_plane_builder);
  op4_stats.AddStatValue(program_id_stat, 1);
  op4_stats.AddStatValue(symbol_id_stat, 4);
  CreateXEvent(&tc_plane_builder, &tc_op_line, "op.1", /*offset_ps=*/15,
               /*duration_ps=*/5, {{StatType::kGroupId, int64_t{1}}});
  CreateXEvent(&tc_plane_builder, &tc_op_line, "op.2", /*offset_ps=*/25,
               /*duration_ps=*/5, {{StatType::kGroupId, int64_t{2}}});
  CreateXEvent(&tc_plane_builder, &tc_op_line, "op.3", /*offset_ps=*/35,
               /*duration_ps=*/5, {{StatType::kGroupId, int64_t{3}}});
  CreateXEvent(&tc_plane_builder, &tc_op_line, "op.4", /*offset_ps=*/45,
               /*duration_ps=*/5, {{StatType::kGroupId, int64_t{4}}});
  XPlane* sc_plane = GetOrCreateTpuXPlane(
      &space, /*device_ordinal=*/1, /*device_type=*/"TPU v4",
      /*peak_tera_flops_per_second=*/0,
      /*peak_hbm_bw_gigabytes_per_second=*/0);
  XPlaneBuilder sc_plane_builder(sc_plane);
  sc_plane_builder.SetId(1);
  sc_plane_builder.SetName(
      absl::StrCat(sc_plane->name(), " SparseCore ", sc_plane->id()));
  XLineBuilder sc_step_line = sc_plane_builder.GetOrCreateLine(0);
  sc_step_line.SetName(tsl::profiler::kSparseCoreStepLineName);
  CreateXEvent(&sc_plane_builder, &sc_step_line, "step.1", /*offset_ps=*/10,
               /*duration_ps=*/10,
               {{StatType::kStepIdleTimePs, int64_t{5}},
                {StatType::kGroupId, int64_t{1}}});
  CreateXEvent(&sc_plane_builder, &sc_step_line, "step.2", /*offset_ps=*/20,
               /*duration_ps=*/10,
               {{StatType::kStepIdleTimePs, int64_t{5}},
                {StatType::kGroupId, int64_t{2}}});
  CreateXEvent(&sc_plane_builder, &sc_step_line, "step.3", /*offset_ps=*/30,
               /*duration_ps=*/10,
               {{StatType::kStepIdleTimePs, int64_t{5}},
                {StatType::kGroupId, int64_t{3}}});
  CreateXEvent(&sc_plane_builder, &sc_step_line, "step.4", /*offset_ps=*/40,
               /*duration_ps=*/10,
               {{StatType::kStepIdleTimePs, int64_t{5}},
                {StatType::kGroupId, int64_t{4}}});
  XLineBuilder sc_module_line = sc_plane_builder.GetOrCreateLine(1);
  sc_module_line.SetName(kSparseCoreModuleLineName);
  CreateXEvent(&sc_plane_builder, &sc_module_line, "module.1", /*offset_ps=*/10,
               /*duration_ps=*/10, {{StatType::kGroupId, int64_t{1}}});
  CreateXEvent(&sc_plane_builder, &sc_module_line, "module.2", /*offset_ps=*/20,
               /*duration_ps=*/10, {{StatType::kGroupId, int64_t{2}}});
  CreateXEvent(&sc_plane_builder, &sc_module_line, "module.3", /*offset_ps=*/30,
               /*duration_ps=*/10, {{StatType::kGroupId, int64_t{3}}});
  CreateXEvent(&sc_plane_builder, &sc_module_line, "module.4", /*offset_ps=*/40,
               /*duration_ps=*/10, {{StatType::kGroupId, int64_t{4}}});
  XLineBuilder sc_op_line = sc_plane_builder.GetOrCreateLine(2);
  sc_op_line.SetName(kSparseCoreOpLineName);
  CreateXEvent(&sc_plane_builder, &sc_op_line, "scs op.1", /*offset_ps=*/15,
               /*duration_ps=*/5, {{StatType::kGroupId, int64_t{1}}});
  CreateXEvent(&sc_plane_builder, &sc_op_line, "scs op.2", /*offset_ps=*/25,
               /*duration_ps=*/5, {{StatType::kGroupId, int64_t{2}}});
  CreateXEvent(&sc_plane_builder, &sc_op_line, "scs op.3", /*offset_ps=*/35,
               /*duration_ps=*/5, {{StatType::kGroupId, int64_t{3}}});
  CreateXEvent(&sc_plane_builder, &sc_op_line, "scs op.4", /*offset_ps=*/45,
               /*duration_ps=*/5, {{StatType::kGroupId, int64_t{4}}});
  ASSERT_OK_AND_ASSIGN(OpStats op_stats,
                       ConvertXSpaceToOpStats(
                           space, OpStatsOptions{.generate_op_metrics_db = true,
                                                 .generate_step_db = true}));
  EXPECT_EQ(op_stats.device_op_metrics_db().total_time_ps(), 40);
  EXPECT_EQ(op_stats.device_op_metrics_db().total_op_time_ps(), 20);
  EXPECT_EQ(op_stats.step_db().step_sequence_size(), 4);
  EXPECT_EQ(op_stats.hlo_metrics_db_complete_steps_only().total_time_ps(), 40);
  EXPECT_EQ(op_stats.hlo_metrics_db_complete_steps_only().total_op_time_ps(),
            20);
}

TEST(ConvertXPlaneToOpStats, HandleInputPipelineSlownessCausingDeviceIdleness) {
  auto space = std::make_unique<XSpace>();
  constexpr int64_t kGroupId = 1;

  // Create a TPU XPlane with a single step and a single compute op.
  XPlaneBuilder tpu_plane_builder(
      GetOrCreateTpuXPlane(space.get(), /*device_ordinal=*/0, "TPU V4", 0, 0));
  tpu_plane_builder.SetId(0);
  tpu_plane_builder.ReserveLines(2);

  // TPU Step Line
  XLineBuilder tpu_step_line = tpu_plane_builder.GetOrCreateLine(0);
  tpu_step_line.SetName(tsl::profiler::kStepLineName);
  CreateXEvent(&tpu_plane_builder, &tpu_step_line, "Step 1", /*offset_ps=*/1000,
               /*duration_ps=*/10000, {{StatType::kGroupId, kGroupId}});

  // TPU XLA Op Line
  XLineBuilder tpu_op_line = tpu_plane_builder.GetOrCreateLine(1);
  tpu_op_line.SetName(kXlaOpLineName);
  CreateXEventMetadata(&tpu_plane_builder, "op.1",
                       {{StatType::kHloCategory, "arithmetic"},
                        {StatType::kProgramId, 1},
                        {StatType::kSymbolId, 1},
                        {StatType::kFlops, 1000}});
  CreateXEvent(&tpu_plane_builder, &tpu_op_line, "op.1",
               /*offset_ps=*/2000,
               /*duration_ps=*/8000, {{StatType::kGroupId, kGroupId}});

  // Create a Host XPlane with a single input pipeline op.
  XPlaneBuilder host_plane_builder(GetOrCreateHostXPlane(space.get()));
  host_plane_builder.ReserveLines(1);

  // Host Main Thread Line
  XLineBuilder host_main_thread = host_plane_builder.GetOrCreateLine(0);
  host_main_thread.SetName("main");
  CreateXEvent(&host_plane_builder, &host_main_thread,
               "Iterator::Batch::Map::TFRecord",
               /*offset_ps=*/500,
               /*duration_ps=*/2300,
               {{StatType::kGroupId, kGroupId},
                {StatType::kInputPipelineStageId, 1},
                {StatType::kInputPipelineStageName, "TFRecord"}});

  ASSERT_OK_AND_ASSIGN(
      OpStats op_stats,
      ConvertXSpaceToOpStats(*space,
                             OpStatsOptions{.generate_op_metrics_db = true,
                                            .generate_step_db = true}));
  EXPECT_EQ(op_stats.step_db().step_sequence_size(), 1);
  EXPECT_EQ(op_stats.step_db().step_sequence(0).step_info_per_core_size(), 1);
  auto step_info_per_core =
      op_stats.step_db().step_sequence(0).step_info_per_core();
  auto step_info = step_info_per_core[0];
  GenericStepBreakdown step_breakdown;
  ASSERT_TRUE(step_info.step_breakdown().UnpackTo(&step_breakdown));
  auto category_ps = step_breakdown.category_ps();
  ASSERT_TRUE(category_ps.contains("IDLE"));
  EXPECT_EQ(step_breakdown.category_ps().at("IDLE"), 0);
  ASSERT_TRUE(category_ps.contains("infeed"));
  EXPECT_EQ(step_breakdown.category_ps().at("infeed"), 2000);
  ASSERT_TRUE(category_ps.contains("arithmetic"));
  EXPECT_EQ(step_breakdown.category_ps().at("arithmetic"), 8000);
}

TEST(ConvertXPlaneToOpStats,
     MissingExpectedEventsDoesNotPopulateDisaggregatedServingLatency) {
  XSpace space;
  XPlaneBuilder host_plane_builder(GetOrCreateHostXPlane(&space));
  // Not a wiz inference request, so disaggregated_serving_latency should be
  // empty.
  ASSERT_OK_AND_ASSIGN(OpStats op_stats,
                       ConvertXSpaceToOpStats(space, OpStatsOptions()));
  ASSERT_FALSE(op_stats.has_disaggregated_serving_latency());

  // Make it a wiz inference request but without jit_generate events.
  XLineBuilder line = host_plane_builder.GetOrCreateLine(0);
  CreateXEvent(&host_plane_builder, &line, "WizServable", 0, 100);
  ASSERT_OK_AND_ASSIGN(op_stats,
                       ConvertXSpaceToOpStats(space, OpStatsOptions()));
  ASSERT_FALSE(op_stats.has_disaggregated_serving_latency());
}

TEST(ConvertXPlaneToOpStats, PopulateDisaggregatedServingLatency) {
  XSpace space;
  XPlaneBuilder host_plane_builder(GetOrCreateHostXPlane(&space));
  XLineBuilder host_line = host_plane_builder.GetOrCreateLine(0);
  CreateXEvent(&host_plane_builder, &host_line, "WizServable", 0, 100);

  XPlaneBuilder device_plane_builder(
      GetOrCreateTpuXPlane(&space, 0, "TPU V4", 0, 0));
  XLineBuilder device_line = device_plane_builder.GetOrCreateLine(0);
  device_line.SetName(kXlaModuleLineName);
  int duration_ps = 2000000;
  CreateXEvent(&device_plane_builder, &device_line, "jit_generate", 100,
               duration_ps);

  ASSERT_OK_AND_ASSIGN(OpStats op_stats,
                       ConvertXSpaceToOpStats(space, OpStatsOptions()));
  ASSERT_TRUE(op_stats.has_disaggregated_serving_latency());
  ASSERT_EQ(op_stats.disaggregated_serving_latency().num_decode_steps(), 1);
  double expected_avg_duration_us =
      tsl::profiler::PicoToMicro(duration_ps);
  ASSERT_EQ(
      op_stats.disaggregated_serving_latency().decode_step_time_us().avg(),
      expected_avg_duration_us);
}

TEST(ConvertXPlaneToOpStats, ConstructHCDutyCycleTrackerFromCustomCall) {
  XSpace space;
  XPlane* device_plane = GetOrCreateTpuXPlane(
      &space, /*device_ordinal=*/0, /*device_type=*/"TPU v4",
      /*peak_tera_flops_per_second=*/0,
      /*peak_hbm_bw_gigabytes_per_second=*/0);
  XPlaneBuilder device_plane_builder(device_plane);
  XLineBuilder op_line = device_plane_builder.GetOrCreateLine(0);
  op_line.SetName(kXlaOpLineName);

  // CustomCall with flops = 0 -> off duty due to High Confidence definition
  CreateXEventMetadata(&device_plane_builder, "custom_call_0_flops",
                       {{StatType::kHloCategory,
                         xla::HloOpcodeString(xla::HloOpcode::kCustomCall)}});
  CreateXEvent(&device_plane_builder, &op_line, "custom_call_0_flops",
               /*offset_ps=*/10,
               /*duration_ps=*/10, {{StatType::kFlops, int64_t{0}}});

  // CustomCall with flops > 0 -> on duty
  CreateXEventMetadata(&device_plane_builder, "custom_call_with_flops",
                       {{StatType::kHloCategory,
                         xla::HloOpcodeString(xla::HloOpcode::kCustomCall)}});
  CreateXEvent(&device_plane_builder, &op_line, "custom_call_with_flops",
               /*offset_ps=*/20,
               /*duration_ps=*/10, {{StatType::kFlops, int64_t{5}}});

  // CustomCall without flops stat -> off duty due to High Confidence
  CreateXEventMetadata(&device_plane_builder, "custom_call_no_flops",
                       {{StatType::kHloCategory,
                         xla::HloOpcodeString(xla::HloOpcode::kCustomCall)}});
  CreateXEvent(&device_plane_builder, &op_line, "custom_call_no_flops",
               /*offset_ps=*/30,
               /*duration_ps=*/10);

  XLineBuilder xla_module_line = device_plane_builder.GetOrCreateLine(1);
  xla_module_line.SetName(kXlaModuleLineName);
  CreateXEvent(&device_plane_builder, &xla_module_line, "module.1",
               /*offset_ps=*/5,
               /*duration_ps=*/50);

  XPlaneVisitor visitor = tsl::profiler::CreateTfXPlaneVisitor(device_plane);
  DutyCycleTracker tracker = ConstructHCDutyCycleTracker(visitor);

  // High confidence tracker filters out unknown custom calls (off-duty).
  // Active times: 20-30 (custom_call_with_flops) => 10ps total
  EXPECT_EQ(tracker.GetActiveTimePs(), 10);
  // idle time is 50ps (module) - 10ps (active) = 40ps
  EXPECT_EQ(tracker.GetIdleTimePs(), 40);
}

TEST(ConvertXPlaneToOpStats, ConstructHCDutyCycleTrackerFromCustomCallWithIci) {
  XSpace space;
  XPlane* device_plane = GetOrCreateTpuXPlane(
      &space, /*device_ordinal=*/0, /*device_type=*/"TPU v4",
      /*peak_tera_flops_per_second=*/0,
      /*peak_hbm_bw_gigabytes_per_second=*/0);
  XPlaneBuilder device_plane_builder(device_plane);
  XLineBuilder op_line = device_plane_builder.GetOrCreateLine(0);
  op_line.SetName(kXlaOpLineName);

  // CustomCall with uses_ici stat = 1 -> on duty (offset 10-20)
  CreateXEventMetadata(&device_plane_builder, "custom_call_with_ici",
                       {{StatType::kHloCategory,
                         xla::HloOpcodeString(xla::HloOpcode::kCustomCall)}});
  CreateXEvent(&device_plane_builder, &op_line, "custom_call_with_ici",
               /*offset_ps=*/10,
               /*duration_ps=*/10, {{StatType::kUsesIci, int64_t{1}}});

  // CustomCall with uses_ici = 0, no flops, no model flops -> off duty for
  // HC (offset 20-30)
  CreateXEventMetadata(&device_plane_builder, "custom_call_no_ici_no_flops",
                       {{StatType::kHloCategory,
                         xla::HloOpcodeString(xla::HloOpcode::kCustomCall)}});
  CreateXEvent(&device_plane_builder, &op_line, "custom_call_no_ici_no_flops",
               /*offset_ps=*/20,
               /*duration_ps=*/10, {{StatType::kUsesIci, int64_t{0}}});

  // CustomCall with uses_ici = 1, with flops -> on duty (offset 30-40)
  CreateXEventMetadata(&device_plane_builder, "custom_call_with_ici_and_flops",
                       {{StatType::kHloCategory,
                         xla::HloOpcodeString(xla::HloOpcode::kCustomCall)}});
  CreateXEvent(
      &device_plane_builder, &op_line, "custom_call_with_ici_and_flops",
      /*offset_ps=*/30,
      /*duration_ps=*/10,
      {{StatType::kUsesIci, int64_t{1}}, {StatType::kFlops, int64_t{5}}});

  // Regular CustomCall, no ici, no flops -> off duty for HC (offset 40-50)
  CreateXEventMetadata(&device_plane_builder, "custom_call_no_ici",
                       {{StatType::kHloCategory,
                         xla::HloOpcodeString(xla::HloOpcode::kCustomCall)}});
  CreateXEvent(&device_plane_builder, &op_line, "custom_call_no_ici",
               /*offset_ps=*/40,
               /*duration_ps=*/10);

  // CustomCall with uses_ici = 0, but with model_flops -> on duty
  // (offset 50-60)
  CreateXEventMetadata(&device_plane_builder, "custom_call_with_model_flops",
                       {{StatType::kHloCategory,
                         xla::HloOpcodeString(xla::HloOpcode::kCustomCall)}});
  CreateXEvent(
      &device_plane_builder, &op_line, "custom_call_with_model_flops",
      /*offset_ps=*/50,
      /*duration_ps=*/10,
      {{StatType::kUsesIci, int64_t{0}}, {StatType::kModelFlops, int64_t{5}}});

  // Megacore event overlapping with on duty op -> should not double count
  // (offset 15-25)
  CreateXEventMetadata(&device_plane_builder, "megacore_op",
                       {{StatType::kHloCategory, "megacore"}});
  CreateXEvent(&device_plane_builder, &op_line, "megacore_op",
               /*offset_ps=*/15,
               /*duration_ps=*/10);

  // Non-custom call (category "dot") -> on duty based on compute category;
  // custom-call FLOPs/ICI checks do not apply (offset 60-70)
  CreateXEventMetadata(&device_plane_builder, "dot",
                       {{StatType::kHloCategory, "dot"}});
  CreateXEvent(
      &device_plane_builder, &op_line, "dot",
      /*offset_ps=*/60,
      /*duration_ps=*/10,
      {{StatType::kUsesIci, int64_t{1}}, {StatType::kFlops, int64_t{10}}});

  // Non-custom call (category "add") without flops -> on duty based on
  // compute category unless IsOffDutyOp returns true (offset 70-80)
  CreateXEventMetadata(&device_plane_builder, "add",
                       {{StatType::kHloCategory, "add"}});
  CreateXEvent(&device_plane_builder, &op_line, "add",
               /*offset_ps=*/70,
               /*duration_ps=*/10, {{StatType::kUsesIci, int64_t{1}}});

  XLineBuilder xla_module_line = device_plane_builder.GetOrCreateLine(1);
  xla_module_line.SetName(kXlaModuleLineName);
  CreateXEvent(&device_plane_builder, &xla_module_line, "module.1",
               /*offset_ps=*/5,
               /*duration_ps=*/90);  // 5 to 95 (total length 90)

  // Under High Confidence (HC) Duty Cycle tracking:
  // - CustomCalls without FLOPs and without ICI are treated as off-duty
  //   (filtered out). Thus custom_call_no_ici_no_flops (20-30ps) and
  //   custom_call_no_ici (40-50ps) are excluded.
  // - Standard compute ops ("dot", "add", "megacore") remain on-duty based
  //   on their HLO category.
  //
  // Active Time Calculation:
  // - 10-25ps: custom_call_with_ici (10-20ps) merged with overlapping
  //   megacore_op (15-25ps) = 15ps
  // - 30-40ps: custom_call_with_ici_and_flops = 10ps
  // - 50-60ps: custom_call_with_model_flops = 10ps
  // - 60-70ps: dot = 10ps
  // - 70-80ps: add = 10ps
  //
  // Total Active Time = 15ps + 10ps + 10ps + 10ps + 10ps = 55ps.
  // Total Module Time = 90ps (5-95ps).
  // Total Idle Time = 90ps - 55ps = 35ps.

  XPlaneVisitor visitor = tsl::profiler::CreateTfXPlaneVisitor(device_plane);
  DutyCycleTracker tracker = ConstructHCDutyCycleTracker(visitor);

  EXPECT_EQ(tracker.GetActiveTimePs(), 55);
  EXPECT_EQ(tracker.GetIdleTimePs(), 35);
}

TEST(ConvertXPlaneToOpStats,
     ConstructDutyCycleTrackerCarvesOutNestedSyncWaitChildFromParent) {
  XSpace space;
  XPlane* device_plane = GetOrCreateTpuXPlane(
      &space, /*device_ordinal=*/0, /*device_type=*/"TPU v5p",
      /*peak_tera_flops_per_second=*/0,
      /*peak_hbm_bw_gigabytes_per_second=*/0);
  XPlaneBuilder device_plane_builder(device_plane);
  XLineBuilder op_line = device_plane_builder.GetOrCreateLine(0);
  op_line.SetName(kXlaOpLineName);

  // Parent op at [10, 110) (duration 100ps)
  CreateXEventMetadata(&device_plane_builder, "parent_op",
                       {{StatType::kHloCategory, "fusion"}});
  CreateXEvent(&device_plane_builder, &op_line, "parent_op",
               /*offset_ps=*/10, /*duration_ps=*/100);

  // Nested call-done child at [30, 90) (duration 60ps, off-duty sync wait)
  CreateXEventMetadata(&device_plane_builder, "call-done.1",
                       {{StatType::kHloCategory,
                         xla::HloOpcodeString(xla::HloOpcode::kAsyncDone)}});
  CreateXEvent(&device_plane_builder, &op_line, "call-done.1",
               /*offset_ps=*/30, /*duration_ps=*/60);

  XLineBuilder xla_module_line = device_plane_builder.GetOrCreateLine(1);
  xla_module_line.SetName(kXlaModuleLineName);
  CreateXEvent(&device_plane_builder, &xla_module_line, "module.1",
               /*offset_ps=*/10, /*duration_ps=*/100);

  XPlaneVisitor visitor = tsl::profiler::CreateTfXPlaneVisitor(device_plane);
  DutyCycleTracker tracker = ConstructDutyCycleTracker(visitor);

  // Active time should be [10, 30) + [90, 110) = 40ps, and idle time 60ps.
  EXPECT_EQ(tracker.GetActiveTimePs(), 40);
  EXPECT_EQ(tracker.GetIdleTimePs(), 60);
}

TEST(ConvertXPlaneToOpStats,
     ConstructDutyCycleTrackersMatchSyncWaitsOnShortOpNameOnly) {
  XSpace space;
  XPlane* device_plane = GetOrCreateTpuXPlane(
      &space, /*device_ordinal=*/0, /*device_type=*/"TPU v5p",
      /*peak_tera_flops_per_second=*/0,
      /*peak_hbm_bw_gigabytes_per_second=*/0);
  XPlaneBuilder device_plane_builder(device_plane);
  XLineBuilder op_line = device_plane_builder.GetOrCreateLine(0);
  op_line.SetName(kXlaOpLineName);

  // HLO op events are named by their full HLO text and display the short op
  // name. The sync wait at [10, 40) has no off-duty category, so only its short
  // name marks it idle.
  constexpr absl::string_view kCallDoneHloText =
      "%call-done.3 = f32[8]{0} call-done(f32[8]{0} %call-start.3)";
  device_plane_builder.GetOrCreateEventMetadata(kCallDoneHloText)
      ->set_display_name("call-done.3");
  CreateXEvent(&device_plane_builder, &op_line, kCallDoneHloText,
               /*offset_ps=*/10, /*duration_ps=*/30);

  // The fusion at [40, 100) consumes the call-done result, so its HLO text
  // names %call-done.3 as an operand. It does real work and stays active.
  constexpr absl::string_view kFusionHloText =
      "%fusion.5 = f32[8]{0} fusion(f32[8]{0} %call-done.3), kind=kLoop";
  CreateXEventMetadata(&device_plane_builder, kFusionHloText,
                       {{StatType::kHloCategory, "loop fusion"}});
  device_plane_builder.GetOrCreateEventMetadata(kFusionHloText)
      ->set_display_name("fusion.5");
  CreateXEvent(&device_plane_builder, &op_line, kFusionHloText,
               /*offset_ps=*/40, /*duration_ps=*/60);

  XPlaneVisitor visitor = tsl::profiler::CreateTfXPlaneVisitor(device_plane);
  DutyCycleTrackers trackers = ConstructDutyCycleTrackers(visitor);

  EXPECT_EQ(trackers.duty_cycle_tracker.GetActiveTimePs(), 60);
  EXPECT_EQ(trackers.duty_cycle_tracker.GetIdleTimePs(), 30);
  EXPECT_EQ(trackers.duty_cycle_hc_tracker.GetActiveTimePs(), 60);
  EXPECT_EQ(trackers.duty_cycle_hc_tracker.GetIdleTimePs(), 30);
}

TEST(ConvertXPlaneToOpStats,
     ConstructHCDutyCycleTrackerKeepsSparseCoreCustomCallsActive) {
  XSpace space;
  XPlane* sc_plane = GetOrCreateTpuXPlane(
      &space, /*device_ordinal=*/0, /*device_type=*/"TPU v5p",
      /*peak_tera_flops_per_second=*/0,
      /*peak_hbm_bw_gigabytes_per_second=*/0);
  XPlaneBuilder sc_plane_builder(sc_plane);
  XLineBuilder op_line = sc_plane_builder.GetOrCreateLine(0);
  op_line.SetName(kSparseCoreOpLineName);

  // A custom call without flops counts as idle in the high confidence tracker
  // only on the TensorCore XLA Ops line. On SparseCore lines it stays active.
  CreateXEventMetadata(&sc_plane_builder, "custom_call_no_flops",
                       {{StatType::kHloCategory,
                         xla::HloOpcodeString(xla::HloOpcode::kCustomCall)}});
  CreateXEvent(&sc_plane_builder, &op_line, "custom_call_no_flops",
               /*offset_ps=*/10, /*duration_ps=*/40);

  XPlaneVisitor visitor = tsl::profiler::CreateTfXPlaneVisitor(sc_plane);
  DutyCycleTracker tracker = ConstructHCDutyCycleTracker(visitor);

  EXPECT_EQ(tracker.GetActiveTimePs(), 40);
  EXPECT_EQ(tracker.GetIdleTimePs(), 0);
}

TEST(ConvertXPlaneToOpStats,
     ConstructDutyCycleTrackerUnionsTecOpLinesAndSkipsOtherTecLines) {
  XSpace space;
  XPlane* sc_plane = GetOrCreateTpuXPlane(
      &space, /*device_ordinal=*/0, /*device_type=*/"TPU v5p",
      /*peak_tera_flops_per_second=*/0,
      /*peak_hbm_bw_gigabytes_per_second=*/0);
  XPlaneBuilder sc_plane_builder(sc_plane);
  CreateXEventMetadata(&sc_plane_builder, "fusion.1",
                       {{StatType::kHloCategory, "fusion"}});

  // Parallel work on the Sparse Core Ops line and two TEC lines: [10, 40),
  // [30, 60) and [50, 70) merge into [10, 70).
  XLineBuilder op_line = sc_plane_builder.GetOrCreateLine(0);
  op_line.SetName(kSparseCoreOpLineName);
  CreateXEvent(&sc_plane_builder, &op_line, "fusion.1", /*offset_ps=*/10,
               /*duration_ps=*/30);
  XLineBuilder tec0_line = sc_plane_builder.GetOrCreateLine(1);
  tec0_line.SetName("TEC 0");
  CreateXEvent(&sc_plane_builder, &tec0_line, "fusion.1", /*offset_ps=*/30,
               /*duration_ps=*/30);
  XLineBuilder tec15_line = sc_plane_builder.GetOrCreateLine(2);
  tec15_line.SetName("TEC 15");
  CreateXEvent(&sc_plane_builder, &tec15_line, "fusion.1", /*offset_ps=*/50,
               /*duration_ps=*/20);

  // A TraceMe line shares the "TEC " prefix but holds no ops.
  XLineBuilder tec_traceme_line = sc_plane_builder.GetOrCreateLine(3);
  tec_traceme_line.SetName("TEC TraceMe 0");
  CreateXEvent(&sc_plane_builder, &tec_traceme_line, "annotation",
               /*offset_ps=*/0, /*duration_ps=*/100);

  XPlaneVisitor visitor = tsl::profiler::CreateTfXPlaneVisitor(sc_plane);
  DutyCycleTracker tracker = ConstructDutyCycleTracker(visitor);

  EXPECT_EQ(tracker.GetActiveTimePs(), 60);
  EXPECT_EQ(tracker.GetIdleTimePs(), 0);
}

TEST(ConvertXPlaneToOpStats,
     ConstructDutyCycleTrackerCountsUncategorizedSparseCoreParentsByChildren) {
  XSpace space;
  XPlane* sc_plane = GetOrCreateTpuXPlane(
      &space, /*device_ordinal=*/0, /*device_type=*/"TPU v5p",
      /*peak_tera_flops_per_second=*/0,
      /*peak_hbm_bw_gigabytes_per_second=*/0);
  XPlaneBuilder sc_plane_builder(sc_plane);
  XLineBuilder tec0_line = sc_plane_builder.GetOrCreateLine(0);
  tec0_line.SetName("TEC 0");

  // An HLO module event [0, 100) on a TEC line has no HLO category. Even when
  // it contains a hardware barrier wait [5, 15) before the active fusion
  // [20, 50), only the fusion [20, 50) inside it counts as active, not the
  // gaps around it.
  CreateXEvent(&sc_plane_builder, &tec0_line, "module.1(1)", /*offset_ps=*/0,
               /*duration_ps=*/100);
  CreateXEvent(&sc_plane_builder, &tec0_line, "Barrier Wait:3", /*offset_ps=*/5,
               /*duration_ps=*/10);
  CreateXEventMetadata(&sc_plane_builder, "fusion.1",
                       {{StatType::kHloCategory, "fusion"}});
  CreateXEvent(&sc_plane_builder, &tec0_line, "fusion.1", /*offset_ps=*/20,
               /*duration_ps=*/30);
  // An op without an HLO category and without children still counts.
  CreateXEvent(&sc_plane_builder, &tec0_line, "scs op.7", /*offset_ps=*/60,
               /*duration_ps=*/10);
  // An unsymbolized parent op [100, 200) wrapping only off-duty hardware sync
  // waits ("Sync Wait:5" [120, 140) and "SFence" [140, 150)) keeps its
  // unoccluded intervals [100, 120) and [150, 200) = 70ps active.
  CreateXEvent(&sc_plane_builder, &tec0_line, "scs op.8", /*offset_ps=*/100,
               /*duration_ps=*/100);
  CreateXEvent(&sc_plane_builder, &tec0_line, "Sync Wait:5", /*offset_ps=*/120,
               /*duration_ps=*/20);
  CreateXEvent(&sc_plane_builder, &tec0_line, "SFence", /*offset_ps=*/140,
               /*duration_ps=*/10);

  XPlaneVisitor visitor = tsl::profiler::CreateTfXPlaneVisitor(sc_plane);
  DutyCycleTrackers trackers = ConstructDutyCycleTrackers(visitor);

  EXPECT_EQ(trackers.duty_cycle_tracker.GetActiveTimePs(), 110);
  EXPECT_EQ(trackers.duty_cycle_tracker.GetIdleTimePs(), 90);
  EXPECT_EQ(trackers.duty_cycle_hc_tracker.GetActiveTimePs(), 110);
}

TEST(ConvertXPlaneToOpStats, ConstructDutyCycleTrackerCarvesOutNestedInfeed) {
  XSpace space;
  XPlane* device_plane = GetOrCreateTpuXPlane(
      &space, /*device_ordinal=*/0, /*device_type=*/"TPU v5p",
      /*peak_tera_flops_per_second=*/0,
      /*peak_hbm_bw_gigabytes_per_second=*/0);
  XPlaneBuilder device_plane_builder(device_plane);
  XLineBuilder op_line = device_plane_builder.GetOrCreateLine(0);
  op_line.SetName(kXlaOpLineName);

  // Any off-duty op nested in an active op is carved out of it, not only sync
  // waits: here an infeed [20, 50) inside a while loop [0, 100).
  CreateXEventMetadata(&device_plane_builder, "while.1",
                       {{StatType::kHloCategory, tsl::profiler::kHloWhile}});
  CreateXEvent(&device_plane_builder, &op_line, "while.1", /*offset_ps=*/0,
               /*duration_ps=*/100);
  CreateXEventMetadata(&device_plane_builder, "infeed.1",
                       {{StatType::kHloCategory, tsl::profiler::kHloInfeed}});
  CreateXEvent(&device_plane_builder, &op_line, "infeed.1", /*offset_ps=*/20,
               /*duration_ps=*/30);

  XPlaneVisitor visitor = tsl::profiler::CreateTfXPlaneVisitor(device_plane);
  DutyCycleTracker tracker = ConstructDutyCycleTracker(visitor);

  EXPECT_EQ(tracker.GetActiveTimePs(), 70);
  EXPECT_EQ(tracker.GetIdleTimePs(), 30);
}

TEST(ConvertXPlaneToOpStats,
     SparseCoreStepWithStepLineCountsSyncWaitsAndGapsAsIdle) {
  XSpace space;
  constexpr int64_t kStep1 = 1;
  constexpr int64_t kStep2 = 2;

  // TC Plane 0: steps [0, 1000) and [1000, 2000)
  XPlaneBuilder tc_plane_builder(
      GetOrCreateTpuXPlane(&space, /*device_ordinal=*/0, "TPU v5p", 0, 0));
  tc_plane_builder.SetId(0);
  XLineBuilder tc_step_line = tc_plane_builder.GetOrCreateLine(0);
  tc_step_line.SetName(tsl::profiler::kStepLineName);
  CreateXEvent(&tc_plane_builder, &tc_step_line, "Step 1", /*offset_ps=*/0,
               /*duration_ps=*/1000, {{StatType::kGroupId, kStep1}});
  CreateXEvent(&tc_plane_builder, &tc_step_line, "Step 2", /*offset_ps=*/1000,
               /*duration_ps=*/1000, {{StatType::kGroupId, kStep2}});
  XLineBuilder tc_op_line = tc_plane_builder.GetOrCreateLine(1);
  tc_op_line.SetName(kXlaOpLineName);
  CreateXEventMetadata(&tc_plane_builder, "tc_dot",
                       {{StatType::kHloCategory, "dot"}});
  CreateXEvent(&tc_plane_builder, &tc_op_line, "tc_dot", /*offset_ps=*/0,
               /*duration_ps=*/400, {{StatType::kGroupId, kStep1}});
  CreateXEvent(&tc_plane_builder, &tc_op_line, "tc_dot", /*offset_ps=*/1000,
               /*duration_ps=*/400, {{StatType::kGroupId, kStep2}});

  // SC Plane 1: its own steps, with ops on the Sparse Core Ops line and on
  // two TEC lines.
  XPlaneBuilder sc_plane_builder(
      GetOrCreateTpuXPlane(&space, /*device_ordinal=*/1, "TPU v5p", 0, 0));
  sc_plane_builder.SetId(1);
  sc_plane_builder.SetName("/device:TPU:0 SparseCore 0");
  XLineBuilder sc_step_line = sc_plane_builder.GetOrCreateLine(0);
  sc_step_line.SetName(tsl::profiler::kSparseCoreStepLineName);
  CreateXEvent(&sc_plane_builder, &sc_step_line, "Step 1", /*offset_ps=*/0,
               /*duration_ps=*/1000, {{StatType::kGroupId, kStep1}});
  CreateXEvent(&sc_plane_builder, &sc_step_line, "Step 2", /*offset_ps=*/1000,
               /*duration_ps=*/1000, {{StatType::kGroupId, kStep2}});
  CreateXEventMetadata(&sc_plane_builder, "sc_fusion",
                       {{StatType::kHloCategory, "fusion"},
                        {StatType::kProgramId, 1},
                        {StatType::kSymbolId, 1}});
  CreateXEventMetadata(&sc_plane_builder, "call-done.1",
                       {{StatType::kHloCategory,
                         xla::HloOpcodeString(xla::HloOpcode::kAsyncDone)},
                        {StatType::kProgramId, 1},
                        {StatType::kSymbolId, 2}});
  CreateXEventMetadata(&sc_plane_builder, "sc_infeed",
                       {{StatType::kHloCategory, tsl::profiler::kHloInfeed},
                        {StatType::kProgramId, 1},
                        {StatType::kSymbolId, 3}});
  XLineBuilder sc_op_line = sc_plane_builder.GetOrCreateLine(1);
  sc_op_line.SetName(kSparseCoreOpLineName);
  // Step 1: active op [100, 200), sync wait [200, 500), infeed [500, 600).
  CreateXEvent(&sc_plane_builder, &sc_op_line, "sc_fusion", /*offset_ps=*/100,
               /*duration_ps=*/100, {{StatType::kGroupId, kStep1}});
  CreateXEvent(&sc_plane_builder, &sc_op_line, "call-done.1",
               /*offset_ps=*/200, /*duration_ps=*/300,
               {{StatType::kGroupId, kStep1}});
  CreateXEvent(&sc_plane_builder, &sc_op_line, "sc_infeed", /*offset_ps=*/500,
               /*duration_ps=*/100, {{StatType::kGroupId, kStep1}});
  // Step 2: only a sync wait [1100, 1400) and an infeed [1500, 1600).
  CreateXEvent(&sc_plane_builder, &sc_op_line, "call-done.1",
               /*offset_ps=*/1100, /*duration_ps=*/300,
               {{StatType::kGroupId, kStep2}});
  CreateXEvent(&sc_plane_builder, &sc_op_line, "sc_infeed", /*offset_ps=*/1500,
               /*duration_ps=*/100, {{StatType::kGroupId, kStep2}});
  // TEC 0 [150, 300) and TEC 1 [250, 350) run in parallel and extend the
  // step 1 active union to [100, 350).
  XLineBuilder tec0_line = sc_plane_builder.GetOrCreateLine(2);
  tec0_line.SetName("TEC 0");
  CreateXEvent(&sc_plane_builder, &tec0_line, "tec_task_0", /*offset_ps=*/150,
               /*duration_ps=*/150);
  XLineBuilder tec1_line = sc_plane_builder.GetOrCreateLine(3);
  tec1_line.SetName("TEC 1");
  CreateXEvent(&sc_plane_builder, &tec1_line, "tec_task_1", /*offset_ps=*/250,
               /*duration_ps=*/100);

  ASSERT_OK_AND_ASSIGN(OpStats op_stats,
                       ConvertXSpaceToOpStats(
                           space, OpStatsOptions{.generate_op_metrics_db = true,
                                                 .generate_step_db = true}));
  ASSERT_EQ(op_stats.step_db().step_sequence_size(), 2);
  uint32_t sc_core_id = kSparseCoreIndexStart + 1;
  for (int step = 0; step < 2; ++step) {
    SCOPED_TRACE(absl::StrCat("step ", step + 1));
    const auto& step_info_per_core =
        op_stats.step_db().step_sequence(step).step_info_per_core();
    ASSERT_TRUE(step_info_per_core.contains(sc_core_id));
    GenericStepBreakdown sc_breakdown;
    ASSERT_TRUE(step_info_per_core.at(sc_core_id)
                    .step_breakdown()
                    .UnpackTo(&sc_breakdown));
    const auto& category_ps = sc_breakdown.category_ps();
    // Step 1 is active for the union [100, 350) = 250ps and step 2 is never
    // active. The infeed (100ps) stays, and the rest of each 1000ps step,
    // including the sync wait, is idle.
    EXPECT_EQ(category_ps.at(kIdle), step == 0 ? 650 : 900);
    EXPECT_EQ(category_ps.at(tsl::profiler::kHloInfeed), 100);
    EXPECT_FALSE(
        category_ps.contains(xla::HloOpcodeString(xla::HloOpcode::kAsyncDone)));
  }
}

TEST(ConvertXPlaneToOpStats,
     SparseCoreStepWithStepLineAndTecOnlyOpsPreservesMarkerSpan) {
  XSpace space;
  constexpr int64_t kStep1 = 1;

  // TC Plane 0: Step 1 [0, 1000)
  XPlaneBuilder tc_plane_builder(
      GetOrCreateTpuXPlane(&space, /*device_ordinal=*/0, "TPU v5p", 0, 0));
  tc_plane_builder.SetId(0);
  XLineBuilder tc_step_line = tc_plane_builder.GetOrCreateLine(0);
  tc_step_line.SetName(tsl::profiler::kStepLineName);
  CreateXEvent(&tc_plane_builder, &tc_step_line, "Step 1", /*offset_ps=*/0,
               /*duration_ps=*/1000, {{StatType::kGroupId, kStep1}});
  XLineBuilder tc_op_line = tc_plane_builder.GetOrCreateLine(1);
  tc_op_line.SetName(kXlaOpLineName);
  CreateXEventMetadata(&tc_plane_builder, "tc_dot",
                       {{StatType::kHloCategory, "dot"}});
  CreateXEvent(&tc_plane_builder, &tc_op_line, "tc_dot", /*offset_ps=*/0,
               /*duration_ps=*/400, {{StatType::kGroupId, kStep1}});

  // SC Plane 1: Sparse Core Steps marker [100, 900) (800ps), no Sparse Core Ops
  // events, and TEC 0 active [200, 500) (300ps).
  XPlaneBuilder sc_plane_builder(
      GetOrCreateTpuXPlane(&space, /*device_ordinal=*/1, "TPU v5p", 0, 0));
  sc_plane_builder.SetId(1);
  sc_plane_builder.SetName("/device:TPU:0 SparseCore 0");
  XLineBuilder sc_step_line = sc_plane_builder.GetOrCreateLine(0);
  sc_step_line.SetName(tsl::profiler::kSparseCoreStepLineName);
  CreateXEvent(&sc_plane_builder, &sc_step_line, "Step 1", /*offset_ps=*/100,
               /*duration_ps=*/800, {{StatType::kGroupId, kStep1}});
  XLineBuilder tec0_line = sc_plane_builder.GetOrCreateLine(1);
  tec0_line.SetName("TEC 0");
  CreateXEvent(&sc_plane_builder, &tec0_line, "tec_task_0", /*offset_ps=*/200,
               /*duration_ps=*/300);

  ASSERT_OK_AND_ASSIGN(OpStats op_stats,
                       ConvertXSpaceToOpStats(
                           space, OpStatsOptions{.generate_op_metrics_db = true,
                                                 .generate_step_db = true}));
  ASSERT_EQ(op_stats.step_db().step_sequence_size(), 1);
  uint32_t sc_core_id = kSparseCoreIndexStart + 1;
  const auto& step_info_per_core =
      op_stats.step_db().step_sequence(0).step_info_per_core();
  ASSERT_TRUE(step_info_per_core.contains(sc_core_id));
  const StepInfoResult& sc_step_info = step_info_per_core.at(sc_core_id);
  EXPECT_EQ(sc_step_info.begin_ps(), 100);
  EXPECT_EQ(sc_step_info.duration_ps(), 800);
  GenericStepBreakdown sc_breakdown;
  ASSERT_TRUE(sc_step_info.step_breakdown().UnpackTo(&sc_breakdown));
  EXPECT_EQ(sc_breakdown.category_ps().at(
                std::string(tsl::profiler::kSparseCoreOpLineName)),
            300);
  EXPECT_EQ(sc_breakdown.category_ps().at(kIdle), 500);
}

TEST(ConvertXPlaneToOpStats,
     SparseCoreStepDbUnionsTecLinesAndExcludesSyncWait) {
  XSpace space;
  constexpr int64_t kStep1 = 1;
  constexpr int64_t kStep2 = 2;

  // TC Plane 0: Step 1 [0, 100) and Step 2 [100, 200)
  XPlaneBuilder tc_plane_builder(
      GetOrCreateTpuXPlane(&space, /*device_ordinal=*/0, "TPU v5p", 0, 0));
  tc_plane_builder.SetId(0);
  XLineBuilder tc_step_line = tc_plane_builder.GetOrCreateLine(0);
  tc_step_line.SetName(tsl::profiler::kStepLineName);
  CreateXEvent(&tc_plane_builder, &tc_step_line, "Step 1", /*offset_ps=*/0,
               /*duration_ps=*/100, {{StatType::kGroupId, kStep1}});
  CreateXEvent(&tc_plane_builder, &tc_step_line, "Step 2", /*offset_ps=*/100,
               /*duration_ps=*/100, {{StatType::kGroupId, kStep2}});
  XLineBuilder tc_op_line = tc_plane_builder.GetOrCreateLine(1);
  tc_op_line.SetName(kXlaOpLineName);
  CreateXEventMetadata(&tc_plane_builder, "tc_dot",
                       {{StatType::kHloCategory, "dot"}});
  CreateXEvent(&tc_plane_builder, &tc_op_line, "tc_dot", /*offset_ps=*/0,
               /*duration_ps=*/40, {{StatType::kGroupId, kStep1}});
  CreateXEvent(&tc_plane_builder, &tc_op_line, "tc_dot", /*offset_ps=*/100,
               /*duration_ps=*/80, {{StatType::kGroupId, kStep2}});

  // SC Plane 1: No Sparse Core Steps line; activity in Step 1 across Sparse
  // Core Ops and TEC 0 / TEC 1 with overlapping intervals and an off-duty
  // call-done wait, and idle (only a sync wait) in Step 2.
  XPlaneBuilder sc_plane_builder(
      GetOrCreateTpuXPlane(&space, /*device_ordinal=*/1, "TPU v5p", 0, 0));
  sc_plane_builder.SetId(1);
  sc_plane_builder.SetName("/device:TPU:0 SparseCore 0");
  XLineBuilder sc_op_line = sc_plane_builder.GetOrCreateLine(0);
  sc_op_line.SetName(kSparseCoreOpLineName);
  // Active SC op [10, 40) (30ps) in Step 1
  CreateXEvent(&sc_plane_builder, &sc_op_line, "sc_active_op", /*offset_ps=*/10,
               /*duration_ps=*/30, {{StatType::kGroupId, kStep1}});
  // Off-duty SC sync wait [40, 80) (40ps, excluded) in Step 1
  CreateXEvent(&sc_plane_builder, &sc_op_line, "call-done.sc", /*offset_ps=*/40,
               /*duration_ps=*/40, {{StatType::kGroupId, kStep1}});
  // Off-duty SC sync wait [110, 150) (40ps, excluded) in Step 2
  CreateXEvent(&sc_plane_builder, &sc_op_line, "call-done.sc",
               /*offset_ps=*/110, /*duration_ps=*/40,
               {{StatType::kGroupId, kStep2}});

  // TEC 0 line: active task [20, 60) (overlaps [20, 40) with sc_active_op,
  // extends union to [10, 60) = 50ps)
  XLineBuilder tec0_line = sc_plane_builder.GetOrCreateLine(1);
  tec0_line.SetName("TEC 0");
  CreateXEvent(&sc_plane_builder, &tec0_line, "tec_task_0", /*offset_ps=*/20,
               /*duration_ps=*/40);

  // TEC 1 line: active task [30, 70) (extends union to [10, 70) = 60ps)
  XLineBuilder tec1_line = sc_plane_builder.GetOrCreateLine(2);
  tec1_line.SetName("TEC 1");
  CreateXEvent(&sc_plane_builder, &tec1_line, "tec_task_1", /*offset_ps=*/30,
               /*duration_ps=*/40);

  ASSERT_OK_AND_ASSIGN(OpStats op_stats,
                       ConvertXSpaceToOpStats(
                           space, OpStatsOptions{.generate_op_metrics_db = true,
                                                 .generate_step_db = true}));
  ASSERT_EQ(op_stats.step_db().step_sequence_size(), 2);
  uint32_t sc_core_id = kSparseCoreIndexStart + 1;
  // Step 1: union of [10, 40), [20, 60), [30, 70) = [10, 70) = 60ps active,
  // 40ps idle.
  {
    const auto& step_info_per_core =
        op_stats.step_db().step_sequence(0).step_info_per_core();
    ASSERT_TRUE(step_info_per_core.contains(sc_core_id));
    SparseCoreStepBreakdown sc_breakdown;
    ASSERT_TRUE(step_info_per_core.at(sc_core_id)
                    .step_breakdown()
                    .UnpackTo(&sc_breakdown));
    EXPECT_EQ(sc_breakdown.sc_compute_ps(), 60);
    EXPECT_EQ(sc_breakdown.sc_busy_ps(), 60);
    EXPECT_EQ(sc_breakdown.sc_idle_ps(), 40);
    PerTpuStepDetails step_details = ComputeTpuPerStepDataAcrossCores(
        op_stats.step_db().step_sequence(0), op_stats.core_id_to_details());
    EXPECT_DOUBLE_EQ(step_details.sc_compute_time_ms(),
                     tsl::profiler::PicoToMilli(60));
    EXPECT_DOUBLE_EQ(step_details.sc_active_time_ms(),
                     tsl::profiler::PicoToMilli(60));
  }
  // Step 2: 0ps active, 100ps idle on the active SparseCore.
  {
    const auto& step_info_per_core =
        op_stats.step_db().step_sequence(1).step_info_per_core();
    ASSERT_TRUE(step_info_per_core.contains(sc_core_id));
    SparseCoreStepBreakdown sc_breakdown;
    ASSERT_TRUE(step_info_per_core.at(sc_core_id)
                    .step_breakdown()
                    .UnpackTo(&sc_breakdown));
    EXPECT_EQ(sc_breakdown.sc_compute_ps(), 0);
    EXPECT_EQ(sc_breakdown.sc_busy_ps(), 0);
    EXPECT_EQ(sc_breakdown.sc_idle_ps(), 100);
    PerTpuStepDetails step_details = ComputeTpuPerStepDataAcrossCores(
        op_stats.step_db().step_sequence(1), op_stats.core_id_to_details());
    EXPECT_DOUBLE_EQ(step_details.sc_compute_time_ms(), 0.0);
    EXPECT_DOUBLE_EQ(step_details.sc_active_time_ms(), 0.0);
    EXPECT_DOUBLE_EQ(step_details.sc_idle_time_ms(),
                     tsl::profiler::PicoToMilli(100));
    EXPECT_DOUBLE_EQ(step_details.sc_step_time_ms(),
                     tsl::profiler::PicoToMilli(100));
  }
}
}  // namespace
}  // namespace profiler
}  // namespace tensorflow
