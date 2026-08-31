/* Copyright 2024 The OpenXLA Authors. All Rights Reserved.

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

#include "xprof/convert/xplane_to_utilization_viewer.h"

#include <cstdint>
#include <limits>
#include <string>

#include "testing/base/public/gmock.h"
#include "<gtest/gtest.h>"
#include "nlohmann/json.hpp"
#include "xla/tsl/profiler/utils/xplane_builder.h"
#include "xla/tsl/profiler/utils/xplane_schema.h"
#include "tsl/profiler/protobuf/xplane.pb.h"
#include "xprof/convert/tool_options.h"
#include "xprof/utils/tpu_counter_ids_v6e.h"
#include "xprof/utils/tpu_counter_ids_v7x.h"

namespace xprof {
namespace {

using ::nlohmann::json;
using ::tensorflow::profiler::ToolOptions;
using ::tensorflow::profiler::XPlane;
using ::testing::AllOf;
using ::testing::Eq;
using ::testing::Gt;
using ::testing::HasSubstr;
using ::tsl::profiler::GetStatTypeStr;
using ::tsl::profiler::StatType;
using ::tsl::profiler::XPlaneBuilder;
using ::tsl::profiler::XSpace;

TEST(ConvertXSpaceToUtilizationViewerTest, BasicTpuV7x) {
  XSpace space;
  XPlane* plane = space.add_planes();
  plane->set_name("/device:TPU:0");
  XPlaneBuilder builder(plane);

  builder.AddStatValue(
      *builder.GetOrCreateStatMetadata(GetStatTypeStr(StatType::kDeviceId)), 0);
  builder.AddStatValue(*builder.GetOrCreateStatMetadata(
                           GetStatTypeStr(StatType::kDeviceTypeString)),
                       "TPU v7x");

  auto line_builder = builder.GetOrCreateLine(0);  // Sample 0

  using Tpu7x = TpuCounterIdsTpu7x;
  uint64_t cycles_id = Tpu7x::
      VF_CHIP_DIE0_PWRMGR_PWRMGR_TC_THROTTLE_CORE_DEBUG_STATS_UNPRIVILEGED_CYCLE_COUNT;  // NOLINT
  uint64_t scalar_inst_id = Tpu7x::
      VF_CHIP_DIE0_TC_TCS_TC_MISC_TCS_STATS_TCS_STATS_COUNTERS_UNPRIVILEGED_COUNT_SCALAR_ALU_INSTRUCTION_0;  // NOLINT

  {
    auto event_builder =
        line_builder.AddEvent(*builder.GetOrCreateEventMetadata("CYCLES"));
    event_builder.AddStatValue(*builder.GetOrCreateStatMetadata(GetStatTypeStr(
                                   StatType::kPerformanceCounterId)),
                               cycles_id);
    event_builder.AddStatValue(*builder.GetOrCreateStatMetadata(
                                   GetStatTypeStr(StatType::kCounterValue)),
                               1000.0);
  }

  {
    auto event_builder = line_builder.AddEvent(
        *builder.GetOrCreateEventMetadata("SCALAR_ALU_0"));
    event_builder.AddStatValue(*builder.GetOrCreateStatMetadata(GetStatTypeStr(
                                   StatType::kPerformanceCounterId)),
                               scalar_inst_id);
    event_builder.AddStatValue(*builder.GetOrCreateStatMetadata(
                                   GetStatTypeStr(StatType::kCounterValue)),
                               500.0);
  }

  ASSERT_OK_AND_ASSIGN(std::string json_str,
                       ConvertXSpaceToUtilizationViewer(space));
  EXPECT_THAT(json_str, AllOf(HasSubstr("Scalar Unit"), HasSubstr("1000"),
                              HasSubstr("500")));
}

TEST(ConvertXSpaceToUtilizationViewerTest, VpuUtilTpuV7x) {
  XSpace space;
  XPlane* plane = space.add_planes();
  plane->set_name("/device:TPU:0");
  XPlaneBuilder builder(plane);

  builder.AddStatValue(
      *builder.GetOrCreateStatMetadata(GetStatTypeStr(StatType::kDeviceId)), 0);
  builder.AddStatValue(*builder.GetOrCreateStatMetadata(
                           GetStatTypeStr(StatType::kDeviceTypeString)),
                       "TPU v7x");

  auto line_builder = builder.GetOrCreateLine(0);  // Sample 0

  using Tpu7x = TpuCounterIdsTpu7x;
  uint64_t cycles_id = Tpu7x::
      VF_CHIP_DIE0_PWRMGR_PWRMGR_TC_THROTTLE_CORE_DEBUG_STATS_UNPRIVILEGED_CYCLE_COUNT;  // NOLINT

  uint64_t vpu_fadd_0_id = Tpu7x::
      VF_CHIP_DIE0_TC_TCS_TC_MISC_TCS_STATS_TCS_STATS_COUNTERS_UNPRIVILEGED_COUNT_VPU_VALU_FADD_OPS_0;  // NOLINT

  {
    auto event_builder =
        line_builder.AddEvent(*builder.GetOrCreateEventMetadata("CYCLES"));
    event_builder.AddStatValue(*builder.GetOrCreateStatMetadata(GetStatTypeStr(
                                   StatType::kPerformanceCounterId)),
                               cycles_id);
    event_builder.AddStatValue(*builder.GetOrCreateStatMetadata(
                                   GetStatTypeStr(StatType::kCounterValue)),
                               1000.0);
  }

  {
    auto event_builder =
        line_builder.AddEvent(*builder.GetOrCreateEventMetadata("VPU_FADD_0"));
    event_builder.AddStatValue(*builder.GetOrCreateStatMetadata(GetStatTypeStr(
                                   StatType::kPerformanceCounterId)),
                               vpu_fadd_0_id);
    event_builder.AddStatValue(*builder.GetOrCreateStatMetadata(
                                   GetStatTypeStr(StatType::kCounterValue)),
                               250.0);
  }

  ASSERT_OK_AND_ASSIGN(std::string json_str,
                       ConvertXSpaceToUtilizationViewer(space));
  EXPECT_THAT(json_str, AllOf(HasSubstr("VPU Util"), HasSubstr("250"),
                              HasSubstr("4000")));
}

TEST(ConvertXSpaceToUtilizationViewerTest, VpuUtilTpuV6E) {
  XSpace space;
  XPlane* plane = space.add_planes();
  plane->set_name("/device:TPU:0");
  XPlaneBuilder builder(plane);

  builder.AddStatValue(
      *builder.GetOrCreateStatMetadata(GetStatTypeStr(StatType::kDeviceId)), 0);
  builder.AddStatValue(*builder.GetOrCreateStatMetadata(
                           GetStatTypeStr(StatType::kDeviceTypeString)),
                       "TPU v6 Lite");

  auto line_builder = builder.GetOrCreateLine(0);  // Sample 0

  using Tpu6e = TpuCounterIdsTpu6e;
  uint64_t cycles_id = Tpu6e::
      VF_CHIP_TC_TCS_TC_MISC_TCS_STATS_TCS_STATS_COUNTERS_UNPRIVILEGED_COUNT_CYCLES;  // NOLINT

  uint64_t vpu_fadd_0_id = Tpu6e::
      VF_CHIP_TC_TCS_TC_MISC_TCS_STATS_TCS_STATS_COUNTERS_UNPRIVILEGED_COUNT_VPU_VALU_FADD_OPS_0;  // NOLINT

  {
    auto event_builder =
        line_builder.AddEvent(*builder.GetOrCreateEventMetadata("CYCLES"));
    event_builder.AddStatValue(*builder.GetOrCreateStatMetadata(GetStatTypeStr(
                                   StatType::kPerformanceCounterId)),
                               cycles_id);
    event_builder.AddStatValue(*builder.GetOrCreateStatMetadata(
                                   GetStatTypeStr(StatType::kCounterValue)),
                               1000.0);
  }

  {
    auto event_builder =
        line_builder.AddEvent(*builder.GetOrCreateEventMetadata("VPU_FADD_0"));
    event_builder.AddStatValue(*builder.GetOrCreateStatMetadata(GetStatTypeStr(
                                   StatType::kPerformanceCounterId)),
                               vpu_fadd_0_id);
    event_builder.AddStatValue(*builder.GetOrCreateStatMetadata(
                                   GetStatTypeStr(StatType::kCounterValue)),
                               250.0);
  }

  ASSERT_OK_AND_ASSIGN(std::string json_str,
                       ConvertXSpaceToUtilizationViewer(space));
  EXPECT_THAT(json_str, AllOf(HasSubstr("VPU Util"), HasSubstr("250"),
                              HasSubstr("4000")));
}

TEST(ConvertXSpaceToUtilizationViewerTest, CounterValueOutOfBoundsOfUint64) {
  XSpace space;
  XPlane* plane = space.add_planes();
  plane->set_name("/device:TPU:0");
  XPlaneBuilder builder(plane);

  builder.AddStatValue(
      *builder.GetOrCreateStatMetadata(GetStatTypeStr(StatType::kDeviceId)), 0);
  builder.AddStatValue(*builder.GetOrCreateStatMetadata(
                           GetStatTypeStr(StatType::kDeviceTypeString)),
                       "TPU v7x");

  auto line_builder = builder.GetOrCreateLine(0);  // Sample 0

  using Tpu7x = TpuCounterIdsTpu7x;
  uint64_t cycles_id = Tpu7x::
      VF_CHIP_DIE0_PWRMGR_PWRMGR_TC_THROTTLE_CORE_DEBUG_STATS_UNPRIVILEGED_CYCLE_COUNT;  // NOLINT
  uint64_t scalar_inst_0_id = Tpu7x::
      VF_CHIP_DIE0_TC_TCS_TC_MISC_TCS_STATS_TCS_STATS_COUNTERS_UNPRIVILEGED_COUNT_SCALAR_ALU_INSTRUCTION_0;  // NOLINT
  uint64_t scalar_inst_1_id = Tpu7x::
      VF_CHIP_DIE0_TC_TCS_TC_MISC_TCS_STATS_TCS_STATS_COUNTERS_UNPRIVILEGED_COUNT_SCALAR_ALU_INSTRUCTION_1;  // NOLINT

  auto event_builder =
      line_builder.AddEvent(*builder.GetOrCreateEventMetadata("CYCLES"));
  event_builder.AddStatValue(*builder.GetOrCreateStatMetadata(GetStatTypeStr(
                                 StatType::kPerformanceCounterId)),
                             cycles_id);
  event_builder.AddStatValue(
      *builder.GetOrCreateStatMetadata(GetStatTypeStr(StatType::kCounterValue)),
      1000.0);

  event_builder =
      line_builder.AddEvent(*builder.GetOrCreateEventMetadata("SCALAR_ALU_0"));
  event_builder.AddStatValue(*builder.GetOrCreateStatMetadata(GetStatTypeStr(
                                 StatType::kPerformanceCounterId)),
                             scalar_inst_0_id);
  event_builder.AddStatValue(
      *builder.GetOrCreateStatMetadata(GetStatTypeStr(StatType::kCounterValue)),
      -500.0);

  event_builder =
      line_builder.AddEvent(*builder.GetOrCreateEventMetadata("SCALAR_ALU_1"));
  event_builder.AddStatValue(*builder.GetOrCreateStatMetadata(GetStatTypeStr(
                                 StatType::kPerformanceCounterId)),
                             scalar_inst_1_id);
  event_builder.AddStatValue(
      *builder.GetOrCreateStatMetadata(GetStatTypeStr(StatType::kCounterValue)),
      2.0e19);

  ASSERT_OK_AND_ASSIGN(std::string json_str,
                       ConvertXSpaceToUtilizationViewer(space));
  EXPECT_THAT(json_str, AllOf(HasSubstr("Scalar Unit"), HasSubstr("1.84467"),
                              HasSubstr("e+19")));
}

TEST(ConvertXSpaceToUtilizationViewerTest, UnsupportedDeviceIgnored) {
  XSpace space;
  XPlane* plane = space.add_planes();
  plane->set_name("/device:TPU:0");
  XPlaneBuilder builder(plane);

  builder.AddStatValue(
      *builder.GetOrCreateStatMetadata(GetStatTypeStr(StatType::kDeviceId)), 0);
  builder.AddStatValue(*builder.GetOrCreateStatMetadata(
                           GetStatTypeStr(StatType::kDeviceTypeString)),
                       "TPU v5e");

  ASSERT_OK_AND_ASSIGN(std::string json_str,
                       ConvertXSpaceToUtilizationViewer(space));
  EXPECT_THAT(json_str, HasSubstr("\"rows\":[]"));
}

TEST(ConvertXSpaceToUtilizationViewerTest, NaNCounterValueHandledSafely) {
  XSpace space;
  XPlane* plane = space.add_planes();
  plane->set_name("/device:TPU:0");
  XPlaneBuilder builder(plane);

  builder.AddStatValue(
      *builder.GetOrCreateStatMetadata(GetStatTypeStr(StatType::kDeviceId)), 0);
  builder.AddStatValue(*builder.GetOrCreateStatMetadata(
                           GetStatTypeStr(StatType::kDeviceTypeString)),
                       "TPU v7x");

  auto line_builder = builder.GetOrCreateLine(0);

  using Tpu7x = TpuCounterIdsTpu7x;
  uint64_t cycles_id = Tpu7x::
      VF_CHIP_DIE0_PWRMGR_PWRMGR_TC_THROTTLE_CORE_DEBUG_STATS_UNPRIVILEGED_CYCLE_COUNT;  // NOLINT
  uint64_t scalar_inst_0_id = Tpu7x::
      VF_CHIP_DIE0_TC_TCS_TC_MISC_TCS_STATS_TCS_STATS_COUNTERS_UNPRIVILEGED_COUNT_SCALAR_ALU_INSTRUCTION_0;  // NOLINT

  auto event_builder =
      line_builder.AddEvent(*builder.GetOrCreateEventMetadata("CYCLES"));
  event_builder.AddStatValue(*builder.GetOrCreateStatMetadata(GetStatTypeStr(
                                 StatType::kPerformanceCounterId)),
                             cycles_id);
  event_builder.AddStatValue(
      *builder.GetOrCreateStatMetadata(GetStatTypeStr(StatType::kCounterValue)),
      1000.0);

  event_builder =
      line_builder.AddEvent(*builder.GetOrCreateEventMetadata("SCALAR_ALU_0"));
  event_builder.AddStatValue(*builder.GetOrCreateStatMetadata(GetStatTypeStr(
                                 StatType::kPerformanceCounterId)),
                             scalar_inst_0_id);
  event_builder.AddStatValue(
      *builder.GetOrCreateStatMetadata(GetStatTypeStr(StatType::kCounterValue)),
      std::numeric_limits<double>::quiet_NaN());

  ASSERT_OK_AND_ASSIGN(std::string json_str,
                       ConvertXSpaceToUtilizationViewer(space));
  EXPECT_THAT(json_str, AllOf(HasSubstr("Scalar Unit"), HasSubstr("0")));
}

// =============================================================================
// Kernel Utilization Tests
// =============================================================================

TEST(ConvertXSpaceToKernelUtilizationTest, BasicTpuV7xWithKernelAttribution) {
  XSpace space;
  XPlane* plane = space.add_planes();
  plane->set_name("/device:TPU:0");
  XPlaneBuilder builder(plane);

  builder.AddStatValue(
      *builder.GetOrCreateStatMetadata(GetStatTypeStr(StatType::kDeviceId)), 0);
  builder.AddStatValue(*builder.GetOrCreateStatMetadata(
                           GetStatTypeStr(StatType::kDeviceTypeString)),
                       "TPU v7x");

  // Timeline line with PALLAS kernel name
  auto timeline_line = builder.GetOrCreateLine(100);
  timeline_line.SetName("PALLAS");
  auto timeline_event = timeline_line.AddEvent(
      *builder.GetOrCreateEventMetadata("pallas_matmul_fwd"));
  timeline_event.SetDurationNs(10000);

  // Counter line with counters_0
  auto line_builder = builder.GetOrCreateLine(0);
  line_builder.SetName("counters_0");

  using Tpu7x = TpuCounterIdsTpu7x;
  uint64_t cycles_id = Tpu7x::
      VF_CHIP_DIE0_PWRMGR_PWRMGR_TC_THROTTLE_CORE_DEBUG_STATS_UNPRIVILEGED_CYCLE_COUNT;  // NOLINT
  uint64_t mxu_busy_id = Tpu7x::
      VF_CHIP_DIE0_TC_TCS_TC_MISC_TCS_STATS_TCS_STATS_COUNTERS_UNPRIVILEGED_COUNT_MXU_BUSY_1;  // NOLINT
  uint64_t mxu_bf16_id = Tpu7x::
      VF_CHIP_DIE0_TC_TCS_TC_MISC_TCS_STATS_TCS_STATS_COUNTERS_UNPRIVILEGED_COUNT_MATMUL_VREG_BF16_MXU_0;  // NOLINT

  {
    auto event_builder =
        line_builder.AddEvent(*builder.GetOrCreateEventMetadata("CYCLES"));
    event_builder.AddStatValue(*builder.GetOrCreateStatMetadata(GetStatTypeStr(
                                   StatType::kPerformanceCounterId)),
                               cycles_id);
    event_builder.AddStatValue(*builder.GetOrCreateStatMetadata(
                                   GetStatTypeStr(StatType::kCounterValue)),
                               10000.0);
    event_builder.SetDurationNs(10000);
  }

  {
    auto event_builder =
        line_builder.AddEvent(*builder.GetOrCreateEventMetadata("MXU_BUSY"));
    event_builder.AddStatValue(*builder.GetOrCreateStatMetadata(GetStatTypeStr(
                                   StatType::kPerformanceCounterId)),
                               mxu_busy_id);
    event_builder.AddStatValue(*builder.GetOrCreateStatMetadata(
                                   GetStatTypeStr(StatType::kCounterValue)),
                               16000.0);
  }

  {
    auto event_builder =
        line_builder.AddEvent(*builder.GetOrCreateEventMetadata("MXU_BF16"));
    event_builder.AddStatValue(*builder.GetOrCreateStatMetadata(GetStatTypeStr(
                                   StatType::kPerformanceCounterId)),
                               mxu_bf16_id);
    event_builder.AddStatValue(*builder.GetOrCreateStatMetadata(
                                   GetStatTypeStr(StatType::kCounterValue)),
                               8000.0);
  }

  ASSERT_OK_AND_ASSIGN(std::string json_str,
                       ConvertXSpaceToKernelUtilization(space));
  json result = json::parse(json_str);

  EXPECT_THAT(result["status"], Eq("SUCCESS"));
  ASSERT_THAT(result["devices"].size(), Eq(1));
  EXPECT_THAT(result["devices"][0]["device_id"], Eq(0));
  ASSERT_THAT(result["devices"][0]["kernels"].size(), Eq(1));

  const auto& kernel = result["devices"][0]["kernels"][0];
  EXPECT_THAT(kernel["kernel_name"], Eq("pallas_matmul_fwd"));
  EXPECT_THAT(kernel["mxu_utilization"].get<double>(), Gt(0.0));
  EXPECT_THAT(kernel["mxu_cycles_breakdown"]["BF16"].get<double>(), Eq(100.0));
  EXPECT_THAT(kernel["mxu_is_anomaly"], Eq(false));
}

TEST(ConvertXSpaceToKernelUtilizationTest, TpuV6eWithDurationOverride) {
  XSpace space;
  XPlane* plane = space.add_planes();
  plane->set_name("/device:TPU:0");
  XPlaneBuilder builder(plane);

  builder.AddStatValue(
      *builder.GetOrCreateStatMetadata(GetStatTypeStr(StatType::kDeviceId)), 0);
  builder.AddStatValue(*builder.GetOrCreateStatMetadata(
                           GetStatTypeStr(StatType::kDeviceTypeString)),
                       "TPU v6 Lite");

  auto line_builder = builder.GetOrCreateLine(0);
  line_builder.SetName("counters_custom_flash_attention");

  using Tpu6e = TpuCounterIdsTpu6e;
  uint64_t vpu_fadd_0_id = Tpu6e::
      VF_CHIP_TC_TCS_TC_MISC_TCS_STATS_TCS_STATS_COUNTERS_UNPRIVILEGED_COUNT_VPU_VALU_FADD_OPS_0;  // NOLINT

  {
    auto event_builder =
        line_builder.AddEvent(*builder.GetOrCreateEventMetadata("VPU_VALU"));
    event_builder.AddStatValue(*builder.GetOrCreateStatMetadata(GetStatTypeStr(
                                   StatType::kPerformanceCounterId)),
                               vpu_fadd_0_id);
    event_builder.AddStatValue(*builder.GetOrCreateStatMetadata(
                                   GetStatTypeStr(StatType::kCounterValue)),
                               500.0);
    event_builder.SetDurationNs(5000);
  }

  ToolOptions options;
  options["duration_us"] = std::string("10.0");
  options["kernel"] = std::string("custom_flash_attention");

  ASSERT_OK_AND_ASSIGN(std::string json_str,
                       ConvertXSpaceToKernelUtilization(space, options));
  json result = json::parse(json_str);

  EXPECT_THAT(result["status"], Eq("SUCCESS"));
  ASSERT_THAT(result["devices"][0]["kernels"].size(), Eq(1));
  EXPECT_THAT(result["devices"][0]["kernels"][0]["kernel_name"],
              Eq("custom_flash_attention"));
  EXPECT_THAT(result["devices"][0]["kernels"][0]["duration_us"], Eq(10.0));
}

TEST(ConvertXSpaceToKernelUtilizationTest, UnsupportedDeviceIgnored) {
  XSpace space;
  XPlane* plane = space.add_planes();
  plane->set_name("/device:TPU:0");
  XPlaneBuilder builder(plane);

  builder.AddStatValue(
      *builder.GetOrCreateStatMetadata(GetStatTypeStr(StatType::kDeviceId)), 0);
  builder.AddStatValue(*builder.GetOrCreateStatMetadata(
                           GetStatTypeStr(StatType::kDeviceTypeString)),
                       "TPU v5e");

  ASSERT_OK_AND_ASSIGN(std::string json_str,
                       ConvertXSpaceToKernelUtilization(space));
  json result = json::parse(json_str);
  EXPECT_THAT(result["status"], Eq("SUCCESS"));
  EXPECT_THAT(result["devices"].size(), Eq(0));
}

TEST(ConvertXSpaceToKernelUtilizationTest, DeviceIdAndKernelFiltering) {
  XSpace space;
  XPlane* plane0 = space.add_planes();
  plane0->set_name("/device:TPU:0");
  XPlaneBuilder b0(plane0);
  b0.AddStatValue(
      *b0.GetOrCreateStatMetadata(GetStatTypeStr(StatType::kDeviceId)), 0);
  b0.AddStatValue(
      *b0.GetOrCreateStatMetadata(GetStatTypeStr(StatType::kDeviceTypeString)),
      "TPU v7x");

  auto line0 = b0.GetOrCreateLine(0);
  line0.SetName("counters_matmul");
  auto ev0 = line0.AddEvent(*b0.GetOrCreateEventMetadata("CYCLES"));
  auto* id_meta = b0.GetOrCreateStatMetadata(
      GetStatTypeStr(StatType::kPerformanceCounterId));
  auto* val_meta =
      b0.GetOrCreateStatMetadata(GetStatTypeStr(StatType::kCounterValue));
  ev0.AddStatValue(
      *id_meta,
      static_cast<uint64_t>(
          TpuCounterIdsTpu7x::
              VF_CHIP_DIE0_PWRMGR_PWRMGR_TC_THROTTLE_CORE_DEBUG_STATS_UNPRIVILEGED_CYCLE_COUNT));  // NOLINT
  ev0.AddStatValue(*val_meta, 1000.0);

  ToolOptions options;
  // Filter for non-existent device ID 1
  options["device_id"] = 1;
  {
    ASSERT_OK_AND_ASSIGN(std::string json_str,
                         ConvertXSpaceToKernelUtilization(space, options));
    json result = json::parse(json_str);
    EXPECT_THAT(result["devices"].size(), Eq(0));
  }

  // Filter for matching device ID 0 but non-matching kernel name
  options["device_id"] = 0;
  options["kernel"] = std::string("convolution");
  {
    ASSERT_OK_AND_ASSIGN(std::string json_str,
                         ConvertXSpaceToKernelUtilization(space, options));
    json result = json::parse(json_str);
    ASSERT_THAT(result["devices"].size(), Eq(1));
    EXPECT_THAT(result["devices"][0]["kernels"].size(), Eq(0));
  }
}

TEST(ConvertXSpaceToKernelUtilizationTest, AnomalyAndNanSafety) {
  XSpace space;
  XPlane* plane = space.add_planes();
  plane->set_name("/device:TPU:0");
  XPlaneBuilder builder(plane);

  builder.AddStatValue(
      *builder.GetOrCreateStatMetadata(GetStatTypeStr(StatType::kDeviceId)), 0);
  builder.AddStatValue(*builder.GetOrCreateStatMetadata(
                           GetStatTypeStr(StatType::kDeviceTypeString)),
                       "TPU v7x");

  auto line_builder = builder.GetOrCreateLine(0);
  line_builder.SetName("counters_anomalous_kernel");

  using Tpu7x = TpuCounterIdsTpu7x;
  uint64_t cycles_id = Tpu7x::
      VF_CHIP_DIE0_PWRMGR_PWRMGR_TC_THROTTLE_CORE_DEBUG_STATS_UNPRIVILEGED_CYCLE_COUNT;  // NOLINT
  uint64_t mxu_busy_2_id = Tpu7x::
      VF_CHIP_DIE0_TC_TCS_TC_MISC_TCS_STATS_TCS_STATS_COUNTERS_UNPRIVILEGED_COUNT_MXU_BUSY_2;  // NOLINT
  uint64_t vpu_id = Tpu7x::
      VF_CHIP_DIE0_TC_TCS_TC_MISC_TCS_STATS_TCS_STATS_COUNTERS_UNPRIVILEGED_COUNT_VPU_VALU_FADD_OPS_0;  // NOLINT

  auto* perf_id_meta = builder.GetOrCreateStatMetadata(
      GetStatTypeStr(StatType::kPerformanceCounterId));
  auto* cnt_val_meta =
      builder.GetOrCreateStatMetadata(GetStatTypeStr(StatType::kCounterValue));

  // Cycles = 1000
  {
    auto ev =
        line_builder.AddEvent(*builder.GetOrCreateEventMetadata("CYCLES"));
    ev.AddStatValue(*perf_id_meta, cycles_id);
    ev.AddStatValue(*cnt_val_meta, 1000.0);
  }

  // MXU Busy = 5000 (exceeds 1000 cycles -> anomaly)
  {
    auto ev =
        line_builder.AddEvent(*builder.GetOrCreateEventMetadata("MXU_BUSY"));
    ev.AddStatValue(*perf_id_meta, mxu_busy_2_id);
    ev.AddStatValue(*cnt_val_meta, 5000.0);
  }

  // VPU value with NaN (should be handled safely)
  {
    auto ev =
        line_builder.AddEvent(*builder.GetOrCreateEventMetadata("VPU_NAN"));
    ev.AddStatValue(*perf_id_meta, vpu_id);
    ev.AddStatValue(*cnt_val_meta, std::numeric_limits<double>::quiet_NaN());
  }

  ASSERT_OK_AND_ASSIGN(std::string json_str,
                       ConvertXSpaceToKernelUtilization(space));
  json result = json::parse(json_str);

  EXPECT_THAT(result["status"], Eq("SUCCESS"));
  ASSERT_THAT(result["devices"].size(), Eq(1));
  ASSERT_THAT(result["devices"][0]["kernels"].size(), Eq(1));

  const auto& kernel = result["devices"][0]["kernels"][0];
  EXPECT_THAT(kernel["kernel_name"], Eq("anomalous_kernel"));
  EXPECT_THAT(kernel["mxu_utilization"].get<double>(), Gt(100.0));
  EXPECT_THAT(kernel["mxu_is_anomaly"], Eq(true));
}

TEST(ConvertXSpaceToKernelUtilizationTest,
     MultiCoreOtherMetricsAggregationTpuV7x) {
  XSpace space;
  XPlane* plane = space.add_planes();
  plane->set_name("/device:TPU:0");
  XPlaneBuilder builder(plane);

  builder.AddStatValue(
      *builder.GetOrCreateStatMetadata(GetStatTypeStr(StatType::kDeviceId)), 0);
  builder.AddStatValue(*builder.GetOrCreateStatMetadata(
                           GetStatTypeStr(StatType::kDeviceTypeString)),
                       "TPU v7x");

  auto line_builder = builder.GetOrCreateLine(0);
  line_builder.SetName("counters_multicore_kernel");

  using Tpu7x = TpuCounterIdsTpu7x;
  uint64_t die0_cycles_id = Tpu7x::
      VF_CHIP_DIE0_PWRMGR_PWRMGR_TC_THROTTLE_CORE_DEBUG_STATS_UNPRIVILEGED_CYCLE_COUNT;  // NOLINT
  uint64_t die1_cycles_id = Tpu7x::
      VF_CHIP_DIE1_PWRMGR_PWRMGR_TC_THROTTLE_CORE_DEBUG_STATS_UNPRIVILEGED_CYCLE_COUNT;  // NOLINT
  uint64_t vpu_fadd_0_id = Tpu7x::
      VF_CHIP_DIE0_TC_TCS_TC_MISC_TCS_STATS_TCS_STATS_COUNTERS_UNPRIVILEGED_COUNT_VPU_VALU_FADD_OPS_0;  // NOLINT

  auto* perf_id_meta = builder.GetOrCreateStatMetadata(
      GetStatTypeStr(StatType::kPerformanceCounterId));
  auto* cnt_val_meta =
      builder.GetOrCreateStatMetadata(GetStatTypeStr(StatType::kCounterValue));

  // Die 0 cycles = 1000
  {
    auto ev =
        line_builder.AddEvent(*builder.GetOrCreateEventMetadata("DIE0_CYCLES"));
    ev.AddStatValue(*perf_id_meta, die0_cycles_id);
    ev.AddStatValue(*cnt_val_meta, 1000.0);
  }

  // Die 1 cycles = 1000
  {
    auto ev =
        line_builder.AddEvent(*builder.GetOrCreateEventMetadata("DIE1_CYCLES"));
    ev.AddStatValue(*perf_id_meta, die1_cycles_id);
    ev.AddStatValue(*cnt_val_meta, 1000.0);
  }

  // Die 0 VPU = 250 (core 0 peak 4000, core 1 peak 4000 -> 250/8000 = 3.13%)
  {
    auto ev =
        line_builder.AddEvent(*builder.GetOrCreateEventMetadata("VPU_FADD_0"));
    ev.AddStatValue(*perf_id_meta, vpu_fadd_0_id);
    ev.AddStatValue(*cnt_val_meta, 250.0);
  }

  ASSERT_OK_AND_ASSIGN(std::string json_str,
                       ConvertXSpaceToKernelUtilization(space));
  json result = json::parse(json_str);

  EXPECT_THAT(result["status"], Eq("SUCCESS"));
  ASSERT_THAT(result["devices"].size(), Eq(1));
  ASSERT_THAT(result["devices"][0]["kernels"].size(), Eq(1));

  const auto& kernel = result["devices"][0]["kernels"][0];
  EXPECT_THAT(kernel["kernel_name"], Eq("multicore_kernel"));
  EXPECT_THAT(kernel["other_metrics"]["VPU Utilization"].get<double>(),
              Eq(3.13));
}

TEST(ConvertXSpaceToKernelUtilizationTest, DeviceIdFallbackFromPlaneName) {
  XSpace space;
  XPlane* plane = space.add_planes();
  plane->set_name("/device:TPU:3");
  XPlaneBuilder builder(plane);

  // Intentionally omit StatType::kDeviceId to verify plane name
  // parsing fallback.
  builder.AddStatValue(*builder.GetOrCreateStatMetadata(
                           GetStatTypeStr(StatType::kDeviceTypeString)),
                       "TPU v7x");

  auto line_builder = builder.GetOrCreateLine(0);
  line_builder.SetName("counters_test_kernel");

  using Tpu7x = TpuCounterIdsTpu7x;
  uint64_t cycles_id = Tpu7x::
      VF_CHIP_DIE0_PWRMGR_PWRMGR_TC_THROTTLE_CORE_DEBUG_STATS_UNPRIVILEGED_CYCLE_COUNT;  // NOLINT

  auto* perf_id_meta = builder.GetOrCreateStatMetadata(
      GetStatTypeStr(StatType::kPerformanceCounterId));
  auto* cnt_val_meta =
      builder.GetOrCreateStatMetadata(GetStatTypeStr(StatType::kCounterValue));

  auto ev = line_builder.AddEvent(*builder.GetOrCreateEventMetadata("CYCLES"));
  ev.AddStatValue(*perf_id_meta, cycles_id);
  ev.AddStatValue(*cnt_val_meta, 1000.0);

  ASSERT_OK_AND_ASSIGN(std::string json_str,
                       ConvertXSpaceToKernelUtilization(space));
  json result = json::parse(json_str);

  EXPECT_THAT(result["status"], Eq("SUCCESS"));
  ASSERT_THAT(result["devices"].size(), Eq(1));
  EXPECT_THAT(result["devices"][0]["device_id"], Eq(3));
  ASSERT_THAT(result["devices"][0]["kernels"].size(), Eq(1));
  EXPECT_THAT(result["devices"][0]["kernels"][0]["kernel_name"],
              Eq("test_kernel"));
}

TEST(ConvertXSpaceToKernelUtilizationTest,
     MultiCoreHbmBandwidthAggregationTpuV7x) {
  XSpace space;
  XPlane* plane = space.add_planes();
  plane->set_name("/device:TPU:0");
  XPlaneBuilder builder(plane);

  builder.AddStatValue(
      *builder.GetOrCreateStatMetadata(GetStatTypeStr(StatType::kDeviceId)), 0);
  builder.AddStatValue(*builder.GetOrCreateStatMetadata(
                           GetStatTypeStr(StatType::kDeviceTypeString)),
                       "TPU v7x");

  auto line_builder = builder.GetOrCreateLine(0);
  line_builder.SetName("counters_hbm_kernel");

  using Tpu7x = TpuCounterIdsTpu7x;
  uint64_t die0_cycles_id = Tpu7x::
      VF_CHIP_DIE0_PWRMGR_PWRMGR_TC_THROTTLE_CORE_DEBUG_STATS_UNPRIVILEGED_CYCLE_COUNT;  // NOLINT
  uint64_t die0_misc_cycles_id = Tpu7x::
      VF_CHIP_DIE0_TC_TCS_TC_MISC_TCS_STATS_TCS_STATS_COUNTERS_UNPRIVILEGED_COUNT_CYCLES;  // NOLINT
  uint64_t die1_cycles_id = Tpu7x::
      VF_CHIP_DIE1_PWRMGR_PWRMGR_TC_THROTTLE_CORE_DEBUG_STATS_UNPRIVILEGED_CYCLE_COUNT;  // NOLINT
  uint64_t die1_misc_cycles_id = Tpu7x::
      VF_CHIP_DIE1_TC_TCS_TC_MISC_TCS_STATS_TCS_STATS_COUNTERS_UNPRIVILEGED_COUNT_CYCLES;  // NOLINT
  uint64_t die0_wr_req_id = Tpu7x::
      VF_CHIP_DIE0_HBM_0_SS_HBMC_0_CMN_HI_FREQ_STATS_COUNTERS_UNPRIVILEGED_WR_REQ_PS0;  // NOLINT

  auto* perf_id_meta = builder.GetOrCreateStatMetadata(
      GetStatTypeStr(StatType::kPerformanceCounterId));
  auto* cnt_val_meta =
      builder.GetOrCreateStatMetadata(GetStatTypeStr(StatType::kCounterValue));

  {
    auto ev = line_builder.AddEvent(
        *builder.GetOrCreateEventMetadata("DIE0_CYCLES"));
    ev.AddStatValue(*perf_id_meta, die0_cycles_id);
    ev.AddStatValue(*cnt_val_meta, 10000.0);
  }
  {
    auto ev = line_builder.AddEvent(
        *builder.GetOrCreateEventMetadata("DIE0_MISC_CYCLES"));
    ev.AddStatValue(*perf_id_meta, die0_misc_cycles_id);
    ev.AddStatValue(*cnt_val_meta, 10000.0);
  }
  {
    auto ev = line_builder.AddEvent(
        *builder.GetOrCreateEventMetadata("DIE1_CYCLES"));
    ev.AddStatValue(*perf_id_meta, die1_cycles_id);
    ev.AddStatValue(*cnt_val_meta, 10000.0);
  }
  {
    auto ev = line_builder.AddEvent(
        *builder.GetOrCreateEventMetadata("DIE1_MISC_CYCLES"));
    ev.AddStatValue(*perf_id_meta, die1_misc_cycles_id);
    ev.AddStatValue(*cnt_val_meta, 10000.0);
  }
  {
    auto ev = line_builder.AddEvent(
        *builder.GetOrCreateEventMetadata("HBM_WR"));
    ev.AddStatValue(*perf_id_meta, die0_wr_req_id);
    ev.AddStatValue(*cnt_val_meta, 50000.0);
  }

  ASSERT_OK_AND_ASSIGN(std::string json_str,
                       ConvertXSpaceToKernelUtilization(space));
  json result = json::parse(json_str);

  EXPECT_THAT(result["status"], Eq("SUCCESS"));
  ASSERT_THAT(result["devices"].size(), Eq(1));
  const auto& kernel = result["devices"][0]["kernels"][0];
  EXPECT_TRUE(kernel["other_metrics"].contains("HBM Bandwidth Utilization"));
  EXPECT_FALSE(kernel["other_metrics"].contains("HBM Rd+Wr - core 0"));
  EXPECT_FALSE(kernel["other_metrics"].contains("HBM Rd+Wr - core 1"));
}

}  // namespace
}  // namespace xprof
