/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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
#include <string>

#include "testing/base/public/gmock.h"
#include "<gtest/gtest.h>"
#include "xla/tsl/profiler/utils/xplane_builder.h"
#include "xla/tsl/profiler/utils/xplane_schema.h"
#include "tsl/profiler/protobuf/xplane.pb.h"
#include "xprof/utils/tpu_counter_ids_v7.h"
#include "xprof/utils/tpu_counter_ids_v7x.h"

namespace xprof {
namespace {

using ::testing::AllOf;
using ::testing::HasSubstr;
using ::tensorflow::profiler::XPlane;
using ::tsl::profiler::XPlaneBuilder;
using ::tsl::profiler::XSpace;

TEST(ConvertXSpaceToUtilizationViewerTest, BasicTpuV7x) {
  XSpace space;
  XPlane* plane = space.add_planes();
  plane->set_name("/device:TPU:0");
  XPlaneBuilder builder(plane);

  using tsl::profiler::GetStatTypeStr;
  using tsl::profiler::StatType;

  builder.AddStatValue(
      *builder.GetOrCreateStatMetadata(GetStatTypeStr(StatType::kDeviceId)), 0);
  builder.AddStatValue(*builder.GetOrCreateStatMetadata(
                           GetStatTypeStr(StatType::kDeviceTypeString)),
                       "TPU v7x");

  auto line_builder = builder.GetOrCreateLine(0);  // Sample 0

  // Add some counters
  // Use V7x Counter IDs
  // Cycle count
  // NOLINTBEGIN
  using Tpu7x = TpuCounterIdsTpu7x;
  // V7x uses PWRMGR cycle counter in ComputeTpuv7GenericTcUnitUtilization
  uint64_t cycles_id = Tpu7x::
      VF_CHIP_DIE0_PWRMGR_PWRMGR_TC_THROTTLE_CORE_DEBUG_STATS_UNPRIVILEGED_CYCLE_COUNT;
  uint64_t scalar_inst_id = Tpu7x::
      VF_CHIP_DIE0_TC_TCS_TC_MISC_TCS_STATS_TCS_STATS_COUNTERS_UNPRIVILEGED_COUNT_SCALAR_ALU_INSTRUCTION_0;
  // NOLINTEND
  // Event for Cycles
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

  // Event for Scalar Inst
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

  ASSERT_OK_AND_ASSIGN(std::string json,
                       ConvertXSpaceToUtilizationViewer(space));
  // Check for "Scalar Unit" metric name which is added by
  // ComputeTpuv7GenericTcUnitUtilization should have 1000 cycles (achieved or
  // peak base)
  EXPECT_THAT(json, AllOf(HasSubstr("Scalar Unit"), HasSubstr("1000"),
                          HasSubstr("500")));
}

TEST(ConvertXSpaceToUtilizationViewerTest, VpuUtilTpuV7x) {
  XSpace space;
  XPlane* plane = space.add_planes();
  plane->set_name("/device:TPU:0");
  XPlaneBuilder builder(plane);

  using tsl::profiler::GetStatTypeStr;
  using tsl::profiler::StatType;

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

  ASSERT_OK_AND_ASSIGN(std::string json,
                       ConvertXSpaceToUtilizationViewer(space));
  EXPECT_THAT(json, AllOf(HasSubstr("VPU Util"), HasSubstr("250"),
                          HasSubstr("4000")));
}

TEST(ConvertXSpaceToUtilizationViewerTest, VpuUtilTpuV7) {
  XSpace space;
  XPlane* plane = space.add_planes();
  plane->set_name("/device:TPU:0");
  XPlaneBuilder builder(plane);

  using tsl::profiler::GetStatTypeStr;
  using tsl::profiler::StatType;

  builder.AddStatValue(
      *builder.GetOrCreateStatMetadata(GetStatTypeStr(StatType::kDeviceId)), 0);
  builder.AddStatValue(*builder.GetOrCreateStatMetadata(
                           GetStatTypeStr(StatType::kDeviceTypeString)),
                       "TPU v7");

  auto line_builder = builder.GetOrCreateLine(0);  // Sample 0

  using Tpu7 = TpuCounterIdsTpu7;
  uint64_t cycles_id = Tpu7::
      VF_CHIP_TC_TCS_TC_MISC_TCS_STATS_TCS_STATS_COUNTERS_UNPRIVILEGED_COUNT_CYCLES;  // NOLINT

  uint64_t vpu_fadd_0_id = Tpu7::
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

  ASSERT_OK_AND_ASSIGN(std::string json,
                       ConvertXSpaceToUtilizationViewer(space));
  EXPECT_THAT(json, AllOf(HasSubstr("VPU Util"), HasSubstr("250"),
                          HasSubstr("4000")));
}

TEST(ConvertXSpaceToUtilizationViewerTest, CounterValueOutOfBoundsOfUint64) {
  XSpace space;
  XPlane* plane = space.add_planes();
  plane->set_name("/device:TPU:0");
  XPlaneBuilder builder(plane);

  using tsl::profiler::GetStatTypeStr;
  using tsl::profiler::StatType;

  builder.AddStatValue(
      *builder.GetOrCreateStatMetadata(GetStatTypeStr(StatType::kDeviceId)), 0);
  builder.AddStatValue(*builder.GetOrCreateStatMetadata(
                           GetStatTypeStr(StatType::kDeviceTypeString)),
                       "TPU v7x");

  auto line_builder = builder.GetOrCreateLine(0);  // Sample 0

  using Tpu7x = TpuCounterIdsTpu7x;
  // V7x uses PWRMGR cycle counter in ComputeTpuv7GenericTcUnitUtilization
  uint64_t cycles_id = Tpu7x::
      VF_CHIP_DIE0_PWRMGR_PWRMGR_TC_THROTTLE_CORE_DEBUG_STATS_UNPRIVILEGED_CYCLE_COUNT;  // NOLINT
  uint64_t scalar_inst_0_id = Tpu7x::
      VF_CHIP_DIE0_TC_TCS_TC_MISC_TCS_STATS_TCS_STATS_COUNTERS_UNPRIVILEGED_COUNT_SCALAR_ALU_INSTRUCTION_0;  // NOLINT
  uint64_t scalar_inst_1_id = Tpu7x::
      VF_CHIP_DIE0_TC_TCS_TC_MISC_TCS_STATS_TCS_STATS_COUNTERS_UNPRIVILEGED_COUNT_SCALAR_ALU_INSTRUCTION_1;  // NOLINT

  // Set standard CYCLES to 1000.
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

  ASSERT_OK_AND_ASSIGN(std::string json,
                       ConvertXSpaceToUtilizationViewer(space));
  // "Scalar Unit" achieved value will be COUNTER(SCALAR_ALU_INSTRUCTION_0) +
  // COUNTER(SCALAR_ALU_INSTRUCTION_1). Clamped to 0 + 18446744073709551615ULL =
  // 18446744073709551615ULL. As a double, this is ~1.84467e+19, so that should
  // be in the JSON.
  EXPECT_THAT(json, AllOf(HasSubstr("Scalar Unit"), HasSubstr("1.84467"),
                          HasSubstr("e+19")));
}

}  // namespace
}  // namespace xprof
