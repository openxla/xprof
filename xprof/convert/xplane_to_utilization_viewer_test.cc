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

#include <string>

#include "<gtest/gtest.h>"
#include "absl/strings/match.h"
#include "xla/tsl/profiler/utils/xplane_builder.h"
#include "xla/tsl/profiler/utils/xplane_schema.h"
#include "tsl/profiler/protobuf/xplane.pb.h"
#include "xprof/utils/tpu_counter_ids_v7x.h"

namespace xprof {
namespace {

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

  auto result = ConvertXSpaceToUtilizationViewer(space);
  ASSERT_TRUE(result.ok());

  // Verify JSON contains expected strings
  std::string json = result.value();
  // Check for "Scalar Unit" metric name which is added by
  // ComputeTpuv7GenericTcUnitUtilization
  EXPECT_TRUE(absl::StrContains(json, "Scalar Unit"));
  EXPECT_TRUE(
      absl::StrContains(json, "1000"));  // Cycles (achieved or peak base)
  EXPECT_TRUE(absl::StrContains(json, "500"));
}

}  // namespace
}  // namespace xprof
