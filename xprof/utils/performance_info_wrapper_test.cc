/* Copyright 2025 The TensorFlow Authors. All Rights Reserved.

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

#include "xprof/utils/performance_info_wrapper.h"

#include <cstdint>
#include <memory>
#include <utility>

#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/tsl/platform/test.h"
#include "tsl/platform/protobuf.h"
#include "xprof/utils/cost_utils.h"
#include "xprof/utils/hlo_cost_analysis_wrapper.h"
#include "xprof/utils/hlo_module_map.h"
#include "xprof/utils/xprof_gpu_cost_analysis.h"

namespace tensorflow {
namespace profiler {
namespace {

using ::testing::UnorderedElementsAre;
#if defined(PLATFORM_GOOGLE)
using ::testing::EqualsProto;
using ::testing::proto::IgnoringRepeatedFieldOrdering;
#endif  // PLATFORM_GOOGLE

TEST(PerformanceInfoWrapper, Test16BitPricision) {
  absl::string_view hlo_text = R"hlo(
HloModule test_module
ENTRY test {
  x = bf16[2,4]{1,0} parameter(0)
  y = bf16[2,4]{1,0} parameter(1)
  ROOT add = f32[2,4]{1,0} convolution(x,y), dim_labels=012_012->012
}
)hlo";
  auto hlo_module = xla::ParseAndReturnUnverifiedModule(hlo_text).value();
  const xla::HloInstruction* root =
      hlo_module->entry_computation()->root_instruction();
  std::unique_ptr<HloCostAnalysisWrapper> cost_analysis_wrapper =
      CreateXprofGpuCostAnalysis();
  ASSERT_OK(InitializeHloCostAnalysis(
      *hlo_module, *cost_analysis_wrapper->GetXlaCostAnalysis()));
  std::unique_ptr<PerformanceInfoWrapper> performance_info_wrapper =
      PerformanceInfoWrapper::Create(cost_analysis_wrapper.get(), root);
  EXPECT_EQ(performance_info_wrapper->DeviceFlops(),
            performance_info_wrapper->ModelFlops());
  EXPECT_EQ(performance_info_wrapper->ComputationalPrimitiveBitwidth(), 16);
  EXPECT_GT(performance_info_wrapper->DeviceFlops(), 0);
}

TEST(PerformanceInfoWrapper, Test4BitPricision) {
  absl::string_view hlo_text = R"hlo(
HloModule test_module
ENTRY test {
  x = s4[2,4]{1,0} parameter(0)
  y = s4[2,4]{1,0} parameter(1)
  ROOT add = f32[2,4]{1,0} convolution(x,y), dim_labels=012_012->012
}
)hlo";
  auto hlo_module = xla::ParseAndReturnUnverifiedModule(hlo_text).value();
  const xla::HloInstruction* root =
      hlo_module->entry_computation()->root_instruction();
  std::unique_ptr<HloCostAnalysisWrapper> cost_analysis_wrapper =
      CreateXprofGpuCostAnalysis();
  ASSERT_OK(InitializeHloCostAnalysis(
      *hlo_module, *cost_analysis_wrapper->GetXlaCostAnalysis()));
  std::unique_ptr<PerformanceInfoWrapper> performance_info_wrapper =
      PerformanceInfoWrapper::Create(cost_analysis_wrapper.get(), root);
  EXPECT_EQ(performance_info_wrapper->DeviceFlops(),
            performance_info_wrapper->ModelFlops() / 4);
  EXPECT_EQ(performance_info_wrapper->ComputationalPrimitiveBitwidth(), 4);
  EXPECT_GT(performance_info_wrapper->DeviceFlops(), 0);
}

TEST(PerformanceInfoWrapper, TestInputBitwidths) {
  absl::string_view hlo_text = R"hlo(
HloModule test_module
ENTRY test {
  x = s16[2,4]{1,0} parameter(0)
  y = s4[2,4]{1,0} parameter(1)
  ROOT add = f32[2,4]{1,0} convolution(x,y), dim_labels=012_012->012
}
)hlo";
  auto hlo_module = xla::ParseAndReturnUnverifiedModule(hlo_text).value();
  const xla::HloInstruction* root =
      hlo_module->entry_computation()->root_instruction();
  std::unique_ptr<HloCostAnalysisWrapper> cost_analysis_wrapper =
      CreateXprofGpuCostAnalysis();
  ASSERT_OK(InitializeHloCostAnalysis(
      *hlo_module, *cost_analysis_wrapper->GetXlaCostAnalysis()));
  std::unique_ptr<PerformanceInfoWrapper> performance_info_wrapper =
      PerformanceInfoWrapper::Create(cost_analysis_wrapper.get(), root);
  // Expect the input bitwidths to be 16 and 4 based on the graph created above.
  EXPECT_THAT(performance_info_wrapper->InputBitwidths(),
              UnorderedElementsAre(16, 4));
}

TEST(PerformanceInfoWrapper, TestMemoryAccessed) {
  auto performance_info =
      std::make_unique<PerformanceInfoWrapper::PerfInfoType>();
  tsl::protobuf::TextFormat::ParseFromString(
      R"pb(
        flops: 1000000
        bytes_accessed: 100
        memory_accessed_breakdown {
          is_read: true
          memory_space: 1
          bytes_accessed: 200
        }
        memory_accessed_breakdown {
          is_read: false
          memory_space: 1
          bytes_accessed: 300
        }
      )pb",
      performance_info.get());
  std::unique_ptr<PerformanceInfoWrapper> performance_info_wrapper =
      PerformanceInfoWrapper::Create(std::move(performance_info));
#if defined(PLATFORM_GOOGLE)
  EXPECT_THAT(performance_info_wrapper->GetMemmoryAccessBreakdown(),
              IgnoringRepeatedFieldOrdering(EqualsProto(R"pb(
                memory_accessed {
                  operation_type: READ
                  memory_space: 1
                  bytes_accessed: 200
                }
                memory_accessed {
                  operation_type: WRITE
                  memory_space: 1
                  bytes_accessed: 300
                })pb")));
#endif  // PLATFORM_GOOGLE
}

// ValidHloCost: HloCostAnalysis uses -1 as "no cost"; only that sentinel is
// remapped to 0. Other values pass through unchanged.
TEST(ValidHloCost, SentinelNegativeOneMapsToZero) {
  EXPECT_EQ(ValidHloCost(-1), 0);
}

TEST(ValidHloCost, PositiveAndZeroPassthrough) {
  EXPECT_EQ(ValidHloCost(0), 0);
  EXPECT_EQ(ValidHloCost(1), 1);
  EXPECT_EQ(ValidHloCost(42), 42);
  // Large values must not be clamped or truncated.
  constexpr int64_t kLarge = int64_t{1} << 50;
  EXPECT_EQ(ValidHloCost(kLarge), kLarge);
}

TEST(ValidHloCost, NonSentinelNegativesPassthrough) {
  // Non-sentinel negatives are adjustments for higher-precision analysis.
  EXPECT_EQ(ValidHloCost(-2), -2);
  EXPECT_EQ(ValidHloCost(-100), -100);
}

// PerformanceInfoWrapper must apply ValidHloCost on flops/bytes without
// mutating the owned PerformanceInfo proto.
TEST(PerformanceInfoWrapper, SentinelCostsMapToZero) {
  auto performance_info =
      std::make_unique<PerformanceInfoWrapper::PerfInfoType>();
  performance_info->set_flops(-1);
  performance_info->set_bytes_accessed(-1);
  const PerformanceInfoWrapper::PerfInfoType* raw = performance_info.get();

  std::unique_ptr<PerformanceInfoWrapper> wrapper =
      PerformanceInfoWrapper::Create(std::move(performance_info));

  EXPECT_EQ(wrapper->ModelFlops(), 0);
  EXPECT_EQ(wrapper->BytesAccessed(), 0);
  EXPECT_EQ(wrapper->flops(), 0);
  EXPECT_EQ(wrapper->bytes_accessed(), 0);
  // Underlying proto is unchanged (no mutation).
  EXPECT_EQ(raw->flops(), -1);
  EXPECT_EQ(raw->bytes_accessed(), -1);
}

TEST(PerformanceInfoWrapper, PositiveCostsPassthrough) {
  auto performance_info =
      std::make_unique<PerformanceInfoWrapper::PerfInfoType>();
  constexpr int64_t kFlops = int64_t{1} << 40;
  constexpr int64_t kBytes = 123456789;
  performance_info->set_flops(kFlops);
  performance_info->set_bytes_accessed(kBytes);

  std::unique_ptr<PerformanceInfoWrapper> wrapper =
      PerformanceInfoWrapper::Create(std::move(performance_info));

  EXPECT_EQ(wrapper->ModelFlops(), kFlops);
  EXPECT_EQ(wrapper->BytesAccessed(), kBytes);
  EXPECT_EQ(wrapper->flops(), kFlops);
  EXPECT_EQ(wrapper->bytes_accessed(), kBytes);
}

}  // namespace
}  // namespace profiler
}  // namespace tensorflow
