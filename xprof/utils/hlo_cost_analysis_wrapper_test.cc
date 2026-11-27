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

#include "xprof/utils/hlo_cost_analysis_wrapper.h"

#include <cstdint>
#include <memory>
#include <tuple>
#include <vector>

#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/tsl/platform/test.h"
#include "plugin/xprof/protobuf/op_metrics.pb.h"

namespace tensorflow {
namespace profiler {
namespace {
using ::testing::ElementsAre;

using MemorySpace = PerformanceInfo::MemoryAccessed::MemorySpace;

// Wrapper with a fixed mapping of the XProf memory spaces to XLA memory
// spaces 0 to 3.
class FakeHloCostAnalysisWrapper : public HloCostAnalysisWrapper {
 public:
  FakeHloCostAnalysisWrapper()
      : cost_analysis_(std::make_unique<xla::HloCostAnalysis>()) {}

  xla::HloCostAnalysis* GetXlaCostAnalysis() const override {
    return cost_analysis_.get();
  }

 protected:
  MemorySpaceMap GetMemorySpaceMapping() const override {
    return {{PerformanceInfo::MemoryAccessed::VMEM, 2},
            {PerformanceInfo::MemoryAccessed::CMEM, 1},
            {PerformanceInfo::MemoryAccessed::HBM, 0},
            {PerformanceInfo::MemoryAccessed::UNKNOWN, 3}};
  }

  int64_t GetDeviceFlopsAdjustment(
      const xla::HloInstruction& hlo) const override {
    return 0;
  }

 private:
  std::unique_ptr<xla::HloCostAnalysis> cost_analysis_;
};

TEST(TpuHloCostAnalysisTest, GetInputBitwidths) {
  absl::string_view hlo_text = R"hlo(
HloModule test_module
ENTRY test {
   x = bf16[2,4]{1,0} parameter(0)
   y = s4[2,4]{1,0} parameter(1)
   ROOT add = f32[2,4]{1,0} convolution(x,y), dim_labels=012_012->012
   }
)hlo";
  auto hlo_module = xla::ParseAndReturnUnverifiedModule(hlo_text).value();
  const xla::HloInstruction* root =
      hlo_module->entry_computation()->root_instruction();
  EXPECT_THAT(GetInputBitwidths(*root), ElementsAre(16, 4));
}

TEST(HloCostAnalysisWrapperTest, MemoryAccessedBreakdownIsInMemorySpaceOrder) {
  // Each operand is in a different XLA memory space. The output is in memory
  // space 2.
  absl::string_view hlo_text = R"hlo(
HloModule test_module
ENTRY test {
   a = f32[4]{0} parameter(0)
   b = f32[4]{0:S(1)} parameter(1)
   c = f32[4]{0:S(2)} parameter(2)
   d = f32[4]{0:S(3)} parameter(3)
   ROOT concat = f32[16]{0:S(2)} concatenate(a, b, c, d), dimensions={0}
   }
)hlo";
  auto hlo_module = xla::ParseAndReturnUnverifiedModule(hlo_text).value();
  FakeHloCostAnalysisWrapper wrapper;
  ASSERT_TRUE(hlo_module->entry_computation()
                  ->Accept(wrapper.GetXlaCostAnalysis())
                  .ok());
  const xla::HloInstruction* root =
      hlo_module->entry_computation()->root_instruction();

  std::unique_ptr<PerformanceInfo> performance_info =
      wrapper.GeneratePerformanceInfo(*root);

  std::vector<std::tuple<MemorySpace, bool, int64_t>> breakdown;
  for (const PerformanceInfo::MemoryAccessed& memory_accessed :
       performance_info->memory_accessed_breakdown()) {
    breakdown.emplace_back(memory_accessed.memory_space(),
                           memory_accessed.is_read(),
                           memory_accessed.bytes_accessed());
  }
  // GetMemorySpaceMapping() lists the memory spaces in reverse order, and
  // MemorySpaceMap orders them by MemorySpace.
  EXPECT_THAT(
      breakdown,
      ElementsAre(
          std::make_tuple(PerformanceInfo::MemoryAccessed::UNKNOWN, true, 16),
          std::make_tuple(PerformanceInfo::MemoryAccessed::HBM, true, 16),
          std::make_tuple(PerformanceInfo::MemoryAccessed::CMEM, true, 16),
          std::make_tuple(PerformanceInfo::MemoryAccessed::VMEM, true, 16),
          std::make_tuple(PerformanceInfo::MemoryAccessed::VMEM, false, 64)));
}
}  // namespace
}  // namespace profiler
}  // namespace tensorflow
