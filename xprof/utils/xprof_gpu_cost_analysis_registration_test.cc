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

#include "testing/base/public/gmock.h"
#include "<gtest/gtest.h>"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xprof/utils/hlo_cost_analysis_wrapper.h"
#include "xprof/utils/hlo_module_map.h"
#include "xprof/utils/xprof_gpu_cost_analysis_types.h"

namespace tensorflow {
namespace profiler {
namespace {

class XprofGpuCostAnalysisRegistrationTest
    : public xla::HloHardwareIndependentTestBase {};

TEST_F(XprofGpuCostAnalysisRegistrationTest, CallRegisteredFactory) {
  auto& registry = GetHloCostAnalysisWrapperRegistry();
  ASSERT_NE(&registry, nullptr);
  auto cost_analysis_factory =
      registry.Get(kXprofGpuCostAnalysisName);
  EXPECT_TRUE(cost_analysis_factory);

  XprofGpuCostAnalysisConfig
      xprof_gpu_cost_analysis_config;
  auto cost_analysis = cost_analysis_factory(&xprof_gpu_cost_analysis_config);
  EXPECT_NE(cost_analysis, nullptr);

  // Testing the implementation of the cost analysis with a dummy HLO module
  const char* hlo_text = R"hlo(
    HloModule test_module
    ENTRY test {
      ROOT fusion = f32[2,4]{1,0} fusion(f32[2,4]{1,0}), kind=kOutput, calls=
      {
       x = bf16[2,4]{1,0} parameter(0)
       y = bf16[2,4]{1,0} parameter(1)
       add = f32[2,4]{1,0} convolution(x,y), dim_labels=012_012->012
      }
    }
  )hlo";
  auto hlo_module = xla::ParseAndReturnUnverifiedModule(hlo_text).value();
  ASSERT_OK(InitializeHloCostAnalysis(
      *hlo_module, *cost_analysis->GetXlaCostAnalysis()));
  const xla::HloInstruction* root =
      hlo_module->entry_computation()->root_instruction();
  EXPECT_EQ(cost_analysis->GetDeviceFlops(*root),
            cost_analysis->GetModelFlops(*root))
      << " for " << root->ToString();
}

}  // namespace
}  // namespace profiler
}  // namespace tensorflow
