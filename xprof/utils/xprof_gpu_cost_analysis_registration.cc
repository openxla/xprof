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


#include <memory>

#include "xprof/utils/function_registry.h"
#include "xprof/utils/hlo_cost_analysis_wrapper.h"
#include "xprof/utils/xprof_gpu_cost_analysis.h"
#include "xprof/utils/xprof_gpu_cost_analysis_types.h"


namespace tensorflow {
namespace profiler {

namespace {

const auto kXprofGpuCostAnalysisRegistration = RegisterOrDie(
    &GetHloCostAnalysisWrapperRegistry(), kXprofGpuCostAnalysisName,
    [](const CostAnalysisConfig* cost_analysis_config)
        -> std::unique_ptr<tensorflow::profiler::HloCostAnalysisWrapper> {
      return CreateXprofGpuCostAnalysis();
    }
);

}  // namespace


}  // namespace profiler
}  // namespace tensorflow
