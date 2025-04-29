#include <memory>

#include "base/googleinit.h"
#include "absl/log/log.h"
#include "xprof/utils/hlo_cost_analysis_wrapper.h"
#include "xprof/utils/xprof_gpu_cost_analysis.h"

namespace {

const char* const kXprofGpuCostAnalysisName = "XprofGpuCostAnalysis";

const auto kXprofGpuCostAnalysisRegistration =
    tensorflow::profiler::GetHloCostAnalysisWrapperRegistry().Register(
        kXprofGpuCostAnalysisName,
        [](const tensorflow::profiler::UsageData& usage_data)
            -> std::unique_ptr<tensorflow::profiler::HloCostAnalysisWrapper> {
          return tensorflow::profiler::CreateXprofGpuCostAnalysis();
        });
}  // namespace
REGISTER_MODULE_INITIALIZER(xprof_gpu_cost_analysis_registration, {
  if (kXprofGpuCostAnalysisRegistration) {
    LOG(INFO) << "Successfully registered XprofGpuCostAnalysis";
  } else {
    LOG(ERROR) << "Failed to register XprofGpuCostAnalysis";
  }
});
