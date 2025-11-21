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

#ifndef THIRD_PARTY_XPROF_CONVERT_SMART_SUGGESTION_MOCK_TOOL_DATA_PROVIDER_H_
#define THIRD_PARTY_XPROF_CONVERT_SMART_SUGGESTION_MOCK_TOOL_DATA_PROVIDER_H_

#include <string>

#include "testing/base/public/gmock.h"
#include "absl/status/statusor.h"
#include "xprof/convert/smart_suggestion/tool_data_provider.h"
#include "plugin/xprof/protobuf/event_time_fraction_analyzer.pb.h"
#include "plugin/xprof/protobuf/input_pipeline.pb.h"
#include "plugin/xprof/protobuf/overview_page.pb.h"

namespace tensorflow {
namespace profiler {

// Mock ToolDataProvider
class MockToolDataProvider : public ToolDataProvider {
 public:
  MOCK_METHOD(absl::StatusOr<const OverviewPage*>, GetOverviewPage, (),
              (override));
  MOCK_METHOD(absl::StatusOr<const InputPipelineAnalysisResult*>,
              GetInputPipelineAnalysisResult, (), (override));
  MOCK_METHOD(absl::StatusOr<const EventTimeFractionAnalyzerResult*>,
              GetEventTimeFractionAnalyzerResult, (const std::string&),
              (override));
};

}  // namespace profiler
}  // namespace tensorflow

#endif  // THIRD_PARTY_XPROF_CONVERT_SMART_SUGGESTION_MOCK_TOOL_DATA_PROVIDER_H_
