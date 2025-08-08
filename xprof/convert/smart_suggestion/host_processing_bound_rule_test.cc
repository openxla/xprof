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

#include "xprof/convert/smart_suggestion/host_processing_bound_rule.h"

#include <memory>
#include <optional>
#include <utility>

#include "testing/base/public/gmock.h"
#include "<gtest/gtest.h>"
#include "absl/status/statusor.h"
#include "xprof/convert/smart_suggestion/signal_provider.h"
#include "xprof/convert/smart_suggestion/tool_data_provider.h"
#include "plugin/xprof/protobuf/input_pipeline.pb.h"
#include "plugin/xprof/protobuf/overview_page.pb.h"
#include "plugin/xprof/protobuf/smart_suggestion.pb.h"

namespace tensorflow {
namespace profiler {
namespace {

using ::testing::Eq;
using ::testing::Return;
using ::testing::status::IsOkAndHolds;

// Mock ToolDataProvider
class MockToolDataProvider : public ToolDataProvider {
 public:
  MOCK_METHOD(absl::StatusOr<const OverviewPage*>, GetOverviewPage, (),
              (override));
  MOCK_METHOD(absl::StatusOr<const InputPipelineAnalysisResult*>,
              GetInputPipelineAnalysisResult, (), (override));
};

TEST(HostProcessingBoundRuleTest, MeetsConditions) {
  auto mock_tool_data_provider = std::make_unique<MockToolDataProvider>();
  InputPipelineAnalysisResult result;
  // Scenario: Input bound, and high non-enqueue time
  result.set_input_percent(20.0);  // Input bound
  result.mutable_input_time_breakdown()->set_enqueue_us(10.0);
  // High non-enqueue
  result.mutable_input_time_breakdown()->set_demanded_file_read_us(90.0);

  EXPECT_CALL(*mock_tool_data_provider, GetInputPipelineAnalysisResult())
      .WillRepeatedly(Return(&result));

  SignalProvider signal_provider(std::move(mock_tool_data_provider));
  HostProcessingBoundRule rule;

  absl::StatusOr<std::optional<SmartSuggestion>> suggestion =
      rule.Apply(signal_provider);
  EXPECT_THAT(suggestion, IsOkAndHolds(testing::Not(Eq(std::nullopt))));
  EXPECT_EQ((*suggestion)->rule_name(), "HostProcessingBoundRule");
  EXPECT_THAT((*suggestion)->suggestion_text(),
              testing::HasSubstr("18.0% of the total step time"));
}

TEST(HostProcessingBoundRuleTest, NotInputBound) {
  auto mock_tool_data_provider = std::make_unique<MockToolDataProvider>();
  InputPipelineAnalysisResult result;
  // Scenario: Not input bound
  result.set_input_percent(5.0);  // Not input bound
  result.mutable_input_time_breakdown()->set_enqueue_us(10.0);
  result.mutable_input_time_breakdown()->set_demanded_file_read_us(90.0);

  EXPECT_CALL(*mock_tool_data_provider, GetInputPipelineAnalysisResult())
      .WillRepeatedly(Return(&result));

  SignalProvider signal_provider(std::move(mock_tool_data_provider));
  HostProcessingBoundRule rule;

  absl::StatusOr<std::optional<SmartSuggestion>> suggestion =
      rule.Apply(signal_provider);
  EXPECT_THAT(suggestion, IsOkAndHolds(Eq(std::nullopt)));
}

TEST(HostProcessingBoundRuleTest, InputBoundButNotHostProcessingBound) {
  auto mock_tool_data_provider = std::make_unique<MockToolDataProvider>();
  InputPipelineAnalysisResult result;
  // Scenario: Input bound, but low non-enqueue time
  result.set_input_percent(20.0);  // Input bound
  result.mutable_input_time_breakdown()->set_enqueue_us(90.0);
  // Low non-enqueue
  result.mutable_input_time_breakdown()->set_demanded_file_read_us(10.0);

  EXPECT_CALL(*mock_tool_data_provider, GetInputPipelineAnalysisResult())
      .WillRepeatedly(Return(&result));

  SignalProvider signal_provider(std::move(mock_tool_data_provider));
  HostProcessingBoundRule rule;

  absl::StatusOr<std::optional<SmartSuggestion>> suggestion =
      rule.Apply(signal_provider);
  EXPECT_THAT(suggestion, IsOkAndHolds(Eq(std::nullopt)));
}

}  // namespace
}  // namespace profiler
}  // namespace tensorflow
