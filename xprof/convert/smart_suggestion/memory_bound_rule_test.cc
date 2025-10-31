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

#include "xprof/convert/smart_suggestion/memory_bound_rule.h"

#include <memory>
#include <optional>
#include <utility>
#include <string>
#include <vector>

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
  MOCK_METHOD(absl::StatusOr<std::vector<float>>,
              GetEventTimeFractionEachStep, (const std::string&), (override));
};

TEST(MemoryBoundRuleTest, MeetsConditions) {
  auto mock_tool_data_provider = std::make_unique<MockToolDataProvider>();
  OverviewPage overview_page;
  overview_page.mutable_analysis()
      ->set_memory_bw_utilization_relative_to_hw_limit_percent(71.0);
  overview_page.mutable_analysis()->set_mxu_utilization_percent(49.0);

  EXPECT_CALL(*mock_tool_data_provider, GetOverviewPage())
      .WillRepeatedly(Return(&overview_page));

  SignalProvider signal_provider(std::move(mock_tool_data_provider));
  MemoryBoundRule rule;

  absl::StatusOr<std::optional<SmartSuggestion>> suggestion =
      rule.Apply(signal_provider);
  EXPECT_THAT(suggestion, IsOkAndHolds(testing::Not(Eq(std::nullopt))));
  EXPECT_EQ((*suggestion)->rule_name(), "MemoryBoundRule");
  EXPECT_THAT((*suggestion)->suggestion_text(),
              testing::HasSubstr(
                  "71.0%</b> and low MXU utilization of <b>49.0%"));
}

TEST(MemoryBoundRuleTest, HbmUtilizationTooLow) {
  auto mock_tool_data_provider = std::make_unique<MockToolDataProvider>();
  OverviewPage overview_page;
  overview_page.mutable_analysis()
      ->set_memory_bw_utilization_relative_to_hw_limit_percent(69.0);
  overview_page.mutable_analysis()->set_mxu_utilization_percent(49.0);

  EXPECT_CALL(*mock_tool_data_provider, GetOverviewPage())
      .WillRepeatedly(Return(&overview_page));

  SignalProvider signal_provider(std::move(mock_tool_data_provider));
  MemoryBoundRule rule;

  absl::StatusOr<std::optional<SmartSuggestion>> suggestion =
      rule.Apply(signal_provider);
  EXPECT_THAT(suggestion, IsOkAndHolds(Eq(std::nullopt)));
}

TEST(MemoryBoundRuleTest, MxuUtilizationTooHigh) {
  auto mock_tool_data_provider = std::make_unique<MockToolDataProvider>();
  OverviewPage overview_page;
  overview_page.mutable_analysis()
      ->set_memory_bw_utilization_relative_to_hw_limit_percent(71.0);
  overview_page.mutable_analysis()->set_mxu_utilization_percent(51.0);

  EXPECT_CALL(*mock_tool_data_provider, GetOverviewPage())
      .WillRepeatedly(Return(&overview_page));

  SignalProvider signal_provider(std::move(mock_tool_data_provider));
  MemoryBoundRule rule;

  absl::StatusOr<std::optional<SmartSuggestion>> suggestion =
      rule.Apply(signal_provider);
  EXPECT_THAT(suggestion, IsOkAndHolds(Eq(std::nullopt)));
}

}  // namespace
}  // namespace profiler
}  // namespace tensorflow
