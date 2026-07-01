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

#include "xprof/convert/smart_suggestion/tensor_core_idle_bound_rule.h"

#include <memory>
#include <optional>
#include <utility>
#include <vector>
#include <string>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/statusor.h"
#include "xprof/convert/smart_suggestion/mock_tool_data_provider.h"
#include "xprof/convert/smart_suggestion/signal_provider.h"
#include "plugin/xprof/protobuf/input_pipeline.pb.h"
#include "plugin/xprof/protobuf/smart_suggestion.pb.h"
#include "plugin/xprof/protobuf/tpu_input_pipeline.pb.h"

namespace tensorflow {
namespace profiler {
namespace {

using ::testing::Eq;
using ::testing::Return;
using ::testing::status::IsOkAndHolds;

TEST(TensorCoreIdleBoundRuleTest, MeetsConditions) {
  auto mock_tool_data_provider = std::make_unique<MockToolDataProvider>();
  InputPipelineAnalysisResult input_pipeline_analysis;
  input_pipeline_analysis.mutable_step_time_summary()->set_average(100.0);
  TpuStepTimeBreakdown step_time_breakdown;
  step_time_breakdown.mutable_tc_idle_ms_summary()->set_average(11.0);
  input_pipeline_analysis.mutable_step_time_breakdown()->PackFrom(
      step_time_breakdown);

  EXPECT_CALL(*mock_tool_data_provider, GetInputPipelineAnalysisResult())
      .WillRepeatedly(Return(&input_pipeline_analysis));

  OverviewPage overview_page;
  overview_page.mutable_analysis()->set_mxu_utilization_percent(49.0);
  overview_page.mutable_analysis()
      ->set_memory_bw_utilization_relative_to_hw_limit_percent(49.0);
  EXPECT_CALL(*mock_tool_data_provider, GetOverviewPage())
      .WillRepeatedly(Return(&overview_page));

  SignalProvider signal_provider(std::move(mock_tool_data_provider));
  TensorCoreIdleBoundRule rule;

  absl::StatusOr<std::optional<SmartSuggestion>> suggestion =
      rule.Apply(signal_provider);
  EXPECT_THAT(suggestion, IsOkAndHolds(testing::Not(Eq(std::nullopt))));
  EXPECT_EQ((*suggestion)->rule_name(), "TensorCoreIdleBoundRule");
  EXPECT_THAT((*suggestion)->suggestion_text(),
              testing::HasSubstr("High TensorCore idle time percentage of "
                                 "<b>11.0%</b>"));
}

TEST(TensorCoreIdleBoundRuleTest, IdleTimeTooLow) {
  auto mock_tool_data_provider = std::make_unique<MockToolDataProvider>();
  InputPipelineAnalysisResult input_pipeline_analysis;
  input_pipeline_analysis.mutable_step_time_summary()->set_average(100.0);
  TpuStepTimeBreakdown step_time_breakdown;
  step_time_breakdown.mutable_tc_idle_ms_summary()->set_average(9.0);
  input_pipeline_analysis.mutable_step_time_breakdown()->PackFrom(
      step_time_breakdown);

  EXPECT_CALL(*mock_tool_data_provider, GetInputPipelineAnalysisResult())
      .WillRepeatedly(Return(&input_pipeline_analysis));

  OverviewPage overview_page;
  overview_page.mutable_analysis()->set_mxu_utilization_percent(49.0);
  overview_page.mutable_analysis()
      ->set_memory_bw_utilization_relative_to_hw_limit_percent(49.0);
  EXPECT_CALL(*mock_tool_data_provider, GetOverviewPage())
      .WillRepeatedly(Return(&overview_page));

  SignalProvider signal_provider(std::move(mock_tool_data_provider));
  TensorCoreIdleBoundRule rule;

  absl::StatusOr<std::optional<SmartSuggestion>> suggestion =
      rule.Apply(signal_provider);
  EXPECT_THAT(suggestion, IsOkAndHolds(Eq(std::nullopt)));
}

TEST(TensorCoreIdleBoundRuleTest, NoTpuStepTimeBreakdownField) {
  auto mock_tool_data_provider = std::make_unique<MockToolDataProvider>();
  InputPipelineAnalysisResult input_pipeline_analysis;
  input_pipeline_analysis.mutable_step_time_summary()->set_average(100.0);
  GenericStepTimeBreakdown step_time_breakdown;
  step_time_breakdown.mutable_input_ms_summary()->set_average(9.0);
  input_pipeline_analysis.mutable_step_time_breakdown()->PackFrom(
      step_time_breakdown);

  EXPECT_CALL(*mock_tool_data_provider, GetInputPipelineAnalysisResult())
      .WillRepeatedly(Return(&input_pipeline_analysis));

  SignalProvider signal_provider(std::move(mock_tool_data_provider));
  TensorCoreIdleBoundRule rule;

  absl::StatusOr<std::optional<SmartSuggestion>> suggestion =
      rule.Apply(signal_provider);
  EXPECT_THAT(suggestion, IsOkAndHolds(Eq(std::nullopt)));
}

TEST(TensorCoreIdleBoundRuleTest, HbmAndMxuUtilizationTooHigh) {
  auto mock_tool_data_provider = std::make_unique<MockToolDataProvider>();
  InputPipelineAnalysisResult input_pipeline_analysis;
  input_pipeline_analysis.mutable_step_time_summary()->set_average(100.0);
  TpuStepTimeBreakdown step_time_breakdown;
  step_time_breakdown.mutable_tc_idle_ms_summary()->set_average(11.0);
  input_pipeline_analysis.mutable_step_time_breakdown()->PackFrom(
      step_time_breakdown);

  EXPECT_CALL(*mock_tool_data_provider, GetInputPipelineAnalysisResult())
      .WillRepeatedly(Return(&input_pipeline_analysis));

  OverviewPage overview_page;
  overview_page.mutable_analysis()->set_mxu_utilization_percent(51.0);
  overview_page.mutable_analysis()
      ->set_memory_bw_utilization_relative_to_hw_limit_percent(51.0);
  EXPECT_CALL(*mock_tool_data_provider, GetOverviewPage())
      .WillRepeatedly(Return(&overview_page));

  SignalProvider signal_provider(std::move(mock_tool_data_provider));
  TensorCoreIdleBoundRule rule;

  absl::StatusOr<std::optional<SmartSuggestion>> suggestion =
      rule.Apply(signal_provider);
  EXPECT_THAT(suggestion, IsOkAndHolds(Eq(std::nullopt)));
}

}  // namespace
}  // namespace profiler
}  // namespace tensorflow
