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

#include "xprof/convert/smart_suggestion/sparse_core_bound_rule.h"

#include <memory>
#include <optional>
#include <utility>
#include <vector>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/statusor.h"
#include "xprof/convert/smart_suggestion/mock_tool_data_provider.h"
#include "xprof/convert/smart_suggestion/signal_provider.h"
#include "plugin/xprof/protobuf/input_pipeline.pb.h"
#include "plugin/xprof/protobuf/smart_suggestion.pb.h"
#include "plugin/xprof/protobuf/tpu_input_pipeline.pb.h"

namespace tensorflow {
namespace profiler {
namespace {

using ::testing::DoubleEq;
using ::testing::DoubleNear;
using ::testing::Eq;
using ::testing::Return;
using ::testing::status::IsOkAndHolds;

TEST(SparseCoreBoundRuleTest, MeetsConditions) {
  auto mock_tool_data_provider = std::make_unique<MockToolDataProvider>();
  InputPipelineAnalysisResult input_pipeline_analysis;
  input_pipeline_analysis.mutable_step_time_summary()->set_average(100.0);
  TpuStepTimeBreakdown step_time_breakdown;
  step_time_breakdown.mutable_sparse_core_step_summary()
      ->mutable_sc_step_time_ms_summary()
      ->set_average(11.0);
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
  SparseCoreBoundRule rule;

  absl::StatusOr<std::optional<SmartSuggestion>> suggestion =
      rule.Apply(signal_provider);
  EXPECT_THAT(suggestion, IsOkAndHolds(testing::Not(Eq(std::nullopt))));
  EXPECT_EQ((*suggestion)->rule_name(), "SparseCoreBoundRule");
  EXPECT_THAT((*suggestion)->suggestion_text(),
              testing::HasSubstr(
                  "11.0% of the total step time </b> is spent on SparseCore"));
}

TEST(SparseCoreBoundRuleTest, IdleTimeTooLow) {
  auto mock_tool_data_provider = std::make_unique<MockToolDataProvider>();
  InputPipelineAnalysisResult input_pipeline_analysis;
  input_pipeline_analysis.mutable_step_time_summary()->set_average(100.0);
  TpuStepTimeBreakdown step_time_breakdown;
  step_time_breakdown.mutable_sparse_core_step_summary()
      ->mutable_sc_step_time_ms_summary()
      ->set_average(9.0);
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
  SparseCoreBoundRule rule;

  absl::StatusOr<std::optional<SmartSuggestion>> suggestion =
      rule.Apply(signal_provider);
  EXPECT_THAT(suggestion, IsOkAndHolds(Eq(std::nullopt)));
}

TEST(SparseCoreBoundRuleTest, NoTpuStepTimeBreakdownField) {
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
  SparseCoreBoundRule rule;

  absl::StatusOr<std::optional<SmartSuggestion>> suggestion =
      rule.Apply(signal_provider);
  EXPECT_THAT(suggestion, IsOkAndHolds(Eq(std::nullopt)));
}

TEST(SparseCoreBoundRuleTest, HbmAndMxuUtilizationTooHigh) {
  auto mock_tool_data_provider = std::make_unique<MockToolDataProvider>();
  InputPipelineAnalysisResult input_pipeline_analysis;
  input_pipeline_analysis.mutable_step_time_summary()->set_average(100.0);
  TpuStepTimeBreakdown step_time_breakdown;
  step_time_breakdown.mutable_sparse_core_step_summary()
      ->mutable_sc_step_time_ms_summary()
      ->set_average(11.0);
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
  SparseCoreBoundRule rule;

  absl::StatusOr<std::optional<SmartSuggestion>> suggestion =
      rule.Apply(signal_provider);
  EXPECT_THAT(suggestion, IsOkAndHolds(Eq(std::nullopt)));
}

TEST(SparseCoreBoundRuleTest, ExcludesSparseCoreIdleTimeFromActivePercentage) {
  auto mock_tool_data_provider = std::make_unique<MockToolDataProvider>();
  InputPipelineAnalysisResult input_pipeline_analysis;
  input_pipeline_analysis.mutable_step_time_summary()->set_average(100.0);
  TpuStepTimeBreakdown step_time_breakdown;
  step_time_breakdown.mutable_sparse_core_step_summary()
      ->mutable_sc_step_time_ms_summary()
      ->set_average(50.0);
  step_time_breakdown.mutable_sparse_core_step_summary()
      ->mutable_sc_idle_ms_summary()
      ->set_average(45.0);
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
  SparseCoreBoundRule rule;

  absl::StatusOr<std::optional<SmartSuggestion>> suggestion =
      rule.Apply(signal_provider);
  EXPECT_THAT(suggestion, IsOkAndHolds(Eq(std::nullopt)));
}

TEST(SparseCoreBoundRuleTest, SparseCoreTimePercentSubtractsIdleTime) {
  auto mock_tool_data_provider = std::make_unique<MockToolDataProvider>();
  InputPipelineAnalysisResult input_pipeline_analysis;
  input_pipeline_analysis.mutable_step_time_summary()->set_average(100.0);
  TpuStepTimeBreakdown step_time_breakdown;
  SparseCoreStepSummary* sc_summary =
      step_time_breakdown.mutable_sparse_core_step_summary();
  sc_summary->mutable_sc_step_time_ms_summary()->set_average(50.0);
  sc_summary->mutable_sc_idle_ms_summary()->set_average(30.0);
  // Compute time is ignored when the SparseCore step time is set.
  sc_summary->mutable_sc_compute_ms_summary()->set_average(7.0);
  input_pipeline_analysis.mutable_step_time_breakdown()->PackFrom(
      step_time_breakdown);
  EXPECT_CALL(*mock_tool_data_provider, GetInputPipelineAnalysisResult())
      .WillOnce(Return(&input_pipeline_analysis));

  SignalProvider signal_provider(std::move(mock_tool_data_provider));

  EXPECT_THAT(signal_provider.GetSparseCoreTimePercent(),
              IsOkAndHolds(DoubleNear(20.0, 1e-9)));
}

TEST(SparseCoreBoundRuleTest, SparseCoreTimePercentFallsBackToComponentSum) {
  auto mock_tool_data_provider = std::make_unique<MockToolDataProvider>();
  InputPipelineAnalysisResult input_pipeline_analysis;
  input_pipeline_analysis.mutable_step_time_summary()->set_average(100.0);
  TpuStepTimeBreakdown step_time_breakdown;
  SparseCoreStepSummary* sc_summary =
      step_time_breakdown.mutable_sparse_core_step_summary();
  sc_summary->mutable_sc_compute_ms_summary()->set_average(10.0);
  sc_summary->mutable_sc_infeed_ms_summary()->set_average(5.0);
  sc_summary->mutable_sc_outfeed_ms_summary()->set_average(3.0);
  input_pipeline_analysis.mutable_step_time_breakdown()->PackFrom(
      step_time_breakdown);
  EXPECT_CALL(*mock_tool_data_provider, GetInputPipelineAnalysisResult())
      .WillOnce(Return(&input_pipeline_analysis));

  SignalProvider signal_provider(std::move(mock_tool_data_provider));

  EXPECT_THAT(signal_provider.GetSparseCoreTimePercent(),
              IsOkAndHolds(DoubleNear(18.0, 1e-9)));
}

TEST(SparseCoreBoundRuleTest, SparseCoreTimePercentClampsNegativeActiveTime) {
  auto mock_tool_data_provider = std::make_unique<MockToolDataProvider>();
  InputPipelineAnalysisResult input_pipeline_analysis;
  input_pipeline_analysis.mutable_step_time_summary()->set_average(100.0);
  TpuStepTimeBreakdown step_time_breakdown;
  SparseCoreStepSummary* sc_summary =
      step_time_breakdown.mutable_sparse_core_step_summary();
  sc_summary->mutable_sc_step_time_ms_summary()->set_average(20.0);
  sc_summary->mutable_sc_idle_ms_summary()->set_average(35.0);
  input_pipeline_analysis.mutable_step_time_breakdown()->PackFrom(
      step_time_breakdown);
  EXPECT_CALL(*mock_tool_data_provider, GetInputPipelineAnalysisResult())
      .WillOnce(Return(&input_pipeline_analysis));

  SignalProvider signal_provider(std::move(mock_tool_data_provider));

  EXPECT_THAT(signal_provider.GetSparseCoreTimePercent(),
              IsOkAndHolds(DoubleEq(0.0)));
}

TEST(SparseCoreBoundRuleTest,
     SparseCoreTimePercentIsZeroWithoutPositiveStepTime) {
  for (double step_time_ms : {0.0, -10.0}) {
    auto mock_tool_data_provider = std::make_unique<MockToolDataProvider>();
    InputPipelineAnalysisResult input_pipeline_analysis;
    input_pipeline_analysis.mutable_step_time_summary()->set_average(
        step_time_ms);
    TpuStepTimeBreakdown step_time_breakdown;
    step_time_breakdown.mutable_sparse_core_step_summary()
        ->mutable_sc_step_time_ms_summary()
        ->set_average(50.0);
    input_pipeline_analysis.mutable_step_time_breakdown()->PackFrom(
        step_time_breakdown);
    EXPECT_CALL(*mock_tool_data_provider, GetInputPipelineAnalysisResult())
        .WillOnce(Return(&input_pipeline_analysis));

    SignalProvider signal_provider(std::move(mock_tool_data_provider));

    EXPECT_THAT(signal_provider.GetSparseCoreTimePercent(),
                IsOkAndHolds(DoubleEq(0.0)))
        << "Failed for step_time_ms: " << step_time_ms;
  }
}

}  // namespace
}  // namespace profiler
}  // namespace tensorflow
