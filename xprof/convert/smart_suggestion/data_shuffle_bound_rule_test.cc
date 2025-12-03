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

#include "xprof/convert/smart_suggestion/data_shuffle_bound_rule.h"

#include <memory>
#include <utility>

#include "testing/base/public/gmock.h"
#include "<gtest/gtest.h>"
#include "absl/status/status.h"
#include "tensorflow/core/profiler/protobuf/op_metrics.pb.h"
#include "tensorflow/core/profiler/protobuf/op_stats.pb.h"
#include "tensorflow/core/profiler/protobuf/steps_db.pb.h"
#include "xprof/convert/smart_suggestion/mock_tool_data_provider.h"
#include "xprof/convert/smart_suggestion/signal_provider.h"
#include "plugin/xprof/protobuf/smart_suggestion.pb.h"

namespace tensorflow {
namespace profiler {
namespace {

using ::testing::Return;
using ::testing::status::StatusIs;

class DataShuffleBoundRuleTest : public ::testing::Test {
 protected:
  void SetUp() override {
    auto mock_tool_data_provider = std::make_unique<MockToolDataProvider>();
    mock_tool_data_provider_ = mock_tool_data_provider.get();
    signal_provider_ =
        std::make_unique<SignalProvider>(std::move(mock_tool_data_provider));
    data_shuffle_bound_rule_ = std::make_unique<DataShuffleBoundRule>();
  }

  MockToolDataProvider* mock_tool_data_provider_;
  std::unique_ptr<SignalProvider> signal_provider_;
  std::unique_ptr<DataShuffleBoundRule> data_shuffle_bound_rule_;
};

TEST_F(DataShuffleBoundRuleTest, MeetsConditionsAboveThreshold) {
  tensorflow::profiler::OpStats op_stats;
  auto* step_db = op_stats.mutable_step_db();
  auto* step1 = step_db->add_step_sequence();
  (*step1->mutable_step_info_per_core())[0].set_duration_ps(100);
  auto* metrics1 = step1->mutable_hlo_metrics_db()->add_metrics_db();
  metrics1->set_category("shuffle");
  metrics1->set_self_time_ps(24);

  auto* step2 = step_db->add_step_sequence();
  (*step2->mutable_step_info_per_core())[0].set_duration_ps(100);
  auto* metrics2 = step2->mutable_hlo_metrics_db()->add_metrics_db();
  metrics2->set_category("gather");
  metrics2->set_self_time_ps(37);

  EXPECT_CALL(*mock_tool_data_provider_, GetOpStats())
      .WillOnce(Return(&op_stats));

  EXPECT_TRUE(data_shuffle_bound_rule_->MeetsConditions(*signal_provider_));
}

TEST_F(DataShuffleBoundRuleTest, MeetsConditionsBelowThreshold) {
  tensorflow::profiler::OpStats op_stats;
  auto* step_db = op_stats.mutable_step_db();
  auto* step1 = step_db->add_step_sequence();
  (*step1->mutable_step_info_per_core())[0].set_duration_ps(100);
  auto* metrics1 = step1->mutable_hlo_metrics_db()->add_metrics_db();
  metrics1->set_category("shuffle");
  metrics1->set_self_time_ps(1);

  auto* step2 = step_db->add_step_sequence();
  (*step2->mutable_step_info_per_core())[0].set_duration_ps(100);
  auto* metrics2 = step2->mutable_hlo_metrics_db()->add_metrics_db();
  metrics2->set_category("gather");
  metrics2->set_self_time_ps(2);

  EXPECT_CALL(*mock_tool_data_provider_, GetOpStats())
      .WillOnce(Return(&op_stats));

  EXPECT_FALSE(data_shuffle_bound_rule_->MeetsConditions(*signal_provider_));
}

TEST_F(DataShuffleBoundRuleTest, MeetsConditionsError) {
  EXPECT_CALL(*mock_tool_data_provider_, GetOpStats())
      .WillOnce(Return(absl::InternalError("Test Error")));

  EXPECT_FALSE(data_shuffle_bound_rule_->MeetsConditions(*signal_provider_));
}

TEST_F(DataShuffleBoundRuleTest, GenerateSuggestionError) {
  EXPECT_CALL(*mock_tool_data_provider_, GetOpStats())
      .WillOnce(Return(absl::InternalError("Test Error")));

  EXPECT_THAT(data_shuffle_bound_rule_->GenerateSuggestion(*signal_provider_),
              StatusIs(absl::StatusCode::kInternal));
}

}  // namespace
}  // namespace profiler
}  // namespace tensorflow
