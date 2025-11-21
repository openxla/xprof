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

#include "xprof/convert/smart_suggestion/collective_bound_rule.h"

#include <memory>
#include <utility>
#include <vector>

#include "testing/base/public/gmock.h"
#include "<gtest/gtest.h>"
#include "absl/status/status.h"
#include "xprof/convert/smart_suggestion/mock_tool_data_provider.h"
#include "xprof/convert/smart_suggestion/signal_provider.h"
#include "plugin/xprof/protobuf/smart_suggestion.pb.h"
#include "plugin/xprof/protobuf/tf_op_stats.pb.h"
#include "tensorflow/core/profiler/protobuf/op_stats.pb.h"
#include "plugin/xprof/protobuf/op_metrics.pb.h"
#include "plugin/xprof/protobuf/steps_db.pb.h"

namespace tensorflow {
namespace profiler {
namespace {

using ::testing::Return;
using ::testing::status::StatusIs;

class CollectiveBoundRuleTest : public ::testing::Test {
 protected:
  void SetUp() override {
    auto mock_tool_data_provider = std::make_unique<MockToolDataProvider>();
    mock_tool_data_provider_ = mock_tool_data_provider.get();
    signal_provider_ =
        std::make_unique<SignalProvider>(std::move(mock_tool_data_provider));
    collective_bound_rule_ = std::make_unique<CollectiveBoundRule>();
  }

  MockToolDataProvider* mock_tool_data_provider_;
  std::unique_ptr<SignalProvider> signal_provider_;
  std::unique_ptr<CollectiveBoundRule> collective_bound_rule_;
};

TEST_F(CollectiveBoundRuleTest, MockCallTest) {
  xprof::TfOpStats tf_op_stats;
  auto* step_db = tf_op_stats.mutable_op_stats()->mutable_step_db();
  auto* step = step_db->add_step_sequence();
  (*step->mutable_step_info_per_core())[0].set_duration_ps(100);
  auto* metrics = step->mutable_hlo_metrics_db()->add_metrics_db();
  metrics->set_category("all-reduce");
  metrics->set_self_time_ps(5);

  EXPECT_CALL(*mock_tool_data_provider_, GetTfOpStats())
      .WillOnce(Return(&tf_op_stats));
  auto result = signal_provider_->GetAvgCollectiveTimePercent();
  EXPECT_OK(result);
  EXPECT_NEAR(*result, 5.0, 0.001);
}

TEST_F(CollectiveBoundRuleTest, MeetsConditionsAboveThreshold) {
  xprof::TfOpStats tf_op_stats;
  auto* step_db = tf_op_stats.mutable_op_stats()->mutable_step_db();
  auto* step1 = step_db->add_step_sequence();
  (*step1->mutable_step_info_per_core())[0].set_duration_ps(100);
  auto* metrics1 = step1->mutable_hlo_metrics_db()->add_metrics_db();
  metrics1->set_category("all-reduce");
  metrics1->set_self_time_ps(5);

  auto* step2 = step_db->add_step_sequence();
  (*step2->mutable_step_info_per_core())[0].set_duration_ps(100);
  auto* metrics2 = step2->mutable_hlo_metrics_db()->add_metrics_db();
  metrics2->set_category("all-reduce");
  metrics2->set_self_time_ps(6);

  EXPECT_CALL(*mock_tool_data_provider_, GetTfOpStats())
      .WillOnce(Return(&tf_op_stats));

  EXPECT_TRUE(collective_bound_rule_->MeetsConditions(*signal_provider_));
}

TEST_F(CollectiveBoundRuleTest, MeetsConditionsBelowThreshold) {
  xprof::TfOpStats tf_op_stats;
  auto* step_db = tf_op_stats.mutable_op_stats()->mutable_step_db();
  auto* step1 = step_db->add_step_sequence();
  (*step1->mutable_step_info_per_core())[0].set_duration_ps(100);
  auto* metrics1 = step1->mutable_hlo_metrics_db()->add_metrics_db();
  metrics1->set_category("all-reduce");
  metrics1->set_self_time_ps(1);

  auto* step2 = step_db->add_step_sequence();
  (*step2->mutable_step_info_per_core())[0].set_duration_ps(100);
  auto* metrics2 = step2->mutable_hlo_metrics_db()->add_metrics_db();
  metrics2->set_category("all-reduce");
  metrics2->set_self_time_ps(2);

  EXPECT_CALL(*mock_tool_data_provider_, GetTfOpStats())
      .WillOnce(Return(&tf_op_stats));

  EXPECT_FALSE(collective_bound_rule_->MeetsConditions(*signal_provider_));
}

TEST_F(CollectiveBoundRuleTest, MeetsConditionsError) {
  EXPECT_CALL(*mock_tool_data_provider_, GetTfOpStats())
      .WillOnce(Return(absl::InternalError("Test Error")));

  EXPECT_FALSE(collective_bound_rule_->MeetsConditions(*signal_provider_));
}

TEST_F(CollectiveBoundRuleTest, GenerateSuggestionError) {
  EXPECT_CALL(*mock_tool_data_provider_, GetTfOpStats())
      .WillOnce(Return(absl::InternalError("Test Error")));

  EXPECT_THAT(collective_bound_rule_->GenerateSuggestion(*signal_provider_),
              StatusIs(absl::StatusCode::kInternal));
}

}  // namespace
}  // namespace profiler
}  // namespace tensorflow
