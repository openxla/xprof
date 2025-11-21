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
  EXPECT_CALL(*mock_tool_data_provider_, GetCollectiveTimeFractionEachStep())
      .WillOnce(Return(std::vector<float>{0.05}));
  auto result = signal_provider_->GetAvgCollectiveTimePercent();
  EXPECT_OK(result);
}

TEST_F(CollectiveBoundRuleTest, MeetsConditionsAboveThreshold) {
  EXPECT_CALL(*mock_tool_data_provider_, GetCollectiveTimeFractionEachStep())
      .WillOnce(Return(std::vector<float>{0.05, 0.06}));

  EXPECT_TRUE(collective_bound_rule_->MeetsConditions(*signal_provider_));
}

TEST_F(CollectiveBoundRuleTest, MeetsConditionsBelowThreshold) {
  EXPECT_CALL(*mock_tool_data_provider_, GetCollectiveTimeFractionEachStep())
      .WillOnce(Return(std::vector<float>{0.01, 0.02}));

  EXPECT_FALSE(collective_bound_rule_->MeetsConditions(*signal_provider_));
}

TEST_F(CollectiveBoundRuleTest, MeetsConditionsError) {
  EXPECT_CALL(*mock_tool_data_provider_, GetCollectiveTimeFractionEachStep())
      .WillOnce(Return(absl::InternalError("Test Error")));

  EXPECT_FALSE(collective_bound_rule_->MeetsConditions(*signal_provider_));
}

TEST_F(CollectiveBoundRuleTest, GenerateSuggestionError) {
  EXPECT_CALL(*mock_tool_data_provider_, GetCollectiveTimeFractionEachStep())
      .WillOnce(Return(absl::InternalError("Test Error")));

  EXPECT_THAT(collective_bound_rule_->GenerateSuggestion(*signal_provider_),
              StatusIs(absl::StatusCode::kInternal));
}

}  // namespace
}  // namespace profiler
}  // namespace tensorflow
