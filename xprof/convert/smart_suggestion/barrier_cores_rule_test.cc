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

#include "xprof/convert/smart_suggestion/barrier_cores_rule.h"

#include <memory>
#include <optional>
#include <utility>

#include "testing/base/public/gmock.h"
#include "<gtest/gtest.h>"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xprof/convert/smart_suggestion/mock_tool_data_provider.h"
#include "xprof/convert/smart_suggestion/signal_provider.h"
#include "plugin/xprof/protobuf/smart_suggestion.pb.h"

namespace tensorflow {
namespace profiler {
namespace {

using ::testing::Eq;
using ::testing::HasSubstr;
using ::testing::Return;
using ::testing::status::IsOkAndHolds;

TEST(BarrierCoresRuleTest, MeetsConditions) {
  auto mock_tool_data_provider = std::make_unique<MockToolDataProvider>();
  EventTimeFractionAnalyzerResult result;
  result.add_event_time_fractions(0.15);
  result.add_event_time_fractions(0.25);
  // Average is (0.15+0.25)/2 = 0.2, which is 20%. This is > 10%.
  EXPECT_CALL(*mock_tool_data_provider,
              GetEventTimeFractionAnalyzerResult(kSpecialOpName))
      .WillRepeatedly(Return(&result));

  SignalProvider signal_provider(std::move(mock_tool_data_provider));
  BarrierCoresRule rule;

  absl::StatusOr<std::optional<SmartSuggestion>> suggestion =
      rule.Apply(signal_provider);
  EXPECT_THAT(suggestion, IsOkAndHolds(testing::Not(Eq(std::nullopt))));
  EXPECT_EQ((*suggestion)->rule_name(), "BarrierCoresRule");
  EXPECT_THAT((*suggestion)->suggestion_text(),
     HasSubstr("20.0% of each step time"));
}

TEST(BarrierCoresRuleTest, NotSpecialOpBound) {
  auto mock_tool_data_provider = std::make_unique<MockToolDataProvider>();
  EventTimeFractionAnalyzerResult result;
  result.add_event_time_fractions(0.01);
  result.add_event_time_fractions(0.02);
  // Average is (0.01+0.02)/2 = 0.015, which is 1.5%. This is < 10%.
  EXPECT_CALL(*mock_tool_data_provider,
              GetEventTimeFractionAnalyzerResult(kSpecialOpName))
      .WillRepeatedly(Return(&result));

  SignalProvider signal_provider(std::move(mock_tool_data_provider));
  BarrierCoresRule rule;

  absl::StatusOr<std::optional<SmartSuggestion>> suggestion =
      rule.Apply(signal_provider);
  EXPECT_THAT(suggestion, IsOkAndHolds(Eq(std::nullopt)));
}

TEST(BarrierCoresRuleTest, ErrorFetchingPercentile) {
  auto mock_tool_data_provider = std::make_unique<MockToolDataProvider>();
  EXPECT_CALL(*mock_tool_data_provider,
              GetEventTimeFractionAnalyzerResult(kSpecialOpName))
      .WillRepeatedly(Return(absl::InternalError("Failed to get percentile")));

  SignalProvider signal_provider(std::move(mock_tool_data_provider));
  BarrierCoresRule rule;

  absl::StatusOr<std::optional<SmartSuggestion>> suggestion =
      rule.Apply(signal_provider);
  EXPECT_THAT(suggestion, IsOkAndHolds(Eq(std::nullopt)));
}

}  // namespace
}  // namespace profiler
}  // namespace tensorflow
