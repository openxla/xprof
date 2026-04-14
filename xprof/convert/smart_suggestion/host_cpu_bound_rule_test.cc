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

#include "xprof/convert/smart_suggestion/host_cpu_bound_rule.h"

#include <memory>
#include <optional>
#include <utility>

#include "testing/base/public/gmock.h"
#include "<gtest/gtest.h>"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xprof/convert/smart_suggestion/mock_tool_data_provider.h"
#include "xprof/convert/smart_suggestion/signal_provider.h"
#include "plugin/xprof/protobuf/event_time_fraction_analyzer.pb.h"
#include "plugin/xprof/protobuf/smart_suggestion.pb.h"

namespace tensorflow {
namespace profiler {
namespace {

using ::testing::Eq;
using ::testing::HasSubstr;
using ::testing::Return;
using ::testing::status::IsOkAndHolds;

TEST(HostCPUBoundRuleTest, MeetsConditions) {
  auto mock_tool_data_provider = std::make_unique<MockToolDataProvider>();

  EventTimeFractionAnalyzerResult infeed_result;
  EventTimeFractionPerHost host0;
  host0.set_hostname("host0");
  host0.add_event_time_fractions(0.10);  // 10%
  infeed_result.mutable_host_event_time_fractions()->insert({"host0", host0});

  EventTimeFractionAnalyzerResult enqueue_result;
  EventTimeFractionPerHost host0_enqueue;
  host0_enqueue.set_hostname("host0");
  host0_enqueue.add_event_time_fractions(0.40);  // 40%
  enqueue_result.mutable_host_event_time_fractions()->insert(
      {"host0", host0_enqueue});

  EXPECT_CALL(*mock_tool_data_provider,
              GetEventTimeFractionAnalyzerResult(kInfeedOpName))
      .WillRepeatedly(Return(&infeed_result));
  EXPECT_CALL(*mock_tool_data_provider,
              GetEventTimeFractionAnalyzerResult(kEnqueueDeviceOpName))
      .WillRepeatedly(Return(&enqueue_result));

  SignalProvider signal_provider(std::move(mock_tool_data_provider));
  HostCPUBoundRule rule;

  absl::StatusOr<std::optional<SmartSuggestion>> suggestion =
      rule.Apply(signal_provider);
  EXPECT_THAT(suggestion, IsOkAndHolds(testing::Not(Eq(std::nullopt))));
  EXPECT_EQ((*suggestion)->rule_name(), "HostCPUBoundRule");
  EXPECT_THAT((*suggestion)->suggestion_text(), HasSubstr("host CPU"));
  EXPECT_THAT((*suggestion)->suggestion_text(), HasSubstr("10.0%"));
  EXPECT_THAT((*suggestion)->suggestion_text(), HasSubstr("40.0%"));
  EXPECT_THAT((*suggestion)->suggestion_text(), HasSubstr("host0"));
}

TEST(HostCPUBoundRuleTest, NotHostCPUBound_EnqueueLow) {
  auto mock_tool_data_provider = std::make_unique<MockToolDataProvider>();

  EventTimeFractionAnalyzerResult infeed_result;
  EventTimeFractionPerHost host0;
  host0.set_hostname("host0");
  host0.add_event_time_fractions(0.99);  // 99%
  infeed_result.mutable_host_event_time_fractions()->insert({"host0", host0});

  EventTimeFractionAnalyzerResult enqueue_result;
  EventTimeFractionPerHost host1;
  host1.set_hostname("host1");
  host1.add_event_time_fractions(0.01);  // 1% < 30% threshold
  enqueue_result.mutable_host_event_time_fractions()->insert({"host1", host1});

  EXPECT_CALL(*mock_tool_data_provider,
              GetEventTimeFractionAnalyzerResult(kInfeedOpName))
      .WillRepeatedly(Return(&infeed_result));
  EXPECT_CALL(*mock_tool_data_provider,
              GetEventTimeFractionAnalyzerResult(kEnqueueDeviceOpName))
      .WillRepeatedly(Return(&enqueue_result));

  SignalProvider signal_provider(std::move(mock_tool_data_provider));
  HostCPUBoundRule rule;

  absl::StatusOr<std::optional<SmartSuggestion>> suggestion =
      rule.Apply(signal_provider);
  EXPECT_THAT(suggestion, IsOkAndHolds(Eq(std::nullopt)));
}

TEST(HostCPUBoundRuleTest, NotHostCPUBound_InfeedMissing) {
  auto mock_tool_data_provider = std::make_unique<MockToolDataProvider>();

  EventTimeFractionAnalyzerResult infeed_result;  // Empty

  EventTimeFractionAnalyzerResult enqueue_result;
  EventTimeFractionPerHost host1;
  host1.set_hostname("host1");
  host1.add_event_time_fractions(0.50);  // 50%
  enqueue_result.mutable_host_event_time_fractions()->insert({"host1", host1});

  EXPECT_CALL(*mock_tool_data_provider,
              GetEventTimeFractionAnalyzerResult(kInfeedOpName))
      .WillRepeatedly(Return(&infeed_result));
  EXPECT_CALL(*mock_tool_data_provider,
              GetEventTimeFractionAnalyzerResult(kEnqueueDeviceOpName))
      .WillRepeatedly(Return(&enqueue_result));

  SignalProvider signal_provider(std::move(mock_tool_data_provider));
  HostCPUBoundRule rule;

  absl::StatusOr<std::optional<SmartSuggestion>> suggestion =
      rule.Apply(signal_provider);
  EXPECT_THAT(suggestion, IsOkAndHolds(Eq(std::nullopt)));
}

TEST(HostCPUBoundRuleTest, ErrorFetchingOneSignal) {
  auto mock_tool_data_provider = std::make_unique<MockToolDataProvider>();

  EventTimeFractionAnalyzerResult infeed_result;
  EventTimeFractionPerHost host0;
  host0.set_hostname("host0");
  host0.add_event_time_fractions(0.99);
  infeed_result.mutable_host_event_time_fractions()->insert({"host0", host0});

  EXPECT_CALL(*mock_tool_data_provider,
              GetEventTimeFractionAnalyzerResult(kInfeedOpName))
      .WillRepeatedly(Return(&infeed_result));
  EXPECT_CALL(*mock_tool_data_provider,
              GetEventTimeFractionAnalyzerResult(kEnqueueDeviceOpName))
      .WillRepeatedly(Return(absl::InternalError("Failed")));

  SignalProvider signal_provider(std::move(mock_tool_data_provider));
  HostCPUBoundRule rule;

  absl::StatusOr<std::optional<SmartSuggestion>> suggestion =
      rule.Apply(signal_provider);
  EXPECT_THAT(suggestion, IsOkAndHolds(Eq(std::nullopt)));
}

}  // namespace
}  // namespace profiler
}  // namespace tensorflow
