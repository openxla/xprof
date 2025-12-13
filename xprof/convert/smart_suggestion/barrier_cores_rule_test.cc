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
#include "plugin/xprof/protobuf/event_time_fraction_analyzer.pb.h"
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
  EventTimeFractionPerChip chip0;
  chip0.set_id("chip0");
  chip0.add_event_time_fractions(0.15);
  chip0.add_event_time_fractions(0.25);
  result.mutable_chip_event_time_fractions()->insert({"chip0", chip0});
  EventTimeFractionPerChip chip1;
  chip1.set_id("chip1");
  chip1.add_event_time_fractions(0.05);
  chip1.add_event_time_fractions(0.35);
  result.mutable_chip_event_time_fractions()->insert({"chip1", chip1});
  // Average is (0.15+0.25+0.05+0.35)/4 = 0.2, which is 20%. This is > 10%.
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
  EventTimeFractionPerChip chip0;
  chip0.set_id("chip0");
  chip0.add_event_time_fractions(0.01);
  chip0.add_event_time_fractions(0.02);
  result.mutable_chip_event_time_fractions()->insert({"chip0", chip0});
  EventTimeFractionPerChip chip1;
  chip1.set_id("chip1");
  chip1.add_event_time_fractions(0.05);
  chip1.add_event_time_fractions(0.25);
  result.mutable_chip_event_time_fractions()->insert({"chip1", chip1});
  // Average is (0.01+0.02+0.05+0.25)/4 = 0.015, which is 1.5%. This is < 10%.
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

TEST(BarrierCoresRuleTest, MeetsConditionsWithStragglers) {
  auto mock_tool_data_provider = std::make_unique<MockToolDataProvider>();
  EventTimeFractionAnalyzerResult result;
  // Host 0: (50+50)/2 = 50%
  EventTimeFractionPerHost host0;
  host0.set_hostname("host0");
  host0.add_event_time_fractions(0.50);
  host0.add_event_time_fractions(0.50);
  result.mutable_host_event_time_fractions()->insert({"host0", host0});

  // Host 1: (50+50)/2 = 50%
  EventTimeFractionPerHost host1;
  host1.set_hostname("host1");
  host1.add_event_time_fractions(0.50);
  host1.add_event_time_fractions(0.50);
  result.mutable_host_event_time_fractions()->insert({"host1", host1});

  // Host 2: (50+50)/2 = 50%
  EventTimeFractionPerHost host2;
  host2.set_hostname("host2");
  host2.add_event_time_fractions(0.50);
  host2.add_event_time_fractions(0.50);
  result.mutable_host_event_time_fractions()->insert({"host2", host2});

  // Host 3: (90+90)/2 = 10% (Straggler)
  EventTimeFractionPerHost host3;
  host3.set_hostname("host3");
  host3.add_event_time_fractions(0.10);
  host3.add_event_time_fractions(0.10);
  result.mutable_host_event_time_fractions()->insert({"host3", host3});

  EventTimeFractionPerChip chip0;
  chip0.set_id("chip0");
  // Assuming each host has 1 chip so all fractions go to chip0 at chip-level.
  for (int i = 0; i < 6; ++i) {
    chip0.add_event_time_fractions(0.50);
  }
  for (int j = 0; j < 2; ++j) {
    chip0.add_event_time_fractions(0.10);
  }
  result.mutable_chip_event_time_fractions()->insert({"chip0", chip0});

  EXPECT_CALL(*mock_tool_data_provider,
              GetEventTimeFractionAnalyzerResult(kSpecialOpName))
      .WillRepeatedly(Return(&result));

  SignalProvider signal_provider(std::move(mock_tool_data_provider));
  BarrierCoresRule rule;

  absl::StatusOr<std::optional<SmartSuggestion>> suggestion =
      rule.Apply(signal_provider);
  EXPECT_THAT(suggestion, IsOkAndHolds(testing::Not(Eq(std::nullopt))));
  EXPECT_THAT((*suggestion)->suggestion_text(),
              HasSubstr("<li>Host <b>host3</b> average barrier-cores time "
                        "fraction: <b>10.0%</b></li>"));
  EXPECT_THAT((*suggestion)->suggestion_text(),
              HasSubstr("Investigate Stragglers"));
}

}  // namespace
}  // namespace profiler
}  // namespace tensorflow
