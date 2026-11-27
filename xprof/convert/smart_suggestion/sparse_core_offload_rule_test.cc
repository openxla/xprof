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

#include "xprof/convert/smart_suggestion/sparse_core_offload_rule.h"

#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xprof/convert/smart_suggestion/mock_tool_data_provider.h"
#include "xprof/convert/smart_suggestion/signal_provider.h"
#include "plugin/xprof/protobuf/memory_profile.pb.h"
#include "plugin/xprof/protobuf/op_profile.pb.h"
#include "plugin/xprof/protobuf/smart_suggestion.pb.h"

namespace tensorflow {
namespace profiler {
namespace {

using ::testing::DoubleEq;
using ::testing::DoubleNear;
using ::testing::Eq;
using ::testing::HasSubstr;
using ::testing::Return;
using ::testing::status::IsOkAndHolds;
using ::testing::status::StatusIs;

class SparseCoreOffloadRuleTest : public ::testing::Test {
 protected:
  void SetUp() override {
    auto mock_tool_data_provider = std::make_unique<MockToolDataProvider>();
    mock_tool_data_provider_ = mock_tool_data_provider.get();
    signal_provider_ =
        std::make_unique<SignalProvider>(std::move(mock_tool_data_provider));
    sparse_core_offload_rule_ = std::make_unique<SparseCoreOffloadRule>();
  }

  MockToolDataProvider* mock_tool_data_provider_;
  std::unique_ptr<SignalProvider> signal_provider_;
  std::unique_ptr<SparseCoreOffloadRule> sparse_core_offload_rule_;
};

TEST_F(SparseCoreOffloadRuleTest, MeetsConditions) {
  op_profile::Profile op_profile;
  auto* root_node = op_profile.mutable_by_program_exclude_idle();
  root_node->mutable_metrics()->set_raw_time(1000);
  auto* child = root_node->add_children();
  auto* async_done_node = child->add_children();
  async_done_node->set_name("async-done");
  auto* all_reduce_node = async_done_node->add_children();
  all_reduce_node->mutable_metrics()->set_raw_time(110);  // 11% async-done time
  all_reduce_node->set_name("all-reduce.0");

  MemoryProfile memory_profile;
  PerAllocatorMemoryProfile per_allocator_profile;
  MemoryProfileSummary memory_profile_summary;
  memory_profile_summary.set_memory_capacity(200);
  MemoryAggregationStats memory_aggregation_stats;
  memory_aggregation_stats.set_peak_bytes_in_use(80);  // 40% memory utilization
  *memory_profile_summary.mutable_peak_stats() = memory_aggregation_stats;
  *per_allocator_profile.mutable_profile_summary() = memory_profile_summary;
  memory_profile.mutable_memory_profile_per_allocator()->insert(
      {"0", per_allocator_profile});

  EXPECT_CALL(*mock_tool_data_provider_, GetOpProfile())
      .WillRepeatedly(Return(&op_profile));
  EXPECT_CALL(*mock_tool_data_provider_, GetMemoryProfile())
      .WillRepeatedly(Return(&memory_profile));

  EXPECT_TRUE(sparse_core_offload_rule_->MeetsConditions(*signal_provider_));
}

TEST_F(SparseCoreOffloadRuleTest, AsyncDoneTooLow) {
  op_profile::Profile op_profile;
  auto* root_node = op_profile.mutable_by_program_exclude_idle();
  root_node->mutable_metrics()->set_raw_time(1000);
  auto* child = root_node->add_children();
  auto* async_done_node = child->add_children();
  async_done_node->set_name("async-done");
  auto* all_reduce_node = async_done_node->add_children();
  all_reduce_node->mutable_metrics()->set_raw_time(90);  // 9% async-done time
  all_reduce_node->set_name("all-reduce.0");

  MemoryProfile memory_profile;
  PerAllocatorMemoryProfile per_allocator_profile;
  MemoryProfileSummary memory_profile_summary;
  memory_profile_summary.set_memory_capacity(200);
  MemoryAggregationStats memory_aggregation_stats;
  memory_aggregation_stats.set_peak_bytes_in_use(80);  // 40% memory utilization
  *memory_profile_summary.mutable_peak_stats() = memory_aggregation_stats;
  *per_allocator_profile.mutable_profile_summary() = memory_profile_summary;
  memory_profile.mutable_memory_profile_per_allocator()->insert(
      {"0", per_allocator_profile});

  EXPECT_CALL(*mock_tool_data_provider_, GetOpProfile())
      .WillRepeatedly(Return(&op_profile));
  EXPECT_CALL(*mock_tool_data_provider_, GetMemoryProfile())
      .WillRepeatedly(Return(&memory_profile));

  EXPECT_FALSE(sparse_core_offload_rule_->MeetsConditions(*signal_provider_));
}

TEST_F(SparseCoreOffloadRuleTest, MemoryUtilizationTooHigh) {
  op_profile::Profile op_profile;
  auto* root_node = op_profile.mutable_by_program_exclude_idle();
  root_node->mutable_metrics()->set_raw_time(1000);
  auto* child = root_node->add_children();
  auto* async_done_node = child->add_children();
  async_done_node->set_name("async-done");
  auto* all_reduce_node = async_done_node->add_children();
  all_reduce_node->mutable_metrics()->set_raw_time(110);  // 11% async-done time
  all_reduce_node->set_name("all-reduce.0");

  MemoryProfile memory_profile;
  PerAllocatorMemoryProfile per_allocator_profile;
  MemoryProfileSummary memory_profile_summary;
  memory_profile_summary.set_memory_capacity(200);
  MemoryAggregationStats memory_aggregation_stats;
  // 60% memory utilization
  memory_aggregation_stats.set_peak_bytes_in_use(120);
  *memory_profile_summary.mutable_peak_stats() = memory_aggregation_stats;
  *per_allocator_profile.mutable_profile_summary() = memory_profile_summary;
  memory_profile.mutable_memory_profile_per_allocator()->insert(
      {"0", per_allocator_profile});

  EXPECT_CALL(*mock_tool_data_provider_, GetOpProfile())
      .WillRepeatedly(Return(&op_profile));
  EXPECT_CALL(*mock_tool_data_provider_, GetMemoryProfile())
      .WillRepeatedly(Return(&memory_profile));

  EXPECT_FALSE(sparse_core_offload_rule_->MeetsConditions(*signal_provider_));
}

TEST_F(SparseCoreOffloadRuleTest, GenerateSuggestion) {
  op_profile::Profile op_profile;
  auto* root_node = op_profile.mutable_by_program_exclude_idle();
  root_node->mutable_metrics()->set_raw_time(1000);
  auto* child = root_node->add_children();
  auto* async_done_node = child->add_children();
  async_done_node->set_name("async-done");
  auto* all_reduce_node = async_done_node->add_children();
  all_reduce_node->mutable_metrics()->set_raw_time(110);  // 11% async-done time
  all_reduce_node->set_name("all-reduce.0");

  MemoryProfile memory_profile;
  PerAllocatorMemoryProfile per_allocator_profile;
  MemoryProfileSummary memory_profile_summary;
  memory_profile_summary.set_memory_capacity(200);
  MemoryAggregationStats memory_aggregation_stats;
  memory_aggregation_stats.set_peak_bytes_in_use(80);  // 40% memory utilization
  *memory_profile_summary.mutable_peak_stats() = memory_aggregation_stats;
  *per_allocator_profile.mutable_profile_summary() = memory_profile_summary;
  memory_profile.mutable_memory_profile_per_allocator()->insert(
      {"0", per_allocator_profile});

  EXPECT_CALL(*mock_tool_data_provider_, GetOpProfile())
      .WillRepeatedly(Return(&op_profile));
  EXPECT_CALL(*mock_tool_data_provider_, GetMemoryProfile())
      .WillRepeatedly(Return(&memory_profile));

  absl::StatusOr<std::optional<SmartSuggestion>> suggestion =
      sparse_core_offload_rule_->Apply(*signal_provider_);
  EXPECT_THAT(suggestion, IsOkAndHolds(testing::Not(Eq(std::nullopt))));
  EXPECT_EQ((*suggestion)->rule_name(), "SparseCoreOffloadRule");
  EXPECT_THAT((*suggestion)->suggestion_text(),
              HasSubstr("11.0%</b> of time is spent on async-done operations"
                        " with low memory utilization of <b>40.0%</b>"));
}

TEST_F(SparseCoreOffloadRuleTest, CountsRaggedAllToAllAndReduceGather) {
  op_profile::Profile op_profile;
  auto* root_node = op_profile.mutable_by_program_exclude_idle();
  root_node->mutable_metrics()->set_raw_time(1000);
  auto* child = root_node->add_children();
  auto* async_done_node = child->add_children();
  async_done_node->set_name("async-done");
  auto* ragged_all_to_all_node = async_done_node->add_children();
  ragged_all_to_all_node->set_name("ragged-all-to-all.1");
  ragged_all_to_all_node->mutable_metrics()->set_raw_time(100);
  auto* reduce_gather_node = async_done_node->add_children();
  reduce_gather_node->set_name("reduce-gather.2");
  reduce_gather_node->mutable_metrics()->set_raw_time(50);
  auto* other_node = async_done_node->add_children();
  other_node->set_name("fusion.3");
  other_node->mutable_metrics()->set_raw_time(200);

  EXPECT_CALL(*mock_tool_data_provider_, GetOpProfile())
      .WillRepeatedly(Return(&op_profile));

  // Only the ragged-all-to-all and reduce-gather children count: 15%.
  EXPECT_THAT(signal_provider_->GetAsyncDoneOpPercent(),
              IsOkAndHolds(DoubleNear(15.0, 1e-9)));
}

TEST_F(SparseCoreOffloadRuleTest,
       GetAsyncDoneOpPercentMatchesAllCollectivePrefixes) {
  for (const auto* op_name :
       {"all-gather.1", "all-reduce.1", "all-to-all.1", "ragged-all-to-all.1",
        "reduce-gather.1", "reduce-scatter.1", "%all-reduce-done.2",
        "all-reduce.3 and its duplicate(s)", "all-reduce and its duplicate(s)",
        "all-reduce-done and its duplicate(s)"}) {
    op_profile::Profile op_profile;
    auto* root_node = op_profile.mutable_by_program_exclude_idle();
    root_node->mutable_metrics()->set_raw_time(1000);
    auto* child = root_node->add_children();
    auto* async_done_node = child->add_children();
    async_done_node->set_name("async-done (OFFLOAD_COLLECTIVE)");
    auto* collective_node = async_done_node->add_children();
    collective_node->mutable_metrics()->set_raw_time(150);
    collective_node->set_name(op_name);

    EXPECT_CALL(*mock_tool_data_provider_, GetOpProfile())
        .WillOnce(Return(&op_profile));

    EXPECT_THAT(signal_provider_->GetAsyncDoneOpPercent(),
                IsOkAndHolds(DoubleEq(15.0)))
        << "Failed for op_name: " << op_name;
  }
}

TEST_F(SparseCoreOffloadRuleTest, GetAsyncDoneOpPercentSumsAllPrograms) {
  op_profile::Profile op_profile;
  auto* root_node = op_profile.mutable_by_program_exclude_idle();
  root_node->mutable_metrics()->set_raw_time(1000);
  auto* first_async_done_node = root_node->add_children()->add_children();
  first_async_done_node->set_name("async-done");
  auto* first_collective_node = first_async_done_node->add_children();
  first_collective_node->set_name("all-reduce.1");
  first_collective_node->mutable_metrics()->set_raw_time(100);
  auto* second_async_done_node = root_node->add_children()->add_children();
  second_async_done_node->set_name("async-done");
  auto* second_collective_node = second_async_done_node->add_children();
  second_collective_node->set_name("all-gather.2");
  second_collective_node->mutable_metrics()->set_raw_time(50);

  EXPECT_CALL(*mock_tool_data_provider_, GetOpProfile())
      .WillOnce(Return(&op_profile));

  EXPECT_THAT(signal_provider_->GetAsyncDoneOpPercent(),
              IsOkAndHolds(DoubleEq(15.0)));
}

TEST_F(SparseCoreOffloadRuleTest,
       GetAsyncDoneOpPercentIgnoresNonMatchingCategoriesAndOps) {
  op_profile::Profile op_profile;
  auto* root_node = op_profile.mutable_by_program_exclude_idle();
  root_node->mutable_metrics()->set_raw_time(1000);
  auto* program_node = root_node->add_children();
  auto* other_category_node = program_node->add_children();
  other_category_node->set_name("async-done-other");
  auto* hidden_collective_node = other_category_node->add_children();
  hidden_collective_node->set_name("all-reduce.1");
  hidden_collective_node->mutable_metrics()->set_raw_time(100);
  auto* async_done_node = program_node->add_children();
  async_done_node->set_name("async-done");
  for (const auto* op_name :
       {"", "-done", "%-done", "fusion-done.1", "all-reduce-start.2"}) {
    auto* op_node = async_done_node->add_children();
    op_node->set_name(op_name);
    op_node->mutable_metrics()->set_raw_time(100);
  }

  EXPECT_CALL(*mock_tool_data_provider_, GetOpProfile())
      .WillOnce(Return(&op_profile));

  EXPECT_THAT(signal_provider_->GetAsyncDoneOpPercent(),
              IsOkAndHolds(DoubleEq(0.0)));
}

TEST_F(SparseCoreOffloadRuleTest,
       GetAsyncDoneOpPercentReturnsZeroWithoutPositiveRawTime) {
  for (double raw_time :
       {0.0, -1.0, std::numeric_limits<double>::quiet_NaN()}) {
    op_profile::Profile op_profile;
    auto* root_node = op_profile.mutable_by_program_exclude_idle();
    root_node->mutable_metrics()->set_raw_time(raw_time);
    auto* async_done_node = root_node->add_children()->add_children();
    async_done_node->set_name("async-done");
    auto* collective_node = async_done_node->add_children();
    collective_node->set_name("all-reduce.1");
    collective_node->mutable_metrics()->set_raw_time(100);

    EXPECT_CALL(*mock_tool_data_provider_, GetOpProfile())
        .WillOnce(Return(&op_profile));

    EXPECT_THAT(signal_provider_->GetAsyncDoneOpPercent(),
                IsOkAndHolds(DoubleEq(0.0)))
        << "Failed for raw_time: " << raw_time;
  }
}

void AddAllocator(absl::string_view memory_id, int64_t memory_capacity,
                  int64_t peak_bytes_in_use, MemoryProfile& memory_profile) {
  PerAllocatorMemoryProfile per_allocator_profile;
  MemoryProfileSummary* summary =
      per_allocator_profile.mutable_profile_summary();
  summary->set_memory_capacity(memory_capacity);
  summary->mutable_peak_stats()->set_peak_bytes_in_use(peak_bytes_in_use);
  memory_profile.mutable_memory_profile_per_allocator()->insert(
      {std::string(memory_id), per_allocator_profile});
}

TEST_F(SparseCoreOffloadRuleTest,
       GetPeakMemoryUtilizationIsUnavailableWithoutDeviceCapacity) {
  MemoryProfile empty_profile;
  EXPECT_CALL(*mock_tool_data_provider_, GetMemoryProfile())
      .WillOnce(Return(&empty_profile));
  EXPECT_THAT(signal_provider_->GetPeakMemoryUtilization(),
              StatusIs(absl::StatusCode::kUnavailable));

  MemoryProfile host_only_profile;
  AddAllocator("Host", 1000, 100, host_only_profile);
  AddAllocator("cpu_0", 1000, 100, host_only_profile);
  AddAllocator("hbm_0", 0, 100, host_only_profile);
  EXPECT_CALL(*mock_tool_data_provider_, GetMemoryProfile())
      .WillOnce(Return(&host_only_profile));
  EXPECT_THAT(signal_provider_->GetPeakMemoryUtilization(),
              StatusIs(absl::StatusCode::kUnavailable));
}

TEST_F(SparseCoreOffloadRuleTest,
       GetPeakMemoryUtilizationSkipsHostAllocatorsIgnoringCase) {
  MemoryProfile memory_profile;
  AddAllocator("hbm_0", 200, 100, memory_profile);
  for (const auto* host_id :
       {"host_pageable", "Host", "HOST", "cpu_0", "Cpu_1", "CPU"}) {
    AddAllocator(host_id, 10000, 100, memory_profile);
  }

  EXPECT_CALL(*mock_tool_data_provider_, GetMemoryProfile())
      .WillOnce(Return(&memory_profile));
  EXPECT_THAT(signal_provider_->GetPeakMemoryUtilization(),
              IsOkAndHolds(DoubleEq(50.0)));
}

TEST_F(SparseCoreOffloadRuleTest,
       GetPeakMemoryUtilizationSkipsZeroCapacityAndClampsNegativePeak) {
  MemoryProfile memory_profile;
  AddAllocator("hbm_0", 200, 100, memory_profile);
  AddAllocator("hbm_1", 0, 400, memory_profile);
  AddAllocator("hbm_2", 200, -100, memory_profile);

  EXPECT_CALL(*mock_tool_data_provider_, GetMemoryProfile())
      .WillOnce(Return(&memory_profile));
  // Only hbm_0 and hbm_2 count, with the negative peak clamped: 100 / 400.
  EXPECT_THAT(signal_provider_->GetPeakMemoryUtilization(),
              IsOkAndHolds(DoubleEq(25.0)));
}

TEST_F(SparseCoreOffloadRuleTest,
       DoesNotMeetConditionsWithoutDeviceMemoryProfile) {
  op_profile::Profile op_profile;
  auto* root_node = op_profile.mutable_by_program_exclude_idle();
  root_node->mutable_metrics()->set_raw_time(1000);
  auto* async_done_node = root_node->add_children()->add_children();
  async_done_node->set_name("async-done");
  auto* all_reduce_node = async_done_node->add_children();
  all_reduce_node->set_name("all-reduce.0");
  all_reduce_node->mutable_metrics()->set_raw_time(200);  // 20% async-done.
  MemoryProfile empty_profile;

  EXPECT_CALL(*mock_tool_data_provider_, GetOpProfile())
      .WillRepeatedly(Return(&op_profile));
  EXPECT_CALL(*mock_tool_data_provider_, GetMemoryProfile())
      .WillRepeatedly(Return(&empty_profile));

  EXPECT_FALSE(sparse_core_offload_rule_->MeetsConditions(*signal_provider_));
}

}  // namespace
}  // namespace profiler
}  // namespace tensorflow
