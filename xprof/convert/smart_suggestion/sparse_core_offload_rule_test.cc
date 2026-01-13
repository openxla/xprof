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

#include <memory>
#include <optional>
#include <utility>

#include "testing/base/public/gmock.h"
#include "<gtest/gtest.h>"
#include "absl/status/statusor.h"
#include "xprof/convert/smart_suggestion/mock_tool_data_provider.h"
#include "xprof/convert/smart_suggestion/signal_provider.h"
#include "plugin/xprof/protobuf/memory_profile.pb.h"
#include "plugin/xprof/protobuf/op_profile.pb.h"
#include "plugin/xprof/protobuf/smart_suggestion.pb.h"

namespace tensorflow {
namespace profiler {
namespace {

using ::testing::Eq;
using ::testing::HasSubstr;
using ::testing::Return;
using ::testing::status::IsOkAndHolds;

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

}  // namespace
}  // namespace profiler
}  // namespace tensorflow
