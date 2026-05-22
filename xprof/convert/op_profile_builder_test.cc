/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#include "xprof/convert/op_profile_builder.h"

#include "testing/base/public/gmock.h"
#include "<gtest/gtest.h>"
#include "plugin/xprof/protobuf/op_metrics.pb.h"
#include "plugin/xprof/protobuf/op_profile.pb.h"

namespace tensorflow {
namespace profiler {
namespace {

TEST(OpProfileBuilderTest, SparseCoreMetricsSubtraction) {
  OpProfileOptions options = {OpProfileGrouping::kByCategory,
                              /*group_by_deduplicated_name=*/true,
                              /*children_per_node=*/10};
  op_profile::Node root;
  OpProfileBuilder builder(options, &root, nullptr);

  // Mock parent TensorCore offload OpMetrics
  OpMetrics parent_metrics;
  parent_metrics.set_name("TensorCoreOp");
  parent_metrics.set_category("fusion");
  parent_metrics.set_time_ps(100);
  parent_metrics.set_self_time_ps(50);  // children_time_ps > 0
  parent_metrics.set_flops(1000);
  parent_metrics.set_model_flops(1000);
  parent_metrics.set_flops_v2(1000.0);
  parent_metrics.set_model_flops_v2(1000.0);
  parent_metrics.set_bytes_accessed(2000);
  parent_metrics.set_occurrences(1);

  // Mock SparseCore child OpMetrics
  OpMetrics* child_metrics =
      parent_metrics.mutable_children()->add_metrics_db();
  child_metrics->set_name("SparseCoreChild");
  child_metrics->set_category("sparse");
  child_metrics->set_time_ps(50);
  child_metrics->set_self_time_ps(50);
  child_metrics->set_flops(600);
  child_metrics->set_model_flops(600);
  child_metrics->set_flops_v2(600.0);
  child_metrics->set_model_flops_v2(600.0);
  child_metrics->set_bytes_accessed(1500);
  child_metrics->set_core_type(OpMetrics_TpuCoreType_SPARSE_CORE);
  child_metrics->set_occurrences(1);

  // Process the mock metrics
  builder.AddOp(parent_metrics);

  // Finalize the profile tree
  builder.Finalize(
      /*peak_gigaflops_per_second_per_core=*/1.0,
      /*peak_mem_gibibytes_per_second_per_core=*/{1.0, 1.0, 1.0, 1.0},
      /*total_time_ps=*/100);

  // Verify tree structure and metric subtraction
  ASSERT_EQ(root.children_size(), 1);
  const op_profile::Node& category_node = root.children(0);
  EXPECT_EQ(category_node.name(), "fusion");

  ASSERT_EQ(category_node.children_size(), 1);
  const op_profile::Node& parent_node = category_node.children(0);
  EXPECT_EQ(parent_node.name(), "TensorCoreOp");

  // Verify the parent node's metrics were reduced by the child's metrics
  EXPECT_EQ(parent_node.metrics().raw_flops(), 400.0);
  EXPECT_EQ(parent_node.metrics().bf16_flops(), 400.0);

  // Verify the child was inserted and retained its original metrics
  ASSERT_EQ(parent_node.children_size(), 1);
  const op_profile::Node& child_node = parent_node.children(0);
  EXPECT_EQ(child_node.name(), "SparseCoreChild");
  EXPECT_EQ(child_node.metrics().raw_flops(), 600.0);
  EXPECT_EQ(child_node.metrics().bf16_flops(), 600.0);
}

}  // namespace
}  // namespace profiler
}  // namespace tensorflow
