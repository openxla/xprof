/* Copyright 2026 The OpenXLA Authors. All Rights Reserved.

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

#include "xprof/convert/op_metrics_db_combiner.h"

#include <cstdint>
#include <string>

#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "plugin/xprof/protobuf/op_metrics.pb.h"

namespace tensorflow {
namespace profiler {
namespace {

constexpr uint64_t kModuleId = 1;
constexpr absl::string_view kParentName = "scatter_offload_async_start.95";

// Adds a SparseCore child op to `parent`.
void AddSparseCoreChild(absl::string_view name, uint64_t occurrences,
                        uint64_t time_ps, OpMetrics* parent) {
  OpMetricsDb* children = parent->mutable_children();
  children->set_total_time_ps(children->total_time_ps() + time_ps);
  OpMetrics* child = children->add_metrics_db();
  child->set_hlo_module_id(kModuleId);
  child->set_name(std::string(name));
  child->set_category("async-start");
  child->set_core_type(OpMetrics::SPARSE_CORE);
  child->set_occurrences(occurrences);
  child->set_time_ps(time_ps);
  child->set_self_time_ps(time_ps);
  child->set_num_cores(1);
}

// Returns one core's OpMetricsDb with a single TensorCore offload op.
OpMetricsDb MakeCoreDb() {
  OpMetricsDb db;
  db.set_total_time_ps(1000000000);
  OpMetrics* parent = db.add_metrics_db();
  parent->set_hlo_module_id(kModuleId);
  parent->set_name(std::string(kParentName));
  parent->set_category("async-start");
  parent->set_core_type(OpMetrics::TENSOR_CORE);
  parent->set_occurrences(36);
  parent->set_time_ps(1000);
  parent->set_self_time_ps(1000);
  parent->set_num_cores(1);
  return db;
}

TEST(OpMetricsDbCombinerTest, ChildrenAreSummedAcrossCores) {
  constexpr int kNumCores = 4;
  constexpr uint64_t kChildTimePs = 803500000;
  OpMetricsDb combined;
  OpMetricsDbCombiner combiner(&combined);
  for (int core = 0; core < kNumCores; ++core) {
    OpMetricsDb core_db = MakeCoreDb();
    AddSparseCoreChild("scatter_offload_custom_fusion.155", /*occurrences=*/36,
                       kChildTimePs, core_db.mutable_metrics_db(0));
    combiner.Combine(core_db);
  }

  ASSERT_EQ(combined.metrics_db_size(), 1);
  const OpMetrics& parent = combined.metrics_db(0);
  EXPECT_EQ(parent.occurrences(), 36 * kNumCores);
  ASSERT_EQ(parent.children().metrics_db_size(), 1);
  EXPECT_EQ(parent.children().total_time_ps(), kChildTimePs * kNumCores);
  const OpMetrics& child = parent.children().metrics_db(0);
  EXPECT_EQ(child.name(), "scatter_offload_custom_fusion.155");
  EXPECT_EQ(child.core_type(), OpMetrics::SPARSE_CORE);
  EXPECT_EQ(child.occurrences(), 36 * kNumCores);
  EXPECT_EQ(child.time_ps(), kChildTimePs * kNumCores);
  EXPECT_EQ(child.self_time_ps(), kChildTimePs * kNumCores);
  EXPECT_EQ(child.num_cores(), kNumCores);
}

TEST(OpMetricsDbCombinerTest, ChildrenWithDifferentNamesAreKeptSeparate) {
  OpMetricsDb combined;
  OpMetricsDbCombiner combiner(&combined);
  OpMetricsDb core0 = MakeCoreDb();
  AddSparseCoreChild("child_a", /*occurrences=*/2, /*time_ps=*/100,
                     core0.mutable_metrics_db(0));
  OpMetricsDb core1 = MakeCoreDb();
  AddSparseCoreChild("child_a", /*occurrences=*/3, /*time_ps=*/200,
                     core1.mutable_metrics_db(0));
  AddSparseCoreChild("child_b", /*occurrences=*/5, /*time_ps=*/400,
                     core1.mutable_metrics_db(0));
  combiner.Combine(core0);
  combiner.Combine(core1);

  ASSERT_EQ(combined.metrics_db_size(), 1);
  const OpMetricsDb& children = combined.metrics_db(0).children();
  ASSERT_EQ(children.metrics_db_size(), 2);
  EXPECT_EQ(children.total_time_ps(), 700);
  EXPECT_FALSE(children.has_precision_stats());
  for (const OpMetrics& child : children.metrics_db()) {
    if (child.name() == "child_a") {
      EXPECT_EQ(child.occurrences(), 5);
      EXPECT_EQ(child.time_ps(), 300);
      EXPECT_EQ(child.num_cores(), 2);
    } else {
      EXPECT_EQ(child.name(), "child_b");
      EXPECT_EQ(child.occurrences(), 5);
      EXPECT_EQ(child.time_ps(), 400);
      EXPECT_EQ(child.num_cores(), 1);
    }
  }
}

TEST(OpMetricsDbCombinerTest, SingleSourceChildrenAreCopiedUnchanged) {
  OpMetricsDb core_db = MakeCoreDb();
  AddSparseCoreChild("child_a", /*occurrences=*/2, /*time_ps=*/100,
                     core_db.mutable_metrics_db(0));
  OpMetricsDb combined;
  OpMetricsDbCombiner combiner(&combined);
  combiner.Combine(core_db, /*update_num_cores=*/false);

  ASSERT_EQ(combined.metrics_db_size(), 1);
  const OpMetricsDb& children = combined.metrics_db(0).children();
  ASSERT_EQ(children.metrics_db_size(), 1);
  EXPECT_EQ(children.total_time_ps(), 100);
  EXPECT_EQ(children.metrics_db(0).occurrences(), 2);
  EXPECT_EQ(children.metrics_db(0).time_ps(), 100);
  EXPECT_EQ(children.metrics_db(0).num_cores(), 1);
  EXPECT_FALSE(children.has_precision_stats());
}

TEST(OpMetricsDbCombinerTest, ParentWithoutChildrenHasNoChildren) {
  OpMetricsDb combined;
  OpMetricsDbCombiner combiner(&combined);
  combiner.Combine(MakeCoreDb());
  combiner.Combine(MakeCoreDb());

  ASSERT_EQ(combined.metrics_db_size(), 1);
  EXPECT_FALSE(combined.metrics_db(0).has_children());
}

}  // namespace
}  // namespace profiler
}  // namespace tensorflow
