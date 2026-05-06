/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

#include "xprof/convert/flat_op_metrics_db_combiner.h"

#include <cstdint>
#include <string>

#include "net/proto2/contrib/parse_proto/parse_text_proto.h"
#include "<gtest/gtest.h>"
#include "absl/strings/str_cat.h"
#include "plugin/xprof/protobuf/flat_op_metrics.pb.h"

namespace tensorflow {
namespace profiler {
namespace {

using ::google::protobuf::contrib::parse_proto::ParseTextProtoOrDie;

class FlatOpMetricsDbCombinerTest : public ::testing::Test {
 protected:
  FlatOpMetricsDb dst_;
};

// -----------------------------------------------------------------------------
// Test Case 1: Combine Empty Databases
// -----------------------------------------------------------------------------
TEST_F(FlatOpMetricsDbCombinerTest, CombineEmptyDb) {
  FlatOpMetricsDb src;
  FlatOpMetricsDbCombiner combiner(&dst_);
  combiner.Combine(src);

  EXPECT_EQ(dst_.op_instances_size(), 0);
  EXPECT_EQ(dst_.total_time_ps(), 0);
}

// -----------------------------------------------------------------------------
// Test Case 2: Combine Non-Overlapping Databases
// -----------------------------------------------------------------------------
TEST_F(FlatOpMetricsDbCombinerTest, CombineNonOverlapping) {
  FlatOpMetricsDb src = ParseTextProtoOrDie(R"pb(
    op_instances {
      hlo_module_id: 1
      hlo_name: "op1"
      occurrences: 1
      time_ps: 100
    }
  )pb");

  dst_ = ParseTextProtoOrDie(R"pb(
    op_instances {
      hlo_module_id: 2
      hlo_name: "op2"
      occurrences: 1
      time_ps: 200
    }
  )pb");

  FlatOpMetricsDbCombiner combiner(&dst_);
  combiner.Combine(src);

  EXPECT_EQ(dst_.op_instances_size(), 2);

  // Verify op1
  const auto* op1 = &dst_.op_instances(1);  // Assuming added at the end
  if (op1->hlo_name() != "op1") op1 = &dst_.op_instances(0);
  EXPECT_EQ(op1->hlo_name(), "op1");
  EXPECT_DOUBLE_EQ(op1->time_ps(), 100);

  // Verify op2
  const auto* op2 = &dst_.op_instances(0);
  if (op2->hlo_name() != "op2") op2 = &dst_.op_instances(1);
  EXPECT_EQ(op2->hlo_name(), "op2");
  EXPECT_DOUBLE_EQ(op2->time_ps(), 200);
}

// -----------------------------------------------------------------------------
// Test Case 3: Combine Overlapping Databases
// -----------------------------------------------------------------------------
TEST_F(FlatOpMetricsDbCombinerTest, CombineOverlapping) {
  FlatOpMetricsDb src = ParseTextProtoOrDie(R"pb(
    op_instances {
      hlo_module_id: 1
      hlo_name: "op1"
      occurrences: 2
      time_ps: 150
      self_time_ps: 100
      flops: 1000
      bytes_accessed: 500
    }
  )pb");

  dst_ = ParseTextProtoOrDie(R"pb(
    op_instances {
      hlo_module_id: 1
      hlo_name: "op1"
      occurrences: 1
      time_ps: 100
      self_time_ps: 50
      flops: 500
      bytes_accessed: 250
    }
  )pb");

  FlatOpMetricsDbCombiner combiner(&dst_);
  combiner.Combine(src);

  ASSERT_EQ(dst_.op_instances_size(), 1);
  const auto& metric = dst_.op_instances(0);
  EXPECT_EQ(metric.hlo_name(), "op1");
  EXPECT_EQ(metric.occurrences(), 3);
  EXPECT_DOUBLE_EQ(metric.time_ps(), 250);
  EXPECT_DOUBLE_EQ(metric.self_time_ps(), 150);
  EXPECT_EQ(metric.flops(), 1500);
  EXPECT_DOUBLE_EQ(metric.bytes_accessed(), 750);
}

// -----------------------------------------------------------------------------
// Test Case 4: Combine Memory Breakdown
// -----------------------------------------------------------------------------
TEST_F(FlatOpMetricsDbCombinerTest, CombineMemoryBreakdown) {
  FlatOpMetricsDb src = ParseTextProtoOrDie(R"pb(
    op_instances {
      hlo_module_id: 1
      hlo_name: "op1"
      occurrences: 1
      memory_accessed_breakdown {
        memory_space: 1
        operation_type: READ
        bytes_accessed: 100
      }
      memory_accessed_breakdown {
        memory_space: 2
        operation_type: WRITE
        bytes_accessed: 200
      }
    }
  )pb");

  dst_ = ParseTextProtoOrDie(R"pb(
    op_instances {
      hlo_module_id: 1
      hlo_name: "op1"
      occurrences: 1
      memory_accessed_breakdown {
        memory_space: 1
        operation_type: READ
        bytes_accessed: 50
      }
    }
  )pb");

  FlatOpMetricsDbCombiner combiner(&dst_);
  combiner.Combine(src);

  ASSERT_EQ(dst_.op_instances_size(), 1);
  const auto& metric = dst_.op_instances(0);
  ASSERT_EQ(metric.memory_accessed_breakdown_size(), 2);

  auto find_breakdown = [&](uint64_t space,
                            FlatOpMetrics::MemoryAccessed::OperationType type) {
    for (const auto& b : metric.memory_accessed_breakdown()) {
      if (b.memory_space() == space && b.operation_type() == type) {
        return b.bytes_accessed();
      }
    }
    return uint64_t{0};
  };

  EXPECT_EQ(find_breakdown(1, FlatOpMetrics::MemoryAccessed::READ), 150);
  EXPECT_EQ(find_breakdown(2, FlatOpMetrics::MemoryAccessed::WRITE), 200);
}

// -----------------------------------------------------------------------------
// Test Case 5: Test Update Num Cores
// -----------------------------------------------------------------------------
TEST_F(FlatOpMetricsDbCombinerTest, TestUpdateNumCores) {
  FlatOpMetricsDb src = ParseTextProtoOrDie(R"pb(
    op_instances {
      hlo_module_id: 1
      hlo_name: "op1"
      occurrences: 1
      num_cores: 2
    }
  )pb");

  dst_ = ParseTextProtoOrDie(R"pb(
    op_instances {
      hlo_module_id: 1
      hlo_name: "op1"
      occurrences: 1
      num_cores: 1
    }
  )pb");

  // Test with update_num_cores = true
  {
    FlatOpMetricsDb dst_clone = dst_;
    FlatOpMetricsDbCombiner combiner(&dst_clone);
    combiner.Combine(src, /*update_num_cores=*/true);
    EXPECT_EQ(dst_clone.op_instances(0).num_cores(), 3);
  }

  // Test with update_num_cores = false
  {
    FlatOpMetricsDb dst_clone = dst_;
    FlatOpMetricsDbCombiner combiner(&dst_clone);
    combiner.Combine(src, /*update_num_cores=*/false);
    // Should not change from dst
    EXPECT_EQ(dst_clone.op_instances(0).num_cores(), 1);
  }
}

// -----------------------------------------------------------------------------
// Test Case 6: New Op Num Cores (Fixes double counting)
// -----------------------------------------------------------------------------
TEST_F(FlatOpMetricsDbCombinerTest, NewOpNumCores) {
  FlatOpMetricsDb src = ParseTextProtoOrDie(R"pb(
    op_instances {
      hlo_module_id: 1
      hlo_name: "op1"
      occurrences: 1
      num_cores: 2
    }
  )pb");

  FlatOpMetricsDbCombiner combiner(&dst_);
  combiner.Combine(src, /*update_num_cores=*/true);

  ASSERT_EQ(dst_.op_instances_size(), 1);
  const auto& metric = dst_.op_instances(0);
  EXPECT_EQ(metric.hlo_name(), "op1");
  EXPECT_EQ(metric.num_cores(), 2);
}

// -----------------------------------------------------------------------------
// Test Case 7: Complex Combine with Many Ops and Fields
// -----------------------------------------------------------------------------
TEST_F(FlatOpMetricsDbCombinerTest, ComplexCombine) {
  FlatOpMetricsDb src = ParseTextProtoOrDie(R"pb(
    total_time_ps: 1000
    idle_time_ps: 200
    op_instances {
      hlo_module_id: 1
      hlo_name: "op1"
      occurrences: 5
      time_ps: 500
      self_time_ps: 300
      flops: 5000
      bytes_accessed: 2000
      min_time_ps: 50
    }
    op_instances {
      hlo_module_id: 2
      hlo_name: "op2"
      occurrences: 2
      time_ps: 300
      self_time_ps: 300
      flops: 2000
      bytes_accessed: 1000
      min_time_ps: 100
    }
  )pb");

  dst_ = ParseTextProtoOrDie(R"pb(
    total_time_ps: 2000
    idle_time_ps: 400
    op_instances {
      hlo_module_id: 1
      hlo_name: "op1"
      occurrences: 2
      time_ps: 200
      self_time_ps: 100
      flops: 2000
      bytes_accessed: 800
      min_time_ps: 80
    }
    op_instances {
      hlo_module_id: 3
      hlo_name: "op3"
      occurrences: 1
      time_ps: 100
      self_time_ps: 100
    }
  )pb");

  FlatOpMetricsDbCombiner combiner(&dst_);
  combiner.Combine(src);

  EXPECT_EQ(dst_.total_time_ps(), 3000);
  EXPECT_EQ(dst_.idle_time_ps(), 600);
  EXPECT_EQ(dst_.op_instances_size(), 3);

  auto find_op = [&](const std::string& name) -> const FlatOpMetrics* {
    for (const auto& op : dst_.op_instances()) {
      if (op.hlo_name() == name) return &op;
    }
    return nullptr;
  };

  // Verify op1
  const auto* op1 = find_op("op1");
  ASSERT_NE(op1, nullptr);
  EXPECT_EQ(op1->occurrences(), 7);
  EXPECT_DOUBLE_EQ(op1->time_ps(), 700);
  EXPECT_DOUBLE_EQ(op1->self_time_ps(), 400);
  EXPECT_EQ(op1->flops(), 7000);
  EXPECT_DOUBLE_EQ(op1->bytes_accessed(), 2800);
  EXPECT_DOUBLE_EQ(op1->min_time_ps(), 50);  // min of 50 and 80

  // Verify op2
  const auto* op2 = find_op("op2");
  ASSERT_NE(op2, nullptr);
  EXPECT_EQ(op2->occurrences(), 2);
  EXPECT_DOUBLE_EQ(op2->time_ps(), 300);

  // Verify op3
  const auto* op3 = find_op("op3");
  ASSERT_NE(op3, nullptr);
  EXPECT_EQ(op3->occurrences(), 1);
  EXPECT_DOUBLE_EQ(op3->time_ps(), 100);
}

// -----------------------------------------------------------------------------
// Test Case 8: Stress Test with Many Ops
// -----------------------------------------------------------------------------
TEST_F(FlatOpMetricsDbCombinerTest, StressTestManyOps) {
  FlatOpMetricsDb src;
  FlatOpMetricsDb dst;

  const int kNumOps = 500;
  for (int i = 0; i < kNumOps; ++i) {
    std::string name = absl::StrCat("op", i);

    auto* src_op = src.add_op_instances();
    src_op->set_hlo_module_id(1);
    src_op->set_hlo_name(name);
    src_op->set_occurrences(1);
    src_op->set_time_ps(100);

    if (i % 2 == 0) {
      auto* dst_op = dst.add_op_instances();
      dst_op->set_hlo_module_id(1);
      dst_op->set_hlo_name(name);
      dst_op->set_occurrences(1);
      dst_op->set_time_ps(50);
    }
  }

  FlatOpMetricsDbCombiner combiner(&dst);
  combiner.Combine(src);

  EXPECT_EQ(dst.op_instances_size(), kNumOps);

  // Verify results
  // We need to find the ops in dst. Since they might not be in order (depending
  // on how combiner works), let's use a map for verification or just assume
  // they are in order if combiner preserves order or appends. Combiner appends
  // new ops. Overlapping ops are updated in place. So the order in dst will be:
  // original dst ops (even indices), then new src ops (odd indices). Let's use
  // a helper to find ops to be safe.
  auto find_op = [&](const FlatOpMetricsDb& db,
                     const std::string& name) -> const FlatOpMetrics* {
    for (const auto& op : db.op_instances()) {
      if (op.hlo_name() == name) return &op;
    }
    return nullptr;
  };

  for (int i = 0; i < kNumOps; ++i) {
    std::string name = absl::StrCat("op", i);
    const auto* op = find_op(dst, name);
    ASSERT_NE(op, nullptr) << "Op not found: " << name;
    if (i % 2 == 0) {
      EXPECT_EQ(op->occurrences(), 2) << "Op: " << name;
      EXPECT_DOUBLE_EQ(op->time_ps(), 150) << "Op: " << name;
    } else {
      EXPECT_EQ(op->occurrences(), 1) << "Op: " << name;
      EXPECT_DOUBLE_EQ(op->time_ps(), 100) << "Op: " << name;
    }
  }
}

}  // namespace
}  // namespace profiler
}  // namespace tensorflow
