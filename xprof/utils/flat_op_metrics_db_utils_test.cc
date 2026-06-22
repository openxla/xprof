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

#include "xprof/utils/flat_op_metrics_db_utils.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "<gtest/gtest.h>"
#include "absl/strings/str_cat.h"
#include "xla/tsl/profiler/utils/xplane_builder.h"
#include "xla/tsl/profiler/utils/xplane_schema.h"
#include "xla/tsl/profiler/utils/xplane_visitor.h"
#include "tsl/profiler/protobuf/xplane.pb.h"
#include "plugin/xprof/protobuf/flat_op_metrics.pb.h"
#include "plugin/xprof/protobuf/op_metrics.pb.h"

namespace tensorflow {
namespace profiler {
namespace {

using ::tensorflow::profiler::XEventMetadata;
using ::tensorflow::profiler::XPlane;
using ::tensorflow::profiler::XStatMetadata;
using ::tsl::profiler::XEventBuilder;
using ::tsl::profiler::XEventVisitor;
using ::tsl::profiler::XLineBuilder;
using ::tsl::profiler::XPlaneBuilder;
using ::tsl::profiler::XPlaneVisitor;

// Helper to create a dummy XPlane for testing.
XPlane CreateTestXPlane() {
  XPlane plane;
  XPlaneBuilder plane_builder(&plane);
  plane_builder.SetName("test_plane");

  XLineBuilder line_builder = plane_builder.GetOrCreateLine(0);
  line_builder.SetName("test_line");

  return plane;
}

class XEventsFlatOpMetricsDbBuilderTest : public ::testing::Test {
 protected:
  void SetUp() override {
    builder_ = std::make_unique<XEventsFlatOpMetricsDbBuilder>();
  }

  XEventBuilder CreateXEvent(XPlaneBuilder& plane_builder,
                             XLineBuilder& line_builder, absl::string_view name,
                             uint64_t ts, uint64_t dur, uint64_t program_id,
                             uint64_t symbol_id) {
    XEventMetadata* meta = plane_builder.GetOrCreateEventMetadata(name);

    XStatMetadata* program_id_meta =
        plane_builder.GetOrCreateStatMetadata("program_id");
    XStatMetadata* symbol_id_meta =
        plane_builder.GetOrCreateStatMetadata("symbol_id");

    // Add stats to metadata if they are not already there.
    bool has_program_id = false;
    bool has_symbol_id = false;
    for (const auto& stat : meta->stats()) {
      if (stat.metadata_id() == program_id_meta->id()) has_program_id = true;
      if (stat.metadata_id() == symbol_id_meta->id()) has_symbol_id = true;
    }

    if (!has_program_id) {
      auto* stat = meta->add_stats();
      stat->set_metadata_id(program_id_meta->id());
      stat->set_uint64_value(program_id);
    }

    if (!has_symbol_id) {
      auto* stat = meta->add_stats();
      stat->set_metadata_id(symbol_id_meta->id());
      stat->set_uint64_value(symbol_id);
    }

    XEventBuilder ev = line_builder.AddEvent(*meta);
    ev.SetTimestampPs(ts);
    ev.SetDurationPs(dur);

    return ev;
  }

  std::unique_ptr<XEventsFlatOpMetricsDbBuilder> builder_;
};

// -----------------------------------------------------------------------------
// Test Case 1: Basic Event Processing
// -----------------------------------------------------------------------------
TEST_F(XEventsFlatOpMetricsDbBuilderTest, BasicEventProcessing) {
  XPlane plane = CreateTestXPlane();
  XPlaneBuilder plane_builder(&plane);
  XLineBuilder line_builder = plane_builder.GetOrCreateLine(0);

  CreateXEvent(plane_builder, line_builder, "op1", 1000, 500, 1, 1);

  XPlaneVisitor plane_visitor(&plane, {}, {tsl::profiler::FindStatType});
  XEventVisitor event_visitor(&plane_visitor, &plane.lines(0),
                              &plane.lines(0).events(0));

  builder_->AddOpMetric(event_visitor);
  FlatOpMetricsDb db = builder_->Finalize();

  ASSERT_EQ(db.op_instances_size(), 1);
  const auto& metric = db.op_instances(0);
  EXPECT_EQ(metric.hlo_name(), "op1");
  EXPECT_EQ(metric.occurrences(), 1);
  EXPECT_DOUBLE_EQ(metric.time_ps(), 500);
  EXPECT_DOUBLE_EQ(metric.self_time_ps(), 500);
}

// -----------------------------------------------------------------------------
// Test Case 2: Aggregation of Multiple Instances
// -----------------------------------------------------------------------------
TEST_F(XEventsFlatOpMetricsDbBuilderTest, AggregateMultipleInstances) {
  XPlane plane = CreateTestXPlane();
  XPlaneBuilder plane_builder(&plane);
  XLineBuilder line_builder = plane_builder.GetOrCreateLine(0);

  CreateXEvent(plane_builder, line_builder, "op1", 1000, 500, 1, 1);
  CreateXEvent(plane_builder, line_builder, "op1", 2000, 300, 1, 1);

  XPlaneVisitor plane_visitor(&plane, {}, {tsl::profiler::FindStatType});

  XEventVisitor ev1(&plane_visitor, &plane.lines(0), &plane.lines(0).events(0));
  XEventVisitor ev2(&plane_visitor, &plane.lines(0), &plane.lines(0).events(1));

  builder_->AddOpMetric(ev1);
  builder_->AddOpMetric(ev2);
  FlatOpMetricsDb db = builder_->Finalize();

  ASSERT_EQ(db.op_instances_size(), 1);
  const auto& metric = db.op_instances(0);
  EXPECT_EQ(metric.hlo_name(), "op1");
  EXPECT_EQ(metric.occurrences(), 2);
  EXPECT_DOUBLE_EQ(metric.time_ps(), 800);
  EXPECT_DOUBLE_EQ(metric.min_time_ps(), 300);
}

// -----------------------------------------------------------------------------
// Test Case 3: Self-Time from Event Stats
// -----------------------------------------------------------------------------
TEST_F(XEventsFlatOpMetricsDbBuilderTest, SelfTimeFromEventStats) {
  XPlane plane = CreateTestXPlane();
  XPlaneBuilder plane_builder(&plane);
  XLineBuilder line_builder = plane_builder.GetOrCreateLine(0);

  XEventBuilder ev =
      CreateXEvent(plane_builder, line_builder, "op1", 1000, 1000, 1, 1);

  XStatMetadata* self_dur_meta =
      plane_builder.GetOrCreateStatMetadata("self_duration_ps");
  ev.AddStatValue(*self_dur_meta, int64_t{600});

  XPlaneVisitor plane_visitor(&plane, {}, {tsl::profiler::FindStatType});
  XEventVisitor event_visitor(&plane_visitor, &plane.lines(0),
                              &plane.lines(0).events(0));

  builder_->AddOpMetric(event_visitor);
  FlatOpMetricsDb db = builder_->Finalize();

  ASSERT_EQ(db.op_instances_size(), 1);
  const auto& metric = db.op_instances(0);
  EXPECT_DOUBLE_EQ(metric.time_ps(), 1000);
  EXPECT_DOUBLE_EQ(metric.self_time_ps(), 600);
}

// -----------------------------------------------------------------------------
// Test Case 4: Topological Sorting with Manual Metrics
// -----------------------------------------------------------------------------
TEST_F(XEventsFlatOpMetricsDbBuilderTest, TopologicalSortManualMetrics) {
  FlatOpMetrics parent;
  parent.set_op_id(1);
  parent.set_hlo_name("parent");
  parent.set_time_ps(1000);
  parent.set_self_time_ps(600);
  parent.set_occurrences(1);

  FlatOpMetrics child;
  child.set_op_id(2);
  child.set_parent_op_id(1);
  child.set_hlo_name("child");
  child.set_time_ps(400);
  child.set_self_time_ps(400);
  child.set_occurrences(1);

  XEventsFlatOpMetricsDbBuilder::OpKey key_parent{1, 1};
  XEventsFlatOpMetricsDbBuilder::OpKey key_child{1, 2};

  builder_->AddOpMetric(child, key_child);
  builder_->AddOpMetric(parent, key_parent);

  FlatOpMetricsDb db = builder_->FinalizeSorted();

  ASSERT_EQ(db.op_instances_size(), 2);
  EXPECT_EQ(db.op_instances(0).hlo_name(), "parent");
  EXPECT_EQ(db.op_instances(1).hlo_name(), "child");
}

// -----------------------------------------------------------------------------
// Test Case 5: Deeply Nested Hierarchy Sorting
// -----------------------------------------------------------------------------
TEST_F(XEventsFlatOpMetricsDbBuilderTest, DeeplyNestedHierarchySorting) {
  auto create_metric = [](uint64_t id, uint64_t parent_id,
                          const std::string& name) {
    FlatOpMetrics m;
    m.set_op_id(id);
    m.set_parent_op_id(parent_id);
    m.set_hlo_name(name);
    m.set_occurrences(1);
    return m;
  };

  FlatOpMetrics m1 = create_metric(1, 0, "level1");
  FlatOpMetrics m2 = create_metric(2, 1, "level2");
  FlatOpMetrics m3 = create_metric(3, 2, "level3");
  FlatOpMetrics m4 = create_metric(4, 3, "level4");
  FlatOpMetrics m5 = create_metric(5, 4, "level5");

  XEventsFlatOpMetricsDbBuilder::OpKey k1{1, 1};
  XEventsFlatOpMetricsDbBuilder::OpKey k2{1, 2};
  XEventsFlatOpMetricsDbBuilder::OpKey k3{1, 3};
  XEventsFlatOpMetricsDbBuilder::OpKey k4{1, 4};
  XEventsFlatOpMetricsDbBuilder::OpKey k5{1, 5};

  builder_->AddOpMetric(m5, k5);
  builder_->AddOpMetric(m4, k4);
  builder_->AddOpMetric(m3, k3);
  builder_->AddOpMetric(m2, k2);
  builder_->AddOpMetric(m1, k1);

  FlatOpMetricsDb db = builder_->FinalizeSorted();

  ASSERT_EQ(db.op_instances_size(), 5);
  EXPECT_EQ(db.op_instances(0).hlo_name(), "level1");
  EXPECT_EQ(db.op_instances(1).hlo_name(), "level2");
  EXPECT_EQ(db.op_instances(2).hlo_name(), "level3");
  EXPECT_EQ(db.op_instances(3).hlo_name(), "level4");
  EXPECT_EQ(db.op_instances(4).hlo_name(), "level5");
}

// -----------------------------------------------------------------------------
// Test Case 7: Multiple Programs
// -----------------------------------------------------------------------------
TEST_F(XEventsFlatOpMetricsDbBuilderTest, MultiplePrograms) {
  XPlane plane = CreateTestXPlane();
  XPlaneBuilder plane_builder(&plane);
  XLineBuilder line_builder = plane_builder.GetOrCreateLine(0);

  CreateXEvent(plane_builder, line_builder, "op1", 1000, 500, 1, 1);
  CreateXEvent(plane_builder, line_builder, "op2", 2000, 300, 2, 1);

  XPlaneVisitor plane_visitor(&plane, {}, {tsl::profiler::FindStatType});

  XEventVisitor ev1(&plane_visitor, &plane.lines(0), &plane.lines(0).events(0));
  XEventVisitor ev2(&plane_visitor, &plane.lines(0), &plane.lines(0).events(1));

  builder_->AddOpMetric(ev1);
  builder_->AddOpMetric(ev2);
  FlatOpMetricsDb db = builder_->Finalize();

  EXPECT_EQ(db.op_instances_size(), 2);
}

// -----------------------------------------------------------------------------
// Test Case 8: Memory Breakdown Parsing
// -----------------------------------------------------------------------------
TEST_F(XEventsFlatOpMetricsDbBuilderTest, MemoryBreakdownParsing) {
  XPlane plane = CreateTestXPlane();
  XPlaneBuilder plane_builder(&plane);
  XLineBuilder line_builder = plane_builder.GetOrCreateLine(0);

  CreateXEvent(plane_builder, line_builder, "op1", 1000, 500, 1, 1);

  tensorflow::profiler::MemoryAccessBreakdown breakdown;
  auto* mem = breakdown.add_memory_accessed();
  mem->set_operation_type(
      tensorflow::profiler::OpMetrics::MemoryAccessed::READ);
  mem->set_memory_space(1);

  mem->set_bytes_accessed(100);

  std::string serialized_breakdown;
  ASSERT_TRUE(breakdown.SerializeToString(&serialized_breakdown));

  XEventMetadata* meta = plane_builder.GetOrCreateEventMetadata("op1");
  XStatMetadata* breakdown_meta =
      plane_builder.GetOrCreateStatMetadata("memory_access_breakdown");
  auto* stat = meta->add_stats();
  stat->set_metadata_id(breakdown_meta->id());
  stat->set_bytes_value(serialized_breakdown);

  XPlaneVisitor plane_visitor(&plane, {}, {tsl::profiler::FindStatType});
  XEventVisitor event_visitor(&plane_visitor, &plane.lines(0),
                              &plane.lines(0).events(0));

  builder_->AddOpMetric(event_visitor);
  FlatOpMetricsDb db = builder_->Finalize();

  ASSERT_EQ(db.op_instances_size(), 1);
  const auto& metric = db.op_instances(0);
  ASSERT_EQ(metric.memory_accessed_breakdown_size(), 1);
  EXPECT_EQ(metric.memory_accessed_breakdown(0).operation_type(),
            FlatOpMetrics::MemoryAccessed::READ);
  EXPECT_EQ(metric.memory_accessed_breakdown(0).memory_space(), 1);
  EXPECT_EQ(metric.memory_accessed_breakdown(0).bytes_accessed(), 100);
}

// -----------------------------------------------------------------------------
// Test Case 9: Performance Metrics Stats
// -----------------------------------------------------------------------------
TEST_F(XEventsFlatOpMetricsDbBuilderTest, PerformanceMetricsStats) {
  XPlane plane = CreateTestXPlane();
  XPlaneBuilder plane_builder(&plane);
  XLineBuilder line_builder = plane_builder.GetOrCreateLine(0);
  CreateXEvent(plane_builder, line_builder, "op1", 1000, 500, 1, 1);

  XEventMetadata* meta = plane_builder.GetOrCreateEventMetadata("op1");

  auto* stat1 = meta->add_stats();
  stat1->set_metadata_id(plane_builder.GetOrCreateStatMetadata("flops")->id());
  stat1->set_int64_value(1000);

  auto* stat2 = meta->add_stats();
  stat2->set_metadata_id(
      plane_builder.GetOrCreateStatMetadata("model_flops")->id());
  stat2->set_int64_value(2000);

  auto* stat3 = meta->add_stats();
  XStatMetadata* bytes_accessed_meta =
      plane_builder.GetOrCreateStatMetadata("bytes_accessed");
  stat3->set_metadata_id(bytes_accessed_meta->id());
  stat3->set_int64_value(3000);

  XPlaneVisitor plane_visitor(&plane, {}, {tsl::profiler::FindStatType});
  XEventVisitor event_visitor(&plane_visitor, &plane.lines(0),
                              &plane.lines(0).events(0));

  builder_->AddOpMetric(event_visitor);
  FlatOpMetricsDb db = builder_->Finalize();

  ASSERT_EQ(db.op_instances_size(), 1);
  const auto& metric = db.op_instances(0);
  EXPECT_EQ(metric.flops(), 1000);
  EXPECT_EQ(metric.model_flops(), 2000);
  EXPECT_EQ(metric.bytes_accessed(), 3000);
}

// -----------------------------------------------------------------------------
// Test Case 10: Stress Test with Many Events
// -----------------------------------------------------------------------------
TEST_F(XEventsFlatOpMetricsDbBuilderTest, StressTestManyEvents) {
  XPlane plane = CreateTestXPlane();
  XPlaneBuilder plane_builder(&plane);
  XLineBuilder line_builder = plane_builder.GetOrCreateLine(0);

  const int kNumEvents = 1000;
  for (int i = 0; i < kNumEvents; ++i) {
    uint64_t program_id = 1;
    uint64_t symbol_id = (i % 10) + 1;  // 10 distinct ops
    std::string name = absl::StrCat("op", symbol_id);
    CreateXEvent(plane_builder, line_builder, name, i * 1000, 500, program_id,
                 symbol_id);
  }

  XPlaneVisitor plane_visitor(&plane, {}, {tsl::profiler::FindStatType});

  for (int i = 0; i < kNumEvents; ++i) {
    XEventVisitor event_visitor(&plane_visitor, &plane.lines(0),
                                &plane.lines(0).events(i));
    builder_->AddOpMetric(event_visitor);
  }

  FlatOpMetricsDb db = builder_->Finalize();

  EXPECT_EQ(db.op_instances_size(), 10);
  for (const auto& metric : db.op_instances()) {
    EXPECT_EQ(metric.occurrences(), kNumEvents / 10);
    EXPECT_DOUBLE_EQ(metric.time_ps(), (kNumEvents / 10) * 500);
  }
}

// -----------------------------------------------------------------------------
// Test Case 11: Complex Hierarchy with Multiple Roots
// -----------------------------------------------------------------------------
TEST_F(XEventsFlatOpMetricsDbBuilderTest, ComplexHierarchyMultipleRoots) {
  auto create_metric = [](uint64_t id, uint64_t parent_id,
                          const std::string& name) {
    FlatOpMetrics m;
    m.set_op_id(id);
    m.set_parent_op_id(parent_id);
    m.set_hlo_name(name);
    m.set_occurrences(1);
    return m;
  };

  // Tree 1
  FlatOpMetrics m1 = create_metric(1, 0, "r1");
  FlatOpMetrics m2 = create_metric(2, 1, "r1_c1");
  FlatOpMetrics m3 = create_metric(3, 1, "r1_c2");
  FlatOpMetrics m4 = create_metric(4, 2, "r1_c1_c1");

  // Tree 2
  FlatOpMetrics m5 = create_metric(5, 0, "r2");
  FlatOpMetrics m6 = create_metric(6, 5, "r2_c1");

  XEventsFlatOpMetricsDbBuilder::OpKey k1{1, 1};
  XEventsFlatOpMetricsDbBuilder::OpKey k2{1, 2};
  XEventsFlatOpMetricsDbBuilder::OpKey k3{1, 3};
  XEventsFlatOpMetricsDbBuilder::OpKey k4{1, 4};
  XEventsFlatOpMetricsDbBuilder::OpKey k5{1, 5};
  XEventsFlatOpMetricsDbBuilder::OpKey k6{1, 6};

  builder_->AddOpMetric(m4, k4);
  builder_->AddOpMetric(m6, k6);
  builder_->AddOpMetric(m1, k1);
  builder_->AddOpMetric(m5, k5);
  builder_->AddOpMetric(m3, k3);
  builder_->AddOpMetric(m2, k2);

  FlatOpMetricsDb db = builder_->FinalizeSorted();

  ASSERT_EQ(db.op_instances_size(), 6);

  auto find_index = [&](const std::string& name) {
    for (int i = 0; i < db.op_instances_size(); ++i) {
      if (db.op_instances(i).hlo_name() == name) return i;
    }
    return -1;
  };

  int idx_r1 = find_index("r1");
  int idx_r1_c1 = find_index("r1_c1");
  int idx_r1_c2 = find_index("r1_c2");
  int idx_r1_c1_c1 = find_index("r1_c1_c1");
  int idx_r2 = find_index("r2");
  int idx_r2_c1 = find_index("r2_c1");

  EXPECT_LT(idx_r1, idx_r1_c1);
  EXPECT_LT(idx_r1, idx_r1_c2);
  EXPECT_LT(idx_r1_c1, idx_r1_c1_c1);
  EXPECT_LT(idx_r2, idx_r2_c1);
}

// -----------------------------------------------------------------------------
// Tests for FlatOpMetricsDbBuilder
// -----------------------------------------------------------------------------

class FlatOpMetricsDbBuilderTest : public ::testing::Test {
 protected:
  FlatOpMetricsDb db_;
};

class TestFlatOpMetricsDbBuilder : public FlatOpMetricsDbBuilder {
 public:
  using FlatOpMetricsDbBuilder::FlatOpMetricsDbBuilder;
  using FlatOpMetricsDbBuilder::LookupOrInsertNewFlatOpMetrics;
};

TEST_F(FlatOpMetricsDbBuilderTest, LookupOrInsertNew) {
  TestFlatOpMetricsDbBuilder builder(&db_);

  FlatOpMetrics* op1 = builder.LookupOrInsertNewFlatOpMetrics(1, "op1");
  ASSERT_NE(op1, nullptr);
  EXPECT_EQ(op1->hlo_module_id(), 1);
  EXPECT_EQ(op1->hlo_name(), "op1");

  // Lookup again should return the same pointer.
  FlatOpMetrics* op1_again = builder.LookupOrInsertNewFlatOpMetrics(1, "op1");
  EXPECT_EQ(op1, op1_again);

  EXPECT_EQ(db_.op_instances_size(), 1);
}

TEST_F(FlatOpMetricsDbBuilderTest, ConstructorPopulatesMap) {
  // Pre-populate DB
  auto* op = db_.add_op_instances();
  op->set_hlo_module_id(1);
  op->set_hlo_name("op1");

  TestFlatOpMetricsDbBuilder builder(&db_);

  // Lookup should find the existing one.
  FlatOpMetrics* found_op = builder.LookupOrInsertNewFlatOpMetrics(1, "op1");
  EXPECT_EQ(found_op, op);
  EXPECT_EQ(db_.op_instances_size(), 1);
}

TEST_F(FlatOpMetricsDbBuilderTest, StressTest) {
  TestFlatOpMetricsDbBuilder builder(&db_);
  const int kNumOps = 1000;
  for (int i = 0; i < kNumOps; ++i) {
    builder.LookupOrInsertNewFlatOpMetrics(i, absl::StrCat("op", i));
  }

  EXPECT_EQ(db_.op_instances_size(), kNumOps);

  for (int i = 0; i < kNumOps; ++i) {
    FlatOpMetrics* op =
        builder.LookupOrInsertNewFlatOpMetrics(i, absl::StrCat("op", i));
    EXPECT_EQ(op->hlo_module_id(), i);
    EXPECT_EQ(op->hlo_name(), absl::StrCat("op", i));
  }
}

TEST_F(FlatOpMetricsDbBuilderTest, DetailedMultiProgramMultiOpTest) {
  TestFlatOpMetricsDbBuilder builder(&db_);

  const int kNumPrograms = 5;
  const int kOpsPerProgram = 50;

  for (int p = 0; p < kNumPrograms; ++p) {
    for (int o = 0; o < kOpsPerProgram; ++o) {
      std::string op_name = absl::StrCat("program_", p, "_op_", o);
      FlatOpMetrics* op = builder.LookupOrInsertNewFlatOpMetrics(p, op_name);
      op->set_occurrences(1);
      op->set_time_ps(100.0 * (o + 1));
      op->set_self_time_ps(50.0 * (o + 1));
    }
  }

  EXPECT_EQ(db_.op_instances_size(), kNumPrograms * kOpsPerProgram);

  for (int p = 0; p < kNumPrograms; ++p) {
    for (int o = 0; o < kOpsPerProgram; ++o) {
      std::string op_name = absl::StrCat("program_", p, "_op_", o);
      FlatOpMetrics* op = builder.LookupOrInsertNewFlatOpMetrics(p, op_name);
      EXPECT_EQ(op->hlo_module_id(), p);
      EXPECT_EQ(op->hlo_name(), op_name);
      EXPECT_DOUBLE_EQ(op->time_ps(), 100.0 * (o + 1));
      EXPECT_DOUBLE_EQ(op->self_time_ps(), 50.0 * (o + 1));
    }
  }
}

TEST_F(FlatOpMetricsDbBuilderTest, ModifyReturnedPointer) {
  TestFlatOpMetricsDbBuilder builder(&db_);
  FlatOpMetrics* op = builder.LookupOrInsertNewFlatOpMetrics(1, "op1");
  op->set_occurrences(5);
  op->set_time_ps(500);

  ASSERT_EQ(db_.op_instances_size(), 1);
  EXPECT_EQ(db_.op_instances(0).occurrences(), 5);
  EXPECT_DOUBLE_EQ(db_.op_instances(0).time_ps(), 500);
}

TEST_F(FlatOpMetricsDbBuilderTest, EdgeCases) {
  TestFlatOpMetricsDbBuilder builder(&db_);

  // Empty name
  FlatOpMetrics* op1 = builder.LookupOrInsertNewFlatOpMetrics(1, "");
  EXPECT_EQ(op1->hlo_name(), "");

  // Zero ID
  FlatOpMetrics* op2 = builder.LookupOrInsertNewFlatOpMetrics(0, "op2");
  EXPECT_EQ(op2->hlo_module_id(), 0);

  // Large ID
  uint64_t large_id = 18446744073709551615ULL;  // Max uint64
  FlatOpMetrics* op3 = builder.LookupOrInsertNewFlatOpMetrics(large_id, "op3");
  EXPECT_EQ(op3->hlo_module_id(), large_id);
}

TEST_F(FlatOpMetricsDbBuilderTest, ComprehensiveDbBuild) {
  TestFlatOpMetricsDbBuilder builder(&db_);

  struct RawMetric {
    uint64_t mod_id;
    std::string name;
    double time;
    double self_time;
    int occurrences;
  };

  std::vector<RawMetric> raw_metrics = {
      {1, "op1", 1000, 500, 2},
      {1, "op2", 500, 500, 1},
      {2, "op1", 2000, 1000, 1},
      {2, "op3", 300, 300, 3},
  };

  for (const auto& rm : raw_metrics) {
    FlatOpMetrics* op =
        builder.LookupOrInsertNewFlatOpMetrics(rm.mod_id, rm.name);
    op->set_time_ps(rm.time);
    op->set_self_time_ps(rm.self_time);
    op->set_occurrences(rm.occurrences);
  }

  ASSERT_EQ(db_.op_instances_size(), 4);

  for (const auto& rm : raw_metrics) {
    FlatOpMetrics* op =
        builder.LookupOrInsertNewFlatOpMetrics(rm.mod_id, rm.name);
    EXPECT_DOUBLE_EQ(op->time_ps(), rm.time);
    EXPECT_DOUBLE_EQ(op->self_time_ps(), rm.self_time);
    EXPECT_EQ(op->occurrences(), rm.occurrences);
  }
}

TEST_F(XEventsFlatOpMetricsDbBuilderTest, FinalizeWithTotalTime) {
  XPlane plane = CreateTestXPlane();
  XPlaneBuilder plane_builder(&plane);
  XLineBuilder line_builder = plane_builder.GetOrCreateLine(0);

  CreateXEvent(plane_builder, line_builder, "op1", 1000, 500, 1, 1);

  XPlaneVisitor plane_visitor(&plane, {}, {tsl::profiler::FindStatType});
  XEventVisitor event_visitor(&plane_visitor, &plane.lines(0),
                              &plane.lines(0).events(0));

  builder_->AddOpMetric(event_visitor);
  FlatOpMetricsDb db = builder_->Finalize(1000);

  // We expect 2 ops: "op1" and "IDLE"
  ASSERT_EQ(db.op_instances_size(), 2);

  // Find idle op
  const FlatOpMetrics* idle_op = nullptr;
  for (const auto& op : db.op_instances()) {
    if (IsIdleOp(op)) {
      idle_op = &op;
    }
  }
  ASSERT_NE(idle_op, nullptr);
  EXPECT_EQ(idle_op->time_ps(), 500);
  EXPECT_EQ(db.total_time_ps(), 1000);
}

TEST_F(XEventsFlatOpMetricsDbBuilderTest, CustomCallMetricsScaling) {
  XPlane plane = CreateTestXPlane();
  XPlaneBuilder plane_builder(&plane);
  XLineBuilder line_builder = plane_builder.GetOrCreateLine(0);

  XEventBuilder ev = CreateXEvent(plane_builder, line_builder, "custom_call_op",
                                  1000, 500, 1, 1);
  ev.SetNumOccurrences(2);

  XEventMetadata* meta =
      plane_builder.GetOrCreateEventMetadata("custom_call_op");
  XStatMetadata* category_meta =
      plane_builder.GetOrCreateStatMetadata("hlo_category");
  auto* stat_cat = meta->add_stats();
  stat_cat->set_metadata_id(category_meta->id());
  stat_cat->set_str_value("custom-call");

  XStatMetadata* flops_meta = plane_builder.GetOrCreateStatMetadata("flops");
  ev.AddStatValue(*flops_meta, int64_t{100});

  XPlaneVisitor plane_visitor(&plane, {}, {tsl::profiler::FindStatType});
  XEventVisitor event_visitor(&plane_visitor, &plane.lines(0),
                              &plane.lines(0).events(0));

  builder_->AddOpMetric(event_visitor);
  FlatOpMetricsDb db = builder_->Finalize();

  ASSERT_EQ(db.op_instances_size(), 1);
  const auto& metric = db.op_instances(0);
  EXPECT_EQ(metric.hlo_name(), "custom_call_op");
  EXPECT_EQ(metric.flops(), 200);
}

// -----------------------------------------------------------------------------
// Test Case 12: Source Information Parsing
// -----------------------------------------------------------------------------
TEST_F(XEventsFlatOpMetricsDbBuilderTest, SourceInformationParsing) {
  XPlane plane = CreateTestXPlane();
  XPlaneBuilder plane_builder(&plane);
  XLineBuilder line_builder = plane_builder.GetOrCreateLine(0);

  CreateXEvent(plane_builder, line_builder, "op_with_source", 1000, 500, 1, 1);

  XEventMetadata* meta =
      plane_builder.GetOrCreateEventMetadata("op_with_source");

  XStatMetadata* source_meta = plane_builder.GetOrCreateStatMetadata("source");
  auto* stat_source = meta->add_stats();
  stat_source->set_metadata_id(source_meta->id());
  stat_source->set_str_value("filename.cc:123");

  XStatMetadata* stack_meta =
      plane_builder.GetOrCreateStatMetadata("source_stack");
  auto* stat_stack = meta->add_stats();
  stat_stack->set_metadata_id(stack_meta->id());
  stat_stack->set_str_value("stack_frame_contents");

  XPlaneVisitor plane_visitor(&plane, {}, {tsl::profiler::FindStatType});
  XEventVisitor event_visitor(&plane_visitor, &plane.lines(0),
                              &plane.lines(0).events(0));

  builder_->AddOpMetric(event_visitor);
  FlatOpMetricsDb db = builder_->Finalize();

  ASSERT_EQ(db.op_instances_size(), 1);
  const auto& metric = db.op_instances(0);
  EXPECT_EQ(metric.hlo_name(), "op_with_source");
  ASSERT_TRUE(metric.has_source_info());
  EXPECT_EQ(metric.source_info().file_name(), "filename.cc");
  EXPECT_EQ(metric.source_info().line_number(), 123);
  EXPECT_EQ(metric.source_info().stack_frame(), "stack_frame_contents");
}

// -----------------------------------------------------------------------------
// Tests for CreateTfMetricsDbFromDeviceOpMetricsDb
// -----------------------------------------------------------------------------

TEST(CreateTfMetricsDbFromDeviceOpMetricsDbTest, AggregationAndIdle) {
  FlatOpMetricsDb device_db;
  device_db.set_total_op_time_ps(1000);
  device_db.set_total_time_ps(1500);

  // Op 1 with provenance
  auto* op1 = device_db.add_op_instances();
  op1->set_hlo_name("hlo1");
  op1->set_provenance("TfOp1:TfOpType1");
  op1->set_time_ps(500);
  op1->set_self_time_ps(400);
  op1->set_flops(100);
  op1->set_flops_v2(101);
  op1->set_model_flops(102);
  op1->set_model_flops_v2(103);
  op1->set_bytes_accessed(1000);
  op1->set_occurrences(1);

  // Op 2 with same provenance
  auto* op2 = device_db.add_op_instances();
  op2->set_hlo_name("hlo2");
  op2->set_provenance("TfOp1:TfOpType1");
  op2->set_time_ps(300);
  op2->set_self_time_ps(300);
  op2->set_flops(50);
  op2->set_flops_v2(51);
  op2->set_model_flops(52);
  op2->set_model_flops_v2(53);
  op2->set_bytes_accessed(500);
  op2->set_occurrences(2);

  // Op 3 with empty provenance
  auto* op3 = device_db.add_op_instances();
  op3->set_hlo_name("hlo3");
  op3->set_time_ps(200);
  op3->set_self_time_ps(200);
  op3->set_flops(20);
  op3->set_bytes_accessed(200);
  op3->set_occurrences(1);

  // Idle Op
  auto* idle_op = device_db.add_op_instances();
  idle_op->set_category(kIdle);
  idle_op->set_time_ps(500);
  idle_op->set_self_time_ps(500);

  // Test with_idle = true
  {
    FlatOpMetricsDb tf_db =
        CreateTfMetricsDbFromDeviceOpMetricsDb(device_db, /*with_idle=*/true);

    EXPECT_EQ(tf_db.total_op_time_ps(), 1000);
    EXPECT_EQ(tf_db.total_time_ps(), 1500);
    ASSERT_EQ(tf_db.op_instances_size(), 3);

    // Find ops
    const FlatOpMetrics* tf_op1 = nullptr;
    const FlatOpMetrics* tf_op3 = nullptr;
    const FlatOpMetrics* tf_idle = nullptr;

    for (const auto& op : tf_db.op_instances()) {
      if (op.hlo_name() == "TfOp1")
        tf_op1 = &op;
      else if (op.hlo_name() == "hlo3")
        tf_op3 = &op;
      else if (op.hlo_name() == kIdle)
        tf_idle = &op;
    }

    ASSERT_NE(tf_op1, nullptr);
    EXPECT_EQ(tf_op1->category(), "TfOpType1");
    EXPECT_EQ(tf_op1->time_ps(), 800);          // 500 + 300
    EXPECT_EQ(tf_op1->self_time_ps(), 700);     // 400 + 300
    EXPECT_EQ(tf_op1->flops(), 150);            // 100 + 50
    EXPECT_EQ(tf_op1->flops_v2(), 152);         // 101 + 51
    EXPECT_EQ(tf_op1->model_flops(), 154);      // 102 + 52
    EXPECT_EQ(tf_op1->model_flops_v2(), 156);   // 103 + 53
    EXPECT_EQ(tf_op1->bytes_accessed(), 1500);  // 1000 + 500
    EXPECT_EQ(tf_op1->occurrences(), 2);         // max(1, 2)

    ASSERT_NE(tf_op3, nullptr);
    EXPECT_EQ(tf_op3->category(), "Unknown");
    EXPECT_EQ(tf_op3->time_ps(), 200);

    ASSERT_NE(tf_idle, nullptr);
    EXPECT_EQ(tf_idle->category(), kIdle);
    EXPECT_EQ(tf_idle->time_ps(), 500);
  }

  // Test with_idle = false
  {
    FlatOpMetricsDb tf_db =
        CreateTfMetricsDbFromDeviceOpMetricsDb(device_db, /*with_idle=*/false);

    EXPECT_EQ(tf_db.total_op_time_ps(), 1000);
    EXPECT_EQ(tf_db.total_time_ps(), 1000);  // Should be total_op_time_ps
    ASSERT_EQ(tf_db.op_instances_size(), 2);  // Idle should be skipped

    const FlatOpMetrics* tf_idle = nullptr;
    for (const auto& op : tf_db.op_instances()) {
      if (op.hlo_name() == kIdle) tf_idle = &op;
    }
    EXPECT_EQ(tf_idle, nullptr);
  }
}

}  // namespace
}  // namespace profiler
}  // namespace tensorflow
