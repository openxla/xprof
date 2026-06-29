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

#include "xprof/convert/xplane_to_flat_op_metrics_db.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "testing/base/public/gmock.h"
#include "<gtest/gtest.h>"
#include "absl/container/flat_hash_map.h"
#include "xla/tsl/profiler/utils/xplane_builder.h"
#include "xla/tsl/profiler/utils/xplane_schema.h"
#include "tsl/profiler/protobuf/xplane.pb.h"
#include "xprof/convert/xplane_to_op_metrics_db.h"
#include "plugin/xprof/protobuf/flat_op_metrics.pb.h"
#include "plugin/xprof/protobuf/op_metrics.pb.h"
#include "xprof/utils/hlo_module_map.h"

namespace tensorflow {
namespace profiler {
namespace {

using ::testing::Eq;
using ::testing::SizeIs;
using ::tsl::profiler::StatType;
using ::tsl::profiler::XEventBuilder;
using ::tsl::profiler::XLineBuilder;
using ::tsl::profiler::XPlaneBuilder;

class XPlaneToFlatOpMetricsDbTest : public ::testing::Test {
 protected:
  void SetUp() override {
    xplane_ = std::make_unique<XPlane>();
    plane_builder_ = std::make_unique<XPlaneBuilder>(xplane_.get());
  }

  std::unique_ptr<XPlane> xplane_;
  std::unique_ptr<XPlaneBuilder> plane_builder_;

  // Helper to add a module event to the plane.
  // Now explicitly sets kDeviceOffsetPs and kDeviceDurationPs in picoseconds.
  XEventBuilder AddModuleEvent(XLineBuilder& line, const std::string& name,
                               int64_t start_ns, int64_t duration_ns,
                               uint64_t offload_core_id, uint64_t tc_start_id) {
    XEventBuilder builder =
        line.AddEvent(*plane_builder_->GetOrCreateEventMetadata(name));
    builder.SetTimestampNs(start_ns);
    builder.SetDurationNs(duration_ns);

    // Set explicit device stats to avoid fallback issues.
    builder.AddStatValue(*plane_builder_->GetOrCreateStatMetadata(
                             GetStatTypeStr(StatType::kDeviceOffsetPs)),
                         start_ns * 1000);
    builder.AddStatValue(*plane_builder_->GetOrCreateStatMetadata(
                             GetStatTypeStr(StatType::kDeviceDurationPs)),
                         duration_ns * 1000);

    builder.AddStatValue(*plane_builder_->GetOrCreateStatMetadata(
                             GetStatTypeStr(StatType::kOffloadCoreId)),
                         offload_core_id);
    builder.AddStatValue(*plane_builder_->GetOrCreateStatMetadata(
                             GetStatTypeStr(StatType::kTcOffloadStartId)),
                         tc_start_id);
    return builder;
  }

  // Helper to add an HLO op event to the plane.
  // Now explicitly sets kDeviceOffsetPs and kDeviceDurationPs in picoseconds.
  XEventBuilder AddHloOpEvent(XLineBuilder& line, const std::string& name,
                              int64_t start_ns, int64_t duration_ns,
                              uint64_t program_id, uint64_t symbol_id) {
    XEventBuilder builder =
        line.AddEvent(*plane_builder_->GetOrCreateEventMetadata(name));
    builder.SetTimestampNs(start_ns);
    builder.SetDurationNs(duration_ns);

    // Set explicit device stats to avoid fallback issues.
    builder.AddStatValue(*plane_builder_->GetOrCreateStatMetadata(
                             GetStatTypeStr(StatType::kDeviceOffsetPs)),
                         start_ns * 1000);
    builder.AddStatValue(*plane_builder_->GetOrCreateStatMetadata(
                             GetStatTypeStr(StatType::kDeviceDurationPs)),
                         duration_ns * 1000);

    tsl::profiler::XStatsBuilder<XEventMetadata> event_metadata(
        plane_builder_->GetOrCreateEventMetadata(name), plane_builder_.get());
    event_metadata.AddStatValue(*plane_builder_->GetOrCreateStatMetadata(
                                    GetStatTypeStr(StatType::kProgramId)),
                                program_id);
    event_metadata.AddStatValue(*plane_builder_->GetOrCreateStatMetadata(
                                    GetStatTypeStr(StatType::kSymbolId)),
                                symbol_id);
    return builder;
  }

  // Helper to add a step event to the plane.
  XEventBuilder AddStepEvent(XLineBuilder* line, const std::string& name,
                             int64_t start_ns, int64_t duration_ns) {
    XEventBuilder builder =
        line->AddEvent(*plane_builder_->GetOrCreateEventMetadata(name));
    builder.SetTimestampNs(start_ns);
    builder.SetDurationNs(duration_ns);

    builder.AddStatValue(*plane_builder_->GetOrCreateStatMetadata(
                             GetStatTypeStr(StatType::kDeviceOffsetPs)),
                         start_ns * 1000);
    builder.AddStatValue(*plane_builder_->GetOrCreateStatMetadata(
                             GetStatTypeStr(StatType::kDeviceDurationPs)),
                         duration_ns * 1000);
    return builder;
  }
};

// Test Case 1: Empty XPlane should result in an empty metrics map.
TEST_F(XPlaneToFlatOpMetricsDbTest, EmptyPlane) {
  absl::flat_hash_map<std::pair<uint64_t, uint64_t>, FlatOpMetricsDb>
      sparse_core_metrics_map;

  ConvertSparseCoreDeviceTraceXPlaneToFlatOpMetricsDb(*xplane_,
                                                      sparse_core_metrics_map);

  EXPECT_THAT(sparse_core_metrics_map, SizeIs(0));
}

// Test Case 2: A single module with a single operation.
TEST_F(XPlaneToFlatOpMetricsDbTest, SingleModuleSingleOp) {
  XLineBuilder module_line = plane_builder_->GetOrCreateLine(1);
  module_line.SetName(tsl::profiler::kSparseCoreModuleLineName);

  XLineBuilder op_line = plane_builder_->GetOrCreateLine(2);
  op_line.SetName(tsl::profiler::kSparseCoreOpLineName);

  // Add one module event.
  AddModuleEvent(module_line, "Module1", 1000, 500, 0, 123);

  // Add one op event that falls within the module's timespan.
  AddHloOpEvent(op_line, "Op1", 1100, 200, 1, 2);

  absl::flat_hash_map<std::pair<uint64_t, uint64_t>, FlatOpMetricsDb>
      sparse_core_metrics_map;

  ConvertSparseCoreDeviceTraceXPlaneToFlatOpMetricsDb(*xplane_,
                                                      sparse_core_metrics_map);

  // We expect one entry in the map for key (0, 123).
  ASSERT_THAT(sparse_core_metrics_map, SizeIs(1));
  auto it = sparse_core_metrics_map.find({0, 123});
  ASSERT_NE(it, sparse_core_metrics_map.end());

  const FlatOpMetricsDb& db = it->second;
  // Expect Op1 and IDLE.
  ASSERT_THAT(db.op_instances(), SizeIs(2));

  const FlatOpMetrics& op1 = db.op_instances(0);
  EXPECT_THAT(op1.hlo_name(), Eq("Op1"));
  EXPECT_THAT(op1.time_ps(), Eq(200000));  // 200 ns to ps
  EXPECT_THAT(op1.self_time_ps(), Eq(200000));
  EXPECT_THAT(op1.core_type(), Eq(FlatOpMetrics::SPARSE_CORE));

  const FlatOpMetrics& idle = db.op_instances(1);
  EXPECT_THAT(idle.hlo_name(), Eq("IDLE"));
  EXPECT_THAT(idle.time_ps(),
              Eq(300000));  // Total (500) - Op (200) = 300 ns = 300000 ps
}

// Test Case 3: Multiple modules should be separated correctly.
TEST_F(XPlaneToFlatOpMetricsDbTest, MultipleModules) {
  XLineBuilder module_line = plane_builder_->GetOrCreateLine(1);
  module_line.SetName(tsl::profiler::kSparseCoreModuleLineName);

  XLineBuilder op_line = plane_builder_->GetOrCreateLine(2);
  op_line.SetName(tsl::profiler::kSparseCoreOpLineName);

  // Module 1: Core 0, Start ID 123
  AddModuleEvent(module_line, "Module1", 1000, 500, 0, 123);
  // Module 2: Core 1, Start ID 456
  AddModuleEvent(module_line, "Module2", 2000, 600, 1, 456);

  // Op for Module 1
  AddHloOpEvent(op_line, "Op1", 1100, 100, 1, 2);
  // Op for Module 2
  AddHloOpEvent(op_line, "Op2", 2100, 150, 1, 3);

  absl::flat_hash_map<std::pair<uint64_t, uint64_t>, FlatOpMetricsDb>
      sparse_core_metrics_map;

  ConvertSparseCoreDeviceTraceXPlaneToFlatOpMetricsDb(*xplane_,
                                                      sparse_core_metrics_map);

  ASSERT_THAT(sparse_core_metrics_map, SizeIs(2));

  // Check Module 1
  {
    auto it = sparse_core_metrics_map.find({0, 123});
    ASSERT_NE(it, sparse_core_metrics_map.end());
    const FlatOpMetricsDb& db = it->second;
    ASSERT_THAT(db.op_instances(), SizeIs(2));
    EXPECT_THAT(db.op_instances(0).hlo_name(), Eq("Op1"));
    EXPECT_THAT(db.op_instances(1).hlo_name(), Eq("IDLE"));
    EXPECT_THAT(db.op_instances(1).time_ps(),
                Eq(400000));  // 500 - 100 = 400 ns
  }

  // Check Module 2
  {
    auto it = sparse_core_metrics_map.find({1, 456});
    ASSERT_NE(it, sparse_core_metrics_map.end());
    const FlatOpMetricsDb& db = it->second;
    ASSERT_THAT(db.op_instances(), SizeIs(2));
    EXPECT_THAT(db.op_instances(0).hlo_name(), Eq("Op2"));
    EXPECT_THAT(db.op_instances(1).hlo_name(), Eq("IDLE"));
    EXPECT_THAT(db.op_instances(1).time_ps(),
                Eq(450000));  // 600 - 150 = 450 ns
  }
}

// Test Case 4: Nested operations to verify topological sort and self-time.
TEST_F(XPlaneToFlatOpMetricsDbTest, NestedOpsAndTopologicalSort) {
  XLineBuilder module_line = plane_builder_->GetOrCreateLine(1);
  module_line.SetName(tsl::profiler::kSparseCoreModuleLineName);

  XLineBuilder op_line = plane_builder_->GetOrCreateLine(2);
  op_line.SetName(tsl::profiler::kSparseCoreOpLineName);

  // Add one module event.
  AddModuleEvent(module_line, "Module1", 1000, 1000, 0, 123);

  // Add nested op events.
  // Parent Op: spans from 1100 to 1600 (duration 500)
  AddHloOpEvent(op_line, "ParentOp", 1100, 500, 1, 10);

  // Child Op 1: spans from 1200 to 1300 (duration 100). Nested in ParentOp.
  AddHloOpEvent(op_line, "ChildOp1", 1200, 100, 1, 11);

  // Child Op 2: spans from 1400 to 1550 (duration 150). Nested in ParentOp.
  AddHloOpEvent(op_line, "ChildOp2", 1400, 150, 1, 12);

  absl::flat_hash_map<std::pair<uint64_t, uint64_t>, FlatOpMetricsDb>
      sparse_core_metrics_map;

  ConvertSparseCoreDeviceTraceXPlaneToFlatOpMetricsDb(*xplane_,
                                                      sparse_core_metrics_map);

  ASSERT_THAT(sparse_core_metrics_map, SizeIs(1));
  const FlatOpMetricsDb& db = sparse_core_metrics_map.begin()->second;

  // Expect ParentOp, ChildOp1, ChildOp2, and IDLE.
  // Due to topological sort, Parent should come first, then children.
  ASSERT_THAT(db.op_instances(), SizeIs(4));

  const FlatOpMetrics& op0 = db.op_instances(0);
  EXPECT_THAT(op0.hlo_name(), Eq("ParentOp"));
  EXPECT_THAT(op0.time_ps(), Eq(500000));  // 500 ns
  // Self time = Total (500) - Children (100 + 150) = 250 ns = 250000 ps
  EXPECT_THAT(op0.self_time_ps(), Eq(250000));

  const FlatOpMetrics& op1 = db.op_instances(1);
  const FlatOpMetrics& op2 = db.op_instances(2);

  if (op1.hlo_name() == "ChildOp1") {
    EXPECT_THAT(op1.time_ps(), Eq(100000));
    EXPECT_THAT(op1.self_time_ps(), Eq(100000));
    EXPECT_THAT(op2.hlo_name(), Eq("ChildOp2"));
    EXPECT_THAT(op2.time_ps(), Eq(150000));
    EXPECT_THAT(op2.self_time_ps(), Eq(150000));
  } else {
    EXPECT_THAT(op1.hlo_name(), Eq("ChildOp2"));
    EXPECT_THAT(op1.time_ps(), Eq(150000));
    EXPECT_THAT(op1.self_time_ps(), Eq(150000));
    EXPECT_THAT(op2.hlo_name(), Eq("ChildOp1"));
    EXPECT_THAT(op2.time_ps(), Eq(100000));
    EXPECT_THAT(op2.self_time_ps(), Eq(100000));
  }

  const FlatOpMetrics& idle = db.op_instances(3);
  EXPECT_THAT(idle.hlo_name(), Eq("IDLE"));
  // Total module time (1000) - Total Op time (500) = 500 ns = 500000 ps
  EXPECT_THAT(idle.time_ps(), Eq(500000));
}

// Test Case 5: Ops outside any module should be ignored.
TEST_F(XPlaneToFlatOpMetricsDbTest, IgnoreOpsOutsideModule) {
  XLineBuilder module_line = plane_builder_->GetOrCreateLine(1);
  module_line.SetName(tsl::profiler::kSparseCoreModuleLineName);

  XLineBuilder op_line = plane_builder_->GetOrCreateLine(2);
  op_line.SetName(tsl::profiler::kSparseCoreOpLineName);

  // Add one module event.
  AddModuleEvent(module_line, "Module1", 1000, 500, 0, 123);

  // Add an op event that falls WITHIN the module's timespan.
  AddHloOpEvent(op_line, "ValidOp", 1100, 100, 1, 2);

  // Add an op event that falls OUTSIDE the module's timespan.
  AddHloOpEvent(op_line, "IgnoredOp", 2000, 100, 1, 3);

  absl::flat_hash_map<std::pair<uint64_t, uint64_t>, FlatOpMetricsDb>
      sparse_core_metrics_map;

  ConvertSparseCoreDeviceTraceXPlaneToFlatOpMetricsDb(*xplane_,
                                                      sparse_core_metrics_map);

  ASSERT_THAT(sparse_core_metrics_map, SizeIs(1));
  const FlatOpMetricsDb& db = sparse_core_metrics_map.begin()->second;

  // Expect only ValidOp and IDLE.
  ASSERT_THAT(db.op_instances(), SizeIs(2));
  EXPECT_THAT(db.op_instances(0).hlo_name(), Eq("ValidOp"));
  EXPECT_THAT(db.op_instances(1).hlo_name(), Eq("IDLE"));
}

// Test Case: Operations that partially overlap with a module's timespan
// should be ignored.
TEST_F(XPlaneToFlatOpMetricsDbTest, PartiallyOverlappingOpsIgnored) {
  XLineBuilder module_line = plane_builder_->GetOrCreateLine(1);
  module_line.SetName(tsl::profiler::kSparseCoreModuleLineName);

  XLineBuilder op_line = plane_builder_->GetOrCreateLine(2);
  op_line.SetName(tsl::profiler::kSparseCoreOpLineName);

  // Add one module event: 1000 ns to 2000 ns.
  AddModuleEvent(module_line, "Module1", 1000, 1000, 0, 123);

  // Add a valid op event that falls WITHIN the module's timespan.
  AddHloOpEvent(op_line, "ValidOp", 1100, 100, 1, 2);

  // Add an op event that starts BEFORE the module but ends WITHIN it.
  AddHloOpEvent(op_line, "PartialStartOp", 900, 200, 1, 3);

  // Add an op event that starts WITHIN the module but ends AFTER it.
  AddHloOpEvent(op_line, "PartialEndOp", 1900, 200, 1, 4);

  absl::flat_hash_map<std::pair<uint64_t, uint64_t>, FlatOpMetricsDb>
      sparse_core_metrics_map;

  ConvertSparseCoreDeviceTraceXPlaneToFlatOpMetricsDb(*xplane_,
                                                      sparse_core_metrics_map);

  ASSERT_THAT(sparse_core_metrics_map, SizeIs(1));
  const FlatOpMetricsDb& db = sparse_core_metrics_map.begin()->second;

  // Expect only ValidOp and IDLE.
  ASSERT_THAT(db.op_instances(), SizeIs(2));
  EXPECT_THAT(db.op_instances(0).hlo_name(), Eq("ValidOp"));
  EXPECT_THAT(db.op_instances(1).hlo_name(), Eq("IDLE"));
  // Total module time (1000) - ValidOp (100) = 900 ns = 900000 ps.
  EXPECT_THAT(db.op_instances(1).time_ps(), Eq(900000));
}

// Test Case 6: Multiple op lines should be processed correctly.
TEST_F(XPlaneToFlatOpMetricsDbTest, MultipleOpLines) {
  XLineBuilder module_line = plane_builder_->GetOrCreateLine(1);
  module_line.SetName(tsl::profiler::kSparseCoreModuleLineName);

  XLineBuilder op_line1 = plane_builder_->GetOrCreateLine(2);
  op_line1.SetName(tsl::profiler::kSparseCoreOpLineName);

  XLineBuilder op_line2 = plane_builder_->GetOrCreateLine(3);
  op_line2.SetName(tsl::profiler::kSparseCoreOpLineName);

  // Add one module event.
  AddModuleEvent(module_line, "Module1", 1000, 1000, 0, 123);

  // Add ops on line 1.
  AddHloOpEvent(op_line1, "OpLine1_1", 1100, 100, 1, 2);
  AddHloOpEvent(op_line1, "OpLine1_2", 1300, 100, 1, 3);

  // Add ops on line 2.
  AddHloOpEvent(op_line2, "OpLine2_1", 1200, 100, 1, 4);
  AddHloOpEvent(op_line2, "OpLine2_2", 1400, 100, 1, 5);

  absl::flat_hash_map<std::pair<uint64_t, uint64_t>, FlatOpMetricsDb>
      sparse_core_metrics_map;

  ConvertSparseCoreDeviceTraceXPlaneToFlatOpMetricsDb(*xplane_,
                                                      sparse_core_metrics_map);

  ASSERT_THAT(sparse_core_metrics_map, SizeIs(1));
  const FlatOpMetricsDb& db = sparse_core_metrics_map.begin()->second;

  // Expect 4 ops and IDLE.
  ASSERT_THAT(db.op_instances(), SizeIs(5));

  std::vector<std::string> op_names;
  for (int i = 0; i < 4; ++i) {
    op_names.push_back(db.op_instances(i).hlo_name());
  }
  EXPECT_THAT(op_names,
              ::testing::UnorderedElementsAre("OpLine1_1", "OpLine1_2",
                                              "OpLine2_1", "OpLine2_2"));

  EXPECT_THAT(db.op_instances(4).hlo_name(), Eq("IDLE"));
  // Total module time (1000) - Ops (100*4 = 400) = 600 ns = 600000 ps.
  EXPECT_THAT(db.op_instances(4).time_ps(), Eq(600000));
}

// Test Case 7: Deeply nested operations to further verify self-time and tree
// structure.
TEST_F(XPlaneToFlatOpMetricsDbTest, DeeplyNestedOps) {
  XLineBuilder module_line = plane_builder_->GetOrCreateLine(1);
  module_line.SetName(tsl::profiler::kSparseCoreModuleLineName);

  XLineBuilder op_line = plane_builder_->GetOrCreateLine(2);
  op_line.SetName(tsl::profiler::kSparseCoreOpLineName);

  // Add one module event.
  AddModuleEvent(module_line, "Module1", 1000, 2000, 0, 123);

  // Level 1 Parent: 1100 - 1900 (duration 800)
  AddHloOpEvent(op_line, "L1_Parent", 1100, 800, 1, 10);

  // Level 2 Child: 1200 - 1700 (duration 500)
  AddHloOpEvent(op_line, "L2_Child", 1200, 500, 1, 11);

  // Level 3 Grandchild: 1300 - 1500 (duration 200)
  AddHloOpEvent(op_line, "L3_Grandchild", 1300, 200, 1, 12);

  absl::flat_hash_map<std::pair<uint64_t, uint64_t>, FlatOpMetricsDb>
      sparse_core_metrics_map;

  ConvertSparseCoreDeviceTraceXPlaneToFlatOpMetricsDb(*xplane_,
                                                      sparse_core_metrics_map);

  ASSERT_THAT(sparse_core_metrics_map, SizeIs(1));
  const FlatOpMetricsDb& db = sparse_core_metrics_map.begin()->second;

  ASSERT_THAT(db.op_instances(), SizeIs(4));  // L1, L2, L3, IDLE

  // Topological sort should put L1 first, then L2, then L3.
  const FlatOpMetrics& l1 = db.op_instances(0);
  EXPECT_THAT(l1.hlo_name(), Eq("L1_Parent"));
  EXPECT_THAT(l1.time_ps(), Eq(800000));
  // Self time = 800 - 500 = 300 ns = 300000 ps
  EXPECT_THAT(l1.self_time_ps(), Eq(300000));

  const FlatOpMetrics& l2 = db.op_instances(1);
  EXPECT_THAT(l2.hlo_name(), Eq("L2_Child"));
  EXPECT_THAT(l2.time_ps(), Eq(500000));
  // Self time = 500 - 200 = 300 ns = 300000 ps
  EXPECT_THAT(l2.self_time_ps(), Eq(300000));

  const FlatOpMetrics& l3 = db.op_instances(2);
  EXPECT_THAT(l3.hlo_name(), Eq("L3_Grandchild"));
  EXPECT_THAT(l3.time_ps(), Eq(200000));
  EXPECT_THAT(l3.self_time_ps(), Eq(200000));

  const FlatOpMetrics& idle = db.op_instances(3);
  EXPECT_THAT(idle.hlo_name(), Eq("IDLE"));
  // Total (2000) - L1 (800) = 1200 ns = 1200000 ps
  EXPECT_THAT(idle.time_ps(), Eq(1200000));
}

TEST_F(XPlaneToFlatOpMetricsDbTest, ModuleEventMissingStatsIgnored) {
  XLineBuilder module_line = plane_builder_->GetOrCreateLine(1);
  module_line.SetName(tsl::profiler::kSparseCoreModuleLineName);

  XLineBuilder op_line = plane_builder_->GetOrCreateLine(2);
  op_line.SetName(tsl::profiler::kSparseCoreOpLineName);

  // Add a module event with ONLY offload_core_id.
  XEventBuilder module_builder = module_line.AddEvent(
      *plane_builder_->GetOrCreateEventMetadata("Module1"));
  module_builder.SetTimestampNs(1000);
  module_builder.SetDurationNs(500);
  module_builder.AddStatValue(*plane_builder_->GetOrCreateStatMetadata(
                                  GetStatTypeStr(StatType::kDeviceOffsetPs)),
                              int64_t{1000000});
  module_builder.AddStatValue(*plane_builder_->GetOrCreateStatMetadata(
                                  GetStatTypeStr(StatType::kDeviceDurationPs)),
                              int64_t{500000});
  module_builder.AddStatValue(*plane_builder_->GetOrCreateStatMetadata(
                                  GetStatTypeStr(StatType::kOffloadCoreId)),
                              uint64_t{0});
  // Missing kTcOffloadStartId!

  // Add an op event that falls WITHIN the module's timespan.
  AddHloOpEvent(op_line, "ValidOp", 1100, 100, 1, 2);

  absl::flat_hash_map<std::pair<uint64_t, uint64_t>, FlatOpMetricsDb>
      sparse_core_metrics_map;

  ConvertSparseCoreDeviceTraceXPlaneToFlatOpMetricsDb(*xplane_,
                                                      sparse_core_metrics_map);

  // Since the module was ignored due to missing stat, the op should also be
  // ignored.
  EXPECT_TRUE(sparse_core_metrics_map.empty());
}

// Test Case: A module event missing kOffloadCoreId should be ignored.
TEST_F(XPlaneToFlatOpMetricsDbTest, ModuleEventMissingOffloadCoreIdIgnored) {
  XLineBuilder module_line = plane_builder_->GetOrCreateLine(1);
  module_line.SetName(tsl::profiler::kSparseCoreModuleLineName);

  XLineBuilder op_line = plane_builder_->GetOrCreateLine(2);
  op_line.SetName(tsl::profiler::kSparseCoreOpLineName);

  // Add a module event with ONLY kTcOffloadStartId.
  XEventBuilder module_builder = module_line.AddEvent(
      *plane_builder_->GetOrCreateEventMetadata("Module1"));
  module_builder.SetTimestampNs(1000);
  module_builder.SetDurationNs(500);
  module_builder.AddStatValue(*plane_builder_->GetOrCreateStatMetadata(
                                  GetStatTypeStr(StatType::kDeviceOffsetPs)),
                              int64_t{1000000});
  module_builder.AddStatValue(*plane_builder_->GetOrCreateStatMetadata(
                                  GetStatTypeStr(StatType::kDeviceDurationPs)),
                              int64_t{500000});
  module_builder.AddStatValue(*plane_builder_->GetOrCreateStatMetadata(
                                  GetStatTypeStr(StatType::kTcOffloadStartId)),
                              uint64_t{123});
  // Missing kOffloadCoreId!

  // Add an op event that falls WITHIN the module's timespan.
  AddHloOpEvent(op_line, "ValidOp", 1100, 100, 1, 2);

  absl::flat_hash_map<std::pair<uint64_t, uint64_t>, FlatOpMetricsDb>
      sparse_core_metrics_map;

  ConvertSparseCoreDeviceTraceXPlaneToFlatOpMetricsDb(*xplane_,
                                                      sparse_core_metrics_map);

  // Since the module was ignored due to missing stat, the op should also be
  // ignored.
  EXPECT_TRUE(sparse_core_metrics_map.empty());
}

TEST_F(XPlaneToFlatOpMetricsDbTest, GpuEquivalenceTest) {
  XLineBuilder stream1 = plane_builder_->GetOrCreateLine(1);
  stream1.SetName("Stream #1");
  XLineBuilder stream2 = plane_builder_->GetOrCreateLine(2);
  stream2.SetName("Stream #2");

  // XlaOp event on stream1
  XEventBuilder xla_op =
      stream1.AddEvent(*plane_builder_->GetOrCreateEventMetadata("xla_op_1"));
  xla_op.SetTimestampNs(1000);
  xla_op.SetDurationNs(500);
  xla_op.AddStatValue(*plane_builder_->GetOrCreateStatMetadata(
                          GetStatTypeStr(StatType::kDeviceOffsetPs)),
                      int64_t{1000000});
  xla_op.AddStatValue(*plane_builder_->GetOrCreateStatMetadata(
                          GetStatTypeStr(StatType::kDeviceDurationPs)),
                      int64_t{500000});
  tsl::profiler::XStatsBuilder<XEventMetadata> xla_op_meta(
      plane_builder_->GetOrCreateEventMetadata("xla_op_1"),
      plane_builder_.get());
  xla_op_meta.AddStatValue(*plane_builder_->GetOrCreateStatMetadata(
                               GetStatTypeStr(StatType::kProgramId)),
                           uint64_t{10});
  xla_op_meta.AddStatValue(*plane_builder_->GetOrCreateStatMetadata(
                               GetStatTypeStr(StatType::kSymbolId)),
                           uint64_t{20});
  xla_op_meta.AddStatValue(*plane_builder_->GetOrCreateStatMetadata(
                               GetStatTypeStr(StatType::kHloOp)),
                           "hlo_op_1");
  xla_op_meta.AddStatValue(*plane_builder_->GetOrCreateStatMetadata(
                               GetStatTypeStr(StatType::kHloCategory)),
                           "hlo_cat_1");
  xla_op_meta.AddStatValue(*plane_builder_->GetOrCreateStatMetadata(
                               GetStatTypeStr(StatType::kFlops)),
                           int64_t{12345});
  xla_op_meta.AddStatValue(*plane_builder_->GetOrCreateStatMetadata(
                               GetStatTypeStr(StatType::kBytesAccessed)),
                           int64_t{67890});

  // Nested TfOp on stream 1
  XEventBuilder child_tf_op = stream1.AddEvent(
      *plane_builder_->GetOrCreateEventMetadata("child_tf_op"));
  child_tf_op.SetTimestampNs(1100);
  child_tf_op.SetDurationNs(100);
  child_tf_op.AddStatValue(*plane_builder_->GetOrCreateStatMetadata(
                               GetStatTypeStr(StatType::kDeviceOffsetPs)),
                           int64_t{1100000});
  child_tf_op.AddStatValue(*plane_builder_->GetOrCreateStatMetadata(
                               GetStatTypeStr(StatType::kDeviceDurationPs)),
                           int64_t{100000});
  tsl::profiler::XStatsBuilder<XEventMetadata> child_tf_op_meta(
      plane_builder_->GetOrCreateEventMetadata("child_tf_op"),
      plane_builder_.get());
  child_tf_op_meta.AddStatValue(
      *plane_builder_->GetOrCreateStatMetadata(GetStatTypeStr(StatType::kTfOp)),
      "TfOpCategory:ChildTfOpName");

  // TfOp event on stream2 (overlapping with idle gap on stream1)
  XEventBuilder tf_op =
      stream2.AddEvent(*plane_builder_->GetOrCreateEventMetadata("tf_op_1"));
  tf_op.SetTimestampNs(2000);
  tf_op.SetDurationNs(600);
  tf_op.AddStatValue(*plane_builder_->GetOrCreateStatMetadata(
                         GetStatTypeStr(StatType::kDeviceOffsetPs)),
                     int64_t{2000000});
  tf_op.AddStatValue(*plane_builder_->GetOrCreateStatMetadata(
                         GetStatTypeStr(StatType::kDeviceDurationPs)),
                     int64_t{600000});
  tsl::profiler::XStatsBuilder<XEventMetadata> tf_op_meta(
      plane_builder_->GetOrCreateEventMetadata("tf_op_1"),
      plane_builder_.get());
  tf_op_meta.AddStatValue(
      *plane_builder_->GetOrCreateStatMetadata(GetStatTypeStr(StatType::kTfOp)),
      "TfOpCategory:TfOpName");
  tf_op.AddStatValue(*plane_builder_->GetOrCreateStatMetadata(
                         GetStatTypeStr(StatType::kIsEager)),
                     int64_t{1});

  HloModuleMap hlo_module_map;

  OpMetricsDb legacy_db =
      ConvertDeviceTraceXPlaneToOpMetricsDb(*xplane_, hlo_module_map);
  FlatOpMetricsDb new_db =
      ConvertDeviceTraceXPlaneToFlatOpMetricsDb(*xplane_, hlo_module_map);

  absl::flat_hash_map<std::string, const FlatOpMetrics*> new_db_map;
  for (const auto& op : new_db.op_instances()) {
    new_db_map[op.hlo_name()] = &op;
  }

  for (const auto& legacy_op : legacy_db.metrics_db()) {
    auto it = new_db_map.find(legacy_op.name());
    ASSERT_NE(it, new_db_map.end())
        << "Missing op in flat DB: " << legacy_op.name();
    const FlatOpMetrics* new_op = it->second;

    EXPECT_EQ(legacy_op.occurrences(), new_op->occurrences())
        << legacy_op.name();
    EXPECT_EQ(legacy_op.time_ps(), new_op->time_ps()) << legacy_op.name();
    EXPECT_EQ(legacy_op.self_time_ps(), new_op->self_time_ps())
        << legacy_op.name();
    EXPECT_EQ(legacy_op.flops(), new_op->flops()) << legacy_op.name();
    EXPECT_EQ(legacy_op.bytes_accessed(), new_op->bytes_accessed())
        << legacy_op.name();
    EXPECT_EQ(legacy_op.is_eager(), new_op->is_eager()) << legacy_op.name();
    EXPECT_EQ(legacy_op.category(), new_op->category()) << legacy_op.name();
    EXPECT_EQ(legacy_op.provenance(), new_op->provenance()) << legacy_op.name();
  }

  EXPECT_EQ(legacy_db.metrics_db_size(), new_db.op_instances_size());
  EXPECT_EQ(legacy_db.total_op_time_ps(), new_db.total_op_time_ps());
  EXPECT_EQ(legacy_db.total_time_ps(), new_db.total_time_ps());
}

TEST_F(XPlaneToFlatOpMetricsDbTest, HostInfeedEnqueueMetrics) {
  XLineBuilder host_line = plane_builder_->GetOrCreateLine(1);
  host_line.SetName("Host Threads");

  auto add_infeed = [&](int64_t start_ns, int64_t duration_ns) {
    XEventBuilder builder = host_line.AddEvent(
        *plane_builder_->GetOrCreateEventMetadata("InfeedEnqueueTuple"));
    builder.SetTimestampNs(start_ns);
    builder.SetDurationNs(duration_ns);
    builder.AddStatValue(
        *plane_builder_->GetOrCreateStatMetadata(
            GetStatTypeStr(StatType::kTfOp)),
        "InfeedEnqueueTuple:InfeedEnqueueTuple");
  };

  add_infeed(1000, 1000);
  add_infeed(3000, 1000);
  add_infeed(6000, 1000);

  FlatOpMetricsDb db = ConvertHostThreadsXPlaneToFlatOpMetricsDb(*xplane_);

  EXPECT_EQ(db.total_host_infeed_enq_duration_ps(), 2000000);
  EXPECT_EQ(db.total_host_infeed_enq_start_timestamp_ps_diff(), 5000000);
}

}  // namespace
}  // namespace profiler
}  // namespace tensorflow
