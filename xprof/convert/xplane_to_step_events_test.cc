/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "xprof/convert/xplane_to_step_events.h"

#include <cstdint>
#include <vector>

#include "<gtest/gtest.h>"
#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/profiler/utils/group_events.h"
#include "xla/tsl/profiler/utils/preprocess_xplane.h"
#include "xla/tsl/profiler/utils/tf_xplane_visitor.h"
#include "xla/tsl/profiler/utils/timespan.h"
#include "xla/tsl/profiler/utils/xplane_builder.h"
#include "xla/tsl/profiler/utils/xplane_schema.h"
#include "xla/tsl/profiler/utils/xplane_test_utils.h"
#include "tsl/profiler/lib/context_types.h"
#include "tsl/profiler/protobuf/xplane.pb.h"
#include "plugin/xprof/protobuf/op_metrics.pb.h"
#include "xprof/utils/derived_timeline.h"
#include "xprof/utils/event_span.h"
#include "xprof/utils/op_metrics_db_utils.h"

namespace tensorflow {
namespace profiler {
namespace {

using ::tsl::profiler::HostEventType;
using ::tsl::profiler::XEventBuilder;
using ::tsl::profiler::XLineBuilder;
using ::tsl::profiler::XPlaneBuilder;
using ::tsl::profiler::XStatsBuilder;

const FlatOpMetrics* FindFlatOp(const FlatOpMetricsDb& db,
                               absl::string_view name) {
  for (const auto& op : db.op_instances()) {
    if (op.hlo_name() == name) return &op;
  }
  return nullptr;
}

// Tests with a sample profile with two steps captured on the host but only one
// step on the device. On the host, each step consists of TraceContext ->
// FunctionRun -> ExecutorState::Process -> matmul. On the host, each step
// consists of matmul. The host's step db should be created only for the step
// observed on the host.
TEST(ConvertXPlaneToOpStats, CpuOnlyStepDbTest) {
  constexpr int64_t kFirstStepNum = 123;
  constexpr int64_t kSecondStepNum = 456;
  constexpr int64_t kFirstStepId = 0;
  constexpr int64_t kSecondStepId = 1;
  constexpr int64_t kFirstCorrelationId = 100;
  constexpr int64_t kSecondCorrelationId = 200;

  XSpace space;
  XPlane* host_plane = tsl::profiler::GetOrCreateHostXPlane(&space);
  XPlaneBuilder host_plane_builder(host_plane);
  host_plane_builder.ReserveLines(2);

  auto main_thread = host_plane_builder.GetOrCreateLine(0);
  CreateXEvent(&host_plane_builder, &main_thread, HostEventType::kTraceContext,
               0, 100, {{StatType::kStepNum, kFirstStepNum}});
  CreateXEvent(&host_plane_builder, &main_thread, HostEventType::kFunctionRun,
               10, 90,
               {{StatType::kStepId, kFirstStepId},
                {StatType::kProducerType, int64_t{1}},
                {StatType::kProducerId, kFirstStepId}});
  CreateXEvent(&host_plane_builder, &main_thread, HostEventType::kTraceContext,
               300, 100, {{StatType::kStepNum, kSecondStepNum}});
  CreateXEvent(&host_plane_builder, &main_thread, HostEventType::kFunctionRun,
               310, 90,
               {{StatType::kStepId, kSecondStepId},
                {StatType::kProducerType, int64_t{1}},
                {StatType::kProducerId, kSecondStepId}});

  auto tf_executor_thread = host_plane_builder.GetOrCreateLine(1);
  CreateXEvent(&host_plane_builder, &tf_executor_thread,
               HostEventType::kExecutorStateProcess, 20, 20,
               {{StatType::kStepId, kFirstStepId},
                {StatType::kConsumerType, int64_t{1}},
                {StatType::kConsumerId, kFirstStepId}});
  CreateXEvent(&host_plane_builder, &tf_executor_thread, "matmul", 30, 10,
               {{StatType::kCorrelationId, kFirstCorrelationId}});
  CreateXEvent(&host_plane_builder, &tf_executor_thread,
               HostEventType::kExecutorStateProcess, 320, 20,
               {{StatType::kStepId, kSecondStepId},
                {StatType::kConsumerType, int64_t{1}},
                {StatType::kConsumerId, kSecondStepId}});
  CreateXEvent(&host_plane_builder, &tf_executor_thread, "matmul", 330, 10,
               {{StatType::kCorrelationId, kSecondCorrelationId}});

  XPlane* device_plane = space.add_planes();
  XPlaneBuilder device_plane_builder(device_plane);
  device_plane_builder.ReserveLines(1);

  auto stream = device_plane_builder.GetOrCreateLine(0);
  CreateXEvent(&device_plane_builder, &stream, "matmul", 50, 40,
               {{StatType::kCorrelationId, kFirstCorrelationId}});

  tsl::profiler::EventForest event_forest;
  tsl::profiler::GroupTfEvents(&space, &event_forest);
  DeriveStepEventsFromGroups(event_forest.GetGroupMetadataMap(), device_plane);
  StepEvents device_step_events =
      ConvertDeviceTraceXPlaneToStepEvents(*device_plane);
  EXPECT_EQ(device_step_events.size(), 1);
  EXPECT_EQ(device_step_events[0].Events().size(), 1);
  StepEvents host_step_events =
      ConvertHostThreadsXPlaneToStepEvents(*host_plane, &device_step_events);
  // Should contain only the step which is also present on the device.
  EXPECT_EQ(host_step_events.size(), 1);
  // TraceContext should be added as a step marker.
  EXPECT_EQ(host_step_events[0].Markers().size(), 1);
  // FunctionRun shouldn't be added.
  EXPECT_EQ(host_step_events[0].Events().size(), 2);
}

TEST(ConvertXPlaneToStepEvents, TpuDevicePlaneToStepEvents) {
  XPlane raw_plane;
  XPlaneBuilder plane(&raw_plane);
  int64_t device_id = 1;
  plane.SetId(device_id);
  plane.SetName("/device:TPU:0");
  XLineBuilder op_line = plane.GetOrCreateLine(0);
  op_line.SetName(tsl::profiler::kXlaOpLineName);
  const XStatMetadata& program_id_stat =
      *plane.GetOrCreateStatMetadata(GetStatTypeStr(StatType::kProgramId));
  const XStatMetadata& symbol_id_stat =
      *plane.GetOrCreateStatMetadata(GetStatTypeStr(StatType::kSymbolId));
  const XStatMetadata& group_id_stat =
      *plane.GetOrCreateStatMetadata(GetStatTypeStr(StatType::kGroupId));
  {
    XEventMetadata* event_metadata =
        plane.GetOrCreateEventMetadata("op_long_name");
    event_metadata->set_display_name("op_name");
    XStatsBuilder<XEventMetadata> stats(event_metadata, &plane);
    stats.AddStatValue(program_id_stat, 1);
    stats.AddStatValue(symbol_id_stat, 1);
    {
      XEventBuilder event = op_line.AddEvent(*event_metadata);
      event.SetOffsetPs(0);
      event.SetDurationPs(50);
      event.AddStatValue(group_id_stat, 1);
    }
    {
      XEventBuilder event = op_line.AddEvent(*event_metadata);
      event.SetOffsetPs(100);
      event.SetDurationPs(50);
      event.AddStatValue(group_id_stat, 2);
    }
  }
  {
    XEventMetadata* event_metadata =
        plane.GetOrCreateEventMetadata("op_long_name2");
    event_metadata->set_display_name("op_name2");
    XStatsBuilder<XEventMetadata> stats(event_metadata, &plane);
    stats.AddStatValue(program_id_stat, 1);
    stats.AddStatValue(symbol_id_stat, 2);
    XEventBuilder event = op_line.AddEvent(*event_metadata);
    event.SetOffsetPs(50);
    event.SetDurationPs(50);
    event.AddStatValue(group_id_stat, 1);
  }
  XLineBuilder step_line = plane.GetOrCreateLine(1);
  step_line.SetName(tsl::profiler::kStepLineName);
  {
    XEventMetadata* event_metadata = plane.CreateEventMetadata();
    XStatsBuilder<XEventMetadata> stats(event_metadata, &plane);
    {
      XEventBuilder event = step_line.AddEvent(*event_metadata);
      event.SetOffsetPs(0);
      event.SetDurationPs(100);
      event.AddStatValue(group_id_stat, 1);
    }
    {
      XEventBuilder event = step_line.AddEvent(*event_metadata);
      event.SetOffsetPs(100);
      event.SetDurationPs(100);
      event.AddStatValue(group_id_stat, 2);
    }
  }

  StepEvents step_events = ConvertDeviceTraceXPlaneToStepEvents(raw_plane);
  EXPECT_EQ(step_events.size(), 2);
  EXPECT_TRUE(step_events.contains(1));
  StepDetails step_1 = step_events[/*group_id=*/1];
  ASSERT_TRUE(step_1.PerCoreOpMetricsDb().contains(device_id));
  EXPECT_EQ(step_1.PerCoreOpMetricsDb().at(device_id).metrics_db_size(), 2);
  EXPECT_EQ(step_1.Markers().size(), 1);
  EXPECT_TRUE(step_events.contains(2));
  StepDetails step_2 = step_events[/*group_id=*/2];
  ASSERT_TRUE(step_2.PerCoreOpMetricsDb().contains(device_id));
  EXPECT_EQ(step_2.PerCoreOpMetricsDb().at(device_id).metrics_db_size(), 1);
  EXPECT_EQ(step_2.Markers().size(), 1);
}

TEST(ConvertXPlaneToStepEvents, TpuDevicePlaneToFlatStepEvents) {
  XPlane raw_plane;
  XPlaneBuilder plane(&raw_plane);
  int64_t device_id = 1;
  plane.SetId(device_id);
  plane.SetName("/device:TPU:0");
  XLineBuilder op_line = plane.GetOrCreateLine(0);
  op_line.SetName(tsl::profiler::kXlaOpLineName);
  const XStatMetadata& program_id_stat =
      *plane.GetOrCreateStatMetadata(GetStatTypeStr(StatType::kProgramId));
  const XStatMetadata& symbol_id_stat =
      *plane.GetOrCreateStatMetadata(GetStatTypeStr(StatType::kSymbolId));
  const XStatMetadata& group_id_stat =
      *plane.GetOrCreateStatMetadata(GetStatTypeStr(StatType::kGroupId));
    // Op 1 (symbol 1)
    XEventMetadata* op1_meta = plane.GetOrCreateEventMetadata("op1_long");
    op1_meta->set_display_name("op1");
    {
      XStatsBuilder<XEventMetadata> stats(op1_meta, &plane);
      stats.AddStatValue(program_id_stat, 1);
      stats.AddStatValue(symbol_id_stat, 1);
    }

    // Op 2 (symbol 2)
    XEventMetadata* op2_meta = plane.GetOrCreateEventMetadata("op2_long");
    op2_meta->set_display_name("op2");
    {
      XStatsBuilder<XEventMetadata> stats(op2_meta, &plane);
      stats.AddStatValue(program_id_stat, 1);
      stats.AddStatValue(symbol_id_stat, 2);
    }

    // Op 3 (symbol 3)
    XEventMetadata* op3_meta = plane.GetOrCreateEventMetadata("op3_long");
    op3_meta->set_display_name("op3");
    {
      XStatsBuilder<XEventMetadata> stats(op3_meta, &plane);
      stats.AddStatValue(program_id_stat, 1);
      stats.AddStatValue(symbol_id_stat, 3);
    }

    // Nested 1 (symbol 4)
    XEventMetadata* nested1_meta =
        plane.GetOrCreateEventMetadata("nested1_long");
    nested1_meta->set_display_name("nested1");
    {
      XStatsBuilder<XEventMetadata> stats(nested1_meta, &plane);
      stats.AddStatValue(program_id_stat, 1);
      stats.AddStatValue(symbol_id_stat, 4);
    }

    // Nested 2 (symbol 5)
    XEventMetadata* nested2_meta =
        plane.GetOrCreateEventMetadata("nested2_long");
    nested2_meta->set_display_name("nested2");
    {
      XStatsBuilder<XEventMetadata> stats(nested2_meta, &plane);
      stats.AddStatValue(program_id_stat, 1);
      stats.AddStatValue(symbol_id_stat, 5);
    }

    // Nested 3 (symbol 6)
    XEventMetadata* nested3_meta =
        plane.GetOrCreateEventMetadata("nested3_long");
    nested3_meta->set_display_name("nested3");
    {
      XStatsBuilder<XEventMetadata> stats(nested3_meta, &plane);
      stats.AddStatValue(program_id_stat, 1);
      stats.AddStatValue(symbol_id_stat, 6);
    }

    // Nested 4 (symbol 7)
    XEventMetadata* nested4_meta =
        plane.GetOrCreateEventMetadata("nested4_long");
    nested4_meta->set_display_name("nested4");
    {
      XStatsBuilder<XEventMetadata> stats(nested4_meta, &plane);
      stats.AddStatValue(program_id_stat, 1);
      stats.AddStatValue(symbol_id_stat, 7);
    }

    // Group 1 events
    // op1 (0-50)
    {
      XEventBuilder event = op_line.AddEvent(*op1_meta);
      event.SetOffsetPs(0);
      event.SetDurationPs(50);
      event.AddStatValue(group_id_stat, 1);
    }
    // nested1 (10-30)
    {
      XEventBuilder event = op_line.AddEvent(*nested1_meta);
      event.SetOffsetPs(10);
      event.SetDurationPs(20);
      event.AddStatValue(group_id_stat, 1);
    }
    // op2 (50-100)
    {
      XEventBuilder event = op_line.AddEvent(*op2_meta);
      event.SetOffsetPs(50);
      event.SetDurationPs(50);
      event.AddStatValue(group_id_stat, 1);
    }
    // nested2 (60-90)
    {
      XEventBuilder event = op_line.AddEvent(*nested2_meta);
      event.SetOffsetPs(60);
      event.SetDurationPs(30);
      event.AddStatValue(group_id_stat, 1);
    }

    // Group 2 events
    // op1 (100-150)
    {
      XEventBuilder event = op_line.AddEvent(*op1_meta);
      event.SetOffsetPs(100);
      event.SetDurationPs(50);
      event.AddStatValue(group_id_stat, 2);
    }
    // nested3 (110-140)
    {
      XEventBuilder event = op_line.AddEvent(*nested3_meta);
      event.SetOffsetPs(110);
      event.SetDurationPs(30);
      event.AddStatValue(group_id_stat, 2);
    }
    // op3 (150-200)
    {
      XEventBuilder event = op_line.AddEvent(*op3_meta);
      event.SetOffsetPs(150);
      event.SetDurationPs(50);
      event.AddStatValue(group_id_stat, 2);
    }
    // nested4 (160-190)
    {
      XEventBuilder event = op_line.AddEvent(*nested4_meta);
      event.SetOffsetPs(160);
      event.SetDurationPs(30);
      event.AddStatValue(group_id_stat, 2);
    }
  XLineBuilder step_line = plane.GetOrCreateLine(1);
  step_line.SetName(tsl::profiler::kStepLineName);
  {
    XEventMetadata* event_metadata = plane.CreateEventMetadata();
    XStatsBuilder<XEventMetadata> stats(event_metadata, &plane);
    {
      XEventBuilder event = step_line.AddEvent(*event_metadata);
      event.SetOffsetPs(0);
      event.SetDurationPs(100);
      event.AddStatValue(group_id_stat, 1);
    }
    {
      XEventBuilder event = step_line.AddEvent(*event_metadata);
      event.SetOffsetPs(100);
      event.SetDurationPs(100);
      event.AddStatValue(group_id_stat, 2);
    }
  }

  StepEvents step_events = ConvertDeviceTraceXPlaneToStepEvents(
      raw_plane, /*for_flat_profile=*/true);
  EXPECT_EQ(step_events.size(), 2);
  EXPECT_TRUE(step_events.contains(1));
  StepDetails step_1 = step_events[/*group_id=*/1];
  EXPECT_FALSE(step_1.PerCoreOpMetricsDb().contains(device_id));
  ASSERT_TRUE(step_1.PerCoreFlatOpMetricsDb().contains(device_id));

  const auto& flat_db1 = step_1.PerCoreFlatOpMetricsDb().at(device_id);
  EXPECT_EQ(flat_db1.op_instances_size(), 4);

  const auto* op1_g1 = FindFlatOp(flat_db1, "op1");
  ASSERT_NE(op1_g1, nullptr);
  EXPECT_EQ(op1_g1->time_ps(), 50);
  EXPECT_EQ(op1_g1->self_time_ps(), 30);  // 50 - 20 (nested1)

  const auto* nested1_g1 = FindFlatOp(flat_db1, "nested1");
  ASSERT_NE(nested1_g1, nullptr);
  EXPECT_EQ(nested1_g1->time_ps(), 20);
  EXPECT_EQ(nested1_g1->self_time_ps(), 20);

  const auto* op2_g1 = FindFlatOp(flat_db1, "op2");
  ASSERT_NE(op2_g1, nullptr);
  EXPECT_EQ(op2_g1->time_ps(), 50);
  EXPECT_EQ(op2_g1->self_time_ps(), 20);  // 50 - 30 (nested2)

  const auto* nested2_g1 = FindFlatOp(flat_db1, "nested2");
  ASSERT_NE(nested2_g1, nullptr);
  EXPECT_EQ(nested2_g1->time_ps(), 30);
  EXPECT_EQ(nested2_g1->self_time_ps(), 30);

  EXPECT_TRUE(step_events.contains(2));
  StepDetails step_2 = step_events[/*group_id=*/2];
  EXPECT_FALSE(step_2.PerCoreOpMetricsDb().contains(device_id));
  ASSERT_TRUE(step_2.PerCoreFlatOpMetricsDb().contains(device_id));

  const auto& flat_db2 = step_2.PerCoreFlatOpMetricsDb().at(device_id);
  EXPECT_EQ(flat_db2.op_instances_size(), 4);

  const auto* op1_g2 = FindFlatOp(flat_db2, "op1");
  ASSERT_NE(op1_g2, nullptr);
  EXPECT_EQ(op1_g2->time_ps(), 50);
  EXPECT_EQ(op1_g2->self_time_ps(), 20);  // 50 - 30 (nested3)

  const auto* nested3_g2 = FindFlatOp(flat_db2, "nested3");
  ASSERT_NE(nested3_g2, nullptr);
  EXPECT_EQ(nested3_g2->time_ps(), 30);
  EXPECT_EQ(nested3_g2->self_time_ps(), 30);

  const auto* op3_g2 = FindFlatOp(flat_db2, "op3");
  ASSERT_NE(op3_g2, nullptr);
  EXPECT_EQ(op3_g2->time_ps(), 50);
  EXPECT_EQ(op3_g2->self_time_ps(), 20);  // 50 - 30 (nested4)

  const auto* nested4_g2 = FindFlatOp(flat_db2, "nested4");
  ASSERT_NE(nested4_g2, nullptr);
  EXPECT_EQ(nested4_g2->time_ps(), 30);
  EXPECT_EQ(nested4_g2->self_time_ps(), 30);
}

TEST(ConvertXPlaneToStepEvents, SparseCoreShouldHaveStepMarkers) {
  XSpace space;
  XPlane* sc_plane =
      tsl::profiler::GetOrCreateTpuXPlane(&space, 0, "TPUv3", 0, 0, 1);
  XPlaneBuilder sc_plane_builder(sc_plane);
  sc_plane_builder.SetId(1);
  XLineBuilder step_line = sc_plane_builder.GetOrCreateLine(0);
  step_line.SetName(tsl::profiler::kSparseCoreStepLineName);
  tsl::profiler::CreateXEventMetadata(&sc_plane_builder, "Step 1");
  tsl::profiler::CreateXEvent(&sc_plane_builder, &step_line,
                              /*event_name=*/"Step 1",
                              /*offset_ps=*/0, /*duration_ps=*/1000,
                              {{StatType::kGroupId, int64_t{1}}});
  XLineBuilder op_line = sc_plane_builder.GetOrCreateLine(1);
  op_line.SetName(tsl::profiler::kSparseCoreOpLineName);
  tsl::profiler::CreateXEventMetadata(
      &sc_plane_builder, "sparse_op",
      {{StatType::kProgramId, int64_t{1}}, {StatType::kSymbolId, int64_t{1}}});
  tsl::profiler::CreateXEvent(&sc_plane_builder, &op_line,
                              /*event_name=*/"sparse_op",
                              /*offset_ps=*/100, /*duration_ps=*/880,
                              {{StatType::kGroupId, int64_t{1}}});
  StepEvents step_events = ConvertDeviceTraceXPlaneToStepEvents(*sc_plane);
  EXPECT_EQ(step_events.size(), 1);
  EXPECT_TRUE(step_events.contains(1));
  StepDetails step_1 = step_events[/*group_id=*/1];
  EXPECT_EQ(step_1.Markers().size(), 1);
  EXPECT_EQ(step_1.StepTime(), tsl::profiler::Timespan(0, 1000));
  OpMetricsDb op_metrics_db =
      step_1.PerCoreOpMetricsDb()[/*core_id=*/1 + kSparseCoreIndexStart];
  ASSERT_EQ(op_metrics_db.metrics_db_size(), 1);  // Sparse op
  EXPECT_FALSE(step_1.PerCoreFlatOpMetricsDb().contains(
      /*core_id=*/1 + kSparseCoreIndexStart));
  const OpMetrics* sparse_core_op = &op_metrics_db.metrics_db(0);
  EXPECT_EQ(sparse_core_op->name(), "sparse_op");
  EXPECT_EQ(sparse_core_op->time_ps(), 880);
  EXPECT_EQ(sparse_core_op->self_time_ps(), 880);
  EXPECT_EQ(sparse_core_op->hlo_module_id(), 1);
}

TEST(ConvertXPlaneToStepEvents, SparseCoreShouldHaveStepMarkersFlat) {
  XSpace space;
  XPlane* sc_plane =
      tsl::profiler::GetOrCreateTpuXPlane(&space, 0, "TPUv3", 0, 0, 1);
  XPlaneBuilder sc_plane_builder(sc_plane);
  sc_plane_builder.SetId(1);
  XLineBuilder step_line = sc_plane_builder.GetOrCreateLine(0);
  step_line.SetName(tsl::profiler::kSparseCoreStepLineName);
  tsl::profiler::CreateXEventMetadata(&sc_plane_builder, "Step 1");
  tsl::profiler::CreateXEvent(&sc_plane_builder, &step_line,
                              /*event_name=*/"Step 1",
                              /*offset_ps=*/0, /*duration_ps=*/1000,
                              {{StatType::kGroupId, int64_t{1}}});
  XLineBuilder op_line = sc_plane_builder.GetOrCreateLine(1);
  op_line.SetName(tsl::profiler::kSparseCoreOpLineName);
  tsl::profiler::CreateXEventMetadata(
      &sc_plane_builder, "sparse_op",
      {{StatType::kProgramId, int64_t{1}}, {StatType::kSymbolId, int64_t{1}}});
  tsl::profiler::CreateXEvent(&sc_plane_builder, &op_line,
                              /*event_name=*/"sparse_op",
                              /*offset_ps=*/100, /*duration_ps=*/880,
                              {{StatType::kGroupId, int64_t{1}}});

  StepEvents step_events = ConvertDeviceTraceXPlaneToStepEvents(
      *sc_plane, /*for_flat_profile=*/true);
  EXPECT_EQ(step_events.size(), 1);
  EXPECT_TRUE(step_events.contains(1));
  StepDetails step_1 = step_events[/*group_id=*/1];
  EXPECT_EQ(step_1.Markers().size(), 1);
  EXPECT_EQ(step_1.StepTime(), tsl::profiler::Timespan(0, 1000));
  EXPECT_FALSE(step_1.PerCoreOpMetricsDb().contains(
      /*core_id=*/1 + kSparseCoreIndexStart));
  ASSERT_TRUE(step_1.PerCoreFlatOpMetricsDb().contains(
      /*core_id=*/1 + kSparseCoreIndexStart));
  FlatOpMetricsDb flat_op_metrics_db =
      step_1.PerCoreFlatOpMetricsDb().at(/*core_id=*/1 + kSparseCoreIndexStart);
  ASSERT_EQ(flat_op_metrics_db.op_instances_size(), 1);
}

TEST(ConvertXPlaneToStepEvents, TpuDevicePlaneNoStepLine) {
  XPlane raw_plane;
  XPlaneBuilder plane(&raw_plane);
  int64_t device_id = 1;
  plane.SetId(device_id);
  plane.SetName("/device:TPU:0");

  // Empty step line.
  XLineBuilder step_line = plane.GetOrCreateLine(0);
  step_line.SetName(tsl::profiler::kStepLineName);

  // Non-empty op line.
  XLineBuilder op_line = plane.GetOrCreateLine(1);
  op_line.SetName(tsl::profiler::kXlaOpLineName);
  const XStatMetadata& program_id_stat =
      *plane.GetOrCreateStatMetadata(GetStatTypeStr(StatType::kProgramId));
  const XStatMetadata& symbol_id_stat =
      *plane.GetOrCreateStatMetadata(GetStatTypeStr(StatType::kSymbolId));
  const XStatMetadata& group_id_stat =
      *plane.GetOrCreateStatMetadata(GetStatTypeStr(StatType::kGroupId));
  {
    XEventMetadata* event_metadata =
        plane.GetOrCreateEventMetadata("op_long_name");
    event_metadata->set_display_name("op_name");
    XStatsBuilder<XEventMetadata> stats(event_metadata, &plane);
    stats.AddStatValue(program_id_stat, 1);
    stats.AddStatValue(symbol_id_stat, 1);
    {
      XEventBuilder event = op_line.AddEvent(*event_metadata);
      event.SetOffsetPs(0);
      event.SetDurationPs(50);
      event.AddStatValue(group_id_stat, 1);
    }
    {
      XEventBuilder event = op_line.AddEvent(*event_metadata);
      event.SetOffsetPs(100);
      event.SetDurationPs(50);
      event.AddStatValue(group_id_stat, 2);
    }
  }
  {
    XEventMetadata* event_metadata =
        plane.GetOrCreateEventMetadata("op_long_name2");
    event_metadata->set_display_name("op_name2");
    XStatsBuilder<XEventMetadata> stats(event_metadata, &plane);
    stats.AddStatValue(program_id_stat, 1);
    stats.AddStatValue(symbol_id_stat, 2);
    {
      XEventBuilder event = op_line.AddEvent(*event_metadata);
      event.SetOffsetPs(50);
      event.SetDurationPs(50);
      event.AddStatValue(group_id_stat, 1);
    }
    {
      XEventBuilder event = op_line.AddEvent(*event_metadata);
      event.SetOffsetPs(150);
      event.SetDurationPs(50);
      event.AddStatValue(group_id_stat, 2);
    }
  }

  StepEvents step_events = ConvertDeviceTraceXPlaneToStepEvents(raw_plane);
  EXPECT_EQ(step_events.size(), 0);
}

TEST(ConvertXPlaneToOpStats, CpuOnlyPyGrainSingleProcessInputPipelineTest) {
  XSpace space;
  XPlane* host_plane = tsl::profiler::GetOrCreateHostXPlane(&space);
  XPlaneBuilder host_plane_builder(host_plane);
  host_plane_builder.ReserveLines(5);
  auto main_thread = host_plane_builder.GetOrCreateLine(0);
  CreateXEvent(
      &host_plane_builder, &main_thread, "read_data 1", 0, 100,
      {{StatType::kStepNum, int64_t{1}}, {StatType::kIsRoot, int64_t{1}}});
  CreateXEvent(&host_plane_builder, &main_thread,
               "_BatchDatasetIterator.__next__", 1, 90,
               {{StatType::kInputPipelineStageName, "Batch"}});
  CreateXEvent(&host_plane_builder, &main_thread,
               "FirstFitPackDatasetIterator.__next__", 2, 88,
               {{StatType::kInputPipelineStageName, "FirstFitPack"}});
  CreateXEvent(&host_plane_builder, &main_thread,
               "PrefetchDatasetIterator.__next__", 3, 9,
               {{StatType::kInputPipelineStageName, "Prefetch"},
                {StatType::kProducerType, int64_t{9999}},
                {StatType::kProducerId, int64_t{1}}});
  CreateXEvent(&host_plane_builder, &main_thread,
               "PrefetchDatasetIterator.__next__", 13, 9,
               {{StatType::kInputPipelineStageName, "Prefetch"},
                {StatType::kProducerType, int64_t{9999}},
                {StatType::kProducerId, int64_t{2}}});
  CreateXEvent(&host_plane_builder, &main_thread,
               "PrefetchDatasetIterator.__next__", 23, 50,
               {{StatType::kInputPipelineStageName, "Prefetch"},
                {StatType::kProducerType, int64_t{9999}},
                {StatType::kProducerId, int64_t{3}}});
  CreateXEvent(&host_plane_builder, &main_thread,
               "PrefetchDatasetIterator.__next__", 74, 10,
               {{StatType::kInputPipelineStageName, "Prefetch"},
                {StatType::kProducerType, int64_t{9999}},
                {StatType::kProducerId, int64_t{4}}});
  CreateXEvent(&host_plane_builder, &main_thread,
               "PrefetchDatasetIterator.__next__", 85, 5,
               {{StatType::kInputPipelineStageName, "Prefetch"},
                {StatType::kProducerType, int64_t{9999}},
                {StatType::kProducerId, int64_t{5}}});
  CreateXEvent(
      &host_plane_builder, &main_thread, HostEventType::kTpuSystemExecute, 92,
      1,
      {{StatType::kProducerType,
        static_cast<int64_t>(tsl::profiler::ContextType::kTfrtTpuRuntime)},
       {StatType::kProducerId, int64_t{1}}});
  auto runtime_thread = host_plane_builder.GetOrCreateLine(1);
  CreateXEvent(
      &host_plane_builder, &runtime_thread,
      "tpu::System::Execute=>IssueSequencedEvent", 94, 4,
      {{StatType::kConsumerType,
        static_cast<int64_t>(tsl::profiler::ContextType::kTfrtTpuRuntime)},
       {StatType::kConsumerId, int64_t{1}}});
  CreateXEvent(&host_plane_builder, &runtime_thread,
               HostEventType::kDoEnqueueProgram, 95, 2,
               {{StatType::kRunId, int64_t{1}},
                {StatType::kQueueId, int64_t{0}},
                {StatType::kDeviceOrdinal, int64_t{0}}});
  for (int i = 1; i <= 5; ++i) {
    auto worker_thread = host_plane_builder.GetOrCreateLine(i + 1);
    worker_thread.SetName(absl::StrCat("WorkerThread-", i));
    CreateXEvent(&host_plane_builder, &worker_thread, "MapDataset.__getitem__",
                 2, 2,
                 {{StatType::kInputPipelineStageName, "MapDataset"},
                  {StatType::kConsumerType, int64_t{9999}},
                  {StatType::kConsumerId, int64_t{i}}});
  }

  XPlane* device_plane =
      tsl::profiler::GetOrCreateTpuXPlane(&space, 0, "TPUv4", 0, 0);
  XPlaneBuilder device_plane_builder(device_plane);
  device_plane_builder.ReserveLines(3);

  auto steps = device_plane_builder.GetOrCreateLine(0);
  steps.SetName(tsl::profiler::kStepLineName);
  auto modules = device_plane_builder.GetOrCreateLine(1);
  modules.SetName(tsl::profiler::kXlaModuleLineName);
  auto ops = device_plane_builder.GetOrCreateLine(2);
  ops.SetName(tsl::profiler::kXlaOpLineName);
  CreateXEvent(&device_plane_builder, &steps, "read_data 1", 0, 100, {});
  CreateXEvent(&device_plane_builder, &modules, "jit_multiply(1)", 97, 2,
               {{StatType::kRunId, int64_t{1}},
                {StatType::kReplicaId, int64_t{0}},
                {StatType::kQueueId, int64_t{0}}});
  CreateXEvent(&device_plane_builder, &ops, "multiply.1", 98, 1, {});

  tsl::profiler::EventForest event_forest;
  tsl::profiler::PreprocessXSpace(&space);
  tsl::profiler::GroupTpuEventsOSS(&space, {device_plane}, &event_forest);
  EXPECT_EQ(event_forest.GetGroupMetadataMap().size(), 1);
  StepEvents device_step_events =
      ConvertDeviceTraceXPlaneToStepEvents(*device_plane);
  EXPECT_EQ(device_step_events.size(), 1);
  XPlaneVisitor host_plane_visitor =
      tsl::profiler::CreateTfXPlaneVisitor(host_plane);
  host_plane_visitor.ForEachLine([&](const XLineVisitor& line) {
    line.ForEachEvent([&](const XEventVisitor& event) {
      SCOPED_TRACE(absl::StrCat(line.Name(), " ", event.Name()));
      // All events should be grouped and have `group_id` set.
      ASSERT_TRUE(event.GetStat(StatType::kGroupId).has_value());
      EXPECT_EQ(event.GetStat(StatType::kGroupId)->IntOrUintValue(), 0);
    });
  });
  XPlaneVisitor device_plane_visitor =
      tsl::profiler::CreateTfXPlaneVisitor(device_plane);
  device_plane_visitor.ForEachLine([&](const XLineVisitor& line) {
    line.ForEachEvent([&](const XEventVisitor& event) {
      SCOPED_TRACE(absl::StrCat(line.Name(), " ", event.Name()));
      // All events should be grouped and have `group_id` set.
      ASSERT_TRUE(event.GetStat(StatType::kGroupId).has_value());
      EXPECT_EQ(event.GetStat(StatType::kGroupId)->IntOrUintValue(), 0);
    });
  });
  StepEvents host_step_events =
      ConvertHostThreadsXPlaneToStepEvents(*host_plane, &device_step_events);
  // Should contain only the step which is also present on the device.
  EXPECT_EQ(host_step_events.size(), 1);
  // TraceContext should be added as a step marker.
  EXPECT_EQ(host_step_events[0].Markers().size(), 1);
  // FunctionRun shouldn't be added.
  EXPECT_EQ(host_step_events[0].Events().size(), 15);
  // Check that the new event is classified as HOST_WAIT_INPUT
  std::vector<const EventTypeSpan*> host_wait_input_events;
  for (const auto& event : host_step_events[0].Events()) {
    if (event.type == EventType::HOST_WAIT_INPUT) {
      host_wait_input_events.push_back(&event);
    }
  }
  ASSERT_EQ(host_wait_input_events.size(), 1);
  EXPECT_EQ(host_wait_input_events[0]->span, tsl::profiler::Timespan(1, 90));
}
}  // namespace
}  // namespace profiler
}  // namespace tensorflow
