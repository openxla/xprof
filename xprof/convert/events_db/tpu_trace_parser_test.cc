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

#include "xprof/convert/events_db/tpu_trace_parser.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "testing/base/public/gmock.h"
#include "<gtest/gtest.h>"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/tsl/profiler/utils/xplane_builder.h"
#include "xla/tsl/profiler/utils/xplane_schema.h"
#include "tsl/profiler/protobuf/xplane.pb.h"
#include "xprof/convert/events_db/event_utils.h"
#include "xprof/convert/events_db/record_consumer.h"
#include "xprof/convert/events_db/schema.h"
#include "xprof/convert/events_db/tpu_component.h"
#include "xprof/utils/hlo_cost_analysis_wrapper.h"
#include "xprof/utils/hlo_module_map.h"
#include "xprof/utils/xprof_gpu_cost_analysis.h"

namespace xprof::events_db::internal {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;

TEST(TpuTraceParserTest, ParsesTensorCoreAndCalculatesSelfTime) {
  tensorflow::profiler::XPlane plane;
  plane.set_name("/device:TPU:0");
  tsl::profiler::XPlaneBuilder plane_builder(&plane);

  // Line 1: Step Line
  tsl::profiler::XLineBuilder step_line =
      plane_builder.GetOrCreateLine(TpuComponent::kTensorCoreStepCounter);
  step_line.SetName(tsl::profiler::kStepLineName);
  tsl::profiler::XEventBuilder step_event =
      step_line.AddEvent(*plane_builder.GetOrCreateEventMetadata("0"));
  step_event.SetTimestampNs(0);
  step_event.SetDurationNs(10000);

  // Line 2: XLA Modules
  tsl::profiler::XLineBuilder module_line =
      plane_builder.GetOrCreateLine(TpuComponent::kTensorCoreHloModule);
  module_line.SetName(tsl::profiler::kXlaModuleLineName);
  tsl::profiler::XEventBuilder module_event = module_line.AddEvent(
      *plane_builder.GetOrCreateEventMetadata("jit_func(42)"));
  module_event.SetTimestampNs(0);
  module_event.SetDurationNs(10000);

  // Line 3: XLA Ops with nested parent & child
  tsl::profiler::XLineBuilder op_line =
      plane_builder.GetOrCreateLine(TpuComponent::kTensorCoreHLO);
  op_line.SetName(tsl::profiler::kXlaOpLineName);

  // Parent op: [1000, 9000) (dur = 8000)
  tensorflow::profiler::XEventMetadata* parent_metadata =
      plane_builder.GetOrCreateEventMetadata("fusion_parent");
  parent_metadata->set_display_name("fusion_parent");
  tsl::profiler::XEventBuilder parent_op = op_line.AddEvent(*parent_metadata);
  parent_op.SetTimestampNs(1000);
  parent_op.SetDurationNs(8000);

  // Child op: [2000, 5000) (dur = 3000)
  tensorflow::profiler::XEventMetadata* child_metadata =
      plane_builder.GetOrCreateEventMetadata("dot_child");
  child_metadata->set_display_name("dot_child");
  tsl::profiler::XEventBuilder child_op = op_line.AddEvent(*child_metadata);
  child_op.SetTimestampNs(2000);
  child_op.SetDurationNs(3000);

  tensorflow::profiler::HloModuleMap hlo_module_map;
  absl::string_view hlo_text = R"hlo(
    HloModule jit_func

    fused_comp {
      p0 = f32[128,256]{1,0} parameter(0)
      p1 = f32[256,512]{1,0} parameter(1)
      ROOT res = f32[128,512]{1,0} dot(p0, p1), lhs_contracting_dims={1}, rhs_contracting_dims={0}
    }

    ENTRY main {
      arg0 = f32[128,256]{1,0} parameter(0)
      arg1 = f32[256,512]{1,0} parameter(1)
      ROOT fusion_parent = f32[128,512]{1,0} fusion(arg0, arg1), kind=kCustom, calls=fused_comp, metadata={op_name="model/dense/MatMul" op_type="MatMul"}
    }
    )hlo";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::HloModule> hlo_module,
                       xla::ParseAndReturnUnverifiedModule(hlo_text));
  std::unique_ptr<tensorflow::profiler::HloCostAnalysisWrapper>
      cost_analysis_wrapper =
          tensorflow::profiler::CreateXprofGpuCostAnalysis();
  hlo_module_map.try_emplace(
      42, tensorflow::profiler::HloModuleWrapper(
              std::move(hlo_module), std::move(cost_analysis_wrapper)));

  Schema schema;
  FieldIndices indices(schema);
  std::vector<Record> parsed_records;

  EXPECT_THAT(ParseTpuTrace(plane, hlo_module_map, indices,
                            [&](Record& record) -> absl::StatusOr<StepControl> {
                              parsed_records.push_back(record);
                              return StepControl::kContinue;
                            }),
              IsOkAndHolds(ParseStatus::kComplete));

  // Total events parsed: Step (1), Module (1), Parent Op (1), Child Op (1) = 4
  ASSERT_EQ(parsed_records.size(), 4);

  const Record* parent_rec = nullptr;
  const Record* child_rec = nullptr;
  for (const Record& r : parsed_records) {
    if (r[indices.kernel_name] == "fusion_parent") {
      parent_rec = &r;
    } else if (r[indices.kernel_name] == "dot_child") {
      child_rec = &r;
    }
  }

  ASSERT_NE(parent_rec, nullptr);
  ASSERT_NE(child_rec, nullptr);

  EXPECT_EQ(parent_rec->Get(indices.device), "TPU:0");
  EXPECT_EQ(parent_rec->Get(indices.stream_id), TpuComponent::kTensorCoreHLO);
  EXPECT_EQ(parent_rec->Get(indices.category), tsl::profiler::kXlaOpLineName);
  EXPECT_EQ(parent_rec->Get(indices.step), "step:0");
  EXPECT_EQ(parent_rec->Get(indices.hlo_module), "jit_func(42)");
  EXPECT_FALSE(parent_rec->HasField(indices.tf_op_name));
  EXPECT_FALSE(parent_rec->HasField(indices.tf_op_type));
  EXPECT_NE(parent_rec->Get(indices.hlo_fingerprint), 0);
  ASSERT_TRUE(parent_rec->HasField(indices.flops));
  EXPECT_GT(parent_rec->Get(indices.flops), 0);
  ASSERT_TRUE(parent_rec->HasField(indices.memory_accessed));
  EXPECT_GT(parent_rec->Get(indices.memory_accessed), 0);

  ASSERT_TRUE(parent_rec->HasField(indices.input_tensors));
  const std::vector<std::string>& in_tensors =
      parent_rec->Get(indices.input_tensors);
  ASSERT_EQ(in_tensors.size(), 2);
  EXPECT_EQ(in_tensors[0], "f32[128,256]");
  EXPECT_EQ(in_tensors[1], "f32[256,512]");

  ASSERT_TRUE(parent_rec->HasField(indices.output_tensors));
  const std::vector<std::string>& out_tensors =
      parent_rec->Get(indices.output_tensors);
  ASSERT_EQ(out_tensors.size(), 1);
  EXPECT_EQ(out_tensors[0], "f32[128,512]");

  EXPECT_EQ(parent_rec->Get(indices.start_ns), 1000);
  EXPECT_EQ(parent_rec->Get(indices.end_ns), 9000);
  // Self-time of parent should have child deducted: 8000 - 3000 = 5000
  EXPECT_EQ(parent_rec->Get(indices.self_time_ns), 5000);

  EXPECT_EQ(child_rec->Get(indices.start_ns), 2000);
  EXPECT_EQ(child_rec->Get(indices.end_ns), 5000);
  EXPECT_EQ(child_rec->Get(indices.self_time_ns), 3000);
}

TEST(TpuTraceParserTest, ParsesTensorCoreWithoutPerfInfo) {
  tensorflow::profiler::XPlane plane;
  plane.set_name("/device:TPU:0");
  tsl::profiler::XPlaneBuilder plane_builder(&plane);

  tsl::profiler::XLineBuilder module_line =
      plane_builder.GetOrCreateLine(TpuComponent::kTensorCoreHloModule);
  module_line.SetName(tsl::profiler::kXlaModuleLineName);
  tsl::profiler::XEventBuilder module_event = module_line.AddEvent(
      *plane_builder.GetOrCreateEventMetadata("jit_func(42)"));
  module_event.SetTimestampNs(0);
  module_event.SetDurationNs(10000);

  tsl::profiler::XLineBuilder op_line =
      plane_builder.GetOrCreateLine(TpuComponent::kTensorCoreHLO);
  op_line.SetName(tsl::profiler::kXlaOpLineName);
  tensorflow::profiler::XEventMetadata* op_metadata =
      plane_builder.GetOrCreateEventMetadata("custom_op");
  op_metadata->set_display_name("custom_op");
  tsl::profiler::XEventBuilder op_event = op_line.AddEvent(*op_metadata);
  op_event.SetTimestampNs(1000);
  op_event.SetDurationNs(2000);

  tensorflow::profiler::HloModuleMap hlo_module_map;
  absl::string_view hlo_text = R"hlo(
    HloModule jit_func

    ENTRY main {
      p0 = f32[64,64]{1,0} parameter(0)
      ROOT custom_op = f32[64,64]{1,0} custom-call(p0), custom_call_target="noop", metadata={op_name="model/op" op_type="Custom"}
    }
    )hlo";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::HloModule> hlo_module,
                       xla::ParseAndReturnUnverifiedModule(hlo_text));
  // Emplace HloModuleWrapper without cost analysis -> perf_info == nullptr
  hlo_module_map.try_emplace(42, std::move(hlo_module));

  Schema schema;
  FieldIndices indices(schema);
  std::vector<Record> parsed_records;

  EXPECT_THAT(ParseTpuTrace(plane, hlo_module_map, indices,
                            [&](Record& record) -> absl::StatusOr<StepControl> {
                              parsed_records.push_back(record);
                              return StepControl::kContinue;
                            }),
              IsOkAndHolds(ParseStatus::kComplete));

  ASSERT_EQ(parsed_records.size(), 2);
  const Record& op_rec = parsed_records[1];
  EXPECT_EQ(op_rec[indices.kernel_name], "custom_op");
  EXPECT_FALSE(op_rec.HasField(indices.tf_op_name));
  EXPECT_FALSE(op_rec.HasField(indices.tf_op_type));
  EXPECT_FALSE(op_rec.HasField(indices.flops));
  EXPECT_FALSE(op_rec.HasField(indices.memory_accessed));
}

TEST(TpuTraceParserTest, ParsesTensorCoreWithVariousLineNames) {
  tensorflow::profiler::XPlane plane;
  plane.set_name("/device:TPU:0");
  tsl::profiler::XPlaneBuilder plane_builder(&plane);

  // Line 1: "Tensor Core"
  tsl::profiler::XLineBuilder tc_line =
      plane_builder.GetOrCreateLine(TpuComponent::kTensorCore);
  tc_line.SetName("Tensor Core");
  tsl::profiler::XEventBuilder tc_event =
      tc_line.AddEvent(*plane_builder.GetOrCreateEventMetadata("tc_kernel"));
  tc_event.SetTimestampNs(100);
  tc_event.SetDurationNs(50);

  // Line 2: "XLA TraceMe"
  tsl::profiler::XLineBuilder traceme_line =
      plane_builder.GetOrCreateLine(TpuComponent::kTensorCoreTraceMe);
  traceme_line.SetName("XLA TraceMe");
  tsl::profiler::XEventBuilder traceme_event = traceme_line.AddEvent(
      *plane_builder.GetOrCreateEventMetadata("traceme_kernel"));
  traceme_event.SetTimestampNs(200);
  traceme_event.SetDurationNs(50);

  // Line 3: "Step"
  tsl::profiler::XLineBuilder step_line =
      plane_builder.GetOrCreateLine(TpuComponent::kTensorCoreStepCounter);
  step_line.SetName("Step");
  tsl::profiler::XEventBuilder step_event =
      step_line.AddEvent(*plane_builder.GetOrCreateEventMetadata("step_1"));
  step_event.SetTimestampNs(300);
  step_event.SetDurationNs(50);

  // Line 4: "XLA Modules"
  tsl::profiler::XLineBuilder mod_line =
      plane_builder.GetOrCreateLine(TpuComponent::kTensorCoreHloModule);
  mod_line.SetName("XLA Modules");
  tsl::profiler::XEventBuilder mod_event =
      mod_line.AddEvent(*plane_builder.GetOrCreateEventMetadata("module_1"));
  mod_event.SetTimestampNs(400);
  mod_event.SetDurationNs(50);

  // Line 5: "XLA Ops" with DisplayName
  tsl::profiler::XLineBuilder ops_line =
      plane_builder.GetOrCreateLine(TpuComponent::kTensorCoreHLO);
  ops_line.SetName("XLA Ops");
  tensorflow::profiler::XEventMetadata* ops_event_metadata =
      plane_builder.GetOrCreateEventMetadata("op_raw_name");
  ops_event_metadata->set_display_name("op_display_name");
  tsl::profiler::XEventBuilder ops_event =
      ops_line.AddEvent(*ops_event_metadata);
  ops_event.SetTimestampNs(500);
  ops_event.SetDurationNs(50);

  // Line 6: "TC Overlay"
  tsl::profiler::XLineBuilder overlay_line =
      plane_builder.GetOrCreateLine(TpuComponent::kTensorCoreOverlay);
  overlay_line.SetName("TC Overlay");
  tsl::profiler::XEventBuilder overlay_event = overlay_line.AddEvent(
      *plane_builder.GetOrCreateEventMetadata("tc_overlay_event"));
  overlay_event.SetTimestampNs(600);
  overlay_event.SetDurationNs(50);

  // Line 7: Unparsed line (should be skipped)
  tsl::profiler::XLineBuilder unknown_line = plane_builder.GetOrCreateLine(999);
  unknown_line.SetName("Unknown Line");
  tsl::profiler::XEventBuilder unknown_event = unknown_line.AddEvent(
      *plane_builder.GetOrCreateEventMetadata("unknown_event"));
  unknown_event.SetTimestampNs(700);
  unknown_event.SetDurationNs(50);

  tensorflow::profiler::HloModuleMap hlo_module_map;
  Schema schema;
  FieldIndices indices(schema);
  std::vector<Record> parsed_records;

  EXPECT_THAT(ParseTpuTrace(plane, hlo_module_map, indices,
                            [&](Record& record) -> absl::StatusOr<StepControl> {
                              parsed_records.push_back(record);
                              return StepControl::kContinue;
                            }),
              IsOkAndHolds(ParseStatus::kComplete));

  ASSERT_EQ(parsed_records.size(), 6);
  EXPECT_EQ(parsed_records[0][indices.kernel_name], "tc_kernel");
  EXPECT_EQ(parsed_records[0][indices.stream_id], TpuComponent::kTensorCore);
  EXPECT_EQ(parsed_records[0][indices.category], "Tensor Core");
  EXPECT_EQ(parsed_records[1][indices.kernel_name], "traceme_kernel");
  EXPECT_EQ(parsed_records[1][indices.stream_id],
            TpuComponent::kTensorCoreTraceMe);
  EXPECT_EQ(parsed_records[1][indices.category], "XLA TraceMe");
  EXPECT_EQ(parsed_records[2][indices.kernel_name], "step:step_1");
  EXPECT_EQ(parsed_records[2][indices.stream_id],
            TpuComponent::kTensorCoreStepCounter);
  EXPECT_EQ(parsed_records[2][indices.category], "Step");
  EXPECT_EQ(parsed_records[3][indices.kernel_name], "HLO Module:module_1");
  EXPECT_EQ(parsed_records[3][indices.stream_id],
            TpuComponent::kTensorCoreHloModule);
  EXPECT_EQ(parsed_records[3][indices.category], "XLA Modules");
  EXPECT_EQ(parsed_records[4][indices.kernel_name], "op_raw_name");
  EXPECT_EQ(parsed_records[4][indices.hlo_op], "op_display_name");
  EXPECT_EQ(parsed_records[4][indices.stream_id], TpuComponent::kTensorCoreHLO);
  EXPECT_EQ(parsed_records[4][indices.category], "XLA Ops");
  EXPECT_EQ(parsed_records[5][indices.kernel_name],
            "TC Overlay:tc_overlay_event");
  EXPECT_EQ(parsed_records[5][indices.stream_id],
            TpuComponent::kTensorCoreOverlay);
  EXPECT_EQ(parsed_records[5][indices.category], "TC Overlay");
}

TEST(TpuTraceParserTest, TensorCoreParentSelfTimeUnderflowProtected) {
  tensorflow::profiler::XPlane plane;
  plane.set_name("/device:TPU:0");
  tsl::profiler::XPlaneBuilder plane_builder(&plane);

  tsl::profiler::XLineBuilder op_line =
      plane_builder.GetOrCreateLine(TpuComponent::kTensorCoreHLO);
  op_line.SetName(tsl::profiler::kXlaOpLineName);

  // Parent op: [1000, 6000) (dur = 5000)
  tsl::profiler::XEventBuilder parent_op =
      op_line.AddEvent(*plane_builder.GetOrCreateEventMetadata("parent_op"));
  parent_op.SetTimestampNs(1000);
  parent_op.SetDurationNs(5000);

  // Child op 1: [1500, 4500) (dur = 3000) -> parent self-time becomes 2000
  tsl::profiler::XEventBuilder child_op1 =
      op_line.AddEvent(*plane_builder.GetOrCreateEventMetadata("child_op1"));
  child_op1.SetTimestampNs(1500);
  child_op1.SetDurationNs(3000);

  // Child op 2: [2000, 5000) (dur = 3000) -> deducts 3000 from 2000 ->
  // underflow replaces with 0
  tsl::profiler::XEventBuilder child_op2 =
      op_line.AddEvent(*plane_builder.GetOrCreateEventMetadata("child_op2"));
  child_op2.SetTimestampNs(2000);
  child_op2.SetDurationNs(3000);

  tensorflow::profiler::HloModuleMap hlo_module_map;
  Schema schema;
  FieldIndices indices(schema);
  std::vector<Record> parsed_records;

  EXPECT_THAT(ParseTpuTrace(plane, hlo_module_map, indices,
                            [&](Record& record) -> absl::StatusOr<StepControl> {
                              parsed_records.push_back(record);
                              return StepControl::kContinue;
                            }),
              IsOkAndHolds(ParseStatus::kComplete));

  ASSERT_EQ(parsed_records.size(), 3);
  const Record* parent_op_rec = nullptr;
  const Record* child_op1_rec = nullptr;
  const Record* child_op2_rec = nullptr;
  for (const Record& r : parsed_records) {
    if (r[indices.kernel_name] == "parent_op") {
      parent_op_rec = &r;
    } else if (r[indices.kernel_name] == "child_op1") {
      child_op1_rec = &r;
    } else if (r[indices.kernel_name] == "child_op2") {
      child_op2_rec = &r;
    }
  }
  ASSERT_NE(parent_op_rec, nullptr);
  ASSERT_NE(child_op1_rec, nullptr);
  ASSERT_NE(child_op2_rec, nullptr);

  // Underflow should be replaced with 0
  EXPECT_EQ(parent_op_rec->Get(indices.self_time_ns), 0);
  EXPECT_EQ(child_op1_rec->Get(indices.self_time_ns), 3000);
  EXPECT_EQ(child_op2_rec->Get(indices.self_time_ns), 3000);
}

TEST(TpuTraceParserTest, TensorCoreXlaOpWithoutModuleContextOrStepContext) {
  tensorflow::profiler::XPlane plane;
  plane.set_name("/device:TPU:0");
  tsl::profiler::XPlaneBuilder plane_builder(&plane);

  // XLA Module without valid numeric ID
  tsl::profiler::XLineBuilder module_line =
      plane_builder.GetOrCreateLine(TpuComponent::kTensorCoreHloModule);
  module_line.SetName(tsl::profiler::kXlaModuleLineName);
  tsl::profiler::XEventBuilder module_event = module_line.AddEvent(
      *plane_builder.GetOrCreateEventMetadata("invalid_module_name"));
  module_event.SetTimestampNs(0);
  module_event.SetDurationNs(5000);

  // Op 1 in range of module, but module has invalid ID
  tsl::profiler::XLineBuilder op_line =
      plane_builder.GetOrCreateLine(TpuComponent::kTensorCoreHLO);
  op_line.SetName(tsl::profiler::kXlaOpLineName);
  tsl::profiler::XEventBuilder op_event1 =
      op_line.AddEvent(*plane_builder.GetOrCreateEventMetadata("op_1"));
  op_event1.SetTimestampNs(1000);
  op_event1.SetDurationNs(1000);

  // Op 2 outside range of module (no module context, no step context)
  tsl::profiler::XEventBuilder op_event2 =
      op_line.AddEvent(*plane_builder.GetOrCreateEventMetadata("op_2"));
  op_event2.SetTimestampNs(8000);
  op_event2.SetDurationNs(1000);

  tensorflow::profiler::HloModuleMap hlo_module_map;
  Schema schema;
  FieldIndices indices(schema);
  std::vector<Record> parsed_records;

  EXPECT_THAT(ParseTpuTrace(plane, hlo_module_map, indices,
                            [&](Record& record) -> absl::StatusOr<StepControl> {
                              parsed_records.push_back(record);
                              return StepControl::kContinue;
                            }),
              IsOkAndHolds(ParseStatus::kComplete));

  ASSERT_EQ(parsed_records.size(), 3);
  const Record& op1_rec = parsed_records[1];
  EXPECT_EQ(op1_rec[indices.hlo_module], "invalid_module_name");
  EXPECT_FALSE(op1_rec.HasField(indices.step));
  EXPECT_FALSE(op1_rec.HasField(indices.tf_op_name));

  const Record& op2_rec = parsed_records[2];
  EXPECT_FALSE(op2_rec.HasField(indices.hlo_module));
  EXPECT_FALSE(op2_rec.HasField(indices.step));
}

TEST(TpuTraceParserTest, TensorCoreStopsEarlyWhenRequested) {
  tensorflow::profiler::XPlane plane;
  plane.set_name("/device:TPU:0");
  tsl::profiler::XPlaneBuilder plane_builder(&plane);
  tsl::profiler::XLineBuilder line_builder =
      plane_builder.GetOrCreateLine(TpuComponent::kTensorCore);
  line_builder.SetName("Tensor Core");

  for (int i = 0; i < 5; ++i) {
    tsl::profiler::XEventBuilder event_builder = line_builder.AddEvent(
        *plane_builder.GetOrCreateEventMetadata(absl::StrCat("Event_", i)));
    event_builder.SetTimestampNs(100 * (i + 1));
    event_builder.SetDurationNs(10);
  }

  tensorflow::profiler::HloModuleMap hlo_module_map;
  Schema schema;
  FieldIndices indices(schema);
  int records_seen = 0;

  EXPECT_THAT(ParseTpuTrace(plane, hlo_module_map, indices,
                            [&](Record& record) -> absl::StatusOr<StepControl> {
                              ++records_seen;
                              if (records_seen == 2) {
                                return StepControl::kStop;
                              }
                              return StepControl::kContinue;
                            }),
              IsOkAndHolds(ParseStatus::kStoppedEarly));

  EXPECT_EQ(records_seen, 2);
}

TEST(TpuTraceParserTest, TensorCorePropagatesConsumerError) {
  tensorflow::profiler::XPlane plane;
  plane.set_name("/device:TPU:0");
  tsl::profiler::XPlaneBuilder plane_builder(&plane);
  tsl::profiler::XLineBuilder line_builder =
      plane_builder.GetOrCreateLine(TpuComponent::kTensorCore);
  line_builder.SetName("Tensor Core");
  tsl::profiler::XEventBuilder event_builder =
      line_builder.AddEvent(*plane_builder.GetOrCreateEventMetadata("Event_0"));
  event_builder.SetTimestampNs(100);
  event_builder.SetDurationNs(10);

  tensorflow::profiler::HloModuleMap hlo_module_map;
  Schema schema;
  FieldIndices indices(schema);

  EXPECT_THAT(ParseTpuTrace(plane, hlo_module_map, indices,
                            [&](Record& record) -> absl::StatusOr<StepControl> {
                              return absl::InternalError("tpu tc error");
                            }),
              StatusIs(absl::StatusCode::kInternal, "tpu tc error"));
}

TEST(TpuTraceParserTest, ParsesSparseCoreEvents) {
  tensorflow::profiler::XPlane plane;
  plane.set_name("/device:TPU:0 SparseCore 0");
  tsl::profiler::XPlaneBuilder plane_builder(&plane);

  // Line 1: kSparseCoreModuleLineName
  tsl::profiler::XLineBuilder mod_line =
      plane_builder.GetOrCreateLine(TpuComponent::kSparseCoreModule);
  mod_line.SetName(tsl::profiler::kSparseCoreModuleLineName);
  tsl::profiler::XEventBuilder mod_event =
      mod_line.AddEvent(*plane_builder.GetOrCreateEventMetadata("sc_module"));
  mod_event.SetTimestampNs(1000);
  mod_event.SetDurationNs(100);

  // Line 2: kSparseCoreOpLineName
  tsl::profiler::XLineBuilder op_line =
      plane_builder.GetOrCreateLine(TpuComponent::kSparseCoreOps);
  op_line.SetName(tsl::profiler::kSparseCoreOpLineName);
  tsl::profiler::XEventBuilder op_event = op_line.AddEvent(
      *plane_builder.GetOrCreateEventMetadata("embedding_lookup"));
  op_event.SetTimestampNs(2000);
  op_event.SetDurationNs(1500);

  // Line 3: "Sparse Core Syncs"
  tsl::profiler::XLineBuilder sync_line =
      plane_builder.GetOrCreateLine(TpuComponent::kSparseCoreSyncs);
  sync_line.SetName("Sparse Core Syncs");
  tsl::profiler::XEventBuilder sync_event =
      sync_line.AddEvent(*plane_builder.GetOrCreateEventMetadata("sc_sync"));
  sync_event.SetTimestampNs(4000);
  sync_event.SetDurationNs(200);

  // Line 4: kSparseCoreStepLineName
  tsl::profiler::XLineBuilder step_line =
      plane_builder.GetOrCreateLine(TpuComponent::kSparseCoreStepCounter);
  step_line.SetName(tsl::profiler::kSparseCoreStepLineName);
  tsl::profiler::XEventBuilder step_event =
      step_line.AddEvent(*plane_builder.GetOrCreateEventMetadata("sc_step_0"));
  step_event.SetTimestampNs(5000);
  step_event.SetDurationNs(300);

  // Line 5: "SC Overlay"
  tsl::profiler::XLineBuilder overlay_line =
      plane_builder.GetOrCreateLine(TpuComponent::kSparseCoreOverlay);
  overlay_line.SetName("SC Overlay");
  tsl::profiler::XEventBuilder overlay_event = overlay_line.AddEvent(
      *plane_builder.GetOrCreateEventMetadata("sc_overlay_event"));
  overlay_event.SetTimestampNs(6000);
  overlay_event.SetDurationNs(400);

  // Line 6: "TEC 0"
  tsl::profiler::XLineBuilder tec_line =
      plane_builder.GetOrCreateLine(TpuComponent::kSparseCoreTecBase);
  tec_line.SetName("TEC 0");
  tsl::profiler::XEventBuilder tec_event =
      tec_line.AddEvent(*plane_builder.GetOrCreateEventMetadata("tec_event"));
  tec_event.SetTimestampNs(7000);
  tec_event.SetDurationNs(500);

  // Line 7: Other line
  tsl::profiler::XLineBuilder other_line = plane_builder.GetOrCreateLine(999);
  other_line.SetName("Custom SC Line");
  tsl::profiler::XEventBuilder other_event = other_line.AddEvent(
      *plane_builder.GetOrCreateEventMetadata("other_event"));
  other_event.SetTimestampNs(8000);
  other_event.SetDurationNs(600);

  tensorflow::profiler::HloModuleMap hlo_module_map;
  Schema schema;
  FieldIndices indices(schema);
  std::vector<Record> parsed_records;

  EXPECT_THAT(ParseTpuTrace(plane, hlo_module_map, indices,
                            [&](Record& record) -> absl::StatusOr<StepControl> {
                              parsed_records.push_back(record);
                              return StepControl::kContinue;
                            }),
              IsOkAndHolds(ParseStatus::kComplete));

  ASSERT_EQ(parsed_records.size(), 7);
  EXPECT_EQ(parsed_records[0][indices.device], "TPU:0 SparseCore 0");
  EXPECT_EQ(parsed_records[0][indices.kernel_name], "HLO Module:sc_module");
  EXPECT_EQ(parsed_records[1][indices.kernel_name], "embedding_lookup");
  EXPECT_EQ(parsed_records[1][indices.start_ns], 2000);
  EXPECT_EQ(parsed_records[1][indices.end_ns], 3500);
  EXPECT_EQ(parsed_records[2][indices.kernel_name], "sc_sync");
  EXPECT_EQ(parsed_records[3][indices.kernel_name], "step:sc_step_0");
  EXPECT_EQ(parsed_records[4][indices.kernel_name],
            "SC Overlay:sc_overlay_event");
  EXPECT_EQ(parsed_records[5][indices.kernel_name], "tec_event");
  EXPECT_FALSE(parsed_records[6].HasField(indices.kernel_name));
}

TEST(TpuTraceParserTest, SparseCoreStopsEarlyWhenRequested) {
  tensorflow::profiler::XPlane plane;
  plane.set_name("/device:TPU:0 SparseCore 0");
  tsl::profiler::XPlaneBuilder plane_builder(&plane);
  tsl::profiler::XLineBuilder line_builder =
      plane_builder.GetOrCreateLine(TpuComponent::kSparseCoreOps);
  line_builder.SetName(tsl::profiler::kSparseCoreOpLineName);

  for (int i = 0; i < 5; ++i) {
    tsl::profiler::XEventBuilder event_builder = line_builder.AddEvent(
        *plane_builder.GetOrCreateEventMetadata(absl::StrCat("Event_", i)));
    event_builder.SetTimestampNs(100 * (i + 1));
    event_builder.SetDurationNs(10);
  }

  tensorflow::profiler::HloModuleMap hlo_module_map;
  Schema schema;
  FieldIndices indices(schema);
  int records_seen = 0;

  EXPECT_THAT(ParseTpuTrace(plane, hlo_module_map, indices,
                            [&](Record& record) -> absl::StatusOr<StepControl> {
                              ++records_seen;
                              if (records_seen == 2) {
                                return StepControl::kStop;
                              }
                              return StepControl::kContinue;
                            }),
              IsOkAndHolds(ParseStatus::kStoppedEarly));

  EXPECT_EQ(records_seen, 2);
}

TEST(TpuTraceParserTest, SparseCorePropagatesConsumerError) {
  tensorflow::profiler::XPlane plane;
  plane.set_name("/device:TPU:0 SparseCore 0");
  tsl::profiler::XPlaneBuilder plane_builder(&plane);
  tsl::profiler::XLineBuilder line_builder =
      plane_builder.GetOrCreateLine(TpuComponent::kSparseCoreOps);
  line_builder.SetName(tsl::profiler::kSparseCoreOpLineName);
  tsl::profiler::XEventBuilder event_builder =
      line_builder.AddEvent(*plane_builder.GetOrCreateEventMetadata("Event_0"));
  event_builder.SetTimestampNs(100);
  event_builder.SetDurationNs(10);

  tensorflow::profiler::HloModuleMap hlo_module_map;
  Schema schema;
  FieldIndices indices(schema);

  EXPECT_THAT(ParseTpuTrace(plane, hlo_module_map, indices,
                            [&](Record& record) -> absl::StatusOr<StepControl> {
                              return absl::InternalError("tpu sc error");
                            }),
              StatusIs(absl::StatusCode::kInternal, "tpu sc error"));
}

TEST(TpuTraceParserTest, ParsesNonCoreEvents) {
  tensorflow::profiler::XPlane plane;
  plane.set_name("/device:CUSTOM_TPU:0");
  tsl::profiler::XPlaneBuilder plane_builder(&plane);

  tsl::profiler::XLineBuilder line_builder = plane_builder.GetOrCreateLine(1);
  line_builder.SetName("NonCore Line");
  tsl::profiler::XEventBuilder event_builder = line_builder.AddEvent(
      *plane_builder.GetOrCreateEventMetadata("noncore_event"));
  event_builder.SetTimestampNs(1000);
  event_builder.SetDurationNs(500);

  tensorflow::profiler::HloModuleMap hlo_module_map;
  Schema schema;
  FieldIndices indices(schema);
  std::vector<Record> parsed_records;

  EXPECT_THAT(ParseTpuTrace(plane, hlo_module_map, indices,
                            [&](Record& record) -> absl::StatusOr<StepControl> {
                              parsed_records.push_back(record);
                              return StepControl::kContinue;
                            }),
              IsOkAndHolds(ParseStatus::kComplete));

  ASSERT_EQ(parsed_records.size(), 1);
  EXPECT_EQ(parsed_records[0][indices.device], "CUSTOM_TPU:0");
  EXPECT_FALSE(parsed_records[0].HasField(indices.kernel_name));
  EXPECT_EQ(parsed_records[0][indices.start_ns], 1000);
  EXPECT_EQ(parsed_records[0][indices.end_ns], 1500);
}

TEST(TpuTraceParserTest, NonCoreStopsEarlyWhenRequested) {
  tensorflow::profiler::XPlane plane;
  plane.set_name("/device:CUSTOM_TPU:0");
  tsl::profiler::XPlaneBuilder plane_builder(&plane);
  tsl::profiler::XLineBuilder line_builder = plane_builder.GetOrCreateLine(1);

  for (int i = 0; i < 5; ++i) {
    tsl::profiler::XEventBuilder event_builder = line_builder.AddEvent(
        *plane_builder.GetOrCreateEventMetadata(absl::StrCat("Event_", i)));
    event_builder.SetTimestampNs(100 * (i + 1));
    event_builder.SetDurationNs(10);
  }

  tensorflow::profiler::HloModuleMap hlo_module_map;
  Schema schema;
  FieldIndices indices(schema);
  int records_seen = 0;

  EXPECT_THAT(ParseTpuTrace(plane, hlo_module_map, indices,
                            [&](Record& record) -> absl::StatusOr<StepControl> {
                              ++records_seen;
                              if (records_seen == 2) {
                                return StepControl::kStop;
                              }
                              return StepControl::kContinue;
                            }),
              IsOkAndHolds(ParseStatus::kStoppedEarly));

  EXPECT_EQ(records_seen, 2);
}

TEST(TpuTraceParserTest, NonCorePropagatesConsumerError) {
  tensorflow::profiler::XPlane plane;
  plane.set_name("/device:CUSTOM_TPU:0");
  tsl::profiler::XPlaneBuilder plane_builder(&plane);
  tsl::profiler::XLineBuilder line_builder = plane_builder.GetOrCreateLine(1);
  tsl::profiler::XEventBuilder event_builder =
      line_builder.AddEvent(*plane_builder.GetOrCreateEventMetadata("Event_0"));
  event_builder.SetTimestampNs(100);
  event_builder.SetDurationNs(10);

  tensorflow::profiler::HloModuleMap hlo_module_map;
  Schema schema;
  FieldIndices indices(schema);

  EXPECT_THAT(ParseTpuTrace(plane, hlo_module_map, indices,
                            [&](Record& record) -> absl::StatusOr<StepControl> {
                              return absl::InternalError("tpu noncore error");
                            }),
              StatusIs(absl::StatusCode::kInternal, "tpu noncore error"));
}

}  // namespace
}  // namespace xprof::events_db::internal
