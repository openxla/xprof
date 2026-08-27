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

#include "xprof/convert/events_db/gpu_trace_parser.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "testing/base/public/gmock.h"
#include "<gtest/gtest.h>"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/tsl/profiler/utils/group_events.h"
#include "xla/tsl/profiler/utils/trace_utils.h"
#include "xla/tsl/profiler/utils/xplane_builder.h"
#include "xla/tsl/profiler/utils/xplane_schema.h"
#include "tsl/profiler/protobuf/xplane.pb.h"
#include "xprof/convert/events_db/event_utils.h"
#include "xprof/convert/events_db/record_consumer.h"
#include "xprof/convert/events_db/schema.h"
#include "xprof/utils/hlo_cost_analysis_wrapper.h"
#include "xprof/utils/hlo_module_map.h"
#include "xprof/utils/xprof_gpu_cost_analysis.h"

namespace xprof::events_db::internal {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;
using ::testing::ElementsAre;

class GpuTraceParserTest : public testing::Test {
 protected:
  GpuTraceParserTest() : indices(schema) {}

  tensorflow::profiler::XPlane CreateGpuPlane(int plane_id = 0) {
    tensorflow::profiler::XPlane plane;
    plane.set_id(plane_id);
    plane.set_name(absl::StrCat("/device:GPU:", plane_id));
    return plane;
  }

  absl::Status AddHloModule(
      uint64_t program_id, absl::string_view hlo_text,
      std::unique_ptr<tensorflow::profiler::HloCostAnalysisWrapper>
          cost_analysis = nullptr) {
    ASSIGN_OR_RETURN(std::unique_ptr<xla::HloModule> hlo_module,
                     xla::ParseAndReturnUnverifiedModule(hlo_text));
    if (cost_analysis == nullptr) {
      hlo_module_map.try_emplace(program_id, std::move(hlo_module));
    } else {
      hlo_module_map.try_emplace(
          program_id, tensorflow::profiler::HloModuleWrapper(
                          std::move(hlo_module), std::move(cost_analysis)));
    }
    return absl::OkStatus();
  }

  absl::StatusOr<std::vector<Record>> Parse(
      const tensorflow::profiler::XPlane& plane) {
    std::vector<Record> records;
    RETURN_IF_ERROR(
        ParseGpuTrace(plane, hlo_module_map, group_metadata_map, indices,
                      [&](Record& record) -> absl::StatusOr<StepControl> {
                        records.push_back(std::move(record));
                        return StepControl::kContinue;
                      })
            .status());
    return records;
  }

  void TestStopsEarly(absl::string_view line_name = "") {
    tensorflow::profiler::XPlane plane;
    tsl::profiler::XPlaneBuilder plane_builder(&plane);
    tsl::profiler::XLineBuilder line_builder = plane_builder.GetOrCreateLine(1);
    if (!line_name.empty()) line_builder.SetName(line_name);

    for (int i = 0; i < 5; ++i) {
      tsl::profiler::XEventBuilder event_builder = line_builder.AddEvent(
          *plane_builder.GetOrCreateEventMetadata(absl::StrCat("Event_", i)));
      event_builder.SetTimestampNs(100 * (i + 1));
      event_builder.SetDurationNs(10);
    }

    int records_seen = 0;
    EXPECT_THAT(
        ParseGpuTrace(plane, hlo_module_map, group_metadata_map, indices,
                      [&](Record&) -> absl::StatusOr<StepControl> {
                        ++records_seen;
                        return records_seen == 2 ? StepControl::kStop
                                                 : StepControl::kContinue;
                      }),
        IsOkAndHolds(ParseStatus::kStoppedEarly));
    EXPECT_EQ(records_seen, 2);
  }

  Schema schema;
  FieldIndices indices;
  tensorflow::profiler::HloModuleMap hlo_module_map;
  tsl::profiler::GroupMetadataMap group_metadata_map;
};

TEST_F(GpuTraceParserTest, ParsesGpuKernelEvents) {
  tensorflow::profiler::XPlane plane = CreateGpuPlane();
  tsl::profiler::XPlaneBuilder plane_builder(&plane);
  tsl::profiler::XLineBuilder line_builder = plane_builder.GetOrCreateLine(1);
  line_builder.SetName("Stream #1");

  tsl::profiler::XEventBuilder event_builder = line_builder.AddEvent(
      *plane_builder.GetOrCreateEventMetadata("void gemm_kernel<float>(...)"));
  event_builder.SetTimestampNs(10000);
  event_builder.SetDurationNs(4000);
  event_builder.AddStatValue(
      *plane_builder.GetOrCreateStatMetadata(
          tsl::profiler::GetStatTypeStr(tsl::profiler::StatType::kGroupId)),
      static_cast<int64_t>(1));
  event_builder.AddStatValue(
      *plane_builder.GetOrCreateStatMetadata(tsl::profiler::GetStatTypeStr(
          tsl::profiler::StatType::kCorrelationId)),
      static_cast<uint64_t>(1234));
  event_builder.AddStatValue(
      *plane_builder.GetOrCreateStatMetadata(tsl::profiler::GetStatTypeStr(
          tsl::profiler::StatType::kKernelDetails)),
      "registers=32, shared_memory=1024");
  event_builder.AddStatValue(
      *plane_builder.GetOrCreateStatMetadata(
          tsl::profiler::GetStatTypeStr(tsl::profiler::StatType::kTfOp)),
      "model/dense/MatMul:MatMul");
  event_builder.AddStatValue(
      *plane_builder.GetOrCreateStatMetadata(tsl::profiler::GetStatTypeStr(
          tsl::profiler::StatType::kTensorShapes)),
      "32,64;64,128");
  event_builder.AddStatValue(
      *plane_builder.GetOrCreateStatMetadata(
          tsl::profiler::GetStatTypeStr(tsl::profiler::StatType::kSourceInfo)),
      "train.py:100");
  event_builder.AddStatValue(
      *plane_builder.GetOrCreateStatMetadata("custom_stat"), "custom_val");

  group_metadata_map[1] = {.name = "step:0"};

  ASSERT_OK_AND_ASSIGN(std::vector<Record> parsed_records, Parse(plane));
  ASSERT_EQ(parsed_records.size(), 1);
  const Record& record = parsed_records[0];

  EXPECT_EQ(record[indices.device], "gpu:0");
  EXPECT_EQ(record[indices.stream_id], 1);
  EXPECT_EQ(record[indices.kernel_name], "void gemm_kernel<float>(...)");
  EXPECT_EQ(record[indices.start_ns], 10000);
  EXPECT_EQ(record[indices.end_ns], 14000);
  EXPECT_EQ(record[indices.self_time_ns], 4000);
  EXPECT_EQ(record[indices.category], "device");
  EXPECT_EQ(record[indices.step], "step:0");
  EXPECT_EQ(record[indices.correlation_id], 1234);
  EXPECT_EQ(record[indices.kernel_details], "registers=32, shared_memory=1024");
  ASSERT_TRUE(record.HasField(indices.tf_op_name));
  EXPECT_EQ(record[indices.tf_op_name], "model/dense/MatMul");
  ASSERT_TRUE(record.HasField(indices.tf_op_type));
  EXPECT_EQ(record[indices.tf_op_type], "MatMul");
  ASSERT_TRUE(record.HasField(indices.input_tensors));
  const std::vector<std::string>& in_tensors = record[indices.input_tensors];
  EXPECT_THAT(in_tensors, ElementsAre("32,64", "64,128"));
  ASSERT_TRUE(record.HasField(indices.trace_args));
  EXPECT_EQ(record[indices.trace_args],
            "source=train.py:100,custom_stat=custom_val");
  EXPECT_FALSE(record.HasField(indices.hlo_op));
  EXPECT_FALSE(record.HasField(indices.hlo_module));
  EXPECT_FALSE(record.HasField(indices.hlo_fingerprint));
  EXPECT_FALSE(record.HasField(indices.flops));
  EXPECT_FALSE(record.HasField(indices.memory_accessed));
}

TEST_F(GpuTraceParserTest, ParsesGpuKernelEventsWithoutTensorShapes) {
  tensorflow::profiler::XPlane plane = CreateGpuPlane();
  tsl::profiler::XPlaneBuilder plane_builder(&plane);
  tsl::profiler::XLineBuilder line_builder = plane_builder.GetOrCreateLine(1);
  line_builder.SetName("Stream #1");

  tsl::profiler::XEventBuilder event_builder = line_builder.AddEvent(
      *plane_builder.GetOrCreateEventMetadata("void gemm_kernel<float>(...)"));
  event_builder.SetTimestampNs(10000);
  event_builder.SetDurationNs(4000);
  event_builder.AddStatValue(
      *plane_builder.GetOrCreateStatMetadata(
          tsl::profiler::GetStatTypeStr(tsl::profiler::StatType::kTfOp)),
      "model/dense/MatMul:MatMul");

  ASSERT_OK_AND_ASSIGN(std::vector<Record> parsed_records, Parse(plane));
  ASSERT_EQ(parsed_records.size(), 1);
  const Record& record = parsed_records[0];

  ASSERT_TRUE(record.HasField(indices.tf_op_name));
  EXPECT_EQ(record[indices.tf_op_name], "model/dense/MatMul");
  ASSERT_TRUE(record.HasField(indices.tf_op_type));
  EXPECT_EQ(record[indices.tf_op_type], "MatMul");
  EXPECT_FALSE(record.HasField(indices.input_tensors));
}

TEST_F(GpuTraceParserTest, ParsesGpuXlaOpWithHloDetails) {
  tensorflow::profiler::XPlane plane = CreateGpuPlane();
  tsl::profiler::XPlaneBuilder plane_builder(&plane);
  tsl::profiler::XLineBuilder line_builder = plane_builder.GetOrCreateLine(1);
  line_builder.SetName("Stream #1");

  tsl::profiler::XEventBuilder event_builder = line_builder.AddEvent(
      *plane_builder.GetOrCreateEventMetadata("custom_kernel"));
  event_builder.SetTimestampNs(10000);
  event_builder.SetDurationNs(4000);
  event_builder.AddStatValue(
      *plane_builder.GetOrCreateStatMetadata(
          tsl::profiler::GetStatTypeStr(tsl::profiler::StatType::kHloOp)),
      "fusion_1");
  event_builder.AddStatValue(
      *plane_builder.GetOrCreateStatMetadata(
          tsl::profiler::GetStatTypeStr(tsl::profiler::StatType::kHloModule)),
      "test_module");
  event_builder.AddStatValue(
      *plane_builder.GetOrCreateStatMetadata(
          tsl::profiler::GetStatTypeStr(tsl::profiler::StatType::kProgramId)),
      static_cast<uint64_t>(100));
  event_builder.AddStatValue(
      *plane_builder.GetOrCreateStatMetadata(
          tsl::profiler::GetStatTypeStr(tsl::profiler::StatType::kGroupId)),
      static_cast<int64_t>(999));

  absl::string_view hlo_text = R"hlo(
    HloModule test_module

    fused_comp {
      p0 = f32[64,64]{1,0} parameter(0)
      ROOT res = f32[64,128]{1,0} custom-call(p0), custom_call_target="dense"
    }

    ENTRY main {
      param0 = f32[64,64]{1,0} parameter(0)
      ROOT fusion_1 = f32[64,128]{1,0} fusion(param0), kind=kCustom, calls=fused_comp, metadata={op_name="model/layer/Dense" op_type="Dense"}
    }
    )hlo";
  ASSERT_OK(AddHloModule(100, hlo_text));

  ASSERT_OK_AND_ASSIGN(std::vector<Record> parsed_records, Parse(plane));
  ASSERT_EQ(parsed_records.size(), 1);
  const Record& record = parsed_records[0];

  ASSERT_TRUE(record.HasField(indices.hlo_op));
  EXPECT_EQ(record[indices.hlo_op], "fusion_1");
  ASSERT_TRUE(record.HasField(indices.hlo_module));
  EXPECT_EQ(record[indices.hlo_module], "test_module");
  ASSERT_TRUE(record.HasField(indices.hlo_fingerprint));
  EXPECT_NE(record[indices.hlo_fingerprint], 0);
  ASSERT_TRUE(record.HasField(indices.tf_op_name));
  EXPECT_EQ(record[indices.tf_op_name], "model/layer/Dense");
  ASSERT_TRUE(record.HasField(indices.tf_op_type));
  EXPECT_EQ(record[indices.tf_op_type], "Dense");
  EXPECT_FALSE(record.HasField(indices.step));

  const std::vector<std::string>& in_tensors = record[indices.input_tensors];
  ASSERT_EQ(in_tensors.size(), 1);
  EXPECT_EQ(in_tensors[0], "f32[64,64]");

  const std::vector<std::string>& out_tensors = record[indices.output_tensors];
  ASSERT_EQ(out_tensors.size(), 1);
  EXPECT_EQ(out_tensors[0], "f32[64,128]");
  EXPECT_FALSE(record.HasField(indices.flops));
  EXPECT_FALSE(record.HasField(indices.memory_accessed));
}

TEST_F(GpuTraceParserTest, ParsesGpuXlaOpWithPerformanceInfo) {
  tensorflow::profiler::XPlane plane = CreateGpuPlane();
  tsl::profiler::XPlaneBuilder plane_builder(&plane);
  tsl::profiler::XLineBuilder line_builder = plane_builder.GetOrCreateLine(1);
  line_builder.SetName("Stream #1");

  tsl::profiler::XEventBuilder event_builder = line_builder.AddEvent(
      *plane_builder.GetOrCreateEventMetadata("custom_kernel"));
  event_builder.SetTimestampNs(10000);
  event_builder.SetDurationNs(4000);
  event_builder.AddStatValue(
      *plane_builder.GetOrCreateStatMetadata(
          tsl::profiler::GetStatTypeStr(tsl::profiler::StatType::kHloOp)),
      "dot_1");
  event_builder.AddStatValue(
      *plane_builder.GetOrCreateStatMetadata(
          tsl::profiler::GetStatTypeStr(tsl::profiler::StatType::kHloModule)),
      "test_module");
  event_builder.AddStatValue(
      *plane_builder.GetOrCreateStatMetadata(
          tsl::profiler::GetStatTypeStr(tsl::profiler::StatType::kProgramId)),
      static_cast<uint64_t>(100));

  absl::string_view hlo_text = R"hlo(
    HloModule test_module

    ENTRY main {
      param0 = f32[64,64]{1,0} parameter(0)
      param1 = f32[64,128]{1,0} parameter(1)
      ROOT dot_1 = f32[64,128]{1,0} dot(param0, param1), lhs_contracting_dims={1}, rhs_contracting_dims={0}, metadata={op_name="model/layer/Dense" op_type="Dense"}
    }
    )hlo";
  ASSERT_OK(AddHloModule(100, hlo_text,
                         tensorflow::profiler::CreateXprofGpuCostAnalysis()));

  ASSERT_OK_AND_ASSIGN(std::vector<Record> parsed_records, Parse(plane));
  ASSERT_EQ(parsed_records.size(), 1);
  const Record& record = parsed_records[0];

  ASSERT_TRUE(record.HasField(indices.hlo_op));
  EXPECT_EQ(record[indices.hlo_op], "dot_1");
  ASSERT_TRUE(record.HasField(indices.hlo_module));
  EXPECT_EQ(record[indices.hlo_module], "test_module");
  ASSERT_TRUE(record.HasField(indices.hlo_fingerprint));
  EXPECT_NE(record[indices.hlo_fingerprint], 0);
  ASSERT_TRUE(record.HasField(indices.tf_op_name));
  EXPECT_EQ(record[indices.tf_op_name], "model/layer/Dense");
  ASSERT_TRUE(record.HasField(indices.tf_op_type));
  EXPECT_EQ(record[indices.tf_op_type], "Dense");

  ASSERT_TRUE(record.HasField(indices.input_tensors));
  const std::vector<std::string>& in_tensors = record[indices.input_tensors];
  ASSERT_EQ(in_tensors.size(), 2);
  EXPECT_EQ(in_tensors[0], "f32[64,64]");
  EXPECT_EQ(in_tensors[1], "f32[64,128]");

  ASSERT_TRUE(record.HasField(indices.output_tensors));
  const std::vector<std::string>& out_tensors = record[indices.output_tensors];
  ASSERT_EQ(out_tensors.size(), 1);
  EXPECT_EQ(out_tensors[0], "f32[64,128]");

  ASSERT_TRUE(record.HasField(indices.flops));
  EXPECT_EQ(record[indices.flops], 1048576);
  ASSERT_TRUE(record.HasField(indices.memory_accessed));
  EXPECT_EQ(record[indices.memory_accessed], 81920);
}

TEST_F(GpuTraceParserTest, ParsesGpuXlaOpWithoutHloDetails) {
  tensorflow::profiler::XPlane plane = CreateGpuPlane();
  tsl::profiler::XPlaneBuilder plane_builder(&plane);
  tsl::profiler::XLineBuilder line_builder = plane_builder.GetOrCreateLine(1);
  line_builder.SetName("Stream #1");

  tsl::profiler::XEventBuilder event_builder = line_builder.AddEvent(
      *plane_builder.GetOrCreateEventMetadata("custom_kernel"));
  event_builder.SetTimestampNs(10000);
  event_builder.SetDurationNs(4000);
  event_builder.AddStatValue(
      *plane_builder.GetOrCreateStatMetadata(
          tsl::profiler::GetStatTypeStr(tsl::profiler::StatType::kHloOp)),
      "fusion_scope::fusion_unknown");

  ASSERT_OK_AND_ASSIGN(std::vector<Record> parsed_records, Parse(plane));
  ASSERT_EQ(parsed_records.size(), 1);
  const Record& record = parsed_records[0];

  ASSERT_TRUE(record.HasField(indices.hlo_op));
  EXPECT_EQ(record[indices.hlo_op], "fusion_unknown");
  EXPECT_FALSE(record.HasField(indices.tf_op_name));
  EXPECT_FALSE(record.HasField(indices.tf_op_type));
  EXPECT_FALSE(record.HasField(indices.hlo_fingerprint));
  EXPECT_FALSE(record.HasField(indices.flops));
  EXPECT_FALSE(record.HasField(indices.memory_accessed));
}

TEST_F(GpuTraceParserTest, ParsesXlaModuleAndXlaOpLines) {
  tensorflow::profiler::XPlane plane = CreateGpuPlane();
  tsl::profiler::XPlaneBuilder plane_builder(&plane);

  tsl::profiler::XLineBuilder module_line_builder =
      plane_builder.GetOrCreateLine(1);
  module_line_builder.SetName(
      absl::StrCat(tsl::profiler::kXlaModuleLineName, " #1"));
  tsl::profiler::XEventBuilder module_event_builder =
      module_line_builder.AddEvent(
          *plane_builder.GetOrCreateEventMetadata("module_event"));
  module_event_builder.SetTimestampNs(10000);
  module_event_builder.SetDurationNs(5000);

  tsl::profiler::XLineBuilder op_line_builder =
      plane_builder.GetOrCreateLine(2);
  op_line_builder.SetName(absl::StrCat(tsl::profiler::kXlaOpLineName, " #1"));
  tsl::profiler::XEventBuilder op_event_builder = op_line_builder.AddEvent(
      *plane_builder.GetOrCreateEventMetadata("op_event"));
  op_event_builder.SetTimestampNs(12000);
  op_event_builder.SetDurationNs(2000);

  ASSERT_OK_AND_ASSIGN(std::vector<Record> parsed_records, Parse(plane));
  ASSERT_EQ(parsed_records.size(), 2);

  const Record& module_record = parsed_records[0];
  EXPECT_EQ(module_record[indices.device], "gpu:0");
  EXPECT_EQ(module_record[indices.stream_id], 1);
  EXPECT_EQ(module_record[indices.category],
            absl::StrCat(tsl::profiler::kXlaModuleLineName, " #1"));
  EXPECT_EQ(module_record[indices.kernel_name], "HLO Module:module_event");
  EXPECT_EQ(module_record[indices.start_ns], 10000);
  EXPECT_EQ(module_record[indices.end_ns], 15000);
  EXPECT_EQ(module_record[indices.self_time_ns], 5000);

  const Record& op_record = parsed_records[1];
  EXPECT_EQ(op_record[indices.device], "gpu:0");
  EXPECT_EQ(op_record[indices.stream_id], 2);
  EXPECT_EQ(op_record[indices.category],
            absl::StrCat(tsl::profiler::kXlaOpLineName, " #1"));
  EXPECT_EQ(op_record[indices.kernel_name], "op_event");
  EXPECT_EQ(op_record[indices.start_ns], 12000);
  EXPECT_EQ(op_record[indices.end_ns], 14000);
  EXPECT_EQ(op_record[indices.self_time_ns], 2000);
}

TEST_F(GpuTraceParserTest, SkipsDerivedLines) {
  tensorflow::profiler::XPlane plane = CreateGpuPlane();
  tsl::profiler::XPlaneBuilder plane_builder(&plane);
  tsl::profiler::XLineBuilder line_builder =
      plane_builder.GetOrCreateLine(tsl::profiler::kThreadIdStepInfo);
  line_builder.SetName("Steps");

  tsl::profiler::XEventBuilder event_builder =
      line_builder.AddEvent(*plane_builder.GetOrCreateEventMetadata("step 0"));
  event_builder.SetTimestampNs(10000);
  event_builder.SetDurationNs(4000);

  ASSERT_OK_AND_ASSIGN(std::vector<Record> parsed_records, Parse(plane));
  EXPECT_TRUE(parsed_records.empty());
}

TEST_F(GpuTraceParserTest, SkipsCounterEventsLine) {
  tensorflow::profiler::XPlane plane = CreateGpuPlane();
  tsl::profiler::XPlaneBuilder plane_builder(&plane);
  tsl::profiler::XLineBuilder line_builder = plane_builder.GetOrCreateLine(1);
  line_builder.SetName(tsl::profiler::kCounterEventsLineName);
  tsl::profiler::XEventBuilder event_builder =
      line_builder.AddEvent(*plane_builder.GetOrCreateEventMetadata("counter"));
  event_builder.SetTimestampNs(10000);
  event_builder.SetDurationNs(4000);

  ASSERT_OK_AND_ASSIGN(std::vector<Record> parsed_records, Parse(plane));
  EXPECT_TRUE(parsed_records.empty());
}

TEST_F(GpuTraceParserTest, StopsEarlyWhenRequested) { TestStopsEarly(); }

TEST_F(GpuTraceParserTest, StopsEarlyOnXlaLineWhenRequested) {
  TestStopsEarly(absl::StrCat(tsl::profiler::kXlaModuleLineName, " #1"));
}

TEST_F(GpuTraceParserTest, PropagatesConsumerError) {
  tensorflow::profiler::XPlane plane;
  tsl::profiler::XPlaneBuilder plane_builder(&plane);
  tsl::profiler::XLineBuilder line_builder = plane_builder.GetOrCreateLine(1);
  tsl::profiler::XEventBuilder event_builder =
      line_builder.AddEvent(*plane_builder.GetOrCreateEventMetadata("Event_0"));
  event_builder.SetTimestampNs(100);
  event_builder.SetDurationNs(10);

  EXPECT_THAT(ParseGpuTrace(plane, hlo_module_map, group_metadata_map, indices,
                            [](Record&) -> absl::StatusOr<StepControl> {
                              return absl::InternalError("consumer error");
                            }),
              StatusIs(absl::StatusCode::kInternal, "consumer error"));
}

TEST_F(GpuTraceParserTest, PropagatesConsumerErrorOnXlaLine) {
  tensorflow::profiler::XPlane plane = CreateGpuPlane();
  tsl::profiler::XPlaneBuilder plane_builder(&plane);
  tsl::profiler::XLineBuilder line_builder = plane_builder.GetOrCreateLine(1);
  line_builder.SetName(absl::StrCat(tsl::profiler::kXlaModuleLineName, " #1"));
  tsl::profiler::XEventBuilder event_builder =
      line_builder.AddEvent(*plane_builder.GetOrCreateEventMetadata("Event_0"));
  event_builder.SetTimestampNs(100);
  event_builder.SetDurationNs(10);

  EXPECT_THAT(ParseGpuTrace(plane, hlo_module_map, group_metadata_map, indices,
                            [](Record&) -> absl::StatusOr<StepControl> {
                              return absl::InternalError("xla consumer error");
                            }),
              StatusIs(absl::StatusCode::kInternal, "xla consumer error"));
}

}  // namespace
}  // namespace xprof::events_db::internal
