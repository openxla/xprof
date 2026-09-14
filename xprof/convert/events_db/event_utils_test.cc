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

#include "xprof/convert/events_db/event_utils.h"

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "xla/tsl/profiler/utils/tf_xplane_visitor.h"
#include "xla/tsl/profiler/utils/xplane_builder.h"
#include "xla/tsl/profiler/utils/xplane_schema.h"
#include "xla/tsl/profiler/utils/xplane_visitor.h"
#include "tsl/profiler/protobuf/xplane.pb.h"
#include "xprof/convert/events_db/schema.h"

namespace xprof::events_db::internal {
namespace {

using ::testing::Eq;
using ::testing::Optional;

TEST(FieldIndicesTest, ValidIndices) {
  Schema schema;
  FieldIndices indices(schema);
  EXPECT_TRUE(indices.device.untyped().is_valid());
  EXPECT_TRUE(indices.stream_id.untyped().is_valid());
  EXPECT_TRUE(indices.thread_id.untyped().is_valid());
  EXPECT_TRUE(indices.thread_name.untyped().is_valid());
  EXPECT_TRUE(indices.correlation_id.untyped().is_valid());
  EXPECT_TRUE(indices.kernel_name.untyped().is_valid());
  EXPECT_TRUE(indices.kernel_details.untyped().is_valid());
  EXPECT_TRUE(indices.tf_op_name.untyped().is_valid());
  EXPECT_TRUE(indices.tf_op_type.untyped().is_valid());
  EXPECT_TRUE(indices.hlo_op.untyped().is_valid());
  EXPECT_TRUE(indices.hlo_module.untyped().is_valid());
  EXPECT_TRUE(indices.start_ns.untyped().is_valid());
  EXPECT_TRUE(indices.end_ns.untyped().is_valid());
  EXPECT_TRUE(indices.self_time_ns.untyped().is_valid());
  EXPECT_TRUE(indices.step.untyped().is_valid());
  EXPECT_TRUE(indices.category.untyped().is_valid());
  EXPECT_TRUE(indices.flops.untyped().is_valid());
  EXPECT_TRUE(indices.memory_accessed.untyped().is_valid());
  EXPECT_TRUE(indices.input_tensors.untyped().is_valid());
  EXPECT_TRUE(indices.output_tensors.untyped().is_valid());
  EXPECT_TRUE(indices.trace_args.untyped().is_valid());
  EXPECT_TRUE(indices.hlo_fingerprint.untyped().is_valid());
  EXPECT_TRUE(indices.flow.untyped().is_valid());
  EXPECT_TRUE(indices.source_line.untyped().is_valid());
  EXPECT_TRUE(indices.dcn_src_slice_id.untyped().is_valid());
  EXPECT_TRUE(indices.dcn_dst_slice_id.untyped().is_valid());
  EXPECT_TRUE(indices.dcn_src_logical_device_id.untyped().is_valid());
  EXPECT_TRUE(indices.dcn_dst_logical_device_id.untyped().is_valid());
  EXPECT_TRUE(indices.dcn_collective_name.untyped().is_valid());
  EXPECT_TRUE(indices.dcn_duration_us.untyped().is_valid());
  EXPECT_TRUE(indices.dcn_payload_size_bytes.untyped().is_valid());
}

TEST(EventUtilsTest, GetTpuDeviceName) {
  EXPECT_EQ(GetTpuDeviceName("/device:TPU:0"), "TPU:0");
  EXPECT_EQ(GetTpuDeviceName("/device:TPU_SPARSE_CORE:1"), "TPU_SPARSE_CORE:1");
  // Unexpected / non-prefixed inputs returned unchanged
  EXPECT_EQ(GetTpuDeviceName("TPU:1"), "TPU:1");
  EXPECT_EQ(GetTpuDeviceName("CustomPlane"), "CustomPlane");
  EXPECT_EQ(GetTpuDeviceName(""), "");
}

TEST(EventUtilsTest, GetProgramIdFromHloModuleName) {
  EXPECT_THAT(GetProgramIdFromHloModuleName("jit_model_eval(12345)"),
              Optional(12345));
  EXPECT_THAT(GetProgramIdFromHloModuleName("train_step(0)"), Optional(0));

  // Unexpected / non-conforming inputs return nullopt
  EXPECT_THAT(GetProgramIdFromHloModuleName("non_conforming_module"),
              Eq(std::nullopt));
  EXPECT_THAT(GetProgramIdFromHloModuleName("model()"), Eq(std::nullopt));
  EXPECT_THAT(GetProgramIdFromHloModuleName("model(abc)"), Eq(std::nullopt));
  EXPECT_THAT(GetProgramIdFromHloModuleName(""), Eq(std::nullopt));
  // Overflowing uint64_t value returned as nullopt
  EXPECT_THAT(GetProgramIdFromHloModuleName("model(18446744073709551616)"),
              Eq(std::nullopt));
}

TEST(EventUtilsTest, FieldIndicesRegistration) {
  Schema schema;
  FieldIndices indices(schema);

  Record record;
  record[indices.device] = "gpu:0";
  record[indices.start_ns] = uint64_t{1000};
  record[indices.end_ns] = uint64_t{2000};

  EXPECT_EQ(record[indices.device], "gpu:0");
  EXPECT_EQ(record[indices.start_ns], 1000);
  EXPECT_EQ(record[indices.end_ns], 2000);
}

TEST(EventUtilsTest, FormatTraceArgsEmpty) {
  const std::vector<std::pair<absl::string_view, std::string>> empty_pairs;
  EXPECT_EQ(FormatTraceArgs(empty_pairs), "");
}

TEST(EventUtilsTest, FormatTraceArgsSinglePair) {
  const std::vector<std::pair<absl::string_view, std::string>> pairs = {
      {"key", "value"},
  };
  EXPECT_EQ(FormatTraceArgs(pairs), "key=value");
}

TEST(EventUtilsTest, FormatTraceArgsMultiplePairs) {
  const std::vector<std::pair<absl::string_view, std::string>> pairs = {
      {"counter", "42"},
      {"model", "resnet"},
      {"phase", "train"},
  };
  EXPECT_EQ(FormatTraceArgs(pairs), "counter=42,model=resnet,phase=train");
}

TEST(EventUtilsTest, FormatTraceArgsDuplicateKeysAndEmptyValues) {
  const std::vector<std::pair<absl::string_view, std::string>> pairs = {
      {"tag", ""},
      {"tag", "second_val"},
  };
  EXPECT_EQ(FormatTraceArgs(pairs), "tag=,tag=second_val");
}

TEST(EventUtilsTest, ExtractCommonInfo) {
  tensorflow::profiler::XPlane plane;
  tsl::profiler::XPlaneBuilder plane_builder(&plane);
  tsl::profiler::XLineBuilder line_builder = plane_builder.GetOrCreateLine(1);
  line_builder.SetName("Tensor Core");

  tsl::profiler::XEventBuilder event_builder =
      line_builder.AddEvent(*plane_builder.GetOrCreateEventMetadata("matmul"));
  event_builder.SetTimestampNs(1000);
  event_builder.SetDurationNs(500);
  event_builder.AddStatValue(
      *plane_builder.GetOrCreateStatMetadata(
          tsl::profiler::GetStatTypeStr(tsl::profiler::StatType::kFlops)),
      static_cast<int64_t>(1000000));
  event_builder.AddStatValue(
      *plane_builder.GetOrCreateStatMetadata(tsl::profiler::GetStatTypeStr(
          tsl::profiler::StatType::kBytesAccessed)),
      static_cast<int64_t>(2048));
  event_builder.AddStatValue(
      *plane_builder.GetOrCreateStatMetadata(
          tsl::profiler::GetStatTypeStr(tsl::profiler::StatType::kTfOp)),
      "model/dense/MatMul:MatMul");
  event_builder.AddStatValue(
      *plane_builder.GetOrCreateStatMetadata(
          tsl::profiler::GetStatTypeStr(tsl::profiler::StatType::kSourceInfo)),
      "file.cc:123");
  // Stat with recognized StatType not handled in the switch (falls through to
  // key_value_pairs).
  event_builder.AddStatValue(
      *plane_builder.GetOrCreateStatMetadata(tsl::profiler::GetStatTypeStr(
          tsl::profiler::StatType::kCorrelationId)),
      static_cast<uint64_t>(999));
  // Custom stats with no StatType.
  event_builder.AddStatValue(
      *plane_builder.GetOrCreateStatMetadata("custom_counter"), 42);
  event_builder.AddStatValue(
      *plane_builder.GetOrCreateStatMetadata("extra_param"), "abc");

  Schema schema;
  FieldIndices indices(schema);
  Record record;

  tsl::profiler::XPlaneVisitor plane_visitor =
      tsl::profiler::CreateTfXPlaneVisitor(&plane);
  plane_visitor.ForEachLine([&](const tsl::profiler::XLineVisitor& line) {
    line.ForEachEvent([&](const tsl::profiler::XEventVisitor& event) {
      ExtractCommonInfo("TPU:0", line, event, indices, record);
    });
  });

  EXPECT_EQ(record[indices.device], "TPU:0");
  EXPECT_EQ(record[indices.category], "Tensor Core");
  EXPECT_EQ(record[indices.stream_id], 1);
  EXPECT_EQ(record[indices.start_ns], 1000);
  EXPECT_EQ(record[indices.end_ns], 1500);
  EXPECT_EQ(record[indices.self_time_ns], 500);
  EXPECT_EQ(record[indices.flops], 1000000);
  EXPECT_EQ(record[indices.memory_accessed], 2048);
  EXPECT_EQ(record[indices.tf_op_name], "model/dense/MatMul");
  EXPECT_EQ(record[indices.tf_op_type], "MatMul");
  EXPECT_EQ(record[indices.source_line], "file.cc:123");
  EXPECT_EQ(record[indices.trace_args],
            "correlation_id=999,custom_counter=42,extra_param=abc");
}

TEST(EventUtilsTest, ExtractCommonInfoWithoutTraceArgs) {
  tensorflow::profiler::XPlane plane;
  tsl::profiler::XPlaneBuilder plane_builder(&plane);
  tsl::profiler::XLineBuilder line_builder = plane_builder.GetOrCreateLine(1);
  line_builder.SetName("Compute Line");

  tsl::profiler::XEventBuilder event_builder =
      line_builder.AddEvent(*plane_builder.GetOrCreateEventMetadata("op"));
  event_builder.SetTimestampNs(2000);
  event_builder.SetDurationNs(100);

  Schema schema;
  FieldIndices indices(schema);
  Record record;

  tsl::profiler::XPlaneVisitor plane_visitor =
      tsl::profiler::CreateTfXPlaneVisitor(&plane);
  plane_visitor.ForEachLine([&](const tsl::profiler::XLineVisitor& line) {
    line.ForEachEvent([&](const tsl::profiler::XEventVisitor& event) {
      ExtractCommonInfo("GPU:0", line, event, indices, record);
    });
  });

  EXPECT_EQ(record[indices.device], "GPU:0");
  EXPECT_EQ(record[indices.category], "Compute Line");
  EXPECT_EQ(record[indices.stream_id], 1);
  EXPECT_EQ(record[indices.start_ns], 2000);
  EXPECT_EQ(record[indices.end_ns], 2100);
  EXPECT_EQ(record[indices.self_time_ns], 100);
  EXPECT_FALSE(record.HasField(indices.trace_args));
}

TEST(EventUtilsTest, ExtractDcnEvent) {
  tensorflow::profiler::XPlane plane;
  tsl::profiler::XPlaneBuilder plane_builder(&plane);
  tsl::profiler::XLineBuilder line_builder = plane_builder.GetOrCreateLine(0);

  tsl::profiler::XEventMetadata* event_metadata =
      plane_builder.GetOrCreateEventMetadata(1);
  event_metadata->set_name(std::string(tsl::profiler::kMegaScaleDcnReceive));

  tsl::profiler::XEventBuilder event_builder =
      line_builder.AddEvent(*event_metadata);
  event_builder.SetTimestampNs(100000);
  event_builder.AddStatValue(
      *plane_builder.GetOrCreateStatMetadata("dcn_label"),
      "all-reduce.273_312");
  event_builder.AddStatValue(
      *plane_builder.GetOrCreateStatMetadata("dcn_source_slice_id"), 2);
  event_builder.AddStatValue(
      *plane_builder.GetOrCreateStatMetadata("dcn_source_per_slice_device_id"),
      3);
  event_builder.AddStatValue(
      *plane_builder.GetOrCreateStatMetadata("dcn_destination_slice_id"), 1);
  event_builder.AddStatValue(*plane_builder.GetOrCreateStatMetadata(
                                 "dcn_destination_per_slice_device_id"),
                             3);
  event_builder.AddStatValue(
      *plane_builder.GetOrCreateStatMetadata("dcn_chunk"), 0);
  event_builder.AddStatValue(
      *plane_builder.GetOrCreateStatMetadata("dcn_loop_index"), 24);
  event_builder.AddStatValue(
      *plane_builder.GetOrCreateStatMetadata("duration_us"), 50);
  event_builder.AddStatValue(
      *plane_builder.GetOrCreateStatMetadata("payload_size_bytes"), 32768);

  Schema schema;
  FieldIndices indices(schema);
  Record record;

  tsl::profiler::XPlaneVisitor plane_visitor =
      tsl::profiler::CreateTfXPlaneVisitor(&plane);
  plane_visitor.ForEachLine([&](const tsl::profiler::XLineVisitor& line) {
    line.ForEachEvent([&](const tsl::profiler::XEventVisitor& event) {
      ExtractDcnEvent(event, indices, record);
    });
  });

  EXPECT_EQ(record[indices.dcn_collective_name], "all-reduce.273_312");
  EXPECT_EQ(record[indices.dcn_src_slice_id], 2);
  EXPECT_EQ(record[indices.dcn_src_logical_device_id], 3);
  EXPECT_EQ(record[indices.dcn_dst_slice_id], 1);
  EXPECT_EQ(record[indices.dcn_dst_logical_device_id], 3);
  EXPECT_EQ(record[indices.dcn_duration_us], 50);
  EXPECT_EQ(record[indices.dcn_payload_size_bytes], 32768);
}

TEST(EventUtilsTest, ExtractDcnEventNonDcnIgnored) {
  tensorflow::profiler::XPlane plane;
  tsl::profiler::XPlaneBuilder plane_builder(&plane);
  tsl::profiler::XLineBuilder line_builder = plane_builder.GetOrCreateLine(0);
  tsl::profiler::XEventMetadata* event_metadata =
      plane_builder.GetOrCreateEventMetadata(1);
  event_metadata->set_name("regular_host_event");
  tsl::profiler::XEventBuilder event_builder =
      line_builder.AddEvent(*event_metadata);
  event_builder.SetTimestampNs(100000);

  Schema schema;
  FieldIndices indices(schema);
  Record record;

  tsl::profiler::XPlaneVisitor plane_visitor =
      tsl::profiler::CreateTfXPlaneVisitor(&plane);
  plane_visitor.ForEachLine([&](const tsl::profiler::XLineVisitor& line) {
    line.ForEachEvent([&](const tsl::profiler::XEventVisitor& event) {
      ExtractDcnEvent(event, indices, record);
    });
  });

  EXPECT_FALSE(record.HasField(indices.dcn_collective_name));
}

TEST(EventUtilsTest, ExtractDcnEventInvalidSkipped) {
  tensorflow::profiler::XPlane plane;
  tsl::profiler::XPlaneBuilder plane_builder(&plane);
  tsl::profiler::XLineBuilder line_builder = plane_builder.GetOrCreateLine(0);

  tsl::profiler::XEventMetadata* event_metadata =
      plane_builder.GetOrCreateEventMetadata(1);
  event_metadata->set_name(std::string(tsl::profiler::kMegaScaleDcnReceive));

  tsl::profiler::XEventBuilder event_builder =
      line_builder.AddEvent(*event_metadata);
  event_builder.SetTimestampNs(100000);
  // Missing required DCN fields, leading to DCN_MESSAGE_INVALID_BAD_KEY
  event_builder.AddStatValue(
      *plane_builder.GetOrCreateStatMetadata("dcn_label"),
      "invalid_collective");

  Schema schema;
  FieldIndices indices(schema);
  Record record;

  tsl::profiler::XPlaneVisitor plane_visitor =
      tsl::profiler::CreateTfXPlaneVisitor(&plane);
  plane_visitor.ForEachLine([&](const tsl::profiler::XLineVisitor& line) {
    line.ForEachEvent([&](const tsl::profiler::XEventVisitor& event) {
      ExtractDcnEvent(event, indices, record);
    });
  });

  // Since message is invalid, no DCN fields should be populated.
  EXPECT_FALSE(record.HasField(indices.dcn_collective_name));
  EXPECT_FALSE(record.HasField(indices.dcn_src_slice_id));
  EXPECT_FALSE(record.HasField(indices.dcn_dst_slice_id));
  EXPECT_FALSE(record.HasField(indices.dcn_src_logical_device_id));
  EXPECT_FALSE(record.HasField(indices.dcn_dst_logical_device_id));
  EXPECT_FALSE(record.HasField(indices.dcn_duration_us));
  EXPECT_FALSE(record.HasField(indices.dcn_payload_size_bytes));
}

}  // namespace
}  // namespace xprof::events_db::internal
