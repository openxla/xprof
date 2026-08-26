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

#include "testing/base/public/gmock.h"
#include "<gtest/gtest.h>"
#include "absl/strings/string_view.h"
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

}  // namespace
}  // namespace xprof::events_db::internal
