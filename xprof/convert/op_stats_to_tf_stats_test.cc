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

#include "xprof/convert/op_stats_to_tf_stats.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>

#include "gtest/gtest.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/service/hlo.pb.h"
#include "xla/tsl/profiler/utils/math_utils.h"
#include "xla/tsl/profiler/utils/xplane_builder.h"
#include "xla/tsl/profiler/utils/xplane_schema.h"
#include "xla/tsl/profiler/utils/xplane_test_utils.h"
#include "tsl/profiler/protobuf/xplane.pb.h"
#include "xprof/convert/xplane_to_op_stats.h"
#include "plugin/xprof/protobuf/op_stats.pb.h"
#include "plugin/xprof/protobuf/tf_stats.pb.h"
#include "xprof/utils/kernel_stats_utils.h"

namespace tensorflow {
namespace profiler {
namespace {

using ::tsl::profiler::StatType;
using ::tsl::profiler::XEventBuilder;
using ::tsl::profiler::XLineBuilder;
using ::tsl::profiler::XPlaneBuilder;

XEventBuilder AddTensorFlowOpEvent(std::string&& tf_op_fullname,
                                   int64_t start_timestamp_ns,
                                   int64_t duration_ns, bool on_device,
                                   absl::string_view kernel_name,
                                   XPlaneBuilder* plane, XLineBuilder* line) {
  absl::string_view name = on_device ? kernel_name : tf_op_fullname;
  XEventBuilder event = line->AddEvent(*plane->GetOrCreateEventMetadata(name));
  event.SetTimestampNs(start_timestamp_ns);
  event.SetDurationNs(duration_ns);
  if (!on_device) return event;
  event.AddStatValue(
      *plane->GetOrCreateStatMetadata(GetStatTypeStr(StatType::kTfOp)),
      *plane->GetOrCreateStatMetadata(std::move(tf_op_fullname)));
  return event;
}

void AddTensorFlowOpEventWithKernelDetails(std::string&& tf_op_fullname,
                                           int64_t start_timestamp_ns,
                                           int64_t duration_ns, bool on_device,
                                           absl::string_view kernel_name,
                                           absl::string_view kernel_details,
                                           XPlaneBuilder* plane,
                                           XLineBuilder* line) {
  XEventBuilder event =
      AddTensorFlowOpEvent(std::move(tf_op_fullname), start_timestamp_ns,
                           duration_ns, on_device, kernel_name, plane, line);
  if (!on_device) return;
  event.ParseAndAddStatValue(*plane->GetOrCreateStatMetadata("kernel_details"),
                             kernel_details);
}

TEST(OpStatsToTfStats, GpuTfStats) {
  // TfOp1 has kernel1 and kernel2; TfOp2 has kernel3;
  // TfOp3 has kernel4 and kernel5 and is TensorCore eligible.
  static constexpr char kTfOp1[] = "TfOp1";
  static constexpr char kTfOp2[] = "TfOp2";
  static constexpr char kTfOp3[] = "Conv2D";
  static constexpr char kKernel1[] = "kernel1";
  static constexpr char kKernel2[] = "kernel2";
  static constexpr char kKernel3[] = "kernel3";
  // Kernel4 is a kernel using TensorCore
  static constexpr char kKernel4[] = "volta_fp16_s884gemm";
  static constexpr char kKernel5[] = "kernel5";
  constexpr int64_t kKernel1StartNs = 100000;
  constexpr int64_t kKernel1DurationNs = 8000;
  constexpr int64_t kKernel2StartNs = 110000;
  constexpr int64_t kKernel2DurationNs = 10000;
  constexpr int64_t kKernel3StartNs = 120000;
  constexpr int64_t kKernel3DurationNs = 10000;
  constexpr int64_t kKernel4StartNs = 130000;
  constexpr int64_t kKernel4DurationNs = 10000;
  constexpr int64_t kKernel5StartNs = 150000;
  constexpr int64_t kKernel5DurationNs = 10000;

  // Mock kernel details for both kernel4 and kernel5.
  const std::string kKernelDetails = R"MULTI(regs:32
static_shared:0
dynamic_shared:16384
grid:2,1,1
block:32,1,1
occ_pct:100)MULTI";

  XSpace space;
  XPlaneBuilder device_plane(
      tsl::profiler::GetOrCreateGpuXPlane(&space, /*device_ordinal=*/0));
  device_plane.AddStatValue(
      *device_plane.GetOrCreateStatMetadata(
          GetStatTypeStr(StatType::kDevVendor)),
      tsl::profiler::kDeviceVendorNvidia);
  XLineBuilder stream1 = device_plane.GetOrCreateLine(/*line_id=*/10);
  AddTensorFlowOpEvent(absl::StrCat(kTfOp1, ":", kTfOp1), kKernel1StartNs,
                       kKernel1DurationNs, /*on_device=*/true, kKernel1,
                       &device_plane, &stream1);
  AddTensorFlowOpEvent(absl::StrCat(kTfOp1, ":", kTfOp1), kKernel2StartNs,
                       kKernel2DurationNs, /*on_device=*/true, kKernel2,
                       &device_plane, &stream1);
  XLineBuilder stream2 = device_plane.GetOrCreateLine(/*line_id=*/20);
  AddTensorFlowOpEvent(absl::StrCat(kTfOp1, ":", kTfOp1), kKernel1StartNs,
                       kKernel1DurationNs, /*on_device=*/true, kKernel1,
                       &device_plane, &stream2);
  AddTensorFlowOpEvent(absl::StrCat(kTfOp1, ":", kTfOp1), kKernel2StartNs,
                       kKernel2DurationNs, /*on_device=*/true, kKernel2,
                       &device_plane, &stream2);
  AddTensorFlowOpEvent(absl::StrCat(kTfOp2, ":", kTfOp2), kKernel3StartNs,
                       kKernel3DurationNs, /*on_device=*/true, kKernel3,
                       &device_plane, &stream2);
  AddTensorFlowOpEventWithKernelDetails(
      absl::StrCat(kTfOp3, ":", kTfOp3), kKernel4StartNs, kKernel4DurationNs,
      /*on_device=*/true, kKernel4, kKernelDetails, &device_plane, &stream2);
  AddTensorFlowOpEventWithKernelDetails(
      absl::StrCat(kTfOp3, ":", kTfOp3), kKernel5StartNs, kKernel5DurationNs,
      /*on_device=*/true, kKernel5, kKernelDetails, &device_plane, &stream2);

  OpStatsOptions options;
  options.generate_kernel_stats_db = true;
  options.generate_op_metrics_db = true;
  auto op_stats_or = ConvertXSpaceToOpStats(space, options);
  ASSERT_TRUE(op_stats_or.ok()) << op_stats_or.status();
  OpStats op_stats = std::move(op_stats_or).value();
  const TfStatsDatabase tf_stats = ConvertOpStatsToTfStats(op_stats);

  EXPECT_EQ(tf_stats.device_type(), op_stats.run_environment().device_type());

  // TfOp1, TfOp3, TfOp2, Idle
  EXPECT_EQ(4, tf_stats.with_idle().tf_stats_record_size());

  const TfStatsRecord& record_0 = tf_stats.with_idle().tf_stats_record(0);
  EXPECT_EQ(kTfOp1, record_0.op_name());
  EXPECT_EQ(kTfOp1, record_0.op_type());
  EXPECT_EQ(2, record_0.occurrences());
  EXPECT_EQ(tsl::profiler::NanoToMicro(kKernel1DurationNs) * 2 +
                tsl::profiler::NanoToMicro(kKernel2DurationNs) * 2,
            record_0.total_self_time_in_us());

  const TfStatsRecord& record_1 = tf_stats.with_idle().tf_stats_record(1);
  EXPECT_EQ(kTfOp3, record_1.op_name());
  EXPECT_EQ(kTfOp3, record_1.op_type());
  EXPECT_EQ(1, record_1.occurrences());
  EXPECT_EQ(tsl::profiler::NanoToMicro(kKernel4DurationNs) +
                tsl::profiler::NanoToMicro(kKernel5DurationNs),
            record_1.total_self_time_in_us());
  // GPU TensorCore utilization is 0.5 because kernel4 is using TensorCore and
  // kernel5 is not using TensorCore, and they have the same duration.
  EXPECT_DOUBLE_EQ(0.5, record_1.gpu_tensorcore_utilization());

  const TfStatsRecord& record_2 = tf_stats.with_idle().tf_stats_record(2);
  EXPECT_EQ(kTfOp2, record_2.op_name());
  EXPECT_EQ(kTfOp2, record_2.op_type());
  EXPECT_EQ(1, record_2.occurrences());
  EXPECT_EQ(tsl::profiler::NanoToMicro(kKernel3DurationNs),
            record_2.total_self_time_in_us());
}

TEST(OpStatsToTfStats, XlaOpNameJoinsKernelStatsToFrameworkOpStats) {
  static constexpr char kOpName[] = "jit(foo)/custom";
  static constexpr char kHloOpName[] = "add.1";
  static constexpr uint64_t kProgramId = 1;
  static constexpr char kHlo[] = R"(
HloModule test_module

ENTRY main {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT add.1 = f32[] add(lhs, rhs), metadata={op_name="jit(foo)/custom"}
}
)";
  auto hlo_module_or = xla::ParseAndReturnUnverifiedModule(kHlo);
  ASSERT_TRUE(hlo_module_or.ok()) << hlo_module_or.status();
  std::unique_ptr<xla::HloModule> hlo_module =
      std::move(hlo_module_or).value();
  xla::HloProto hlo_proto;
  *hlo_proto.mutable_hlo_module() = hlo_module->ToProto();

  XSpace space;
  XPlaneBuilder metadata_plane(space.add_planes());
  metadata_plane.SetName(tsl::profiler::kMetadataPlaneName);
  tsl::profiler::XEventMetadata* hlo_metadata =
      metadata_plane.GetOrCreateEventMetadata(kProgramId);
  hlo_metadata->set_name(
      absl::StrCat(hlo_proto.hlo_module().name(), "(", kProgramId, ")"));
  tsl::profiler::XStatsBuilder<tsl::profiler::XEventMetadata>
      hlo_metadata_stats(hlo_metadata, &metadata_plane);
  hlo_metadata_stats.AddStatValue(
      *metadata_plane.GetOrCreateStatMetadata(
          GetStatTypeStr(StatType::kHloProto)),
      hlo_proto);

  XPlaneBuilder device_plane(
      tsl::profiler::GetOrCreateGpuXPlane(&space, /*device_ordinal=*/0));
  device_plane.AddStatValue(
      *device_plane.GetOrCreateStatMetadata(
          GetStatTypeStr(StatType::kDevVendor)),
      tsl::profiler::kDeviceVendorNvidia);
  XLineBuilder stream = device_plane.GetOrCreateLine(/*line_id=*/10);
  const std::string kernel_details = R"MULTI(regs:32
static_shared:0
dynamic_shared:0
grid:1,1,1
block:1,1,1
occ_pct:100)MULTI";
  auto add_kernel = [&](absl::string_view kernel_name, int64_t offset_ns,
                        int64_t duration_ns) {
    XEventBuilder event =
        stream.AddEvent(*device_plane.GetOrCreateEventMetadata(kernel_name));
    event.SetTimestampNs(offset_ns);
    event.SetDurationNs(duration_ns);
    event.AddStatValue(
        *device_plane.GetOrCreateStatMetadata(
            GetStatTypeStr(StatType::kTfOp)),
        *device_plane.GetOrCreateStatMetadata("XlaModule"));
    event.AddStatValue(
        *device_plane.GetOrCreateStatMetadata(
            GetStatTypeStr(StatType::kHloOp)),
        *device_plane.GetOrCreateStatMetadata(kHloOpName));
    event.AddStatValue(
        *device_plane.GetOrCreateStatMetadata(
            GetStatTypeStr(StatType::kProgramId)),
        kProgramId);
    event.ParseAndAddStatValue(
        *device_plane.GetOrCreateStatMetadata(
            GetStatTypeStr(StatType::kKernelDetails)),
        kernel_details);
  };
  add_kernel("volta_fp16_s884gemm", /*offset_ns=*/100000,
             /*duration_ns=*/80);
  add_kernel("helper_kernel", /*offset_ns=*/100080, /*duration_ns=*/20);

  OpStatsOptions options;
  options.generate_kernel_stats_db = true;
  options.generate_op_metrics_db = true;
  auto op_stats_or = ConvertXSpaceToOpStats(space, options);
  ASSERT_TRUE(op_stats_or.ok()) << op_stats_or.status();
  OpStats op_stats = std::move(op_stats_or).value();

  ASSERT_EQ(op_stats.kernel_stats_db().reports_size(), 2);
  const KernelReport& tensor_core_kernel =
      op_stats.kernel_stats_db().reports(0);
  EXPECT_EQ(tensor_core_kernel.name(), "volta_fp16_s884gemm");
  EXPECT_EQ(tensor_core_kernel.op_name(), kOpName);
  EXPECT_TRUE(tensor_core_kernel.is_kernel_using_tensor_core());
  EXPECT_TRUE(tensor_core_kernel.is_op_tensor_core_eligible());
  const KernelReport& helper_kernel = op_stats.kernel_stats_db().reports(1);
  EXPECT_EQ(helper_kernel.name(), "helper_kernel");
  EXPECT_EQ(helper_kernel.op_name(), kOpName);
  EXPECT_FALSE(helper_kernel.is_kernel_using_tensor_core());
  EXPECT_FALSE(helper_kernel.is_op_tensor_core_eligible());

  KernelStatsByOpName kernel_stats_by_op_name =
      GroupKernelReportsByOpName(op_stats.kernel_stats_db());
  ASSERT_EQ(kernel_stats_by_op_name.size(), 1);
  const OpLevelKernelStats& op_kernel_stats =
      kernel_stats_by_op_name.at(kOpName);
  EXPECT_TRUE(op_kernel_stats.is_op_tensor_core_eligible);
  EXPECT_EQ(op_kernel_stats.tensor_core_duration_ns, 80);
  EXPECT_EQ(op_kernel_stats.total_duration_ns, 100);

  const TfStatsDatabase tf_stats = ConvertOpStatsToTfStats(op_stats);
  const TfStatsRecord* op_record = nullptr;
  for (const TfStatsRecord& record :
       tf_stats.with_idle().tf_stats_record()) {
    if (record.op_name() == kOpName) {
      op_record = &record;
      break;
    }
  }
  ASSERT_NE(op_record, nullptr);
  EXPECT_DOUBLE_EQ(0.8, op_record->gpu_tensorcore_utilization());
}

}  // namespace
}  // namespace profiler
}  // namespace tensorflow
