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

#include "xprof/convert/xplane_to_op_metrics_db.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>

#include "testing/base/public/gmock.h"
#include "<gtest/gtest.h>"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/tsl/platform/types.h"
#include "xla/tsl/profiler/utils/math_utils.h"
#include "xla/tsl/profiler/utils/xplane_builder.h"
#include "xla/tsl/profiler/utils/xplane_schema.h"
#include "xla/tsl/profiler/utils/xplane_test_utils.h"
#include "tsl/profiler/protobuf/xplane.pb.h"
#include "plugin/xprof/protobuf/op_metrics.pb.h"
#include "xprof/utils/hlo_cost_analysis_wrapper.h"
#include "xprof/utils/hlo_module_map.h"
#include "xprof/utils/op_metrics_db_utils.h"
#include "xprof/utils/xprof_gpu_cost_analysis.h"

namespace tensorflow {
namespace profiler {
namespace {

using ::tsl::uint64;
using ::tsl::profiler::StatType;
using ::tsl::profiler::XEventBuilder;
using ::tsl::profiler::XEventMetadata;
using ::tsl::profiler::XLineBuilder;
using ::tsl::profiler::XPlaneBuilder;
using ::tsl::profiler::XStatsBuilder;

#if defined(PLATFORM_GOOGLE)
// NOLINTNEXTLINE: clang-tidy missing-includes
using ::testing::EqualsProto;
using ::testing::proto::IgnoringRepeatedFieldOrdering;
#endif

void AddTensorFlowTpuOpEvent(std::string&& name, std::string&& tf_op_fullname,
                             int64_t start_timestamp_ns, int64_t duration_ns,
                             std::string&& hlo_category, uint64 flops,
                             uint64 bytes_accessed, int64_t occurrences,
                             int64_t self_duration, int64_t program_id,
                             int64_t symbol_id, double time_scale_multiplier,
                             XPlaneBuilder* plane, XLineBuilder* line) {
  XEventBuilder event = line->AddEvent(*plane->GetOrCreateEventMetadata(name));
  event.SetTimestampNs(start_timestamp_ns);
  event.SetDurationNs(duration_ns);
  event.SetNumOccurrences(occurrences);
  XStatsBuilder<XEventMetadata> event_metadata(
      plane->GetOrCreateEventMetadata(name), plane);
  event_metadata.AddStatValue(
      *plane->GetOrCreateStatMetadata(GetStatTypeStr(StatType::kTfOp)),
      tf_op_fullname);
  event_metadata.AddStatValue(
      *plane->GetOrCreateStatMetadata(GetStatTypeStr(StatType::kHloCategory)),
      hlo_category);
  event_metadata.AddStatValue(
      *plane->GetOrCreateStatMetadata(GetStatTypeStr(StatType::kFlops)), flops);
  event_metadata.AddStatValue(
      *plane->GetOrCreateStatMetadata(GetStatTypeStr(StatType::kSymbolId)),
      symbol_id);
  event_metadata.AddStatValue(
      *plane->GetOrCreateStatMetadata(GetStatTypeStr(StatType::kProgramId)),
      program_id);
  XStatMetadata* time_scale_multiplier_stat = plane->GetOrCreateStatMetadata(
      GetStatTypeStr(StatType::kTimeScaleMultiplier));
  event.AddStatValue(*time_scale_multiplier_stat, time_scale_multiplier);
}

void AddTensorFlowOpEvent(std::string&& tf_op_fullname,
                          int64_t start_timestamp_ns, int64_t duration_ns,
                          bool on_device, absl::string_view kernel_name,
                          XPlaneBuilder* plane, XLineBuilder* line) {
  absl::string_view name = on_device ? kernel_name : tf_op_fullname;
  XEventBuilder event = line->AddEvent(*plane->GetOrCreateEventMetadata(name));
  event.SetTimestampNs(start_timestamp_ns);
  event.SetDurationNs(duration_ns);
  if (!on_device) return;
  event.AddStatValue(
      *plane->GetOrCreateStatMetadata(GetStatTypeStr(StatType::kTfOp)),
      *plane->GetOrCreateStatMetadata(std::move(tf_op_fullname)));
}

void AddXlaCpuOpEvent(std::string&& hlo_op_name, std::string&& tf_op,
                      int64_t start_timestamp_ns, int64_t duration_ns,
                      XPlaneBuilder* plane, XLineBuilder* line) {
  XEventBuilder event =
      line->AddEvent(*plane->GetOrCreateEventMetadata(hlo_op_name));
  event.SetTimestampNs(start_timestamp_ns);
  event.SetDurationNs(duration_ns);
  event.ParseAndAddStatValue(
      *plane->GetOrCreateStatMetadata(GetStatTypeStr(StatType::kTfOp)), tf_op);
}

TEST(ConvertXPlaneToOpMetricsDb, HostOpMetricsDb) {
  static constexpr char kTfOp1[] = "TfOp1";
  static constexpr char kTfOp2[] = "TfOp2";
  constexpr int64_t kTfOp1StartNs = 100000;
  constexpr int64_t kTfOp1DurationNs = 8000;
  constexpr int64_t kTfOp2StartNs = 110000;
  constexpr int64_t kTfOp2DurationNs = 10000;

  XSpace xspace;
  XPlane* xplane = tsl::profiler::GetOrCreateHostXPlane(&xspace);
  XPlaneBuilder host_plane(xplane);
  XLineBuilder thread1 = host_plane.GetOrCreateLine(/*line_id=*/10);
  AddTensorFlowOpEvent(absl::StrCat(kTfOp1, ":", kTfOp1), kTfOp1StartNs,
                       kTfOp1DurationNs, /*on_device=*/false,
                       /*kernel_name=*/"", &host_plane, &thread1);
  XLineBuilder thread2 = host_plane.GetOrCreateLine(/*line_id=*/20);
  AddTensorFlowOpEvent(absl::StrCat(kTfOp1, ":", kTfOp1), kTfOp1StartNs,
                       kTfOp1DurationNs, /*on_device=*/false,
                       /*kernel_name=*/"", &host_plane, &thread2);
  AddTensorFlowOpEvent(absl::StrCat(kTfOp2, ":", kTfOp2), kTfOp2StartNs,
                       kTfOp2DurationNs, /*on_device=*/false,
                       /*kernel_name=*/"", &host_plane, &thread2);

  OpMetricsDb op_metrics = ConvertHostThreadsXPlaneToOpMetricsDb(*xplane);
  // Op1, Op2, Idle.
  EXPECT_EQ(3, op_metrics.metrics_db_size());
  uint64 total_op_duration =
      tsl::profiler::NanoToPico(kTfOp1DurationNs * 2 + kTfOp2DurationNs);
  EXPECT_EQ(total_op_duration, op_metrics.total_op_time_ps());
  uint64 total_duration = tsl::profiler::NanoToPico(
      kTfOp2StartNs - kTfOp1StartNs + kTfOp2DurationNs + kTfOp1DurationNs);
  EXPECT_EQ(total_duration, op_metrics.total_time_ps());

  // Verifies OpMetricsDb is built correctly.
  const OpMetrics& op_1 = op_metrics.metrics_db().at(0);
  EXPECT_EQ(kTfOp1, op_1.name());
  EXPECT_EQ(kTfOp1, op_1.category());
  EXPECT_EQ(2, op_1.occurrences());
  EXPECT_EQ(tsl::profiler::NanoToPico(kTfOp1DurationNs) * 2, op_1.time_ps());

  const OpMetrics& idle = op_metrics.metrics_db().at(1);
  EXPECT_EQ(kIdle, idle.name());
  EXPECT_EQ(kIdle, idle.category());
  // Idle time is the gap between Op2 start and the end of Op1, which is 2000ns.
  EXPECT_EQ(tsl::profiler::NanoToPico(2000), idle.time_ps());

  const OpMetrics& op_2 = op_metrics.metrics_db().at(2);
  EXPECT_EQ(kTfOp2, op_2.name());
  EXPECT_EQ(kTfOp2, op_2.category());
  EXPECT_EQ(1, op_2.occurrences());
  EXPECT_EQ(tsl::profiler::NanoToPico(kTfOp2DurationNs), op_2.time_ps());
}

TEST(ConvertXPlaneToOpMetricsDb, DeviceOpMetricsDb) {
  // TfOp1 has kernel1 and kernel2; TfOp2 has kernel3.
  static constexpr char kTfOp1[] = "TfOp1";
  static constexpr char kTfOp2[] = "TfOp2";
  static constexpr char kKernel1[] = "kernel1";
  static constexpr char kKernel2[] = "kernel2";
  static constexpr char kKernel3[] = "kernel3";
  constexpr int64_t kKernel1StartNs = 100000;
  constexpr int64_t kKernel1DurationNs = 8000;
  constexpr int64_t kKernel2StartNs = 110000;
  constexpr int64_t kKernel2DurationNs = 10000;
  constexpr int64_t kKernel3StartNs = 120000;
  constexpr int64_t kKernel3DurationNs = 10000;

  XSpace xspace;
  XPlane* xplane =
      tsl::profiler::GetOrCreateGpuXPlane(&xspace, /*device_ordinal=*/0);
  XPlaneBuilder device_plane(xplane);
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
  HloModuleMap hlo_module_map;
  tensorflow::profiler::HloCostAnalysisWrapper::Factory create_cost_analysis =
      []() { return tensorflow::profiler::CreateXprofGpuCostAnalysis(); };
  ProcessHloModuleMapFromXSpace(hlo_module_map, &xspace, create_cost_analysis);
  OpMetricsDb op_metrics =
      ConvertDeviceTraceXPlaneToOpMetricsDb(*xplane, hlo_module_map);

  // kernel1, kernel2, kernel3, Idle.
  EXPECT_EQ(4, op_metrics.metrics_db_size());
  uint64 total_op_duration = tsl::profiler::NanoToPico(
      kKernel1DurationNs * 2 + kKernel2DurationNs * 2 + kKernel3DurationNs);
  EXPECT_EQ(total_op_duration, op_metrics.total_op_time_ps());
  // For device, the total_duration for each device is the total duration
  // merged from all GPU streams, which is from 100000 to 130000.
  uint64 total_duration = tsl::profiler::NanoToPico(
      kKernel3StartNs + kKernel3DurationNs - kKernel1StartNs);
  EXPECT_EQ(std::max(total_duration, total_op_duration),
            op_metrics.total_time_ps());

  // Verifies OpMetricsDb is built correctly.
  const OpMetrics& op_1 = op_metrics.metrics_db().at(0);
  EXPECT_EQ(absl::StrCat(kTfOp1, "/", kKernel1), op_1.name());
  EXPECT_EQ(kTfOp1, op_1.category());
  EXPECT_EQ(2, op_1.occurrences());
  EXPECT_EQ(tsl::profiler::NanoToPico(kKernel1DurationNs) * 2, op_1.time_ps());

  const OpMetrics& op_2 = op_metrics.metrics_db().at(1);
  EXPECT_EQ(absl::StrCat(kTfOp1, "/", kKernel2), op_2.name());
  EXPECT_EQ(kTfOp1, op_2.category());
  EXPECT_EQ(2, op_2.occurrences());
  EXPECT_EQ(tsl::profiler::NanoToPico(kKernel2DurationNs) * 2, op_2.time_ps());

  const OpMetrics& op_3 = op_metrics.metrics_db().at(2);
  EXPECT_EQ(absl::StrCat(kTfOp2, "/", kKernel3), op_3.name());
  EXPECT_EQ(kTfOp2, op_3.category());
  EXPECT_EQ(1, op_3.occurrences());
  EXPECT_EQ(tsl::profiler::NanoToPico(kKernel3DurationNs), op_3.time_ps());

  const OpMetrics& idle = op_metrics.metrics_db().at(3);
  EXPECT_EQ(kIdle, idle.name());
  EXPECT_EQ(kIdle, idle.category());
  // GPU is always busy in this example.
  EXPECT_EQ(tsl::profiler::NanoToPico(0), idle.time_ps());
}

TEST(ConvertXPlaneToOpMetricsDb, TpuDeviceOpMetricsDb) {
  XSpace xspace;
  XPlane* xplane = tsl::profiler::GetOrCreateTpuXPlane(
      &xspace, /*device_ordinal=*/0, "TPU V4",
      /*peak_tera_flops_per_second=*/0,
      /*peak_hbm_bw_gigabytes_per_second=*/0);
  XPlaneBuilder device_plane(xplane);
  XLineBuilder stream1 = device_plane.GetOrCreateLine(/*line_id=*/10);
  stream1.SetName(tsl::profiler::kTensorFlowOpLineName);
  AddTensorFlowTpuOpEvent("MatMul", "while:MatMul", 0, 10, "MatMul", 34, 45, 2,
                          5, 1, 1, 2.0, &device_plane, &stream1);
  OpMetricsDb op_metrics = ConvertTpuDeviceTraceXPlaneToOpMetricsDb(*xplane);
#if defined(PLATFORM_GOOGLE)
  EXPECT_THAT(op_metrics, IgnoringRepeatedFieldOrdering(EqualsProto(
                              R"pb(metrics_db {
                                     hlo_module_id: 1
                                     self_time_ps: 10000
                                     flops: 68
                                     model_flops: 68
                                     num_cores: 1
                                     occurrences: 2
                                     name: "MatMul"
                                     time_ps: 10000
                                     normalized_time_ps: 20000
                                     category: "MatMul"
                                     provenance: "while:MatMul"
                                     min_time_ps: 10000
                                   }
                                   metrics_db { name: "IDLE" category: "IDLE" }
                                   total_time_ps: 10000
                                   total_op_time_ps: 10000
                                   normalized_total_op_time_ps: 20000
                              )pb")));
#endif
}

TEST(ConvertXPlaneToOpMetricsDb, HostXPlaneWithXlaOps) {
  XPlane xplane;
  XPlaneBuilder plane(&xplane);
  XLineBuilder line = plane.GetOrCreateLine(/*line_id=*/10);
  AddXlaCpuOpEvent("xla_op", "tf_op", 100000, 8000, &plane, &line);
  AddXlaCpuOpEvent("xla_op2", "tf_op2", 110000, 10000, &plane, &line);
  OpMetricsDb op_metrics = ConvertHostThreadsXPlaneToOpMetricsDb(xplane);
#if defined(PLATFORM_GOOGLE)
  EXPECT_THAT(op_metrics, EqualsProto(R"pb(metrics_db {
                                             self_time_ps: 8000000
                                             occurrences: 1
                                             name: "tf_op"
                                             time_ps: 8000000
                                           }
                                           metrics_db {
                                             self_time_ps: 10000000
                                             occurrences: 1
                                             name: "tf_op2"
                                             time_ps: 10000000
                                           }
                                           metrics_db {
                                             self_time_ps: 2000000
                                             name: "IDLE"
                                             time_ps: 2000000
                                             category: "IDLE"
                                           }
                                           total_time_ps: 20000000
                                           total_op_time_ps: 18000000
                                           precision_stats {}
              )pb"));
#endif
}

TEST(ConvertXPlaneToOpMetricsDb, HostXPlaneWithInputPipelineTracemeOps) {
  XPlane xplane;
  XPlaneBuilder plane(&xplane);
  XLineBuilder line = plane.GetOrCreateLine(/*line_id=*/10);
  tsl::profiler::CreateXEvent(
      &plane, &line, "ShuffleMapDataset", /*offset_ps=*/100000000,
      /*duration_ps=*/10000000,
      {{StatType::kInputPipelineStageId, 1},
       {StatType::kInputPipelineStageCategory, "preprocessing"}});
  tsl::profiler::CreateXEvent(
      &plane, &line, "MapMapDataset", /*offset_ps=*/100000000,
      /*duration_ps=*/8000000,
      {{StatType::kInputPipelineStageId, 2},
       {StatType::kInputPipelineStageCategory, "preprocessing"}});
  tsl::profiler::CreateXEvent(
      &plane, &line, "ShuffleMapDataset", /*offset_ps=*/120000000,
      /*duration_ps=*/10000000,
      {{StatType::kInputPipelineStageId, 3},
       {StatType::kInputPipelineStageCategory, "preprocessing"}});
  tsl::profiler::CreateXEvent(
      &plane, &line, "MapMapDataset", /*offset_ps=*/120000000,
      /*duration_ps=*/8000000,
      {{StatType::kInputPipelineStageId, 4},
       {StatType::kInputPipelineStageCategory, "preprocessing"}});

  OpMetricsDb op_metrics = ConvertHostThreadsXPlaneToOpMetricsDb(xplane);
#if defined(PLATFORM_GOOGLE)
  EXPECT_THAT(op_metrics, IgnoringRepeatedFieldOrdering(
                              EqualsProto(R"pb(metrics_db {
                                                 self_time_ps: 2000000
                                                 occurrences: 1
                                                 name: "ShuffleMapDataset"
                                                 category: "preprocessing"
                                                 hlo_module_id: 1
                                                 time_ps: 10000000
                                               }
                                               metrics_db {
                                                 self_time_ps: 8000000
                                                 occurrences: 1
                                                 name: "MapMapDataset"
                                                 category: "preprocessing"
                                                 hlo_module_id: 2
                                                 time_ps: 8000000
                                               }
                                               metrics_db {
                                                 self_time_ps: 2000000
                                                 occurrences: 1
                                                 name: "ShuffleMapDataset"
                                                 category: "preprocessing"
                                                 hlo_module_id: 3
                                                 time_ps: 10000000
                                               }
                                               metrics_db {
                                                 self_time_ps: 8000000
                                                 occurrences: 1
                                                 name: "MapMapDataset"
                                                 category: "preprocessing"
                                                 hlo_module_id: 4
                                                 time_ps: 8000000
                                               }
                                               metrics_db {
                                                 self_time_ps: 10000000
                                                 name: "IDLE"
                                                 time_ps: 10000000
                                                 category: "IDLE"
                                               }
                                               total_time_ps: 30000000
                                               total_op_time_ps: 20000000
                                               precision_stats {}
                              )pb")));
#endif
}

TEST(ConvertXPlaneToOpMetricsDb, DeviceOpMetricsDbWithNullPerformanceInfo) {
  std::string hlo_string = R"(
    HloModule TestModule

    fused_computation {
      param_0 = f32[3,3]{1,0} parameter(0)
      param_1 = f32[1,1]{1,0} parameter(1)
      convolution.1 = f32[3,3]{1,0} convolution(param_0, param_1), dim_labels=bf_oi->bf
      param_2 = f32[3,3]{1,0} parameter(2)
      ROOT add.1 = f32[3,3]{1,0} add(convolution.1, param_2)
    }

    ENTRY test {
      input0 = f32[3,3]{1,0} parameter(0)
      filter = f32[1,1]{1,0} parameter(1)
      input1 = f32[3,3]{1,0} parameter(2)
      ROOT fusion.1 = f32[3,3]{1,0} fusion(input0, filter, input1), kind=kCustom, calls=fused_computation
    }
  )";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::HloModule> hlo_module,
                       xla::ParseAndReturnUnverifiedModule(hlo_string));
  HloModuleMap hlo_module_map;
  hlo_module_map.try_emplace(
      /*program_id=*/1,
      HloModuleWrapper(std::move(hlo_module), /*cost_analysis=*/nullptr));
  XSpace xspace;
  XPlane* xplane =
      tsl::profiler::GetOrCreateGpuXPlane(&xspace, /*device_ordinal=*/0);
  XPlaneBuilder device_plane(xplane);
  XLineBuilder stream1 = device_plane.GetOrCreateLine(/*line_id=*/10);
  tsl::profiler::CreateXEvent(
      &device_plane, &stream1, "Add", /*offset_ps=*/100,
      /*duration_ps=*/10,
      {{StatType::kHloOp, "xla::op::add.1"}, {StatType::kProgramId, 1}});

  OpMetricsDb op_metrics =
      ConvertDeviceTraceXPlaneToOpMetricsDb(*xplane, hlo_module_map);

  EXPECT_EQ(2, op_metrics.metrics_db_size());
  OpMetrics op = op_metrics.metrics_db().at(0);
  EXPECT_EQ(op.name(), "add.1");
  EXPECT_EQ(op.occurrences(), 1);
  EXPECT_EQ(op.time_ps(), 10);
  EXPECT_EQ(op.flops(), 0);
  OpMetrics idle = op_metrics.metrics_db().at(1);
  EXPECT_EQ(idle.name(), "IDLE");
  EXPECT_EQ(idle.category(), "IDLE");
}

}  // namespace
}  // namespace profiler
}  // namespace tensorflow
