/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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
#include "xprof/convert/op_stats_to_input_pipeline_analysis.h"

#include <cstdint>
#include <string>

#include "google/protobuf/any.pb.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/tsl/profiler/utils/timespan.h"
#include "<gtest/gtest.h>"
#include "tsl/platform/protobuf.h"
#include "plugin/xprof/protobuf/op_stats.pb.h"
#include "plugin/xprof/protobuf/steps_db.pb.h"
#include "xprof/utils/event_span.h"
#include "xprof/utils/op_metrics_db_utils.h"

namespace tensorflow {
namespace profiler {
namespace {

using ::tensorflow::profiler::CoreDetails;
using ::tensorflow::profiler::OpMetricsDb;
using ::tensorflow::profiler::StepDatabaseResult;
using ::tensorflow::profiler::StepEvents;

TEST(TfOpStatsToInputPipelineAnalysisTest,
     AttributeHostInputTimeToTCWhenInfeedMissing) {
  uint64_t step_num = 1;
  tensorflow::profiler::StepDetails step_details;
  step_details.AddEvent(tensorflow::profiler::EventTypeSpan(
      tensorflow::profiler::EventType::HOST_WAIT_INPUT,
      tsl::profiler::Timespan::FromEndPoints(50, 100)));
  step_details.AddEvent(tensorflow::profiler::EventTypeSpan(
      tensorflow::profiler::EventType::HOST_TO_DEVICE,
      tsl::profiler::Timespan::FromEndPoints(110, 200)));
  step_details.AddEvent(tensorflow::profiler::EventTypeSpan(
      tensorflow::profiler::EventType::HOST_TO_DEVICE,
      tsl::profiler::Timespan::FromEndPoints(430, 500)));
  StepEvents host_step_events = {{step_num, step_details}};
  StepDatabaseResult step_db;
  tensorflow::profiler::PerCoreStepInfo* pcsi = step_db.add_step_sequence();
  pcsi->set_step_num(step_num);
  auto& sipc_map = *pcsi->mutable_step_info_per_core();
  tensorflow::profiler::StepInfoResult& sir = sipc_map[/* core_id= */ 2];
  sir.set_step_num(step_num);
  sir.set_begin_ps(40);
  sir.set_duration_ps(1000);
  tensorflow::profiler::GenericStepBreakdown step_breakdown;
  google::protobuf::Map<std::string, uint64_t>& category_ps =
      *step_breakdown.mutable_category_ps();
  category_ps[tensorflow::profiler::kIdle] = 300;
  category_ps[xla::HloOpcodeString(xla::HloOpcode::kMultiply)] = 300;
  category_ps[xla::HloOpcodeString(xla::HloOpcode::kAllGather)] = 300;
  category_ps[xla::HloOpcodeString(xla::HloOpcode::kAsyncStart)] = 50;
  category_ps[xla::HloOpcodeString(xla::HloOpcode::kAsyncDone)] = 50;
  sir.mutable_step_breakdown()->PackFrom(step_breakdown);
  google::protobuf::Map<uint32_t, CoreDetails> core_details_map;
  MayFixTpuStepAnalysis(host_step_events, OpMetricsDb(), step_db,
                        core_details_map);
  tensorflow::profiler::GenericStepBreakdown updated_step_breakdown;
  sir.step_breakdown().UnpackTo(&updated_step_breakdown);
  const google::protobuf::Map<std::string, uint64_t>& updated_category_ps =
      updated_step_breakdown.category_ps();
  EXPECT_EQ(updated_category_ps.at(tensorflow::profiler::kIdle), 90);
  ASSERT_TRUE(updated_category_ps.contains(
      xla::HloOpcodeString(xla::HloOpcode::kInfeed)));
  EXPECT_EQ(
      updated_category_ps.at(xla::HloOpcodeString(xla::HloOpcode::kInfeed)),
      210);
}

TEST(TfOpStatsToInputPipelineAnalysisTest,
     AttributeHostInputTimeToTCWhenInfeedMissingMultiCore) {
  uint64_t step_num = 1;
  tensorflow::profiler::StepDetails step_details;
  step_details.AddEvent(tensorflow::profiler::EventTypeSpan(
      tensorflow::profiler::EventType::HOST_WAIT_INPUT,
      tsl::profiler::Timespan::FromEndPoints(50, 100)));
  step_details.AddEvent(tensorflow::profiler::EventTypeSpan(
      tensorflow::profiler::EventType::HOST_TO_DEVICE,
      tsl::profiler::Timespan::FromEndPoints(110, 200)));
  step_details.AddEvent(tensorflow::profiler::EventTypeSpan(
      tensorflow::profiler::EventType::HOST_TO_DEVICE,
      tsl::profiler::Timespan::FromEndPoints(430, 500)));
  StepEvents host_step_events = {{step_num, step_details}};
  StepDatabaseResult step_db;
  tensorflow::profiler::PerCoreStepInfo* pcsi = step_db.add_step_sequence();
  pcsi->set_step_num(step_num);
  google::protobuf::Map<uint32_t, tensorflow::profiler::StepInfoResult>& sipc_map =
      *pcsi->mutable_step_info_per_core();
  tensorflow::profiler::StepInfoResult& sir = sipc_map[/* core_id= */ 2];
  sir.set_step_num(step_num);
  sir.set_begin_ps(40);
  sir.set_duration_ps(1000);
  tensorflow::profiler::GenericStepBreakdown step_breakdown;
  google::protobuf::Map<std::string, uint64_t>& category_ps =
      *step_breakdown.mutable_category_ps();
  category_ps[tensorflow::profiler::kIdle] = 300;
  category_ps[xla::HloOpcodeString(xla::HloOpcode::kMultiply)] = 300;
  category_ps[xla::HloOpcodeString(xla::HloOpcode::kAllGather)] = 300;
  category_ps[xla::HloOpcodeString(xla::HloOpcode::kAsyncStart)] = 50;
  category_ps[xla::HloOpcodeString(xla::HloOpcode::kAsyncDone)] = 50;
  sir.mutable_step_breakdown()->PackFrom(step_breakdown);
  tensorflow::profiler::StepInfoResult& sir2 = sipc_map[/* core_id= */ 1];
  sir2.set_step_num(step_num);
  sir2.set_begin_ps(45);
  sir2.set_duration_ps(900);
  tensorflow::profiler::GenericStepBreakdown step_breakdown2;
  google::protobuf::Map<std::string, uint64_t>& category_ps2 =
      *step_breakdown2.mutable_category_ps();
  category_ps2[tensorflow::profiler::kIdle] = 250;
  category_ps2[xla::HloOpcodeString(xla::HloOpcode::kMultiply)] = 300;
  category_ps2[xla::HloOpcodeString(xla::HloOpcode::kAllGather)] = 250;
  category_ps2[xla::HloOpcodeString(xla::HloOpcode::kAsyncStart)] = 50;
  category_ps2[xla::HloOpcodeString(xla::HloOpcode::kAsyncDone)] = 50;
  sir2.mutable_step_breakdown()->PackFrom(step_breakdown2);
  google::protobuf::Map<uint32_t, CoreDetails> core_details_map;
  OpMetricsDb device_op_metrics_db;
  MayFixTpuStepAnalysis(host_step_events, device_op_metrics_db, step_db,
                        core_details_map);
  tensorflow::profiler::GenericStepBreakdown updated_step_breakdown;
  sir.step_breakdown().UnpackTo(&updated_step_breakdown);
  const google::protobuf::Map<std::string, uint64_t>& updated_category_ps =
      updated_step_breakdown.category_ps();
  EXPECT_EQ(updated_category_ps.at(tensorflow::profiler::kIdle), 48);
  ASSERT_TRUE(updated_category_ps.contains(
      xla::HloOpcodeString(xla::HloOpcode::kInfeed)));
  EXPECT_EQ(
      updated_category_ps.at(xla::HloOpcodeString(xla::HloOpcode::kInfeed)),
      252);
  tensorflow::profiler::GenericStepBreakdown updated_step_breakdown2;
  sir2.step_breakdown().UnpackTo(&updated_step_breakdown2);
  const google::protobuf::Map<std::string, uint64_t>& updated_category_ps2 =
      updated_step_breakdown2.category_ps();
  EXPECT_EQ(updated_category_ps2.at(tensorflow::profiler::kIdle), 40);
  ASSERT_TRUE(updated_category_ps2.contains(
      xla::HloOpcodeString(xla::HloOpcode::kInfeed)));
  EXPECT_EQ(
      updated_category_ps2.at(xla::HloOpcodeString(xla::HloOpcode::kInfeed)),
      210);
}

TEST(TfOpStatsToInputPipelineAnalysisTest,
     SkipMayFixTpuStepAnalysisWhenInfeedExists) {
  uint64_t step_num = 1;
  tensorflow::profiler::StepDetails step_details;
  step_details.AddEvent(tensorflow::profiler::EventTypeSpan(
      tensorflow::profiler::EventType::HOST_WAIT_INPUT,
      tsl::profiler::Timespan::FromEndPoints(50, 100)));
  step_details.AddEvent(tensorflow::profiler::EventTypeSpan(
      tensorflow::profiler::EventType::HOST_TO_DEVICE,
      tsl::profiler::Timespan::FromEndPoints(110, 200)));
  step_details.AddEvent(tensorflow::profiler::EventTypeSpan(
      tensorflow::profiler::EventType::HOST_TO_DEVICE,
      tsl::profiler::Timespan::FromEndPoints(430, 500)));
  StepEvents host_step_events = {{step_num, step_details}};
  StepDatabaseResult step_db;
  tensorflow::profiler::PerCoreStepInfo* pcsi = step_db.add_step_sequence();
  pcsi->set_step_num(step_num);
  google::protobuf::Map<uint32_t, tensorflow::profiler::StepInfoResult>& sipc_map =
      *pcsi->mutable_step_info_per_core();
  tensorflow::profiler::StepInfoResult& sir = sipc_map[/* core_id= */ 2];
  sir.set_step_num(step_num);
  sir.set_begin_ps(40);
  sir.set_duration_ps(1000);
  tensorflow::profiler::GenericStepBreakdown step_breakdown;
  google::protobuf::Map<std::string, uint64_t>& category_ps =
      *step_breakdown.mutable_category_ps();
  category_ps[tensorflow::profiler::kIdle] = 300;
  category_ps[xla::HloOpcodeString(xla::HloOpcode::kMultiply)] = 300;
  category_ps[xla::HloOpcodeString(xla::HloOpcode::kAllGather)] = 300;
  category_ps[xla::HloOpcodeString(xla::HloOpcode::kAsyncStart)] = 50;
  category_ps[xla::HloOpcodeString(xla::HloOpcode::kInfeed)] = 50;
  sir.mutable_step_breakdown()->PackFrom(step_breakdown);
  google::protobuf::Map<uint32_t, CoreDetails> core_details_map;
  OpMetricsDb device_op_metrics_db;
  device_op_metrics_db.add_metrics_db()->set_category(
      std::string(xla::HloOpcodeString(xla::HloOpcode::kInfeed)));
  MayFixTpuStepAnalysis(host_step_events, device_op_metrics_db, step_db,
                        core_details_map);
  tensorflow::profiler::GenericStepBreakdown updated_step_breakdown;
  sir.step_breakdown().UnpackTo(&updated_step_breakdown);
  const google::protobuf::Map<std::string, uint64_t>& updated_category_ps =
      updated_step_breakdown.category_ps();
  EXPECT_EQ(updated_category_ps.at(tensorflow::profiler::kIdle), 300);
  EXPECT_EQ(
      updated_category_ps.at(xla::HloOpcodeString(xla::HloOpcode::kInfeed)),
      50);
}

TEST(TfOpStatsToInputPipelineAnalysisTest, EnsureSparseCoreStepsSetStepNumber) {
  PerCoreStepInfo per_core_step_info;
  per_core_step_info.set_step_num(1);
  google::protobuf::Map<uint32_t, StepInfoResult>& step_info_per_core =
      *per_core_step_info.mutable_step_info_per_core();
  StepInfoResult& step_info =
      step_info_per_core[/* core_id= */ kSparseCoreIndexStart + 1];
  step_info.set_step_num(1);
  step_info.set_begin_ps(100);
  step_info.set_duration_ps(1000);
  GenericStepBreakdown sparse_core_step_breakdown;
  google::protobuf::Map<std::string, uint64_t>& category_ps =
      *sparse_core_step_breakdown.mutable_category_ps();
  category_ps[tensorflow::profiler::kIdle] = 500;
  category_ps["sparse_core_busy_ops"] = 500;
  step_info.mutable_step_breakdown()->PackFrom(sparse_core_step_breakdown);

  google::protobuf::Map<uint32_t, CoreDetails> core_details_map;
  CoreDetails& core_details =
      core_details_map[/* core_id= */ kSparseCoreIndexStart + 1];
  core_details.set_is_sparse_core(true);

  PerTpuStepDetails per_step_data =
      ComputeTpuPerStepDataAcrossCores(per_core_step_info, core_details_map);
  EXPECT_EQ(per_step_data.step_number(), 1);
}

}  // namespace
}  // namespace profiler
}  // namespace tensorflow
