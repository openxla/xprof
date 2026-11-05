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
#include <gtest/gtest.h>
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/tsl/profiler/utils/timespan.h"
#include "tsl/platform/protobuf.h"
#include "xprof/convert/data_table_utils.h"
#include "plugin/xprof/protobuf/hardware_types.pb.h"
#include "plugin/xprof/protobuf/input_pipeline.pb.h"
#include "plugin/xprof/protobuf/op_stats.pb.h"
#include "plugin/xprof/protobuf/steps_db.pb.h"
#include "plugin/xprof/protobuf/tpu_input_pipeline.pb.h"
#include "xprof/utils/event_span.h"
#include "xprof/utils/op_metrics_db_utils.h"

namespace tensorflow {
namespace profiler {
namespace {

using ::tensorflow::profiler::CoreDetails;
using ::tensorflow::profiler::OpMetricsDb;
using ::tensorflow::profiler::StepDatabaseResult;
using ::tensorflow::profiler::StepEvents;

constexpr uint64_t kPicosPerMilli = 1000000000;

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
  tsl::protobuf::Map<std::string, uint64_t>& category_ps =
      *step_breakdown.mutable_category_ps();
  category_ps[tensorflow::profiler::kIdle] = 300;
  category_ps[xla::HloOpcodeString(xla::HloOpcode::kMultiply)] = 300;
  category_ps[xla::HloOpcodeString(xla::HloOpcode::kAllGather)] = 300;
  category_ps[xla::HloOpcodeString(xla::HloOpcode::kAsyncStart)] = 50;
  category_ps[xla::HloOpcodeString(xla::HloOpcode::kAsyncDone)] = 50;
  sir.mutable_step_breakdown()->PackFrom(step_breakdown);
  tsl::protobuf::Map<uint32_t, CoreDetails> core_details_map;
  MayFixTpuStepAnalysis(host_step_events, OpMetricsDb(), step_db,
                        core_details_map);
  tensorflow::profiler::GenericStepBreakdown updated_step_breakdown;
  sir.step_breakdown().UnpackTo(&updated_step_breakdown);
  const tsl::protobuf::Map<std::string, uint64_t>& updated_category_ps =
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
  tsl::protobuf::Map<uint32_t, tensorflow::profiler::StepInfoResult>& sipc_map =
      *pcsi->mutable_step_info_per_core();
  tensorflow::profiler::StepInfoResult& sir = sipc_map[/* core_id= */ 2];
  sir.set_step_num(step_num);
  sir.set_begin_ps(40);
  sir.set_duration_ps(1000);
  tensorflow::profiler::GenericStepBreakdown step_breakdown;
  tsl::protobuf::Map<std::string, uint64_t>& category_ps =
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
  tsl::protobuf::Map<std::string, uint64_t>& category_ps2 =
      *step_breakdown2.mutable_category_ps();
  category_ps2[tensorflow::profiler::kIdle] = 250;
  category_ps2[xla::HloOpcodeString(xla::HloOpcode::kMultiply)] = 300;
  category_ps2[xla::HloOpcodeString(xla::HloOpcode::kAllGather)] = 250;
  category_ps2[xla::HloOpcodeString(xla::HloOpcode::kAsyncStart)] = 50;
  category_ps2[xla::HloOpcodeString(xla::HloOpcode::kAsyncDone)] = 50;
  sir2.mutable_step_breakdown()->PackFrom(step_breakdown2);
  tsl::protobuf::Map<uint32_t, CoreDetails> core_details_map;
  OpMetricsDb device_op_metrics_db;
  MayFixTpuStepAnalysis(host_step_events, device_op_metrics_db, step_db,
                        core_details_map);
  tensorflow::profiler::GenericStepBreakdown updated_step_breakdown;
  sir.step_breakdown().UnpackTo(&updated_step_breakdown);
  const tsl::protobuf::Map<std::string, uint64_t>& updated_category_ps =
      updated_step_breakdown.category_ps();
  EXPECT_EQ(updated_category_ps.at(tensorflow::profiler::kIdle), 48);
  ASSERT_TRUE(updated_category_ps.contains(
      xla::HloOpcodeString(xla::HloOpcode::kInfeed)));
  EXPECT_EQ(
      updated_category_ps.at(xla::HloOpcodeString(xla::HloOpcode::kInfeed)),
      252);
  tensorflow::profiler::GenericStepBreakdown updated_step_breakdown2;
  sir2.step_breakdown().UnpackTo(&updated_step_breakdown2);
  const tsl::protobuf::Map<std::string, uint64_t>& updated_category_ps2 =
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
  tsl::protobuf::Map<uint32_t, tensorflow::profiler::StepInfoResult>& sipc_map =
      *pcsi->mutable_step_info_per_core();
  tensorflow::profiler::StepInfoResult& sir = sipc_map[/* core_id= */ 2];
  sir.set_step_num(step_num);
  sir.set_begin_ps(40);
  sir.set_duration_ps(1000);
  tensorflow::profiler::GenericStepBreakdown step_breakdown;
  tsl::protobuf::Map<std::string, uint64_t>& category_ps =
      *step_breakdown.mutable_category_ps();
  category_ps[tensorflow::profiler::kIdle] = 300;
  category_ps[xla::HloOpcodeString(xla::HloOpcode::kMultiply)] = 300;
  category_ps[xla::HloOpcodeString(xla::HloOpcode::kAllGather)] = 300;
  category_ps[xla::HloOpcodeString(xla::HloOpcode::kAsyncStart)] = 50;
  category_ps[xla::HloOpcodeString(xla::HloOpcode::kInfeed)] = 50;
  sir.mutable_step_breakdown()->PackFrom(step_breakdown);
  tsl::protobuf::Map<uint32_t, CoreDetails> core_details_map;
  OpMetricsDb device_op_metrics_db;
  device_op_metrics_db.add_metrics_db()->set_category(
      std::string(xla::HloOpcodeString(xla::HloOpcode::kInfeed)));
  MayFixTpuStepAnalysis(host_step_events, device_op_metrics_db, step_db,
                        core_details_map);
  tensorflow::profiler::GenericStepBreakdown updated_step_breakdown;
  sir.step_breakdown().UnpackTo(&updated_step_breakdown);
  const tsl::protobuf::Map<std::string, uint64_t>& updated_category_ps =
      updated_step_breakdown.category_ps();
  EXPECT_EQ(updated_category_ps.at(tensorflow::profiler::kIdle), 300);
  EXPECT_EQ(
      updated_category_ps.at(xla::HloOpcodeString(xla::HloOpcode::kInfeed)),
      50);
}

TEST(TfOpStatsToInputPipelineAnalysisTest, EnsureSparseCoreStepsSetStepNumber) {
  PerCoreStepInfo per_core_step_info;
  per_core_step_info.set_step_num(1);
  tsl::protobuf::Map<uint32_t, StepInfoResult>& step_info_per_core =
      *per_core_step_info.mutable_step_info_per_core();
  StepInfoResult& step_info =
      step_info_per_core[/* core_id= */ kSparseCoreIndexStart + 1];
  step_info.set_step_num(1);
  step_info.set_begin_ps(100);
  step_info.set_duration_ps(1000);
  GenericStepBreakdown sparse_core_step_breakdown;
  tsl::protobuf::Map<std::string, uint64_t>& category_ps =
      *sparse_core_step_breakdown.mutable_category_ps();
  category_ps[tensorflow::profiler::kIdle] = 500;
  step_info.mutable_step_breakdown()->PackFrom(sparse_core_step_breakdown);

  tsl::protobuf::Map<uint32_t, CoreDetails> core_details_map;
  CoreDetails& core_details =
      core_details_map[/* core_id= */ kSparseCoreIndexStart + 1];
  core_details.set_is_sparse_core(true);

  PerTpuStepDetails per_step_data =
      ComputeTpuPerStepDataAcrossCores(per_core_step_info, core_details_map);
  EXPECT_EQ(per_step_data.step_number(), 1);
}

TEST(TfOpStatsToInputPipelineAnalysisTest,
     ComputeTpuPerStepDataAcrossCoresSparseCoreOnly) {
  PerCoreStepInfo per_core_step_info;
  per_core_step_info.set_step_num(1);
  tsl::protobuf::Map<uint32_t, StepInfoResult>& step_info_per_core =
      *per_core_step_info.mutable_step_info_per_core();

  // Add a SparseCore step
  uint32_t sparse_core_id = kSparseCoreIndexStart + 1;
  StepInfoResult& sc_step_info = step_info_per_core[sparse_core_id];
  sc_step_info.set_step_num(1);
  sc_step_info.set_begin_ps(100);
  sc_step_info.set_duration_ps(1000 * 1e9);
  GenericStepBreakdown sparse_core_step_breakdown;
  tsl::protobuf::Map<std::string, uint64_t>& sc_category_ps =
      *sparse_core_step_breakdown.mutable_category_ps();
  sc_category_ps[tensorflow::profiler::kIdle] = 200 * 1e9;
  sc_step_info.mutable_step_breakdown()->PackFrom(sparse_core_step_breakdown);

  // Setup CoreDetails for the SparseCore
  tsl::protobuf::Map<uint32_t, CoreDetails> core_details_map;
  CoreDetails& sc_core_details = core_details_map[sparse_core_id];
  sc_core_details.set_is_sparse_core(true);

  PerTpuStepDetails per_step_data =
      ComputeTpuPerStepDataAcrossCores(per_core_step_info, core_details_map);

  // Verify that TensorCore related fields are not UINT64_MAX and are 0.
  EXPECT_EQ(per_step_data.tc_compute_time_ms(), 0);
  EXPECT_EQ(per_step_data.tc_infeed_time_ms(), 0);
  EXPECT_EQ(per_step_data.tc_outfeed_time_ms(), 0);
  EXPECT_EQ(per_step_data.tc_idle_time_ms(), 0);
  EXPECT_EQ(per_step_data.all_reduce_compute_time_ms(), 0);
  EXPECT_EQ(per_step_data.all_reduce_sync_time_ms(), 0);
  EXPECT_EQ(per_step_data.scv0_compute_time_ms(), 0);
  EXPECT_EQ(per_step_data.scv0_infeed_time_ms(), 0);
  EXPECT_EQ(per_step_data.host_transfer_ms(), 0);

  // Also check SparseCore fields are populated correctly.
  EXPECT_EQ(per_step_data.sc_compute_time_ms(), 800);
  EXPECT_EQ(per_step_data.sc_idle_time_ms(), 200);
  EXPECT_EQ(per_step_data.sc_outfeed_time_ms(), 0);
  EXPECT_EQ(per_step_data.sc_infeed_time_ms(), 0);
  EXPECT_EQ(per_step_data.sc_step_time_ms(), 1000);
  EXPECT_EQ(per_step_data.tc_active_time_ms(), 0);
  EXPECT_EQ(per_step_data.sc_active_time_ms(), 800);
  EXPECT_EQ(per_step_data.active_step_time_ms(), 800);
}

TEST(TfOpStatsToInputPipelineAnalysisTest,
     ComputeTpuPerStepDataAcrossCoresExcludesScSyncWaitAndComputesActiveTime) {
  PerCoreStepInfo per_core_step_info;
  per_core_step_info.set_step_num(1);
  tsl::protobuf::Map<uint32_t, StepInfoResult>& step_info_per_core =
      *per_core_step_info.mutable_step_info_per_core();

  // Add a TensorCore step (1000 ms total: 600 ms compute, 100 ms infeed,
  // 200 ms async-done/call-done, 100 ms idle -> 600 ms active).
  uint32_t tc_core_id = 0;
  StepInfoResult& tc_step_info = step_info_per_core[tc_core_id];
  tc_step_info.set_step_num(1);
  tc_step_info.set_begin_ps(0);
  tc_step_info.set_duration_ps(1000 * 1000000000ULL);
  GenericStepBreakdown tc_breakdown;
  tsl::protobuf::Map<std::string, uint64_t>& tc_cat =
      *tc_breakdown.mutable_category_ps();
  tc_cat[xla::HloOpcodeString(xla::HloOpcode::kMultiply)] = 600 * 1000000000ULL;
  tc_cat[xla::HloOpcodeString(xla::HloOpcode::kInfeed)] = 100 * 1000000000ULL;
  tc_cat[xla::HloOpcodeString(xla::HloOpcode::kAsyncDone)] =
      200 * 1000000000ULL;
  tc_cat[tensorflow::profiler::kIdle] = 100 * 1000000000ULL;
  tc_step_info.mutable_step_breakdown()->PackFrom(tc_breakdown);

  // Add a SparseCore step via GenericStepBreakdown (1000 ms total:
  // 500 ms compute, 300 ms async-done/call-done, 100 ms infeed, 100 ms idle
  // -> 500 ms active).
  uint32_t sc_core_id = kSparseCoreIndexStart + 1;
  StepInfoResult& sc_step_info = step_info_per_core[sc_core_id];
  sc_step_info.set_step_num(1);
  sc_step_info.set_begin_ps(0);
  sc_step_info.set_duration_ps(1000 * 1000000000ULL);
  GenericStepBreakdown sc_breakdown;
  tsl::protobuf::Map<std::string, uint64_t>& sc_cat =
      *sc_breakdown.mutable_category_ps();
  sc_cat[xla::HloOpcodeString(xla::HloOpcode::kAsyncDone)] =
      300 * 1000000000ULL;
  sc_cat[xla::HloOpcodeString(xla::HloOpcode::kInfeed)] = 100 * 1000000000ULL;
  sc_cat[tensorflow::profiler::kIdle] = 100 * 1000000000ULL;
  sc_step_info.mutable_step_breakdown()->PackFrom(sc_breakdown);

  tsl::protobuf::Map<uint32_t, CoreDetails> core_details_map;
  core_details_map[tc_core_id].set_is_sparse_core(false);
  core_details_map[sc_core_id].set_is_sparse_core(true);

  PerTpuStepDetails per_step_data =
      ComputeTpuPerStepDataAcrossCores(per_core_step_info, core_details_map);

  // Legacy tc_compute_time_ms/tc_idle_time_ms preserve historical semantics
  // for MayFixTpuStepAnalysis, while tc_active_time_ms excludes async-done.
  EXPECT_NEAR(per_step_data.tc_compute_time_ms(), 800.0, 1e-6);
  EXPECT_NEAR(per_step_data.tc_idle_time_ms(), 100.0, 1e-6);
  EXPECT_NEAR(per_step_data.tc_infeed_time_ms(), 100.0, 1e-6);
  EXPECT_NEAR(per_step_data.sc_compute_time_ms(), 500.0, 1e-6);
  EXPECT_NEAR(per_step_data.sc_idle_time_ms(), 400.0, 1e-6);
  EXPECT_NEAR(per_step_data.sc_infeed_time_ms(), 100.0, 1e-6);
  EXPECT_NEAR(per_step_data.tc_active_time_ms(), 600.0, 1e-6);
  EXPECT_NEAR(per_step_data.sc_active_time_ms(), 500.0, 1e-6);
  EXPECT_NEAR(per_step_data.active_step_time_ms(), 600.0, 1e-6);
}

TEST(TfOpStatsToInputPipelineAnalysisTest,
     MayFixTpuStepAnalysisFlatOpMetricsDb) {
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
  tsl::protobuf::Map<uint32_t, tensorflow::profiler::StepInfoResult>& sipc_map =
      *pcsi->mutable_step_info_per_core();
  tensorflow::profiler::StepInfoResult& sir = sipc_map[/* core_id= */ 2];
  sir.set_step_num(step_num);
  sir.set_begin_ps(40);
  sir.set_duration_ps(1000);
  tensorflow::profiler::GenericStepBreakdown step_breakdown;
  tsl::protobuf::Map<std::string, uint64_t>& category_ps =
      *step_breakdown.mutable_category_ps();
  category_ps[tensorflow::profiler::kIdle] = 300;
  category_ps[xla::HloOpcodeString(xla::HloOpcode::kMultiply)] = 300;
  category_ps[xla::HloOpcodeString(xla::HloOpcode::kAllGather)] = 300;
  category_ps[xla::HloOpcodeString(xla::HloOpcode::kAsyncStart)] = 50;
  category_ps[xla::HloOpcodeString(xla::HloOpcode::kAsyncDone)] = 50;
  sir.mutable_step_breakdown()->PackFrom(step_breakdown);
  tsl::protobuf::Map<uint32_t, CoreDetails> core_details_map;
  FlatOpMetricsDb flat_device_op_metrics_db;
  MayFixTpuStepAnalysis(host_step_events, flat_device_op_metrics_db, step_db,
                        core_details_map);
  tensorflow::profiler::GenericStepBreakdown updated_step_breakdown;
  sir.step_breakdown().UnpackTo(&updated_step_breakdown);
  const tsl::protobuf::Map<std::string, uint64_t>& updated_category_ps =
      updated_step_breakdown.category_ps();
  EXPECT_EQ(updated_category_ps.at(tensorflow::profiler::kIdle), 90);
  ASSERT_TRUE(updated_category_ps.contains(
      xla::HloOpcodeString(xla::HloOpcode::kInfeed)));
  EXPECT_EQ(
      updated_category_ps.at(xla::HloOpcodeString(xla::HloOpcode::kInfeed)),
      210);
}

TEST(TfOpStatsToInputPipelineAnalysisTest,
     SkipMayFixTpuStepAnalysisWhenInfeedExistsFlatOpMetricsDb) {
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
  tsl::protobuf::Map<uint32_t, tensorflow::profiler::StepInfoResult>& sipc_map =
      *pcsi->mutable_step_info_per_core();
  tensorflow::profiler::StepInfoResult& sir = sipc_map[/* core_id= */ 2];
  sir.set_step_num(step_num);
  sir.set_begin_ps(40);
  sir.set_duration_ps(1000);
  tensorflow::profiler::GenericStepBreakdown step_breakdown;
  tsl::protobuf::Map<std::string, uint64_t>& category_ps =
      *step_breakdown.mutable_category_ps();
  category_ps[tensorflow::profiler::kIdle] = 300;
  category_ps[xla::HloOpcodeString(xla::HloOpcode::kMultiply)] = 300;
  category_ps[xla::HloOpcodeString(xla::HloOpcode::kAllGather)] = 300;
  category_ps[xla::HloOpcodeString(xla::HloOpcode::kAsyncStart)] = 50;
  category_ps[xla::HloOpcodeString(xla::HloOpcode::kInfeed)] = 50;
  sir.mutable_step_breakdown()->PackFrom(step_breakdown);
  tsl::protobuf::Map<uint32_t, CoreDetails> core_details_map;
  FlatOpMetricsDb flat_device_op_metrics_db;
  flat_device_op_metrics_db.add_op_instances()->set_category(
      std::string(xla::HloOpcodeString(xla::HloOpcode::kInfeed)));
  MayFixTpuStepAnalysis(host_step_events, flat_device_op_metrics_db, step_db,
                        core_details_map);
  tensorflow::profiler::GenericStepBreakdown updated_step_breakdown;
  sir.step_breakdown().UnpackTo(&updated_step_breakdown);
  const tsl::protobuf::Map<std::string, uint64_t>& updated_category_ps =
      updated_step_breakdown.category_ps();
  EXPECT_EQ(updated_category_ps.at(tensorflow::profiler::kIdle), 300);
  EXPECT_EQ(
      updated_category_ps.at(xla::HloOpcodeString(xla::HloOpcode::kInfeed)),
      50);
}

TEST(TfOpStatsToInputPipelineAnalysisTest,
     ComputeTpuPerStepDataAcrossCoresUsesSparseCoreStepBreakdownBusyPs) {
  PerCoreStepInfo per_core_step_info;
  per_core_step_info.set_step_num(1);
  tsl::protobuf::Map<uint32_t, StepInfoResult>& step_info_per_core =
      *per_core_step_info.mutable_step_info_per_core();

  uint32_t sc_core_id = kSparseCoreIndexStart + 1;
  StepInfoResult& sc_step_info = step_info_per_core[sc_core_id];
  sc_step_info.set_step_num(1);
  sc_step_info.set_begin_ps(0);
  sc_step_info.set_duration_ps(1000 * 1000000000ULL);
  SparseCoreStepBreakdown sc_breakdown;
  sc_breakdown.set_sc_compute_ps(450 * 1000000000ULL);
  sc_breakdown.set_sc_busy_ps(450 * 1000000000ULL);
  sc_breakdown.set_sc_infeed_ps(50 * 1000000000ULL);
  sc_breakdown.set_sc_idle_ps(500 * 1000000000ULL);
  sc_step_info.mutable_step_breakdown()->PackFrom(sc_breakdown);

  tsl::protobuf::Map<uint32_t, CoreDetails> core_details_map;
  core_details_map[sc_core_id].set_is_sparse_core(true);

  PerTpuStepDetails per_step_data =
      ComputeTpuPerStepDataAcrossCores(per_core_step_info, core_details_map);

  EXPECT_NEAR(per_step_data.sc_compute_time_ms(), 450.0, 1e-6);
  EXPECT_NEAR(per_step_data.sc_active_time_ms(), 450.0, 1e-6);
  EXPECT_NEAR(per_step_data.sc_idle_time_ms(), 500.0, 1e-6);
  EXPECT_NEAR(per_step_data.sc_infeed_time_ms(), 50.0, 1e-6);
  EXPECT_NEAR(per_step_data.active_step_time_ms(), 450.0, 1e-6);
}

TEST(TfOpStatsToInputPipelineAnalysisTest,
     ComputeTpuPerStepDataAcrossCoresAveragesTcBusyAndClampsToDuration) {
  PerCoreStepInfo per_core_step_info;
  per_core_step_info.set_step_num(1);
  tsl::protobuf::Map<uint32_t, StepInfoResult>& step_info_per_core =
      *per_core_step_info.mutable_step_info_per_core();

  // Core 1 has 800 ms of optimal step time (1000 - 100 infeed - 100 idle) and
  // 750 ms of TensorCore busy time.
  StepInfoResult& core1_step_info = step_info_per_core[/*core_id=*/1];
  core1_step_info.set_step_num(1);
  core1_step_info.set_duration_ps(1000 * kPicosPerMilli);
  TpuStepBreakdown core1_breakdown;
  core1_breakdown.set_infeed_duration_ps(100 * kPicosPerMilli);
  core1_breakdown.set_tc_idle_ps(100 * kPicosPerMilli);
  core1_breakdown.set_wait_for_megacore_fusion_peer_duration_ps(50 *
                                                                kPicosPerMilli);
  core1_breakdown.set_tc_busy_ps(750 * kPicosPerMilli);
  core1_step_info.mutable_step_breakdown()->PackFrom(core1_breakdown);

  // Core 2 reports 600 ms of TensorCore busy time.
  StepInfoResult& core2_step_info = step_info_per_core[/*core_id=*/2];
  core2_step_info.set_step_num(1);
  core2_step_info.set_duration_ps(1000 * kPicosPerMilli);
  TpuStepBreakdown core2_breakdown;
  core2_breakdown.set_tc_busy_ps(600 * kPicosPerMilli);
  core2_step_info.mutable_step_breakdown()->PackFrom(core2_breakdown);

  PerTpuStepDetails per_step_data = ComputeTpuPerStepDataAcrossCores(
      per_core_step_info, tsl::protobuf::Map<uint32_t, CoreDetails>());

  EXPECT_NEAR(per_step_data.tc_compute_time_ms(), 800.0, 1e-6);
  // The step's active time is the average across cores: (750 + 600) / 2.
  EXPECT_NEAR(per_step_data.tc_active_time_ms(), 675.0, 1e-6);
  EXPECT_NEAR(per_step_data.active_step_time_ms(), 675.0, 1e-6);
}

TEST(TfOpStatsToInputPipelineAnalysisTest,
     ComputeTpuPerStepDataAcrossCoresClampsInconsistentBreakdowns) {
  PerCoreStepInfo per_core_step_info;
  per_core_step_info.set_step_num(1);
  tsl::protobuf::Map<uint32_t, StepInfoResult>& step_info_per_core =
      *per_core_step_info.mutable_step_info_per_core();

  // A zero-length TensorCore step must not divide by zero.
  StepInfoResult& empty_step_info = step_info_per_core[/*core_id=*/1];
  empty_step_info.set_step_num(1);
  TpuStepBreakdown empty_breakdown;
  empty_breakdown.set_infeed_duration_ps(10);
  empty_step_info.mutable_step_breakdown()->PackFrom(empty_breakdown);

  // Waits longer than the step must not wrap the optimal step time around.
  StepInfoResult& tc_step_info = step_info_per_core[/*core_id=*/2];
  tc_step_info.set_step_num(1);
  tc_step_info.set_duration_ps(100);
  TpuStepBreakdown tc_breakdown;
  tc_breakdown.set_infeed_duration_ps(150);
  tc_step_info.mutable_step_breakdown()->PackFrom(tc_breakdown);

  // SparseCore infeed longer than the step must not wrap compute time around.
  StepInfoResult& sc_step_info =
      step_info_per_core[/*core_id=*/kSparseCoreIndexStart + 1];
  sc_step_info.set_step_num(1);
  sc_step_info.set_duration_ps(100);
  SparseCoreStepBreakdown sc_breakdown;
  sc_breakdown.set_sc_infeed_ps(120);
  sc_step_info.mutable_step_breakdown()->PackFrom(sc_breakdown);

  PerTpuStepDetails per_step_data = ComputeTpuPerStepDataAcrossCores(
      per_core_step_info, tsl::protobuf::Map<uint32_t, CoreDetails>());

  EXPECT_DOUBLE_EQ(per_step_data.infeed_percent_maximum(), 150.0);
  EXPECT_EQ(per_step_data.tc_compute_time_ms(), 0);
  EXPECT_EQ(per_step_data.tc_active_time_ms(), 0);
  EXPECT_EQ(per_step_data.sc_compute_time_ms(), 0);
  EXPECT_EQ(per_step_data.sc_active_time_ms(), 0);
}

TEST(TfOpStatsToInputPipelineAnalysisTest,
     ConvertOpStatsToInputPipelineAnalysisSummarizesActiveTimes) {
  OpStats op_stats;
  op_stats.mutable_run_environment()->set_hardware_type(TPU);
  PerCoreStepInfo& per_core_step_info =
      *op_stats.mutable_step_db()->add_step_sequence();
  per_core_step_info.set_step_num(1);
  tsl::protobuf::Map<uint32_t, StepInfoResult>& step_info_per_core =
      *per_core_step_info.mutable_step_info_per_core();

  StepInfoResult& tc_step_info = step_info_per_core[/*core_id=*/1];
  tc_step_info.set_step_num(1);
  tc_step_info.set_duration_ps(1000 * kPicosPerMilli);
  TpuStepBreakdown tc_breakdown;
  tc_breakdown.set_tc_busy_ps(600 * kPicosPerMilli);
  tc_step_info.mutable_step_breakdown()->PackFrom(tc_breakdown);

  // The SparseCore has no CoreDetails, so only its step data shows that the
  // profile has a SparseCore.
  StepInfoResult& sc_step_info =
      step_info_per_core[/*core_id=*/kSparseCoreIndexStart + 1];
  sc_step_info.set_step_num(1);
  sc_step_info.set_duration_ps(1000 * kPicosPerMilli);
  SparseCoreStepBreakdown sc_breakdown;
  sc_breakdown.set_sc_busy_ps(700 * kPicosPerMilli);
  sc_breakdown.set_sc_idle_ps(300 * kPicosPerMilli);
  sc_step_info.mutable_step_breakdown()->PackFrom(sc_breakdown);

  InputPipelineAnalysisResult result =
      ConvertOpStatsToInputPipelineAnalysis(op_stats);

  TpuStepTimeBreakdown breakdown;
  ASSERT_TRUE(result.step_time_breakdown().UnpackTo(&breakdown));
  EXPECT_NEAR(breakdown.tc_active_ms_summary().average(), 600.0, 1e-6);
  EXPECT_NEAR(breakdown.active_step_time_ms_summary().average(), 700.0, 1e-6);
  ASSERT_TRUE(breakdown.has_sparse_core_step_summary());
  const SparseCoreStepSummary& sc_summary =
      breakdown.sparse_core_step_summary();
  EXPECT_NEAR(sc_summary.sc_active_ms_summary().average(), 700.0, 1e-6);
  EXPECT_NEAR(sc_summary.sc_step_time_ms_summary().average(), 1000.0, 1e-6);
}

TEST(TfOpStatsToInputPipelineAnalysisTest,
     ConvertOpStatsToInputPipelineAnalysisKeepsZeroActiveTimeForWaitingCore) {
  OpStats op_stats;
  op_stats.mutable_run_environment()->set_hardware_type(TPU);
  PerCoreStepInfo& per_core_step_info =
      *op_stats.mutable_step_db()->add_step_sequence();
  per_core_step_info.set_step_num(1);
  tsl::protobuf::Map<uint32_t, StepInfoResult>& step_info_per_core =
      *per_core_step_info.mutable_step_info_per_core();

  StepInfoResult& tc_step_info = step_info_per_core[/*core_id=*/1];
  tc_step_info.set_step_num(1);
  tc_step_info.set_duration_ps(1000 * kPicosPerMilli);
  // Core 1 (GenericStepBreakdown) only waits: 600 ms on async-done and 400 ms
  // idle.
  GenericStepBreakdown tc_breakdown;
  tsl::protobuf::Map<std::string, uint64_t>& category_ps =
      *tc_breakdown.mutable_category_ps();
  category_ps[xla::HloOpcodeString(xla::HloOpcode::kAsyncDone)] =
      600 * kPicosPerMilli;
  category_ps[tensorflow::profiler::kIdle] = 400 * kPicosPerMilli;
  tc_step_info.mutable_step_breakdown()->PackFrom(tc_breakdown);

  // Core 2 (TpuStepBreakdown) also has 0 tc_busy_ps and 400 ms idle.
  StepInfoResult& tc2_step_info = step_info_per_core[/*core_id=*/2];
  tc2_step_info.set_step_num(1);
  tc2_step_info.set_duration_ps(1000 * kPicosPerMilli);
  TpuStepBreakdown tc2_breakdown;
  tc2_breakdown.set_tc_busy_ps(0);
  tc2_breakdown.set_tc_idle_ps(400 * kPicosPerMilli);
  tc2_step_info.mutable_step_breakdown()->PackFrom(tc2_breakdown);

  // SparseCore (SparseCoreStepBreakdown) also has 0 sc_busy_ps and 400 ms idle.
  StepInfoResult& sc_step_info =
      step_info_per_core[/*core_id=*/kSparseCoreIndexStart + 1];
  sc_step_info.set_step_num(1);
  sc_step_info.set_duration_ps(1000 * kPicosPerMilli);
  SparseCoreStepBreakdown sc_breakdown;
  sc_breakdown.set_sc_busy_ps(0);
  sc_breakdown.set_sc_idle_ps(400 * kPicosPerMilli);
  sc_step_info.mutable_step_breakdown()->PackFrom(sc_breakdown);

  InputPipelineAnalysisResult result =
      ConvertOpStatsToInputPipelineAnalysis(op_stats);

  TpuStepTimeBreakdown breakdown;
  ASSERT_TRUE(result.step_time_breakdown().UnpackTo(&breakdown));
  // Compute time still counts the async-done wait, but active time does not,
  // and it is not replaced by compute time when it is zero.
  EXPECT_NEAR(breakdown.tc_compute_ms_summary().average(), 600.0, 1e-6);
  EXPECT_EQ(breakdown.tc_active_ms_summary().average(), 0);
  EXPECT_EQ(breakdown.active_step_time_ms_summary().average(), 0);
  const SparseCoreStepSummary& sc_summary =
      breakdown.sparse_core_step_summary();
  EXPECT_NEAR(sc_summary.sc_compute_ms_summary().average(), 600.0, 1e-6);
  EXPECT_EQ(sc_summary.sc_active_ms_summary().average(), 0);
}

TEST(TfOpStatsToInputPipelineAnalysisTest,
     ComputeTpuPerStepDataUsesActiveCategoryWhenSparseCoreIoOverlaps) {
  PerCoreStepInfo per_core_step_info;
  per_core_step_info.set_step_num(1);
  StepInfoResult& sc_step_info =
      (*per_core_step_info
            .mutable_step_info_per_core())[kSparseCoreIndexStart + 1];
  sc_step_info.set_step_num(1);
  sc_step_info.set_duration_ps(1000 * kPicosPerMilli);
  GenericStepBreakdown sc_generic;
  tsl::protobuf::Map<std::string, uint64_t>& category_ps =
      *sc_generic.mutable_category_ps();
  category_ps[xla::HloOpcodeString(xla::HloOpcode::kInfeed)] =
      500 * kPicosPerMilli;
  category_ps[xla::HloOpcodeString(xla::HloOpcode::kOutfeed)] =
      500 * kPicosPerMilli;
  category_ps["Sparse Core Ops"] = 600 * kPicosPerMilli;
  sc_step_info.mutable_step_breakdown()->PackFrom(sc_generic);

  PerTpuStepDetails per_step_data = ComputeTpuPerStepDataAcrossCores(
      per_core_step_info, tsl::protobuf::Map<uint32_t, CoreDetails>());

  EXPECT_NEAR(per_step_data.sc_compute_time_ms(), 600.0, 1e-6);
  EXPECT_NEAR(per_step_data.sc_active_time_ms(), 600.0, 1e-6);
  EXPECT_NEAR(per_step_data.active_step_time_ms(), 600.0, 1e-6);
}

TEST(TfOpStatsToInputPipelineAnalysisTest,
     GenerateTpuInputPipelineAnalysisDataTableAddsActiveTimes) {
  TpuStepTimeBreakdown breakdown;
  breakdown.mutable_tc_active_ms_summary()->set_average(20);
  breakdown.mutable_active_step_time_ms_summary()->set_average(25);
  SparseCoreStepSummary& sc_summary =
      *breakdown.mutable_sparse_core_step_summary();
  // Minimum is 0 (e.g. when one step has no SparseCore events), but average >
  // 0.
  sc_summary.mutable_sc_step_time_ms_summary()->set_minimum(0);
  sc_summary.mutable_sc_step_time_ms_summary()->set_average(40);
  sc_summary.mutable_sc_compute_ms_summary()->set_average(30);
  sc_summary.mutable_sc_active_ms_summary()->set_average(25);
  InputPipelineAnalysisResult result;
  result.mutable_step_time_breakdown()->PackFrom(breakdown);

  DataTable data_table = GenerateTpuInputPipelineAnalysisDataTable(result);

  const auto& properties = data_table.GetCustomProperties();
  EXPECT_EQ(properties.at("tc_active_ms_average"), "20.00");
  EXPECT_EQ(properties.at("active_steptime_ms_average"), "25.00");
  EXPECT_EQ(properties.at("sc_compute_ms_average"), "30.00");
  EXPECT_EQ(properties.at("sc_active_ms_average"), "25.00");
  EXPECT_EQ(properties.at("sc_step_time_ms_average"), "40.0");
}

TEST(TfOpStatsToInputPipelineAnalysisTest, UnrecognizedOpIsUnknown) {
  OpMetricsDb host_tf_metrics_db;
  auto* metrics = host_tf_metrics_db.add_metrics_db();
  metrics->set_name("UnknownOp::UnknownCategory");
  metrics->set_category("Dataset");
  metrics->set_self_time_ps(1000000);  // 1 us

  InputPipelineAnalysisResult result;
  GenerateHostResult(host_tf_metrics_db, &result);

  EXPECT_GT(result.input_time_breakdown().unclassified_non_enqueue_us(), 0);
  EXPECT_EQ(result.input_op_details(0).category(), "Unknown");
}

TEST(TfOpStatsToInputPipelineAnalysisTest, RecognizedOpIsCorrect) {
  OpMetricsDb host_tf_metrics_db;
  auto* metrics = host_tf_metrics_db.add_metrics_db();
  metrics->set_name("Iterator::TFRecord");
  metrics->set_category("Dataset");
  metrics->set_self_time_ps(1000000);  // 1 us

  InputPipelineAnalysisResult result;
  GenerateHostResult(host_tf_metrics_db, &result);

  EXPECT_GT(result.input_time_breakdown().demanded_file_read_us(), 0);
  EXPECT_EQ(result.input_op_details(0).category(), "Demanded file read");
}

TEST(TfOpStatsToInputPipelineAnalysisTest, ArrayRecordIsCorrect) {
  OpMetricsDb host_tf_metrics_db;
  auto* metrics = host_tf_metrics_db.add_metrics_db();
  metrics->set_name("Iterator::ArrayRecord");
  metrics->set_category("Dataset");
  metrics->set_self_time_ps(1000000);  // 1 us

  InputPipelineAnalysisResult result;
  GenerateHostResult(host_tf_metrics_db, &result);

  EXPECT_GT(result.input_time_breakdown().demanded_file_read_us(), 0);
  EXPECT_EQ(result.input_op_details(0).category(), "Demanded file read");
}

}  // namespace
}  // namespace profiler
}  // namespace tensorflow
