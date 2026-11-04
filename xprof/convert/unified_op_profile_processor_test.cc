/* Copyright 2026 The OpenXLA Authors. All Rights Reserved.

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

#include "xprof/convert/unified_op_profile_processor.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status_matchers.h"
#include "xla/tsl/platform/env.h"
#include "tsl/platform/path.h"
#include "tsl/profiler/protobuf/xplane.pb.h"
#include "xprof/convert/file_utils.h"
#include "xprof/convert/op_profile_processor.h"
#include "xprof/convert/repository.h"
#include "xprof/convert/tool_options.h"
#include "xprof/convert/unified_profile_processor_factory.h"
#include "xprof/convert/unified_tools_registration.h"
#include "plugin/xprof/protobuf/flat_op_metrics.pb.h"
#include "plugin/xprof/protobuf/op_metrics.pb.h"
#include "plugin/xprof/protobuf/op_profile.pb.h"
#include "plugin/xprof/protobuf/op_stats.pb.h"

namespace xprof {
namespace {

using ::tensorflow::profiler::FlatOpMetrics;
using ::tensorflow::profiler::OpMetrics;
using ::tensorflow::profiler::OpMetricsDb;
using ::tensorflow::profiler::OpStats;
using ::tensorflow::profiler::PerfEnv;
using ::tensorflow::profiler::SessionSnapshot;
using ::tensorflow::profiler::ToolOptions;
using ::tensorflow::profiler::XEvent;
using ::tensorflow::profiler::XEventMetadata;
using ::tensorflow::profiler::XLine;
using ::tensorflow::profiler::XPlane;
using ::tensorflow::profiler::XSpace;
using ::tensorflow::profiler::XStat;
using ::testing::HasSubstr;
using ::testing::Not;

TEST(UnifiedOpProfileProcessorTest, MinimalTest) {
  RegisterUnifiedToolRegistrations();
  ToolOptions options;
  auto processor = UnifiedProfileProcessorFactory::GetInstance().Create(
      "op_profile", options);
  ASSERT_NE(processor, nullptr);

  std::string session_dir = tsl::io::JoinPath(
      testing::TempDir(), "unified_op_profile_processor_test");
  ASSERT_OK(tsl::Env::Default()->RecursivelyCreateDir(session_dir));
  std::string xspace_path =
      tsl::io::JoinPath(session_dir, "test_host.xplane.pb");
  XSpace dummy_space;
  ASSERT_OK(WriteBinaryProto(xspace_path, dummy_space));

  std::vector<std::string> xspace_paths = {xspace_path};
  ASSERT_OK_AND_ASSIGN(
      auto session_snapshot,
      SessionSnapshot::Create(xspace_paths, /*xspaces=*/std::nullopt));

  EXPECT_OK(processor->ProcessSession(session_snapshot, options));
}

TEST(UnifiedOpProfileProcessorTest, FlatOpMetricsDbTest) {
  RegisterUnifiedToolRegistrations();
  ToolOptions options;
  options["use_flat_metric"] = true;

  auto processor = UnifiedProfileProcessorFactory::GetInstance().Create(
      "op_profile", options);
  ASSERT_NE(processor, nullptr);

  std::string session_dir = tsl::io::JoinPath(
      testing::TempDir(), "unified_op_profile_processor_flat_test");
  ASSERT_OK(tsl::Env::Default()->RecursivelyCreateDir(session_dir));
  std::string xspace_path =
      tsl::io::JoinPath(session_dir, "test_host.xplane.pb");
  XSpace dummy_space;
  ASSERT_OK(WriteBinaryProto(xspace_path, dummy_space));

  std::vector<std::string> xspace_paths = {xspace_path};
  ASSERT_OK_AND_ASSIGN(
      auto session_snapshot,
      SessionSnapshot::Create(xspace_paths, /*xspaces=*/std::nullopt));

  EXPECT_OK(processor->ProcessSession(session_snapshot, options));
}

OpStats CreateTestOpStatsWithProgram(uint64_t program_id) {
  OpStats op_stats;
  op_stats.mutable_run_environment()->set_device_type("TPU");
  PerfEnv* perf_env = op_stats.mutable_perf_env();
  perf_env->set_peak_tera_flops_per_second(10.0);
  perf_env->add_peak_bws_giga_bytes_per_second(100.0);
  perf_env->add_peak_bws_giga_bytes_per_second(100.0);
  perf_env->add_peak_bws_giga_bytes_per_second(100.0);
  OpMetricsDb* db = op_stats.mutable_device_op_metrics_db();
  db->set_total_time_ps(1000);
  db->set_total_op_time_ps(1000);
  OpMetrics* op = db->add_metrics_db();
  op->set_name("fusion.1");
  op->set_category("fusion");
  op->set_time_ps(1000);
  op->set_self_time_ps(1000);
  op->set_occurrences(1);
  op->set_hlo_module_id(program_id);
  (*op_stats.mutable_program_id_to_name_map())[program_id] = "jit_step";
  return op_stats;
}

XSpace CreateInferenceServingXSpace(uint64_t program_id) {
  constexpr int64_t kSharedBatchSchedulerContext = 4;
  constexpr int64_t kTpuLaunchContext = 12;

  auto add_event_meta = [](XPlane* plane, int64_t id,
                           const std::string& name) -> XEventMetadata* {
    XEventMetadata& meta = (*plane->mutable_event_metadata())[id];
    meta.set_id(id);
    meta.set_name(name);
    return &meta;
  };
  auto add_stat_meta = [](XPlane* plane, int64_t id, const std::string& name) {
    auto& meta = (*plane->mutable_stat_metadata())[id];
    meta.set_id(id);
    meta.set_name(name);
  };
  auto add_event = [](XLine* line, int64_t meta_id, int64_t offset_ps,
                      int64_t duration_ps) -> XEvent* {
    XEvent* event = line->add_events();
    event->set_metadata_id(meta_id);
    event->set_offset_ps(offset_ps);
    event->set_duration_ps(duration_ps);
    return event;
  };
  auto add_int_stat = [](auto* holder, int64_t meta_id, int64_t val) {
    XStat* stat = holder->add_stats();
    stat->set_metadata_id(meta_id);
    stat->set_int64_value(val);
  };
  auto add_double_stat = [](auto* holder, int64_t meta_id, double val) {
    XStat* stat = holder->add_stats();
    stat->set_metadata_id(meta_id);
    stat->set_double_value(val);
  };
  auto add_str_stat = [](auto* holder, int64_t meta_id,
                         const std::string& val) {
    XStat* stat = holder->add_stats();
    stat->set_metadata_id(meta_id);
    stat->set_str_value(val);
  };

  XSpace space;
  XPlane* host_plane = space.add_planes();
  host_plane->set_id(1);
  host_plane->set_name("/host:CPU");
  add_event_meta(host_plane, 1, "SessionRun");
  add_event_meta(host_plane, 2, "ScheduleWithoutSplit");
  add_event_meta(host_plane, 3, "ProcessBatch");
  add_event_meta(host_plane, 4, "ConcatInputTensors");
  add_stat_meta(host_plane, 1, "_r");
  add_stat_meta(host_plane, 2, "model_id");
  add_stat_meta(host_plane, 3, "_pt");
  add_stat_meta(host_plane, 4, "_p");
  add_stat_meta(host_plane, 5, "batching_input_task_size");
  add_stat_meta(host_plane, 6, "_ct");
  add_stat_meta(host_plane, 7, "_c");
  add_stat_meta(host_plane, 8, "padding_amount");
  add_stat_meta(host_plane, 9, "batch_size_after_padding");

  XLine* request_line = host_plane->add_lines();
  request_line->set_id(1);
  request_line->set_name("RequestThread");
  XEvent* session_run = add_event(request_line, 1, 100, 20000);
  add_int_stat(session_run, 1, 1);
  add_str_stat(session_run, 2, "/models/ranking_model:1");
  XEvent* schedule = add_event(request_line, 2, 200, 200);
  add_int_stat(schedule, 3, kSharedBatchSchedulerContext);
  add_int_stat(schedule, 4, 1);
  add_int_stat(schedule, 5, 8);

  XLine* batch_line = host_plane->add_lines();
  batch_line->set_id(2);
  batch_line->set_name("BatchThread");
  XEvent* process_batch = add_event(batch_line, 3, 500, 15000);
  add_int_stat(process_batch, 6, kSharedBatchSchedulerContext);
  add_int_stat(process_batch, 7, 1);
  add_int_stat(process_batch, 1, 2);
  add_int_stat(process_batch, 3, kTpuLaunchContext);
  add_int_stat(process_batch, 4, 42);
  XEvent* concat = add_event(batch_line, 4, 600, 100);
  add_int_stat(concat, 8, 0);
  add_int_stat(concat, 9, 8);

  XPlane* device_plane = space.add_planes();
  device_plane->set_id(2);
  device_plane->set_name("/device:TPU:0");
  add_stat_meta(device_plane, 1, "peak_teraflops_per_second");
  add_stat_meta(device_plane, 2, "peak_hbm_bw_gigabytes_per_second");
  add_stat_meta(device_plane, 3, "device_type_string");
  add_stat_meta(device_plane, 4, "_ct");
  add_stat_meta(device_plane, 5, "_c");
  add_stat_meta(device_plane, 6, "program_id");
  add_stat_meta(device_plane, 7, "symbol_id");
  add_stat_meta(device_plane, 8, "hlo_category");
  add_stat_meta(device_plane, 9, "flops");
  add_stat_meta(device_plane, 10, "bytes_accessed");
  add_double_stat(device_plane, 1, 275.0);
  add_double_stat(device_plane, 2, 1200.0);
  add_str_stat(device_plane, 3, "TPU v4");

  add_event_meta(device_plane, 1,
                 "jit_serve(" + std::to_string(program_id) + ")");
  XEventMetadata* op_meta = add_event_meta(device_plane, 2, "fusion.1");
  add_int_stat(op_meta, 6, static_cast<int64_t>(program_id));
  add_int_stat(op_meta, 7, 1);
  add_str_stat(op_meta, 8, "fusion");
  add_int_stat(op_meta, 9, 1000000);
  add_int_stat(op_meta, 10, 100000);

  XLine* module_line = device_plane->add_lines();
  module_line->set_id(1);
  module_line->set_name("XLA Modules");
  XEvent* module_event = add_event(module_line, 1, 1000, 10000);
  add_int_stat(module_event, 4, kTpuLaunchContext);
  add_int_stat(module_event, 5, 42);

  XLine* ops_line = device_plane->add_lines();
  ops_line->set_id(2);
  ops_line->set_name("XLA Ops");
  add_event(ops_line, 2, 2000, 8000);
  return space;
}

TEST(UnifiedOpProfileProcessorTest, ProtoOutputWhenTqxIsOutPb) {
  std::string session_dir =
      tsl::io::JoinPath(testing::TempDir(), "unified_op_profile_pb_test");
  ASSERT_OK(tsl::Env::Default()->RecursivelyCreateDir(session_dir));
  std::string xspace_path =
      tsl::io::JoinPath(session_dir, "test_host.xplane.pb");
  XSpace dummy_space;
  ASSERT_OK(WriteBinaryProto(xspace_path, dummy_space));
  ASSERT_OK_AND_ASSIGN(
      auto session_snapshot,
      SessionSnapshot::Create({xspace_path}, /*xspaces=*/std::nullopt));

  OpStats op_stats = CreateTestOpStatsWithProgram(1234);

  ToolOptions pb_options;
  pb_options["tqx"] = std::string("out:pb");
  UnifiedOpProfileProcessor processor(pb_options);
  ASSERT_OK(
      processor.ProcessCombinedOpStats(session_snapshot, op_stats, pb_options));
  EXPECT_EQ(processor.GetContentType(), "application/x-protobuf");

  tensorflow::profiler::op_profile::Profile profile;
  ASSERT_TRUE(profile.ParseFromString(processor.GetData()));
  ASSERT_EQ(profile.by_program().children_size(), 1);
  EXPECT_EQ(profile.by_program().children(0).name(), "jit_step(1234)");
}

TEST(UnifiedOpProfileProcessorTest,
     ServingMetadataComputedEndToEndFromInferenceXSpace) {
  constexpr uint64_t kProgramId = 5621224425449793704ULL;
  std::string session_dir = tsl::io::JoinPath(
      testing::TempDir(), "unified_op_profile_e2e_inference_test");
  ASSERT_OK(tsl::Env::Default()->RecursivelyCreateDir(session_dir));
  std::string xspace_path =
      tsl::io::JoinPath(session_dir, "serving_host.xplane.pb");
  ASSERT_OK(
      WriteBinaryProto(xspace_path, CreateInferenceServingXSpace(kProgramId)));
  ASSERT_OK_AND_ASSIGN(
      auto session_snapshot,
      SessionSnapshot::Create({xspace_path}, /*xspaces=*/std::nullopt));

  // Default options omit programServingMetadata.
  ToolOptions default_options;
  UnifiedOpProfileProcessor default_processor(default_options);
  ASSERT_OK(
      default_processor.ProcessSession(session_snapshot, default_options));
  EXPECT_THAT(default_processor.GetData(),
              Not(HasSubstr("programServingMetadata")));

  // Opt-in populates programServingMetadata for both unified and legacy
  // processors.
  ToolOptions options;
  options["program_serving_metadata"] = true;

  UnifiedOpProfileProcessor unified_processor(options);
  ASSERT_OK(unified_processor.ProcessSession(session_snapshot, options));
  EXPECT_THAT(unified_processor.GetData(),
              HasSubstr("\"programServingMetadata\""));
  EXPECT_THAT(unified_processor.GetData(),
              HasSubstr("\"/models/ranking_model:1\""));
  EXPECT_THAT(unified_processor.GetData(), HasSubstr("\"batchSize\":8"));

  OpProfileProcessor legacy_processor(options);
  ASSERT_OK(legacy_processor.ProcessSession(session_snapshot, options));
  EXPECT_THAT(legacy_processor.GetData(),
              HasSubstr("\"programServingMetadata\""));
  EXPECT_THAT(legacy_processor.GetData(),
              HasSubstr("\"/models/ranking_model:1\""));
  EXPECT_THAT(legacy_processor.GetData(), HasSubstr("\"batchSize\":8"));

  // Also verify use_flat_metric=true with program_serving_metadata=true.
  OpStats flat_op_stats = CreateTestOpStatsWithProgram(kProgramId);
  FlatOpMetrics* flat_op =
      flat_op_stats.mutable_flat_device_op_metrics_db()->add_op_instances();
  flat_op->set_hlo_name("fusion.1");
  flat_op->set_op_id(1);
  flat_op->set_category("fusion");
  flat_op->set_time_ps(1000);
  flat_op->set_self_time_ps(1000);
  flat_op->set_occurrences(1);
  flat_op->set_hlo_module_id(kProgramId);
  flat_op_stats.mutable_flat_device_op_metrics_db()->set_total_time_ps(1000);
  flat_op_stats.mutable_flat_device_op_metrics_db()->set_total_op_time_ps(1000);
  ToolOptions flat_options = options;
  flat_options["use_flat_metric"] = true;
  UnifiedOpProfileProcessor flat_processor(flat_options);
  ASSERT_OK(flat_processor.ProcessCombinedOpStats(
      session_snapshot, flat_op_stats, flat_options));
  EXPECT_THAT(flat_processor.GetData(),
              HasSubstr("\"programServingMetadata\""));
  EXPECT_THAT(flat_processor.GetData(),
              HasSubstr("\"/models/ranking_model:1\""));

  // Training runs skip serving metadata even when opted in.
  OpStats training_op_stats = CreateTestOpStatsWithProgram(kProgramId);
  training_op_stats.mutable_run_environment()->set_is_training(true);
  UnifiedOpProfileProcessor training_unified_processor(options);
  ASSERT_OK(training_unified_processor.ProcessCombinedOpStats(
      session_snapshot, training_op_stats, options));
  EXPECT_THAT(training_unified_processor.GetData(),
              Not(HasSubstr("programServingMetadata")));

  // Unreadable XSpace file degrades gracefully without failing Op Profile.
  ASSERT_OK_AND_ASSIGN(
      auto missing_snapshot,
      SessionSnapshot::Create(
          {tsl::io::JoinPath(session_dir, "missing_host.xplane.pb")},
          /*xspaces=*/std::nullopt));
  UnifiedOpProfileProcessor fallback_processor(options);
  ASSERT_OK(fallback_processor.ProcessCombinedOpStats(
      missing_snapshot, CreateTestOpStatsWithProgram(kProgramId), options));
  EXPECT_THAT(fallback_processor.GetData(),
              Not(HasSubstr("programServingMetadata")));
  EXPECT_THAT(fallback_processor.GetData(),
              HasSubstr("jit_step(" + std::to_string(kProgramId) + ")"));
}

}  // namespace
}  // namespace xprof
