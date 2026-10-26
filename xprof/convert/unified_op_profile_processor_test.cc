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
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "xla/tsl/platform/env.h"
#include "tsl/platform/path.h"
#include "tsl/profiler/protobuf/xplane.pb.h"
#include "xprof/convert/file_utils.h"
#include "xprof/convert/op_profile_builder.h"
#include "xprof/convert/repository.h"
#include "xprof/convert/tool_options.h"
#include "xprof/convert/unified_profile_processor_factory.h"
#include "xprof/convert/unified_session_snapshot.h"
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
using ::tensorflow::profiler::ProgramServingMetadataMap;
using ::tensorflow::profiler::SessionSnapshot;
using ::tensorflow::profiler::ToolOptions;
using ::tensorflow::profiler::XSpace;
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

class FakeUnifiedOpProfileProcessor : public UnifiedOpProfileProcessor {
 public:
  explicit FakeUnifiedOpProfileProcessor(const ToolOptions& options,
                                         bool fail_hook = false)
      : UnifiedOpProfileProcessor(options), fail_hook_(fail_hook) {}

  int hook_calls() const { return hook_calls_; }

 protected:
  absl::StatusOr<ProgramServingMetadataMap> GetProgramServingMetadata(
      const XprofSessionSnapshot& session_snapshot,
      const OpStats& combined_op_stats) override {
    ++hook_calls_;
    if (fail_hook_) {
      return absl::InternalError("Simulated metadata failure");
    }
    ProgramServingMetadataMap metadata;
    metadata[1234].set_model_id("/models/unified_model:1");
    metadata[1234].set_batch_size(16);
    metadata[1234].set_num_batches(9);
    return metadata;
  }

 private:
  bool fail_hook_ = false;
  int hook_calls_ = 0;
};

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

TEST(UnifiedOpProfileProcessorTest,
     ServingMetadataHookOnlyCalledWhenOptInParamIsTrue) {
  std::string session_dir = tsl::io::JoinPath(
      testing::TempDir(), "unified_op_profile_serving_metadata_test");
  ASSERT_OK(tsl::Env::Default()->RecursivelyCreateDir(session_dir));
  std::string xspace_path =
      tsl::io::JoinPath(session_dir, "test_host.xplane.pb");
  XSpace dummy_space;
  ASSERT_OK(WriteBinaryProto(xspace_path, dummy_space));
  ASSERT_OK_AND_ASSIGN(
      auto session_snapshot,
      SessionSnapshot::Create({xspace_path}, /*xspaces=*/std::nullopt));

  OpStats op_stats = CreateTestOpStatsWithProgram(1234);

  // Without program_serving_metadata=true, hook is not called and JSON has no
  // programServingMetadata.
  ToolOptions default_options;
  FakeUnifiedOpProfileProcessor processor_default(default_options);
  ASSERT_OK(processor_default.ProcessCombinedOpStats(session_snapshot, op_stats,
                                                     default_options));
  EXPECT_EQ(processor_default.hook_calls(), 0);
  EXPECT_THAT(processor_default.GetData(),
              Not(HasSubstr("programServingMetadata")));

  // With program_serving_metadata=true, hook is called and JSON includes
  // programServingMetadata.
  ToolOptions opt_in_options;
  opt_in_options["program_serving_metadata"] = true;
  FakeUnifiedOpProfileProcessor processor_opt_in(opt_in_options);
  ASSERT_OK(processor_opt_in.ProcessCombinedOpStats(session_snapshot, op_stats,
                                                    opt_in_options));
  EXPECT_EQ(processor_opt_in.hook_calls(), 1);
  EXPECT_THAT(processor_opt_in.GetData(),
              HasSubstr("\"programServingMetadata\""));
  EXPECT_THAT(processor_opt_in.GetData(),
              HasSubstr("\"/models/unified_model:1\""));

  // Also verify use_flat_metric=true with program_serving_metadata=true.
  FlatOpMetrics* flat_op =
      op_stats.mutable_flat_device_op_metrics_db()->add_op_instances();
  flat_op->set_hlo_name("fusion.1");
  flat_op->set_op_id(1);
  flat_op->set_category("fusion");
  flat_op->set_time_ps(1000);
  flat_op->set_self_time_ps(1000);
  flat_op->set_occurrences(1);
  flat_op->set_hlo_module_id(1234);
  op_stats.mutable_flat_device_op_metrics_db()->set_total_time_ps(1000);
  op_stats.mutable_flat_device_op_metrics_db()->set_total_op_time_ps(1000);
  ToolOptions flat_opt_in_options;
  flat_opt_in_options["use_flat_metric"] = true;
  flat_opt_in_options["program_serving_metadata"] = true;
  FakeUnifiedOpProfileProcessor processor_flat_opt_in(flat_opt_in_options);
  ASSERT_OK(processor_flat_opt_in.ProcessCombinedOpStats(
      session_snapshot, op_stats, flat_opt_in_options));
  EXPECT_EQ(processor_flat_opt_in.hook_calls(), 1);
  EXPECT_THAT(processor_flat_opt_in.GetData(),
              HasSubstr("\"programServingMetadata\""));
  EXPECT_THAT(processor_flat_opt_in.GetData(),
              HasSubstr("\"/models/unified_model:1\""));

  // The base hook returns an empty map, so opting in on the base processor
  // produces no programServingMetadata.
  UnifiedOpProfileProcessor base_processor(opt_in_options);
  ASSERT_OK(base_processor.ProcessCombinedOpStats(session_snapshot, op_stats,
                                                  opt_in_options));
  EXPECT_THAT(base_processor.GetData(),
              Not(HasSubstr("programServingMetadata")));
  EXPECT_THAT(base_processor.GetData(), HasSubstr("jit_step(1234)"));

  // When the hook returns an error, ProcessCombinedOpStats degrades gracefully
  // and still succeeds.
  FakeUnifiedOpProfileProcessor processor_failing(opt_in_options,
                                                  /*fail_hook=*/true);
  ASSERT_OK(processor_failing.ProcessCombinedOpStats(session_snapshot, op_stats,
                                                     opt_in_options));
  EXPECT_EQ(processor_failing.hook_calls(), 1);
  EXPECT_THAT(processor_failing.GetData(),
              Not(HasSubstr("programServingMetadata")));
  EXPECT_THAT(processor_failing.GetData(), HasSubstr("jit_step(1234)"));
}

}  // namespace
}  // namespace xprof
