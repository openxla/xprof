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

#include "xprof/convert/unified_roofline_model_processor.h"

#include <memory>
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
#include "xprof/convert/repository.h"
#include "xprof/convert/tool_options.h"
#include "xprof/convert/unified_profile_processor.h"
#include "xprof/convert/unified_profile_processor_factory.h"
#include "xprof/convert/unified_tools_registration.h"
#include "plugin/xprof/protobuf/hardware_types.pb.h"
#include "plugin/xprof/protobuf/op_metrics.pb.h"
#include "plugin/xprof/protobuf/op_stats.pb.h"

namespace xprof {
namespace {

using ::tensorflow::profiler::OpMetrics;
using ::tensorflow::profiler::OpMetricsDb;
using ::tensorflow::profiler::OpStats;
using ::tensorflow::profiler::SessionSnapshot;
using ::tensorflow::profiler::ToolOptions;
using ::tensorflow::profiler::XSpace;
using ::testing::HasSubstr;
using ::testing::IsEmpty;
using ::testing::Not;

TEST(UnifiedRooflineModelProcessorTest, MinimalTest) {
  RegisterUnifiedToolRegistrations();
  ToolOptions options;
  std::unique_ptr<UnifiedProfileProcessor> processor =
      UnifiedProfileProcessorFactory::GetInstance().Create("roofline_model",
                                                           options);
  ASSERT_NE(processor, nullptr);

  std::string session_dir = tsl::io::JoinPath(
      testing::TempDir(), "unified_roofline_model_processor_test");
  ASSERT_OK(tsl::Env::Default()->RecursivelyCreateDir(session_dir));
  std::string xspace_path =
      tsl::io::JoinPath(session_dir, "test_host.xplane.pb");
  XSpace dummy_space;
  ASSERT_OK(WriteBinaryProto(xspace_path, dummy_space));

  std::vector<std::string> xspace_paths = {xspace_path};
  ASSERT_OK_AND_ASSIGN(
      SessionSnapshot session_snapshot,
      SessionSnapshot::Create(xspace_paths, /*xspaces=*/std::nullopt));

  EXPECT_OK(processor->ProcessSession(session_snapshot, options));
  EXPECT_EQ(processor->GetContentType(), "application/json");
  EXPECT_THAT(processor->GetData(), Not(IsEmpty()));
}

TEST(UnifiedRooflineModelProcessorTest, FlatOpMetricsDbTest) {
  RegisterUnifiedToolRegistrations();
  ToolOptions options;
  options["use_flat_metric"] = true;
  options["apply_time_scale_multiplier"] = true;

  std::unique_ptr<UnifiedProfileProcessor> processor =
      UnifiedProfileProcessorFactory::GetInstance().Create("roofline_model",
                                                           options);
  ASSERT_NE(processor, nullptr);

  std::string session_dir = tsl::io::JoinPath(
      testing::TempDir(), "unified_roofline_model_processor_flat_test");
  ASSERT_OK(tsl::Env::Default()->RecursivelyCreateDir(session_dir));
  std::string xspace_path =
      tsl::io::JoinPath(session_dir, "test_host.xplane.pb");
  XSpace dummy_space;
  ASSERT_OK(WriteBinaryProto(xspace_path, dummy_space));

  std::vector<std::string> xspace_paths = {xspace_path};
  ASSERT_OK_AND_ASSIGN(
      SessionSnapshot session_snapshot,
      SessionSnapshot::Create(xspace_paths, /*xspaces=*/std::nullopt));

  EXPECT_OK(processor->ProcessSession(session_snapshot, options));
  EXPECT_EQ(processor->GetContentType(), "application/json");
  EXPECT_THAT(processor->GetData(), Not(IsEmpty()));
}

// The OpStats combiner always creates the flat op metrics submessage, even when
// flat metrics were not requested and only the legacy `device_op_metrics_db`
// is populated. The processor must not treat the mere presence of an empty
// flat DB as a signal to read from it, or every per-op record is dropped.
TEST(UnifiedRooflineModelProcessorTest,
     UsesLegacyOpMetricsDbWhenFlatDbIsPresentButEmpty) {
  ToolOptions options;  // `use_flat_metric` is not requested.
  UnifiedRooflineModelProcessor processor(options);

  std::string session_dir = tsl::io::JoinPath(
      testing::TempDir(), "unified_roofline_model_processor_legacy_db_test");
  ASSERT_OK(tsl::Env::Default()->RecursivelyCreateDir(session_dir));
  std::string xspace_path =
      tsl::io::JoinPath(session_dir, "test_host.xplane.pb");
  XSpace dummy_space;
  ASSERT_OK(WriteBinaryProto(xspace_path, dummy_space));
  std::vector<std::string> xspace_paths = {xspace_path};
  ASSERT_OK_AND_ASSIGN(
      SessionSnapshot session_snapshot,
      SessionSnapshot::Create(xspace_paths, /*xspaces=*/std::nullopt));

  OpStats op_stats;
  op_stats.mutable_run_environment()->set_hardware_type(
      tensorflow::profiler::TPU);
  OpMetricsDb* device_op_metrics_db = op_stats.mutable_device_op_metrics_db();
  device_op_metrics_db->set_total_time_ps(1000);
  device_op_metrics_db->set_total_op_time_ps(1000);
  OpMetrics* op_metrics = device_op_metrics_db->add_metrics_db();
  op_metrics->set_name("fusion.123");
  op_metrics->set_category("convolution");
  op_metrics->set_occurrences(1);
  op_metrics->set_time_ps(1000);
  op_metrics->set_self_time_ps(1000);
  op_metrics->set_flops(100);
  op_metrics->set_bytes_accessed(10);
  // Present but empty, mirroring the OpStats combiner output.
  op_stats.mutable_flat_device_op_metrics_db();
  ASSERT_TRUE(op_stats.has_flat_device_op_metrics_db());

  ASSERT_OK(
      processor.ProcessCombinedOpStats(session_snapshot, op_stats, options));
  EXPECT_THAT(processor.GetData(), HasSubstr("fusion.123"));
}

}  // namespace
}  // namespace xprof
