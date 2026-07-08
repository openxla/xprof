// Copyright 2026 The OpenXLA Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ==============================================================================

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "file/base/filesystem.h"
#include "file/base/options.h"
#include "file/base/path.h"
#include "testing/base/public/gmock.h"
#include "<gtest/gtest.h>"
#include "absl/status/status.h"
#include "third_party/jsoncpp/include/json/reader.h"
#include "third_party/jsoncpp/include/json/value.h"
#include "tsl/profiler/protobuf/xplane.pb.h"
#include "xprof/convert/base_op_stats_processor.h"
#include "xprof/convert/file_utils.h"
#include "xprof/convert/repository.h"
#include "xprof/convert/tool_options.h"
#include "xprof/convert/unified_profile_processor_factory.h"
#include "xprof/convert/unified_tools_registration.h"
#include "plugin/xprof/protobuf/op_stats.pb.h"

namespace xprof {
namespace {

using ::tensorflow::profiler::SessionSnapshot;
using ::tensorflow::profiler::ToolOptions;
using ::tensorflow::profiler::XSpace;

class UnifiedHloStatsProcessorTest : public ::testing::Test {
 protected:
  void SetUp() override {
    session_dir_ =
        file::JoinPath(testing::TempDir(), "unified_hlo_stats_processor_test");
    file::RecursivelyDelete(session_dir_, file::Defaults()).IgnoreError();
    ASSERT_OK(file::CreateDir(session_dir_, file::Defaults()));
    RegisterUnifiedToolRegistrations();
  }

  void TearDown() override {
    file::RecursivelyDelete(session_dir_, file::Defaults()).IgnoreError();
  }

  std::string session_dir_;
  ToolOptions options_;
};

TEST_F(UnifiedHloStatsProcessorTest, MinimalTest) {
  auto processor = UnifiedProfileProcessorFactory::GetInstance().Create(
      "hlo_stats", options_);
  ASSERT_NE(processor, nullptr);

  std::string xspace_path = file::JoinPath(session_dir_, "test_host.xplane.pb");
  XSpace dummy_space;
  ASSERT_OK(WriteBinaryProto(xspace_path, dummy_space));

  ASSERT_OK_AND_ASSIGN(SessionSnapshot session_snapshot,
                       SessionSnapshot::Create({xspace_path}, std::nullopt));

  EXPECT_EQ(processor->GetData(), "");  // Initially empty
}

TEST_F(UnifiedHloStatsProcessorTest, ProcessCombinedOpStatsTest) {
  auto processor = UnifiedProfileProcessorFactory::GetInstance().Create(
      "hlo_stats", options_);
  ASSERT_NE(processor, nullptr);

  auto* hlo_processor = dynamic_cast<BaseOpStatsProcessor*>(processor.get());
  ASSERT_NE(hlo_processor, nullptr);

  std::string xspace_path = file::JoinPath(session_dir_, "test_host.xplane.pb");
  XSpace dummy_space;
  ASSERT_OK(WriteBinaryProto(xspace_path, dummy_space));

  ASSERT_OK_AND_ASSIGN(SessionSnapshot session_snapshot,
                       SessionSnapshot::Create({xspace_path}, std::nullopt));

  tensorflow::profiler::OpStats combined_op_stats;
  combined_op_stats.mutable_run_environment()->set_device_type("TPU");

  auto& db = *(combined_op_stats.mutable_device_op_metrics_db());
  auto& tc_metric = *db.add_metrics_db();
  tc_metric.set_name("tc-op");
  tc_metric.set_core_type(tensorflow::profiler::OpMetrics::TENSOR_CORE);
  tc_metric.set_occurrences(1);
  tc_metric.set_self_time_ps(100000);

  ASSERT_OK(hlo_processor->ProcessCombinedOpStats(session_snapshot,
                                                  combined_op_stats, options_));

  std::string output_str = processor->GetData();
  EXPECT_FALSE(output_str.empty());

  Json::Value json;
  Json::Reader reader;
  ASSERT_TRUE(reader.parse(output_str, json));
  ASSERT_EQ(json["rows"].size(), 1);
  EXPECT_EQ(json["rows"][0]["c"][4]["v"].asString(), "tc-op");
  EXPECT_EQ(json["rows"][0]["c"][26]["v"].asString(), "TensorCore");
}

}  // namespace
}  // namespace xprof
