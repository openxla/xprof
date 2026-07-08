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
#include "net/proto2/contrib/parse_proto/parse_text_proto.h"
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

using ::google::protobuf::contrib::parse_proto::ParseTextProtoOrDie;
using ::tensorflow::profiler::SessionSnapshot;
using ::tensorflow::profiler::ToolOptions;
using ::tensorflow::profiler::XSpace;
using ::testing::IsEmpty;
using ::testing::Not;
using ::testing::SizeIs;

class UnifiedOverviewPageProcessorTest : public ::testing::Test {
 protected:
  void SetUp() override {
    session_dir_ = file::JoinPath(testing::TempDir(),
                                  "unified_overview_page_processor_test");
    file::RecursivelyDelete(session_dir_, file::Defaults()).IgnoreError();
    ASSERT_OK(file::CreateDir(session_dir_, file::Defaults()));
    RegisterUnifiedToolRegistrations();
  }

  void TearDown() override {
    file::RecursivelyDelete(session_dir_, file::Defaults()).IgnoreError();
  }

  std::string session_dir_;
};

TEST_F(UnifiedOverviewPageProcessorTest, MinimalTest) {
  ToolOptions options;
  auto processor = UnifiedProfileProcessorFactory::GetInstance().Create(
      "overview_page", options);
  ASSERT_NE(processor, nullptr);

  std::string xspace_path = file::JoinPath(session_dir_, "test_host.xplane.pb");
  XSpace dummy_space;
  ASSERT_OK(WriteBinaryProto(xspace_path, dummy_space));

  ASSERT_OK_AND_ASSIGN(auto session_snapshot,
                       SessionSnapshot::Create({xspace_path}, std::nullopt));

  EXPECT_EQ(processor->GetData(), "");  // Initially empty
}

TEST_F(UnifiedOverviewPageProcessorTest, ProcessCombinedOpStatsTrainingTest) {
  ToolOptions options;
  auto processor = UnifiedProfileProcessorFactory::GetInstance().Create(
      "overview_page", options);
  ASSERT_NE(processor, nullptr);

  auto* overview_processor =
      dynamic_cast<BaseOpStatsProcessor*>(processor.get());
  ASSERT_NE(overview_processor, nullptr);

  std::string xspace_path = file::JoinPath(session_dir_, "test_host.xplane.pb");
  XSpace dummy_space;
  ASSERT_OK(WriteBinaryProto(xspace_path, dummy_space));

  ASSERT_OK_AND_ASSIGN(auto session_snapshot,
                       SessionSnapshot::Create({xspace_path}, std::nullopt));

  tensorflow::profiler::OpStats combined_op_stats = ParseTextProtoOrDie(R"pb(
    run_environment {
      device_type: "TPU"
      is_training: true
    }
  )pb");

  ASSERT_OK(overview_processor->ProcessCombinedOpStats(
      session_snapshot, combined_op_stats, options));

  std::string output_str = processor->GetData();
  EXPECT_FALSE(output_str.empty());

  Json::Value json;
  Json::Reader reader;
  ASSERT_TRUE(reader.parse(output_str, json));
  EXPECT_THAT(json, SizeIs(7));
  // Index 2 is run_environment_data_table. Verify custom property is_training.
  EXPECT_EQ(json[2]["p"]["is_training"].asString(), "true");
}

TEST_F(UnifiedOverviewPageProcessorTest, ProcessCombinedOpStatsInferenceTest) {
  ToolOptions options;
  auto processor = UnifiedProfileProcessorFactory::GetInstance().Create(
      "overview_page", options);
  ASSERT_NE(processor, nullptr);

  auto* overview_processor =
      dynamic_cast<BaseOpStatsProcessor*>(processor.get());
  ASSERT_NE(overview_processor, nullptr);

  std::string xspace_path = file::JoinPath(session_dir_, "test_host.xplane.pb");
  XSpace dummy_space;
  ASSERT_OK(WriteBinaryProto(xspace_path, dummy_space));

  ASSERT_OK_AND_ASSIGN(auto session_snapshot,
                       SessionSnapshot::Create({xspace_path}, std::nullopt));

  tensorflow::profiler::OpStats combined_op_stats = ParseTextProtoOrDie(R"pb(
    run_environment {
      device_type: "TPU"
      is_training: false
    }
  )pb");

  ASSERT_OK(overview_processor->ProcessCombinedOpStats(
      session_snapshot, combined_op_stats, options));

  std::string output_str = processor->GetData();
  EXPECT_THAT(output_str, Not(IsEmpty()));

  Json::Value json;
  Json::Reader reader;
  ASSERT_TRUE(reader.parse(output_str, json));
  EXPECT_THAT(json, SizeIs(7));
  // Index 2 is run_environment_data_table. Verify custom property is_training.
  EXPECT_EQ(json[2]["p"]["is_training"].asString(), "false");
}

}  // namespace
}  // namespace xprof
