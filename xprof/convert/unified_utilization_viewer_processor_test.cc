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

#include <cstddef>
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
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "third_party/jsoncpp/include/json/reader.h"
#include "third_party/jsoncpp/include/json/value.h"
#include "google/protobuf/arena.h"
#include "tsl/profiler/protobuf/xplane.pb.h"
#include "xprof/convert/file_utils.h"
#include "xprof/convert/repository.h"
#include "xprof/convert/tool_options.h"
#include "xprof/convert/unified_profile_processor.h"
#include "xprof/convert/unified_profile_processor_factory.h"
#include "xprof/convert/unified_session_snapshot.h"
#include "xprof/convert/unified_tools_registration.h"

namespace xprof {
namespace {

using ::tensorflow::profiler::SessionSnapshot;
using ::tensorflow::profiler::ToolOptions;
using ::tensorflow::profiler::XSpace;

class MockXprofSessionSnapshot : public XprofSessionSnapshot {
 public:
  MOCK_METHOD(size_t, XSpaceSize, (), (const, override));
  MOCK_METHOD((absl::StatusOr<tensorflow::profiler::XSpace*>), GetXSpace,
              (size_t index, google::protobuf::Arena* arena), (const, override));
  MOCK_METHOD(std::string, GetHostname, (size_t index), (const, override));
  MOCK_METHOD(absl::string_view, GetSessionRunDir, (), (const, override));
  MOCK_METHOD(absl::StatusOr<std::string>, GetHostDataFileName,
              (tensorflow::profiler::StoredDataType data_type,
               absl::string_view host),
              (const, override));
};

class UnifiedUtilizationViewerProcessorTest : public ::testing::Test {
 protected:
  void SetUp() override {
    session_dir_ =
        file::JoinPath(testing::TempDir(), "unified_utilization_viewer_test");
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

TEST_F(UnifiedUtilizationViewerProcessorTest, EmptyXSpaceTest) {
  auto processor = UnifiedProfileProcessorFactory::GetInstance().Create(
      "utilization_viewer", options_);
  ASSERT_NE(processor, nullptr);

  std::string xspace_path = file::JoinPath(session_dir_, "test_host.xplane.pb");
  XSpace dummy_space;
  ASSERT_OK(WriteBinaryProto(xspace_path, dummy_space));

  ASSERT_OK_AND_ASSIGN(SessionSnapshot session_snapshot,
                       SessionSnapshot::Create({xspace_path}, std::nullopt));

  ASSERT_OK(processor->ProcessSession(session_snapshot, options_));

  std::string output_str = processor->GetData();
  EXPECT_FALSE(output_str.empty());
  EXPECT_EQ(processor->GetContentType(), "application/json");

  Json::Value json;
  Json::Reader reader;
  ASSERT_TRUE(reader.parse(output_str, json));
  ASSERT_TRUE(json.isMember("rows"));
  EXPECT_EQ(json["rows"].size(), 0);
}

TEST_F(UnifiedUtilizationViewerProcessorTest, NoXSpaceTest) {
  auto processor = UnifiedProfileProcessorFactory::GetInstance().Create(
      "utilization_viewer", options_);
  ASSERT_NE(processor, nullptr);

  testing::NiceMock<MockXprofSessionSnapshot> session_snapshot;
  EXPECT_CALL(session_snapshot, XSpaceSize())
      .WillRepeatedly(testing::Return(0));

  absl::Status status = processor->ProcessSession(session_snapshot, options_);
  EXPECT_EQ(status.code(), absl::StatusCode::kNotFound);
  EXPECT_THAT(status.message(),
              ::testing::HasSubstr("No XSpace found in the session."));
}

}  // namespace
}  // namespace xprof
