/* Copyright 2025 The OpenXLA Authors. All Rights Reserved.

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

#include "xprof/convert/xplane_to_hlo.h"

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "testing/base/public/gmock.h"
#include "<gtest/gtest.h>"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/service/hlo.pb.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/file_system.h"
#include "tsl/platform/path.h"
#include "xprof/convert/file_utils.h"
#include "xprof/convert/repository.h"
#include "xprof/convert/tool_options.h"

namespace tensorflow {
namespace profiler {
namespace {

using ::absl_testing::StatusIs;
using ::testing::Eq;
using ::testing::HasSubstr;

class XPlaneToHloTest : public testing::Test {
 protected:
  void SetUp() override {
    std::string temp_dir = testing::TempDir();
    profile_dir_ =
        tsl::io::JoinPath(temp_dir, "log/plugins/profile/hlo_proto_test_dir");
    CHECK_OK(tsl::Env::Default()->RecursivelyCreateDir(profile_dir_));
    xplane_path_ = tsl::io::JoinPath(profile_dir_, "hostname0.xplane.pb");
    std::unique_ptr<tsl::WritableFile> xplane_file;
    CHECK_OK(
        tsl::Env::Default()->NewAppendableFile(xplane_path_, &xplane_file));
  }

  void WriteDummyHloWithNode(absl::string_view module_name,
                             absl::string_view node_name) {
    xla::HloProto hlo;
    hlo.mutable_hlo_module()->set_name(std::string(module_name));
    xla::HloComputationProto* computation =
        hlo.mutable_hlo_module()->add_computations();
    computation->add_instructions()->set_name(std::string(node_name));
    std::string path = tsl::io::JoinPath(
        profile_dir_, absl::StrCat(module_name, ".hlo_proto.pb"));
    ASSERT_OK(xprof::WriteBinaryProto(path, hlo));
  }

  absl::StatusOr<SessionSnapshot> CreateSnapshot() {
    return SessionSnapshot::Create({xplane_path_}, /*xspaces=*/std::nullopt);
  }

  std::string profile_dir_;
  std::string xplane_path_;
};

TEST_F(XPlaneToHloTest, GetHloProtoByNodeNameSuccess) {
  // Arrange
  // Write an HloProto without an hlo_module.
  xla::HloProto empty_hlo;
  std::string empty_hlo_path =
      tsl::io::JoinPath(profile_dir_, "empty_module.hlo_proto.pb");
  ASSERT_OK(xprof::WriteBinaryProto(empty_hlo_path, empty_hlo));
  WriteDummyHloWithNode("other_module", "other_node");
  WriteDummyHloWithNode("my_module", "my_target_node");
  ASSERT_OK_AND_ASSIGN(SessionSnapshot session_snapshot, CreateSnapshot());

  // Act
  absl::StatusOr<xla::HloProto> hlo_proto_or =
      GetHloProtoByNodeName(session_snapshot, "my_target_node");

  // Assert
  ASSERT_OK(hlo_proto_or.status());
  EXPECT_THAT(hlo_proto_or->hlo_module().name(), Eq("my_module"));
}

TEST_F(XPlaneToHloTest, GetHloProtoByNodeNameNotFound) {
  // Arrange
  // Write a bad HloProto file.
  std::string bad_hlo_path =
      tsl::io::JoinPath(profile_dir_, "bad_module.hlo_proto.pb");
  ASSERT_OK(tsl::WriteStringToFile(tsl::Env::Default(), bad_hlo_path,
                                   "this is not a valid proto"));
  WriteDummyHloWithNode("my_module", "other_node");
  ASSERT_OK_AND_ASSIGN(SessionSnapshot session_snapshot, CreateSnapshot());

  // Act
  absl::StatusOr<xla::HloProto> hlo_proto_or =
      GetHloProtoByNodeName(session_snapshot, "my_target_node");

  // Assert
  EXPECT_THAT(hlo_proto_or,
              StatusIs(absl::StatusCode::kNotFound, HasSubstr("not found")));
}

TEST_F(XPlaneToHloTest, GetHloProtoByNodeNameIgnoresOtherFiles) {
  // Arrange
  // Write a dummy file that shouldn't be read.
  std::string other_file_path =
      tsl::io::JoinPath(profile_dir_, "some_other_file.txt");
  ASSERT_OK(tsl::WriteStringToFile(tsl::Env::Default(), other_file_path,
                                   "not a proto"));

  // Also include a file ending in .hlo_proto.pb to cover ConsumeSuffix.
  std::string dummy_file_path =
      tsl::io::JoinPath(profile_dir_, ".hlo_proto.pb");
  ASSERT_OK(tsl::WriteStringToFile(tsl::Env::Default(), dummy_file_path,
                                   "not a proto"));
  ASSERT_OK_AND_ASSIGN(SessionSnapshot session_snapshot, CreateSnapshot());

  // Act
  absl::StatusOr<xla::HloProto> hlo_proto_or =
      GetHloProtoByNodeName(session_snapshot, "my_target_node");

  // Assert
  EXPECT_FALSE(hlo_proto_or.ok());
}

TEST_F(XPlaneToHloTest, GetHloProtoByNodeNameInvalidDir) {
  // Arrange
  std::string temp_dir = testing::TempDir();
  std::string invalid_profile_dir =
      tsl::io::JoinPath(temp_dir, "non_existent_dir");
  std::string invalid_xplane_path =
      tsl::io::JoinPath(invalid_profile_dir, "hostname0.xplane.pb");

  absl::StatusOr<SessionSnapshot> session_snapshot_or =
      SessionSnapshot::Create({invalid_xplane_path}, /*xspaces=*/std::nullopt);
  ASSERT_OK(session_snapshot_or.status());
  const SessionSnapshot& session_snapshot = *session_snapshot_or;

  // Act
  absl::StatusOr<xla::HloProto> hlo_proto_or =
      GetHloProtoByNodeName(session_snapshot, "my_target_node");

  // Assert
  EXPECT_THAT(hlo_proto_or, StatusIs(absl::StatusCode::kNotFound));
}

TEST_F(XPlaneToHloTest, GetHloProtoByProgramIdSuccess) {
  // Arrange
  // Write a dummy HloProto that matches.
  xla::HloProto dummy_hlo;
  dummy_hlo.mutable_hlo_module()->set_name("my_module_prog_123");

  std::string hlo_path =
      tsl::io::JoinPath(profile_dir_, "my_module_prog_123.hlo_proto.pb");
  ASSERT_OK(xprof::WriteBinaryProto(hlo_path, dummy_hlo));
  ASSERT_OK_AND_ASSIGN(SessionSnapshot session_snapshot, CreateSnapshot());

  // Act
  absl::StatusOr<xla::HloProto> hlo_proto_or =
      GetHloProtoByProgramId(session_snapshot, "prog_123");

  // Assert
  ASSERT_OK(hlo_proto_or.status());
  EXPECT_THAT(hlo_proto_or->hlo_module().name(), Eq("my_module_prog_123"));
}

TEST_F(XPlaneToHloTest, GetHloProtoByProgramIdNotFound) {
  // Arrange
  ASSERT_OK_AND_ASSIGN(SessionSnapshot session_snapshot, CreateSnapshot());

  // Act
  absl::StatusOr<xla::HloProto> hlo_proto_or =
      GetHloProtoByProgramId(session_snapshot, "prog_456");

  // Assert
  EXPECT_THAT(hlo_proto_or,
              StatusIs(absl::StatusCode::kNotFound, HasSubstr("not found")));
}

TEST_F(XPlaneToHloTest, GetHloProtoByOptionsSuccessModule) {
  // Arrange
  xla::HloProto dummy_hlo;
  dummy_hlo.mutable_hlo_module()->set_name("my_module");

  std::string hlo_path =
      tsl::io::JoinPath(profile_dir_, "my_module.hlo_proto.pb");
  ASSERT_OK(xprof::WriteBinaryProto(hlo_path, dummy_hlo));
  ASSERT_OK_AND_ASSIGN(SessionSnapshot session_snapshot, CreateSnapshot());
  ToolOptions options;
  options.insert(
      {std::string(tensorflow::profiler::kModuleNameOption), "my_module"});

  // Act
  absl::StatusOr<xla::HloProto> hlo_proto_or =
      GetHloProtoByOptions(session_snapshot, options);

  // Assert
  ASSERT_OK(hlo_proto_or.status());
  EXPECT_THAT(hlo_proto_or->hlo_module().name(), Eq("my_module"));
}

TEST_F(XPlaneToHloTest, GetHloProtoByOptionsSuccessProgramId) {
  // Arrange
  xla::HloProto dummy_hlo;
  dummy_hlo.mutable_hlo_module()->set_name("my_module_prog_123");

  std::string hlo_path =
      tsl::io::JoinPath(profile_dir_, "my_module_prog_123.hlo_proto.pb");
  ASSERT_OK(xprof::WriteBinaryProto(hlo_path, dummy_hlo));
  ASSERT_OK_AND_ASSIGN(SessionSnapshot session_snapshot, CreateSnapshot());
  ToolOptions options;
  options["program_id"] = "prog_123";

  // Act
  absl::StatusOr<xla::HloProto> hlo_proto_or =
      GetHloProtoByOptions(session_snapshot, options);

  // Assert
  ASSERT_OK(hlo_proto_or.status());
  EXPECT_THAT(hlo_proto_or->hlo_module().name(), Eq("my_module_prog_123"));
}

TEST_F(XPlaneToHloTest, GetHloProtoByOptionsFailure) {
  // Arrange
  ASSERT_OK_AND_ASSIGN(SessionSnapshot session_snapshot, CreateSnapshot());
  ToolOptions options;

  // Act
  absl::StatusOr<xla::HloProto> hlo_proto_or =
      GetHloProtoByOptions(session_snapshot, options);

  // Assert
  EXPECT_THAT(hlo_proto_or, StatusIs(absl::StatusCode::kInvalidArgument,
                                     HasSubstr("Can not load")));
}

}  // namespace
}  // namespace profiler
}  // namespace tensorflow
