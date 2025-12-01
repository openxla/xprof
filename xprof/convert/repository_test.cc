/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "xprof/convert/repository.h"

#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "testing/base/public/gmock.h"
#include "<gtest/gtest.h>"
#include "absl/status/status.h"
#include "google/protobuf/arena.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/file_system.h"
#include "xla/tsl/platform/status.h"
#include "tsl/platform/path.h"
#include "tsl/profiler/protobuf/xplane.pb.h"
#include "plugin/xprof/protobuf/op_stats.pb.h"

namespace tensorflow {
namespace profiler {
namespace {

using ::testing::Eq;

TEST(Repository, GetHostName) {
  auto session_snapshot_or =
      SessionSnapshot::Create({"log/plugins/profile/hostname0.xplane.pb",
                               "log/plugins/profile/hostname1.xplane.pb"},
                              /*xspaces=*/std::nullopt);
  TF_CHECK_OK(session_snapshot_or.status());
  EXPECT_THAT(session_snapshot_or.value().GetHostname(0), Eq("hostname0"));
  EXPECT_THAT(session_snapshot_or.value().GetHostname(1), Eq("hostname1"));
  EXPECT_TRUE(session_snapshot_or.value().HasAccessibleRunDir());
}

TEST(Repository, GetHostNameWithPeriods) {
  auto session_snapshot_or =
      SessionSnapshot::Create({"log/plugins/profile/127.0.0.1_6009.xplane.pb"},
                              /*xspaces=*/std::nullopt);
  TF_CHECK_OK(session_snapshot_or.status());
  EXPECT_THAT(session_snapshot_or.value().GetHostname(0), Eq("127.0.0.1_6009"));
  EXPECT_TRUE(session_snapshot_or.value().HasAccessibleRunDir());
}

TEST(Repository, GetSpaceByHostName) {
  std::vector<std::unique_ptr<XSpace>> xspaces;
  // prepare host 1.
  auto space1 = std::make_unique<XSpace>();
  *(space1->add_hostnames()) = "hostname1";
  // with index 0 which shouldn't impact the space finding by name.
  xspaces.push_back(std::move(space1));

  // prepare host 0.
  auto space0 = std::make_unique<XSpace>();
  *(space0->add_hostnames()) = "hostname0";
  // with index 1 which shouldn't impact the space finding by name.
  xspaces.push_back(std::move(space0));

  auto session_snapshot_or =
      SessionSnapshot::Create({"log/plugins/profile/hostname1.xplane.pb",
                               "log/plugins/profile/hostname0.xplane.pb"},
                              std::move(xspaces));
  TF_CHECK_OK(session_snapshot_or.status());
  google::protobuf::Arena arena;
  auto xspace0_or =
      session_snapshot_or.value().GetXSpaceByName("hostname0", &arena);
  TF_CHECK_OK(xspace0_or.status());
  google::protobuf::Arena arena2;
  auto xspace1_or =
      session_snapshot_or.value().GetXSpaceByName("hostname1", &arena2);
  EXPECT_FALSE(session_snapshot_or.value().HasAccessibleRunDir());
  TF_CHECK_OK(xspace1_or.status());
  EXPECT_THAT(xspace0_or.value()->hostnames(0), Eq("hostname0"));
  EXPECT_THAT(xspace1_or.value()->hostnames(0), Eq("hostname1"));
}

TEST(Repository, GetSSTableFile) {
  auto session_snapshot_or =
      SessionSnapshot::Create({"log/plugins/profile/hostname0.xplane.pb"},
                              /*xspaces=*/std::nullopt);
  TF_CHECK_OK(session_snapshot_or.status());
  auto sstable_path =
      session_snapshot_or.value().GetFilePath("trace_viewer@", "hostname0");
  auto not_found_path =
      session_snapshot_or.value().GetFilePath("memory_viewer", "hostname0");
  EXPECT_THAT(sstable_path, Eq("log/plugins/profile/hostname0.SSTABLE"));
  EXPECT_THAT(not_found_path, Eq(std::nullopt));
}

TEST(Repository, GetSSTableFileWithXSpace) {
  std::vector<std::unique_ptr<XSpace>> xspaces;
  // prepare host 0.
  auto space0 = std::make_unique<XSpace>();
  *(space0->add_hostnames()) = "hostname0";
  // with index 1 which shouldn't impact the space finding by name.
  xspaces.push_back(std::move(space0));
  auto session_snapshot_or = SessionSnapshot::Create(
      {"log/plugins/profile/hostname0.xplane.pb"}, std::move(xspaces));
  TF_CHECK_OK(session_snapshot_or.status());
  auto file_path_init_by_xspace =
      session_snapshot_or.value().GetFilePath("trace_viewer@", "hostname0");
  // The file path should be disabled in this mode.
  EXPECT_THAT(file_path_init_by_xspace, Eq(std::nullopt));
}

TEST(Repository, MismatchedXSpaceAndPath) {
  std::vector<std::unique_ptr<XSpace>> xspaces;
  // prepare host 1.
  auto space1 = std::make_unique<XSpace>();
  *(space1->add_hostnames()) = "hostname1";
  // with index 0 which shouldn't impact the space finding by name.
  xspaces.push_back(std::move(space1));

  // prepare host 0.
  auto space0 = std::make_unique<XSpace>();
  *(space0->add_hostnames()) = "hostname0";
  // with index 1 which shouldn't impact the space finding by name.
  xspaces.push_back(std::move(space0));

  auto session_snapshot_or =
      SessionSnapshot::Create({"log/plugins/profile/hostname0.xplane.pb",
                               "log/plugins/profile/hostname1.xplane.pb"},
                              std::move(xspaces));
  auto error =
      "The hostname of xspace path and preloaded xpace don't match at index: "
      "0. \nThe host name of xpace path is hostname0 but the host name of "
      "preloaded xpace is hostname1.";
  EXPECT_THAT(session_snapshot_or.status().code(),
              Eq(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(session_snapshot_or.status().message(), error);
}

TEST(Repository, ClearCacheFiles) {
  // Create a temp directory for the test.
  auto temp_dir = ::testing::TempDir();
  auto profile_dir = tsl::io::JoinPath(temp_dir, "log/plugins/profile");
  TF_CHECK_OK(tsl::Env::Default()->RecursivelyCreateDir(profile_dir));
  auto xplane_path = tsl::io::JoinPath(profile_dir, "hostname0.xplane.pb");
  std::unique_ptr<tsl::WritableFile> xplane_file;
  TF_CHECK_OK(
      tsl::Env::Default()->NewAppendableFile(xplane_path, &xplane_file));

  std::vector<std::unique_ptr<XSpace>> xspaces;
  // prepare host 0.
  auto space0 = std::make_unique<XSpace>();
  *(space0->add_hostnames()) = "hostname0";
  // with index 1 which shouldn't impact the space finding by name.
  xspaces.push_back(std::move(space0));
  auto session_snapshot_or =
      SessionSnapshot::Create({xplane_path}, /*xspaces=*/std::nullopt);
  TF_CHECK_OK(session_snapshot_or.status());
  EXPECT_TRUE(session_snapshot_or.value().HasAccessibleRunDir());

  // Generate Dummy HLO OpStats file.
  OpStats op_stats;
  op_stats.set_allocated_run_environment(new RunEnvironment());
  TF_CHECK_OK(session_snapshot_or.value().WriteBinaryProto(
      StoredDataType::OP_STATS, "hostname0", op_stats));
  auto opt_statsfile_path = session_snapshot_or.value().GetHostDataFilePath(
      StoredDataType::OP_STATS, "hostname0");
  EXPECT_TRUE(opt_statsfile_path.value().has_value());

  // Check that the cache file should be deleted
  TF_CHECK_OK(session_snapshot_or.value().ClearCacheFiles());
  opt_statsfile_path = session_snapshot_or.value().GetHostDataFilePath(
      StoredDataType::OP_STATS, "hostname0");
  EXPECT_FALSE(opt_statsfile_path.value().has_value());
  EXPECT_TRUE(tsl::Env::Default()->FileExists(xplane_path).ok());
}

}  // namespace
}  // namespace profiler
}  // namespace tensorflow
