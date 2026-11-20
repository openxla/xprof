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

#include "plugin/xprof/worker/worker_service.h"

#if !defined(_WIN32)
#include <sys/stat.h>
#include <unistd.h>
#endif

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>
#include <thread>  // NOLINT(build/c++11)
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/cleanup/cleanup.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/notification.h"
#include "absl/time/time.h"
#include "grpc/grpc_security_constants.h"
#include "grpcpp/client_context.h"
#include "grpcpp/create_channel.h"
#include "grpcpp/security/credentials.h"
#include "grpcpp/server_context.h"
#include "grpcpp/support/status.h"
#include "xla/tsl/platform/env.h"
#include "tsl/platform/path.h"
#include "tsl/profiler/protobuf/xplane.pb.h"
#include "xprof/convert/file_utils.h"
#include "xprof/convert/profile_processor.h"
#include "xprof/convert/profile_processor_factory.h"
#include "xprof/convert/repository.h"
#include "xprof/convert/tool_options.h"
#include "plugin/xprof/protobuf/worker_service.grpc.pb.h"
#include "plugin/xprof/worker/grpc_server.h"

namespace xprof {
namespace profiler {
namespace {

using ::absl_testing::IsOk;
using ::absl_testing::StatusIs;
using ::tensorflow::profiler::SessionSnapshot;
using ::tensorflow::profiler::ToolOptions;
using ::tensorflow::profiler::XSpace;
using ::testing::HasSubstr;
using ::testing::IsEmpty;
using ::testing::Not;

::grpc::Status CallWorkerGetProfileData(
    ProfileWorkerServiceImpl& service, absl::string_view session_id,
    absl::string_view tool_name,
    ::xprof::pywrap::WorkerProfileDataResponse* response) {
  ::grpc::ServerContext context;
  ::xprof::pywrap::WorkerProfileDataRequest request;
  request.mutable_origin_request()->set_session_id(session_id);
  request.mutable_origin_request()->set_tool_name(tool_name);
  return service.GetProfileData(&context, &request, response);
}

::grpc::Status CallStubGetProfileData(
    int port, absl::string_view session_id, absl::string_view tool_name,
    ::xprof::pywrap::WorkerProfileDataResponse* response,
    absl::string_view large_param_value = "") {
  auto channel = ::grpc::CreateChannel(
      absl::StrCat("127.0.0.1:", port),
      ::grpc::experimental::LocalCredentials(LOCAL_TCP));
  auto stub =
      ::xprof::pywrap::grpc::XprofAnalysisWorkerService::NewStub(channel);
  ::grpc::ClientContext context;
  ::xprof::pywrap::WorkerProfileDataRequest request;
  request.mutable_origin_request()->set_session_id(session_id);
  request.mutable_origin_request()->set_tool_name(tool_name);
  if (!large_param_value.empty()) {
    (*request.mutable_origin_request()->mutable_parameters())["padding"] =
        std::string(large_param_value);
  }
  return stub->GetProfileData(&context, request, response);
}

TEST(GrpcServerAddressTest, FormatsIpv4Ipv6WhitespaceAndDefaultAddresses) {
  using internal::FormatGrpcServerAddress;
  EXPECT_EQ(FormatGrpcServerAddress("", 50051), "127.0.0.1:50051");
  EXPECT_EQ(FormatGrpcServerAddress("   \t\n ", 50051), "127.0.0.1:50051");
  EXPECT_EQ(FormatGrpcServerAddress("0.0.0.0", 50051), "0.0.0.0:50051");
  EXPECT_EQ(FormatGrpcServerAddress(" 10.1.2.3 ", 50051), "10.1.2.3:50051");
  EXPECT_EQ(FormatGrpcServerAddress("::1", 50051), "[::1]:50051");
  EXPECT_EQ(FormatGrpcServerAddress(" ::1 ", 50051), "[::1]:50051");
  EXPECT_EQ(FormatGrpcServerAddress("::", 50051), "[::]:50051");
  EXPECT_EQ(FormatGrpcServerAddress("[::1]", 50051), "[::1]:50051");
  EXPECT_EQ(FormatGrpcServerAddress(" [::1] ", 50051), "[::1]:50051");
}

class ProfileWorkerServiceSecurityTest : public testing::Test {
 protected:
  void SetUp() override {
    internal::ResetWorkerAllowedRootDirsForTesting();
    const testing::TestInfo* const test_info =
        testing::UnitTest::GetInstance()->current_test_info();
    base_dir_ =
        tsl::io::JoinPath(testing::TempDir(),
                          absl::StrCat("worker_security_", test_info->name()));
    CHECK_OK(tsl::Env::Default()->RecursivelyCreateDir(base_dir_));
  }

  void TearDown() override {
    internal::ResetWorkerAllowedRootDirsForTesting();
    int64_t undeleted_files = 0;
    int64_t undeleted_dirs = 0;
    tsl::Env::Default()
        ->DeleteRecursively(base_dir_, &undeleted_files, &undeleted_dirs)
        .IgnoreError();
  }

  std::string base_dir_;
};

TEST_F(ProfileWorkerServiceSecurityTest,
       RejectsRequestWhenNoAllowedRootConfigured) {
  XSpace space;
  space.add_planes()->set_name("test_plane");
  std::string xspace_path = tsl::io::JoinPath(base_dir_, "host1.xplane.pb");
  ASSERT_THAT(xprof::WriteBinaryProto(xspace_path, space), IsOk());

  auto service =
      internal::CreateProfileWorkerServiceWithEnvForTesting(1, "", "");
  ::xprof::pywrap::WorkerProfileDataResponse response;
  ::grpc::Status status = CallWorkerGetProfileData(*service, xspace_path,
                                                   "overview_page", &response);
  EXPECT_EQ(status.error_code(), ::grpc::StatusCode::PERMISSION_DENIED);

  EXPECT_FALSE(AddWorkerAllowedRootDir("/"));
  EXPECT_FALSE(AddWorkerAllowedRootDir(xspace_path));
  std::string nul_root =
      absl::StrCat(base_dir_, absl::string_view("\0", 1), "extra");
  EXPECT_FALSE(AddWorkerAllowedRootDir(nul_root));
  EXPECT_FALSE(AddWorkerAllowedRootDir("\\\\attacker.example\\share"));
  EXPECT_FALSE(AddWorkerAllowedRootDir("//attacker.example/share"));

  ProfileWorkerServiceImpl root_slash_service(1, "/");
  EXPECT_EQ(CallWorkerGetProfileData(root_slash_service, xspace_path,
                                     "overview_page", &response)
                .error_code(),
            ::grpc::StatusCode::PERMISSION_DENIED);
  EXPECT_THAT(tsl::Env::Default()->FileExists(
                  tsl::io::JoinPath(base_dir_, "host1.op_stats_v2.pb")),
              StatusIs(absl::StatusCode::kNotFound));

  // Even after a valid root is added to the global allowlist, an instance
  // configured with "/", a regular file, an embedded NUL, or a nonexistent root
  // must fail closed instead of falling through to the global allowlist.
  EXPECT_TRUE(AddWorkerAllowedRootDir(base_dir_));
  EXPECT_EQ(CallWorkerGetProfileData(root_slash_service, xspace_path,
                                     "overview_page", &response)
                .error_code(),
            ::grpc::StatusCode::PERMISSION_DENIED);
  ProfileWorkerServiceImpl file_root_service(1, xspace_path);
  EXPECT_EQ(CallWorkerGetProfileData(file_root_service, xspace_path,
                                     "overview_page", &response)
                .error_code(),
            ::grpc::StatusCode::PERMISSION_DENIED);
  ProfileWorkerServiceImpl nul_root_service(1, nul_root);
  EXPECT_EQ(CallWorkerGetProfileData(nul_root_service, xspace_path,
                                     "overview_page", &response)
                .error_code(),
            ::grpc::StatusCode::PERMISSION_DENIED);
  ProfileWorkerServiceImpl missing_root_service(
      1, tsl::io::JoinPath(base_dir_, "missing_root"));
  EXPECT_EQ(CallWorkerGetProfileData(missing_root_service, xspace_path,
                                     "overview_page", &response)
                .error_code(),
            ::grpc::StatusCode::PERMISSION_DENIED);

  auto env_slash_service =
      internal::CreateProfileWorkerServiceWithEnvForTesting(1, "", "/");
  EXPECT_EQ(CallWorkerGetProfileData(*env_slash_service, xspace_path,
                                     "overview_page", &response)
                .error_code(),
            ::grpc::StatusCode::PERMISSION_DENIED);

  auto env_empty_service =
      internal::CreateProfileWorkerServiceWithEnvForTesting(1, "", "");
  ::grpc::Status env_empty_status = CallWorkerGetProfileData(
      *env_empty_service, xspace_path, "overview_page", &response);
  EXPECT_TRUE(env_empty_status.ok()) << env_empty_status.error_message();
}

TEST_F(ProfileWorkerServiceSecurityTest,
       RejectsPathOutsideAllowedRootAndPrefixCollision) {
  std::string allowed_dir = tsl::io::JoinPath(base_dir_, "allowed_root");
  std::string other_dir = tsl::io::JoinPath(base_dir_, "other_root");
  std::string evil_prefix_dir =
      tsl::io::JoinPath(base_dir_, "allowed_root_evil");
  ASSERT_THAT(tsl::Env::Default()->RecursivelyCreateDir(allowed_dir), IsOk());
  ASSERT_THAT(tsl::Env::Default()->RecursivelyCreateDir(other_dir), IsOk());
  ASSERT_THAT(tsl::Env::Default()->RecursivelyCreateDir(evil_prefix_dir),
              IsOk());

  XSpace space;
  space.add_planes()->set_name("test_plane");
  std::string other_xspace = tsl::io::JoinPath(other_dir, "host.xplane.pb");
  std::string evil_xspace =
      tsl::io::JoinPath(evil_prefix_dir, "host.xplane.pb");
  ASSERT_THAT(xprof::WriteBinaryProto(other_xspace, space), IsOk());
  ASSERT_THAT(xprof::WriteBinaryProto(evil_xspace, space), IsOk());

  ProfileWorkerServiceImpl service(1, allowed_dir);
  EXPECT_TRUE(AddWorkerAllowedRootDir(other_dir));
  ::xprof::pywrap::WorkerProfileDataResponse response;

  EXPECT_EQ(CallWorkerGetProfileData(service, other_xspace, "overview_page",
                                     &response)
                .error_code(),
            ::grpc::StatusCode::PERMISSION_DENIED);
  EXPECT_THAT(tsl::Env::Default()->FileExists(
                  tsl::io::JoinPath(other_dir, "host.op_stats_v2.pb")),
              StatusIs(absl::StatusCode::kNotFound));

  EXPECT_EQ(
      CallWorkerGetProfileData(service, evil_xspace, "overview_page", &response)
          .error_code(),
      ::grpc::StatusCode::PERMISSION_DENIED);
  EXPECT_THAT(tsl::Env::Default()->FileExists(
                  tsl::io::JoinPath(evil_prefix_dir, "host.op_stats_v2.pb")),
              StatusIs(absl::StatusCode::kNotFound));
}

TEST_F(ProfileWorkerServiceSecurityTest,
       EnforcesInstanceRemoteBucketPrefixAndRejectsTraversal) {
  ProfileWorkerServiceImpl service(1, "gs://allowed-bucket/profiles///");
  ::xprof::pywrap::WorkerProfileDataResponse response;

  for (absl::string_view denied_remote_path : {
           "gs://other-bucket/profiles/host.xplane.pb",
           "gs://allowed-bucket/profiles_evil/host.xplane.pb",
           "gs://allowed-bucket/profiles//host.xplane.pb",
           "gs://allowed-bucket/profiles\\sub/host.xplane.pb",
       }) {
    EXPECT_EQ(CallWorkerGetProfileData(service, denied_remote_path,
                                       "overview_page", &response)
                  .error_code(),
              ::grpc::StatusCode::PERMISSION_DENIED)
        << "remote_path=" << denied_remote_path;
  }

  for (absl::string_view traversal_remote_path : {
           "gs://allowed-bucket/profiles/../other/host.xplane.pb",
           "gs://allowed-bucket/profiles/./host.xplane.pb",
           "gs://allowed-bucket/profiles\\..\\other\\host.xplane.pb",
       }) {
    ::grpc::Status status = CallWorkerGetProfileData(
        service, traversal_remote_path, "overview_page", &response);
    EXPECT_EQ(status.error_code(), ::grpc::StatusCode::INVALID_ARGUMENT)
        << "remote_path=" << traversal_remote_path;
    EXPECT_THAT(status.error_message(), HasSubstr("Path traversal"));
  }

  std::string remote_nul_path =
      absl::StrCat("gs://allowed-bucket/profiles/run1/host",
                   absl::string_view("\0", 1), "evil.xplane.pb");
  for (absl::string_view invalid_remote_path : {
           absl::string_view("gs://allowed-bucket/profiles/run1/secret.txt"),
           absl::string_view(
               "gs://allowed-bucket/profiles/run1/.hidden.xplane.pb"),
           absl::string_view(remote_nul_path),
       }) {
    ::grpc::Status status = CallWorkerGetProfileData(
        service, invalid_remote_path, "overview_page", &response);
    EXPECT_EQ(status.error_code(), ::grpc::StatusCode::INVALID_ARGUMENT);
    EXPECT_THAT(status.error_message(), HasSubstr(".xplane.pb"));
  }

  ::grpc::Status allowed_status = CallWorkerGetProfileData(
      service, "gs://allowed-bucket/profiles/run1/host.xplane.pb",
      "unknown_tool", &response);
  EXPECT_EQ(allowed_status.error_code(), ::grpc::StatusCode::INVALID_ARGUMENT);
  EXPECT_THAT(allowed_status.error_message(),
              HasSubstr("Can not find tool: unknown_tool"));

  for (absl::string_view malformed_root : {
           "gs:///",
           "gs:////tenant-bucket",
           "gs://allowed-bucket/profiles/..",
           "gs://allowed-bucket/a//b",
           "gs://allowed-bucket\\evil/profiles",
       }) {
    EXPECT_FALSE(AddWorkerAllowedRootDir(malformed_root))
        << "malformed_root=" << malformed_root;
    ProfileWorkerServiceImpl malformed_service(1, malformed_root);
    EXPECT_EQ(CallWorkerGetProfileData(
                  malformed_service,
                  "gs://allowed-bucket/profiles/run1/host.xplane.pb",
                  "unknown_tool", &response)
                  .error_code(),
              ::grpc::StatusCode::PERMISSION_DENIED)
        << "malformed_root=" << malformed_root;
  }

  EXPECT_FALSE(AddWorkerAllowedRootDir("gs://"));
}

TEST_F(ProfileWorkerServiceSecurityTest,
       SupportsGcsSchemeRootAndBlocksNonGcsPaths) {
  XSpace space;
  space.add_planes()->set_name("test_plane");
  std::string xspace_path =
      tsl::io::JoinPath(base_dir_, "local_host.xplane.pb");
  ASSERT_THAT(xprof::WriteBinaryProto(xspace_path, space), IsOk());

  ProfileWorkerServiceImpl gcs_scheme_service(1, "gs://");
  auto env_gcs_scheme_service =
      internal::CreateProfileWorkerServiceWithEnvForTesting(1, "", "gs://");

  ::xprof::pywrap::WorkerProfileDataResponse response;
  for (ProfileWorkerServiceImpl* scheme_service :
       {&gcs_scheme_service, env_gcs_scheme_service.get()}) {
    for (absl::string_view allowed_gcs_path : {
             "gs://tenant-bucket/run1/host.xplane.pb",
             "gs://tenant-bucket/host.xplane.pb",
         }) {
      ::grpc::Status status = CallWorkerGetProfileData(
          *scheme_service, allowed_gcs_path, "unknown_tool", &response);
      EXPECT_EQ(status.error_code(), ::grpc::StatusCode::INVALID_ARGUMENT);
      EXPECT_THAT(status.error_message(),
                  HasSubstr("Can not find tool: unknown_tool"));
    }
    for (absl::string_view denied_scheme_path : {
             absl::string_view(xspace_path),
             absl::string_view("gs:///host.xplane.pb"),
             absl::string_view("gs://tenant-bucket//host.xplane.pb"),
             absl::string_view("gs://tenant-bucket\\evil/host.xplane.pb"),
         }) {
      EXPECT_EQ(CallWorkerGetProfileData(*scheme_service, denied_scheme_path,
                                         "unknown_tool", &response)
                    .error_code(),
                ::grpc::StatusCode::PERMISSION_DENIED)
          << "path=" << denied_scheme_path;
    }
    EXPECT_EQ(CallWorkerGetProfileData(*scheme_service, "gs://host.xplane.pb",
                                       "unknown_tool", &response)
                  .error_code(),
              ::grpc::StatusCode::INVALID_ARGUMENT);
  }
}

TEST_F(ProfileWorkerServiceSecurityTest, EnforcesDynamicRemoteRootsAndReset) {
  auto dynamic_service =
      internal::CreateProfileWorkerServiceWithEnvForTesting(1, "", "");
  ::xprof::pywrap::WorkerProfileDataResponse response;

  EXPECT_FALSE(AddWorkerAllowedRootDir("gs://"));
  EXPECT_FALSE(AddWorkerAllowedRootDir("gs://allowed-bucket/profiles/.."));
  EXPECT_FALSE(AddWorkerAllowedRootDir("gs://allowed-bucket/a//b"));
  EXPECT_EQ(CallWorkerGetProfileData(*dynamic_service,
                                     "gs://tenant-bucket/run1/host.xplane.pb",
                                     "unknown_tool", &response)
                .error_code(),
            ::grpc::StatusCode::PERMISSION_DENIED);

  EXPECT_TRUE(AddWorkerAllowedRootDir("gs://allowed-bucket/profiles///"));

  for (absl::string_view denied_dynamic_path : {
           "gs://other-bucket/profiles/host.xplane.pb",
           "gs://allowed-bucket/profiles_evil/host.xplane.pb",
           "gs://allowed-bucket/profiles//host.xplane.pb",
       }) {
    EXPECT_EQ(CallWorkerGetProfileData(*dynamic_service, denied_dynamic_path,
                                       "overview_page", &response)
                  .error_code(),
              ::grpc::StatusCode::PERMISSION_DENIED)
        << "remote_path=" << denied_dynamic_path;
  }

  for (absl::string_view allowed_dynamic_path : {
           "gs://allowed-bucket/profiles/run1/host.xplane.pb",
       }) {
    ::grpc::Status status = CallWorkerGetProfileData(
        *dynamic_service, allowed_dynamic_path, "unknown_tool", &response);
    EXPECT_EQ(status.error_code(), ::grpc::StatusCode::INVALID_ARGUMENT);
    EXPECT_THAT(status.error_message(),
                HasSubstr("Can not find tool: unknown_tool"));
  }

  internal::ResetWorkerAllowedRootDirsForTesting();
  EXPECT_EQ(
      CallWorkerGetProfileData(
          *dynamic_service, "gs://allowed-bucket/profiles/run1/host.xplane.pb",
          "unknown_tool", &response)
          .error_code(),
      ::grpc::StatusCode::PERMISSION_DENIED);
}

TEST_F(ProfileWorkerServiceSecurityTest,
       RejectsPathTraversalAndNonXplaneExtensions) {
  std::string non_xplane_path = tsl::io::JoinPath(base_dir_, "passwd");
  ASSERT_THAT(tsl::WriteStringToFile(tsl::Env::Default(), non_xplane_path, "x"),
              IsOk());
  std::string hidden_host_path = tsl::io::JoinPath(base_dir_, ".xplane.pb");
  ASSERT_THAT(
      tsl::WriteStringToFile(tsl::Env::Default(), hidden_host_path, "x"),
      IsOk());
  std::string dot_prefixed_host_path =
      tsl::io::JoinPath(base_dir_, ".hidden.xplane.pb");
  ASSERT_THAT(
      tsl::WriteStringToFile(tsl::Env::Default(), dot_prefixed_host_path, "x"),
      IsOk());
  std::string empty_xplane_path =
      tsl::io::JoinPath(base_dir_, "empty.xplane.pb");
  ASSERT_THAT(
      tsl::WriteStringToFile(tsl::Env::Default(), empty_xplane_path, ""),
      IsOk());
  std::string dir_xplane_path = tsl::io::JoinPath(base_dir_, "dir.xplane.pb");
  ASSERT_THAT(tsl::Env::Default()->RecursivelyCreateDir(dir_xplane_path),
              IsOk());
  XSpace space;
  space.add_planes()->set_name("test_plane");
  std::string valid_xplane_path =
      tsl::io::JoinPath(base_dir_, "valid.xplane.pb");
  ASSERT_THAT(xprof::WriteBinaryProto(valid_xplane_path, space), IsOk());
  std::string uncleaned_path = absl::StrCat(base_dir_, "//valid.xplane.pb");
  std::string nonexistent_file_path =
      tsl::io::JoinPath(base_dir_, "nonexistent_host.xplane.pb");
  std::string nonexistent_dir_path =
      tsl::io::JoinPath(base_dir_, "nonexistent_dir/host.xplane.pb");
  std::string traversal_path =
      absl::StrCat(base_dir_, "/../other/host.xplane.pb");
  std::string nul_byte_path = absl::StrCat(
      base_dir_, "/host", absl::string_view("\0", 1), "evil.xplane.pb");

  ProfileWorkerServiceImpl service(1, base_dir_);
  ::xprof::pywrap::WorkerProfileDataResponse response;

  ::grpc::ServerContext context;
  ::xprof::pywrap::WorkerProfileDataRequest request;
  EXPECT_EQ(service.GetProfileData(&context, nullptr, &response).error_code(),
            ::grpc::StatusCode::INVALID_ARGUMENT);
  EXPECT_EQ(service.GetProfileData(&context, &request, nullptr).error_code(),
            ::grpc::StatusCode::INVALID_ARGUMENT);

  for (absl::string_view invalid_session_id : {
           absl::string_view(""),
           absl::string_view(traversal_path),
           absl::string_view(non_xplane_path),
           absl::string_view("gs://allowed-bucket/profiles/run1/secret.txt"),
           absl::string_view(hidden_host_path),
           absl::string_view(dot_prefixed_host_path),
           absl::string_view("relative/host.xplane.pb"),
           absl::string_view("/host.xplane.pb"),
           absl::string_view("\\\\attacker.example\\share\\host.xplane.pb"),
           absl::string_view("//attacker.example/share/host.xplane.pb"),
           absl::string_view(uncleaned_path),
           absl::string_view(nonexistent_file_path),
           absl::string_view(empty_xplane_path),
           absl::string_view(dir_xplane_path),
           absl::string_view(nul_byte_path),
       }) {
    EXPECT_EQ(CallWorkerGetProfileData(service, invalid_session_id,
                                       "overview_page", &response)
                  .error_code(),
              ::grpc::StatusCode::INVALID_ARGUMENT)
        << "session_id=" << invalid_session_id;
  }

#if !defined(_WIN32)
  std::string fifo_path = tsl::io::JoinPath(base_dir_, "named_pipe.xplane.pb");
  ASSERT_EQ(::mkfifo(fifo_path.c_str(), 0600), 0);
  ::grpc::Status fifo_status =
      CallWorkerGetProfileData(service, fifo_path, "overview_page", &response);
  EXPECT_EQ(fifo_status.error_code(), ::grpc::StatusCode::INVALID_ARGUMENT);
  EXPECT_THAT(fifo_status.error_message(), HasSubstr("non-empty XPlane file"));
#endif

  ::grpc::Status nonexistent_file_status = CallWorkerGetProfileData(
      service, nonexistent_file_path, "overview_page", &response);
  EXPECT_EQ(nonexistent_file_status.error_code(),
            ::grpc::StatusCode::INVALID_ARGUMENT);
  EXPECT_THAT(nonexistent_file_status.error_message(),
              HasSubstr("does not exist"));

  ::grpc::Status empty_file_status = CallWorkerGetProfileData(
      service, empty_xplane_path, "overview_page", &response);
  EXPECT_EQ(empty_file_status.error_code(),
            ::grpc::StatusCode::INVALID_ARGUMENT);
  EXPECT_THAT(empty_file_status.error_message(),
              HasSubstr("non-empty XPlane file"));

  EXPECT_EQ(CallWorkerGetProfileData(service, nonexistent_dir_path,
                                     "overview_page", &response)
                .error_code(),
            ::grpc::StatusCode::PERMISSION_DENIED);
}

TEST_F(ProfileWorkerServiceSecurityTest, RejectsSymlinkDirectoryAndFileEscape) {
#if !defined(_WIN32)
  std::string allowed_dir = tsl::io::JoinPath(base_dir_, "allowed_dir");
  std::string outside_dir = tsl::io::JoinPath(base_dir_, "outside_dir");
  ASSERT_THAT(tsl::Env::Default()->RecursivelyCreateDir(allowed_dir), IsOk());
  ASSERT_THAT(tsl::Env::Default()->RecursivelyCreateDir(outside_dir), IsOk());

  XSpace space;
  space.add_planes()->set_name("test_plane");
  std::string outside_xplane = tsl::io::JoinPath(outside_dir, "host.xplane.pb");
  ASSERT_THAT(xprof::WriteBinaryProto(outside_xplane, space), IsOk());
  std::string outside_secret = tsl::io::JoinPath(outside_dir, "secret.txt");
  ASSERT_THAT(
      tsl::WriteStringToFile(tsl::Env::Default(), outside_secret, "secret"),
      IsOk());
  std::string inside_secret = tsl::io::JoinPath(allowed_dir, "secret.txt");
  ASSERT_THAT(
      tsl::WriteStringToFile(tsl::Env::Default(), inside_secret, "secret"),
      IsOk());
  std::string inside_xplane =
      tsl::io::JoinPath(allowed_dir, "real_host.xplane.pb");
  ASSERT_THAT(xprof::WriteBinaryProto(inside_xplane, space), IsOk());

  std::string escape_dir_link = tsl::io::JoinPath(allowed_dir, "escape_dir");
  ASSERT_EQ(::symlink(outside_dir.c_str(), escape_dir_link.c_str()), 0);
  std::string escape_xplane_link =
      tsl::io::JoinPath(allowed_dir, "escape_xplane.xplane.pb");
  ASSERT_EQ(::symlink(outside_xplane.c_str(), escape_xplane_link.c_str()), 0);
  std::string escape_secret_link =
      tsl::io::JoinPath(allowed_dir, "escape_secret.xplane.pb");
  ASSERT_EQ(::symlink(outside_secret.c_str(), escape_secret_link.c_str()), 0);
  std::string inside_secret_link =
      tsl::io::JoinPath(allowed_dir, "inside_secret.xplane.pb");
  ASSERT_EQ(::symlink(inside_secret.c_str(), inside_secret_link.c_str()), 0);
  std::string inside_xplane_link =
      tsl::io::JoinPath(allowed_dir, "alias_host.xplane.pb");
  ASSERT_EQ(::symlink(inside_xplane.c_str(), inside_xplane_link.c_str()), 0);

  ProfileWorkerServiceImpl service(1, allowed_dir);
  ::xprof::pywrap::WorkerProfileDataResponse response;

  EXPECT_EQ(CallWorkerGetProfileData(
                service, tsl::io::JoinPath(escape_dir_link, "host.xplane.pb"),
                "overview_page", &response)
                .error_code(),
            ::grpc::StatusCode::PERMISSION_DENIED);
  EXPECT_THAT(tsl::Env::Default()->FileExists(
                  tsl::io::JoinPath(outside_dir, "host.op_stats_v2.pb")),
              StatusIs(absl::StatusCode::kNotFound));

  EXPECT_EQ(CallWorkerGetProfileData(service, escape_xplane_link,
                                     "overview_page", &response)
                .error_code(),
            ::grpc::StatusCode::PERMISSION_DENIED);
  EXPECT_THAT(tsl::Env::Default()->FileExists(tsl::io::JoinPath(
                  allowed_dir, "escape_xplane.op_stats_v2.pb")),
              StatusIs(absl::StatusCode::kNotFound));

  // Symlink escaping the allowed root must fail with PERMISSION_DENIED even
  // when its target does not have an XPlane extension, avoiding an out-of-root
  // target-extension oracle.
  EXPECT_EQ(CallWorkerGetProfileData(service, escape_secret_link,
                                     "overview_page", &response)
                .error_code(),
            ::grpc::StatusCode::PERMISSION_DENIED);

  // Symlink staying within the allowed root to a non-XPlane file fails with
  // INVALID_ARGUMENT on target filename validation.
  ::grpc::Status inside_secret_status = CallWorkerGetProfileData(
      service, inside_secret_link, "overview_page", &response);
  EXPECT_EQ(inside_secret_status.error_code(),
            ::grpc::StatusCode::INVALID_ARGUMENT);
  EXPECT_THAT(inside_secret_status.error_message(),
              HasSubstr("Session file target must be"));

  // Symlink staying within the allowed root to a valid XPlane file succeeds.
  ::grpc::Status inside_xplane_status = CallWorkerGetProfileData(
      service, inside_xplane_link, "overview_page", &response);
  EXPECT_TRUE(inside_xplane_status.ok())
      << inside_xplane_status.error_message();
#endif
}

TEST_F(ProfileWorkerServiceSecurityTest,
       RegisterAndResolveStartupWorkerRootHandlesDynamicAndOverrideModes) {
  std::string startup_dir = tsl::io::JoinPath(base_dir_, "startup_dir");
  std::string startup_nested_dir = tsl::io::JoinPath(startup_dir, "run1");
  std::string dynamic_dir = tsl::io::JoinPath(base_dir_, "dynamic_dir");
  std::string unregistered_dir =
      tsl::io::JoinPath(base_dir_, "unregistered_dir");
  std::string env_dir = tsl::io::JoinPath(base_dir_, "env_dir");
  ASSERT_THAT(tsl::Env::Default()->RecursivelyCreateDir(startup_nested_dir),
              IsOk());
  ASSERT_THAT(tsl::Env::Default()->RecursivelyCreateDir(dynamic_dir), IsOk());
  ASSERT_THAT(tsl::Env::Default()->RecursivelyCreateDir(unregistered_dir),
              IsOk());
  ASSERT_THAT(tsl::Env::Default()->RecursivelyCreateDir(env_dir), IsOk());

  XSpace space;
  space.add_planes()->set_name("test_plane");
  std::string startup_xspace =
      tsl::io::JoinPath(startup_nested_dir, "host.xplane.pb");
  std::string dynamic_xspace = tsl::io::JoinPath(dynamic_dir, "host.xplane.pb");
  std::string unregistered_xspace =
      tsl::io::JoinPath(unregistered_dir, "host.xplane.pb");
  std::string env_xspace = tsl::io::JoinPath(env_dir, "host.xplane.pb");
  ASSERT_THAT(xprof::WriteBinaryProto(startup_xspace, space), IsOk());
  ASSERT_THAT(xprof::WriteBinaryProto(dynamic_xspace, space), IsOk());
  ASSERT_THAT(xprof::WriteBinaryProto(unregistered_xspace, space), IsOk());
  ASSERT_THAT(xprof::WriteBinaryProto(env_xspace, space), IsOk());

  EXPECT_FALSE(AddWorkerAllowedRootDir(""));
  EXPECT_FALSE(AddWorkerAllowedRootDir("/"));
  EXPECT_FALSE(AddWorkerAllowedRootDir("gs://"));
  EXPECT_FALSE(AddWorkerAllowedRootDir("gs:///"));
  EXPECT_FALSE(AddWorkerAllowedRootDir("gs:////tenant-bucket"));
  EXPECT_FALSE(
      AddWorkerAllowedRootDir(tsl::io::JoinPath(base_dir_, "missing_dir")));
  EXPECT_TRUE(AddWorkerAllowedRootDir("gs://allowed-bucket/runs"));
  internal::ResetWorkerAllowedRootDirsForTesting();

  EXPECT_EQ(
      internal::RegisterAndResolveStartupWorkerRootWithEnvForTesting("", ""),
      "");
  EXPECT_EQ(internal::RegisterAndResolveStartupWorkerRootWithEnvForTesting(
                "gs://", ""),
            "gs://");

  // Valid startup logdir without XPROF_WORKER_LOGDIR registers startup_dir in
  // the global allowlist and returns "" so nested and dynamic session roots
  // also work.
  std::string startup_instance_root =
      internal::RegisterAndResolveStartupWorkerRootWithEnvForTesting(
          startup_dir, "");
  EXPECT_EQ(startup_instance_root, "");
  auto startup_service = internal::CreateProfileWorkerServiceWithEnvForTesting(
      1, startup_instance_root, "");
  ::xprof::pywrap::WorkerProfileDataResponse response;
  ::grpc::Status startup_status = CallWorkerGetProfileData(
      *startup_service, startup_xspace, "overview_page", &response);
  EXPECT_TRUE(startup_status.ok()) << startup_status.error_message();
  EXPECT_EQ(CallWorkerGetProfileData(*startup_service, dynamic_xspace,
                                     "overview_page", &response)
                .error_code(),
            ::grpc::StatusCode::PERMISSION_DENIED);
  EXPECT_TRUE(AddWorkerAllowedRootDir(dynamic_dir));
  ::grpc::Status dynamic_status = CallWorkerGetProfileData(
      *startup_service, dynamic_xspace, "overview_page", &response);
  EXPECT_TRUE(dynamic_status.ok()) << dynamic_status.error_message();
  EXPECT_EQ(CallWorkerGetProfileData(*startup_service, unregistered_xspace,
                                     "overview_page", &response)
                .error_code(),
            ::grpc::StatusCode::PERMISSION_DENIED);

  // Invalid startup logdir ("/") is forwarded as "/" and fails closed even
  // when valid directories exist in the global allowlist.
  std::string slash_instance_root =
      internal::RegisterAndResolveStartupWorkerRootWithEnvForTesting("/", "");
  EXPECT_EQ(slash_instance_root, "/");
  ProfileWorkerServiceImpl slash_service(1, slash_instance_root);
  EXPECT_EQ(CallWorkerGetProfileData(slash_service, startup_xspace,
                                     "overview_page", &response)
                .error_code(),
            ::grpc::StatusCode::PERMISSION_DENIED);

  // When XPROF_WORKER_LOGDIR is set, explicit startup logdir is returned as
  // the authoritative instance root without modifying the global allowlist.
  internal::ResetWorkerAllowedRootDirsForTesting();
  EXPECT_EQ(internal::RegisterAndResolveStartupWorkerRootWithEnvForTesting(
                "", env_dir),
            "");
  std::string override_instance_root =
      internal::RegisterAndResolveStartupWorkerRootWithEnvForTesting(
          startup_dir, env_dir);
  EXPECT_EQ(override_instance_root, startup_dir);
  auto override_service = internal::CreateProfileWorkerServiceWithEnvForTesting(
      1, override_instance_root, env_dir);

  ::grpc::Status override_status = CallWorkerGetProfileData(
      *override_service, startup_xspace, "overview_page", &response);
  EXPECT_TRUE(override_status.ok()) << override_status.error_message();
  EXPECT_EQ(CallWorkerGetProfileData(*override_service, env_xspace,
                                     "overview_page", &response)
                .error_code(),
            ::grpc::StatusCode::PERMISSION_DENIED);

  // Confirm startup_dir was not inserted into the global allowlist while
  // XPROF_WORKER_LOGDIR was set.
  auto global_check_service =
      internal::CreateProfileWorkerServiceWithEnvForTesting(1, "", "");
  EXPECT_EQ(CallWorkerGetProfileData(*global_check_service, startup_xspace,
                                     "overview_page", &response)
                .error_code(),
            ::grpc::StatusCode::PERMISSION_DENIED);
}

TEST_F(ProfileWorkerServiceSecurityTest,
       AcceptsValidXplaneClampsConcurrencyAndPropagatesMapErrors) {
  std::string run_dir = tsl::io::JoinPath(base_dir_, "run1");
  ASSERT_THAT(tsl::Env::Default()->RecursivelyCreateDir(run_dir), IsOk());

  XSpace space;
  space.add_planes()->set_name("test_plane");
  std::string xspace_path = tsl::io::JoinPath(run_dir, "host1.xplane.pb");
  ASSERT_THAT(xprof::WriteBinaryProto(xspace_path, space), IsOk());
  std::string riegeli_path = tsl::io::JoinPath(run_dir, "host2.xplane.riegeli");
  ASSERT_THAT(
      tsl::WriteStringToFile(tsl::Env::Default(), riegeli_path, "riegeli"),
      IsOk());

  // Non-positive max_concurrent_requests is clamped to 1 rather than crashing
  // or deadlocking.
  ProfileWorkerServiceImpl service(0, absl::StrCat(base_dir_, "///"));
  ::xprof::pywrap::WorkerProfileDataResponse response;
  ::grpc::Status status = CallWorkerGetProfileData(service, xspace_path,
                                                   "overview_page", &response);
  EXPECT_TRUE(status.ok()) << status.error_message();
  EXPECT_THAT(response.output(), Not(IsEmpty()));
  EXPECT_THAT(tsl::Env::Default()->FileExists(
                  tsl::io::JoinPath(run_dir, "host1.op_stats_v2.pb")),
              IsOk());

  ::grpc::Status riegeli_status = CallWorkerGetProfileData(
      service, riegeli_path, "overview_page", &response);
  EXPECT_TRUE(riegeli_status.ok()) << riegeli_status.error_message();
  EXPECT_THAT(tsl::Env::Default()->FileExists(
                  tsl::io::JoinPath(run_dir, "host2.op_stats_v2.pb")),
              IsOk());

  auto env_service = internal::CreateProfileWorkerServiceWithEnvForTesting(
      -5, "", base_dir_);
  ::grpc::Status env_status = CallWorkerGetProfileData(
      *env_service, xspace_path, "overview_page", &response);
  EXPECT_TRUE(env_status.ok()) << env_status.error_message();

  // Verify Map() non-OK status propagates cleanly through ToGrpcStatus.
  class FailingMapProcessor final : public xprof::ProfileProcessor {
   public:
    absl::StatusOr<std::string> Map(
        const std::string& /*xspace_path*/) override {
      return absl::DataLossError("corrupt xspace payload");
    }
    absl::StatusOr<std::string> Map(const SessionSnapshot& /*session_snapshot*/,
                                    const std::string& /*hostname*/,
                                    const XSpace& /*xspace*/) override {
      return absl::UnimplementedError("unused");
    }
    absl::Status Reduce(
        const SessionSnapshot& /*session_snapshot*/,
        const std::vector<std::string>& /*map_output_files*/) override {
      return absl::OkStatus();
    }
  };
  constexpr absl::string_view kFailingToolName = "test_failing_map_tool";
  xprof::ProfileProcessorFactory::GetInstance().Register(
      kFailingToolName, [](const ToolOptions& /*options*/) {
        return std::make_unique<FailingMapProcessor>();
      });
  auto factory_cleanup = absl::MakeCleanup([kFailingToolName] {
    xprof::ProfileProcessorFactory::GetInstance().Register(
        kFailingToolName,
        [](const ToolOptions& /*options*/)
            -> std::unique_ptr<xprof::ProfileProcessor> { return nullptr; });
  });
  ::grpc::Status failing_status = CallWorkerGetProfileData(
      service, xspace_path, kFailingToolName, &response);
  EXPECT_EQ(failing_status.error_code(), ::grpc::StatusCode::DATA_LOSS);
  EXPECT_THAT(failing_status.error_message(),
              HasSubstr("corrupt xspace payload"));
}

TEST_F(ProfileWorkerServiceSecurityTest,
       RevalidatesSessionPathAfterSemaphoreWait) {
#if !defined(_WIN32)
  class BlockingMapProcessor final : public xprof::ProfileProcessor {
   public:
    BlockingMapProcessor(absl::Notification* entered,
                         absl::Notification* release,
                         std::atomic<bool>* arm_next_block,
                         std::atomic<int>* map_calls)
        : entered_(entered),
          release_(release),
          arm_next_block_(arm_next_block),
          map_calls_(map_calls) {}

    absl::StatusOr<std::string> Map(const std::string& xspace_path) override {
      map_calls_->fetch_add(1, std::memory_order_acq_rel);
      if (arm_next_block_->exchange(false, std::memory_order_acq_rel)) {
        entered_->Notify();
        release_->WaitForNotification();
      }
      return absl::StrCat(xspace_path, ".mapped");
    }

    absl::StatusOr<std::string> Map(const SessionSnapshot& /*session_snapshot*/,
                                    const std::string& /*hostname*/,
                                    const XSpace& /*xspace*/) override {
      return absl::UnimplementedError("unused");
    }

    absl::Status Reduce(
        const SessionSnapshot& /*session_snapshot*/,
        const std::vector<std::string>& /*map_output_files*/) override {
      return absl::OkStatus();
    }

   private:
    absl::Notification* entered_;
    absl::Notification* release_;
    std::atomic<bool>* arm_next_block_;
    std::atomic<int>* map_calls_;
  };

  std::string allowed_dir = tsl::io::JoinPath(base_dir_, "allowed_dir");
  std::string outside_dir = tsl::io::JoinPath(base_dir_, "outside_dir");
  ASSERT_THAT(tsl::Env::Default()->RecursivelyCreateDir(allowed_dir), IsOk());
  ASSERT_THAT(tsl::Env::Default()->RecursivelyCreateDir(outside_dir), IsOk());

  XSpace space;
  space.add_planes()->set_name("test_plane");
  std::string first_xspace = tsl::io::JoinPath(allowed_dir, "first.xplane.pb");
  std::string second_xspace =
      tsl::io::JoinPath(allowed_dir, "second.xplane.pb");
  std::string third_xspace = tsl::io::JoinPath(allowed_dir, "third.xplane.pb");
  std::string outside_xplane =
      tsl::io::JoinPath(outside_dir, "outside.xplane.pb");
  ASSERT_THAT(xprof::WriteBinaryProto(first_xspace, space), IsOk());
  ASSERT_THAT(xprof::WriteBinaryProto(second_xspace, space), IsOk());
  ASSERT_THAT(xprof::WriteBinaryProto(third_xspace, space), IsOk());
  ASSERT_THAT(xprof::WriteBinaryProto(outside_xplane, space), IsOk());

  absl::Notification first_entered;
  absl::Notification release_first;
  std::atomic<bool> arm_next_block{true};
  std::atomic<int> map_calls{0};
  constexpr absl::string_view kBlockingToolName =
      "test_revalidation_blocking_tool";
  xprof::ProfileProcessorFactory::GetInstance().Register(
      kBlockingToolName, [&](const ToolOptions& /*options*/) {
        return std::make_unique<BlockingMapProcessor>(
            &first_entered, &release_first, &arm_next_block, &map_calls);
      });
  auto factory_cleanup = absl::MakeCleanup([kBlockingToolName] {
    xprof::ProfileProcessorFactory::GetInstance().Register(
        kBlockingToolName,
        [](const ToolOptions& /*options*/)
            -> std::unique_ptr<xprof::ProfileProcessor> { return nullptr; });
  });

  ProfileWorkerServiceImpl service(1, allowed_dir);
  std::atomic<int> queued_waiters{0};
  absl::Notification both_queued;
  service.SetQueueWaitCallbackForTesting([&] {
    if (queued_waiters.fetch_add(1, std::memory_order_acq_rel) + 1 == 2) {
      both_queued.Notify();
    }
  });

  ::xprof::pywrap::WorkerProfileDataResponse first_response;
  ::grpc::Status first_status;
  std::thread first_thread;
  ::xprof::pywrap::WorkerProfileDataResponse swapped_response;
  ::grpc::Status swapped_status;
  std::thread swapped_thread;
  ::xprof::pywrap::WorkerProfileDataResponse valid_queued_response;
  ::grpc::Status valid_queued_status;
  std::thread valid_queued_thread;

  auto threads_cleanup = absl::MakeCleanup([&] {
    if (!release_first.HasBeenNotified()) {
      release_first.Notify();
    }
    if (first_thread.joinable()) first_thread.join();
    if (swapped_thread.joinable()) swapped_thread.join();
    if (valid_queued_thread.joinable()) valid_queued_thread.join();
  });

  first_thread = std::thread([&] {
    first_status = CallWorkerGetProfileData(service, first_xspace,
                                            kBlockingToolName, &first_response);
  });
  ASSERT_TRUE(first_entered.WaitForNotificationWithTimeout(absl::Seconds(5)));

  swapped_thread = std::thread([&] {
    swapped_status = CallWorkerGetProfileData(
        service, second_xspace, kBlockingToolName, &swapped_response);
  });

  valid_queued_thread = std::thread([&] {
    valid_queued_status = CallWorkerGetProfileData(
        service, third_xspace, kBlockingToolName, &valid_queued_response);
  });

  ASSERT_TRUE(both_queued.WaitForNotificationWithTimeout(absl::Seconds(5)));
  std::string tmp_second_link =
      tsl::io::JoinPath(allowed_dir, "second_tmp.xplane.pb");
  ASSERT_EQ(::symlink(outside_xplane.c_str(), tmp_second_link.c_str()), 0);
  ASSERT_EQ(::rename(tmp_second_link.c_str(), second_xspace.c_str()), 0);

  release_first.Notify();
  first_thread.join();
  swapped_thread.join();
  valid_queued_thread.join();

  EXPECT_TRUE(first_status.ok()) << first_status.error_message();
  EXPECT_EQ(swapped_status.error_code(), ::grpc::StatusCode::PERMISSION_DENIED);
  EXPECT_THAT(swapped_response.output(), IsEmpty());
  EXPECT_TRUE(valid_queued_status.ok()) << valid_queued_status.error_message();
  EXPECT_THAT(valid_queued_response.output(),
              HasSubstr("third.xplane.pb.mapped"));
  EXPECT_EQ(map_calls.load(std::memory_order_acquire), 2);

  // Clearing the callback with an empty std::function must stop invoking the
  // prior callback on subsequent queued requests.
  service.SetQueueWaitCallbackForTesting({});
  EXPECT_EQ(queued_waiters.load(std::memory_order_acquire), 2);
#endif
}

class GrpcServerLoopbackTest : public testing::Test {
 protected:
  void SetUp() override {
    internal::ShutdownGrpcServerForTesting();
    internal::ResetWorkerAllowedRootDirsForTesting();
    const testing::TestInfo* const test_info =
        testing::UnitTest::GetInstance()->current_test_info();
    base_dir_ = tsl::io::JoinPath(
        testing::TempDir(), absl::StrCat("grpc_loopback_", test_info->name()));
    allowed_dir_ = tsl::io::JoinPath(base_dir_, "allowed_a");
    second_allowed_dir_ = tsl::io::JoinPath(base_dir_, "allowed_b");
    outside_dir_ = tsl::io::JoinPath(base_dir_, "outside");
    CHECK_OK(tsl::Env::Default()->RecursivelyCreateDir(allowed_dir_));
    CHECK_OK(tsl::Env::Default()->RecursivelyCreateDir(second_allowed_dir_));
    CHECK_OK(tsl::Env::Default()->RecursivelyCreateDir(outside_dir_));
  }

  void TearDown() override {
    internal::ShutdownGrpcServerForTesting();
    internal::ResetWorkerAllowedRootDirsForTesting();
    int64_t undeleted_files = 0;
    int64_t undeleted_dirs = 0;
    tsl::Env::Default()
        ->DeleteRecursively(base_dir_, &undeleted_files, &undeleted_dirs)
        .IgnoreError();
  }

  std::string base_dir_;
  std::string allowed_dir_;
  std::string second_allowed_dir_;
  std::string outside_dir_;
};

TEST_F(GrpcServerLoopbackTest,
       ServesLoopbackRequestsEnforcesMaxMessageSizeAndShutsDownCleanly) {
  XSpace space;
  space.add_planes()->set_name("test_plane");
  std::string allowed_xspace =
      tsl::io::JoinPath(allowed_dir_, "host1.xplane.pb");
  std::string second_xspace =
      tsl::io::JoinPath(second_allowed_dir_, "host2.xplane.pb");
  std::string outside_xspace =
      tsl::io::JoinPath(outside_dir_, "outside.xplane.pb");
  ASSERT_THAT(xprof::WriteBinaryProto(allowed_xspace, space), IsOk());
  ASSERT_THAT(xprof::WriteBinaryProto(second_xspace, space), IsOk());
  ASSERT_THAT(xprof::WriteBinaryProto(outside_xspace, space), IsOk());

  const int first_port =
      internal::InitializeGrpcServerForTesting(0, 1, allowed_dir_, "127.0.0.1");
  ASSERT_GT(first_port, 0);

  ::xprof::pywrap::WorkerProfileDataResponse response;
  ::grpc::Status allowed_status = CallStubGetProfileData(
      first_port, allowed_xspace, "overview_page", &response);
  EXPECT_TRUE(allowed_status.ok()) << allowed_status.error_message();
  EXPECT_THAT(response.output(), Not(IsEmpty()));

  EXPECT_EQ(CallStubGetProfileData(first_port, outside_xspace, "overview_page",
                                   &response)
                .error_code(),
            ::grpc::StatusCode::PERMISSION_DENIED);

  // Requests exceeding the 4 MiB receive limit must be rejected with
  // RESOURCE_EXHAUSTED.
  std::string oversized_payload(4 * 1024 * 1024 + 1024, 'x');
  EXPECT_EQ(CallStubGetProfileData(first_port, allowed_xspace, "overview_page",
                                   &response, oversized_payload)
                .error_code(),
            ::grpc::StatusCode::RESOURCE_EXHAUSTED);

  // Calling InitializeGrpcServer replaces and shuts down the previous server.
  InitializeGrpcServer(0, 1, second_allowed_dir_);
  EXPECT_EQ(CallStubGetProfileData(first_port, allowed_xspace, "overview_page",
                                   &response)
                .error_code(),
            ::grpc::StatusCode::UNAVAILABLE);

  const int second_port = internal::InitializeGrpcServerForTesting(
      0, 1, second_allowed_dir_, "127.0.0.1");
  ASSERT_GT(second_port, 0);
  ::grpc::Status second_status = CallStubGetProfileData(
      second_port, second_xspace, "overview_page", &response);
  EXPECT_TRUE(second_status.ok()) << second_status.error_message();
  EXPECT_EQ(CallStubGetProfileData(second_port, allowed_xspace, "overview_page",
                                   &response)
                .error_code(),
            ::grpc::StatusCode::PERMISSION_DENIED);

  // Invalid bind host fails cleanly with -1 and shuts down the prior server.
  EXPECT_EQ(internal::InitializeGrpcServerForTesting(0, 1, allowed_dir_,
                                                     "[invalid"),
            -1);
  EXPECT_EQ(CallStubGetProfileData(second_port, second_xspace, "overview_page",
                                   &response)
                .error_code(),
            ::grpc::StatusCode::UNAVAILABLE);

  // Explicit ShutdownGrpcServerForTesting stops an active server.
  const int third_port =
      internal::InitializeGrpcServerForTesting(0, 1, allowed_dir_, "127.0.0.1");
  ASSERT_GT(third_port, 0);
  ::grpc::Status third_status = CallStubGetProfileData(
      third_port, allowed_xspace, "overview_page", &response);
  EXPECT_TRUE(third_status.ok()) << third_status.error_message();
  internal::ShutdownGrpcServerForTesting();
  EXPECT_EQ(CallStubGetProfileData(third_port, allowed_xspace, "overview_page",
                                   &response)
                .error_code(),
            ::grpc::StatusCode::UNAVAILABLE);
}

}  // namespace
}  // namespace profiler
}  // namespace xprof
