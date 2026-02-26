/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

#include "xprof/convert/file_utils.h"

#include <algorithm>
#include <cstdint>
#include <string>

#include "testing/base/public/gmock.h"
#include "<gtest/gtest.h>"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "tsl/profiler/protobuf/xplane.pb.h"
#include "xprof/convert/file_utils_internal.h"
#include "xprof/convert/storage_client_interface.h"

namespace xprof {
namespace {

using ::testing::_;
using ::testing::Eq;
using ::testing::Return;
using ::xprof::internal::ParseGcsPath;
using ::xprof::internal::ReadBinaryProtoWithClient;
using ::xprof::internal::WriteBinaryProtoWithClient;

class MockStorageClient : public internal::StorageClientInterface {
 public:
  MOCK_METHOD(absl::StatusOr<std::uint64_t>, GetObjectSize,
              (const std::string& bucket, const std::string& object),
              (override));
  MOCK_METHOD(absl::Status, ReadObject,
              (const std::string& bucket, const std::string& object,
               std::uint64_t start, std::uint64_t end, char* buffer),
              (override));
  MOCK_METHOD(absl::Status, WriteObject,
              (const std::string& bucket, const std::string& object,
               const std::string& contents),
              (override));
};

TEST(FileUtilsTest, ParseGcsPath_GsPrefix) {
  std::string bucket;
  std::string object;
  TF_EXPECT_OK(
      ParseGcsPath("gs://my-bucket/path/to/object.hlo", &bucket, &object));
  EXPECT_THAT(bucket, Eq("my-bucket"));
  EXPECT_THAT(object, Eq("path/to/object.hlo"));
}

TEST(FileUtilsTest, ParseGcsPath_BigstorePrefix) {
  std::string bucket;
  std::string object;
  TF_EXPECT_OK(
      ParseGcsPath("/bigstore/my-bucket/path/to/object.hlo", &bucket, &object));
  EXPECT_THAT(bucket, Eq("my-bucket"));
  EXPECT_THAT(object, Eq("path/to/object.hlo"));
}

TEST(FileUtilsTest, ParseGcsPath_Invalid) {
  std::string bucket;
  std::string object;
  EXPECT_FALSE(ParseGcsPath("s3://my-bucket/object", &bucket, &object).ok());
  EXPECT_FALSE(ParseGcsPath("gs://my-bucket", &bucket, &object).ok());
  EXPECT_FALSE(ParseGcsPath("gs:///object", &bucket, &object).ok());
}

TEST(FileUtilsTest, ReadBinaryProtoWithClient_Success) {
  MockStorageClient client;
  constexpr absl::string_view kContent = "XSpace content";

  EXPECT_CALL(client, GetObjectSize("bucket", "object"))
      .WillOnce(Return(kContent.size()));

  EXPECT_CALL(client, ReadObject("bucket", "object", 0, kContent.size(), _))
      .WillOnce([kContent](const std::string&, const std::string&,
                           std::uint64_t, std::uint64_t, char* buf) {
        std::copy(kContent.begin(), kContent.end(), buf);
        return absl::OkStatus();
      });

  tensorflow::profiler::XSpace xspace;
  const absl::Status status =
      ReadBinaryProtoWithClient(client, "gs://bucket/object", &xspace);
  // Expect kDataLoss because "XSpace content" is not a valid serialized proto.
  EXPECT_THAT(status.code(), Eq(absl::StatusCode::kDataLoss));
}

TEST(FileUtilsTest, WriteBinaryProtoWithClient_Success) {
  MockStorageClient client;
  tensorflow::profiler::XSpace xspace;
  xspace.add_hostnames("test-host");

  std::string expected_contents;
  xspace.SerializeToString(&expected_contents);

  EXPECT_CALL(client, WriteObject("bucket", "object", expected_contents))
      .WillOnce(Return(absl::OkStatus()));

  TF_EXPECT_OK(
      WriteBinaryProtoWithClient(client, "gs://bucket/object", xspace));
}

}  // namespace
}  // namespace xprof
