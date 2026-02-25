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
#include <cstddef>
#include <memory>
#include <string>
#include <utility>

#include "testing/base/public/gmock.h"
#include "<gtest/gtest.h>"
#include "absl/status/status.h"
#include "google/cloud/status_or.h"  // from @com_github_googlecloudplatform_google_cloud_cpp
#include "google/cloud/storage/client.h"  // from @com_github_googlecloudplatform_google_cloud_cpp
#include "google/cloud/storage/internal/http_response.h"  // from @com_github_googlecloudplatform_google_cloud_cpp
#include "google/cloud/storage/internal/object_read_source.h"  // from @com_github_googlecloudplatform_google_cloud_cpp
#include "google/cloud/storage/object_metadata.h"  // from @com_github_googlecloudplatform_google_cloud_cpp
#include "google/cloud/storage/testing/mock_client.h"  // from @com_github_googlecloudplatform_google_cloud_cpp
#include "xla/tsl/lib/core/status_test_util.h"
#include "tsl/profiler/protobuf/xplane.pb.h"

namespace xprof {
namespace {

using ::testing::_;
using ::testing::Eq;
using ::testing::Return;
using ::xprof::internal::ParseGcsPath;
using ::xprof::internal::ReadBinaryProtoWithClient;
namespace gcs = ::google::cloud::storage;
namespace gcs_testing = ::google::cloud::storage::testing;

TEST(FileUtilsTest, ParseGcsPath_GsPrefix) {
  std::string bucket, object;
  TF_EXPECT_OK(
      ParseGcsPath("gs://my-bucket/path/to/object.hlo", &bucket, &object));
  EXPECT_THAT(bucket, Eq("my-bucket"));
  EXPECT_THAT(object, Eq("path/to/object.hlo"));
}

TEST(FileUtilsTest, ParseGcsPath_BigstorePrefix) {
  std::string bucket, object;
  TF_EXPECT_OK(
      ParseGcsPath("/bigstore/my-bucket/path/to/object.hlo", &bucket, &object));
  EXPECT_THAT(bucket, Eq("my-bucket"));
  EXPECT_THAT(object, Eq("path/to/object.hlo"));
}

TEST(FileUtilsTest, ParseGcsPath_Invalid) {
  std::string bucket, object;
  EXPECT_FALSE(ParseGcsPath("s3://my-bucket/object", &bucket, &object).ok());
  EXPECT_FALSE(ParseGcsPath("gs://my-bucket", &bucket, &object).ok());
  EXPECT_FALSE(ParseGcsPath("gs:///object", &bucket, &object).ok());
}

TEST(FileUtilsTest, ReadBinaryProtoWithClient_Success) {
  auto mock = std::make_shared<gcs_testing::MockClient>();
  gcs::Client client = gcs_testing::UndecoratedClientFromMock(mock);

  std::string bucket = "bucket";
  std::string object = "object";
  std::string content = "XSpace content";

  gcs::ObjectMetadata metadata;
  metadata.set_size(content.size());

  EXPECT_CALL(*mock, GetObjectMetadata(_))
      .WillOnce(Return(google::cloud::StatusOr<gcs::ObjectMetadata>(metadata)));

  auto mock_source = std::make_unique<gcs_testing::MockObjectReadSource>();
  EXPECT_CALL(*mock_source, IsOpen()).WillRepeatedly(Return(true));
  EXPECT_CALL(*mock_source, Read(_, _))
      .WillOnce([content](char* buf, std::size_t n) {
        std::copy(content.begin(), content.end(), buf);
        return gcs::internal::ReadSourceResult{
            content.size(), gcs::internal::HttpResponse{200, "", {}}};
      });

  EXPECT_CALL(*mock, ReadObject(_))
      .WillOnce(Return(google::cloud::StatusOr<
                       std::unique_ptr<gcs::internal::ObjectReadSource>>(
          std::move(mock_source))));

  tensorflow::profiler::XSpace xspace;
  absl::Status status =
      ReadBinaryProtoWithClient(client, "gs://bucket/object", &xspace);
  // The ReadBinaryProtoWithClient function first reads the data and then tries
  // to parse it as a binary proto. Since "XSpace content" is not a valid
  // serialized proto, the parsing will fail with kDataLoss. This expectation
  // confirms that the download was successful and the code proceeded to the
  // parsing stage.
  EXPECT_THAT(status.code(), Eq(absl::StatusCode::kDataLoss));
}

}  // namespace
}  // namespace xprof
