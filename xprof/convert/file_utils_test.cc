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

#include <string>

#include "<gtest/gtest.h>"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/env.h"
#include "tsl/profiler/protobuf/xplane.pb.h"

namespace xprof {
namespace {

TEST(FileUtilsTest, WriteAndReadBinaryProto_Success) {
  std::string test_file = absl::StrCat(testing::TempDir(), "/test.xspace.pb");
  tensorflow::profiler::XSpace xspace;
  xspace.add_hostnames("test-host");

  TF_EXPECT_OK(WriteBinaryProto(test_file, xspace));

  tensorflow::profiler::XSpace xspace_read;
  TF_EXPECT_OK(ReadBinaryProto(test_file, &xspace_read));

  EXPECT_EQ(xspace_read.hostnames_size(), 1);
  EXPECT_EQ(xspace_read.hostnames(0), "test-host");
}

TEST(FileUtilsTest, ReadBinaryProto_InvalidProto) {
  std::string test_file = absl::StrCat(testing::TempDir(), "/invalid_proto");
  TF_EXPECT_OK(
      tsl::WriteStringToFile(tsl::Env::Default(), test_file, "invalid"));

  tensorflow::profiler::XSpace xspace;
  absl::Status status = ReadBinaryProto(test_file, &xspace);
  EXPECT_EQ(status.code(), absl::StatusCode::kDataLoss);
}

}  // namespace
}  // namespace xprof
