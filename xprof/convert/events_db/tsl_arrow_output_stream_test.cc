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

#include "xprof/convert/events_db/tsl_arrow_output_stream.h"

#include <cstdint>
#include <memory>
#include <string>

#include "testing/base/public/gmock.h"
#include "<gtest/gtest.h>"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "arrow/result.h"  // from @arrow
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/file_system.h"

namespace xprof::events_db::internal {
namespace {

using ::absl_testing::StatusIs;
using ::testing::HasSubstr;

TEST(TslArrowOutputStreamTest, WriteFlushTellAndReadBack) {
  const std::string file_path =
      absl::StrCat(testing::TempDir(), "/test_output.bin");

  ASSERT_OK_AND_ASSIGN(std::shared_ptr<TslArrowOutputStream> stream,
                       TslArrowOutputStream::Open(file_path));
  ASSERT_NE(stream, nullptr);

  EXPECT_FALSE(stream->closed());
  EXPECT_EQ(stream->mode(),
            arrow::io::FileMode::WRITE);  // NOLINT(misc-include-cleaner)
  EXPECT_EQ(stream->Tell(), arrow::Result<int64_t>(0));

  const std::string payload1 = "Hello, ";
  ASSERT_TRUE(stream->Write(payload1.data(), payload1.size()).ok());
  EXPECT_EQ(stream->Tell(), arrow::Result<int64_t>(payload1.size()));

  const std::string payload2 = "Arrow Parquet on TSL!";
  ASSERT_TRUE(stream->Write(payload2.data(), payload2.size()).ok());
  EXPECT_EQ(stream->Tell(),
            arrow::Result<int64_t>(payload1.size() + payload2.size()));

  ASSERT_TRUE(stream->Flush().ok());
  ASSERT_TRUE(stream->Close().ok());
  EXPECT_TRUE(stream->closed());

  EXPECT_FALSE(stream->Write("extra", 5).ok());

  std::string contents;
  ASSERT_OK(tsl::ReadFileToString(tsl::Env::Default(), file_path, &contents));
  EXPECT_EQ(contents, "Hello, Arrow Parquet on TSL!");
}

TEST(TslArrowOutputStreamTest, OpenInvalidPathReturnsError) {
  EXPECT_THAT(TslArrowOutputStream::Open("/nonexistent_dir_12345/sub/test.bin"),
              StatusIs(absl::StatusCode::kNotFound));
}

class CustomTestEnv final : public tsl::EnvWrapper {
 public:
  CustomTestEnv() : tsl::EnvWrapper(tsl::Env::Default()) {}

  absl::Status GetFileSystemForFile(absl::string_view fname,
                                    tsl::FileSystem** result) override {
    return absl::UnimplementedError("custom env called");
  }
};

TEST(TslArrowOutputStreamTest, OpenWithCustomEnvUsesProvidedEnv) {
  CustomTestEnv custom_env;
  EXPECT_THAT(TslArrowOutputStream::Open("/some/test/path.bin", &custom_env),
              StatusIs(absl::StatusCode::kUnimplemented,
                       HasSubstr("custom env called")));
}

TEST(TslArrowOutputStreamTest, DestructorClosesFileAutomatically) {
  const std::string file_path =
      absl::StrCat(testing::TempDir(), "/auto_close.bin");
  {
    ASSERT_OK_AND_ASSIGN(std::shared_ptr<TslArrowOutputStream> stream,
                         TslArrowOutputStream::Open(file_path));
    ASSERT_TRUE(stream->Write("auto close", 10).ok());
    // Destructor runs here without explicit Close()
  }
  std::string contents;
  ASSERT_OK(tsl::ReadFileToString(tsl::Env::Default(), file_path, &contents));
  EXPECT_EQ(contents, "auto close");
}

class CloseTrackingWritableFile final : public tsl::WritableFile {
 public:
  explicit CloseTrackingWritableFile(int* close_count)
      : close_count_(close_count) {}

  absl::Status Append(absl::string_view data) override {
    return absl::OkStatus();
  }
  absl::Status Flush() override { return absl::OkStatus(); }
  absl::Status Close() override {
    if (close_count_ != nullptr) ++(*close_count_);
    return absl::OkStatus();
  }
  absl::Status Sync() override { return absl::OkStatus(); }

 private:
  int* close_count_;
};

TEST(TslArrowOutputStreamTest, DestructorClosesFileWhenNotExplicitlyClosed) {
  int close_count = 0;
  {
    const std::shared_ptr<TslArrowOutputStream> stream =
        std::make_shared<TslArrowOutputStream>(
            std::make_unique<CloseTrackingWritableFile>(&close_count));
    EXPECT_EQ(close_count, 0);
  }
  EXPECT_EQ(close_count, 1);
}

TEST(TslArrowOutputStreamTest, DestructorDoesNotCloseFileWhenAlreadyClosed) {
  int close_count = 0;
  {
    const std::shared_ptr<TslArrowOutputStream> stream =
        std::make_shared<TslArrowOutputStream>(
            std::make_unique<CloseTrackingWritableFile>(&close_count));
    ASSERT_TRUE(stream->Close().ok());
    EXPECT_EQ(close_count, 1);
  }
  EXPECT_EQ(close_count, 1);
}

class FailingWritableFile final : public tsl::WritableFile {
 public:
  absl::Status Append(absl::string_view data) override {
    return absl::InternalError("Write error");
  }
  absl::Status Flush() override { return absl::InternalError("Flush error"); }
  absl::Status Close() override { return absl::InternalError("Close error"); }
  absl::Status Sync() override { return absl::OkStatus(); }
};

TEST(TslArrowOutputStreamTest, ErrorPropagationAndEdgeCases) {
  const std::shared_ptr<TslArrowOutputStream> stream =
      std::make_shared<TslArrowOutputStream>(
          std::make_unique<FailingWritableFile>());
  EXPECT_TRUE(stream->Write(nullptr, 0).ok());
  EXPECT_FALSE(stream->Write("data", 4).ok());
  EXPECT_FALSE(stream->Flush().ok());
  EXPECT_FALSE(stream->Close().ok());
  EXPECT_TRUE(stream->closed());
  EXPECT_TRUE(stream->Close().ok());
  EXPECT_FALSE(stream->Write("data", 4).ok());
  EXPECT_FALSE(stream->Flush().ok());
  EXPECT_FALSE(stream->Tell().ok());
}

TEST(TslArrowOutputStreamTest, WriteNegativeNbytesReturnsError) {
  const std::string file_path =
      absl::StrCat(testing::TempDir(), "/negative_bytes_test.bin");
  ASSERT_OK_AND_ASSIGN(std::shared_ptr<TslArrowOutputStream> stream,
                       TslArrowOutputStream::Open(file_path));
  EXPECT_THAT(stream->Write("data", -1).message(), HasSubstr("negative"));
}

TEST(TslArrowOutputStreamTest, WriteNullDataPointerReturnsError) {
  const std::string file_path =
      absl::StrCat(testing::TempDir(), "/null_data_pointer_test.bin");
  ASSERT_OK_AND_ASSIGN(std::shared_ptr<TslArrowOutputStream> stream,
                       TslArrowOutputStream::Open(file_path));
  EXPECT_THAT(stream->Write(nullptr, 1).message(), HasSubstr("null data"));
}

}  // namespace
}  // namespace xprof::events_db::internal
