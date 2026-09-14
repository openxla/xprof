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

#include "xprof/convert/events_db/python/pywrap_events_db_c_api.h"

#include <cstdint>
#include <fstream>
#include <ios>
#include <iterator>
#include <optional>
#include <string>
#include <string_view>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/algorithm/container.h"
#include "absl/strings/str_cat.h"
#include "arrow/util/compression.h"  // from @arrow
#include "arrow/util/type_fwd.h"  // from @arrow
#include "tsl/platform/path.h"
#include "tsl/platform/protobuf.h"
#include "tsl/profiler/protobuf/xplane.pb.h"
#include "xprof/convert/events_db/parquet_record_consumer.h"

namespace {

using ::testing::EndsWith;
using ::testing::HasSubstr;
using ::testing::NotNull;
using ::testing::StartsWith;
using ::xprof::events_db::ParquetExportOptions;

std::string CreateTempParquetPath(std::string_view filename) {
  return tsl::io::JoinPath(testing::TempDir(), filename);
}

void CreateTestXSpaceFile(std::string_view path) {
  constexpr std::string_view kXSpaceText = R"pb(
    planes {
      name: "/host:CPU"
      lines {
        id: 1
        name: "Thread1"
        events { metadata_id: 1 offset_ps: 1000000 duration_ps: 5000000 }
        events { metadata_id: 2 offset_ps: 3000000 duration_ps: 0 }
      }
      event_metadata {
        key: 1
        value { name: "test_step" }
      }
    }
  )pb";
  tensorflow::profiler::XSpace xspace;
  ASSERT_TRUE(tsl::protobuf::TextFormat::ParseFromString(kXSpaceText, &xspace));
  std::ofstream ofs(std::string(path), std::ios::binary);
  ASSERT_TRUE(xspace.SerializeToOstream(&ofs));
}

// ---------------------------------------------------------------------------
// Parametric Tests for Conversion Options
// ---------------------------------------------------------------------------

class XSpaceToParquetOptionsTest
    : public testing::TestWithParam<ParquetExportOptions> {};

TEST_P(XSpaceToParquetOptionsTest, ConvertsSuccessfullyWithParametricOptions) {
  const ParquetExportOptions& options = GetParam();
  std::string sanitized_test_name =
      testing::UnitTest::GetInstance()->current_test_info()->name();
  absl::c_replace(sanitized_test_name, '/', '_');
  const std::string input_path = CreateTempParquetPath(
      absl::StrCat("input_", sanitized_test_name, ".xplane.pb"));
  CreateTestXSpaceFile(input_path);
  const std::string output_path = CreateTempParquetPath(
      absl::StrCat("output_", sanitized_test_name, ".parquet"));

  const std::string codec_name =
      options.compression_type.has_value()
          ? arrow::util::Codec::GetCodecAsString(*options.compression_type)
          : "";

  char* error = nullptr;
  const bool success = XProfEventsDbXSpaceToParquet(
      input_path.c_str(), output_path.c_str(), options.batch_size.value_or(0),
      options.compression_type.has_value() ? codec_name.c_str() : nullptr,
      options.compression_level.value_or(-1),
      options.max_record_count.value_or(0), &error);

  EXPECT_TRUE(success);
  EXPECT_EQ(error, nullptr);
  XProfEventsDbFreeString(error);

  // Verify that the parquet output file exists and has valid magic bytes.
  std::ifstream file(output_path, std::ios::binary);
  ASSERT_TRUE(file.is_open());
  const std::string content((std::istreambuf_iterator<char>(file)),
                            std::istreambuf_iterator<char>());
  EXPECT_THAT(content, StartsWith("PAR1"));
  EXPECT_THAT(content, EndsWith("PAR1"));
  EXPECT_THAT(content, HasSubstr("test_step"));
}

std::string ParquetExportOptionsTestName(
    const testing::TestParamInfo<ParquetExportOptions>& info) {
  std::string name;
  if (info.param.compression_type.has_value()) {
    name += arrow::util::Codec::GetCodecAsString(*info.param.compression_type);
  } else {
    name += "DefaultCodec";
  }
  if (info.param.batch_size.has_value()) {
    absl::StrAppend(&name, "_Batch", *info.param.batch_size);
  } else {
    absl::StrAppend(&name, "_DefaultBatch");
  }
  if (info.param.compression_level.has_value()) {
    absl::StrAppend(&name, "_Level", *info.param.compression_level);
  } else {
    absl::StrAppend(&name, "_DefaultLevel");
  }
  if (info.param.max_record_count.has_value()) {
    absl::StrAppend(&name, "_MaxRecords", *info.param.max_record_count);
  } else {
    absl::StrAppend(&name, "_AllRecords");
  }
  return name;
}

INSTANTIATE_TEST_SUITE_P(
    AllOptionCombinations, XSpaceToParquetOptionsTest,
    testing::Values(
        // Default options (nullopt for max_records, compression_type, level,
        // and batch_size)
        ParquetExportOptions{
            /*max_record_count=*/std::nullopt,
            /*batch_size=*/std::nullopt,
            /*compression_type=*/std::nullopt,
            /*compression_level=*/std::nullopt,
        },
        // Standard default batch size (65536) with nullopt compression
        ParquetExportOptions{
            /*max_record_count=*/std::nullopt,
            /*batch_size=*/std::optional<uint32_t>(65536),
            /*compression_type=*/std::nullopt,
            /*compression_level=*/std::nullopt,
        },
        // Explicit SNAPPY compression with custom batch size
        ParquetExportOptions{
            /*max_record_count=*/std::nullopt,
            /*batch_size=*/std::optional<uint32_t>(512),
            /*compression_type=*/arrow::Compression::SNAPPY,
            /*compression_level=*/std::nullopt,
        },
        // Explicit ZSTD compression with default compression level
        ParquetExportOptions{
            /*max_record_count=*/std::nullopt,
            /*batch_size=*/std::optional<uint32_t>(1024),
            /*compression_type=*/arrow::Compression::ZSTD,
            /*compression_level=*/std::nullopt,
        },
        // Explicit ZSTD compression with explicit level and max record count
        // limit
        ParquetExportOptions{
            /*max_record_count=*/10,
            /*batch_size=*/std::optional<uint32_t>(512),
            /*compression_type=*/arrow::Compression::ZSTD,
            /*compression_level=*/3,
        },
        // Max record count early stopping test
        ParquetExportOptions{
            /*max_record_count=*/1,
            /*batch_size=*/std::optional<uint32_t>(65536),
            /*compression_type=*/arrow::Compression::SNAPPY,
            /*compression_level=*/std::nullopt,
        }),
    ParquetExportOptionsTestName);

TEST(XSpaceToParquetTest, EmptyCompressionStringTreatedAsDefault) {
  const std::string input_path =
      CreateTempParquetPath("input_empty_compression_string.xplane.pb");
  CreateTestXSpaceFile(input_path);
  const std::string output_path =
      CreateTempParquetPath("output_empty_compression_string.parquet");

  char* error = nullptr;
  const bool success = XProfEventsDbXSpaceToParquet(
      input_path.c_str(), output_path.c_str(), /*batch_size=*/0,
      /*compression_type=*/"", /*compression_level=*/-1,
      /*max_record_count=*/0, &error);

  EXPECT_TRUE(success);
  EXPECT_EQ(error, nullptr);
  XProfEventsDbFreeString(error);

  std::ifstream file(output_path, std::ios::binary);
  ASSERT_TRUE(file.is_open());
  const std::string content((std::istreambuf_iterator<char>(file)),
                            std::istreambuf_iterator<char>());
  EXPECT_THAT(content, StartsWith("PAR1"));
  EXPECT_THAT(content, EndsWith("PAR1"));
  EXPECT_THAT(content, HasSubstr("test_step"));
}

// ---------------------------------------------------------------------------
// Parametric Tests for Invalid Input / Output Paths
// ---------------------------------------------------------------------------

struct InvalidPathParams {
  const char* input_path;
  std::optional<std::string> output_path;
  std::string expected_substring;
  std::string test_name;
};

class XSpaceToParquetInvalidPathTest
    : public testing::TestWithParam<InvalidPathParams> {};

TEST_P(XSpaceToParquetInvalidPathTest, ReturnsExpectedError) {
  const InvalidPathParams& params = GetParam();

  std::string temp_output_path;
  const char* output_path = nullptr;
  if (params.output_path.has_value()) {
    if (params.output_path->empty()) {
      output_path = "";
    } else {
      temp_output_path = CreateTempParquetPath(*params.output_path);
      output_path = temp_output_path.c_str();
    }
  }

  char* error = nullptr;
  const bool success = XProfEventsDbXSpaceToParquet(
      params.input_path, output_path,
      /*batch_size=*/0, /*compression_type=*/nullptr,
      /*compression_level=*/-1, /*max_record_count=*/0, &error);

  EXPECT_FALSE(success);
  ASSERT_THAT(error, NotNull());
  EXPECT_THAT(error, HasSubstr(params.expected_substring));
  XProfEventsDbFreeString(error);
}

INSTANTIATE_TEST_SUITE_P(
    PathErrors, XSpaceToParquetInvalidPathTest,
    testing::Values(
        InvalidPathParams{
            /*input_path=*/nullptr,
            /*output_path=*/"null_input.parquet",
            /*expected_substring=*/"input_path must not be null or empty",
            /*test_name=*/"NullInputPath",
        },
        InvalidPathParams{
            /*input_path=*/"",
            /*output_path=*/"empty_input.parquet",
            /*expected_substring=*/"input_path must not be null or empty",
            /*test_name=*/"EmptyInputPath",
        },
        InvalidPathParams{
            /*input_path=*/"/valid/input.pb",
            /*output_path=*/std::nullopt,
            /*expected_substring=*/"output_path must not be null or empty",
            /*test_name=*/"NullOutputPath",
        },
        InvalidPathParams{
            /*input_path=*/"/valid/input.pb",
            /*output_path=*/"",
            /*expected_substring=*/"output_path must not be null or empty",
            /*test_name=*/"EmptyOutputPath",
        },
        InvalidPathParams{
            /*input_path=*/"/path/does/not/exist/missing.xplane.pb",
            /*output_path=*/"non_existent_input.parquet",
            /*expected_substring=*/"missing.xplane.pb",
            /*test_name=*/"NonExistentInputFile",
        }),
    [](const testing::TestParamInfo<InvalidPathParams>& info) {
      return info.param.test_name;
    });

// ---------------------------------------------------------------------------
// Tests for Invalid Codec & Memory Deallocation
// ---------------------------------------------------------------------------

TEST(XSpaceToParquetTest, InvalidCompressionTypeReturnsError) {
  const std::string input_path =
      CreateTempParquetPath("input_invalid_compression.xplane.pb");
  CreateTestXSpaceFile(input_path);
  const std::string output_path =
      CreateTempParquetPath("invalid_codec.parquet");

  char* error = nullptr;
  const bool success = XProfEventsDbXSpaceToParquet(
      input_path.c_str(), output_path.c_str(), /*batch_size=*/0,
      /*compression_type=*/"UNSUPPORTED_CODEC_XYZ",
      /*compression_level=*/-1, /*max_record_count=*/0, &error);

  EXPECT_FALSE(success);
  ASSERT_THAT(error, NotNull());
  EXPECT_THAT(error, HasSubstr("Invalid compression type"));
  XProfEventsDbFreeString(error);
}

TEST(XSpaceToParquetTest, CompressionLevelWithoutTypeReturnsError) {
  const std::string input_path =
      CreateTempParquetPath("input_compression_level.xplane.pb");
  CreateTestXSpaceFile(input_path);
  const std::string output_path =
      CreateTempParquetPath("compression_level_no_type.parquet");

  char* error = nullptr;
  const bool success = XProfEventsDbXSpaceToParquet(
      input_path.c_str(), output_path.c_str(), /*batch_size=*/0,
      /*compression_type=*/nullptr,
      /*compression_level=*/3, /*max_record_count=*/0, &error);

  EXPECT_FALSE(success);
  ASSERT_THAT(error, NotNull());
  EXPECT_THAT(
      error,
      HasSubstr("compression_level requires compression_type to be set"));
  XProfEventsDbFreeString(error);
}

TEST(XSpaceToParquetTest, NullOutErrorMessageSafelyIgnoredOnSuccess) {
  const std::string input_path =
      CreateTempParquetPath("input_null_out_error.xplane.pb");
  CreateTestXSpaceFile(input_path);
  const std::string output_path =
      CreateTempParquetPath("output_null_out_error.parquet");

  const bool success = XProfEventsDbXSpaceToParquet(
      input_path.c_str(), output_path.c_str(), /*batch_size=*/0,
      /*compression_type=*/nullptr, /*compression_level=*/-1,
      /*max_record_count=*/0, /*out_error_message=*/nullptr);

  EXPECT_TRUE(success);
}

TEST(XSpaceToParquetTest, NullOutErrorMessageSafelyIgnoredOnFailure) {
  const std::string output_path =
      CreateTempParquetPath("output_null_out_error_fail.parquet");

  const bool success = XProfEventsDbXSpaceToParquet(
      /*input_path=*/nullptr, output_path.c_str(), /*batch_size=*/0,
      /*compression_type=*/nullptr, /*compression_level=*/-1,
      /*max_record_count=*/0, /*out_error_message=*/nullptr);

  EXPECT_FALSE(success);
}

TEST(XSpaceToParquetTest, FreeStringHandlesNullSafely) {
  // FreeString(nullptr) should be a safe no-op like free(nullptr).
  XProfEventsDbFreeString(nullptr);
}

TEST(XSpaceToParquetTest, MaxRecordCountTruncatesOutput) {
  const std::string input_path =
      CreateTempParquetPath("truncate_input.xplane.pb");
  CreateTestXSpaceFile(input_path);

  const std::string output_path_1 =
      CreateTempParquetPath("output_max_record_count_1.parquet");
  char* error_1 = nullptr;
  ASSERT_TRUE(XProfEventsDbXSpaceToParquet(
      input_path.c_str(), output_path_1.c_str(), /*batch_size=*/0,
      /*compression_type=*/nullptr, /*compression_level=*/-1,
      /*max_record_count=*/1, &error_1));
  EXPECT_EQ(error_1, nullptr);
  XProfEventsDbFreeString(error_1);

  const std::string output_path_all =
      CreateTempParquetPath("output_max_record_count_all.parquet");
  char* error_all = nullptr;
  ASSERT_TRUE(XProfEventsDbXSpaceToParquet(
      input_path.c_str(), output_path_all.c_str(), /*batch_size=*/0,
      /*compression_type=*/nullptr, /*compression_level=*/-1,
      /*max_record_count=*/0, &error_all));
  EXPECT_EQ(error_all, nullptr);
  XProfEventsDbFreeString(error_all);

  std::ifstream file_1(output_path_1, std::ios::binary);
  ASSERT_TRUE(file_1.is_open());
  const std::string content_1((std::istreambuf_iterator<char>(file_1)),
                              std::istreambuf_iterator<char>());

  std::ifstream file_all(output_path_all, std::ios::binary);
  ASSERT_TRUE(file_all.is_open());
  const std::string content_all((std::istreambuf_iterator<char>(file_all)),
                                std::istreambuf_iterator<char>());

  EXPECT_THAT(content_1, StartsWith("PAR1"));
  EXPECT_THAT(content_1, EndsWith("PAR1"));
  EXPECT_THAT(content_1, HasSubstr("test_step"));

  EXPECT_THAT(content_all, StartsWith("PAR1"));
  EXPECT_THAT(content_all, EndsWith("PAR1"));
  EXPECT_THAT(content_all, HasSubstr("test_step"));

  EXPECT_LT(content_1.size(), content_all.size());
}

TEST(XSpaceToParquetTest, ZeroMaxRecordCountExportsAllRecords) {
  const std::string input_path =
      CreateTempParquetPath("all_records_input.xplane.pb");
  CreateTestXSpaceFile(input_path);

  const std::string output_path =
      CreateTempParquetPath("output_max_record_count_0.parquet");
  char* error = nullptr;
  const bool success = XProfEventsDbXSpaceToParquet(
      input_path.c_str(), output_path.c_str(), /*batch_size=*/0,
      /*compression_type=*/nullptr, /*compression_level=*/-1,
      /*max_record_count=*/0, &error);

  EXPECT_TRUE(success);
  EXPECT_EQ(error, nullptr) << "error: " << (error ? error : "");
  XProfEventsDbFreeString(error);

  std::ifstream file(output_path, std::ios::binary);
  ASSERT_TRUE(file.is_open());
  const std::string content((std::istreambuf_iterator<char>(file)),
                            std::istreambuf_iterator<char>());
  EXPECT_THAT(content, StartsWith("PAR1"));
  EXPECT_THAT(content, EndsWith("PAR1"));
  EXPECT_THAT(content, HasSubstr("test_step"));
}

}  // namespace
