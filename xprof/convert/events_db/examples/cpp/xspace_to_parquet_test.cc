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

#include <fstream>
#include <ios>
#include <iterator>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "file/base/path.h"
#include "testing/base/public/gmock.h"
#include "<gtest/gtest.h>"
#include "absl/strings/str_cat.h"
#include "xla/tsl/platform/subprocess.h"
#include "tsl/platform/protobuf.h"
#include "tsl/profiler/protobuf/xplane.pb.h"

namespace {

using ::testing::EndsWith;
using ::testing::HasSubstr;
using ::testing::StartsWith;

std::string GetBinaryPath() {
  constexpr std::string_view kRelativeBinaryPath =
      "org_xprof/xprof/convert/events_db/examples/cpp/xspace_to_parquet";
  return file::JoinPath(testing::SrcDir(), kRelativeBinaryPath);
}

std::string CreateTempFilePath(std::string_view filename) {
  return file::JoinPath(testing::TempDir(), filename);
}

void CreateEmptyFile(std::string_view path) {
  std::ofstream ofs(std::string(path), std::ios::binary);
}

void CreateTestXSpaceFile(std::string_view path) {
  constexpr std::string_view kXSpaceText = R"pb(
    planes {
      name: "/host:CPU"
      lines {
        id: 1
        name: "Thread1"
        events { metadata_id: 1 offset_ps: 1000000 duration_ps: 5000000 }
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

int RunCommand(const std::vector<std::string>& argv,
               std::string* stderr_output = nullptr,
               std::string* stdout_output = nullptr) {
  tsl::SubProcess proc;
  proc.SetProgram(argv[0], argv);
  if (stdout_output != nullptr) {
    proc.SetChannelAction(tsl::CHAN_STDOUT, tsl::ACTION_PIPE);
  } else {
    proc.SetChannelAction(tsl::CHAN_STDOUT, tsl::ACTION_CLOSE);
  }
  if (stderr_output != nullptr) {
    proc.SetChannelAction(tsl::CHAN_STDERR, tsl::ACTION_PIPE);
  } else {
    proc.SetChannelAction(tsl::CHAN_STDERR, tsl::ACTION_CLOSE);
  }
  if (!proc.Start()) return -1;
  return proc.Communicate(nullptr, stdout_output, stderr_output);
}

class CompressionTest
    : public testing::TestWithParam<std::optional<std::string>> {};

TEST_P(CompressionTest, ConvertsEmptyXSpace) {
  const std::string input_path = CreateTempFilePath("empty.xplane.pb");
  const std::string output_path = CreateTempFilePath("out.parquet");
  CreateEmptyFile(input_path);

  std::vector<std::string> args = {
      GetBinaryPath(), absl::StrCat("--input_path=", input_path),
      absl::StrCat("--output_path=", output_path), "--batch_size=512",
      "--max_record_count=100"};
  if (GetParam().has_value()) {
    args.push_back(absl::StrCat("--compression_type=", *GetParam()));
  }

  EXPECT_EQ(RunCommand(args), 0);

  // Verify that the generated parquet file is valid and contains Parquet magic
  // bytes.
  std::ifstream file(output_path, std::ios::binary);
  const std::string content((std::istreambuf_iterator<char>(file)),
                            std::istreambuf_iterator<char>());
  EXPECT_THAT(content, StartsWith("PAR1"));
  EXPECT_THAT(content, EndsWith("PAR1"));
}

INSTANTIATE_TEST_SUITE_P(
    CompressionTypes, CompressionTest,
    testing::Values<std::optional<std::string>>(std::nullopt, "", "SNAPPY"),
    [](const testing::TestParamInfo<std::optional<std::string>>& info)
        -> std::string {
      if (!info.param.has_value()) return "Default";
      if (info.param->empty()) return "ExplicitEmpty";
      return *info.param;
    });

TEST(XSpaceToParquetTest,
     ConvertXSpaceToParquetReturnsErrorIfInputPathIsEmpty) {
  const std::string output_path = CreateTempFilePath("out.parquet");
  EXPECT_NE(RunCommand({GetBinaryPath(), "--input_path=",
                        absl::StrCat("--output_path=", output_path)}),
            0);
}

TEST(XSpaceToParquetTest,
     ConvertXSpaceToParquetReturnsErrorIfOutputPathIsEmpty) {
  const std::string input_path = CreateTempFilePath("in.xplane.pb");
  CreateEmptyFile(input_path);
  EXPECT_NE(
      RunCommand({GetBinaryPath(), absl::StrCat("--input_path=", input_path),
                  "--output_path="}),
      0);
}

TEST(XSpaceToParquetTest, MissingInputPathReturnsError) {
  const std::string output_path = CreateTempFilePath("out.parquet");
  EXPECT_NE(RunCommand(
                {GetBinaryPath(), absl::StrCat("--output_path=", output_path)}),
            0);
}

TEST(XSpaceToParquetTest, MissingOutputPathReturnsError) {
  const std::string input_path = CreateTempFilePath("in.xplane.pb");
  CreateEmptyFile(input_path);
  EXPECT_NE(
      RunCommand({GetBinaryPath(), absl::StrCat("--input_path=", input_path)}),
      0);
}

TEST(XSpaceToParquetTest, TooManyCommandLineArgumentsReturnsError) {
  const std::string input_path = CreateTempFilePath("in.xplane.pb");
  const std::string output_path = CreateTempFilePath("out.parquet");
  CreateEmptyFile(input_path);
  EXPECT_NE(
      RunCommand({GetBinaryPath(), absl::StrCat("--input_path=", input_path),
                  absl::StrCat("--output_path=", output_path),
                  "unexpected_arg"}),
      0);
}

TEST(XSpaceToParquetTest, InvalidCompressionTypeReturnsError) {
  const std::string input_path = CreateTempFilePath("in.xplane.pb");
  const std::string output_path = CreateTempFilePath("out.parquet");
  CreateEmptyFile(input_path);
  EXPECT_NE(
      RunCommand({GetBinaryPath(), absl::StrCat("--input_path=", input_path),
                  absl::StrCat("--output_path=", output_path),
                  "--compression_type=INVALID_CODEC"}),
      0);
}

TEST(XSpaceToParquetTest, ConsumerBuildFailureReturnsError) {
  const std::string input_path = CreateTempFilePath("in.xplane.pb");
  const std::string output_path = CreateTempFilePath("out.parquet");
  CreateEmptyFile(input_path);

  std::string stderr_output;
  EXPECT_NE(RunCommand(
                {GetBinaryPath(), absl::StrCat("--input_path=", input_path),
                 absl::StrCat("--output_path=", output_path), "--batch_size=0"},
                &stderr_output),
            0);
  EXPECT_THAT(
      stderr_output,
      HasSubstr(
          "Conversion failed: INVALID_ARGUMENT: batch_size must be positive."));
}

TEST(XSpaceToParquetTest, ParseXSpaceFailureReturnsError) {
  const std::string input_path = CreateTempFilePath("nonexistent.xplane.pb");
  const std::string output_path = CreateTempFilePath("out.parquet");

  std::string stderr_output;
  EXPECT_NE(
      RunCommand({GetBinaryPath(), absl::StrCat("--input_path=", input_path),
                  absl::StrCat("--output_path=", output_path)},
                 &stderr_output),
      0);
  EXPECT_THAT(stderr_output, HasSubstr("Conversion failed: "));
}

TEST(XSpaceToParquetTest, MaxRecordCountStopsEarly) {
  const std::string input_path = CreateTempFilePath("in.xplane.pb");
  const std::string output_path = CreateTempFilePath("out.parquet");
  CreateTestXSpaceFile(input_path);

  std::string stdout_output;
  EXPECT_EQ(
      RunCommand(
          {GetBinaryPath(), absl::StrCat("--input_path=", input_path),
           absl::StrCat("--output_path=", output_path), "--max_record_count=0"},
          /*stderr_output=*/nullptr, &stdout_output),
      0);
  EXPECT_THAT(stdout_output, HasSubstr("ParseStatus.STOPPED_EARLY"));
}

TEST(XSpaceToParquetTest, SnappyWithCompressionLevelReturnsError) {
  const std::string input_path = CreateTempFilePath("in.xplane.pb");
  const std::string output_path = CreateTempFilePath("out.parquet");
  CreateTestXSpaceFile(input_path);

  std::string stderr_output;
  EXPECT_NE(
      RunCommand({GetBinaryPath(), absl::StrCat("--input_path=", input_path),
                  absl::StrCat("--output_path=", output_path),
                  "--compression_type=SNAPPY", "--compression_level=5"},
                 &stderr_output),
      0);
  EXPECT_THAT(
      stderr_output,
      HasSubstr("Codec 'snappy' doesn't support setting a compression level."));
}

}  // namespace
