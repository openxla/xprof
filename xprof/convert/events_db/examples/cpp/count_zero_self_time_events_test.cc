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
#include <string>
#include <string_view>
#include <vector>

#include "file/base/path.h"
#include "net/proto2/contrib/parse_proto/parse_text_proto.h"
#include "testing/base/public/gmock.h"
#include "<gtest/gtest.h>"
#include "absl/strings/str_cat.h"
#include "xla/tsl/platform/subprocess.h"
#include "tsl/profiler/protobuf/xplane.pb.h"

namespace {

using ::google::protobuf::contrib::parse_proto::ParseTextProtoOrDie;
using ::testing::HasSubstr;

std::string GetBinaryPath() {
  constexpr std::string_view kRelativeBinaryPath =
      "org_xprof/xprof/convert/events_db/examples/cpp/"
      "count_zero_self_time_events";
  return file::JoinPath(testing::SrcDir(), kRelativeBinaryPath);
}

std::string CreateTempFilePath(std::string_view filename) {
  return file::JoinPath(testing::TempDir(), filename);
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
  const tensorflow::profiler::XSpace xspace = ParseTextProtoOrDie(kXSpaceText);
  std::ofstream ofs(std::string(path), std::ios::binary);
  ASSERT_TRUE(xspace.SerializeToOstream(&ofs));
}

int RunCommand(const std::vector<std::string>& argv,
               std::string* stdout_output = nullptr,
               std::string* stderr_output = nullptr) {
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

TEST(CountZeroSelfTimeEventsTest, CountsZeroSelfTimeEventsWithTestData) {
  const std::string input_path = CreateTempFilePath("test.xplane.pb");
  CreateTestXSpaceFile(input_path);

  std::string stdout_output;
  EXPECT_EQ(
      RunCommand({GetBinaryPath(), absl::StrCat("--input_path=", input_path)},
                 &stdout_output),
      0);

  EXPECT_THAT(stdout_output,
              HasSubstr("Counting events with zero self_time_ns"));
  EXPECT_THAT(stdout_output, HasSubstr("Successfully finished parsing in"));
  EXPECT_THAT(stdout_output, HasSubstr("with parse status: COMPLETE"));
  EXPECT_THAT(stdout_output, HasSubstr("Total records processed: 2"));
  EXPECT_THAT(stdout_output, HasSubstr("Zero self_time_ns events: 1"));
}

TEST(CountZeroSelfTimeEventsTest, MissingInputPathReturnsError) {
  std::string stderr_output;
  EXPECT_NE(
      RunCommand({GetBinaryPath()}, /*stdout_output=*/nullptr, &stderr_output),
      0);
  EXPECT_THAT(stderr_output, HasSubstr("input_path cannot be empty."));
}

TEST(CountZeroSelfTimeEventsTest, EmptyInputPathReturnsError) {
  std::string stderr_output;
  EXPECT_NE(RunCommand({GetBinaryPath(), "--input_path="},
                       /*stdout_output=*/nullptr, &stderr_output),
            0);
  EXPECT_THAT(stderr_output, HasSubstr("input_path cannot be empty."));
}

TEST(CountZeroSelfTimeEventsTest, NonexistentInputPathReturnsError) {
  std::string stderr_output;
  EXPECT_NE(RunCommand({GetBinaryPath(),
                        "--input_path=/nonexistent/path/trace.xplane.pb"},
                       /*stdout_output=*/nullptr, &stderr_output),
            0);
  EXPECT_THAT(stderr_output, HasSubstr("Parsing failed: "));
}

TEST(CountZeroSelfTimeEventsTest, TooManyCommandLineArgumentsReturnsError) {
  const std::string input_path = CreateTempFilePath("test.xplane.pb");
  CreateTestXSpaceFile(input_path);

  std::string stderr_output;
  EXPECT_NE(
      RunCommand({GetBinaryPath(), absl::StrCat("--input_path=", input_path),
                  "unexpected_arg"},
                 /*stdout_output=*/nullptr, &stderr_output),
      0);
  EXPECT_THAT(stderr_output, HasSubstr("Too many command-line arguments."));
}

}  // namespace
