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

#include "xprof/convert/trace_view_options.h"

#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/file_system.h"
#include "xla/tsl/profiler/utils/timespan.h"
#include "xprof/convert/tool_options.h"
#include "xprof/convert/trace_viewer/trace_events.h"
#include "xprof/convert/trace_viewer/trace_options.h"
#include "xprof/convert/xplane_to_trace_container.h"
#include "plugin/xprof/protobuf/trace_events.pb.h"

namespace tensorflow {
namespace profiler {
namespace {

using ::absl_testing::StatusIs;
using ::testing::DoubleEq;
using ::testing::Optional;
using ::tsl::profiler::Timespan;

TEST(TraceViewOptionsTest, GetTraceViewOptionValidOptions) {
  // Arrange
  ToolOptions options;
  options["start_time_ms"] = "100.0";
  options["end_time_ms"] = "200.0";
  options["resolution"] = "1000";
  options["event_name"] = "test_event";
  options["search_prefix"] = "prefix";
  options["duration_ms"] = "50.0";
  options["unique_id"] = "123";
  options["format"] = "proto";
  options["search_metadata"] = true;

  // Act
  absl::StatusOr<TraceViewOption> trace_option = GetTraceViewOption(options);

  // Assert
  ASSERT_OK(trace_option);
  EXPECT_DOUBLE_EQ(trace_option->start_time_ms, 100.0);
  EXPECT_THAT(trace_option->end_time_ms, Optional(DoubleEq(200.0)));
  EXPECT_EQ(trace_option->resolution, 1000);
  EXPECT_EQ(trace_option->event_name, "test_event");
  EXPECT_EQ(trace_option->search_prefix, "prefix");
  EXPECT_DOUBLE_EQ(trace_option->duration_ms, 50.0);
  EXPECT_EQ(trace_option->unique_id, 123);
  EXPECT_EQ(trace_option->format, "proto");
  EXPECT_TRUE(trace_option->search_metadata);
}

TEST(TraceViewOptionsTest, GetTraceViewOptionDefaultOptions) {
  // Arrange
  ToolOptions options;

  // Act
  absl::StatusOr<TraceViewOption> trace_option = GetTraceViewOption(options);

  // Assert
  ASSERT_OK(trace_option);
  EXPECT_DOUBLE_EQ(trace_option->start_time_ms, 0.0);
  EXPECT_EQ(trace_option->end_time_ms, std::nullopt);
  EXPECT_EQ(trace_option->resolution, 0);
  EXPECT_EQ(trace_option->event_name, "");
  EXPECT_EQ(trace_option->search_prefix, "");
  EXPECT_DOUBLE_EQ(trace_option->duration_ms, 0.0);
  EXPECT_EQ(trace_option->unique_id, 0);
  EXPECT_EQ(trace_option->format, "json");
  EXPECT_FALSE(trace_option->search_metadata);
}

TEST(TraceViewOptionsTest, GetTraceViewOptionStartTimeOnly) {
  // Arrange
  ToolOptions options;
  options["start_time_ms"] = "100.0";

  // Act
  absl::StatusOr<TraceViewOption> trace_option = GetTraceViewOption(options);

  // Assert
  ASSERT_OK(trace_option);
  EXPECT_DOUBLE_EQ(trace_option->start_time_ms, 100.0);
  EXPECT_EQ(trace_option->end_time_ms, std::nullopt);
}

TEST(TraceViewOptionsTest, GetTraceViewOptionExplicitZeroEndTime) {
  // Arrange
  ToolOptions options;
  options["start_time_ms"] = "0.0";
  options["end_time_ms"] = "0.0";

  // Act
  absl::StatusOr<TraceViewOption> trace_option = GetTraceViewOption(options);

  // Assert
  ASSERT_OK(trace_option);
  EXPECT_DOUBLE_EQ(trace_option->start_time_ms, 0.0);
  EXPECT_THAT(trace_option->end_time_ms, Optional(DoubleEq(0.0)));
}

TEST(TraceViewOptionsTest, GetTraceViewOptionInvalidDouble) {
  // Arrange
  ToolOptions options;
  options["start_time_ms"] = "invalid";

  // Act
  absl::StatusOr<TraceViewOption> trace_option = GetTraceViewOption(options);

  // Assert
  EXPECT_FALSE(trace_option.ok());
}

TEST(TraceViewOptionsTest, GetTraceViewOptionInvalidInt) {
  // Arrange
  ToolOptions options;
  options["resolution"] = "invalid";

  // Act
  absl::StatusOr<TraceViewOption> trace_option = GetTraceViewOption(options);

  // Assert
  EXPECT_FALSE(trace_option.ok());
}

TEST(TraceViewOptionsTest, GetTraceViewOptionDoubleAsInt) {
  // Arrange
  ToolOptions options;
  options["resolution"] = "1000.0";

  // Act
  absl::StatusOr<TraceViewOption> trace_option = GetTraceViewOption(options);

  // Assert
  ASSERT_OK(trace_option);
  EXPECT_EQ(trace_option->resolution, 1000);
}

struct InvalidTimeBoundCase {
  std::string test_name;
  std::string value;
};

class GetTraceViewOptionInvalidTimeBoundTest
    : public ::testing::TestWithParam<InvalidTimeBoundCase> {};

TEST_P(GetTraceViewOptionInvalidTimeBoundTest, RejectsInvalidStartTimeMs) {
  ToolOptions options;
  options["start_time_ms"] = GetParam().value;

  EXPECT_THAT(GetTraceViewOption(options),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST_P(GetTraceViewOptionInvalidTimeBoundTest, RejectsInvalidEndTimeMs) {
  ToolOptions options;
  options["end_time_ms"] = GetParam().value;

  EXPECT_THAT(GetTraceViewOption(options),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

INSTANTIATE_TEST_SUITE_P(
    InvalidTimeBounds, GetTraceViewOptionInvalidTimeBoundTest,
    ::testing::Values(InvalidTimeBoundCase{"NegativeWhole", "-1.0"},
                      InvalidTimeBoundCase{"NegativeSubMillisecond", "-0.001"},
                      InvalidTimeBoundCase{"NaN", "nan"},
                      InvalidTimeBoundCase{"PositiveInfinity", "inf"},
                      InvalidTimeBoundCase{"NegativeInfinity", "-inf"},
                      InvalidTimeBoundCase{"LargeOverflow", "1e20"},
                      InvalidTimeBoundCase{"ExactUint64OverflowBoundary",
                                           "18446744073.709552"}),
    [](const ::testing::TestParamInfo<InvalidTimeBoundCase>& info) {
      return info.param.test_name;
    });

TEST(TraceViewOptionsTest, GetTraceViewOptionAcceptsMaxValidTimeMs) {
  ToolOptions options;
  options["start_time_ms"] = "0.0";
  options["end_time_ms"] = "18446744073.709545";

  absl::StatusOr<TraceViewOption> trace_option = GetTraceViewOption(options);

  ASSERT_OK(trace_option);
  EXPECT_DOUBLE_EQ(trace_option->start_time_ms, 0.0);
  EXPECT_THAT(trace_option->end_time_ms,
              Optional(DoubleEq(18446744073.709545)));
}

absl::StatusOr<TraceEventsLevelDbFilePaths> CreateTestTraceLevelDbFiles(
    absl::string_view file_prefix) {
  TraceEventsContainer input_container;
  Device* device = input_container.MutableDevice(1);
  device->set_device_id(1);
  device->set_name("TestDevice");
  Resource* resource = &(*device->mutable_resources())[1];
  resource->set_resource_id(1);
  resource->set_name("TestResource");

  // Event 1: [1.0 ms, 1.1 ms], Event 2: [1.2 ms, 1.25 ms].
  input_container.AddCompleteEvent(
      "Event1", /*resource_id=*/1, /*device_id=*/1,
      Timespan::FromEndPoints(1000000000ULL, 1100000000ULL));
  input_container.AddCompleteEvent(
      "Event2", /*resource_id=*/1, /*device_id=*/1,
      Timespan::FromEndPoints(1200000000ULL, 1250000000ULL));

  const std::string base = absl::StrCat(::testing::TempDir(), "/", file_prefix);
  TraceEventsLevelDbFilePaths file_paths = {
      .trace_events_file_path = absl::StrCat(base, "_events.ldb"),
      .trace_events_metadata_file_path = absl::StrCat(base, "_metadata.ldb"),
      .trace_events_prefix_trie_file_path = absl::StrCat(base, "_trie.ldb"),
  };

  std::unique_ptr<tsl::WritableFile> events_file;
  std::unique_ptr<tsl::WritableFile> metadata_file;
  std::unique_ptr<tsl::WritableFile> trie_file;
  if (absl::Status status = tsl::Env::Default()->NewWritableFile(
          file_paths.trace_events_file_path, &events_file);
      !status.ok()) {
    return status;
  }
  if (absl::Status status = tsl::Env::Default()->NewWritableFile(
          file_paths.trace_events_metadata_file_path, &metadata_file);
      !status.ok()) {
    return status;
  }
  if (absl::Status status = tsl::Env::Default()->NewWritableFile(
          file_paths.trace_events_prefix_trie_file_path, &trie_file);
      !status.ok()) {
    return status;
  }
  if (absl::Status status = input_container.StoreAsLevelDbTables(
          std::move(events_file), std::move(metadata_file),
          std::move(trie_file));
      !status.ok()) {
    return status;
  }
  return file_paths;
}

TEST(LoadTraceEventsContainerTest, LoadsEventsWithinValidWindow) {
  absl::StatusOr<TraceEventsLevelDbFilePaths> file_paths =
      CreateTestTraceLevelDbFiles("valid_window");
  ASSERT_OK(file_paths);

  TraceViewOption full_window;
  full_window.start_time_ms = 0.0;
  full_window.end_time_ms = 2.0;
  TraceEventsContainer loaded_full;
  ASSERT_OK(LoadTraceEventsContainer(*file_paths, full_window, TraceOptions(),
                                     &loaded_full));
  EXPECT_EQ(loaded_full.NumEvents(), 2);

  TraceViewOption open_ended_window;
  open_ended_window.start_time_ms = 1.15;
  TraceEventsContainer loaded_open_ended;
  ASSERT_OK(LoadTraceEventsContainer(*file_paths, open_ended_window,
                                     TraceOptions(), &loaded_open_ended));
  EXPECT_EQ(loaded_open_ended.NumEvents(), 1);
}

struct EmptyWindowLoadCase {
  std::string test_name;
  double start_time_ms;
  std::optional<double> end_time_ms;
};

class LoadTraceEventsContainerInvalidOrEmptyBoundsTest
    : public ::testing::TestWithParam<EmptyWindowLoadCase> {};

TEST_P(LoadTraceEventsContainerInvalidOrEmptyBoundsTest,
       ReturnsEmptyEventsWithoutOverflow) {
  const EmptyWindowLoadCase& test_case = GetParam();
  absl::StatusOr<TraceEventsLevelDbFilePaths> file_paths =
      CreateTestTraceLevelDbFiles(absl::StrCat("empty_", test_case.test_name));
  ASSERT_OK(file_paths);

  TraceViewOption trace_option;
  trace_option.start_time_ms = test_case.start_time_ms;
  trace_option.end_time_ms = test_case.end_time_ms;

  TraceEventsContainer loaded;
  ASSERT_OK(LoadTraceEventsContainer(*file_paths, trace_option, TraceOptions(),
                                     &loaded));
  EXPECT_EQ(loaded.NumEvents(), 0);
}

double Uint64OverflowBoundaryMs() {
  constexpr double kUint64Limit =
      static_cast<double>(std::numeric_limits<uint64_t>::max());
  double boundary = kUint64Limit / 1E9;
  if (boundary * 1E9 < kUint64Limit) {
    boundary =
        std::nextafter(boundary, std::numeric_limits<double>::infinity());
  }
  return boundary;
}

INSTANTIATE_TEST_SUITE_P(
    InvalidOrEmptyBounds, LoadTraceEventsContainerInvalidOrEmptyBoundsTest,
    ::testing::Values(
        EmptyWindowLoadCase{"InvertedBounds", 2.0, 1.0},
        EmptyWindowLoadCase{"NegativeStartWithEnd", -1.0, 2.0},
        EmptyWindowLoadCase{"NegativeStartOpenEnded", -1.0, std::nullopt},
        EmptyWindowLoadCase{"NegativeEndWithZeroStart", 0.0, -1.0},
        EmptyWindowLoadCase{"NaNStart",
                            std::numeric_limits<double>::quiet_NaN(), 2.0},
        EmptyWindowLoadCase{"NaNEnd", 0.0,
                            std::numeric_limits<double>::quiet_NaN()},
        EmptyWindowLoadCase{"PositiveInfStart",
                            std::numeric_limits<double>::infinity(),
                            std::nullopt},
        EmptyWindowLoadCase{"PositiveInfEnd", 0.0,
                            std::numeric_limits<double>::infinity()},
        EmptyWindowLoadCase{"NegativeInfStart",
                            -std::numeric_limits<double>::infinity(), 2.0},
        EmptyWindowLoadCase{"NegativeInfEnd", 0.0,
                            -std::numeric_limits<double>::infinity()},
        EmptyWindowLoadCase{"OverflowingStart", 1e20, std::nullopt},
        EmptyWindowLoadCase{"OverflowingEnd", 0.0, 1e20},
        EmptyWindowLoadCase{"ExactOverflowBoundaryStart",
                            Uint64OverflowBoundaryMs(), std::nullopt},
        EmptyWindowLoadCase{"ExactOverflowBoundaryEnd", 0.0,
                            Uint64OverflowBoundaryMs()}),
    [](const ::testing::TestParamInfo<EmptyWindowLoadCase>& info) {
      return info.param.test_name;
    });

}  // namespace
}  // namespace profiler
}  // namespace tensorflow
