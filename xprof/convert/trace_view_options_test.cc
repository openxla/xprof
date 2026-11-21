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
#include <optional>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/profiler/utils/timespan.h"
#include "xprof/convert/tool_options.h"
#include "xprof/convert/trace_viewer/trace_events.h"
#include "xprof/convert/trace_viewer/trace_events_test_utils.h"
#include "xprof/convert/trace_viewer/trace_options.h"
#include "xprof/convert/xplane_to_trace_container.h"
#include "plugin/xprof/protobuf/trace_events.pb.h"

namespace tensorflow {
namespace profiler {
namespace {

using ::absl_testing::StatusIs;
using ::testing::DoubleEq;
using ::testing::ElementsAre;
using ::testing::HasSubstr;
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
  EXPECT_THAT(trace_option,
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("start_time_ms must be a number")));
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

TEST(TraceViewOptionsTest, GetTraceViewOptionAcceptsZeroFloatUint64Options) {
  ToolOptions options;
  options["resolution"] = "0.0";
  options["unique_id"] = "0.0";

  absl::StatusOr<TraceViewOption> parsed = GetTraceViewOption(options);

  ASSERT_OK(parsed);
  EXPECT_EQ(parsed->resolution, 0);
  EXPECT_EQ(parsed->unique_id, 0);
}

TEST(TraceViewOptionsTest,
     GetTraceViewOptionAcceptsPositiveFloatUint64Options) {
  ToolOptions options;
  options["resolution"] = "100.5";
  options["unique_id"] = "42.0";

  absl::StatusOr<TraceViewOption> parsed = GetTraceViewOption(options);

  ASSERT_OK(parsed);
  EXPECT_EQ(parsed->resolution, 100);
  EXPECT_EQ(parsed->unique_id, 42);
}

TEST(TraceViewOptionsTest, GetTraceViewOptionAcceptsMaxUint64Options) {
  ToolOptions options;
  options["resolution"] = "18446744073709551615";
  options["unique_id"] = "18446744073709551615";

  absl::StatusOr<TraceViewOption> parsed = GetTraceViewOption(options);

  ASSERT_OK(parsed);
  EXPECT_EQ(parsed->resolution, std::numeric_limits<uint64_t>::max());
  EXPECT_EQ(parsed->unique_id, std::numeric_limits<uint64_t>::max());
}

struct ValidUniqueIdCase {
  std::string test_name;
  std::string value;
  uint64_t expected;
};

class GetTraceViewOptionValidUniqueIdTest
    : public testing::TestWithParam<ValidUniqueIdCase> {};

TEST_P(GetTraceViewOptionValidUniqueIdTest, AcceptsExactIntegralUniqueId) {
  ToolOptions options;
  options["unique_id"] = GetParam().value;

  absl::StatusOr<TraceViewOption> parsed = GetTraceViewOption(options);

  ASSERT_OK(parsed);
  EXPECT_EQ(parsed->unique_id, GetParam().expected);
}

INSTANTIATE_TEST_SUITE_P(
    ValidUniqueIds, GetTraceViewOptionValidUniqueIdTest,
    testing::Values(
        ValidUniqueIdCase{"ZeroFloat", "0.0", 0},
        ValidUniqueIdCase{"IntegralFloat", "7.0", 7},
        ValidUniqueIdCase{"LeadingPlusFloat", "+42.0", 42},
        ValidUniqueIdCase{"PositiveExponentIntegral", "1.5e1", 15},
        ValidUniqueIdCase{"NegativeExponentIntegral", "100e-2", 1},
        ValidUniqueIdCase{"ZeroHugeExponent", "0e99999999999999999999", 0},
        ValidUniqueIdCase{"AboveDoubleExactInt", "9007199254740993.0",
                          (uint64_t{1} << 53) + 1},
        ValidUniqueIdCase{"Uint64MaxFloat", "18446744073709551615.0",
                          std::numeric_limits<uint64_t>::max()}),
    [](const testing::TestParamInfo<ValidUniqueIdCase>& info) {
      return info.param.test_name;
    });

struct InvalidUniqueIdOnlyCase {
  std::string test_name;
  std::string value;
};

class GetTraceViewOptionInvalidUniqueIdOnlyTest
    : public testing::TestWithParam<InvalidUniqueIdOnlyCase> {};

TEST_P(GetTraceViewOptionInvalidUniqueIdOnlyTest,
       RejectsFractionalOrMalformedUniqueId) {
  ToolOptions options;
  options["unique_id"] = GetParam().value;

  EXPECT_THAT(
      GetTraceViewOption(options),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("unique_id")));
}

INSTANTIATE_TEST_SUITE_P(
    InvalidUniqueIds, GetTraceViewOptionInvalidUniqueIdOnlyTest,
    testing::Values(
        InvalidUniqueIdOnlyCase{"HalfFraction", "7.5"},
        InvalidUniqueIdOnlyCase{"SubDoubleUlpsFraction", "7.0000000000000001"},
        InvalidUniqueIdOnlyCase{"NegativeExponentFraction", "1e-1"},
        InvalidUniqueIdOnlyCase{"UnderflowExponentFraction", "1e-1000"},
        InvalidUniqueIdOnlyCase{"TrailingZerosStillFractional", "100e-3"},
        InvalidUniqueIdOnlyCase{"NegativeZeroFloat", "-0.0"},
        InvalidUniqueIdOnlyCase{"HexFloat", "0x1p4"},
        InvalidUniqueIdOnlyCase{"HugePositiveExponent",
                                "1e+999999999999999999999"},
        InvalidUniqueIdOnlyCase{"HugeNegativeExponent",
                                "1e-999999999999999999999"},
        InvalidUniqueIdOnlyCase{"Uint64OverflowFloat",
                                "18446744073709551616.0"},
        InvalidUniqueIdOnlyCase{"Empty", ""},
        InvalidUniqueIdOnlyCase{"DotOnly", "."},
        InvalidUniqueIdOnlyCase{"MissingExponentDigits", "1e"},
        InvalidUniqueIdOnlyCase{"MissingSignedExponentDigits", "1e+"}),
    [](const testing::TestParamInfo<InvalidUniqueIdOnlyCase>& info) {
      return info.param.test_name;
    });

struct InvalidUint64OptionCase {
  std::string test_name;
  std::string value;
};

class GetTraceViewOptionInvalidUint64OptionTest
    : public testing::TestWithParam<InvalidUint64OptionCase> {};

TEST_P(GetTraceViewOptionInvalidUint64OptionTest, RejectsInvalidResolution) {
  ToolOptions options;
  options["resolution"] = GetParam().value;

  EXPECT_THAT(GetTraceViewOption(options),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST_P(GetTraceViewOptionInvalidUint64OptionTest, RejectsInvalidUniqueId) {
  ToolOptions options;
  options["unique_id"] = GetParam().value;

  EXPECT_THAT(GetTraceViewOption(options),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

INSTANTIATE_TEST_SUITE_P(
    InvalidUint64Options, GetTraceViewOptionInvalidUint64OptionTest,
    testing::Values(InvalidUint64OptionCase{"NegativeInt", "-1"},
                    InvalidUint64OptionCase{"NegativeFloat", "-1.5"},
                    InvalidUint64OptionCase{"NaN", "nan"},
                    InvalidUint64OptionCase{"PositiveInfinity", "inf"},
                    InvalidUint64OptionCase{"NegativeInfinity", "-inf"},
                    InvalidUint64OptionCase{"LargeOverflow", "1e30"},
                    InvalidUint64OptionCase{"ExactUint64LimitFloat",
                                            "18446744073709551616.0"},
                    InvalidUint64OptionCase{"ExactUint64LimitInt",
                                            "18446744073709551616"}),
    [](const testing::TestParamInfo<InvalidUint64OptionCase>& info) {
      return info.param.test_name;
    });

struct InvalidTimeBoundCase {
  std::string test_name;
  std::string value;
};

class GetTraceViewOptionInvalidTimeBoundTest
    : public testing::TestWithParam<InvalidTimeBoundCase> {};

TEST_P(GetTraceViewOptionInvalidTimeBoundTest, RejectsInvalidStartTimeMs) {
  ToolOptions options;
  options["start_time_ms"] = GetParam().value;

  EXPECT_THAT(
      GetTraceViewOption(options),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("start_time_ms must be finite, non-negative")));
}

TEST_P(GetTraceViewOptionInvalidTimeBoundTest, RejectsInvalidEndTimeMs) {
  ToolOptions options;
  options["end_time_ms"] = GetParam().value;

  EXPECT_THAT(GetTraceViewOption(options),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("end_time_ms must be finite, non-negative")));
}

TEST_P(GetTraceViewOptionInvalidTimeBoundTest, RejectsInvalidDurationMs) {
  ToolOptions options;
  options["duration_ms"] = GetParam().value;

  EXPECT_THAT(GetTraceViewOption(options),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("duration_ms must be finite, non-negative")));
}

INSTANTIATE_TEST_SUITE_P(
    InvalidTimeBounds, GetTraceViewOptionInvalidTimeBoundTest,
    testing::Values(InvalidTimeBoundCase{"NegativeWhole", "-1.0"},
                    InvalidTimeBoundCase{"NegativeSubMillisecond", "-0.001"},
                    InvalidTimeBoundCase{"NaN", "nan"},
                    InvalidTimeBoundCase{"PositiveInfinity", "inf"},
                    InvalidTimeBoundCase{"NegativeInfinity", "-inf"},
                    InvalidTimeBoundCase{"LargeOverflow", "1e20"},
                    InvalidTimeBoundCase{"DoubleProductOverflow", "1e305"},
                    InvalidTimeBoundCase{"ExactUint64OverflowBoundary",
                                         "18446744073.709552"}),
    [](const testing::TestParamInfo<InvalidTimeBoundCase>& info) {
      return info.param.test_name;
    });

TEST(TraceViewOptionsTest, GetTraceViewOptionAcceptsMaxValidTimeMs) {
  ToolOptions options;
  options["start_time_ms"] = "0.0";
  options["end_time_ms"] = "18446744073.709545";
  options["duration_ms"] = "18446744073.709545";

  absl::StatusOr<TraceViewOption> trace_option = GetTraceViewOption(options);

  ASSERT_OK(trace_option);
  EXPECT_DOUBLE_EQ(trace_option->start_time_ms, 0.0);
  EXPECT_THAT(trace_option->end_time_ms,
              Optional(DoubleEq(18446744073.709545)));
  EXPECT_DOUBLE_EQ(trace_option->duration_ms, 18446744073.709545);
}

TEST(TraceViewOptionsTest,
     GetTraceViewOptionRejectsInt64OverflowBoundsForEventName) {
  {
    ToolOptions options;
    options["event_name"] = "Event1";
    options["start_time_ms"] = "10000000000.0";
    EXPECT_THAT(GetTraceViewOption(options),
                StatusIs(absl::StatusCode::kInvalidArgument,
                         HasSubstr("start_time_ms")));
  }
  {
    ToolOptions options;
    options["event_name"] = "Event1";
    options["start_time_ms"] = "9223372036.854776";
    EXPECT_THAT(GetTraceViewOption(options),
                StatusIs(absl::StatusCode::kInvalidArgument,
                         HasSubstr("start_time_ms")));
  }
  {
    ToolOptions options;
    options["event_name"] = "Event1";
    options["duration_ms"] = "10000000000.0";
    EXPECT_THAT(
        GetTraceViewOption(options),
        StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("duration_ms")));
  }
  {
    ToolOptions options;
    options["event_name"] = "Event1";
    options["unique_id"] = "9223372036854775808";
    EXPECT_THAT(
        GetTraceViewOption(options),
        StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("unique_id")));
  }
  {
    ToolOptions options;
    options["event_name"] = "Event1";
    options["unique_id"] = "18446744073709551615";
    EXPECT_THAT(
        GetTraceViewOption(options),
        StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("unique_id")));
  }
}

TEST(TraceViewOptionsTest,
     GetTraceViewOptionAcceptsMaxInt64BoundsForEventName) {
  ToolOptions options;
  options["event_name"] = "Event1";
  options["start_time_ms"] = "9223372036.854774";
  options["duration_ms"] = "9223372036.854774";
  options["unique_id"] = "9223372036854775807";

  absl::StatusOr<TraceViewOption> parsed = GetTraceViewOption(options);

  ASSERT_OK(parsed);
  EXPECT_DOUBLE_EQ(parsed->start_time_ms, 9223372036.854774);
  EXPECT_DOUBLE_EQ(parsed->duration_ms, 9223372036.854774);
  EXPECT_EQ(parsed->unique_id,
            static_cast<uint64_t>(std::numeric_limits<int64_t>::max()));
}

TEST(TraceViewOptionsTest,
     GetTraceViewOptionAllowsUint64BoundsWithoutEventName) {
  ToolOptions options;
  options["start_time_ms"] = "10000000000.0";
  options["end_time_ms"] = "18446744073.709545";
  options["duration_ms"] = "10000000000.0";
  options["unique_id"] = "18446744073709551615";

  absl::StatusOr<TraceViewOption> parsed = GetTraceViewOption(options);

  ASSERT_OK(parsed);
  EXPECT_DOUBLE_EQ(parsed->start_time_ms, 10000000000.0);
  EXPECT_THAT(parsed->end_time_ms, Optional(DoubleEq(18446744073.709545)));
  EXPECT_EQ(parsed->unique_id, std::numeric_limits<uint64_t>::max());
}

TraceEventsContainer MakeTestTraceContainer() {
  TraceEventsContainer container;
  Device* device = container.MutableDevice(1);
  device->set_device_id(1);
  device->set_name("TestDevice");
  Resource* resource = &(*device->mutable_resources())[1];
  resource->set_resource_id(1);
  resource->set_name("TestResource");
  return container;
}

absl::StatusOr<TraceEventsLevelDbFilePaths> CreateTestTraceLevelDbFiles(
    absl::string_view file_prefix) {
  TraceEventsContainer input_container = MakeTestTraceContainer();

  // Event 1: [1.0 ms, 1.1 ms], Event 2: [1.2 ms, 1.25 ms].
  input_container.AddCompleteEvent(
      "Event1", /*resource_id=*/1, /*device_id=*/1,
      Timespan::FromEndPoints(1000000000ULL, 1100000000ULL));
  input_container.AddCompleteEvent(
      "Event2", /*resource_id=*/1, /*device_id=*/1,
      Timespan::FromEndPoints(1200000000ULL, 1250000000ULL));

  return StoreTraceContainerToLevelDbTables(testing::TempDir(), file_prefix,
                                            input_container);
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

TEST(LoadTraceEventsContainerTest,
     ExactInstantAndSubPicosecondNonInvertedWindowsLoadOverlappingEvent) {
  absl::StatusOr<TraceEventsLevelDbFilePaths> file_paths =
      CreateTestTraceLevelDbFiles("exact_instant_window");
  ASSERT_OK(file_paths);

  auto collect_event_names = [](const TraceEventsContainer& container) {
    std::vector<std::string> names;
    container.ForAllEvents(
        [&](const TraceEvent& event) { names.push_back(event.name()); });
    return names;
  };

  // Exact-instant non-zero window [1.0 ms, 1.0 ms] overlaps only Event1.
  TraceViewOption exact_instant;
  exact_instant.start_time_ms = 1.0;
  exact_instant.end_time_ms = 1.0;
  TraceEventsContainer loaded_exact;
  ASSERT_OK(LoadTraceEventsContainer(*file_paths, exact_instant, TraceOptions(),
                                     &loaded_exact));
  EXPECT_EQ(loaded_exact.NumEvents(), 1);
  EXPECT_THAT(collect_event_names(loaded_exact), ElementsAre("Event1"));

  // Non-inverted sub-picosecond window (both truncate to 1000000000 ps, with
  // start_time_ms < end_time_ms) behaves as an exact-instant window at 1.0 ms.
  TraceViewOption sub_ps_non_inverted;
  sub_ps_non_inverted.start_time_ms = 1.0;
  sub_ps_non_inverted.end_time_ms = 1.0 + 1e-10;
  TraceEventsContainer loaded_sub_ps;
  ASSERT_OK(LoadTraceEventsContainer(*file_paths, sub_ps_non_inverted,
                                     TraceOptions(), &loaded_sub_ps));
  EXPECT_EQ(loaded_sub_ps.NumEvents(), 1);
  EXPECT_THAT(collect_event_names(loaded_sub_ps), ElementsAre("Event1"));

  // Exact-instant window [1.2 ms, 1.2 ms] scans earlier Event1 [1.0, 1.1] ms
  // from LevelDB and must exclude it, loading only Event2.
  TraceViewOption late_exact_instant;
  late_exact_instant.start_time_ms = 1.2;
  late_exact_instant.end_time_ms = 1.2;
  TraceEventsContainer loaded_late_exact;
  ASSERT_OK(LoadTraceEventsContainer(*file_paths, late_exact_instant,
                                     TraceOptions(), &loaded_late_exact));
  EXPECT_EQ(loaded_late_exact.NumEvents(), 1);
  EXPECT_THAT(collect_event_names(loaded_late_exact), ElementsAre("Event2"));

  // Non-inverted sub-picosecond window [1.2 ms, 1.2 + 1e-10 ms] also excludes
  // earlier Event1 and loads only Event2.
  TraceViewOption late_sub_ps_non_inverted;
  late_sub_ps_non_inverted.start_time_ms = 1.2;
  late_sub_ps_non_inverted.end_time_ms = 1.2 + 1e-10;
  TraceEventsContainer loaded_late_sub_ps;
  ASSERT_OK(LoadTraceEventsContainer(*file_paths, late_sub_ps_non_inverted,
                                     TraceOptions(), &loaded_late_sub_ps));
  EXPECT_EQ(loaded_late_sub_ps.NumEvents(), 1);
  EXPECT_THAT(collect_event_names(loaded_late_sub_ps), ElementsAre("Event2"));
}

struct InvertedWindowBoundsCase {
  std::string test_name;
  double start_time_ms;
  double end_time_ms;
};

class LoadTraceEventsContainerInvertedBoundsTest
    : public testing::TestWithParam<InvertedWindowBoundsCase> {};

TEST_P(LoadTraceEventsContainerInvertedBoundsTest,
       ReturnsEmptyEventsWhilePreservingMetadata) {
  const InvertedWindowBoundsCase& test_case = GetParam();
  absl::StatusOr<TraceEventsLevelDbFilePaths> file_paths =
      CreateTestTraceLevelDbFiles(
          absl::StrCat("inverted_", test_case.test_name));
  ASSERT_OK(file_paths);

  TraceViewOption option;
  option.start_time_ms = test_case.start_time_ms;
  option.end_time_ms = test_case.end_time_ms;

  TraceEventsContainer loaded;
  EXPECT_OK(
      LoadTraceEventsContainer(*file_paths, option, TraceOptions(), &loaded));
  EXPECT_EQ(loaded.NumEvents(), 0);
  EXPECT_EQ(loaded.trace().min_timestamp_ps(), 1000000000ULL);
  EXPECT_EQ(loaded.trace().max_timestamp_ps(), 1250000000ULL);
  ASSERT_TRUE(loaded.trace().devices().contains(1));
  EXPECT_EQ(loaded.trace().devices().at(1).name(), "TestDevice");
}

INSTANTIATE_TEST_SUITE_P(
    InvertedWindowBounds, LoadTraceEventsContainerInvertedBoundsTest,
    testing::Values(InvertedWindowBoundsCase{"Macroscopic", 2.0, 1.0},
                    // Both truncate to 1000000000 ps in MilliToPico, but
                    // start_time_ms > end_time_ms before truncation.
                    InvertedWindowBoundsCase{"SubPicosecond", 1.0 + 1e-10,
                                             1.0}),
    [](const testing::TestParamInfo<InvertedWindowBoundsCase>& info) {
      return info.param.test_name;
    });

struct InvalidWindowBoundsCase {
  std::string test_name;
  double start_time_ms;
  std::optional<double> end_time_ms;
  std::string expected_option_name;
};

class LoadTraceEventsContainerInvalidBoundsTest
    : public testing::TestWithParam<InvalidWindowBoundsCase> {};

TEST_P(LoadTraceEventsContainerInvalidBoundsTest,
       RejectsMalformedOrOverflowingBoundsWithoutOverflow) {
  const InvalidWindowBoundsCase& test_case = GetParam();
  TraceViewOption trace_option;
  trace_option.start_time_ms = test_case.start_time_ms;
  trace_option.end_time_ms = test_case.end_time_ms;

  TraceEventsContainer loaded;
  EXPECT_THAT(LoadTraceEventsContainer(TraceEventsLevelDbFilePaths{},
                                       trace_option, TraceOptions(), &loaded),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr(test_case.expected_option_name)));
}

template <typename IntT>
double OverflowBoundaryMs() {
  constexpr double kLimit =
      static_cast<double>(std::numeric_limits<IntT>::max());
  double boundary = kLimit / 1E9;
  if (boundary * 1E9 < kLimit) {
    boundary =
        std::nextafter(boundary, std::numeric_limits<double>::infinity());
  }
  return boundary;
}

INSTANTIATE_TEST_SUITE_P(
    InvalidWindowBounds, LoadTraceEventsContainerInvalidBoundsTest,
    testing::Values(
        InvalidWindowBoundsCase{"NegativeStartWithEnd", -1.0, 2.0,
                                "start_time_ms"},
        InvalidWindowBoundsCase{"NegativeStartOpenEnded", -1.0, std::nullopt,
                                "start_time_ms"},
        InvalidWindowBoundsCase{"NegativeEndWithZeroStart", 0.0, -1.0,
                                "end_time_ms"},
        InvalidWindowBoundsCase{"NaNStart",
                                std::numeric_limits<double>::quiet_NaN(), 2.0,
                                "start_time_ms"},
        InvalidWindowBoundsCase{"NaNEnd", 0.0,
                                std::numeric_limits<double>::quiet_NaN(),
                                "end_time_ms"},
        InvalidWindowBoundsCase{"PositiveInfStart",
                                std::numeric_limits<double>::infinity(),
                                std::nullopt, "start_time_ms"},
        InvalidWindowBoundsCase{"PositiveInfEnd", 0.0,
                                std::numeric_limits<double>::infinity(),
                                "end_time_ms"},
        InvalidWindowBoundsCase{"NegativeInfStart",
                                -std::numeric_limits<double>::infinity(), 2.0,
                                "start_time_ms"},
        InvalidWindowBoundsCase{"NegativeInfEnd", 0.0,
                                -std::numeric_limits<double>::infinity(),
                                "end_time_ms"},
        InvalidWindowBoundsCase{"OverflowingStart", 1e20, std::nullopt,
                                "start_time_ms"},
        InvalidWindowBoundsCase{"OverflowingEnd", 0.0, 1e20, "end_time_ms"},
        InvalidWindowBoundsCase{"DoubleProductOverflowStart", 1e305,
                                std::nullopt, "start_time_ms"},
        InvalidWindowBoundsCase{"DoubleProductOverflowEnd", 0.0, 1e305,
                                "end_time_ms"},
        InvalidWindowBoundsCase{"ExactOverflowBoundaryStart",
                                OverflowBoundaryMs<uint64_t>(), std::nullopt,
                                "start_time_ms"},
        InvalidWindowBoundsCase{"ExactOverflowBoundaryEnd", 0.0,
                                OverflowBoundaryMs<uint64_t>(), "end_time_ms"}),
    [](const testing::TestParamInfo<InvalidWindowBoundsCase>& info) {
      return info.param.test_name;
    });

TEST(LoadTraceEventsContainerTest,
     LoadsFullEventWithValidStartTimeAndDuration) {
  absl::StatusOr<TraceEventsLevelDbFilePaths> file_paths =
      CreateTestTraceLevelDbFiles("valid_full_event");
  ASSERT_OK(file_paths);

  TraceViewOption event_option;
  event_option.event_name = "Event1";
  event_option.start_time_ms = 1.0;
  event_option.duration_ms = 0.1;

  TraceEventsContainer loaded;
  ASSERT_OK(LoadTraceEventsContainer(*file_paths, event_option, TraceOptions(),
                                     &loaded));
  EXPECT_EQ(loaded.NumEvents(), 1);

  // Sub-picosecond inputs just below Event1's [1.0 ms, +0.1 ms] key must round
  // up to [1000000000, 100000000] ps rather than truncate.
  TraceViewOption rounded_event_option;
  rounded_event_option.event_name = "Event1";
  rounded_event_option.start_time_ms = 1.0 - 1e-10;
  rounded_event_option.duration_ms = 0.1 - 1e-10;

  TraceEventsContainer loaded_rounded;
  ASSERT_OK(LoadTraceEventsContainer(*file_paths, rounded_event_option,
                                     TraceOptions(), &loaded_rounded));
  EXPECT_EQ(loaded_rounded.NumEvents(), 1);

  // Verify the zero lower boundary for both start_time_ms and duration_ms.
  TraceEventsContainer zero_duration_input = MakeTestTraceContainer();
  zero_duration_input.AddCompleteEvent("InstantEvent", /*resource_id=*/1,
                                       /*device_id=*/1,
                                       Timespan::FromEndPoints(0ULL, 0ULL));
  absl::StatusOr<TraceEventsLevelDbFilePaths> zero_duration_paths =
      StoreTraceContainerToLevelDbTables(
          testing::TempDir(), "zero_duration_event", zero_duration_input);
  ASSERT_OK(zero_duration_paths);

  TraceViewOption zero_duration_option;
  zero_duration_option.event_name = "InstantEvent";
  zero_duration_option.start_time_ms = 0.0;
  zero_duration_option.duration_ms = 0.0;

  TraceEventsContainer loaded_zero_duration;
  EXPECT_OK(LoadTraceEventsContainer(*zero_duration_paths, zero_duration_option,
                                     TraceOptions(), &loaded_zero_duration));
  EXPECT_EQ(loaded_zero_duration.NumEvents(), 1);
}

struct InvalidEventLookupCase {
  std::string test_name;
  double start_time_ms;
  double duration_ms;
  uint64_t unique_id;
  std::string expected_option_name;
};

double MaxValidInt64TimeMs() {
  return std::nextafter(OverflowBoundaryMs<int64_t>(), 0.0);
}

TEST(LoadTraceEventsContainerTest,
     AcceptsFullEventLookupAtSignedInt64UpperBoundaries) {
  const double max_valid_ms = MaxValidInt64TimeMs();
  const uint64_t max_valid_ps =
      static_cast<uint64_t>(std::round(max_valid_ms * 1E9));

  TraceEventsContainer boundary_input = MakeTestTraceContainer();
  boundary_input.AddCompleteEvent(
      "BoundaryEvent", /*resource_id=*/1,
      /*device_id=*/1, Timespan::FromEndPoints(max_valid_ps, max_valid_ps));
  absl::StatusOr<TraceEventsLevelDbFilePaths> file_paths =
      StoreTraceContainerToLevelDbTables(
          testing::TempDir(), "int64_boundary_event", boundary_input);
  ASSERT_OK(file_paths);

  TraceViewOption exact_boundary_option;
  exact_boundary_option.event_name = "BoundaryEvent";
  exact_boundary_option.start_time_ms = max_valid_ms;
  exact_boundary_option.duration_ms = 0.0;
  exact_boundary_option.unique_id = 0;

  TraceEventsContainer loaded_boundary;
  ASSERT_OK(LoadTraceEventsContainer(*file_paths, exact_boundary_option,
                                     TraceOptions(), &loaded_boundary));
  EXPECT_EQ(loaded_boundary.NumEvents(), 1);

  // Max valid `start_time_ms`, `duration_ms`, and `unique_id == INT64_MAX` pass
  // argument validation without overflow and reach LevelDB lookup (`NotFound`).
  TraceViewOption max_int64_option;
  max_int64_option.event_name = "MissingEvent";
  max_int64_option.start_time_ms = max_valid_ms;
  max_int64_option.duration_ms = max_valid_ms;
  max_int64_option.unique_id =
      static_cast<uint64_t>(std::numeric_limits<int64_t>::max());

  TraceEventsContainer loaded_missing;
  EXPECT_THAT(LoadTraceEventsContainer(*file_paths, max_int64_option,
                                       TraceOptions(), &loaded_missing),
              StatusIs(absl::StatusCode::kNotFound));
}

class LoadTraceEventsContainerInvalidEventLookupTest
    : public testing::TestWithParam<InvalidEventLookupCase> {};

TEST_P(LoadTraceEventsContainerInvalidEventLookupTest,
       RejectsInvalidEventLookupArgumentsWithoutOverflow) {
  const InvalidEventLookupCase& test_case = GetParam();
  TraceViewOption trace_option;
  trace_option.event_name = "Event1";
  trace_option.start_time_ms = test_case.start_time_ms;
  trace_option.duration_ms = test_case.duration_ms;
  trace_option.unique_id = test_case.unique_id;

  TraceEventsContainer loaded;
  EXPECT_THAT(LoadTraceEventsContainer(TraceEventsLevelDbFilePaths{},
                                       trace_option, TraceOptions(), &loaded),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr(test_case.expected_option_name)));
}

INSTANTIATE_TEST_SUITE_P(
    InvalidEventLookups, LoadTraceEventsContainerInvalidEventLookupTest,
    testing::Values(
        InvalidEventLookupCase{"NegativeStartTime", -1.0, 0.1, 0,
                               "start_time_ms"},
        InvalidEventLookupCase{"NegativeDuration", 1.0, -0.1, 0, "duration_ms"},
        InvalidEventLookupCase{"NaNStartTime",
                               std::numeric_limits<double>::quiet_NaN(), 0.1, 0,
                               "start_time_ms"},
        InvalidEventLookupCase{"NaNDuration", 1.0,
                               std::numeric_limits<double>::quiet_NaN(), 0,
                               "duration_ms"},
        InvalidEventLookupCase{"PositiveInfStartTime",
                               std::numeric_limits<double>::infinity(), 0.1, 0,
                               "start_time_ms"},
        InvalidEventLookupCase{"PositiveInfDuration", 1.0,
                               std::numeric_limits<double>::infinity(), 0,
                               "duration_ms"},
        InvalidEventLookupCase{"NegativeInfStartTime",
                               -std::numeric_limits<double>::infinity(), 0.1, 0,
                               "start_time_ms"},
        InvalidEventLookupCase{"NegativeInfDuration", 1.0,
                               -std::numeric_limits<double>::infinity(), 0,
                               "duration_ms"},
        InvalidEventLookupCase{"OverflowingStartTime", 1e20, 0.1, 0,
                               "start_time_ms"},
        InvalidEventLookupCase{"OverflowingDuration", 1.0, 1e20, 0,
                               "duration_ms"},
        InvalidEventLookupCase{"DoubleProductOverflowStartTime", 1e305, 0.1, 0,
                               "start_time_ms"},
        InvalidEventLookupCase{"DoubleProductOverflowDuration", 1.0, 1e305, 0,
                               "duration_ms"},
        InvalidEventLookupCase{"ExactUint64OverflowBoundaryStartTime",
                               OverflowBoundaryMs<uint64_t>(), 0.1, 0,
                               "start_time_ms"},
        InvalidEventLookupCase{"ExactUint64OverflowBoundaryDuration", 1.0,
                               OverflowBoundaryMs<uint64_t>(), 0,
                               "duration_ms"},
        InvalidEventLookupCase{"BetweenInt64AndUint64StartTime", 10000000000.0,
                               0.1, 0, "start_time_ms"},
        InvalidEventLookupCase{"BetweenInt64AndUint64Duration", 1.0,
                               10000000000.0, 0, "duration_ms"},
        InvalidEventLookupCase{"ExactInt64OverflowBoundaryStartTime",
                               OverflowBoundaryMs<int64_t>(), 0.1, 0,
                               "start_time_ms"},
        InvalidEventLookupCase{"ExactInt64OverflowBoundaryDuration", 1.0,
                               OverflowBoundaryMs<int64_t>(), 0, "duration_ms"},
        InvalidEventLookupCase{
            "UniqueIdExceedsInt64MaxByOne", 1.0, 0.1,
            static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) + 1,
            "unique_id"},
        InvalidEventLookupCase{"UniqueIdUint64Max", 1.0, 0.1,
                               std::numeric_limits<uint64_t>::max(),
                               "unique_id"}),
    [](const testing::TestParamInfo<InvalidEventLookupCase>& info) {
      return info.param.test_name;
    });

}  // namespace
}  // namespace profiler
}  // namespace tensorflow
