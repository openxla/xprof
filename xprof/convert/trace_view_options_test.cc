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

#include <string>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/statusor.h"
#include "xprof/convert/tool_options.h"

namespace tensorflow {
namespace profiler {
namespace {

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
  EXPECT_DOUBLE_EQ(trace_option->end_time_ms, 200.0);
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
  EXPECT_DOUBLE_EQ(trace_option->end_time_ms, 0.0);
  EXPECT_EQ(trace_option->resolution, 0);
  EXPECT_EQ(trace_option->event_name, "");
  EXPECT_EQ(trace_option->search_prefix, "");
  EXPECT_DOUBLE_EQ(trace_option->duration_ms, 0.0);
  EXPECT_EQ(trace_option->unique_id, 0);
  EXPECT_EQ(trace_option->format, "json");
  EXPECT_FALSE(trace_option->search_metadata);
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

}  // namespace
}  // namespace profiler
}  // namespace tensorflow
