/* Copyright 2026 The OpenXLA Authors. All Rights Reserved.

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

#include "xprof/convert/grpc_metrics/json_utils.h"

#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "absl/strings/string_view.h"

namespace xprof::grpc_metrics {
namespace {

// This test suite verifies the helper functions in json_utils.h
// which are used to parse JSON fragments from gRPC metrics.

TEST(JsonUtilsTest, ExtractJsonValue) {
  // Test extracting string values.
  const absl::string_view json1 = R"({"key": "value"})";
  EXPECT_EQ(ExtractJsonValue(json1, "key"), "value");

  // Test extracting numeric values as raw strings.
  const absl::string_view json2 = R"({"key": 123})";
  EXPECT_EQ(ExtractJsonValue(json2, "key"), "123");

  // Test extracting arrays.
  const absl::string_view json3 = R"({"key": [1, 2, 3]})";
  EXPECT_EQ(ExtractJsonValue(json3, "key"), "[1, 2, 3]");

  // Test missing key.
  const absl::string_view json4 = R"({"other": "value"})";
  EXPECT_EQ(ExtractJsonValue(json4, "key"), "");

  // Test key as prefix (should not match).
  const absl::string_view json5 = R"({"key_suffix": "value"})";
  EXPECT_EQ(ExtractJsonValue(json5, "key"), "");

  // Test key with underscore suffix matching.
  const absl::string_view json6 = R"({"key_": "value"})";
  EXPECT_EQ(ExtractJsonValue(json6, "key"), "value");
}

TEST(JsonUtilsTest, ParseJsonArray) {
  // Test standard array.
  const std::vector<std::string> expected1 = {"1", "2", "3"};
  EXPECT_EQ(ParseJsonArray("[1, 2, 3]"), expected1);

  // Test string array with quotes.
  const std::vector<std::string> expected2 = {"a", "b", "c"};
  EXPECT_EQ(ParseJsonArray(R"(["a", "b", "c"])"), expected2);

  // Test array with whitespace.
  const std::vector<std::string> expected3 = {"a", "b"};
  EXPECT_EQ(ParseJsonArray(R"(  [ "a" ,  "b" ] )"), expected3);

  // Test empty array.
  EXPECT_TRUE(ParseJsonArray("[]").empty());
  EXPECT_TRUE(ParseJsonArray("").empty());
}

TEST(JsonUtilsTest, GetNumFromJson) {
  // Test valid double.
  const absl::string_view json1 = R"({"key": 123.45})";
  EXPECT_DOUBLE_EQ(GetNumFromJson(json1, "key"), 123.45);

  // Test boolean true.
  const absl::string_view json2 = R"({"key": true})";
  EXPECT_DOUBLE_EQ(GetNumFromJson(json2, "key"), 1.0);

  // Test boolean false.
  const absl::string_view json3 = R"({"key": false})";
  EXPECT_DOUBLE_EQ(GetNumFromJson(json3, "key"), 0.0);

  // Test missing key returns default.
  const absl::string_view json4 = R"({"other": 123.45})";
  EXPECT_DOUBLE_EQ(GetNumFromJson(json4, "key", 9.9), 9.9);

  // Test invalid double returns default.
  const absl::string_view json5 = R"({"key": "not_a_number"})";
  EXPECT_DOUBLE_EQ(GetNumFromJson(json5, "key", 9.9), 9.9);
}

TEST(JsonUtilsTest, GetIdFromJson) {
  // Test string ID.
  const absl::string_view json1 = R"({"key": "my_id"})";
  EXPECT_EQ(GetIdFromJson(json1, "key"), "my_id");

  // Test numeric ID.
  const absl::string_view json2 = R"({"key": 123})";
  EXPECT_EQ(GetIdFromJson(json2, "key"), "123");

  // Test missing ID.
  const absl::string_view json3 = R"({"other": "my_id"})";
  EXPECT_EQ(GetIdFromJson(json3, "key"), "");
}

}  // namespace
}  // namespace xprof::grpc_metrics
