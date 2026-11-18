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

#include "xprof/convert/tool_options.h"

#include <optional>
#include <string>

#include <gtest/gtest.h>

namespace tensorflow {
namespace profiler {
namespace {

TEST(ToolOptionsTest, GetParamBool) {
  ToolOptions options;
  options["bool_true"] = true;
  options["bool_false"] = false;
  options["int_zero"] = 0;
  options["int_one"] = 1;
  options["int_two"] = 2;
  options["str_true"] = std::string("true");
  options["str_false"] = std::string("false");
  options["str_zero"] = std::string("0");
  options["str_one"] = std::string("1");
  options["str_two"] = std::string("2");
  options["str_empty"] = std::string("");
  options["str_garbage"] = std::string("garbage");

  EXPECT_EQ(GetParam<bool>(options, "bool_true"), std::optional<bool>(true));
  EXPECT_EQ(GetParam<bool>(options, "bool_false"), std::optional<bool>(false));
  EXPECT_EQ(GetParam<bool>(options, "int_zero"), std::optional<bool>(false));
  EXPECT_EQ(GetParam<bool>(options, "int_one"), std::optional<bool>(true));
  EXPECT_EQ(GetParam<bool>(options, "int_two"), std::optional<bool>(true));
  EXPECT_EQ(GetParam<bool>(options, "str_true"), std::optional<bool>(true));
  EXPECT_EQ(GetParam<bool>(options, "str_false"), std::optional<bool>(false));
  EXPECT_EQ(GetParam<bool>(options, "str_zero"), std::optional<bool>(false));
  EXPECT_EQ(GetParam<bool>(options, "str_one"), std::optional<bool>(true));
  EXPECT_EQ(GetParam<bool>(options, "str_two"), std::optional<bool>(true));

  EXPECT_EQ(GetParam<bool>(options, "str_empty"), std::nullopt);
  EXPECT_EQ(GetParam<bool>(options, "str_garbage"), std::nullopt);
  EXPECT_EQ(GetParam<bool>(options, "missing"), std::nullopt);

  EXPECT_FALSE(GetParamWithDefault<bool>(options, "str_empty", false));
  EXPECT_TRUE(GetParamWithDefault<bool>(options, "str_empty", true));
  EXPECT_FALSE(GetParamWithDefault<bool>(options, "str_garbage", false));
  EXPECT_TRUE(GetParamWithDefault<bool>(options, "str_garbage", true));
}

TEST(ToolOptionsTest, GetParamInt) {
  ToolOptions options;
  options["int_val"] = 10;
  options["str_int"] = std::string("10");
  options["str_neg"] = std::string("-5");
  options["str_invalid"] = std::string("abc");
  options["str_bool"] = std::string("true");
  options["bool_val"] = true;

  EXPECT_EQ(GetParam<int>(options, "int_val"), std::optional<int>(10));
  EXPECT_EQ(GetParam<int>(options, "str_int"), std::optional<int>(10));
  EXPECT_EQ(GetParam<int>(options, "str_neg"), std::optional<int>(-5));
  EXPECT_EQ(GetParam<int>(options, "str_invalid"), std::nullopt);
  EXPECT_EQ(GetParam<int>(options, "str_bool"), std::nullopt);
  EXPECT_EQ(GetParam<int>(options, "bool_val"), std::nullopt);
  EXPECT_EQ(GetParam<int>(options, "missing"), std::nullopt);

  EXPECT_EQ(GetParamWithDefault<int>(options, "str_invalid", 3), 3);
}

TEST(ToolOptionsTest, GetParamString) {
  ToolOptions options;
  options["str_val"] = std::string("hello");
  options["int_val"] = 42;
  options["bool_true"] = true;
  options["bool_false"] = false;

  EXPECT_EQ(GetParam<std::string>(options, "str_val"),
            std::optional<std::string>("hello"));
  EXPECT_EQ(GetParam<std::string>(options, "int_val"),
            std::optional<std::string>("42"));
  EXPECT_EQ(GetParam<std::string>(options, "bool_true"),
            std::optional<std::string>("true"));
  EXPECT_EQ(GetParam<std::string>(options, "bool_false"),
            std::optional<std::string>("false"));
  EXPECT_EQ(GetParam<std::string>(options, "missing"), std::nullopt);
}

}  // namespace
}  // namespace profiler
}  // namespace tensorflow
