/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#ifndef XPROF_CONVERT_TOOL_OPTIONS_H_
#define XPROF_CONVERT_TOOL_OPTIONS_H_

#include <optional>
#include <string>
#include <variant>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_cat.h"

namespace tensorflow {
namespace profiler {

// Default parameter constants for graph viewer.
static constexpr char kGraphTypeName[] = "graph";
static constexpr char kAdjacentNodes[] = "adj_nodes";
static constexpr char kShortTxtTypeName[] = "short_txt";
static constexpr char kLongTxtTypeName[] = "long_txt";
static constexpr char kJsonTypeName[] = "json";
static constexpr char kProtoTypeName[] = "pb";
static constexpr char kProtoTextTypeName[] = "pbtxt";
static constexpr char kDefaultFormatString[] = "url";
static constexpr int kDefaultWidth = 3;
static constexpr int kDefaultShowMetadata = 0;
static constexpr int kDefaultMergeFusion = 0;

using ToolOptions =
    absl::flat_hash_map<std::string, std::variant<bool, int, std::string>>;

// Helper function to get parameter from tool options.
template <typename T>
std::optional<T> GetParam(const ToolOptions& options, const std::string& key) {
  const auto iter = options.find(key);
  if (iter == options.end()) {
    return std::nullopt;
  }

  const T* result = std::get_if<T>(&iter->second);
  if (!result) {
    return std::nullopt;
  }
  return *result;
}

// Helper function to get parameter from tool options with default value.
template <typename T>
T GetParamWithDefault(const ToolOptions& options, const std::string& key,
                      const T& default_param) {
  if (auto param = GetParam<T>(options, key)) {
    return *param;
  }
  return default_param;
}

inline std::string DebugString(const ToolOptions& options) {
  std::string output;
  for (const auto& [k, v] : options) {
    absl::StrAppend(
        &output, k, ":",
        std::visit([](const auto& value) { return absl::StrCat(value); }, v),
        ":", v.index(), ";");
  }
  return absl::StrCat("{", output, "}");
}

}  // namespace profiler
}  // namespace tensorflow

#endif  // XPROF_CONVERT_TOOL_OPTIONS_H_
