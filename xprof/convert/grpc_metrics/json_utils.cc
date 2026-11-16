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

#include <cstddef>
#include <string>
#include <vector>

#include "absl/strings/ascii.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"

namespace xprof::grpc_metrics {

absl::string_view ExtractJsonValue(absl::string_view json,
                                   absl::string_view key) {
  size_t pos = 0;
  while (true) {
    pos = json.find(key, pos);
    if (pos == absl::string_view::npos) return "";
    if (pos > 0 && json[pos - 1] == '"') {
      size_t p = pos + key.length();
      if (p < json.length() && json[p] == '_') p++;
      if (p < json.length() && json[p] == '"') {
        p++;
        while (p < json.length() && (json[p] == ' ' || json[p] == '\t')) p++;
        if (p < json.length() && json[p] == ':') {
          pos = p + 1;
          break;
        }
      }
    }
    pos += key.length();
  }
  while (pos < json.length() && (json[pos] == ' ' || json[pos] == '\t')) pos++;
  if (pos >= json.length()) return "";

  size_t end_pos = pos;
  if (json[pos] == '"') {
    pos++;
    end_pos = json.find('"', pos);
    if (end_pos == absl::string_view::npos) return "";
    return json.substr(pos, end_pos - pos);
  } else if (json[pos] == '[') {
    int bracket_count = 1;
    end_pos++;
    while (end_pos < json.length() && bracket_count > 0) {
      if (json[end_pos] == '[')
        bracket_count++;
      else if (json[end_pos] == ']')
        bracket_count--;
      end_pos++;
    }
    if (bracket_count == 0) {
      return json.substr(pos, end_pos - pos);
    }
    return "";
  } else {
    while (end_pos < json.length() && json[end_pos] != ',' &&
           json[end_pos] != '}' && json[end_pos] != ' ' &&
           json[end_pos] != '\n') {
      end_pos++;
    }
    return json.substr(pos, end_pos - pos);
  }
}

std::vector<std::string> ParseJsonArray(absl::string_view array_str) {
  std::vector<std::string> result;
  absl::string_view content = absl::StripAsciiWhitespace(array_str);
  if (content.empty()) return result;
  if (content[0] == '[' && content.back() == ']') {
    content = content.substr(1, content.length() - 2);
  }
  std::vector<absl::string_view> parts = absl::StrSplit(content, ',');
  result.reserve(parts.size());
  for (auto part : parts) {
    part = absl::StripAsciiWhitespace(part);
    if (!part.empty()) {
      if (part[0] == '"' && part.back() == '"') {
        part = part.substr(1, part.length() - 2);
      }
      result.push_back(std::string(part));
    }
  }
  return result;
}

double GetNumFromJson(absl::string_view json, absl::string_view key,
                      double default_val) {
  absl::string_view valStr = ExtractJsonValue(json, key);
  if (valStr.empty()) return default_val;
  if (valStr == "true") return 1.0;
  if (valStr == "false") return 0.0;
  double val;
  if (absl::SimpleAtod(valStr, &val)) {
    return val;
  }
  return default_val;
}

std::string GetIdFromJson(absl::string_view json, absl::string_view key) {
  absl::string_view val =
      absl::StripAsciiWhitespace(ExtractJsonValue(json, key));
  while (val.size() >= 2 && ((val.front() == '"' && val.back() == '"') ||
                             (val.front() == '\'' && val.back() == '\''))) {
    val = val.substr(1, val.size() - 2);
  }
  return std::string(val);
}

}  // namespace xprof::grpc_metrics
