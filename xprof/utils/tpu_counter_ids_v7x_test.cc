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

#include "xprof/utils/tpu_counter_ids_v7x.h"

#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/strings/ascii.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/platform/test.h"

namespace xprof {
namespace {

struct ParsedCounter {
  std::string name;
  uint64_t id = 0;
};

// Strips // line comments, then collects `NAME = <decimal>` assignments from
// the TpuCounterIdsTpu7x enum body. Handles values on the same line as the name
// or on a following line (as in tpu_counter_ids_v7x.h).
std::vector<ParsedCounter> ParseTpuCounterIdAssignments(
    absl::string_view header_text) {
  std::string without_comments;
  without_comments.reserve(header_text.size());
  for (absl::string_view line : absl::StrSplit(header_text, '\n')) {
    size_t comment = line.find("//");
    if (comment != absl::string_view::npos) {
      line = line.substr(0, comment);
    }
    absl::StrAppend(&without_comments, line, "\n");
  }

  const size_t enum_pos = without_comments.find("enum TpuCounterIdsTpu7x");
  if (enum_pos == std::string::npos) {
    return {};
  }
  const size_t brace_open = without_comments.find('{', enum_pos);
  if (brace_open == std::string::npos) {
    return {};
  }
  const size_t brace_close = without_comments.find('}', brace_open + 1);
  if (brace_close == std::string::npos) {
    return {};
  }
  const absl::string_view body = absl::string_view(without_comments)
                                     .substr(brace_open + 1,
                                             brace_close - brace_open - 1);

  std::vector<ParsedCounter> counters;
  for (absl::string_view entry : absl::StrSplit(body, ',')) {
    entry = absl::StripAsciiWhitespace(entry);
    if (entry.empty()) {
      continue;
    }
    const size_t eq = entry.find('=');
    if (eq == absl::string_view::npos) {
      continue;
    }
    absl::string_view name = absl::StripAsciiWhitespace(entry.substr(0, eq));
    absl::string_view val_str =
        absl::StripAsciiWhitespace(entry.substr(eq + 1));
    // Guard against accidental extra '=' fragments.
    if (val_str.find('=') != absl::string_view::npos) {
      continue;
    }
    if (name.empty() || val_str.empty()) {
      continue;
    }
    uint64_t id = 0;
    if (!absl::SimpleAtoi(val_str, &id)) {
      continue;
    }
    counters.push_back(ParsedCounter{std::string(name), id});
  }
  return counters;
}

std::string ReadFileOrEmpty(const std::string& path) {
  std::ifstream in(path);
  if (!in) {
    return {};
  }
  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

// Resolves tpu_counter_ids_v7x.h under Bazel runfiles or a local checkout.
std::string FindV7xHeaderPath() {
  const char* test_srcdir = std::getenv("TEST_SRCDIR");
  const char* test_workspace = std::getenv("TEST_WORKSPACE");
  std::vector<std::string> candidates;
  if (test_srcdir != nullptr) {
    if (test_workspace != nullptr) {
      candidates.push_back(absl::StrCat(test_srcdir, "/", test_workspace,
                                        "/xprof/utils/tpu_counter_ids_v7x.h"));
    }
    candidates.push_back(
        absl::StrCat(test_srcdir, "/org_xprof/xprof/utils/tpu_counter_ids_v7x.h"));
  }
  candidates.push_back("xprof/utils/tpu_counter_ids_v7x.h");
  candidates.push_back("tpu_counter_ids_v7x.h");

  for (const std::string& path : candidates) {
    std::ifstream in(path);
    if (in.good()) {
      return path;
    }
  }
  return {};
}

TEST(TpuCounterIdsV7xTest, ParserCollectsSameLineAndWrappedAssignments) {
  constexpr absl::string_view kSnippet = R"(
enum TpuCounterIdsTpu7x : uint64_t {
  FOO = 1,
  BAR =
      2,
  BAZ = 3,
};
)";
  const auto counters = ParseTpuCounterIdAssignments(kSnippet);
  ASSERT_EQ(counters.size(), 3);
  EXPECT_EQ(counters[0].name, "FOO");
  EXPECT_EQ(counters[0].id, 1u);
  EXPECT_EQ(counters[1].name, "BAR");
  EXPECT_EQ(counters[1].id, 2u);
  EXPECT_EQ(counters[2].name, "BAZ");
  EXPECT_EQ(counters[2].id, 3u);
}

TEST(TpuCounterIdsV7xTest, ParserIgnoresLineComments) {
  constexpr absl::string_view kSnippet = R"(
enum TpuCounterIdsTpu7x : uint64_t {
  // FOO = 999,
  BAR = 2,  // trailing
};
)";
  const auto counters = ParseTpuCounterIdAssignments(kSnippet);
  ASSERT_EQ(counters.size(), 1);
  EXPECT_EQ(counters[0].name, "BAR");
  EXPECT_EQ(counters[0].id, 2u);
}

TEST(TpuCounterIdsV7xTest, HeaderNumericIdsAreUnique) {
  const std::string path = FindV7xHeaderPath();
  ASSERT_FALSE(path.empty()) << "Could not locate tpu_counter_ids_v7x.h";
  const std::string text = ReadFileOrEmpty(path);
  ASSERT_FALSE(text.empty()) << "Failed to read " << path;

  const auto counters = ParseTpuCounterIdAssignments(text);
  // Guard against an empty parse (regex/format drift) silently passing.
  ASSERT_GT(counters.size(), 1000)
      << "Expected thousands of TPU v7x counter IDs in " << path;

  absl::flat_hash_map<uint64_t, std::string> id_to_name;
  absl::flat_hash_set<std::string> names;
  std::vector<std::string> duplicate_ids;
  std::vector<std::string> duplicate_names;

  for (const ParsedCounter& c : counters) {
    if (!names.insert(c.name).second) {
      duplicate_names.push_back(c.name);
    }
    auto [it, inserted] = id_to_name.emplace(c.id, c.name);
    if (!inserted) {
      duplicate_ids.push_back(absl::StrCat(c.name, "=", c.id, " collides with ",
                                           it->second));
    }
  }

  EXPECT_TRUE(duplicate_names.empty())
      << "Duplicate enum enumerator names: "
      << absl::StrJoin(duplicate_names, ", ");
  EXPECT_TRUE(duplicate_ids.empty())
      << "Duplicate counter numeric IDs:\n"
      << absl::StrJoin(duplicate_ids, "\n");
  EXPECT_EQ(id_to_name.size(), counters.size());
  EXPECT_EQ(names.size(), counters.size());
}

TEST(TpuCounterIdsV7xTest, HeaderCompilesAndExposesDistinctSampleIds) {
  // Smoke-check a few well-known entries still map to distinct hardware IDs.
  EXPECT_NE(
      static_cast<uint64_t>(
          VF_CHIP_DIE0_PWRMGR_PWRMGR_TC_THROTTLE_CORE_DEBUG_STATS_UNPRIVILEGED_CLOCKS_SKIPPED),
      static_cast<uint64_t>(
          VF_CHIP_DIE0_PWRMGR_PWRMGR_TC_THROTTLE_CORE_DEBUG_STATS_UNPRIVILEGED_CYCLE_COUNT));
  EXPECT_EQ(
      static_cast<uint64_t>(
          VF_CHIP_DIE0_PWRMGR_PWRMGR_TC_THROTTLE_CORE_DEBUG_STATS_UNPRIVILEGED_CLOCKS_SKIPPED),
      2847490064ull);
  EXPECT_EQ(
      static_cast<uint64_t>(
          VF_CHIP_DIE0_PWRMGR_PWRMGR_TC_THROTTLE_CORE_DEBUG_STATS_UNPRIVILEGED_CYCLE_COUNT),
      2847490056ull);
}

}  // namespace
}  // namespace xprof
