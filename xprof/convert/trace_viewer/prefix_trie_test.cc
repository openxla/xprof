#include "xprof/convert/trace_viewer/prefix_trie.h"

#include <memory>
#include <string>
#include <vector>

#include "testing/base/public/gmock.h"
#include "<gtest/gtest.h>"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/file_system.h"

namespace {

using ::tensorflow::profiler::LoadTrieAsLevelDbTableAndSearch;
using ::tensorflow::profiler::PrefixSearchResult;
using ::tensorflow::profiler::PrefixTrie;
using ::testing::FieldsAre;
using ::testing::IsEmpty;
using ::testing::UnorderedElementsAreArray;
using ::testing::UnorderedPointwise;

std::string GetTempFilename() {
  return absl::StrCat(::testing::TempDir(), "/prefix_trie.ptldb2");
}

// Custom matcher to compare PrefixSearchResult structs.
MATCHER(PrefixSearchResultFieldsAreEqual, "") {
  const PrefixSearchResult& a = std::get<0>(arg);
  const PrefixSearchResult& b = std::get<1>(arg);
  return ExplainMatchResult(
      FieldsAre(b.key, UnorderedElementsAreArray(b.terminal_key_ids)), a,
      result_listener);
}

TEST(PrefixTrieTest, PrefixSearchOnEmptyTrie) {
  PrefixTrie trie;

  std::unique_ptr<tsl::WritableFile> file;
  ASSERT_OK(tsl::Env::Default()->NewWritableFile(GetTempFilename(), &file));
  ASSERT_OK(trie.SaveAsLevelDbTable(file));

  auto results = LoadTrieAsLevelDbTableAndSearch(GetTempFilename(), "a");
  ASSERT_OK(results);
  EXPECT_THAT(*results, IsEmpty());
}

TEST(PrefixTrieTest, PrefixSearchOnTrieWithSingleKey) {
  PrefixTrie trie;
  trie.Insert("abc", "id00");

  std::unique_ptr<tsl::WritableFile> file;
  ASSERT_OK(tsl::Env::Default()->NewWritableFile(GetTempFilename(), &file));
  ASSERT_OK(trie.SaveAsLevelDbTable(file));

  auto results = LoadTrieAsLevelDbTableAndSearch(GetTempFilename(), "a");
  ASSERT_OK(results);
  std::vector<PrefixSearchResult> expected_results = {
      {.key = "abc", .terminal_key_ids = {"id00"}}};
  EXPECT_THAT(*results, UnorderedPointwise(PrefixSearchResultFieldsAreEqual(),
                                           expected_results));
}

class PrefixTrieWithDataTest : public ::testing::Test {
 protected:
  void SetUp() override {
    filename_ = GetTempFilename();
    PrefixTrie trie;
    trie.Insert("abc", "id00");
    trie.Insert("abcd", "id01");
    trie.Insert("abce", "id02");
    trie.Insert("a", "id03");
    trie.Insert("abcd", "id04");
    trie.Insert("def", "id05");

    std::unique_ptr<tsl::WritableFile> file;
    ASSERT_OK(tsl::Env::Default()->NewWritableFile(filename_, &file));
    ASSERT_OK(trie.SaveAsLevelDbTable(file));
  }

  std::string filename_;
};

TEST_F(PrefixTrieWithDataTest,
       PrefixSearchOnTrieWithMultipleKeysMatchingPrefix) {
  auto results = LoadTrieAsLevelDbTableAndSearch(filename_, "abc");
  ASSERT_OK(results);
  std::vector<PrefixSearchResult> expected_results = {
      {.key = "abc", .terminal_key_ids = {"id00"}},
      {.key = "abcd", .terminal_key_ids = {"id01", "id04"}},
      {.key = "abce", .terminal_key_ids = {"id02"}}};
  EXPECT_THAT(*results, UnorderedPointwise(PrefixSearchResultFieldsAreEqual(),
                                           expected_results));
}

TEST_F(PrefixTrieWithDataTest,
       PrefixSearchOnTrieWithMultipleKeysPrefixNotFound) {
  auto results = LoadTrieAsLevelDbTableAndSearch(filename_, "abd");
  ASSERT_OK(results);
  EXPECT_THAT(*results, IsEmpty());
}

TEST_F(PrefixTrieWithDataTest, PrefixSearchOnTrieWithMultipleKeysEmptyPrefix) {
  auto results = LoadTrieAsLevelDbTableAndSearch(filename_, "");
  ASSERT_OK(results);
  std::vector<PrefixSearchResult> expected_results = {
      {.key = "abc", .terminal_key_ids = {"id00"}},
      {.key = "abcd", .terminal_key_ids = {"id01", "id04"}},
      {.key = "abce", .terminal_key_ids = {"id02"}},
      {.key = "a", .terminal_key_ids = {"id03"}},
      {.key = "def", .terminal_key_ids = {"id05"}}};
  EXPECT_THAT(*results, UnorderedPointwise(PrefixSearchResultFieldsAreEqual(),
                                           expected_results));
}

TEST_F(PrefixTrieWithDataTest,
       PrefixSearchOnTrieWithMultipleKeysPrefixIsAnExactMatch) {
  auto results = LoadTrieAsLevelDbTableAndSearch(filename_, "abce");
  ASSERT_OK(results);
  std::vector<PrefixSearchResult> expected_results = {
      {.key = "abce", .terminal_key_ids = {"id02"}}};
  EXPECT_THAT(*results, UnorderedPointwise(PrefixSearchResultFieldsAreEqual(),
                                           expected_results));
}

TEST(PrefixTrieTest, PrefixTrieSearchWithFileNotFound) {
  auto results =
      LoadTrieAsLevelDbTableAndSearch("non_existent_file.ptldb2", "a");
  EXPECT_FALSE(results.ok());
  EXPECT_EQ(results.status().code(), absl::StatusCode::kNotFound);
}
}  // namespace
