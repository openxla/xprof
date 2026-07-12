#include "xprof/convert/trace_viewer/prefix_trie.h"

#include <memory>
#include <string>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
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
  ASSERT_OK(trie.SaveAsLevelDbTable(file.get()));

  auto results = LoadTrieAsLevelDbTableAndSearch(GetTempFilename(), "a");
  ASSERT_OK(results);
  EXPECT_THAT(*results, IsEmpty());
}

TEST(PrefixTrieTest, PrefixSearchOnTrieWithSingleKey) {
  PrefixTrie trie;
  trie.Insert("abc", "id00");

  std::unique_ptr<tsl::WritableFile> file;
  ASSERT_OK(tsl::Env::Default()->NewWritableFile(GetTempFilename(), &file));
  ASSERT_OK(trie.SaveAsLevelDbTable(file.get()));

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
    ASSERT_OK(trie.SaveAsLevelDbTable(file.get()));
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
  EXPECT_EQ(results.status().code(), absl::StatusCode::kNotFound);
}

// Stress-tests the iterative destroy (and optional LevelDB save) path that
// replaced recursive walkers. A single key of length kDepth creates a chain of
// that many child nodes; naive recursion would risk stack overflow.
TEST(PrefixTrieTest, DeepChainInsertDestroyAndSave) {
  constexpr int kDepth = 2000;
  const std::string deep_key(kDepth, 'x');
  const std::string deep_id = "deep_id";
  const std::string filename =
      absl::StrCat(::testing::TempDir(), "/prefix_trie_deep.ptldb2");

  {
    PrefixTrie trie;
    trie.Insert(deep_key, deep_id);
    // A second id on the same deep key exercises multi-id terminals on a leaf.
    trie.Insert(deep_key, "deep_id_2");
    // A short key that is a strict prefix of the deep chain.
    trie.Insert(deep_key.substr(0, 16), "prefix_id");

    std::unique_ptr<tsl::WritableFile> file;
    ASSERT_OK(tsl::Env::Default()->NewWritableFile(filename, &file));
    ASSERT_OK(trie.SaveAsLevelDbTable(file.get()));
  }  // trie destructor must not stack-overflow on a depth-2000 chain

  // Full-key search.
  {
    auto results = LoadTrieAsLevelDbTableAndSearch(filename, deep_key);
    ASSERT_OK(results);
    std::vector<PrefixSearchResult> expected_results = {
        {.key = deep_key, .terminal_key_ids = {deep_id, "deep_id_2"}}};
    EXPECT_THAT(*results, UnorderedPointwise(PrefixSearchResultFieldsAreEqual(),
                                             expected_results));
  }

  // Prefix search: both the short terminal and the deep terminal match.
  {
    auto results =
        LoadTrieAsLevelDbTableAndSearch(filename, deep_key.substr(0, 8));
    ASSERT_OK(results);
    std::vector<PrefixSearchResult> expected_results = {
        {.key = deep_key.substr(0, 16), .terminal_key_ids = {"prefix_id"}},
        {.key = deep_key, .terminal_key_ids = {deep_id, "deep_id_2"}}};
    EXPECT_THAT(*results, UnorderedPointwise(PrefixSearchResultFieldsAreEqual(),
                                             expected_results));
  }
}

// Pure insert+destroy (no LevelDB I/O): verifies iterative ~PrefixTrieNode on
// a deep single-child spine without relying on the save path.
TEST(PrefixTrieTest, DeepChainInsertAndDestroy) {
  constexpr int kDepth = 2500;
  const std::string deep_key(kDepth, 'z');
  PrefixTrie trie;
  trie.Insert(deep_key, "id");
  // Leaving scope runs the iterative destructor across kDepth nodes.
}
}  // namespace
