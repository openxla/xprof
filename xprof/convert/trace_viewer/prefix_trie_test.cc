#include "xprof/convert/trace_viewer/prefix_trie.h"

#include <memory>
#include <string>
#include <vector>

#include "testing/base/public/gmock.h"
#include "<gtest/gtest.h>"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "xla/tsl/lib/io/iterator.h"
#include "xla/tsl/lib/io/table.h"
#include "xla/tsl/lib/io/table_options.h"
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

TEST_F(PrefixTrieWithDataTest, TotalEntriesInTable) {
  uint64_t file_size;
  ASSERT_OK(tsl::Env::Default()->GetFileSize(filename_, &file_size));

  tsl::FileSystem* file_system;
  ASSERT_OK(tsl::Env::Default()->GetFileSystemForFile(filename_, &file_system));

  std::unique_ptr<tsl::RandomAccessFile> random_access_file;
  ASSERT_OK(file_system->NewRandomAccessFile(filename_, &random_access_file));

  tsl::table::Options options;
  options.block_size = 20 * 1024 * 1024;
  tsl::table::Table* table = nullptr;
  ASSERT_OK(tsl::table::Table::Open(options, random_access_file.get(),
                                     file_size, &table));
  std::unique_ptr<tsl::table::Table> table_deleter(table);
  std::unique_ptr<tsl::table::Iterator> iterator(table->NewIterator());

  int count = 0;
  iterator->SeekToFirst();
  while (iterator->Valid()) {
    count++;
    iterator->Next();
  }

  // Unique keys inserted: "a", "abc", "abcd", "abce", "def".
  // Total = 5.
  EXPECT_EQ(count, 5);
}
}  // namespace
