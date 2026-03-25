#include "xprof/convert/trace_viewer/prefix_trie.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/string_view.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "xla/tsl/lib/io/iterator.h"
#include "xla/tsl/lib/io/table.h"
#include "xla/tsl/lib/io/table_builder.h"
#include "xla/tsl/lib/io/table_options.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/file_system.h"

namespace tensorflow {
namespace profiler {

void PrefixTrie::Insert(absl::string_view key, absl::string_view id) {
  map_[std::string(key)].emplace_back(id);
}

absl::Status PrefixTrie::SaveAsLevelDbTable(tsl::WritableFile* file) {
  tsl::table::Options options;
  options.block_size = 20 * 1024 * 1024;
  options.compression = tsl::table::kSnappyCompression;
  tsl::table::TableBuilder builder(options, file);

  // Collect and sort keys to satisfy TableBuilder's strictly increasing order.
  std::vector<absl::string_view> keys;
  keys.reserve(map_.size());
  for (const auto& [key, _] : map_) {
    keys.push_back(key);
  }
  std::sort(keys.begin(), keys.end());

  for (const auto& key : keys) {
    const auto& ids = map_.find(std::string(key))->second;
    PrefixTrieNodeProto proto;
    proto.mutable_terminal_key_ids()->Add(ids.begin(), ids.end());
    builder.Add(key, proto.SerializeAsString());
  }

  TF_RETURN_IF_ERROR(builder.Finish());
  absl::string_view filename;
  TF_RETURN_IF_ERROR(file->Name(&filename));
  TF_RETURN_WITH_CONTEXT_IF_ERROR(
      file->Close(), "Failed to save prefix trie to file: ", filename);
  return absl::OkStatus();
}

absl::StatusOr<std::vector<PrefixSearchResult>> LoadTrieAsLevelDbTableAndSearch(
    absl::string_view filename, absl::string_view prefix) {
  std::vector<PrefixSearchResult> results;
  uint64_t file_size;
  TF_RETURN_IF_ERROR(
      tsl::Env::Default()->GetFileSize(std::string(filename), &file_size));

  tsl::FileSystem* file_system;
  TF_RETURN_IF_ERROR(tsl::Env::Default()->GetFileSystemForFile(
      std::string(filename), &file_system));

  std::unique_ptr<tsl::RandomAccessFile> random_access_file;
  TF_RETURN_IF_ERROR(file_system->NewRandomAccessFile(std::string(filename),
                                                      &random_access_file));

  tsl::table::Options options;
  options.block_size = 20 * 1024 * 1024;
  tsl::table::Table* table = nullptr;
  TF_RETURN_IF_ERROR(tsl::table::Table::Open(options, random_access_file.get(),
                                             file_size, &table));
  std::unique_ptr<tsl::table::Table> table_deleter(table);
  std::unique_ptr<tsl::table::Iterator> iterator(table->NewIterator());

  iterator->Seek(prefix);
  while (iterator->Valid()) {
    std::string key = std::string(iterator->key());
    if (absl::StartsWith(key, prefix)) {
      PrefixTrieNodeProto proto;
      if (!proto.ParseFromString(iterator->value())) {
        return absl::InternalError("Failed to parse PrefixTrieNodeProto");
      }
      if (!proto.terminal_key_ids().empty()) {
        results.push_back({.key = key,
                           .terminal_key_ids = std::vector<std::string>(
                               proto.terminal_key_ids().begin(),
                               proto.terminal_key_ids().end())});
      }
      iterator->Next();
    } else {
      break;
    }
  }

  return results;
}

}  // namespace profiler
}  // namespace tensorflow
