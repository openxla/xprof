#include "xprof/convert/trace_viewer/prefix_trie.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/btree_map.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
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

PrefixTrieNode::~PrefixTrieNode() {
  std::vector<std::unique_ptr<PrefixTrieNode>> children_to_delete;
  auto delete_children =
      [&children_to_delete](
          absl::btree_map<char, std::unique_ptr<PrefixTrieNode>>& children) {
        for (auto& [_, child] : children) {
          if (child != nullptr) {
            children_to_delete.push_back(std::move(child));
          }
        }
      };

  delete_children(this->children);
  children.clear();

  while (!children_to_delete.empty()) {
    std::unique_ptr<PrefixTrieNode> current =
        std::move(children_to_delete.back());
    children_to_delete.pop_back();
    delete_children(current->children);
  }
}

PrefixTrieNodeProto PrefixTrieNode::ToProto() const {
  PrefixTrieNodeProto proto;
  proto.mutable_terminal_key_ids()->Add(terminal_key_ids.begin(),
                                        terminal_key_ids.end());
  return proto;
}

void PrefixTrie::Insert(absl::string_view key, absl::string_view id) {
  PrefixTrieNode* node = &root_;
  for (const char c : key) {
    auto& child = node->children[c];
    if (child == nullptr) {
      child = std::make_unique<PrefixTrieNode>();
    }
    node = child.get();
  }
  node->terminal_key_ids.emplace_back(id);
}

void IterateTrieAndSaveToLevelDbTable(PrefixTrieNode* node, std::string key,
                                      tsl::table::TableBuilder& builder) {
  std::vector<std::pair<PrefixTrieNode*, std::string>> stack;
  stack.push_back({node, std::move(key)});

  while (!stack.empty()) {
    auto [node, key] = stack.back();
    stack.pop_back();

    auto proto = node->ToProto();
    builder.Add(key, proto.SerializeAsString());
    for (auto it = node->children.rbegin(); it != node->children.rend(); ++it) {
      stack.push_back(
          {it->second.get(), absl::StrCat(key, std::string(1, it->first))});
    }
  }
}

absl::Status PrefixTrie::SaveAsLevelDbTable(tsl::WritableFile* file) {
  tsl::table::Options options;
  options.block_size = 20 * 1024 * 1024;
  options.compression = tsl::table::kSnappyCompression;
  tsl::table::TableBuilder builder(options, file);
  IterateTrieAndSaveToLevelDbTable(&root_, "", builder);
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
