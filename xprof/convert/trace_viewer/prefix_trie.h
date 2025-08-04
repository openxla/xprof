#ifndef THIRD_PARTY_XPROF_CONVERT_TRACE_VIEWER_PREFIX_TRIE_H_
#define THIRD_PARTY_XPROF_CONVERT_TRACE_VIEWER_PREFIX_TRIE_H_

#include <memory>
#include <string>
#include <vector>

#include "absl/container/btree_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/platform/file_system.h"
#include "plugin/xprof/protobuf/prefix_trie_node.pb.h"

namespace tensorflow {
namespace profiler {

// A node in the prefix trie data structure.
struct PrefixTrieNode {
  std::string key;
  // The ids of the keys that are termindated at this node.
  std::vector<std::string> terminal_key_ids;
  absl::btree_map<char, std::unique_ptr<PrefixTrieNode>> children;

  explicit PrefixTrieNode(std::string key) : key(key) {}

  PrefixTrieNode(const PrefixTrieNode&) = delete;
  PrefixTrieNode& operator=(const PrefixTrieNode&) = delete;

  // Converts the node to a proto which is then serialized to a file.
  PrefixTrieNodeProto ToProto() const;
};

// The result of a prefix search.
struct PrefixSearchResult {
  std::string key;
  std::vector<std::string> terminal_key_ids;
};

// Loads the provided leveldb table and finds all keys with the given prefix. It
// returns the key and the terminal key ids of all the satisfying keys.
absl::StatusOr<std::vector<PrefixSearchResult>> LoadTrieAsLevelDbTableAndSearch(
    absl::string_view filename, absl::string_view prefix);

// Manages a prefix trie data structure. The trie is built in memory and once
// built, it is saved as a leveldb table. The leveldb table can then be loaded
// and used to find all terminating key ids with a given prefix.
class PrefixTrie {
 public:
  PrefixTrie() { root_ = std::make_unique<PrefixTrieNode>(""); }
  // Inserts a key into the trie. The id is stored as a terminal key id at the
  // leaf node of the key.
  void Insert(absl::string_view key, absl::string_view id);
  // Saves the trie as a leveldb table.
  absl::Status SaveAsLevelDbTable(std::unique_ptr<tsl::WritableFile>& file);

 private:
  std::unique_ptr<PrefixTrieNode> root_;
};

}  // namespace profiler
}  // namespace tensorflow

#endif  // THIRD_PARTY_XPROF_CONVERT_TRACE_VIEWER_PREFIX_TRIE_H_
