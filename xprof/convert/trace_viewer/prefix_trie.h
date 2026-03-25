#ifndef THIRD_PARTY_XPROF_CONVERT_TRACE_VIEWER_PREFIX_TRIE_H_
#define THIRD_PARTY_XPROF_CONVERT_TRACE_VIEWER_PREFIX_TRIE_H_

#include <memory>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/platform/file_system.h"
#include "plugin/xprof/protobuf/prefix_trie_node.pb.h"

namespace tensorflow {
namespace profiler {

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
  // Inserts a key into the trie. The id is stored as a terminal key id at the
  // leaf node of the key.
  void Insert(absl::string_view key, absl::string_view id);

  // Saves the trie as a leveldb table where each node's key becomes a LevelDB
  // table key and the value is the serialized terminal key ids. This format is
  // optimized for prefix searches using LevelDB's iterator.
  absl::Status SaveAsLevelDbTable(tsl::WritableFile* file);

 private:
  absl::flat_hash_map<std::string, std::vector<std::string>> map_;
};

}  // namespace profiler
}  // namespace tensorflow

#endif  // THIRD_PARTY_XPROF_CONVERT_TRACE_VIEWER_PREFIX_TRIE_H_
