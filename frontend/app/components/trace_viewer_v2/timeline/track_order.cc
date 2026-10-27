#include "frontend/app/components/trace_viewer_v2/timeline/track_order.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <numeric>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/functional/function_ref.h"
#include "absl/strings/ascii.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"

namespace traceviewer {
namespace {

using SiblingKeyView = std::pair<absl::string_view, int>;

// The half-open range [begin, end) of tracks that form a subtree.
struct Subtree {
  int begin = 0;
  int end = 0;
};

// Returns the range [begin, end) of tracks that are descendants of `parent`.
Subtree ChildRange(absl::Span<const TrackNode> tracks, int parent) {
  if (parent < 0) return {0, static_cast<int>(tracks.size())};
  return {parent + 1, TrackSubtreeEnd(tracks, parent)};
}

// Partitions `tracks[begin, end)` into contiguous subtrees of siblings at the
// shallowest nesting level in that range.
std::vector<Subtree> SplitIntoSiblingSubtrees(
    absl::Span<const TrackNode> tracks, int begin, int end) {
  std::vector<Subtree> subtrees;
  int i = begin;
  while (i < end) {
    const int subtree_end = TrackSubtreeEnd(tracks, i);
    subtrees.push_back({i, subtree_end});
    i = subtree_end;
  }
  return subtrees;
}

// Computes the (name, ordinal) key of each sibling subtree in `subtrees`.
std::vector<SiblingKeyView> ComputeSiblingKeys(
    absl::Span<const TrackNode> tracks, absl::Span<const Subtree> subtrees) {
  std::vector<SiblingKeyView> keys;
  keys.reserve(subtrees.size());
  absl::flat_hash_map<absl::string_view, int> ordinals;
  for (const Subtree& subtree : subtrees) {
    const absl::string_view name = tracks[subtree.begin].name;
    keys.push_back({name, ordinals[name]++});
  }
  return keys;
}

// Compares two track names case-insensitively, breaking ties by exact byte
// order so that alphabetical sorting is deterministic without allocating.
int CompareNamesCaseInsensitive(absl::string_view a, absl::string_view b) {
  const size_t min_len = std::min(a.size(), b.size());
  for (size_t i = 0; i < min_len; ++i) {
    const unsigned char ca =
        static_cast<unsigned char>(absl::ascii_tolower(a[i]));
    const unsigned char cb =
        static_cast<unsigned char>(absl::ascii_tolower(b[i]));
    if (ca != cb) return ca < cb ? -1 : 1;
  }
  if (a.size() != b.size()) return a.size() < b.size() ? -1 : 1;
  return a.compare(b);
}

// Returns the index in `subtrees` of the subtree rooted at `root`, or -1.
int FindSubtree(absl::Span<const Subtree> subtrees, int root) {
  const auto it = absl::c_find_if(subtrees, [root](const Subtree& subtree) {
    return subtree.begin == root;
  });
  return it == subtrees.end() ? -1 : static_cast<int>(it - subtrees.begin());
}

// Returns the (name, ordinal) key of `tracks[index]` among its siblings.
SiblingKeyView GetSiblingKey(absl::Span<const TrackNode> tracks,
                             absl::Span<const int> parents, int index) {
  const int parent = parents[index];
  const Subtree range = ChildRange(tracks, parent);
  const std::vector<Subtree> siblings =
      SplitIntoSiblingSubtrees(tracks, range.begin, range.end);
  const int sibling_index = FindSubtree(siblings, index);
  if (sibling_index < 0) return {tracks[index].name, 0};
  return ComputeSiblingKeys(tracks, siblings)[sibling_index];
}

// Returns the root track index of each subtree in `subtrees`.
std::vector<int> SubtreeRoots(absl::Span<const Subtree> subtrees) {
  std::vector<int> roots;
  roots.reserve(subtrees.size());
  for (const Subtree& subtree : subtrees) {
    roots.push_back(subtree.begin);
  }
  return roots;
}

// The JSON writer and reader below are used instead of a JSON library, so that
// the WASM binary needs no exception handling and no document model.

// Appends `value` to `out` as a JSON string literal.
void AppendJsonString(std::string& out, absl::string_view value) {
  constexpr char kHexDigits[] = "0123456789abcdef";
  out.push_back('"');
  for (const unsigned char c : value) {
    switch (c) {
      case '"':
        out.append(R"(\")");
        break;
      case '\\':
        out.append(R"(\\)");
        break;
      case '\b':
        out.append(R"(\b)");
        break;
      case '\f':
        out.append(R"(\f)");
        break;
      case '\n':
        out.append(R"(\n)");
        break;
      case '\r':
        out.append(R"(\r)");
        break;
      case '\t':
        out.append(R"(\t)");
        break;
      default:
        if (c < 0x20) {
          out.append(R"(\u00)");
          out.push_back(kHexDigits[(c >> 4) & 0xF]);
          out.push_back(kHexDigits[c & 0xF]);
        } else {
          out.push_back(static_cast<char>(c));
        }
        break;
    }
  }
  out.push_back('"');
}

// Appends the Unicode code point `cp` to `out`, encoded in UTF-8.
void AppendCodePointUtf8(std::string& out, uint32_t cp) {
  if (cp <= 0x7F) {
    out.push_back(static_cast<char>(cp));
  } else if (cp <= 0x7FF) {
    out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  } else if (cp <= 0xFFFF) {
    out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
    out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  } else if (cp <= 0x10FFFF) {
    out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
    out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  }
}

// Caps the nesting depth of unknown objects and arrays that `SkipValue()`
// skips, which bounds its recursion.
constexpr int kMaxSkippedValueDepth = 32;

// Minimal JSON reader for `TrackOrder::Deserialize()`, used instead of
// `nlohmann::json` so the WASM binary has no exception handling. The methods
// that read a token skip any leading whitespace.
class JsonParser {
 public:
  explicit JsonParser(absl::string_view input) : input_(input) {}

  // Advances past any ASCII whitespace.
  void SkipWhitespace() {
    while (pos_ < input_.size() && absl::ascii_isspace(input_[pos_])) {
      ++pos_;
    }
  }

  // Returns true if all input has been consumed.
  bool AtEnd() {
    SkipWhitespace();
    return pos_ == input_.size();
  }

  // Consumes `expected` character if it is the next non-whitespace character.
  bool Consume(char expected) {
    SkipWhitespace();
    if (pos_ < input_.size() && input_[pos_] == expected) {
      ++pos_;
      return true;
    }
    return false;
  }

  // Consumes `literal` string if it matches the next non-whitespace characters.
  bool ConsumeLiteral(absl::string_view literal) {
    SkipWhitespace();
    if (pos_ + literal.size() <= input_.size() &&
        input_.substr(pos_, literal.size()) == literal) {
      pos_ += literal.size();
      return true;
    }
    return false;
  }

  // Returns the next non-whitespace character without consuming it, or '\0'.
  char Peek() {
    SkipWhitespace();
    return pos_ < input_.size() ? input_[pos_] : '\0';
  }

  // Parses a JSON string literal and unescapes it into `out`.
  bool ParseString(std::string& out) {
    out.clear();
    if (!Consume('"')) return false;
    while (pos_ < input_.size()) {
      const unsigned char c = static_cast<unsigned char>(input_[pos_++]);
      if (c == '"') return true;
      if (c < 0x20) return false;
      if (c != '\\') {
        out.push_back(static_cast<char>(c));
        continue;
      }
      if (pos_ >= input_.size()) return false;
      const char esc = input_[pos_++];
      switch (esc) {
        case '"':
        case '\\':
        case '/':
          out.push_back(esc);
          break;
        case 'b':
          out.push_back('\b');
          break;
        case 'f':
          out.push_back('\f');
          break;
        case 'n':
          out.push_back('\n');
          break;
        case 'r':
          out.push_back('\r');
          break;
        case 't':
          out.push_back('\t');
          break;
        case 'u': {
          uint32_t code_unit = 0;
          if (!ParseHex4(code_unit)) return false;
          if (code_unit >= 0xD800 && code_unit <= 0xDBFF) {
            if (pos_ + 2 > input_.size() || input_[pos_] != '\\' ||
                input_[pos_ + 1] != 'u') {
              return false;
            }
            pos_ += 2;
            uint32_t low = 0;
            if (!ParseHex4(low) || low < 0xDC00 || low > 0xDFFF) return false;
            code_unit =
                0x10000 + (((code_unit - 0xD800) << 10) | (low - 0xDC00));
          } else if (code_unit >= 0xDC00 && code_unit <= 0xDFFF) {
            return false;
          }
          AppendCodePointUtf8(out, code_unit);
          break;
        }
        default:
          return false;
      }
    }
    return false;
  }

  // Parses a non-negative decimal integer without leading zeros.
  bool ParseNonNegativeInt(int& out) {
    SkipWhitespace();
    if (pos_ >= input_.size() || !absl::ascii_isdigit(input_[pos_])) {
      return false;
    }
    if (input_[pos_] == '0' && pos_ + 1 < input_.size() &&
        absl::ascii_isdigit(input_[pos_ + 1])) {
      return false;
    }
    int64_t value = 0;
    while (pos_ < input_.size() && absl::ascii_isdigit(input_[pos_])) {
      value = value * 10 + (input_[pos_++] - '0');
      if (value > std::numeric_limits<int>::max()) return false;
    }
    out = static_cast<int>(value);
    return true;
  }

  // Skips any valid JSON value (object, array, string, number, boolean, or
  // null) so unknown or future fields can be ignored without failing.
  bool SkipValue(int depth = 0) {
    if (depth > kMaxSkippedValueDepth) return false;
    const char next = Peek();
    if (next == '"') {
      std::string ignored;
      return ParseString(ignored);
    }
    if (next == '{') {
      Consume('{');
      if (Consume('}')) return true;
      while (true) {
        std::string key;
        if (!ParseString(key) || !Consume(':') || !SkipValue(depth + 1)) {
          return false;
        }
        if (Consume('}')) return true;
        if (!Consume(',')) return false;
      }
    }
    if (next == '[') {
      Consume('[');
      if (Consume(']')) return true;
      while (true) {
        if (!SkipValue(depth + 1)) return false;
        if (Consume(']')) return true;
        if (!Consume(',')) return false;
      }
    }
    if (ConsumeLiteral("true") || ConsumeLiteral("false") ||
        ConsumeLiteral("null")) {
      return true;
    }
    if (next == '-' || absl::ascii_isdigit(next)) {
      if (next == '-') ++pos_;
      if (pos_ >= input_.size() || !absl::ascii_isdigit(input_[pos_])) {
        return false;
      }
      if (input_[pos_] == '0' && pos_ + 1 < input_.size() &&
          absl::ascii_isdigit(input_[pos_ + 1])) {
        return false;
      }
      while (pos_ < input_.size() && absl::ascii_isdigit(input_[pos_])) {
        ++pos_;
      }
      if (pos_ < input_.size() && input_[pos_] == '.') {
        ++pos_;
        if (pos_ >= input_.size() || !absl::ascii_isdigit(input_[pos_])) {
          return false;
        }
        while (pos_ < input_.size() && absl::ascii_isdigit(input_[pos_])) {
          ++pos_;
        }
      }
      if (pos_ < input_.size() &&
          (input_[pos_] == 'e' || input_[pos_] == 'E')) {
        ++pos_;
        if (pos_ < input_.size() &&
            (input_[pos_] == '+' || input_[pos_] == '-')) {
          ++pos_;
        }
        if (pos_ >= input_.size() || !absl::ascii_isdigit(input_[pos_])) {
          return false;
        }
        while (pos_ < input_.size() && absl::ascii_isdigit(input_[pos_])) {
          ++pos_;
        }
      }
      return true;
    }
    return false;
  }

 private:
  bool ParseHex4(uint32_t& out) {
    if (pos_ + 4 > input_.size()) return false;
    uint32_t val = 0;
    for (int i = 0; i < 4; ++i) {
      const char c = input_[pos_++];
      val <<= 4;
      if (c >= '0' && c <= '9') {
        val |= static_cast<uint32_t>(c - '0');
      } else if (c >= 'a' && c <= 'f') {
        val |= static_cast<uint32_t>(c - 'a' + 10);
      } else if (c >= 'A' && c <= 'F') {
        val |= static_cast<uint32_t>(c - 'A' + 10);
      } else {
        return false;
      }
    }
    out = val;
    return true;
  }

  absl::string_view input_;
  size_t pos_ = 0;
};

// Returns the display order of the sibling subtrees that start at the
// indices `roots` of `tracks`, as indices into `roots`, ordered by `ranks`
// (see `RankChildren()`). Returns the natural order if `ranks` is empty.
std::vector<int> SortSiblings(absl::Span<const TrackNode> tracks,
                              absl::Span<const int> roots,
                              absl::Span<const int> ranks) {
  std::vector<int> positions(roots.size());
  std::iota(positions.begin(), positions.end(), 0);
  if (ranks.empty()) return positions;
  // In DFS pre-order, the nesting levels of consecutive siblings never
  // increase. Siblings are only reordered among those with the same level, so
  // that no track ever becomes part of another track's subtree. The sort is
  // stable, so siblings that compare equal keep their natural order.
  absl::c_stable_sort(positions, [&](int a, int b) {
    const TrackNode& track_a = tracks[roots[a]];
    const TrackNode& track_b = tracks[roots[b]];
    if (track_a.nesting_level != track_b.nesting_level) {
      return track_a.nesting_level > track_b.nesting_level;
    }
    return ranks[a] < ranks[b];
  });
  return positions;
}

}  // namespace

std::vector<int> ComputeTrackParents(absl::Span<const TrackNode> tracks) {
  std::vector<int> parents(tracks.size(), -1);
  // The chain of ancestors of the current track, with strictly increasing
  // nesting levels.
  std::vector<int> ancestors;
  for (int i = 0; i < static_cast<int>(tracks.size()); ++i) {
    while (!ancestors.empty() &&
           tracks[ancestors.back()].nesting_level >= tracks[i].nesting_level) {
      ancestors.pop_back();
    }
    if (!ancestors.empty()) parents[i] = ancestors.back();
    ancestors.push_back(i);
  }
  return parents;
}

int TrackSubtreeEnd(absl::Span<const TrackNode> tracks, int index) {
  const int size = static_cast<int>(tracks.size());
  if (index < 0 || index >= size) return std::clamp(index, 0, size);
  int end = index + 1;
  while (end < size &&
         tracks[end].nesting_level > tracks[index].nesting_level) {
    ++end;
  }
  return end;
}

TrackOrder::PresentParents TrackOrder::CollectPresentParents(
    absl::Span<const TrackNode> tracks) {
  PresentParents present;
  const std::vector<int> parents = ComputeTrackParents(tracks);
  for (int i = 0; i + 1 < static_cast<int>(tracks.size()); ++i) {
    if (tracks[i + 1].nesting_level <= tracks[i].nesting_level) continue;
    std::vector<absl::string_view> name_path;
    std::vector<SiblingKeyView> key_path;
    for (int ancestor = i; ancestor >= 0; ancestor = parents[ancestor]) {
      name_path.push_back(tracks[ancestor].name);
      key_path.push_back(GetSiblingKey(tracks, parents, ancestor));
    }
    absl::c_reverse(name_path);
    absl::c_reverse(key_path);
    present.name_paths.insert(std::move(name_path));
    present.key_paths.insert(std::move(key_path));
  }
  return present;
}

absl::string_view TrackOrder::ModeName(Mode mode) {
  switch (mode) {
    case Mode::kCustom:
      return "custom";
    case Mode::kTraceOrder:
      return "trace_order";
    case Mode::kAlphabeticalAsc:
      return "alphabetical_asc";
    case Mode::kAlphabeticalDesc:
      return "alphabetical_desc";
  }
  return "custom";
}

std::optional<TrackOrder::Mode> TrackOrder::ParseModeName(
    absl::string_view name) {
  for (const Mode mode : {Mode::kCustom, Mode::kTraceOrder,
                          Mode::kAlphabeticalAsc, Mode::kAlphabeticalDesc}) {
    if (name == ModeName(mode)) return mode;
  }
  // Accept "default" as an alias for trace order for backward compatibility.
  if (name == "default") return Mode::kTraceOrder;
  return std::nullopt;
}
bool TrackOrder::ParentPathLess(const ParentPath& a, const ParentPath& b) {
  const size_t min_size = std::min(a.size(), b.size());
  for (size_t i = 0; i < min_size; ++i) {
    if (a[i].name != b[i].name) return a[i].name < b[i].name;
    if (a[i].ordinal != b[i].ordinal) return a[i].ordinal < b[i].ordinal;
  }
  return a.size() < b.size();
}

const std::vector<TrackOrder::SiblingKey>* TrackOrder::FindStoredOrder(
    const ParentPath& parent_path, absl::Span<const TrackNode> tracks,
    int begin, int end, const PresentParents& present) const {
  if (sibling_orders_.empty()) return nullptr;

  // 1. Exact (name, ordinal) ancestor path match.
  if (const auto it = sibling_orders_.find(parent_path);
      it != sibling_orders_.end()) {
    return &it->second;
  }

  // 2. Same ancestor names (ignoring ordinals) at the same nesting depth.
  // Candidates whose exact path belongs to a parent present in the current
  // tracks are skipped so that reordering one parent does not change others
  // with the same name.
  const std::vector<SiblingKey>* best_name_match = nullptr;
  const ParentPath* best_name_match_path = nullptr;
  for (const auto& [candidate_path, candidate_order] : sibling_orders_) {
    if (candidate_path.size() != parent_path.size()) continue;
    bool same_names = true;
    for (size_t i = 0; i < parent_path.size(); ++i) {
      if (candidate_path[i].name != parent_path[i].name) {
        same_names = false;
        break;
      }
    }
    if (!same_names) continue;

    std::vector<SiblingKeyView> candidate_key_path;
    candidate_key_path.reserve(candidate_path.size());
    for (const SiblingKey& key : candidate_path) {
      candidate_key_path.push_back({key.name, key.ordinal});
    }
    if (present.key_paths.contains(candidate_key_path)) continue;

    if (best_name_match_path == nullptr ||
        ParentPathLess(candidate_path, *best_name_match_path)) {
      best_name_match = &candidate_order;
      best_name_match_path = &candidate_path;
    }
  }
  if (best_name_match != nullptr) return best_name_match;

  // 3. Cross-session fallback: if `parent_path` is a non-root parent and a
  // stored entry at the same depth comes from a parent whose names are absent
  // from the current tracks (for example, "/device:TPU:0" in one session and
  // "/device:GPU:0" in another), match the stored order with the most child
  // names in common. Ties go to the entry with the smallest parent path.
  if (parent_path.empty()) return nullptr;
  const std::vector<Subtree> subtrees =
      SplitIntoSiblingSubtrees(tracks, begin, end);
  absl::flat_hash_set<absl::string_view> child_names;
  child_names.reserve(subtrees.size());
  for (const Subtree& subtree : subtrees) {
    if (!tracks[subtree.begin].name.empty()) {
      child_names.insert(tracks[subtree.begin].name);
    }
  }
  if (child_names.empty()) return nullptr;

  const std::vector<SiblingKey>* best_overlap_order = nullptr;
  const ParentPath* best_overlap_path = nullptr;
  int best_overlap_count = 0;
  for (const auto& [candidate_path, candidate_order] : sibling_orders_) {
    if (candidate_path.size() != parent_path.size()) continue;
    std::vector<absl::string_view> candidate_names;
    candidate_names.reserve(candidate_path.size());
    for (const SiblingKey& key : candidate_path) {
      candidate_names.push_back(key.name);
    }
    // Do not borrow a stored order that belongs to another parent actually
    // present in the current session's trace.
    if (present.name_paths.contains(candidate_names)) continue;

    int overlap = 0;
    absl::flat_hash_set<absl::string_view> counted;
    for (const SiblingKey& child_key : candidate_order) {
      if (child_names.contains(child_key.name) &&
          counted.insert(child_key.name).second) {
        ++overlap;
      }
    }
    if (overlap > best_overlap_count ||
        (overlap > 0 && overlap == best_overlap_count &&
         best_overlap_path != nullptr &&
         ParentPathLess(candidate_path, *best_overlap_path))) {
      best_overlap_count = overlap;
      best_overlap_order = &candidate_order;
      best_overlap_path = &candidate_path;
    }
  }
  return best_overlap_order;
}

std::vector<int> TrackOrder::RankChildren(
    absl::Span<const SiblingKeyView> keys,
    const std::vector<SiblingKey>* stored_order) {
  if (stored_order == nullptr) return {};
  const int stored_size = static_cast<int>(stored_order->size());
  absl::flat_hash_map<SiblingKeyView, int> stored_ranks;
  absl::flat_hash_map<absl::string_view, int> stored_name_ranks;
  stored_ranks.reserve(stored_size);
  stored_name_ranks.reserve(stored_size);
  for (int rank = 0; rank < stored_size; ++rank) {
    const SiblingKey& key = (*stored_order)[rank];
    stored_ranks.try_emplace(SiblingKeyView(key.name, key.ordinal), rank);
    stored_name_ranks.try_emplace(key.name, rank);
  }
  const absl::flat_hash_set<SiblingKeyView> present_keys(keys.begin(),
                                                         keys.end());
  std::vector<int> ranks(keys.size());
  for (int i = 0; i < static_cast<int>(keys.size()); ++i) {
    if (const auto rank = stored_ranks.find(keys[i]);
        rank != stored_ranks.end()) {
      ranks[i] = rank->second;
      continue;
    }
    // A child whose exact key is not stored takes the place of the first
    // stored sibling with the same name, if that sibling is absent. This keeps
    // the order of a track whose ordinal changed because a sibling with the
    // same name before it disappeared.
    if (const auto name_rank = stored_name_ranks.find(keys[i].first);
        name_rank != stored_name_ranks.end()) {
      const SiblingKey& stored_key = (*stored_order)[name_rank->second];
      if (!present_keys.contains(
              SiblingKeyView(stored_key.name, stored_key.ordinal))) {
        ranks[i] = name_rank->second;
        continue;
      }
    }
    ranks[i] = stored_size + i;
  }
  return ranks;
}

std::vector<int> TrackOrder::SortSiblings(absl::Span<const TrackNode> tracks,
                                          absl::Span<const int> roots,
                                          absl::Span<const int> ranks) const {
  std::vector<int> positions(roots.size());
  std::iota(positions.begin(), positions.end(), 0);
  if (mode_ == Mode::kTraceOrder ||
      (mode_ == Mode::kCustom && ranks.empty())) {
    return positions;
  }
  // In DFS pre-order, the nesting levels of consecutive siblings never
  // increase. Siblings are only reordered among those with the same level, so
  // that no track ever becomes part of another track's subtree. The sort is
  // stable, so siblings that compare equal keep their natural order.
  absl::c_stable_sort(positions, [&](int a, int b) {
    const TrackNode& track_a = tracks[roots[a]];
    const TrackNode& track_b = tracks[roots[b]];
    if (track_a.nesting_level != track_b.nesting_level) {
      return track_a.nesting_level > track_b.nesting_level;
    }
    switch (mode_) {
      case Mode::kAlphabeticalAsc:
        return CompareNamesCaseInsensitive(track_a.name, track_b.name) < 0;
      case Mode::kAlphabeticalDesc:
        return CompareNamesCaseInsensitive(track_a.name, track_b.name) > 0;
      case Mode::kCustom:
      case Mode::kTraceOrder:
        return ranks[a] < ranks[b];
    }
    return false;
  });
  return positions;
}
std::vector<int> TrackOrder::DisplayOrder(
    absl::Span<const TrackNode> tracks) const {
  std::vector<int> display_order;
  if (mode_ == Mode::kTraceOrder ||
      (mode_ == Mode::kCustom && sibling_orders_.empty())) {
    display_order.resize(tracks.size());
    std::iota(display_order.begin(), display_order.end(), 0);
    return display_order;
  }
  display_order.reserve(tracks.size());
  const PresentParents present = mode_ == Mode::kCustom
                                     ? CollectPresentParents(tracks)
                                     : PresentParents();
  ParentPath parent_path;
  AppendSiblings(tracks, 0, static_cast<int>(tracks.size()), parent_path,
                 present, display_order);
  return display_order;
}

void TrackOrder::AppendSiblings(absl::Span<const TrackNode> tracks, int begin,
                                int end, ParentPath& parent_path,
                                const PresentParents& present,
                                std::vector<int>& display_order,
                                std::vector<ParentPath>* visited_parents,
                                SiblingOrders* displayed_orders) const {
  if (visited_parents != nullptr) {
    visited_parents->push_back(parent_path);
  }
  const std::vector<Subtree> subtrees =
      SplitIntoSiblingSubtrees(tracks, begin, end);
  const std::vector<SiblingKeyView> keys = ComputeSiblingKeys(tracks, subtrees);
  const std::vector<int> ranks =
      mode_ == Mode::kCustom
          ? RankChildren(keys, FindStoredOrder(parent_path, tracks, begin, end,
                                               present))
          : std::vector<int>();
  const std::vector<int> positions =
      SortSiblings(tracks, SubtreeRoots(subtrees), ranks);

  if (displayed_orders != nullptr && !absl::c_is_sorted(positions)) {
    std::vector<SiblingKey>& displayed_order = (*displayed_orders)[parent_path];
    displayed_order.reserve(positions.size());
    for (const int i : positions) {
      displayed_order.push_back({std::string(keys[i].first), keys[i].second});
    }
  }

  for (const int i : positions) {
    const Subtree& subtree = subtrees[i];
    display_order.push_back(subtree.begin);
    if (subtree.end - subtree.begin > 1) {
      parent_path.push_back({std::string(keys[i].first), keys[i].second});
      AppendSiblings(tracks, subtree.begin + 1, subtree.end, parent_path,
                     present, display_order, visited_parents, displayed_orders);
      parent_path.pop_back();
    }
  }
}

bool TrackOrder::Move(absl::Span<const TrackNode> tracks, int source,
                      int target, DropPosition position) {
  const int size = static_cast<int>(tracks.size());
  if (source < 0 || source >= size || target < 0 || target >= size ||
      source == target ||
      tracks[source].nesting_level != tracks[target].nesting_level) {
    return false;
  }
  const std::vector<int> parents = ComputeTrackParents(tracks);
  const int parent = parents[source];
  if (parents[target] != parent) return false;

  const Subtree range = ChildRange(tracks, parent);
  const std::vector<Subtree> siblings =
      SplitIntoSiblingSubtrees(tracks, range.begin, range.end);
  const int source_index = FindSubtree(siblings, source);
  const int target_index = FindSubtree(siblings, target);
  if (source_index < 0 || target_index < 0) return false;
  const std::vector<SiblingKeyView> keys = ComputeSiblingKeys(tracks, siblings);

  ParentPath parent_path;
  for (int ancestor = parent; ancestor >= 0; ancestor = parents[ancestor]) {
    const auto [name, ordinal] = GetSiblingKey(tracks, parents, ancestor);
    parent_path.push_back({std::string(name), ordinal});
  }
  absl::c_reverse(parent_path);

  // The siblings in their current display order, computed in the same way as
  // `DisplayOrder()` does.
  const PresentParents present = CollectPresentParents(tracks);
  const std::vector<SiblingKey>* const stored_order =
      mode_ == Mode::kCustom
          ? FindStoredOrder(parent_path, tracks, range.begin, range.end,
                            present)
          : nullptr;
  const std::vector<int> ranks = RankChildren(keys, stored_order);
  const std::vector<int> displayed =
      SortSiblings(tracks, SubtreeRoots(siblings), ranks);

  std::vector<int> moved = displayed;
  moved.erase(absl::c_find(moved, source_index));
  auto insert_at = absl::c_find(moved, target_index);
  if (position == DropPosition::kAfter) ++insert_at;
  moved.insert(insert_at, source_index);
  if (moved == displayed) return false;

  // The new order of the children: the present ones in their new display
  // order, with each stored child that is absent from `tracks` kept right
  // after the child that it follows in the stored order, so that it returns
  // to its place if it comes back.
  const int stored_size =
      stored_order == nullptr ? 0 : static_cast<int>(stored_order->size());
  std::vector<int> child_of_entry(stored_size, -1);
  for (int i = 0; i < static_cast<int>(ranks.size()); ++i) {
    if (ranks[i] < stored_size) child_of_entry[ranks[i]] = i;
  }
  // `absent_after[i + 1]` holds the absent children that follow child `i`, and
  // `absent_after[0]` holds the ones that come first.
  std::vector<std::vector<SiblingKey>> absent_after(siblings.size() + 1);
  int previous_child = -1;
  for (int entry = 0; entry < stored_size; ++entry) {
    if (child_of_entry[entry] >= 0) {
      previous_child = child_of_entry[entry];
    } else {
      absent_after[previous_child + 1].push_back((*stored_order)[entry]);
    }
  }
  std::vector<SiblingKey> new_order = std::move(absent_after[0]);
  for (const int i : moved) {
    new_order.push_back({std::string(keys[i].first), keys[i].second});
    absl::c_move(absent_after[i + 1], std::back_inserter(new_order));
  }

  if (mode_ != Mode::kCustom) {
    // Starts the custom orders from the order that `mode_` displays.
    // Replace only the entries for parent paths that exist in `tracks`,
    // preserving saved orders for absent parents.
    std::vector<ParentPath> visited_parents;
    SiblingOrders displayed_orders;
    ParentPath root_path;
    std::vector<int> display_order;
    AppendSiblings(tracks, 0, size, root_path, present, display_order,
                   &visited_parents, &displayed_orders);
    for (const ParentPath& path : visited_parents) {
      sibling_orders_.erase(path);
    }
    for (auto& [path, order] : displayed_orders) {
      sibling_orders_.insert_or_assign(std::move(path), std::move(order));
    }
    mode_ = Mode::kCustom;
  }

  // Drop the entry once the children are back in their natural order, so that
  // the common case of no custom order stays on the fast path.
  const bool is_natural_order = absl::c_equal(
      new_order, keys, [](const SiblingKey& key, const SiblingKeyView& view) {
        return key.name == view.first && key.ordinal == view.second;
      });
  if (is_natural_order) {
    sibling_orders_.erase(parent_path);
    // If a fallback order would still match after removing the exact entry,
    // explicitly store the natural order so that the fallback does not
    // override it.
    if (FindStoredOrder(parent_path, tracks, range.begin, range.end, present) !=
        nullptr) {
      sibling_orders_.insert_or_assign(std::move(parent_path),
                                       std::move(new_order));
    }
  } else {
    sibling_orders_.insert_or_assign(std::move(parent_path),
                                     std::move(new_order));
  }
  return true;
}

std::string TrackOrder::Serialize() const {
  if (empty()) return "";

  std::vector<const std::pair<const ParentPath, std::vector<SiblingKey>>*>
      entries;
  entries.reserve(sibling_orders_.size());
  for (const auto& entry : sibling_orders_) {
    entries.push_back(&entry);
  }
  // Sorts the entries by parent path, so that the output is deterministic.
  absl::c_sort(entries, [](const auto* a, const auto* b) {
    return ParentPathLess(a->first, b->first);
  });

  auto append_key_list = [](std::string& out,
                             absl::Span<const SiblingKey> keys) {
    out.push_back('[');
    for (size_t i = 0; i < keys.size(); ++i) {
      if (i > 0) out.push_back(',');
      out.push_back('[');
      AppendJsonString(out, keys[i].name);
      absl::StrAppend(&out, ",", keys[i].ordinal, "]");
    }
    out.push_back(']');
  };

  std::string out = R"({"mode":)";
  AppendJsonString(out, ModeName(mode_));
  out.append(R"(,"orders":[)");
  for (size_t i = 0; i < entries.size(); ++i) {
    if (i > 0) out.push_back(',');
    out.append(R"({"parent":)");
    append_key_list(out, entries[i]->first);
    out.append(R"(,"order":)");
    append_key_list(out, entries[i]->second);
    out.push_back('}');
  }
  out.append("]}");
  return out;
}

bool TrackOrder::Deserialize(absl::string_view json) {
  json = absl::StripAsciiWhitespace(json);
  if (json.empty()) {
    sibling_orders_.clear();
    return true;
  }

  JsonParser parser(json);

  // Parses the comma-separated members of an object, after its opening brace,
  // up to and including its closing brace. Calls `parse_member` with the name
  // of each member, positioned at its value.
  auto parse_object_members =
      [&parser](absl::FunctionRef<bool(absl::string_view)> parse_member) {
        if (parser.Consume('}')) return true;
        while (true) {
          std::string name;
          if (!parser.ParseString(name) || !parser.Consume(':') ||
              !parse_member(name)) {
            return false;
          }
          if (parser.Consume('}')) return true;
          if (!parser.Consume(',')) return false;
        }
      };

  auto parse_key_list = [&parser](std::vector<SiblingKey>& out) {
    out.clear();
    if (!parser.Consume('[')) return false;
    if (parser.Consume(']')) return true;
    while (true) {
      if (!parser.Consume('[')) return false;
      std::string name;
      int ordinal = 0;
      if (!parser.ParseString(name)) return false;
      if (parser.Consume(',')) {
        if (!parser.ParseNonNegativeInt(ordinal)) return false;
      }
      if (!parser.Consume(']')) return false;
      out.push_back({std::move(name), ordinal});
      if (parser.Consume(']')) return true;
      if (!parser.Consume(',')) return false;
    }
  };

  Mode mode = Mode::kCustom;
  SiblingOrders parsed_orders;
  auto parse_root_member = [&](absl::string_view name) {
    if (name == "mode") {
      std::string mode_name;
      if (!parser.ParseString(mode_name)) return false;
      mode = ParseModeName(mode_name).value_or(Mode::kCustom);
      return true;
    }
    if (name != "orders") return parser.SkipValue();
    if (!parser.Consume('[')) return false;
    if (parser.Consume(']')) return true;
    while (true) {
      if (!parser.Consume('{')) return false;
      ParentPath parent_path;
      std::vector<SiblingKey> order;
      bool has_order = false;
      const bool ok = parse_object_members([&](absl::string_view member) {
        if (member == "parent") return parse_key_list(parent_path);
        if (member == "order") {
          has_order = true;
          return parse_key_list(order);
        }
        return parser.SkipValue();
      });
      if (!ok) return false;
      if (has_order) {
        // Drop duplicates within the same sibling list, keeping the first
        // appearance of each key, so malformed or hand-edited JSON cannot
        // produce invalid ranks.
        absl::flat_hash_set<SiblingKey> seen;
        std::vector<SiblingKey> deduplicated;
        deduplicated.reserve(order.size());
        for (SiblingKey& key : order) {
          if (seen.insert(key).second) {
            deduplicated.push_back(std::move(key));
          }
        }
        parsed_orders.insert_or_assign(std::move(parent_path),
                                       std::move(deduplicated));
      }
      if (parser.Consume(']')) return true;
      if (!parser.Consume(',')) return false;
    }
  };

  if (!parser.Consume('{') || !parse_object_members(parse_root_member) ||
      !parser.AtEnd()) {
    return false;
  }
  mode_ = mode;
  sibling_orders_ = std::move(parsed_orders);
  return true;
}

}  // namespace traceviewer
