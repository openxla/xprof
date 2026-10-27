#ifndef THIRD_PARTY_XPROF_FRONTEND_APP_COMPONENTS_TRACE_VIEWER_V2_TIMELINE_TRACK_ORDER_H_
#define THIRD_PARTY_XPROF_FRONTEND_APP_COMPONENTS_TRACE_VIEWER_V2_TIMELINE_TRACK_ORDER_H_

#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"

namespace traceviewer {

// Minimal node data used by `TrackOrder` to reason about timeline tracks.
struct TrackNode {
  int nesting_level = 0;
  std::string name;
};

// Computes the parent index of each track in DFS pre-order. Returns -1 for
// root tracks.
std::vector<int> ComputeTrackParents(absl::Span<const TrackNode> tracks);

// Returns the index one past the last track of the subtree rooted at
// `tracks[index]`. If `index` is out of range, returns it clamped to
// [0, `tracks.size()`].
int TrackSubtreeEnd(absl::Span<const TrackNode> tracks, int index);

// A user-defined order of timeline tracks.
//
// Tracks are organized in a hierarchy (processes, threads, lanes) and laid out
// in depth-first pre-order. `TrackOrder` lets users reorder sibling tracks at
// any level without breaking the tree structure: each moved track carries its
// entire subtree with it, and siblings are only reordered among those with the
// same nesting level.
//
// Because the viewer loads trace data incrementally and tracks may appear or
// disappear as the user pans, `TrackOrder` identifies tracks by name rather
// than by index. Siblings with identical names are disambiguated by an
// ordinal (0 for the first sibling with that name, 1 for the second, etc.).
//
// Reordering is applied on top of the natural order of the data:
//  - Any parent whose children have not been explicitly reordered displays its
//    children in their natural order.
//  - If new children appear under a reordered parent, they are displayed after
//    the reordered children, in their natural order.
//  - An empty `TrackOrder` is a no-op: `DisplayOrder()` returns the identity
//    permutation.
class TrackOrder {
 public:
  // How the children of every track are ordered.
  enum class Mode {
    // Orders the children by the custom order of their parent, if any (see
    // `Move()`), and otherwise in their natural order.
    kCustom,
    // Orders all the tracks in their natural order (the order defined in the
    // trace data). The custom orders are kept, so that switching back to
    // `Mode::kCustom` restores them.
    kTraceOrder,
    // Sorts the children by name, from A to Z, ignoring ASCII case.
    kAlphabeticalAsc,
    // Sorts the children by name, from Z to A, ignoring ASCII case.
    kAlphabeticalDesc,
  };

  // Where a moved track is placed relative to its target sibling.
  enum class DropPosition { kBefore, kAfter };

  TrackOrder() = default;

  // Returns the stable name of `mode` that is used in the serialized JSON and
  // in the JavaScript API: "custom", "trace_order", "alphabetical_asc", or
  // "alphabetical_desc".
  static absl::string_view ModeName(Mode mode);

  // Parses a name returned by `ModeName()`. Returns `std::nullopt` if `name`
  // is not a known mode name.
  static std::optional<Mode> ParseModeName(absl::string_view name);

  Mode mode() const { return mode_; }
  void set_mode(Mode mode) { mode_ = mode; }

  // Returns true if any parent has a custom order of its children, whether or
  // not `mode()` displays it.
  bool HasCustomOrders() const { return !sibling_orders_.empty(); }

  // Returns true if this order is in its initial state: `Mode::kCustom` with
  // no custom orders, which displays all the tracks in their natural order.
  bool empty() const {
    return mode_ == Mode::kCustom && sibling_orders_.empty();
  }

  // Forgets all custom orders and resets `mode()` to `Mode::kCustom`.
  void Clear() {
    sibling_orders_.clear();
    mode_ = Mode::kCustom;
  }

  // Returns a permutation of the indices of `tracks` (which must be in DFS
  // pre-order) that lists the tracks in display order: each track is followed
  // by its subtree, and the children of every parent are ordered according to
  // `mode()`. Returns the identity permutation if the tracks are displayed in
  // their natural order.
  std::vector<int> DisplayOrder(absl::Span<const TrackNode> tracks) const;

  // Moves the subtree rooted at `tracks[source]` right before its sibling
  // `tracks[target]` (`DropPosition::kBefore`), or right after it and its
  // subtree (`DropPosition::kAfter`), in the current display order. Siblings
  // are tracks with the same parent and nesting level.
  //
  // If `mode()` is not `Mode::kCustom`, first replaces the custom orders for
  // parents present in `tracks` with the order that `mode()` displays and
  // switches to `Mode::kCustom`, so that no other track changes position while
  // preserving saved orders for absent parents.
  //
  // Returns true if the display order changed. Returns false and changes
  // nothing if the indices are out of range, the tracks are not siblings, or
  // the track is already at that position.
  bool Move(absl::Span<const TrackNode> tracks, int source, int target,
            DropPosition position);

  // Serializes `mode()` and the custom orders into compact, deterministic
  // JSON, for example:
  //
  //   {"mode":"custom","orders":[
  //     {"parent":[["/device:TPU:0",0]],"order":[["XLA Ops",0],["Steps",0]]}]}
  //
  // Each key is a track name and how many earlier siblings have the same name.
  // Returns an empty string if `empty()`, as there is nothing to save, which
  // `Deserialize()` reads back as no custom order.
  std::string Serialize() const;

  // Restores `mode()` and the custom orders from JSON produced by
  // `Serialize()`. Missing members get their defaults (no `"mode"`: custom,
  // no `"orders"` or no `"order"`: no custom order, no `"parent"`: the root
  // tracks, no ordinal: 0), an unknown mode is read as custom, and unknown
  // members are skipped, so that orders saved by other versions of the viewer
  // can be read. Returns false and changes nothing if `json` is malformed.
  bool Deserialize(absl::string_view json);

 private:
  // A (name, ordinal) key of a track among its siblings.
  struct SiblingKey {
    std::string name;
    int ordinal = 0;

    bool operator==(const SiblingKey& other) const {
      return ordinal == other.ordinal && name == other.name;
    }

    template <typename H>
    friend H AbslHashValue(H h, const SiblingKey& key) {
      return H::combine(std::move(h), key.name, key.ordinal);
    }
  };

  // A (name, ordinal) key of a track among its siblings, as a string view.
  using SiblingKeyView = std::pair<absl::string_view, int>;

  // Identifies a parent track by the keys of the tracks on the path from its
  // root track down to it. The empty path stands for the (virtual) parent of
  // the root tracks.
  using ParentPath = std::vector<SiblingKey>;

  // The custom order of the children of each parent.
  using SiblingOrders =
      absl::flat_hash_map<ParentPath, std::vector<SiblingKey>>;

  // The parents in the current tracks (see `CollectPresentParents()` in the .cc
  // file).
  struct PresentParents {
    // The names of the tracks on the path from the root down to each parent.
    absl::flat_hash_set<std::vector<absl::string_view>> name_paths;
    // The sibling keys of the tracks on the path from the root down to each
    // parent.
    absl::flat_hash_set<std::vector<SiblingKeyView>> key_paths;
  };

  // Collects the paths of the parent tracks present in `tracks`.
  static PresentParents CollectPresentParents(
      absl::Span<const TrackNode> tracks);

  // Orders parent paths lexicographically by the names and ordinals of their
  // tracks.
  static bool ParentPathLess(const ParentPath& a, const ParentPath& b);

  // Returns the rank of each of the children with sibling keys `keys` in
  // `stored_order`: the index of the entry that the child matches, either
  // exactly or, if its key is not stored, by the first stored entry with the
  // same name, provided that entry is absent from `keys`. Children that match
  // no entry rank after all the entries, in their natural order. Returns an
  // empty vector if `stored_order` is null.
  static std::vector<int> RankChildren(
      absl::Span<const SiblingKeyView> keys,
      const std::vector<SiblingKey>* stored_order);

  // Looks up the custom order of the children of `parent_path`, whose
  // children are `tracks[begin, end)`. If there is no exact match, falls back
  // to the custom order of a parent with the same names but other ordinals,
  // and then, for a non-root parent, to the custom order at the same depth
  // with the most child names in common. Both fallbacks only use custom orders
  // of parents that are absent from the current tracks (`present`), which
  // covers other traces, and also a parent that is missing from the fetched
  // data for a while. Returns null if there is no match.
  const std::vector<SiblingKey>* FindStoredOrder(
      const ParentPath& parent_path, absl::Span<const TrackNode> tracks,
      int begin, int end, const PresentParents& present) const;

  // Returns the display order of the sibling subtrees that start at the
  // indices `roots` of `tracks`, as indices into `roots`. `ranks` (see
  // `RankChildren()`) is used in `Mode::kCustom`, unless it is empty.
  std::vector<int> SortSiblings(absl::Span<const TrackNode> tracks,
                                absl::Span<const int> roots,
                                absl::Span<const int> ranks) const;

  // Appends the sibling subtrees in `tracks[begin, end)`, the children of
  // `parent_path`, to `display_order` in display order, recursing into their
  // subtrees. If `visited_parents` and `displayed_orders` are not null, records
  // each visited parent and adds the display order of its children if not in
  // natural order.
  void AppendSiblings(absl::Span<const TrackNode> tracks, int begin, int end,
                      ParentPath& parent_path, const PresentParents& present,
                      std::vector<int>& display_order,
                      std::vector<ParentPath>* visited_parents = nullptr,
                      SiblingOrders* displayed_orders = nullptr) const;

  Mode mode_ = Mode::kCustom;
  SiblingOrders sibling_orders_;
};

}  // namespace traceviewer

#endif  // THIRD_PARTY_XPROF_FRONTEND_APP_COMPONENTS_TRACE_VIEWER_V2_TIMELINE_TRACK_ORDER_H_
