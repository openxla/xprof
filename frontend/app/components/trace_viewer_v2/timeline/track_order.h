#ifndef THIRD_PARTY_XPROF_FRONTEND_APP_COMPONENTS_TRACE_VIEWER_V2_TIMELINE_TRACK_ORDER_H_
#define THIRD_PARTY_XPROF_FRONTEND_APP_COMPONENTS_TRACE_VIEWER_V2_TIMELINE_TRACK_ORDER_H_

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
  // Where a moved track is placed relative to its target sibling.
  enum class DropPosition { kBefore, kAfter };

  TrackOrder() = default;

  // Returns true if no custom orders have been set.
  bool empty() const { return sibling_orders_.empty(); }

  // Removes all custom orders, returning all tracks to their natural order.
  void Clear() { sibling_orders_.clear(); }

  // Returns a permutation of indices into `tracks` that orders tracks
  // according to the custom order. The returned vector has the same size as
  // `tracks`.
  std::vector<int> DisplayOrder(absl::Span<const TrackNode> tracks) const;

  // Moves the sibling subtree rooted at `source` before or after `target`.
  //
  // `source` and `target` must be distinct sibling tracks (same parent and
  // same nesting level). The entire subtree rooted at `source` moves with it.
  //
  // Returns true if the order changed, or false if the move was invalid
  // (e.g., different parents, different nesting levels, or out of range) or
  // would not change the current display order.
  bool Move(absl::Span<const TrackNode> tracks, int source, int target,
            DropPosition position);

  // Serializes the custom order to a compact JSON string.
  //
  // The format stores each parent path and the ordered list of its children:
  //
  //   {"orders":[
  //     {"parent":[["/device:TPU:0",0]],"order":[["XLA Ops",0],["Steps",0]]}]}
  //
  // Each key is a track name and how many earlier siblings have the same name.
  // Returns an empty string if `empty()`, as there is nothing to save, which
  // `Deserialize()` reads back as no custom order.
  std::string Serialize() const;

  // Restores the custom orders from `json`, produced by `Serialize()`. To read
  // orders saved by other versions of the viewer, missing members get default
  // values and unknown members are skipped: a missing `"parent"` stands for the
  // root tracks, a missing ordinal is 0, and entries without an `"order"` are
  // ignored. Returns false and changes nothing if `json` is malformed.
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

  // Appends the sibling subtrees in `tracks[begin, end)`, the children of
  // `parent_path`, to `display_order` in display order, recursing into their
  // subtrees.
  void AppendSiblings(absl::Span<const TrackNode> tracks, int begin, int end,
                      ParentPath& parent_path, const PresentParents& present,
                      std::vector<int>& display_order) const;

  SiblingOrders sibling_orders_;
};

}  // namespace traceviewer

#endif  // THIRD_PARTY_XPROF_FRONTEND_APP_COMPONENTS_TRACE_VIEWER_V2_TIMELINE_TRACK_ORDER_H_
