#ifndef THIRD_PARTY_XPROF_FRONTEND_APP_COMPONENTS_TRACE_VIEWER_V2_TIMELINE_TIMELINE_H_
#define THIRD_PARTY_XPROF_FRONTEND_APP_COMPONENTS_TRACE_VIEWER_V2_TIMELINE_TIMELINE_H_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <limits>
#include <map>
#include <optional>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/functional/any_invocable.h"
#include "absl/functional/function_ref.h"
#include "absl/log/check.h"
#include "absl/numeric/bits.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "imgui.h"
#include "imgui_internal.h"
#include "tsl/profiler/lib/context_types.h"
#include "frontend/app/components/trace_viewer_v2/animation.h"
#include "frontend/app/components/trace_viewer_v2/color/colors.h"
#include "frontend/app/components/trace_viewer_v2/event_data.h"
#include "frontend/app/components/trace_viewer_v2/timeline/constants.h"
#include "frontend/app/components/trace_viewer_v2/timeline/time_range.h"
#include "frontend/app/components/trace_viewer_v2/timeline/timeline_hlo_dependencies.h"
#include "frontend/app/components/trace_viewer_v2/trace_helper/trace_event.h"

namespace traceviewer {

namespace FlowCategoryFilter {
static constexpr int kAll = -1;
static constexpr int kNone = -2;
}  // namespace FlowCategoryFilter

enum class MouseMode {
  // Select events in an area (legacy mode 1)
  kSelect = 1,
  // Pan the view (legacy mode 2)
  kPan = 2,
  // Zoom the view (legacy mode 3)
  kZoom = 3,
  // Create or move markers (legacy mode 4)
  kTiming = 4
};

// Represents a rectangle on the screen.
struct EventRect {
  Pixel left = 0.0f;
  Pixel top = 0.0f;
  Pixel right = 0.0f;
  Pixel bottom = 0.0f;
};

struct DeleteButtonLayout {
  ImVec2 button_pos;
  ImRect hover_rect;
  bool text_fits = false;
};

struct CounterData {
  std::vector<Microseconds> timestamps;
  std::vector<double> values;
  double min_value = std::numeric_limits<double>::max();
  double max_value = std::numeric_limits<double>::lowest();
  std::string event_stats;
  CounterUnit unit = CounterUnit::kNone;
};

// Represents a stable key for identifying a group/track across data reloads.
struct GroupKey {
  int nesting_level = 0;
  std::string name;
  std::string parent_name;

  bool operator==(const GroupKey& other) const {
    return nesting_level == other.nesting_level && name == other.name &&
           parent_name == other.parent_name;
  }

  bool operator<(const GroupKey& other) const {
    return std::tie(nesting_level, name, parent_name) <
           std::tie(other.nesting_level, other.name, other.parent_name);
  }

  template <typename H>
  friend H AbslHashValue(H h, const GroupKey& k) {
    return H::combine(std::move(h), k.nesting_level, k.name, k.parent_name);
  }
};

// Represents a grouping of timeline tracks, such as processes, threads, or
// counters.
struct Group {
  enum class Type { kFlame, kCounter };
  Type type = Type::kFlame;
  std::string name;
  std::string subtitle;
  // The start level of the groups of complete events.
  // For flame groups, we increment the group level by real events' levels.
  // For counter groups, we increment the group level by 1.
  int start_level = 0;
  int nesting_level = 0;
  bool expanded = false;
  // Parent index in groups vector, or -1 for top-level processes.
  int parent_index = -1;
  // List of child process/thread indices in the groups vector.
  // Typically 2-10 children per process group.
  std::vector<int> child_indices = {};

  // Stable index in the original sequential order.
  int original_index = -1;

  // Number of timeline event levels occupied by this track.
  int level_count = 0;
  // Indicates if this group has nested child tracks.
  bool has_children = false;

  // Cached layout offset (screen Y coordinate in pixels).
  mutable Pixel offset = 0.0f;
  // Cached full height (in pixels) of the track based on level count.
  mutable Pixel height = 0.0f;
  // Indicates if the track is visible (not hidden by a collapsed parent).
  mutable bool visible = true;

  ProcessId pid = 0;
  ThreadId tid = 0;
};

struct FlowLine {
  Microseconds source_ts = 0.0;

  Microseconds target_ts = 0.0;

  int source_level = 0;

  int target_level = 0;

  uint32_t color = traceviewer::kBlackColor;
  tsl::profiler::ContextType category = tsl::profiler::ContextType::kGeneric;
};

// Group IDs of flame-chart entries, stored as sorted, disjoint, half-open
// runs of entry indices that share one group ID. An entry index outside
// every run has no group ID.
class GroupIdRuns {
 public:
  struct Run {
    uint32_t begin = 0;  // First entry index of the run.
    uint32_t end = 0;    // One past the last entry index of the run.
    int64_t group_id = 0;
    friend bool operator==(const Run&, const Run&) = default;
  };

  // Records `group_id` for entry `index`. Indices must strictly increase
  // across calls; gaps mean "no group ID".
  void Append(size_t index, int64_t group_id) {
    DCHECK_LT(index, std::numeric_limits<uint32_t>::max());
    DCHECK(runs_.empty() || index >= runs_.back().end);
    if (index >= std::numeric_limits<uint32_t>::max() ||
        (!runs_.empty() && index < runs_.back().end)) {
      return;
    }
    const uint32_t i = static_cast<uint32_t>(index);
    if (!runs_.empty() && runs_.back().end == i &&
        runs_.back().group_id == group_id) {
      ++runs_.back().end;
      return;
    }
    runs_.push_back({.begin = i, .end = i + 1, .group_id = group_id});
  }

  // Returns the group ID of entry `index`, or nullopt if it has none.
  std::optional<int64_t> Find(size_t index) const {
    const auto it = std::upper_bound(
        runs_.begin(), runs_.end(), index,
        [](size_t i, const Run& run) { return i < run.begin; });
    if (it == runs_.begin()) {
      return std::nullopt;
    }
    const auto prev_it = std::prev(it);
    if (index >= prev_it->end) {
      return std::nullopt;
    }
    return prev_it->group_id;
  }

  bool empty() const { return runs_.empty(); }
  void clear() { runs_.clear(); }
  void release() { std::vector<Run>().swap(runs_); }
  size_t AllocatedBytes() const { return runs_.capacity() * sizeof(Run); }
  absl::Span<const Run> runs() const { return runs_; }
  friend bool operator==(const GroupIdRuns&, const GroupIdRuns&) = default;

 private:
  std::vector<Run> runs_;
};
static_assert(sizeof(GroupIdRuns::Run) == 16);

// Non-zero table indices (such as HLO op or HLO module indices) of
// flame-chart entries, stored as sorted, disjoint, half-open runs of entry
// indices that share one uint32_t value. An entry index outside every run
// has value 0.
class IndexRuns {
 public:
  struct Run {
    uint32_t begin = 0;  // First entry index of the run.
    uint32_t end = 0;    // One past the last entry index of the run.
    uint32_t value = 0;
    friend bool operator==(const Run&, const Run&) = default;
  };

  void Append(size_t index, uint32_t value) {
    if (value == 0) {
      return;
    }
    DCHECK_LT(index, std::numeric_limits<uint32_t>::max());
    DCHECK(runs_.empty() || index >= runs_.back().end);
    if (index >= std::numeric_limits<uint32_t>::max() ||
        (!runs_.empty() && index < runs_.back().end)) {
      return;
    }
    const uint32_t i = static_cast<uint32_t>(index);
    if (!runs_.empty() && runs_.back().end == i &&
        runs_.back().value == value) {
      ++runs_.back().end;
      return;
    }
    runs_.push_back({.begin = i, .end = i + 1, .value = value});
  }

  std::optional<uint32_t> Find(size_t index) const {
    const auto it = std::upper_bound(
        runs_.begin(), runs_.end(), index,
        [](size_t i, const Run& run) { return i < run.begin; });
    if (it == runs_.begin()) {
      return std::nullopt;
    }
    const auto prev_it = std::prev(it);
    if (index >= prev_it->end) {
      return std::nullopt;
    }
    return prev_it->value;
  }

  bool empty() const { return runs_.empty(); }
  void clear() { runs_.clear(); }
  void release() { std::vector<Run>().swap(runs_); }
  size_t AllocatedBytes() const { return runs_.capacity() * sizeof(Run); }
  absl::Span<const Run> runs() const { return runs_; }
  friend bool operator==(const IndexRuns&, const IndexRuns&) = default;

 private:
  std::vector<Run> runs_;
};
static_assert(sizeof(IndexRuns::Run) == 12);

// Compact chunked table of self-time overrides for non-leaf flame-chart
// entries whose children subtract from total duration. Leaf entries have
// self_time == total_time and are not stored.
class SparseSelfTimes {
 public:
  struct Record {
    uint32_t entry_index = 0;
    uint32_t self_time_lo = 0;
    uint32_t self_time_hi = 0;

    Microseconds GetSelfTime() const {
      const uint64_t bits = static_cast<uint64_t>(self_time_lo) |
                            (static_cast<uint64_t>(self_time_hi) << 32);
      Microseconds val = 0.0;
      std::memcpy(&val, &bits, sizeof(val));
      return val;
    }

    void SetSelfTime(Microseconds val) {
      uint64_t bits = 0;
      std::memcpy(&bits, &val, sizeof(bits));
      self_time_lo = static_cast<uint32_t>(bits);
      self_time_hi = static_cast<uint32_t>(bits >> 32);
    }

    friend bool operator==(const Record&, const Record&) = default;
  };
  static_assert(sizeof(Record) == 12);
  static_assert(alignof(Record) == 4);

  static constexpr uint32_t kNone = std::numeric_limits<uint32_t>::max();
  static constexpr size_t kBlockShift = 16;
  static constexpr size_t kBlockSize = size_t{1} << kBlockShift;  // 65,536
  static constexpr size_t kBlockMask = kBlockSize - 1;

  uint32_t Append(size_t entry_index, Microseconds self_time) {
    DCHECK_LT(entry_index, std::numeric_limits<uint32_t>::max());
    DCHECK(size_ == 0 || entry_index > RecordAt(size_ - 1).entry_index);
    if (entry_index >= std::numeric_limits<uint32_t>::max() ||
        (size_ > 0 && entry_index <= RecordAt(size_ - 1).entry_index)) {
      return kNone;
    }
    const size_t block_idx = size_ >> kBlockShift;
    if (block_idx == blocks_.size()) {
      std::vector<Record> block;
      block.reserve(kBlockSize);
      blocks_.push_back(std::move(block));
    }
    Record rec{.entry_index = static_cast<uint32_t>(entry_index)};
    rec.SetSelfTime(self_time);
    blocks_[block_idx].push_back(rec);
    return static_cast<uint32_t>(size_++);
  }

  void Subtract(uint32_t record_index, Microseconds child_duration) {
    if (record_index >= size_) return;
    Record& rec = RecordAt(record_index);
    rec.SetSelfTime(
        std::max<Microseconds>(0.0, rec.GetSelfTime() - child_duration));
  }

  std::optional<Microseconds> Find(size_t entry_index) const {
    if (size_ == 0 || entry_index > std::numeric_limits<uint32_t>::max()) {
      return std::nullopt;
    }
    const uint32_t target = static_cast<uint32_t>(entry_index);
    size_t lo = 0;
    size_t hi = size_;
    while (lo < hi) {
      const size_t mid = lo + ((hi - lo) >> 1);
      const Record& rec = RecordAt(mid);
      if (rec.entry_index < target) {
        lo = mid + 1;
      } else {
        hi = mid;
      }
    }
    if (lo < size_) {
      const Record& rec = RecordAt(lo);
      if (rec.entry_index == target) {
        return rec.GetSelfTime();
      }
    }
    return std::nullopt;
  }

  bool empty() const { return size_ == 0; }
  size_t size() const { return size_; }
  void clear() {
    blocks_.clear();
    size_ = 0;
  }
  void release() {
    std::vector<std::vector<Record>>().swap(blocks_);
    size_ = 0;
  }
  size_t AllocatedBytes() const {
    size_t bytes = blocks_.capacity() * sizeof(std::vector<Record>);
    for (const std::vector<Record>& block : blocks_) {
      bytes += block.capacity() * sizeof(Record);
    }
    return bytes;
  }

 private:
  Record& RecordAt(size_t record_index) {
    return blocks_[record_index >> kBlockShift][record_index & kBlockMask];
  }
  const Record& RecordAt(size_t record_index) const {
    return blocks_[record_index >> kBlockShift][record_index & kBlockMask];
  }

  std::vector<std::vector<Record>> blocks_;
  size_t size_ = 0;
};

// Compact bitset of non-negative event indices used for search match
// highlighting without allocating hash-set nodes per matching event.
class EntryBitset {
 public:
  void Reserve(size_t max_entries) {
    const size_t needed_words = (max_entries + 63) >> 6;
    if (needed_words > words_.size()) {
      words_.resize(needed_words, 0);
    }
  }

  void insert(int index) {
    if (index < 0) return;
    const size_t word_idx = static_cast<size_t>(index) >> 6;
    const uint64_t mask = uint64_t{1} << (static_cast<size_t>(index) & 63);
    if (word_idx >= words_.size()) {
      words_.resize(word_idx + 1, 0);
    }
    if ((words_[word_idx] & mask) == 0) {
      words_[word_idx] |= mask;
      ++size_;
    }
  }

  void push_back(int index) { insert(index); }

  bool contains(int index) const {
    if (index < 0) return false;
    const size_t word_idx = static_cast<size_t>(index) >> 6;
    if (word_idx >= words_.size()) return false;
    const uint64_t mask = uint64_t{1} << (static_cast<size_t>(index) & 63);
    return (words_[word_idx] & mask) != 0;
  }

  template <typename Fn>
  void ForEachSetBit(Fn&& fn) const {
    for (size_t word_idx = 0; word_idx < words_.size(); ++word_idx) {
      uint64_t word = words_[word_idx];
      while (word != 0) {
        const int bit = absl::countr_zero(word);
        const int index = static_cast<int>((word_idx << 6) + bit);
        if constexpr (std::is_convertible_v<decltype(fn(index)), bool>) {
          if (!fn(index)) return;
        } else {
          fn(index);
        }
        word &= (word - 1);
      }
    }
  }

  void IntersectWith(const EntryBitset& other) {
    const size_t min_words = std::min(words_.size(), other.words_.size());
    size_t new_size = 0;
    for (size_t i = 0; i < min_words; ++i) {
      words_[i] &= other.words_[i];
      new_size += static_cast<size_t>(absl::popcount(words_[i]));
    }
    for (size_t i = min_words; i < words_.size(); ++i) {
      words_[i] = 0;
    }
    size_ = new_size;
  }

  void UnionWith(const EntryBitset& other) {
    if (other.words_.size() > words_.size()) {
      words_.resize(other.words_.size(), 0);
    }
    size_t new_size = 0;
    for (size_t i = 0; i < other.words_.size(); ++i) {
      words_[i] |= other.words_[i];
      new_size += static_cast<size_t>(absl::popcount(words_[i]));
    }
    for (size_t i = other.words_.size(); i < words_.size(); ++i) {
      new_size += static_cast<size_t>(absl::popcount(words_[i]));
    }
    size_ = new_size;
  }

  void clear() {
    words_.clear();
    size_ = 0;
  }

  void release() {
    std::vector<uint64_t>().swap(words_);
    size_ = 0;
  }

  bool empty() const { return size_ == 0; }
  size_t size() const { return size_; }

 private:
  std::vector<uint64_t> words_;
  size_t size_ = 0;
};

// Represents a half-open [start, end) XLA module interval on a process.
struct ModuleInterval {
  Microseconds start = 0.0;
  Microseconds end = 0.0;
  uint32_t module_index = 0;

  friend bool operator==(const ModuleInterval&,
                         const ModuleInterval&) = default;
};

// Holds all the data required to render a flame chart and counter lines,
// including event timing, grouping information, and mappings between levels
// and events.
struct FlameChartTimelineData {
  struct FlowLineRange {
    uint32_t begin = 0;
    uint32_t end = 0;
    friend bool operator==(const FlowLineRange&,
                           const FlowLineRange&) = default;
  };

  // Compact per-track rule for read-time HLO op and HLO module derivation on
  // the protobuf ingestion path, avoiding dense or per-event RLE HLO storage.
  struct HloTrackRule {
    size_t begin_entry = 0;
    size_t end_entry = 0;
    uint32_t default_module_index = 0;
    uint32_t module_intervals_index = std::numeric_limits<uint32_t>::max();
    bool is_xla_ops = false;
    bool use_module_intervals = false;
  };

  // Global cumulative flame chart row index across all process, counter,
  // thread, and async tracks. Capped at 65,534; events at level < 0 or
  // level >= std::numeric_limits<uint16_t>::max() (65,535) are dropped.
  std::vector<uint16_t> entry_levels;
  std::vector<Microseconds> entry_total_times;
  std::vector<Microseconds> entry_self_times;
  std::vector<Microseconds> entry_start_times;
  std::vector<uint32_t> entry_name_indices;
  absl::flat_hash_map<int, uint32_t> non_zero_uids_by_event_index;
  std::vector<uint32_t> entry_hlo_module_indices;
  std::vector<uint32_t> entry_hlo_op_indices;
  std::vector<std::string> entry_names;
  std::vector<EventId> entry_event_ids;
  std::vector<absl::flat_hash_map<std::string, std::string>> entry_args;
  std::vector<Group> groups;
  std::vector<int> level_event_indices;
  std::vector<size_t> level_offsets;
  std::vector<FlowLine> flow_lines;
  // Map from event_id to list of flow ids that connect to this event.
  absl::flat_hash_map<EventId, std::vector<std::string>> flow_ids_by_event_id;
  // Map from event index to flow id that connects to this event.
  absl::flat_hash_map<int, uint64_t> flow_ids_by_event_index;
  // Map from flow_id to list of flow lines that belong to this flow.
  absl::flat_hash_map<std::string, std::vector<FlowLine>> flow_lines_by_flow_id;
  // Contiguous [begin, end) slices into `flow_lines` keyed by numeric flow_id
  // on the protobuf ingestion path, avoiding duplicate FlowLine copies and
  // heap-allocated string keys.
  absl::flat_hash_map<uint64_t, FlowLineRange> flow_line_ranges_by_flow_id;
  // A map from group index to counter data.
  // We use group index instead of PID as the key because a process (PID) can
  // have multiple counter tracks associated with it. The group index uniquely
  // identifies each track within the `groups` vector.
  std::map<int, CounterData> counter_data_by_group_index;
  std::vector<std::string> interned_string_pool;
  std::vector<std::string> hlo_module_table;
  std::vector<std::string> hlo_op_table;
  GroupIdRuns sparse_entry_group_ids;
  IndexRuns sparse_entry_hlo_module_indices;
  IndexRuns sparse_entry_hlo_op_indices;
  std::vector<std::vector<ModuleInterval>> hlo_module_intervals;
  std::vector<HloTrackRule> hlo_track_rules;
  SparseSelfTimes sparse_self_times;
  absl::flat_hash_map<size_t, absl::flat_hash_map<std::string, std::string>>
      sparse_entry_args;
  uint32_t dropped_level_cap_events = 0;

  // Reserves capacity for per-entry dense columns. When
  // `reserve_legacy_names_and_ids` is true (default), also reserves legacy
  // dense columns used by the `TraceEvent` path.
  void Reserve(size_t n, bool reserve_legacy_names_and_ids = true) {
    entry_levels.reserve(n);
    entry_total_times.reserve(n);
    entry_start_times.reserve(n);
    entry_name_indices.reserve(n);
    if (reserve_legacy_names_and_ids) {
      entry_self_times.reserve(n);
      entry_hlo_module_indices.reserve(n);
      entry_hlo_op_indices.reserve(n);
      entry_names.reserve(n);
      entry_event_ids.reserve(n);
    }
  }

  // Invokes `fn` on every per-entry dense column vector. Shared column registry
  // for entry release, compaction, and invariant validation.
  template <typename Fn>
  void ForEachEntryColumn(Fn&& fn) {
    fn(entry_levels);
    fn(entry_total_times);
    fn(entry_self_times);
    fn(entry_start_times);
    fn(entry_name_indices);
    fn(entry_hlo_module_indices);
    fn(entry_hlo_op_indices);
    fn(entry_names);
    fn(entry_event_ids);
    fn(entry_args);
  }

  template <typename Fn>
  void ForEachEntryColumn(Fn&& fn) const {
    fn(entry_levels);
    fn(entry_total_times);
    fn(entry_self_times);
    fn(entry_start_times);
    fn(entry_name_indices);
    fn(entry_hlo_module_indices);
    fn(entry_hlo_op_indices);
    fn(entry_names);
    fn(entry_event_ids);
    fn(entry_args);
  }

  // Invokes `fn` on every event-index-keyed sparse map.
  // Run tables (`sparse_entry_group_ids`, `sparse_entry_hlo_*_indices`,
  // `sparse_self_times`) are handled explicitly during release.
  template <typename Fn>
  void ForEachSparseEntryMap(Fn&& fn) {
    fn(non_zero_uids_by_event_index);
    fn(flow_ids_by_event_index);
    fn(sparse_entry_args);
  }

  template <typename Fn>
  void ForEachSparseEntryMap(Fn&& fn) const {
    fn(non_zero_uids_by_event_index);
    fn(flow_ids_by_event_index);
    fn(sparse_entry_args);
  }

  // Returns the self-time of entry `index`, falling back to `sparse_self_times`
  // or `entry_total_times[index]` when `entry_self_times` is not populated.
  Microseconds GetEntrySelfTime(size_t index) const {
    if (index < entry_self_times.size()) {
      return entry_self_times[index];
    }
    if (const std::optional<Microseconds> sparse_self =
            sparse_self_times.Find(index);
        sparse_self.has_value()) {
      return *sparse_self;
    }
    if (index < entry_total_times.size()) {
      return entry_total_times[index];
    }
    return 0.0;
  }

  // Finds the per-track HLO derivation rule covering `index`, or nullptr.
  const HloTrackRule* FindHloTrackRule(size_t index) const {
    if (hlo_track_rules.empty()) return nullptr;
    const auto it =
        std::upper_bound(hlo_track_rules.begin(), hlo_track_rules.end(), index,
                         [](size_t target, const HloTrackRule& r) {
                           return target < r.end_entry;
                         });
    if (it != hlo_track_rules.end() && index >= it->begin_entry) {
      return &(*it);
    }
    return nullptr;
  }

  // Returns the HLO op table index of entry `index` (or `entry_name_indices`
  // on `"XLA Ops"` tracks when derived on read; 0 if none).
  uint32_t GetEntryHloOpIndex(size_t index) const {
    if (index < entry_hlo_op_indices.size()) {
      return entry_hlo_op_indices[index];
    }
    if (const std::optional<uint32_t> explicit_op =
            sparse_entry_hlo_op_indices.Find(index);
        explicit_op.has_value()) {
      return *explicit_op;
    }
    if (const HloTrackRule* rule = FindHloTrackRule(index);
        rule != nullptr && rule->is_xla_ops &&
        index < entry_name_indices.size()) {
      return entry_name_indices[index];
    }
    return 0;
  }

  // Returns the HLO module table index of entry `index` (0 if default/none).
  uint32_t GetEntryHloModuleIndex(size_t index) const {
    if (index < entry_hlo_module_indices.size()) {
      return entry_hlo_module_indices[index];
    }
    if (const std::optional<uint32_t> explicit_mod =
            sparse_entry_hlo_module_indices.Find(index);
        explicit_mod.has_value()) {
      return *explicit_mod;
    }
    if (const HloTrackRule* rule = FindHloTrackRule(index); rule != nullptr) {
      uint32_t mod_idx = rule->default_module_index;
      if (rule->use_module_intervals &&
          rule->module_intervals_index < hlo_module_intervals.size() &&
          index < entry_start_times.size()) {
        const std::vector<ModuleInterval>& intervals =
            hlo_module_intervals[rule->module_intervals_index];
        const Microseconds ts = entry_start_times[index];
        const auto it = std::lower_bound(
            intervals.begin(), intervals.end(), ts,
            [](const ModuleInterval& m, Microseconds target_ts) {
              return m.end <= target_ts;
            });
        if (it != intervals.end() && ts >= it->start && ts < it->end) {
          mod_idx = it->module_index;
        }
      }
      return mod_idx;
    }
    return 0;
  }

  // Returns the HLO op name of entry `index` (empty if none).
  absl::string_view GetEntryHloOp(size_t index) const {
    if (index < entry_hlo_op_indices.size()) {
      const uint32_t op_idx = entry_hlo_op_indices[index];
      return op_idx < hlo_op_table.size()
                 ? absl::string_view(hlo_op_table[op_idx])
                 : "";
    }
    if (const std::optional<uint32_t> explicit_op =
            sparse_entry_hlo_op_indices.Find(index);
        explicit_op.has_value()) {
      return *explicit_op < hlo_op_table.size()
                 ? absl::string_view(hlo_op_table[*explicit_op])
                 : "";
    }
    if (const HloTrackRule* rule = FindHloTrackRule(index);
        rule != nullptr && rule->is_xla_ops &&
        index < entry_name_indices.size()) {
      const uint32_t name_idx = entry_name_indices[index];
      if (name_idx < interned_string_pool.size()) {
        return interned_string_pool[name_idx];
      }
    }
    return "";
  }

  // Returns the HLO module name of entry `index` (`kHloModuleDefault` when
  // module index is 0 and HLO module table is populated).
  absl::string_view GetEntryHloModule(size_t index) const {
    const uint32_t mod_idx = GetEntryHloModuleIndex(index);
    if (mod_idx < hlo_module_table.size() &&
        !hlo_module_table[mod_idx].empty()) {
      return hlo_module_table[mod_idx];
    }
    return kHloModuleDefault;
  }

  // Returns a pointer to the group containing the given level, or nullptr if
  // the level falls outside all group intervals.
  const Group* FindGroupForLevel(int level) const {
    if (level < 0 || groups.empty()) {
      return nullptr;
    }
    auto it = std::upper_bound(
        groups.begin(), groups.end(), level,
        [](int lvl, const Group& g) { return lvl < g.start_level; });
    if (it == groups.begin()) {
      return nullptr;
    }
    const auto& grp = *std::prev(it);
    if (grp.level_count > 0 && level >= grp.start_level + grp.level_count) {
      return nullptr;
    }
    return &grp;
  }

  int total_levels() const {
    return level_offsets.empty() ? 0
                                 : static_cast<int>(level_offsets.size() - 1);
  }

  absl::Span<const int> level_events(int level) const {
    if (level < 0 || level >= total_levels()) {
      return {};
    }
    const size_t start = level_offsets[level];
    const size_t end = level_offsets[level + 1];
    if (end < start || end > level_event_indices.size()) {
      return {};
    }
    return absl::Span<const int>(level_event_indices.data() + start,
                                 end - start);
  }

  bool HasEntryArgs(size_t index) const {
    if (index < entry_args.size() || sparse_entry_args.contains(index)) {
      return true;
    }
    return index < entry_name_indices.size() || index < entry_names.size();
  }

  absl::flat_hash_map<std::string, std::string> GetEntryArgs(
      size_t index) const {
    if (index < entry_args.size()) {
      return entry_args[index];
    }
    absl::flat_hash_map<std::string, std::string> args;
    if (auto it = sparse_entry_args.find(index);
        it != sparse_entry_args.end()) {
      args = it->second;
    } else if (index >= entry_name_indices.size() &&
               index >= entry_names.size()) {
      return args;
    }
    if (auto it = non_zero_uids_by_event_index.find(static_cast<int>(index));
        it != non_zero_uids_by_event_index.end()) {
      args.try_emplace("uid", absl::StrCat(it->second));
    }
    if (const std::optional<int64_t> group_id =
            sparse_entry_group_ids.Find(index);
        group_id.has_value()) {
      args.try_emplace("group_id", absl::StrCat(*group_id));
    }
    const absl::string_view hlo_op = GetEntryHloOp(index);
    if (!hlo_op.empty()) {
      args[std::string(kHloOp)] = std::string(hlo_op);
    }
    const bool has_hlo_module_metadata =
        index < entry_hlo_module_indices.size() ||
        !sparse_entry_hlo_module_indices.empty() || !hlo_module_table.empty();
    if (has_hlo_module_metadata) {
      args[std::string(kHloModule)] = std::string(GetEntryHloModule(index));
    }
    return args;
  }
};

// Renders an interactive timeline visualization for trace events, handling
// zooming, panning, and rendering of events grouped into lanes.
class Timeline {
 public:
  void CalculateAndEmitMetrics_for_test() { CalculateAndEmitMetrics(); }
  EntryBitset& selected_event_indices_for_test() {
    return selected_event_indices_;
  }
  std::vector<std::pair<int, int>>& selected_counter_points_for_test() {
    return selected_counter_points_;
  }
  void SetPlaybackState(bool is_playing, double current_progress_us,
                        double play_speed) {
    if (!timeline_player_enabled_) return;
    is_playing_ = is_playing;
    if (current_progress_us >= 0) {
      current_play_time_ = visible_range().start() + current_progress_us;
    }
    play_speed_ = play_speed;
    if (redraw_callback_) redraw_callback_();
  }

  struct SearchResult {
    EventId event_id;
    int level;
    Microseconds start_time;
    Microseconds duration;
    ProcessId pid;
    ThreadId tid;
    std::string name;
    int loaded_index = -1;
  };

  struct GroupRelativeInfo {
    Group* parent = nullptr;
    std::vector<Group>* siblings = nullptr;
    int index_in_siblings = -1;
  };
  // A callback function to handle events from the timeline. The first argument
  // is the event type string. The second argument, EventData, is the payload
  // dispatched as the `detail` of a `CustomEvent` on the `window` object.
  // The callback is expected to be lightweight and non-blocking, as it will be
  // called on the main thread.
  using EventCallback =
      absl::AnyInvocable<void(absl::string_view, const EventData&) const>;
  using RedrawCallback = absl::AnyInvocable<void()>;

  Timeline(ColorPalette& palette) : palette_(palette) {}
  // This is necessary because MockTimeline in the tests inherits from Timeline.
  virtual ~Timeline() = default;

  // For testing only
  float get_copy_notification_timer_for_test() const {
    return copy_notification_timer_;
  }
  const std::string& get_copied_track_name_for_test() const {
    return copied_track_name_;
  }
  ImU32 get_group_accent_color_for_test(int group_index) const {
    return GetGroupAccentColor(group_index);
  }
  const std::vector<bool>& get_group_striped_for_test() const {
    return group_striped_;
  }
  float get_bounds_notification_timer_for_test() const {
    return bounds_notification_timer_;
  }
  const std::string& get_bounds_notification_message_for_test() const {
    return bounds_notification_message_;
  }
  bool get_should_restore_scroll_for_test() const {
    return should_restore_scroll_;
  }
  // Sets the pending navigation event ID for testing.
  void set_pending_navigation_event_id_for_test(std::optional<EventId> id) {
    pending_navigation_event_id_ = id;
  }
  // Gets the pending navigation event ID for testing.
  std::optional<EventId> get_pending_navigation_event_id_for_test() const {
    return pending_navigation_event_id_;
  }
  const EntryBitset& get_matching_event_indices_for_test() const {
    return matching_event_indices_;
  }
  void set_selection_start_pos_for_test(std::optional<ImVec2> pos) {
    selection_start_pos_ = pos;
  }
  void set_current_timeline_width_for_test(Pixel width) {
    current_timeline_width_ = width;
  }
  void emit_viewport_changed_for_test(const TimeRange& range) {
    EmitViewportChanged(range);
  }
  void zoom_for_test(float zoom_factor, Microseconds pivot) {
    Zoom(zoom_factor, pivot);
  }
  // Applies a deterministic vertical scroll offset (in pixels) for tests,
  // reusing the production scroll-restore path so the offset takes effect
  // on the next Draw().
  void set_scroll_offset_for_test(Pixel scroll_y) {
    last_scroll_y_ = scroll_y;
    should_restore_scroll_ = true;
  }
  int get_pending_reorder_source_for_test() const {
    return pending_reorder_source_;
  }
  void set_pending_reorder_source_for_test(int source) {
    pending_reorder_source_ = source;
  }
  int get_pending_reorder_target_for_test() const {
    return pending_reorder_target_;
  }
  void set_pending_reorder_target_for_test(int target) {
    pending_reorder_target_ = target;
  }
  bool get_pending_reorder_drop_after_for_test() const {
    return pending_reorder_drop_after_;
  }
  void set_pending_reorder_drop_after_for_test(bool drop_after) {
    pending_reorder_drop_after_ = drop_after;
  }
  Pixel get_reorder_preview_line_y_for_test() const {
    return reorder_preview_line_y_;
  }
  void set_reorder_preview_line_y_for_test(Pixel y) {
    reorder_preview_line_y_ = y;
  }
  void set_group_offsets_for_test(const std::vector<Pixel>& offsets) {
    group_offsets_ = offsets;
  }
  void set_group_heights_for_test(const std::vector<Pixel>& heights) {
    group_heights_ = heights;
  }
  void set_label_width_for_test(Pixel width) {
    label_width_ = width;
  }
  bool get_is_resizing_label_column_for_test() const {
    return is_resizing_label_column_;
  }
  void set_track_management_enabled_for_test(bool enabled) {
    track_management_enabled_ = enabled;
  }


  // The provided callback is stored and invoked during the lifetime of this
  // `Timeline` instance. Any captured references must outlive the `Timeline`
  // instance.
  void set_event_callback(EventCallback callback) {
    event_callback_ = std::move(callback);
  }
  void set_redraw_callback(RedrawCallback callback) {
    redraw_callback_ = std::move(callback);
  }
  void RequestRedraw() {
    if (redraw_callback_) {
      redraw_callback_();
    }
  }

  uint64_t filter_epoch() const { return filter_epoch_; }

  bool IsEventVisibleUnderFilter(int event_index) const {
    return !has_active_event_filter_ ||
           filtered_event_indices_.contains(event_index);
  }

  // Sets the search query to highlight events matching the query.
  // The search is case-insensitive.
  void SetSearchQuery(absl::string_view query);

  // Sets the search results from the given parsed trace events.
  void SetSearchResults(const ParsedTraceEvents& search_results);

  int get_search_results_count() const {
    return !search_results_.empty()
               ? static_cast<int>(search_results_.size())
               : static_cast<int>(matching_event_indices_.size());
  }
  const std::vector<SearchResult>& get_search_results_for_test() const;
  int get_current_search_result_index() const {
    return current_search_result_index_;
  }

  // Sets the visible time range. If animate is true, the transition to the
  // new range will be animated, otherwise it will snap to the new time range.
  // Animation is useful for smoothing out transitions caused by user actions
  // like zooming to a selection.
  void SetVisibleRange(const TimeRange& range, bool animate = false);
  const TimeRange& visible_range() const { return *visible_range_; }
  const TimeRange& visible_range_target() const {
    return visible_range_.target();
  }

  void AddSelectedTimeRange(const TimeRange& range) {
    selected_time_ranges_.push_back(range);
  }
  const std::vector<TimeRange>& selected_time_ranges() const {
    return selected_time_ranges_;
  }

  const std::optional<TimeRange>& current_selected_time_range() const {
    return current_selected_time_range_;
  }

  void set_fetched_data_time_range(const TimeRange& range) {
    fetched_data_time_range_ = range;
    // If the last fetch request range is empty, it means we haven't made any
    // incremental loading yet. In this case, we initialize it to the fetched
    // data range to prevent immediate redundant fetches upon the first update.
    if (last_fetch_request_range_.duration() == 0) {
      last_fetch_request_range_ = range;
    }
  }
  const TimeRange& fetched_data_time_range() const {
    return fetched_data_time_range_;
  }

  const TimeRange& last_fetch_request_range() const {
    return last_fetch_request_range_;
  }

  // Calculates and initializes the last_fetch_request_range_ using the same
  // logic as MaybeRequestData (scaling, min-bounds, and ConstrainTimeRange).
  void InitializeLastFetchRequestRange(const TimeRange& visible_range);

  void set_data_time_range(const TimeRange& range) {
    if (data_time_range_ != range) {
      data_time_range_ = range;
      minimap_schedule_dirty_ = true;
      SyncMinimapState();
    }
  }
  const TimeRange& data_time_range() const { return data_time_range_; }

  // Releases per-entry columns, sparse maps, CSR buffers, flows, string pools,
  // entry colors, and mip pyramids before a new FlameChartTimelineData is built
  // or committed, while preserving track group metadata and scroll/selection
  // state across reloads.
  void ReleaseTimelineEntriesPreservingState();

  void SetTimelineData(FlameChartTimelineData data);
  void UpdateTimelineData(
      absl::FunctionRef<FlameChartTimelineData()> builder);
  const FlameChartTimelineData& timeline_data() const { return timeline_data_; }

  static constexpr size_t kDefaultMaxUncompactedOverviewEntries = 50'000'000;
  static constexpr size_t kDefaultMaxUncompactedOverviewBytes =
      1536ULL * 1024 * 1024;
  static constexpr size_t kMaxUncompactedOverviewStashEntries = 200'000;

  void set_max_uncompacted_overview_entries_for_test(size_t max_entries) {
    max_uncompacted_overview_entries_ = max_entries;
  }
  size_t max_uncompacted_overview_entries() const {
    return max_uncompacted_overview_entries_;
  }
  void set_max_uncompacted_overview_bytes_for_test(size_t max_bytes) {
    max_uncompacted_overview_bytes_ = max_bytes;
  }
  size_t max_uncompacted_overview_bytes() const {
    return max_uncompacted_overview_bytes_;
  }
  size_t EstimateRetainedEntryBytes() const;

  void MarkCurrentDataAsFullOverview() { is_full_overview_data_ = true; }
  void PrepareForFilteredRefetch() {
    if (is_full_overview_data_ && !saved_full_overview_.has_value()) {
      stash_full_overview_on_next_update_ = true;
    }
  }
  bool stash_full_overview_on_next_update() const {
    return stash_full_overview_on_next_update_;
  }
  bool HasSavedFullOverview() const { return saved_full_overview_.has_value(); }
  bool RestoreSavedFullOverview();
  void ClearSavedFullOverview() {
    saved_full_overview_.reset();
    is_full_overview_data_ = false;
    stash_full_overview_on_next_update_ = false;
  }

  int selected_event_index() const { return selected_event_index_; }
  int selected_group_index() const { return selected_group_index_; }
  int selected_counter_index() const { return selected_counter_index_; }

  size_t GetEntryCount() const {
    return !timeline_data_.entry_start_times.empty()
               ? timeline_data_.entry_start_times.size()
               : (!timeline_data_.entry_name_indices.empty()
                      ? timeline_data_.entry_name_indices.size()
                      : timeline_data_.entry_names.size());
  }
  absl::string_view GetEventName(int event_index) const;

  const std::vector<Pixel>& GetVisibleLevelOffsets() const {
    return visible_level_offsets_;
  }

  void set_mpmd_pipeline_view_enabled(bool enabled) {
    mpmd_pipeline_view_enabled_ = enabled;
  }
  bool mpmd_pipeline_view_enabled() const {
    return mpmd_pipeline_view_enabled_;
  }

  // The unit of the x-axis. For `TimeAxisUnit::kUnitless` (e.g. scheduled
  // bundle numbers), the ruler shows integer positions instead of times.
  void set_time_axis_unit(TimeAxisUnit unit) {
    if (time_axis_unit_ != unit) {
      time_axis_unit_ = unit;
      minimap_schedule_dirty_ = true;
      SyncMinimapState();
    }
  }
  TimeAxisUnit time_axis_unit() const { return time_axis_unit_; }


  void set_bookmarks_enabled(bool enabled) { bookmarks_enabled_ = enabled; }
  bool bookmarks_enabled() const { return bookmarks_enabled_; }

  // Adds a bookmark at the specified time if one doesn't already exist nearby.
  void AddBookmark(Microseconds time);

  // Removes the specified bookmark.
  void RemoveBookmark(Microseconds time);

  const std::vector<Microseconds>& bookmarks() const { return bookmarks_; }
  // Enables motion in the track sidebar: expand / collapse transitions that
  // slide rows open and closed, rotating chevrons, and hover fades. Disabled by
  // default (rendering snaps to the final state, which keeps tests and
  // screenshots deterministic); the application enables it unless the user
  // prefers reduced motion. Disabling it finishes any motion in flight.
  void set_animations_enabled(bool enabled);
  bool animations_enabled() const { return animations_enabled_; }
  // Returns true while an expand / collapse layout transition is in flight.
  bool is_layout_transition_active() const { return layout_transition_.active; }
  // Returns the (possibly mid-transition) height of each group's row.
  const std::vector<Pixel>& group_heights() const { return group_heights_; }

  void set_track_management_enabled(bool enabled) {
    track_management_enabled_ = enabled;
  }
  bool track_management_enabled() const { return track_management_enabled_; }
  void set_timeline_player_enabled(bool enabled) {
    timeline_player_enabled_ = enabled;
  }
  bool timeline_player_enabled() const { return timeline_player_enabled_; }
  void set_minimap_enabled(bool enabled) {
    if (minimap_enabled_ != enabled) {
      minimap_enabled_ = enabled;
      if (minimap_enabled_) {
        minimap_schedule_dirty_ = true;
        SyncMinimapState();
      }
    }
  }
  bool minimap_enabled() const { return minimap_enabled_; }

  // Whether hovering a flame-chart event shows the built-in tooltip
  // (`name (duration)`). Hosts that draw their own tooltip, or that already
  // paint everything the tooltip would say on the event itself, turn it off.
  // Hover highlighting, the `eventhovered` callback and click selection are
  // unaffected, and counter-track tooltips are separate. On by default and
  // kept by ResetForNewDataset().
  void set_event_tooltip_enabled(bool enabled) {
    if (event_tooltip_enabled_ != enabled) {
      event_tooltip_enabled_ = enabled;
      if (!event_tooltip_enabled_) {
        minimap_schedule_dirty_ = true;
        SyncMinimapState();
      }
    }
  }
  bool event_tooltip_enabled() const { return event_tooltip_enabled_; }

  void set_hlo_dependency_arrows_enabled(bool enabled) {
    hlo_dependency_arrows_enabled_ = enabled;
  }

  // Resolves the HLO operands and consumers of the selected event to events
  // on its track, see ResolveEventDependencies() in
  // timeline_hlo_dependencies.h. The names are resolved again when new
  // timeline data is set while the same op stays selected. Does nothing unless
  // HLO dependency arrows are enabled.
  void SetSelectedEventDependencies(std::vector<std::string> operand_names,
                                    std::vector<std::string> consumer_names);

  // The events that the operands and consumers of the selected event resolved
  // to. Empty if they were set while another event was selected.
  absl::Span<const int> producer_event_indices() const {
    if (dependencies_.event_index != selected_event_index_) return {};
    return dependencies_.producer_indices;
  }
  absl::Span<const int> consumer_event_indices() const {
    if (dependencies_.event_index != selected_event_index_) return {};
    return dependencies_.consumer_indices;
  }

  void set_panning_speed(float speed) { panning_speed_ = speed; }
  float panning_speed() const { return panning_speed_; }

  void set_zoom_speed(float speed) { zoom_speed_ = speed; }
  float zoom_speed() const { return zoom_speed_; }

  bool show_grid() const { return show_grid_; }
  void set_show_grid(bool show_grid) { show_grid_ = show_grid; }

  void set_mouse_wheel_zoom_speed(float speed) {
    mouse_wheel_zoom_speed_ = speed;
  }
  float mouse_wheel_zoom_speed() const { return mouse_wheel_zoom_speed_; }

  // Zooms the visible time range by the given zoom factor, centered around the
  // mouse position, or the center of the visible range if the mouse is outside
  // the trace events area.
  // This method is virtual to allow derived classes to customize or extend
  // zooming behavior.
  virtual void Zoom(float zoom_factor);

  void set_mouse_mode(MouseMode mode) { mouse_mode_ = mode; }
  MouseMode mouse_mode() const { return mouse_mode_; }

  void set_is_incremental_loading(bool is_incremental_loading) {
    is_incremental_loading_ = is_incremental_loading;
  }
  bool is_incremental_loading() const { return is_incremental_loading_; }

  void set_full_trace_loaded(bool full_trace_loaded) {
    full_trace_loaded_ = full_trace_loaded;
  }
  bool full_trace_loaded() const { return full_trace_loaded_; }
  void set_is_compacted(bool is_compacted) { is_compacted_ = is_compacted; }
  bool is_compacted() const { return is_compacted_; }
  void set_window_at_full_resolution(bool window_at_full_resolution) {
    window_at_full_resolution_ = window_at_full_resolution;
  }
  bool window_at_full_resolution() const { return window_at_full_resolution_; }

  // Whether MaybeRequestData() may emit `fetch_data` to refetch a time range at
  // higher resolution. Hosts whose backend ignores the requested range and
  // always returns the full dataset should turn this off.
  void set_incremental_fetch_enabled(bool enabled) {
    incremental_fetch_enabled_ = enabled;
  }
  bool incremental_fetch_enabled() const { return incremental_fetch_enabled_; }

  // Returns the timeline to its pre-first-load state (data, ranges, selection,
  // search, bookmarks, scroll) so a different dataset can be loaded. Host
  // toggles and per-name track preferences are kept.
  void ResetForNewDataset();

  Pixel GetLabelWidth() const { return label_width_; }

  void SetVisibleFlowCategory(int category_id) {
    flow_category_filter_ = category_id;
  }
  void SetVisibleFlowCategories(const std::vector<int>& category_ids);

  void HideTrack(absl::string_view name);

  void Draw();

  void UpdateLevelPositions(const FlameChartTimelineData& data);
  void BuildFlattenedGroups(const FlameChartTimelineData& data);
  void CategorizeGroupsForTrackManagement(
      const FlameChartTimelineData& data,
      std::vector<const Group*>& hidden_groups,
      std::vector<const Group*>& pinned_groups,
      std::vector<const Group*>& all_groups);

  // Expands the minimum necessary tracks to make the event visible.
  void ExpandRelatedTracks(int event_index);

  // Calculates the screen coordinates of the rectangle for an event.
  EventRect CalculateEventRect(Microseconds start, Microseconds end,
                               Pixel screen_x_offset, Pixel screen_y_offset,
                               double px_per_time_unit, int level_in_group,
                               Pixel timeline_width, Pixel event_height,
                               Pixel padding_bottom) const;

  // Calculates the top-left screen coordinates for the event name text.
  ImVec2 CalculateEventTextRect(absl::string_view event_name,
                                const EventRect& event_rect) const;

  // Returns text truncated with ellipsis if it's wider than available_width.
  std::string GetTextForDisplay(absl::string_view event_name,
                                float available_width) const;

  // Converts a pixel offset relative to the start of the visible range to a
  // time.
  Microseconds PixelToTime(Pixel pixel_offset, double px_per_time_unit) const;

  // Converts a time to a pixel offset relative to the start of the visible
  // range.
  Pixel TimeToPixel(Microseconds time, double px_per_time_unit) const;

  // Converts a time value to an absolute screen X coordinate.
  Pixel TimeToScreenX(Microseconds time, Pixel screen_x_offset,
                      double px_per_time_unit) const;

  void ConstrainTimeRange(TimeRange& range);

  // Selects the event with the given index and ensures it is visible.
  // If the event is already fully visible, the viewport is not changed.
  // If the event is out of view, the viewport pans horizontally to reveal it
  // without altering the zoom level.
  void RevealEvent(int event_index);

  // Navigates to and selects the event with the given index, zooming to it.
  void ZoomEvent(int event_index);

  // Navigates selection to the previous event on the current level/track.
  void SelectPreviousEvent();

  // Navigates selection to the next event on the current level/track.
  void SelectNextEvent();

  void NavigateToNextSearchResult();
  void NavigateToPrevSearchResult();

  // Information about timeline ticks for drawing ruler and grid lines.
  struct TickInfo {
    // Time duration between major ticks.
    Microseconds tick_interval;
    // Pixel distance between major ticks.
    Pixel major_tick_dist_px;
    // Time of the first major tick relative to trace start.
    Microseconds first_tick_time_relative;
  };

  // Calculates tick information based on current zoom level (px_per_time_unit).
  TickInfo CalculateTickInfo(double px_per_time_unit_val) const;

  // Formats the label for a major ruler tick at `time_relative` (relative to
  // the trace start). Returns a time string (e.g. "1.5 ms") normally, or the
  // absolute position as a plain integer (e.g. "40") on a unitless axis.
  std::string FormatRulerLabel(Microseconds time_relative) const;

  // Calculates the control points for a cubic Bezier curve used to draw flows.
  static void CalculateBezierControlPoints(float start_x, float start_y,
                                           float end_x, float end_y,
                                           ImVec2& cp0, ImVec2& cp1);

  // Gets the starting level index of the group immediately following the group
  // at the given index. If the given group is the last one, returns the total
  // number of levels.
  static int GetNextGroupStartLevel(const FlameChartTimelineData& data,
                                    int group_index);

  // Checks if the visible time range is close to the edge of the loaded data
  // range. If the user pans or zooms to an area where data might soon be
  // needed (i.e., outside the `preserve` range), this function triggers a data
  // fetch request for a larger range (`fetch` range) to ensure data is
  // available before it becomes visible, providing a smoother user experience.
  // Exposed for testing.
  void MaybeRequestData();
  double px_per_time_unit() const;
  double px_per_time_unit(Pixel timeline_width) const;
  void DrawTimelinePlayerSync();


  // Calculates the layout for the delete button and its hover area.
  // Exposed for testing.
  DeleteButtonLayout GetDeleteButtonLayout(const ImVec2& text_size,
                                           const ImVec2& text_pos,
                                           const ImRect& visible_range_rect,
                                           const ImRect& full_range_rect) const;

  const ColorPalette& GetPalette() const { return palette_; }

  // ---------------------------------------------------------------------------
  // Accessors for testing
  // ---------------------------------------------------------------------------
  Pixel last_scroll_y_for_test() const { return last_scroll_y_; }
  void set_last_scroll_y_for_test(Pixel y) { last_scroll_y_ = y; }
  void set_selected_event_index_for_test(int idx) {
    selected_event_index_ = idx;
  }
  void set_group_visible_for_test(int idx, bool visible) {
    if (idx >= 0 && idx < static_cast<int>(group_visible_.size())) {
      group_visible_[idx] = visible;
    }
  }
  void set_header_all_expanded_for_test(bool expanded) {
    header_all_expanded_ = expanded;
  }
  const Group& header_hidden_for_test() const { return header_hidden_; }
  const Group& header_pinned_for_test() const { return header_pinned_; }
  void set_search_results_for_test(std::vector<SearchResult> results) {
    search_results_ = std::move(results);
  }
  void set_current_search_result_index_for_test(int idx) {
    current_search_result_index_ = idx;
  }
  bool is_level_raw_events_compacted_for_test(int level) const {
    return level >= 0 &&
           static_cast<size_t>(level) < level_mip_pyramids_.size() &&
           level_mip_pyramids_[level].raw_events_compacted;
  }

  struct GpuFlameBatch {
    uint32_t first_instance = 0;
    uint32_t instance_count = 0;
    double visible_start_us = 0.0;
    float px_per_us = 0.0f;
    float screen_x_offset = 0.0f;
    float timeline_width = 0.0f;
    float y_top = 0.0f;
    float y_bottom = 0.0f;
    float min_width_px = 1.0f;
    float padding_right_px = 0.5f;
    float alpha_multiplier = 1.0f;
  };
  void set_force_gpu_flame_batch_mode_for_test(bool enabled) {
    force_gpu_flame_batch_mode_for_test_ = enabled;
  }
  const std::vector<GpuFlameBatch>& emitted_gpu_flame_batches_for_test() const {
    return emitted_gpu_flame_batches_;
  }
  size_t gpu_flame_instances_count_for_test() const {
    return gpu_flame_instances_.size();
  }
  void ExpandAllGroupsForTest() {
    for (Group& group : timeline_data_.groups) {
      group.expanded = true;
    }
    UpdateLevelPositions(timeline_data_);
  }

 protected:
  // Virtual method to allow mocking in tests.
  virtual ImVec2 GetTextSize(absl::string_view text) const {
    return ImGui::CalcTextSize(text.data(), text.data() + text.size());
  }

  // Pans the visible time range by the given pixel amount.
  // This method is virtual to allow derived classes to customize or extend
  // panning behavior.
  virtual void Pan(Pixel pixel_amount);

  // Scrolls the visible time range by the given pixel amount.
  // This method is virtual to allow derived classes to customize or extend
  // panning behavior.
  virtual void Scroll(Pixel pixel_amount);

  // Zooms the visible time range by the given zoom factor around the specified
  // pivot timestamp.
  // This method is virtual to allow derived classes to customize or extend
  // zooming behavior.
  virtual void Zoom(float zoom_factor, Microseconds pivot);

 protected:
  virtual void DrawEventsForLevel(int group_index,
                                  absl::Span<const int> event_indices,
                                  double px_per_time_unit, int level_in_group,
                                  const ImVec2& pos, const ImVec2& max,
                                  Pixel event_height, Pixel padding_bottom);

  virtual void DrawGroup(int group_index, double px_per_time_unit_val,
                         Pixel scroll_y, Pixel window_height);

  // Finds the index of the first visible ancestor (or the group itself if it is
  // visible) for a given group index. Decoupled from scroll-height specific
  // terms to allow general usage.
  int FindFirstVisibleAncestorIndex(int start_idx) const;

  Pixel GetGroupTop(const Group* group) const;
  Pixel GetGroupBottom(const Group* group) const;

  // Returns the cached group visibility array.
  const std::vector<bool>& group_visible() const { return group_visible_; }

  void DrawEvent(int group_index, int event_index, const EventRect& rect,
                 ImDrawList* absl_nonnull draw_list);

  // Returns true if the mouse is over any part (label or canvas) of the row.
  bool IsTrackRowHovered(int group_index) const;
  // Draw the pin / hide icon buttons of a process row with the cursor at the
  // top-left of the button, vertically centering them in `height`. Return true
  // if the pinned / hidden state was toggled.
  bool DrawHideButton(int group_index, Pixel height, bool is_track_hidden);
  bool DrawPinButton(int group_index, Pixel height, bool is_pinned);

 private:
  struct SavedSelectionState {
    EventId event_id = 0;
    ProcessId pid = 0;
    ThreadId tid = 0;
    std::string group_name;
    std::string name;
    Microseconds start = 0.0;
    Microseconds dur = 0.0;
  };

  EntryBitset matching_event_indices_;
  std::optional<SavedSelectionState> saved_selection_state_;

  void CaptureSelectionStateIfNeeded();
  int GetNthMatchingCsrEventIndex(int target_rank) const;
  void NavigateToSearchResult(const SearchResult& result);
  void BackfillGroupLevelCount(FlameChartTimelineData& data);

  // Applies snapping to selected time ranges for the given range.
  void ApplySnapping(TimeRange& range);

  // Finds the nearest event edge to a given time across all tracks.
  void FindNearestEventEdge(Microseconds time, Microseconds threshold,
                            Microseconds& best_diff, Microseconds& snapped_time,
                            bool& snapped) const;

  // Emits an event selected event to JS side.
  void EmitEventSelected(int event_index);
  EventData CreateBaseEventData(int event_index, bool is_hover = false) const;
  // Emits an event hovered event to JS side.
  void EmitEventHovered(int event_index, float mouse_x, float mouse_y);
  // Populates schedule context (totalBundles, bundleCounts, regions,
  // utilization, regionDepth) on `event_data` when `event_tooltip_enabled_` is
  // false.
  void PopulateEventSelectedScheduleDetails(int event_index,
                                            EventData& event_data) const;
  void PopulateCounterSelectedScheduleDetails(int group_index,
                                              size_t counter_index,
                                              Microseconds clicked_bundle,
                                              EventData& event_data) const;
  void PopulateEventHoveredScheduleDetails(int event_index,
                                           EventData& event_data) const;
  void PopulateMultiSelectionScheduleDetails(Microseconds selection_start_us,
                                             Microseconds selection_extent_us,
                                             EventData& event_data) const;
  void PopulateHoverScheduleDetails(int group_index, int event_index,
                                    EventData& event_data) const;
  // Populates schedule context at a discrete VLIW instruction bundle index on
  // the timeline x-axis (e.g. when hovering a counter or process track).
  void PopulateBundleScheduleDetails(Microseconds bundle,
                                     EventData& event_data) const;
  void PopulateBundleEvents(Microseconds bundle, int selected_event_index,
                            EventData& event_data) const;
  void PopulateSelectedScheduleDetails(int group_index, int event_index,
                                       EventData& event_data) const;
  void SyncMinimapState();
  void PopulateMinimapScheduleSummary(Microseconds data_start,
                                      Microseconds data_end,
                                      EventData& payload) const;
  Microseconds GetScheduleEndBundle() const;
  // Emits viewport changed event to JS side.
  void EmitViewportChanged(const TimeRange& range);
  // Emits mouse mode changed event to JS side.
  void EmitMouseModeChanged();
  void ShowNavigationWarningNotification(absl::string_view message);

 protected:
  GroupRelativeInfo FindGroupRelatives(Group* target_group);

 private:
  // Draws the timeline ruler UI (background, horizontal line, labels, ticks).
  void DrawRulerUI(const TickInfo& info, Pixel timeline_width);

  // Draws a section header row (All, Pinned or Hidden) in the timeline. The
  // whole row toggles the section. Returns true if layout update is needed.
  bool DrawHeaderRow(const Group* group_ptr,
                     const ImVec2& tracks_start_screen_pos, Pixel group_top,
                     Pixel group_bottom);

  // Copies a track's name to the clipboard and shows the copy notification.
  void CopyTrackName(int group_index);

  // --- Motion -------------------------------------------------------------
  // Returns true if the group's row should be drawn: it is visible, or it is
  // folding away / sliding in as part of a layout transition.
  bool IsGroupDrawn(int group_index) const;
  // Returns the height to draw a row's events at: `full_height`, or less while
  // the row is opening or folding during a layout transition.
  Pixel AnimatedRowHeight(int group_index, Pixel full_height) const;
  // Prepares the motion for an expand / collapse toggle of the row identified
  // by `chevron_key` (a group index, or a virtual header id) whose expanded
  // state is about to flip from `was_expanded`: the next layout update is
  // animated and the chevron starts rotating.
  void BeginExpandToggleMotion(int chevron_key, bool was_expanded);
  // Returns the chevron "openness" of a row in [0, 1] (0 = collapsed, pointing
  // right; 1 = expanded, pointing down), advancing its rotation if animating.
  // Call at most once per row per frame.
  float ChevronOpenness(int chevron_key, bool expanded);
  // Returns the hover emphasis of the item `id` in [0, 1], fading it in and out
  // instead of snapping. Call at most once per item per frame.
  float HoverFade(ImGuiID id, bool hovered);
  // Advances the layout transition by `delta_time` seconds, writing the
  // interpolated offsets and heights into the live layout vectors.
  void AdvanceLayoutTransition(float delta_time);
  // Snaps the layout transition to its end state.
  void FinishLayoutTransition();
  // Stores a newly computed layout, animating to it if a toggle requested it.
  void ApplyLayout(const FlameChartTimelineData& data,
                   std::vector<Pixel> group_offsets,
                   std::vector<Pixel> group_heights,
                   std::vector<Pixel> level_offsets,
                   const std::vector<bool>& group_visible,
                   const Pixel (&from_header_offsets)[3]);
  // Turns the rows entering or leaving in a layout transition into an
  // accordion: instead of being squashed, they keep their height and slide in
  // from / out under the row above them, behind a moving clip edge.
  void SetUpAccordionRuns(const FlameChartTimelineData& data,
                          const std::vector<bool>& was_drawn,
                          const std::vector<bool>& was_visible,
                          const std::vector<bool>& will_be_visible);
  // Returns the screen-space top clip edge of a row sliding in or out during a
  // layout transition, or nullopt if the row is not clipped.
  std::optional<Pixel> AccordionClipTop(int group_index) const;

  // Returns the accent color of a track (used by collapsed-track cues): the
  // dominant event color for flame tracks, the chart color for process and
  // counter tracks, or a neutral color for tracks without events.
  ImU32 GetGroupAccentColor(int group_index) const;

  // Recomputes `group_accent_colors_` from the current events and colors.
  void RebuildGroupAccentColors();

  // Draws a standard track row in the timeline: the row background, the label
  // (chevron, name, subtitle and count chip) and the events.
  // Returns true if layout update is needed.
  bool DrawTrackRow(int group_index, const ImVec2& tracks_start_pos,
                    const ImVec2& tracks_start_screen_pos,
                    Pixel content_region_avail_width,
                    double px_per_time_unit_val, Pixel scroll_y,
                    Pixel window_height);

  // Handles clicking on the timeline area of a process track header to
  // expand/collapse. Returns true if layout update is needed.
  bool HandleProcessTrackHeaderClick(int group_index, Group& group,
                                     const ImVec2& tracks_start_screen_pos,
                                     Pixel content_region_avail_width);

 protected:
  // Handles drag-and-drop source/target and
  // item hover check for a track label row.
  bool HandleTrackDragAndDrop(int group_index, Group& group,
                              const ImVec2& tracks_start_pos,
                              const ImVec2& tracks_start_screen_pos,
                              Pixel group_height, Pixel hover_zone_width);
  // Handles track hover, drag-and-drop, and reorder tooltip for a track row.
  void HandleTrackDragAndDropHoverAndFeedback(
      int group_index, Group& group, const ImVec2& tracks_start_pos,
      const ImVec2& tracks_start_screen_pos, Pixel group_height);

 private:
  // Draws vertical grid lines across the background of the tracks.
  // `viewport_bottom` is the y-coordinate of the bottom of the viewport, used
  // to draw vertical grid lines across the tracks.
  void DrawVerticalGridLines(const TickInfo& info, Pixel timeline_width,
                             Pixel viewport_bottom);

  void DrawEventName(absl::string_view event_name, const EventRect& rect,
                     ImDrawList* absl_nonnull draw_list,
                     ImU32 text_color) const;
  void DrawEventTextOnly(int event_index, const EventRect& rect,
                         ImDrawList* absl_nonnull draw_list) const;

  void DrawCounterTooltip(int group_index, const CounterData& counter_data,
                          double px_per_time_unit_val, const ImVec2& pos,
                          Pixel height, float y_ratio, ImDrawList* draw_list);

  void DrawCounterTrack(int group_index, const CounterData& counter_data,
                        double px_per_time_unit_val, const ImVec2& pos,
                        Pixel height);

  bool DrawTrackManagementButtons(int group_index, const Group& group,
                                  const ImVec2& tracks_start_pos,
                                  Pixel centereable_height);

  void DrawGroupPreview(int group_index, double px_per_time_unit_val);
  void DrawFlameGroupPreview(int start_level, int end_level,
                             double px_per_time_unit_val, const ImVec2& pos,
                             Pixel group_height, ImDrawList* draw_list);
  void DrawUtilizationAreaChart(int group_index, int start_level, int end_level,
                                double px_per_time_unit_val, const ImVec2& pos,
                                Pixel group_height, ImDrawList* draw_list);
  void AccumulateClippedRangeOccupancy(Pixel x_start, Pixel x_end,
                                       float density, int num_bins);
  bool TryAccumulateProcessUtilization(int start_level, int end_level,
                                       double px_per_time_unit_val,
                                       int num_bins);
  void AccumulateLevelUtilization(int level, double px_per_time_unit_val,
                                  int num_bins);

  // Draws a single flow line.
  void DrawSingleFlow(const FlowLine& flow, Pixel timeline_x_start,
                      Pixel timeline_y_start, double px_per_time,
                      ImDrawList* draw_list);

  // Draws flow lines connecting events. Each flow line is rendered as a Bezier
  // curve connecting a start point (time and level) to an end point (time and
  // level).
  void DrawFlows(Pixel timeline_width, Pixel timeline_y_start);

  // Draws an arrow from the end of `from_event` to the start of `to_event`,
  // as a Bezier curve like a flow line, with an arrowhead at the end.
  void DrawDependencyArrow(int from_event, int to_event, ImU32 color,
                           Pixel timeline_x_start, Pixel timeline_y_start,
                           double px_per_time, ImDrawList* draw_list);

  // Draws arrows from the resolved HLO operands of the selected event to it,
  // and from it to its resolved consumers (see SetSelectedEventDependencies).
  // At most kMaxHloDependencyArrowsPerSide are drawn on each side.
  void DrawDependencyArrows(Pixel timeline_width, Pixel timeline_y_start);

  // Draws a single selected time range.
  // The `show_delete_button` will be false for the currently selected time
  // time range.
  void DrawSelectedTimeRange(const TimeRange& range, Pixel timeline_width,
                             double px_per_time_unit_val,
                             bool show_delete_button = true,
                             std::optional<size_t> range_index = std::nullopt);
  void DrawDeleteButton(ImDrawList* draw_list, const ImVec2& button_pos,
                        const ImRect& hover_rect, const TimeRange& range);

  // Draws all the selected time ranges, including the current selected range.
  void DrawSelectedTimeRanges(Pixel timeline_width,
                              double px_per_time_unit_val);

  // Handles keyboard input for panning and zooming.
  // Returns true if any interaction occurred.
  bool HandleKeyboard();

  // Handles mouse wheel input for scrolling.
  // Returns true if any interaction occurred.
  bool HandleWheel();

  // Processes any pending vertical scroll request to reveal a specific event.
  void ProcessPendingScroll();

  // Handles deselection of events when clicking on an empty area.
  void HandleEventDeselection();

  // Updates the search results based on the current search query.
  void RecomputeSearchResults();

  // Handles mouse input for creating curtains.
  // Returns true if any interaction occurred.
  bool HandleMouse();

  void HandleMouseDown(Pixel timeline_origin_x);
  void HandleMouseDrag(Pixel timeline_origin_x);
  void HandleMouseRelease();

  // Helper to handle bookmark addition upon mouse release.
  // Returns true if a bookmark was successfully added.
  bool HandleBookmarkAddition(bool is_click);
  // Helper to handle selection or time range addition upon mouse release.
  // Returns true if a selection was made or a time range was added.
  bool HandleSelectionOrTimeRangeAddition();

  // Computes the bounding time range of the active selection, handling either
  // multiple selection (via selected_event_indices_) or single selection
  // (via selected_event_index_). Returns std::nullopt if no event is selected.
  std::optional<TimeRange> GetSelectionTimeRange() const;

  void FindSelectedEvents(const ImRect& selection_rect);
  void CalculateAndEmitMetrics();
  void DrawSelectionRectangle();

  // Draws a notification toast at the bottom of the timeline.
  void DrawToast(absl::string_view message, float& timer, float base_y_offset);

  // Draws all the bookmarks as vertical lines.
  void DrawBookmarks(Pixel timeline_width, double px_per_time_unit_val);

  // Draws a generic close (x) button. Returns true if the button was clicked.
  bool DrawCloseButton(ImDrawList* draw_list, const ImVec2& button_pos,
                       const ImRect& hover_rect);

  // Helper to calculate the timeline area.
  ImRect GetTimelineArea() const;

  void RebuildEntryColors();
  void BuildLevelMipPyramids();
  void BuildLevelMipPyramidForLevel(int lvl);
  void EnsureLevelMipPyramid(int lvl);
  void CompactTimelineDataToMipOverview();
  static void ReleaseEntryMemory(FlameChartTimelineData& d);
  void UploadGpuFlameInstances();
  bool HasGpuFlameInstanceBuffer() const;
  void AddGpuFlameBatch(ImDrawList* draw_list, const GpuFlameBatch& batch);
  int FindEventAtLevelAndTime(int level, Microseconds ts) const;
  bool IsFlowVisibleUnderFilter(const FlowLine& flow) const;
  ImU32 GetEventColor(int event_index) const;

  // Private static constants.
  static constexpr ImGuiWindowFlags kImGuiWindowFlags =
      ImGuiWindowFlags_NoScrollWithMouse | ImGuiWindowFlags_NoCollapse |
      ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
      ImGuiWindowFlags_NoMove;
  static constexpr ImGuiWindowFlags kTrackFlags =
      ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse;

  // Synthetic groups for headers.
  Group header_all_{.name = kAllHeaderName,
                    .nesting_level = kHeaderNestingLevel};
  Group header_hidden_{.name = kHiddenHeaderName,
                       .nesting_level = kHeaderNestingLevel};
  Group header_pinned_{.name = kPinnedHeaderName,
                       .nesting_level = kHeaderNestingLevel};
  // Y coordinate offsets of section headers cached from layout computation.
  Pixel header_all_offset_ = 0.0f;
  Pixel header_hidden_offset_ = 0.0f;
  Pixel header_pinned_offset_ = 0.0f;
  // Persistent expansion/collapse states of section headers.
  bool header_hidden_expanded_ = false;
  bool header_all_expanded_ = true;
  bool header_pinned_expanded_ = true;
  // Caching counts of unhidden, hidden, and pinned process tracks.
  int all_processes_count_ = 0;
  int hidden_processes_count_ = 0;
  int pinned_processes_count_ = 0;

  FlameChartTimelineData timeline_data_;
  std::vector<ImU32> entry_colors_;
  // The dominant event color of each flame track, indexed by group index, or 0
  // if the track has no events (or is not a flame track). Rebuilt together
  // with `entry_colors_`.
  std::vector<ImU32> group_accent_colors_;
  uint64_t cached_trace_colors_version_ = std::numeric_limits<uint64_t>::max();
  std::vector<float> utilization_bins_;

 public:
  struct LevelMipTile {
    Microseconds min_start = 0.0;
    Microseconds max_end = 0.0;
    int dominant_event_index = -1;
    float total_occupied_dur = 0.0f;
    uint32_t count = 0;
    ImU32 blended_color = 0;
  };
  static_assert(sizeof(LevelMipTile) == 32);
  struct LevelMipLevel {
    Microseconds bin_width_us = 0.0;
    std::vector<LevelMipTile> tiles;
    uint32_t gpu_instance_offset = 0;
  };
  struct LevelMipPyramid {
    static constexpr int kNumMipLevels = 3;
    bool has_pyramid = false;
    bool raw_events_compacted = false;
    Microseconds level_min_ts = 0.0;
    Microseconds level_max_ts = 0.0;
    size_t visible_event_count = 0;
    LevelMipLevel levels[kNumMipLevels];
    uint32_t raw_gpu_instance_offset = UINT32_MAX;
    uint64_t built_filter_epoch = std::numeric_limits<uint64_t>::max();

    int SelectMipLevel(Microseconds max_bin_width_us) const {
      for (int m = kNumMipLevels - 1; m >= 0; --m) {
        const LevelMipLevel& mip = levels[m];
        if (!mip.tiles.empty() && mip.bin_width_us <= max_bin_width_us) {
          return m;
        }
      }
      return -1;
    }
  };
  struct ProcessUtilizationMip {
    Microseconds min_ts = 0.0;
    Microseconds bin_width_us = 0.0;
    std::vector<float> occupied_us_bins;
  };
  struct ProcessUtilizationPyramid {
    int start_level = -1;
    int end_level = -1;
    ProcessUtilizationMip levels[LevelMipPyramid::kNumMipLevels];

    int SelectMipLevel(Microseconds max_bin_width_us) const {
      for (int m = LevelMipPyramid::kNumMipLevels - 1; m >= 0; --m) {
        const ProcessUtilizationMip& mip = levels[m];
        if (!mip.occupied_us_bins.empty() &&
            mip.bin_width_us <= max_bin_width_us) {
          return m;
        }
      }
      return -1;
    }
  };
  const std::vector<LevelMipPyramid>& get_level_mip_pyramids_for_test() const {
    return level_mip_pyramids_;
  }
  void ensure_level_mip_pyramid_for_test(int lvl) {
    EnsureLevelMipPyramid(lvl);
  }
  void ensure_process_utilization_pyramids_for_test() {
    if (process_utilization_pyramids_dirty_) {
      BuildProcessUtilizationPyramids();
    }
  }
  const std::vector<ProcessUtilizationPyramid>&
  get_process_utilization_pyramids_for_test() const {
    return process_utilization_pyramids_;
  }
  bool is_flow_visible_under_filter_for_test(const FlowLine& flow) const {
    return IsFlowVisibleUnderFilter(flow);
  }
  ImU32 get_event_color_for_test(int event_index) const {
    return GetEventColor(event_index);
  }

 private:
  static constexpr size_t kMinEventsForMipPyramid = 64;
  static constexpr size_t kMinVisibleEventsForMipDraw = 128;
  static constexpr double kMaxMipBinWidthPx = 1.25;
  static constexpr double kMaxPreviewMipBinWidthPx = 2.0;
  static constexpr double kMaxProcessUtilizationMipBinWidthPx = 2.5;
  static constexpr size_t kMinEventsForProcessUtilizationPyramid = 256;
  static constexpr int kMipBinCounts[LevelMipPyramid::kNumMipLevels] = {
      2048, 512, 128};
  std::vector<LevelMipPyramid> level_mip_pyramids_;

  void BuildProcessUtilizationPyramids();
  std::optional<ProcessUtilizationPyramid>
  BuildProcessUtilizationPyramidForRange(int proc_start, int proc_end) const;
  std::vector<ProcessUtilizationPyramid> process_utilization_pyramids_;
  bool process_utilization_pyramids_dirty_ = true;

  struct CachedGpuFlameInstance {
    float start_hi = 0.0f;
    float start_lo = 0.0f;
    float duration = 0.0f;
    uint32_t color = 0;
  };
  std::vector<CachedGpuFlameInstance> gpu_flame_instances_;
  std::vector<GpuFlameBatch> emitted_gpu_flame_batches_;
  bool force_gpu_flame_batch_mode_for_test_ = false;

  struct SavedFullOverview {
    FlameChartTimelineData data;
    std::vector<ImU32> entry_colors;
    uint64_t cached_trace_colors_version = std::numeric_limits<uint64_t>::max();
    std::vector<LevelMipPyramid> level_mip_pyramids;
    std::vector<ProcessUtilizationPyramid> process_utilization_pyramids;
    std::vector<CachedGpuFlameInstance> gpu_flame_instances;
    bool process_utilization_pyramids_dirty = true;
    TimeRange fetched_data_time_range = TimeRange::Zero();
  };
  std::optional<SavedFullOverview> saved_full_overview_;
  bool is_full_overview_data_ = false;
  bool stash_full_overview_on_next_update_ = false;

  // TODO - b/444026851: Set the label width based on the real screen width.
  Pixel label_width_ = kDefaultLabelWidth;
  // The width of the timeline track area in pixels. Calculated in Draw() and
  // cached for use in interaction handlers (Zoom, Pan).
  Pixel current_timeline_width_ = 0.0f;
  // The screen Y position of the ruler, used for culling text.
  Pixel ruler_screen_y_ = 0.0f;

  ImVec2 tracks_start_screen_pos_ = {0.0f, 0.0f};
  // Whether the user is currently resizing the label column.
  bool is_resizing_label_column_ = false;

  // Stores the relative Y coordinate offset of the center of each level from
  // the top of the track area. Precalculated upon updates to the tree state.
  std::vector<Pixel> visible_level_offsets_;
  // Initialized to {0.0f} to represent the total height of 0 groups.
  // This prevents memory access out of bounds when `group_offsets_.back()`
  // is called during rendering before timeline_data_ has been fully loaded.
  std::vector<Pixel> group_offsets_ = {0.0f};
  std::vector<Pixel> group_heights_;
  // Stores whether each group is visible (not hidden by a collapsed parent).
  // Precalculated in UpdateLevelPositions.
  std::vector<bool> group_visible_;
  // Motion state (see set_animations_enabled).
  struct LayoutTransition {
    bool active = false;
    // Whether the content grows (expanding) or shrinks (collapsing); picks the
    // easing curve.
    bool expanding = true;
    float elapsed = 0.0f;
    float duration = 0.0f;
    std::vector<Pixel> from_group_offsets;
    std::vector<Pixel> to_group_offsets;
    std::vector<Pixel> from_group_heights;
    std::vector<Pixel> to_group_heights;
    std::vector<Pixel> from_level_offsets;
    std::vector<Pixel> to_level_offsets;
    // The real end-state layout. `to_group_offsets` / `to_group_heights` may
    // differ for rows leaving the screen, which animate to a tucked-away
    // position instead (see SetUpAccordionRuns).
    std::vector<Pixel> final_group_offsets;
    std::vector<Pixel> final_group_heights;
    // Rows drawn during the transition: visible before or after it.
    std::vector<bool> drawn;
    // Accordion clipping of rows sliding in or out (see SetUpAccordionRuns):
    // whether each row is clipped, and its clip edge (in track content
    // coordinates) at the start, end and current point of the transition.
    std::vector<bool> clipped;
    std::vector<Pixel> from_clip_tops;
    std::vector<Pixel> to_clip_tops;
    std::vector<Pixel> clip_tops;
    // Hidden, Pinned and All section header offsets.
    Pixel from_header_offsets[3] = {};
    Pixel to_header_offsets[3] = {};
  };
  bool animations_enabled_ = false;
  // Set by BeginExpandToggleMotion; makes the next UpdateLevelPositions
  // animate.
  bool animate_next_layout_ = false;
  // Set while drawing a frame if any motion still needs more frames.
  bool motion_in_progress_ = false;
  LayoutTransition layout_transition_;
  // In-flight chevron rotations, keyed by group index or virtual header id.
  absl::flat_hash_map<int, float> chevron_openness_;
  // In-flight hover fades, keyed by ImGui item id. Only items that are
  // hovered or fading out have entries, so this stays tiny.
  absl::flat_hash_map<ImGuiID, float> hover_fades_;

  // Stores whether each visible child track gets the alternating (zebra) row
  // stripe. The alternation restarts below every process header so the first
  // child of each process is never striped. Precalculated in
  // UpdateLevelPositions.
  std::vector<bool> group_striped_;

  // The visible time range in microseconds in the timeline. It is initialized
  // to {0, 0} by the `TimeRange` default constructor.
  // This range is updated through `SetVisibleRange`.
  // User interactions like panning and zooming also cause updates to this
  // range.
  Animated<TimeRange> visible_range_;
  // The total time range [min_time, max_time] in microseconds of the loaded
  // trace data. This range is set when trace data is processed.
  TimeRange fetched_data_time_range_ = TimeRange::Zero();
  // The total time range [min_time, max_time] in microseconds of the entire
  // trace. This might be larger than fetched_data_time_range_ if only a part
  // of the trace is loaded. This is used as the boundaries for constraining
  // panning and zooming.
  TimeRange data_time_range_ = TimeRange::Zero();

  // The index of the group of the currently selected event (flame or counter),
  // or -1 if no event is selected.
  int selected_group_index_ = -1;
  // The index of the currently selected event, or -1 if no event is selected.
  int selected_event_index_ = -1;
  // The index of the currently selected counter event in the counter data, or
  // -1 if no counter event is selected.
  int selected_counter_index_ = -1;
  int64_t selected_counter_bundle_ = -1;
  bool should_restore_scroll_ = false;
  float last_scroll_y_ = 0.0f;

  EventCallback event_callback_ = [](absl::string_view, const EventData&) {};
  // Flag to track if an event was clicked in the current frame. This is used
  // to detect clicks in empty areas for deselection logic.
  bool event_clicked_this_frame_ = false;

  // Whether the user is currently dragging the mouse on the timeline.
  bool is_dragging_ = false;
  // Controls which flow categories are visible:
  //   `FlowCategoryFilter::kAll`: Show all categories.
  //   `FlowCategoryFilter::kNone`: Show no categories.
  //   `>=0`: Show only the specific category with this ID.
  int flow_category_filter_ = FlowCategoryFilter::kNone;
  // Stores the set of flow category IDs that are currently visible.
  absl::flat_hash_set<int> visible_flow_categories_;
  // Whether the current drag operation is a selection (Shift + Drag).
  // If false, the drag operation is a pan/scroll.
  // This flag is latched at the start of the drag.
  bool is_selecting_ = false;
  // Whether the user is currently resizing an existing time range.
  struct TimeRangeResizingState {
    size_t range_index;
    bool is_start_edge;
  };
  std::optional<TimeRangeResizingState> time_range_resizing_state_;

  MouseMode mouse_mode_ = MouseMode::kPan;

  int hovered_event_index_ = -1;
  int last_reported_hovered_event_index_ = -1;
  // The index of the hovered counter or process group when no flamechart event
  // is hovered, or -1 if none is hovered.
  int hovered_group_index_ = -1;
  int last_reported_hovered_group_index_ = -1;
  // Discrete VLIW instruction bundle index (`floor(timestamp_us)`) on the
  // timeline x-axis when hovering a counter or process track without an active
  // flamechart slice, or -1 if none is hovered.
  int64_t hovered_bundle_ = -1;
  int64_t last_reported_hovered_bundle_ = -1;
  double hovered_counter_value_ = 0.0;
  bool bookmarks_enabled_ = false;
  bool track_management_enabled_ = false;
  bool timeline_player_enabled_ = false;
  bool minimap_enabled_ = false;
  // See set_event_tooltip_enabled().
  bool event_tooltip_enabled_ = true;
  bool hlo_dependency_arrows_enabled_ = false;
  // See SetSelectedEventDependencies().
  EventDependencies dependencies_;

  float panning_speed_ = kPanningSpeed;
  float zoom_speed_ = kZoomSpeed;
  float mouse_wheel_zoom_speed_ = kMouseWheelZoomSpeed;

  bool mpmd_pipeline_view_enabled_ = false;
  TimeAxisUnit time_axis_unit_ = TimeAxisUnit::kTime;

  // The index of the event to scroll to in the next Draw call.
  int event_index_to_scroll_to_ = -1;

  std::vector<TimeRange> selected_time_ranges_;
  std::vector<Microseconds> bookmarks_;

  std::optional<ImVec2> selection_start_pos_;
  std::optional<ImVec2> selection_end_pos_;
  EntryBitset selected_event_indices_;
  std::vector<std::pair<int, int>> selected_counter_points_;

  Microseconds drag_start_time_ = 0.0;
  std::optional<TimeRange> current_selected_time_range_;

  // Initialize to true to prevent sending request in the initial load where
  // JS side is already fetching the data.
  bool is_incremental_loading_ = true;
  bool full_trace_loaded_ = false;
  bool is_compacted_ = false;
  bool window_at_full_resolution_ = false;
  size_t max_uncompacted_overview_entries_ =
      kDefaultMaxUncompactedOverviewEntries;
  size_t max_uncompacted_overview_bytes_ = kDefaultMaxUncompactedOverviewBytes;

  bool incremental_fetch_enabled_ = true;

  // Stores the last requested data range to prevent redundant refetches when
  // the returned data is empty or sparse (and thus fetched_data_time_range_
  // doesn't cover the full requested range).
  TimeRange last_fetch_request_range_ = TimeRange::Zero();

  Pixel reorder_preview_line_y_ = -1.0f;
  int pending_reorder_source_ = -1;
  int pending_reorder_target_ = -1;
  bool pending_reorder_drop_after_ = false;

  EntryBitset filtered_event_indices_;
  bool has_active_event_filter_ = false;
  uint64_t filter_epoch_ = 0;

  std::string search_query_lower_;
  mutable std::vector<SearchResult> search_results_;
  int current_search_result_index_ = -1;
  std::optional<EventId> pending_navigation_event_id_;

  // Stores the remaining time (in seconds) to display the bounds
  // notification toast.
  float bounds_notification_timer_ = 0.0f;
  // The message to show in the bounds notification toast.
  std::string bounds_notification_message_ = "";

  // Stores the remaining time (in seconds) to display the "Copied!"
  // notification.
  float copy_notification_timer_ = 0.0f;
  // The name of the track that was recently copied to the clipboard.
  std::string copied_track_name_ = "";
  // The group index of the track that was recently copied to the clipboard,
  // used to briefly highlight its label. -1 if none.
  int copied_track_group_index_ = -1;
  RedrawCallback redraw_callback_;
  // Current color palette.
  ColorPalette& palette_;

  // Timeline Player states

  bool is_playing_ = false;

  Microseconds current_play_time_ = -1.0;
  double previous_play_time_ = -2.0;
  double previous_visible_start_ = -2.0;
  double previous_visible_duration_ = -2.0;

  double play_speed_ = 1.0;
  bool show_grid_ = true;

  bool minimap_schedule_dirty_ = true;
  bool has_minimap_schedule_ = false;
  Microseconds last_minimap_data_start_ = -1.0;
  Microseconds last_minimap_data_end_ = -1.0;
  Microseconds last_minimap_visible_start_ = -1.0;
  Microseconds last_minimap_visible_end_ = -1.0;
  Pixel last_minimap_label_width_ = -1.0f;
  uint64_t last_minimap_palette_version_ = std::numeric_limits<uint64_t>::max();
  uint64_t last_minimap_trace_version_ = std::numeric_limits<uint64_t>::max();

 protected:
  absl::flat_hash_set<std::string> hidden_track_names_;
  absl::flat_hash_set<std::string> pinned_track_names_;

  // Flattened sequence of virtual headers and group tracks.
  // Pre-calculated in UpdateLevelPositions to avoid CPU overhead in Draw().
  // This is the primary data structure used for rendering data as it
  // currently appears in the timeline, including collapsed groups and hidden
  // tracks.
  std::vector<const Group*> flattened_groups_;
};

}  // namespace traceviewer
#endif  // THIRD_PARTY_XPROF_FRONTEND_APP_COMPONENTS_TRACE_VIEWER_V2_TIMELINE_TIMELINE_H_
