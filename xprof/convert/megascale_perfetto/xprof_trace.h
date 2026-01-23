#ifndef THIRD_PARTY_XPROF_CONVERT_MEGASCALE_PERFETTO_XPROF_TRACE_H_
#define THIRD_PARTY_XPROF_CONVERT_MEGASCALE_PERFETTO_XPROF_TRACE_H_

#include <cstddef>
#include <cstdint>
#include <string>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/lib/gtl/int_type.h"

// XprofTrace is a container for holding the events from an Xprof trace.
// It facilitates performant processing (vs using an XSpace proto) and is
// intended to be used by TraceProcessor.

namespace xprof::megascale {

// -----------------------------------------------------------------------------
// String Interning
// -----------------------------------------------------------------------------

// Represents an index into the XprofTrace::string_table.
TSL_LIB_GTL_DEFINE_INT_TYPE(StringId, size_t);

// A helper to store unique strings once and refer to them by integer ID.
class StringTable {
 public:
  // Returns existing ID or creates a new one.
  StringId Intern(absl::string_view s) {
    auto it = index_.find(s);
    if (it != index_.end()) {
      return it->second;
    }
    StringId id(store_.size());
    store_.push_back(std::string(s));
    index_.emplace(store_.back(), id);
    return id;
  }

  absl::string_view Get(StringId id) const { return store_[id.value()]; }

 private:
  std::vector<std::string> store_;
  absl::flat_hash_map<std::string, StringId> index_;
};

// -----------------------------------------------------------------------------
// Event Arguments
// -----------------------------------------------------------------------------

// An arg value can be a number or a reference to a string in the table.
using ArgValue = std::variant<int64_t, uint64_t, double, StringId>;

struct Arg {
  StringId key;  // The key is also interned (e.g. "step_number", "formula")
  ArgValue value;
};

// -----------------------------------------------------------------------------
// Flows
// -----------------------------------------------------------------------------

enum class FlowDirection { kSource, kSink };

struct Flow {
  int64_t id;
  FlowDirection direction;
  bool is_terminating = false;
};

// -----------------------------------------------------------------------------
// Event Structure
// -----------------------------------------------------------------------------

struct Event {
  std::string name;

  int64_t timestamp_ps;
  int64_t duration_ps;

  std::vector<Arg> args;

  // Logic Fields (Populated by trace processor)
  int64_t run_id = -1;
  std::vector<Flow> flows;
};

struct Track {
  std::string name;
  std::vector<Event> events;
};

template <typename T>
struct CounterTrack {
  std::string name;
  std::vector<int64_t> timestamps_ps;
  std::vector<T> values;
};

// -----------------------------------------------------------------------------
// Root Container
// -----------------------------------------------------------------------------

struct XprofTrace {
  StringTable string_table;

  // Data sourced from "/device:TPU:X" planes.
  // Key: TPU ID (e.g., 0, 1)
  // Value: The standard XLA tracks (Steps, Ops, Modules)
  absl::flat_hash_map<int64_t, std::vector<Track>> tpu_fragments;

  // Data sourced from "/device:CUSTOM:Megascale Trace" plane.
  // Key: TPU ID (derived from graph_key)
  // Value: The Megascale specific tracks
  absl::flat_hash_map<int64_t, std::vector<Track>> megascale_fragments;

  // Counter tracks sourced from various sources.
  CounterTrack<int64_t> rx_counter;
  CounterTrack<int64_t> tx_counter;
  CounterTrack<double> rx_bw_counter;
  CounterTrack<double> tx_bw_counter;
};

}  // namespace xprof::megascale

#endif  // THIRD_PARTY_XPROF_CONVERT_MEGASCALE_PERFETTO_XPROF_TRACE_H_
