/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef XPROF_UTILS_XPLANE_LITE_H_
#define XPROF_UTILS_XPLANE_LITE_H_

#include <cstdint>
#include <deque>
#include <string>
#include <utility>
#include <variant>
#include <vector>

// Plain-struct counterparts of the XSpace protos defined in
// third_party/tensorflow/tsl/profiler/protobuf/xplane.proto.
//
// They mirror the proto fields and semantics but are laid out for fast
// construction and iteration: events are stored contiguously, metadata is
// stored in dense vectors indexed by id, and strings referenced by metadata and
// stats are stored once per plane in XPlaneLite::string_table and referred to
// by index. Each XPlaneLite owns all of its strings, so planes can be built
// independently (e.g. one per thread) without synchronization.

namespace tensorflow {
namespace profiler {

// Mirrors XStat.
struct XStatLite {
  // Mirrors XStat.str_value: index into XPlaneLite::string_table.
  struct StrValue {
    uint32_t index = 0;
  };
  // Mirrors XStat.bytes_value: index into XPlaneLite::string_table.
  struct BytesValue {
    uint32_t index = 0;
  };
  // Mirrors XStat.ref_value: the id of an XStatMetadataLite whose name holds
  // the string value.
  struct RefValue {
    uint64_t value = 0;
  };

  // XStatMetadataLite::id of the corresponding metadata.
  int64_t metadata_id = 0;

  // Mirrors the XStat.value oneof. std::monostate means no value is set.
  std::variant<std::monostate, double, uint64_t, int64_t, StrValue, BytesValue,
               RefValue>
      value;
};

// Mirrors XEvent.
//
// Kept trivially copyable and 24 bytes so that events of a line can be stored
// contiguously. XEvent.num_occurrences (aggregated events) is not represented,
// and per-event stats are stored out of line in XLineLite::event_stats.
struct XEventLite {
  // XEventMetadataLite::id of the corresponding metadata.
  int64_t metadata_id = 0;
  // Start time of the event in picoseconds, as offset from
  // XLineLite::timestamp_ns.
  int64_t offset_ps = 0;
  // Duration of the event in picoseconds. Can be zero for an instant event.
  int64_t duration_ps = 0;
};

// Mirrors XLine.
struct XLineLite {
  // Id of this line. All lines with the same id are the same timeline.
  int64_t id = 0;
  // Lines with the same display_id are grouped in the same trace viewer row.
  int64_t display_id = 0;
  std::string name;
  // Name of this line to display in trace viewer.
  std::string display_name;
  // Start time of this line in nanoseconds since the UNIX epoch.
  // XEventLite::offset_ps is relative to this timestamp.
  int64_t timestamp_ns = 0;
  // Profiling duration for this line in picoseconds.
  int64_t duration_ps = 0;
  // Events of this line, sorted by (offset_ps ascending, duration_ps
  // descending). Events may be nested but must not partially overlap.
  std::vector<XEventLite> events;
  // Stats attached to individual events, as (index into `events`, stat).
  // Only used by lines that carry per-event values, such as counters.
  std::vector<std::pair<uint32_t, XStatLite>> event_stats;
};

// Mirrors XEventMetadata.
struct XEventMetadataLite {
  // Index of this metadata in XPlaneLite::event_metadata.
  int64_t id = 0;
  // Index into XPlaneLite::string_table.
  uint32_t name_index = 0;
  // Name of the event shown in trace viewer. Index into
  // XPlaneLite::string_table; 0 (the empty string) means unset.
  uint32_t display_name_index = 0;
};

// Mirrors XStatMetadata.
struct XStatMetadataLite {
  // Index of this metadata in XPlaneLite::stat_metadata.
  int64_t id = 0;
  // Index into XPlaneLite::string_table.
  uint32_t name_index = 0;
  // Index into XPlaneLite::string_table; 0 (the empty string) means unset.
  uint32_t description_index = 0;
};

// Mirrors XPlane.
struct XPlaneLite {
  int64_t id = 0;
  std::string name;
  // Parallel timelines of this plane.
  std::vector<XLineLite> lines;
  // Dense replacement for XPlane.event_metadata: entry i has id i. Entry 0 is
  // a placeholder because metadata ids start at 1.
  std::vector<XEventMetadataLite> event_metadata;
  // Dense replacement for XPlane.stat_metadata: entry i has id i. Entry 0 is
  // a placeholder because metadata ids start at 1.
  std::vector<XStatMetadataLite> stat_metadata;
  // Stats associated with this plane, e.g. device capabilities.
  std::vector<XStatLite> stats;
  // Owns every string referenced by index from this plane's metadata and
  // stats. Each distinct string is stored once. Entry 0 is the empty string.
  std::vector<std::string> string_table = {std::string()};
};

// Mirrors XSpace.
struct XSpaceLite {
  // A deque, not a vector: adding a plane at the end never moves the existing
  // planes, so pointers and references to them stay valid (like the
  // RepeatedPtrField in XSpace). Builders rely on this.
  std::deque<XPlaneLite> planes;
  // Errors (if any) in the generation of planes.
  std::vector<std::string> errors;
  // Warnings (if any) in the generation of planes.
  std::vector<std::string> warnings;
  // Hostnames that the planes are generated from.
  std::vector<std::string> hostnames;
};

}  // namespace profiler
}  // namespace tensorflow

#endif  // XPROF_UTILS_XPLANE_LITE_H_
