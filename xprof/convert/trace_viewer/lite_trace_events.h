#ifndef THIRD_PARTY_XPROF_CONVERT_TRACE_VIEWER_LITE_TRACE_EVENTS_H_
#define THIRD_PARTY_XPROF_CONVERT_TRACE_VIEWER_LITE_TRACE_EVENTS_H_

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/functional/any_invocable.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "google/protobuf/arena.h"
#include "xla/tsl/platform/file_system.h"
#include "xla/tsl/profiler/utils/group_events.h"
#include "xla/tsl/profiler/utils/timespan.h"
#include "xla/tsl/profiler/utils/xplane_visitor.h"
#include "tsl/profiler/lib/context_types.h"
#include "tsl/profiler/protobuf/xplane.pb.h"
#include "plugin/xprof/protobuf/trace_events.pb.h"

namespace tensorflow {
namespace profiler {

struct TraceTrackMetadata {
  uint32_t resource_id = UINT32_MAX;
  uint32_t device_id = 0;
  std::shared_ptr<const tsl::profiler::XPlaneVisitor> plane_visitor;
  uint64_t name_ref = 0;

  bool operator==(const TraceTrackMetadata& other) const {
    return device_id == other.device_id &&
           resource_id == other.resource_id &&
           name_ref == other.name_ref;
  }

  template <typename H>
  friend H AbslHashValue(H h, const TraceTrackMetadata& meta) {
    return H::combine(std::move(h), meta.device_id, meta.resource_id,
                      meta.name_ref);
  }
};

struct TraceEventLite {
  uint64_t timestamp_ps = 0;
  uint64_t duration_ps = 0;
  uint64_t flow_id = UINT64_MAX;
  const tsl::profiler::XLine* line = nullptr;
  const TraceTrackMetadata* metadata = nullptr;
  uint32_t event_idx = 0;
  uint32_t serial = 0;
  uint8_t flow_entry_type = 0;
  bool is_async = false;
};

struct DmaFlowMetadata {
  uint64_t kilobytes = 0;
  uint64_t src_address = 0;
  uint64_t dst_address = 0;
  uint64_t last_write_start_cycle = 0;
};

struct DeduplicatedFusionMetadata {
  std::string duplicate_of;
};

struct LiteIngestionMetadata {
  absl::flat_hash_map<uint64_t, DmaFlowMetadata> dma_flow_map;
  tsl::profiler::GroupMetadataMap group_metadata_map;
  absl::flat_hash_map<std::pair<const tsl::profiler::XLine*, uint32_t>,
                      DeduplicatedFusionMetadata>
      dedup_fusion_map;
  absl::flat_hash_map<std::string, std::string> dedup_name_to_hlo_text;
};

struct TraceEventLiteContainer {
  std::vector<std::unique_ptr<TraceTrackMetadata>> tracks_metadata;
  absl::flat_hash_map<TraceTrackMetadata, std::vector<TraceEventLite>>
      counter_events;
  absl::flat_hash_map<TraceTrackMetadata, std::vector<TraceEventLite>>
      complete_events;
  absl::flat_hash_map<uint64_t, std::string> name_table;
  Trace trace;

  // Global processed metadata annotations (populated post-merge, not merged in
  // MergeLiteTraceEventsContainers)
  LiteIngestionMetadata metadata;
};

void MergeLiteTraceEventsContainers(TraceEventLiteContainer* src,
                                    TraceEventLiteContainer* dst);

class LiteTraceEventsContainerManager {
 public:
  LiteTraceEventsContainerManager() = default;

  void Reserve(int32_t num_containers) {
    containers_.reserve(num_containers);
  }

  TraceEventLiteContainer* Add() {
    containers_.push_back(std::make_unique<TraceEventLiteContainer>());
    return containers_.back().get();
  }

  void MergeAll(TraceEventLiteContainer* dst) {
    for (auto& src : containers_) {
      MergeLiteTraceEventsContainers(src.get(), dst);
    }
    containers_.clear();
  }

 private:
  std::vector<std::unique_ptr<TraceEventLiteContainer>> containers_;
};

struct TraceEventLiteComparator {
  explicit TraceEventLiteComparator(const TraceEventLiteContainer& c)
      : container(c) {}

  bool operator()(const TraceEventLite& a, const TraceEventLite& b) const {
    return operator()(&a, &b);
  }

  bool operator()(const TraceEventLite* a, const TraceEventLite* b) const {
    if (a->timestamp_ps != b->timestamp_ps) {
      return a->timestamp_ps < b->timestamp_ps;
    }
    if (a->duration_ps != b->duration_ps) {
      return a->duration_ps > b->duration_ps;
    }

    const auto* meta_a = a->metadata;
    const auto* meta_b = b->metadata;

    if (meta_a->device_id != meta_b->device_id) {
      return meta_a->device_id < meta_b->device_id;
    }

    bool is_resource_a = (meta_a->resource_id != UINT32_MAX) && !a->is_async;
    bool is_resource_b = (meta_b->resource_id != UINT32_MAX) && !b->is_async;

    if (is_resource_a && !is_resource_b) return true;
    if (!is_resource_a && is_resource_b) return false;

    if (is_resource_a) {
      return meta_a->resource_id < meta_b->resource_id;
    }

    return container.name_table.at(meta_a->name_ref) <
           container.name_table.at(meta_b->name_ref);
  }

  const TraceEventLiteContainer& container;
};

template <typename HashFn = std::hash<absl::string_view>>
uint64_t MaybeInternString(
    absl::string_view str,
    absl::flat_hash_map<uint64_t, std::string>* local_name_table) {
  uint64_t fp = HashFn()(str);
  local_name_table->try_emplace(fp, str);
  return fp;
}

template <typename HashFn = std::hash<absl::string_view>>
void MaybeInternEventName(
    TraceEvent* event, absl::string_view name,
    absl::flat_hash_map<uint64_t, std::string>* local_name_table) {
  static constexpr size_t kNameInternThreshold = 32;
  if (name.size() > kNameInternThreshold) {
    event->set_name_ref(MaybeInternString<HashFn>(name, local_name_table));
  } else {
    event->set_name(name.data(), name.size());
  }
}

template <typename HashFn = std::hash<absl::string_view>, typename RawDataType>
void MaybeInternTraceArgument(
    RawDataType* raw_data,
    absl::flat_hash_map<uint64_t, std::string>* local_name_table) {
  if (raw_data->has_args()) {
    for (auto& arg : *raw_data->mutable_args()->mutable_arg()) {
      constexpr size_t kTraceArgInternThreshold = 16;
      if (arg.has_str_value() &&
          arg.str_value().size() > kTraceArgInternThreshold) {
        if (arg.name() == "long_name" || arg.name() == "hlo_text") {
          arg.set_ref_value(MaybeInternString<HashFn>(
              absl::StrCat("@@", arg.str_value()), local_name_table));
        } else {
          arg.set_ref_value(
              MaybeInternString<HashFn>(arg.str_value(), local_name_table));
        }
      }
    }
  }
}

template <typename HashFn = std::hash<absl::string_view>, typename RawDataType>
void CreateCounterEvent(absl::string_view name, uint32_t device_id,
                        uint64_t timestamp_ps, const RawDataType& raw_data,
                        std::optional<int64_t> serial, TraceEvent* event) {
  event->set_name(name.data(), name.size());
  event->set_device_id(device_id);
  event->set_timestamp_ps(timestamp_ps);
  DCHECK(raw_data.has_args());
  DCHECK_EQ(raw_data.args().arg_size(), 1);
  DCHECK(raw_data.args().arg(0).has_uint_value() ||
         raw_data.args().arg(0).has_double_value());
  raw_data.SerializePartialToString(event->mutable_raw_data());
  if (serial && *serial > 0) {
    event->set_serial(static_cast<uint32_t>(*serial));
  }
}

template <typename HashFn = std::hash<absl::string_view>, typename RawDataType>
void CreateAsyncEvent(
    absl::string_view name, uint32_t device_id,
    const tsl::profiler::Timespan& span, uint64_t flow_id,
    TraceEvent::FlowEntryType flow_entry_type,
    tsl::profiler::ContextType flow_category, RawDataType* raw_data,
    std::optional<int64_t> group_id, std::optional<int64_t> serial,
    absl::flat_hash_map<uint64_t, std::string>* local_name_table,
    TraceEvent* event) {
  MaybeInternEventName<HashFn>(event, name, local_name_table);
  event->set_device_id(device_id);
  event->set_timestamp_ps(span.begin_ps());
  if (span.duration_ps() > 0) {
    event->set_duration_ps(span.duration_ps());
  }
  event->set_flow_id(flow_id);
  event->set_flow_entry_type(flow_entry_type);
  event->set_flow_category(static_cast<uint32_t>(flow_category));
  if (raw_data) {
    MaybeInternTraceArgument<HashFn>(raw_data, local_name_table);
    raw_data->SerializePartialToString(event->mutable_raw_data());
    if (event->raw_data().empty()) event->clear_raw_data();
  }
  if (group_id.has_value()) {
    event->set_group_id(*group_id);
  }
  if (serial && *serial > 0) {
    event->set_serial(static_cast<uint32_t>(*serial));
  }
}

template <typename HashFn = std::hash<absl::string_view>, typename RawDataType>
void CreateFlowEvent(
    absl::string_view name, uint32_t resource_id, uint32_t device_id,
    const tsl::profiler::Timespan& span, uint64_t flow_id,
    TraceEvent::FlowEntryType flow_entry_type,
    tsl::profiler::ContextType flow_category, RawDataType* raw_data,
    std::optional<int64_t> group_id, std::optional<int64_t> serial,
    absl::flat_hash_map<uint64_t, std::string>* local_name_table,
    TraceEvent* event) {
  MaybeInternEventName<HashFn>(event, name, local_name_table);
  event->set_device_id(device_id);
  event->set_resource_id(resource_id);
  event->set_timestamp_ps(span.begin_ps());
  if (span.duration_ps() > 0) {
    event->set_duration_ps(span.duration_ps());
  }
  event->set_flow_id(flow_id);
  event->set_flow_entry_type(flow_entry_type);
  event->set_flow_category(static_cast<uint32_t>(flow_category));
  if (raw_data) {
    MaybeInternTraceArgument<HashFn>(raw_data, local_name_table);
    raw_data->SerializePartialToString(event->mutable_raw_data());
    if (event->raw_data().empty()) event->clear_raw_data();
  }
  if (group_id.has_value()) {
    event->set_group_id(*group_id);
  }
  if (serial && *serial > 0) {
    event->set_serial(static_cast<uint32_t>(*serial));
  }
}

template <typename HashFn = std::hash<absl::string_view>, typename RawDataType>
void CreateCompleteEvent(
    absl::string_view name, uint32_t resource_id, uint32_t device_id,
    const tsl::profiler::Timespan& span, RawDataType* raw_data,
    std::optional<int64_t> group_id, std::optional<int64_t> serial,
    absl::flat_hash_map<uint64_t, std::string>* local_name_table,
    TraceEvent* event) {
  MaybeInternEventName<HashFn>(event, name, local_name_table);
  event->set_device_id(device_id);
  event->set_resource_id(resource_id);
  event->set_timestamp_ps(span.begin_ps());
  if (span.duration_ps() > 0) {
    event->set_duration_ps(span.duration_ps());
  }
  if (raw_data) {
    MaybeInternTraceArgument<HashFn>(raw_data, local_name_table);
    raw_data->SerializePartialToString(event->mutable_raw_data());
    if (event->raw_data().empty()) event->clear_raw_data();
  }
  if (group_id.has_value()) {
    event->set_group_id(*group_id);
  }
  if (serial && *serial > 0) {
    event->set_serial(static_cast<uint32_t>(*serial));
  }
}

size_t NumEvents(const TraceEventLiteContainer& container);

using TraceEventConverterFn = absl::AnyInvocable<absl::Status(
    const tsl::profiler::XEventVisitor& event_visitor,
    const TraceEventLite& lite_event,
    absl::flat_hash_map<uint64_t, std::string>* local_name_table,
    TraceEvent* full_event, google::protobuf::Arena* arena) const>;

void MaybeAddEventUniqueId(const std::vector<TraceEventLite*>& all_events);

std::vector<std::vector<const TraceEventLite*>> LiteTraceEventsByLevel(
    TraceEventLiteContainer* container);

std::vector<std::vector<const TraceEventLite*>> GetLiteTraceEventsByLevel(
    const TraceEventLiteContainer& container);

absl::Status StoreLiteEventsAsLevelDbTables(
    TraceEventLiteContainer* container,
    const TraceEventConverterFn& converter_fn,
    std::unique_ptr<tsl::WritableFile>& trace_events_file,
    std::unique_ptr<tsl::WritableFile>& trace_events_metadata_file,
    std::unique_ptr<tsl::WritableFile>& trace_events_prefix_trie_file);

absl::Status CreateAndSavePrefixTrieLite(
    tsl::WritableFile* trace_events_prefix_trie_file,
    const std::vector<std::vector<const tensorflow::profiler::TraceEventLite*>>&
        events_by_level,
    const tensorflow::profiler::TraceEventLiteContainer& container);

absl::Status DoStoreLiteEventsAsTraceEventsAndMetadataLevelDbTables(
    std::unique_ptr<tsl::WritableFile>& trace_events_file,
    std::unique_ptr<tsl::WritableFile>& trace_events_metadata_file,
    const std::vector<std::vector<const TraceEventLite*>>& events_by_level,
    TraceEventLiteContainer* container,
    const TraceEventConverterFn& converter_fn);

absl::Status DoStoreLiteEventsAsLevelDbTables(
    const std::vector<std::vector<const TraceEventLite*>>& events_by_level,
    TraceEventLiteContainer* container,
    const TraceEventConverterFn& converter_fn,
    std::unique_ptr<tsl::WritableFile>& trace_events_file,
    std::unique_ptr<tsl::WritableFile>& trace_events_metadata_file,
    std::unique_ptr<tsl::WritableFile>& trace_events_prefix_trie_file);

}  // namespace profiler
}  // namespace tensorflow

#endif  // THIRD_PARTY_XPROF_CONVERT_TRACE_VIEWER_LITE_TRACE_EVENTS_H_
