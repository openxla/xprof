#include "xprof/convert/trace_viewer/delta_series/trace_data_to_compressed_delta_series_proto.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/string_view.h"
#include "tsl/profiler/lib/context_types.h"
#include "xprof/convert/trace_viewer/trace_events.h"
#include "plugin/xprof/protobuf/trace_data_response.pb.h"

namespace tensorflow {
namespace profiler {

DeltaSeriesProtoConverter::DeltaSeriesProtoConverter(
    const Trace* trace, CounterExtractor counter_extractor,
    const DeltaSeriesProtoConversionOptions& options)
    : trace_(trace),
      counter_extractor_(std::move(counter_extractor)),
      options_(options) {}

uint32_t DeltaSeriesProtoConverter::MaybeInternString(absl::string_view str) {
  auto it = interned_strings_map_.find(str);
  if (it != interned_strings_map_.end()) {
    return it->second;
  }
  uint32_t interned_string_id = interned_strings_.size();
  interned_strings_.push_back(std::string(str));
  interned_strings_map_[str] = interned_string_id;
  return interned_string_id;
}

xprof::TraceMetadata DeltaSeriesProtoConverter::GetTraceMetadata() const {
  xprof::TraceMetadata metadata;
  for (const auto& [device_id, device] : trace_->devices()) {
    xprof::Process* process = metadata.add_processes();
    process->set_id(device_id);
    if (device.has_name()) {
      process->set_name(device.name());
    }
    uint32_t sort_index = device_id;
    if (auto it = mpmd_sort_indices_.find(device_id);
        it != mpmd_sort_indices_.end()) {
      sort_index = it->second;
    }
    process->set_sort_index(sort_index);
    for (const auto& [resource_id, resource] : device.resources()) {
      xprof::Thread* thread = process->add_threads();
      thread->set_id(resource_id);
      if (resource.has_name()) {
        thread->set_name(resource.name());
      }
      thread->set_sort_index(resource_id);
    }
  }
  return metadata;
}

void DeltaSeriesProtoConverter::PopulateTraceEventMetadata(
    const TraceEvent& event, xprof::TraceEventMetadata* metadata) {
  if (event.has_flow_id()) {
    metadata->set_flow_id(event.flow_id());
  }
  if (event.has_flow_category()) {
    tsl::profiler::ContextType type =
        tsl::profiler::GetSafeContextType(event.flow_category());
    if (type != tsl::profiler::ContextType::kGeneric &&
        type != tsl::profiler::ContextType::kLegacy) {
      const char* category = tsl::profiler::GetContextTypeString(type);
      metadata->set_flow_category(MaybeInternString(category));
    }
  }
  if (event.has_group_id()) {
    metadata->set_group_id(event.group_id());
  }
  if (event.has_serial()) {
    metadata->set_serial(event.serial());
  }
}

void DeltaSeriesProtoConverter::AddCompleteEventTrack(
    uint32_t pid, uint64_t tid, const TraceEventTrack& events,
    xprof::TraceDataResponse* response) {
  xprof::TraceEventSeries* series = response->add_complete_events();
  xprof::TraceEventSeriesMetadata* series_metadata = series->mutable_metadata();
  series_metadata->set_process_id(pid);
  series_metadata->set_thread_id(tid);

  uint64_t last_timestamp = 0;
  for (const auto* event : events) {
    uint64_t current_timestamp = event->timestamp_ps();
    series->add_deltas(current_timestamp - last_timestamp);
    last_timestamp = current_timestamp;

    series->add_durations(event->duration_ps());

    std::string event_name = event->has_name_ref()
                                 ? trace_->name_table().at(event->name_ref())
                                 : event->name();
    series->add_name_refs(MaybeInternString(event_name));

    PopulateTraceEventMetadata(*event, series->add_event_metadata());
  }
}

void DeltaSeriesProtoConverter::AddAsyncEventTrack(
    uint32_t pid, absl::string_view name, const TraceEventTrack& events,
    xprof::TraceDataResponse* response) {
  xprof::TraceEventSeries* series = response->add_async_events();
  xprof::TraceEventSeriesMetadata* series_metadata = series->mutable_metadata();
  series_metadata->set_process_id(pid);
  series_metadata->set_name_ref(MaybeInternString(name));

  uint64_t last_timestamp = 0;
  for (const auto* event : events) {
    uint64_t current_timestamp = event->timestamp_ps();
    series->add_deltas(current_timestamp - last_timestamp);
    last_timestamp = current_timestamp;

    series->add_durations(event->duration_ps());
    PopulateTraceEventMetadata(*event, series->add_event_metadata());
  }
}

void DeltaSeriesProtoConverter::AddCounterEventTrack(
    uint32_t pid, absl::string_view name, const TraceEventTrack& events,
    xprof::TraceDataResponse* response) {
  xprof::TraceEventSeries* series = response->add_counter_events();
  xprof::TraceEventSeriesMetadata* series_metadata = series->mutable_metadata();
  series_metadata->set_process_id(pid);
  series_metadata->set_name_ref(MaybeInternString(name));

  uint64_t last_timestamp = 0;
  for (const auto* event : events) {
    uint64_t current_timestamp = event->timestamp_ps();
    series->add_deltas(current_timestamp - last_timestamp);
    last_timestamp = current_timestamp;

    xprof::TraceEventMetadata* event_metadata = series->add_event_metadata();
    if (event->has_raw_data()) {
      counter_extractor_(event->raw_data(), event_metadata);
    }
  }
}

}  // namespace profiler
}  // namespace tensorflow
