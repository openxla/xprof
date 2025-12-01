/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#ifndef THIRD_PARTY_XPROF_CONVERT_TRACE_VIEWER_TRACE_EVENTS_H_
#define THIRD_PARTY_XPROF_CONVERT_TRACE_VIEWER_TRACE_EVENTS_H_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iterator>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/base/optimization.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/functional/bind_front.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/optional.h"
#include "xla/tsl/lib/io/iterator.h"
#include "xla/tsl/lib/io/table.h"
#include "xla/tsl/lib/io/table_builder.h"
#include "xla/tsl/lib/io/table_options.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/file_system.h"
#include "xla/tsl/platform/macros.h"
#include "xla/tsl/profiler/utils/timespan.h"
#include "tsl/platform/cpu_info.h"
#include "tsl/profiler/lib/context_types.h"
#include "xprof/convert/trace_viewer/prefix_trie.h"
#include "xprof/convert/trace_viewer/trace_events_filter_interface.h"
#include "xprof/convert/trace_viewer/trace_events_util.h"
#include "xprof/convert/trace_viewer/trace_viewer_visibility.h"
#include "xprof/convert/xprof_thread_pool_executor.h"
#include "plugin/xprof/protobuf/task.pb.h"
#include "plugin/xprof/protobuf/trace_events.pb.h"

namespace tensorflow {
namespace profiler {

// A track of events in the trace-viewer.
using TraceEventTrack = std::vector<TraceEvent*>;

using ResourceValue = std::variant<uint64_t, absl::string_view>;

static constexpr absl::string_view kTraceMetadataKey = "/trace";
// Constants used by the LevelDB Table-based efficient trace viewer storage.
static constexpr absl::string_view kLevelKey("123456789ABCDEFGHIJKLMNOPQ");

constexpr uint64_t kLayerResolutions[] = {
    1000000000000ull,  // 1 second.
    100000000000ull,  10000000000ull, 1000000000ull, 100000000ull,
    10000000ull,      1000000ull,     100000ull,     10000ull,
    1000ull,          100ull,         10ull,         1ull,
};

constexpr int NumLevels() { return TF_ARRAYSIZE(kLayerResolutions); }
static constexpr size_t kLevelDbKeyLength = 10;
static constexpr int kSearchParallelizationThreshold = 100;

// Merge-sorts the given event tracks. Each track must be sorted.
std::vector<TraceEvent*> MergeEventTracks(
    const std::vector<const TraceEventTrack*>& event_tracks);

absl::Status DoStoreAsLevelDbTable(
    std::unique_ptr<tsl::WritableFile>& file, const Trace& trace,
    const std::vector<std::vector<const TraceEvent*>>& events_by_level,
    std::function<std::optional<TraceEvent>(const TraceEvent*)>
        generate_event_copy_fn);

absl::Status DoStoreAsLevelDbTables(
    const std::vector<std::vector<const TraceEvent*>>& events_by_level,
    const Trace& trace, std::unique_ptr<tsl::WritableFile>& trace_events_file,
    std::unique_ptr<tsl::WritableFile>& trace_events_metadata_file,
    std::unique_ptr<tsl::WritableFile>& trace_events_prefix_trie_file);

// Generates a copy of the event to be persisted in the trace events file.
// This is the copy of the passed event without the timestamp_ps field.
std::optional<TraceEvent> GenerateTraceEventCopyForPersistingFullEvent(
    const TraceEvent* event);

// Generates a copy of the event to be persisted in the trace events file.
// This is the copy of the passed event without the raw_data and timestamp_ps
// fields.
std::optional<TraceEvent>
GenerateTraceEventCopyForPersistingEventWithoutMetadata(
    const TraceEvent* event);

// It generates a copy of the event to be persisted in the trace events metadata
// file. This only has the raw_data field set.
std::optional<TraceEvent> GenerateTraceEventCopyForPersistingOnlyMetadata(
    const TraceEvent* event);

// Opens the level db table from the given filename. The table is owned by the
// caller.
absl::Status OpenLevelDbTable(const std::string& filename,
                              tsl::table::Table** table,
                              std::unique_ptr<tsl::RandomAccessFile>& file);

TraceEvent::EventType GetTraceEventType(const TraceEvent& event);

struct TraceEventsLevelDbFilePaths {
  std::string trace_events_file_path;
  std::string trace_events_metadata_file_path;
  std::string trace_events_prefix_trie_file_path;
};

uint64_t TimestampFromLevelDbTableKey(absl::string_view level_db_table_key);

uint64_t LayerResolutionPs(unsigned level);

std::string LevelDbTableKey(int zoom_level, uint64_t timestamp,
                            uint64_t repetition);

bool ReadTraceMetadata(tsl::table::Iterator* iterator,
                       absl::string_view metadata_key, Trace* trace);

void PurgeIrrelevantEntriesInTraceNameTable(
    Trace& trace,
    const absl::flat_hash_set<uint64_t>& required_event_references);

template <typename RawDataType>
absl::StatusOr<absl::flat_hash_set<uint64_t>>
GetEventReferencesInTraceNameTable(const TraceEvent& event) {
  absl::flat_hash_set<uint64_t> event_references;
  if (event.has_name_ref()) {
    event_references.insert(event.name_ref());
  }
  if (event.has_raw_data()) {
    RawDataType raw_data;
    if (!raw_data.ParseFromString(event.raw_data())) {
      return absl::InternalError("Failed to parse raw_data for event: " +
                                 event.name());
    }
    for (const auto& arg : raw_data.args().arg()) {
      if (arg.has_ref_value()) {
        event_references.insert(arg.ref_value());
      }
    }
  }
  return event_references;
}

template <typename RawDataType>
absl::Status DoLoadFromLevelDbTable(
    const TraceEventsLevelDbFilePaths& file_paths,
    std::unique_ptr<TraceEventsFilterInterface> filter,
    std::unique_ptr<TraceVisibilityFilter> visibility_filter,
    int64_t filter_by_visibility_threshold, Trace& trace,
    bool& filter_by_visibility,
    const std::function<TraceEvent*(const TraceEvent&)>& copy_event_to_arena,
    const std::function<void(TraceEvent*)>& add_arena_event) {
  std::string filename = file_paths.trace_events_file_path;
  bool trace_events_metadata_file_exists = false;
  if (!file_paths.trace_events_metadata_file_path.empty()) {
    trace_events_metadata_file_exists =
        tsl::Env::Default()
            ->FileExists(file_paths.trace_events_metadata_file_path)
            .ok();
  }
  uint64_t file_size;
  TF_RETURN_IF_ERROR(tsl::Env::Default()->GetFileSize(filename, &file_size));

  tsl::FileSystem* file_system;
  TF_RETURN_IF_ERROR(
      tsl::Env::Default()->GetFileSystemForFile(filename, &file_system));

  std::unique_ptr<tsl::RandomAccessFile> file;
  TF_RETURN_IF_ERROR(file_system->NewRandomAccessFile(filename, &file));

  tsl::table::Options options;
  options.block_size = 20 * 1024 * 1024;
  tsl::table::Table* table = nullptr;
  TF_RETURN_IF_ERROR(
      tsl::table::Table::Open(options, file.get(), file_size, &table));
  std::unique_ptr<tsl::table::Table> table_deleter(table);
  std::unique_ptr<tsl::table::Iterator> iterator(table->NewIterator());
  if (iterator == nullptr) return tsl::errors::Unknown("Could not open table");

  // Read the metadata.
  iterator->SeekToFirst();
  if (!ReadTraceMetadata(iterator.get(), kTraceMetadataKey, &trace)) {
    return absl::UnknownError(
        "Could not parse Trace proto to read trace metadata");
  }

  if (filter) filter->SetUp(trace);

  tsl::profiler::Timespan visible_span;
  uint64_t container_resolution_ps = 0;

  filter_by_visibility = filter_by_visibility_threshold == -1LL ||
                         !trace.has_num_events() ||
                         trace.num_events() >= filter_by_visibility_threshold;
  if (visibility_filter) {
    if (!filter_by_visibility) {
      // disable streaming
      visibility_filter->UpdateVisibility(0);
    }
    visibility_filter->SetUp(trace);
    visible_span = visibility_filter->VisibleSpan();
    container_resolution_ps = visibility_filter->ResolutionPs();
  } else {
    visible_span = TraceSpan(trace);
  }

  // Read events at the different zoom levels.
  std::vector<std::unique_ptr<std::vector<TraceEvent*>>> loaded_events_by_level;
  size_t filtered = 0;
  TraceEvent event;  // Declared outside of the loop to avoid repeated calls to
                     // the constructor and destructor in the loop body. Cleared
                     // by every call to ParseFromCord.
  for (int i = 0;; ++i) {
    loaded_events_by_level.emplace_back(
        std::make_unique<std::vector<TraceEvent*>>());
    auto& loaded_events = *loaded_events_by_level.back();
    uint64_t resolution_ps = LayerResolutionPs(i);
    // Seek to the first element that might be in range. For the initial zoom
    // level, we don't know any bounds as events might be arbitrarily large.
    uint64_t min_timestamp_ps = 0;
    if (i > 0 && visible_span.begin_ps() > LayerResolutionPs(i - 1)) {
      min_timestamp_ps = visible_span.begin_ps() - LayerResolutionPs(i - 1);
    }
    iterator->Seek(LevelDbTableKey(i, i == 0 ? 0 : min_timestamp_ps, 0));
    while (iterator->Valid() && iterator->key().at(0) == kLevelKey[i]) {
      auto serialized_event = iterator->value();
      if (!event.ParseFromArray(serialized_event.data(),
                                serialized_event.size())) {
        return tsl::errors::Unknown("Could not parse TraceEvent proto");
      }
      uint64_t timestamp = TimestampFromLevelDbTableKey(iterator->key());
      event.set_timestamp_ps(timestamp);
      if (event.timestamp_ps() > visible_span.end_ps()) {
        // This (and all following) events are outside of our window.
        break;
      }
      // Filter before copying to the arena as it does not require sorting.
      if (!filter || !filter->Filter(event)) {
        loaded_events.push_back(copy_event_to_arena(event));
      } else {
        ++filtered;
      }
      iterator->Next();
    }
    if (container_resolution_ps >= resolution_ps) {
      // No need to read further, the resolution we just loaded already exceeds
      // the desired resolution.
      break;
    }
  }

  // We have loaded events from different zoom levels. Sort them by timestamp
  // so visibility filtering works as expected.
  std::vector<TraceEvent*> loaded_events;
  nway_merge(loaded_events_by_level, std::back_inserter(loaded_events),
             TraceEventsComparator());
  loaded_events_by_level.clear();

  LOG(INFO) << "Loaded " << loaded_events.size() << " events after filtering "
            << filtered << " events from LevelDb fast file: " << filename;
  size_t visible_events_count = 0;
  for (TraceEvent* event : loaded_events) {
    if (!visibility_filter || !visibility_filter->Filter(*event)) {
      if (trace_events_metadata_file_exists && !event->has_raw_data()) {
        RawDataType raw_data;
        tensorflow::profiler::TraceEventArguments::Argument* arg =
            raw_data.mutable_args()->add_arg();
        arg->set_name("uid");
        arg->set_int_value(event->serial());
        raw_data.SerializePartialToString(event->mutable_raw_data());
      }
      add_arena_event(event);
      ++visible_events_count;
    }
  }
  LOG(INFO) << "Added " << visible_events_count
            << " visible events from LevelDb fast file: " << filename;
  return absl::OkStatus();
}

template <typename RawDataType>
absl::Status DoSearchInLevelDbTable(
    const TraceEventsLevelDbFilePaths& file_paths,
    absl::string_view event_name_prefix,
    std::unique_ptr<TraceEventsFilterInterface> filter, Trace& trace,
    const std::function<TraceEvent*(const TraceEvent&)>& copy_event_to_arena,
    const std::function<void(TraceEvent*)>& add_arena_event) {
  auto executor =
      std::make_unique<XprofThreadPoolExecutor>("DoSearchInLevelDbTable", 2);
  std::vector<PrefixSearchResult> search_results;
  absl::Status search_results_status;
  executor->Execute([&file_paths, &event_name_prefix, &search_results,
                     &search_results_status] {
    absl::StatusOr<std::vector<PrefixSearchResult>> search_results_or =
        LoadTrieAsLevelDbTableAndSearch(
            file_paths.trace_events_prefix_trie_file_path, event_name_prefix);
    search_results_status = search_results_or.status();
    // This is to maintain the backward compatibility for old profiles where
    // prefix trie file is not present.
    if (search_results_status.code() == absl::StatusCode::kNotFound) {
      LOG(INFO) << "Prefix trie file not found: "
                << file_paths.trace_events_prefix_trie_file_path;
      search_results = {};
      search_results_status = absl::OkStatus();
      return;
    }
    if (!search_results_status.ok()) {
      LOG(ERROR) << "Failed to load and search from the prefix trie file: "
                 << file_paths.trace_events_prefix_trie_file_path
                 << " with status: " << search_results_status;
      return;
    }
    search_results = std::move(search_results_or.value());
  });

  tsl::table::Table* trace_events_table = nullptr;
  std::unique_ptr<tsl::RandomAccessFile> trace_events_file;
  absl::Status open_trace_events_table_status;
  executor->Execute([&file_paths, &trace_events_table, &trace_events_file,
                     &open_trace_events_table_status] {
    open_trace_events_table_status =
        OpenLevelDbTable(file_paths.trace_events_file_path, &trace_events_table,
                         trace_events_file);
  });
  executor->JoinAll();

  TF_RETURN_IF_ERROR(open_trace_events_table_status);
  std::unique_ptr<tsl::table::Table> trace_events_table_deleter(
      trace_events_table);
  std::unique_ptr<tsl::table::Iterator> trace_events_iterator(
      trace_events_table->NewIterator());

  TF_RETURN_IF_ERROR(search_results_status);

  if (trace_events_iterator == nullptr) {
    return absl::UnknownError("Could not open trace events table");
  }

  trace_events_iterator->SeekToFirst();
  if (!ReadTraceMetadata(trace_events_iterator.get(), kTraceMetadataKey,
                         &trace)) {
    return absl::UnknownError(
        "Could not parse Trace proto to read trace metadata");
  }
  if (filter) filter->SetUp(trace);

  std::vector<std::string> event_ids;
  for (const auto& search_result : search_results) {
    event_ids.insert(event_ids.end(), search_result.terminal_key_ids.begin(),
                     search_result.terminal_key_ids.end());
  }

  if (event_ids.empty()) {
    LOG(INFO) << "Matched 0 events from LevelDb fast file: "
              << file_paths.trace_events_file_path;
    return absl::OkStatus();
  }

  std::sort(event_ids.begin(), event_ids.end());
  const int num_threads =
      std::min(tsl::port::MaxParallelism(),
               event_ids.size() < kSearchParallelizationThreshold
                   ? 1
                   : static_cast<int>(event_ids.size()));

  executor = std::make_unique<XprofThreadPoolExecutor>("SearchEventRetrieval",
                                                       num_threads);
  std::vector<std::vector<TraceEvent>> thread_events(num_threads);
  std::vector<absl::Status> thread_statuses(num_threads);

  for (int i = 0; i < num_threads; ++i) {
    executor->Execute([&, i] {
      size_t start = (event_ids.size() * i) / num_threads;
      size_t end = (event_ids.size() * (i + 1)) / num_threads;

      std::unique_ptr<tsl::table::Iterator> iterator(
          trace_events_table->NewIterator());
      if (iterator == nullptr) {
        thread_statuses[i] =
            absl::UnknownError("Could not create table iterator");
        return;
      }

      for (size_t j = start; j < end; ++j) {
        iterator->Seek(event_ids[j]);
        if (!iterator->Valid()) {
          LOG(ERROR) << "Could not find trace event id: " << event_ids[j]
                     << "in the trace events table";
          continue;
        }
        TraceEvent event;
        auto serialized_event = iterator->value();
        if (!event.ParseFromArray(serialized_event.data(),
                                  serialized_event.size())) {
          LOG(ERROR) << "Could not parse TraceEvent proto for trace event id: "
                     << event_ids[j];
          continue;
        }
        uint64_t timestamp = TimestampFromLevelDbTableKey(event_ids[j]);
        event.set_timestamp_ps(timestamp);
        thread_events[i].push_back(std::move(event));
      }
      thread_statuses[i] = absl::OkStatus();
    });
  }
  executor->JoinAll();

  absl::Status final_status = absl::OkStatus();
  for (const auto& status : thread_statuses) {
    final_status.Update(status);
  }
  if (!final_status.ok()) {
    LOG(ERROR) << "Failed to search events: " << final_status;
    return final_status;
  }

  size_t matched_events_count = 0;
  for (auto& events : thread_events) {
    for (auto& event : events) {
      if ((!filter || !filter->Filter(event))) {
        if (!event.has_raw_data()) {
          RawDataType raw_data;
          tensorflow::profiler::TraceEventArguments::Argument* arg =
              raw_data.mutable_args()->add_arg();
          arg->set_name("uid");
          arg->set_int_value(event.serial());
          raw_data.SerializePartialToString(event.mutable_raw_data());
        }
        add_arena_event(copy_event_to_arena(event));
        ++matched_events_count;
      }
    }
  }

  LOG(INFO) << "Matched " << matched_events_count
            << " events from LevelDb fast file: "
            << file_paths.trace_events_file_path;
  return absl::OkStatus();
}

// Read full event from the level db trace events and trace events metadata
// tables. We iterate over all the zoom levels and try finding the event with
// the given timestamp and unique id for this zoom level. If the required event
// is found, we add it to the arena.
template <typename RawDataType>
absl::Status DoReadFullEventFromLevelDbTable(
    const std::string& trace_events_metadata_filename,
    const std::string& trace_events_filename, absl::string_view event_name,
    int64_t timestamp_ps, int64_t duration_ps, int64_t unique_id, Trace& trace,
    const std::function<TraceEvent*(const TraceEvent&)>& copy_event_to_arena,
    const std::function<void(TraceEvent*)>& add_arena_event) {
  tsl::table::Table* trace_events_table = nullptr;
  tsl::table::Table* trace_events_metadata_table = nullptr;
  std::unique_ptr<tsl::RandomAccessFile> trace_events_file;
  std::unique_ptr<tsl::RandomAccessFile> trace_events_metadata_file;
  auto executor = std::make_unique<XprofThreadPoolExecutor>(
      "ReadFullEventFromLevelDbTable", /*num_threads=*/2);
  absl::Status trace_events_status;
  absl::Status trace_events_metadata_status;
  executor->Execute([&trace_events_filename, &trace_events_table,
                     &trace_events_file, &trace_events_status]() {
    trace_events_status = OpenLevelDbTable(
        trace_events_filename, &trace_events_table, trace_events_file);
  });
  executor->Execute([&trace_events_metadata_filename,
                     &trace_events_metadata_table, &trace_events_metadata_file,
                     &trace_events_metadata_status]() {
    trace_events_metadata_status = OpenLevelDbTable(
        trace_events_metadata_filename, &trace_events_metadata_table,
        trace_events_metadata_file);
  });
  executor->JoinAll();
  trace_events_status.Update(trace_events_metadata_status);
  TF_RETURN_IF_ERROR(trace_events_status);

  std::unique_ptr<tsl::table::Table> trace_events_table_deleter(
      trace_events_table);
  std::unique_ptr<tsl::table::Table> trace_events_metadata_table_deleter(
      trace_events_metadata_table);
  std::unique_ptr<tsl::table::Iterator> trace_events_iterator(
      trace_events_table->NewIterator());
  std::unique_ptr<tsl::table::Iterator> trace_events_metadata_iterator(
      trace_events_metadata_table->NewIterator());
  if (trace_events_iterator == nullptr ||
      trace_events_metadata_iterator == nullptr) {
    return absl::UnknownError("Could not open table");
  }

  trace_events_iterator->SeekToFirst();
  if (!ReadTraceMetadata(trace_events_iterator.get(), kTraceMetadataKey,
                         &trace)) {
    return absl::UnknownError("Could not parse Trace proto");
  }

  for (int zoom_level = 0; zoom_level < NumLevels(); ++zoom_level) {
    std::string level_db_table_key =
        LevelDbTableKey(zoom_level, timestamp_ps, unique_id);
    trace_events_iterator->Seek(level_db_table_key);
    if (trace_events_iterator->Valid() &&
        trace_events_iterator->key() == level_db_table_key) {
      TraceEvent event;
      if (!event.ParseFromArray(trace_events_iterator->value().data(),
                                trace_events_iterator->value().size())) {
        return absl::UnknownError("Could not parse TraceEvent proto");
      }
      if (event.has_name_ref()) {
        auto it = trace.name_table().find(event.name_ref());
        if (it != trace.name_table().end()) {
          event.set_name(it->second);
        }
      }
      if (event.name() != event_name || event.duration_ps() != duration_ps) {
        continue;
      }
      trace_events_metadata_iterator->Seek(level_db_table_key);
      TraceEvent event_metadata;
      if (trace_events_metadata_iterator->Valid() &&
          trace_events_metadata_iterator->key() == level_db_table_key) {
        if (!event_metadata.ParseFromArray(
                trace_events_metadata_iterator->value().data(),
                trace_events_metadata_iterator->value().size())) {
          return absl::UnknownError("Could not parse TraceEvent proto");
        }
      }
      event.set_timestamp_ps(timestamp_ps);
      event.set_raw_data(event_metadata.raw_data());
      auto event_references_in_trace_name_table_or =
          GetEventReferencesInTraceNameTable<RawDataType>(event);
      if (!event_references_in_trace_name_table_or.ok()) {
        return event_references_in_trace_name_table_or.status();
      }
      absl::flat_hash_set<uint64_t> event_references_in_trace_name_table =
          event_references_in_trace_name_table_or.value();
      PurgeIrrelevantEntriesInTraceNameTable(
          trace, event_references_in_trace_name_table);
      add_arena_event(copy_event_to_arena(event));
      return absl::OkStatus();
    }
  }
  return absl::NotFoundError("Event not found");
}

// Reads the trace metadata from a file with given path
absl::Status ReadFileTraceMetadata(std::string& filepath, Trace* trace);

std::vector<std::vector<const TraceEvent*>> GetEventsByLevel(
    const Trace& trace, std::vector<TraceEvent*>& events);

// Return the minimum duration an event can have in `level`.
uint64_t LayerResolutionPs(unsigned level);

// Returns <lower, upper> bounds (in picoseconds) for the level that an event
// with `duration_ps` would go into. (upper >= duration_ps > lower)
std::pair<uint64_t, uint64_t> GetLevelBoundsForDuration(uint64_t duration_ps);

struct EventFactory {
  TraceEvent* Create() {
    events.push_back(std::make_unique<TraceEvent>());
    return events.back().get();
  }
  std::vector<std::unique_ptr<TraceEvent>> events;
};

struct DefaultStdHash {
  size_t operator()(absl::string_view input) {
    return std::hash<absl::string_view>()(input);
  }
};

template <typename EventFactory, typename RawData,
          typename Hash = DefaultStdHash>
class TraceEventsContainerBase {
 public:
  TraceEventsContainerBase() {
    arenas_.insert(std::make_shared<EventFactory>());
  }

  // Movable but non-copyable.
  TraceEventsContainerBase(TraceEventsContainerBase&&) = default;
  TraceEventsContainerBase& operator=(TraceEventsContainerBase&&) = default;
  TraceEventsContainerBase(const TraceEventsContainerBase&) = delete;
  TraceEventsContainerBase& operator=(const TraceEventsContainerBase&) = delete;

  void Merge(TraceEventsContainerBase&& other, int host_id);

  // Creates a TraceEvent prefilled with the given values.
  void AddCompleteEvent(absl::string_view name, uint64_t resource_id,
                        uint32_t device_id, tsl::profiler::Timespan timespan,
                        RawData* raw_data = nullptr,
                        std::optional<int64_t> group_id = std::nullopt,
                        std::optional<int64_t> serial = std::nullopt) {
    TraceEvent* event = CreateArenaEvent();
    MaybeInternEventName(event, name);
    event->set_resource_id(resource_id);
    event->set_device_id(device_id);
    event->set_timestamp_ps(timespan.begin_ps());
    if (timespan.duration_ps() != 0) {
      event->set_duration_ps(timespan.duration_ps());
    }
    if (raw_data) {
      MaybeInternTraceArgument(raw_data);
      raw_data->SerializePartialToString(event->mutable_raw_data());
      if (event->raw_data().empty()) event->clear_raw_data();
    }
    if (group_id) {
      event->set_group_id(*group_id);
    }
    if (serial && *serial > 0) {
      event->set_serial(static_cast<uint32_t>(*serial));
    }
    AddArenaEvent(event);
  }

  // Similar to above, but the TraceEvent also has an associated flow_id and
  // flow_entry_type, to make it part of a flow.
  void AddFlowEvent(absl::string_view name, uint64_t resource_id,
                    uint32_t device_id, tsl::profiler::Timespan timespan,
                    uint64_t flow_id, TraceEvent::FlowEntryType flow_entry_type,
                    tsl::profiler::ContextType flow_category =
                        tsl::profiler::ContextType::kGeneric,
                    RawData* raw_data = nullptr,
                    std::optional<int64_t> group_id = std::nullopt,
                    std::optional<int64_t> serial = std::nullopt) {
    TraceEvent* event = CreateArenaEvent();
    MaybeInternEventName(event, name);
    event->set_resource_id(resource_id);
    event->set_device_id(device_id);
    event->set_timestamp_ps(timespan.begin_ps());
    if (timespan.duration_ps() != 0) {
      event->set_duration_ps(timespan.duration_ps());
    }
    event->set_flow_id(flow_id);
    event->set_flow_entry_type(flow_entry_type);
    event->set_flow_category(static_cast<uint32_t>(flow_category));
    if (raw_data) {
      MaybeInternTraceArgument(raw_data);
      raw_data->SerializePartialToString(event->mutable_raw_data());
      if (event->raw_data().empty()) event->clear_raw_data();
    }
    if (group_id) {
      event->set_group_id(*group_id);
    }
    if (serial && *serial > 0) {
      event->set_serial(static_cast<uint32_t>(*serial));
    }
    AddArenaEvent(event);
  }

  // Similar to above, but the "async" TraceEvent don't have a resource id, its
  // name is used as "async channel" which are used as "thread" name. It has an
  // associated unique flow_id and flow_entry_type to signal asynchronous
  // start and end events and match up between them.
  void AddAsyncEvent(absl::string_view name, uint32_t device_id,
                     tsl::profiler::Timespan timespan, uint64_t flow_id,
                     TraceEvent::FlowEntryType flow_entry_type,
                     tsl::profiler::ContextType flow_category =
                         tsl::profiler::ContextType::kGeneric,
                     RawData* raw_data = nullptr,
                     std::optional<int64_t> group_id = std::nullopt,
                     std::optional<int64_t> serial = std::nullopt) {
    TraceEvent* event = CreateArenaEvent();
    MaybeInternEventName(event, name);
    event->set_device_id(device_id);
    event->set_timestamp_ps(timespan.begin_ps());
    if (timespan.duration_ps() != 0) {
      event->set_duration_ps(timespan.duration_ps());
    }
    event->set_flow_id(flow_id);
    event->set_flow_entry_type(flow_entry_type);
    event->set_flow_category(static_cast<uint32_t>(flow_category));
    if (raw_data) {
      MaybeInternTraceArgument(raw_data);
      raw_data->SerializePartialToString(event->mutable_raw_data());
      if (event->raw_data().empty()) event->clear_raw_data();
    }
    if (group_id) {
      event->set_group_id(*group_id);
    }
    if (serial && *serial > 0) {
      event->set_serial(static_cast<int32_t>(*serial));
    }
    AddArenaEvent(event);
  }

  // Similar to above, but the TraceEvent also has an associated counter name
  // and value in RawData.args. Counter events are per device, so no resource_id
  // is passed.
  void AddCounterEvent(absl::string_view name, uint32_t device_id,
                       uint64_t timestamp_ps, const RawData& raw_data,
                       std::optional<int64_t> serial = std::nullopt) {
    TraceEvent* event = CreateArenaEvent();
    event->set_name(name.data(), name.size());
    event->set_device_id(device_id);
    // Do not set resource_id for counter events, they are per device.
    event->set_timestamp_ps(timestamp_ps);
    DCHECK(raw_data.has_args());
    DCHECK_EQ(raw_data.args().arg_size(), 1);
    DCHECK(raw_data.args().arg(0).has_uint_value() ||
           raw_data.args().arg(0).has_double_value());
    raw_data.SerializePartialToString(event->mutable_raw_data());
    if (serial && *serial > 0) {
      event->set_serial(static_cast<uint32_t>(*serial));
    }
    AddArenaEvent(event);
  }

  // Returns a device descriptor.
  Device* MutableDevice(uint32_t device_id) {
    return &(*trace_.mutable_devices())[device_id];
  }

  // Returns a resource descriptor,
  Resource* MutableResource(uint64_t resource_id, uint32_t device_id) {
    Device* device = MutableDevice(device_id);
    return &(*device->mutable_resources())[resource_id];
  }

  // Adds metadata events to set the name of each device and resource.
  // The arguments are callbacks that return the names given ids.
  // This must be called after all AddEvent calls, and no more AddEvent
  // calls should be made after calling AddMetadataEvents.
  void AddMetadataEvents(
      const std::function<std::string(uint32_t /*device_id*/)>& device_name,
      const std::function<std::string(
          uint32_t /*device_id*/, uint64_t /*resource_id*/)>& resource_name) {
    for (const auto& id_and_device : events_by_device_) {
      uint32_t device_id = id_and_device.first;
      auto& device = (*trace_.mutable_devices())[device_id];
      device.set_device_id(device_id);
      device.set_name(device_name(device_id));
      const DeviceEvents& device_events = id_and_device.second;
      for (const auto& id_and_resource : device_events.events_by_resource) {
        uint64_t resource_id = id_and_resource.first;
        auto& resource = (*device.mutable_resources())[resource_id];
        resource.set_resource_id(resource_id);
        resource.set_name(resource_name(device_id, resource_id));
        resource.set_num_events(id_and_resource.second.size());
      }
    }
  }

  // Adds task metadata for the given host.
  void AddTask(int host_id, const Task& task) {
    (*trace_.mutable_tasks())[host_id] = task;
  }

  // Stores the contents of this container in a level-db sstable file.
  absl::Status StoreAsLevelDbTable(
      std::unique_ptr<tsl::WritableFile> file) const {
    Trace trace = trace_;
    trace.set_num_events(NumEvents());
    auto events_by_level = EventsByLevel();
    return DoStoreAsLevelDbTable(file, trace, events_by_level,
                                 GenerateTraceEventCopyForPersistingFullEvent);
  }

  // Stores the contents of this container in three level-db sstable files. The
  // first file contains the full events except the metadata, the second
  // file contains only the metadata and the third file contains the prefix
  // trie index over trace event names that would be used for fast prefix
  // search of events.
  absl::Status StoreAsLevelDbTables(
      std::unique_ptr<tsl::WritableFile> trace_events_file,
      std::unique_ptr<tsl::WritableFile> trace_events_metadata_file,
      std::unique_ptr<tsl::WritableFile> trace_events_prefix_trie_file) const {
    Trace trace = trace_;
    trace.set_num_events(NumEvents());
    auto events_by_level = EventsByLevel();
    return DoStoreAsLevelDbTables(events_by_level, trace, trace_events_file,
                                  trace_events_metadata_file,
                                  trace_events_prefix_trie_file);
  }

  std::vector<std::vector<const TraceEvent*>> GetTraceEventsByLevel() const {
    return EventsByLevel();
  }

  // Loads the contents of this container from a level-db sstable file.
  // In order to be efficient, requires resolution__ to be set.
  // If span_ is not set, it is initialized from the loaded trace_.
  absl::Status LoadFromLevelDbTable(
      const TraceEventsLevelDbFilePaths& trace_events_level_db_file_paths,
      std::unique_ptr<TraceEventsFilterInterface> filter = nullptr,
      std::unique_ptr<TraceVisibilityFilter> visibility = nullptr,
      int64_t filter_by_visibility_threshold = -1LL) {
    return DoLoadFromLevelDbTable<RawData>(
        trace_events_level_db_file_paths, std::move(filter),
        std::move(visibility), filter_by_visibility_threshold, trace_,
        filter_by_visibility_,
        absl::bind_front(&TraceEventsContainerBase::CopyEventToArena, this),
        absl::bind_front(&TraceEventsContainerBase::AddArenaEvent, this));
  }

  // Finds all event ids matching the given event name prefix using the prefix
  // trie sstable file then loads the full events from the level-db sstable
  // file.
  absl::Status SearchInLevelDbTable(
      const TraceEventsLevelDbFilePaths& trace_events_level_db_file_paths,
      absl::string_view event_name_prefix,
      std::unique_ptr<TraceEventsFilterInterface> filter = nullptr) {
    return DoSearchInLevelDbTable<RawData>(
        trace_events_level_db_file_paths, event_name_prefix, std::move(filter),
        trace_,
        absl::bind_front(&TraceEventsContainerBase::CopyEventToArena, this),
        absl::bind_front(&TraceEventsContainerBase::AddArenaEvent, this));
  }

  // Reads full event from level-db sstable files.
  absl::Status ReadFullEventFromLevelDbTable(
      const std::string& trace_events_metadata_filename,
      const std::string& trace_events_filename, absl::string_view event_name,
      int64_t timestamp_ps, int64_t duration_ps, int64_t unique_id) {
    return DoReadFullEventFromLevelDbTable<RawData>(
        trace_events_metadata_filename, trace_events_filename, event_name,
        timestamp_ps, duration_ps, unique_id, trace_,
        absl::bind_front(&TraceEventsContainerBase::CopyEventToArena, this),
        absl::bind_front(&TraceEventsContainerBase::AddArenaEvent, this));
  }

  // Calls 'callback' with all events stored in this container.
  template <typename Callback>
  void ForAllEvents(Callback callback) const {
    for (const auto& [device_id, device] : events_by_device_) {
      for (const auto& [counter_name, events] : device.counter_events_by_name) {
        for (auto* event : events) {
          callback(*event);
        }
      }
      for (const auto& [resource_id, events] : device.events_by_resource) {
        for (auto* event : events) {
          callback(*event);
        }
      }
    }
  }

  // Calls 'callback' with the first event of the first resource for each
  // device.
  template <typename Callback>
  void ForAllDeviceFirstEvents(Callback callback) const {
    for (const auto& [device_id, device] : events_by_device_) {
      if (!device.events_by_resource.empty() &&
          !device.events_by_resource.begin()->second.empty()) {
        callback(*device.events_by_resource.begin()->second.front());
      }
    }
  }

  // Calls 'callback' with all event tracks stored in this container.
  template <typename Callback>
  void ForAllTracks(Callback callback) const {
    for (const auto& [device_id, device] : events_by_device_) {
      for (const auto& [counter_name, events] : device.counter_events_by_name) {
        if (!events.empty()) {
          if (ABSL_PREDICT_FALSE(!callback(device_id, counter_name, events)))
            return;
        }
      }
      for (const auto& [resource_id, events] : device.events_by_resource) {
        if (!events.empty()) {
          if (ABSL_PREDICT_FALSE(!callback(device_id, resource_id, events)))
            return;
        }
      }
    }
  }

  // Calls 'callback' with all event tracks stored in this container.
  template <typename Callback>
  void ForAllMutableTracks(Callback callback) const {
    for (auto& [device_id, device] : events_by_device_) {
      for (auto& [counter_name, events] : device.counter_events_by_name) {
        if (!events.empty()) {
          callback(device_id, counter_name, &events);
        }
      }
      for (auto& [resource_id, events] : device.events_by_resource) {
        if (!events.empty()) {
          callback(device_id, resource_id, &events);
        }
      }
    }
  }

  // Calls 'callback' with all event flows stored in this container.
  template <typename Callback>
  void ForAllFlows(Callback callback) const {
    absl::flat_hash_map<uint64_t /*flow_id*/, TraceEventFlow> flows;
    for (const auto& [device_id, device] : events_by_device_) {
      // Counter events are not flow events.
      for (const auto& [resource_id, events] : device.events_by_resource) {
        for (auto* event : events) {
          if (event->has_flow_id()) flows[event->flow_id()].push_back(event);
        }
      }
    }
    for (auto& [flow_id, combined_flow] : flows) {
      // If the flow_id is reused, split into individual flows.
      for (auto& flow : SplitEventFlow(std::move(combined_flow))) {
        callback(flow_id, flow);
      }
    }
  }

  // Returns the metadata for this trace container.
  const Trace& trace() const { return trace_; }

  // Returns the number of events.
  size_t NumEvents() const {
    size_t count = 0;
    for (const auto& [device_id, device] : events_by_device_) {
      for (const auto& [counter_name, events] : device.counter_events_by_name) {
        count += events.size();
      }
      for (const auto& [resource_id, events] : device.events_by_resource) {
        count += events.size();
      }
    }
    return count;
  }

  // Returns the number of tracks.
  size_t NumTracks() const {
    return std::accumulate(
        events_by_device_.begin(), events_by_device_.end(), 0,
        [](const size_t tracks, const std::pair<uint32_t, DeviceEvents> item) {
          return tracks + item.second.counter_events_by_name.size() +
                 item.second.events_by_resource.size();
        });
  }

  bool FilterByVisibility() const { return filter_by_visibility_; }

 protected:
  // Allocates an event in the first of the arenas_.
  TraceEvent* CreateArenaEvent() { return (*arenas_.begin())->Create(); }

  // Copies event into arenas_.
  TraceEvent* CopyEventToArena(const TraceEvent& event) {
    TraceEvent* copy = CreateArenaEvent();
    *copy = event;
    return copy;
  }

  // Helper function to merge top-level trace metadata.
  void MergeTrace(const Trace& other_trace);

  // Adds an event from arenas_ to events_by_device_.
  void AddArenaEvent(TraceEvent* event) {
    ExpandTraceSpan(EventSpan(*event), &trace_);
    DeviceEvents& device_events = events_by_device_[event->device_id()];
    if (!event->has_resource_id()) {
      device_events.counter_events_by_name[event->name()].push_back(event);
    } else {
      device_events.events_by_resource[event->resource_id()].push_back(event);
    }
  }

  // Returns all events grouped by visibility level.
  std::vector<std::vector<const TraceEvent*>> EventsByLevel() const {
    std::vector<TraceEvent*> events = SortedEvents();
    return GetEventsByLevel(trace_, events);
  }

  // Returns all events sorted using TraceEventsComparator.
  // Helper for EventsByLevel().
  // REQUIRED: All events have been added and SortTracks() has been called.
  std::vector<TraceEvent*> SortedEvents() const {
    std::vector<const TraceEventTrack*> event_tracks;
    event_tracks.reserve(NumTracks());
    ForAllMutableTracks([&event_tracks](uint32_t device_id,
                                        ResourceValue resource_id,
                                        TraceEventTrack* events) {
      event_tracks.push_back(events);
    });
    return MergeEventTracks(event_tracks);
  }

  uint64_t MaybeInternString(absl::string_view name) {
    uint64_t fp = hash_(name);
    auto& it = (*trace_.mutable_name_table())[fp];
    if (it.empty()) {
      it = name;
    }
    return fp;
  }

  void MaybeInternEventName(TraceEvent* event, absl::string_view name) {
    static constexpr size_t kNameInternThreshold = 32;
    if (name.size() > kNameInternThreshold) {
      event->set_name_ref(MaybeInternString(name));
    } else {
      event->set_name(name.data(), name.size());
    }
  }

  void MaybeInternTraceArgument(RawData* raw_data) {
    if (raw_data->has_args()) {
      for (auto& arg : *raw_data->mutable_args()->mutable_arg()) {
        constexpr size_t kTraceArgInternThreshold = 16;
        if (arg.has_str_value() &&
            arg.str_value().size() > kTraceArgInternThreshold) {
          // Use name table to string intern the trace argument.
          if (arg.name() == "long_name" || arg.name() == "hlo_text") {
            // Also mark it as potential stack frame.
            arg.set_ref_value(MaybeInternString("@@" + arg.str_value()));
          } else {
            arg.set_ref_value(MaybeInternString(arg.str_value()));
          }
        }
      }
    }
  }

  // Events shown within a single device.
  struct DeviceEvents {
    // Counter events, which are per-device (don't have resource_id), and are
    // plotted in different tracks for each counter name.
    absl::flat_hash_map<std::string, TraceEventTrack> counter_events_by_name;

    // Complete events and flow events, mapped by resource_id.
    std::map<uint64_t, TraceEventTrack> events_by_resource;
  };

  // Events, mapped by device_id.
  mutable std::map<uint32_t, DeviceEvents> events_by_device_;

  // Indicator on if visibility filtering is applied or not
  // Currently skip visibility filtering only applies to ssTable
  bool filter_by_visibility_ = true;

  // The arenas containing events constructed in this container or in containers
  // that have been merged into this container.
  using Arenas = absl::flat_hash_set<std::shared_ptr<EventFactory>>;
  Arenas arenas_;

  Trace trace_;
  Hash hash_;
};

}  // namespace profiler
}  // namespace tensorflow

#endif  // THIRD_PARTY_XPROF_CONVERT_TRACE_VIEWER_TRACE_EVENTS_H_
