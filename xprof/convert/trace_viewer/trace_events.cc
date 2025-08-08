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
#include "xprof/convert/trace_viewer/trace_events.h"

#include <stddef.h>

#include <cstdint>
#include <cstring>
#include <functional>
#include <iterator>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/internal/endian.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/lib/io/iterator.h"
#include "xla/tsl/lib/io/table.h"
#include "xla/tsl/lib/io/table_builder.h"
#include "xla/tsl/lib/io/table_options.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/file_system.h"
#include "xla/tsl/platform/macros.h"
#include "xla/tsl/platform/types.h"
#include "xla/tsl/profiler/utils/timespan.h"
#include "xprof/convert/trace_viewer/prefix_trie.h"
#include "xprof/convert/trace_viewer/trace_events_util.h"
#include "xprof/convert/trace_viewer/trace_viewer_visibility.h"
#include "xprof/convert/xprof_thread_pool_executor.h"
#include "plugin/xprof/protobuf/trace_events.pb.h"
#include "plugin/xprof/protobuf/trace_events_raw.pb.h"

namespace tensorflow {
namespace profiler {
using tsl::kint64max;

namespace {

constexpr uint64_t kLayerResolutions[] = {
    1000000000000ull,  // 1 second.
    100000000000ull,  10000000000ull, 1000000000ull, 100000000ull,
    10000000ull,      1000000ull,     100000ull,     10000ull,
    1000ull,          100ull,         10ull,         1ull,
};

constexpr int NumLevels() { return TF_ARRAYSIZE(kLayerResolutions); }
static constexpr size_t kLevelDbKeyLength = 10;


// Returns the total number of events.
inline int32_t NumEvents(
    const std::vector<const TraceEventTrack*>& event_tracks) {
  int32_t num_events = 0;
  for (const auto* track : event_tracks) {
    num_events += track->size();
  }
  return num_events;
}

// Mark events with duplicated timestamp with different serial. This is to
// help front end to deduplicate events during streaming mode. The uniqueness
// is guaranteed by the tuple <device_id, timestamp_ps, serial_number>.
// REQUIRES: events is sorted by timestamp_ps
void MaybeAddEventUniqueId(std::vector<TraceEvent*>& events) {
  uint64_t last_ts = UINT64_MAX;
  uint64_t serial = 0;
  for (TraceEvent* event : events) {
    if (event->timestamp_ps() == last_ts) {
      event->set_serial(++serial);
    } else {
      serial = 0;
    }
    last_ts = event->timestamp_ps();
  }
}

}  // namespace

bool ReadTraceMetadata(tsl::table::Iterator* iterator,
                       absl::string_view metadata_key, Trace* trace) {
  if (!iterator->Valid()) return false;
  if (iterator->key() != metadata_key) return false;
  auto serialized_trace = iterator->value();
  return trace->ParseFromArray(serialized_trace.data(),
                               serialized_trace.size());
}

uint64_t TimestampFromLevelDbTableKey(absl::string_view level_db_table_key) {
  DCHECK_EQ(level_db_table_key.size(), kLevelDbKeyLength);
  uint64_t value;  // big endian representation of timestamp.
  memcpy(&value, level_db_table_key.data() + 1, sizeof(uint64_t));
  return absl::big_endian::ToHost64(value);
}

// Level Db table don't allow duplicated keys, so we add a tie break at the last
// bytes. the format is zoom[1B] + timestamp[8B] + repetition[1B]
std::string LevelDbTableKey(int zoom_level, uint64_t timestamp,
                            uint64_t repetition) {
  if (repetition >= 256) return std::string();
  std::string output(kLevelDbKeyLength, 0);
  char* ptr = output.data();
  ptr[0] = kLevelKey[zoom_level];
  // The big-endianness preserve the monotonic order of timestamp when convert
  // to lexigraphical order (of Sstable key namespace).
  uint64_t timestamp_bigendian = absl::big_endian::FromHost64(timestamp);
  memcpy(ptr + 1, &timestamp_bigendian, sizeof(uint64_t));
  ptr[9] = repetition;
  return output;
}

uint64_t LayerResolutionPs(unsigned level) {
  // This sometimes gets called in a tight loop, so levels are precomputed.
  return level >= NumLevels() ? 0 : kLayerResolutions[level];
}

std::pair<uint64_t, uint64_t> GetLevelBoundsForDuration(uint64_t duration_ps) {
  if (duration_ps == 0 || duration_ps > kLayerResolutions[0]) {
    return std::make_pair(kLayerResolutions[0], kint64max);
  }
  for (int i = 1; i < NumLevels(); ++i) {
    if (duration_ps > kLayerResolutions[i]) {
      return std::make_pair(kLayerResolutions[i], kLayerResolutions[i - 1]);
    }
  }
  // Tiny (non-zero) event. Put it in the bottom bucket. ([0, 1ps])
  return std::make_pair(0, 1);
}

std::vector<TraceEvent*> MergeEventTracks(
    const std::vector<const TraceEventTrack*>& event_tracks) {
  std::vector<TraceEvent*> events;
  events.reserve(NumEvents(event_tracks));
  nway_merge(event_tracks, std::back_inserter(events), TraceEventsComparator());
  return events;
}

std::vector<std::vector<const TraceEvent*>> GetEventsByLevel(
    const Trace& trace, std::vector<TraceEvent*>& events) {
  MaybeAddEventUniqueId(events);

  constexpr int kNumLevels = NumLevels();

  // Track visibility per zoom level.
  tsl::profiler::Timespan trace_span = TraceSpan(trace);
  std::vector<TraceViewerVisibility> visibility_by_level;
  visibility_by_level.reserve(kNumLevels);
  for (int zoom_level = 0; zoom_level < kNumLevels - 1; ++zoom_level) {
    visibility_by_level.emplace_back(trace_span, LayerResolutionPs(zoom_level));
  }

  std::vector<std::vector<const TraceEvent*>> events_by_level(kNumLevels);
  for (const TraceEvent* event : events) {
    int zoom_level = 0;
    // Find the smallest zoom level on which we can distinguish this event.
    for (; zoom_level < kNumLevels - 1; ++zoom_level) {
      if (visibility_by_level[zoom_level].VisibleAtResolution(*event)) {
        break;
      }
    }
    events_by_level[zoom_level].push_back(event);
    // Record the visibility of this event in all higher zoom levels.
    // An event on zoom level N can make events at zoom levels >N invisible.
    for (++zoom_level; zoom_level < kNumLevels - 1; ++zoom_level) {
      visibility_by_level[zoom_level].SetVisibleAtResolution(*event);
    }
  }
  return events_by_level;
}

absl::Status ReadFileTraceMetadata(std::string& filepath, Trace* trace) {
  // 1. Open the file.
  uint64_t file_size;
  TF_RETURN_IF_ERROR(tsl::Env::Default()->GetFileSize(filepath, &file_size));

  tsl::FileSystem* file_system;
  TF_RETURN_IF_ERROR(
      tsl::Env::Default()->GetFileSystemForFile(filepath, &file_system));

  std::unique_ptr<tsl::RandomAccessFile> file;
  TF_RETURN_IF_ERROR(file_system->NewRandomAccessFile(filepath, &file));

  tsl::table::Options options;
  options.block_size = 20 * 1024 * 1024;
  tsl::table::Table* table = nullptr;
  TF_RETURN_IF_ERROR(
      tsl::table::Table::Open(options, file.get(), file_size, &table));
  std::unique_ptr<tsl::table::Table> table_deleter(table);

  std::unique_ptr<tsl::table::Iterator> iterator(table->NewIterator());
  if (iterator == nullptr) return absl::UnknownError("Could not open table");

  // 2. Read the metadata.
  iterator->SeekToFirst();
  if (!ReadTraceMetadata(iterator.get(), kTraceMetadataKey, trace)) {
    return absl::UnknownError("Could not parse Trace proto");
  }
  return absl::OkStatus();
}

absl::Status CreateAndSavePrefixTrie(
    std::unique_ptr<tsl::WritableFile>& trace_events_prefix_trie_file,
    const std::vector<std::vector<const TraceEvent*>>& events_by_level) {
  PrefixTrie prefix_trie;
  for (int zoom_level = 0; zoom_level < events_by_level.size(); ++zoom_level) {
    for (const TraceEvent* event : events_by_level[zoom_level]) {
      std::string event_id =
          LevelDbTableKey(zoom_level, event->timestamp_ps(), event->serial());
      if (!event_id.empty()) {
        prefix_trie.Insert(event->name(), event_id);
      }
    }
  }
  return prefix_trie.SaveAsLevelDbTable(trace_events_prefix_trie_file);
}

absl::Status DoStoreAsLevelDbTables(
    std::unique_ptr<tsl::WritableFile>& trace_events_file,
    std::unique_ptr<tsl::WritableFile>& trace_events_metadata_file,
    std::unique_ptr<tsl::WritableFile>& trace_events_prefix_trie_file,
    const Trace& trace,
    const std::vector<std::vector<const TraceEvent*>>& events_by_level) {
  auto executor = std::make_unique<XprofThreadPoolExecutor>(
      "StoreTraceEventsAndTraceEventsMetadataLevelDbTables", /*num_threads=*/3);
  absl::Status trace_events_status, trace_events_metadata_status;
  executor->Execute(
      [&trace_events_file, &trace, &events_by_level, &trace_events_status]() {
        trace_events_status =
            DoStoreAsLevelDbTable(trace_events_file, trace, events_by_level,
              GenerateTraceEventCopyForPersistingEventWithoutMetadata);
      });
  executor->Execute([&trace_events_metadata_file, &events_by_level, &trace,
                     &trace_events_metadata_status]() {
    trace_events_metadata_status = DoStoreAsLevelDbTable(
        trace_events_metadata_file, trace, events_by_level,
        GenerateTraceEventCopyForPersistingOnlyMetadata);
  });
  absl::Status trace_events_prefix_trie_status;
  executor->Execute([&trace_events_prefix_trie_file, &events_by_level,
                     &trace_events_prefix_trie_status]() {
    trace_events_prefix_trie_status =
        CreateAndSavePrefixTrie(trace_events_prefix_trie_file, events_by_level);
  });
  executor->JoinAll();
  trace_events_status.Update(trace_events_metadata_status);
  trace_events_status.Update(trace_events_prefix_trie_status);
  return trace_events_status;
}

TraceEvent GenerateTraceEventCopyForPersistingFullEvent(
    const TraceEvent* event) {
  TraceEvent event_copy = *event;
  // To reduce file size, clear the timestamp from the value. It is
  // redundant info because the timestamp is part of the key.
  event_copy.clear_timestamp_ps();
  return event_copy;
}

TraceEvent GenerateTraceEventCopyForPersistingEventWithoutMetadata(
    const TraceEvent* event) {
  TraceEvent event_copy = *event;
  // To reduce file size, clear the timestamp from the value. It is
  // redundant info because the timestamp is part of the key.
  event_copy.clear_timestamp_ps();
  // To reduce file size, clear the raw data from the value. It is
  // redundant info because the raw data is stored in the metadata file.
  event_copy.clear_raw_data();
  return event_copy;
}

TraceEvent GenerateTraceEventCopyForPersistingOnlyMetadata(
    const TraceEvent* event) {
  TraceEvent event_copy;
  event_copy.set_raw_data(event->raw_data());
  return event_copy;
}

// Store the contents of this container in an sstable file. The format is as
// follows:
//
// key                     | value
// trace                   | The Trace-proto trace_
// 0<timestamp><serial>    | Event at timestamp visible at a 10ms resolution
// 1<timestamp><serial>    | Event at timestamp visible at a 1ms resolution
// ...
// 7<timestamp><serial>    | Event at timestamp visible at a 1ns resolution
//
// Note that each event only appears exactly once, at the first layer it's
// eligible for.
absl::Status DoStoreAsLevelDbTable(
    std::unique_ptr<tsl::WritableFile>& file, const Trace& trace,
    const std::vector<std::vector<const TraceEvent*>>& events_by_level,
    std::function<TraceEvent(const TraceEvent*)> generate_event_copy_fn) {
  LOG(INFO) << "Storing " << trace.num_events()
            << " events to LevelDb table fast file: ";
  tsl::table::Options options;
  options.block_size = 20 * 1024 * 1024;
  options.compression = tsl::table::kSnappyCompression;
  tsl::table::TableBuilder builder(options, file.get());

  builder.Add(kTraceMetadataKey, trace.SerializeAsString());

  size_t num_of_events_dropped = 0;  // Due to too many timestamp repetitions.
  for (int zoom_level = 0; zoom_level < events_by_level.size(); ++zoom_level) {
    // The key of level db table have to be monotonically increasing, therefore
    // we make the timestamp repetition count as the last byte of key as tie
    // breaker. The hidden assumption was that there are not too many identical
    // timestamp per resolution, (if there are such duplications, we dropped
    // them if it overflow the last byte).
    for (const TraceEvent* event : events_by_level[zoom_level]) {
      uint64_t timestamp = event->timestamp_ps();
      std::string key =
          LevelDbTableKey(zoom_level, timestamp, event->serial());
      if (!key.empty()) {
        TraceEvent event_copy = generate_event_copy_fn(event);
        builder.Add(key, event_copy.SerializeAsString());
      } else {
        ++num_of_events_dropped;
      }
    }
  }
  absl::string_view filename;
  TF_RETURN_IF_ERROR(file->Name(&filename));
  LOG(INFO) << "Storing " << trace.num_events() - num_of_events_dropped
            << " as LevelDb table fast file: " << filename << " with "
            << num_of_events_dropped << " events dropped.";

  TF_RETURN_IF_ERROR(builder.Finish());
  return file->Close();
}

absl::Status OpenLevelDbTable(const std::string& filename,
                               tsl::table::Table** table,
                               std::unique_ptr<tsl::RandomAccessFile>& file) {
  uint64_t file_size;
  TF_RETURN_IF_ERROR(tsl::Env::Default()->GetFileSize(filename, &file_size));
  tsl::FileSystem* file_system;
  TF_RETURN_IF_ERROR(
      tsl::Env::Default()->GetFileSystemForFile(filename, &file_system));
  TF_RETURN_IF_ERROR(file_system->NewRandomAccessFile(filename, &file));
  tsl::table::Options options;
  options.block_size = 20 * 1024 * 1024;
  TF_RETURN_IF_ERROR(
      tsl::table::Table::Open(options, file.get(), file_size, table));
  return absl::OkStatus();
}

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
      if (!trace_events_metadata_iterator->Valid() ||
          trace_events_metadata_iterator->key() != level_db_table_key) {
        return absl::UnknownError("Could not find metadata for event");
      }
      TraceEvent event_metadata;
      if (!event_metadata.ParseFromArray(
              trace_events_metadata_iterator->value().data(),
              trace_events_metadata_iterator->value().size())) {
        return absl::UnknownError("Could not parse TraceEvent proto");
      }
      event.set_timestamp_ps(timestamp_ps);
      event.set_raw_data(event_metadata.raw_data());
      add_arena_event(copy_event_to_arena(event));
      return absl::OkStatus();
    }
  }
  return absl::NotFoundError("Event not found");
}

}  // namespace profiler
}  // namespace tensorflow
