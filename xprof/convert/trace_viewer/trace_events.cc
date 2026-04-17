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

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iterator>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/numeric/bits.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "google/protobuf/arena.h"
#include "google/protobuf/io/coded_stream.h"
#include "xla/tsl/lib/io/iterator.h"
#include "xla/tsl/lib/io/table.h"
#include "xla/tsl/lib/io/table_builder.h"
#include "xla/tsl/lib/io/table_options.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/file_system.h"
#include "xla/tsl/profiler/utils/timespan.h"
#include "tsl/platform/cpu_info.h"
#include "xprof/convert/trace_viewer/prefix_trie.h"
#include "xprof/convert/trace_viewer/trace_events_util.h"
#include "xprof/convert/trace_viewer/trace_viewer_visibility.h"
#include "xprof/convert/xprof_thread_pool_executor.h"
#include "plugin/xprof/protobuf/trace_events.pb.h"
#include "plugin/xprof/protobuf/trace_events_raw.pb.h"

namespace tensorflow {
namespace profiler {

namespace {

// Returns the total number of events.
inline int32_t NumEvents(
    const std::vector<const TraceEventTrack*>& event_tracks) {
  int32_t num_events = 0;
  for (const auto* track : event_tracks) {
    num_events += track->size();
  }
  return num_events;
}


// Appends all events from src into dst.
inline void AppendEvents(TraceEventTrack&& src, TraceEventTrack* dst) {
  if (dst->empty()) {
    *dst = std::move(src);
  } else {
    absl::c_move(src, std::back_inserter(*dst));
  }
}

template <typename Modifier>
absl::Status SerializeWithReusableEvent(const TraceEvent& event,
                                        TraceEvent& reusable_event_copy,
                                        std::string& output,
                                        Modifier modifier) {
  reusable_event_copy = event;
  modifier(&reusable_event_copy);

  output.clear();
  return reusable_event_copy.AppendToString(&output)
             ? absl::OkStatus()
             : absl::InternalError("Failed to serialize trace event");
}

}  // namespace

// Mark events with duplicated timestamp with different serial. This is to
// help front end to deduplicate events during streaming mode. The uniqueness
// is guaranteed by the tuple <device_id, timestamp_ps, serial_number>.
void MaybeAddEventUniqueId(
    const std::vector<const TraceEventTrack*>& event_tracks) {
  std::vector<TraceEvent*> all_events;
  std::vector<const std::vector<TraceEvent*>*> tracks_ptrs;
  tracks_ptrs.reserve(event_tracks.size());
  for (const auto* track : event_tracks) {
    if (!track->empty()) {
      tracks_ptrs.push_back(track);
    }
  }
  nway_merge(tracks_ptrs, std::back_inserter(all_events),
             TraceEventsComparator());

  uint64_t last_ts = UINT64_MAX;
  uint64_t serial = 0;
  for (TraceEvent* event : all_events) {
    if (event->timestamp_ps() == last_ts) {
      event->set_serial(++serial);
    } else {
      serial = 0;
    }
    last_ts = event->timestamp_ps();
  }
}

TraceEvent::EventType GetTraceEventType(const TraceEvent& event) {
  return event.has_resource_id() ? TraceEvent::EVENT_TYPE_COMPLETE
         : event.has_flow_id()   ? TraceEvent::EVENT_TYPE_ASYNC
                                 : TraceEvent::EVENT_TYPE_COUNTER;
}

bool ReadTraceMetadata(tsl::table::Iterator* iterator,
                       absl::string_view metadata_key, Trace* trace) {
  if (!iterator->Valid()) return false;
  if (iterator->key() != metadata_key) return false;
  auto serialized_trace = iterator->value();
  return trace->ParseFromString(serialized_trace);
}

uint64_t TimestampFromLevelDbTableKey(absl::string_view level_db_table_key) {
  DCHECK_EQ(level_db_table_key.size(), kLevelDbKeyLength);
  uint64_t value;  // big endian representation of timestamp.
  memcpy(&value, level_db_table_key.data() + 1, sizeof(uint64_t));
  if constexpr (absl::endian::native == absl::endian::little) {
    return absl::byteswap<uint64_t>(value);
  } else {
    return value;
  }
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
  // to lexicographical order (of Sstable key namespace).
  uint64_t timestamp_bigendian;
  if constexpr (absl::endian::native == absl::endian::little) {
    timestamp_bigendian = absl::byteswap<uint64_t>(timestamp);
  } else {
    timestamp_bigendian = timestamp;
  }

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
    return std::make_pair(kLayerResolutions[0],
                          std::numeric_limits<int64_t>::max());
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
    const Trace& trace,
    const std::vector<const TraceEventTrack*>& event_tracks) {
  int num_threads = std::min(tsl::port::MaxParallelism(),
                             static_cast<int>(event_tracks.size()));
  if (num_threads <= 0) num_threads = 1;

  // Pass 1: Extract flows in parallel.
  std::vector<std::vector<const TraceEvent*>> track_flow_events(
      event_tracks.size());
  {
    auto executor = std::make_unique<XprofThreadPoolExecutor>(
        "EventsByLevelParallel_Pass1", num_threads);
    for (int i = 0; i < num_threads; ++i) {
      executor->Execute([&, i] {
        size_t start = (event_tracks.size() * i) / num_threads;
        size_t end = (event_tracks.size() * (i + 1)) / num_threads;
        for (size_t j = start; j < end; ++j) {
          const TraceEventTrack* track = event_tracks[j];
          for (const TraceEvent* event : *track) {
            if (event->has_flow_id()) {
              track_flow_events[j].push_back(event);
            }
          }
        }
      });
    }
  }

  // Merge and sort flows using N-way merge.
  std::vector<const TraceEvent*> flow_events;
  std::vector<const std::vector<const TraceEvent*>*> track_flow_events_ptrs;
  track_flow_events_ptrs.reserve(track_flow_events.size());
  for (const auto& vec : track_flow_events) {
    if (!vec.empty()) {
      track_flow_events_ptrs.push_back(&vec);
    }
  }
  nway_merge(track_flow_events_ptrs, std::back_inserter(flow_events),
             TraceEventsComparator());

  // Calculate flow visibility.
  constexpr int kNumLevels = NumLevels();
  tsl::profiler::Timespan trace_span = TraceSpan(trace);
  std::vector<TraceViewerVisibility> visibility_by_level;
  visibility_by_level.reserve(kNumLevels);
  for (int zoom_level = 0; zoom_level < kNumLevels - 1; ++zoom_level) {
    visibility_by_level.emplace_back(trace_span, LayerResolutionPs(zoom_level));
  }

  std::vector<absl::flat_hash_map<uint64_t, bool>> flow_visibility_by_level(
      kNumLevels);

  for (const TraceEvent* event : flow_events) {
    int zoom_level = 0;
    for (; zoom_level < kNumLevels - 1; ++zoom_level) {
      bool visible =
          visibility_by_level[zoom_level].VisibleAtResolution(*event);
      flow_visibility_by_level[zoom_level].try_emplace(event->flow_id(),
                                                       visible);
      if (visible) {
        break;
      }
    }
    for (++zoom_level; zoom_level < kNumLevels - 1; ++zoom_level) {
      visibility_by_level[zoom_level].SetVisibleAtResolution(*event);
      flow_visibility_by_level[zoom_level].try_emplace(event->flow_id(), true);
    }
  }

  // Pass 2: Parallel track processing.
  // track_events_by_level[track_index][zoom_level]
  std::vector<std::vector<std::vector<const TraceEvent*>>>
      track_events_by_level(
          event_tracks.size(),
          std::vector<std::vector<const TraceEvent*>>(kNumLevels));
  {
    auto executor = std::make_unique<XprofThreadPoolExecutor>(
        "EventsByLevelParallel_Pass2", num_threads);
    for (int i = 0; i < num_threads; ++i) {
      executor->Execute([&, i] {
        size_t start = (event_tracks.size() * i) / num_threads;
        size_t end = (event_tracks.size() * (i + 1)) / num_threads;

        for (size_t j = start; j < end; ++j) {
          const TraceEventTrack* track = event_tracks[j];

          std::vector<TraceViewerVisibility> track_visibility_by_level;
          track_visibility_by_level.reserve(kNumLevels);
          for (int zoom_level = 0; zoom_level < kNumLevels - 1; ++zoom_level) {
            track_visibility_by_level.emplace_back(
                trace_span, LayerResolutionPs(zoom_level));
          }

          for (const TraceEvent* event : *track) {
            int zoom_level = 0;

            if (event->has_flow_id()) {
              for (; zoom_level < kNumLevels - 1; ++zoom_level) {
                auto it =
                    flow_visibility_by_level[zoom_level].find(event->flow_id());
                if (it != flow_visibility_by_level[zoom_level].end() &&
                    it->second) {
                  break;
                }
              }
            } else {
              for (; zoom_level < kNumLevels - 1; ++zoom_level) {
                if (track_visibility_by_level[zoom_level].VisibleAtResolution(
                        *event)) {
                  break;
                }
              }
            }

            track_events_by_level[j][zoom_level].push_back(event);

            for (++zoom_level; zoom_level < kNumLevels - 1; ++zoom_level) {
              track_visibility_by_level[zoom_level].SetVisibleAtResolution(
                  *event);
            }
          }
        }
      });
    }
  }

  // Final Merge using N-way merge per level.
  std::vector<std::vector<const TraceEvent*>> events_by_level(kNumLevels);
  for (int zoom_level = 0; zoom_level < kNumLevels; ++zoom_level) {
    std::vector<const std::vector<const TraceEvent*>*> level_events_ptrs;
    level_events_ptrs.reserve(event_tracks.size());
    for (size_t j = 0; j < event_tracks.size(); ++j) {
      if (!track_events_by_level[j][zoom_level].empty()) {
        level_events_ptrs.push_back(&track_events_by_level[j][zoom_level]);
      }
    }
    nway_merge(level_events_ptrs,
               std::back_inserter(events_by_level[zoom_level]),
               TraceEventsComparator());
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
    tsl::WritableFile* trace_events_prefix_trie_file,
    const std::vector<std::vector<const TraceEvent*>>& events_by_level) {
  absl::Time start_time = absl::Now();
  PrefixTrie prefix_trie;
  for (int zoom_level = 0; zoom_level < events_by_level.size(); ++zoom_level) {
    for (const TraceEvent* event : events_by_level[zoom_level]) {
      uint64_t timestamp = event->timestamp_ps();
      std::string event_id =
          LevelDbTableKey(zoom_level, timestamp, event->serial());
      if (!event_id.empty()) {
        prefix_trie.Insert(event->name(), event_id);
      }
    }
  }
  absl::string_view filename;
  TF_RETURN_IF_ERROR(trace_events_prefix_trie_file->Name(&filename));
  LOG(INFO) << "Prefix trie created to be stored in file: " << filename
            << " took " << absl::Now() - start_time;
  return prefix_trie.SaveAsLevelDbTable(trace_events_prefix_trie_file);
}

absl::Status DoStoreAsLevelDbTables(
    const std::vector<std::vector<const TraceEvent*>>& events_by_level,
    const Trace& trace, std::unique_ptr<tsl::WritableFile>& trace_events_file,
    std::unique_ptr<tsl::WritableFile>& trace_events_metadata_file,
    std::unique_ptr<tsl::WritableFile>& trace_events_prefix_trie_file) {
  std::unique_ptr<XprofThreadPoolExecutor> executor =
      std::make_unique<XprofThreadPoolExecutor>("StoreAsLevelDbTables",
                                                /*num_threads=*/3);
  absl::Status trace_events_status, trace_events_metadata_status;
  executor->Execute(
      [&trace_events_file, &trace, &events_by_level, &trace_events_status]() {
        trace_events_status = DoStoreAsLevelDbTable(
            trace_events_file, trace, events_by_level,
            SerializeTraceEventForPersistingEventWithoutMetadata);
      });
  executor->Execute([&trace_events_metadata_file, &events_by_level, &trace,
                     &trace_events_metadata_status]() {
    trace_events_metadata_status = DoStoreAsLevelDbTable(
        trace_events_metadata_file, trace, events_by_level,
        SerializeTraceEventForPersistingOnlyMetadata);
  });
  absl::Status trace_events_prefix_trie_status;
  executor->Execute([&trace_events_prefix_trie_file, &events_by_level,
                     &trace_events_prefix_trie_status]() {
    trace_events_prefix_trie_status = CreateAndSavePrefixTrie(
        trace_events_prefix_trie_file.get(), events_by_level);
  });
  executor->JoinAll();
  trace_events_status.Update(trace_events_metadata_status);
  trace_events_status.Update(trace_events_prefix_trie_status);
  return trace_events_status;
}

absl::Status SerializeTraceEventForPersistingFullEvent(
    const TraceEvent& event, TraceEvent& reusable_event_copy,
    std::string& output) {
  return SerializeWithReusableEvent(
      event, reusable_event_copy, output,
      [](TraceEvent* copy) { copy->clear_timestamp_ps(); });
}

absl::Status SerializeTraceEventForPersistingEventWithoutMetadata(
    const TraceEvent& event, TraceEvent& reusable_event_copy,
    std::string& output) {
  return SerializeWithReusableEvent(
      event, reusable_event_copy, output, [](TraceEvent* copy) {
        copy->clear_timestamp_ps();
        if (GetTraceEventType(*copy) != TraceEvent::EVENT_TYPE_COUNTER) {
          copy->clear_raw_data();
        }
      });
}

absl::Status SerializeTraceEventForPersistingOnlyMetadata(
    const TraceEvent& event, TraceEvent& reusable_event, std::string& output) {
  // Counter events are stored in the trace events file itself and do not
  // require a metadata copy.
  if (GetTraceEventType(event) == TraceEvent::EVENT_TYPE_COUNTER) {
    return absl::NotFoundError("No metadata found for counter event");
  }

  // Reconstruct explicitly avoiding a deep copy inside
  // SerializeWithReusableEvent to save substantial memory allocation
  // operations.
  reusable_event.Clear();
  reusable_event.set_raw_data(event.raw_data());
  output.clear();
  return reusable_event.AppendToString(&output)
             ? absl::OkStatus()
             : absl::InternalError("Failed to serialize trace event");
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
    SerializeEventFn serialize_event_fn) {
  absl::Time start_time = absl::Now();
  tsl::table::Options options;
  options.block_size = 20 * 1024 * 1024;
  options.compression = tsl::table::kSnappyCompression;
  tsl::table::TableBuilder builder(options, file.get());

  builder.Add(kTraceMetadataKey, trace.SerializeAsString());

  size_t num_of_events_dropped = 0;  // Due to too many timestamp repetitions.
  google::protobuf::Arena arena;
  TraceEvent* reusable_event = google::protobuf::Arena::Create<TraceEvent>(&arena);
  std::string buffer;
  for (int zoom_level = 0; zoom_level < events_by_level.size(); ++zoom_level) {
    // The key of level db table have to be monotonically increasing, therefore
    // we make the timestamp repetition count as the last byte of key as tie
    // breaker. The hidden assumption was that there are not too many identical
    // timestamp per resolution, (if there are such duplications, we dropped
    // them if it overflow the last byte).
    for (const TraceEvent* event : events_by_level[zoom_level]) {
      uint64_t timestamp = event->timestamp_ps();
      std::string key = LevelDbTableKey(zoom_level, timestamp, event->serial());
      if (!key.empty()) {
        absl::Status status =
            serialize_event_fn(*event, *reusable_event, buffer);
        if (status.ok()) {
          builder.Add(key, buffer);
        } else if (!absl::IsNotFound(status)) {
          return status;
        }
      } else {
        ++num_of_events_dropped;
      }
    }
  }
  absl::string_view filename;
  TF_RETURN_IF_ERROR(file->Name(&filename));
  LOG(INFO) << "Storing " << trace.num_events() - num_of_events_dropped
            << " as LevelDb table fast file: " << filename << " with "
            << num_of_events_dropped << " events dropped"
            << " took " << absl::Now() - start_time;

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

void PurgeIrrelevantEntriesInTraceNameTable(
    Trace& trace,
    const absl::flat_hash_set<uint64_t>& required_event_references) {
  google::protobuf::Map<uint64_t, std::string> new_name_table;
  for (const auto& reference : required_event_references) {
    if (trace.name_table().contains(reference)) {
      new_name_table.insert({reference, trace.name_table().at(reference)});
    }
  }
  trace.mutable_name_table()->swap(new_name_table);
}

template <typename EventFactory, typename RawData, typename Hash>
void TraceEventsContainerBase<EventFactory, RawData, Hash>::MergeTrace(
    const Trace& other_trace) {
  trace_.mutable_tasks()->insert(other_trace.tasks().begin(),
                                 other_trace.tasks().end());
  trace_.mutable_name_table()->insert(other_trace.name_table().begin(),
                                      other_trace.name_table().end());
  if (other_trace.has_min_timestamp_ps() &&
      other_trace.has_max_timestamp_ps()) {
    ExpandTraceSpan(TraceSpan(other_trace), &trace_);
  }
  trace_.set_num_events(trace_.num_events() + other_trace.num_events());
}

template <typename EventFactory, typename RawData, typename Hash>
void TraceEventsContainerBase<EventFactory, RawData, Hash>::Merge(
    TraceEventsContainerBase&& other, int host_id) {
  if (this == &other) return;
  if (other.NumEvents() == 0 && other.trace().devices().empty()) return;

  const int kMaxDevicesPerHost = 1000;
  absl::flat_hash_map<uint32_t, uint32_t> other_to_this_device_id_map;
  auto& this_device_map = *trace_.mutable_devices();

  // Handle device id collisions.
  // TODO: b/452643006 - Check if this logic can be moved to
  // xplane_to_trace_container.
  for (const auto& [other_id, other_device] : other.trace().devices()) {
    LOG(WARNING) << "Remapping device id " << other_id << " for host "
                 << host_id << " to "
                 << other_id + host_id * kMaxDevicesPerHost;
    uint32_t target_id = other_id + host_id * kMaxDevicesPerHost;
    other_to_this_device_id_map[other_id] = target_id;

    Device device_copy = other_device;
    device_copy.set_device_id(target_id);

    this_device_map.insert({target_id, device_copy});
  }

  other.ForAllMutableTracks([this, &other_to_this_device_id_map](
                                uint32_t other_device_id,
                                ResourceValue resource_id_or_counter_name,
                                TraceEventTrack* track) {
    uint32_t this_device_id = other_to_this_device_id_map.at(other_device_id);
    for (TraceEvent* event : *track) {
      event->set_device_id(this_device_id);
    }
    DeviceEvents& device = this->events_by_device_[this_device_id];
    if (const uint64_t* resource_id =
            std::get_if<uint64_t>(&resource_id_or_counter_name)) {
      AppendEvents(std::move(*track), &device.events_by_resource[*resource_id]);
    } else if (const absl::string_view* counter_name =
                   std::get_if<absl::string_view>(
                       &resource_id_or_counter_name)) {
      AppendEvents(std::move(*track),
                   &device.counter_events_by_name[*counter_name]);
    }
  });

  MergeTrace(other.trace());
  arenas_.insert(std::make_move_iterator(other.arenas_.begin()),
                 std::make_move_iterator(other.arenas_.end()));
  other.arenas_.clear();
  other.events_by_device_.clear();
  other.trace_.Clear();
}


// Explicit instantiations for the common case.
template class TraceEventsContainerBase<EventFactory, RawData>;

}  // namespace profiler
}  // namespace tensorflow
