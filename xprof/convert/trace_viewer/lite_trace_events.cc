#include "xprof/convert/trace_viewer/lite_trace_events.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iterator>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "google/protobuf/arena.h"
#include "xla/tsl/lib/io/table_builder.h"
#include "xla/tsl/lib/io/table_options.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/profiler/utils/timespan.h"
#include "xla/tsl/profiler/utils/xplane_schema.h"
#include "xla/tsl/profiler/utils/xplane_visitor.h"
#include "tsl/platform/cpu_info.h"
#include "xprof/convert/trace_viewer/lite_trace_events_visibility.h"
#include "xprof/convert/trace_viewer/prefix_trie.h"
#include "xprof/convert/trace_viewer/trace_events.h"
#include "xprof/convert/trace_viewer/trace_events_util.h"
#include "xprof/convert/xprof_thread_pool_executor.h"
#include "plugin/xprof/protobuf/trace_events_raw.pb.h"

namespace tensorflow {
namespace profiler {


size_t NumEvents(const TraceEventLiteContainer& container) {
  size_t count = 0;
  for (const auto& [metadata, events] : container.complete_events) {
    count += events.size();
  }
  for (const auto& [metadata, events] : container.counter_events) {
    count += events.size();
  }
  return count;
}

void MaybeAddEventUniqueId(const std::vector<TraceEventLite*>& all_events) {
  uint64_t last_ts = UINT64_MAX;
  uint32_t serial = 0;
  for (TraceEventLite* event : all_events) {
    if (event->timestamp_ps == last_ts) {
      event->serial = ++serial;
    } else {
      serial = 0;
      last_ts = event->timestamp_ps;
    }
  }
}

std::vector<const TraceEventLite*> ExtractFlowEventsParallelLite(
    const TraceEventLiteContainer& container, int num_threads) {
  size_t total_tracks =
      container.complete_events.size() + container.counter_events.size();

  std::vector<std::vector<const TraceEventLite*>> track_flow_events(
      total_tracks);

  {
    XprofThreadPoolExecutor executor("LiteEventsByLevelParallel_Pass1",
                                     num_threads);
    size_t track_idx = 0;

    for (const auto& [meta, events] : container.complete_events) {
      executor.Execute([&events, &dest = track_flow_events[track_idx]]() {
        for (const auto& ev : events) {
          if (ev.flow_id != UINT64_MAX) {
            dest.push_back(&ev);
          }
        }
      });
      track_idx++;
    }

    for (const auto& [meta, events] : container.counter_events) {
      executor.Execute([&events, &dest = track_flow_events[track_idx]]() {
        for (const auto& ev : events) {
          if (ev.flow_id != UINT64_MAX) {
            dest.push_back(&ev);
          }
        }
      });
      track_idx++;
    }
  }

  std::vector<const std::vector<const TraceEventLite*>*> track_flow_events_ptrs;
  track_flow_events_ptrs.reserve(track_flow_events.size());
  for (const auto& vec : track_flow_events) {
    if (!vec.empty()) {
      track_flow_events_ptrs.push_back(&vec);
    }
  }

  std::vector<const TraceEventLite*> flow_events;
  nway_merge(track_flow_events_ptrs, std::back_inserter(flow_events),
             TraceEventLiteComparator(container));
  return flow_events;
}

std::vector<absl::flat_hash_map<uint64_t, bool>> CalculateFlowVisibilityLite(
    const std::vector<const TraceEventLite*>& flow_events,
    const TraceEventLiteContainer& container,
    tsl::profiler::Timespan trace_span) {
  constexpr int kNumLevels = NumLevels();
  std::vector<TraceViewerVisibilityLite> visibility_by_level;
  visibility_by_level.reserve(kNumLevels);
  for (int zoom_level = 0; zoom_level < kNumLevels - 1; ++zoom_level) {
    visibility_by_level.emplace_back(trace_span, LayerResolutionPs(zoom_level));
  }

  std::vector<absl::flat_hash_map<uint64_t, bool>> flow_visibility_by_level(
      kNumLevels);

  for (const auto* ref : flow_events) {
    const auto& lite_event = *ref;
    const auto& metadata = *lite_event.metadata;

    int zoom_level = 0;
    for (; zoom_level < kNumLevels - 1; ++zoom_level) {
      bool visible = visibility_by_level[zoom_level].VisibleAtResolution(
          lite_event, metadata);

      flow_visibility_by_level[zoom_level].try_emplace(
          lite_event.flow_id, visible);
      if (visible) break;
    }
    for (++zoom_level; zoom_level < kNumLevels - 1; ++zoom_level) {
      visibility_by_level[zoom_level].SetVisibleAtResolution(lite_event,
                                                             metadata);
      flow_visibility_by_level[zoom_level].try_emplace(
          lite_event.flow_id, true);
    }
  }
  return flow_visibility_by_level;
}

std::vector<std::vector<std::vector<const TraceEventLite*>>>
ProcessTrackEventsParallelLite(
    const TraceEventLiteContainer& container,
    const std::vector<absl::flat_hash_map<uint64_t, bool>>&
        flow_visibility_by_level,
    tsl::profiler::Timespan trace_span, int num_threads) {
  constexpr int kNumLevels = NumLevels();
  size_t total_tracks =
      container.complete_events.size() + container.counter_events.size();

  std::vector<std::vector<std::vector<const TraceEventLite*>>>
      track_events_by_level(
          total_tracks,
          std::vector<std::vector<const TraceEventLite*>>(kNumLevels));

  {
    XprofThreadPoolExecutor executor("EventsByLevelLiteParallel_Pass2",
                                     num_threads);

    auto process_track =
        [&](const std::vector<TraceEventLite>& events,
            const TraceTrackMetadata& metadata,
            std::vector<std::vector<const TraceEventLite*>>& dest_slices) {
          executor.Execute([&events, &metadata, &flow_visibility_by_level,
                            trace_span, &dest_slices]() {
            std::vector<TraceViewerVisibilityLite> local_visibility_by_level;
            local_visibility_by_level.reserve(kNumLevels);
            for (int zoom_level = 0; zoom_level < kNumLevels - 1;
                 ++zoom_level) {
              local_visibility_by_level.emplace_back(
                  trace_span, LayerResolutionPs(zoom_level));
            }

            for (size_t event_idx = 0; event_idx < events.size(); ++event_idx) {
              const auto& lite_event = events[event_idx];
              int zoom_level = 0;
              if (lite_event.flow_id != UINT64_MAX) {
                for (; zoom_level < kNumLevels - 1; ++zoom_level) {
                  auto it = flow_visibility_by_level[zoom_level].find(
                      lite_event.flow_id);
                  if (it != flow_visibility_by_level[zoom_level].end() &&
                      it->second) {
                    break;
                  }
                  if (lite_event.duration_ps >= LayerResolutionPs(zoom_level)) {
                    break;
                  }
                }
                dest_slices[zoom_level].push_back(&lite_event);
                if (zoom_level < kNumLevels - 1) {
                  local_visibility_by_level[zoom_level].SetVisibleAtResolution(
                      lite_event, metadata);
                }
              } else {
                for (; zoom_level < kNumLevels - 1; ++zoom_level) {
                  if (local_visibility_by_level[zoom_level].VisibleAtResolution(
                          lite_event, metadata)) {
                    break;
                  }
                }
                dest_slices[zoom_level].push_back(&lite_event);
              }

              for (++zoom_level; zoom_level < kNumLevels - 1; ++zoom_level) {
                local_visibility_by_level[zoom_level].SetVisibleAtResolution(
                    lite_event, metadata);
              }
            }
          });
        };

    size_t track_idx = 0;

    // 1. Schedule complete event tracks
    for (const auto& [metadata, events] : container.complete_events) {
      process_track(events, metadata, track_events_by_level[track_idx++]);
    }

    // 2. Schedule counter event tracks
    for (const auto& [metadata, events] : container.counter_events) {
      process_track(events, metadata, track_events_by_level[track_idx++]);
    }
  }

  return track_events_by_level;
}

std::vector<std::vector<const TraceEventLite*>> GetLiteTraceEventsByLevel(
    const TraceEventLiteContainer& container) {
  tsl::profiler::Timespan trace_span = tsl::profiler::Timespan::FromEndPoints(
      container.trace.min_timestamp_ps(), container.trace.max_timestamp_ps());

  int num_tracks =
      container.complete_events.size() + container.counter_events.size();
  int num_threads = std::min(tsl::port::MaxParallelism(), num_tracks);
  if (num_threads <= 0) num_threads = 1;

  std::vector<const TraceEventLite*> flow_events =
      ExtractFlowEventsParallelLite(container, num_threads);

  std::vector<absl::flat_hash_map<uint64_t, bool>> flow_visibility_by_level =
      CalculateFlowVisibilityLite(flow_events, container, trace_span);

  std::vector<std::vector<std::vector<const TraceEventLite*>>>
      track_events_by_level = ProcessTrackEventsParallelLite(
          container, flow_visibility_by_level, trace_span, num_threads);

  constexpr int kNumLevels = NumLevels();
  std::vector<std::vector<const TraceEventLite*>> events_by_level(kNumLevels);
  for (int zoom_level = 0; zoom_level < kNumLevels; ++zoom_level) {
    std::vector<const std::vector<const TraceEventLite*>*> level_events_ptrs;
    level_events_ptrs.reserve(num_tracks);
    for (size_t j = 0; j < num_tracks; ++j) {
      if (!track_events_by_level[j][zoom_level].empty()) {
        level_events_ptrs.push_back(&track_events_by_level[j][zoom_level]);
      }
    }
    nway_merge(level_events_ptrs,
               std::back_inserter(events_by_level[zoom_level]),
               TraceEventLiteComparator(container));
  }

  return events_by_level;
}

std::vector<std::vector<const TraceEventLite*>> LiteTraceEventsByLevel(
    TraceEventLiteContainer* container) {
  XprofThreadPoolExecutor executor("LiteTraceEventsByLevelExecutor", 2);

  constexpr int kNumLevels = NumLevels();
  std::vector<std::vector<const TraceEventLite*>> events_by_level(kNumLevels);
  std::vector<TraceEventLite*> all_events;
  all_events.reserve(NumEvents(*container));

  executor.Execute(
      [&]() { events_by_level = GetLiteTraceEventsByLevel(*container); });

  executor.Execute([&]() {
    std::vector<std::vector<TraceEventLite*>> track_pointers;
    track_pointers.reserve(container->complete_events.size() +
                           container->counter_events.size());

    auto collect_track_pointers = [&](auto& events_map) {
      for (auto& [metadata, events] : events_map) {
        if (!events.empty()) {
          track_pointers.emplace_back();
          auto& ptrs = track_pointers.back();
          ptrs.reserve(events.size());
          for (auto& ev : events) {
            ptrs.push_back(&ev);
          }
        }
      }
    };

    collect_track_pointers(container->complete_events);
    collect_track_pointers(container->counter_events);

    std::vector<const std::vector<TraceEventLite*>*> tracks_ptrs;
    tracks_ptrs.reserve(track_pointers.size());
    for (const auto& ptrs : track_pointers) {
      tracks_ptrs.push_back(&ptrs);
    }

    nway_merge(tracks_ptrs, std::back_inserter(all_events),
               TraceEventLiteComparator(*container));
  });

  executor.JoinAll();

  MaybeAddEventUniqueId(all_events);

  return events_by_level;
}

absl::Status DoStoreLiteEventsAsTraceEventsAndMetadataLevelDbTables(
    std::unique_ptr<tsl::WritableFile>& trace_events_file,
    std::unique_ptr<tsl::WritableFile>& trace_events_metadata_file,
    const std::vector<std::vector<const TraceEventLite*>>& events_by_level,
    TraceEventLiteContainer* container,
    const TraceEventConverterFn& converter_fn) {
  absl::Time start_time = absl::Now();
  tsl::table::Options options;
  options.block_size = 20 * 1024 * 1024;
  options.compression = tsl::table::kSnappyCompression;
  tsl::table::TableBuilder trace_events_builder(options,
                                                trace_events_file.get());
  tsl::table::TableBuilder trace_events_metadata_builder(
      options, trace_events_metadata_file.get());

  constexpr size_t kChunkSize = 10000;
  constexpr size_t kMaxBufferedChunks = 20;

  size_t total_chunks = 0;
  std::vector<size_t> chunks_per_level(events_by_level.size());
  for (size_t zoom_level = 0; zoom_level < events_by_level.size();
       ++zoom_level) {
    size_t num_events = events_by_level[zoom_level].size();
    chunks_per_level[zoom_level] = (num_events + kChunkSize - 1) / kChunkSize;
    total_chunks += chunks_per_level[zoom_level];
  }

  struct CompletedChunk {
    std::vector<std::pair<std::string, std::string>> trace_events_data;
    std::vector<std::pair<std::string, std::string>> trace_events_metadata_data;
    absl::flat_hash_map<uint64_t, std::string> name_table;
  };

  struct SharedState {
    // Mutex protecting the shared state and used for condition variables.
    absl::Mutex mu;
    // Boolean flag for each chunk indicating that the worker has finished
    // writing data to completed_chunks.
    std::vector<bool> chunk_ready;
    // Pre-allocated vector to store serialized data for each chunk.
    std::vector<CompletedChunk> completed_chunks;
    // Stores the first error encountered by any worker thread.
    absl::Status status;
    // Total number of events dropped across all chunks.
    size_t dropped_events = 0;
  };

  auto shared_state = std::make_shared<SharedState>();
  shared_state->completed_chunks.resize(total_chunks);
  shared_state->chunk_ready.resize(total_chunks, false);
  absl::Status writer_status = absl::OkStatus();

  struct TaskDescriptor {
    int zoom_level;
    size_t chunk_idx;
    const std::vector<const TraceEventLite*>* level_events;
  };
  std::vector<TaskDescriptor> task_descriptors;
  task_descriptors.reserve(total_chunks);

  for (int zoom_level = 0; zoom_level < events_by_level.size(); ++zoom_level) {
    const auto& level_events = events_by_level[zoom_level];
    size_t num_chunks = chunks_per_level[zoom_level];
    for (size_t chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx) {
      task_descriptors.push_back({zoom_level, chunk_idx, &level_events});
    }
  }

  {
    XprofThreadPoolExecutor executor("SerializationPool");

    auto submit_task = [&](size_t idx) {
      const auto& desc = task_descriptors[idx];
      executor.Execute([idx, desc, shared_state, &converter_fn]() {
        {
          absl::MutexLock lock(shared_state->mu);
          if (!shared_state->status.ok()) return;
        }

        size_t start_idx = desc.chunk_idx * kChunkSize;
        size_t end_idx =
            std::min(start_idx + kChunkSize, desc.level_events->size());

        std::vector<std::pair<std::string, std::string>>
            chunk_trace_events_data;
        std::vector<std::pair<std::string, std::string>>
            chunk_trace_events_metadata_data;
        chunk_trace_events_data.reserve(end_idx - start_idx);
        chunk_trace_events_metadata_data.reserve(end_idx - start_idx);

        google::protobuf::Arena reusable_event_arena;
        TraceEvent* reusable_event =
            google::protobuf::Arena::Create<TraceEvent>(&reusable_event_arena);
        std::string trace_events_buffer;
        std::string trace_events_metadata_buffer;
        size_t dropped = 0;

        absl::flat_hash_map<uint64_t, std::string> local_name_table;

        // Pre-allocate temporary arena on stack per event with 2KB size. If
        // event size exceeds the stack buffer size, fallback to heap
        // allocation.
        char stack_buffer[2048];
        google::protobuf::ArenaOptions options;
        options.initial_block = stack_buffer;
        options.initial_block_size = sizeof(stack_buffer);
        google::protobuf::Arena event_arena(options);

        for (size_t i = start_idx; i < end_idx; ++i) {
          event_arena.Reset();

          const auto* ref = (*desc.level_events)[i];
          const auto& lite_event = *ref;
          const auto& metadata = *lite_event.metadata;

          tsl::profiler::XEventVisitor event_visitor(
              metadata.plane_visitor.get(), lite_event.line,
              &lite_event.line->events(lite_event.event_idx));

          TraceEvent* full_event =
              google::protobuf::Arena::Create<TraceEvent>(&event_arena);

          absl::Status status =
              converter_fn(event_visitor, lite_event, &local_name_table,
                           full_event, &event_arena);
          if (!status.ok()) {
            absl::MutexLock lock(shared_state->mu);
            if (shared_state->status.ok()) {
              shared_state->status = status;
            }
            return;
          }

          std::string key = LevelDbTableKey(
              desc.zoom_level, lite_event.timestamp_ps, lite_event.serial);

          if (!key.empty()) {
            absl::Status trace_events_status =
                SerializeTraceEventForPersistingEventWithoutMetadata(
                    *full_event, *reusable_event, trace_events_buffer);
            absl::Status trace_events_metadata_status =
                SerializeTraceEventForPersistingOnlyMetadata(
                    *full_event, *reusable_event, trace_events_metadata_buffer);

            if (trace_events_status.ok()) {
              chunk_trace_events_data.push_back({key, trace_events_buffer});
            } else {
              absl::MutexLock lock(shared_state->mu);
              if (shared_state->status.ok()) {
                shared_state->status = trace_events_status;
              }
              return;
            }
            if (trace_events_metadata_status.ok()) {
              chunk_trace_events_metadata_data.push_back(
                  {std::move(key), trace_events_metadata_buffer});
            } else if (!absl::IsNotFound(trace_events_metadata_status)) {
              absl::MutexLock lock(shared_state->mu);
              if (shared_state->status.ok()) {
                shared_state->status = trace_events_metadata_status;
              }
              return;
            }
          } else {
            ++dropped;
          }
        }

        shared_state->completed_chunks[idx] = {
            std::move(chunk_trace_events_data),
            std::move(chunk_trace_events_metadata_data),
            std::move(local_name_table)};

        {
          absl::MutexLock lock(shared_state->mu);
          shared_state->chunk_ready[idx] = true;
          shared_state->dropped_events += dropped;
        }
      });
    };

    size_t next_chunk_to_submit = 0;
    for (size_t i = 0; i < std::min(kMaxBufferedChunks, total_chunks); ++i) {
      submit_task(next_chunk_to_submit++);
    }

    absl::Status writer1_status = absl::OkStatus();
    absl::Status writer2_status = absl::OkStatus();

    XprofThreadPoolExecutor writer_executor("WriterPool", 2);

    // Writer 1 (Events + Name Table + Sliding Window)
    writer_executor.Execute([&]() {
      for (size_t i = 0; i < total_chunks; ++i) {
        std::vector<std::pair<std::string, std::string>> trace_events_data;
        absl::flat_hash_map<uint64_t, std::string> name_table;

        struct IsReadyArgs {
          SharedState* state;
          size_t idx;
        } args{shared_state.get(), i};

        {
          absl::MutexLock lock(shared_state->mu);
          shared_state->mu.Await(absl::Condition(
              +[](void* arg) -> bool {
                auto* a = static_cast<IsReadyArgs*>(arg);
                return a->state->chunk_ready[a->idx] || !a->state->status.ok();
              },
              &args));

          if (!shared_state->status.ok()) {
            writer1_status = shared_state->status;
            break;
          }

          trace_events_data =
              std::move(shared_state->completed_chunks[i].trace_events_data);
          name_table = std::move(shared_state->completed_chunks[i].name_table);
        }

        for (const auto& kv : trace_events_data) {
          trace_events_builder.Add(kv.first, kv.second);
        }

        for (const auto& [fp, str] : name_table) {
          container->trace.mutable_name_table()->insert({fp, str});
        }

        if (next_chunk_to_submit < total_chunks) {
          submit_task(next_chunk_to_submit++);
        }
      }
    });

    // Writer 2 (Metadata)
    writer_executor.Execute([&]() {
      for (size_t i = 0; i < total_chunks; ++i) {
        std::vector<std::pair<std::string, std::string>>
            trace_events_metadata_data;

        struct IsReadyArgs {
          SharedState* state;
          size_t idx;
        } args{shared_state.get(), i};

        {
          absl::MutexLock lock(shared_state->mu);
          shared_state->mu.Await(absl::Condition(
              +[](void* arg) -> bool {
                auto* a = static_cast<IsReadyArgs*>(arg);
                return a->state->chunk_ready[a->idx] || !a->state->status.ok();
              },
              &args));

          if (!shared_state->status.ok()) {
            writer2_status = shared_state->status;
            break;
          }

          trace_events_metadata_data = std::move(
              shared_state->completed_chunks[i].trace_events_metadata_data);
        }

        for (const auto& kv : trace_events_metadata_data) {
          trace_events_metadata_builder.Add(kv.first, kv.second);
        }
      }
    });

    writer_executor.JoinAll();

    if (!writer1_status.ok()) {
      writer_status = writer1_status;
    } else if (!writer2_status.ok()) {
      writer_status = writer2_status;
    }
  }

  TF_RETURN_IF_ERROR(writer_status);

  size_t num_of_events_dropped = shared_state->dropped_events;

  absl::string_view trace_events_filename;
  TF_RETURN_IF_ERROR(trace_events_file->Name(&trace_events_filename));
  absl::string_view trace_events_metadata_filename;
  TF_RETURN_IF_ERROR(
      trace_events_metadata_file->Name(&trace_events_metadata_filename));

  LOG(INFO) << "Storing "
            << container->trace.num_events() - num_of_events_dropped
            << " as LevelDb tables. Fast file: " << trace_events_filename
            << ", Metadata file: " << trace_events_metadata_filename
            << " with " << num_of_events_dropped << " events dropped"
            << " took " << absl::Now() - start_time;

  // Write the complete unified Trace metadata under the key "trace" at the very
  // end of the fast table!
  container->trace.set_num_events(NumEvents(*container));
  trace_events_builder.Add("trace", container->trace.SerializeAsString());

  TF_RETURN_IF_ERROR(trace_events_builder.Finish());
  TF_RETURN_IF_ERROR(trace_events_metadata_builder.Finish());

  TF_RETURN_IF_ERROR(trace_events_file->Close());
  TF_RETURN_IF_ERROR(trace_events_metadata_file->Close());

  return absl::OkStatus();
}

// Helper to resolve the event name, matching legacy step_name and display_name
// rules
absl::string_view GetEventName(
    const tsl::profiler::XEventVisitor& event_visitor) {
  absl::string_view event_name = event_visitor.HasDisplayName()
                                     ? event_visitor.DisplayName()
                                     : event_visitor.Name();

  std::optional<tsl::profiler::XStatVisitor> step_name_stat =
      event_visitor.GetStat(tsl::profiler::StatType::kStepName);
  if (!step_name_stat.has_value()) {
    step_name_stat =
        event_visitor.Metadata().GetStat(tsl::profiler::StatType::kStepName);
  }
  if (step_name_stat.has_value()) {
    event_name = step_name_stat->StrOrRefValue();
  }
  return event_name;
}

absl::Status DoStoreLiteEventsAsLevelDbTables(
    const std::vector<std::vector<const TraceEventLite*>>& events_by_level,
    TraceEventLiteContainer* container,
    const TraceEventConverterFn& converter_fn,
    std::unique_ptr<tsl::WritableFile>& trace_events_file,
    std::unique_ptr<tsl::WritableFile>& trace_events_metadata_file,
    std::unique_ptr<tsl::WritableFile>& trace_events_prefix_trie_file) {
  absl::Status trace_events_status;
  absl::Status trace_events_prefix_trie_status;

  XprofThreadPoolExecutor executor("StoreLiteEventsAsLevelDbTables", 2);
  executor.Execute([&] {
    trace_events_status =
        DoStoreLiteEventsAsTraceEventsAndMetadataLevelDbTables(
            trace_events_file, trace_events_metadata_file, events_by_level,
            container, converter_fn);
  });
  if (trace_events_prefix_trie_file) {
    executor.Execute([&] {
      trace_events_prefix_trie_status =
          CreateAndSavePrefixTrieLite(trace_events_prefix_trie_file.get(),
                                      events_by_level, *container);
    });
  }

  executor.JoinAll();

  TF_RETURN_IF_ERROR(trace_events_status);
  TF_RETURN_IF_ERROR(trace_events_prefix_trie_status);

  return absl::OkStatus();
}

absl::string_view GetEventNameForTrie(
    const TraceEventLite* lite_event,
    const TraceEventLiteContainer& container) {
  const auto& metadata = *lite_event->metadata;
  if (metadata.resource_id == UINT32_MAX) {
    auto it = container.name_table.find(metadata.name_ref);
    if (it != container.name_table.end()) {
      return it->second;
    }
    return "";
  }
  const auto& event = lite_event->line->events(lite_event->event_idx);
  tsl::profiler::XEventVisitor event_visitor(metadata.plane_visitor.get(),
                                             lite_event->line, &event);
  return GetEventName(event_visitor);
}

absl::Status CreateAndSavePrefixTrieLite(
    tsl::WritableFile* trace_events_prefix_trie_file,
    const std::vector<std::vector<const tensorflow::profiler::TraceEventLite*>>&
        events_by_level,
    const tensorflow::profiler::TraceEventLiteContainer& container) {
  absl::Time start_time = absl::Now();
  PrefixTrie prefix_trie;

  constexpr size_t kChunkSize = 10000;
  constexpr size_t kMaxBufferedChunks = 20;

  size_t total_chunks = 0;
  std::vector<size_t> chunks_per_level(events_by_level.size());
  for (int zoom_level = 0; zoom_level < events_by_level.size(); ++zoom_level) {
    size_t num_events = events_by_level[zoom_level].size();
    chunks_per_level[zoom_level] = (num_events + kChunkSize - 1) / kChunkSize;
    total_chunks += chunks_per_level[zoom_level];
  }

  struct CompletedChunk {
    std::vector<std::pair<absl::string_view, std::string>> trie_data;
  };

  struct SharedState {
    // Mutex protecting the shared state and used for condition variables.
    absl::Mutex mu;
    // Boolean flag for each chunk indicating that the worker has finished
    // writing data to completed_chunks.
    std::vector<bool> chunk_ready;
    // Pre-allocated vector to store resolved trie data for each chunk.
    std::vector<CompletedChunk> completed_chunks;
    // Stores the first error encountered by any worker thread.
    absl::Status status;
  };

  auto shared_state = std::make_shared<SharedState>();
  shared_state->completed_chunks.resize(total_chunks);
  shared_state->chunk_ready.resize(total_chunks, false);
  absl::Status writer_status = absl::OkStatus();

  struct TaskDescriptor {
    int zoom_level;
    size_t chunk_idx;
    const std::vector<const TraceEventLite*>* level_events;
  };
  std::vector<TaskDescriptor> task_descriptors;
  task_descriptors.reserve(total_chunks);

  for (int zoom_level = 0; zoom_level < events_by_level.size(); ++zoom_level) {
    const auto& level_events = events_by_level[zoom_level];
    size_t num_chunks = chunks_per_level[zoom_level];
    for (size_t chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx) {
      task_descriptors.push_back({zoom_level, chunk_idx, &level_events});
    }
  }

  {
    XprofThreadPoolExecutor executor("TrieResolutionPool");

    auto submit_task = [&](size_t idx) {
      const auto& desc = task_descriptors[idx];
      executor.Execute([idx, desc, shared_state, &container]() {
        {
          absl::MutexLock lock(shared_state->mu);
          if (!shared_state->status.ok()) return;
        }

        size_t start_idx = desc.chunk_idx * kChunkSize;
        size_t end_idx =
            std::min(start_idx + kChunkSize, desc.level_events->size());

        std::vector<std::pair<absl::string_view, std::string>> chunk_trie_data;
        chunk_trie_data.reserve(end_idx - start_idx);

        for (size_t i = start_idx; i < end_idx; ++i) {
          const auto* lite_event = (*desc.level_events)[i];

          absl::string_view event_name =
              GetEventNameForTrie(lite_event, container);

          std::string event_id = LevelDbTableKey(
              desc.zoom_level, lite_event->timestamp_ps, lite_event->serial);

          if (!event_id.empty() && !event_name.empty()) {
            chunk_trie_data.push_back({event_name, std::move(event_id)});
          }
        }

        shared_state->completed_chunks[idx] = {std::move(chunk_trie_data)};

        {
          absl::MutexLock lock(shared_state->mu);
          shared_state->chunk_ready[idx] = true;
        }
      });
    };

    size_t next_chunk_to_submit = 0;
    for (size_t i = 0; i < std::min(kMaxBufferedChunks, total_chunks); ++i) {
      submit_task(next_chunk_to_submit++);
    }

    // Sequential Inserter loop inside Trie Thread
    for (size_t i = 0; i < total_chunks; ++i) {
      CompletedChunk completed_chunk;
      struct IsReadyArgs {
        SharedState* state;
        size_t idx;
      } args{shared_state.get(), i};

      {
        absl::MutexLock lock(shared_state->mu);
        shared_state->mu.Await(absl::Condition(
            +[](void* arg) -> bool {
              auto* a = static_cast<IsReadyArgs*>(arg);
              return a->state->chunk_ready[a->idx] || !a->state->status.ok();
            },
            &args));

        if (!shared_state->status.ok()) {
          writer_status = shared_state->status;
          break;
        }

        completed_chunk = std::move(shared_state->completed_chunks[i]);
      }

      for (const auto& [name, id] : completed_chunk.trie_data) {
        prefix_trie.Insert(name, id);
      }

      if (next_chunk_to_submit < total_chunks) {
        submit_task(next_chunk_to_submit++);
      }
    }
  }

  TF_RETURN_IF_ERROR(writer_status);

  absl::string_view filename;
  TF_RETURN_IF_ERROR(trace_events_prefix_trie_file->Name(&filename));
  LOG(INFO) << "Prefix trie created to be stored in file: " << filename
            << " took " << absl::Now() - start_time;
  return prefix_trie.SaveAsLevelDbTable(trace_events_prefix_trie_file);
}

absl::Status StoreLiteEventsAsLevelDbTables(
    TraceEventLiteContainer* container,
    const TraceEventConverterFn& converter_fn,
    std::unique_ptr<tsl::WritableFile>& trace_events_file,
    std::unique_ptr<tsl::WritableFile>& trace_events_metadata_file,
    std::unique_ptr<tsl::WritableFile>& trace_events_prefix_trie_file) {
  absl::Time start_time = absl::Now();
  container->trace.set_num_events(NumEvents(*container));

  auto events_by_level = LiteTraceEventsByLevel(container);

  absl::string_view trace_events_file_name;
  TF_RETURN_IF_ERROR(trace_events_file->Name(&trace_events_file_name));
  LOG(INFO) << "Preprocess events for storing as leveldb table: "
            << trace_events_file_name << "Time: " << absl::Now() - start_time;

  return DoStoreLiteEventsAsLevelDbTables(
      events_by_level, container, converter_fn, trace_events_file,
      trace_events_metadata_file, trace_events_prefix_trie_file);
}

void MergeLiteTraceEventsContainers(TraceEventLiteContainer* src,
                                    TraceEventLiteContainer* dst) {
  dst->tracks_metadata.insert(
      dst->tracks_metadata.end(),
      std::make_move_iterator(src->tracks_metadata.begin()),
      std::make_move_iterator(src->tracks_metadata.end()));
  src->tracks_metadata.clear();

  for (auto& [metadata, events] : src->complete_events) {
    auto [it, inserted] =
        dst->complete_events.try_emplace(metadata, std::move(events));
    if (!inserted) {
      auto& dst_events = it->second;
      dst_events.insert(dst_events.end(),
                        std::make_move_iterator(events.begin()),
                        std::make_move_iterator(events.end()));
    }
  }

  for (auto& [metadata, events] : src->counter_events) {
    auto [it, inserted] =
        dst->counter_events.try_emplace(metadata, std::move(events));
    if (!inserted) {
      auto& dst_events = it->second;
      dst_events.insert(dst_events.end(),
                        std::make_move_iterator(events.begin()),
                        std::make_move_iterator(events.end()));
    }
  }

  dst->name_table.insert(src->name_table.begin(), src->name_table.end());

  for (auto& [dev_id, src_dev] : *src->trace.mutable_devices()) {
    auto& dst_dev = (*dst->trace.mutable_devices())[dev_id];
    if (dst_dev.name().empty()) {
      dst_dev = std::move(src_dev);
    } else {
      for (auto& [res_id, src_res] : *src_dev.mutable_resources()) {
        auto& dst_res = (*dst_dev.mutable_resources())[res_id];
        if (dst_res.name().empty()) {
          dst_res = std::move(src_res);
        } else {
          dst_res.set_num_events(dst_res.num_events() + src_res.num_events());
        }
      }
    }
  }

  for (auto& [task_id, src_task] : *src->trace.mutable_tasks()) {
    dst->trace.mutable_tasks()->try_emplace(task_id, std::move(src_task));
  }

  dst->trace.mutable_name_table()->insert(src->trace.name_table().begin(),
                                          src->trace.name_table().end());

  if (src->trace.has_min_timestamp_ps()) {
    if (!dst->trace.has_min_timestamp_ps() ||
        src->trace.min_timestamp_ps() < dst->trace.min_timestamp_ps()) {
      dst->trace.set_min_timestamp_ps(src->trace.min_timestamp_ps());
    }
  }
  if (src->trace.has_max_timestamp_ps()) {
    if (!dst->trace.has_max_timestamp_ps() ||
        src->trace.max_timestamp_ps() > dst->trace.max_timestamp_ps()) {
      dst->trace.set_max_timestamp_ps(src->trace.max_timestamp_ps());
    }
  }
}

}  // namespace profiler
}  // namespace tensorflow
