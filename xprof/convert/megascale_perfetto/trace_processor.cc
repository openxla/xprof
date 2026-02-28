#include "xprof/convert/megascale_perfetto/trace_processor.h"

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/strings/match.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/strings/strip.h"
#include "re2/re2.h"
#include "xla/tsl/lib/gtl/map_util.h"
#include "xprof/convert/megascale_perfetto/xprof_trace.h"

namespace xprof::megascale {
namespace {
// -----------------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------------

// Returns the string value of an Arg key if it exists and is a StringId.
bool FindArgString(const Event& event, const XprofTrace& trace,
                   absl::string_view key_str, absl::string_view* out_string) {
  for (const auto& arg : event.args) {
    if (trace.string_table.Get(arg.key) == key_str) {
      if (std::holds_alternative<StringId>(arg.value)) {
        *out_string = trace.string_table.Get(std::get<StringId>(arg.value));
        return true;
      }
    }
  }
  return false;
}

// Returns the int value of an Arg key if it exists.
bool FindArgInt(const Event& event, const XprofTrace& trace,
                absl::string_view key_str, int64_t* out_value) {
  for (const auto& arg : event.args) {
    if (trace.string_table.Get(arg.key) == key_str) {
      if (std::holds_alternative<int64_t>(arg.value)) {
        *out_value = std::get<int64_t>(arg.value);
        return true;
      }
    }
  }
  return false;
}

// Returns the uint value of an Arg key if it exists.
bool FindArgUint(const Event& event, const XprofTrace& trace,
                 absl::string_view key_str, uint64_t* out_value) {
  for (const auto& arg : event.args) {
    if (trace.string_table.Get(arg.key) == key_str) {
      if (std::holds_alternative<uint64_t>(arg.value)) {
        *out_value = std::get<uint64_t>(arg.value);
        return true;
      }
    }
  }
  return false;
}

// Sorts events by timestamp (primary) and duration (secondary, descending).
void SortEventsInTrack(Track& track) {
  std::sort(track.events.begin(), track.events.end(),
            [](const Event& a, const Event& b) {
              if (a.timestamp_ps != b.timestamp_ps) {
                return a.timestamp_ps < b.timestamp_ps;
              }
              return a.duration_ps > b.duration_ps;
            });
}

absl::string_view ExtractRendezvousFromLongName(const Event& event,
                                                const XprofTrace& trace) {
  absl::string_view long_name;
  if (!FindArgString(event, trace, "long_name", &long_name)) {
    return "";
  }
  static constexpr LazyRE2 kRendezvousRe = {
      "_xla_host_transfer_rendezvous=\"([^\"]*)\""};
  absl::string_view rendezvous_name;
  if (RE2::PartialMatch(long_name, *kRendezvousRe, &rendezvous_name)) {
    return rendezvous_name;
  }
  return "";
}

absl::string_view ExtractRendezvousFromGraphKey(const Event& event,
                                                const XprofTrace& trace) {
  absl::string_view graph_key;
  if (!FindArgString(event, trace, "graph_key", &graph_key)) {
    return "";
  }
  static constexpr LazyRE2 kGraphKeyRendezvousRe = {
      R"(device_\d+_gid_([^$]+)\$.*)"};
  absl::string_view rendezvous_name;
  if (RE2::FullMatch(graph_key, *kGraphKeyRendezvousRe, &rendezvous_name)) {
    return rendezvous_name;
  }
  return "";
}

int64_t ExtractHloId(absl::string_view hlo_name) {
  static constexpr LazyRE2 kHloId = {R"re([a-zA-Z_-]+\.(\d+))re"};
  int64_t id;
  if (RE2::FullMatch(hlo_name, *kHloId, &id)) {
    return id;
  }
  return -1;
}

void RenameTrack(Track& track) {
  uint64_t device_id;
  absl::string_view name_part;
  if (RE2::FullMatch(track.name, "device_(\\d+)_gid_(.*)", &device_id,
                     &name_part)) {
    track.name = absl::StrCat(name_part, " (", device_id, ")");
  } else if (track.name == "Steps") {
    track.name = "1. Steps";
  } else if (track.name == "XLA Modules") {
    track.name = "2. XLA Modules";
  } else if (track.name == "XLA Ops") {
    track.name = "3. XLA Ops";
  } else if (track.name == "XLA TraceMe") {
    track.name = "4. XLA TraceMe";
  }
}

// -----------------------------------------------------------------------------
// Flow Queue System
// -----------------------------------------------------------------------------

// Manages FIFO queues of Flow IDs for different consumers.
// Keys are strings like "tpu_0_run_1_chan_5".
class FlowQueueMap {
 public:
  void Push(const std::string& key, int64_t flow_id) {
    map_[key].push_back(flow_id);
  }

  // Pops the next flow ID. Returns -1 if empty.
  int64_t Pop(absl::string_view key) {
    auto it = map_.find(key);
    if (it == map_.end() || it->second.empty()) {
      return -1;
    }
    int64_t id = it->second.front();
    it->second.erase(it->second.begin());
    return id;
  }

 private:
  absl::flat_hash_map<std::string, std::vector<int64_t>> map_;
};

}  // namespace

// -----------------------------------------------------------------------------
// TraceProcessor Implementation
// -----------------------------------------------------------------------------

void TraceProcessor::Process() {
  SortEvents();
  AssignRunIds();
  MarkLastDmaEvents();
  ResolveFlows();
  AddNetworkCounters();
  ModifyTrackNames();
}

void TraceProcessor::SortEvents() {
  for (auto& [tpu_id, tracks] : trace_.tpu_fragments) {
    for (auto& track : tracks) {
      SortEventsInTrack(track);
    }
  }
  for (auto& [tpu_id, tracks] : trace_.megascale_fragments) {
    for (auto& track : tracks) {
      SortEventsInTrack(track);
    }
  }
}

void TraceProcessor::AssignRunIds() {
  // Map: TPU ID -> List of {Timestamp, RunID}
  absl::flat_hash_map<int64_t, std::vector<std::pair<int64_t, int64_t>>>
      tpu_run_map;

  // 1. Build map from "XLA Modules"
  for (auto& [tpu_id, tracks] : trace_.tpu_fragments) {
    for (auto& track : tracks) {
      if (!absl::StrContains(track.name, "XLA Modules")) {
        continue;
      }
      auto& runs = tpu_run_map[tpu_id];
      int64_t run_id_counter = 1;  // For modules that don't have run_id.
      for (Event& event : track.events) {
        uint64_t run_id_temp;
        int64_t run_id;
        if (FindArgUint(event, trace_, "run_id", &run_id_temp)) {
          run_id = static_cast<int64_t>(run_id_temp);
        } else {
          run_id = run_id_counter++;
          event.args.push_back({trace_.string_table.Intern("run_id"), run_id});
        }
        event.run_id = run_id;
        runs.push_back({event.timestamp_ps, run_id});
      }
    }
  }

  // 2. Assign IDs
  auto assign_ids = [&](auto& fragments) {
    for (auto& [tpu_id, tracks] : fragments) {
      auto it = tpu_run_map.find(tpu_id);
      if (it == tpu_run_map.end() || it->second.empty()) {
        continue;
      }
      const auto& runs = it->second;

      for (auto& track : tracks) {
        if (absl::StrContains(track.name, "XLA Modules")) {
          continue;
        }
        for (auto& event : track.events) {
          auto run_it =
              std::upper_bound(runs.begin(), runs.end(),
                               std::make_pair(event.timestamp_ps, int64_t{0}),
                               [](const std::pair<int64_t, int64_t>& a,
                                  const std::pair<int64_t, int64_t>& b) {
                                 return a.first < b.first;
                               });

          if (run_it != runs.begin()) {
            event.run_id = std::prev(run_it)->second;
            event.args.push_back(
                {trace_.string_table.Intern("run_id"), event.run_id});
          }
        }
      }
    }
  };

  assign_ids(trace_.tpu_fragments);
  assign_ids(trace_.megascale_fragments);
}

void TraceProcessor::MarkLastDmaEvents() {
  static constexpr LazyRE2 kExecutionEventRe = {R"(device_\d+_gid_.*)"};
  for (auto& [tpu_id, tracks] : trace_.megascale_fragments) {
    for (auto& track : tracks) {
      std::vector<size_t> execution_event_indices;
      for (size_t i = 0; i < track.events.size(); ++i) {
        if (RE2::FullMatch(track.events[i].name, *kExecutionEventRe)) {
          execution_event_indices.push_back(i);
        }
      }

      for (size_t i : execution_event_indices) {
        Event& exec_event = track.events[i];
        int64_t execution_end_ps =
            exec_event.timestamp_ps + exec_event.duration_ps;
        Event* last_h2d_event = nullptr;
        Event* last_d2h_event = nullptr;

        for (size_t j = i + 1; j < track.events.size(); ++j) {
          if (track.events[j].timestamp_ps >= execution_end_ps) {
            break;
          }
          if (track.events[j].name == "HostToDevice END") {
            last_h2d_event = &track.events[j];
          }
          if (track.events[j].name == "DeviceToHost END") {
            last_d2h_event = &track.events[j];
          }
        }

        if (last_h2d_event != nullptr) {
          last_h2d_event->args.push_back(
              {trace_.string_table.Intern("is_last_instance"), int64_t{1}});
        }
        if (last_d2h_event != nullptr) {
          last_d2h_event->args.push_back(
              {trace_.string_table.Intern("is_last_instance"), int64_t{1}});
        }
      }
    }
  }
}

void TraceProcessor::ResolveFlows() {
  // map key: tpu_id, run_id, hlo_id
  absl::flat_hash_map<std::string, absl::string_view> send_hlo_to_rendezvous;
  absl::flat_hash_map<std::string, absl::string_view> recv_hlo_to_rendezvous;

  // Separate queues for different consumer types to ensure the right ID goes to
  // the right event type, even if counts mismatch slightly.
  FlowQueueMap q_send_to_d2h;
  FlowQueueMap q_d2h_to_send_done;
  FlowQueueMap q_h2d_to_recv_done;
  FlowQueueMap q_recv_to_recv_done;
  FlowQueueMap q_recv_to_h2d;
  FlowQueueMap q_send_to_send_done;
  FlowQueueMap q_send_to_recv_done;

  auto make_key = [](int64_t tpu, int64_t run, absl::string_view rendezvous) {
    return absl::StrCat(tpu, "_", run, "_", rendezvous);
  };
  auto make_hlo_key = [](int64_t tpu, int64_t run, int64_t hlo_id) {
    return absl::StrCat(tpu, "_", run, "_", hlo_id);
  };

  auto push_flow = [&](FlowQueueMap& queue, const std::string& key,
                       Event& producer) {
    int64_t fid = next_flow_id_++;
    producer.flows.push_back({fid, FlowDirection::kSource});
    producer.args.push_back({trace_.string_table.Intern("flow_out"), fid});
    queue.Push(key, fid);
  };

  auto pop_flow = [&](FlowQueueMap& queue, const std::string& key,
                      Event& consumer) {
    int64_t fid = queue.Pop(key);
    if (fid != -1) {
      consumer.args.push_back({trace_.string_table.Intern("flow_in"), fid});
      consumer.flows.push_back({fid, FlowDirection::kSink});
    }
    return fid;
  };

  // Helper to visit XLA ops in TPU fragments.
  auto visit_tpu_ops = [&](auto visitor) {
    for (auto& [tpu_id, tracks] : trace_.tpu_fragments) {
      for (auto& track : tracks) {
        if (!absl::StrContains(track.name, "XLA Ops")) {
          continue;
        }
        for (auto& event : track.events) {
          if (event.run_id == -1) {
            continue;
          }
          visitor(tpu_id, track, event);
        }
      }
    }
  };

  // Helper to visit all events in Megascale fragments.
  auto visit_megascale = [&](auto visitor) {
    for (auto& [tpu_id, tracks] : trace_.megascale_fragments) {
      for (auto& track : tracks) {
        for (auto& event : track.events) {
          if (event.run_id == -1) {
            continue;
          }
          visitor(tpu_id, event);
        }
      }
    }
  };

  // ---------------------------------------------------------------------------
  // Pass 1: Producers
  // ---------------------------------------------------------------------------
  // TPU Events: 'send' and 'recv'.
  visit_tpu_ops([&](int64_t tpu_id, Track& track, Event& event) {
    bool is_send;
    if (absl::StartsWith(event.name, "send.")) {
      is_send = true;
    } else if (absl::StartsWith(event.name, "recv.")) {
      is_send = false;
    } else {
      return;  // Skip other events.
    }

    int64_t hlo_id = ExtractHloId(event.name);
    if (hlo_id == -1) {
      return;
    }
    absl::string_view rendezvous = ExtractRendezvousFromLongName(event, trace_);
    if (rendezvous.empty()) {
      return;
    }
    std::string key = make_key(tpu_id, event.run_id, rendezvous);
    std::string hlo_key = make_hlo_key(tpu_id, event.run_id, hlo_id);

    if (is_send) {
      send_hlo_to_rendezvous[hlo_key] = rendezvous;
      push_flow(q_send_to_send_done, key, event);
      push_flow(q_send_to_recv_done, key, event);
      push_flow(q_send_to_d2h, key, event);
    } else {
      recv_hlo_to_rendezvous[hlo_key] = rendezvous;
      push_flow(q_recv_to_recv_done, key, event);
      push_flow(q_recv_to_h2d, key, event);
    }
  });

  // Megascale DMA events: D2H and H2D.
  // Note: We also process the consumers here since we have all necessary info.
  visit_megascale([&](int64_t tpu_id, Event& event) {
    bool is_d2h;
    bool is_start;
    if (event.name == "DeviceToHost START") {
      is_d2h = true;
      is_start = true;
    } else if (event.name == "DeviceToHost END") {
      is_d2h = true;
      is_start = false;
    } else if (event.name == "HostToDevice START") {
      is_d2h = false;
      is_start = true;
    } else if (event.name == "HostToDevice END") {
      is_d2h = false;
      is_start = false;
    } else {
      return;  // Skip this event.
    }

    absl::string_view rendezvous = ExtractRendezvousFromGraphKey(event, trace_);
    if (rendezvous.empty()) {
      return;
    }
    std::string key = make_key(tpu_id, event.run_id, rendezvous);

    if (is_start) {
      // We're only interested in the first START event.
      if (int64_t action_idx = -1;
          !FindArgInt(event, trace_, "action_index", &action_idx) ||
          action_idx != 0) {
        return;
      }
      if (is_d2h) {
        pop_flow(q_send_to_d2h, key, event);
      } else {
        pop_flow(q_recv_to_h2d, key, event);
      }
    } else {  // END event
      // We're only interested in the last END event.
      if (int64_t is_last = 0;
          !FindArgInt(event, trace_, "is_last_instance", &is_last) ||
          is_last != 1) {
        return;
      }
      if (!is_d2h) {
        // Reduce last H2D END duration slightly. This is needed because
        // Perfetto will show the flow going out of the parent slice (megascale
        // graph event) if this is the last event in the action graph and its
        // end time matches the end time of the parent slice.
        event.duration_ps = std::max(event.duration_ps - 1000, int64_t{0});
      }
      if (is_d2h) {
        push_flow(q_d2h_to_send_done, key, event);
      } else {
        push_flow(q_h2d_to_recv_done, key, event);
      }
    }
  });

  // ---------------------------------------------------------------------------
  // Pass 2: Consumers (send-done, recv-done)
  // ---------------------------------------------------------------------------
  visit_tpu_ops([&](int64_t tpu_id, Track& track, Event& event) {
    bool is_send_done;
    if (absl::StartsWith(event.name, "send-done.")) {
      is_send_done = true;
    } else if (absl::StartsWith(event.name, "recv-done.")) {
      is_send_done = false;
    } else {
      return;  // Skip other events.
    }

    int64_t hlo_id = ExtractHloId(event.name);
    if (hlo_id == -1) {
      return;
    }
    std::string hlo_key = make_hlo_key(tpu_id, event.run_id, hlo_id);
    absl::string_view rendezvous =
        is_send_done
            ? tsl::gtl::FindWithDefault(send_hlo_to_rendezvous, hlo_key, "")
            : tsl::gtl::FindWithDefault(recv_hlo_to_rendezvous, hlo_key, "");
    if (rendezvous.empty()) return;
    std::string key = make_key(tpu_id, event.run_id, rendezvous);

    if (is_send_done) {
      pop_flow(q_send_to_send_done, key, event);
      pop_flow(q_d2h_to_send_done, key, event);
    } else {
      pop_flow(q_send_to_recv_done, key, event);
      pop_flow(q_recv_to_recv_done, key, event);
      // Instead of attaching the flows to the recv-done event, let's create an
      // instant "recv-done END" event that begins right after the recv-done
      // finishes and attach the flows to it. We do this because the H2D events
      // may end after the recv-done has started and Perfetto does not handle
      // that case well. (end time of producer slice is later than start time of
      // consumer slice)

      Event instant_event;
      instant_event.name = absl::StrCat(event.name, " END");
      instant_event.timestamp_ps =
          event.timestamp_ps + event.duration_ps + 1000;
      instant_event.duration_ps = 0;
      instant_event.run_id = event.run_id;
      instant_event.args.push_back(
          {trace_.string_table.Intern("run_id"), event.run_id});
      // Add a flow from the recv-done to the recv-done END.
      int64_t fid_internal = next_flow_id_++;
      event.args.push_back(
          {trace_.string_table.Intern("flow_out"), fid_internal});
      event.flows.push_back({fid_internal, FlowDirection::kSource});
      instant_event.args.push_back(
          {trace_.string_table.Intern("flow_in"), fid_internal});
      instant_event.flows.push_back({fid_internal, FlowDirection::kSink});

      pop_flow(q_h2d_to_recv_done, key, instant_event);

      track.events.push_back(std::move(instant_event));
    }
  });
}

void TraceProcessor::AddNetworkCounters() {
  // Filter out unreasonable spikes which likely come from bad duration data.
  // We don't expect to see a very large jump in bandwidth usage between two
  // consecutive data points.
  constexpr double kMaxRateDeltaGbps = 200.0;
  // Pair contents: <timestamp_ps, value_delta>
  std::vector<std::pair<int64_t, int64_t>> rx_deltas;
  std::vector<std::pair<int64_t, int64_t>> tx_deltas;
  std::vector<std::pair<int64_t, double>> rx_bw_deltas;
  std::vector<std::pair<int64_t, double>> tx_bw_deltas;

  for (auto& [tpu_id, tracks] : trace_.megascale_fragments) {
    for (const auto& track : tracks) {
      for (const auto& event : track.events) {
        if (event.name == "NetworkReceive END") {
          int64_t latency_us = 0;
          uint64_t latency_uint = 0;
          if (FindArgUint(event, trace_, "network_transport_latency_us",
                          &latency_uint)) {
            latency_us = static_cast<int64_t>(latency_uint);
          } else {
            continue;
          }

          absl::string_view buffer_sizes;
          if (!FindArgString(event, trace_, "buffer_sizes", &buffer_sizes)) {
            LOG_EVERY_N(WARNING, 1000)
                << "NetworkReceive END event missing buffer_sizes";
            continue;
          }

          static constexpr LazyRE2 kBufferSizeRe = {R"(\$c\d+=(\d+))"};
          absl::string_view input(buffer_sizes);
          int64_t bytes;
          int chunks = 0;
          while (RE2::FindAndConsume(&input, *kBufferSizeRe, &bytes)) {
            chunks++;
            int64_t end_time_ps = event.timestamp_ps + event.duration_ps;
            int64_t start_time_ps = end_time_ps - (latency_us * 1000000);

            rx_deltas.push_back({start_time_ps, bytes});
            rx_deltas.push_back({end_time_ps, -bytes});

            if (latency_us > 0) {
              double rate_gbps =
                  (static_cast<double>(bytes) * 8.0) / (latency_us * 1000.0);
              if (rate_gbps <= kMaxRateDeltaGbps) {
                rx_bw_deltas.push_back({start_time_ps, rate_gbps});
                rx_bw_deltas.push_back({end_time_ps, -rate_gbps});
              } else {
                LOG_EVERY_POW_2(WARNING)
                    << "NetworkReceive END event has unreasonable bandwidth: "
                    << rate_gbps << " Gbps, bytes: " << bytes
                    << ", network_transport_latency_us: " << latency_us;
              }
            }
          }
          if (chunks == 0) {
            LOG_EVERY_N(WARNING, 1000)
                << "NetworkReceive END failed to parse buffer_sizes: "
                << buffer_sizes;
          }
        } else if (event.name == "NetworkSend END") {
          int64_t duration_ns = 0;
          // We're using action_duration_ns here because we don't have access to
          // network_transport_latency_us for NetworkSend events. If/when that
          // becomes available then we should update this code.
          if (!FindArgInt(event, trace_, "action_duration_ns", &duration_ns)) {
            LOG_EVERY_N(WARNING, 1000)
                << "NetworkSend END missing action_duration_ns";
            continue;
          }

          absl::string_view buffer_sizes;
          if (!FindArgString(event, trace_, "buffer_sizes", &buffer_sizes)) {
            LOG_EVERY_N(WARNING, 1000)
                << "NetworkSend END missing buffer_sizes";
            continue;
          }

          static constexpr LazyRE2 kBufferSizeRe = {
              R"(s(\w+)->(\w+)d.*?\$c\d+=(\d+))"};
          absl::string_view input(buffer_sizes);
          std::string src, dst;
          int64_t bytes;
          int chunks = 0;
          while (
              RE2::FindAndConsume(&input, *kBufferSizeRe, &src, &dst, &bytes)) {
            // Ignore loopback transfers.
            if (src == dst) {
              continue;
            }
            chunks++;
            int64_t end_time_ps = event.timestamp_ps + event.duration_ps;
            int64_t start_time_ps = end_time_ps - (duration_ns * 1000);
            tx_deltas.push_back({start_time_ps, bytes});
            tx_deltas.push_back({end_time_ps, -bytes});

            if (duration_ns > 0) {
              double rate_gbps = (static_cast<double>(bytes) * 8.0) /
                                 static_cast<double>(duration_ns);
              if (rate_gbps <= kMaxRateDeltaGbps) {
                tx_bw_deltas.push_back({start_time_ps, rate_gbps});
                tx_bw_deltas.push_back({end_time_ps, -rate_gbps});
              } else {
                LOG_EVERY_POW_2(WARNING)
                    << "NetworkSend END event has unreasonable bandwidth: "
                    << rate_gbps << " Gbps, bytes: " << bytes
                    << ", duration_ns: " << duration_ns;
              }
            }
          }
        }
      }
    }
  }

  // Sort `deltas` and group them by timestamp such that we have one data point
  // per timestamp then write them to `track`. Set the track's name to `name`.
  auto process_deltas = [&](auto& deltas, auto& track, absl::string_view name) {
    if (deltas.empty()) {
      return;
    }
    std::sort(deltas.begin(), deltas.end());
    track.name = name;
    using ValueType = typename std::remove_reference<
        decltype(deltas)>::type::value_type::second_type;
    ValueType current_value = 0;
    for (size_t i = 0; i < deltas.size(); ++i) {
      current_value += deltas[i].second;
      if (i + 1 < deltas.size() && deltas[i + 1].first == deltas[i].first) {
        continue;
      }
      track.timestamps_ps.push_back(deltas[i].first);
      track.values.push_back(current_value);
    }
  };

  process_deltas(rx_deltas, trace_.rx_counter, "Outstanding Bytes RX");
  process_deltas(tx_deltas, trace_.tx_counter, "Outstanding Bytes TX");
  process_deltas(rx_bw_deltas, trace_.rx_bw_counter, "Bandwidth RX (Gbps)");
  process_deltas(tx_bw_deltas, trace_.tx_bw_counter, "Bandwidth TX (Gbps)");
}

void TraceProcessor::ModifyTrackNames() {
  for (auto& [tpu_id, tracks] : trace_.tpu_fragments) {
    for (auto& track : tracks) {
      RenameTrack(track);
    }
  }
  for (auto& [tpu_id, tracks] : trace_.megascale_fragments) {
    for (auto& track : tracks) {
      RenameTrack(track);
    }
  }
}

}  // namespace xprof::megascale
