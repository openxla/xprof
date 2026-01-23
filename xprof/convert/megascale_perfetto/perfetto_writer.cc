#include "xprof/convert/megascale_perfetto/perfetto_writer.h"

#include <cstddef>
#include <cstdint>
#include <string>

#include "absl/container/btree_set.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "protos/perfetto/trace/interned_data/interned_data.pb.h"
#include "protos/perfetto/trace/profiling/profile_common.pb.h"
#include "protos/perfetto/trace/trace.pb.h"
#include "protos/perfetto/trace/trace_packet.pb.h"
#include "protos/perfetto/trace/track_event/counter_descriptor.pb.h"
#include "protos/perfetto/trace/track_event/debug_annotation.pb.h"
#include "protos/perfetto/trace/track_event/track_descriptor.pb.h"
#include "protos/perfetto/trace/track_event/track_event.pb.h"
#ifndef PLATFORM_WINDOWS
#include "google/protobuf/io/gzip_stream.h"
#endif
#include "google/protobuf/io/zero_copy_stream_impl_lite.h"
#include "xprof/convert/megascale_perfetto/xprof_trace.h"

namespace xprof::megascale {

namespace {
using ::perfetto::protos::DebugAnnotation;
using ::perfetto::protos::DebugAnnotationName;
using ::perfetto::protos::EventName;
using ::perfetto::protos::InternedString;
using ::perfetto::protos::Trace;
using ::perfetto::protos::TracePacket;
using ::perfetto::protos::TrackDescriptor;
using ::perfetto::protos::TrackEvent;

// Use a single sequence ID to allow interning across the whole file.
constexpr uint32_t kTrustedPacketSequenceId = 1;

class WriterContext {
 public:
  WriterContext(const XprofTrace& ir, Trace* trace) : ir_(ir), trace_(trace) {}

  void Write() {
    WriteCounters();

    absl::btree_set<int64_t> device_ids_set;
    for (const auto& [device_id, tracks] : ir_.tpu_fragments) {
      device_ids_set.insert(device_id);
    }
    for (const auto& [device_id, tracks] : ir_.megascale_fragments) {
      device_ids_set.insert(device_id);
    }

    if (device_ids_set.empty()) {
      return;
    }

    uint64_t tpus_uuid =
        WriteTrackDescriptor(next_track_uuid_++, "2. TPUs", /*parent_uuid=*/0);

    for (int64_t device_id : device_ids_set) {
      WriteDevice(device_id, tpus_uuid);
    }
  }

 private:
  void WriteDevice(int64_t device_id, uint64_t parent_uuid) {
    uint64_t uuid = next_track_uuid_++;
    WriteTrackDescriptor(uuid, absl::StrCat("/device:TPU:", device_id),
                         parent_uuid);
    if (ir_.tpu_fragments.contains(device_id)) {
      for (const auto& track : ir_.tpu_fragments.at(device_id)) {
        WriteTrack(track, uuid);
      }
    }
    if (ir_.megascale_fragments.contains(device_id)) {
      uint64_t megascale_parent_uuid =
          WriteTrackDescriptor(next_track_uuid_++, "Megascale", uuid);
      for (const auto& track : ir_.megascale_fragments.at(device_id)) {
        WriteTrack(track, megascale_parent_uuid);
      }
    }
  }

  void WriteCounters() {
    if (ir_.rx_counter.values.empty() && ir_.tx_counter.values.empty() &&
        ir_.rx_bw_counter.values.empty() && ir_.tx_bw_counter.values.empty()) {
      return;
    }
    uint64_t parent_uuid =
        WriteTrackDescriptor(next_track_uuid_++, "1. Network",
                             /*parent_uuid=*/0);
    if (!ir_.rx_counter.values.empty()) {
      WriteCounterTrack(ir_.rx_counter, parent_uuid, "bytes",
                        "outstanding_bytes");
    }
    if (!ir_.tx_counter.values.empty()) {
      WriteCounterTrack(ir_.tx_counter, parent_uuid, "bytes",
                        "outstanding_bytes");
    }
    if (!ir_.rx_bw_counter.values.empty()) {
      WriteCounterTrack(ir_.rx_bw_counter, parent_uuid, "Gbps", "bandwidth");
    }
    if (!ir_.tx_bw_counter.values.empty()) {
      WriteCounterTrack(ir_.tx_bw_counter, parent_uuid, "Gbps", "bandwidth");
    }
  }

  void WriteTrack(const Track& track, uint64_t parent_uuid) {
    uint64_t uuid = next_track_uuid_++;
    WriteTrackDescriptor(uuid, track.name, parent_uuid);
    for (const auto& event : track.events) {
      WriteEvent(event, uuid);
    }
  }

  template <typename T>
  void WriteCounterTrack(const CounterTrack<T>& track, uint64_t parent_uuid,
                         absl::string_view unit = "bytes",
                         absl::string_view y_axis_share_key = "") {
    uint64_t uuid = next_track_uuid_++;
    WriteTrackDescriptor(uuid, track.name, parent_uuid, /*is_counter=*/true,
                         unit, y_axis_share_key);
    for (size_t i = 0; i < track.timestamps_ps.size(); ++i) {
      WriteCounterEvent(track.timestamps_ps[i], track.values[i], uuid);
    }
  }

  uint64_t WriteTrackDescriptor(uint64_t uuid, absl::string_view name,
                                uint64_t parent_uuid, bool is_counter = false,
                                absl::string_view counter_unit = "bytes",
                                absl::string_view y_axis_share_key = "") {
    TracePacket* packet = NewPacket();
    TrackDescriptor* descriptor = packet->mutable_track_descriptor();
    descriptor->set_uuid(uuid);
    descriptor->set_name(name);
    if (parent_uuid != 0) {
      descriptor->set_parent_uuid(parent_uuid);
    }
    if (is_counter) {
      auto* counter_descriptor = descriptor->mutable_counter();
      counter_descriptor->set_unit_name(counter_unit);
      if (!y_axis_share_key.empty()) {
        counter_descriptor->set_y_axis_share_key(y_axis_share_key);
      }
    }
    return uuid;
  }

  template <typename T>
  void WriteCounterEvent(int64_t timestamp_ps, T value, uint64_t track_uuid) {
    TracePacket* packet = NewPacket();
    packet->set_timestamp(timestamp_ps / 1000);
    if (packet->trusted_packet_sequence_id() == 0) {
      packet->set_trusted_packet_sequence_id(kTrustedPacketSequenceId);
    }
    TrackEvent* te = packet->mutable_track_event();
    te->set_track_uuid(track_uuid);
    te->set_type(TrackEvent::TYPE_COUNTER);
    if constexpr (std::is_same_v<T, int64_t>) {
      te->set_counter_value(value);
    } else if constexpr (std::is_same_v<T, double>) {
      te->set_double_counter_value(value);
    }
  }

  void WriteEvent(const Event& event, uint64_t track_uuid) {
    bool is_instant = (event.duration_ps == 0);

    // 1. Begin (or Instant) Packet
    TracePacket* packet = NewPacket();
    packet->set_timestamp(event.timestamp_ps / 1000);
    TrackEvent* te = packet->mutable_track_event();
    te->set_track_uuid(track_uuid);
    te->set_type(is_instant ? TrackEvent::TYPE_INSTANT
                            : TrackEvent::TYPE_SLICE_BEGIN);

    // Name: Intern the event name on the fly (it is not interned in IR)
    te->set_name_iid(GetOrInternEventName(packet, event.name));

    // Args: Keys and StringValues are interned in IR, map them to Perfetto
    // IIDs
    auto set_annotation_value = [&](const ArgValue& value,
                                    DebugAnnotation* dbg) {
      if (std::holds_alternative<int64_t>(value)) {
        dbg->set_int_value(std::get<int64_t>(value));
      } else if (std::holds_alternative<uint64_t>(value)) {
        dbg->set_uint_value(std::get<uint64_t>(value));
      } else if (std::holds_alternative<double>(value)) {
        dbg->set_double_value(std::get<double>(value));
      } else if (std::holds_alternative<StringId>(value)) {
        StringId val_id = std::get<StringId>(value);
        dbg->set_string_value_iid(GetOrInternAnnotationValue(packet, val_id));
      }
    };

    absl::flat_hash_map<StringId, std::vector<const ArgValue*>> grouped_args;
    for (const auto& arg : event.args) {
      grouped_args[arg.key].push_back(&arg.value);
    }

    for (const auto& [key, values] : grouped_args) {
      DebugAnnotation* dbg = te->add_debug_annotations();
      dbg->set_name_iid(GetOrInternAnnotationName(packet, key));
      // If a key repeats, treat its values as an array.
      if (values.size() == 1) {
        set_annotation_value(*values[0], dbg);
      } else {
        for (const ArgValue* val : values) {
          DebugAnnotation* array_entry = dbg->add_array_values();
          set_annotation_value(*val, array_entry);
        }
      }
    }

    // Flows:
    for (const auto& flow : event.flows) {
      if (flow.direction == FlowDirection::kSink) {
        if (flow.is_terminating) {
          te->add_terminating_flow_ids(flow.id);
        } else {
          te->add_flow_ids(flow.id);
        }
      } else if (is_instant && flow.direction == FlowDirection::kSource) {
        te->add_flow_ids(flow.id);
      }
    }

    // 2. End Packet (if not Instant)
    if (is_instant) {
      return;
    }

    packet = NewPacket();
    packet->set_timestamp((event.timestamp_ps + event.duration_ps) / 1000);
    te = packet->mutable_track_event();
    te->set_track_uuid(track_uuid);
    te->set_type(TrackEvent::TYPE_SLICE_END);

    // Flows: Attach kSource flows to the End event
    for (const auto& flow : event.flows) {
      if (flow.direction == FlowDirection::kSource) {
        te->add_flow_ids(flow.id);
      }
    }
  }

  TracePacket* NewPacket() {
    auto* p = trace_->add_packet();
    p->set_trusted_packet_sequence_id(kTrustedPacketSequenceId);
    if (first_packet_) {
      p->set_sequence_flags(TracePacket::SEQ_INCREMENTAL_STATE_CLEARED |
                            TracePacket::SEQ_NEEDS_INCREMENTAL_STATE);
      first_packet_ = false;
    } else {
      p->set_sequence_flags(TracePacket::SEQ_NEEDS_INCREMENTAL_STATE);
    }
    return p;
  }

  // --- Interning Logic ---

  uint64_t GetOrInternEventName(TracePacket* packet, absl::string_view name) {
    auto it = event_name_to_iid_.find(name);
    if (it != event_name_to_iid_.end()) {
      return it->second;
    }

    uint64_t iid = next_iid_++;
    event_name_to_iid_[name] = iid;
    EventName* entry = packet->mutable_interned_data()->add_event_names();
    entry->set_iid(iid);
    entry->set_name(name);
    return iid;
  }

  uint64_t GetOrInternAnnotationName(TracePacket* packet, StringId key_id) {
    // Check cache using IR StringId
    auto it = annotation_name_to_iid_.find(key_id);
    if (it != annotation_name_to_iid_.end()) {
      return it->second;
    }

    // Not found, emit InternedData
    uint64_t iid = next_iid_++;
    annotation_name_to_iid_[key_id] = iid;

    absl::string_view str = ir_.string_table.Get(key_id);
    DebugAnnotationName* entry =
        packet->mutable_interned_data()->add_debug_annotation_names();
    entry->set_iid(iid);
    entry->set_name(str);
    return iid;
  }

  uint64_t GetOrInternAnnotationValue(TracePacket* packet, StringId val_id) {
    auto it = annotation_value_to_iid_.find(val_id);
    if (it != annotation_value_to_iid_.end()) {
      return it->second;
    }

    uint64_t iid = next_iid_++;
    annotation_value_to_iid_[val_id] = iid;

    absl::string_view str = ir_.string_table.Get(val_id);
    InternedString* entry =
        packet->mutable_interned_data()->add_debug_annotation_string_values();
    entry->set_iid(iid);
    entry->set_str(str);
    return iid;
  }

  const XprofTrace& ir_;
  Trace* trace_;

  uint64_t next_track_uuid_ = 1;
  uint64_t next_iid_ = 1;
  bool first_packet_ = true;

  // Maps to cache Perfetto IIDs.
  // We use StringId as key for args since they are already unique in IR.
  // We use std::string as key for event names since they are raw in IR Event.
  absl::flat_hash_map<std::string, uint64_t> event_name_to_iid_;
  absl::flat_hash_map<StringId, uint64_t> annotation_name_to_iid_;
  absl::flat_hash_map<StringId, uint64_t> annotation_value_to_iid_;
};

void Write(const XprofTrace& trace, Trace* out_proto) {
  WriterContext ctx(trace, out_proto);
  ctx.Write();
}

}  // namespace

absl::Status PerfettoWriter::WriteToCord(const XprofTrace& trace,
                                         absl::Cord* output,
                                         bool compressed_output) {
  perfetto::protos::Trace trace_proto;
  Write(trace, &trace_proto);
  if (compressed_output) {
#ifdef PLATFORM_WINDOWS
    LOG(WARNING) << "Compression is not supported on Windows.";
#else
    google::protobuf::io::CordOutputStream stream;
    google::protobuf::io::GzipOutputStream gzip_stream(&stream);
    if (!trace_proto.SerializeToZeroCopyStream(&gzip_stream) ||
        !gzip_stream.Close()) {
      return absl::InternalError("Failed to serialize to gzip stream");
    }
    output->Append(stream.Consume());
    return absl::OkStatus();
#endif
  }
  if (!trace_proto.AppendToString(output)) {
    return absl::InternalError("Failed to serialize to cord");
  }
  return absl::OkStatus();
}

absl::StatusOr<std::string> PerfettoWriter::WriteToString(
    const XprofTrace& trace, bool compressed_output) {
  perfetto::protos::Trace trace_proto;
  Write(trace, &trace_proto);
  if (compressed_output) {
#ifdef PLATFORM_WINDOWS
    LOG(WARNING) << "Compression is not supported on Windows.";
#else
    std::string output;
    google::protobuf::io::StringOutputStream stream(&output);
    google::protobuf::io::GzipOutputStream gzip_stream(&stream);
    if (!trace_proto.SerializeToZeroCopyStream(&gzip_stream) ||
        !gzip_stream.Close()) {
      return absl::InternalError("Failed to serialize to gzip stream");
    }
    return output;
#endif
  }
  return trace_proto.SerializeAsString();
}

}  // namespace xprof::megascale
