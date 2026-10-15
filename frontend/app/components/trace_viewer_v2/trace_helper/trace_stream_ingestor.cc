#include "frontend/app/components/trace_viewer_v2/trace_helper/trace_stream_ingestor.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "google/protobuf/arena.h"
#include "xprof/convert/trace_viewer/delta_series/zstd_compression.h"
#include "frontend/app/components/trace_viewer_v2/timeline/data_provider.h"
#include "frontend/app/components/trace_viewer_v2/timeline/timeline.h"
#include "plugin/xprof/protobuf/trace_data_response.pb.h"

namespace traceviewer {
namespace {

constexpr uint32_t kMaxCompressedFrameBytes = 32U * 1024U * 1024U;

uint32_t ReadUint32LE(const char* ptr) {
  return static_cast<uint32_t>(static_cast<uint8_t>(ptr[0])) |
         (static_cast<uint32_t>(static_cast<uint8_t>(ptr[1])) << 8) |
         (static_cast<uint32_t>(static_cast<uint8_t>(ptr[2])) << 16) |
         (static_cast<uint32_t>(static_cast<uint8_t>(ptr[3])) << 24);
}

}  // namespace

void StreamingTraceIngestor::Begin(uint64_t stream_id) {
  Abort();
  active_ = true;
  stream_id_ = stream_id;
  status_ = absl::OkStatus();
}

void StreamingTraceIngestor::Abort() {
  active_ = false;
  stream_id_ = 0;
  seen_header_ = false;
  seen_trailer_ = false;
  total_received_events_ = 0;
  status_ = absl::OkStatus();
  std::string().swap(frame_buffer_);
  frame_read_offset_ = 0;
  message_arena_.Reset();
  header_response_.Clear();
  std::vector<TrackState>().swap(tracks_);
  std::vector<std::string>().swap(complete_wire_payloads_);
}

absl::Status StreamingTraceIngestor::PushBytes(
    absl::Span<const uint8_t> bytes) {
  if (!active_) {
    return absl::FailedPreconditionError("Trace stream is not active");
  }
  if (!status_.ok()) {
    return status_;
  }
  if (bytes.empty()) {
    return absl::OkStatus();
  }

  frame_buffer_.append(reinterpret_cast<const char*>(bytes.data()),
                       bytes.size());

  while (frame_buffer_.size() - frame_read_offset_ >= 4) {
    const uint32_t frame_len =
        ReadUint32LE(frame_buffer_.data() + frame_read_offset_);
    if (frame_len == 0 || frame_len > kMaxCompressedFrameBytes) {
      status_ = absl::DataLossError(
          absl::StrCat("Invalid compressed frame length: ", frame_len));
      return status_;
    }
    if (frame_buffer_.size() - frame_read_offset_ < 4 + frame_len) {
      break;
    }

    const absl::string_view compressed_frame(
        frame_buffer_.data() + frame_read_offset_ + 4, frame_len);
    frame_read_offset_ += 4 + frame_len;

    status_ = ProcessCompressedFrame(compressed_frame);
    if (!status_.ok()) {
      return status_;
    }
  }

  if (frame_read_offset_ == frame_buffer_.size()) {
    frame_buffer_.clear();
    frame_read_offset_ = 0;
  } else if (frame_read_offset_ >= 65536 &&
             frame_read_offset_ * 2 >= frame_buffer_.size()) {
    frame_buffer_.erase(0, frame_read_offset_);
    frame_read_offset_ = 0;
  }

  return absl::OkStatus();
}

absl::Status StreamingTraceIngestor::ProcessCompressedFrame(
    absl::string_view compressed_frame) {
  absl::StatusOr<std::string> decompressed =
      tensorflow::profiler::ZstdCompression::Decompress(compressed_frame);
  if (!decompressed.ok()) {
    return decompressed.status();
  }

  xprof::TraceStreamMessage* msg =
      google::protobuf::Arena::Create<xprof::TraceStreamMessage>(&message_arena_);
  if (!msg->ParseFromArray(decompressed->data(),
                           static_cast<int>(decompressed->size()))) {
    message_arena_.Reset();
    return absl::DataLossError("Failed to parse TraceStreamMessage");
  }
  std::string().swap(*decompressed);

  absl::Status result = absl::OkStatus();
  switch (msg->payload_case()) {
    case xprof::TraceStreamMessage::kHeader:
      result = OnHeader(msg->header());
      break;
    case xprof::TraceStreamMessage::kChunk:
      result = OnChunk(msg->chunk());
      break;
    case xprof::TraceStreamMessage::kTrailer:
      result = OnTrailer(msg->trailer());
      break;
    case xprof::TraceStreamMessage::PAYLOAD_NOT_SET:
      result = absl::DataLossError("TraceStreamMessage payload is not set");
      break;
  }
  message_arena_.Reset();
  return result;
}

absl::Status StreamingTraceIngestor::OnHeader(
    const xprof::TraceStreamHeader& header) {
  if (seen_header_ || seen_trailer_) {
    return absl::DataLossError("Unexpected TraceStreamHeader frame");
  }
  seen_header_ = true;

  header_response_.Clear();
  if (header.has_metadata()) {
    *header_response_.mutable_metadata() = header.metadata();
  }
  for (const xprof::DetailItem& detail : header.details()) {
    *header_response_.add_details() = detail;
  }
  if (header.has_full_timespan_start_ps()) {
    header_response_.set_full_timespan_start_ps(
        header.full_timespan_start_ps());
  }
  if (header.has_full_timespan_end_ps()) {
    header_response_.set_full_timespan_end_ps(header.full_timespan_end_ps());
  }
  header_response_.mutable_interned_strings()->Reserve(
      header.interned_strings_size());
  for (const std::string& s : header.interned_strings()) {
    header_response_.add_interned_strings(s);
  }

  tracks_.clear();
  tracks_.reserve(header.tracks_size());
  for (int i = 0; i < header.tracks_size(); ++i) {
    const xprof::TrackManifest& tm = header.tracks(i);
    if (tm.track_id() != static_cast<uint32_t>(i)) {
      return absl::DataLossError(
          absl::StrCat("TrackManifest track_id mismatch: expected ", i,
                       ", got ", tm.track_id()));
    }
    TrackState state{
        .kind = tm.kind(),
        .process_id = tm.process_id(),
        .thread_id = tm.thread_id(),
        .name_ref = tm.name_ref(),
        .expected_event_count = tm.event_count(),
        .ordered = tm.ordered(),
    };
    if (tm.kind() == xprof::TrackManifest::TRACK_KIND_ASYNC &&
        tm.event_count() == 0) {
      xprof::TraceEventSeries* empty_async =
          header_response_.add_async_events();
      empty_async->mutable_metadata()->set_process_id(tm.process_id());
      empty_async->mutable_metadata()->set_thread_id(tm.thread_id());
      empty_async->mutable_metadata()->set_name_ref(tm.name_ref());
    } else if (tm.kind() == xprof::TrackManifest::TRACK_KIND_COUNTER &&
               tm.event_count() == 0) {
      xprof::TraceEventSeries* empty_cnt =
          header_response_.add_counter_events();
      empty_cnt->mutable_metadata()->set_process_id(tm.process_id());
      empty_cnt->mutable_metadata()->set_thread_id(tm.thread_id());
      empty_cnt->mutable_metadata()->set_name_ref(tm.name_ref());
    }
    tracks_.push_back(state);
  }

  return absl::OkStatus();
}

absl::Status StreamingTraceIngestor::OnChunk(
    const xprof::TraceEventChunk& chunk) {
  if (!seen_header_ || seen_trailer_) {
    return absl::DataLossError("Unexpected TraceEventChunk frame");
  }

  if (chunk.appended_interned_strings_size() > 0) {
    const uint32_t current_string_count =
        static_cast<uint32_t>(header_response_.interned_strings_size());
    if (chunk.string_table_base_index() != current_string_count) {
      return absl::DataLossError(absl::StrCat(
          "TraceEventChunk string_table_base_index mismatch: expected ",
          current_string_count, ", got ", chunk.string_table_base_index()));
    }
    for (const std::string& s : chunk.appended_interned_strings()) {
      header_response_.add_interned_strings(s);
    }
  }

  for (const xprof::TrackSegment& seg : chunk.segments()) {
    const uint32_t track_id = seg.track_id();
    if (track_id >= tracks_.size()) {
      return absl::DataLossError(absl::StrCat(
          "TrackSegment references out-of-range track_id ", track_id));
    }
    TrackState& track = tracks_[track_id];
    if (track.sealed) {
      return absl::DataLossError(absl::StrCat(
          "TrackSegment received for already-sealed track ", track_id));
    }
    if (seg.first_event_ordinal() != track.received_events) {
      return absl::DataLossError(
          absl::StrCat("TrackSegment first_event_ordinal mismatch on track ",
                       track_id, ": expected ", track.received_events, ", got ",
                       seg.first_event_ordinal()));
    }

    const xprof::TraceEventSeries& series = seg.series();
    const int n = series.deltas_size();
    if (track.received_events + static_cast<uint64_t>(n) >
        track.expected_event_count) {
      return absl::DataLossError(
          absl::StrCat("Track ", track_id, " exceeded manifest event_count (",
                       track.expected_event_count, ")"));
    }

    const uint64_t prev_end_ts_ps = track.last_ts_ps;
    uint64_t cur_ts_ps = 0;
    for (int i = 0; i < n; ++i) {
      const uint64_t delta_ps = series.deltas(i);
      if (track.kind == xprof::TrackManifest::TRACK_KIND_COMPLETE) {
        const int64_t signed_delta = static_cast<int64_t>(delta_ps);
        if (signed_delta < 0) {
          const uint64_t abs_d = static_cast<uint64_t>(-(signed_delta + 1)) + 1;
          cur_ts_ps = (cur_ts_ps >= abs_d) ? (cur_ts_ps - abs_d) : 0;
        } else {
          cur_ts_ps += delta_ps;
        }
      } else {
        cur_ts_ps += delta_ps;
      }

      const uint64_t dur_ps =
          (track.kind == xprof::TrackManifest::TRACK_KIND_COUNTER)
              ? 0
              : (i < series.durations_size() ? series.durations(i) : 0);
      const uint32_t name_ref =
          (track.kind == xprof::TrackManifest::TRACK_KIND_COMPLETE)
              ? (i < series.name_refs_size() ? series.name_refs(i)
                                             : series.metadata().name_ref())
              : track.name_ref;

      if (track.ordered && track.has_last_event) {
        if (cur_ts_ps < track.last_ts_ps ||
            (track.kind != xprof::TrackManifest::TRACK_KIND_COUNTER &&
             cur_ts_ps == track.last_ts_ps && dur_ps > track.last_dur_ps)) {
          return absl::DataLossError(absl::StrCat(
              "Ordered track ", track_id, " violated monotonicity at ordinal ",
              track.received_events + static_cast<uint64_t>(i)));
        }
      }
      track.has_last_event = true;
      track.last_ts_ps = cur_ts_ps;
      track.last_dur_ps = dur_ps;
      track.checksum = tensorflow::profiler::UpdateTrackEventChecksum(
          track.checksum, cur_ts_ps, dur_ps, name_ref);
    }

    if (n > 0) {
      if (track.kind == xprof::TrackManifest::TRACK_KIND_COMPLETE) {
        std::string wire_slice;
        if (!series.SerializeToString(&wire_slice)) {
          return absl::InternalError(
              "Failed to serialize complete track segment");
        }
        complete_wire_payloads_.push_back(std::move(wire_slice));
      } else if (track.kind == xprof::TrackManifest::TRACK_KIND_ASYNC) {
        *header_response_.add_async_events() = series;
      } else if (track.kind == xprof::TrackManifest::TRACK_KIND_COUNTER) {
        if (track.counter_series_index < 0) {
          track.counter_series_index = header_response_.counter_events_size();
          *header_response_.add_counter_events() = series;
        } else {
          xprof::TraceEventSeries* dst =
              header_response_.mutable_counter_events(
                  track.counter_series_index);
          for (int k = 0; k < n; ++k) {
            const uint64_t d = (k == 0) ? (series.deltas(0) - prev_end_ts_ps)
                                        : series.deltas(k);
            dst->add_deltas(d);
            if (k < series.event_metadata_size()) {
              *dst->add_event_metadata() = series.event_metadata(k);
            } else {
              dst->add_event_metadata();
            }
          }
        }
      }
    }

    track.received_events += static_cast<uint64_t>(n);
    total_received_events_ += static_cast<uint64_t>(n);
  }

  for (const uint32_t sealed_id : chunk.sealed_track_ids()) {
    if (sealed_id >= tracks_.size()) {
      return absl::DataLossError(absl::StrCat(
          "sealed_track_ids references invalid track ", sealed_id));
    }
    TrackState& track = tracks_[sealed_id];
    if (track.sealed) {
      return absl::DataLossError(
          absl::StrCat("Track ", sealed_id, " sealed more than once"));
    }
    if (track.received_events != track.expected_event_count) {
      return absl::DataLossError(absl::StrCat(
          "Track ", sealed_id, " sealed with event count ",
          track.received_events, ", expected ", track.expected_event_count));
    }
    track.sealed = true;
  }

  return absl::OkStatus();
}

absl::Status StreamingTraceIngestor::OnTrailer(
    const xprof::TraceStreamTrailer& trailer) {
  if (!seen_header_ || seen_trailer_) {
    return absl::DataLossError("Unexpected TraceStreamTrailer frame");
  }
  if (trailer.total_events() != total_received_events_) {
    return absl::DataLossError(
        absl::StrCat("TraceStreamTrailer total_events mismatch: expected ",
                     total_received_events_, ", got ", trailer.total_events()));
  }
  const uint32_t current_string_count =
      static_cast<uint32_t>(header_response_.interned_strings_size());
  if (trailer.final_string_count() != current_string_count) {
    return absl::DataLossError(absl::StrCat(
        "TraceStreamTrailer final_string_count mismatch: expected ",
        current_string_count, ", got ", trailer.final_string_count()));
  }
  if (trailer.track_event_counts_size() != static_cast<int>(tracks_.size()) ||
      trailer.track_event_checksums_size() !=
          static_cast<int>(tracks_.size())) {
    return absl::DataLossError(
        "TraceStreamTrailer track count/checksum array size mismatch");
  }

  for (size_t t = 0; t < tracks_.size(); ++t) {
    const TrackState& track = tracks_[t];
    if (!track.sealed) {
      return absl::DataLossError(
          absl::StrCat("Track ", t, " was not sealed before trailer"));
    }
    if (trailer.track_event_counts(static_cast<int>(t)) !=
            track.received_events ||
        track.received_events != track.expected_event_count) {
      return absl::DataLossError(absl::StrCat(
          "TraceStreamTrailer track_event_counts mismatch on track ", t));
    }
    if (trailer.track_event_checksums(static_cast<int>(t)) != track.checksum) {
      return absl::DataLossError(absl::StrCat(
          "TraceStreamTrailer track_event_checksums mismatch on track ", t));
    }
  }

  seen_trailer_ = true;
  return absl::OkStatus();
}

absl::Status StreamingTraceIngestor::Finish(
    DataProvider& data_provider, Timeline& timeline,
    std::optional<std::pair<Milliseconds, Milliseconds>> visible_range_from_url,
    std::vector<std::pair<std::string, bool>>* details_out) {
  if (!active_) {
    return absl::FailedPreconditionError("Trace stream is not active");
  }
  if (!status_.ok()) {
    const absl::Status err = status_;
    Abort();
    return err;
  }
  if (!seen_header_ || !seen_trailer_ ||
      frame_read_offset_ != frame_buffer_.size()) {
    Abort();
    return absl::DataLossError(
        "Truncated trace stream: missing header/trailer or trailing bytes");
  }

  std::vector<std::pair<std::string, bool>> extracted_details;
  if (details_out != nullptr) {
    extracted_details.reserve(header_response_.details_size());
    for (const xprof::DetailItem& detail : header_response_.details()) {
      extracted_details.emplace_back(detail.name(), detail.value());
    }
  }

  std::vector<absl::string_view> complete_slices;
  complete_slices.reserve(complete_wire_payloads_.size());
  for (const std::string& payload : complete_wire_payloads_) {
    complete_slices.push_back(payload);
  }

  auto release_buffers = [this, &complete_slices]() {
    complete_slices.clear();
    complete_slices.shrink_to_fit();
    std::vector<std::string>().swap(complete_wire_payloads_);
    header_response_.Clear();
    std::vector<TrackState>().swap(tracks_);
  };

  if (!data_provider.ProcessStreamedTraceEvents(
          header_response_, complete_slices, release_buffers, timeline,
          visible_range_from_url)) {
    Abort();
    return absl::DataLossError("DataProvider failed to process streamed trace");
  }

  if (details_out != nullptr) {
    *details_out = std::move(extracted_details);
  }
  Abort();
  return absl::OkStatus();
}

}  // namespace traceviewer
