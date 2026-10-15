#ifndef THIRD_PARTY_XPROF_FRONTEND_APP_COMPONENTS_TRACE_VIEWER_V2_TRACE_HELPER_TRACE_STREAM_INGESTOR_H_
#define THIRD_PARTY_XPROF_FRONTEND_APP_COMPONENTS_TRACE_VIEWER_V2_TRACE_HELPER_TRACE_STREAM_INGESTOR_H_

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "google/protobuf/arena.h"
#include "frontend/app/components/trace_viewer_v2/timeline/data_provider.h"
#include "frontend/app/components/trace_viewer_v2/timeline/timeline.h"
#include "frontend/app/components/trace_viewer_v2/trace_helper/trace_event.h"
#include "plugin/xprof/protobuf/trace_data_response.pb.h"

namespace traceviewer {

// Ingests a framed, zstd-compressed `TraceStreamMessage` byte stream
// incrementally (`P1`–`P12`). Arbitrary byte chunk boundaries (from 1 byte up
// to multi-megabyte network chunks) are reassembled into length-prefixed frames
// (`[uint32_le compressed_size][zstd_compressed_bytes]`), decompressed,
// validated against the `TrackManifest` and `TraceStreamTrailer` (`P10`
// counts and XXH64 prime-mixing checksums), and committed into `DataProvider`
// and `Timeline`. On any framing, decompression, monotonicity, or trailer
// verification error, `Finish` discards all staged state without mutating
// `DataProvider` or `Timeline`.
class StreamingTraceIngestor {
 public:
  StreamingTraceIngestor() = default;

  // Starts or resets a streaming ingestion session with `stream_id`.
  void Begin(uint64_t stream_id = 0);

  // Pushes an arbitrary slice of network bytes into the frame reassembler and
  // processes all complete frames contained within the buffered stream.
  absl::Status PushBytes(absl::Span<const uint8_t> bytes);

  absl::Status PushBytes(absl::string_view bytes) {
    return PushBytes(absl::MakeConstSpan(
        reinterpret_cast<const uint8_t*>(bytes.data()), bytes.size()));
  }

  // Verifies that a valid `TraceStreamHeader` and `TraceStreamTrailer` were
  // received with zero trailing bytes and commits the ingested trace into
  // `data_provider` and `timeline`. On failure, aborts and leaves
  // `data_provider` and `timeline` untouched.
  absl::Status Finish(
      DataProvider& data_provider, Timeline& timeline,
      std::optional<std::pair<Milliseconds, Milliseconds>>
          visible_range_from_url = std::nullopt,
      std::vector<std::pair<std::string, bool>>* details_out = nullptr);

  // Discards all buffered state and marks the stream inactive.
  void Abort();

  uint64_t stream_id() const { return stream_id_; }

 private:
  struct TrackState {
    xprof::TrackManifest::TrackKind kind =
        xprof::TrackManifest::TRACK_KIND_UNSPECIFIED;
    uint32_t process_id = 0;
    uint64_t thread_id = 0;
    uint32_t name_ref = 0;
    uint64_t expected_event_count = 0;
    bool ordered = true;
    uint64_t received_events = 0;
    uint64_t checksum = 0;
    bool has_last_event = false;
    uint64_t last_ts_ps = 0;
    uint64_t last_dur_ps = 0;
    int counter_series_index = -1;
    bool sealed = false;
  };

  absl::Status ProcessCompressedFrame(absl::string_view compressed_frame);
  absl::Status OnHeader(const xprof::TraceStreamHeader& header);
  absl::Status OnChunk(const xprof::TraceEventChunk& chunk);
  absl::Status OnTrailer(const xprof::TraceStreamTrailer& trailer);

  bool active_ = false;
  uint64_t stream_id_ = 0;
  bool seen_header_ = false;
  bool seen_trailer_ = false;
  uint64_t total_received_events_ = 0;
  absl::Status status_ = absl::OkStatus();

  std::string frame_buffer_;
  size_t frame_read_offset_ = 0;
  google::protobuf::Arena message_arena_;

  xprof::TraceDataResponse header_response_;
  std::vector<TrackState> tracks_;
  std::vector<std::string> complete_wire_payloads_;
};

}  // namespace traceviewer

#endif  // THIRD_PARTY_XPROF_FRONTEND_APP_COMPONENTS_TRACE_VIEWER_V2_TRACE_HELPER_TRACE_STREAM_INGESTOR_H_
