#ifndef THIRD_PARTY_XPROF_CONVERT_TRACE_VIEWER_DELTA_SERIES_ZSTD_COMPRESSION_H_
#define THIRD_PARTY_XPROF_CONVERT_TRACE_VIEWER_DELTA_SERIES_ZSTD_COMPRESSION_H_

#include <cstdint>
#include <string>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"

namespace tensorflow {
namespace profiler {

// Updates a 64-bit rolling checksum over a track event's
// (timestamp_ps, duration_ps, name_ref) tuple for stream verification.
inline uint64_t UpdateTrackEventChecksum(uint64_t hash, uint64_t timestamp_ps,
                                         uint64_t duration_ps,
                                         uint32_t name_ref) {
  constexpr uint64_t kPrime1 = 0x9E3779B185EBCA87ULL;
  constexpr uint64_t kPrime2 = 0xC2B2AE3D27D4EB4FULL;
  constexpr uint64_t kPrime3 = 0x165667B19E3779F9ULL;
  const auto rotl64 = [](uint64_t x, int r) -> uint64_t {
    return (x << r) | (x >> (64 - r));
  };
  hash ^= timestamp_ps * kPrime2;
  hash = rotl64(hash, 31) * kPrime1;
  hash ^= duration_ps * kPrime3;
  hash = rotl64(hash, 27) * kPrime1 + 0x85EBCA77C2B2AE63ULL;
  hash ^= static_cast<uint64_t>(name_ref) * kPrime2;
  hash = rotl64(hash, 33) * kPrime1;
  return hash;
}

// Provides ZStandard compression and decompression utilities.
class ZstdCompression {
 public:
  // Compresses the provided input string using Zstandard.
  // Returns the compressed bytes on success, or an error status on failure.
  static absl::StatusOr<std::string> Compress(absl::string_view input);

  // Decompresses the provided Zstandard-compressed string.
  // Returns the decompressed bytes on success, or an error status on failure.
  static absl::StatusOr<std::string> Decompress(
      absl::string_view compressed_input);
};

}  // namespace profiler
}  // namespace tensorflow

#endif  // THIRD_PARTY_XPROF_CONVERT_TRACE_VIEWER_DELTA_SERIES_ZSTD_COMPRESSION_H_
