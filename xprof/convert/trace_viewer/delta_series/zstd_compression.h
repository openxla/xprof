#ifndef THIRD_PARTY_XPROF_CONVERT_TRACE_VIEWER_DELTA_SERIES_ZSTD_COMPRESSION_H_
#define THIRD_PARTY_XPROF_CONVERT_TRACE_VIEWER_DELTA_SERIES_ZSTD_COMPRESSION_H_

#include <string>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"

namespace tensorflow {
namespace profiler {

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
