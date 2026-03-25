#include "xprof/convert/trace_viewer/delta_series/zstd_compression.h"

#include <cstddef>
#include <cstdint>
#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
// NOTE: "zstd.h" is used instead of "third_party/zstdlib/zstd.h" to maintain
// compilation compatibility with open-source Bazel builds (using @net_zstd).
#include "zstd.h"  // NOLINT(build/include)

namespace tensorflow {
namespace profiler {

absl::StatusOr<std::string> ZstdCompression::Compress(absl::string_view input) {
  size_t bound = ZSTD_compressBound(input.size());
  std::string compressed(bound, '\0');

  // compression level 1 is standard/fast
  size_t compressed_size =
      ZSTD_compress(&compressed[0], bound, input.data(), input.size(), 1);
  if (ZSTD_isError(compressed_size)) {
    return absl::InternalError("Zstd compression failed.");
  }

  compressed.resize(compressed_size);
  return compressed;
}

absl::StatusOr<std::string> ZstdCompression::Decompress(
    absl::string_view compressed_input) {
  uint64_t decompressed_size = ZSTD_getFrameContentSize(
      compressed_input.data(), compressed_input.size());

  if (decompressed_size == ZSTD_CONTENTSIZE_UNKNOWN ||
      decompressed_size == ZSTD_CONTENTSIZE_ERROR) {
    return absl::InternalError(
        "Zstd decompression failed: unknown or error frame content size.");
  }

  std::string decompressed(decompressed_size, '\0');
  size_t actual_size =
      ZSTD_decompress(&decompressed[0], decompressed_size,
                      compressed_input.data(), compressed_input.size());
  if (ZSTD_isError(actual_size)) {
    return absl::InternalError("Zstd decompression failed.");
  }

  decompressed.resize(actual_size);
  return decompressed;
}

}  // namespace profiler
}  // namespace tensorflow
