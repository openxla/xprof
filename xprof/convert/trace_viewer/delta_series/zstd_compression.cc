#include "xprof/convert/trace_viewer/delta_series/zstd_compression.h"

#include <cstddef>
#include <cstdint>
#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/resize_and_overwrite.h"
#include "absl/strings/string_view.h"
// NOTE: "zstd.h" is used instead of "third_party/zstdlib/zstd.h" to maintain
// compilation compatibility with open-source Bazel builds (using @net_zstd).
#include "zstd.h"  // NOLINT(build/include)

namespace tensorflow {
namespace profiler {

absl::StatusOr<std::string> ZstdCompression::Compress(absl::string_view input) {
  size_t bound = ZSTD_compressBound(input.size());
  std::string compressed;
  size_t compressed_size = 0;
  absl::StringResizeAndOverwrite(
      compressed, bound, [&](char* buf, size_t buf_size) {
        compressed_size =
            ZSTD_compress(buf, buf_size, input.data(), input.size(), 1);
        return ZSTD_isError(compressed_size) ? 0 : compressed_size;
      });
  if (ZSTD_isError(compressed_size)) {
    return absl::InternalError("Zstd compression failed.");
  }

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

  std::string decompressed;
  size_t actual_size = 0;
  absl::StringResizeAndOverwrite(
      decompressed, static_cast<size_t>(decompressed_size),
      [&](char* buf, size_t buf_size) {
        actual_size = ZSTD_decompress(buf, buf_size, compressed_input.data(),
                                      compressed_input.size());
        return ZSTD_isError(actual_size) ? 0 : actual_size;
      });
  if (ZSTD_isError(actual_size)) {
    return absl::InternalError("Zstd decompression failed.");
  }

  return decompressed;
}

}  // namespace profiler
}  // namespace tensorflow
