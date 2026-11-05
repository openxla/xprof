#ifndef THIRD_PARTY_XPROF_CONVERT_TRACE_VIEWER_DELTA_SERIES_ZSTD_COMPRESSION_H_
#define THIRD_PARTY_XPROF_CONVERT_TRACE_VIEWER_DELTA_SERIES_ZSTD_COMPRESSION_H_

#include <cstddef>
#include <memory>
#include <string>

#include "absl/base/attributes.h"
#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"

// Forward declaration of Zstd decompression context structure to avoid
// exposing zstd.h in the header.
struct ZSTD_DCtx_s;

namespace tensorflow {
namespace profiler {

// 4-byte magic prefix identifying a length-delimited multi-frame Zstd stream.
inline constexpr absl::string_view kXChunkMagic = "XCHN";

// Holds uninitialized heap-allocated memory without zero-initialization
// overhead.
struct UninitializedByteBuffer {
  std::unique_ptr<char[]> data;
  size_t size = 0;

  absl::string_view view() const ABSL_ATTRIBUTE_LIFETIME_BOUND {
    return absl::string_view(data.get(), size);
  }
};

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

  // Decompresses the provided Zstandard-compressed string into an
  // UninitializedByteBuffer without zero-filling.
  static absl::StatusOr<UninitializedByteBuffer> DecompressToBuffer(
      absl::string_view compressed_input);
};

// Reusable streaming decompression context managing a persistent ZSTD_DCtx
// and an uninitialized reusable output buffer (default 8 MiB).
// Single-threaded per instance.
class ZstdDecompressorContext {
 public:
  static constexpr size_t kDefaultBufferSize = 8 * 1024 * 1024;  // 8 MiB

  explicit ZstdDecompressorContext(
      size_t initial_capacity = kDefaultBufferSize);
  ~ZstdDecompressorContext();

  // Move-only.
  ZstdDecompressorContext(ZstdDecompressorContext&& other) noexcept;
  ZstdDecompressorContext& operator=(ZstdDecompressorContext&& other) noexcept;
  ZstdDecompressorContext(const ZstdDecompressorContext&) = delete;
  ZstdDecompressorContext& operator=(const ZstdDecompressorContext&) = delete;

  // Decompresses into the internal reusable buffer. The returned string_view is
  // valid until the next call to Decompress() on this context instance.
  absl::StatusOr<absl::string_view> Decompress(
      absl::string_view compressed_input);

  // Decompresses into a standalone UninitializedByteBuffer using this context's
  // reusable ZSTD_DCtx state.
  absl::StatusOr<UninitializedByteBuffer> DecompressToBuffer(
      absl::string_view compressed_input);

  // Streams `compressed_input` through the fixed-capacity reusable buffer
  // (default 8 MiB) via ZSTD_decompressStream without growing `buffer_` to
  // the full frame size, invoking `on_chunk` for each decoded window. If
  // `on_chunk` returns a non-OK status, aborts immediately and returns it.
  absl::Status DecompressStream(
      absl::string_view compressed_input,
      absl::FunctionRef<absl::Status(absl::string_view)> on_chunk);

  // Shrinks the internal buffer back to kDefaultBufferSize if a large frame
  // caused it to grow beyond the default capacity.
  void ShrinkToDefaultCapacity();

  size_t capacity() const { return capacity_; }

 private:
  struct ZSTD_DCtx_s* dctx_ = nullptr;
  std::unique_ptr<char[]> buffer_;
  size_t capacity_ = 0;
};

using StreamingZstdDecompressor = ZstdDecompressorContext;

}  // namespace profiler
}  // namespace tensorflow

#endif  // THIRD_PARTY_XPROF_CONVERT_TRACE_VIEWER_DELTA_SERIES_ZSTD_COMPRESSION_H_
