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
// NOTE: "zstd.h" is used instead of "third_party/zstdlib/zstd.h" to maintain
// compilation compatibility with open-source Bazel builds (using @net_zstd).
#include "zstd.h"  // NOLINT(build/include)

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

  // Decompresses into the internal reusable buffer. The returned string_view
  // remains valid until the next call to Decompress(), DecompressStream(),
  // ShrinkToDefaultCapacity(), or ReleaseBuffer(), or until move-assignment or
  // destruction of the owning context.
  absl::StatusOr<absl::string_view> Decompress(
      absl::string_view compressed_input) ABSL_ATTRIBUTE_LIFETIME_BOUND;

  // Streams `compressed_input` through the fixed-capacity reusable buffer
  // (default 8 MiB) via ZSTD_decompressStream without growing `buffer_` to
  // the full frame size, invoking `on_chunk` for each decoded window. The view
  // passed to `on_chunk` is valid only during that call. If `on_chunk` or
  // decompression returns a non-OK status, aborts immediately without rolling
  // back chunks already passed to `on_chunk`.
  absl::Status DecompressStream(
      absl::string_view compressed_input,
      absl::FunctionRef<absl::Status(absl::string_view)> on_chunk);

  // Shrinks the internal buffer back to kDefaultBufferSize if a large frame
  // caused it to grow beyond the default capacity.
  void ShrinkToDefaultCapacity();

  // Releases the internal reusable buffer and resets capacity to 0.
  void ReleaseBuffer();

  size_t capacity() const { return capacity_; }

 private:
  struct DCtxDeleter {
    void operator()(ZSTD_DCtx* dctx) const { ZSTD_freeDCtx(dctx); }
  };

  std::unique_ptr<ZSTD_DCtx, DCtxDeleter> dctx_;
  std::unique_ptr<char[]> buffer_;
  size_t capacity_ = 0;
};

}  // namespace profiler
}  // namespace tensorflow

#endif  // THIRD_PARTY_XPROF_CONVERT_TRACE_VIEWER_DELTA_SERIES_ZSTD_COMPRESSION_H_
