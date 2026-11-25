#include "xprof/convert/trace_viewer/delta_series/zstd_compression.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>

#include "absl/functional/function_ref.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
// NOTE: "zstd.h" is used instead of "third_party/zstdlib/zstd.h" to maintain
// compilation compatibility with open-source Bazel builds (using @net_zstd).
#include "zstd.h"  // NOLINT(build/include)

namespace tensorflow {
namespace profiler {

absl::StatusOr<std::string> ZstdCompression::Compress(absl::string_view input) {
  ZstdCompressorContext context(/*compression_level=*/1);
  std::string compressed;
  absl::Status status = context.Compress(input, &compressed);
  if (!status.ok()) {
    return status;
  }
  return compressed;
}

namespace {

// Validates and returns the uncompressed frame content size from
// `compressed_input`, or an error if the frame header is missing or invalid.
absl::StatusOr<size_t> GetValidatedDecompressedSize(
    absl::string_view compressed_input) {
  const uint64_t decompressed_size = ZSTD_getFrameContentSize(
      compressed_input.data(), compressed_input.size());

  if (decompressed_size == ZSTD_CONTENTSIZE_UNKNOWN ||
      decompressed_size == ZSTD_CONTENTSIZE_ERROR) {
    return absl::InternalError(
        "Zstd decompression failed: unknown or error frame content size.");
  }

#if SIZE_MAX < UINT64_MAX
  if (decompressed_size > static_cast<uint64_t>(SIZE_MAX)) {
    return absl::InternalError(
        "Zstd decompressed size exceeds platform size_t address space.");
  }
#endif

  return static_cast<size_t>(decompressed_size);
}

}  // namespace

absl::StatusOr<std::string> ZstdCompression::Decompress(
    absl::string_view compressed_input) {
  absl::StatusOr<size_t> decompressed_size =
      GetValidatedDecompressedSize(compressed_input);
  if (!decompressed_size.ok()) {
    return decompressed_size.status();
  }

  std::string out(*decompressed_size, '\0');
  size_t actual_size =
      ZSTD_decompress(&out[0], *decompressed_size, compressed_input.data(),
                      compressed_input.size());
  if (ZSTD_isError(actual_size)) {
    return absl::InternalError("Zstd decompression failed.");
  }
  return out;
}

absl::StatusOr<UninitializedByteBuffer> ZstdCompression::DecompressToBuffer(
    absl::string_view compressed_input) {
  const absl::StatusOr<size_t> decompressed_size =
      GetValidatedDecompressedSize(compressed_input);
  if (!decompressed_size.ok()) {
    return decompressed_size.status();
  }

  // Default-initialization leaves memory uninitialized. Skip allocation when
  // size is 0 so data remains nullptr.
  std::unique_ptr<char[]> buf;
  if (*decompressed_size > 0) {
    buf = absl::make_unique_for_overwrite<char[]>(*decompressed_size);
  }
  const size_t actual_size =
      ZSTD_decompress(buf.get(), *decompressed_size, compressed_input.data(),
                      compressed_input.size());
  if (ZSTD_isError(actual_size)) {
    return absl::InternalError("Zstd decompression failed.");
  }
  return UninitializedByteBuffer{std::move(buf), actual_size};
}

ZstdCompressorContext::ZstdCompressorContext(int compression_level)
    : cctx_(ZSTD_createCCtx()) {
  if (cctx_ != nullptr) {
    ZSTD_CCtx_setParameter(cctx_.get(), ZSTD_c_compressionLevel,
                           compression_level);
    ZSTD_CCtx_setParameter(cctx_.get(), ZSTD_c_checksumFlag, 1);
  }
}

ZstdCompressorContext::~ZstdCompressorContext() = default;

ZstdCompressorContext::ZstdCompressorContext(
    ZstdCompressorContext&& other) noexcept
    : cctx_(std::move(other.cctx_)),
      scratch_buf_(std::move(other.scratch_buf_)),
      scratch_capacity_(std::exchange(other.scratch_capacity_, 0)) {}

ZstdCompressorContext& ZstdCompressorContext::operator=(
    ZstdCompressorContext&& other) noexcept {
  if (this != &other) {
    cctx_ = std::move(other.cctx_);
    scratch_buf_ = std::move(other.scratch_buf_);
    scratch_capacity_ = std::exchange(other.scratch_capacity_, 0);
  }
  return *this;
}

absl::StatusOr<size_t> ZstdCompressorContext::CheckedCompressBound(
    size_t input_size) {
  const size_t bound = ZSTD_compressBound(input_size);
  if (ZSTD_isError(bound)) {
    return absl::InvalidArgumentError(
        "Input exceeds Zstd maximum compressible size.");
  }
  return bound;
}

absl::Status ZstdCompressorContext::Compress(absl::string_view input,
                                             std::string* out) {
  if (out == nullptr) {
    return absl::InvalidArgumentError("Output buffer is null.");
  }
  if (cctx_ == nullptr) {
    return absl::InternalError("Zstd CCtx not initialized.");
  }
  const absl::StatusOr<size_t> bound = CheckedCompressBound(input.size());
  if (!bound.ok()) {
    return bound.status();
  }
  if (scratch_capacity_ < *bound) {
    // Free the old scratch buffer before allocating so old and new buffers
    // are never live on the heap simultaneously.
    scratch_buf_.reset();
    scratch_capacity_ = 0;
    scratch_buf_ = absl::make_unique_for_overwrite<char[]>(*bound);
    scratch_capacity_ = *bound;
  }
  const size_t compressed_size =
      ZSTD_compress2(cctx_.get(), scratch_buf_.get(), *bound,
                     input.empty() ? "" : input.data(), input.size());
  if (ZSTD_isError(compressed_size)) {
    return absl::InternalError("Zstd compression failed.");
  }
  out->assign(scratch_buf_.get(), compressed_size);
  return absl::OkStatus();
}

ZstdDecompressorContext::ZstdDecompressorContext(size_t initial_capacity)
    : dctx_(ZSTD_createDCtx()),
      buffer_(initial_capacity > 0
                  ? absl::make_unique_for_overwrite<char[]>(initial_capacity)
                  : nullptr),
      capacity_(initial_capacity) {}

ZstdDecompressorContext::~ZstdDecompressorContext() = default;

ZstdDecompressorContext::ZstdDecompressorContext(
    ZstdDecompressorContext&& other) noexcept
    : dctx_(std::move(other.dctx_)),
      buffer_(std::move(other.buffer_)),
      capacity_(std::exchange(other.capacity_, 0)) {}

ZstdDecompressorContext& ZstdDecompressorContext::operator=(
    ZstdDecompressorContext&& other) noexcept {
  if (this != &other) {
    dctx_ = std::move(other.dctx_);
    buffer_ = std::move(other.buffer_);
    capacity_ = std::exchange(other.capacity_, 0);
  }
  return *this;
}

absl::StatusOr<absl::string_view> ZstdDecompressorContext::Decompress(
    absl::string_view compressed_input) {
  if (dctx_ == nullptr) {
    return absl::InternalError("Zstd DCtx not initialized.");
  }

  const absl::StatusOr<size_t> decompressed_size =
      GetValidatedDecompressedSize(compressed_input);
  if (!decompressed_size.ok()) {
    return decompressed_size.status();
  }

  if (*decompressed_size > capacity_) {
    // Free the old scratch buffer before allocating so old and new buffers
    // are never live on the heap simultaneously.
    buffer_.reset();
    capacity_ = 0;
    buffer_ = absl::make_unique_for_overwrite<char[]>(*decompressed_size);
    capacity_ = *decompressed_size;
  }

  const size_t actual_size =
      ZSTD_decompressDCtx(dctx_.get(), buffer_.get(), *decompressed_size,
                          compressed_input.data(), compressed_input.size());
  if (ZSTD_isError(actual_size)) {
    return absl::InternalError("Zstd decompression failed.");
  }

  return absl::string_view(buffer_.get(), actual_size);
}

void ZstdDecompressorContext::ShrinkToDefaultCapacity() {
  if (capacity_ > kDefaultBufferSize) {
    buffer_.reset();
    capacity_ = 0;
    buffer_ = absl::make_unique_for_overwrite<char[]>(kDefaultBufferSize);
    capacity_ = kDefaultBufferSize;
  }
}

void ZstdDecompressorContext::ReleaseBuffer() {
  buffer_.reset();
  capacity_ = 0;
}

absl::Status ZstdDecompressorContext::DecompressStream(
    absl::string_view compressed_input,
    absl::FunctionRef<absl::Status(absl::string_view)> on_chunk) {
  if (dctx_ == nullptr) {
    return absl::InternalError("Zstd DCtx not initialized.");
  }
  if (compressed_input.empty()) {
    return absl::InvalidArgumentError("Compressed input is empty.");
  }
  if (buffer_ == nullptr) {
    capacity_ = kDefaultBufferSize;
    buffer_ = absl::make_unique_for_overwrite<char[]>(capacity_);
  }

  // ZSTD_initDStream cannot fail on a valid non-null ZSTD_DCtx.
  ZSTD_initDStream(dctx_.get());

  ZSTD_inBuffer in_buf = {compressed_input.data(), compressed_input.size(), 0};
  while (in_buf.pos < in_buf.size) {
    ZSTD_outBuffer out_buf = {buffer_.get(), capacity_, 0};
    const size_t ret = ZSTD_decompressStream(dctx_.get(), &out_buf, &in_buf);
    if (ZSTD_isError(ret)) {
      return absl::InternalError("Zstd stream decompression error.");
    }
    if (out_buf.pos > 0) {
      absl::Status status =
          on_chunk(absl::string_view(buffer_.get(), out_buf.pos));
      if (!status.ok()) {
        return status;
      }
    }
    // If the frame is incomplete and all input has been consumed, Zstd cannot
    // make further progress (valid frames with buffered output hold back a byte
    // so `in_buf.pos < in_buf.size` until flushed).
    if (ret != 0 && in_buf.pos == in_buf.size) {
      return absl::InternalError(
          "Zstd stream decompression error: truncated compressed input.");
    }
  }

  return absl::OkStatus();
}

}  // namespace profiler
}  // namespace tensorflow
