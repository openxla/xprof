#include "xprof/convert/trace_viewer/delta_series/zstd_compression.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>

#include "absl/functional/function_ref.h"
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
  absl::StatusOr<size_t> decompressed_size =
      GetValidatedDecompressedSize(compressed_input);
  if (!decompressed_size.ok()) {
    return decompressed_size.status();
  }

  if (*decompressed_size == 0) {
    size_t actual_size = ZSTD_decompress(nullptr, 0, compressed_input.data(),
                                         compressed_input.size());
    if (ZSTD_isError(actual_size)) {
      return absl::InternalError("Zstd decompression failed.");
    }
    return UninitializedByteBuffer{nullptr, 0};
  }

  // C++17 default-initialization (new char[] without parentheses) leaves memory
  // uninitialized, avoiding zero-fill overhead.
  auto buf = std::unique_ptr<char[]>(new char[*decompressed_size]);
  size_t actual_size =
      ZSTD_decompress(buf.get(), *decompressed_size, compressed_input.data(),
                      compressed_input.size());
  if (ZSTD_isError(actual_size)) {
    return absl::InternalError("Zstd decompression failed.");
  }

  return UninitializedByteBuffer{std::move(buf), actual_size};
}

ZstdDecompressorContext::ZstdDecompressorContext(size_t initial_capacity)
    : dctx_(ZSTD_createDCtx()),
      buffer_(initial_capacity > 0 ? new char[initial_capacity] : nullptr),
      capacity_(initial_capacity) {}

ZstdDecompressorContext::~ZstdDecompressorContext() {
  ZSTD_freeDCtx(dctx_);
}

ZstdDecompressorContext::ZstdDecompressorContext(
    ZstdDecompressorContext&& other) noexcept
    : dctx_(other.dctx_),
      buffer_(std::move(other.buffer_)),
      capacity_(other.capacity_) {
  other.dctx_ = nullptr;
  other.capacity_ = 0;
}

ZstdDecompressorContext& ZstdDecompressorContext::operator=(
    ZstdDecompressorContext&& other) noexcept {
  if (this != &other) {
    ZSTD_freeDCtx(dctx_);
    dctx_ = other.dctx_;
    buffer_ = std::move(other.buffer_);
    capacity_ = other.capacity_;
    other.dctx_ = nullptr;
    other.capacity_ = 0;
  }
  return *this;
}

absl::StatusOr<absl::string_view> ZstdDecompressorContext::Decompress(
    absl::string_view compressed_input) {
  if (dctx_ == nullptr) {
    return absl::InternalError("Zstd DCtx not initialized.");
  }

  absl::StatusOr<size_t> decompressed_size =
      GetValidatedDecompressedSize(compressed_input);
  if (!decompressed_size.ok()) {
    return decompressed_size.status();
  }

  if (*decompressed_size > capacity_) {
    buffer_ = std::unique_ptr<char[]>(new char[*decompressed_size]);
    capacity_ = *decompressed_size;
  }

  size_t actual_size =
      ZSTD_decompressDCtx(dctx_, buffer_.get(), *decompressed_size,
                          compressed_input.data(), compressed_input.size());
  if (ZSTD_isError(actual_size)) {
    return absl::InternalError("Zstd decompression failed.");
  }

  return absl::string_view(buffer_.get(), actual_size);
}

void ZstdDecompressorContext::ShrinkToDefaultCapacity() {
  if (capacity_ > kDefaultBufferSize) {
    buffer_ = std::unique_ptr<char[]>(new char[kDefaultBufferSize]);
    capacity_ = kDefaultBufferSize;
  }
}

absl::StatusOr<UninitializedByteBuffer>
ZstdDecompressorContext::DecompressToBuffer(
    absl::string_view compressed_input) {
  if (dctx_ == nullptr) {
    return absl::InternalError("Zstd DCtx not initialized.");
  }

  absl::StatusOr<size_t> decompressed_size =
      GetValidatedDecompressedSize(compressed_input);
  if (!decompressed_size.ok()) {
    return decompressed_size.status();
  }

  if (*decompressed_size == 0) {
    size_t actual_size = ZSTD_decompressDCtx(
        dctx_, nullptr, 0, compressed_input.data(), compressed_input.size());
    if (ZSTD_isError(actual_size)) {
      return absl::InternalError("Zstd decompression failed.");
    }
    return UninitializedByteBuffer{nullptr, 0};
  }

  auto buf = std::unique_ptr<char[]>(new char[*decompressed_size]);
  size_t actual_size =
      ZSTD_decompressDCtx(dctx_, buf.get(), *decompressed_size,
                          compressed_input.data(), compressed_input.size());
  if (ZSTD_isError(actual_size)) {
    return absl::InternalError("Zstd decompression failed.");
  }

  return UninitializedByteBuffer{std::move(buf), actual_size};
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
    buffer_ = std::unique_ptr<char[]>(new char[capacity_]);
  }

  // ZSTD_initDStream cannot fail on a valid non-null ZSTD_DCtx.
  ZSTD_initDStream(dctx_);

  ZSTD_inBuffer in_buf = {compressed_input.data(), compressed_input.size(), 0};
  size_t ret = 1;
  while (in_buf.pos < in_buf.size || ret != 0) {
    ZSTD_outBuffer out_buf = {buffer_.get(), capacity_, 0};
    ret = ZSTD_decompressStream(dctx_, &out_buf, &in_buf);
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
