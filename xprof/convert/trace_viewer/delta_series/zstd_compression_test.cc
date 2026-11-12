#include "xprof/convert/trace_viewer/delta_series/zstd_compression.h"

#ifdef __linux__
#include <sys/resource.h>
#endif

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>

#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"

namespace tensorflow {
namespace profiler {
namespace {

// Valid Zstd frame with Single_Segment_flag=0 and FCS_field_size=0, so
// ZSTD_getFrameContentSize returns ZSTD_CONTENTSIZE_UNKNOWN.
constexpr char kUnknownContentSizeFrame[] =
    "\x28\xb5\x2f\xfd\x00\x00\x01\x00\x00";

void MoveAssignContext(ZstdDecompressorContext& dst,
                       ZstdDecompressorContext& src) {
  dst = std::move(src);
}

TEST(ZstdCompressionTest, CompressAndDecompressSuccessfully) {
  absl::string_view original_string =
      "Hello, Zstd! This is a test string to be compressed. "
      "Repeating it makes it more compressible... "
      "Repeating it makes it more compressible... "
      "Repeating it makes it more compressible...";

  absl::StatusOr<std::string> compressed_result =
      ZstdCompression::Compress(original_string);
  ASSERT_TRUE(compressed_result.ok());
  EXPECT_GT(compressed_result->size(), 0);

  absl::StatusOr<std::string> decompressed_result =
      ZstdCompression::Decompress(*compressed_result);
  ASSERT_TRUE(decompressed_result.ok());
  EXPECT_EQ(*decompressed_result, original_string);
}

TEST(ZstdCompressionTest, EmptyString) {
  absl::string_view original_string = "";

  absl::StatusOr<std::string> compressed_result =
      ZstdCompression::Compress(original_string);
  ASSERT_TRUE(compressed_result.ok());

  absl::StatusOr<std::string> decompressed_result =
      ZstdCompression::Decompress(*compressed_result);
  ASSERT_TRUE(decompressed_result.ok());
  EXPECT_EQ(*decompressed_result, original_string);
}

TEST(ZstdCompressionTest, DecompressInvalidInputFails) {
  absl::string_view invalid_compressed_string = "Not compressed data at all";

  absl::StatusOr<std::string> decompressed_result =
      ZstdCompression::Decompress(invalid_compressed_string);
  EXPECT_FALSE(decompressed_result.ok());
  EXPECT_EQ(decompressed_result.status().code(), absl::StatusCode::kInternal);
}

TEST(ZstdCompressionTest, DecompressToBufferRoundTripAndCorruptFrameTest) {
  // 1. Text payload round-trip
  absl::string_view original_string =
      "Performance optimization with zero-fill elimination and SIMD128 "
      "support. "
      "Repeating payload to make compression effective. "
      "Repeating payload to make compression effective. "
      "Repeating payload to make compression effective.";
  auto compressed = ZstdCompression::Compress(original_string);
  ASSERT_TRUE(compressed.ok());

  auto decompressed = ZstdCompression::DecompressToBuffer(*compressed);
  ASSERT_TRUE(decompressed.ok());
  EXPECT_EQ(decompressed->size, original_string.size());
  EXPECT_EQ(decompressed->view(), original_string);

  // 2. Empty string round-trip
  auto empty_compressed = ZstdCompression::Compress("");
  ASSERT_TRUE(empty_compressed.ok());
  auto empty_decompressed =
      ZstdCompression::DecompressToBuffer(*empty_compressed);
  ASSERT_TRUE(empty_decompressed.ok());
  EXPECT_EQ(empty_decompressed->data, nullptr);
  EXPECT_EQ(empty_decompressed->size, 0);
  EXPECT_TRUE(empty_decompressed->view().empty());

  // 3. Corrupt frame failure
  auto corrupt_result =
      ZstdCompression::DecompressToBuffer("not a valid zstd frame");
  EXPECT_FALSE(corrupt_result.ok());
  EXPECT_EQ(corrupt_result.status().code(), absl::StatusCode::kInternal);

  // 4. Truncated frame failure
  if (compressed->size() > 5) {
    absl::string_view truncated =
        absl::string_view(compressed->data(), compressed->size() - 5);
    auto truncated_result = ZstdCompression::DecompressToBuffer(truncated);
    EXPECT_FALSE(truncated_result.ok());
  }
}

TEST(ZstdDecompressorContextTest,
     StreamingContextMultiplePayloadsAndBufferReuse) {
  ZstdDecompressorContext context;
  EXPECT_GE(context.capacity(), ZstdDecompressorContext::kDefaultBufferSize);

  // Payload A
  std::string payload_a(1024 * 128, 'A');
  auto comp_a = ZstdCompression::Compress(payload_a);
  ASSERT_TRUE(comp_a.ok());

  // Payload B
  std::string payload_b(1024 * 256, 'B');
  auto comp_b = ZstdCompression::Compress(payload_b);
  ASSERT_TRUE(comp_b.ok());

  // Payload C (Empty)
  auto comp_c = ZstdCompression::Compress("");
  ASSERT_TRUE(comp_c.ok());

  size_t initial_cap = context.capacity();

  // Decompress A
  auto res_a = context.Decompress(*comp_a);
  ASSERT_TRUE(res_a.ok());
  EXPECT_EQ(*res_a, payload_a);
  EXPECT_EQ(context.capacity(), initial_cap);  // Reused buffer without realloc

  // Decompress B
  auto res_b = context.Decompress(*comp_b);
  ASSERT_TRUE(res_b.ok());
  EXPECT_EQ(*res_b, payload_b);
  EXPECT_EQ(context.capacity(), initial_cap);  // Still reused

  // Decompress C (Empty)
  auto res_c = context.Decompress(*comp_c);
  ASSERT_TRUE(res_c.ok());
  EXPECT_TRUE(res_c->empty());

  // DecompressToBuffer on context (allocates distinct UninitializedByteBuffer)
  auto buf_a = context.DecompressToBuffer(*comp_a);
  ASSERT_TRUE(buf_a.ok());
  EXPECT_EQ(buf_a->view(), payload_a);

  auto buf_b = context.DecompressToBuffer(*comp_b);
  ASSERT_TRUE(buf_b.ok());
  EXPECT_EQ(buf_b->view(), payload_b);

  // Corrupt input on context returns error gracefully
  auto corrupt_res = context.Decompress("invalid zstd frame data");
  EXPECT_FALSE(corrupt_res.ok());
  EXPECT_EQ(corrupt_res.status().code(), absl::StatusCode::kInternal);
}

TEST(ZstdDecompressorContextTest, BufferGrowsBeyondDefaultCapacity) {
  // Use a small initial capacity to test buffer growth
  ZstdDecompressorContext context(1024);  // 1 KiB initial
  EXPECT_EQ(context.capacity(), 1024);

  // 64 KiB payload exceeds 1 KiB initial capacity
  std::string large_payload(64 * 1024, 'X');
  auto comp_large = ZstdCompression::Compress(large_payload);
  ASSERT_TRUE(comp_large.ok());

  auto res = context.Decompress(*comp_large);
  ASSERT_TRUE(res.ok());
  EXPECT_EQ(*res, large_payload);
  EXPECT_GE(context.capacity(), 64 * 1024);
}

TEST(ZstdDecompressorContextTest, DecompressStreamBoundedWindowReconstruction) {
  ZstdDecompressorContext context(1024);
  EXPECT_EQ(context.capacity(), 1024);

  // 64 KiB payload with varying characters to test multi-chunk streaming.
  std::string original_payload;
  original_payload.reserve(64 * 1024);
  for (int i = 0; i < 64 * 1024; ++i) {
    original_payload.push_back(static_cast<char>('a' + (i % 26)));
  }

  auto comp_res = ZstdCompression::Compress(original_payload);
  ASSERT_TRUE(comp_res.ok());

  std::string reconstructed;
  int chunk_count = 0;
  absl::Status status =
      context.DecompressStream(*comp_res, [&](absl::string_view chunk) {
        ++chunk_count;
        reconstructed.append(chunk.data(), chunk.size());
        EXPECT_LE(chunk.size(), 1024);
        return absl::OkStatus();
      });

  ASSERT_TRUE(status.ok()) << status.message();
  EXPECT_EQ(reconstructed, original_payload);
  EXPECT_GE(chunk_count, 64);
  EXPECT_EQ(context.capacity(), 1024);
}

TEST(ZstdDecompressorContextTest,
     DecompressStreamEarlyCancellationAndContextReuse) {
  ZstdDecompressorContext context(1024);

  std::string original_payload(64 * 1024, 'K');
  auto comp_res = ZstdCompression::Compress(original_payload);
  ASSERT_TRUE(comp_res.ok());

  // 1. Cancel on second chunk.
  int chunks_seen = 0;
  absl::Status cancel_status =
      context.DecompressStream(*comp_res, [&](absl::string_view chunk) {
        ++chunks_seen;
        if (chunks_seen == 2) {
          return absl::CancelledError("abort");
        }
        return absl::OkStatus();
      });

  EXPECT_TRUE(absl::IsCancelled(cancel_status));
  EXPECT_EQ(chunks_seen, 2);

  // 2. Context reuse immediately afterwards on a valid frame succeeds
  // completely.
  std::string reconstructed;
  absl::Status ok_status =
      context.DecompressStream(*comp_res, [&](absl::string_view chunk) {
        reconstructed.append(chunk.data(), chunk.size());
        return absl::OkStatus();
      });

  ASSERT_TRUE(ok_status.ok()) << ok_status.message();
  EXPECT_EQ(reconstructed, original_payload);
  EXPECT_EQ(context.capacity(), 1024);
}

TEST(ZstdCompressionTest, DecompressErrorBranchesAndUnknownContentSize) {
  const absl::string_view unknown_size_frame(
      kUnknownContentSizeFrame, sizeof(kUnknownContentSizeFrame) - 1);

  const absl::StatusOr<std::string> unknown_res =
      ZstdCompression::Decompress(unknown_size_frame);
  EXPECT_EQ(unknown_res.status().code(), absl::StatusCode::kInternal);

  const absl::StatusOr<UninitializedByteBuffer> unknown_buf_res =
      ZstdCompression::DecompressToBuffer(unknown_size_frame);
  EXPECT_EQ(unknown_buf_res.status().code(), absl::StatusCode::kInternal);

  // Truncate the last byte of an empty frame so ZSTD_getFrameContentSize
  // still returns 0 while ZSTD_decompress detects truncation.
  const absl::StatusOr<std::string> empty_compressed =
      ZstdCompression::Compress("");
  ASSERT_TRUE(empty_compressed.ok());
  ASSERT_GT(empty_compressed->size(), 1);
  const absl::string_view corrupt_empty(empty_compressed->data(),
                                        empty_compressed->size() - 1);

  EXPECT_EQ(ZstdCompression::Decompress(corrupt_empty).status().code(),
            absl::StatusCode::kInternal);
  EXPECT_EQ(ZstdCompression::DecompressToBuffer(corrupt_empty).status().code(),
            absl::StatusCode::kInternal);

  // Truncate a non-empty frame so ZSTD_getFrameContentSize returns > 0 while
  // ZSTD_decompress returns an error.
  const absl::StatusOr<std::string> compressed =
      ZstdCompression::Compress("Repeatable payload for truncation testing.");
  ASSERT_TRUE(compressed.ok());
  ASSERT_GT(compressed->size(), 5);
  const absl::string_view truncated(compressed->data(), compressed->size() - 5);
  EXPECT_EQ(ZstdCompression::Decompress(truncated).status().code(),
            absl::StatusCode::kInternal);
}

TEST(ZstdDecompressorContextTest, MoveSemanticsAndMovedFromErrors) {
  const std::string payload = "Move semantics test payload.";
  const absl::StatusOr<std::string> compressed =
      ZstdCompression::Compress(payload);
  ASSERT_TRUE(compressed.ok());

  ZstdDecompressorContext ctx1(/*initial_capacity=*/1024);
  ZstdDecompressorContext ctx2(std::move(ctx1));

  // Moved-from ctx1 has null dctx_ and zero capacity, failing on all entry
  // points.
  // NOLINTNEXTLINE(bugprone-use-after-move)
  EXPECT_EQ(ctx1.capacity(), 0);
  // NOLINTNEXTLINE(bugprone-use-after-move)
  EXPECT_EQ(ctx1.Decompress(*compressed).status().code(),
            absl::StatusCode::kInternal);
  // NOLINTNEXTLINE(bugprone-use-after-move)
  EXPECT_EQ(ctx1.DecompressToBuffer(*compressed).status().code(),
            absl::StatusCode::kInternal);
  // NOLINTNEXTLINE(bugprone-use-after-move)
  const absl::Status moved_stream_status = ctx1.DecompressStream(
      *compressed, [](absl::string_view) { return absl::OkStatus(); });
  EXPECT_EQ(moved_stream_status.code(), absl::StatusCode::kInternal);

  EXPECT_EQ(ctx2.capacity(), 1024);
  const absl::StatusOr<absl::string_view> res2 = ctx2.Decompress(*compressed);
  ASSERT_TRUE(res2.ok());
  EXPECT_EQ(*res2, payload);

  // Move-assign into an active context (dctx_ != nullptr).
  ZstdDecompressorContext ctx3(/*initial_capacity=*/512);
  ctx3 = std::move(ctx2);
  EXPECT_EQ(ctx3.capacity(), 1024);
  // NOLINTNEXTLINE(bugprone-use-after-move)
  EXPECT_EQ(ctx2.capacity(), 0);
  // NOLINTNEXTLINE(bugprone-use-after-move)
  EXPECT_EQ(ctx2.Decompress(*compressed).status().code(),
            absl::StatusCode::kInternal);
  const absl::StatusOr<absl::string_view> res3 = ctx3.Decompress(*compressed);
  ASSERT_TRUE(res3.ok());
  EXPECT_EQ(*res3, payload);

  // Move-assign into a moved-from context (dctx_ == nullptr).
  ctx1 = std::move(ctx3);
  EXPECT_EQ(ctx1.capacity(), 1024);
  // NOLINTNEXTLINE(bugprone-use-after-move)
  EXPECT_EQ(ctx3.capacity(), 0);
  // NOLINTNEXTLINE(bugprone-use-after-move)
  EXPECT_EQ(ctx3.Decompress(*compressed).status().code(),
            absl::StatusCode::kInternal);
  const absl::StatusOr<absl::string_view> res1 = ctx1.Decompress(*compressed);
  ASSERT_TRUE(res1.ok());
  EXPECT_EQ(*res1, payload);

  // Self-move assignment is a safe no-op.
  MoveAssignContext(ctx1, ctx1);
  const absl::StatusOr<absl::string_view> res_self =
      ctx1.Decompress(*compressed);
  ASSERT_TRUE(res_self.ok());
  EXPECT_EQ(*res_self, payload);
}

TEST(ZstdDecompressorContextTest,
     DecompressAndDecompressToBufferErrorAndShrinkBranches) {
  ZstdDecompressorContext context;
  const absl::string_view unknown_size_frame(
      kUnknownContentSizeFrame, sizeof(kUnknownContentSizeFrame) - 1);

  const absl::StatusOr<std::string> empty_compressed =
      ZstdCompression::Compress("");
  ASSERT_TRUE(empty_compressed.ok());
  ASSERT_GT(empty_compressed->size(), 1);
  const absl::string_view corrupt_empty(empty_compressed->data(),
                                        empty_compressed->size() - 1);

  const absl::StatusOr<std::string> non_empty_compressed =
      ZstdCompression::Compress("Payload to test truncated decompression.");
  ASSERT_TRUE(non_empty_compressed.ok());
  ASSERT_GT(non_empty_compressed->size(), 5);
  const absl::string_view truncated_non_empty(non_empty_compressed->data(),
                                              non_empty_compressed->size() - 5);

  // Context::Decompress error paths.
  EXPECT_EQ(context.Decompress(unknown_size_frame).status().code(),
            absl::StatusCode::kInternal);
  EXPECT_EQ(context.Decompress(corrupt_empty).status().code(),
            absl::StatusCode::kInternal);
  EXPECT_EQ(context.Decompress(truncated_non_empty).status().code(),
            absl::StatusCode::kInternal);

  // Context::DecompressToBuffer empty, corrupt, unknown-size, and truncated.
  const absl::StatusOr<UninitializedByteBuffer> empty_buf =
      context.DecompressToBuffer(*empty_compressed);
  ASSERT_TRUE(empty_buf.ok());
  EXPECT_EQ(empty_buf->data, nullptr);
  EXPECT_EQ(empty_buf->size, 0);
  EXPECT_TRUE(empty_buf->view().empty());

  EXPECT_EQ(context.DecompressToBuffer("invalid frame").status().code(),
            absl::StatusCode::kInternal);
  EXPECT_EQ(context.DecompressToBuffer(unknown_size_frame).status().code(),
            absl::StatusCode::kInternal);
  EXPECT_EQ(context.DecompressToBuffer(corrupt_empty).status().code(),
            absl::StatusCode::kInternal);
  EXPECT_EQ(context.DecompressToBuffer(truncated_non_empty).status().code(),
            absl::StatusCode::kInternal);

  // ShrinkToDefaultCapacity is a no-op when capacity <= kDefaultBufferSize.
  context.ShrinkToDefaultCapacity();
  EXPECT_EQ(context.capacity(), ZstdDecompressorContext::kDefaultBufferSize);

  // Growing beyond kDefaultBufferSize and shrinking back restores default cap.
  const std::string oversized_payload(
      ZstdDecompressorContext::kDefaultBufferSize + 1024, 'Z');
  const absl::StatusOr<std::string> oversized_compressed =
      ZstdCompression::Compress(oversized_payload);
  ASSERT_TRUE(oversized_compressed.ok());
  const absl::StatusOr<absl::string_view> oversized_res =
      context.Decompress(*oversized_compressed);
  ASSERT_TRUE(oversized_res.ok());
  EXPECT_EQ(*oversized_res, oversized_payload);
  EXPECT_GT(context.capacity(), ZstdDecompressorContext::kDefaultBufferSize);

  context.ShrinkToDefaultCapacity();
  EXPECT_EQ(context.capacity(), ZstdDecompressorContext::kDefaultBufferSize);
}

TEST(ZstdDecompressorContextTest,
     DecompressStreamZeroInitConcatenatedEmptyExactBoundaryAndTruncated) {
  // Zero initial capacity lazily allocates kDefaultBufferSize on
  // DecompressStream.
  ZstdDecompressorContext zero_cap_ctx(/*initial_capacity=*/0);
  EXPECT_EQ(zero_cap_ctx.capacity(), 0);

  // Compressed empty payload yields 0 output bytes (out_buf.pos == 0).
  const absl::StatusOr<std::string> empty_compressed =
      ZstdCompression::Compress("");
  ASSERT_TRUE(empty_compressed.ok());
  const absl::StatusOr<absl::string_view> zero_cap_empty =
      zero_cap_ctx.Decompress(*empty_compressed);
  ASSERT_TRUE(zero_cap_empty.ok());
  EXPECT_TRUE(zero_cap_empty->empty());
  EXPECT_EQ(zero_cap_ctx.capacity(), 0);
  int empty_chunks = 0;
  ASSERT_TRUE(zero_cap_ctx
                  .DecompressStream(*empty_compressed,
                                    [&](absl::string_view) {
                                      ++empty_chunks;
                                      return absl::OkStatus();
                                    })
                  .ok());
  EXPECT_EQ(empty_chunks, 0);
  EXPECT_EQ(zero_cap_ctx.capacity(),
            ZstdDecompressorContext::kDefaultBufferSize);

  // Concatenated Zstd frames exercise ret == 0 with in_buf.pos < in_buf.size.
  const std::string part1 = "first-frame-data-";
  const std::string part2 = "second-frame-data";
  const absl::StatusOr<std::string> comp1 = ZstdCompression::Compress(part1);
  const absl::StatusOr<std::string> comp2 = ZstdCompression::Compress(part2);
  ASSERT_TRUE(comp1.ok());
  ASSERT_TRUE(comp2.ok());
  const std::string concatenated = *comp1 + *comp2;
  std::string concatenated_out;
  ASSERT_TRUE(zero_cap_ctx
                  .DecompressStream(concatenated,
                                    [&](absl::string_view chunk) {
                                      concatenated_out.append(chunk.data(),
                                                              chunk.size());
                                      return absl::OkStatus();
                                    })
                  .ok());
  EXPECT_EQ(concatenated_out, part1 + part2);

  // Exact multiple of buffer capacity (2048 bytes with 1024-byte buffer)
  // exercises in_buf.pos == in_buf.size && out_buf.pos == out_buf.size while
  // ret != 0.
  ZstdDecompressorContext small_ctx(/*initial_capacity=*/1024);
  std::string exact_payload;
  exact_payload.reserve(2048);
  for (int i = 0; i < 2048; ++i) {
    exact_payload.push_back(static_cast<char>('a' + (i % 26)));
  }
  const absl::StatusOr<std::string> exact_comp =
      ZstdCompression::Compress(exact_payload);
  ASSERT_TRUE(exact_comp.ok());
  std::string exact_reconstructed;
  ASSERT_TRUE(small_ctx
                  .DecompressStream(*exact_comp,
                                    [&](absl::string_view chunk) {
                                      exact_reconstructed.append(chunk.data(),
                                                                 chunk.size());
                                      return absl::OkStatus();
                                    })
                  .ok());
  EXPECT_EQ(exact_reconstructed, exact_payload);

  // Truncated valid frame triggers incomplete-stream error.
  ASSERT_GT(exact_comp->size(), 5);
  const absl::string_view truncated_stream(exact_comp->data(),
                                           exact_comp->size() - 5);
  const absl::Status truncated_status = small_ctx.DecompressStream(
      truncated_stream, [](absl::string_view) { return absl::OkStatus(); });
  EXPECT_EQ(truncated_status.code(), absl::StatusCode::kInternal);

  // Frame truncated right after a 16-byte non-last raw block fills a 16-byte
  // output buffer (`out_buf.pos == out_buf.size` with `ret != 0` and
  // `in_buf.pos == in_buf.size`), then fails on the following read pass.
  ZstdDecompressorContext exact_block_ctx(/*initial_capacity=*/16);
  const std::string block_boundary_truncated(
      "\x28\xb5\x2f\xfd\x00\x00\x80\x00\x00"
      "0123456789abcdef",
      25);
  std::string partial_out;
  const absl::Status block_boundary_status = exact_block_ctx.DecompressStream(
      block_boundary_truncated, [&](absl::string_view chunk) {
        partial_out.append(chunk.data(), chunk.size());
        return absl::OkStatus();
      });
  EXPECT_EQ(partial_out, "0123456789abcdef");
  EXPECT_EQ(block_boundary_status.code(), absl::StatusCode::kInternal);

  // Valid 32-byte checksum-free RLE frame streamed with a 16-byte output
  // buffer flushes across two passes without false truncation detection.
  // Header: magic (4B), FHD=0x00, WD=0x00; Last RLE block of 32 bytes
  // ((32 << 3) | (1 << 1) | 1 = 0x000103) of 'M'.
  const std::string multi_pass_frame(
      "\x28\xb5\x2f\xfd\x00\x00\x03\x01\x00M", 10);
  std::string multi_pass_out;
  int pass_count = 0;
  const absl::Status multi_pass_status = exact_block_ctx.DecompressStream(
      multi_pass_frame, [&](absl::string_view chunk) {
        ++pass_count;
        multi_pass_out.append(chunk.data(), chunk.size());
        return absl::OkStatus();
      });
  ASSERT_TRUE(multi_pass_status.ok()) << multi_pass_status.message();
  EXPECT_EQ(multi_pass_out, std::string(32, 'M'));
  EXPECT_EQ(pass_count, 2);
}

TEST(ZstdDecompressorContextTest, DecompressStreamEmptyAndCorruptInput) {
  ZstdDecompressorContext context;

  // Empty compressed input -> InvalidArgument
  absl::Status empty_status = context.DecompressStream(
      "", [](absl::string_view) { return absl::OkStatus(); });
  EXPECT_TRUE(absl::IsInvalidArgument(empty_status));

  // Corrupt compressed input -> InternalError
  absl::Status corrupt_status = context.DecompressStream(
      "invalid compressed data",
      [](absl::string_view) { return absl::OkStatus(); });
  EXPECT_FALSE(corrupt_status.ok());
  EXPECT_EQ(corrupt_status.code(), absl::StatusCode::kInternal);
}

#ifdef __linux__
TEST(ZstdDecompressorContextTest,
     MoveAssignmentAndDestructionReclaimZstdContext) {
  constexpr int kIterations = 2000;
  constexpr int64_t kMaxRssGrowthKb = 32 * 1024;  // 32 MiB.

  struct rusage baseline_usage {};
  ASSERT_EQ(getrusage(RUSAGE_SELF, &baseline_usage), 0);
  const int64_t baseline_rss_kb =
      static_cast<int64_t>(baseline_usage.ru_maxrss);
  ASSERT_GT(baseline_rss_kb, 0);

  ZstdDecompressorContext dst(/*initial_capacity=*/0);
  for (int i = 0; i < kIterations; ++i) {
    ZstdDecompressorContext src(/*initial_capacity=*/0);
    dst = std::move(src);
  }
  for (int i = 0; i < kIterations; ++i) {
    ZstdDecompressorContext scoped(/*initial_capacity=*/0);
  }

  struct rusage after_usage {};
  ASSERT_EQ(getrusage(RUSAGE_SELF, &after_usage), 0);
  const int64_t after_rss_kb = static_cast<int64_t>(after_usage.ru_maxrss);
  ASSERT_GT(after_rss_kb, 0);
  EXPECT_LT(after_rss_kb - baseline_rss_kb, kMaxRssGrowthKb);
}
#endif  // __linux__

}  // namespace
}  // namespace profiler
}  // namespace tensorflow
