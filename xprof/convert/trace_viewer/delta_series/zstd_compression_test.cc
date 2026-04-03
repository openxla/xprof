#include "xprof/convert/trace_viewer/delta_series/zstd_compression.h"

#include <string>

#include "<gtest/gtest.h>"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"

namespace tensorflow {
namespace profiler {
namespace {

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

}  // namespace
}  // namespace profiler
}  // namespace tensorflow
