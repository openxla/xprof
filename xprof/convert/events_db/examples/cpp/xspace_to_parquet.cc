/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// Example CLI to convert an XSpace trace into an Events DB Parquet file.
//
// Usage:
//   Only the first two flags are required.
//   ```shell
//   bazel run \
//       //xprof/convert/events_db/examples/cpp:xspace_to_parquet \
//       -- \
//       --input_path=/path/to/trace.xplane.pb \
//       --output_path=/path/to/events.parquet \
//       --batch_size=65536 \
//       --compression_type=SNAPPY \
//       --max_record_count=10
//   ```

#include <cstdint>
#include <iostream>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "third_party/arrow/result.h"
#include "third_party/arrow/util/compression.h"
#include "third_party/arrow/util/type_fwd.h"
#include "tsl/platform/init_main.h"
#include "xprof/convert/events_db/parquet_record_consumer.h"
#include "xprof/convert/events_db/record_consumer.h"
#include "xprof/convert/events_db/schema.h"
#include "xprof/convert/events_db/xspace_parser.h"

namespace {

std::string_view ParseStatusToString(xprof::events_db::ParseStatus status) {
  switch (status) {
    case xprof::events_db::ParseStatus::kComplete:
      return "ParseStatus.COMPLETE";
    case xprof::events_db::ParseStatus::kStoppedEarly:
      return "ParseStatus.STOPPED_EARLY";
    default:
      return "ParseStatus.UNKNOWN";  // Should never happen.
  }
}

absl::StatusOr<std::optional<arrow::Compression::type>> GetCompressionType(
    std::string_view compression_type_str) {
  if (compression_type_str.empty()) return std::nullopt;
  arrow::Result<arrow::Compression::type> codec =
      arrow::util::Codec::GetCompressionType(
          absl::AsciiStrToLower(compression_type_str));
  if (codec.ok()) return *codec;
  return absl::InvalidArgumentError(
      absl::StrCat("Invalid compression type: '", compression_type_str, "'"));
}

}  // namespace

ABSL_FLAG(std::string, input_path, "",
          "Path to input XSpace/trace file (e.g., .xplane.pb).");
ABSL_FLAG(std::string, output_path, "",
          "Path to output Events DB Parquet file.");
ABSL_FLAG(uint32_t, batch_size, 65536,
          "Number of records buffered before flushing a batch to disk.");
ABSL_FLAG(std::string, compression_type, "",
          "Compression codec applied to Parquet data pages (e.g., SNAPPY, "
          "ZSTD, GZIP).");
ABSL_FLAG(std::optional<int>, compression_level, std::nullopt,
          "Compressor-specific compression level (e.g. 1-22 for ZSTD, 1-9 for "
          "GZIP).");
ABSL_FLAG(std::optional<uint64_t>, max_record_count, std::nullopt,
          "If set, at most this many records will be written before stopping.");

int main(int argc, char* argv[]) {
  tsl::port::InitMain(
      "Converts an XSpace trace file into an Events DB Parquet file.\n"
      "Usage:\n"
      "  xspace_to_parquet --input_path=<path> --output_path=<path> [flags]",
      &argc, &argv);

  const std::vector<char*> positional_args = absl::ParseCommandLine(argc, argv);
  if (positional_args.size() > 1) {
    std::cerr << "Too many command-line arguments.\n";
    return 1;
  }

  const std::string input_path = absl::GetFlag(FLAGS_input_path);
  if (input_path.empty()) {
    std::cerr << "input_path cannot be empty.\n";
    return 1;
  }

  const std::string output_path = absl::GetFlag(FLAGS_output_path);
  if (output_path.empty()) {
    std::cerr << "output_path cannot be empty.\n";
    return 1;
  }

  const uint32_t batch_size = absl::GetFlag(FLAGS_batch_size);
  const std::optional<int> compression_level =
      absl::GetFlag(FLAGS_compression_level);
  const std::optional<uint64_t> max_record_count =
      absl::GetFlag(FLAGS_max_record_count);

  const std::string flag_compression_type =
      absl::GetFlag(FLAGS_compression_type);
  const absl::StatusOr<std::optional<arrow::Compression::type>>
      compression_type = GetCompressionType(flag_compression_type);
  if (!compression_type.ok()) {
    std::cerr << "Invalid compression type: " << compression_type.status()
              << "\n";
    return 1;
  }

  const std::string compression_type_str =
      compression_type->has_value() ? flag_compression_type : "not set";
  const std::string compression_level_str =
      compression_level.has_value() ? absl::StrCat(*compression_level)
                                    : "not set";
  const std::string max_record_count_str = max_record_count.has_value()
                                               ? absl::StrCat(*max_record_count)
                                               : "not set";

  const absl::Time start_time = absl::Now();

  std::cout << "Converting XSpace trace to Events DB...\n"
            << "  Input XSpace: '" << input_path << "'\n"
            << "  Output Parquet: '" << output_path << "'\n"
            << "  Batch Size: " << batch_size << "\n"
            << "  Compression Type: " << compression_type_str << "\n"
            << "  Compression Level: " << compression_level_str << "\n"
            << "  Max Record Count: " << max_record_count_str << "\n";

  xprof::events_db::Schema schema;
  xprof::events_db::ParquetExportOptions parquet_options;
  parquet_options.batch_size = batch_size;
  parquet_options.compression_type = *compression_type;
  parquet_options.compression_level = compression_level;
  parquet_options.max_record_count = max_record_count;

  absl::StatusOr<xprof::events_db::ParquetRecordConsumer> consumer =
      xprof::events_db::ParquetRecordConsumer::Build(
          schema, output_path, std::move(parquet_options));
  if (!consumer.ok()) {
    std::cerr << "Conversion failed: " << consumer.status() << "\n";
    return 1;
  }

  const absl::StatusOr<xprof::events_db::ParseStatus> parse_status =
      xprof::events_db::ParseXSpace(input_path, schema, *consumer);
  if (!parse_status.ok()) {
    std::cerr << "Conversion failed: " << parse_status.status() << "\n";
    return 1;
  }

  const absl::Duration elapsed = absl::Now() - start_time;
  std::cout << absl::StrFormat(
      "Successfully finished parsing in %.2fs with status: %s\n",
      absl::ToDoubleSeconds(elapsed), ParseStatusToString(*parse_status));

  return 0;
}
